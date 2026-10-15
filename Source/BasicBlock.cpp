#include <atomic>
#include "Profiler.h"
#include "BasicBlock.h"
#include "MemStream.h"
#include "offsetof_def.h"
#include "MipsJitter.h"
#include "Jitter_CodeGenFactory.h"
#include "ee/VUShared.h"
#ifdef __EMSCRIPTEN__
#include <cstdarg>
#include <cstdio>
#include <string>
#include <cstring>
#include <vector>
#include "BlockCache.h"
#include "xxhash.h"
#include "Jitter_CodeGen_Wasm.h"

//VU blocks keep context symbols (VF/VI registers, flags, pipeline state) in wasm locals.
bool g_vuContextPromotion = true;
//Differential test: every VU block also gets a reference version (no promotion); both run from
//the same state and their results are compared. See CBasicBlock::ExecuteDiff.
bool g_vuDiffMode = false;
thread_local bool g_vuDiffSuppressKick = false;
//Name every JIT function (wasm name section) so CPU profiles tell blocks apart
bool g_jitFunctionNames = false;
std::atomic<uint32> g_vuDiffRuns = 0;
std::atomic<uint32> g_vuDiffMismatches = 0;
std::mutex g_vuDiffLogMutex;
std::string g_vuDiffLog;

static void VuDiffLog(const char* format, ...)
{
	char line[256];
	va_list args;
	va_start(args, format);
	vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	printf("%s", line);
	std::lock_guard<std::mutex> lock(g_vuDiffLogMutex);
	if(g_vuDiffLog.size() < 8192) g_vuDiffLog += line;
}
#endif

#if defined(AOT_BUILD_CACHE) || defined(AOT_USE_CACHE)
#define AOT_ENABLED
#endif

#ifdef AOT_ENABLED

#include "xxhash.h"

#include "StdStream.h"
#include "StdStreamUtils.h"

#endif

#ifdef VTUNE_ENABLED
#include <jitprofiling.h>
#include "string_format.h"
#endif

#ifdef AOT_USE_CACHE

#pragma pack(push, 1)
struct AOT_BLOCK
{
	AOT_BLOCK_KEY key;
	void* fct;
};
#pragma pack(pop)

extern "C"
{
	extern AOT_BLOCK _aot_firstBlock;
	extern uint32 _aot_blockCount;
}

#endif

#define INVALID_LINK_SLOT (~0U)

CBasicBlock::CBasicBlock(CMIPS& context, uint32 begin, uint32 end, BLOCK_CATEGORY category)
    : m_begin(begin)
    , m_end(end)
    , m_category(category)
    , m_context(context)
#ifdef AOT_USE_CACHE
    , m_function(nullptr)
#endif
{
	assert(m_end >= m_begin);
	for(uint32 i = 0; i < LINK_SLOT_MAX; i++)
	{
#ifdef _DEBUG
		m_linkBlock[i] = nullptr;
#endif
		m_linkBlockTrampolineOffset[i] = INVALID_LINK_SLOT;
	}
}

#ifdef AOT_BUILD_CACHE

Framework::CStdStream* CBasicBlock::m_aotBlockOutputStream(nullptr);
std::mutex CBasicBlock::m_aotBlockOutputStreamMutex;

void CBasicBlock::SetAotBlockOutputStream(Framework::CStdStream* outputStream)
{
	assert(m_aotBlockOutputStream == nullptr || outputStream == nullptr);
	m_aotBlockOutputStream = outputStream;
}

#endif

#ifdef PROFILE
std::atomic<uint32> g_blockCompileCount = 0;
std::atomic<uint32> g_vuBlockExecCount = 0;
#endif

//Block chaining switch (Module.setEeBlockChaining), read when blocks are compiled
bool g_eeBlockChaining = true;

bool CBasicBlock::UsesBlockChaining() const
{
#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	return g_eeBlockChaining && (m_context.m_blockFnLookup != nullptr) && !IsEmpty();
#else
	return false;
#endif
}

void CBasicBlock::Compile()
{
#ifdef PROFILE
	static const auto compileZone = CProfiler::GetInstance().RegisterZone("COMPILE");
	CProfilerZone profilerZone(compileZone);
	g_blockCompileCount++;
#endif
#ifndef AOT_USE_CACHE

#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	//Persistent block cache: reuse the wasm bytes generated for the same code in an earlier run.
	//Code is position independent (context-relative accesses, PC-relative branches, import table
	//indexes fixed at startup), so the key is the block contents plus everything else that changes codegen.
	BlockCache::KEY cacheKey = {};
	extern int g_vuExperiment;
	//(The VU differential test always compiles, so every block gets its reference code)
	bool useBlockCache = BlockCache::IsEnabled() && !IsEmpty() && (g_vuExperiment == 0) && !(g_vuDiffMode && (m_category == BLOCK_CATEGORY_PS2_VU));
	if(useBlockCache)
	{
		uint32 wordCount = ((m_end - m_begin) / 4) + 1;
		thread_local std::vector<uint32> words;
		words.resize(wordCount);
		for(uint32 i = 0; i < wordCount; i++)
		{
			words[i] = m_context.m_pMemoryMap->GetInstruction(m_begin + (i * 4));
		}
		auto xxHash = XXH3_128bits(words.data(), wordCount * 4);
		cacheKey.category = m_category;
		cacheKey.size = wordCount * 4;
		cacheKey.hints = m_blockCompileHints;
		cacheKey.variant = m_cacheVariant | (UsesContextPromotion() ? 0 : 0x40000000) | (IsSuperBlock() ? 0x20000000 : 0) | (UsesBlockChaining() ? 0x10000000 : 0);
		cacheKey.hash[0] = xxHash.low64;
		cacheKey.hash[1] = xxHash.high64;

		thread_local std::vector<uint8> cachedCode;
		if(BlockCache::Find(cacheKey, cachedCode))
		{
#ifdef PROFILE
			static const auto compileWasmZone = CProfiler::GetInstance().RegisterZone("COMPILE_WASM");
			CProfilerZone profilerZone(compileWasmZone);
#endif
			m_function = CMemoryFunction(cachedCode.data(), cachedCode.size());
			return;
		}
	}
#endif

	Framework::CMemStream stream;
	CompileToStream(stream, UsesContextPromotion());

#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	//(Not the executor's empty block: it compiles and runs the real block, which has its own test)
	if(g_vuDiffMode && (m_category == BLOCK_CATEGORY_PS2_VU) && UsesContextPromotion() && !IsEmpty())
	{
		//Superblocks are checked against the reference blocks run one by one (see CVuBasicBlock)
		if(!IsSuperBlock())
		{
			Framework::CMemStream refStream;
			CompileToStream(refStream, false);
			m_refFunction = CMemoryFunction(refStream.GetBuffer(), refStream.GetSize());
		}
		m_diffExecute = true;
	}
#endif

	{
#ifdef PROFILE
		static const auto compileWasmZone = CProfiler::GetInstance().RegisterZone("COMPILE_WASM");
		CProfilerZone profilerZone(compileWasmZone);
#endif
		m_function = CMemoryFunction(stream.GetBuffer(), stream.GetSize());
	}

#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	if(useBlockCache && m_cacheable)
	{
		BlockCache::Add(cacheKey, stream.GetBuffer(), stream.GetSize());
	}
#endif

#ifdef VTUNE_ENABLED
	if(iJIT_IsProfilingActive() == iJIT_SAMPLING_ON)
	{
		iJIT_Method_Load jmethod = {};
		jmethod.method_id = iJIT_GetNewMethodID();
		jmethod.class_file_name = "";
		jmethod.source_file_name = __FILE__;

		jmethod.method_load_address = m_function.GetCode();
		jmethod.method_size = m_function.GetSize();
		jmethod.line_number_size = 0;

		auto functionName = string_format("BasicBlock_0x%08X_0x%08X", m_begin, m_end);

		jmethod.method_name = const_cast<char*>(functionName.c_str());
		iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, reinterpret_cast<void*>(&jmethod));
	}
#endif

#endif

#ifdef AOT_ENABLED

	size_t blockSize = ((m_end - m_begin) / 4) + 1;
	size_t blockSizeByte = blockSize * 4;
	auto blockData = new uint32[blockSize];

	if(!IsEmpty())
	{
		for(uint32 i = 0; i < blockSize; i++)
		{
			blockData[i] = m_context.m_pMemoryMap->GetInstruction(m_begin + (i * 4));
		}
	}
	else
	{
		//The empty block has no data per se
		assert(blockSize == 1);
		blockData[0] = ~0;
	}

	auto xxHash = XXH3_128bits(blockData, blockSizeByte);
	uint128 hash;
	memcpy(&hash, &xxHash, sizeof(xxHash));

#endif

#ifdef AOT_USE_CACHE

	AOT_BLOCK* blocksBegin = &_aot_firstBlock;
	AOT_BLOCK* blocksEnd = blocksBegin + _aot_blockCount;

	AOT_BLOCK blockRef = {{m_category, hash, blockSizeByte}, nullptr};

	static const auto blockComparer =
	    [](const AOT_BLOCK& item1, const AOT_BLOCK& item2) {
		    return item1.key < item2.key;
	    };

	//	assert(std::is_sorted(blocksBegin, blocksEnd, blockComparer));

	bool blockExists = std::binary_search(blocksBegin, blocksEnd, blockRef, blockComparer);
	auto blockIterator = std::lower_bound(blocksBegin, blocksEnd, blockRef, blockComparer);

	assert(blockExists);
	assert(blockIterator != blocksEnd);
	assert(blockIterator->key.hash == hash);
	assert(blockIterator->key.size == blockSizeByte);

	m_function = reinterpret_cast<void (*)(void*)>(blockIterator->fct);

#endif

#ifdef AOT_BUILD_CACHE
	if(m_aotBlockOutputStream)
	{
		std::lock_guard<std::mutex> lock(m_aotBlockOutputStreamMutex);

		m_aotBlockOutputStream->Write32(m_category);
		m_aotBlockOutputStream->Write(&hash, sizeof(hash));
		m_aotBlockOutputStream->Write32(blockSizeByte);
		m_aotBlockOutputStream->Write(blockData, blockSizeByte);
	}
#endif
}

void CBasicBlock::AddBlockCompileHints(uint32 compileHints)
{
	m_blockCompileHints |= compileHints;
}

void CBasicBlock::AddCacheVariant(uint32 variant)
{
	m_cacheVariant |= variant;
}

void CBasicBlock::CompileRange(CMipsJitter* jitter)
{
	if(IsEmpty())
	{
		jitter->JumpTo(reinterpret_cast<void*>(&EmptyBlockHandler));
		return;
	}

	bool loopsOnItself = [&]() {
		if(m_begin == m_end)
		{
			return false;
		}
		uint32 branchInstAddr = m_end - 4;
		uint32 inst = m_context.m_pMemoryMap->GetInstruction(branchInstAddr);
		if(m_context.m_pArch->IsInstructionBranch(&m_context, branchInstAddr, inst) != MIPS_BRANCH_NORMAL)
		{
			return false;
		}
		uint32 target = m_context.m_pArch->GetInstructionEffectiveAddress(&m_context, branchInstAddr, inst);
		if(target == MIPS_INVALID_PC)
		{
			return false;
		}
		return target == m_begin;
	}();

	m_context.m_pArch->SetCompileHints(m_blockCompileHints);

	CompileProlog(jitter);
	jitter->MarkFirstBlockLabel();

	for(uint32 address = m_begin; address <= m_end; address += 4)
	{
		m_context.m_pArch->CompileInstruction(
		    address,
		    jitter,
		    &m_context, address - m_begin);
		//Sanity check
		assert(jitter->IsStackEmpty());
	}

	jitter->MarkLastBlockLabel();
	CompileEpilog(jitter, loopsOnItself);
}

void CBasicBlock::CompileProlog(CMipsJitter* jitter)
{
#ifdef DEBUGGER_INCLUDED
	if(HasBreakpoint())
	{
		jitter->PushCtx();
		jitter->Call(reinterpret_cast<void*>(&BreakpointFilter), 1, Jitter::CJitter::RETURN_VALUE_32);

		jitter->PushCst(0);
		jitter->BeginIf(Jitter::CONDITION_EQ);
		{
			jitter->JumpTo(reinterpret_cast<void*>(&BreakpointHandler));
		}
		jitter->EndIf();
	}
#endif
}

void CBasicBlock::CompileEpilog(CMipsJitter* jitter, bool loopsOnItself)
{
	//Update cycle quota
	jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
	jitter->PushCst(((m_end - m_begin) / 4) + 1);
	jitter->Sub();
	jitter->PullRel(offsetof(CMIPS, m_State.cycleQuota));

	jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
	jitter->PushCst(0);
	jitter->BeginIf(Jitter::CONDITION_LE);
	{
		jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
		jitter->PushCst(MIPS_EXCEPTION_STATUS_QUOTADONE);
		jitter->Or();
		jitter->PullRel(offsetof(CMIPS, m_State.nHasException));
	}
	jitter->EndIf();

	//We probably don't need to pay for this since we know in advance if there's a branch
	jitter->PushCst(MIPS_INVALID_PC);
	jitter->PushRel(offsetof(CMIPS, m_State.nDelayedJumpAddr));
	jitter->BeginIf(Jitter::CONDITION_NE);
	{
		jitter->PushRel(offsetof(CMIPS, m_State.nDelayedJumpAddr));
		jitter->PullRel(offsetof(CMIPS, m_State.nPC));

		jitter->PushCst(MIPS_INVALID_PC);
		jitter->PullRel(offsetof(CMIPS, m_State.nDelayedJumpAddr));

		if(loopsOnItself)
		{
			jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
			jitter->PushCst(0);
			jitter->BeginIf(Jitter::CONDITION_EQ);
			{
				jitter->Goto(jitter->GetFirstBlockLabel());
			}
			jitter->EndIf();
		}
		else
		{
#if !defined(AOT_BUILD_CACHE) && !defined(__EMSCRIPTEN__)
			jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
			jitter->PushCst(0);
			jitter->BeginIf(Jitter::CONDITION_EQ);
			{
				jitter->JumpToDynamic(reinterpret_cast<void*>(&BranchBlockTrampoline));
			}
			jitter->EndIf();
#endif
		}
	}
	jitter->Else();
	{
		jitter->PushRel(offsetof(CMIPS, m_State.nPC));
		jitter->PushCst(m_end - m_begin + 4);
		jitter->Add();
		jitter->PullRel(offsetof(CMIPS, m_State.nPC));

#if !defined(AOT_BUILD_CACHE) && !defined(__EMSCRIPTEN__)
		jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
		jitter->PushCst(0);
		jitter->BeginIf(Jitter::CONDITION_EQ);
		{
			jitter->JumpToDynamic(reinterpret_cast<void*>(&NextBlockTrampoline));
		}
		jitter->EndIf();
#endif
	}
	jitter->EndIf();

	if(UsesBlockChaining())
	{
		//Same as one more iteration of the executor loop: if there is no exception (cycle quota
		//exhausted, syscall, idle loop, ...), look up the block at nPC and tail call it. No block
		//there (index 0): return, the executor compiles it. Blocks deleted by the software write
		//protection have their index cleared, and nPC is the running block's address at any time.
		jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
		jitter->PushCst(0);
		jitter->BeginIf(Jitter::CONDITION_EQ);
		{
			jitter->PushRelRef(offsetof(CMIPS, m_blockFnLookup));
			jitter->PushRel(offsetof(CMIPS, m_State.nPC));
			jitter->Srl(16);
			jitter->PushCst(0x1FFF); //EE address mask (0x1FFFFFFF) >> 16
			jitter->And();
			jitter->LoadRefFromRefIdx();
			jitter->PushRel(offsetof(CMIPS, m_State.nPC));
			jitter->Srl(2);
			jitter->PushCst(0x3FFF);
			jitter->And();
			jitter->LoadFromRefIdx();
			jitter->JumpToIndirect();
		}
		jitter->EndIf();
	}
}

void CBasicBlock::CompileToStream(Framework::CMemStream& stream, bool contextPromotion)
{
	//One jitter per thread (VU1 can compile on its own thread)
	static thread_local CMipsJitter* jitter = nullptr;
	if(jitter == nullptr)
	{
		Jitter::CCodeGen* codeGen = Jitter::CreateCodeGen();
		jitter = new CMipsJitter(codeGen);
	}

#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	auto wasmCodeGen = static_cast<Jitter::CCodeGen_Wasm*>(jitter->GetCodeGen());
	wasmCodeGen->SetContextPromotion(contextPromotion);
	if(g_jitFunctionNames)
	{
		const char* prefix = (m_category == BLOCK_CATEGORY_PS2_VU) ? (m_diffDataSize > 0x1000 ? "vu1" : "vu0") : (m_category == BLOCK_CATEGORY_PS2_EE) ? "ee" : "iop";
		char name[64];
		snprintf(name, sizeof(name), "%s%s%s_%08X_%08X", prefix, contextPromotion ? "p" : "", IsSuperBlock() ? "s" : "", m_begin, m_end);
		wasmCodeGen->SetFunctionName(name);
	}
#endif
	//Flag value forwarding: VU blocks only (straight-line FMAC code), part of the optimized VU codegen
	m_compileOptimized = (m_category == BLOCK_CATEGORY_PS2_VU) && contextPromotion;
	VUShared::ResetFlagForwarding(m_compileOptimized);
	jitter->GetCodeGen()->SetExternalSymbolReferencedHandler([&](auto symbol, auto offset, auto refType) { this->HandleExternalFunctionReference(symbol, offset, refType); });
	jitter->SetStream(&stream);
	jitter->Begin();
	CompileRange(jitter);
	jitter->End();
}

bool CBasicBlock::UsesContextPromotion() const
{
#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	return (m_category == BLOCK_CATEGORY_PS2_VU) && g_vuContextPromotion && !m_forceReference;
#else
	return false;
#endif
}

#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
void VuDiffCompare(uint32 begin, uint32 end, const MIPSSTATE& refState, const MIPSSTATE& newState, const uint8* refMem, const uint8* newMem, uint32 dataSize)
{
	g_vuDiffRuns++;
	bool stateDiffers = memcmp(&refState, &newState, sizeof(MIPSSTATE)) != 0;
	bool memDiffers = memcmp(refMem, newMem, dataSize) != 0;
	if(stateDiffers || memDiffers)
	{
		uint32 count = g_vuDiffMismatches++;
		if(count < 50)
		{
			auto ref = reinterpret_cast<const uint32*>(&refState);
			auto cur = reinterpret_cast<const uint32*>(&newState);
			for(uint32 i = 0; i < sizeof(MIPSSTATE) / 4; i++)
			{
				if(ref[i] != cur[i])
				{
					VuDiffLog("VUDIFF block %08X-%08X (size %u) state offset 0x%X: ref %08X new %08X\n",
					          begin, end, dataSize, i * 4, ref[i], cur[i]);
				}
			}
			if(memDiffers)
			{
				VuDiffLog("VUDIFF block %08X-%08X data memory differs\n", begin, end);
			}
		}
	}
}

void CBasicBlock::ExecuteDiff()
{
	uint32 dataSize = m_diffDataSize;
	auto dataMem = reinterpret_cast<uint8*>(m_context.m_vuMem);
	static thread_local MIPSSTATE initState, refState;
	static thread_local std::vector<uint8> initMem, refMem;
	initMem.resize(dataSize);
	refMem.resize(dataSize);

	memcpy(&initState, &m_context.m_State, sizeof(MIPSSTATE));
	memcpy(initMem.data(), dataMem, dataSize);

	g_vuDiffSuppressKick = true;
	m_refFunction(&m_context);
	g_vuDiffSuppressKick = false;

	memcpy(&refState, &m_context.m_State, sizeof(MIPSSTATE));
	memcpy(refMem.data(), dataMem, dataSize);
	memcpy(&m_context.m_State, &initState, sizeof(MIPSSTATE));
	memcpy(dataMem, initMem.data(), dataSize);

	m_function(&m_context);

	VuDiffCompare(m_begin, m_end, refState, m_context.m_State, refMem.data(), dataMem, dataSize);
}
#endif

void CBasicBlock::Execute()
{
#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	if(m_diffExecute)
	{
		ExecuteDiff();
		return;
	}
#endif
	m_function(&m_context);

	assert(m_context.m_State.nGPR[0].nV0 == 0);
	assert(m_context.m_State.nGPR[0].nV1 == 0);
	assert(m_context.m_State.nGPR[0].nV2 == 0);
	assert(m_context.m_State.nGPR[0].nV3 == 0);
	assert((m_context.m_State.nPC & 3) == 0);
	assert(m_context.m_State.nCOP2[0].nV0 == 0x00000000);
	assert(m_context.m_State.nCOP2[0].nV1 == 0x00000000);
	assert(m_context.m_State.nCOP2[0].nV2 == 0x00000000);
	assert(m_context.m_State.nCOP2[0].nV3 == 0x3F800000);
	assert(m_context.m_State.nCOP2VI[0] == 0);
}

uint32 CBasicBlock::GetBeginAddress() const
{
	return m_begin;
}

uint32 CBasicBlock::GetEndAddress() const
{
	return m_end;
}

bool CBasicBlock::IsCompiled() const
{
#ifndef AOT_USE_CACHE
	return !m_function.IsEmpty();
#else
	return (m_function != nullptr);
#endif
}

bool CBasicBlock::IsEmpty() const
{
	return (m_begin == MIPS_INVALID_PC) &&
	       (m_end == MIPS_INVALID_PC);
}

uint32 CBasicBlock::GetRecycleCount() const
{
	return m_recycleCount;
}

void CBasicBlock::SetRecycleCount(uint32 recycleCount)
{
	m_recycleCount = recycleCount;
}

bool CBasicBlock::HasLinkSlot(LINK_SLOT linkSlot) const
{
	return m_linkBlockTrampolineOffset[linkSlot] != INVALID_LINK_SLOT;
}

BlockOutLinkPointer CBasicBlock::GetOutLink(LINK_SLOT linkSlot) const
{
	assert(linkSlot < LINK_SLOT_MAX);
	return m_outLinks[linkSlot];
}

void CBasicBlock::SetOutLink(LINK_SLOT linkSlot, BlockOutLinkPointer link)
{
	assert(linkSlot < LINK_SLOT_MAX);
	m_outLinks[linkSlot] = link;
}

void CBasicBlock::LinkBlock(LINK_SLOT linkSlot, CBasicBlock* otherBlock)
{
#if !defined(AOT_ENABLED) && !defined(__EMSCRIPTEN__)
	assert(!IsEmpty());
	assert(!otherBlock->IsEmpty());
	assert(linkSlot < LINK_SLOT_MAX);
	assert(m_linkBlockTrampolineOffset[linkSlot] != INVALID_LINK_SLOT);
#ifdef _DEBUG
	assert(m_linkBlock[linkSlot] == nullptr);
	m_linkBlock[linkSlot] = otherBlock;
#endif
	auto patchValue = reinterpret_cast<uintptr_t>(otherBlock->m_function.GetCode());
	auto code = reinterpret_cast<uint8*>(m_function.GetCode());
	m_function.BeginModify();
	*reinterpret_cast<uintptr_t*>(code + m_linkBlockTrampolineOffset[linkSlot]) = patchValue;
	m_function.EndModify();
#endif //!AOT_ENABLED && !__EMSCRIPTEN__
}

void CBasicBlock::UnlinkBlock(LINK_SLOT linkSlot)
{
#if !defined(AOT_ENABLED) && !defined(__EMSCRIPTEN__)
	assert(!IsEmpty());
	assert(linkSlot < LINK_SLOT_MAX);
	assert(m_linkBlockTrampolineOffset[linkSlot] != INVALID_LINK_SLOT);
#ifdef _DEBUG
	assert(m_linkBlock[linkSlot] != nullptr);
	m_linkBlock[linkSlot] = nullptr;
#endif
	auto patchValue = (linkSlot == LINK_SLOT_NEXT) ? reinterpret_cast<uintptr_t>(&NextBlockTrampoline) : reinterpret_cast<uintptr_t>(&BranchBlockTrampoline);
	auto code = reinterpret_cast<uint8*>(m_function.GetCode());
	m_function.BeginModify();
	*reinterpret_cast<uintptr_t*>(code + m_linkBlockTrampolineOffset[linkSlot]) = patchValue;
	m_function.EndModify();
#endif //!AOT_ENABLED && !__EMSCRIPTEN__
}

void CBasicBlock::HandleExternalFunctionReference(uintptr_t symbol, uint32 offset, Jitter::CCodeGen::SYMBOL_REF_TYPE refType)
{
	if(symbol == reinterpret_cast<uintptr_t>(&NextBlockTrampoline))
	{
		assert(refType == Jitter::CCodeGen::SYMBOL_REF_TYPE::NATIVE_POINTER);
		assert(m_linkBlockTrampolineOffset[LINK_SLOT_NEXT] == INVALID_LINK_SLOT);
		m_linkBlockTrampolineOffset[LINK_SLOT_NEXT] = offset;
	}
	else if(symbol == reinterpret_cast<uintptr_t>(&BranchBlockTrampoline))
	{
		assert(refType == Jitter::CCodeGen::SYMBOL_REF_TYPE::NATIVE_POINTER);
		assert(m_linkBlockTrampolineOffset[LINK_SLOT_BRANCH] == INVALID_LINK_SLOT);
		m_linkBlockTrampolineOffset[LINK_SLOT_BRANCH] = offset;
	}
}

void CBasicBlock::CopyFunctionFrom(const std::shared_ptr<CBasicBlock>& other)
{
#ifndef AOT_USE_CACHE
	m_function = other->m_function.CreateInstance();
#if defined(__EMSCRIPTEN__) && !defined(AOT_ENABLED)
	if(!other->m_refFunction.IsEmpty())
	{
		m_refFunction = other->m_refFunction.CreateInstance();
	}
	m_diffExecute = other->m_diffExecute;
#endif
	std::copy(std::begin(other->m_linkBlockTrampolineOffset), std::end(other->m_linkBlockTrampolineOffset), m_linkBlockTrampolineOffset);
#ifdef _DEBUG
	std::copy(std::begin(other->m_linkBlock), std::end(other->m_linkBlock), m_linkBlock);
#endif
	if(
	    HasLinkSlot(LINK_SLOT_NEXT)
#ifdef _DEBUG
	    && m_linkBlock[LINK_SLOT_NEXT]
#endif
	)
	{
		UnlinkBlock(LINK_SLOT_NEXT);
	}
	if(
	    HasLinkSlot(LINK_SLOT_BRANCH)
#ifdef _DEBUG
	    && m_linkBlock[LINK_SLOT_BRANCH]
#endif
	)
	{
		UnlinkBlock(LINK_SLOT_BRANCH);
	}
#else
	m_function = basicBlock->m_function;
#endif
}

#ifdef DEBUGGER_INCLUDED

bool CBasicBlock::HasBreakpoint() const
{
	return m_context.HasBreakpointInRange(GetBeginAddress(), GetEndAddress());
}

uint32 CBasicBlock::BreakpointFilter(CMIPS* context)
{
	return context->m_executor->FilterBreakpoint();
}

void CBasicBlock::BreakpointHandler(CMIPS* context)
{
	assert(context->m_State.nHasException == MIPS_EXCEPTION_NONE);
	context->m_State.nHasException = MIPS_EXCEPTION_BREAKPOINT;
}

#endif

void EmptyBlockHandler(CMIPS* context)
{
	context->m_emptyBlockHandler(context);
}

void NextBlockTrampoline(CMIPS* context)
{
}

void BranchBlockTrampoline(CMIPS* context)
{
}
