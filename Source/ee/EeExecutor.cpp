#include "EeExecutor.h"
#include "../Ps2Const.h"
#include "AlignedAlloc.h"
#include "EeBasicBlock.h"
#include "MA_EE.h"
#include "xxhash.h"
#include "../Profiler.h"
#include "offsetof_def.h"
#include <algorithm>
#if defined(__EMSCRIPTEN__)
#include "WasmModuleBuilder.h"
#endif

#if defined(__unix__) || defined(__ANDROID__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

#if defined(__APPLE__)

#include <TargetConditionals.h>

#if TARGET_CPU_ARM
#define DISABLE_PROTECTION
#define STATE_FLAVOR ARM_THREAD_STATE32
#define STATE_FLAVOR_COUNT ARM_THREAD_STATE32_COUNT
#elif TARGET_CPU_ARM64
#define STATE_FLAVOR ARM_THREAD_STATE64
#define STATE_FLAVOR_COUNT ARM_THREAD_STATE64_COUNT
#elif TARGET_CPU_X86
#define STATE_FLAVOR x86_THREAD_STATE32
#define STATE_FLAVOR_COUNT x86_THREAD_STATE32_COUNT
#elif TARGET_CPU_X86_64
#define STATE_FLAVOR x86_THREAD_STATE64
#define STATE_FLAVOR_COUNT x86_THREAD_STATE64_COUNT
#else
#error Unsupported CPU architecture
#endif

#if TARGET_OS_TV
#define DISABLE_PROTECTION
#endif

#endif

static CEeExecutor* g_eeExecutor = nullptr;

CEeExecutor::CEeExecutor(CMIPS& context, uint8* ram)
    : CGenericMipsExecutor(context, 0x20000000, BLOCK_CATEGORY_PS2_EE)
    , m_ram(ram)
{
	m_pageSize = framework_getpagesize();
#if defined(__EMSCRIPTEN__)
	m_pageSize = MIPS_PAGE_SIZE;
	m_protectedPages.resize(PS2::EE_RAM_SIZE / MIPS_PAGE_SIZE, 0);
	m_context.m_writeWatchHandler = &CEeExecutor::WriteWatchHandler;
	m_context.m_blockFnLookup = m_blockLookup.GetFnTable();
#endif
}

#if defined(__EMSCRIPTEN__)

//Blocks (or region functions) called by the executor loop (Module.eeDispatchCount, benchmarks)
uint64 g_eeDispatchCount = 0;

int CEeExecutor::Execute(int cycles)
{
	m_executing = true;
#if !defined(PROFILE) && !defined(DEBUGGER_INCLUDED)
	//Same loop as CGenericMipsExecutor::Execute, but the call goes through the function index table
	//(kept for block chaining): one dependent load less per block than going through the CBasicBlock.
	//Index 0 (no block yet) takes the usual path, which compiles it.
	m_context.m_State.cycleQuota = cycles;
	uint32** fnTable = m_blockLookup.GetFnTable();
	while(m_context.m_State.nHasException == 0)
	{
		uint32 address = m_context.m_State.nPC & m_addressMask;
		uint32 fctIdx = fnTable[address >> 16][(address & 0xFFFF) / 4];
		g_eeDispatchCount++;
		if(fctIdx != 0)
		{
			reinterpret_cast<void (*)(CMIPS*)>(static_cast<uintptr_t>(fctIdx))(&m_context);
		}
		else
		{
			m_blockLookup.FindBlockAt(address)->Execute();
		}
	}
	m_context.m_State.nHasException &= ~MIPS_EXCEPTION_STATUS_QUOTADONE;
	int result = m_context.m_State.cycleQuota;
#else
	int result = CGenericMipsExecutor::Execute(cycles);
#endif
	m_executing = false;
	return result;
}

void CEeExecutor::WriteWatchHandler(CMIPS* context, uint32 address, uint32 size)
{
	//Only called from JIT code (store slow path), so a block is executing
	auto executor = static_cast<CEeExecutor*>(context->m_executor.get());
	if((address >= PS2::EE_RAM_SIZE) && executor->m_ioWriteHandler)
	{
		executor->m_ioWriteHandler(address);
	}
	executor->InvalidateWrittenRange(address, size, true);
}

void CEeExecutor::NotifyRamWritten(uint32 address, uint32 size)
{
	if(!g_eeExecutor) return;
	g_eeExecutor->InvalidateWrittenRange(address, size, g_eeExecutor->m_executing);
}

extern uint32 g_eeBatchSize;

void CEeExecutor::QueueForBatch(const BasicBlockPtr& block)
{
	if(block->GetBatchModule().empty()) return;
	m_batchPending.push_back(block);
	if(m_batchPending.size() >= g_eeBatchSize) FlushBatch();
}

extern uint32 g_eeRegionSpan;

void CEeExecutor::FlushBatch()
{
	std::vector<BasicBlockPtr> blocks;
	for(const auto& weakBlock : m_batchPending)
	{
		auto block = weakBlock.lock();
		if(!block || block->GetBatchModule().empty()) continue;
		blocks.push_back(std::move(block));
	}
	m_batchPending.clear();
	if(blocks.empty()) return;
	//Ascending addresses: regions are runs of blocks within g_eeRegionSpan bytes of the first one
	std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) { return a->GetBeginAddress() < b->GetBeginAddress(); });
	std::vector<CWasmModuleBuilder::MODULE_BYTES> modules;
	for(const auto& block : blocks)
	{
		modules.push_back({block->GetBatchModule().data(), static_cast<uint32>(block->GetBatchModule().size())});
	}
	static const uint32 MAX_REGION_BLOCKS = 64;
	std::vector<CWasmModuleBuilder::BATCH_FUNCTION> functions;
	std::vector<uint32> exports(blocks.size());
	auto isLive = [&](uint32 i) { return m_blockLookup.FindBlockAt(blocks[i]->GetBeginAddress()) == blocks[i].get(); };
	for(uint32 i = 0; i < blocks.size();)
	{
		CWasmModuleBuilder::BATCH_FUNCTION function;
		uint32 first = blocks[i]->GetBeginAddress();
		uint32 next = i + 1;
		function.modules.push_back(i);
		function.starts.push_back(first);
		if((g_eeRegionSpan != 0) && isLive(i))
		{
			while((next < blocks.size()) && (function.modules.size() < MAX_REGION_BLOCKS) &&
			      ((blocks[next]->GetBeginAddress() - first) < g_eeRegionSpan) &&
			      (blocks[next]->GetBeginAddress() != function.starts.back()) && isLive(next))
			{
				function.modules.push_back(next);
				function.starts.push_back(blocks[next]->GetBeginAddress());
				next++;
			}
		}
		for(uint32 member : function.modules) exports[member] = static_cast<uint32>(functions.size());
		functions.push_back(std::move(function));
		i = next;
	}
	CWasmModuleBuilder::REGION_PARAMS regionParams;
	regionParams.pcOffset = offsetof(CMIPS, m_State.nPC);
	regionParams.exceptionOffset = offsetof(CMIPS, m_State.nHasException);
	regionParams.generationOffset = offsetof(CMIPS, m_blockGeneration);
	regionParams.addressMask = m_addressMask;
	bool hasRegions = functions.size() != blocks.size();
	auto batch = hasRegions ? CWasmModuleBuilder::BuildBatchModule(modules, functions, exports, regionParams) : std::vector<uint8>();
	if(batch.empty())
	{
		//No regions, or a block the region builder doesn't handle: one function per block
		hasRegions = false;
		batch = CWasmModuleBuilder::BuildBatchModule(modules);
	}
	auto fctIds = batch.empty() ? std::vector<int>() : CMemoryFunction::CreateWasmBatch(batch.data(), batch.size(), static_cast<uint32>(blocks.size()));
	for(uint32 i = 0; i < blocks.size(); i++)
	{
		if(fctIds.empty())
		{
			blocks[i]->GetBatchModule() = std::vector<uint8>();
			continue;
		}
		blocks[i]->ReplaceWasmFunction(fctIds[i]);
		m_blockLookup.UpdateFunctionIndex(blocks[i].get());
	}
	if(!hasRegions || fctIds.empty()) return;
	for(const auto& function : functions)
	{
		if(function.modules.size() < 2) continue;
		REGION region;
		for(uint32 member : function.modules)
		{
			region.end = std::max(region.end, blocks[member]->GetEndAddress());
			region.members.push_back(blocks[member]);
		}
		m_regions.emplace(function.starts[0], std::move(region));
	}
}

void CEeExecutor::DissolveRegions(uint32 start, uint32 end)
{
	//ponytail: linear scan; invalidations are rare in steady state. Index by address if they show up in profiles.
	for(auto regionIterator = m_regions.begin(); regionIterator != m_regions.end();)
	{
		const auto& region = regionIterator->second;
		if(!RangesOverlap(regionIterator->first, region.end, start, end - 1))
		{
			regionIterator++;
			continue;
		}
		for(const auto& weakMember : region.members)
		{
			auto member = weakMember.lock();
			if(!member) continue;
			member->RecreateFunction();
			m_blockLookup.UpdateFunctionIndex(member.get());
		}
		regionIterator = m_regions.erase(regionIterator);
	}
}

void CEeExecutor::InvalidateWrittenRange(uint32 address, uint32 size, bool executing)
{
	if(size == 0) return;
	if(address >= PS2::EE_RAM_SIZE) return;
	uint32 end = std::min<uint32>(address + size, PS2::EE_RAM_SIZE);
	for(uint32 page = address / MIPS_PAGE_SIZE; page <= (end - 1) / MIPS_PAGE_SIZE; page++)
	{
		if(m_protectedPages[page])
		{
#ifdef PROFILE
			PROFILE_COUNT(PCNT_EE_INVAL, 1);
			CProfiler::GetInstance().Log("W %08X %u pc=%08X\n", address, size, m_context.m_State.nPC);
#endif
			ClearActiveBlocksInRange(page * MIPS_PAGE_SIZE, (page + 1) * MIPS_PAGE_SIZE, executing);
		}
	}
}

#endif

void CEeExecutor::SetBlockFpRoundingModes(BlockFpRoundingModeMap blockFpRoundingModes)
{
	m_blockFpRoundingModes = std::move(blockFpRoundingModes);
}

void CEeExecutor::SetBlockFpUseAccurateAddSub(BlockFpUseAccurateAddSubSet blockFpUseAccurateAddSub)
{
	m_blockFpUseAccurateAddSub = std::move(blockFpUseAccurateAddSub);
}

void CEeExecutor::SetIdleLoopBlocks(IdleLoopBlockMap idleLoopBlocks)
{
	m_idleLoopBlocks = std::move(idleLoopBlocks);
}

void CEeExecutor::AddExceptionHandler()
{
	assert(g_eeExecutor == nullptr);
	g_eeExecutor = this;

#ifdef DISABLE_PROTECTION
	return;
#endif

#if defined(_WIN32)
	m_handler = AddVectoredExceptionHandler(TRUE, &CEeExecutor::HandleException);
	assert(m_handler != NULL);
#elif defined(__unix__) || defined(__ANDROID__)
	struct sigaction sigAction;
	sigAction.sa_handler = nullptr;
	sigAction.sa_sigaction = &HandleException;
	sigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&sigAction.sa_mask);
	int result = sigaction(SIGSEGV, &sigAction, nullptr);
	assert(result >= 0);
#elif defined(__APPLE__) && !TARGET_OS_TV
	if(!m_running)
	{
		kern_return_t result = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &m_port);
		assert(result == KERN_SUCCESS);

		m_running = true;
		m_handlerThread = std::thread([this]() { HandlerThreadProc(); });
	}

	AttachExceptionHandlerToThread();
#endif
}

void CEeExecutor::RemoveExceptionHandler()
{
#ifndef DISABLE_PROTECTION

#if defined(_WIN32)
	RemoveVectoredExceptionHandler(m_handler);
#elif defined(__APPLE__) && !TARGET_OS_TV
	m_running = false;
	m_handlerThread.join();
#endif

#endif //!DISABLE_PROTECTION

	g_eeExecutor = nullptr;
}

void CEeExecutor::AttachExceptionHandlerToThread()
{
	//Only necessary for macOS and iOS since the handler is set on a thread basis
#if defined(__APPLE__) && !TARGET_OS_TV
	assert(m_running);

	auto result = mach_port_insert_right(mach_task_self(), m_port, m_port, MACH_MSG_TYPE_MAKE_SEND);
	assert(result == KERN_SUCCESS);

	result = thread_set_exception_ports(mach_thread_self(), EXC_MASK_BAD_ACCESS, m_port, EXCEPTION_STATE | MACH_EXCEPTION_CODES, STATE_FLAVOR);
	assert(result == KERN_SUCCESS);

	result = mach_port_mod_refs(mach_task_self(), m_port, MACH_PORT_RIGHT_SEND, -1);
	assert(result == KERN_SUCCESS);
#endif
}

void CEeExecutor::Reset()
{
	SetMemoryProtected(m_ram, PS2::EE_RAM_SIZE, false);
	m_cachedBlocks.clear();
#if defined(__EMSCRIPTEN__)
	m_batchPending.clear();
	m_regions.clear();
#endif
	m_blockFpRoundingModes.clear();
	m_idleLoopBlocks.clear();
	CGenericMipsExecutor::Reset();
}

void CEeExecutor::ClearActiveBlocksInRange(uint32 start, uint32 end, bool executing)
{
#if defined(__EMSCRIPTEN__)
	//Region functions on the stack stop at their next block exit; regions with blocks that may be
	//deleted are dissolved so none of their stale blocks can be reached again
	m_context.m_blockGeneration++;
	DissolveRegions(start, end);
#endif
	uint32 rangeSize = end - start;
	SetMemoryProtected(m_ram + start, rangeSize, false);
	CGenericMipsExecutor::ClearActiveBlocksInRange(start, end, executing);
}

BasicBlockPtr CEeExecutor::BlockFactory(CMIPS& context, uint32 start, uint32 end)
{
	uint32 blockSize = (end - start) + 4;

	//Kernel area is below 0x100000 and isn't protected. Some games will write code in there
	//but it is safe to assume that it won't change (code writes some data just besides itself
	//so it keeps generating exceptions, making the game slower)
	if(start >= 0x100000 && start < PS2::EE_RAM_SIZE)
	{
		SetMemoryProtected(m_ram + start, blockSize, true);
	}

	auto blockMemory = reinterpret_cast<uint32*>(alloca(blockSize));
	for(uint32 address = start; address <= end; address += 4)
	{
		uint32 index = (address - start) / 4;
		uint32 opcode = m_context.m_pMemoryMap->GetInstruction(address);
		blockMemory[index] = opcode;
	}

	auto xxHash = XXH3_128bits(blockMemory, blockSize);
	uint128 hash;
	memcpy(&hash, &xxHash, sizeof(xxHash));
	static_assert(sizeof(hash) == sizeof(xxHash));
	auto blockKey = std::make_pair(hash, blockSize);

	bool hasBreakpoint = m_context.HasBreakpointInRange(start, end);

	std::optional<Jitter::CJitter::ROUNDINGMODE> blockFpRoundingModeOverride;
	if(auto blockFpRoundingModeIterator = m_blockFpRoundingModes.find(start);
	   blockFpRoundingModeIterator != std::end(m_blockFpRoundingModes))
	{
		blockFpRoundingModeOverride = blockFpRoundingModeIterator->second;
	}

	bool isIdleLoopBlockOverride = false;
	if(auto idleLoopBlockIterator = m_idleLoopBlocks.find(start);
	   idleLoopBlockIterator != std::end(m_idleLoopBlocks))
	{
		const auto& checkBlockKey = idleLoopBlockIterator->second;
		if(!checkBlockKey.has_value() || (checkBlockKey.value() == blockKey))
		{
			isIdleLoopBlockOverride = true;
		}
	}

	bool fpUseAccurateAddSub = (m_blockFpUseAccurateAddSub.count(start) != 0);

	bool isCacheableBlock = !hasBreakpoint && !blockFpRoundingModeOverride.has_value() && !isIdleLoopBlockOverride && !fpUseAccurateAddSub;
	if(isCacheableBlock)
	{
		auto blockIterator = m_cachedBlocks.find(blockKey);
		if(blockIterator != std::end(m_cachedBlocks))
		{
			const auto& basicBlock(blockIterator->second);
			if(basicBlock->GetBeginAddress() == start && basicBlock->GetEndAddress() == end)
			{
				uint32 recycleCount = basicBlock->GetRecycleCount();
				basicBlock->SetRecycleCount(std::min<uint32>(RECYCLE_NOLINK_THRESHOLD, recycleCount + 1));
				return basicBlock;
			}
			else
			{
				auto result = std::make_shared<CEeBasicBlock>(context, start, end, m_blockCategory);
				result->CopyFunctionFrom(basicBlock);
				return result;
			}
		}
	}

	auto result = std::make_shared<CEeBasicBlock>(context, start, end, m_blockCategory);
	if(blockFpRoundingModeOverride.has_value())
	{
		result->SetFpRoundingMode(blockFpRoundingModeOverride.value());
	}
	if(isIdleLoopBlockOverride)
	{
		result->SetIsIdleLoopBlock();
	}
	if(fpUseAccurateAddSub)
	{
		result->AddBlockCompileHints(CMA_EE::COMPILEHINT_FPU_USE_ACCURATE_ADD_SUB);
	}

	result->Compile();
	if(isCacheableBlock)
	{
		m_cachedBlocks.insert(std::make_pair(blockKey, result));
	}
#if defined(__EMSCRIPTEN__)
	QueueForBatch(result);
#endif
	return result;
}

bool CEeExecutor::HandleAccessFault(intptr_t ptr)
{
	ptrdiff_t addr = reinterpret_cast<uint8*>(ptr) - m_ram;
	if(addr >= 0 && addr < PS2::EE_RAM_SIZE)
	{
		addr &= ~(m_pageSize - 1);
		ClearActiveBlocksInRange(addr, addr + m_pageSize, true);
		return true;
	}
	return false;
}

void CEeExecutor::SetMemoryProtected(void* addr, size_t size, bool protect)
{
#ifdef DISABLE_PROTECTION
	return;
#endif

#if defined(_WIN32)
	DWORD oldProtect = 0;
	BOOL result = VirtualProtect(addr, size, protect ? PAGE_READONLY : PAGE_READWRITE, &oldProtect);
	assert(result == TRUE);
#elif defined(__EMSCRIPTEN__)
	{
		ptrdiff_t offset = reinterpret_cast<uint8*>(addr) - m_ram;
		assert(offset >= 0 && (offset + size) <= PS2::EE_RAM_SIZE);
		if(size == 0) return;
		uint32 firstPage = static_cast<uint32>(offset) / MIPS_PAGE_SIZE;
		uint32 lastPage = static_cast<uint32>(offset + size - 1) / MIPS_PAGE_SIZE;
		static const uint32 mirrorBases[] = {0x00000000, 0x20000000, 0x30000000, 0x80000000};
		for(uint32 page = firstPage; page <= lastPage; page++)
		{
			if(m_protectedPages[page] == protect) continue;
			m_protectedPages[page] = protect;
			void* pagePtr = protect ? nullptr : (m_ram + (page * MIPS_PAGE_SIZE));
			for(uint32 mirrorBase : mirrorBases)
			{
				m_context.m_pageLookupWrite[(mirrorBase / MIPS_PAGE_SIZE) + page] = pagePtr;
			}
		}
	}
#elif defined(__unix__) || defined(__ANDROID__) || defined(__APPLE__)
	uintptr_t addrValue = reinterpret_cast<uintptr_t>(addr) & ~(m_pageSize - 1);
	addr = reinterpret_cast<void*>(addrValue);
	size = size + (m_pageSize - 1) & ~(m_pageSize - 1);
	int result = mprotect(addr, size, protect ? PROT_READ : PROT_READ | PROT_WRITE);
	assert(result >= 0);
#else
	assert(false);
#endif
}

#if defined(_WIN32)

LONG WINAPI CEeExecutor::HandleException(_EXCEPTION_POINTERS* exceptionInfo)
{
	return g_eeExecutor->HandleExceptionInternal(exceptionInfo);
}

LONG CEeExecutor::HandleExceptionInternal(_EXCEPTION_POINTERS* exceptionInfo)
{
	auto exceptionRecord = exceptionInfo->ExceptionRecord;
	if(exceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
	{
		if(HandleAccessFault(exceptionRecord->ExceptionInformation[1]))
		{
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__unix__) || defined(__ANDROID__)

void CEeExecutor::HandleException(int sigId, siginfo_t* sigInfo, void* baseContext)
{
	g_eeExecutor->HandleExceptionInternal(sigId, sigInfo, baseContext);
}

void CEeExecutor::HandleExceptionInternal(int sigId, siginfo_t* sigInfo, void* baseContext)
{
	if(sigId != SIGSEGV) return;
	if(HandleAccessFault(reinterpret_cast<intptr_t>(sigInfo->si_addr)))
	{
		return;
	}
	signal(SIGSEGV, SIG_DFL);
}

#elif defined(__APPLE__) && !TARGET_OS_TV

void CEeExecutor::HandlerThreadProc()
{
#pragma pack(push, 4)
	struct INPUT_MESSAGE
	{
		mach_msg_header_t head;
		NDR_record_t ndr;
		exception_type_t exception;
		mach_msg_type_number_t codeCount;
		intptr_t code[2];
		int flavor;
		mach_msg_type_number_t stateCount;
		natural_t state[STATE_FLAVOR_COUNT];
		mach_msg_trailer_t trailer;
	};
	struct OUTPUT_MESSAGE
	{
		mach_msg_header_t head;
		NDR_record_t ndr;
		kern_return_t result;
		int flavor;
		mach_msg_type_number_t stateCount;
		natural_t state[STATE_FLAVOR_COUNT];
	};
#pragma pack(pop)

	while(m_running)
	{
		kern_return_t result = KERN_SUCCESS;

		INPUT_MESSAGE inMsg;
		result = mach_msg(&inMsg.head, MACH_RCV_MSG | MACH_RCV_LARGE | MACH_RCV_TIMEOUT, 0, sizeof(inMsg), m_port, 1000, MACH_PORT_NULL);
		if(result == MACH_RCV_TIMED_OUT) continue;
		assert(result == KERN_SUCCESS);

		assert(inMsg.head.msgh_id == 2406); //MACH_EXCEPTION_RAISE_RPC
		assert(inMsg.flavor == STATE_FLAVOR);
		assert(inMsg.stateCount == STATE_FLAVOR_COUNT);

		bool success = HandleAccessFault(inMsg.code[1]);

		OUTPUT_MESSAGE outMsg;
		outMsg.head.msgh_bits = MACH_MSGH_BITS(MACH_MSGH_BITS_REMOTE(inMsg.head.msgh_bits), 0);
		outMsg.head.msgh_remote_port = inMsg.head.msgh_remote_port;
		outMsg.head.msgh_local_port = MACH_PORT_NULL;
		outMsg.head.msgh_id = inMsg.head.msgh_id + 100;
		outMsg.head.msgh_size = sizeof(outMsg);
		outMsg.ndr = inMsg.ndr;

		if(success)
		{
			outMsg.result = KERN_SUCCESS;
			outMsg.flavor = STATE_FLAVOR;
			outMsg.stateCount = STATE_FLAVOR_COUNT;
			memcpy(outMsg.state, inMsg.state, STATE_FLAVOR_COUNT * sizeof(natural_t));
		}
		else
		{
			outMsg.result = KERN_FAILURE;
			outMsg.flavor = 0;
			outMsg.stateCount = 0;
		}

		result = mach_msg(&outMsg.head, MACH_SEND_MSG | MACH_RCV_LARGE, sizeof(outMsg), 0, MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
		assert(result == KERN_SUCCESS);
	}
}

#endif
