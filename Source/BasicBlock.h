#pragma once

#include "MIPS.h"
#include "MemoryFunction.h"
#include "MemStream.h"
#include "Jitter_Interpreter.h"
#include "BlockCache.h"
#ifdef AOT_BUILD_CACHE
#include "StdStream.h"
#include <mutex>
#endif

enum BLOCK_CATEGORY : uint32
{
	BLOCK_CATEGORY_UNKNOWN = 0,
	BLOCK_CATEGORY_PS2_EE = 0x65650000,
	BLOCK_CATEGORY_PS2_IOP = 0x696F7000,
	BLOCK_CATEGORY_PS2_VU = 0x76750000,
	BLOCK_CATEGORY_PSP = 0x50535000,
};

#pragma pack(push, 1)
struct AOT_BLOCK_KEY
{
	BLOCK_CATEGORY category;
	uint128 hash;
	uint32 size;

	bool operator<(const AOT_BLOCK_KEY& k2) const
	{
		const auto& k1 = (*this);
		return std::tie(k1.category, k1.hash, k1.size) <
		       std::tie(k2.category, k2.hash, k2.size);
	}
};
#pragma pack(pop)
static_assert(sizeof(AOT_BLOCK_KEY) == 0x18, "AOT_BLOCK_KEY must be 24 bytes long.");

namespace Jitter
{
	class CJitter;
};

namespace AsyncCompile
{
	struct JOB;
};

extern "C"
{
	void EmptyBlockHandler(CMIPS*);
	void NextBlockTrampoline(CMIPS*);
	void BranchBlockTrampoline(CMIPS*);
}

enum LINK_SLOT
{
	LINK_SLOT_NEXT,
	LINK_SLOT_BRANCH,
	LINK_SLOT_MAX,
};

//Block outgoing link
struct BLOCK_OUT_LINK
{
	LINK_SLOT slot;    //slot used in the source block
	uint32 srcAddress; //address of source block
	bool live;         //live if linked to another block, otherwise, link is pending
};

//Block outgoing links map (key: target link address, value: struct describing link status)
typedef std::multimap<uint32, BLOCK_OUT_LINK> BlockOutLinkMap;

//When block linking is used, each basic block will maintain pointers
//to their outgoing link definitions inside the map
typedef BlockOutLinkMap::iterator BlockOutLinkPointer;

class CBasicBlock : public std::enable_shared_from_this<CBasicBlock>
{
public:
	CBasicBlock(CMIPS&, uint32 = MIPS_INVALID_PC, uint32 = MIPS_INVALID_PC, BLOCK_CATEGORY = BLOCK_CATEGORY_UNKNOWN);
	virtual ~CBasicBlock() = default;
	void Execute();
	void Compile();
	virtual void CompileRange(CMipsJitter*);

	void AddBlockCompileHints(uint32);
	//Extra codegen-affecting state (besides instruction words and compile hints) for the persistent block cache key.
	void AddCacheVariant(uint32);

	uint32 GetBeginAddress() const;
	uint32 GetEndAddress() const;
	bool IsCompiled() const;
	bool IsEmpty() const;

	uint32 GetRecycleCount() const;
	void SetRecycleCount(uint32);

	bool HasLinkSlot(LINK_SLOT) const;
	BlockOutLinkPointer GetOutLink(LINK_SLOT) const;
	void SetOutLink(LINK_SLOT, BlockOutLinkPointer);

	void LinkBlock(LINK_SLOT, CBasicBlock*);
	void UnlinkBlock(LINK_SLOT);

#ifdef AOT_BUILD_CACHE
	static void SetAotBlockOutputStream(Framework::CStdStream*);
#endif

	void CopyFunctionFrom(const std::shared_ptr<CBasicBlock>& basicBlock);

#if defined(__EMSCRIPTEN__) && !defined(AOT_USE_CACHE)
	//Module batching: wasm module bytes of the compiled function, kept until the block is batched
	std::vector<uint8>& GetBatchModule()
	{
		return m_batchModule;
	}
	void ReplaceWasmFunction(int fctId)
	{
		m_function.ReplaceWasmFunction(fctId);
		m_chainFunctionIndex = GetFunctionIndex();
		m_batchModule = std::vector<uint8>();
	}
	//Async compile (AsyncCompile.h): the job of a block that still runs in the interpreter (null otherwise)
	const std::shared_ptr<AsyncCompile::JOB>& GetAsyncJob() const
	{
		return m_asyncJob;
	}
	//Takes the compiled function of a block that ran in the interpreter
	void SetCompiledFunction(CMemoryFunction&&);
	//Moves the pending blocks' own modules into one batch module (fewer, larger modules: better code
	//locality in the browser's JIT). Returns the blocks that now call a batched function.
	static std::vector<std::shared_ptr<CBasicBlock>> BatchModules(std::vector<std::weak_ptr<CBasicBlock>>&);
	//Offset of m_chainFunctionIndex (index of the compiled function, read by VU block chaining)
	static size_t GetChainFunctionIndexOffset();
	//Index of the compiled function in the wasm function table
	uint32 GetFunctionIndex() const
	{
		return static_cast<uint32>(reinterpret_cast<uintptr_t>(m_function.GetCode()));
	}
#endif

protected:
	uint32 m_begin;
	uint32 m_end;
	BLOCK_CATEGORY m_category;
	CMIPS& m_context;

	uint32 m_blockCompileHints = 0;
	uint32 m_cacheVariant = 0;
	//Cleared by CompileRange when the generated code depends on more than the block's own words.
	bool m_cacheable = true;

	void CompileToStream(Framework::CMemStream&, bool contextPromotion);
	//CompileToStream up to the end of the front end (statements in the jitter, not compiled yet)
	void BeginCompile(CMipsJitter*, bool contextPromotion);
#if defined(__EMSCRIPTEN__)
	std::vector<uint8> m_batchModule;
#endif
	bool UsesContextPromotion() const;
	bool UsesModuleBatching() const;
	bool MaySetDelayedJump() const;
	virtual bool IsSuperBlock() const
	{
		return false;
	}
#if defined(__EMSCRIPTEN__)
	//Async compile: run from the Jitter's statements until the compiled function arrives
	Jitter::InterpreterProgramPtr m_interpreterProgram;
	std::shared_ptr<AsyncCompile::JOB> m_asyncJob;
	//Front end on this thread, the rest on the helper thread; false: not possible, compile as usual
	bool TryCompileAsync(const BlockCache::KEY*);
	void SubmitAsyncJob(CMipsJitter*, bool hot);
	uint32 m_interpretCount = 0;
	BlockCache::KEY m_asyncCacheKey = {};
	bool m_asyncAddToCache = false;
	virtual void ExecuteDiff();
	CMemoryFunction m_refFunction;
	bool m_diffExecute = false;
#endif
	//Set while compiling: true with the optimized VU codegen (context promotion and friends)
	bool m_compileOptimized = false;
#if defined(__EMSCRIPTEN__)

public:
	//Size of the VU data memory, saved/restored by the differential test
	uint32 m_diffDataSize = 0;
	//Compile with the reference (unoptimized) codegen, used by the differential test
	bool m_forceReference = false;

protected:
#endif

	virtual void CompileProlog(CMipsJitter*);
	virtual void CompileEpilog(CMipsJitter*, bool);
	//Block chaining (WebAssembly): the epilog tail calls the next block when there is no exception
	bool UsesBlockChaining() const;

private:
	void HandleExternalFunctionReference(uintptr_t, uint32, Jitter::CCodeGen::SYMBOL_REF_TYPE);

#ifdef DEBUGGER_INCLUDED
	bool HasBreakpoint() const;
	static uint32 BreakpointFilter(CMIPS*);
	static void BreakpointHandler(CMIPS*);
#endif

#ifdef AOT_BUILD_CACHE
	static Framework::CStdStream* m_aotBlockOutputStream;
	static std::mutex m_aotBlockOutputStreamMutex;
#endif

protected:
#ifndef AOT_USE_CACHE
	CMemoryFunction m_function;
#else
	void (*m_function)(void*);
#endif

private:
	uint32 m_chainFunctionIndex = 0;
	uint32 m_recycleCount = 0;
	BlockOutLinkPointer m_outLinks[LINK_SLOT_MAX];
	uint32 m_linkBlockTrampolineOffset[LINK_SLOT_MAX];
#ifdef _DEBUG
	CBasicBlock* m_linkBlock[LINK_SLOT_MAX];
#endif
};
