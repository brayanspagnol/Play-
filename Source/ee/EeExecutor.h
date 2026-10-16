#pragma once

#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <thread>
#elif defined(__unix__)
#include <signal.h>
#endif

#include <optional>

#include "../GenericMipsExecutor.h"

class CEeExecutor : public CGenericMipsExecutor<BlockLookupTwoWay>
{
public:
	using CachedBlockKey = std::pair<uint128, uint32>;
	using IdleLoopBlockMap = std::map<uint32, std::optional<CachedBlockKey>>;
	using BlockFpUseAccurateAddSubSet = std::set<uint32>;
	using BlockFpRoundingModeMap = std::map<uint32, Jitter::CJitter::ROUNDINGMODE>;

	CEeExecutor(CMIPS&, uint8*);
	virtual ~CEeExecutor() = default;

	void SetBlockFpRoundingModes(BlockFpRoundingModeMap);
	void SetBlockFpUseAccurateAddSub(BlockFpUseAccurateAddSubSet);
	void SetIdleLoopBlocks(IdleLoopBlockMap);

	void AddExceptionHandler();
	void RemoveExceptionHandler();

	void AttachExceptionHandlerToThread();

	void Reset() override;
	void ClearActiveBlocksInRange(uint32, uint32, bool) override;

	BasicBlockPtr BlockFactory(CMIPS&, uint32, uint32) override;

#if defined(__EMSCRIPTEN__)
	int Execute(int) override;

	//Software write protection of code pages (no mprotect on WebAssembly): JIT stores to
	//protected pages take the slow path (null write page table entry) and invalidate the
	//blocks of that page, like the access fault handler does on desktop platforms.
	//C++ code writing EE RAM directly (DMA, HLE IOP modules) must call NotifyRamWritten.
	static void NotifyRamWritten(uint32 address, uint32 size);

	//Called before JIT stores outside of RAM/scratchpad (used to sync with the VU1 thread)
	std::function<void(uint32)> m_ioWriteHandler;
#else
	static void NotifyRamWritten(uint32, uint32)
	{
	}
#endif

private:
	typedef std::map<CachedBlockKey, BasicBlockPtr> CachedBlockMap;
	CachedBlockMap m_cachedBlocks;

	IdleLoopBlockMap m_idleLoopBlocks;
	BlockFpUseAccurateAddSubSet m_blockFpUseAccurateAddSub;
	BlockFpRoundingModeMap m_blockFpRoundingModes;

	uint8* m_ram = nullptr;
	size_t m_pageSize = 0;

#if defined(__EMSCRIPTEN__)
	static void WriteWatchHandler(CMIPS*, uint32, uint32);
	void InvalidateWrittenRange(uint32, uint32, bool);
	//Module batching: newly compiled blocks wait here until g_eeBatchSize of them can be moved
	//into one wasm module (fewer, larger modules: better code locality in the browser's JIT)
	void QueueForBatch(const BasicBlockPtr&);
	void FlushBatch();
	std::vector<std::weak_ptr<CBasicBlock>> m_batchPending;

	std::vector<uint8> m_protectedPages;
	bool m_executing = false;
#endif

	bool HandleAccessFault(intptr_t);
	void SetMemoryProtected(void*, size_t, bool);

#if defined(_WIN32)
	static LONG CALLBACK HandleException(_EXCEPTION_POINTERS*);
	LONG HandleExceptionInternal(_EXCEPTION_POINTERS*);

	LPVOID m_handler = NULL;
#elif defined(__unix__) || defined(__ANDROID__)
	static void HandleException(int, siginfo_t*, void*);
	void HandleExceptionInternal(int, siginfo_t*, void*);
#elif defined(__APPLE__) && !TARGET_OS_TV
	void HandlerThreadProc();

	mach_port_t m_port = MACH_PORT_NULL;
	std::thread m_handlerThread;
	std::atomic<bool> m_running = false;
#endif
};
