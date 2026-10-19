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
#include <array>
#include <unordered_map>

#include "../GenericMipsExecutor.h"

//Tier-2 region counters (g_eeRegionStats, Module.eeRegionStats)
enum EE_REGION_STAT
{
	EE_REGION_CREATED,           //regions created (including recreations after invalidation)
	EE_REGION_PARTS,             //their total parts
	EE_REGION_ENTRIES,           //distinct entries that got a region
	EE_REGION_REBUILT_ENTRIES,   //entries whose region was created more than once
	EE_REGION_REBUILT8_ENTRIES,  //entries whose region was created more than 8 times
	EE_REGION_MAX_BUILDS,        //most creations for one entry
	EE_REGION_COMPILED,          //regions compiled (tier-1 cache miss: block cache lookup + module)
	EE_REGION_REUSED,            //regions recreated from m_cachedBlocks (no compile)
	EE_REGION_PAGE_CLEARS,       //protected pages cleared by a write (all blocks)
	EE_REGION_STAT_COUNT,
};

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

#ifdef PROFILE
	//Tier-2 study (notes/tier2-m0.md): counts of block -> next block transitions by exit kind, plus a
	//trace of the first 'traceCapacity' blocks run. Stop returns them with the code pages (see the format there).
	void StartEdgeProfile(uint32 traceCapacity);
	std::vector<uint8> StopEdgeProfile();
#endif
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
	//Tier-2 EE regions (g_eeRegions): plans by entry address, each with a hash of the words of its parts,
	//so recreating a region after invalidation (same code) skips planning
	struct REGION_PLAN
	{
		uint64 hash = 0;
		std::vector<CBasicBlock::SUPERBLOCK_PART> parts;
	};
	void PartitionFunction(uint32) override;
	std::vector<CBasicBlock::SUPERBLOCK_PART> PlanRegion(uint32) const;
	uint64 HashRegionWords(const std::vector<CBasicBlock::SUPERBLOCK_PART>&) const;
	std::unordered_map<uint32, std::vector<REGION_PLAN>> m_regionPlans;
	std::vector<CBasicBlock::SUPERBLOCK_PART> m_pendingRegionParts;
	//Hot entries: a dispatch address sampled g_eeRegionHotSamples times gets a region
	//ponytail: fixed average period; counts restart only on Reset/state load (no decay)
	enum
	{
		REGION_SAMPLE_PERIOD = 64,
	};
	static constexpr uint32 REGION_SAMPLE_SEED = 0x9E3779B9;
	void SampleRegionEntry(uint32);
	void ResetRegionSampling();
	uint32 m_regionSampleCountdown = REGION_SAMPLE_PERIOD;
	uint32 m_regionSampleRng = REGION_SAMPLE_SEED;
	//Sample counts by word address (an unordered_map cost ~1 ms/vblank in GTA)
	//ponytail: aliases addresses 4 MB apart (the first to reach the threshold gets the region); tag it if that matters
	std::vector<uint16> m_regionEntrySamples = std::vector<uint16>(0x100000);
	std::set<uint32> m_regionHotEntries;
	std::unordered_map<uint32, uint32> m_regionBuildCounts;
	//Regions removed by an invalidation while they may be running, kept alive until Execute returns
	std::vector<BasicBlockPtr> m_retiredRegions;

#ifdef PROFILE
	void ProfileBlockExecuted(uint32) override;
	bool m_edgeProfiling = false;
	uint32 m_edgePrev = MIPS_INVALID_PC;
	uint32 m_edgePrevKind = 0;
	//(src << 32 | dst) -> count per exit kind of src: 0 no exception, 1 cycle quota only, 2 other exception
	std::unordered_map<uint64, std::array<uint32, 3>> m_edges;
	//Block start | exit kind, in execution order
	std::vector<uint32> m_edgeTrace;
	size_t m_edgeTraceSize = 0;
#endif
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
