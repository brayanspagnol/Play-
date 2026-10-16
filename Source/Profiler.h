#pragma once

#include <string>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "Singleton.h"
#include "Types.h"

//Sampling profiler: the work thread only publishes its current zone (two stores per zone
//enter/exit), a sampler thread attributes elapsed time to that zone every ~0.5 ms.
//Timing every zone transition was too expensive on WebAssembly (clock reads are JS calls).
class CProfiler : public CSingleton<CProfiler>
{
public:
	typedef uint32 ZoneHandle;

	enum
	{
		MAX_ZONES = 64,
		MAX_PC_SAMPLERS = 2,
	};

	struct ZONE
	{
		std::string name;
		uint64 totalTime = 0;
	};

	typedef std::vector<ZONE> ZoneArray;
	typedef std::unordered_map<uint32, uint32> PcHistogram;

	virtual ~CProfiler() = default;

	ZoneHandle RegisterZone(const char*);

	void CountCurrentZone();

	void EnterZone(ZoneHandle);
	void ExitZone();

	ZoneArray GetStats() const;
	void Reset();

	void SetWorkThread();
	//A second thread with its own zone accounting (VU1 thread), reported as "VU1:<zone>"
	void SetSecondWorkThread();

	//While the work thread is in 'zone', the sampler also records '*pc' in histogram 'slot'.
	void SetPcSampler(unsigned int slot, ZoneHandle zone, const uint32* pc);
	PcHistogram TakePcHistogram(unsigned int slot);

	//Free-form debug log (any thread), drained by the frontend
	void Log(const char*, ...);
	std::string TakeLog();

	//Cheap zone switch for very hot paths (no stack bookkeeping). Switches the calling thread's zone
	//(work thread or second work thread; no-op elsewhere). Only the owner writes it: plain load + store.
	static int32 SwapCurrentZone(int32 zone)
	{
		auto current = t_currentZone;
		if(!current) return -1;
		int32 previous = current->load(std::memory_order_relaxed);
		current->store(zone, std::memory_order_relaxed);
		return previous;
	}

	//Raw per-zone times (ns) since the last Reset, for the VM thread and the second work thread
	void GetZoneTimes(uint64* times, uint64* secondTimes, unsigned int count) const;
	std::vector<std::string> GetZoneNames() const;

	bool IsWorkThread() const
	{
		return std::this_thread::get_id() == m_workThreadId;
	}

	bool IsSecondWorkThread() const
	{
		return std::this_thread::get_id() == m_secondWorkThreadId;
	}

private:
	void SamplerThreadProc();

	std::vector<std::string> m_zoneNames;
	std::atomic<uint64> m_zoneTimes[MAX_ZONES] = {};
	std::stack<ZoneHandle> m_zoneStack;
	std::atomic<int32> m_currentZone = -1;
	std::atomic<uint64> m_secondZoneTimes[MAX_ZONES] = {};
	std::stack<ZoneHandle> m_secondZoneStack;
	std::atomic<int32> m_secondCurrentZone = -1;
	std::thread::id m_secondWorkThreadId;

	struct PC_SAMPLER
	{
		std::atomic<int32> zone = -1;
		std::atomic<const uint32*> pc = nullptr;
		PcHistogram histogram;
	};
	PC_SAMPLER m_pcSamplers[MAX_PC_SAMPLERS];
	std::mutex m_pcSamplersMutex;

	std::mutex m_logMutex;
	std::string m_log;

	std::thread::id m_workThreadId;
	std::thread m_samplerThread;
	static thread_local std::atomic<int32>* t_currentZone;
};

//Event counters for profiling builds (any thread), reported by getProfile() as "counters".
enum PROFILE_COUNTER
{
	PCNT_VIF1_DMA,
	PCNT_VIF1_DMA_QW,
	PCNT_VIF1_MSCAL,
	PCNT_VIF1_UNPACK,
	PCNT_VIF1_MPG,
	PCNT_VIF1_DIRECT,
	PCNT_VIF1_FLUSH,
	PCNT_VIF1_IBIT,
	PCNT_VIF1_MSKPATH3,
	PCNT_VIF1_REGREAD,
	PCNT_VIF1_FIFOREAD,
	PCNT_PATH3_DMA,
	PCNT_PATH3_DMA_QW,
	PCNT_GIF_STAT_READ,
	PCNT_VU1_MEM_EE,
	PCNT_VU1_CMSAR,
	PCNT_VU1_XGKICK,
	PCNT_VU1_SYNC,
	PCNT_EE_BLOCK,
	PCNT_EE_INSN,
	PCNT_EE_INVAL,
	PCNT_EE_JIT,
	PCNT_MAX
};
extern std::atomic<uint32> g_profileCounters[PCNT_MAX];
extern const char* const g_profileCounterNames[PCNT_MAX];
#ifdef PROFILE
#define PROFILE_COUNT(counter, value) g_profileCounters[counter].fetch_add(value, std::memory_order_relaxed)
#else
#define PROFILE_COUNT(counter, value)
#endif

//Per-vblank timeline (all builds, see getTimeline in ui_js/Main.cpp): wall time spent waiting or compiling,
//and event counts, accumulated by the threads that do it and read at every vblank.
enum TIMELINE_TIMER
{
	TLT_LIMITER,    //VM thread: frame limiter sleep
	TLT_VU1SYNC,    //VM thread: waiting for the VU1 thread (CVu1Thread::Sync)
	TLT_RINGFULL,   //VM thread: VU1 ring full
	TLT_GSWAIT,     //VM thread: waiting for the GS thread to complete a call
	TLT_VU1GSWAIT,  //other threads (VU1): waiting for the GS thread
	TLT_DISC,       //VM thread: disc image reads
	TLT_COMPILE,    //VM thread: JIT block compiles (incl. block cache hits)
	TLT_VU1COMPILE, //other threads: JIT block compiles
	TLT_GSBUSY,     //GS thread: processing calls
	TLT_VU1BUSY,    //VU1 thread: processing messages
	TLT_MAX
};
enum TIMELINE_COUNTER
{
	TLC_COMPILES,
	TLC_JITRUNS, //compiles that ran the Jitter (block cache miss)
	TLC_DISCBYTES,
	TLC_DISCXHR, //1 MB blocks fetched by sync XHR
	TLC_MAX
};
extern std::atomic<uint64> g_timelineNs[TLT_MAX];
extern std::atomic<uint32> g_timelineCounts[TLC_MAX];

class CTimelineTimer
{
public:
	CTimelineTimer(TIMELINE_TIMER timer)
	    : m_timer(timer)
	    , m_start(std::chrono::steady_clock::now())
	{
	}
	~CTimelineTimer()
	{
		g_timelineNs[m_timer].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count(), std::memory_order_relaxed);
	}

private:
	TIMELINE_TIMER m_timer;
	std::chrono::steady_clock::time_point m_start;
};

class CProfilerZone
{
public:
	CProfilerZone(CProfiler::ZoneHandle);
	~CProfilerZone();

private:
	bool m_active = false;
};
