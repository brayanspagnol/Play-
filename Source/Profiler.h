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

	//Cheap zone switch for very hot paths (work thread only, no stack bookkeeping)
	int32 SwapCurrentZone(int32 zone)
	{
		return m_currentZone.exchange(zone, std::memory_order_relaxed);
	}

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
	PCNT_MAX
};
extern std::atomic<uint32> g_profileCounters[PCNT_MAX];
extern const char* const g_profileCounterNames[PCNT_MAX];
#ifdef PROFILE
#define PROFILE_COUNT(counter, value) g_profileCounters[counter].fetch_add(value, std::memory_order_relaxed)
#else
#define PROFILE_COUNT(counter, value)
#endif

class CProfilerZone
{
public:
	CProfilerZone(CProfiler::ZoneHandle);
	~CProfilerZone();

private:
	bool m_active = false;
};
