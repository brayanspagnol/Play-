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

private:
	void SamplerThreadProc();

	std::vector<std::string> m_zoneNames;
	std::atomic<uint64> m_zoneTimes[MAX_ZONES] = {};
	std::stack<ZoneHandle> m_zoneStack;
	std::atomic<int32> m_currentZone = -1;

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

class CProfilerZone
{
public:
	CProfilerZone(CProfiler::ZoneHandle);
	~CProfilerZone();

private:
	bool m_active = false;
};
