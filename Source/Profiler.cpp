#include "Profiler.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>

std::atomic<uint32> g_profileCounters[PCNT_MAX];
const char* const g_profileCounterNames[PCNT_MAX] =
    {
        "vif1Dma", "vif1DmaQw", "mscal", "unpack", "mpg", "direct", "flush", "ibit", "mskpath3",
        "vif1RegRead", "vif1FifoRead", "path3Dma", "path3DmaQw", "gifStatRead", "vu1MemEe", "vu1Cmsar",
        "xgkick", "vu1Sync", "eeBlocks", "eeInsns", "eeInval", "eeJit"};

CProfiler::ZoneHandle CProfiler::RegisterZone(const char* name)
{
#ifdef PROFILE
	static std::mutex registerMutex;
	std::lock_guard<std::mutex> registerLock(registerMutex);
	for(unsigned int i = 0; i < m_zoneNames.size(); i++)
	{
		if(m_zoneNames[i] == name) return i;
	}
	assert(m_zoneNames.size() < MAX_ZONES);
	m_zoneNames.push_back(name);
	return static_cast<CProfiler::ZoneHandle>(m_zoneNames.size() - 1);
#else
	return 0;
#endif
}

void CProfiler::CountCurrentZone()
{
}

void CProfiler::EnterZone(ZoneHandle zoneHandle)
{
	bool second = IsSecondWorkThread();
	auto& zoneStack = second ? m_secondZoneStack : m_zoneStack;
	zoneStack.push(zoneHandle);
	(second ? m_secondCurrentZone : m_currentZone).store(zoneHandle, std::memory_order_relaxed);
}

void CProfiler::ExitZone()
{
	bool second = IsSecondWorkThread();
	auto& zoneStack = second ? m_secondZoneStack : m_zoneStack;
	zoneStack.pop();
	(second ? m_secondCurrentZone : m_currentZone).store(zoneStack.empty() ? -1 : zoneStack.top(), std::memory_order_relaxed);
}

CProfiler::ZoneArray CProfiler::GetStats() const
{
	ZoneArray result;
	for(unsigned int i = 0; i < m_zoneNames.size(); i++)
	{
		ZONE zone;
		zone.name = m_zoneNames[i];
		zone.totalTime = m_zoneTimes[i].load(std::memory_order_relaxed);
		result.push_back(zone);
		uint64 secondTime = m_secondZoneTimes[i].load(std::memory_order_relaxed);
		if(secondTime != 0)
		{
			result.push_back({"VU1:" + m_zoneNames[i], secondTime});
		}
	}
	return result;
}

void CProfiler::Reset()
{
	for(auto& zoneTime : m_zoneTimes)
	{
		zoneTime.store(0, std::memory_order_relaxed);
	}
	for(auto& zoneTime : m_secondZoneTimes)
	{
		zoneTime.store(0, std::memory_order_relaxed);
	}
}

void CProfiler::SetWorkThread()
{
	m_workThreadId = std::this_thread::get_id();
#ifdef PROFILE
	if(!m_samplerThread.joinable())
	{
		m_samplerThread = std::thread([this]() { SamplerThreadProc(); });
		m_samplerThread.detach();
	}
#endif
}

void CProfiler::SetSecondWorkThread()
{
	m_secondWorkThreadId = std::this_thread::get_id();
}

void CProfiler::SetPcSampler(unsigned int slot, ZoneHandle zone, const uint32* pc)
{
	assert(slot < MAX_PC_SAMPLERS);
	m_pcSamplers[slot].pc = pc;
	m_pcSamplers[slot].zone = zone;
}

CProfiler::PcHistogram CProfiler::TakePcHistogram(unsigned int slot)
{
	assert(slot < MAX_PC_SAMPLERS);
	std::lock_guard<std::mutex> lock(m_pcSamplersMutex);
	PcHistogram result;
	std::swap(result, m_pcSamplers[slot].histogram);
	return result;
}

void CProfiler::SamplerThreadProc()
{
	auto lastTime = std::chrono::steady_clock::now();
	while(1)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(500));
		auto thisTime = std::chrono::steady_clock::now();
		uint64 elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(thisTime - lastTime).count();
		lastTime = thisTime;
		int32 secondZone = m_secondCurrentZone.load(std::memory_order_relaxed);
		if(secondZone >= 0 && secondZone < MAX_ZONES)
		{
			m_secondZoneTimes[secondZone].fetch_add(elapsed, std::memory_order_relaxed);
		}
		int32 zone = m_currentZone.load(std::memory_order_relaxed);
		if(zone < 0 || zone >= MAX_ZONES) continue;
		m_zoneTimes[zone].fetch_add(elapsed, std::memory_order_relaxed);
		for(auto& sampler : m_pcSamplers)
		{
			auto pc = sampler.pc.load();
			if(sampler.zone == zone && pc)
			{
				std::lock_guard<std::mutex> lock(m_pcSamplersMutex);
				sampler.histogram[*pc]++;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//CProfilerZone

CProfilerZone::CProfilerZone(CProfiler::ZoneHandle handle)
{
#ifdef PROFILE
	m_active = CProfiler::GetInstance().IsWorkThread() || CProfiler::GetInstance().IsSecondWorkThread();
	if(m_active) CProfiler::GetInstance().EnterZone(handle);
#endif
}

CProfilerZone::~CProfilerZone()
{
#ifdef PROFILE
	if(m_active) CProfiler::GetInstance().ExitZone();
#endif
}

void CProfiler::Log(const char* format, ...)
{
	char buffer[2048];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	std::lock_guard<std::mutex> lock(m_logMutex);
	if(m_log.size() < 65536) m_log += buffer;
}

std::string CProfiler::TakeLog()
{
	std::lock_guard<std::mutex> lock(m_logMutex);
	std::string result;
	std::swap(result, m_log);
	return result;
}
