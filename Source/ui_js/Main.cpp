#include <cstdio>
#include <emscripten/bind.h>
#include "Ps2VmJs.h"
#include "GSH_OpenGLJs.h"
#include "SH_WebAudio.h"
#include "input/PH_GenericInput.h"
#include "InputProviderEmscripten.h"
#include "ui_shared/StatsManager.h"
#include "DefaultAppConfig.h"
#include "../Js_DiscImageDeviceStream.h"
#include "PS2VM_Preferences.h"
#include "AppConfig.h"
#include "string_format.h"
#include "BlockCache.h"
#include "ee/Vpu.h"
#include "ee/VUShared.h"
#include <future>
#include <mutex>
#include <set>
#include <map>
#include <atomic>
#include <algorithm>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <cstring>
#include "xxhash.h"
#include "Ps2Const.h"
#include "ee/EeExecutor.h"

#ifdef PROFILE
extern std::atomic<uint32> g_vuBlockExecCount;
extern std::atomic<uint32> g_eeSyscallCounts[0x100];
extern std::atomic<uint32> g_eeInterruptCount;
extern std::atomic<uint32> g_eeSlices;
extern std::atomic<uint32> g_eeIdleSlices;
#endif

CPs2VmJs* g_virtualMachine = nullptr;
CGSHandler::NewFrameEvent::Connection g_gsNewFrameConnection;
std::shared_ptr<CInputProviderEmscripten> g_inputProvider;
Framework::CSignal<void()>::Connection g_vmNewFrameConnection;
Framework::CSignal<void()>::Connection g_vmStateLoadedConnection;

//Profiling/stats snapshot, filled on the emulation thread at every vblank, read by getProfile() on the main thread.
struct PROFILE_ACCUM
{
	uint32 vblanks = 0;
	CPS2VM::CPU_UTILISATION_INFO cpu;
	std::map<std::string, uint64> zoneNs;
	std::string eePcs;
	std::string iopPcs;
};
static std::mutex g_profileMutex;
static PROFILE_ACCUM g_profile;
static double g_profileLastTime = 0;

//Increase of a cumulative counter since the previous call with the same 'last'
template <typename T>
static T TakeDelta(const std::atomic<T>& counter, T& last)
{
	T value = counter.load(std::memory_order_relaxed);
	T delta = value - last;
	last = value;
	return delta;
}

//Per-vblank timeline: one row per emulated vblank (wall time, waits, compiles, disc reads, other threads'
//busy time; zone times in profiling builds), kept in a ring and read with getTimeline(from).
namespace Timeline
{
	enum
	{
		RING_SIZE = 16384, //~5 min at 50-60 vblanks/s
	};
	//Base columns (all builds)
	static const char* const g_baseColumns[] = {
	    "t", "wall", "limiter", "vu1sync", "ringfull", "gswait", "vu1gswait", "disc", "compile", "vu1compile",
	    "gsbusy", "vu1busy", "compiles", "jitruns", "discKB", "discxhr", "ringKB", "eeBusy"};
	enum
	{
		BASE_COLUMNS = sizeof(g_baseColumns) / sizeof(g_baseColumns[0]),
#ifdef PROFILE
		ZONE_COLUMNS = CProfiler::MAX_ZONES,
		COUNTER_COLUMNS = PCNT_MAX,
#else
		ZONE_COLUMNS = 0,
		COUNTER_COLUMNS = 0,
#endif
		COLUMNS = BASE_COLUMNS + COUNTER_COLUMNS + (2 * ZONE_COLUMNS),
	};
	static std::mutex g_mutex;
	static std::vector<float> g_rows; //RING_SIZE * COLUMNS
	static uint64 g_count = 0;        //rows written so far
	static double g_start = 0;        //emscripten_get_now() at the first row
	static double g_startDate = 0;    //Date.now() at the first row
	static double g_lastTime = 0;
	static uint64 g_lastNs[TLT_MAX] = {};
	static uint32 g_lastCounts[TLC_MAX] = {};
#ifdef PROFILE
	static uint32 g_lastProfileCounters[PCNT_MAX] = {};
#endif

	//VM thread, at every vblank (before the profiler's zone times are reset)
	static void Record(const CPS2VM::CPU_UTILISATION_INFO& cpu)
	{
		double now = emscripten_get_now();
		float row[COLUMNS] = {};
		if(g_start == 0)
		{
			g_start = now;
			g_startDate = emscripten_date_now();
			g_lastTime = now;
		}
		row[0] = static_cast<float>(now - g_start);
		row[1] = static_cast<float>(now - g_lastTime);
		g_lastTime = now;
		static const TIMELINE_TIMER timers[] = {TLT_LIMITER, TLT_VU1SYNC, TLT_RINGFULL, TLT_GSWAIT, TLT_VU1GSWAIT, TLT_DISC, TLT_COMPILE, TLT_VU1COMPILE, TLT_GSBUSY, TLT_VU1BUSY};
		unsigned int column = 2;
		for(auto timer : timers)
		{
			row[column++] = static_cast<float>(TakeDelta(g_timelineNs[timer], g_lastNs[timer]) / 1e6);
		}
		row[column++] = static_cast<float>(TakeDelta(g_timelineCounts[TLC_COMPILES], g_lastCounts[TLC_COMPILES]));
		row[column++] = static_cast<float>(TakeDelta(g_timelineCounts[TLC_JITRUNS], g_lastCounts[TLC_JITRUNS]));
		row[column++] = static_cast<float>(TakeDelta(g_timelineCounts[TLC_DISCBYTES], g_lastCounts[TLC_DISCBYTES]) / 1024.0);
		row[column++] = static_cast<float>(TakeDelta(g_timelineCounts[TLC_DISCXHR], g_lastCounts[TLC_DISCXHR]));
		row[column++] = static_cast<float>(g_virtualMachine->m_ee->GetVu1PendingBytes() / 1024.0);
		row[column++] = cpu.eeTotalTicks ? static_cast<float>(100.0 * (cpu.eeTotalTicks - cpu.eeIdleTicks) / cpu.eeTotalTicks) : 0;
		assert(column == BASE_COLUMNS);
#ifdef PROFILE
		for(uint32 i = 0; i < PCNT_MAX; i++)
		{
			row[column++] = static_cast<float>(TakeDelta(g_profileCounters[i], g_lastProfileCounters[i]));
		}
		uint64 zoneNs[CProfiler::MAX_ZONES] = {};
		uint64 secondZoneNs[CProfiler::MAX_ZONES] = {};
		CProfiler::GetInstance().GetZoneTimes(zoneNs, secondZoneNs, CProfiler::MAX_ZONES);
		for(uint32 i = 0; i < CProfiler::MAX_ZONES; i++)
		{
			row[column + i] = static_cast<float>(zoneNs[i] / 1e6);
			row[column + CProfiler::MAX_ZONES + i] = static_cast<float>(secondZoneNs[i] / 1e6);
		}
#endif
		std::lock_guard<std::mutex> lock(g_mutex);
		if(g_rows.empty()) g_rows.resize(RING_SIZE * COLUMNS);
		memcpy(g_rows.data() + (g_count % RING_SIZE) * COLUMNS, row, sizeof(row));
		g_count++;
	}

	//JSON: {"cols":[...],"next":<index after the last row>,"t0":<Date.now() of row t=0>,"rows":[[...],...]}
	//with the rows of index >= from that are still in the ring.
	static std::string Get(double fromValue)
	{
		std::vector<float> rows;
		uint64 first = 0, next = 0;
		double startDate = 0;
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			next = g_count;
			first = std::max<uint64>(static_cast<uint64>(std::max(fromValue, 0.0)), (g_count > RING_SIZE) ? (g_count - RING_SIZE) : 0);
			first = std::min(first, next);
			rows.reserve((next - first) * COLUMNS);
			for(uint64 i = first; i < next; i++)
			{
				const float* row = g_rows.data() + (i % RING_SIZE) * COLUMNS;
				rows.insert(rows.end(), row, row + COLUMNS);
			}
			startDate = g_startDate;
		}
		std::vector<std::string> columns(g_baseColumns, g_baseColumns + BASE_COLUMNS);
		unsigned int usedColumns = BASE_COLUMNS;
#ifdef PROFILE
		for(uint32 i = 0; i < PCNT_MAX; i++) columns.push_back(std::string("n:") + g_profileCounterNames[i]);
		//Every zone slot, so that rows fetched at different times line up (zones registered later are "#<slot>" until then)
		auto zoneNames = CProfiler::GetInstance().GetZoneNames();
		for(const char* prefix : {"z:", "v:"})
		{
			for(uint32 i = 0; i < CProfiler::MAX_ZONES; i++)
			{
				columns.push_back(prefix + ((i < zoneNames.size()) ? zoneNames[i] : string_format("#%u", i)));
			}
		}
		usedColumns = COLUMNS;
#endif
		std::string result = "{\"cols\":[";
		for(size_t i = 0; i < columns.size(); i++)
		{
			result += string_format("%s\"%s\"", i ? "," : "", columns[i].c_str());
		}
		result += string_format("],\"first\":%llu,\"next\":%llu,\"t0\":%.0f,\"rows\":[", static_cast<unsigned long long>(first), static_cast<unsigned long long>(next), startDate);
		auto appendValue = [&](float value, bool comma) {
			if(comma) result += ',';
			if(value == 0)
				result += '0';
			else
				result += string_format("%.2f", value);
		};
		for(uint64 r = 0; r < (next - first); r++)
		{
			const float* row = rows.data() + r * COLUMNS;
			result += r ? ",[" : "[";
			for(unsigned int c = 0; c < usedColumns; c++) appendValue(row[c], c != 0);
			result += ']';
		}
		result += "]}";
		return result;
	}
}

std::string getTimeline(double from)
{
	return Timeline::Get(from);
}

#ifdef PROFILE
static std::string TopPcs(CProfiler::PcHistogram histogram)
{
	std::vector<std::pair<uint32, uint32>> items(histogram.begin(), histogram.end());
	size_t count = std::min<size_t>(items.size(), 8);
	std::partial_sort(items.begin(), items.begin() + count, items.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	uint64 total = 0;
	for(const auto& item : items) total += item.second;
	std::string result;
	for(size_t i = 0; i < count; i++)
	{
		result += string_format("%s[\"%08X\",%.1f]", i ? "," : "", items[i].first, total ? 100.0 * items[i].second / total : 0.0);
	}
	return result;
}
#endif

//Differential tests: hashes of the EE/IOP state and RAM at each of the first N vblanks after the next
//state load (deterministic points in emulated time), see eeHashAfterLoad/eeHashResult.
static std::atomic<int> g_hashRequest = 0;
static int g_hashRemaining = 0;
static std::mutex g_hashMutex;
static std::string g_hashResult;
//Same points: VU0/VU1 state, data and micro memory (VU code changes rarely show in EE state)
static std::string g_vuHashResult;

//Benchmark over a deterministic stretch of emulated time: wall ms between vblanks [from, to) after the next state load
static std::atomic<int> g_timeFrom = 0;
static std::atomic<int> g_timeTo = 0;
static std::atomic<int> g_vblanksSinceLoad = -1;
static double g_timeStart = 0;
static std::atomic<double> g_timeResult = -1;

static void OnVmStateLoaded()
{
	g_hashRemaining = g_hashRequest.exchange(0);
	g_vblanksSinceLoad = (g_timeTo > 0) ? 0 : -1;
}

#ifdef PROFILE
//Tier-2 study: EE block transitions recorded over the timing window (CEeExecutor::StartEdgeProfile),
//read with eeEdgeProfile() once eeTimeResult() is ready
static std::vector<uint8> g_edgeProfile;

static CEeExecutor* GetEeExecutor()
{
	return static_cast<CEeExecutor*>(g_virtualMachine->m_ee->m_EE.m_executor.get());
}
#endif

static void UpdateVblankTiming()
{
	if(g_vblanksSinceLoad < 0) return;
	int count = g_vblanksSinceLoad++;
	if(count == g_timeFrom)
	{
		g_timeStart = emscripten_get_now();
#ifdef PROFILE
		GetEeExecutor()->StartEdgeProfile(8 * 1024 * 1024);
#endif
	}
	if(count == g_timeTo)
	{
		double elapsed = emscripten_get_now() - g_timeStart;
#ifdef PROFILE
		g_edgeProfile = GetEeExecutor()->StopEdgeProfile();
#endif
		g_timeResult = elapsed;
		g_vblanksSinceLoad = -1;
		g_timeTo = 0;
	}
}

//View of the edge profile bytes (empty in non-profiling builds); copy it before the next window
emscripten::val eeEdgeProfile()
{
#ifdef PROFILE
	return emscripten::val(emscripten::typed_memory_view(g_edgeProfile.size(), g_edgeProfile.data()));
#else
	return emscripten::val::null();
#endif
}

static void HashVmState()
{
	auto& ee = *g_virtualMachine->m_ee;
	auto& iop = *g_virtualMachine->m_iop;
	//Component hashes: EE state, EE RAM, SPR, IOP state, IOP RAM (first differing component tells where to look)
	auto h = [](const void* data, size_t size) { return static_cast<unsigned long long>(XXH3_64bits(data, size)) & 0xFFFF; };
	std::lock_guard<std::mutex> lock(g_hashMutex);
	g_hashResult += string_format("%s%08X:%04llX.%04llX.%04llX.%04llX.%04llX", g_hashResult.empty() ? "" : " ", ee.m_EE.m_State.nPC,
	                              h(&ee.m_EE.m_State, sizeof(MIPSSTATE)), h(ee.m_ram, PS2::EE_RAM_SIZE), h(ee.m_spr, PS2::EE_SPR_SIZE),
	                              h(&iop.m_cpu.m_State, sizeof(MIPSSTATE)), h(iop.m_ram, PS2::IOP_RAM_SIZE));
	//(Then EE, VU0, VU1 states without the flag pipeline entries, which skipped flag writes change
	//without changing any value a program can read)
	auto hm = [&h](const MIPSSTATE& state) {
		MIPSSTATE copy = state;
		auto raw = reinterpret_cast<uint8*>(&copy);
		for(const auto* pipe : {&VUShared::g_pipeInfoMac, &VUShared::g_pipeInfoSticky, &VUShared::g_pipeInfoClip})
		{
			memset(raw + pipe->index, 0, sizeof(uint32));
			memset(raw + pipe->valueArray, 0, sizeof(uint32) * FLAG_PIPELINE_SLOTS);
			memset(raw + pipe->timeArray, 0, sizeof(uint32) * FLAG_PIPELINE_SLOTS);
		}
		return h(&copy, sizeof(MIPSSTATE));
	};
	g_vuHashResult += string_format("%s%04llX.%04llX.%04llX.%04llX.%04llX.%04llX/%04llX.%04llX.%04llX", g_vuHashResult.empty() ? "" : " ",
	                                h(&ee.m_VU0.m_State, sizeof(MIPSSTATE)), h(ee.m_vuMem0, PS2::VUMEM0SIZE), h(ee.m_microMem0, PS2::MICROMEM0SIZE),
	                                h(&ee.m_VU1.m_State, sizeof(MIPSSTATE)), h(ee.m_vuMem1, PS2::VUMEM1SIZE), h(ee.m_microMem1, PS2::MICROMEM1SIZE),
	                                hm(ee.m_EE.m_State), hm(ee.m_VU0.m_State), hm(ee.m_VU1.m_State));
}

std::string vuHashResult()
{
	std::lock_guard<std::mutex> lock(g_hashMutex);
	return g_vuHashResult;
}

//Debug: EE MIPSSTATE as hex words (compare two builds)
std::string eeStateDump()
{
	auto& state = g_virtualMachine->m_ee->m_EE.m_State;
	auto words = reinterpret_cast<const uint32*>(&state);
	std::string result;
	for(uint32 i = 0; i < sizeof(MIPSSTATE) / 4; i++) result += string_format("%08X ", words[i]);
	return result;
}

//Runs on the emulation thread (CPS2VM::OnNewFrame, at vblank start)
static void OnVmNewFrame()
{
	if(g_hashRemaining > 0)
	{
		g_hashRemaining--;
		HashVmState();
	}
	UpdateVblankTiming();
	auto cpu = g_virtualMachine->GetCpuUtilisationInfo();
	Timeline::Record(cpu);
	std::lock_guard<std::mutex> lock(g_profileMutex);
	g_profile.vblanks++;
	g_profile.cpu.eeTotalTicks += cpu.eeTotalTicks;
	g_profile.cpu.eeIdleTicks += cpu.eeIdleTicks;
	g_profile.cpu.iopTotalTicks += cpu.iopTotalTicks;
	g_profile.cpu.iopIdleTicks += cpu.iopIdleTicks;
#ifdef PROFILE
	for(const auto& zone : CProfiler::GetInstance().GetStats())
	{
		g_profile.zoneNs[zone.name] += zone.totalTime;
	}
	static uint32 vblankCounter = 0;
	if((++vblankCounter % 50) == 0)
	{
		g_profile.eePcs = TopPcs(CProfiler::GetInstance().TakePcHistogram(0));
		g_profile.iopPcs = TopPcs(CProfiler::GetInstance().TakePcHistogram(1));
	}
#endif
}

//Returns a JSON string with stats accumulated since the previous call.
std::string getProfile()
{
	PROFILE_ACCUM profile;
	{
		std::lock_guard<std::mutex> lock(g_profileMutex);
		profile = g_profile;
		g_profile = PROFILE_ACCUM();
		g_profile.eePcs = profile.eePcs;
		g_profile.iopPcs = profile.iopPcs;
	}
	double now = emscripten_get_now();
	double wallMs = g_profileLastTime ? (now - g_profileLastTime) : 0;
	g_profileLastTime = now;
	std::string zones;
	for(const auto& zone : profile.zoneNs)
	{
		zones += string_format("%s\"%s\":%.1f", zones.empty() ? "" : ",", zone.first.c_str(), zone.second / 1e6);
	}
	uint64 gsBusyNs = 0;
	uint64 vu1BusyNs = 0;
	uint32 compiles = 0;
	std::string extra;
	{
		static uint64 lastGsBusy = 0, lastVu1Busy = 0;
		static uint32 lastCompiles = 0;
		gsBusyNs = TakeDelta(g_timelineNs[TLT_GSBUSY], lastGsBusy);
		vu1BusyNs = TakeDelta(g_timelineNs[TLT_VU1BUSY], lastVu1Busy);
		compiles = TakeDelta(g_timelineCounts[TLC_COMPILES], lastCompiles);
	}
#ifdef PROFILE
	{
		std::string syscalls;
		for(uint32 i = 0; i < 0x100; i++)
		{
			uint32 count = g_eeSyscallCounts[i].exchange(0);
			if(count) syscalls += string_format("%s\"%02X\":%u", syscalls.empty() ? "" : ",", i, count);
		}
		auto log = CProfiler::GetInstance().TakeLog();
		std::string escapedLog;
		for(char c : log)
		{
			if(c == '"' || c == '\\') escapedLog += '\\';
			if(c == '\n') { escapedLog += "\\n"; continue; }
			if(static_cast<unsigned char>(c) >= 0x20) escapedLog += c;
		}
		std::string counters;
		static uint32 lastCounters[PCNT_MAX] = {};
		for(uint32 i = 0; i < PCNT_MAX; i++)
		{
			counters += string_format("%s\"%s\":%u", i ? "," : "", g_profileCounterNames[i], TakeDelta(g_profileCounters[i], lastCounters[i]));
		}
		extra = string_format(",\"counters\":{%s},\"vuBlocks\":%u,\"log\":\"%s\",\"syscalls\":{%s},\"eeInts\":%u,\"eeSlices\":%u,\"eeIdleSlices\":%u",
		                      counters.c_str(), g_vuBlockExecCount.exchange(0), escapedLog.c_str(), syscalls.c_str(), g_eeInterruptCount.exchange(0), g_eeSlices.exchange(0), g_eeIdleSlices.exchange(0));
	}
#endif
	return string_format("{\"wallMs\":%.1f,\"vblanks\":%u,\"eeTotal\":%d,\"eeIdle\":%d,\"iopTotal\":%d,\"iopIdle\":%d,"
	                     "\"gsBusyMs\":%.1f,\"vu1BusyMs\":%.1f,\"compiles\":%u,\"cacheHits\":%u,\"zonesMs\":{%s},\"eePcs\":[%s],\"iopPcs\":[%s]%s}",
	                     wallMs, profile.vblanks, profile.cpu.eeTotalTicks, profile.cpu.eeIdleTicks,
	                     profile.cpu.iopTotalTicks, profile.cpu.iopIdleTicks, gsBusyNs / 1e6, vu1BusyNs / 1e6, compiles, BlockCache::TakeHitCount(),
	                     zones.c_str(), profile.eePcs.c_str(), profile.iopPcs.c_str(), extra.c_str());
}

extern bool g_vuContextPromotion;
extern bool g_vuDiffMode;
extern std::atomic<uint32> g_vuDiffRuns;
extern std::atomic<uint32> g_vuDiffMismatches;

//VU JIT debug options (call before booting): promotion of VU context state into wasm locals,
//and the differential test (each VU block runs with and without promotion, results compared).
extern bool g_jitFunctionNames;

void setVuJitOptions(bool promotion, bool diff, bool names)
{
	g_vuContextPromotion = promotion;
	g_vuDiffMode = diff;
	g_jitFunctionNames = names;
}

extern bool g_vuSuperBlocks;
extern bool g_vuBlockChaining;

//VU block chaining (blocks tail call the next one), used for new blocks from now on
void setVuBlockChaining(bool enabled)
{
	g_vuBlockChaining = enabled;
}

extern bool g_vuSkipFlagWrites;

//VU status/clip flag pipeline writes nothing reads skipped (default on), used for new blocks from now on
void setVuSkipFlagWrites(bool enabled)
{
	g_vuSkipFlagWrites = enabled;
}

//VU superblocks (several VU blocks compiled into one function), used for new blocks from now on
void setVuSuperBlocks(bool enabled)
{
	g_vuSuperBlocks = enabled;
}

extern std::mutex g_vuDiffLogMutex;
extern std::string g_vuDiffLog;

extern std::atomic<uint32> g_vuCaptureRequest;
extern std::atomic<uint32> g_vuReplayRequest;
extern std::mutex g_vuReplayResultMutex;
extern std::string g_vuReplayResult;

//VU1 replay benchmark (see CVpu::RunReplayBenchmark): capture the next count programs, then
//request a replay; the result string appears once VU1 starts its next program.
extern int g_vuReplayExperiment;
//Debug: parts of the optimized VU codegen dropped in the promoted rounds of the next replay
void setVuExperiment(int bits)
{
	g_vuReplayExperiment = bits;
}

void vuCapture(uint32 count)
{
	g_vuCaptureRequest = count;
}

void vuReplay(uint32 iterations)
{
	{
		std::lock_guard<std::mutex> lock(g_vuReplayResultMutex);
		g_vuReplayResult.clear();
	}
	g_vuReplayRequest = iterations;
}

std::string vuReplayResult()
{
	std::lock_guard<std::mutex> lock(g_vuReplayResultMutex);
	return string_format("captures=%u\n%s", CVpu::GetReplayCaptureCount(), g_vuReplayResult.c_str());
}

std::string vuDiffStats()
{
	std::lock_guard<std::mutex> lock(g_vuDiffLogMutex);
	return string_format("runs=%u mismatches=%u\n%s", g_vuDiffRuns.load(), g_vuDiffMismatches.load(), g_vuDiffLog.c_str());
}

//Save states run on the emulation thread; never block the main thread on them
//(the GS thread may need the main thread for proxied GL calls). Poll stateStatus().
static std::future<bool> g_stateFuture;
static int g_stateResult = 1;

//EE block chaining (tail calls between JIT blocks); applies to blocks compiled afterwards, set before boot
extern bool g_eeBlockChaining;
void setEeBlockChaining(bool enabled)
{
	g_eeBlockChaining = enabled;
}

//EE module batching: newly compiled EE blocks are moved into shared wasm modules of this many blocks (0 = off)
extern uint32 g_jitPageAccess;
//Guest loads/stores as single page-table statements (default) or the older if/else Jitter sequence
void setJitPageAccess(uint32 mask)
{
	g_jitPageAccess = mask;
}

extern uint32 g_eeBatchSize;
void setEeBatchSize(uint32 size)
{
	g_eeBatchSize = size;
}

//EE context promotion (MIPS state kept in wasm locals within a block); for blocks compiled afterwards
extern bool g_eeContextPromotion;
void setEeContextPromotion(bool enabled)
{
	g_eeContextPromotion = enabled;
}

//Tier-2 EE regions: up to this many tier-1 blocks per wasm function (0 = off); for blocks created afterwards
extern uint32 g_eeRegionParts;
void setEeRegionParts(uint32 parts)
{
	g_eeRegionParts = parts;
}

//Debug: EE regions count entries, parts and instructions run ("entries parts instructions")
extern bool g_eeRegionCount;
std::string EeRegionRunCounts();
void setEeRegionCount(bool enabled)
{
	g_eeRegionCount = enabled;
}

std::string eeRegionRunCounts()
{
	return EeRegionRunCounts();
}

//JIT memory access fast paths marked likely (wasm branch hints), for blocks compiled afterwards
extern bool g_jitBranchHints;
void setJitBranchHints(bool enabled)
{
	g_jitBranchHints = enabled;
}

//Tier-2 EE regions follow JAL calls into the callee and back
extern bool g_eeRegionCalls;
void setEeRegionCalls(bool enabled)
{
	g_eeRegionCalls = enabled;
}

//Tier-2 EE regions compiled with context promotion (GPRs in wasm locals across the region)
extern bool g_eeRegionPromotion;
void setEeRegionPromotion(bool enabled)
{
	g_eeRegionPromotion = enabled;
}

//Tier-2 EE regions: an entry gets a region after this many samples (one dispatch in 64 is sampled)
extern uint32 g_eeRegionHotSamples;
void setEeRegionHotSamples(uint32 samples)
{
	g_eeRegionHotSamples = std::clamp<uint32>(samples, 1, 0xFFFF);
}

extern uint32 g_eeRegionStats[EE_REGION_STAT_COUNT];
extern uint64 g_eeDispatchCount;
extern uint32 g_eeRegionJitRuns;

//Counters in EE_REGION_STAT order (ee/EeExecutor.h), then the EE dispatch count and region Jitter runs
std::string eeRegionStats()
{
	std::string result;
	for(int i = 0; i < EE_REGION_STAT_COUNT; i++)
	{
		result += string_format("%u ", g_eeRegionStats[i]);
	}
	return result + string_format("%llu %u", static_cast<unsigned long long>(g_eeDispatchCount), g_eeRegionJitRuns);
}

//Automatic EE wait loop detection (see CEeExecutor): on/off, and the detected loops with their counters
extern bool g_eeAutoIdle;
std::string EeAutoIdleStats(bool);
void setEeAutoIdle(bool enabled)
{
	g_eeAutoIdle = enabled;
}

std::string eeAutoIdleStats(bool reset)
{
	return EeAutoIdleStats(reset);
}

extern std::mutex g_eeDumpMutex;
extern std::set<uint32> g_eeDumpRequest;
extern std::map<uint32, std::string> g_eeDumpResult;

//Debug: EE block start addresses (hex, comma separated) whose wasm code is dumped when they compile
void eeDumpRequest(std::string addresses)
{
	std::lock_guard<std::mutex> lock(g_eeDumpMutex);
	size_t pos = 0;
	while(pos < addresses.size())
	{
		size_t end = addresses.find(',', pos);
		if(end == std::string::npos) end = addresses.size();
		g_eeDumpRequest.insert(std::stoul(addresses.substr(pos, end - pos), nullptr, 16));
		pos = end + 1;
	}
}

std::string eeDumpResult()
{
	std::lock_guard<std::mutex> lock(g_eeDumpMutex);
	std::string result;
	for(const auto& entry : g_eeDumpResult) result += entry.second;
	return result;
}

void eeHashAfterLoad(int vblanks)
{
	std::lock_guard<std::mutex> lock(g_hashMutex);
	g_hashResult.clear();
	g_vuHashResult.clear();
	g_hashRequest = vblanks;
}

void eeTimeAfterLoad(int from, int to)
{
	g_timeResult = -1;
	g_timeFrom = from;
	g_timeTo = to;
}

//Vblanks since the state load while a timing window is armed, -1 otherwise (lets a CDP client
//sample thread CPU time over the same window)
int eeTimeVblank()
{
	return g_vblanksSinceLoad;
}

//Wall ms between the vblanks given to eeTimeAfterLoad, -1 until done
double eeTimeResult()
{
	return g_timeResult;
}

std::string eeHashResult()
{
	std::lock_guard<std::mutex> lock(g_hashMutex);
	return g_hashResult;
}

void saveState(std::string path)
{
	g_stateResult = -1;
	g_stateFuture = g_virtualMachine->SaveState(path);
}

void loadState(std::string path)
{
	g_stateResult = -1;
	g_stateFuture = g_virtualMachine->LoadState(path);
}

//-1: pending, 0: failed, 1: done/ok
int stateStatus()
{
	if(g_stateFuture.valid() && g_stateFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
	{
		g_stateResult = g_stateFuture.get() ? 1 : 0;
	}
	return g_stateResult;
}

//VU1 on its own thread (on by default), applied at the next vblank
//GS renderer options by name (bench/debug switch). "fbAlias" (on by default): PSMCT16 draws over
//PSMCT32 frame buffers, destination alpha test and unclamped additive blend, see CGSH_OpenGL::SyncFramebufferAlias.
bool setGsHack(std::string name, bool enabled)
{
	auto gs = static_cast<CGSH_OpenGLJs*>(g_virtualMachine->GetGSHandler());
	if(name == "fbAlias")
	{
		gs->SetFramebufferAliasing(enabled);
		return true;
	}
	return false;
}

void setVu1Threaded(bool enabled)
{
	g_virtualMachine->SetVu1Threaded(enabled);
}

void setFrameLimit(bool enabled)
{
	CAppConfig::GetInstance().SetPreferenceBoolean(PREF_PS2_LIMIT_FRAMERATE, enabled);
	g_virtualMachine->ReloadFrameRateLimit();
}

//Page settings: GS internal resolution (1 = native), forced bilinear texture filtering, widescreen
//(16:9 output) and presentation mode (CGSHandler::PRESENTATION_MODE). The output canvas grows with the
//resolution factor so the page gets the full-resolution frames (640x480 or 854x480 per unit of factor).
void setGsOptions(int resolutionFactor, bool forceBilinear, bool widescreen, int presentationMode)
{
	resolutionFactor = std::clamp(resolutionFactor, 1, 4);
	if(presentationMode < CGSHandler::PRESENTATION_MODE_FILL || presentationMode > CGSHandler::PRESENTATION_MODE_ORIGINAL)
	{
		presentationMode = CGSHandler::PRESENTATION_MODE_FIT;
	}
	auto& config = CAppConfig::GetInstance();
	config.SetPreferenceInteger(PREF_CGSH_OPENGL_RESOLUTION_FACTOR, resolutionFactor);
	config.SetPreferenceBoolean(PREF_CGSH_OPENGL_FORCEBILINEARTEXTURES, forceBilinear);
	config.SetPreferenceBoolean(PREF_CGSHANDLER_WIDESCREEN, widescreen);
	config.SetPreferenceInteger(PREF_CGSHANDLER_PRESENTATION_MODE, presentationMode);
	auto gs = static_cast<CGSH_OpenGLJs*>(g_virtualMachine->GetGSHandler());
	gs->SetOutput((widescreen ? 854 : 640) * resolutionFactor, 480 * resolutionFactor, static_cast<CGSHandler::PRESENTATION_MODE>(presentationMode));
}

//Frame skip for slow machines (0 = off, n = n frames skipped after each drawn one), see CGSH_OpenGLJs.
void setFrameSkip(int count)
{
	static_cast<CGSH_OpenGLJs*>(g_virtualMachine->GetGSHandler())->SetFrameSkip(std::clamp(count, 0, 3));
}

//Frames not drawn because of the frame skip since start (cumulative).
uint32 getSkippedFrames()
{
	return static_cast<CGSH_OpenGLJs*>(g_virtualMachine->GetGSHandler())->GetSkippedFrames();
}

//Replaces the keyboard/gamepad bindings: keys = [[KeyboardEvent.code, PS2 button, positive], ...]
//(button = PS2::CControllerInfo::BUTTON, positive only matters for axes), pad = PS2 button per standard
//gamepad button index (-1 or missing = unbound).
void setInputBindings(emscripten::val keys, emscripten::val pad)
{
	std::vector<CInputProviderEmscripten::KEYBINDING> keyBindings;
	uint32 keyCount = keys["length"].as<uint32>();
	for(uint32 i = 0; i < keyCount; i++)
	{
		auto entry = keys[i];
		int button = entry[1].as<int>();
		if(button < 0 || button >= PS2::CControllerInfo::MAX_BUTTONS) continue;
		keyBindings.push_back({entry[0].as<std::string>(), static_cast<PS2::CControllerInfo::BUTTON>(button), entry[2].as<bool>()});
	}
	int padButtons[CInputProviderEmscripten::MAX_PAD_BUTTONS];
	uint32 padCount = pad["length"].as<uint32>();
	for(uint32 i = 0; i < CInputProviderEmscripten::MAX_PAD_BUTTONS; i++)
	{
		padButtons[i] = (i < padCount) ? pad[i].as<int>() : -1;
	}
	g_inputProvider->SetBindings(std::move(keyBindings), padButtons);
}

void setGamepadEnabled(bool enabled)
{
	g_inputProvider->SetGamepadEnabled(enabled);
}

//EE underclock (numerator/denominator of the real EE clock), like PCSX2's EE cycle rate:
//the EE gets fewer cycles per vblank while IOP, SPU and vblank timing stay real time.
void setEeFrequencyScale(int numerator, int denominator)
{
	if((numerator <= 0) || (denominator <= 0)) return;
	g_virtualMachine->SetEeFrequencyScale(numerator, denominator);
}

int main(int argc, const char** argv)
{
	printf("Play! - Version %s\r\n", PLAY_VERSION);
	return 0;
}

//Browser Gamepad API is only reachable from the main thread: poll it every animation frame.
EM_BOOL gamepadPollCallback(double, void*)
{
	g_inputProvider->PollGamepads();
	return EM_TRUE;
}

EM_BOOL keyboardCallback(int eventType, const EmscriptenKeyboardEvent* keyEvent, void* userData)
{
	if(keyEvent->repeat)
	{
		return true;
	}
	switch(eventType)
	{
	case EMSCRIPTEN_EVENT_KEYDOWN:
		g_inputProvider->OnKeyDown(keyEvent->code);
		break;
	case EMSCRIPTEN_EVENT_KEYUP:
		g_inputProvider->OnKeyUp(keyEvent->code);
		break;
	}
	return true;
}

extern "C" void initVm()
{
	//The GS thread renders on its own OffscreenCanvas (see GSH_OpenGLJs.h); the page shows the frames
	//it receives through Module.onGsFrame, which must be set before this call.
	g_virtualMachine = new CPs2VmJs();
	g_virtualMachine->Initialize();
	g_virtualMachine->CreateGSHandler(CGSH_OpenGLJs::GetFactoryFunction(640, 480));

	{
		//Size here needs to match the size of the canvas in HTML file.

		CGSHandler::PRESENTATION_PARAMS presentationParams;
		presentationParams.mode = CGSHandler::PRESENTATION_MODE_FIT;
		presentationParams.windowWidth = 640;
		presentationParams.windowHeight = 480;

		g_virtualMachine->m_ee->m_gs->SetPresentationParams(presentationParams);
	}

	{
		g_virtualMachine->CreatePadHandler(CPH_GenericInput::GetFactoryFunction());
		auto padHandler = static_cast<CPH_GenericInput*>(g_virtualMachine->GetPadHandler());
		auto& bindingManager = padHandler->GetBindingManager();

		g_inputProvider = std::make_shared<CInputProviderEmscripten>();
		bindingManager.RegisterInputProvider(g_inputProvider);

		//Keyboard and gamepads are merged by the provider, one target per PS2 control.
		for(uint32 button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			auto ps2Button = static_cast<PS2::CControllerInfo::BUTTON>(button);
			bindingManager.SetSimpleBinding(0, ps2Button, CInputProviderEmscripten::MakeBindingTarget(ps2Button));
		}
		bindingManager.ResetBindingValues();
	}

	{
		//Hand samples to the output every 10 SPU blocks (~10 ms) instead of 100 (~100 ms): the worklet keeps
		//~100 ms queued, so it needs a smooth supply.
		CAppConfig::GetInstance().SetPreferenceInteger(PREF_AUDIO_SPUBLOCKCOUNT, 10);
		g_virtualMachine->ReloadSpuBlockCount();
		CSH_WebAudio::Start();
		g_virtualMachine->CreateSoundHandler(CSH_WebAudio::GetFactoryFunction());
	}

	g_virtualMachine->SetVu1Threaded(true);

	g_vmNewFrameConnection = g_virtualMachine->OnNewFrame.Connect(&OnVmNewFrame);
	g_vmStateLoadedConnection = g_virtualMachine->OnStateLoaded.Connect(&OnVmStateLoaded);
	g_gsNewFrameConnection = g_virtualMachine->GetGSHandler()->OnNewFrame.Connect(std::bind(&CStatsManager::OnGsNewFrame, &CStatsManager::GetInstance(), std::placeholders::_1));

	EMSCRIPTEN_RESULT result = EMSCRIPTEN_RESULT_SUCCESS;

	result = emscripten_set_keydown_callback("#outputCanvas", nullptr, false, &keyboardCallback);
	assert(result == EMSCRIPTEN_RESULT_SUCCESS);

	result = emscripten_set_keyup_callback("#outputCanvas", nullptr, false, &keyboardCallback);
	assert(result == EMSCRIPTEN_RESULT_SUCCESS);

	emscripten_request_animation_frame_loop(&gamepadPollCallback, nullptr);

	//Not waited for: the GS thread needs the main thread free to create its WebGL context.
	auto gs = static_cast<CGSH_OpenGLJs*>(g_virtualMachine->GetGSHandler());
	gs->SendGSCall([gs]() { gs->InitializeContext(); });
}

//The page left the game: stop emulating until the next boot (VM thread idle, audio context suspended).
//Booting resets the VM and resumes it.
void pauseVm()
{
	g_virtualMachine->PauseAsync();
	EM_ASM({ Module["audioSetPaused"](true); });
}

//Continues a VM held by pauseVm without rebooting it (e.g. the page held the game while a boot intro played).
void resumeVm()
{
	EM_ASM({ Module["audioSetPaused"](false); });
	g_virtualMachine->Resume();
}

void bootElf(std::string path)
{
	EM_ASM({ Module["audioSetPaused"](false); });
	g_virtualMachine->BootElf(path);
}

void bootDiscImage(std::string path)
{
	EM_ASM({ Module["audioSetPaused"](false); });
	g_virtualMachine->BootDiscImage(path);
}

int getFrames()
{
	return CStatsManager::GetInstance().GetFrames();
}

void clearStats()
{
	CStatsManager::GetInstance().ClearStats();
}

//Persistent JIT block cache (see BlockCache.h). The page owns storage (OPFS): it loads the
//saved entries for a game before booting it and periodically appends what blockCacheTakePending returns.
static uint32 blockCacheLoad(emscripten::val data)
{
	auto bytes = emscripten::convertJSArrayToNumberVector<uint8>(data);
	return BlockCache::Load(bytes.data(), bytes.size());
}

static emscripten::val blockCacheTakePending()
{
	auto pending = BlockCache::TakePending();
	if(pending.empty()) return emscripten::val::null();
	//Copy out of the (shared) wasm heap so the result can be written to OPFS
	return emscripten::val::global("Uint8Array").new_(emscripten::typed_memory_view(pending.size(), pending.data()));
}

static double blockCachePendingSize()
{
	return static_cast<double>(BlockCache::GetPendingSize());
}

EMSCRIPTEN_BINDINGS(Play)
{
	using namespace emscripten;

	function("bootElf", &bootElf);
	function("bootDiscImage", &bootDiscImage);
	function("pauseVm", &pauseVm);
	function("resumeVm", &resumeVm);
	function("setDiscImageUrl", &Js_SetDiscImageUrl);
	function("getFrames", &getFrames);
	function("clearStats", &clearStats);
	function("getProfile", &getProfile);
	function("getTimeline", &getTimeline);
	function("setVuJitOptions", &setVuJitOptions);
	function("setVuSuperBlocks", &setVuSuperBlocks);
	function("vuDiffStats", &vuDiffStats);
	function("vuCapture", &vuCapture);
	function("setVuExperiment", &setVuExperiment);
	function("vuReplay", &vuReplay);
	function("vuReplayResult", &vuReplayResult);
	function("saveState", &saveState);
	function("setEeBlockChaining", &setEeBlockChaining);
	function("eeHashAfterLoad", &eeHashAfterLoad);
	function("eeTimeAfterLoad", &eeTimeAfterLoad);
	function("eeTimeResult", &eeTimeResult);
	function("eeTimeVblank", &eeTimeVblank);
	function("setEeContextPromotion", &setEeContextPromotion);
	function("setEeRegionParts", &setEeRegionParts);
	function("eeRegionStats", &eeRegionStats);
	function("setEeRegionHotSamples", &setEeRegionHotSamples);
	function("setEeRegionPromotion", &setEeRegionPromotion);
	function("setEeRegionCalls", &setEeRegionCalls);
	function("setJitBranchHints", &setJitBranchHints);
	function("setEeRegionCount", &setEeRegionCount);
	function("eeRegionRunCounts", &eeRegionRunCounts);
	function("eeEdgeProfile", &eeEdgeProfile);
	function("setEeAutoIdle", &setEeAutoIdle);
	function("eeAutoIdleStats", &eeAutoIdleStats);
	function("eeHashResult", &eeHashResult);
	function("vuHashResult", &vuHashResult);
	function("setVuBlockChaining", &setVuBlockChaining);
	function("setVuSkipFlagWrites", &setVuSkipFlagWrites);
	function("eeDumpRequest", &eeDumpRequest);
	function("eeStateDump", &eeStateDump);
	function("setEeBatchSize", &setEeBatchSize);
	function("setJitPageAccess", &setJitPageAccess);
	function("eeDumpResult", &eeDumpResult);
	function("loadState", &loadState);
	function("stateStatus", &stateStatus);
	function("setFrameLimit", &setFrameLimit);
	function("setGsOptions", &setGsOptions);
	function("setFrameSkip", &setFrameSkip);
	function("getSkippedFrames", &getSkippedFrames);
	function("setInputBindings", &setInputBindings);
	function("setGamepadEnabled", &setGamepadEnabled);
	function("setVu1Threaded", &setVu1Threaded);
	function("setGsHack", &setGsHack);
	function("setEeFrequencyScale", &setEeFrequencyScale);
	function("blockCacheEnable", &BlockCache::SetEnabled);
	function("blockCacheClear", &BlockCache::Clear);
	function("blockCacheLoad", &blockCacheLoad);
	function("blockCacheTakePending", &blockCacheTakePending);
	function("blockCachePendingSize", &blockCachePendingSize);
}
