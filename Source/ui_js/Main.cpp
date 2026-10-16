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
#include <future>
#include <mutex>
#include <set>
#include <map>
#include <atomic>
#include <algorithm>
#include <emscripten.h>
#include "xxhash.h"
#include "Ps2Const.h"
#include "ee/EeExecutor.h"

#ifdef PROFILE
extern std::atomic<uint64> g_gsThreadBusyNs;
extern std::atomic<uint64> g_vu1ThreadBusyNs;
extern std::atomic<uint32> g_blockCompileCount;
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
#ifdef PROFILE
	gsBusyNs = g_gsThreadBusyNs.exchange(0);
	vu1BusyNs = g_vu1ThreadBusyNs.exchange(0);
	compiles = g_blockCompileCount.exchange(0);
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
		for(uint32 i = 0; i < PCNT_MAX; i++)
		{
			counters += string_format("%s\"%s\":%u", i ? "," : "", g_profileCounterNames[i], g_profileCounters[i].exchange(0));
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
void setVu1Threaded(bool enabled)
{
	g_virtualMachine->SetVu1Threaded(enabled);
}

void setFrameLimit(bool enabled)
{
	CAppConfig::GetInstance().SetPreferenceBoolean(PREF_PS2_LIMIT_FRAMERATE, enabled);
	g_virtualMachine->ReloadFrameRateLimit();
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

void bootElf(std::string path)
{
	g_virtualMachine->BootElf(path);
}

void bootDiscImage(std::string path)
{
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
	function("setDiscImageUrl", &Js_SetDiscImageUrl);
	function("getFrames", &getFrames);
	function("clearStats", &clearStats);
	function("getProfile", &getProfile);
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
	function("eeHashResult", &eeHashResult);
	function("eeEdgeProfile", &eeEdgeProfile);
	function("eeDumpRequest", &eeDumpRequest);
	function("eeStateDump", &eeStateDump);
	function("setEeBatchSize", &setEeBatchSize);
	function("setJitPageAccess", &setJitPageAccess);
	function("eeDumpResult", &eeDumpResult);
	function("loadState", &loadState);
	function("stateStatus", &stateStatus);
	function("setFrameLimit", &setFrameLimit);
	function("setVu1Threaded", &setVu1Threaded);
	function("setEeFrequencyScale", &setEeFrequencyScale);
	function("blockCacheEnable", &BlockCache::SetEnabled);
	function("blockCacheClear", &BlockCache::Clear);
	function("blockCacheLoad", &blockCacheLoad);
	function("blockCacheTakePending", &blockCacheTakePending);
	function("blockCachePendingSize", &blockCachePendingSize);
}
