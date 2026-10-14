#include <cstdio>
#include <emscripten/bind.h>
#include "Ps2VmJs.h"
#include "GSH_OpenGLJs.h"
#include "sound/SH_OpenAL/SH_OpenALProxy.h"
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
#include <map>
#include <atomic>
#include <algorithm>
#include <emscripten.h>

#ifdef PROFILE
extern std::atomic<uint64> g_gsThreadBusyNs;
extern std::atomic<uint32> g_blockCompileCount;
extern std::atomic<uint32> g_vuBlockExecCount;
extern std::atomic<uint32> g_eeSyscallCounts[0x100];
extern std::atomic<uint32> g_eeInterruptCount;
extern std::atomic<uint32> g_eeSlices;
extern std::atomic<uint32> g_eeIdleSlices;
#endif

CPs2VmJs* g_virtualMachine = nullptr;
CGSHandler::NewFrameEvent::Connection g_gsNewFrameConnection;
EMSCRIPTEN_WEBGL_CONTEXT_HANDLE g_context = 0;
std::shared_ptr<CInputProviderEmscripten> g_inputProvider;
CSH_OpenAL* g_soundHandler = nullptr;
Framework::CSignal<void()>::Connection g_vmNewFrameConnection;

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

//Runs on the emulation thread (CPS2VM::OnNewFrame, at vblank start)
static void OnVmNewFrame()
{
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
	uint32 compiles = 0;
	std::string extra;
#ifdef PROFILE
	gsBusyNs = g_gsThreadBusyNs.exchange(0);
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
		extra = string_format(",\"vuBlocks\":%u,\"log\":\"%s\",\"syscalls\":{%s},\"eeInts\":%u,\"eeSlices\":%u,\"eeIdleSlices\":%u",
		                      g_vuBlockExecCount.exchange(0), escapedLog.c_str(), syscalls.c_str(), g_eeInterruptCount.exchange(0), g_eeSlices.exchange(0), g_eeIdleSlices.exchange(0));
	}
#endif
	return string_format("{\"wallMs\":%.1f,\"vblanks\":%u,\"eeTotal\":%d,\"eeIdle\":%d,\"iopTotal\":%d,\"iopIdle\":%d,"
	                     "\"gsBusyMs\":%.1f,\"compiles\":%u,\"cacheHits\":%u,\"zonesMs\":{%s},\"eePcs\":[%s],\"iopPcs\":[%s]%s}",
	                     wallMs, profile.vblanks, profile.cpu.eeTotalTicks, profile.cpu.eeIdleTicks,
	                     profile.cpu.iopTotalTicks, profile.cpu.iopIdleTicks, gsBusyNs / 1e6, compiles, BlockCache::TakeHitCount(),
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

extern std::mutex g_vuDiffLogMutex;
extern std::string g_vuDiffLog;

extern std::atomic<uint32> g_vuCaptureRequest;
extern std::atomic<uint32> g_vuReplayRequest;
extern std::mutex g_vuReplayResultMutex;
extern std::string g_vuReplayResult;

//VU1 replay benchmark (see CVpu::RunReplayBenchmark): capture the next count programs, then
//request a replay; the result string appears once VU1 starts its next program.
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
	EmscriptenWebGLContextAttributes attr;
	emscripten_webgl_init_context_attributes(&attr);
	attr.majorVersion = 2;
	attr.minorVersion = 0;
	attr.alpha = false;
	g_context = emscripten_webgl_create_context("#outputCanvas", &attr);
	assert(g_context >= 0);

	g_virtualMachine = new CPs2VmJs();
	g_virtualMachine->Initialize();
	g_virtualMachine->CreateGSHandler(CGSH_OpenGLJs::GetFactoryFunction(g_context));

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
		g_soundHandler = new CSH_OpenAL();
		g_virtualMachine->CreateSoundHandler(CSH_OpenALProxy::GetFactoryFunction(g_soundHandler));
	}

	g_vmNewFrameConnection = g_virtualMachine->OnNewFrame.Connect(&OnVmNewFrame);
	g_gsNewFrameConnection = g_virtualMachine->GetGSHandler()->OnNewFrame.Connect(std::bind(&CStatsManager::OnGsNewFrame, &CStatsManager::GetInstance(), std::placeholders::_1));

	EMSCRIPTEN_RESULT result = EMSCRIPTEN_RESULT_SUCCESS;

	result = emscripten_set_keydown_callback("#outputCanvas", nullptr, false, &keyboardCallback);
	assert(result == EMSCRIPTEN_RESULT_SUCCESS);

	result = emscripten_set_keyup_callback("#outputCanvas", nullptr, false, &keyboardCallback);
	assert(result == EMSCRIPTEN_RESULT_SUCCESS);

	emscripten_request_animation_frame_loop(&gamepadPollCallback, nullptr);
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
	function("vuDiffStats", &vuDiffStats);
	function("vuCapture", &vuCapture);
	function("vuReplay", &vuReplay);
	function("vuReplayResult", &vuReplayResult);
	function("saveState", &saveState);
	function("loadState", &loadState);
	function("stateStatus", &stateStatus);
	function("setFrameLimit", &setFrameLimit);
	function("setEeFrequencyScale", &setEeFrequencyScale);
	function("blockCacheEnable", &BlockCache::SetEnabled);
	function("blockCacheClear", &BlockCache::Clear);
	function("blockCacheLoad", &blockCacheLoad);
	function("blockCacheTakePending", &blockCacheTakePending);
	function("blockCachePendingSize", &blockCachePendingSize);
}
