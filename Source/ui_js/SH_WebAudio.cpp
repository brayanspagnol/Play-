#include "SH_WebAudio.h"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <emscripten.h>
#include <mutex>
#include <cstdio>
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

//Layout shared with WebAudio.js: 4 uint32 (write, read, overflow, frames), then stereo int16 samples.
//write/read are free-running frame counters; only the VM thread moves write, only the worklet moves read.
struct SAMPLE_RING
{
	enum
	{
		FRAMES = 32768, //743 ms at 44.1 kHz
	};
	std::atomic<uint32> write;
	std::atomic<uint32> read;
	std::atomic<uint32> overflow; //frames dropped because the ring was full (worklet not running)
	uint32 frames;
	int16 samples[FRAMES * 2];
};
static_assert(sizeof(std::atomic<uint32>) == 4, "ring header layout");
static SAMPLE_RING g_ring;

void CSH_WebAudio::Start()
{
	g_ring.frames = SAMPLE_RING::FRAMES;
	MAIN_THREAD_EM_ASM({ Module["webAudioStart"]($0); }, &g_ring);
}

CSoundHandler::FactoryFunction CSH_WebAudio::GetFactoryFunction()
{
	return []() { return new CSH_WebAudio(); };
}

void CSH_WebAudio::Reset()
{
}

//Differential test of the audio output: hash of the first N frames written after ArmPcmHash
static std::mutex g_pcmHashMutex;
static std::atomic<uint32> g_pcmHashRemaining = 0;
static uint32 g_pcmHashFrames = 0;
static XXH3_state_t g_pcmHashState;

void CSH_WebAudio::ArmPcmHash(uint32 frames)
{
	std::lock_guard lock(g_pcmHashMutex);
	g_pcmHashRemaining = frames;
	g_pcmHashFrames = 0;
	XXH3_64bits_reset(&g_pcmHashState);
}

std::string CSH_WebAudio::GetPcmHash()
{
	std::lock_guard lock(g_pcmHashMutex);
	char result[64];
	snprintf(result, sizeof(result), "%u:%016llx", g_pcmHashFrames, static_cast<unsigned long long>(XXH3_64bits_digest(&g_pcmHashState)));
	return result;
}

void CSH_WebAudio::Write(int16* samples, unsigned int sampleCount, unsigned int)
{
	uint32 frames = sampleCount / 2;
	if(g_pcmHashRemaining != 0)
	{
		std::lock_guard lock(g_pcmHashMutex);
		uint32 count = std::min(frames, g_pcmHashRemaining.load());
		XXH3_64bits_update(&g_pcmHashState, samples, count * 2 * sizeof(int16));
		g_pcmHashRemaining -= count;
		g_pcmHashFrames += count;
	}
	uint32 write = g_ring.write.load(std::memory_order_relaxed);
	uint32 read = g_ring.read.load(std::memory_order_acquire);
	uint32 space = SAMPLE_RING::FRAMES - (write - read);
	if(frames > space)
	{
		g_ring.overflow.fetch_add(frames - space, std::memory_order_relaxed);
		frames = space;
	}
	for(uint32 done = 0; done < frames;)
	{
		uint32 index = (write + done) & (SAMPLE_RING::FRAMES - 1);
		uint32 count = std::min(frames - done, SAMPLE_RING::FRAMES - index);
		memcpy(&g_ring.samples[index * 2], samples + done * 2, count * 2 * sizeof(int16));
		done += count;
	}
	g_ring.write.store(write + frames, std::memory_order_release);
}

bool CSH_WebAudio::HasFreeBuffers()
{
	return true;
}

void CSH_WebAudio::RecycleBuffers()
{
}
