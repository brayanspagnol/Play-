#include "SH_WebAudio.h"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <emscripten.h>

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

void CSH_WebAudio::Write(int16* samples, unsigned int sampleCount, unsigned int)
{
	uint32 frames = sampleCount / 2;
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
