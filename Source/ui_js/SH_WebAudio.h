#pragma once

#include <string>
#include "sound/SoundHandler.h"

//Audio output of the web build. Write (VM thread) copies the samples into a lock-free ring in wasm memory;
//an AudioWorklet (WebAudio.js) reads the ring from the shared memory and time-stretches the audio when the
//emulator runs slower than real time. No main thread round trip.
class CSH_WebAudio : public CSoundHandler
{
public:
	//Creates the AudioContext and the worklet. Main thread only.
	static void Start();
	static FactoryFunction GetFactoryFunction();
	//Hash of the first N frames written after the call (differential tests of the SPU output)
	static void ArmPcmHash(uint32);
	static std::string GetPcmHash();

	void Reset() override;
	void Write(int16*, unsigned int, unsigned int) override;
	bool HasFreeBuffers() override;
	void RecycleBuffers() override;
};
