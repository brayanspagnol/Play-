#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "Types.h"
#include "iop/Iop_SpuBase.h"

class CSoundHandler;

//Renders the SPU2 audio on its own thread.
//
//The VM thread keeps running both SPU cores, but in "state only" mode (CSpuBase::Render with a null buffer):
//voice positions, end flags, envelopes, IRQs and sound input block positions advance exactly as before, so
//register reads, IRQs and DMA timing do not change. Before each update block, it queues a copy of both cores.
//This thread replays the block in full on replica cores that keep their own decoder state (ADPCM history,
//decoded samples), an SPU RAM mirror (fed with every DMA/PIO write, in order) and sample cache, mixes the
//cores like CPS2VM::UpdateSpu did and writes the result to the sound handler: the output is the same.
//
//Reverb writes only reach the mirror. SyncBack copies the mirror and the decoder state back into the VM's
//cores (save states, executable reloads), and DMA reads that overlap a reverb work area wait for this thread
//and read the mirror.
class CSpuRenderThread : public Iop::CSpuRamObserver
{
public:
	//blockSize: int16 samples (stereo) per update block
	CSpuRenderThread(Iop::CSpuBase&, Iop::CSpuBase&, uint8* ram, uint32 ramSize, uint32 blockSize, uint32 maxBlockCount, uint32 sampleRate);
	~CSpuRenderThread();

	//VM thread: one update block (CPS2VM::UpdateSpu)
	void RenderBlock(CSoundHandler*, uint32 blockCount);
	//VM thread: the VM's cores and SPU RAM were replaced (reset, state load): the replicas copy them
	//before the next block
	void RequestResync();
	//VM thread: waits for this thread, then copies the mirror (reverb writes) and the decoder state back
	//into the VM's cores and SPU RAM, which then hold the complete state
	void SyncBack();
	//VM thread: waits until every queued block was written to the sound handler
	void Drain();

	void OnRamClear(uint32, uint32) override;
	void OnRamWrite(uint32, uint32) override;
	void OnRamRead(uint32, uint32) override;

private:
	enum
	{
		SLOT_COUNT = 32,
		RAMLOG_CLEAR = 0x80000000,
	};

	struct SLOT
	{
		std::unique_ptr<Iop::CSpuBase> core0;
		std::unique_ptr<Iop::CSpuBase> core1;
		//Entries: uint32 address, uint32 size (| RAMLOG_CLEAR for a cache invalidation), then size bytes for a write
		std::vector<uint8> ramLog;
		CSoundHandler* soundHandler = nullptr;
		uint32 blockCount = 0;
		bool render = false;
	};

	SLOT& OpenSlot();
	void Publish(bool render);
	void Resync();
	void ThreadProc();
	void Process(SLOT&);

	Iop::CSpuBase& m_vmCore0;
	Iop::CSpuBase& m_vmCore1;
	uint8* m_vmRam = nullptr;
	uint32 m_ramSize = 0;
	uint32 m_blockSize = 0;
	uint32 m_sampleRate = 0;

	//Replicas (this thread only, except while it is idle in Resync/SyncBack)
	std::vector<uint8> m_ram;
	Iop::CSpuSampleCache m_sampleCache;
	Iop::CSpuIrqWatcher m_irqWatcher;
	std::unique_ptr<Iop::CSpuBase> m_core0;
	std::unique_ptr<Iop::CSpuBase> m_core1;
	std::vector<int16> m_samples;
	std::vector<int16> m_core1Samples;
	uint32 m_currentBlock = 0;

	SLOT m_slots[SLOT_COUNT];
	//Producer state (VM thread)
	bool m_slotOpen = false;
	bool m_resyncPending = true;
	//SPU RAM range where the mirror may differ from the VM's copy (reverb writes)
	uint32 m_dirtyStart = ~0U;
	uint32 m_dirtyEnd = 0;

	std::mutex m_mutex;
	std::condition_variable m_workCondition;
	std::condition_variable m_idleCondition;
	uint32 m_publishedIndex = 0;
	uint32 m_readIndex = 0;
	bool m_quit = false;
	std::thread m_thread;
};
