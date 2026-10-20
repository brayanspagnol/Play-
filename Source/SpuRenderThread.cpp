#include <cassert>
#include <cfenv>
#include <cstring>
#include <algorithm>
#include "SpuRenderThread.h"
#include "FpUtils.h"
#include "sound/SoundHandler.h"

CSpuRenderThread::CSpuRenderThread(Iop::CSpuBase& core0, Iop::CSpuBase& core1, uint8* ram, uint32 ramSize, uint32 blockSize, uint32 maxBlockCount, uint32 sampleRate)
    : m_vmCore0(core0)
    , m_vmCore1(core1)
    , m_vmRam(ram)
    , m_ramSize(ramSize)
    , m_blockSize(blockSize)
    , m_sampleRate(sampleRate)
    , m_ram(ramSize)
    , m_samples(blockSize * maxBlockCount)
    , m_core1Samples(blockSize)
{
	m_core0 = std::make_unique<Iop::CSpuBase>(m_ram.data(), ramSize, &m_sampleCache, &m_irqWatcher, 0);
	m_core1 = std::make_unique<Iop::CSpuBase>(m_ram.data(), ramSize, &m_sampleCache, &m_irqWatcher, 1);
	for(auto& slot : m_slots)
	{
		//Only assigned to (CSpuBase::operator=), never rendered
		slot.core0 = std::make_unique<Iop::CSpuBase>(nullptr, ramSize, nullptr, nullptr, 0);
		slot.core1 = std::make_unique<Iop::CSpuBase>(nullptr, ramSize, nullptr, nullptr, 1);
	}
	m_vmCore0.SetRamObserver(this);
	m_vmCore1.SetRamObserver(this);
	m_thread = std::thread([this]() { ThreadProc(); });
}

CSpuRenderThread::~CSpuRenderThread()
{
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_workCondition.notify_all();
	m_thread.join();
	m_vmCore0.SetRamObserver(nullptr);
	m_vmCore1.SetRamObserver(nullptr);
}

void CSpuRenderThread::RenderBlock(CSoundHandler* soundHandler, uint32 blockCount)
{
	if(m_resyncPending) Resync();
	auto& slot = OpenSlot();
	*slot.core0 = m_vmCore0;
	*slot.core1 = m_vmCore1;
	slot.soundHandler = soundHandler;
	slot.blockCount = blockCount;
	bool core1Enabled = m_vmCore1.IsEnabled();
	for(auto core : {&m_vmCore0, core1Enabled ? &m_vmCore1 : nullptr})
	{
		uint32 start = 0, end = 0;
		if(core && core->GetReverbWriteRange(start, end))
		{
			m_dirtyStart = std::min(m_dirtyStart, start);
			m_dirtyEnd = std::max(m_dirtyEnd, end);
		}
	}
	Publish(true);
	m_vmCore0.Render(nullptr, m_blockSize);
	if(core1Enabled)
	{
		m_vmCore1.Render(nullptr, m_blockSize);
	}
}

void CSpuRenderThread::RequestResync()
{
	m_resyncPending = true;
}

void CSpuRenderThread::Resync()
{
	Drain();
	//Writes logged since the last block are already in the VM's RAM
	m_slotOpen = false;
	memcpy(m_ram.data(), m_vmRam, m_ramSize);
	m_sampleCache.Clear();
	m_core0->CopyState(m_vmCore0, false);
	m_core1->CopyState(m_vmCore1, false);
	m_dirtyStart = ~0U;
	m_dirtyEnd = 0;
	m_resyncPending = false;
}

void CSpuRenderThread::SyncBack()
{
	if(m_resyncPending) return;
	if(m_slotOpen) Publish(false);
	Drain();
	memcpy(m_vmRam, m_ram.data(), m_ramSize);
	m_vmCore0.CopyDecodeState(*m_core0);
	m_vmCore1.CopyDecodeState(*m_core1);
	m_dirtyStart = ~0U;
	m_dirtyEnd = 0;
}

void CSpuRenderThread::Drain()
{
	std::unique_lock lock(m_mutex);
	m_idleCondition.wait(lock, [&]() { return m_readIndex == m_publishedIndex; });
}

void CSpuRenderThread::OnRamClear(uint32 address, uint32 size)
{
	if(m_resyncPending) return;
	auto& log = OpenSlot().ramLog;
	uint32 header[2] = {address, size | RAMLOG_CLEAR};
	log.insert(log.end(), reinterpret_cast<uint8*>(header), reinterpret_cast<uint8*>(header + 2));
}

void CSpuRenderThread::OnRamWrite(uint32 address, uint32 size)
{
	if(m_resyncPending) return;
	assert((address + size) <= m_ramSize);
	auto& log = OpenSlot().ramLog;
	uint32 header[2] = {address, size};
	log.insert(log.end(), reinterpret_cast<uint8*>(header), reinterpret_cast<uint8*>(header + 2));
	log.insert(log.end(), m_vmRam + address, m_vmRam + address + size);
}

void CSpuRenderThread::OnRamRead(uint32 address, uint32 size)
{
	if(m_resyncPending) return;
	uint32 start = std::max(address, m_dirtyStart);
	uint32 end = std::min(address + size, m_dirtyEnd);
	if(start >= end) return;
	//The guest reads RAM the reverb may have written: take it from the mirror once this thread caught up
	if(m_slotOpen) Publish(false);
	Drain();
	memcpy(m_vmRam + start, m_ram.data() + start, end - start);
}

CSpuRenderThread::SLOT& CSpuRenderThread::OpenSlot()
{
	auto& slot = m_slots[m_publishedIndex % SLOT_COUNT];
	if(!m_slotOpen)
	{
		std::unique_lock lock(m_mutex);
		m_idleCondition.wait(lock, [&]() { return (m_publishedIndex - m_readIndex) < SLOT_COUNT; });
		lock.unlock();
		slot.ramLog.clear();
		m_slotOpen = true;
	}
	return slot;
}

void CSpuRenderThread::Publish(bool render)
{
	assert(m_slotOpen);
	m_slots[m_publishedIndex % SLOT_COUNT].render = render;
	m_slotOpen = false;
	{
		std::lock_guard lock(m_mutex);
		m_publishedIndex++;
	}
	m_workCondition.notify_one();
}

void CSpuRenderThread::ThreadProc()
{
	//Same floating point environment as the VM thread (reverb)
	fesetround(FE_TOWARDZERO);
	FpUtils::SetDenormalHandlingMode();
	std::unique_lock lock(m_mutex);
	while(true)
	{
		m_workCondition.wait(lock, [&]() { return m_quit || (m_readIndex != m_publishedIndex); });
		if(m_quit) break;
		auto& slot = m_slots[m_readIndex % SLOT_COUNT];
		lock.unlock();
		Process(slot);
		lock.lock();
		m_readIndex++;
		m_idleCondition.notify_all();
	}
}

void CSpuRenderThread::Process(SLOT& slot)
{
	for(size_t pos = 0; pos < slot.ramLog.size();)
	{
		uint32 header[2];
		memcpy(header, slot.ramLog.data() + pos, sizeof(header));
		pos += sizeof(header);
		if(header[1] & RAMLOG_CLEAR)
		{
			m_sampleCache.ClearRange(header[0], header[1] & ~RAMLOG_CLEAR);
		}
		else
		{
			memcpy(m_ram.data() + header[0], slot.ramLog.data() + pos, header[1]);
			pos += header[1];
		}
	}
	if(!slot.render) return;

	//Same as the single thread version of CPS2VM::UpdateSpu
	m_core0->CopyState(*slot.core0, true);
	m_core1->CopyState(*slot.core1, true);

	int16* samplesSpu0 = m_samples.data() + (m_blockSize * m_currentBlock);
	m_core0->Render(samplesSpu0, m_blockSize);

	if(m_core1->IsEnabled())
	{
		int16* samplesSpu1 = m_core1Samples.data();
		m_core1->Render(samplesSpu1, m_blockSize);

		for(unsigned int i = 0; i < m_blockSize; i++)
		{
			//Core0 output should be mixed with Core1 at volume
			//corresponding to Core1's AVOL register values
			int32 volume = (i % 2) ? m_core1->m_extInputVolR : m_core1->m_extInputVolL;
			Iop::CSpuBase::MixSamples(samplesSpu0[i], volume, &samplesSpu1[i]);
			samplesSpu0[i] = samplesSpu1[i];
		}
	}

	m_currentBlock++;
	if(m_currentBlock >= slot.blockCount)
	{
		if(slot.soundHandler)
		{
			slot.soundHandler->RecycleBuffers();
			slot.soundHandler->Write(m_samples.data(), m_blockSize * m_currentBlock, m_sampleRate);
		}
		m_currentBlock = 0;
	}
}
