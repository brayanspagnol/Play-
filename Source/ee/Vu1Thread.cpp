#include <cassert>
#include <cstring>
#include <chrono>
#include <future>
#include "Vu1Thread.h"
#include "Vpu.h"
#include "Vif.h"
#include "../Profiler.h"
#include "../FpUtils.h"
#include <cfenv>

#ifdef PROFILE
std::atomic<uint64> g_vu1ThreadBusyNs;
#endif

CVu1Thread::FunctionType CVu1Thread::ThreadInitHandler;

static uint32 AlignSize(uint32 size)
{
	return (size + 0xF) & ~0xF;
}

CVu1Thread::CVu1Thread(CVpu& vpu, CVif& vif)
    : m_vpu(vpu)
    , m_vif(vif)
{
	m_ringStorage.resize(RING_SIZE + 0x10);
	m_ring = reinterpret_cast<uint8*>((reinterpret_cast<uintptr_t>(m_ringStorage.data()) + 0xF) & ~static_cast<uintptr_t>(0xF));
	m_thread = std::thread([this]() { ThreadProc(); });
}

CVu1Thread::~CVu1Thread()
{
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_workCondition.notify_all();
	m_thread.join();
	//Drop pending function messages
	while(m_readPos != m_publishedPos)
	{
		uint32 offset = m_readPos % RING_SIZE;
		auto header = *reinterpret_cast<const MESSAGE_HEADER*>(m_ring + offset);
		if(header.type == MESSAGE_WRAP)
		{
			m_readPos += RING_SIZE - offset;
			continue;
		}
		if(header.type == MESSAGE_FUNCTION)
		{
			delete *reinterpret_cast<FunctionType**>(m_ring + offset + sizeof(MESSAGE_HEADER));
		}
		m_readPos += sizeof(MESSAGE_HEADER) + AlignSize(header.size);
	}
}

void CVu1Thread::SetFlushHandler(FunctionType flushHandler)
{
	m_flushHandler = std::move(flushHandler);
}

uint8* CVu1Thread::Reserve(uint32 payloadSize)
{
	uint32 total = sizeof(MESSAGE_HEADER) + AlignSize(payloadSize);
	assert(total <= (RING_SIZE / 2));
	uint32 offset = m_writePos % RING_SIZE;
	bool wrap = (offset + total) > RING_SIZE;
	uint32 needed = total + (wrap ? (RING_SIZE - offset) : 0);
	if((RING_SIZE - (m_writePos - m_readPos)) < needed)
	{
		//Ring full: wait for the consumer to free some space
		std::unique_lock lock(m_mutex);
		m_producerWaiting = true;
		m_idleCondition.wait(lock, [&]() { return (RING_SIZE - (m_writePos - m_readPos)) >= needed; });
		m_producerWaiting = false;
	}
	if(wrap)
	{
		auto header = reinterpret_cast<MESSAGE_HEADER*>(m_ring + offset);
		header->type = MESSAGE_WRAP;
		header->size = 0;
		m_writePos += RING_SIZE - offset;
		offset = 0;
	}
	return m_ring + offset;
}

void CVu1Thread::Commit(const MESSAGE_HEADER& header)
{
	uint32 offset = m_writePos % RING_SIZE;
	*reinterpret_cast<MESSAGE_HEADER*>(m_ring + offset) = header;
	m_writePos += sizeof(MESSAGE_HEADER) + AlignSize(header.size);
	{
		std::lock_guard lock(m_mutex);
		m_publishedPos = m_writePos;
		m_idle = false;
		m_stalled = false;
	}
	m_workCondition.notify_one();
}

void CVu1Thread::PushVifData(const uint8* data, uint32 size, bool tagIncluded)
{
	auto message = Reserve(size);
	memcpy(message + sizeof(MESSAGE_HEADER), data, size);
	Commit({MESSAGE_VIFDATA, size, tagIncluded ? 1U : 0U, 0});
}

void CVu1Thread::PushStartMicroProgram(uint32 address)
{
	Reserve(0);
	Commit({MESSAGE_STARTMICRO, 0, address, 0});
}

void CVu1Thread::PushFunction(FunctionType function)
{
	auto message = Reserve(sizeof(FunctionType*));
	*reinterpret_cast<FunctionType**>(message + sizeof(MESSAGE_HEADER)) = new FunctionType(std::move(function));
	Commit({MESSAGE_FUNCTION, sizeof(FunctionType*), 0, 0});
}

void CVu1Thread::RunSync(FunctionType function)
{
	if(m_flushHandler) m_flushHandler();
	std::promise<void> done;
	auto future = done.get_future();
	PushFunction([&]() { function(); done.set_value(); });
	future.wait();
	Sync();
}

void CVu1Thread::Sync()
{
	if(m_flushHandler) m_flushHandler();
	std::unique_lock lock(m_mutex);
	auto isDone = [&]() { return m_idle && (m_stalled || (m_readPos == m_publishedPos)); };
	if(isDone()) return;
	PROFILE_COUNT(PCNT_VU1_SYNC, 1);
#ifdef PROFILE
	static const auto syncZone = CProfiler::GetInstance().RegisterZone("VU1SYNC");
	CProfilerZone profilerZone(syncZone);
#endif
	m_idleCondition.wait(lock, isDone);
}

std::unique_lock<std::mutex> CVu1Thread::SyncAndLockGif()
{
	Sync();
	return std::unique_lock<std::mutex>(m_gifMutex);
}

void CVu1Thread::SyncAndDiscard()
{
	Sync();
	std::unique_lock lock(m_mutex);
	if(m_readPos == m_publishedPos) return;
	m_discard = true;
	m_workCondition.notify_all();
	m_idleCondition.wait(lock, [&]() { return !m_discard; });
}

bool CVu1Thread::ProcessMessage(const MESSAGE_HEADER& header, uint8* payload)
{
	switch(header.type)
	{
	case MESSAGE_VIFDATA:
		if(!m_currentStarted)
		{
			m_vif.BeginForwardedData(payload, header.size, header.param0 != 0);
			m_currentStarted = true;
		}
		if(!m_vif.ProcessForwardedData())
		{
			return false;
		}
		m_currentStarted = false;
		break;
	case MESSAGE_STARTMICRO:
		m_vpu.ExecuteMicroProgram(header.param0);
		break;
	case MESSAGE_FUNCTION:
	{
		auto function = *reinterpret_cast<FunctionType**>(payload);
		(*function)();
		delete function;
	}
	break;
	default:
		assert(false);
		break;
	}
	return true;
}

void CVu1Thread::ThreadProc()
{
	//Same FP environment as the VM thread
	fesetround(FE_TOWARDZERO);
	FpUtils::SetDenormalHandlingMode();
	if(ThreadInitHandler) ThreadInitHandler();
	std::unique_lock lock(m_mutex);
	while(!m_quit)
	{
		if(m_discard)
		{
			while(m_readPos != m_publishedPos)
			{
				uint32 offset = m_readPos % RING_SIZE;
				auto header = *reinterpret_cast<const MESSAGE_HEADER*>(m_ring + offset);
				if(header.type == MESSAGE_WRAP)
				{
					m_readPos += RING_SIZE - offset;
					continue;
				}
				if(header.type == MESSAGE_FUNCTION)
				{
					delete *reinterpret_cast<FunctionType**>(m_ring + offset + sizeof(MESSAGE_HEADER));
				}
				m_readPos += sizeof(MESSAGE_HEADER) + AlignSize(header.size);
			}
			m_currentStarted = false;
			m_discard = false;
			m_idleCondition.notify_all();
			continue;
		}
		uint64 published = m_publishedPos;
		if(m_readPos == published)
		{
			m_idle = true;
			m_stalled = false;
			m_idleCondition.notify_all();
			m_workCondition.wait(lock);
			continue;
		}
		m_idle = false;
		m_stalled = false;
		lock.unlock();
		bool stalled = false;
		{
#ifdef PROFILE
			auto busyStart = std::chrono::steady_clock::now();
#endif
			std::lock_guard gifLock(m_gifMutex);
			uint64 readPos = m_readPos;
			while(readPos != published)
			{
				uint32 offset = readPos % RING_SIZE;
				auto header = *reinterpret_cast<const MESSAGE_HEADER*>(m_ring + offset);
				if(header.type != MESSAGE_WRAP)
				{
					if(!ProcessMessage(header, m_ring + offset + sizeof(MESSAGE_HEADER)))
					{
						stalled = true;
						break;
					}
					readPos += sizeof(MESSAGE_HEADER) + AlignSize(header.size);
				}
				else
				{
					readPos += RING_SIZE - offset;
				}
				m_readPos = readPos;
				if(m_producerWaiting)
				{
					std::lock_guard waitLock(m_mutex);
					m_idleCondition.notify_all();
				}
			}
#ifdef PROFILE
			g_vu1ThreadBusyNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - busyStart).count();
#endif
		}
		lock.lock();
		if(stalled)
		{
			//Waiting on the GIF (PATH3 packet in progress or SIGNAL not acknowledged yet): let the
			//VM thread go on, retry when new data comes in or a bit later.
			m_idle = true;
			m_stalled = true;
			m_idleCondition.notify_all();
			m_workCondition.wait_for(lock, std::chrono::milliseconds(1));
		}
	}
}
