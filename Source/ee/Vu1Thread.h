#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "Types.h"

class CVpu;
class CVif;

//Runs VU1 on its own thread (like PCSX2's MTVU).
//
//The emulation (VM) thread keeps decoding the VIF1 stream with a "front" CVif1 (registers, interrupts,
//MSKPATH3, DMA stalls), but skips everything that touches VU1: UNPACK, MPG, DIRECT and microprogram
//execution. Every qword the front consumes is pushed into a ring buffer, and a "back" CVif1 owned by this
//thread decodes the same stream again for real: it writes VU1 memory, runs microprograms and sends
//PATH1 (XGKICK) and PATH2 (DIRECT) data to the GIF. Both decoders see the same qwords in the same
//order, so their state stays in step (the front does not track ROW registers in difference mode).
//
//Sync() makes the VM thread wait until this thread has nothing left to do; it is called before the
//VM thread touches anything this thread owns (VU1 memory/registers, the GIF/GS write buffer for PATH3
//and vblank, save states, resets). The VU1 JIT blocks must be created and destroyed on this thread
//(each wasm worker has its own function table), so executor resets go through RunSync().
class CVu1Thread
{
public:
	typedef std::function<void()> FunctionType;

	CVu1Thread(CVpu&, CVif&);
	~CVu1Thread();

	//Called once on the new thread before anything else (the wasm frontend registers the JIT import table there).
	static FunctionType ThreadInitHandler;

	//Producer side (VM thread only)
	void PushVifData(const uint8*, uint32, bool);
	void PushStartMicroProgram(uint32);
	void PushFunction(FunctionType);
	void RunSync(FunctionType);

	//Waits until the VU1 thread is idle (or stalled on the GIF). Calls the flush handler first so that
	//data the front decoder has consumed but not pushed yet is pushed.
	void Sync();
	//Sync + exclusive access to the GIF/GS register write path.
	std::unique_lock<std::mutex> SyncAndLockGif();
	//Sync, then drops whatever is left (stalled data) before a reset or a state load
	void SyncAndDiscard();

	void SetFlushHandler(FunctionType);

private:
	enum MESSAGE_TYPE : uint32
	{
		MESSAGE_WRAP,
		MESSAGE_VIFDATA,
		MESSAGE_STARTMICRO,
		MESSAGE_FUNCTION,
	};

	struct MESSAGE_HEADER
	{
		uint32 type;
		uint32 size;
		uint32 param0;
		uint32 param1;
	};
	static_assert(sizeof(MESSAGE_HEADER) == 0x10);

	enum
	{
		RING_SIZE = 0x1000000,
	};

	uint8* Reserve(uint32);
	void Commit(const MESSAGE_HEADER&);
	void ThreadProc();
	bool ProcessMessage(const MESSAGE_HEADER&, uint8*);

	CVpu& m_vpu;
	CVif& m_vif;

	std::vector<uint8> m_ringStorage;
	uint8* m_ring = nullptr;
	uint64 m_writePos = 0;              //Producer only
	std::atomic<uint64> m_publishedPos = 0; //Written by producer under m_mutex
	std::atomic<uint64> m_readPos = 0;      //Written by consumer

	std::mutex m_mutex;
	std::condition_variable m_workCondition;
	std::condition_variable m_idleCondition;
	bool m_idle = true;
	bool m_stalled = false;
	bool m_currentStarted = false;
	bool m_quit = false;
	bool m_discard = false;
	std::atomic<bool> m_producerWaiting = false;

	std::mutex m_gifMutex;
	FunctionType m_flushHandler;
	std::thread m_thread;
};
