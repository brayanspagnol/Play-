#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "Types.h"
#include "BlockCache.h"

class CBasicBlock;
class CMipsJitter;

//Compiling without stalling the emulation threads (WebAssembly build: EE, IOP and VU blocks).
//A new block runs its front end (the instruction compilers, cheap) on the VM thread and is then interpreted
//from those statements (Jitter::CInterpreterProgram, same results as the compiled code) while helper
//threads run the optimizer and the wasm code generator. Helpers group finished blocks in batch modules (one
//wasm module for up to 64 blocks); the thread that runs the blocks compiles and instantiates each batch on
//its next Update (wasm functions live in per-worker tables, and a blocking pthread can't receive a module
//compiled elsewhere), a few ms each. So it never waits for the Jitter. Helpers take promoted (often run)
//blocks first, then the newest.
namespace AsyncCompile
{
	enum MODE
	{
		MODE_OFF = 0,       //compile on the VM thread, as before
		MODE_ON = 1,        //interpret new blocks until a helper thread has compiled them
		MODE_INTERPRET = 2, //interpret new blocks forever (testing the interpreter)
	};

	struct JOB
	{
		std::weak_ptr<CBasicBlock> block;
		CMipsJitter* jitter = nullptr;
		BlockCache::KEY cacheKey = {};
		bool addToCache = false;
		std::thread::id owner;           //thread that runs the block: only it can put the function in its wasm table
		bool taken = false;              //helper took it (under the queue mutex)
		std::vector<uint8> code;         //wasm module, set by the helper
		double doneTime = 0;             //when the helper finished it (ms)
	};
	typedef std::shared_ptr<JOB> JobPtr;

	MODE GetMode();
	void SetMode(MODE);
	//Helper threads (default 2)
	void SetHelperCount(uint32);

	//Jitters live in a pool: a job takes one with the block's statements to the helper thread
	CMipsJitter* TakeJitter();
	void ReturnJitter(CMipsJitter*);

	//Room for more jobs: at most MAX_JOBS wait or compile (each keeps its block's statements), except hot ones
	bool HasRoom();
	void Submit(const JobPtr&, bool hot);
	//A block ran this many times in the interpreter: its job goes ahead of the others
	enum
	{
		HOT_THRESHOLD = 32,
		MAX_JOBS = 1024,
	};
	void Promote(const JobPtr&);

	//Thread that submitted jobs (VM or VU1): puts its finished blocks in place when it's worth it (see
	//AsyncCompile.cpp). Executors with a function index table pick the new function up on the block's next run.
	void Update();
	//Stats: queue sizes, helper and swap times (JSON)
	void AddSubmitTime(double ms);
	std::string GetStats();
}
