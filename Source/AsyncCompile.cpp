#include "AsyncCompile.h"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include "BasicBlock.h"
#include "MipsJitter.h"
#include "MemStream.h"
#include "Jitter_CodeGenFactory.h"
#include "Profiler.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include "WasmModuleBuilder.h"

#endif

using namespace AsyncCompile;

namespace
{
	//Finished blocks are put in place together (one wasm module): at most this many per module, and as
	//soon as this many are ready, or there's nothing left to compile, or the oldest has waited this long
	constexpr uint32 MAX_BATCH = 64; //~6 ms to instantiate on the owner thread
	constexpr uint32 MIN_BATCH = 16;
	constexpr double MAX_WAIT_MS = 2.0;

	struct BATCH
	{
		std::thread::id owner;
		std::vector<JobPtr> jobs;
		std::vector<uint8> module; //all the jobs' functions, export "f<index>" (empty: build failed)
	};

	std::atomic<MODE> g_mode = MODE_ON;

	std::mutex g_jitterMutex;
	std::vector<CMipsJitter*> g_freeJitters;

	std::mutex g_mutex;
	std::condition_variable g_condition;
	std::vector<JobPtr> g_waiting;
	std::deque<JobPtr> g_hot;       //promoted (see Promote)
	std::vector<JobPtr> g_finished; //compiled, not in a batch yet (all owners)
	std::vector<BATCH> g_batches;   //ready for their owner's Update
	std::atomic<uint32> g_batchCount = 0; //g_batches.size()
	uint32 g_helpersStarted = 0;
	std::atomic<uint32> g_helperCount = 2;
	std::atomic<uint32> g_jobCount = 0; //submitted, not compiled yet

	//Stats (GetStats)
	std::atomic<uint32> g_swapCount = 0;
	std::atomic<uint32> g_helperJobs = 0;
	std::atomic<uint32> g_newJitters = 0;
	std::atomic<double> g_helperMs = 0, g_swapMs = 0, g_swapMaxMs = 0, g_submitMs = 0, g_submitMaxMs = 0;
	void AddMs(std::atomic<double>& total, std::atomic<double>* max, double ms)
	{
		total.store(total.load(std::memory_order_relaxed) + ms, std::memory_order_relaxed);
		if(max && (ms > max->load(std::memory_order_relaxed))) max->store(ms, std::memory_order_relaxed);
	}

	double Now()
	{
#ifdef __EMSCRIPTEN__
		return emscripten_get_now();
#else
		return 0;
#endif
	}

	void Compile(const JobPtr& job)
	{
		//(Code from the block cache: only goes in a batch)
		if(!job->jitter) return;
		//(Only expired() here: the block must never be destroyed on this thread)
		if(!job->block.expired())
		{
			try
			{
				Framework::CMemStream stream;
				job->jitter->SetStream(&stream);
				job->jitter->End();
				job->code.assign(stream.GetBuffer(), stream.GetBuffer() + stream.GetSize());
				if(job->addToCache)
				{
					BlockCache::Add(job->cacheKey, job->code.data(), job->code.size());
				}
			}
			catch(const std::exception& exception)
			{
				printf("AsyncCompile: %s\r\n", exception.what());
				job->jitter->Abandon();
				job->code.clear();
			}
		}
		else
		{
			job->jitter->Abandon();
		}
		ReturnJitter(job->jitter);
		job->jitter = nullptr;
	}

	//Under g_mutex: takes the finished jobs of one owner that should go in a batch now
	bool TakeBatchJobs(std::vector<JobPtr>& jobs)
	{
		//(Nothing left to start: don't wait for a helper still busy with a big block)
		bool helpersIdle = g_waiting.empty() && g_hot.empty();
		double now = Now();
		for(const auto& candidate : g_finished)
		{
			auto owner = candidate->owner;
			uint32 count = 0;
			bool old = false;
			for(const auto& job : g_finished)
			{
				if(job->owner != owner) continue;
				count++;
				old |= (now - job->doneTime) >= MAX_WAIT_MS;
			}
			if((count < MIN_BATCH) && !helpersIdle && !old) continue;
			for(auto it = g_finished.begin(); (it != g_finished.end()) && (jobs.size() < MAX_BATCH);)
			{
				if((*it)->owner != owner)
				{
					++it;
					continue;
				}
				jobs.push_back(std::move(*it));
				it = g_finished.erase(it);
			}
			return true;
		}
		return false;
	}

	void HelperThreadProc()
	{
		std::unique_lock<std::mutex> lock(g_mutex);
		while(true)
		{
			g_condition.wait(lock, [] { return !g_waiting.empty() || !g_hot.empty(); });
			//Blocks the emulation keeps running in the interpreter first, then the newest (code that just showed
			//up is likely to run again soon)
			JobPtr job;
			if(!g_hot.empty())
			{
				job = std::move(g_hot.front());
				g_hot.pop_front();
			}
			else
			{
				job = std::move(g_waiting.back());
				g_waiting.pop_back();
			}
			//(Promoted jobs are in both lists)
			if(job->taken) continue;
			job->taken = true;
			lock.unlock();

			double startTime = Now();
			Compile(job);
			g_jobCount--;
			job->doneTime = Now();

			lock.lock();
			if(!job->code.empty()) g_finished.push_back(std::move(job));
			//Batches are built here too, so the owner only instantiates them
			std::vector<JobPtr> jobs;
			while(TakeBatchJobs(jobs))
			{
				lock.unlock();
				BATCH batch;
				batch.owner = jobs[0]->owner;
#ifdef __EMSCRIPTEN__
				std::vector<CWasmModuleBuilder::MODULE_BYTES> modules;
				for(const auto& batchJob : jobs) modules.push_back({batchJob->code.data(), static_cast<uint32>(batchJob->code.size())});
				batch.module = CWasmModuleBuilder::BuildBatchModule(modules);
#endif
				batch.jobs = std::move(jobs);
				jobs.clear();
				lock.lock();
				g_batches.push_back(std::move(batch));
				g_batchCount.store(static_cast<uint32>(g_batches.size()), std::memory_order_release);
			}
			AddMs(g_helperMs, nullptr, Now() - startTime);
			g_helperJobs++;
		}
	}
}

MODE AsyncCompile::GetMode()
{
	return g_mode.load(std::memory_order_relaxed);
}

void AsyncCompile::SetMode(MODE mode)
{
	g_mode = mode;
}

void AsyncCompile::SetHelperCount(uint32 count)
{
	//(Takes effect before the first job; helpers are never stopped)
	g_helperCount = std::max<uint32>(count, 1);
}

CMipsJitter* AsyncCompile::TakeJitter()
{
	{
		std::lock_guard<std::mutex> lock(g_jitterMutex);
		if(!g_freeJitters.empty())
		{
			auto jitter = g_freeJitters.back();
			g_freeJitters.pop_back();
			return jitter;
		}
	}
	g_newJitters++;
	return new CMipsJitter(Jitter::CreateCodeGen());
}

void AsyncCompile::ReturnJitter(CMipsJitter* jitter)
{
	std::lock_guard<std::mutex> lock(g_jitterMutex);
	//(Jobs are capped, so this keeps about every jitter ever needed: creating one costs ~0.1 ms)
	if(g_freeJitters.size() < MAX_JOBS)
	{
		g_freeJitters.push_back(jitter);
	}
	else
	{
		delete jitter;
	}
}

bool AsyncCompile::HasRoom()
{
	return g_jobCount.load(std::memory_order_relaxed) < MAX_JOBS;
}

void AsyncCompile::Submit(const JobPtr& job, bool hot)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	for(; g_helpersStarted < g_helperCount; g_helpersStarted++)
	{
		std::thread(&HelperThreadProc).detach();
	}
	job->owner = std::this_thread::get_id();
	g_jobCount++;
	if(hot)
		g_hot.push_back(job);
	else
		g_waiting.push_back(job);
	g_condition.notify_one();
}

void AsyncCompile::Promote(const JobPtr& job)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	if(job->taken) return;
	g_hot.push_back(job);
	g_condition.notify_one();
}

void AsyncCompile::Update()
{
	if(g_batchCount.load(std::memory_order_acquire) == 0) return;
	auto self = std::this_thread::get_id();
	std::vector<BATCH> batches;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		for(auto it = g_batches.begin(); it != g_batches.end();)
		{
			if(it->owner != self)
			{
				++it;
				continue;
			}
			batches.push_back(std::move(*it));
			it = g_batches.erase(it);
		}
		g_batchCount.store(static_cast<uint32>(g_batches.size()), std::memory_order_release);
	}
	if(batches.empty()) return;

#ifdef __EMSCRIPTEN__
	CTimelineTimer timer(CProfiler::GetInstance().IsWorkThread() ? TLT_COMPILE : TLT_VU1COMPILE);
	double swapStart = Now();
	uint32 swapCount = 0;
	for(const auto& batch : batches)
	{
		auto fctIds = batch.module.empty() ? std::vector<int>() : CMemoryFunction::CreateWasmBatch(batch.module.data(), batch.module.size(), static_cast<uint32>(batch.jobs.size()));
		for(uint32 i = 0; i < batch.jobs.size(); i++)
		{
			const auto& job = batch.jobs[i];
			auto function = fctIds.empty() ? CMemoryFunction(job->code.data(), job->code.size()) : CMemoryFunction::FromWasmFunction(fctIds[i], job->code.size());
			auto block = job->block.lock();
			//(A block that was dropped or recompiled since: its function goes, which frees the table slot)
			if(!block || (block->GetAsyncJob() != job)) continue;
			block->SetCompiledFunction(std::move(function));
			swapCount++;
		}
	}
	g_swapCount.fetch_add(swapCount, std::memory_order_relaxed);
	AddMs(g_swapMs, &g_swapMaxMs, Now() - swapStart);
#endif
}

void AsyncCompile::AddSubmitTime(double ms)
{
	AddMs(g_submitMs, &g_submitMaxMs, ms);
}

std::string AsyncCompile::GetStats()
{
	size_t waiting = 0, hot = 0, finished = 0, batches = 0;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		for(const auto& job : g_waiting) waiting += job->taken ? 0 : 1;
		hot = g_hot.size();
		finished = g_finished.size();
		batches = g_batches.size();
	}
	char text[512];
	snprintf(text, sizeof(text), "{\"waiting\":%zu,\"hot\":%zu,\"finished\":%zu,\"batches\":%zu,\"helperJobs\":%u,\"helperMs\":%.0f,\"swaps\":%u,\"swapMs\":%.1f,"
	                             "\"swapMaxMs\":%.1f,\"submitMs\":%.1f,\"submitMaxMs\":%.2f,\"newJitters\":%u}",
	         waiting, hot, finished, batches, g_helperJobs.load(), g_helperMs.load(), g_swapCount.load(), g_swapMs.load(),
	         g_swapMaxMs.load(), g_submitMs.load(), g_submitMaxMs.load(), g_newJitters.load());
	return text;
}
