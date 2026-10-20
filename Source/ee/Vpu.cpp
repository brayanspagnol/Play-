#include "Vpu.h"
#include "make_unique.h"
#include "string_format.h"
#include "Log.h"
#include "../states/RegisterStateFile.h"
#include "../Ps2Const.h"
#include "../FrameDump.h"
#include "Vif.h"
#include "Vif1.h"
#include "GIF.h"

#define LOG_NAME ("ee_vpu")

#define STATE_PATH_REGS_FORMAT ("vpu/vpu_%d.xml")

#define STATE_REGS_VUSTATE ("vuState")
#define STATE_REGS_FBRST ("fbrst")

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <vector>
#include <algorithm>
#include "xxhash.h"
#include "VuExecutor.h"

//VU1 replay benchmark (WebAssembly debug tool): capture the inputs of the next N VU1 microprogram
//starts (registers, micro and data memory), then replay them in a tight loop with the reference
//codegen and with context promotion, timing both and checking that their results are identical.
//Requests come from the main thread; capture and replay run on the VM thread when VU1 starts a
//program. XGKICKs are suppressed during replay and the live VU1 state is restored afterwards.
extern bool g_vuContextPromotion;
extern thread_local bool g_vuDiffSuppressKick;
std::atomic<uint32> g_vuCaptureRequest = 0;
std::atomic<uint32> g_vuReplayRequest = 0;
int g_vuReplayExperiment = 0;
extern int g_vuExperiment;
bool g_vuReplayIoActive = false;
uint32 g_vuReplayTop = 0;
uint32 g_vuReplayItop = 0;
std::mutex g_vuReplayResultMutex;
std::string g_vuReplayResult;

namespace
{
	struct VU_CAPTURE
	{
		uint32 address;
		uint32 top;
		uint32 itop;
		MIPSSTATE state;
		std::vector<uint8> microMem;
		std::vector<uint8> vuMem;
	};
	std::vector<VU_CAPTURE> g_vuCaptures;
}
#endif

CVpu::CVpu(unsigned int number, const VPUINIT& vpuInit, CGIF& gif, CINTC& intc, uint8* ram, uint8* spr)
    : m_number(number)
    , m_vif((number == 0) ? std::make_unique<CVif>(0, *this, intc, ram, spr) : std::make_unique<CVif1>(1, *this, gif, intc, ram, spr))
    , m_microMem(vpuInit.microMem)
    , m_microMemSize((number == 0) ? PS2::MICROMEM0SIZE : PS2::MICROMEM1SIZE)
    , m_vuMem(vpuInit.vuMem)
    , m_vuMemSize((number == 0) ? PS2::VUMEM0SIZE : PS2::VUMEM1SIZE)
    , m_ctx(vpuInit.context)
    , m_gif(gif)
    , m_vuProfilerZone(CProfiler::GetInstance().RegisterZone("VU"))
#ifdef DEBUGGER_INCLUDED
    , m_microMemMiniState(new uint8[(number == 0) ? PS2::MICROMEM0SIZE : PS2::MICROMEM1SIZE])
    , m_vuMemMiniState(new uint8[(number == 0) ? PS2::VUMEM0SIZE : PS2::VUMEM1SIZE])
    , m_topMiniState(0)
    , m_itopMiniState(0)
#endif
{
}

CVpu::~CVpu()
{
#ifdef DEBUGGER_INCLUDED
	delete[] m_microMemMiniState;
	delete[] m_vuMemMiniState;
#endif
}

void CVpu::Execute(int32 quota)
{
	if(m_vuState != VU_STATE_RUNNING) return;

#ifdef PROFILE
	CProfilerZone profilerZone(m_vuProfilerZone);
#endif

	m_ctx->m_executor->Execute(quota);
	switch(m_ctx->m_State.nHasException)
	{
	case MIPS_EXCEPTION_VU_EBIT:
		//E bit encountered
		m_vuState = VU_STATE_READY;
		VuStateChanged(m_vuState);
		break;
	case MIPS_EXCEPTION_VU_TBIT:
	case MIPS_EXCEPTION_VU_DBIT:
		//T/D bit encountered
		{
			bool mustBreak = false;
			mustBreak |= (m_ctx->m_State.nHasException == MIPS_EXCEPTION_VU_TBIT) && (m_fbrst & FBRST_TE);
			mustBreak |= (m_ctx->m_State.nHasException == MIPS_EXCEPTION_VU_DBIT) && (m_fbrst & FBRST_DE);
			if(mustBreak)
			{
				m_vuState = VU_STATE_STOPPED;
				VuStateChanged(m_vuState);
				VuInterruptTriggered();
			}
			else
			{
				m_ctx->m_State.nHasException = 0;
			}
		}
		break;
	default:
		break;
	}
}

#ifdef DEBUGGER_INCLUDED

void CVpu::SaveMiniState()
{
	memcpy(m_microMemMiniState, m_microMem, (m_number == 0) ? PS2::MICROMEM0SIZE : PS2::MICROMEM1SIZE);
	memcpy(m_vuMemMiniState, m_vuMem, (m_number == 0) ? PS2::VUMEM0SIZE : PS2::VUMEM1SIZE);
	memcpy(&m_vuMiniState, &m_ctx->m_State, sizeof(MIPSSTATE));
	m_topMiniState = (m_number == 0) ? 0 : m_vif->GetTOP();
	m_itopMiniState = m_vif->GetITOP();
}

const MIPSSTATE& CVpu::GetVuMiniState() const
{
	return m_vuMiniState;
}

uint8* CVpu::GetVuMemoryMiniState() const
{
	return m_vuMemMiniState;
}

uint8* CVpu::GetMicroMemoryMiniState() const
{
	return m_microMemMiniState;
}

uint32 CVpu::GetVuTopMiniState() const
{
	return m_topMiniState;
}

uint32 CVpu::GetVuItopMiniState() const
{
	return m_itopMiniState;
}

#endif

void CVpu::Reset()
{
	m_vuState = VU_STATE_READY;
	m_ctx->m_executor->Reset();
	m_vif->Reset();
}

void CVpu::SaveState(Framework::CZipArchiveWriter& archive)
{
	{
		auto path = string_format(STATE_PATH_REGS_FORMAT, m_number);
		auto registerFile = std::make_unique<CRegisterStateFile>(path.c_str());
		registerFile->SetRegister32(STATE_REGS_VUSTATE, m_vuState);
		registerFile->SetRegister32(STATE_REGS_FBRST, m_fbrst);
		archive.InsertFile(std::move(registerFile));
	}

	m_vif->SaveState(archive);
}

void CVpu::LoadState(Framework::CZipArchiveReader& archive)
{
	{
		auto path = string_format(STATE_PATH_REGS_FORMAT, m_number);
		CRegisterStateFile registerFile(*archive.BeginReadFile(path.c_str()));
		m_vuState = static_cast<VU_STATE>(registerFile.GetRegister32(STATE_REGS_VUSTATE));
		m_fbrst = registerFile.GetRegister32(STATE_REGS_FBRST);
	}

	m_vif->LoadState(archive);
}

CMIPS& CVpu::GetContext() const
{
	return *m_ctx;
}

uint8* CVpu::GetMicroMemory() const
{
	return m_microMem;
}

uint32 CVpu::GetMicroMemorySize() const
{
	return m_microMemSize;
}

uint8* CVpu::GetVuMemory() const
{
	return m_vuMem;
}

uint32 CVpu::GetVuMemorySize() const
{
	return m_vuMemSize;
}

CVif& CVpu::GetVif()
{
	return *m_vif.get();
}

void CVpu::SetRunToCompletion(bool runToCompletion)
{
	m_runToCompletion = runToCompletion;
}

void CVpu::SetFbrst(uint32 fbrst)
{
	//Only keep DE and TE bits
	m_fbrst = (fbrst & (FBRST_DE | FBRST_TE));
}

void CVpu::ExecuteMicroProgram(uint32 nAddress)
{
	CLog::GetInstance().Print(LOG_NAME, "Starting microprogram execution at 0x%08X.\r\n", nAddress);

	if(m_number == 1) PROFILE_COUNT(PCNT_VIF1_MSCAL, 1);
	m_ctx->m_State.nPC = nAddress;
	m_ctx->m_State.pipeTime = 0;
	m_ctx->m_State.pipeFmacWrite[0] = {};
	m_ctx->m_State.pipeFmacWrite[1] = {};
	m_ctx->m_State.pipeFmacWrite[2] = {};
	m_ctx->m_State.savedNextBlockIntRegIdx = 0;
	m_ctx->m_State.nHasException = 0;

#ifdef DEBUGGER_INCLUDED
	SaveMiniState();
#endif

#ifdef __EMSCRIPTEN__
	if(m_number == 1)
	{
		if(g_vuReplayRequest != 0)
		{
			RunReplayBenchmark(g_vuReplayRequest.exchange(0));
		}
		if(g_vuCaptureRequest != 0)
		{
			g_vuCaptureRequest--;
			VU_CAPTURE capture;
			capture.address = nAddress;
			//Through the VU memory map: with VU1 threaded, TOP/ITOP come from the back decoder
			capture.top = m_ctx->m_pMemoryMap->GetWord(VU_ADDR_TOP);
			capture.itop = m_ctx->m_pMemoryMap->GetWord(VU_ADDR_ITOP);
			memcpy(&capture.state, &m_ctx->m_State, sizeof(MIPSSTATE));
			capture.microMem.assign(m_microMem, m_microMem + m_microMemSize);
			capture.vuMem.assign(m_vuMem, m_vuMem + m_vuMemSize);
			g_vuCaptures.push_back(std::move(capture));
			if(g_vuCaptureRequest == 0)
			{
				//Done: save them so the same programs can be replayed by another build or session
				if(FILE* file = fopen("/work/vucap.bin", "wb"))
				{
					uint32 count = static_cast<uint32>(g_vuCaptures.size());
					fwrite(&count, 4, 1, file);
					for(const auto& saved : g_vuCaptures)
					{
						fwrite(&saved.address, 4, 1, file);
						fwrite(&saved.top, 4, 1, file);
						fwrite(&saved.itop, 4, 1, file);
						fwrite(&saved.state, sizeof(MIPSSTATE), 1, file);
						fwrite(saved.microMem.data(), 1, saved.microMem.size(), file);
						fwrite(saved.vuMem.data(), 1, saved.vuMem.size(), file);
					}
					fclose(file);
				}
			}
		}
	}
#endif

	assert(m_vuState != VU_STATE_RUNNING);
	m_vuState = VU_STATE_RUNNING;
	VuStateChanged(m_vuState);
	for(unsigned int i = 0; m_runToCompletion || (i < 100); i++)
	{
		Execute(5000);
		if(m_vuState != VU_STATE_RUNNING) break;
	}
}

#ifdef __EMSCRIPTEN__
void CVpu::ClearReplayCaptures()
{
	g_vuCaptures.clear();
}

uint32 CVpu::GetReplayCaptureCount()
{
	return static_cast<uint32>(g_vuCaptures.size());
}

void CVpu::RunReplayBenchmark(uint32 iterations)
{
	if(g_vuCaptures.empty())
	{
		//Captures saved by an earlier session (see ExecuteMicroProgram)
		if(FILE* file = fopen("/work/vucap.bin", "rb"))
		{
			uint32 count = 0;
			fread(&count, 4, 1, file);
			for(uint32 i = 0; i < count; i++)
			{
				VU_CAPTURE capture;
				capture.microMem.resize(m_microMemSize);
				capture.vuMem.resize(m_vuMemSize);
				fread(&capture.address, 4, 1, file);
				fread(&capture.top, 4, 1, file);
				fread(&capture.itop, 4, 1, file);
				fread(&capture.state, sizeof(MIPSSTATE), 1, file);
				fread(capture.microMem.data(), 1, m_microMemSize, file);
				fread(capture.vuMem.data(), 1, m_vuMemSize, file);
				g_vuCaptures.push_back(std::move(capture));
			}
			fclose(file);
		}
	}
	if(g_vuCaptures.empty()) return;

	MIPSSTATE liveState;
	memcpy(&liveState, &m_ctx->m_State, sizeof(MIPSSTATE));
	std::vector<uint8> liveMicroMem(m_microMem, m_microMem + m_microMemSize);
	std::vector<uint8> liveVuMem(m_vuMem, m_vuMem + m_vuMemSize);
	bool livePromotion = g_vuContextPromotion;
	bool liveSuperBlocks = g_vuSuperBlocks;

	//Keep captures that share a microprogram together: switching programs invalidates blocks
	std::stable_sort(g_vuCaptures.begin(), g_vuCaptures.end(), [](const VU_CAPTURE& a, const VU_CAPTURE& b) {
		return XXH64(a.microMem.data(), a.microMem.size(), 0) < XXH64(b.microMem.data(), b.microMem.size(), 0);
	});
	auto runCapture = [&](const VU_CAPTURE& capture, bool withHash) -> uint64 {
		memcpy(&m_ctx->m_State, &capture.state, sizeof(MIPSSTATE));
		if(memcmp(m_microMem, capture.microMem.data(), m_microMemSize) != 0)
		{
			memcpy(m_microMem, capture.microMem.data(), m_microMemSize);
			InvalidateMicroProgram();
		}
		memcpy(m_vuMem, capture.vuMem.data(), m_vuMemSize);
		m_ctx->m_State.nPC = capture.address;
		g_vuReplayTop = capture.top;
		g_vuReplayItop = capture.itop;
		for(uint32 i = 0; i < 1000; i++)
		{
			m_ctx->m_executor->Execute(5000);
			uint32 exception = m_ctx->m_State.nHasException;
			if((exception == MIPS_EXCEPTION_VU_EBIT) || (exception == MIPS_EXCEPTION_VU_TBIT) || (exception == MIPS_EXCEPTION_VU_DBIT)) break;
		}
		if(!withHash) return 0;
		//Result fingerprint: registers and data memory
		XXH64_state_t* hashState = XXH64_createState();
		XXH64_reset(hashState, 0);
		XXH64_update(hashState, &m_ctx->m_State, sizeof(MIPSSTATE));
		XXH64_update(hashState, m_vuMem, m_vuMemSize);
		uint64 hash = XXH64_digest(hashState);
		XXH64_freeState(hashState);
		return hash;
	};

	std::string result;
	std::vector<uint64> refHashes;
	g_vuDiffSuppressKick = true;
	g_vuReplayIoActive = true;
	int liveExperiment = g_vuExperiment;
	//Variants alternate (reference, promoted, promoted + superblocks, reference, ...). Each starts from
	//freshly instantiated blocks, so it first runs for a while to let the engine tier the code up.
	//(With experiment bits set, the third variant is "promoted + experiment" instead of superblocks)
	const char* variantNames[3] = {"reference", "promoted", (g_vuReplayExperiment != 0) ? "experiment" : "superblocks"};
	for(int round = 0; round < 9; round++)
	{
		int variant = round % 3;
		g_vuContextPromotion = (variant != 0);
		g_vuSuperBlocks = (variant == 2) && (g_vuReplayExperiment == 0);
		g_vuExperiment = (variant == 2) ? g_vuReplayExperiment : 0;
		static_cast<CVuExecutor*>(m_ctx->m_executor.get())->DropBlocks();
		std::vector<uint64> hashes;
		for(const auto& capture : g_vuCaptures) hashes.push_back(runCapture(capture, true));
		double warmStart = emscripten_get_now();
		while(emscripten_get_now() - warmStart < 1500.0)
		{
			for(const auto& capture : g_vuCaptures) runCapture(capture, false);
		}
#ifdef PROFILE
		extern std::atomic<uint32> g_vuBlockExecCount;
		uint32 blocksBefore = g_vuBlockExecCount;
#endif
		double start = emscripten_get_now();
		for(uint32 i = 0; i < iterations; i++)
		{
			for(const auto& capture : g_vuCaptures) runCapture(capture, false);
		}
		double elapsed = emscripten_get_now() - start;
#ifdef PROFILE
		result += string_format("blocks/program %.1f\n", static_cast<double>(g_vuBlockExecCount - blocksBefore) / (iterations * g_vuCaptures.size()));
#endif
		uint32 mismatches = 0;
		if(round == 0)
		{
			refHashes = hashes;
		}
		else
		{
			for(size_t i = 0; i < hashes.size(); i++) mismatches += (hashes[i] != refHashes[i]) ? 1 : 0;
		}
		result += string_format("%s: %.1f us/program (%u programs x %u), mismatches %u\n",
		                        variantNames[variant], elapsed * 1000.0 / (iterations * g_vuCaptures.size()),
		                        static_cast<uint32>(g_vuCaptures.size()), iterations, mismatches);
	}
	g_vuDiffSuppressKick = false;
	g_vuReplayIoActive = false;
	g_vuExperiment = liveExperiment;

	//Back to the live program
	g_vuContextPromotion = livePromotion;
	g_vuSuperBlocks = liveSuperBlocks;
	memcpy(&m_ctx->m_State, &liveState, sizeof(MIPSSTATE));
	memcpy(m_microMem, liveMicroMem.data(), m_microMemSize);
	memcpy(m_vuMem, liveVuMem.data(), m_vuMemSize);
	static_cast<CVuExecutor*>(m_ctx->m_executor.get())->DropBlocks();

	std::lock_guard<std::mutex> lock(g_vuReplayResultMutex);
	g_vuReplayResult = result;
}
#endif

void CVpu::InvalidateMicroProgram()
{
	m_ctx->m_executor->ClearActiveBlocksInRange(0, (m_number == 0) ? PS2::MICROMEM0SIZE : PS2::MICROMEM1SIZE, false);
}

void CVpu::InvalidateMicroProgram(uint32 start, uint32 end)
{
	m_ctx->m_executor->ClearActiveBlocksInRange(start, end, false);
}

void CVpu::ProcessXgKick(uint32 address)
{
	PROFILE_COUNT(PCNT_VU1_XGKICK, 1);
	address &= 0x3FF;
	address *= 0x10;

	CGsPacketMetadata metadata;
	metadata.pathIndex = 1;
#ifdef DEBUGGER_INCLUDED
	metadata.vuMemPacketAddress = address;
	metadata.vpu1Top = GetVuTopMiniState();
	metadata.vpu1Itop = GetVuItopMiniState();
	memcpy(&metadata.vu1State, &GetVuMiniState(), sizeof(MIPSSTATE));
	memcpy(metadata.vuMem1, GetVuMemoryMiniState(), PS2::VUMEM1SIZE);
	memcpy(metadata.microMem1, GetMicroMemoryMiniState(), PS2::MICROMEM1SIZE);
#endif

	address += m_gif.ProcessSinglePacket(GetVuMemory(), PS2::VUMEM1SIZE, address, PS2::VUMEM1SIZE, metadata);
	if((address == PS2::VUMEM1SIZE) && (m_gif.GetActivePath() == 1))
	{
		address = 0;
		address += m_gif.ProcessSinglePacket(GetVuMemory(), PS2::VUMEM1SIZE, address, PS2::VUMEM1SIZE, metadata);
	}
	assert(m_gif.GetActivePath() == 0);

#ifdef DEBUGGER_INCLUDED
	SaveMiniState();
#endif
}
