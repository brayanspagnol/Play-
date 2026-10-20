#include "Ps2VmJs.h"
#include "Jitter_CodeGen_Wasm.h"
#include "MemoryUtils.h"
#include "BasicBlock.h"
#include "PS2VM_Preferences.h"
#include "AppConfig.h"
#include "ee/Vu1Thread.h"

extern "C" uint32 LWL_Proxy(uint32, uint32, CMIPS*);
extern "C" uint32 LWR_Proxy(uint32, uint32, CMIPS*);
extern "C" uint64 LDL_Proxy(uint32, uint64, CMIPS*);
extern "C" uint64 LDR_Proxy(uint32, uint64, CMIPS*);
extern "C" void SWL_Proxy(uint32, uint32, CMIPS*);
extern "C" void SWR_Proxy(uint32, uint32, CMIPS*);
extern "C" void SDL_Proxy(uint32, uint64, CMIPS*);
extern "C" void SDR_Proxy(uint32, uint64, CMIPS*);
extern "C" void EeAutoIdleCheck(CMIPS*);
extern "C" void EeAutoIdleTaint(CMIPS*);

//Memory access helpers only reach the context through the memory map handlers, which may touch the
//ranges registered in RegisterVolatileContextRanges below and nothing else (FUNCTION_EFFECT_NONE)
static const auto EFFECT_ALL = Jitter::CWasmFunctionRegistry::FUNCTION_EFFECT_ALL;
static const auto EFFECT_NONE = Jitter::CWasmFunctionRegistry::FUNCTION_EFFECT_NONE;

static const struct
{
	uintptr_t function;
	const char* name;
	const char* signature;
	Jitter::CWasmFunctionRegistry::FUNCTION_EFFECT effect;
} g_codeGenFunctions[] =
    {
        {reinterpret_cast<uintptr_t>(&EmptyBlockHandler), "_EmptyBlockHandler", "vi", EFFECT_ALL},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetByteProxy), "_MemoryUtils_GetByteProxy", "iii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetHalfProxy), "_MemoryUtils_GetHalfProxy", "iii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetWordProxy), "_MemoryUtils_GetWordProxy", "iii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetDoubleProxy), "_MemoryUtils_GetDoubleProxy", "jii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetByteProxy), "_MemoryUtils_SetByteProxy", "viii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetHalfProxy), "_MemoryUtils_SetHalfProxy", "viii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetWordProxy), "_MemoryUtils_SetWordProxy", "viii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetDoubleProxy), "_MemoryUtils_SetDoubleProxy", "viji", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&LWL_Proxy), "_LWL_Proxy", "iiii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&LWR_Proxy), "_LWR_Proxy", "iiii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&LDL_Proxy), "_LDL_Proxy", "jiji", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&LDR_Proxy), "_LDR_Proxy", "jiji", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&SWL_Proxy), "_SWL_Proxy", "viii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&SWR_Proxy), "_SWR_Proxy", "viii", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&SDL_Proxy), "_SDL_Proxy", "viji", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&SDR_Proxy), "_SDR_Proxy", "viji", EFFECT_NONE},
        {reinterpret_cast<uintptr_t>(&EeAutoIdleCheck), "_EeAutoIdleCheck", "vi", EFFECT_ALL},
        {reinterpret_cast<uintptr_t>(&EeAutoIdleTaint), "_EeAutoIdleTaint", "vi", EFFECT_ALL},
};

//What the EE memory map handlers (I/O, write watch) may read or write in the CPU context:
//- nPC: idle loop detection in CSubSystem::IOPortReadHandler, current block in ClearActiveBlocksInRange;
//- nHasException: interrupt check and idle detection in the I/O handlers;
//- COP0 (STATUS read by IOPortWriteHandler, CPCOND0 written by the DMAC) and the performance counters;
//- COP2/VU0 state and everything after it: VIF0 writes can start VU0, which copies the COP2 state
//  (CSubSystem::Vu0StateChanged); TLB entries (address translation).
//GPRs, HI/LO, SA, COP1, nDelayedJumpAddr and cycleQuota stay promotable across these helpers.
static void RegisterVolatileContextRanges()
{
	auto add = [](size_t begin, size_t end) {
		Jitter::CCodeGen_Wasm::RegisterVolatileContextRange(static_cast<uint32>(begin), static_cast<uint32>(end - begin));
	};
	add(offsetof(CMIPS, m_State.nPC), offsetof(CMIPS, m_State.nPC) + 4);
	add(offsetof(CMIPS, m_State.nHasException), offsetof(CMIPS, m_State.nHasException) + 4);
	add(offsetof(CMIPS, m_State.nCOP0), offsetof(CMIPS, m_State.nCOP1));
	add(offsetof(CMIPS, m_State.nCOP2), sizeof(MIPSSTATE) + offsetof(CMIPS, m_State));
	//EE region base: the write watch handler makes it odd to stop a running region
	add(offsetof(CMIPS, m_jitFlagScratch[4]), offsetof(CMIPS, m_jitFlagScratch[5]));
}

//Defined by CodeGen (EM_JS): adds a function to this thread's Module.codeGenImportTable
extern "C" int RegisterExternFunction(const char*, const char*, uintptr_t);

void CPs2VmJs::CreateVM()
{
	printf("Initializing PS2VM...\r\n");

	for(const auto& function : g_codeGenFunctions)
	{
		Jitter::CWasmFunctionRegistry::RegisterFunction(function.function, function.name, function.signature, function.effect);
	}
	RegisterVolatileContextRanges();
	//Guest memory accesses: one wasm if/else per load/store with these as the slow path
	Jitter::CCodeGen_Wasm::PAGE_ACCESS_HANDLERS pageAccessHandlers;
	pageAccessHandlers.load8 = reinterpret_cast<uintptr_t>(&MemoryUtils_GetByteProxy);
	pageAccessHandlers.load16 = reinterpret_cast<uintptr_t>(&MemoryUtils_GetHalfProxy);
	pageAccessHandlers.load32 = reinterpret_cast<uintptr_t>(&MemoryUtils_GetWordProxy);
	pageAccessHandlers.load64 = reinterpret_cast<uintptr_t>(&MemoryUtils_GetDoubleProxy);
	pageAccessHandlers.store8 = reinterpret_cast<uintptr_t>(&MemoryUtils_SetByteProxy);
	pageAccessHandlers.store16 = reinterpret_cast<uintptr_t>(&MemoryUtils_SetHalfProxy);
	pageAccessHandlers.store32 = reinterpret_cast<uintptr_t>(&MemoryUtils_SetWordProxy);
	pageAccessHandlers.store64 = reinterpret_cast<uintptr_t>(&MemoryUtils_SetDoubleProxy);
	pageAccessHandlers.pageShift = __builtin_ctz(MIPS_PAGE_SIZE);
	Jitter::CCodeGen_Wasm::SetPageAccessHandlers(pageAccessHandlers);

	//Each wasm worker has its own JS Module: JIT code compiled on the VU1 thread needs the same
	//import table there, built in the same order so the indexes match.
	CVu1Thread::ThreadInitHandler = []() {
		for(const auto& function : g_codeGenFunctions)
		{
			RegisterExternFunction(function.name, function.signature, function.function);
		}
	};

	CPS2VM::CreateVM();
}

void CPs2VmJs::BootElf(std::string path)
{
	m_mailBox.SendCall([this, path]() {
		printf("Loading '%s'...\r\n", path.c_str());
		try
		{
			Reset();
			m_ee->m_os->BootFromFile(path);
		}
		catch(const std::exception& ex)
		{
			printf("Failed to start: %s.\r\n", ex.what());
			return;
		}
		printf("Starting...\r\n");
		ResumeImpl();
	});
}

void CPs2VmJs::BootDiscImage(std::string path)
{
	m_mailBox.SendCall([this, path]() {
		printf("Loading '%s'...\r\n", path.c_str());
		try
		{
			CAppConfig::GetInstance().SetPreferencePath(PREF_PS2_CDROM0_PATH, path);
			Reset();
			m_ee->m_os->BootFromCDROM();
		}
		catch(const std::exception& ex)
		{
			printf("Failed to start: %s.\r\n", ex.what());
			return;
		}
		printf("Starting...\r\n");
		ResumeImpl();
	});
}