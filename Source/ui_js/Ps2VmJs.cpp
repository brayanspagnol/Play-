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

static const struct
{
	uintptr_t function;
	const char* name;
	const char* signature;
} g_codeGenFunctions[] =
    {
        {reinterpret_cast<uintptr_t>(&EmptyBlockHandler), "_EmptyBlockHandler", "vi"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetByteProxy), "_MemoryUtils_GetByteProxy", "iii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetHalfProxy), "_MemoryUtils_GetHalfProxy", "iii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetWordProxy), "_MemoryUtils_GetWordProxy", "iii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_GetDoubleProxy), "_MemoryUtils_GetDoubleProxy", "jii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetByteProxy), "_MemoryUtils_SetByteProxy", "viii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetHalfProxy), "_MemoryUtils_SetHalfProxy", "viii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetWordProxy), "_MemoryUtils_SetWordProxy", "viii"},
        {reinterpret_cast<uintptr_t>(&MemoryUtils_SetDoubleProxy), "_MemoryUtils_SetDoubleProxy", "viji"},
        {reinterpret_cast<uintptr_t>(&LWL_Proxy), "_LWL_Proxy", "iiii"},
        {reinterpret_cast<uintptr_t>(&LWR_Proxy), "_LWR_Proxy", "iiii"},
        {reinterpret_cast<uintptr_t>(&LDL_Proxy), "_LDL_Proxy", "jiji"},
        {reinterpret_cast<uintptr_t>(&LDR_Proxy), "_LDR_Proxy", "jiji"},
        {reinterpret_cast<uintptr_t>(&SWL_Proxy), "_SWL_Proxy", "viii"},
        {reinterpret_cast<uintptr_t>(&SWR_Proxy), "_SWR_Proxy", "viii"},
        {reinterpret_cast<uintptr_t>(&SDL_Proxy), "_SDL_Proxy", "viji"},
        {reinterpret_cast<uintptr_t>(&SDR_Proxy), "_SDR_Proxy", "viji"},
};

//Defined by CodeGen (EM_JS): adds a function to this thread's Module.codeGenImportTable
extern "C" int RegisterExternFunction(const char*, const char*, uintptr_t);

void CPs2VmJs::CreateVM()
{
	printf("Initializing PS2VM...\r\n");

	for(const auto& function : g_codeGenFunctions)
	{
		Jitter::CWasmFunctionRegistry::RegisterFunction(function.function, function.name, function.signature);
	}

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