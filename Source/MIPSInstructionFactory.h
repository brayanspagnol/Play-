#pragma once

#include "Types.h"
#include "MipsJitter.h"

class CMIPS;

enum MIPS_REGSIZE
{
	MIPS_REGSIZE_32 = 0,
	MIPS_REGSIZE_64 = 1,
};

enum MIPS_BRANCH_TYPE
{
	MIPS_BRANCH_NONE = 0,
	MIPS_BRANCH_NORMAL = 1,
	MIPS_BRANCH_NODELAY = 2,
};

class CMIPSInstructionFactory
{
public:
	CMIPSInstructionFactory(MIPS_REGSIZE);
	virtual ~CMIPSInstructionFactory() = default;
	virtual void CompileInstruction(uint32, CMipsJitter*, CMIPS*, uint32) = 0;
	virtual void SetCompileHints(uint32);
	void Illegal();

protected:
	void ComputeMemAccessAddr();
	void ComputeMemAccessAddrNoXlat();
	void ComputeMemAccessRefIdx(uint32, bool forWrite = false);
	void ComputeMemAccessPageRef(bool forWrite = false);
	//Single-statement page table accesses (LoadFromPage/StoreAtPage), when the code generator has them
	//kind: debug bit for g_jitPageAccess (1 load, 2 store, 4/8 LWC1/SWC1, 16/32 LD/SD, 64/128 LQC2/SQC2, 256/512 LQ/SQ)
	bool UsePageAccess(uint32 kind) const;
	//Pushes the page table ref and the effective address
	void PushMemAccessPage(bool forWrite = false);

	void CheckTLBExceptions(bool);
	void CheckTrap();
	void Branch(Jitter::CONDITION);
	void BranchLikely(Jitter::CONDITION);

	void SetupQuickVariables(uint32, CMipsJitter*, CMIPS*, uint32);

	CMipsJitter* m_codeGen = nullptr;
	CMIPS* m_pCtx = nullptr;
	uint32 m_nOpcode = 0;
	uint32 m_nAddress = 0;
	uint32 m_instrPosition = 0;
	uint32 m_compileHints = 0;
	MIPS_REGSIZE m_regSize;
};
