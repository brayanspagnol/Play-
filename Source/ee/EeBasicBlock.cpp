#include "EeBasicBlock.h"
#include "offsetof_def.h"
#include <algorithm>
#include "xxhash.h"
#include <map>
#include <string>
#include <cstdio>

void CEeBasicBlock::SetFpRoundingMode(Jitter::CJitter::ROUNDINGMODE fpRoundingMode)
{
	m_fpRoundingMode = fpRoundingMode;
	AddCacheVariant((static_cast<uint32>(fpRoundingMode) + 1) << 8);
}

void CEeBasicBlock::SetIsIdleLoopBlock()
{
	m_isIdleLoopBlock = true;
	AddCacheVariant(1 << 16);
}

//Wait loop found by CEeExecutor's analysis: when the branch back to the loop head is taken,
//EeAutoIdleCheck decides whether the EE is idle (see EeExecutor.cpp)
void CEeBasicBlock::SetIsAutoIdleLoopBlock()
{
	m_isAutoIdleLoopBlock = true;
	AddCacheVariant(1 << 17);
}

//Start of a path that a wait loop's analysis found unsafe: calls EeAutoIdleTaint when it runs
void CEeBasicBlock::SetIsAutoIdleTaintBlock()
{
	m_isAutoIdleTaintBlock = true;
	AddCacheVariant(1 << 18);
}

extern "C" void EeAutoIdleCheck(CMIPS*);
extern "C" void EeAutoIdleTaint(CMIPS*);

void CEeBasicBlock::CompileProlog(CMipsJitter* jitter)
{
	if(m_fpRoundingMode != DEFAULT_FP_ROUNDING_MODE)
	{
		jitter->FP_SetRoundingMode(m_fpRoundingMode);
	}
	if(m_isAutoIdleTaintBlock)
	{
		jitter->PushCtx();
		jitter->Call(reinterpret_cast<void*>(&EeAutoIdleTaint), 1, Jitter::CJitter::RETURN_VALUE_NONE);
	}
	CBasicBlock::CompileProlog(jitter);
}

void CEeBasicBlock::CompileEpilog(CMipsJitter* jitter, bool loopsOnItself)
{
	if(m_fpRoundingMode != DEFAULT_FP_ROUNDING_MODE)
	{
		jitter->FP_SetRoundingMode(DEFAULT_FP_ROUNDING_MODE);
	}

	if(m_isIdleLoopBlock || IsCodeIdleLoopBlock(m_context, m_begin, m_end))
	{
		jitter->PushCst(MIPS_EXCEPTION_IDLE);
		jitter->PullRel(offsetof(CMIPS, m_State.nHasException));
	}
	else if(m_isAutoIdleLoopBlock)
	{
		jitter->PushCst(MIPS_INVALID_PC);
		jitter->PushRel(offsetof(CMIPS, m_State.nDelayedJumpAddr));
		jitter->BeginIf(Jitter::CONDITION_NE);
		{
			jitter->PushCtx();
			jitter->Call(reinterpret_cast<void*>(&EeAutoIdleCheck), 1, Jitter::CJitter::RETURN_VALUE_NONE);
		}
		jitter->EndIf();
	}

	CBasicBlock::CompileEpilog(jitter, loopsOnItself);
}

bool CEeBasicBlock::IsCodeIdleLoopBlock(CMIPS& context, uint32 begin, uint32 end)
{
	enum OP
	{
		OP_BEQ = 0x04,
		OP_BNE = 0x05,
		OP_ADDIU = 0x09,
		OP_SLTIU = 0x0B,
		OP_ANDI = 0x0C,
		OP_XORI = 0x0E,
		OP_LUI = 0x0F,
		OP_LQ = 0x1E,
		OP_LW = 0x23,
		OP_LBU = 0x24,
	};

	enum
	{
		OP_SPECIAL_SLL = 0x00,
		OP_SPECIAL_ADDU = 0x21,
		OP_SPECIAL_SLT = 0x2A,
		OP_SPECIAL_SLTU = 0x2B,
		OP_SPECIAL_DSLL32 = 0x3C,
		OP_SPECIAL_DSRL32 = 0x3E,
	};

	uint32 endInstructionAddress = end - 4;
	uint32 endInstruction = context.m_pMemoryMap->GetWord(endInstructionAddress);

	//We need a branch at the end of the block
	auto branchType = context.m_pArch->IsInstructionBranch(&context, endInstructionAddress, endInstruction);
	if(branchType != MIPS_BRANCH_NORMAL) return false;

	//Check that the branch target is ourself
	uint32 branchTarget = context.m_pArch->GetInstructionEffectiveAddress(&context, endInstructionAddress, endInstruction);
	if(branchTarget == MIPS_INVALID_PC) return false;
	if(branchTarget != begin) return false;

	uint32 compareRs = 0;
	uint32 compareRt = 0;

	//Check what kind of branching instruction we have.
	{
		uint32 op = (endInstruction >> 26) & 0x3F;
		uint32 rt = (endInstruction >> 16) & 0x1F;
		uint32 rs = (endInstruction >> 21) & 0x1F;

		//We want a BEQ or BNE
		if((op != OP_BEQ) && (op != OP_BNE)) return false;

		compareRs = rs;
		compareRt = rt;
	}

	uint32 defState = 0; //Set of completely new definitions of registers within this block
	uint32 useState = 0; //Set of previous state usage within this block

	//Check all instructions inside to see if we can prove it's waiting for some kind of flag
	for(uint32 address = begin; address <= end; address += 4)
	{
		//Don't check branch instruction as we've checked it already
		if(address == endInstructionAddress) continue;

		uint32 inst = context.m_pMemoryMap->GetWord(address);
		if(inst == 0) continue;
		uint32 special = inst & 0x3F;
		uint32 rd = (inst >> 11) & 0x1F;
		uint32 rt = (inst >> 16) & 0x1F;
		uint32 rs = (inst >> 21) & 0x1F;
		uint32 op = (inst >> 26) & 0x3F;

		uint32 newDef = 0;
		uint32 newUse = 0;

		switch(op)
		{
		case 0x00:
			switch(special)
			{
			case OP_SPECIAL_SLL:
			case OP_SPECIAL_DSLL32:
			case OP_SPECIAL_DSRL32:
				newUse = (1 << rt);
				newDef = (1 << rd);
				break;
			case OP_SPECIAL_ADDU:
			case OP_SPECIAL_SLT:
			case OP_SPECIAL_SLTU:
				newUse = (1 << rs) | (1 << rt);
				newDef = (1 << rd);
				break;
			default:
				//We don't know what this does, let's not take a chance
				return false;
			}
			break;
		case OP_LUI:
			newDef = (1 << rt);
			break;
		case OP_LW:
		case OP_LQ:
		case OP_LBU:
		case OP_ADDIU:
		case OP_SLTIU:
		case OP_XORI:
			newUse = (1 << rs);
			newDef = (1 << rt);
			break;
		default:
			//We don't know what this does, let's not take a chance
			return false;
		}

		//Remove uses from defs within this block
		newUse &= ~defState;
		useState |= newUse;

		//Bail if this defines any state that we previously used
		if(useState & newDef)
		{
			return false;
		}

		defState |= newDef;
	}

	//Just make sure that we've at least defined our comparision register
	FRAMEWORK_MAYBE_UNUSED bool compareRsDefined = defState & (1 << compareRs);
	FRAMEWORK_MAYBE_UNUSED bool compareRtDefined = defState & (1 << compareRt);

	assert(compareRsDefined || compareRtDefined);

	return true;
}

void CEeBasicBlock::SetRegionParts(std::vector<SUPERBLOCK_PART> parts)
{
	m_regionParts = std::move(parts);
	m_cacheSalt = GetRegionLayoutHash(m_regionParts, m_begin);
}

bool CEeBasicBlock::IsSuperBlock() const
{
	return !m_regionParts.empty();
}

uint64 CEeBasicBlock::GetRegionLayoutHash(const std::vector<SUPERBLOCK_PART>& parts, uint32 begin)
{
	if(parts.empty()) return 0;
	//Region code has the part addresses as constants (constant PC): the layout is absolute, plus a format version
	extern bool g_eeRegionCount;
	std::vector<uint32> layout = {begin, REGION_CODE_VERSION, g_eeRegionCount ? 1U : 0U};
	for(const auto& part : parts)
	{
		for(uint32 address : {part.begin, part.end, part.successors[0], part.successors[1]})
		{
			layout.push_back(address);
		}
	}
	return XXH3_64bits(layout.data(), layout.size() * sizeof(uint32));
}

void CEeBasicBlock::GetCodeWords(std::vector<uint32>& words) const
{
	if(m_regionParts.empty())
	{
		CBasicBlock::GetCodeWords(words);
		return;
	}
	//Words between the parts (often data) don't change the code
	words.clear();
	for(const auto& part : m_regionParts)
	{
		for(uint32 address = part.begin; address <= part.end; address += 4)
		{
			words.push_back(m_context.m_pMemoryMap->GetInstruction(address));
		}
	}
}

//Whether running these instructions may call a helper (memory access slow paths, COP0/COP1/COP2,
//MMI helpers...) or raise an exception. Plain integer arithmetic and branches don't.
bool CEeBasicBlock::MayCallOrRaise(CMIPS& context, uint32 begin, uint32 end)
{
	for(uint32 address = begin; address <= end; address += 4)
	{
		uint32 opcode = context.m_pMemoryMap->GetInstruction(address);
		uint32 op = opcode >> 26;
		uint32 rs = (opcode >> 21) & 0x1F;
		uint32 rt = (opcode >> 16) & 0x1F;
		uint32 funct = opcode & 0x3F;
		switch(op)
		{
		case 0x00:
			//SYSCALL, BREAK, MFSA/MTSA, traps, and unused codes may call or raise
			if((funct == 0x0C) || (funct == 0x0D) || (funct == 0x01) || (funct == 0x05) || (funct == 0x0E) ||
			   (funct == 0x15) || (funct == 0x1C) || (funct == 0x1D) || (funct == 0x1E) || (funct == 0x1F) ||
			   (funct == 0x28) || (funct == 0x29) || ((funct >= 0x30) && (funct <= 0x37)) || (funct == 0x39) ||
			   (funct == 0x3D))
			{
				return true;
			}
			break;
		case 0x01:
			//Branches only (no traps, no MTSAB/MTSAH)
			if(!((rt <= 0x03) || ((rt >= 0x10) && (rt <= 0x13)))) return true;
			break;
		case 0x09:
			//ADDIU R0, R0, x: syscall hack (see CMA_MIPSIV::ADDIU)
			if((rs == 0) && (rt == 0)) return true;
			break;
		case 0x02:
		case 0x03:
		case 0x04:
		case 0x05:
		case 0x06:
		case 0x07:
		case 0x08:
		case 0x0A:
		case 0x0B:
		case 0x0C:
		case 0x0D:
		case 0x0E:
		case 0x0F:
		case 0x14:
		case 0x15:
		case 0x16:
		case 0x17:
		case 0x18:
		case 0x19:
			break;
		default:
			return true;
		}
	}
	return false;
}

void CEeBasicBlock::CompileRange(CMipsJitter* jitter)
{
	if(m_regionParts.empty())
	{
		CBasicBlock::CompileRange(jitter);
		return;
	}

	//Region (tier 2): one function, a general control flow graph structured by the wasm backend.
	//  m_jitFlagScratch[4] = entry (an invalidation while the region runs makes it odd, see CEeExecutor)
	//  for each part (parts[0] is the entry):
	//    part label: [nPC = part start, if the part may call a helper: helpers read it]
	//      the part's instructions with nPC as a compile time constant (part start)
	//      cycleQuota -= part size
	//      successor = nDelayedJumpAddr != INVALID ? branch target : fall through; nDelayedJumpAddr = INVALID
	//      if(cycleQuota <= 0) { nHasException |= QUOTADONE; exit to successor }
	//      if(the part may call or raise && (nHasException || invalidated)) exit to successor
	//      goto the successor's part label, or exit to it
	//  exit to T: nPC = T, return
	//This is what the tier-1 blocks do between dispatches, without the context round trips: parts run with
	//promoted symbols (GPRs, HI/LO, cycle quota, ...) in locals, and nPC is only written where it is read.
	const size_t regionOffset = offsetof(CMIPS, m_jitFlagScratch[4]);
	const size_t pcOffset = offsetof(CMIPS, m_State.nPC);
	const size_t delayedJumpOffset = offsetof(CMIPS, m_State.nDelayedJumpAddr);
	uint32 savedBegin = m_begin;
	uint32 savedEnd = m_end;

	const auto& parts = m_regionParts;
	assert(parts[0].begin == savedBegin);
	{
		//(Block dumps list the parts)
		extern thread_local std::string g_superBlockPartsText;
		g_superBlockPartsText.clear();
		for(const auto& part : parts)
		{
			char line[64];
			snprintf(line, sizeof(line), "PART %08X %08X\n", part.begin, part.end);
			g_superBlockPartsText += line;
		}
	}

	jitter->PushCst(savedBegin);
	jitter->PullRel(regionOffset);
	extern bool g_eeRegionCount;
	auto count = [&](uint32 index, uint32 value) {
		if(!g_eeRegionCount) return;
		jitter->PushRel(offsetof(CMIPS, m_jitFlagScratch[index]));
		jitter->PushCst(value);
		jitter->Add();
		jitter->PullRel(offsetof(CMIPS, m_jitFlagScratch[index]));
	};
	count(5, 1);

	std::map<uint32, CMipsJitter::LABEL> partLabels;
	for(const auto& part : parts)
	{
		partLabels[part.begin] = jitter->CreateLabel();
	}
	std::map<uint32, CMipsJitter::LABEL> exitLabels;
	auto exitLabel = [&](uint32 target) {
		auto exitIterator = exitLabels.find(target);
		if(exitIterator != exitLabels.end()) return exitIterator->second;
		auto label = jitter->CreateLabel();
		exitLabels[target] = label;
		return label;
	};
	auto dynamicExitLabel = jitter->CreateLabel();
	bool usesDynamicExit = false;
	auto endLabel = jitter->CreateLabel();

	m_context.m_pArch->SetCompileHints(m_blockCompileHints);

	for(const auto& part : parts)
	{
		jitter->MarkLabel(partLabels[part.begin]);

		m_begin = part.begin;
		m_end = part.end;
		bool mayCall = MayCallOrRaise(m_context, part.begin, part.end);
		if(mayCall)
		{
			jitter->PushCst(part.begin);
			jitter->PullRel(pcOffset);
		}
		count(6, 1);
		count(7, ((part.end - part.begin) / 4) + 1);
		jitter->SetVariableAsConstant(pcOffset, part.begin);
		for(uint32 address = part.begin; address <= part.end; address += 4)
		{
			m_context.m_pArch->CompileInstruction(address, jitter, &m_context, address - part.begin);
			assert(jitter->IsStackEmpty());
		}
		jitter->ClearVariableStatus(pcOffset);
		jitter->MarkLastBlockLabel();

		jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
		jitter->PushCst(((part.end - part.begin) / 4) + 1);
		jitter->Sub();
		jitter->PullRel(offsetof(CMIPS, m_State.cycleQuota));

		//Static branch target of the part (MIPS_INVALID_PC: none or dynamic)
		uint32 takenTarget = MIPS_INVALID_PC;
		bool dynamicTarget = false;
		bool maySetDelayedJump = MaySetDelayedJump();
		if(maySetDelayedJump)
		{
			uint32 branchAddress = part.end;
			uint32 opcode = m_context.m_pMemoryMap->GetInstruction(part.end - 4);
			if((part.end != part.begin) && (m_context.m_pArch->IsInstructionBranch(&m_context, part.end - 4, opcode) == MIPS_BRANCH_NORMAL))
			{
				branchAddress = part.end - 4;
			}
			opcode = m_context.m_pMemoryMap->GetInstruction(branchAddress);
			if(m_context.m_pArch->IsInstructionBranch(&m_context, branchAddress, opcode) != MIPS_BRANCH_NONE)
			{
				takenTarget = m_context.m_pArch->GetInstructionEffectiveAddress(&m_context, branchAddress, opcode);
			}
			//COP0 (ERET...) or a jump to a register: the target is only known at run time
			dynamicTarget = (takenTarget == MIPS_INVALID_PC) || ((opcode >> 26) == 0x10);
			for(uint32 address = part.begin; address < branchAddress; address += 4)
			{
				if((m_context.m_pMemoryMap->GetInstruction(address) >> 26) == 0x10) dynamicTarget = true;
			}
		}

		auto route = [&](uint32 target, bool internalAllowed) {
			auto exit = exitLabel(target);
			jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
			jitter->PushCst(0);
			jitter->BeginIf(Jitter::CONDITION_LE, -1);
			{
				jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
				jitter->PushCst(MIPS_EXCEPTION_STATUS_QUOTADONE);
				jitter->Or();
				jitter->PullRel(offsetof(CMIPS, m_State.nHasException));
				jitter->Goto(exit);
			}
			jitter->EndIf();
			if(mayCall)
			{
				jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
				jitter->PushRel(regionOffset);
				jitter->PushCst(1);
				jitter->And();
				jitter->Or();
				jitter->PushCst(0);
				jitter->BeginIf(Jitter::CONDITION_NE, -1);
				{
					jitter->Goto(exit);
				}
				jitter->EndIf();
			}
			auto partIterator = partLabels.find(target);
			jitter->Goto((internalAllowed && (partIterator != partLabels.end())) ? partIterator->second : exit);
		};

		if(maySetDelayedJump)
		{
			jitter->PushRel(delayedJumpOffset);
			jitter->PushCst(MIPS_INVALID_PC);
			jitter->BeginIf(Jitter::CONDITION_NE);
			{
				if(dynamicTarget)
				{
					usesDynamicExit = true;
					jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
					jitter->PushCst(0);
					jitter->BeginIf(Jitter::CONDITION_LE, -1);
					{
						jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
						jitter->PushCst(MIPS_EXCEPTION_STATUS_QUOTADONE);
						jitter->Or();
						jitter->PullRel(offsetof(CMIPS, m_State.nHasException));
					}
					jitter->EndIf();
					//Returns from calls followed into the region
					if(!part.returns.empty())
					{
						if(mayCall)
						{
							jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
							jitter->PushRel(regionOffset);
							jitter->PushCst(1);
							jitter->And();
							jitter->Or();
							jitter->PushCst(0);
							jitter->BeginIf(Jitter::CONDITION_NE, -1);
							{
								jitter->Goto(dynamicExitLabel);
							}
							jitter->EndIf();
						}
						jitter->PushRel(offsetof(CMIPS, m_State.cycleQuota));
						jitter->PushCst(0);
						jitter->BeginIf(Jitter::CONDITION_GT, 1);
						{
							for(uint32 returnAddress : part.returns)
							{
								auto partIterator = partLabels.find(returnAddress);
								if(partIterator == partLabels.end()) continue;
								jitter->PushRel(delayedJumpOffset);
								jitter->PushCst(returnAddress);
								jitter->BeginIf(Jitter::CONDITION_EQ);
								{
									jitter->PushCst(MIPS_INVALID_PC);
									jitter->PullRel(delayedJumpOffset);
									jitter->Goto(partIterator->second);
								}
								jitter->EndIf();
							}
						}
						jitter->EndIf();
					}
					jitter->Goto(dynamicExitLabel);
				}
				else
				{
					jitter->PushCst(MIPS_INVALID_PC);
					jitter->PullRel(delayedJumpOffset);
					route(takenTarget, takenTarget == part.successors[0]);
				}
			}
			jitter->EndIf();
		}
		uint32 fallTarget = part.end + 4;
		route(fallTarget, fallTarget == part.successors[1]);
	}

	for(const auto& exitPair : exitLabels)
	{
		jitter->MarkLabel(exitPair.second);
		jitter->PushCst(exitPair.first);
		jitter->PullRel(pcOffset);
		jitter->Goto(endLabel);
	}
	if(usesDynamicExit)
	{
		jitter->MarkLabel(dynamicExitLabel);
		jitter->PushRel(delayedJumpOffset);
		jitter->PullRel(pcOffset);
		jitter->PushCst(MIPS_INVALID_PC);
		jitter->PullRel(delayedJumpOffset);
		jitter->Goto(endLabel);
	}
	jitter->MarkLabel(endLabel);

	m_begin = savedBegin;
	m_end = savedEnd;

	if(UsesBlockChaining())
	{
		CompileBlockChaining(jitter);
	}
}
