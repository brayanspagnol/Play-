#include "EeBasicBlock.h"
#include "offsetof_def.h"
#include <algorithm>
#include "xxhash.h"

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

void CEeBasicBlock::CompileProlog(CMipsJitter* jitter)
{
	if(m_fpRoundingMode != DEFAULT_FP_ROUNDING_MODE)
	{
		jitter->FP_SetRoundingMode(m_fpRoundingMode);
	}
	CBasicBlock::CompileProlog(jitter);
}

void CEeBasicBlock::CompileEpilog(CMipsJitter* jitter, bool loopsOnItself)
{
	if(m_fpRoundingMode != DEFAULT_FP_ROUNDING_MODE)
	{
		jitter->FP_SetRoundingMode(DEFAULT_FP_ROUNDING_MODE);
	}

	if(m_isIdleLoopBlock || IsCodeIdleLoopBlock())
	{
		jitter->PushCst(MIPS_EXCEPTION_IDLE);
		jitter->PullRel(offsetof(CMIPS, m_State.nHasException));
	}

	CBasicBlock::CompileEpilog(jitter, loopsOnItself);
}

bool CEeBasicBlock::IsCodeIdleLoopBlock() const
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

	uint32 endInstructionAddress = m_end - 4;
	uint32 endInstruction = m_context.m_pMemoryMap->GetWord(endInstructionAddress);

	//We need a branch at the end of the block
	auto branchType = m_context.m_pArch->IsInstructionBranch(&m_context, endInstructionAddress, endInstruction);
	if(branchType != MIPS_BRANCH_NORMAL) return false;

	//Check that the branch target is ourself
	uint32 branchTarget = m_context.m_pArch->GetInstructionEffectiveAddress(&m_context, endInstructionAddress, endInstruction);
	if(branchTarget == MIPS_INVALID_PC) return false;
	if(branchTarget != m_begin) return false;

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
	for(uint32 address = m_begin; address <= m_end; address += 4)
	{
		//Don't check branch instruction as we've checked it already
		if(address == endInstructionAddress) continue;

		uint32 inst = m_context.m_pMemoryMap->GetWord(address);
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
	std::vector<uint32> layout;
	for(const auto& part : parts)
	{
		for(uint32 address : {part.begin, part.end, part.successors[0], part.successors[1]})
		{
			layout.push_back((address == MIPS_INVALID_PC) ? address : (address - begin));
		}
	}
	return XXH3_64bits(layout.data(), layout.size() * sizeof(uint32));
}

void CEeBasicBlock::CompileRange(CMipsJitter* jitter)
{
	if(m_regionParts.empty())
	{
		CBasicBlock::CompileRange(jitter);
		return;
	}

	//Function layout (a general control flow graph, structured by the wasm backend):
	//  regionBase = nPC (the executor only enters a block at its first address)
	//  for each part, in address order (parts[0] is the entry):
	//    part label: the part's code and epilog, as the tier-1 block (without chaining)
	//    if(nHasException == 0) for each static successor that is a part: if(nPC - regionBase == its offset) goto its label
	//    goto exit
	//  exit: block chaining, as at the end of a tier-1 block
	//Offsets are relative to the entry, so the code is position independent like other blocks. When code is
	//invalidated while the region runs, the executor sets regionBase to an odd value: no comparison matches
	//anymore and the region returns after the current part, as a tier-1 block would.
	const size_t baseOffset = offsetof(CMIPS, m_jitFlagScratch[4]);
	uint32 savedBegin = m_begin;
	uint32 savedEnd = m_end;

	jitter->PushRel(offsetof(CMIPS, m_State.nPC));
	jitter->PullRel(baseOffset);

	CompileProlog(jitter);

	auto parts = m_regionParts;
	std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
	assert(parts[0].begin == savedBegin);

	std::vector<CMipsJitter::LABEL> partLabels;
	for(size_t i = 0; i < parts.size(); i++)
	{
		partLabels.push_back(jitter->CreateLabel());
	}
	auto exitLabel = jitter->CreateLabel();

	m_context.m_pArch->SetCompileHints(m_blockCompileHints);

	for(size_t partIndex = 0; partIndex < parts.size(); partIndex++)
	{
		const auto& part = parts[partIndex];
		jitter->MarkLabel(partLabels[partIndex]);

		m_begin = part.begin;
		m_end = part.end;

		for(uint32 address = m_begin; address <= m_end; address += 4)
		{
			m_context.m_pArch->CompileInstruction(address, jitter, &m_context, address - m_begin);
			assert(jitter->IsStackEmpty());
		}
		jitter->MarkLastBlockLabel();

		m_compilingRegionPart = true;
		CompileEpilog(jitter, false);
		m_compilingRegionPart = false;

		jitter->PushRel(offsetof(CMIPS, m_State.nHasException));
		jitter->PushCst(0);
		jitter->BeginIf(Jitter::CONDITION_EQ);
		{
			for(size_t successorIndex = 0; successorIndex < 2; successorIndex++)
			{
				uint32 successor = part.successors[successorIndex];
				if((successorIndex == 1) && (successor == part.successors[0])) continue;
				for(size_t targetIndex = 0; targetIndex < parts.size(); targetIndex++)
				{
					if(parts[targetIndex].begin != successor) continue;
					jitter->PushRel(offsetof(CMIPS, m_State.nPC));
					jitter->PushRel(baseOffset);
					jitter->Sub();
					jitter->PushCst(successor - savedBegin);
					jitter->BeginIf(Jitter::CONDITION_EQ);
					{
						jitter->Goto(partLabels[targetIndex]);
					}
					jitter->EndIf();
				}
			}
		}
		jitter->EndIf();
		jitter->Goto(exitLabel);
	}
	jitter->MarkLabel(exitLabel);

	m_begin = savedBegin;
	m_end = savedEnd;

	if(UsesBlockChaining())
	{
		CompileBlockChaining(jitter);
	}
}
