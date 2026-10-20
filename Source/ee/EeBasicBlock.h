#pragma once

#include "BasicBlock.h"

class CEeBasicBlock : public CBasicBlock
{
public:
	using CBasicBlock::CBasicBlock;

	void SetFpRoundingMode(Jitter::CJitter::ROUNDINGMODE);
	void SetIsIdleLoopBlock();
	void SetIsAutoIdleLoopBlock();
	void SetIsAutoIdleTaintBlock();

	static bool IsCodeIdleLoopBlock(CMIPS&, uint32 begin, uint32 end);

	//Tier-2 region: the tier-1 blocks in parts (parts[0] starts at this block's begin) compiled into one
	//function with direct jumps between them (see CompileRange). Empty: an ordinary block.
	void SetRegionParts(std::vector<SUPERBLOCK_PART>);
	bool IsSuperBlock() const override;
	void CompileRange(CMipsJitter*) override;
	void GetCodeWords(std::vector<uint32>&) const override;
	//Hash of the part layout (relative to begin), mixed into cache keys of region code
	static uint64 GetRegionLayoutHash(const std::vector<SUPERBLOCK_PART>&, uint32 begin);

protected:
	void CompileProlog(CMipsJitter*) override;
	void CompileEpilog(CMipsJitter*, bool) override;

private:

	static constexpr auto DEFAULT_FP_ROUNDING_MODE = Jitter::CJitter::ROUND_TRUNCATE;
	Jitter::CJitter::ROUNDINGMODE m_fpRoundingMode = DEFAULT_FP_ROUNDING_MODE;

	bool m_isIdleLoopBlock = false;
	std::vector<SUPERBLOCK_PART> m_regionParts;
	bool m_isAutoIdleLoopBlock = false;
	bool m_isAutoIdleTaintBlock = false;
};
