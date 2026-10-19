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

protected:
	void CompileProlog(CMipsJitter*) override;
	void CompileEpilog(CMipsJitter*, bool) override;

private:

	static constexpr auto DEFAULT_FP_ROUNDING_MODE = Jitter::CJitter::ROUND_TRUNCATE;
	Jitter::CJitter::ROUNDINGMODE m_fpRoundingMode = DEFAULT_FP_ROUNDING_MODE;

	bool m_isIdleLoopBlock = false;
	bool m_isAutoIdleLoopBlock = false;
	bool m_isAutoIdleTaintBlock = false;
};
