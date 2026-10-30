#pragma once

#include <functional>
#include <vector>
#include "../BasicBlock.h"

class CVuBasicBlock : public CBasicBlock
{
public:
	CVuBasicBlock(CMIPS&, uint32, uint32, BLOCK_CATEGORY);
	virtual ~CVuBasicBlock() = default;

	bool IsLinkable() const;

	//Superblock: several VU blocks compiled into one function with a dispatch loop on nPC, so
	//block transitions stay inside the function (state in wasm locals). parts[0] is this block's
	//own range, the other parts are blocks reachable from it, all inside [begin, end].
	struct SUPERBLOCK_PART
	{
		uint32 begin = 0;
		uint32 end = 0;
	};
	void SetSuperBlockParts(std::vector<SUPERBLOCK_PART>);
	const std::vector<SUPERBLOCK_PART>& GetSuperBlockParts() const
	{
		return m_superBlockParts;
	}
	bool IsSuperBlock() const override;
	//Differential test of superblocks: runs reference blocks one at a time
	typedef std::function<CBasicBlock*(uint32)> ReferenceBlockProvider;
	void SetReferenceBlockProvider(ReferenceBlockProvider);
	//Never loop inside the function (reference blocks of the superblock differential test)
	void SetNoSelfLoop();

	//Static analysis helpers (also used by the executor to plan superblocks)
	static bool IsConditionalBranch(uint32);
	static bool IsNonConditionalBranch(uint32);

protected:
	void CompileRange(CMipsJitter*) override;
#ifdef __EMSCRIPTEN__
	void ExecuteDiff() override;
#endif

private:
	struct INTEGER_BRANCH_DELAY_INFO
	{
		unsigned int regIndex = 0;
		uint32 saveRegAddress = MIPS_INVALID_PC;
		uint32 useRegAddress = MIPS_INVALID_PC;
	};

	void CompileBody(CMipsJitter*, bool allowSelfLoop);
	void CompileSuperBlock(CMipsJitter*);

	typedef uint32 FmacRegWriteTimes[32][4];
	struct BlockFmacPipelineInfo
	{
		//State of the write times at the end of block
		FmacRegWriteTimes regWriteTimes = {};

		//Delays incurred during every instruction execution
		std::vector<uint32> stallDelays;

		//Pipe time at the end of block
		uint32 pipeTime = 0;
	};

	INTEGER_BRANCH_DELAY_INFO ComputeIntegerBranchDelayInfo(const std::vector<uint32>&) const;
	INTEGER_BRANCH_DELAY_INFO ComputeTrailingIntegerBranchDelayInfo(const std::vector<uint32>&) const;
	bool CheckIsSpecialIntegerLoop(unsigned int) const;
	void ComputeSkipFlagsHints(const std::vector<uint32>&, std::vector<uint32>&) const;
	BlockFmacPipelineInfo ComputeFmacStallDelays(uint32, uint32) const;
	static void EmitXgKick(CMipsJitter*);

	bool m_isLinkable = true;
	bool m_noSelfLoop = false;
	std::vector<SUPERBLOCK_PART> m_superBlockParts;
	ReferenceBlockProvider m_referenceBlockProvider;
};
