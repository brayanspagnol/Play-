#pragma once

#include <map>
#include <vector>
#include "../GenericMipsExecutor.h"
#include "VuBasicBlock.h"

//Compile VU superblocks (several blocks in one function, see CVuBasicBlock), optimized codegen only
extern bool g_vuSuperBlocks;

class CVuExecutor : public CGenericMipsExecutor<BlockLookupOneWay, 8>
{
public:
	using CachedBlockKey = std::pair<uint128, uint32>;
	using BlockNoClampingSet = std::set<CachedBlockKey>;

	CVuExecutor(CMIPS&, uint32);
	virtual ~CVuExecutor() = default;

	void SetBlockNoClamping(BlockNoClampingSet);
	const BlockNoClampingSet& GetBlockNoClamping() const;

	void Reset() override;
	//Drops every compiled block (keeps the per game settings)
	void DropBlocks();
	void ClearActiveBlocksInRange(uint32, uint32, bool) override;

protected:
	typedef std::multimap<CachedBlockKey, BasicBlockPtr> CachedBlockMap;

	struct BLOCK_COMPILE_HINTS
	{
		CachedBlockKey blockKey;
		uint32 hints;
	};

	struct BLOCK_END_INFO
	{
		uint32 end = 0;
		uint32 branchAddress = MIPS_INVALID_PC;
		//Execution can continue at end + 4 (conditional branch, or no branch at all)
		bool fallsThrough = false;
	};

	BasicBlockPtr BlockFactory(CMIPS&, uint32, uint32) override;
	void PartitionFunction(uint32) override;

	BLOCK_END_INFO FindBlockEnd(uint32) const;
	CachedBlockKey ComputeBlockKey(uint32, uint32) const;
	bool CanJoinSuperBlock(uint32, uint32) const;
	std::vector<CVuBasicBlock::SUPERBLOCK_PART> PlanSuperBlock(uint32) const;
	CBasicBlock* GetReferenceBlock(uint32);

	static const BLOCK_COMPILE_HINTS g_blockCompileHints[];
	CachedBlockMap m_cachedBlocks;

	BlockNoClampingSet m_blockNoClamping;

	//Parts of the superblock being created (consumed by BlockFactory)
	std::vector<CVuBasicBlock::SUPERBLOCK_PART> m_pendingSuperBlockParts;
	//Differential test: standalone reference blocks the superblocks are compared against
	std::map<uint32, BasicBlockPtr> m_referenceBlocks;
};
