#pragma once

#include <map>
#include <memory>
#include <unordered_map>
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
#ifdef __EMSCRIPTEN__
	int Execute(int) override;
#endif

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
	//PlanSuperBlock results by (head, key of the code in [head, head + MAX_BLOCK_SIZE)): microprogram uploads
	//invalidate every block, and planning again on each re-creation costs more than the superblock saves
	std::map<std::pair<uint32, CachedBlockKey>, std::vector<CVuBasicBlock::SUPERBLOCK_PART>> m_superBlockPlans;
	//Differential test: standalone reference blocks the superblocks are compared against
	std::map<uint32, BasicBlockPtr> m_referenceBlocks;

#ifdef __EMSCRIPTEN__
	//Block sets per micro memory contents. Writes to micro memory (MPG, EE stores, state loads) don't
	//drop blocks: they mark pages dirty, and before the next run the executor switches to the block set
	//of the new contents (key: hash of the per page hashes), which is empty the first time those
	//contents are seen. Games that swap a few microprograms in and out (GTA SA: several per frame)
	//then find all their blocks again. (Block links are no-ops in WebAssembly, blocks share nothing
	//else that depends on other blocks.)
	struct BLOCK_SET
	{
		std::unique_ptr<CBasicBlock*[]> table;
		BlockStore blocks;
	};
	struct KEY_HASHER
	{
		size_t operator()(const uint128& key) const
		{
			return static_cast<size_t>(key.nD0);
		}
	};
	enum
	{
		BLOCK_SET_PAGE_SIZE = 0x100,
		MAX_BLOCK_SETS = 64,
	};

	void ResetBlockSets();
	void SelectBlockSet();

	std::unordered_map<uint128, BLOCK_SET, KEY_HASHER> m_blockSets;
	//Module batching (see CBasicBlock::BatchModules): new blocks wait here
	std::vector<std::weak_ptr<CBasicBlock>> m_batchPending;
	std::vector<uint128> m_pageHashes;
	std::vector<bool> m_dirtyPages;
	bool m_contentsChanged = true;
	bool m_hasCurrentKey = false;
	uint128 m_currentKey = {};
#endif
};
