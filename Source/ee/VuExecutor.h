#pragma once

#include <map>
#include "../GenericMipsExecutor.h"

class CVuExecutor : public CGenericMipsExecutor<BlockLookupOneWay, 8>
{
public:
	using CachedBlockKey = std::pair<uint128, uint32>;
	using BlockNoClampingSet = std::set<CachedBlockKey>;

	CVuExecutor(CMIPS&, uint32);
	virtual ~CVuExecutor() = default;

	void SetBlockNoClamping(BlockNoClampingSet);

	void Reset() override;
	//Drops every compiled block (keeps per-game settings), used by the replay benchmark
	void ResetBlocks();

protected:
	typedef std::multimap<CachedBlockKey, BasicBlockPtr> CachedBlockMap;

	struct BLOCK_COMPILE_HINTS
	{
		CachedBlockKey blockKey;
		uint32 hints;
	};

	BasicBlockPtr BlockFactory(CMIPS&, uint32, uint32) override;
	void PartitionFunction(uint32) override;

	static const BLOCK_COMPILE_HINTS g_blockCompileHints[];
	CachedBlockMap m_cachedBlocks;

	BlockNoClampingSet m_blockNoClamping;
};
