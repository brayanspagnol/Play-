#include "VuExecutor.h"
#include "VuBasicBlock.h"
#include "VUShared.h"
#include "xxhash.h"

bool g_vuSuperBlocks = false;
extern bool g_vuContextPromotion;

// clang-format off
const CVuExecutor::BLOCK_COMPILE_HINTS CVuExecutor::g_blockCompileHints[] =
{
	// Tri-Ace VU0 decompression code
	{ std::make_pair(uint128{0x3aa43f7c, 0xe932516c, 0x6786472d, 0xf5333e12}, 0x58U), VUShared::COMPILEHINT_USE_ACCURATE_ADDI },
};
// clang-format on

CVuExecutor::CVuExecutor(CMIPS& context, uint32 maxAddress)
    : CGenericMipsExecutor(context, maxAddress, BLOCK_CATEGORY_PS2_VU)
{
}

void CVuExecutor::SetBlockNoClamping(BlockNoClampingSet blockNoClamping)
{
	m_blockNoClamping = std::move(blockNoClamping);
}

const CVuExecutor::BlockNoClampingSet& CVuExecutor::GetBlockNoClamping() const
{
	return m_blockNoClamping;
}

void CVuExecutor::Reset()
{
	m_referenceBlocks.clear();
	m_cachedBlocks.clear();
	m_superBlockPlans.clear();
	m_blockNoClamping.clear();
	CGenericMipsExecutor::Reset();
}

void CVuExecutor::DropBlocks()
{
	m_referenceBlocks.clear();
	m_cachedBlocks.clear();
	m_superBlockPlans.clear();
	CGenericMipsExecutor::Reset();
}

void CVuExecutor::ClearActiveBlocksInRange(uint32 start, uint32 end, bool executing)
{
	for(auto blockIterator = m_referenceBlocks.begin(); blockIterator != m_referenceBlocks.end();)
	{
		const auto& block = blockIterator->second;
		bool overlaps = RangesOverlap(block->GetBeginAddress(), block->GetEndAddress(), start, end);
		blockIterator = overlaps ? m_referenceBlocks.erase(blockIterator) : std::next(blockIterator);
	}
	CGenericMipsExecutor::ClearActiveBlocksInRange(start, end, executing);
}

CVuExecutor::CachedBlockKey CVuExecutor::ComputeBlockKey(uint32 begin, uint32 end) const
{
	uint32 blockSizeByte = ((end - begin) + 4) & ~3U;
	auto map = m_context.m_pMemoryMap->GetInstructionMap(begin);
	assert(m_context.m_pMemoryMap->GetInstructionMap(end) == map);
	uint32 localBegin = begin - map->nStart;
	auto blockMemory = reinterpret_cast<const uint32*>(reinterpret_cast<uint8*>(map->pPointer) + localBegin);
	auto xxHash = XXH3_128bits(blockMemory, blockSizeByte);
	uint128 hash;
	memcpy(&hash, &xxHash, sizeof(xxHash));
	static_assert(sizeof(hash) == sizeof(xxHash));
	return std::make_pair(hash, blockSizeByte);
}

BasicBlockPtr CVuExecutor::BlockFactory(CMIPS& context, uint32 begin, uint32 end)
{
	auto blockKey = ComputeBlockKey(begin, end);
	bool superBlock = !m_pendingSuperBlockParts.empty();
	auto makeBlock = [&]() {
		auto block = std::make_shared<CVuBasicBlock>(context, begin, end, m_blockCategory);
		//VU data memory is as large as micro memory (4 KB VU0, 16 KB VU1)
		block->m_diffDataSize = m_maxAddress;
		if(superBlock)
		{
			block->SetSuperBlockParts(m_pendingSuperBlockParts);
			block->SetReferenceBlockProvider([this](uint32 address) { return GetReferenceBlock(address); });
		}
		return block;
	};

	//Don't use the cached blocks of we have a breakpoint in our block range.
	bool hasBreakpoint = m_context.HasBreakpointInRange(begin, end);
	if(!hasBreakpoint)
	{
		auto beginBlockIterator = m_cachedBlocks.lower_bound(blockKey);
		auto endBlockIterator = m_cachedBlocks.upper_bound(blockKey);
		//Check if we have a block that has the same contents and the same range.
		for(auto blockIterator = beginBlockIterator; blockIterator != endBlockIterator; blockIterator++)
		{
			const auto& basicBlock(blockIterator->second);
			if(static_cast<CVuBasicBlock*>(basicBlock.get())->IsSuperBlock() != superBlock) continue;
			if(basicBlock->GetBeginAddress() == begin && basicBlock->GetEndAddress() == end)
			{
				return basicBlock;
			}
		}
		//Check if we have a block that has the same contents but not the same range. Reuse the code of that block if that's the case.
		//(Superblock code is position independent too: part offsets are relative to the entry address.)
		for(auto blockIterator = beginBlockIterator; blockIterator != endBlockIterator; blockIterator++)
		{
			if(static_cast<CVuBasicBlock*>(blockIterator->second.get())->IsSuperBlock() != superBlock) continue;
			auto result = makeBlock();
			result->CopyFunctionFrom(blockIterator->second);
			m_cachedBlocks.insert(std::make_pair(blockKey, result));
			return result;
		}
	}

	//Totally new block, build it from scratch
	auto result = makeBlock();
	//VU0 and VU1 generate different code (memory address mask)
	result->AddCacheVariant(m_maxAddress);

	auto blockCompileHintsIterator = std::find_if(std::begin(g_blockCompileHints), std::end(g_blockCompileHints),
	                                              [&](const auto& item) { return item.blockKey == blockKey; });
	if(blockCompileHintsIterator != std::end(g_blockCompileHints))
	{
		result->AddBlockCompileHints(blockCompileHintsIterator->hints);
	}
	if(auto blockNoClamping = m_blockNoClamping.find(blockKey);
	   blockNoClamping != std::end(m_blockNoClamping))
	{
		result->AddBlockCompileHints(VUShared::COMPILEHINT_NO_CLAMPING);
	}

	result->Compile();
	if(!hasBreakpoint)
	{
		m_cachedBlocks.insert(std::make_pair(blockKey, result));
	}
	return result;
}

CVuExecutor::BLOCK_END_INFO CVuExecutor::FindBlockEnd(uint32 startAddress) const
{
	BLOCK_END_INFO result;
	uint32 endAddress = std::min<uint32>(startAddress + MAX_BLOCK_SIZE - 4, m_maxAddress - 4);
	result.fallsThrough = true;
	for(uint32 address = startAddress; address < endAddress; address += 8)
	{
		uint32 addrLo = address + 0;
		uint32 addrHi = address + 4;
		uint32 lowerOp = m_context.m_pMemoryMap->GetInstruction(addrLo);
		uint32 upperOp = m_context.m_pMemoryMap->GetInstruction(addrHi);
		auto branchType = m_context.m_pArch->IsInstructionBranch(&m_context, addrLo, lowerOp);
		if(upperOp & VUShared::VU_UPPEROP_BIT_E)
		{
			endAddress = address + 0xC;
			result.fallsThrough = false;
			break;
		}
		else if((branchType == MIPS_BRANCH_NONE) && (upperOp & (VUShared::VU_UPPEROP_BIT_D | VUShared::VU_UPPEROP_BIT_T)))
		{
			endAddress = address + 0x04;
			result.fallsThrough = false;
			break;
		}
		else if(branchType == MIPS_BRANCH_NORMAL)
		{
			result.branchAddress = m_context.m_pArch->GetInstructionEffectiveAddress(&m_context, addrLo, lowerOp);
			result.fallsThrough = CVuBasicBlock::IsConditionalBranch(lowerOp);
			endAddress = address + 0xC;
			break;
		}
		else if(branchType == MIPS_BRANCH_NODELAY)
		{
			//Should never happen
			assert(false);
		}
	}
	assert((endAddress - startAddress) <= MAX_BLOCK_SIZE);
	result.end = endAddress;
	return result;
}

bool CVuExecutor::CanJoinSuperBlock(uint32 begin, uint32 end) const
{
	//Blocks whose code depends on more than their own words (branch or E bit in the delay slot),
	//blocks with per-block compile settings and breakpoints stay standalone
	uint32 lastLo = m_context.m_pMemoryMap->GetInstruction(end - 4);
	uint32 lastHi = m_context.m_pMemoryMap->GetInstruction(end);
	if(CVuBasicBlock::IsConditionalBranch(lastLo)) return false;
	if(lastHi & VUShared::VU_UPPEROP_BIT_E) return false;
	if(m_context.HasBreakpointInRange(begin, end)) return false;
	auto blockKey = ComputeBlockKey(begin, end);
	if(m_blockNoClamping.find(blockKey) != std::end(m_blockNoClamping)) return false;
	for(const auto& compileHints : g_blockCompileHints)
	{
		if(compileHints.blockKey == blockKey) return false;
	}
	return true;
}

std::vector<CVuBasicBlock::SUPERBLOCK_PART> CVuExecutor::PlanSuperBlock(uint32 head) const
{
	//Blocks statically reachable from head (breadth first), all inside [head, head + MAX_BLOCK_SIZE)
	//so that range invalidation (which scans MAX_BLOCK_SIZE back) still finds the superblock.
	enum
	{
		MAX_PARTS = 16,
	};
	std::vector<CVuBasicBlock::SUPERBLOCK_PART> parts;
	uint32 limit = std::min<uint32>(head + MAX_BLOCK_SIZE, m_maxAddress);
	std::vector<uint32> queue = {head};
	//Replay experiment bits 0xC00: part cap 2 / 4 / 8
	extern int g_vuExperiment;
	static const size_t experimentCaps[4] = {MAX_PARTS, 2, 4, 8};
	size_t maxParts = experimentCaps[(g_vuExperiment >> 10) & 3];
	for(size_t i = 0; (i < queue.size()) && (parts.size() < maxParts); i++)
	{
		uint32 begin = queue[i];
		auto endInfo = FindBlockEnd(begin);
		if((endInfo.end >= limit) || !CanJoinSuperBlock(begin, endInfo.end))
		{
			if(i == 0) return {};
			continue;
		}
		CVuBasicBlock::SUPERBLOCK_PART part;
		part.begin = begin;
		part.end = endInfo.end;
		part.successors[0] = endInfo.branchAddress;
		part.successors[1] = endInfo.fallsThrough ? (endInfo.end + 4) : MIPS_INVALID_PC;
		parts.push_back(part);
		auto addSuccessor = [&](uint32 address) {
			if((address < head) || (address >= limit) || (address & 7)) return;
			if(std::find(queue.begin(), queue.end(), address) != queue.end()) return;
			queue.push_back(address);
		};
		if(endInfo.branchAddress != MIPS_INVALID_PC) addSuccessor(endInfo.branchAddress);
		if(endInfo.fallsThrough) addSuccessor(endInfo.end + 4);
	}
	if(parts.size() < 2) return {};

	//The part graph must be reducible (every edge to a part that comes earlier in reverse postorder
	//goes to a part that dominates its source): the wasm backend can't structure anything else.
	size_t partCount = parts.size();
	auto partIndex = [&](uint32 address) -> int {
		for(size_t i = 0; i < partCount; i++)
		{
			if(parts[i].begin == address) return static_cast<int>(i);
		}
		return -1;
	};
	std::vector<uint32> dominators(partCount, (1U << partCount) - 1);
	dominators[0] = 1;
	for(bool changed = true; changed;)
	{
		changed = false;
		for(size_t i = 1; i < partCount; i++)
		{
			uint32 newDominators = (1U << partCount) - 1;
			for(size_t p = 0; p < partCount; p++)
			{
				for(uint32 successor : parts[p].successors)
				{
					if(partIndex(successor) == static_cast<int>(i)) newDominators &= dominators[p];
				}
			}
			newDominators |= (1U << i);
			if(newDominators != dominators[i])
			{
				dominators[i] = newDominators;
				changed = true;
			}
		}
	}
	std::vector<int> rpo(partCount, -1);
	{
		std::vector<int> postorder;
		std::vector<std::pair<int, int>> stack = {{0, 0}};
		std::vector<bool> visited(partCount, false);
		visited[0] = true;
		while(!stack.empty())
		{
			int x = stack.back().first;
			int successorIndex = stack.back().second++;
			if(successorIndex < 2)
			{
				int y = partIndex(parts[x].successors[successorIndex]);
				if((y >= 0) && !visited[y])
				{
					visited[y] = true;
					stack.push_back({y, 0});
				}
				continue;
			}
			postorder.push_back(x);
			stack.pop_back();
		}
		for(size_t i = 0; i < postorder.size(); i++)
		{
			rpo[postorder[i]] = static_cast<int>(postorder.size() - 1 - i);
		}
	}
	for(size_t x = 0; x < partCount; x++)
	{
		for(uint32 successor : parts[x].successors)
		{
			int y = partIndex(successor);
			if((y >= 0) && (rpo[y] <= rpo[x]) && !(dominators[x] & (1U << y))) return {};
		}
	}
	return parts;
}

CBasicBlock* CVuExecutor::GetReferenceBlock(uint32 address)
{
	auto blockIterator = m_referenceBlocks.find(address);
	if(blockIterator != std::end(m_referenceBlocks))
	{
		return blockIterator->second.get();
	}
	auto endInfo = FindBlockEnd(address);
	auto block = std::make_shared<CVuBasicBlock>(m_context, address, endInfo.end, m_blockCategory);
	block->m_diffDataSize = m_maxAddress;
	block->m_forceReference = true;
	//One block run = one superblock part (the executor would see self loops as several runs)
	block->SetNoSelfLoop();
	block->AddCacheVariant(m_maxAddress);
	block->Compile();
	m_referenceBlocks[address] = block;
	return block.get();
}

void CVuExecutor::PartitionFunction(uint32 startAddress)
{
	auto endInfo = FindBlockEnd(startAddress);
	uint32 endAddress = endInfo.end;
	if(g_vuSuperBlocks && g_vuContextPromotion)
	{
		uint32 limit = std::min<uint32>(startAddress + MAX_BLOCK_SIZE, m_maxAddress);
		auto planKey = std::make_pair(startAddress, ComputeBlockKey(startAddress, limit - 4));
		auto planIterator = m_superBlockPlans.find(planKey);
		if(planIterator == std::end(m_superBlockPlans))
		{
			planIterator = m_superBlockPlans.emplace(planKey, PlanSuperBlock(startAddress)).first;
		}
		m_pendingSuperBlockParts = planIterator->second;
		for(const auto& part : m_pendingSuperBlockParts)
		{
			endAddress = std::max(endAddress, part.end);
		}
	}
	CreateBlock(startAddress, endAddress);
	m_pendingSuperBlockParts.clear();
	auto block = static_cast<CVuBasicBlock*>(FindBlockStartingAt(startAddress));
	if(block->IsLinkable())
	{
		SetupBlockLinks(startAddress, endAddress, endInfo.branchAddress);
	}
}
