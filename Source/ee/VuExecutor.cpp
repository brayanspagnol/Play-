#include "VuExecutor.h"
#include "VuBasicBlock.h"
#include "VUShared.h"
#include "xxhash.h"
#include "../Ps2Const.h"

bool g_vuSuperBlocks = true; //Module.setVuSuperBlocks (bench &vusb=0)
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
#ifdef __EMSCRIPTEN__
	ResetBlockSets();
#endif
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
	m_blockNoClamping.clear();
	CGenericMipsExecutor::Reset();
#ifdef __EMSCRIPTEN__
	ResetBlockSets();
#endif
}

void CVuExecutor::DropBlocks()
{
	m_referenceBlocks.clear();
	m_cachedBlocks.clear();
	CGenericMipsExecutor::Reset();
#ifdef __EMSCRIPTEN__
	ResetBlockSets();
#endif
}

#ifdef __EMSCRIPTEN__
void CVuExecutor::ResetBlockSets()
{
	//(The current block set is in the lookup table and m_blocks; it gets a key at the next run)
	m_blockSets.clear();
	m_carriedBlocks.clear();
	m_batchPending.clear();
	uint32 pageCount = m_maxAddress / BLOCK_SET_PAGE_SIZE;
	m_pageHashes.assign(pageCount, uint128{});
	m_dirtyPages.assign(pageCount, true);
	m_contentsChanged = true;
	m_hasCurrentKey = false;
}

void CVuExecutor::SelectBlockSet()
{
	m_contentsChanged = false;
	auto map = m_context.m_pMemoryMap->GetInstructionMap(0);
	auto memory = reinterpret_cast<const uint8*>(map->pPointer);
	//Pages whose contents differ from the ones the current set's blocks were made from
	uint64 changedPages = 0;
	static_assert(PS2::MICROMEM1SIZE / BLOCK_SET_PAGE_SIZE <= 64);
	for(uint32 page = 0; page < m_pageHashes.size(); page++)
	{
		if(!m_dirtyPages[page]) continue;
		m_dirtyPages[page] = false;
		auto pageHash = XXH3_128bits(memory + (page * BLOCK_SET_PAGE_SIZE), BLOCK_SET_PAGE_SIZE);
		if(memcmp(&m_pageHashes[page], &pageHash, sizeof(uint128)) == 0) continue;
		memcpy(&m_pageHashes[page], &pageHash, sizeof(uint128));
		changedPages |= (1ULL << page);
	}
	auto keyHash = XXH3_128bits(m_pageHashes.data(), m_pageHashes.size() * sizeof(uint128));
	uint128 key;
	memcpy(&key, &keyHash, sizeof(uint128));
	if(m_hasCurrentKey && (key == m_currentKey)) return;

	BLOCK_SET next;
	if(auto blockSetIterator = m_blockSets.find(key); blockSetIterator != std::end(m_blockSets))
	{
		next = std::move(blockSetIterator->second);
		m_blockSets.erase(blockSetIterator);
	}
	else
	{
		next.table.reset(m_blockLookup.CreateEmptyTable());
		//Contents not seen before: keep the current set's blocks whose words did not change (a block's
		//code and extent only depend on its own words). Games that keep rewriting part of micro memory
		//(God of War: almost every MPG makes a new combination) then don't repartition the rest.
		if(m_hasCurrentKey)
		{
			auto currentTable = m_blockLookup.GetTable();
			auto rangePages = [](uint32 begin, uint32 end) {
				uint32 firstPage = begin / BLOCK_SET_PAGE_SIZE;
				uint32 lastPage = end / BLOCK_SET_PAGE_SIZE;
				return (~0ULL >> (63 - lastPage)) & (~0ULL << firstPage);
			};
			auto carry = [&](const BasicBlockPtr& block) {
				uint32 begin = block->GetBeginAddress();
				//(A superblock's code depends on its parts' words only, not on the words between them)
				const auto& parts = static_cast<const CVuBasicBlock*>(block.get())->GetSuperBlockParts();
				uint64 blockPages = 0;
				if(parts.size() > 1)
				{
					for(const auto& part : parts) blockPages |= rangePages(part.begin, part.end);
				}
				else
				{
					blockPages = rangePages(begin, block->GetEndAddress());
				}
				if(changedPages & blockPages) return;
				uint32 entry = begin / 4;
				if(currentTable[entry] != block.get()) return;
				next.table[entry] = block.get();
				next.carriedBlocks.push_back(block);
			};
			next.carriedBlocks.reserve(m_blocks.size() + m_carriedBlocks.size());
			for(const auto& block : m_blocks) carry(block);
			for(const auto& block : m_carriedBlocks) carry(block);
		}
	}
	BLOCK_SET previous;
	previous.table.reset(m_blockLookup.ExchangeTable(next.table.release()));
	previous.blocks = std::move(m_blocks);
	previous.carriedBlocks = std::move(m_carriedBlocks);
	m_blocks = std::move(next.blocks);
	m_carriedBlocks = std::move(next.carriedBlocks);
	if(m_hasCurrentKey)
	{
		//ponytail: drops every parked set when full; LRU if a game cycles through more contents than that
		if(m_blockSets.size() >= MAX_BLOCK_SETS) m_blockSets.clear();
		m_blockSets.emplace(m_currentKey, std::move(previous));
	}
	m_currentKey = key;
	m_hasCurrentKey = true;
}

int CVuExecutor::Execute(int cycles)
{
	if(m_contentsChanged) SelectBlockSet();
	//Block chaining reads the block table through the context (with MTVU, two executors share the VU1 context)
	m_context.m_blockFnLookup = reinterpret_cast<uint32**>(m_blockLookup.GetTable());
	return CGenericMipsExecutor::Execute(cycles);
}
#endif

void CVuExecutor::ClearActiveBlocksInRange(uint32 start, uint32 end, bool executing)
{
	for(auto blockIterator = m_referenceBlocks.begin(); blockIterator != m_referenceBlocks.end();)
	{
		const auto& block = blockIterator->second;
		bool overlaps = RangesOverlap(block->GetBeginAddress(), block->GetEndAddress(), start, end);
		blockIterator = overlaps ? m_referenceBlocks.erase(blockIterator) : std::next(blockIterator);
	}
#ifdef __EMSCRIPTEN__
	//The blocks stay with the old contents' block set, see SelectBlockSet
	assert(!executing);
	uint32 firstPage = std::min(start, m_maxAddress - 1) / BLOCK_SET_PAGE_SIZE;
	uint32 lastPage = std::min(end, m_maxAddress - 1) / BLOCK_SET_PAGE_SIZE;
	for(uint32 page = firstPage; page <= lastPage; page++)
	{
		m_dirtyPages[page] = true;
	}
	m_contentsChanged = true;
#else
	CGenericMipsExecutor::ClearActiveBlocksInRange(start, end, executing);
#endif
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
#ifdef __EMSCRIPTEN__
	extern uint32 g_eeBatchSize;
	extern int g_vuExperiment;
	//(Experiment bit 16: no batching)
	if(!result->GetBatchModule().empty() && !(g_vuExperiment & 16))
	{
		m_batchPending.push_back(result);
		if(m_batchPending.size() >= g_eeBatchSize) CBasicBlock::BatchModules(m_batchPending);
	}
#endif
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
	//(Keys hold the size: no content hash unless a key of this size exists)
	uint32 blockSizeByte = ((end - begin) + 4) & ~3U;
	bool keySizeUsed = !m_blockNoClamping.empty();
	for(const auto& compileHints : g_blockCompileHints)
	{
		keySizeUsed |= (compileHints.blockKey.second == blockSizeByte);
	}
	if(!keySizeUsed) return true;
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
	for(size_t i = 0; (i < queue.size()) && (parts.size() < MAX_PARTS); i++)
	{
		uint32 begin = queue[i];
		auto endInfo = FindBlockEnd(begin);
		if((endInfo.end >= limit) || !CanJoinSuperBlock(begin, endInfo.end))
		{
			if(i == 0) return {};
			continue;
		}
		parts.push_back({begin, endInfo.end});
		auto addSuccessor = [&](uint32 address) {
			if((address < head) || (address >= limit) || (address & 7)) return;
			if(std::find(queue.begin(), queue.end(), address) != queue.end()) return;
			queue.push_back(address);
		};
		if(endInfo.branchAddress != MIPS_INVALID_PC) addSuccessor(endInfo.branchAddress);
		if(endInfo.fallsThrough) addSuccessor(endInfo.end + 4);
	}
	if(parts.size() < 2) return {};
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
		m_pendingSuperBlockParts = PlanSuperBlock(startAddress);
		for(const auto& part : m_pendingSuperBlockParts)
		{
			endAddress = std::max(endAddress, part.end);
		}
	}
	CreateBlock(startAddress, endAddress);
	m_pendingSuperBlockParts.clear();
#ifndef __EMSCRIPTEN__
	//(Block links are no-ops in WebAssembly)
	auto block = static_cast<CVuBasicBlock*>(FindBlockStartingAt(startAddress));
	if(block->IsLinkable())
	{
		SetupBlockLinks(startAddress, endAddress, endInfo.branchAddress);
	}
#endif
}
