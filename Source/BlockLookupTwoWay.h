#pragma once

#include <cstring>
#include "Types.h"
#include "BasicBlock.h"

class BlockLookupTwoWay
{
public:
	typedef CBasicBlock* BlockType;

	BlockLookupTwoWay(BlockType emptyBlock, uint32 maxAddress)
	    : m_emptyBlock(emptyBlock)
	{
		m_subTableCount = (maxAddress + SUBTABLE_MASK) / SUBTABLE_SIZE;
		assert(m_subTableCount != 0);
		m_blockTable = new BlockType*[m_subTableCount];
		memset(m_blockTable, 0, sizeof(BlockType*) * m_subTableCount);
#ifdef __EMSCRIPTEN__
		m_fnTable = new uint32*[m_subTableCount];
		for(unsigned int i = 0; i < m_subTableCount; i++)
		{
			m_fnTable[i] = m_emptyFnSubTable;
		}
#endif
	}

	~BlockLookupTwoWay()
	{
		for(unsigned int i = 0; i < m_subTableCount; i++)
		{
			auto subTable = m_blockTable[i];
			if(subTable)
			{
				delete[] subTable;
			}
		}
		delete[] m_blockTable;
#ifdef __EMSCRIPTEN__
		ClearFnTable();
		delete[] m_fnTable;
#endif
	}

	void Clear()
	{
		for(unsigned int i = 0; i < m_subTableCount; i++)
		{
			auto subTable = m_blockTable[i];
			if(subTable)
			{
				delete[] subTable;
				m_blockTable[i] = nullptr;
			}
		}
#ifdef __EMSCRIPTEN__
		ClearFnTable();
#endif
	}

	void AddBlock(BlockType block)
	{
		uint32 address = block->GetBeginAddress();
		uint32 hiAddress = address >> SUBTABLE_BITS;
		uint32 loAddress = address & SUBTABLE_MASK;
		assert(hiAddress < m_subTableCount);
		auto& subTable = m_blockTable[hiAddress];
		if(!subTable)
		{
			const uint32 subTableSize = SUBTABLE_SIZE / INSTRUCTION_SIZE;
			subTable = new BlockType[subTableSize];
			for(uint32 i = 0; i < subTableSize; i++)
			{
				subTable[i] = m_emptyBlock;
			}
		}
		assert(subTable[loAddress / INSTRUCTION_SIZE] == m_emptyBlock);
		subTable[loAddress / INSTRUCTION_SIZE] = block;
#ifdef __EMSCRIPTEN__
		auto& fnSubTable = m_fnTable[hiAddress];
		if(fnSubTable == m_emptyFnSubTable)
		{
			fnSubTable = new uint32[SUBTABLE_SIZE / INSTRUCTION_SIZE]();
		}
		fnSubTable[loAddress / INSTRUCTION_SIZE] = block->GetFunctionIndex();
#endif
	}

	void DeleteBlock(BlockType block)
	{
		uint32 address = block->GetBeginAddress();
		uint32 hiAddress = address >> SUBTABLE_BITS;
		uint32 loAddress = address & SUBTABLE_MASK;
		assert(hiAddress < m_subTableCount);
		auto& subTable = m_blockTable[hiAddress];
		assert(subTable);
		assert(subTable[loAddress / INSTRUCTION_SIZE] != m_emptyBlock);
		subTable[loAddress / INSTRUCTION_SIZE] = m_emptyBlock;
#ifdef __EMSCRIPTEN__
		m_fnTable[hiAddress][loAddress / INSTRUCTION_SIZE] = 0;
#endif
	}

	BlockType FindBlockAt(uint32 address) const
	{
		uint32 hiAddress = address >> SUBTABLE_BITS;
		uint32 loAddress = address & SUBTABLE_MASK;
		assert(hiAddress < m_subTableCount);
		auto& subTable = m_blockTable[hiAddress];
		if(!subTable) return m_emptyBlock;
		auto result = subTable[loAddress / INSTRUCTION_SIZE];
		return result;
	}

#ifdef __EMSCRIPTEN__
	//Function table indexes of the blocks (0: no block), same layout as the block table. Missing
	//subtables point to a shared all-zero one so generated code can index it without checks.
	uint32** GetFnTable() const
	{
		return m_fnTable;
	}
#endif

private:
#ifdef __EMSCRIPTEN__
	void ClearFnTable()
	{
		for(unsigned int i = 0; i < m_subTableCount; i++)
		{
			if(m_fnTable[i] != m_emptyFnSubTable)
			{
				delete[] m_fnTable[i];
				m_fnTable[i] = m_emptyFnSubTable;
			}
		}
	}
#endif

	enum
	{
		SUBTABLE_BITS = 16,
		SUBTABLE_SIZE = (1 << SUBTABLE_BITS),
		SUBTABLE_MASK = (SUBTABLE_SIZE - 1),
		INSTRUCTION_SIZE = 4,
	};

	BlockType m_emptyBlock = nullptr;
	BlockType** m_blockTable = nullptr;
	uint32 m_subTableCount = 0;
#ifdef __EMSCRIPTEN__
	uint32** m_fnTable = nullptr;
	static inline uint32 m_emptyFnSubTable[SUBTABLE_SIZE / INSTRUCTION_SIZE] = {};
#endif
};
