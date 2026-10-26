#include "BlockCache.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "xxhash.h"
#include "MIPS.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include "WasmModuleBuilder.h"
#endif

using namespace BlockCache;

namespace
{
	struct KeyHasher
	{
		size_t operator()(const KEY& key) const
		{
			return static_cast<size_t>(key.hash[0] ^ key.variant ^ key.hints);
		}
	};

#pragma pack(push, 1)
	struct ENTRY_HEADER
	{
		KEY key;
		uint32 codeSize;
		uint64 checksum; //XXH3_64 of the code bytes
	};
#pragma pack(pop)

	//ponytail: hard cap instead of eviction; a game that outgrows it just stops adding entries.
	constexpr size_t MAX_TOTAL_BYTES = 96 * 1024 * 1024;
	constexpr uint32 MAX_CODE_SIZE = 1024 * 1024;

	std::mutex g_mutex;
	std::atomic<bool> g_enabled = false;
	std::atomic<uint32> g_hits = 0;
	struct ENTRY
	{
		std::vector<uint8> code;
		uint32 group = 0; //0: not in a group (added in this session)
		uint32 index = 0; //function index in the group's module
	};

	//Entries of one Load() call, by category, in file order
	struct GROUP
	{
		std::vector<const std::vector<uint8>*> codes;
	};
	//ponytail: fixed size; 64 is what the EE batching uses. Larger groups compile more unused code.
	constexpr uint32 GROUP_SIZE = 64;

	std::unordered_map<KEY, ENTRY, KeyHasher> g_entries;
	std::unordered_map<uint32, GROUP> g_groups;
	uint32 g_nextGroupId = 1; //never reused, so instances made for older groups are never mistaken for new ones
	std::atomic<uint32> g_generation = 0;
	std::vector<uint8> g_pending;
	size_t g_totalBytes = 0;
}

bool KEY::operator==(const KEY& rhs) const
{
	return memcmp(this, &rhs, sizeof(KEY)) == 0;
}

bool BlockCache::IsEnabled()
{
	return g_enabled;
}

void BlockCache::SetEnabled(bool enabled)
{
	g_enabled = enabled;
}

void BlockCache::Clear()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	g_entries.clear();
	g_groups.clear();
	g_generation++;
	g_pending.clear();
	g_pending.shrink_to_fit();
	g_totalBytes = 0;
}

uint32 BlockCache::Load(const uint8* data, size_t size)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	uint32 count = 0;
	size_t pos = 0;
	std::unordered_map<uint32, uint32> openGroups; //category -> group id
	while(size - pos >= sizeof(ENTRY_HEADER))
	{
		ENTRY_HEADER header;
		memcpy(&header, data + pos, sizeof(header));
		const uint8* code = data + pos + sizeof(header);
		if(header.codeSize < 8 || header.codeSize > MAX_CODE_SIZE) break;
		if(size - pos - sizeof(header) < header.codeSize) break;
		if(memcmp(code, "\0asm", 4) != 0) break;
		if(XXH3_64bits(code, header.codeSize) != header.checksum) break;
		pos += sizeof(header) + header.codeSize;
		if(g_totalBytes + header.codeSize > MAX_TOTAL_BYTES) break;
		ENTRY entry;
		entry.code.assign(code, code + header.codeSize);
		auto inserted = g_entries.emplace(header.key, std::move(entry));
		if(inserted.second)
		{
			g_totalBytes += header.codeSize;
			count++;
			auto& groupId = openGroups[header.key.category];
			if((groupId == 0) || (g_groups[groupId].codes.size() == GROUP_SIZE))
			{
				groupId = g_nextGroupId++;
			}
			auto& group = g_groups[groupId];
			auto& newEntry = inserted.first->second;
			newEntry.group = groupId;
			newEntry.index = static_cast<uint32>(group.codes.size());
			group.codes.push_back(&newEntry.code);
		}
	}
	return count;
}

bool BlockCache::Find(const KEY& key, std::vector<uint8>& code)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	auto entryIterator = g_entries.find(key);
	if(entryIterator == std::end(g_entries)) return false;
	code = entryIterator->second.code;
	g_hits++;
	return true;
}

void BlockCache::Add(const KEY& key, const void* code, size_t size)
{
	if(size > MAX_CODE_SIZE) return;
	std::lock_guard<std::mutex> lock(g_mutex);
	if(g_totalBytes + size > MAX_TOTAL_BYTES) return;
	auto codeBytes = reinterpret_cast<const uint8*>(code);
	ENTRY entry;
	entry.code.assign(codeBytes, codeBytes + size);
	auto inserted = g_entries.emplace(key, std::move(entry));
	if(!inserted.second) return;
	g_totalBytes += size;

	ENTRY_HEADER header = {};
	header.key = key;
	header.codeSize = static_cast<uint32>(size);
	header.checksum = XXH3_64bits(code, size);
	auto headerBytes = reinterpret_cast<const uint8*>(&header);
	g_pending.insert(g_pending.end(), headerBytes, headerBytes + sizeof(header));
	g_pending.insert(g_pending.end(), codeBytes, codeBytes + size);
}

std::vector<uint8> BlockCache::TakePending()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	std::vector<uint8> result;
	std::swap(result, g_pending);
	return result;
}

size_t BlockCache::GetPendingSize()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_pending.size();
}

uint32 BlockCache::TakeHitCount()
{
	return g_hits.exchange(0);
}

#ifdef __EMSCRIPTEN__
// clang-format off
EM_JS_DEPS(BlockCacheGroups, "$getEmptyTableSlot,$setWasmTableEntry");
//Group instances of this thread (each thread has its own JS Module and function table)
EM_JS(int, BlockCacheCreateGroup, (uint32 groupId, uintptr_t code, uintptr_t size), {
	try
	{
		let module = new WebAssembly.Module(HEAP8.subarray(code, code + size));
		let instance = new WebAssembly.Instance(module, {
			env: {
				memory: wasmMemory,
				fctTable : Module.codeGenImportTable,
				blockTable : wasmTable
			}
		});
		if(!Module.blockCacheGroups) Module.blockCacheGroups = new Map();
		Module.blockCacheGroups.set(groupId, instance);
		return 1;
	}
	catch(e)
	{
		err("Block cache group module failed: " + e);
		return 0;
	}
});
EM_JS(int, BlockCacheGroupFunction, (uint32 groupId, uint32 index), {
	//A slot of its own (not addFunction, which hands out the same slot again for the same function):
	//two blocks with the same code each own and later free their slot
	let slot = getEmptyTableSlot();
	setWasmTableEntry(slot, Module.blockCacheGroups.get(groupId).exports["f" + index]);
	return slot;
});
EM_JS(void, BlockCacheResetGroups, (), {
	Module.blockCacheGroups = new Map();
});
// clang-format on

int BlockCache::FindFunction(const KEY& key, size_t& codeSize)
{
	thread_local uint32 generation = ~0U;
	thread_local std::unordered_set<uint32> created; //groups instantiated by this thread
	thread_local std::unordered_set<uint32> failed;
	uint32 groupId = 0, index = 0;
	std::vector<uint8> module;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		auto entryIterator = g_entries.find(key);
		if(entryIterator == std::end(g_entries)) return 0;
		groupId = entryIterator->second.group;
		index = entryIterator->second.index;
		codeSize = entryIterator->second.code.size();
		if(groupId == 0) return 0;
		if(generation != g_generation)
		{
			generation = g_generation;
			created.clear();
			failed.clear();
			BlockCacheResetGroups();
		}
		if(failed.count(groupId)) return 0;
		if(!created.count(groupId))
		{
			const auto& group = g_groups[groupId];
			std::vector<CWasmModuleBuilder::MODULE_BYTES> modules;
			modules.reserve(group.codes.size());
			for(const auto* code : group.codes)
			{
				modules.push_back({code->data(), static_cast<uint32>(code->size())});
			}
			module = CWasmModuleBuilder::BuildBatchModule(modules);
		}
	}
	if(!created.count(groupId))
	{
		if(module.empty() || !BlockCacheCreateGroup(groupId, reinterpret_cast<uintptr_t>(module.data()), module.size()))
		{
			failed.insert(groupId);
			return 0;
		}
		created.insert(groupId);
	}
	g_hits++;
	return BlockCacheGroupFunction(groupId, index);
}
#else
int BlockCache::FindFunction(const KEY&, size_t&)
{
	return 0;
}
#endif

std::string BlockCache::GetSalt()
{
	//BLOCKCACHE_SOURCE_HASH: hash of every source file that takes part in code generation (CMakeLists.txt).
	//The layout part catches context offsets that changed through a header the list missed.
	uint32 layout[] = {
	    static_cast<uint32>(sizeof(CMIPS)), static_cast<uint32>(sizeof(MIPSSTATE)),
	    static_cast<uint32>(offsetof(CMIPS, m_State)), static_cast<uint32>(offsetof(CMIPS, m_vuMem)),
	    static_cast<uint32>(offsetof(CMIPS, m_pageLookup)), static_cast<uint32>(offsetof(CMIPS, m_pageLookupWrite)),
	    static_cast<uint32>(offsetof(CMIPS, m_blockFnLookup)), static_cast<uint32>(offsetof(CMIPS, m_jitFlagScratch)),
	    static_cast<uint32>(CODEGEN_VERSION)};
	char text[64];
	snprintf(text, sizeof(text), "-%016llx", static_cast<unsigned long long>(XXH3_64bits(layout, sizeof(layout))));
#ifdef BLOCKCACHE_SOURCE_HASH
	return std::string(BLOCKCACHE_SOURCE_HASH) + text;
#else
	return std::string("nohash") + text;
#endif
}
