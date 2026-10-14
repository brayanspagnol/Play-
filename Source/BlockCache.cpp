#include "BlockCache.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include "xxhash.h"

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
	std::unordered_map<KEY, std::vector<uint8>, KeyHasher> g_entries;
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
	g_pending.clear();
	g_pending.shrink_to_fit();
	g_totalBytes = 0;
}

uint32 BlockCache::Load(const uint8* data, size_t size)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	uint32 count = 0;
	size_t pos = 0;
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
		auto inserted = g_entries.emplace(header.key, std::vector<uint8>(code, code + header.codeSize));
		if(inserted.second)
		{
			g_totalBytes += header.codeSize;
			count++;
		}
	}
	return count;
}

bool BlockCache::Find(const KEY& key, std::vector<uint8>& code)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	auto entryIterator = g_entries.find(key);
	if(entryIterator == std::end(g_entries)) return false;
	code = entryIterator->second;
	g_hits++;
	return true;
}

void BlockCache::Add(const KEY& key, const void* code, size_t size)
{
	if(size > MAX_CODE_SIZE) return;
	std::lock_guard<std::mutex> lock(g_mutex);
	if(g_totalBytes + size > MAX_TOTAL_BYTES) return;
	auto codeBytes = reinterpret_cast<const uint8*>(code);
	auto inserted = g_entries.emplace(key, std::vector<uint8>(codeBytes, codeBytes + size));
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
