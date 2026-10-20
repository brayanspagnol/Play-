#pragma once

#include <vector>
#include "Types.h"

//Persistent cache of JIT output (WebAssembly build). CBasicBlock::Compile looks a block up by
//the hash of its instruction words (plus category, size and compile hints/variant) and, on a hit,
//builds its function from the cached bytes instead of running the Jitter. Storage (OPFS) is
//handled by the JS frontend, which feeds whole cache files in with Load() and drains new entries
//with TakePending(). All functions are thread safe.
namespace BlockCache
{
	//Mixed into every key: bump when the JIT output changes (CodeGen or block compilers), so
	//entries from older code generators are never reused.
	constexpr uint64 CODEGEN_VERSION = 2; //1: context fields as memarg offsets, 2: VU status/clip flag writes nothing reads skipped

#pragma pack(push, 1)
	struct KEY
	{
		uint32 category;
		uint32 size;
		uint32 hints;
		uint32 variant;
		uint64 hash[2];

		bool operator==(const KEY& rhs) const;
	};
#pragma pack(pop)
	static_assert(sizeof(KEY) == 32);

	bool IsEnabled();
	void SetEnabled(bool);
	void Clear();
	//Parses serialized entries (as produced by TakePending), stops at the first bad one.
	//Returns the number of entries accepted.
	uint32 Load(const uint8* data, size_t size);
	bool Find(const KEY&, std::vector<uint8>& code);
	void Add(const KEY&, const void* code, size_t size);
	std::vector<uint8> TakePending();
	size_t GetPendingSize();
	uint32 TakeHitCount();
}
