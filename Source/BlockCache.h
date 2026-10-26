#pragma once

#include <string>
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
	constexpr uint64 CODEGEN_VERSION = 20; //1: context fields as memarg offsets, 2: VU status/clip flag writes nothing reads skipped, 3: MakeSz/ExpandW wasm forms, 14: ITOFn multiplies (not 4: kept apart from other branches sharing the cache), 20: EE link sites

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
	//Like Find, but entries that came from Load() are instantiated in groups: the first hit in a group
	//compiles one module holding the whole group (entries loaded next to each other were compiled next to
	//each other, so they tend to run together), later hits only take a function out of it. Returns a
	//function table index of the calling thread (0: not found; codeSize is set when found). The
	//instances live in the calling thread's JS realm, like every other JIT function.
	int FindFunction(const KEY&, size_t& codeSize);
	//"<source hash>-<layout>": changes only when cached code could be different (see CMakeLists.txt)
	std::string GetSalt();
	void Add(const KEY&, const void* code, size_t size);
	std::vector<uint8> TakePending();
	size_t GetPendingSize();
	uint32 TakeHitCount();
}
