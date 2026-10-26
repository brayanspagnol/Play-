#include "EeExecutor.h"
#include "../Ps2Const.h"
#include "AlignedAlloc.h"
#include "EeBasicBlock.h"
#include "MA_EE.h"
#include "xxhash.h"
#include "../Profiler.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#if defined(__EMSCRIPTEN__)
#include "WasmModuleBuilder.h"
#include "AsyncCompile.h"
#endif

#if defined(__unix__) || defined(__ANDROID__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

#if defined(__APPLE__)

#include <TargetConditionals.h>

#if TARGET_CPU_ARM
#define DISABLE_PROTECTION
#define STATE_FLAVOR ARM_THREAD_STATE32
#define STATE_FLAVOR_COUNT ARM_THREAD_STATE32_COUNT
#elif TARGET_CPU_ARM64
#define STATE_FLAVOR ARM_THREAD_STATE64
#define STATE_FLAVOR_COUNT ARM_THREAD_STATE64_COUNT
#elif TARGET_CPU_X86
#define STATE_FLAVOR x86_THREAD_STATE32
#define STATE_FLAVOR_COUNT x86_THREAD_STATE32_COUNT
#elif TARGET_CPU_X86_64
#define STATE_FLAVOR x86_THREAD_STATE64
#define STATE_FLAVOR_COUNT x86_THREAD_STATE64_COUNT
#else
#error Unsupported CPU architecture
#endif

#if TARGET_OS_TV
#define DISABLE_PROTECTION
#endif

#endif

static CEeExecutor* g_eeExecutor = nullptr;

//Automatic wait loop detection ---------------------------------------------------------------
//When a block ending with a branch back to an earlier address (the loop head) is created, every path
//from the head back to the head is analyzed, following calls (up to 3 deep). An iteration is "safe" if
//it only executes register arithmetic, loads (hardware registers only when reading them has no side
//effect), stores to stack slots below the loop's sp that it writes before reading (frames of called
//functions), FlushCache syscalls, and if every register it reads before writing is left unchanged or
//set to the same constant at the end of the iteration (no counters). A safe iteration that leaves the
//registers as they were cannot change anything until something external (interrupt, DMA, IOP/SIF,
//timer) changes memory: the EE is idle and the rest of its time slice can be skipped, like with a
//GameConfig IdleLoopBlock.
//Paths with anything else (a debug print, a store to memory...) aren't rejected: the block where they
//start (a "taint block") records that the iteration was not safe. The block with the loop branch calls
//EeAutoIdleCheck when it branches back to the head; the EE is idle when the previous pass through the
//check had the same registers and no taint block ran in between.
//Loops that code outside of them jumps into (other than to the head) are rejected, and blocks are
//invalidated when the code read by the analysis changes.
//Counter loops: a register read by the loop may also be a counter (addiu of a constant each iteration,
//or a copy of one) if its value is only copied or compared with something that doesn't change (timeout
//polls, delay loops). There the check skips iterations instead of the slice: after two consecutive
//passes with the expected deltas, it advances the counters by the number of iterations that fit in the
//rest of the slice (or until one of the loop's comparisons involving a counter would change its result,
//e.g. the timeout or the end of a delay) and takes their cycles from the quota. The skipped iterations
//are exactly the ones the EE would have run: same path, same memory, same registers at the end.

bool g_eeAutoIdle = true;

namespace
{
	enum
	{
		COUNTER_MAX_REGS = 4,
		COUNTER_MAX_PREDS = 4,
	};

	enum COUNTER_PRED_OP : uint8
	{
		COUNTER_PRED_SLT,
		COUNTER_PRED_SLTU,
		COUNTER_PRED_EQ,
	};

	//Operand of a comparison in a counter loop: a constant, or a register's value at the start of the
	//iteration plus an offset (32-bit add when the offset isn't 0)
	struct COUNTER_OPERAND
	{
		bool isConst = true;
		uint8 reg = 0;
		int32 value = 0; //Constant (sign extended), or offset
	};

	struct COUNTER_PRED
	{
		COUNTER_PRED_OP op = COUNTER_PRED_EQ;
		COUNTER_OPERAND a;
		COUNTER_OPERAND b;
	};

	//Registers changed by a constant each iteration and comparisons that read them (none: plain wait loop)
	struct COUNTER_LOOP
	{
		uint32 regCount = 0;
		uint8 regs[COUNTER_MAX_REGS];
		int32 steps[COUNTER_MAX_REGS];
		uint32 predCount = 0;
		COUNTER_PRED preds[COUNTER_MAX_PREDS];
		bool ambiguous = false; //Another loop with the same head: no skipping
		uint32 maxIterationCycles = 0;
	};

	struct AUTOIDLE_STAT
	{
		std::atomic<uint32> head = 0;
		uint32 branch = 0;
		uint32 insts = 0;
		uint32 blocks = 0;
		uint32 taints = 0;
		std::atomic<uint32> checks = 0;
		std::atomic<uint32> idles = 0;
		std::atomic<uint32> skips = 0; //Iterations skipped in counter loops
		COUNTER_LOOP counterLoop;
	};
	enum
	{
		AUTOIDLE_MAX_STATS = 64,
		AUTOIDLE_MAX_REJECTS = 512,
	};
	AUTOIDLE_STAT g_autoIdleStats[AUTOIDLE_MAX_STATS];
	std::atomic<uint32> g_autoIdleStatCount = 0;
	std::mutex g_autoIdleRejectMutex;
	std::map<uint32, std::string> g_autoIdleRejects;

	struct AUTOIDLE_STATE
	{
		uint32 head = MIPS_INVALID_PC;
		bool tainted = false;
		bool blocked = false; //Taint blocks compiled before their loop was found are waiting to be recompiled
		uint128 gpr[32];
		uint32 hilo[8];
	};
	AUTOIDLE_STATE g_autoIdleState;

	//Previous pass through a counter loop's check (kept apart so an enclosing wait loop still idles)
	struct COUNTER_STATE
	{
		uint32 head = MIPS_INVALID_PC;
		bool tainted = false;
		int32 quota = 0;
		uint128 gpr[32];
		uint32 hilo[8];
	};
	COUNTER_STATE g_counterState;

	int32 GetCounterStep(const COUNTER_LOOP& loop, uint32 reg)
	{
		for(uint32 i = 0; i < loop.regCount; i++)
		{
			if(loop.regs[i] == reg) return loop.steps[i];
		}
		return 0;
	}

	//Value of a comparison operand at iteration j (0: the next one, -1: the one that just ended). Counters
	//and offsets are 32-bit adds: the value is exact (and moves monotonically) while it stays in int32
	//and, for unsigned compares, keeps the sign it had at iteration -1. Returns false outside that range.
	bool GetCounterOperand(const COUNTER_OPERAND& operand, const COUNTER_LOOP& loop, const MIPSSTATE& state, int64 j, bool isUnsigned, int64& result)
	{
		if(operand.isConst)
		{
			result = operand.value;
			return true;
		}
		const auto& reg = state.nGPR[operand.reg];
		int64 step = GetCounterStep(loop, operand.reg);
		if((step == 0) && (operand.value == 0))
		{
			result = static_cast<int64>(reg.nD0);
			return true;
		}
		int64 base = static_cast<int64>(static_cast<int32>(reg.nV0)) + operand.value;
		int64 first = base - step;
		result = base + (j * step);
		if((step == 0) && ((first < INT32_MIN) || (first > INT32_MAX))) result = static_cast<int32>(static_cast<uint32>(first)); //Wrapped constant offset
		if((step != 0) && ((result < INT32_MIN) || (result > INT32_MAX) || (first < INT32_MIN) || (first > INT32_MAX))) return false;
		if((step != 0) && isUnsigned && ((result < 0) != (first < 0))) return false;
		return true;
	}

	bool EvalCounterPred(const COUNTER_PRED& pred, const COUNTER_LOOP& loop, const MIPSSTATE& state, int64 j, bool& result)
	{
		bool isUnsigned = (pred.op == COUNTER_PRED_SLTU);
		int64 a = 0, b = 0;
		if(!GetCounterOperand(pred.a, loop, state, j, isUnsigned, a) || !GetCounterOperand(pred.b, loop, state, j, isUnsigned, b)) return false;
		switch(pred.op)
		{
		case COUNTER_PRED_SLT:
			result = (a < b);
			break;
		case COUNTER_PRED_SLTU:
			result = (static_cast<uint64>(a) < static_cast<uint64>(b));
			break;
		default:
			result = (a == b);
			break;
		}
		return true;
	}

	//Largest n <= limit such that the comparison gives the same result for iterations 0 to n - 1 as for
	//the one that just ended (-1). In the counters' monotonic range, < can change only once and == can
	//match only once.
	int64 GetCounterPredLimit(const COUNTER_PRED& pred, const COUNTER_LOOP& loop, const MIPSSTATE& state, int64 limit)
	{
		bool observed = false;
		if(!EvalCounterPred(pred, loop, state, -1, observed)) return 0;
		auto same = [&](int64 j) {
			bool result = false;
			return EvalCounterPred(pred, loop, state, j, result) && (result == observed);
		};
		if(pred.op == COUNTER_PRED_EQ)
		{
			//Two equal operands become different on the next iteration (one of them moves)
			if(observed) return 0;
			//First match: the counter operand moves by its step, the other one is fixed
			bool counterA = !pred.a.isConst && (GetCounterStep(loop, pred.a.reg) != 0);
			const auto& counter = counterA ? pred.a : pred.b;
			const auto& other = counterA ? pred.b : pred.a;
			int64 step = GetCounterStep(loop, counter.reg);
			int64 x0 = 0, y = 0;
			if(GetCounterOperand(counter, loop, state, 0, false, x0) && GetCounterOperand(other, loop, state, 0, false, y))
			{
				int64 gap = y - x0;
				if(((gap == 0) || ((gap > 0) == (step > 0))) && ((gap % step) == 0)) limit = std::min(limit, gap / step);
			}
		}
		//Last iteration with the same result (monotonic, and the range check is monotonic too)
		int64 lo = 0, hi = limit;
		while(lo < hi)
		{
			int64 mid = lo + (hi - lo + 1) / 2;
			if(same(mid - 1))
				lo = mid;
			else
				hi = mid - 1;
		}
		return lo;
	}

	void CounterLoopCheck(MIPSSTATE& state, uint32 head, AUTOIDLE_STAT& stat)
	{
		auto& counterState = g_counterState;
		const auto& loop = stat.counterLoop;
		bool canSkip = !counterState.tainted && !loop.ambiguous && (counterState.head == head) && (state.nHasException == MIPS_EXCEPTION_NONE) &&
		               (counterState.quota > state.cycleQuota) && (memcmp(counterState.hilo, state.nHI, sizeof(counterState.hilo)) == 0);
		//Same registers as at the previous pass except the counters, which moved by their step
		for(uint32 reg = 0; canSkip && (reg < 32); reg++)
		{
			const auto& prev = counterState.gpr[reg];
			const auto& curr = state.nGPR[reg];
			int32 step = GetCounterStep(loop, reg);
			if(step == 0)
			{
				canSkip = (memcmp(&prev, &curr, sizeof(curr)) == 0);
				continue;
			}
			canSkip = (static_cast<uint32>(curr.nV0 - prev.nV0) == static_cast<uint32>(step)) &&
			          (curr.nV1 == ((curr.nV0 & 0x80000000) ? ~0U : 0U)) && (curr.nV2 == prev.nV2) && (curr.nV3 == prev.nV3);
		}
		if(canSkip)
		{
			int64 cyclesPerIteration = counterState.quota - state.cycleQuota;
			//The EE stops at the first block that takes the quota to 0 or below: the skipped iterations must
			//leave some, like the ones they replace
			int64 count = (cyclesPerIteration <= loop.maxIterationCycles) ? ((state.cycleQuota - 1) / cyclesPerIteration) : 0;
			for(uint32 i = 0; (i < loop.predCount) && (count > 0); i++)
			{
				count = std::min(count, GetCounterPredLimit(loop.preds[i], loop, state, count));
			}
			if(count > 0)
			{
				for(uint32 i = 0; i < loop.regCount; i++)
				{
					auto& reg = state.nGPR[loop.regs[i]];
					reg.nV0 += static_cast<uint32>(count * loop.steps[i]);
					reg.nV1 = (reg.nV0 & 0x80000000) ? ~0U : 0U;
				}
				state.cycleQuota -= static_cast<int32>(count * cyclesPerIteration);
				stat.skips.fetch_add(static_cast<uint32>(count), std::memory_order_relaxed);
			}
		}
		counterState.head = head;
		counterState.tainted = false;
		counterState.quota = state.cycleQuota;
		memcpy(counterState.gpr, state.nGPR, sizeof(counterState.gpr));
		memcpy(counterState.hilo, state.nHI, sizeof(counterState.hilo));
	}

	AUTOIDLE_STAT* FindAutoIdleStat(uint32 head)
	{
		uint32 count = g_autoIdleStatCount.load(std::memory_order_acquire);
		for(uint32 i = 0; i < count; i++)
		{
			if(g_autoIdleStats[i].head.load(std::memory_order_relaxed) == head) return &g_autoIdleStats[i];
		}
		return nullptr;
	}

	//Returns false if the loop can't be tracked (counter loops need their entry)
	bool AddAutoIdleStat(uint32 head, uint32 branch, uint32 insts, uint32 blocks, uint32 taints, const COUNTER_LOOP& counterLoop)
	{
		if(auto stat = FindAutoIdleStat(head))
		{
			//Analyzed again (code changed) or another loop branching to the same head
			if(stat->branch != branch) stat->counterLoop.ambiguous = true;
			bool ambiguous = stat->counterLoop.ambiguous;
			stat->counterLoop = counterLoop;
			stat->counterLoop.ambiguous = ambiguous;
			return true;
		}
		uint32 index = g_autoIdleStatCount.load(std::memory_order_relaxed);
		if(index >= AUTOIDLE_MAX_STATS) return (counterLoop.regCount == 0);
		auto& stat = g_autoIdleStats[index];
		stat.head = head;
		stat.branch = branch;
		stat.insts = insts;
		stat.blocks = blocks;
		stat.taints = taints;
		stat.counterLoop = counterLoop;
		g_autoIdleStatCount.store(index + 1, std::memory_order_release);
		return true;
	}

	void AddAutoIdleReject(uint32 head, uint32 branch, const char* reason)
	{
		std::lock_guard<std::mutex> lock(g_autoIdleRejectMutex);
		if((g_autoIdleRejects.size() >= AUTOIDLE_MAX_REJECTS) || g_autoIdleRejects.count(head)) return;
		char line[256];
		snprintf(line, sizeof(line), "R %08X %08X %s\n", head, branch, reason);
		g_autoIdleRejects[head] = line;
	}

	//Hardware registers that games poll and whose read has no side effect
	bool IsSafeIoRead(uint32 address)
	{
		switch((address & 0x1FFFFFFF) & ~0xF)
		{
		case 0x10000000: //T0_COUNT
		case 0x10000800: //T1_COUNT
		case 0x10001000: //T2_COUNT
		case 0x10001800: //T3_COUNT
		case 0x10002010: //IPU_CTRL
		case 0x10003020: //GIF_STAT
		case 0x10003800: //VIF0_STAT
		case 0x10003C00: //VIF1_STAT
		case 0x10008000: //D0_CHCR
		case 0x10009000: //D1_CHCR
		case 0x1000A000: //D2_CHCR
		case 0x1000B000: //D3_CHCR
		case 0x1000B400: //D4_CHCR
		case 0x1000C000: //D5_CHCR
		case 0x1000C400: //D6_CHCR
		case 0x1000C800: //D7_CHCR
		case 0x1000D000: //D8_CHCR
		case 0x1000D400: //D9_CHCR
		case 0x1000E010: //D_STAT
		case 0x1000F000: //INTC_STAT
		case 0x1000F010: //INTC_MASK
		case 0x1000F200: //SB_MSCOM
		case 0x1000F210: //SB_SMCOM
		case 0x1000F220: //SB_MSFLG
		case 0x1000F230: //SB_SMFLG
		case 0x1000F240: //SB_CTRL
		case 0x12001000: //GS_CSR
			return true;
		default:
			return false;
		}
	}

	bool IsIoAddress(uint32 address)
	{
		address &= 0x1FFFFFFF;
		return (address >= 0x10000000) && (address < 0x20000000);
	}

	//Syscalls a wait loop may call: FlushCache / iFlushCache (no-ops here)
	bool IsSafeSyscall(uint32 number)
	{
		if(number & 0x80000000) number = 0 - number;
		return (number == 0x64) || (number == 0x68);
	}

	//Target of a branch or jump instruction with a static target, MIPS_INVALID_PC for anything else
	uint32 GetStaticTarget(uint32 address, uint32 inst)
	{
		uint32 op = inst >> 26;
		uint32 rt = (inst >> 16) & 0x1F;
		uint32 target = address + 4 + (static_cast<int16>(inst) * 4);
		switch(op)
		{
		case 0x01: //bltz, bgez, bltzl, bgezl, bltzal, bgezal, bltzall, bgezall
			return ((rt <= 0x03) || ((rt >= 0x10) && (rt <= 0x13))) ? target : MIPS_INVALID_PC;
		case 0x02: //j
		case 0x03: //jal
			return ((address + 4) & 0xF0000000) | ((inst & 0x03FFFFFF) << 2);
		case 0x04: //beq
		case 0x05: //bne
		case 0x06: //blez
		case 0x07: //bgtz
		case 0x14: //beql
		case 0x15: //bnel
		case 0x16: //blezl
		case 0x17: //bgtzl
			return target;
		default:
			return MIPS_INVALID_PC;
		}
	}

	//Abstract value of a register or stack slot in CWaitLoopAnalyzer
	enum VALUE_KIND : uint32
	{
		VALUE_UNKNOWN,
		VALUE_CONST, //value: the constant
		VALUE_ORIG,  //value: register index, the value it had at the start of the iteration
		VALUE_UNDEF, //stack slot not written yet
	};

	struct VALUE
	{
		VALUE_KIND kind = VALUE_UNKNOWN;
		uint32 value = 0;
		int32 offset = 0; //VALUE_ORIG: added to the register (32-bit add, addiu) when not 0
		bool operator==(const VALUE& rhs) const
		{
			return (kind == rhs.kind) && (value == rhs.value) && (offset == rhs.offset);
		}
	};

	class CWaitLoopAnalyzer
	{
	public:
		CWaitLoopAnalyzer(CMIPS& context, uint32 head, uint32 branchAddress)
		    : m_context(context)
		    , m_head(head)
		    , m_regionEnd(branchAddress + 4)
		{
		}

		//Returns nullptr if the loop is a wait loop, otherwise the reason it isn't
		const char* Analyze()
		{
			if(m_head >= PS2::EE_RAM_SIZE) return "head outside RAM";
			if((m_regionEnd - m_head) > MAX_REGION_SIZE) return "region too large";

			STATE initial;
			for(uint32 reg = 0; reg < REG_COUNT; reg++)
			{
				initial.regs[reg] = (reg == 0) ? VALUE{VALUE_CONST, 0} : VALUE{VALUE_ORIG, reg};
			}
			for(auto& slot : initial.slots)
			{
				slot = VALUE{VALUE_UNDEF, 0};
			}
			for(auto& value : m_backEdge)
			{
				value = VALUE{VALUE_UNDEF, 0};
			}
			if(auto reason = Enqueue(NODE_KEY{m_head, m_head, 0, {}}, initial)) return reason;
			uint32 steps = 0;
			while(!m_queue.empty())
			{
				if(++steps > MAX_STEPS) return "too many steps";
				auto key = m_queue.front();
				m_queue.pop_front();
				auto& node = m_nodes[key];
				node.queued = false;
				STATE state = node.state;
				if(auto reason = Step(key, state)) return reason;
			}
			if(!m_hasBackEdge) return "no safe path back to the head";
			//Counters: registers read by the loop that it increments by a constant (addiu), and registers
			//that end the iteration as a copy of one (the same value plus a constant, same step)
			uint64 counters = 0;
			std::array<int32, REG_COUNT> counterSteps = {};
			for(uint32 reg = 1; reg < REG_HI; reg++)
			{
				const auto& value = m_backEdge[reg];
				if((m_liveIn & (1ULL << reg)) && (value.kind == VALUE_ORIG) && (value.value == reg) && (value.offset != 0))
				{
					counters |= (1ULL << reg);
					counterSteps[reg] = value.offset;
				}
			}
			uint64 copies = 0;
			for(uint32 reg = 1; reg < REG_COUNT; reg++)
			{
				const auto& value = m_backEdge[reg];
				if((value.kind != VALUE_ORIG) || (value.value == reg) || !(counters & (1ULL << value.value))) continue;
				if(reg >= REG_HI) return "a counter is copied to HI/LO";
				copies |= (1ULL << reg);
				counterSteps[reg] = counterSteps[value.value];
			}
			//Registers read before being written must be the same at the start of every iteration (or counters)
			for(uint32 reg = 1; reg < REG_COUNT; reg++)
			{
				if(!(m_liveIn & (1ULL << reg))) continue;
				const auto& value = m_backEdge[reg];
				bool same = (value.kind == VALUE_CONST) || ((value.kind == VALUE_ORIG) && (value.value == reg) && (value.offset == 0)) ||
				            ((counters | copies) & (1ULL << reg));
				if(!same) return "a register read by the loop changes every iteration";
			}
			if(auto reason = MakeCounterLoop(counters | copies, counterSteps)) return reason;
			return CheckInteriorEntries();
		}

		const COUNTER_LOOP& GetCounterLoop() const
		{
			return m_counterLoop;
		}

		uint32 GetInstructionCount() const
		{
			std::set<uint32> addresses;
			for(const auto& node : m_nodes) addresses.insert(node.first.pc);
			return static_cast<uint32>(addresses.size());
		}

		uint32 GetBlockCount() const
		{
			std::set<uint32> blocks;
			for(const auto& node : m_nodes) blocks.insert(node.first.blockStart);
			return static_cast<uint32>(blocks.size());
		}

		const std::set<uint32>& GetPages() const
		{
			return m_pages;
		}

		const std::set<uint32>& GetTaintBlocks() const
		{
			return m_taintBlocks;
		}

	private:
		enum
		{
			MAX_REGION_SIZE = 0x400,
			MAX_STEPS = 4000,
			MAX_DEPTH = 3,
			MAX_SLOTS = 32,
			MAX_TAINT_BLOCKS = 4,
			MAX_PREDS = 16,
			SCAN_WINDOW = 0x2000,
			REG_V0 = 2,
			REG_V1 = 3,
			REG_SP = 29,
			REG_RA = 31,
			REG_HI = 32,
			REG_LO = 33,
			REG_COUNT = 34,
		};

		struct NODE_KEY
		{
			uint32 pc;
			uint32 blockStart; //Where the emulator's block containing pc starts on this path
			uint32 depth;
			std::array<uint32, MAX_DEPTH> ret;
			bool operator<(const NODE_KEY& rhs) const
			{
				return std::tie(pc, blockStart, depth, ret) < std::tie(rhs.pc, rhs.blockStart, rhs.depth, rhs.ret);
			}
		};

		struct STATE
		{
			int32 spDelta = 0;
			std::array<VALUE, REG_COUNT> regs;
			std::array<VALUE, MAX_SLOTS> slots;
		};

		struct NODE
		{
			STATE state;
			bool reached = false;
			bool queued = false;
		};

		struct PRED
		{
			COUNTER_PRED_OP op;
			VALUE a;
			VALUE b;
			bool operator==(const PRED& rhs) const
			{
				return (op == rhs.op) && (a == rhs.a) && (b == rhs.b);
			}
		};

		VALUE Meet(const VALUE& a, const VALUE& b)
		{
			if((a.kind == VALUE_UNDEF) || (b.kind == VALUE_UNDEF)) return VALUE{VALUE_UNDEF, 0};
			if(a == b) return a;
			//The registers' values are lost (a counter can't flow there)
			Use(a);
			Use(b);
			return VALUE{VALUE_UNKNOWN, 0};
		}

		//A register's original value is used for something else than a copy, an addiu or a comparison
		void Use(const VALUE& value)
		{
			if(value.kind == VALUE_ORIG) m_escapes |= (1ULL << value.value);
		}

		//Comparison (slt*, branch) of two values: kept for counter loops
		void Compare(COUNTER_PRED_OP op, const VALUE& a, const VALUE& b)
		{
			if((a.kind != VALUE_ORIG) && (b.kind != VALUE_ORIG)) return;
			PRED pred = {op, a, b};
			if(std::find(std::begin(m_preds), std::end(m_preds), pred) != std::end(m_preds)) return;
			if(m_preds.size() >= MAX_PREDS)
			{
				Use(a);
				Use(b);
				return;
			}
			m_preds.push_back(pred);
		}

		const char* MakeCounterLoop(uint64 counters, const std::array<int32, REG_COUNT>& steps)
		{
			if(counters == 0) return nullptr;
			if(counters & m_escapes) return "a counter is used by more than comparisons";
			//An iteration executes each analyzed instruction (in its call context) at most once: bounds
			//the cycles of an iteration, to tell an iteration from leaving and entering the loop again
			if(m_hasInnerLoop) return "counter loop with an inner loop";
			m_counterLoop.maxIterationCycles = static_cast<uint32>(m_nodes.size() * 2); //Delay slots aren't nodes
			auto isCounter = [&](const VALUE& value) { return (value.kind == VALUE_ORIG) && (counters & (1ULL << value.value)); };
			auto makeOperand = [](const VALUE& value) {
				COUNTER_OPERAND operand;
				operand.isConst = (value.kind == VALUE_CONST);
				operand.reg = operand.isConst ? 0 : static_cast<uint8>(value.value);
				operand.value = operand.isConst ? static_cast<int32>(value.value) : value.offset;
				return operand;
			};
			for(const auto& pred : m_preds)
			{
				bool counterA = isCounter(pred.a);
				bool counterB = isCounter(pred.b);
				if(!counterA && !counterB) continue;
				if(counterA && counterB) return "two counters compared";
				const auto& other = counterA ? pred.b : pred.a;
				if((other.kind != VALUE_CONST) && (other.kind != VALUE_ORIG)) return "a counter is compared with a changing value";
				if((other.kind == VALUE_ORIG) && (other.value >= REG_HI)) return "a counter is compared with HI/LO";
				if(m_counterLoop.predCount >= COUNTER_MAX_PREDS) return "too many counter comparisons";
				auto& counterPred = m_counterLoop.preds[m_counterLoop.predCount++];
				counterPred.op = pred.op;
				counterPred.a = makeOperand(pred.a);
				counterPred.b = makeOperand(pred.b);
			}
			for(uint32 reg = 1; reg < REG_HI; reg++)
			{
				if(!(counters & (1ULL << reg))) continue;
				if(m_counterLoop.regCount >= COUNTER_MAX_REGS) return "too many counters";
				m_counterLoop.regs[m_counterLoop.regCount] = static_cast<uint8>(reg);
				m_counterLoop.steps[m_counterLoop.regCount] = steps[reg];
				m_counterLoop.regCount++;
			}
			return nullptr;
		}

		const char* CheckInteriorEntries()
		{
			//Code outside of the loop that jumps into it (other than to the head, or to the loop's
			//condition from just before the head) would make iterations that weren't analyzed
			uint32 scanStart = (m_head > SCAN_WINDOW) ? (m_head - SCAN_WINDOW) : 0;
			uint32 scanEnd = std::min<uint32>(m_regionEnd + SCAN_WINDOW, PS2::EE_RAM_SIZE);
			for(uint32 address = scanStart; address < scanEnd; address += 4)
			{
				if((address >= (m_head - 8)) && (address <= m_regionEnd)) continue;
				uint32 target = GetStaticTarget(address, m_context.m_pMemoryMap->GetInstruction(address));
				if((target > m_head) && (target <= m_regionEnd)) return "code outside of the loop jumps into it";
			}
			return nullptr;
		}

		//use: the value is used for something else than a copy, an addiu or a comparison
		VALUE Read(const STATE& state, uint32 reg, bool use = true)
		{
			const auto& value = state.regs[reg];
			if((value.kind == VALUE_ORIG) && (value.value == reg)) m_liveIn |= (1ULL << reg);
			if(use) Use(value);
			return value;
		}

		//Returns false if the instruction writes sp other than by adding a constant to it
		bool Write(STATE& state, uint32 reg, VALUE value = VALUE{})
		{
			if(reg == 0) return true;
			if(reg == REG_SP) return false;
			state.regs[reg] = value;
			return true;
		}

		int GetSlot(int32 offset)
		{
			auto it = m_slots.find(offset);
			if(it != std::end(m_slots)) return it->second;
			if(m_slots.size() >= MAX_SLOTS) return -1;
			int index = static_cast<int>(m_slots.size());
			m_slots[offset] = index;
			return index;
		}

		//Effect of one non-branch instruction on the state. Returns false if the instruction isn't safe.
		bool Apply(uint32 inst, STATE& state)
		{
			uint32 op = inst >> 26;
			uint32 rs = (inst >> 21) & 0x1F;
			uint32 rt = (inst >> 16) & 0x1F;
			uint32 rd = (inst >> 11) & 0x1F;
			uint32 funct = inst & 0x3F;
			int32 imm = static_cast<int16>(inst);
			uint32 uimm = inst & 0xFFFF;
			switch(op)
			{
			case 0x00:
				switch(funct)
				{
				case 0x00: //sll
				case 0x02: //srl
				case 0x03: //sra
				case 0x38: //dsll
				case 0x3A: //dsrl
				case 0x3B: //dsra
				case 0x3C: //dsll32
				case 0x3E: //dsrl32
				case 0x3F: //dsra32
					if(inst == 0) return true;
					Read(state, rt);
					return Write(state, rd);
				case 0x21: //addu
				case 0x25: //or
				case 0x2D: //daddu
				{
					//Register moves
					auto a = Read(state, rs, false);
					auto b = Read(state, rt, false);
					VALUE result;
					if(rt == 0)
						result = a;
					else if(rs == 0)
						result = b;
					else
					{
						Use(a);
						Use(b);
						if((a.kind == VALUE_CONST) && (b.kind == VALUE_CONST))
							result = VALUE{VALUE_CONST, (funct == 0x25) ? (a.value | b.value) : (a.value + b.value)};
					}
					return Write(state, rd, result);
				}
				case 0x04: //sllv
				case 0x06: //srlv
				case 0x07: //srav
				case 0x14: //dsllv
				case 0x16: //dsrlv
				case 0x17: //dsrav
				case 0x20: //add
				case 0x22: //sub
				case 0x23: //subu
				case 0x24: //and
				case 0x26: //xor
				case 0x27: //nor
				case 0x2C: //dadd
				case 0x2E: //dsub
				case 0x2F: //dsubu
					Read(state, rs);
					Read(state, rt);
					return Write(state, rd);
				case 0x2A: //slt
				case 0x2B: //sltu
					Compare((funct == 0x2A) ? COUNTER_PRED_SLT : COUNTER_PRED_SLTU, Read(state, rs, false), Read(state, rt, false));
					return Write(state, rd);
				case 0x0A: //movz
				case 0x0B: //movn
					Read(state, rs);
					Read(state, rt);
					Read(state, rd);
					return Write(state, rd);
				case 0x0F: //sync
					return true;
				case 0x10: //mfhi
					return Write(state, rd, Read(state, REG_HI, false));
				case 0x12: //mflo
					return Write(state, rd, Read(state, REG_LO, false));
				case 0x11: //mthi
					return Write(state, REG_HI, Read(state, rs, false));
				case 0x13: //mtlo
					return Write(state, REG_LO, Read(state, rs, false));
				case 0x18: //mult
				case 0x19: //multu
					Read(state, rs);
					Read(state, rt);
					return Write(state, REG_HI) && Write(state, REG_LO) && Write(state, rd);
				case 0x1A: //div
				case 0x1B: //divu
					Read(state, rs);
					Read(state, rt);
					return Write(state, REG_HI) && Write(state, REG_LO);
				case 0x0C: //syscall
				{
					auto number = Read(state, REG_V1);
					if((number.kind != VALUE_CONST) || !IsSafeSyscall(number.value)) return false;
					Read(state, 4); //FlushCache's operation
					return Write(state, REG_V0) && Write(state, REG_V1);
				}
				default:
					return false;
				}
			case 0x08: //addi
			case 0x09: //addiu
			case 0x18: //daddi
			case 0x19: //daddiu
			{
				if(rt == REG_SP)
				{
					if(rs != REG_SP) return false;
					state.spDelta += imm;
					return true;
				}
				auto value = Read(state, rs, false);
				if(value.kind == VALUE_CONST) return Write(state, rt, VALUE{VALUE_CONST, value.value + imm});
				//32-bit adds keep track of the register they add to (counters)
				if((value.kind == VALUE_ORIG) && (op <= 0x09)) return Write(state, rt, VALUE{VALUE_ORIG, value.value, value.offset + imm});
				Use(value);
				return Write(state, rt);
			}
			case 0x0D: //ori
			{
				auto value = Read(state, rs);
				return Write(state, rt, (value.kind == VALUE_CONST) ? VALUE{VALUE_CONST, value.value | uimm} : VALUE{});
			}
			case 0x0A: //slti
			case 0x0B: //sltiu
				Compare((op == 0x0A) ? COUNTER_PRED_SLT : COUNTER_PRED_SLTU, Read(state, rs, false), VALUE{VALUE_CONST, static_cast<uint32>(imm)});
				return Write(state, rt);
			case 0x0C: //andi
			case 0x0E: //xori
				Read(state, rs);
				return Write(state, rt);
			case 0x0F: //lui
				return Write(state, rt, VALUE{VALUE_CONST, uimm << 16});
			case 0x1A: //ldl
			case 0x1B: //ldr
			case 0x22: //lwl
			case 0x26: //lwr
				Read(state, rt);
				[[fallthrough]];
			case 0x1E: //lq
			case 0x20: //lb
			case 0x21: //lh
			case 0x23: //lw
			case 0x24: //lbu
			case 0x25: //lhu
			case 0x27: //lwu
			case 0x37: //ld
			{
				auto base = Read(state, rs);
				VALUE result;
				if(rs == REG_SP)
				{
					int32 offset = state.spDelta + imm;
					if(offset < 0)
					{
						//Stack slot of a called function: must have been written in this iteration
						int slot = GetSlot(offset);
						if(slot < 0) return false;
						if(state.slots[slot].kind == VALUE_UNDEF) return false;
						result = state.slots[slot];
					}
				}
				else if(base.kind == VALUE_CONST)
				{
					uint32 address = base.value + imm;
					if(IsIoAddress(address) && !IsSafeIoRead(address)) return false;
				}
				return Write(state, rt, result);
			}
			case 0x1F: //sq
			case 0x28: //sb
			case 0x29: //sh
			case 0x2A: //swl
			case 0x2B: //sw
			case 0x2C: //sdl
			case 0x2D: //sdr
			case 0x2E: //swr
			case 0x3F: //sd
			{
				auto value = Read(state, rt);
				Read(state, rs);
				if(rs != REG_SP) return false;
				int32 size = (op == 0x1F) ? 16 : (op == 0x28) ? 1 : (op == 0x29) ? 2 : ((op == 0x2C) || (op == 0x2D) || (op == 0x3F)) ? 8 : 4;
				int32 offset = state.spDelta + imm;
				if((offset + size) > 0) return false;
				int slot = GetSlot(offset);
				if(slot < 0) return false;
				state.slots[slot] = (size >= 4) ? value : VALUE{};
				return true;
			}
			case 0x2F: //cache
			case 0x33: //pref
				Read(state, rs);
				return true;
			default:
				return false;
			}
		}

		//The path through key.blockStart isn't safe: that block will mark the iteration as not idle
		const char* Taint(const NODE_KEY& key)
		{
			if((key.depth == 0) && (key.blockStart == m_head)) return "unsafe instruction in the loop's first block";
			m_taintBlocks.insert(key.blockStart);
			if(m_taintBlocks.size() > MAX_TAINT_BLOCKS) return "too many unsafe paths";
			return nullptr;
		}

		const char* Step(const NODE_KEY& key, STATE state)
		{
			uint32 pc = key.pc;
			if(pc >= PS2::EE_RAM_SIZE) return Taint(key);
			m_pages.insert(pc / MIPS_PAGE_SIZE);
			uint32 inst = m_context.m_pMemoryMap->GetInstruction(pc);
			uint32 op = inst >> 26;
			uint32 rs = (inst >> 21) & 0x1F;
			uint32 rt = (inst >> 16) & 0x1F;
			uint32 funct = inst & 0x3F;

			bool isJr = (op == 0x00) && (funct == 0x08);
			uint32 target = GetStaticTarget(pc, inst);
			if((op == 0x00) && (funct == 0x09)) return Taint(key); //jalr
			if((op == 0x01) && (rt >= 0x10)) return Taint(key);    //b*zal
			if(isJr && (rs != REG_RA)) return Taint(key);

			if(!isJr && (target == MIPS_INVALID_PC))
			{
				if(!Apply(inst, state)) return Taint(key);
				NODE_KEY next = key;
				next.pc = pc + 4;
				//syscall ends the emulator's block
				if((op == 0x00) && (funct == 0x0C)) next.blockStart = pc + 4;
				return Enqueue(next, state);
			}

			//Branches and jumps, with their delay slot
			if(key.depth != 0)
			{
				if(isJr) Read(state, REG_RA);
			}
			switch(op)
			{
			case 0x01: //bltz, bgez...: rs < 0
				Compare(COUNTER_PRED_SLT, Read(state, rs, false), VALUE{VALUE_CONST, 0});
				break;
			case 0x06: //blez, bgtz...: 0 < rs
			case 0x07:
			case 0x16:
			case 0x17:
				Compare(COUNTER_PRED_SLT, VALUE{VALUE_CONST, 0}, Read(state, rs, false));
				break;
			case 0x04:
			case 0x05:
			case 0x14:
			case 0x15:
				Compare(COUNTER_PRED_EQ, Read(state, rs, false), Read(state, rt, false));
				break;
			}
			uint32 delayInst = m_context.m_pMemoryMap->GetInstruction(pc + 4);
			if(GetStaticTarget(pc + 4, delayInst) != MIPS_INVALID_PC) return Taint(key);
			if(((delayInst >> 26) == 0) && (((delayInst & 0x3F) == 0x08) || ((delayInst & 0x3F) == 0x09) || ((delayInst & 0x3F) == 0x0C))) return Taint(key);
			STATE taken = state;
			if(!Apply(delayInst, taken)) return Taint(key);
			m_pages.insert((pc + 4) / MIPS_PAGE_SIZE);

			if(isJr)
			{
				if(key.depth == 0) return nullptr; //Returns from the function containing the loop: leaves the loop
				NODE_KEY parent = key;
				parent.depth--;
				parent.ret[parent.depth] = 0;
				return Flow(parent, key.ret[key.depth - 1], taken);
			}
			if(op == 0x03) //jal
			{
				if(key.depth >= MAX_DEPTH) return Taint(key);
				Write(taken, REG_RA, VALUE{VALUE_CONST, pc + 8});
				NODE_KEY callee = key;
				callee.ret[callee.depth] = pc + 8;
				callee.depth++;
				return Flow(callee, target, taken);
			}
			//Branch back to something else than the loop's head: inner loop
			if((target <= pc) && ((key.depth != 0) || (target > m_head))) m_hasInnerLoop = true;
			if(auto reason = Flow(key, target, taken)) return reason;
			bool conditional = (op != 0x02) && !(((op & 0x0F) == 0x04) && (rs == 0) && (rt == 0));
			if(conditional)
			{
				bool likely = (op >= 0x14) || ((op == 0x01) && (rt & 2));
				return Flow(key, pc + 8, likely ? state : taken);
			}
			return nullptr;
		}

		//Control transfer to pc: starts a new block
		const char* Flow(const NODE_KEY& key, uint32 pc, const STATE& state)
		{
			if(key.depth == 0)
			{
				if(pc == m_head)
				{
					if(state.spDelta != 0) return "sp not restored at the end of an iteration";
					m_hasBackEdge = true;
					for(uint32 reg = 0; reg < REG_COUNT; reg++)
					{
						m_backEdge[reg] = m_firstBackEdge ? state.regs[reg] : Meet(m_backEdge[reg], state.regs[reg]);
					}
					m_firstBackEdge = false;
					return nullptr;
				}
				//Leaves the loop
				if((pc < m_head) || (pc > m_regionEnd)) return nullptr;
			}
			NODE_KEY next = key;
			next.pc = pc;
			next.blockStart = pc;
			return Enqueue(next, state);
		}

		const char* Enqueue(const NODE_KEY& key, const STATE& state)
		{
			auto& node = m_nodes[key];
			if(!node.reached)
			{
				node.reached = true;
				node.state = state;
			}
			else
			{
				if(node.state.spDelta != state.spDelta) return "sp differs between paths";
				STATE merged = node.state;
				for(uint32 i = 0; i < REG_COUNT; i++) merged.regs[i] = Meet(merged.regs[i], state.regs[i]);
				for(uint32 i = 0; i < MAX_SLOTS; i++) merged.slots[i] = Meet(merged.slots[i], state.slots[i]);
				if((merged.regs == node.state.regs) && (merged.slots == node.state.slots)) return nullptr;
				node.state = merged;
			}
			if(!node.queued)
			{
				node.queued = true;
				m_queue.push_back(key);
			}
			return nullptr;
		}

		CMIPS& m_context;
		uint32 m_head;
		uint32 m_regionEnd;
		std::map<NODE_KEY, NODE> m_nodes;
		std::deque<NODE_KEY> m_queue;
		std::map<int32, int> m_slots;
		std::set<uint32> m_pages;
		std::set<uint32> m_taintBlocks;
		std::array<VALUE, REG_COUNT> m_backEdge;
		bool m_firstBackEdge = true;
		uint64 m_liveIn = 0;
		uint64 m_escapes = 0; //Registers whose original value is used for something else than a copy, an addiu or a comparison
		std::vector<PRED> m_preds;
		COUNTER_LOOP m_counterLoop;
		bool m_hasBackEdge = false;
		bool m_hasInnerLoop = false;
	};
}

//Called by auto idle loop blocks when their branch back to the loop head is taken
extern "C" void EeAutoIdleCheck(CMIPS* context)
{
	auto& autoIdle = g_autoIdleState;
	if(!g_eeAutoIdle || autoIdle.blocked) return;
	auto& state = context->m_State;
	uint32 head = state.nDelayedJumpAddr;
	auto stat = FindAutoIdleStat(head);
	if(stat) stat->checks.fetch_add(1, std::memory_order_relaxed);
	if(stat && (stat->counterLoop.regCount != 0))
	{
		CounterLoopCheck(state, head, *stat);
		return;
	}
	g_counterState.head = MIPS_INVALID_PC;
	static_assert(sizeof(autoIdle.hilo) == (sizeof(state.nHI) * 4));
	if(!autoIdle.tainted && (autoIdle.head == head) && (state.nHasException == MIPS_EXCEPTION_NONE) &&
	   (memcmp(autoIdle.gpr, state.nGPR, sizeof(autoIdle.gpr)) == 0) &&
	   (memcmp(autoIdle.hilo, state.nHI, sizeof(autoIdle.hilo)) == 0))
	{
		state.nHasException = MIPS_EXCEPTION_IDLE;
		if(stat) stat->idles.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	autoIdle.head = head;
	autoIdle.tainted = false;
	memcpy(autoIdle.gpr, state.nGPR, sizeof(autoIdle.gpr));
	memcpy(autoIdle.hilo, state.nHI, sizeof(autoIdle.hilo));
}

//Called by taint blocks: the current iteration of a wait loop isn't idle
extern "C" void EeAutoIdleTaint(CMIPS*)
{
	g_autoIdleState.tainted = true;
	g_counterState.tainted = true;
}

//New EE time slice: counter loops measure the cycles of an iteration within one slice
void EeAutoIdleNewSlice()
{
	g_counterState.head = MIPS_INVALID_PC;
}

//Detected loops ("L head branch instructions blocks taintBlocks checks idles") and rejected candidates
//("R head branch reason"); counters since the previous call with reset
std::string EeAutoIdleStats(bool reset)
{
	std::string result;
	char line[256];
	uint32 count = g_autoIdleStatCount.load(std::memory_order_acquire);
	for(uint32 i = 0; i < count; i++)
	{
		auto& stat = g_autoIdleStats[i];
		uint32 checks = reset ? stat.checks.exchange(0) : stat.checks.load();
		uint32 idles = reset ? stat.idles.exchange(0) : stat.idles.load();
		uint32 skips = reset ? stat.skips.exchange(0) : stat.skips.load();
		snprintf(line, sizeof(line), "L %08X %08X %u %u %u %u %u %u %u\n", stat.head.load(), stat.branch, stat.insts, stat.blocks, stat.taints, checks, idles,
		         stat.counterLoop.regCount, skips);
		result += line;
	}
	std::lock_guard<std::mutex> lock(g_autoIdleRejectMutex);
	for(const auto& reject : g_autoIdleRejects)
	{
		result += reject.second;
	}
	return result;
}

bool CEeExecutor::IsAutoIdleLoop(uint32 start, uint32 end)
{
	if(!g_eeAutoIdle) return false;
	if(end == start) return false;
	uint32 branchAddress = end - 4;
	uint32 inst = m_context.m_pMemoryMap->GetInstruction(branchAddress);
	//Branches and j (not jal, jr, jalr) back to an earlier address
	uint32 op = inst >> 26;
	if((op == 0x00) || (op == 0x03)) return false;
	uint32 head = GetStaticTarget(branchAddress, inst);
	if(head == MIPS_INVALID_PC) return false;
	if((head > branchAddress) || ((branchAddress - head) > 0x400)) return false;
	//Loops already handled by a GameConfig IdleLoopBlock
	auto idleLoopBlock = m_idleLoopBlocks.lower_bound(head);
	if((idleLoopBlock != std::end(m_idleLoopBlocks)) && (idleLoopBlock->first <= branchAddress)) return false;

	CWaitLoopAnalyzer analyzer(m_context, head, branchAddress);
	if(auto reason = analyzer.Analyze())
	{
		AddAutoIdleReject(head, branchAddress, reason);
		return false;
	}
	if(!AddAutoIdleStat(head, branchAddress, analyzer.GetInstructionCount(), analyzer.GetBlockCount(), static_cast<uint32>(analyzer.GetTaintBlocks().size()), analyzer.GetCounterLoop()))
	{
		AddAutoIdleReject(head, branchAddress, "too many loops");
		return false;
	}

	//Taint blocks compiled before are recompiled once the EE leaves the JIT code (no idle until then)
	for(uint32 taintBlock : analyzer.GetTaintBlocks())
	{
		if(!m_autoIdleTaintBlocks.insert(taintBlock).second) continue;
		if(HasBlockAt(taintBlock))
		{
			m_autoIdlePendingInvalidations.push_back(taintBlock);
			g_autoIdleState.blocked = true;
		}
	}

	//Code outside of the block that the decision depends on: the block is invalidated when it changes
	m_autoIdleLoops.erase(std::remove_if(std::begin(m_autoIdleLoops), std::end(m_autoIdleLoops),
	                                     [start](const AUTOIDLE_LOOP& loop) { return loop.blockStart == start; }),
	                      std::end(m_autoIdleLoops));
	AUTOIDLE_LOOP loop;
	loop.blockStart = start;
	for(uint32 page : analyzer.GetPages())
	{
		uint32 pageAddress = page * MIPS_PAGE_SIZE;
		if(pageAddress >= 0x100000) SetMemoryProtected(m_ram + pageAddress, MIPS_PAGE_SIZE, true);
		loop.pages.push_back(page);
	}
	m_autoIdleLoops.push_back(std::move(loop));
	return true;
}

void CEeExecutor::FlushAutoIdleInvalidations()
{
	if(m_autoIdlePendingInvalidations.empty()) return;
	auto blocks = std::move(m_autoIdlePendingInvalidations);
	m_autoIdlePendingInvalidations.clear();
	for(uint32 block : blocks)
	{
		CGenericMipsExecutor::ClearActiveBlocksInRange(block, block + 4, false);
	}
	g_autoIdleState.blocked = false;
}


CEeExecutor::CEeExecutor(CMIPS& context, uint8* ram)
    : CGenericMipsExecutor(context, 0x20000000, BLOCK_CATEGORY_PS2_EE)
    , m_ram(ram)
{
	m_pageSize = framework_getpagesize();
#if defined(__EMSCRIPTEN__)
	m_pageSize = MIPS_PAGE_SIZE;
	m_protectedPages.resize(PS2::EE_RAM_SIZE / MIPS_PAGE_SIZE, 0);
	m_context.m_writeWatchHandler = &CEeExecutor::WriteWatchHandler;
	m_context.m_blockFnLookup = m_blockLookup.GetFnTable();
#endif
}

#if defined(__EMSCRIPTEN__)

int CEeExecutor::Execute(int cycles)
{
#if defined(__EMSCRIPTEN__)
	//Blocks compiled on the helper thread (EE, IOP, VU0: all run on this thread) replace their interpreter
	AsyncCompile::Update();
#endif
	m_executing = true;
#if !defined(PROFILE) && !defined(DEBUGGER_INCLUDED)
	//Same loop as CGenericMipsExecutor::Execute, but the call goes through the function index table
	//(kept for block chaining): one dependent load less per block than going through the CBasicBlock.
	//Index 0 (no block yet) takes the usual path, which compiles it.
	m_context.m_State.cycleQuota = cycles;
	uint32** fnTable = m_blockLookup.GetFnTable();
	while(m_context.m_State.nHasException == 0)
	{
		uint32 address = m_context.m_State.nPC & m_addressMask;
		uint32 fctIdx = fnTable[address >> 16][(address & 0xFFFF) / 4];
		if(fctIdx != 0)
		{
			reinterpret_cast<void (*)(CMIPS*)>(static_cast<uintptr_t>(fctIdx))(&m_context);
		}
		else
		{
			auto block = m_blockLookup.FindBlockAt(address);
			//A block that ran in the interpreter (async compile) and got its compiled function since
			if(!block->IsEmpty() && (block->GetFunctionIndex() != 0)) m_blockLookup.UpdateFunctionIndex(block);
			block->Execute();
		}
	}
	m_context.m_State.nHasException &= ~MIPS_EXCEPTION_STATUS_QUOTADONE;
	int result = m_context.m_State.cycleQuota;
#else
	int result = CGenericMipsExecutor::Execute(cycles);
#endif
	m_executing = false;
	FlushAutoIdleInvalidations();
	return result;
}

void CEeExecutor::WriteWatchHandler(CMIPS* context, uint32 address, uint32 size)
{
	//Only called from JIT code (store slow path), so a block is executing
	auto executor = static_cast<CEeExecutor*>(context->m_executor.get());
	if((address >= PS2::EE_RAM_SIZE) && executor->m_ioWriteHandler)
	{
		executor->m_ioWriteHandler(address);
	}
	executor->InvalidateWrittenRange(address, size, true);
}

void CEeExecutor::NotifyRamWritten(uint32 address, uint32 size)
{
	if(!g_eeExecutor) return;
	g_eeExecutor->InvalidateWrittenRange(address, size, g_eeExecutor->m_executing);
}

extern uint32 g_eeBatchSize;

void CEeExecutor::QueueForBatch(const BasicBlockPtr& block)
{
	if(block->GetBatchModule().empty()) return;
	m_batchPending.push_back(block);
	if(m_batchPending.size() >= g_eeBatchSize) FlushBatch();
}

void CEeExecutor::FlushBatch()
{
	for(const auto& block : CBasicBlock::BatchModules(m_batchPending))
	{
		m_blockLookup.UpdateFunctionIndex(block.get());
	}
}

void CEeExecutor::InvalidateWrittenRange(uint32 address, uint32 size, bool executing)
{
	if(size == 0) return;
	if(address >= PS2::EE_RAM_SIZE) return;
	uint32 end = std::min<uint32>(address + size, PS2::EE_RAM_SIZE);
	for(uint32 page = address / MIPS_PAGE_SIZE; page <= (end - 1) / MIPS_PAGE_SIZE; page++)
	{
		if(m_protectedPages[page])
		{
#ifdef PROFILE
			PROFILE_COUNT(PCNT_EE_INVAL, 1);
			CProfiler::GetInstance().Log("W %08X %u pc=%08X\n", address, size, m_context.m_State.nPC);
#endif
			ClearActiveBlocksInRange(page * MIPS_PAGE_SIZE, (page + 1) * MIPS_PAGE_SIZE, executing);
		}
	}
}

#endif

void CEeExecutor::SetBlockFpRoundingModes(BlockFpRoundingModeMap blockFpRoundingModes)
{
	m_blockFpRoundingModes = std::move(blockFpRoundingModes);
}

void CEeExecutor::SetBlockFpUseAccurateAddSub(BlockFpUseAccurateAddSubSet blockFpUseAccurateAddSub)
{
	m_blockFpUseAccurateAddSub = std::move(blockFpUseAccurateAddSub);
}

void CEeExecutor::SetIdleLoopBlocks(IdleLoopBlockMap idleLoopBlocks)
{
	m_idleLoopBlocks = std::move(idleLoopBlocks);
}

void CEeExecutor::AddExceptionHandler()
{
	assert(g_eeExecutor == nullptr);
	g_eeExecutor = this;

#ifdef DISABLE_PROTECTION
	return;
#endif

#if defined(_WIN32)
	m_handler = AddVectoredExceptionHandler(TRUE, &CEeExecutor::HandleException);
	assert(m_handler != NULL);
#elif defined(__unix__) || defined(__ANDROID__)
	struct sigaction sigAction;
	sigAction.sa_handler = nullptr;
	sigAction.sa_sigaction = &HandleException;
	sigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&sigAction.sa_mask);
	int result = sigaction(SIGSEGV, &sigAction, nullptr);
	assert(result >= 0);
#elif defined(__APPLE__) && !TARGET_OS_TV
	if(!m_running)
	{
		kern_return_t result = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &m_port);
		assert(result == KERN_SUCCESS);

		m_running = true;
		m_handlerThread = std::thread([this]() { HandlerThreadProc(); });
	}

	AttachExceptionHandlerToThread();
#endif
}

void CEeExecutor::RemoveExceptionHandler()
{
#ifndef DISABLE_PROTECTION

#if defined(_WIN32)
	RemoveVectoredExceptionHandler(m_handler);
#elif defined(__APPLE__) && !TARGET_OS_TV
	m_running = false;
	m_handlerThread.join();
#endif

#endif //!DISABLE_PROTECTION

	g_eeExecutor = nullptr;
}

void CEeExecutor::AttachExceptionHandlerToThread()
{
	//Only necessary for macOS and iOS since the handler is set on a thread basis
#if defined(__APPLE__) && !TARGET_OS_TV
	assert(m_running);

	auto result = mach_port_insert_right(mach_task_self(), m_port, m_port, MACH_MSG_TYPE_MAKE_SEND);
	assert(result == KERN_SUCCESS);

	result = thread_set_exception_ports(mach_thread_self(), EXC_MASK_BAD_ACCESS, m_port, EXCEPTION_STATE | MACH_EXCEPTION_CODES, STATE_FLAVOR);
	assert(result == KERN_SUCCESS);

	result = mach_port_mod_refs(mach_task_self(), m_port, MACH_PORT_RIGHT_SEND, -1);
	assert(result == KERN_SUCCESS);
#endif
}

void CEeExecutor::Reset()
{
	SetMemoryProtected(m_ram, PS2::EE_RAM_SIZE, false);
	m_cachedBlocks.clear();
#if defined(__EMSCRIPTEN__)
	m_batchPending.clear();
#endif
	m_blockFpRoundingModes.clear();
	m_idleLoopBlocks.clear();
	m_autoIdleLoops.clear();
	m_autoIdleTaintBlocks.clear();
	m_autoIdlePendingInvalidations.clear();
	g_autoIdleState = AUTOIDLE_STATE();
	g_counterState = COUNTER_STATE();
	CGenericMipsExecutor::Reset();
}

void CEeExecutor::ClearActiveBlocksInRange(uint32 start, uint32 end, bool executing)
{
	uint32 rangeSize = end - start;
	SetMemoryProtected(m_ram + start, rangeSize, false);
	CGenericMipsExecutor::ClearActiveBlocksInRange(start, end, executing);

	//Auto idle loops whose analysis read code in this range are analyzed again
	if(m_autoIdleLoops.empty()) return;
	uint32 firstPage = start / MIPS_PAGE_SIZE;
	uint32 lastPage = (end - 1) / MIPS_PAGE_SIZE;
	std::vector<uint32> staleBlocks;
	auto loopIterator = std::begin(m_autoIdleLoops);
	while(loopIterator != std::end(m_autoIdleLoops))
	{
		bool stale = std::any_of(std::begin(loopIterator->pages), std::end(loopIterator->pages),
		                         [&](uint32 page) { return (page >= firstPage) && (page <= lastPage); });
		if(stale)
		{
			staleBlocks.push_back(loopIterator->blockStart);
			loopIterator = m_autoIdleLoops.erase(loopIterator);
		}
		else
		{
			loopIterator++;
		}
	}
	for(uint32 blockStart : staleBlocks)
	{
		CGenericMipsExecutor::ClearActiveBlocksInRange(blockStart, blockStart + 4, executing);
	}
}

BasicBlockPtr CEeExecutor::BlockFactory(CMIPS& context, uint32 start, uint32 end)
{
	uint32 blockSize = (end - start) + 4;

	//Kernel area is below 0x100000 and isn't protected. Some games will write code in there
	//but it is safe to assume that it won't change (code writes some data just besides itself
	//so it keeps generating exceptions, making the game slower)
	if(start >= 0x100000 && start < PS2::EE_RAM_SIZE)
	{
		SetMemoryProtected(m_ram + start, blockSize, true);
	}

	auto blockMemory = reinterpret_cast<uint32*>(alloca(blockSize));
	for(uint32 address = start; address <= end; address += 4)
	{
		uint32 index = (address - start) / 4;
		uint32 opcode = m_context.m_pMemoryMap->GetInstruction(address);
		blockMemory[index] = opcode;
	}

	auto xxHash = XXH3_128bits(blockMemory, blockSize);
	uint128 hash;
	memcpy(&hash, &xxHash, sizeof(xxHash));
	static_assert(sizeof(hash) == sizeof(xxHash));
	auto blockKey = std::make_pair(hash, blockSize);

	bool hasBreakpoint = m_context.HasBreakpointInRange(start, end);

	std::optional<Jitter::CJitter::ROUNDINGMODE> blockFpRoundingModeOverride;
	if(auto blockFpRoundingModeIterator = m_blockFpRoundingModes.find(start);
	   blockFpRoundingModeIterator != std::end(m_blockFpRoundingModes))
	{
		blockFpRoundingModeOverride = blockFpRoundingModeIterator->second;
	}

	bool isIdleLoopBlockOverride = false;
	if(auto idleLoopBlockIterator = m_idleLoopBlocks.find(start);
	   idleLoopBlockIterator != std::end(m_idleLoopBlocks))
	{
		const auto& checkBlockKey = idleLoopBlockIterator->second;
		if(!checkBlockKey.has_value() || (checkBlockKey.value() == blockKey))
		{
			isIdleLoopBlockOverride = true;
		}
	}

	bool fpUseAccurateAddSub = (m_blockFpUseAccurateAddSub.count(start) != 0);

	bool isAutoIdleLoop = !hasBreakpoint && !isIdleLoopBlockOverride && !CEeBasicBlock::IsCodeIdleLoopBlock(context, start, end) && IsAutoIdleLoop(start, end);
	bool isAutoIdleTaintBlock = (m_autoIdleTaintBlocks.count(start) != 0);

	bool isCacheableBlock = !hasBreakpoint && !blockFpRoundingModeOverride.has_value() && !isIdleLoopBlockOverride && !fpUseAccurateAddSub &&
	                        !isAutoIdleLoop && !isAutoIdleTaintBlock;
	if(isCacheableBlock)
	{
		auto blockIterator = m_cachedBlocks.find(blockKey);
		if(blockIterator != std::end(m_cachedBlocks))
		{
			const auto& basicBlock(blockIterator->second);
			if(basicBlock->GetBeginAddress() == start && basicBlock->GetEndAddress() == end)
			{
				uint32 recycleCount = basicBlock->GetRecycleCount();
				basicBlock->SetRecycleCount(std::min<uint32>(RECYCLE_NOLINK_THRESHOLD, recycleCount + 1));
				return basicBlock;
			}
			else
			{
				auto result = std::make_shared<CEeBasicBlock>(context, start, end, m_blockCategory);
				result->CopyFunctionFrom(basicBlock);
				return result;
			}
		}
	}

	auto result = std::make_shared<CEeBasicBlock>(context, start, end, m_blockCategory);
	if(blockFpRoundingModeOverride.has_value())
	{
		result->SetFpRoundingMode(blockFpRoundingModeOverride.value());
	}
	if(isIdleLoopBlockOverride)
	{
		result->SetIsIdleLoopBlock();
	}
	if(isAutoIdleLoop)
	{
		result->SetIsAutoIdleLoopBlock();
	}
	if(isAutoIdleTaintBlock)
	{
		result->SetIsAutoIdleTaintBlock();
	}
	if(fpUseAccurateAddSub)
	{
		result->AddBlockCompileHints(CMA_EE::COMPILEHINT_FPU_USE_ACCURATE_ADD_SUB);
	}

	result->Compile();
	if(isCacheableBlock)
	{
		m_cachedBlocks.insert(std::make_pair(blockKey, result));
	}
#if defined(__EMSCRIPTEN__)
	QueueForBatch(result);
#endif
	return result;
}

bool CEeExecutor::HandleAccessFault(intptr_t ptr)
{
	ptrdiff_t addr = reinterpret_cast<uint8*>(ptr) - m_ram;
	if(addr >= 0 && addr < PS2::EE_RAM_SIZE)
	{
		addr &= ~(m_pageSize - 1);
		ClearActiveBlocksInRange(addr, addr + m_pageSize, true);
		return true;
	}
	return false;
}

void CEeExecutor::SetMemoryProtected(void* addr, size_t size, bool protect)
{
#ifdef DISABLE_PROTECTION
	return;
#endif

#if defined(_WIN32)
	DWORD oldProtect = 0;
	BOOL result = VirtualProtect(addr, size, protect ? PAGE_READONLY : PAGE_READWRITE, &oldProtect);
	assert(result == TRUE);
#elif defined(__EMSCRIPTEN__)
	{
		ptrdiff_t offset = reinterpret_cast<uint8*>(addr) - m_ram;
		assert(offset >= 0 && (offset + size) <= PS2::EE_RAM_SIZE);
		if(size == 0) return;
		uint32 firstPage = static_cast<uint32>(offset) / MIPS_PAGE_SIZE;
		uint32 lastPage = static_cast<uint32>(offset + size - 1) / MIPS_PAGE_SIZE;
		static const uint32 mirrorBases[] = {0x00000000, 0x20000000, 0x30000000, 0x80000000};
		for(uint32 page = firstPage; page <= lastPage; page++)
		{
			if(m_protectedPages[page] == protect) continue;
			m_protectedPages[page] = protect;
			void* pagePtr = protect ? nullptr : (m_ram + (page * MIPS_PAGE_SIZE));
			for(uint32 mirrorBase : mirrorBases)
			{
				m_context.m_pageLookupWrite[(mirrorBase / MIPS_PAGE_SIZE) + page] = pagePtr;
			}
		}
	}
#elif defined(__unix__) || defined(__ANDROID__) || defined(__APPLE__)
	uintptr_t addrValue = reinterpret_cast<uintptr_t>(addr) & ~(m_pageSize - 1);
	addr = reinterpret_cast<void*>(addrValue);
	size = size + (m_pageSize - 1) & ~(m_pageSize - 1);
	int result = mprotect(addr, size, protect ? PROT_READ : PROT_READ | PROT_WRITE);
	assert(result >= 0);
#else
	assert(false);
#endif
}

#if defined(_WIN32)

LONG WINAPI CEeExecutor::HandleException(_EXCEPTION_POINTERS* exceptionInfo)
{
	return g_eeExecutor->HandleExceptionInternal(exceptionInfo);
}

LONG CEeExecutor::HandleExceptionInternal(_EXCEPTION_POINTERS* exceptionInfo)
{
	auto exceptionRecord = exceptionInfo->ExceptionRecord;
	if(exceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
	{
		if(HandleAccessFault(exceptionRecord->ExceptionInformation[1]))
		{
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__unix__) || defined(__ANDROID__)

void CEeExecutor::HandleException(int sigId, siginfo_t* sigInfo, void* baseContext)
{
	g_eeExecutor->HandleExceptionInternal(sigId, sigInfo, baseContext);
}

void CEeExecutor::HandleExceptionInternal(int sigId, siginfo_t* sigInfo, void* baseContext)
{
	if(sigId != SIGSEGV) return;
	if(HandleAccessFault(reinterpret_cast<intptr_t>(sigInfo->si_addr)))
	{
		return;
	}
	signal(SIGSEGV, SIG_DFL);
}

#elif defined(__APPLE__) && !TARGET_OS_TV

void CEeExecutor::HandlerThreadProc()
{
#pragma pack(push, 4)
	struct INPUT_MESSAGE
	{
		mach_msg_header_t head;
		NDR_record_t ndr;
		exception_type_t exception;
		mach_msg_type_number_t codeCount;
		intptr_t code[2];
		int flavor;
		mach_msg_type_number_t stateCount;
		natural_t state[STATE_FLAVOR_COUNT];
		mach_msg_trailer_t trailer;
	};
	struct OUTPUT_MESSAGE
	{
		mach_msg_header_t head;
		NDR_record_t ndr;
		kern_return_t result;
		int flavor;
		mach_msg_type_number_t stateCount;
		natural_t state[STATE_FLAVOR_COUNT];
	};
#pragma pack(pop)

	while(m_running)
	{
		kern_return_t result = KERN_SUCCESS;

		INPUT_MESSAGE inMsg;
		result = mach_msg(&inMsg.head, MACH_RCV_MSG | MACH_RCV_LARGE | MACH_RCV_TIMEOUT, 0, sizeof(inMsg), m_port, 1000, MACH_PORT_NULL);
		if(result == MACH_RCV_TIMED_OUT) continue;
		assert(result == KERN_SUCCESS);

		assert(inMsg.head.msgh_id == 2406); //MACH_EXCEPTION_RAISE_RPC
		assert(inMsg.flavor == STATE_FLAVOR);
		assert(inMsg.stateCount == STATE_FLAVOR_COUNT);

		bool success = HandleAccessFault(inMsg.code[1]);

		OUTPUT_MESSAGE outMsg;
		outMsg.head.msgh_bits = MACH_MSGH_BITS(MACH_MSGH_BITS_REMOTE(inMsg.head.msgh_bits), 0);
		outMsg.head.msgh_remote_port = inMsg.head.msgh_remote_port;
		outMsg.head.msgh_local_port = MACH_PORT_NULL;
		outMsg.head.msgh_id = inMsg.head.msgh_id + 100;
		outMsg.head.msgh_size = sizeof(outMsg);
		outMsg.ndr = inMsg.ndr;

		if(success)
		{
			outMsg.result = KERN_SUCCESS;
			outMsg.flavor = STATE_FLAVOR;
			outMsg.stateCount = STATE_FLAVOR_COUNT;
			memcpy(outMsg.state, inMsg.state, STATE_FLAVOR_COUNT * sizeof(natural_t));
		}
		else
		{
			outMsg.result = KERN_FAILURE;
			outMsg.flavor = 0;
			outMsg.stateCount = 0;
		}

		result = mach_msg(&outMsg.head, MACH_SEND_MSG | MACH_RCV_LARGE, sizeof(outMsg), 0, MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
		assert(result == KERN_SUCCESS);
	}
}

#endif
