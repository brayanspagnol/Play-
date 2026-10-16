#!/usr/bin/env python3
"""Tier-2 milestone M0a: how many EE block transitions would stay inside tier-2 regions?

Reads an edge profile written by a PROFILE build (CEeExecutor::StopEdgeProfile, fetched from the page
with Module.eeEdgeProfile() after a bench.html &time=from,to window) and simulates the region formation of
notes/tier2-jit-design.md section 5:
  - a region is one entry block E plus the blocks statically reachable from E (fall-through, conditional
    branches, J/B) inside one MIPS function: JR/JALR, ERET, SYSCALL, VCALLMS and calls (JAL/BxxAL) end it;
  - at most --max-blocks blocks and --max-insns instructions, successors within --span bytes of E;
  - only blocks at or above 0x100000 whose pages tier 1 write-protects;
  - every eligible block start gets a region (the tier2=all mode, the ceiling of the sampled mode).
The trace (blocks in execution order) is then replayed: a transition is internal when the running region
contains the next block as a static successor of the current one and the block did not exit with an
exception (cycle quota, syscall, interrupt check, idle). Any other transition goes back to the dispatcher,
which enters the region of the next block if it has one.

usage: regions.py profile.bin [--max-blocks 32] [--max-insns 256] [--span 4096] [--sweep] [--selftest]
"""
import argparse
import collections
import struct
import sys

BRANCH_NORMAL, BRANCH_NODELAY = 1, 2
INVALID_PC = 1
MAX_BLOCK_SIZE = 0x1000
KERNEL_END = 0x100000


class Profile:
    def __init__(self, edges, block_ends, trace, pages, protected, branches):
        self.edges = edges            # (src, dst) -> [count kind 0, 1, 2]
        self.block_ends = block_ends  # start -> end of the tier-1 block (0 if gone)
        self.trace = trace            # block start | exit kind (0 none, 1 quota, 2 other exception)
        self.pages = pages            # page number -> 1024 words
        self.protected = protected    # page number -> bool
        self.branches = branches      # address -> (MIPS_BRANCH_TYPE, effective address)


def load(path):
    data = open(path, "rb").read()
    u = memoryview(data).cast("I")
    pos = 0

    def take(n):
        nonlocal pos
        r = u[pos:pos + n]
        pos += n
        return r

    magic, version = take(2)
    assert magic == 0x304D3245 and version == 1, "not an edge profile"
    (n,) = take(1)
    e = take(n * 5)
    edges = {(e[i], e[i + 1]): [e[i + 2], e[i + 3], e[i + 4]] for i in range(0, n * 5, 5)}
    (n,) = take(1)
    b = take(n * 2)
    block_ends = {b[i]: b[i + 1] for i in range(0, n * 2, 2)}
    (n,) = take(1)
    trace = take(n).tolist()
    (n,) = take(1)
    pages, protected = {}, {}
    for _ in range(n):
        address, prot = take(2)
        pages[address >> 12] = take(1024)
        protected[address >> 12] = bool(prot)
    (n,) = take(1)
    br = take(n * 3)
    branches = {br[i]: (br[i + 1], br[i + 2]) for i in range(0, n * 3, 3)}
    assert pos == len(u), "trailing data"
    return Profile(edges, block_ends, trace, pages, protected, branches)


class Code:
    """Static view of the code: tier-1 block partition and block exits."""

    def __init__(self, prof):
        self.p = prof
        self.cache = {}

    def word(self, address):
        page = self.p.pages.get(address >> 12)
        return None if page is None else page[(address & 0xFFF) >> 2]

    def btype(self, address):
        return self.p.branches.get(address, (0, INVALID_PC))[0]

    def partition(self, start):
        # Same rule as CGenericMipsExecutor::PartitionFunction (end is the last instruction)
        for address in range(start, start + MAX_BLOCK_SIZE, 4):
            t = self.btype(address)
            if t == BRANCH_NORMAL:
                end = address + 4
                if self.btype(end) == BRANCH_NORMAL:
                    end = address
                return end
            if t == BRANCH_NODELAY:
                return address
        return start + MAX_BLOCK_SIZE

    def info(self, start):
        """(end, instruction count, exit class, static successors, eligible)"""
        r = self.cache.get(start)
        if r is not None:
            return r
        if (start >> 12) not in self.p.pages:
            # Outside EE RAM (BIOS stubs): no code exported, take the tier-1 block as it ran; never in a region
            end = self.p.block_ends.get(start) or start
            r = self.cache[start] = (end, (end - start) // 4 + 1, "outside RAM", (), False)
            return r
        end = self.partition(start)
        size = (end - start) // 4 + 1
        pages = range(start >> 12, (end >> 12) + 1)
        eligible = start >= KERNEL_END and all(self.p.protected.get(pg, False) for pg in pages)
        succ = ()
        if end - 4 >= start and self.btype(end - 4) == BRANCH_NORMAL:
            target = self.p.branches[end - 4][1]
            op = self.word(end - 4)
            opc, rs, rt = op >> 26, (op >> 21) & 31, (op >> 16) & 31
            if target == INVALID_PC:
                cls = "indirect"                       # JR / JALR
            elif opc == 3 or (opc == 1 and 0x10 <= rt <= 0x13):
                cls = "call"                           # JAL / BLTZAL / BGEZAL (and likely forms)
            elif opc == 2 or (opc in (4, 0x14) and rs == rt):
                cls, succ = "jump", (target,)          # J / B (BEQ r,r)
            else:
                cls, succ = "cond", (target, end + 4)
        elif self.btype(end) == BRANCH_NORMAL:
            cls = "dsbranch"                           # branch with a branch in its delay slot: tier 1 only
        elif self.btype(end) == BRANCH_NODELAY:
            cls = "nodelay"                            # SYSCALL / ERET / VCALLMS
        else:
            cls, succ = "fall", (end + 4,)
        r = (end, size, cls, succ, eligible)
        self.cache[start] = r
        return r


def form_region(code, entry, max_blocks, max_insns, span):
    """Blocks of the region entered at 'entry' (breadth-first over static successors), or None."""
    end, size, cls, succ, eligible = code.info(entry)
    if not eligible:
        return None
    blocks = {entry}
    insns = size
    queue = collections.deque([entry])
    while queue:
        for s in code.info(queue.popleft())[3]:
            if s in blocks or abs(s - entry) > span:
                continue
            s_size, s_eligible = code.info(s)[1], code.info(s)[4]
            if not s_eligible or len(blocks) + 1 > max_blocks or insns + s_size > max_insns:
                continue
            blocks.add(s)
            insns += s_size
            queue.append(s)
    return frozenset(blocks)


def edge_breakdown(code, edges):
    """Transition counts by kind over all recorded edges (whole window)."""
    kinds = collections.Counter()
    for (src, dst), counts in edges.items():
        cls = code.info(src)[2]
        kinds["exception:quota"] += counts[1]
        kinds["exception:other"] += counts[2]
        if cls == "outside RAM":
            kinds["outside RAM (BIOS)"] += counts[0]
        elif cls in ("indirect", "nodelay"):
            kinds["indirect"] += counts[0]
        elif cls == "call":
            kinds["direct:call"] += counts[0]
        else:
            kinds["direct"] += counts[0]
    return kinds


def simulate(code, trace, max_blocks, max_insns, span):
    regions = {}

    def region(entry):
        r = regions.get(entry, False)
        if r is False:
            r = regions[entry] = form_region(code, entry, max_blocks, max_insns, span)
        return r

    internal = 0
    reasons = collections.Counter()
    entries = collections.Counter()
    internal_by_region = collections.Counter()
    insns_t2 = insns_all = 0
    cur = None
    cur_set = None
    first = trace[0] & ~3
    if region(first):
        cur, cur_set = first, region(first)
    for i in range(len(trace) - 1):
        a, kind, c = trace[i] & ~3, trace[i] & 3, trace[i + 1] & ~3
        a_info = code.info(a)
        size = a_info[1]
        insns_all += size
        if cur is not None:
            insns_t2 += size
        if cur is not None and kind == 0 and c in cur_set and c in a_info[3]:
            internal += 1
            internal_by_region[cur] += 1
            continue
        if kind:
            reasons["exception:quota" if kind == 1 else "exception:other"] += 1
        elif a_info[2] in ("indirect", "nodelay"):
            reasons["indirect"] += 1
        elif a_info[2] == "call":
            reasons["call"] += 1
        elif cur is None or not code.info(c)[4]:
            reasons["not eligible (kernel/unprotected)"] += 1
        elif c not in a_info[3]:
            reasons["not a static successor"] += 1
        else:
            reasons["outside region (caps/span)"] += 1
        r = region(c)
        cur, cur_set = (c, r) if r else (None, None)
        if r:
            entries[c] += 1
    total = len(trace) - 1
    sizes = [(len(regions[e]), sum(code.info(b)[1] for b in regions[e]), n) for e, n in entries.items()]
    n_entries = sum(n for _, _, n in sizes) or 1
    return {
        "transitions": total,
        "internal": internal,
        "internal_share": internal / total if total else 0.0,
        "reasons": reasons,
        "coverage": insns_t2 / insns_all if insns_all else 0.0,
        "regions_entered": len(entries),
        "regions_formed": sum(1 for r in regions.values() if r),
        "blocks_per_region": sum(b * n for b, _, n in sizes) / n_entries,
        "insns_per_region": sum(i * n for _, i, n in sizes) / n_entries,
        "blocks_per_dispatch": total / max(1, total - internal),
        "top_regions": [sum(n for _, n in internal_by_region.most_common(k)) / max(1, internal) for k in (10, 100, 1000)],
    }


def report(path, args):
    prof = load(path)
    code = Code(prof)
    # Our partition must match the tier-1 blocks that ran
    mismatch = [(hex(s), hex(e), hex(code.info(s)[0])) for s, e in prof.block_ends.items() if e and code.info(s)[0] != e]
    total_edges = sum(sum(c) for c in prof.edges.values())
    print(f"{path}: {len(prof.edges)} distinct edges, {total_edges} transitions, {len(prof.block_ends)} blocks "
          f"({len(mismatch)} partition mismatches {mismatch[:3]}), {sum(prof.protected.values())} protected pages, trace {len(prof.trace)}")
    for name, counts in (("window", edge_breakdown(code, prof.edges)), ("trace", trace_breakdown(code, prof.trace))):
        n = sum(counts.values()) or 1
        print(f"  kinds ({name}): " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in sorted(counts.items())))
    bound = static_bound(code, prof.edges)
    print(f"  static bound (direct non-call edge between eligible blocks, any region): {100 * bound:.1f}%")
    configs = [(args.max_blocks, args.max_insns, args.span)]
    if args.sweep:
        configs += [(16, 128, 4096), (64, 512, 4096), (32, 256, 8192), (128, 2048, 65536)]
    results = []
    for mb, mi, sp in configs:
        r = simulate(code, prof.trace, mb, mi, sp)
        results.append(r)
        n = r["transitions"] - r["internal"]
        print(f"  caps {mb} blocks / {mi} insns / span {sp}: internal {100 * r['internal_share']:.1f}% "
              f"of {r['transitions']} transitions, coverage {100 * r['coverage']:.1f}% of instructions, "
              f"{r['regions_entered']} regions entered ({r['blocks_per_region']:.1f} blocks, "
              f"{r['insns_per_region']:.0f} insns each, weighted), {r['blocks_per_dispatch']:.2f} blocks per dispatch")
        print("    internal transitions in the top 10/100/1000 regions: " + "/".join(f"{100 * x:.0f}%" for x in r["top_regions"]))
        print("    external: " + ", ".join(f"{k} {100 * v / r['transitions']:.1f}%" for k, v in r["reasons"].most_common()))
    i = results[0]["internal_share"]
    c = results[0]["coverage"]
    # Section 13 model: EE code E=0.42, dispatcher D=0.25, other O=0.33 of the VM thread
    print(f"  dispatcher: {100 * i:.0f}% fewer dispatches -> about {100 * 0.25 * i:.1f}% of the VM thread "
          f"(D=0.25; 0.23-0.29 measured), before region entry/exit costs")
    for s in (1.3, 1.75, 2.5):
        t = 0.42 * (1 - c) + 0.42 * c / s + 0.25 * (1 - i) + 0.33
        print(f"    model with s={s}: VM thread {t:.2f} -> {1 / t:.2f}x")
    print("  GO" if i >= 0.40 else "  NO-GO (internal transitions < 40%)")
    return results


def trace_breakdown(code, trace):
    kinds = collections.Counter()
    for i in range(len(trace) - 1):
        a, kind = trace[i] & ~3, trace[i] & 3
        if kind:
            kinds["exception:quota" if kind == 1 else "exception:other"] += 1
            continue
        cls = code.info(a)[2]
        kinds["outside RAM (BIOS)" if cls == "outside RAM" else "indirect" if cls in ("indirect", "nodelay")
              else "direct:call" if cls == "call" else "direct"] += 1
    return kinds


def static_bound(code, edges):
    good = total = 0
    for (src, dst), counts in edges.items():
        total += sum(counts)
        s = code.info(src)
        if s[4] and dst in s[3] and code.info(dst)[4]:
            good += counts[0]
    return good / total if total else 0.0


def selftest():
    # One protected page at 0x200000: a loop 0x200000 -> 0x200010 (BNE back to 0x200000), then JR RA at 0x200018
    words = [0] * 1024
    words[2] = (5 << 26) | (1 << 21) | (0xFFFD & 0xFFFF)    # 0x200008: BNE r1, r0, 0x200000
    words[6] = (0 << 26) | (31 << 21) | 8                   # 0x200018: JR RA
    page = struct.pack("<1024I", *words)
    branches = {0x200008: (BRANCH_NORMAL, 0x200000), 0x200018: (BRANCH_NORMAL, INVALID_PC)}
    trace = [0x200000, 0x200000, 0x200000, 0x200010 | 2, 0x200000, 0x200010, 0x50000]
    prof = Profile({}, {}, trace, {0x200: memoryview(page).cast("I")}, {0x200: True}, branches)
    code = Code(prof)
    assert code.info(0x200000)[:4] == (0x20000C, 4, "cond", (0x200000, 0x200010))
    assert code.info(0x200010)[2] == "indirect"
    assert form_region(code, 0x200000, 32, 256, 4096) == {0x200000, 0x200010}
    r = simulate(code, trace, 32, 256, 4096)
    # internal: loop twice and fall through, exception exit, fall through again, JR into the kernel (no region)
    assert r["internal"] == 4 and r["reasons"]["exception:other"] == 1 and r["reasons"]["indirect"] == 1, r
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="*")
    ap.add_argument("--max-blocks", type=int, default=32)
    ap.add_argument("--max-insns", type=int, default=256)
    ap.add_argument("--span", type=int, default=4096)
    ap.add_argument("--sweep", action="store_true", help="also try smaller and larger caps")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.profiles:
        selftest()
    for p in a.profiles:
        report(p, a)
