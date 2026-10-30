# VU1 JIT code in God of War combat: why superblocks did little in game, next cuts (vu1e, 2026-10-30)

Goal: more game frames/s in GoW combat (VU1-thread bound) by cutting VU1 JIT code cost, generic and exact.

Play branch `vu1e` (on web-perf 8edc38c9), CodeGen branch `vu1e` (on 7bcaeab), pushed to the forks.
Clone `.claude/worktrees/vu1e/play-vu1e`, build dir `build_cmake/build/b`. Scratchpad `vu1e/`:
`build.sh <variant> [prof]` (fails on compile errors now), `run.mjs` (modes time / hash / diff / dump,
`prof=<vbl>,<secs>` CPU profiles of all workers, `dump=F000xxxx,...` VU block wasm dumps), `go.sh`
(memory, user-playing and load gates outside the bench lock), `cat.mjs` (VU1 worker categories, leaf
first), `selfin.mjs` (self leaves under a frame), `tools/x64.mjs` (TurboFan x64 of a dumped block:
`node --no-liftoff --no-wasm-lazy-compilation --print-wasm-code tools/x64.mjs block.wasm`), `abab.sh`,
`med.mjs`.

## Step 1: profile of the current build (web-perf 8edc38c9 = vu1d4) in game
Profiling build `public/emu-vu1e0p` (-O1 --profiling-funcs relink), gow-fight, `blockcache=0&vujit=1,0,1`,
20 s from vblank 500, one profile with superblocks (default) and one with `&vusb=0`.

| VU1 worker, % of busy samples | superblocks | `vusb=0` |
|---|---|---|
| JIT code (leaf) | 62.7 | 63.1 |
| XGKICK -> GIF PACKED / GS write buffer | 14.5 | 14.9 |
| VIF unpack | 9.0 | 9.5 |
| VIF decode | 5.1 | 5.1 |
| block lookup / plan / partition | 4.5 | 2.9 |
| VU dispatcher | 2.3 | 2.4 |
| idle | 5.7% of samples | 6.5% |

JIT code relative to VIF work (same work per frame in both): 4.45 (superblocks) vs 4.32 (`vusb=0`).
So in game superblocks did not make the JIT code measurably cheaper, and they added ~1.6 points of lookup
(PlanSuperBlock 1.4% self, BlockFactory hashing the whole superblock extent ~1%, after every micro memory
upload GoW makes).

Why the replay said -28% and the game -2..4%:
1. **Workload mix.** The replay capture was dominated by the 2EE0/2F38 two-block loop. In game that loop
   (2EE0_2F8C + 2EE0_2F74 + 2F38_2F8C) is ~10.5% of VU1 samples with superblocks vs ~14.1% for its blocks
   without (normalized: -23%, consistent with the replay). The rest of the hot code (2B90 loop, 1258..1F1C,
   08C0..0E2C, 2840, 1678 standalone) shows no measurable gain: these are either single blocks, or the
   superblock is entered and left after one or two parts, so it pays the entry loads / exit stores of the
   union of all parts' state (~36 i32 + 25 v128 promoted values in 2B90_2C3C) instead of one block's.
2. **Register pressure inside loop functions.** x64 of the 2B90 superblock (two 9-pair parts + JR parts):
   1367 instructions, 486 of them stack spill stores/reloads (`[rbp-..]`); one 9-pair part is ~350
   instructions inside the loop vs 505 for the standalone block (with its entry loads and exit stores).
   The saving per iteration is real (~30% static) but V8 keeps far more values live than it has registers,
   so a lot of the "state in locals" goes through the stack anyway, plus ~40 phi moves at the loop header.
3. **Lookup overhead** after block-set switches (above), which the replay doesn't have (it sorts captures
   by micro memory contents).

Where the code goes in a hot 9-pair part (x64, 2B90 loop, ~350 instructions): flag pipeline ring writes
(4 stores + index chain + address per FMAC in the last 4 cycles, ~60), stall-check prologue against
pipeFmacWrite (~35), phi moves (~40), clamps (~20), VU memory base reload + address per LQ/SQ (~15),
the rest spread (FMAC ops, blends, MakeSz, Q/DIV, quota check, dispatch). No single item is above ~17%.

## Changes
1. **CodeGen e225a7a**: a temporary reference set once to context + constant (`PushRelAddrRef`) is replaced
   in i32 LoadFromRef/StoreAtRef by the context, with the constant as the memarg offset. The VU flag
   pipeline writes values[idx] and pipeTimes[idx] of the MAC and sticky rings: they now share one address
   (`movl [rdx+0x64c]`, `movl [rdx+0x66c]`) instead of one `lea` per store (26 such `lea`s in the 2B90
   superblock before, 0 after); flag pipeline reads (8 slots x 2 loads) likewise. Same bytes stored.
2. **Play 830a4f79**: block-set carry-over checks only the pages of a superblock's parts (its code depends
   only on their words), not its whole extent; `CanJoinSuperBlock` hashes a part only when a no-clamping or
   compile-hint key of that size exists (the result is unchanged: keys hold the size). `CODEGEN_VERSION` 50.

## Validation
VU differential test (`&vujit=1,1&vu1thread=0&blockcache=0&hash=15`, 60 s, `vuDiffStats()`):
gow-fight runs=4823445 mismatches=0, play runs=357827 mismatches=0.

## Results
Not measured yet: from ~08:40 to 12:00 the box sat below the 2900 MB MemAvailable gate (user Java/Gradle
processes, ~4 GB) with load up to 14, and the session was reaped for low memory. The few runs taken under
that load (gow-fight 8-57 fps for the same build) were thrown away. Timing (gow-fight, mtvu-ffx, car ABAB vs
`emu-vu1d4`) and the hash gate are still to do with `public/emu-vu1e1`.

## Not done / next
- Cross-part flag-write skipping (vu1d item 2): in diff mode flag writes are never skipped
  (`CanSkipFlagWrite` checks `!g_vuDiffMode`), so the diff test would not cover it; the hash gate would be
  the only check. Gain is limited where a part's other successor is a JR exit (the 2B90 loop): only ~1
  of ~4 tail FMACs per part becomes skippable. Larger where the exit part has its own FMAC early (2EE0 loop).
- Stall-check prologue: per-edge constant in loop functions would need per-predecessor code or a runtime
  flag; a branchless form is not shorter (~same instruction count).
- m_vuMem promotion: the base is spilled anyway under this register pressure, gain ~1 load per LQ/SQ.
- The bigger lever remains structural: fewer live values per loop function (V8 spills ~35% of instructions)
  and the GIF PACKED path (14.5%, other agent).
