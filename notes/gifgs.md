# gifgs: GIF PACKED decode on the GS thread

Branch `gifgs` (fork), based on web-perf 8edc38c9. Baseline build: public/emu-vu1dpub (built from 8edc38c9).
Test builds: public/emu-gifgs1 (the change), emu-gifgsv (cross-check build), emu-gifgscnt (qword counters).

## Change
`CGIF::ProcessPacked`: when the tag's register list has no A+D (0x0E) entry, the qwords of the packet (as far
as the current transfer reaches) are copied raw into the GS write buffer behind a two-entry header:
`{PACKED_RUN_MARKER (0xFF), REGS}`, `{(NREG-1) | (first REGS index << 4), count | (Q << 32)}`. The GS thread
decodes them in `SubmitWriteBufferImpl` -> `ProcessPackedRun` with the same per-qword decoder
(`CGSHandler::DecodePackedQword`) that the VU1/EE side keeps using for A+D lists, runs under 4 qwords and a
nearly full write buffer. GIF state after the run (loops, regsTemp, Q from the last ST) is computed on the
producer side. `ProcessWriteBuffer` (SIGNAL/FINISH/LABEL scan) skips runs. A+D writes to register 0xFF are
stored as 0x7F (WriteRegisterImpl masks with 0x7F anyway). Debugger builds keep the old path (frame dumps
read the buffer).

Why it is exact: SIGNAL/FINISH/LABEL and all CSR/IMR effects only come from A+D qwords, which stay on the
producer side in order; non-A+D PACKED writes only ever had effects on the GS thread (WriteRegisterImpl). The
GS thread sees the same writes in the same order with the same Q.

## Checks
- emu-gifgsv: every run also decoded by the old loop on copies of the state and compared (writes, loops,
  regsTemp, Q, address). gow-fight ~1.0M runs, mtvu-ffx ~0.33M, car ~0.13M (300 vblanks each): 0 mismatches.
  Patch kept at scratchpad gifgs/with-verify.diff (not committed).
- Hash gate: see table below.

## How much is movable (emu-gifgscnt, PACKED qwords over vblanks 0-900 after the state load)
| state | in raw runs (GS thread) | A+D qwords (stay) | non-A+D in A+D packets (stay) | short/no room (stay) |
|---|---|---|---|---|
| gow-fight | 225.8M (99.4%) | 0.91M | 0 | 0.46M |
| mtvu-ffx | 189.9M (98.5%) | 1.21M | 1.70M | 0.06M |
| car | 81.3M (88.3%) | 10.7M | 0 | 0 |

So nearly all of the decode moves. What stays on VU1: one 16-byte copy per qword (same buffer bytes as the
old decoded writes), the header, the mailbox submit every 0x1000 entries, and A+D packets.

## Hash gate (`hash=300&nolimit=1`, emu-gifgs1): all match
| state | vu1thread=0 | vu1thread=1 |
|---|---|---|
| car | 55dab76e | 55dab76e |
| play | 1f851c1c | 1f851c1c |
| gow-fight | 77f177fc | |
| gow-intro | b2119704 | |
| mtvu-ffx | 5927b89c | |

## Speed: not settled
All timing runs (2026-10-30) ran with the machine overloaded by other work (two Gradle JVMs at 2-3 cores
each, load average 8-16, swap full, MemAvailable often under 2.9 GB). Same-build runs varied 3x (gow-fight base
8-25 frames/s, where vu1d measured 33.5-35), so the medians below say nothing. Limiter on, default MTVU:

| state (vblanks) | runs | frames/s base -> gifgs1 | VU1 ms/frame | GS ms/frame |
|---|---|---|---|---|
| gow-fight (100-1300) | 4+4 | 22.7 -> 21.7 | 42.5 -> 48.8 | 29.9 -> 38.7 |
| mtvu-ffx (100-900) | 4+4 | 9.6 -> 12.0 | 105 -> 83 | 69 -> 62 |
| car (100-900) | 3+3 | 10.3 -> 11.5 | 93 -> 83 | 72 -> 67 |

Raw lines: scratchpad gifgs/abab1.txt, abab2.txt, abab3.txt. To redo on a quiet machine:
`gifgs/abab.sh <out> vu1dpub gifgs1 3` (gow-fight 100-1300, mtvu-ffx and car 100-900).
Expected effect: VU1 per frame should drop by roughly the ProcessPacked share (vu1c: ~2.7-3 ms of ~28 ms), and
GS per frame should go up by about the same amount (the decode now runs there).
