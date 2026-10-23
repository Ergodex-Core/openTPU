# Architecture review: docs/vla_frequency_plan.md (VLA at >= 1 Hz) and the LLM wide port

Reviewed 2026-10-16 against the `vla` worktree (d4ffbcce), `ternary-bonsai` (c52ba111), the RTL
(otpu_mxu, otpu_actram, the f2-hbm fabric) and a fresh run of `tools/pi05_cost.py --alt`.

## Verdict

The direction is right: scale one slice up (MXU columns, ACT RAM rows, a fused activation
unit, 64 VPU lanes), keep 8 HBM ports, and leave the LLM decode path to the wide port B. The
ISA tally confirms the shape of the argument: today the chunk is MXU-bound (14.5 s of a 25 s
serial sum), and at the plan's f2-vla point the MXU drops to ~0.95 s.

Four load-bearing assumptions do not survive contact with the repo, and they decide whether
the 1 Hz number is real:

1. the two-products-per-DSP packing that MCOLS 64 depends on was already found inexact on
   this MXU's 16-deep cascade (docs/mxu_systolic.md);
2. the ACT RAM cost is width-bound, not capacity-bound: ~1,024 BRAM36 or ~960 URAM at
   MCOLS 64, not ~114 URAM;
3. ACT_ROWS 256 needs an ISA change (MM's row count is an 8-bit field);
4. section 6a's bandwidth table charges weights at ~100 GB/s, but a PORTB 1 slice takes one
   chunk a cycle: 25.6-32 GB/s, whatever the HBM offers.

Plus one the plan does not see: QST's byte-a-cycle DRAM write becomes the vision tower's
bottleneck once the MXU is 16x wider.

## What the tally says (tools/pi05_cost.py --alt, 200 MHz, 558 prefix rows, the ISA's rates)

| configuration | MXU | port B | VPU | QST | QACT | bound | serial sum |
|---|---|---|---|---|---|---|---|
| today (MCOLS 4, ACT_ROWS 16, LANES 8) | 14.50 s | 7.15 s | 2.01 s | 0.37 s | 0.77 s | 14.5 s | 25.1 s |
| LANES 64 alone | 14.50 | 7.15 | 0.26 | 0.34 | 0.10 | 14.5 | 22.4 |
| MCOLS 16, ACT_ROWS 128, LANES 32, TMEM 256K | 3.73 | 1.91 | 0.51 | 0.34 | 0.19 | 3.7 | 6.7 |
| MCOLS 64, ACT_ROWS 128, LANES 64, TMEM 1M (f2-vla without the fused unit) | 0.95 | 0.65 | 0.26 | 0.34 | 0.10 | 0.95 | 2.36 |

MACs a chunk: 1.474 T (the plan says ~2.1 T, at 800-970 prefix rows; the tool's default is
558). Per phase at the f2-vla point: vision is bound by QST (0.288 s) not the MXU (0.137 s);
the prefix by the MXU (0.675 s) with port B at 0.403 s; the expert step by port B (0.016 s
against 0.014 s of MXU), as the plan predicts.

So at 200 MHz a pi0.5 chunk on f2-vla lies between 0.95 s (every unit overlapped) and 2.4 s
(none): 0.4-1.05 Hz. The plan's "~1.4 s + 0.1-0.2 s" sits inside that bracket but assumes
near-perfect overlap and 250 MHz.

## Findings, by weight

### 1. The DSP packing (L1) contradicts the repo's own study, and MCOLS 64 stands on it

docs/mxu_systolic.md: "Two int8 products per DSP48E1 is not exact on a cascade (the study: the
low field overflows after 3 accumulations), so each column has 128 product DSPs ... 138 DSP per
column". The chains are CL = 16 products (otpu_mxu's `CL`), and each chain end is one MX group
sum, so the packed low field must hold a 16-product sum exactly.

On the DSP48E2 the 27-bit A port allows a 19-bit low lane (the second activation at bit 19).
Sixteen products of operands in [-127, 127] (the quantizer's range, opentpu/quant.py) sum to at
most 16 x 16129 = 258,064 < 2^18, so a 16-deep packed chain is exact on the VU47P if both
operands stay off -128, with a one-bit sign fix-up of the high lane. If weights can be -128
anywhere (check the MX int8 block format), it needs CL 8 and twice the chain ends. Either way
the pre-adder is already taken by PAIR's stream select (INMODE via `hi0`), so forming
a1 + (a2 << 19) needs fabric or a redesign of that select.

Why it matters: without packing, MCOLS 64 is 64 x 138 = 8.8K DSPs, the whole part (9,024),
leaving nothing for the VPU and QUANT. MCOLS 32 unpacked (4.4K DSP, 4,096 MAC/cycle) is the
safe point: pi0.5 MXU ~1.9 s, RhinoVLA ~0.9-1.0 s at 200 MHz, which is at or below 1 Hz.
**The packing is the single gate on the 1 Hz claim.** It is in phase 3 of the plan; it should
be phase 1: a packed 16-chain in Verilator, bit-exact against the ISA over the int8 corners,
plus an out-of-context synthesis of one packed column (DSP count, fmax). That is a day's work
and no nightly.

### 2. ACT RAM (L2): width-bound, not capacity-bound

otpu_actram reads one D = 128-byte block per column every cycle (`r_data` is MCOLS x D x 8
bits). Today that is 4 pieces x 4 RAMB36 = 16 BRAM36 per column, placed one block RAM column
beside each DSP column (docs/mxu_systolic.md, "Placement"). At MCOLS 64 that is 1,024 RAMB36
(of 2,016) or, in URAM at 72 bits a port, 15 x 64 = 960 URAM: every URAM on the part.
Rows do not change this; the plan's "~114 URAM" is the capacity count only.

Options, in order of preference: (a) BRAM, 1,024 RAMB36, half the part's BRAM, leaving ~1,000
for the weight FIFOs, TMEM pieces and the trace; (b) pump the ACT RAM at 2x as TMEM_PMP does
(512 RAMB36), another clock-domain item against 250 MHz; (c) MCOLS 32. The budget line in
section 6 lists DSP, URAM and LUT and no BRAM; add it.

The placement scheme (one DSP column and one BRAM column per array column) also does not
extend to 64 array columns across three SLRs; that is the concrete form of the plan's
"wiring across SLRs" risk, and a floorplan sketch (columns per SLR, where the result FIFO and
the drain sit) belongs in phase 3's exit criteria.

### 3. ACT_ROWS 256 is an ISA change

`isa.mm` checks `0 < m < 256`; otpu_mxu asserts `ROWS < 256`; otpu_actram's row ports are 8
bits. tools/pi05_cost.py stops its alternatives at ACT_ROWS 128 for exactly this reason. QST
already carries 16-bit rows, so widening MM's m is a contained change, but it has to be in
phase 1 (software) with its compiler and simulator tests, or the plan should be re-costed at
ACT_ROWS 128: on the 558-row prefix that is 5 weight passes instead of 3, and port B at 0.65 s
against 0.95 s of MXU (the tally) instead of ~0.4 s.

### 4. Section 6a charges weights at a rate the slice cannot take

A slice at PORTB 1 takes one 128-byte chunk a cycle on port B: 25.6 GB/s at 200 MHz, 32 GB/s
at 250 (docs/f2_hbm.md: at NCH 8 each channel carries 3.2 GB/s of the slice's 25.6;
docs/wide_port.md: "More HBM ports only help once a slice takes more than one chunk a cycle").
So "at ~100 GB/s" in the 6a table and "~100 GB/s usable" in L7 do not apply at PORTB 1; 8 ports
buy the 4 GiB and headroom for PORTB 2 / 4, not bandwidth. Recomputed at 32 GB/s the ~16 GB of
a chunk is ~0.5 s of port time, ~35% of the MXU time rather than ~11%, and the board delivers
78-89% of the port (burst200 measurements in wide_port.md), so nearer 0.6 s.

The conclusion "8 ports are enough" still holds, but the margin comes from ACT_ROWS 256
(finding 3) and from PORTB 2, not from the HBM. Two consequences:

- the expert step is port-bound already in the tally (0.016 s port B vs 0.014 s MXU at
  M = 50 under 64 columns), so PORTB 2 is on pi0.5's critical path, not "if the stalls
  matter"; the wide-port lanes being built on ternary-bonsai become a named dependency of
  f2-vla with a merge point, not a cross-reference;
- "a DMA lane for activation tiles" goes through the same otpu_native_dram grant chain unless
  it gets its own channel path; the plan should say which, since the LD before an MM is not
  overlapped with that MM.

### 5. QST is the vision tower's bottleneck at scale (not in the plan)

The profile's QST cost is ceil(n / LANES) + n: the DRAM write of n int8 bytes at a byte a
cycle on port A (opentpu/profile.py). It does not shrink with LANES (vision QST 0.32 s today,
0.29 s at LANES 64) and at the f2-vla point it bounds the vision phase (0.288 s against 0.137 s
of MXU) and is 0.34 s of the chunk's 2.36 s serial sum, more than the whole VPU after L4.
L6 ("QACT / QST at LANES elements a cycle") covers QST's lane side, not its DRAM side.

Fix: QST stores as 128-byte chunks through the DMA / port B write path, or keep SigLIP's
per-view outputs in TMEM and quantize straight into ACT RAM (L10 territory: the 3 views as
rows of one pass). First confirm on the RTL that the byte-a-cycle model is what the hardware
does; if the profile is pessimistic here, the tally's vision line is wrong in the plan's favour.

### 6. The MAC and token counts disagree with the tally

Plan: ~2.1 T MAC at 800-970 prefix rows. Tally: 1.474 T at 558 rows (the LIBERO prompt
shape the tool defaults to). State which prefix length f2-vla is sized for: at 558 the plan is
~30% pessimistic, at 970 ~15% optimistic. RhinoVLA has no tally at all yet, and its ">= 1 Hz on
this variant" is the plan's headline deliverable (phase 7's exit criterion). A shape-only tally
for RhinoVLA (Qwen3-VL ViT, 2.1B LM, 0.4B expert) needs no port and should be in phase 0.

### 7. Overlap and clock carry the Hz claim; commit at 200 MHz

Today's full f2 build is 145K LUT at 200 MHz; the 250 MHz reference on this part (SGMV) is
20K LUT. A 0.6-0.8M LUT design across three SLRs closing 250 MHz on its first nightly is
unlikely; the plan's own fallback says -20%. Commit the numbers at 200 MHz (RhinoVLA ~1.0 Hz
with no margin, pi0.5 ~0.5 Hz) and treat 250 as upside. Phase 3's exit criterion should be
MXU utilization measured on the real prefix shape (558-970 rows, N = 16384 MLPs, the 2048-wide
projections), not "a cut prefix", because the drain and the result FIFO (RF 32 / 64 rows) are
where a 64-column array loses efficiency, and a cut model will not show it.

### 8. Smaller items

- docs/pi05.md is cited five times (sections 0, L9, L10, phase 0; tools/pi05_cost.py cites its
  section 5) and does not exist on `vla` or `pi05`. Write it or point at opentpu/llm/pi05.py.
- Phase order: at LANES 64 the whole VPU is 0.26 s and GELU 0.14 s (tally), so the fused
  activation unit (L3, phase 2) saves ~6% of the serial sum, not "~1.3 s". It is still cheap
  and worth building, but put L4 / L5 / L6 (phase 4) before it and size L3 on the measured
  remainder. Phase 2's exit criterion ("VPU share < 10%") is met by LANES 64 alone.
- L9 (W8A16 at half rate) is only half rate if the packing works; unpacked, int8 x int16 is
  the same one MAC a DSP as int8 x int8. Decide after finding 1.
- Section 1's LLM column is consistent with ternary-bonsai: the board runs Gemma-3-1B at 22.3
  tok/s, 89% of the port-bound 25, so PORTB 2 / 4 is the right lever and the projections (38 /
  70 tok/s) are fair. Both branches need otpu_native_dram to pass more than one chunk a cycle;
  build the lane structure once, on ternary-bonsai, and have f2-vla take it (L7 says so;
  make it a dependency).
- The multi-slice fallback (MCOLS 32 x 2 slices) brings back the collectives the plan was
  avoiding; the better fallback is MCOLS 32 x 1 slice unpacked (4.4K DSP, finding 1), which
  keeps every argument of section 4 and lands RhinoVLA near 1 Hz.

## Recommended phase order

| phase | work | decides |
|---|---|---|
| 0 | pi0.5 tally (done above); RhinoVLA shape tally; confirm QST's DRAM rate on the RTL | the budget per phase |
| 1 | ISA: MM rows to 16 bits (ACT_ROWS 256); QST chunked stores; **packing feasibility**: a packed CL 16 chain in Verilator bit-exact over the int8 corners, one packed column synthesized OOC | MCOLS 64 vs 32; DSP and fmax per column |
| 2 | ACT RAM at the chosen MCOLS: BRAM / pumped / URAM from the width math; a floorplan sketch per SLR | BRAM budget; the SLR plan |
| 3 | MCOLS 8 -> 16 -> 32 (-> 64), ACT_ROWS 256; RTL timing on the full prefix shape | MXU utilization >= 75% at the real shape |
| 4 | LANES 64, TMEM 1M, QACT / QST wide | VPU + QACT + QST < 15% of the serial sum |
| 5 | fused activation unit, sized on phase 4's remainder | bit-exact; the GELU's share gone |
| 6 | PORTB 2 lanes (from ternary-bonsai) and the DMA path | expert step no longer port-bound |
| 7 | RhinoVLA port | action error vs reference |
| 8 | nightly at 200 MHz, then 250 | >= 1 Hz RhinoVLA measured |
| 9 | W8A16 where needed; the MCOLS 96 stretch | LIBERO parity; pi0.5 at 1 Hz |
