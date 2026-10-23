# C2: dual-mode MXU using the existing columns for decode

**Approved for implementation; reference and standalone RTL milestones complete.**
New Codex proposal, 2026-10-18. The user approved parallel work on C1-C3 and requires
openTPU Lens for performance measurement and visualization. See [validation evidence](VALIDATION.md).
Branch: `codex/v4-c2-dual-mode-mxu`. Base: `016e953a`.
See [comparison and approval boundary](README.md).

## Architectural decision

Change the meaning of the same four MXU columns according to the operation:

| Mode | Four physical columns receive | Useful product PEs at M=1 |
|---|---|---:|
| Existing MM / prefill | One weight row, four stationary activation rows | 128 of 512 |
| New decode mode | Four different weight rows, one broadcast activation row | 512 of 512 |

For eligible M=1 operations, column c produces command-local output r+c from full-matrix
weight row n0+r+c. For ordinary MM, the columns keep their present role. The product DSP
chains and 48-bit column-end accumulators are reused.

Claude P2 adds four complete MCOLS=4 arrays: 2,048 product PEs, of which 512 are useful at
M=1. C2 seeks the same useful decode width from 512 product PEs by changing operand routing.
It does not inherit P2's fourfold prefill compute capacity. Compared with C1, C2 retains a
single grouped array and row-group schedule rather than independently advancing PC engines.

## Initial sizing

| Component | Initial comparison point |
|---|---|
| Core | One slice; D=128; MCOLS=4; 200 MHz target |
| Decode ingress | Four 128-byte weight chunks per cycle |
| Weight-path arithmetic ceiling | 102.4 GB/s; 512 useful int8 products/cycle |
| HBM ownership | Four column groups, four PCs per group |
| Weight buffers | 512 chunks per column; 256 KiB total, plus scales/tags |
| Epilogue | Four lanes shared by both modes |
| VPU / quantizer / TMEM | Existing configuration first; changes require separate measurements |

A column requires 25.6 GB/s at full decode duty. Its four PCs have a combined arithmetic peak
of 57.6 GB/s at 32 bytes and 450 MHz per PC; this is scheduling margin, not proof of sustained
throughput. A local four-PC reader feeds only its assigned column. There is no global
32-to-four destination crossbar.

The four input queues initially advance as a group, masking invalid tail columns.
This preserves a manageable array schedule but retains a slowest-column stall risk.
All row chunks/scales and result credits must be reserved before a row group begins if
the product pipeline remains non-stallable.

## Packed image and explicit dispatch

Keep the canonical image for ordinary MM. Create a packed sidecar for eligible static int8
matrices, preserving original quantized bytes and every scale value. For full matrix row n,
full-matrix block k, full_KB blocks and group_pitch=ceil(full_KB/4):

```text
pc     = 4 * (n mod 4) + (k mod 4)
offset = base[pc] + 128 * ((n // 4) * group_pitch + (k // 4))
```

The descriptor stores the full shape, per-PC bases, allocation bounds and original scale
tables. The reference distinguishes packed `k_begin` from canonical MM's scale-table `k0`:
`k0 = matrix.scale_k0 + k_begin`. This matters for column parts whose scale rows retain an
original prefix. Commands preserve n0, N, KB and ab. A submatrix uses full-matrix indices for
addressing and the original block offsets; it must not compute pitch from the subrange KB.
Only aligned n0 groups use the initial fast path; other ranges fall back to canonical MM.

Each 128-byte chunk can be four 32-byte AXI beats. Prefetch across local PCs uses tagged
requests and a bounded per-column reorder queue. Legal bursts respect controller and 4 KiB
boundaries. Do not assume four logical chunks on different PCs form one contiguous burst.

Packed weight capacity is 16 * ceil(N_full/4) * ceil(full_KB/4) * 128 bytes. When N is divisible
by four, KB=9 incurs 33.3% sidecar padding; KB=54 incurs 3.7%. Unused tail slots must never
be issued as logical reads, stored as outputs or supplied to reductions. Retaining canonical
weights adds another copy. The allocator checks PC-local capacity and nonoverlap.

Full Sw and dw metadata is retained in tagged per-column scale caches, with independent
prefetch. A single uncached scale reader is not assumed to supply four rows per cycle.

Use a new versioned packed-MM capability/descriptor and symbolic operation name
`MM_ROWS`; no opcode number is assigned here. The milestone-1 implementation is an offline
reference selector/executor in `opentpu/c2.py`, not an encoded instruction or production
compiler change. All existing MM flag bits are already used.
The initial compiler selects it only for:

- M=1, int8, a supported aligned packed descriptor and valid activation ranges;
- plain output or ACC; no PAIR, RMAX or ASCALE.

M=2/3, prefill, four-bit formats, dynamic KV attention, gathers and unsupported flag/layout
cases use the original MM. UNIT/RSC can initially fall back as well. Targets without the
capability receive canonical programs; unknown commands must fail clearly, never silently
execute as ordinary MM. Update image and target compatibility hashes when implemented.

## Exact arithmetic and changed RTL

Each output retains the existing computation:

```text
acc[c] = exact shifted integer-group sum for weight row n0+r+c
y[c]   = (i2f48(acc[c]) * Sw[n0+r+c]) * (Sa[0] * 2^-16)
T[out+r+c] = y[c]                   # or the existing ACC operation
```

One ACT RAM block read supplies the same activation bytes, offsets and row scale to four
local register sets. No new activation read ports are required for this mode. Preserve
nonzero ab, original k0 indexing, rounding, FTZ, shift-window behavior and 48-bit bounds.

This is a datapath redesign, not just a mode bit:

- Select between the present weight-hop path and independent per-column ingress. Preserve
  product, scale, control and valid alignment through the existing skew and register stages.
- Make weight offsets and Sw independent per column. The current array shares weight-side
  metadata; existing activation-side metadata cannot be relabeled without redesign.
- Carry four Sw values, mode, command ID, row indices and tail-valid bits through the result
  FIFO. It currently stores one shared weight-row scale.
- Route decode results to out+r+c, where r is command-local, not full-matrix row n.
  Existing drain logic maps column j to out+r+j*ors.
- Retire four results/cycle where grants allow. KB=1 and ACC read/modify/write are stress
  cases; fewer epilogue lanes would throttle this design.
- Tag mode on outstanding work. Initially drain completely at a mode transition to avoid
  mixing interpretations of in-flight metadata; charge that lost overlap to performance.

RMAX remains a fallback because four physical columns now contribute to one logical output
row's maximum. Later support must explicitly redesign that state. The same caution applies
to ASCALE and PAIR. Numerical anchors are [ISA MM](../isa.md) and
[otpu_mxu.sv](../../rtl/mxu/otpu_mxu.sv), particularly scale decode, weight skew, result FIFO
and drain address generation. Older systolic documentation includes superseded epilogue
details; the current RTL/ISA are the source of truth.

## Performance hypothesis and rejection criteria

New baseline evidence from `tpubench` commit `112ef960`, `docs/v4/m0-gating-split.md`:
97.2% of the original blocked cycles were incomplete rows, not exhausted result credits.
The stall-0/LAT-30 RTL simulation took 8,221,777 cycles, versus 10,279,178 at DDR-like
stall-20/LAT-20. Neither is an F2 card measurement. C2 must measure its own memory/scale
delivery and cannot claim to recover the original 18.2% by changing row credits.

The ideal advance count for an eligible aligned operation is ceil(N/4)*KB, rather than N*KB.
This does not include row-group bubbles, scale waits, memory latency, epilogue stalls, tails
or mode-switch drains. The nominal weight-only floor for the baseline's approximately
1.005 GB is 9.8 ms if all traffic were eligible and perfectly supplied; that is not a
whole-token forecast.

Measure eligible versus fallback cycles on the actual Gemma program, including its chunked
down projections. Add unchanged vector chains and the new transitions to the critical path.
Retain the same clock, quantization, host/card loop and workload for comparisons.

The product array stays at 512 PEs, but independent ingress, replicated scale logic, FIFOs,
epilogue metadata and HBM adapters add substantial area. No total DSP/LUT estimate or
200 MHz closure claim is made before synthesis. Prefill compute throughput stays at the
existing four-column rate; the mode multiplexers could even reduce achieved frequency.

Reject or revise C2 if mode-switch drains erase decode gains, scale traffic cannot keep up,
or the new per-column ingress cannot meet the chosen clock within the available CL region.
Report prefill, decode, achieved clock and resource use separately when comparing with P2.

## Milestones after approval

1. Add an executable packed-descriptor reference and prove exact agreement with canonical
   MM for distinct per-row Sw/dw, N tails, KB=1/9/54/255, k0/ab offsets, zero/extreme values,
   shift-window boundaries and ACC. Demonstrate explicit fallback for all excluded modes.
2. Implement selectable operand paths, metadata and drain addressing under RTL simulation.
   Stress mode changes with pending reads/results and run legacy format regressions.
3. Measure row credits, per-column FIFO/scale starvation, TMEM grants, mode drains and
   fallback share on actual Gemma traces. Replace C-only Lens re-timing with measured events.
4. Synthesize the ingress/column path, then the complete shell-constrained design; verify
   real HBM layout and traffic before making an F2 performance claim.

Milestone 1 is implemented and validated: byte-preserving sidecars, explicit reference
dispatch/fallback, independent exact arithmetic, and Lens profiles of actual reference
events. Those original profiles retain their serialized arithmetic-slot axis.

The next milestone adds `rtl/mxu/otpu_c2_mxu.sv`: a standalone D128/C4 dual-mode int8
datapath using one product bank, independently stalled operand/scale streams, tagged
result credits, four fp32 epilogue lanes, ACC service and mode-drain control. Its Verilator
cycle traces are exported separately to Lens; see [validation evidence](VALIDATION.md).
Two operand banks now overlap the next same-mode group's fill with current computation.
Each column has two 256-chunk weight banks (512 chunks, matching the proposed 256 KiB
total weight capacity); this standalone harness also buffers ACT packets and metadata.
It does not yet replace production `otpu_mxu`, encode `MM_ROWS`, or implement packed-PC
address generation/HBM. Synthesis, F2 execution and whole-token integration remain open.
