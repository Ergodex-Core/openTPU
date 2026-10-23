# C1 first milestone: row image and arithmetic reference

This records the completed first milestone. The later [bounded RTL milestone](C1-RTL.md)
adds actual kernel simulation; the analytic results below retain their original scope.

User approval: on 2026-10-18 the user explicitly approved parallel C1-C3 implementation and
required openTPU Lens for performance measurement/visualization. This work is confined to
`codex/v4-c1-hbm-row-engines`; it does not modify Claude's branch or either sibling candidate.

## Implemented contract

`opentpu/c1_rows.py` packs canonical `quantize_mxu(..., "int8", 128)` weight and scale rows
into immutable per-PC records without re-quantization. Version 1 descriptors include matrix
shape, ordered physical PC identities, per-PC bases and exclusive allocation limits. Physical
addresses and compact host payload offsets are separate. The ZIP sidecar carries JSON metadata,
raw PC payloads and SHA-256 digests. Canonical scale padding is preserved as well as meaningful
weight/scale bytes; new alignment padding is zero. Empty engines have empty payloads.

`Command` describes resolved n0/N, k0/KB, ACT ab, output address, flags and matrix immutability.
The reference independently executes 32-element integer groups from sidecar bytes, followed
by the established `fp32` conversion/multiply/add helpers. It validates all ranges and the
single-Sa activation contract before committing outputs. Unsupported widths, mutable weights,
M != 1, UNIT/RSC/RMAX/ASCALE/PAIR and four-bit formats require canonical-MM fallback explicitly.
This is an experimental API; there is no new opcode or automatic compiler dispatch yet.

Tests execute real `isasim.Slice.execute(I.mm(...))` calls against the original DRAM image as
the independent oracle. They cover N below/across engine count; nonzero n0/k0/ab; KB 1/6/9/54;
actual QACT snapshots; consecutive chunked ACC operations; negative-shift groups; signed zero,
subnormal, infinity and NaN epilogues; immutable storage; descriptor bounds/version errors;
sidecar corruption and preservation of all TMEM words outside the output range.

## Lens characterization

Generate the deterministic N=257, KB=9 workload and compare each engine layout's arithmetic
against canonical ISA execution:

```sh
python -m tools.c1_validate --output build/lens/c1
python -m opentpu.lens summary build/lens/c1/c1-validation.otpuprof
python -m opentpu.lens open build/lens/c1/c1-validation.otpuprof
```

The command creates a Lens v1 bundle, standalone Lens HTML, the validated 16-engine sidecar,
and `validation.json`. All four layouts reproduce all 257 output words bit-for-bit. The report
separately records measured **CPU reference wall time**, which is not accelerator timing.

`opentpu/c1_profile.py` supplies **analytic-c1** profiles from individual engine events. It
does not divide a baseline trace by engine count. The initial conservative policy allows one
outstanding row/result per engine and requires ordered retirement before that engine's next
request. It charges an activation multicast, entire row headers, selected weight bytes,
first-response latency, epilogue latency and shared retirement. One result credit per engine
bounds storage and exposes slow-PC head-of-line blocking. Default assumptions are 200 MHz,
80-cycle first response, 8 activation bytes/cycle, 9-cycle epilogue and one retirement slot;
ACC charges an additional assumed read slot. The per-row events and assumptions are embedded
in the profile's `c1_model` field.

| Analytic policy, N=257 and KB=9 | Cycles |
|---|---:|
| 1 engine | 32,792 |
| 4 engines | 8,408 |
| 16 engines | 2,312 |
| 16 engines, PC0 delayed by another 160 cycles per row | 5,032 |
| 32 engines | 1,296 |

These are small-kernel policy experiments, **not token throughput, RTL simulation or F2 device
measurements**. They omit burst/bank scheduling, HBM contention, multirow prefetch, detailed
TMEM arbitration, fallback/vector work, CL DMA and host costs. The generic Lens floorplan
depicts the retained backend and is not a C1 placement diagram. Timeline items are modeled
work records rather than executed ISA instructions. The next milestone must compare plausible
bounded prefetch policies and include full dependency-critical costs before making a speedup
claim. No crossover or throughput recommendation follows from this table.

The latest separate M0 evidence was read from `tpubench` commit `112ef960`,
`docs/v4/m0-gating-split.md`, without cherry-picking moving changes: 97.2% of the original raw
blocked counter was incomplete rows, versus 0.1% credits. The stall-0/LAT-30 baseline is
8,221,777 cycles, compared with 10,279,178 at stall-20/LAT-20. These are **RTL memory-model
simulations**, not F2 hardware. C1 therefore does not justify itself by assuming a large
result-credit fix, and does not treat the slower DDR-like trace as an HBM baseline.

## Validation and remaining work

```sh
python -m pytest -q tests/test_c1_rows.py tests/test_mx.py
# 104 passed
python -m pytest -q tests/test_lens.py -k 'isa or candidate'
# 2 passed, 14 deselected
git diff --check
```

The candidate still needs full-model compiler/image integration, automatic fallback dispatch,
an adequately detailed per-PC memory/scheduler model, RTL, shell-constrained resource/timing
reports and device measurements. CL DMA and shell-reserved placement remain explicit F2 work.
No synthesis or cloud deployment was performed for this milestone.
