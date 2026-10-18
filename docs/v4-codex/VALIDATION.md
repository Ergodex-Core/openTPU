# C2 milestone 1: packed reference and Lens evidence

Completed after the user's approval to implement all three proposals in separate worktrees.
All files in this milestone belong to `codex/v4-c2-dual-mode-mxu`.

## What executes

`opentpu/c2.py` packs canonical static int8 allocations into sixteen PC-local byte arrays.
Its versioned descriptor records the original allocation, PC bounds, layout identifier and
a SHA-256 digest over descriptor, weights and complete scale rows. Packing preserves bytes;
there is no new quantization. Arrays are read-only snapshots. The caller promises that a
static canonical allocation remains unchanged; dynamic KV and gather uses explicitly fall
back. This reference does not intercept arbitrary DRAM writes or implement a device ABI.

The selector checks format, flags, shape, target capability, row alignment and resolved
register-based addresses against the descriptor. Unknown descriptor versions/layouts and
non-MM input operations are errors. Unsupported eligible-source cases execute unchanged
`Slice._mm`. Encoded MM instructions and the production compiler are untouched.

Packed execution reads actual PC arrays, broadcasts ACT row zero, computes exact integer
subgroup sums and shifts, and invokes the existing fp32 helpers for the existing epilogue.
It does not unpack into the canonical ISA simulator to obtain its results. Tests compare
entire TMEM bit arrays with canonical MM and also verify DRAM/ACT are unchanged.

The offset contract is explicit: packed block `k_begin+q`, ACT block `ab+q`, canonical
scale offset `matrix.scale_k0+k_begin+q`, full row `n0+r+c`, output `out+r+c`.
This prevents treating a column part's scale prefix or a nonzero row start as an output offset.

## Tests

Run from this worktree with a Python environment containing the project's dependencies:

```sh
python -m pytest -q tests/test_c2.py tests/test_mx.py tests/test_isa.py
```

Result: **86 passed** (40 new C2 cases and 46 existing MX/ISA cases). Eight selected
existing Lens tests also pass: profile/version handling and ISA/summary compatibility.
Lens profile save/load, HTML generation and command-line summary were verified.

Coverage includes KB=1/9/54/255, plain/ACC, distinct Sw/dw, signed extreme and zero weights,
shift-window boundaries, fp32 FTZ/signed zero, nonzero n0/k_begin/scale origin/ab,
register-based addresses, N tails, poisoned physical padding, bounds/overlap, immutable
snapshots, wrong versions and mismatched scale origins. Repository quantization and QACT
also feed the new path. M=2/3 and prefill, every four-bit format, PAIR, RMAX, ASCALE,
UNIT/RSC, missing capability/descriptor, dynamic KV and gathers execute canonical fallback.

## Lens artifacts and reproduction

```sh
PYTHONPATH=. python tools/c2_reference.py --out-dir build/lens/c2
python -m opentpu.lens summary build/lens/c2/c2-reference.otpuprof
python -m opentpu.lens open build/lens/c2/c2-reference.otpuprof
```

The command writes a Lens profile bundle, standalone `c2-reference.html`, and
`c2-reference.json` with seeds, full descriptors, output bits/hashes and event counts.
All eight default cases use repository int8 quantization and QACT, then compare independently
executed packed arithmetic against canonical MM before exporting anything. N=7 gives a
three-column final group; each plain/ACC pair produces these counts:

| KB | Executed advances | Commits | Weight bytes | Useful column fraction |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 896 | 87.5% |
| 9 | 18 | 2 | 8,064 | 87.5% |
| 54 | 108 | 2 | 48,384 | 87.5% |
| 255 | 510 | 2 | 228,480 | 87.5% |

These are synthetic microbenchmarks, not a Gemma performance run. Lens Timeline and
Instructions show each executed block advance, valid mask, actual PC/address reads and
commit. The profile also retains integer accumulators and output bits. Scale payload is
counted separately: `N*(KB+4)` bytes per operation; cache-line transactions are unmodeled.

**The horizontal axis counts serialized reference steps, not measured hardware cycles.**
Each advance and commit occupies one illustrative slot; pipelined overlap, HBM latency,
scale caches, QACT/vector timing, grants and command transitions are not modeled.
The displayed 200 MHz conversion and weight-only roofline are hypothetical. Profiles are
labeled `REFERENCE SLOTS, NOT RTL`, retain 512 peak product PEs, and do not misrepresent
C2 as four replicated arrays. The existing Lens floorplan remains a legacy schematic;
use Timeline/Instructions to inspect this milestone's actual event evidence.

## Remaining work

RTL selectable ingress, independent weight-scale metadata, mode tags and drain addressing;
production compiler/image integration and binary opcode assignment; per-cycle simulation
and HBM/scale-service modeling; then shell-constrained synthesis and physical measurement.
No resource-fit, timing-closure or end-to-end tokens/s claim follows from this milestone.

Baseline interpretation incorporates the newer `tpubench` evidence at commit `112ef960`,
`docs/v4/m0-gating-split.md`: the original blocked loss is mostly memory delivery, not
result credits. Its 8,221,777-cycle stall-0/LAT-30 trace is RTL simulation, not F2 hardware.
