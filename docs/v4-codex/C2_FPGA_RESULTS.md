# C2 FPGA build evidence — 2026-10-19

**No C2 FPGA execution or physical HBM performance result is available yet.**
The first integrated build, from frozen commit
`4f7509123fdd2cf1eba5a2c88289a3e3c2709439`, was stopped after a clock DRC failure.
Clock and CDC qualification fixes are being prepared for a fresh isolated build.
RTL Lens results remain simulations.

## Standalone core synthesis and route

Vivado 2025.2 synthesized and routed `otpu_c2_mxu` for
`xcvu47p-fsvh2892-2-e`, D128/C4/DEPTH256/RF8, using the Xilinx DSP implementation.
The snapshot was prepared from `29038ae2`; all six RTL source hashes also match
the integrated build's `4f750912` source. Concurrent shell/host work was recorded
in the snapshot's git status but was not among these six inputs.

The clock target was 5 ns, with input/output max delays of 0.5 ns and min delays
of zero, setup uncertainty 0.1 ns and hold uncertainty 0.05 ns. No false paths or
multicycle exceptions were added.

| Metric | Synthesis | Routed OOC |
|---|---:|---:|
| Setup slack, ns | +1.652 | +0.161 |
| Hold slack, ns | −0.125 | **−0.130** |
| Register-to-register hold slack, ns | −0.009 | +0.014 |
| DSP48E2 | 564 | 564 |
| RAMB36 / RAMB18 | 120 / 8 | 120 / 8 |
| Logic LUT cells / FF | 11,818 / 14,521 | 10,042 / 13,287 |
| Full timing target met | No | **No** |

The LUT-cell metric excludes SRL and LUTRAM primitives; it is not total slice-LUT
occupancy. The routed OOC stage deliberately returned failure. Its worst hold path is
`cmd_addr[0][18]` to `command_bank_reg[0][addr][0][18]/D`: 0.012 ns data delay
minus 0.030 ns destination clock delay, 0.050 ns uncertainty and 0.062 ns FF hold
requirement gives −0.130 ns. The standalone port has a zero minimum input delay
and an unset physical boundary net. These address upper bits are constant in the
bounded-row kernel; the full image must still close every remaining actual
driver-to-register path. The OOC result is not being relabeled as a pass.

All twelve `check_timing` categories were zero, including unconstrained internal
endpoints and missing input/output delays. All 65,568 routable nets were fully
routed with zero routing errors. DRC and methodology reports retain DSP/RAM
pipelining warnings, the OOC no-routable-load warning and primitive
advisories. Positive internal slack does not qualify the integrated shell clock
network or establish measured F2 operation.

Local evidence is `build/c2-fpga/ooc-final-evidence/`. The compressed source,
flow, manifests, logs and reports have SHA256
`5137d2d8bcf8c13164880973447ecae347bf720e8fdeab2d585dd6d5257aa637`.
Evidence and both checkpoints are preserved using AWS CLI at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/ooc-xdc-fixed/`.

## Integrated F2 build

The first isolated build ran on `i-00a83cab3c81ac10f` under systemd unit
`otpu-c2-full-4f750912`, with an eight-CPU quota, 96 GiB memory limit, six-hour
runtime limit and four Vivado threads. Its preserved directory is
`/home/ubuntu/rachit/codex-c2/source-4f750912/full`. The source checkout is separate
from existing TPU work; no FPGA slot has been changed.

The kernel, eight native HBM ports, PCIS preload and OCL control synthesized with
zero synthesis errors/critical warnings. The custom CDC constraints found 19
crossing FIFOs and one host write counter, with core/main/HBM periods
5.000/4.000/2.222 ns. Integrated synthesis estimated +0.387 ns setup and −0.095 ns
hold slack; these are pre-route estimates based on the subsequently rejected
clock setting. Shell linking additionally emits HDK constraint/context warnings
that remain recorded and under review.
The integrated synthesis utilization report lists 38,897 total LUTs (23,141 logic,
6,642 LUTRAM, 9,114 SRL), 41,865 FFs, 172 RAMB36, 55 RAMB18 and 564 DSPs. This
includes the CL's HBM/control fabric, before linking the static AWS shell.

Vivado reported `AVAL-168`: the inherited MMCM multiplier `4.8` is not a legal
0.125 increment. The build was stopped before placement; its source, reports,
logs and four checkpoints were preserved in S3 under `full-4f750912/`.
Commit `70bb2971` derives a legal multiplier/divider pair of 24/5, preserving
the exact 200/400 MHz core clocks and 450 MHz HBM clock. The builder now checks
legal clock attributes and exact frequencies before invoking Vivado.

A separate read-only probe of the actual optimized checkpoint confirmed that
all eight native write-counter Gray crossings survive optimization despite
disabled kernel writes. Each retains 16 source and two stages of 16 destination
registers, with a path into read issue gating. The corrected build must therefore
constrain and validate **47** Gray groups: 38 FIFO pointer directions, eight
native counters and one host counter. The old 39-group report is insufficient.
The probe also established that Vivado's exception property includes
`-datapath_only`, and that bus-skew report rows wrap across lines; qualification
must parse these actual report forms. Probe evidence is preserved in
`build/c2-fpga/counter-evidence/` and the S3 `full-4f750912/` prefix.

A second read-only probe found 35 inherited shell methodology findings and four
`TIMING-54` scoped-clock findings. The unchanged shell classifier passed all 35;
the inherited DDR/100 MHz context classifier rejected this HBM/200 MHz design.
All core and HBM clock registers are inside `WRAPPER/CL`. Across the six clock
directions, exhaustive boundary queries found only the existing shell-reset
entries to the three core and three HBM synchronizer CLR pins, with no top-level
I/O paths. Vivado returns those exact false paths in cell-source queries both
with and without `-user_ignored`; this is recorded explicitly rather than
interpreted as a timed data crossing. The C2 checker must audit all six exported
destination-period bounds and map the four findings to their actual constraint
IDs. No severity or timing exception is relaxed.

This probe's complete reports, exported constraints, object properties and logs
are in `build/c2-fpga/methodology-evidence/` and S3
`full-4f750912/methodology-evidence.tar.gz`, with SHA256
`c7da174694ff9e07d4f6f1f929fad8481d0bf4ec3f7f90083f9bc64c96fb2125`.

Shell placement/routing and the separate setup, hold, bus-skew, CDC-witness,
routing, DRC and methodology gates remain required. No AFI is published or loaded
unless those gates pass. The planned execution host is the dedicated
`i-0c78de2253f66519d`, with a fresh ownership/idle check and canonical device lock
before loading; the previous image is restored afterward.

## Prepared physical comparison

Exact-reference fixtures cover rows/KB of 31/1, 257/9, 31/54, 31/255 and 3/9.
Both legacy and decode will use identical physical weights and resident ACT/scale
metadata, with three alternating repetitions. The host checks every fp32 result,
signed integer diagnostic and per-PC read count before exporting hardware Lens
profiles. START-to-final-commit cycles exclude preload and readback.

The combined Lens viewer contains 128 profiles; its latest ten are explicitly
synthetic native-service RTL profiles. Their memory-stall findings and physical
measurement priorities are in [C2_NATIVE_PROFILE_ANALYSIS.md](C2_NATIVE_PROFILE_ANALYSIS.md).
The build/benchmark contract is [F2-BENCH.md](F2-BENCH.md).
