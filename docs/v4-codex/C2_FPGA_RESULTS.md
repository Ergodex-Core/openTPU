# C2 FPGA build evidence — 2026-10-19

**No C2 FPGA execution or physical HBM performance result is available yet.**
The integrated F2 image is being built from frozen commit
`4f7509123fdd2cf1eba5a2c88289a3e3c2709439`. RTL Lens results remain simulations.

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
| LUT / FF | 11,818 / 14,521 | 10,042 / 13,287 |
| Full timing target met | No | **No** |

The routed OOC stage deliberately returned failure. Its worst hold path is
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

The isolated build runs on `i-00a83cab3c81ac10f` under systemd unit
`otpu-c2-full-4f750912`, with an eight-CPU quota, 96 GiB memory limit, six-hour
runtime limit and four Vivado threads. Its directory is
`/home/ubuntu/rachit/codex-c2/source-4f750912/full`. The source checkout is separate
from existing TPU work; no FPGA slot has been changed.

The kernel, eight native HBM ports, PCIS preload and OCL control synthesized with
zero synthesis errors/critical warnings. The custom CDC constraints found 19
crossing FIFOs and one host write counter, with core/main/HBM periods
5.000/4.000/2.222 ns. Integrated synthesis estimated +0.387 ns setup and −0.095 ns
hold slack; these are pre-route estimates. Shell linking additionally emits
HDK constraint/context warnings that remain recorded and under review.

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
