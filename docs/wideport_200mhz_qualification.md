# Wide-port 200 MHz qualification checkpoint

2026-10-21 UTC. This is a plan and evidence checkpoint; no new physical build or
hardware programming was performed. The successful 150 MHz image is preserved.

The isolated candidate is `codex/wide-port-200mhz-opt`. Synthesized RTL behavior last changed
at `b22dda7e45143e6cdfbcf6374103a2b44ee9a8f6`; build support was added at
`6fa3c671ec5d373092c62a3fe6f3cdf271834f18`. Replay reporting is at `beb39f0b`,
with the missing-arbitration gate extended to replay-disabled mode at `b4639a9a`.
The owner repositories `openTPU-wp-alexshell` and `openTPU-hwloop` remain untouched.

## Model accounting

The LPN now resolves target variants and uses the candidate's NCH8 read-credit
default of 128. `--rtl-reference DIR` checks the saved run's correctness receipt,
original log digest, counter text, checkpoint config digest, effective RTL
parameters, exact static program and dynamic unit dispatch order. Lens uarch is
not authoritative: its dump used default values rather than the actual board
parameters. Lens and phase files receive consistency checks and new digests;
the original runner did not independently pin their digests.

`TMEMSTAT rpl[p]` counts lost read/replay port-cycles. Different ports' losses
overlap; unit freeze counts and grant-loss counters overlap pipeline/DRAM work.
No sum of those counters is an elapsed-cycle correction. The current LPN has no
structural bank arbiter/replay/backpressure timing. Both PMP modes therefore
have `calibrated=false`, `tok_s=null`, and an explicit missing-timing scope, even
where their average phase error passes 10%.

The saved-fixture comparison at nominal 200 MHz, lat65, adapter14, RD128 is:

| Workload | PMP | LPN cycles | Existing RTL cycles | Error |
|---|---:|---:|---:|---:|
| Gemma 3 1B, 4 layers + head | 1 | 865,859 | 889,451 | -2.65% |
| Gemma 3 1B, 4 layers + head | 0 | 865,859 | 891,870 | -2.92% |
| Qwen3 0.6B, all 28 layers + head | 1 | 1,413,623 | 1,533,159 | -7.80% |
| Qwen3 0.6B, all 28 layers + head | 0 | 1,413,623 | 1,544,070 | -8.45% |

All existing RTL rows are ISA-exact, int8 weights/head, position9, one decode
step. These are simulated cycles, not an on-card clock/throughput qualification.
The unmodeled PMP0 penalties are 2,419 Gemma cycles and 10,911 Qwen cycles.
Replay port-cycle totals are 1,700/15,648 and 12,307/80,035 respectively (PMP1/PMP0).
The full 26-layer Gemma baseline/combined comparison is separate and passed;
the matched PMP0/PMP1 Gemma fixture above covers four layers plus the whole head.

Independent PMP0 prediction requires bank arbitration, replay and unit pipeline
feedback inside the LPN. Aggregate counters and 64-cycle buckets do not identify
exposed delay. Do not fit a coefficient or add instruction freeze counts to
command service times.

`tools/wideport_tmem_resource.py` implements the cycle-stepped eight-port,
eight-bank/two-subbank read resource, including same-word sharing, guests,
pending addresses, replay age and PMP0 landing-write contention. Its write
pipeline helper models IQ-to-pw's two core-edge delay. PMP0's default read-side
mask is 0xf0; pumped modes use 0x0c, so both the side split and write occupancy
matter. Its standalone test matched 768 RTL cycles, including eight-lane reads
and a fixed-side pumped control; the controlled write-conflict window had 34
PMP0 replay port-cycles versus zero in the pumped controls.

The mixed slice experiment is implemented in
`tests/test_wideport_slice_feedback.py`. It uses 13-row KB9 int8 row-lane MM,
an independent 32x256 two-operand V_ADD, native NCH8 memory, stall50/seed7/lat120,
and a 32768-cycle trace cap. TMEM/DRAM match the bit-exact ISA in both modes.
Program, initial DRAM and source digests are identical; the actual compiled
parameter dictionaries differ only in TMEM_PMP. PMP2 is the core-clock model
of pumped TMEM; it is not a physical target or an on-card clock qualification.

| Mixed test observation | PMP0 | PMP2 |
|---|---:|---:|
| RTL run cycles | 4,855 | 4,603 |
| Complete trace cycles, including reset/loader | 5,014 | 4,762 |
| Replay port-cycles | 298 | 5 |
| VPU read-hold cycles | 257 | 0 |
| VPU queued-write grant-denial cycles | 144 | 145 |
| MXU compute advances while drain grant denied | 20 | 25 |
| MXU result pushes while drain grant denied | 3 | 6 |
| Maximum VPU write-buffer entries | 3 | 3 |
| Maximum MXU result-FIFO rows | 1 | 7 |
| Consumer/next-command handoffs | 4 | 4 |

`tools/wideport_unit_feedback.py` derives slice write-priority grants, guest
and replay holds, filtered reads and read takes. It couples the bank resource
to a VPU write FIFO containing addresses/completion bits and the registered
read enable/hold/lookahead, plus MXU operand/scale/result queue counts, row
capacity, independent compute/drain and next-command handoff. Every captured
cycle matches, including the full read-blocker cause matrix: 9,776 cycles,
zero mismatches. This demonstrates why an MXU-wide freeze is incorrect: compute
and completed rows can progress while its TMEM drain is denied.

The final focused EC2 regression passed 54 tests. Both original and final
traces are preserved (`slice-feedback-v1` and `slice-feedback-v2`). The final
regression reuses the same RTL binaries; only Python compile-contract recording
and rejection checks changed. The trace-only final audit additionally checks
each exact read-blocker cause. No full Gemma/Qwen RTL run was repeated.

The opt-in `+tmfeedback=FILE +tmfeedback_limit=32768` block in
`rtl/top/otpu_slice.sv` is under `ifndef SYNTHESIS`; no synthesized behavior was
changed. Records sample before the core edge from the first edge through halt,
with an explicit end/truncation marker. They contain raw/filtered read requests,
read takes, raw/granted/landing writes, replay/blocker masks, group grants,
VPU queue/enable/hold state and producer outputs, MXU queue/compute/drain state,
DRAM readiness and instruction events. The runner pins actual compile parameters,
input/program/source SHA256 digests and native stall settings. Unsupported
GST/rank/bank/SBH/slice contracts fail closed. A window starting after reset
would additionally need initial replay/age and IQ state; current traces start
at reset, so those states are reconstructed and checked cycle by cycle.

This is still a trace-conditioned component model, not a whole-program LPN
predictor. Arithmetic outputs (`cw_words`, `done_i`, `rf_push`), instruction and
operand availability, DRAM ready and the global collective decision remain
external producer inputs. To predict a changed TMEM schedule, generate their
address/valid/latency pipelines from issued commands and DRAM responses; replaying
recorded producer events under a different stall schedule would be invalid.
Gather same-cycle asks at one deterministic LPN cycle barrier, arbitrate, then
advance the relevant stages rather than delaying an entire instruction.

The result FIFO did not reach capacity in this experiment (zero full-capacity
cycles); its full-threshold backpressure branch needs a bounded stress case
before claiming coverage. The whole-model LPN estimates/errors in the first
table remain unchanged. Calibration and tok/s suppression stay active until
producer pipelines and their counterfactual feedback are modeled and validated.

## Physical target and prerequisites

Target `f2-hbm:p4rl8a64tdp`: S1, D128, PORTB4, MCOLS8, row lanes, ALINE64,
NCH8, FIFO1024, WIN16, MXU_E4, TMEM_IMPL3/NSB2/RPL/TDP/GST, PMP0; core200 MHz,
HBM450 MHz, shell250 MHz. PMP0 removes functional fast-clock TMEM endpoints;
it does not prove that the MXU/ACT core-clock paths meet 5 ns. The MMCM's unused
fast output may be trimmed; the flow checks all surviving TMEM RAM clock pins.

Builder `ubuntu@54.174.195.152` already exists; no instance launch is needed.
At 22:17 UTC it had 64 logical CPUs, 234 GiB available memory and 499 GiB free
disk, with one owner C1 Vivado implementation (PID2298083). At the later receipt
collection that PID was absent and memory available was 242 GiB; its build
outcome was not investigated or claimed. No owner job was stopped. Availability
must be rechecked before any separately authorized build.

Vivado2025.2, HDK `/home/ubuntu/aws-fpga`, the target-pinned HDK/IP commits, shell
version0x10212415, shell DCP and HBM XCI exist. License-seat availability is
unverified: at 22:35-22:36 UTC the read-only lmstat queries returned no license
file or seat data; the legacy loader was absent, a query via the standard loader
reported no license file, and sourcing settings exited without a seat result.
No license checkout was attempted. That preflight found no active Vivado/RTL
jobs and 242 GiB available RAM/499 GiB free disk; it is not a reservation.
The candidate is staged in our isolated
`/home/ubuntu/work/codex-wideport-20261021/lpn-accounting-v1` worktree. No CL build
snapshot has been created there. The PMP0 clock/RAM gates passed mocked flow
tests but have not yet been exercised on an actual PMP0 Vivado netlist.

## Sequence after model review and build scheduling

Use a fresh output directory under our own work root and a frozen candidate
revision. Run `boards/f2/build.py --target f2-hbm:p4rl8a64tdp --flow SynthCL`
with `--hdk /home/ubuntu/aws-fpga --out <fresh-synth-directory>`. Review logs,
resource inference, clocks, methodology and the PMP0 RAM/endpoint checks before
proceeding. Record the exact post-synthesis DCP SHA256 and source/input manifests.

Then use a separate fresh directory with `--flow ImplCL --from-synth
<successful-synth-directory> --synth-sha256 <recorded-digest>`. Use the fresh
implementation flow: the prior AWS partition incremental route attempt failed
to open its pb file. Preserve the successful baseline and the owner's runs.

Review routed setup/hold, pulse-width, unconstrained endpoints, CDC/DRC,
congestion/utilization and the post-route validation marker at the exact target
clock. Core-to-core closure is independently necessary after removing pumped
TMEM paths. The old TMEM box was already about 97.5% CLB/94.9% BRAM occupied;
do not tighten it without new congestion/path evidence. Attempting above200 MHz
requires new routed evidence and a supported clock recipe, not extrapolation.

No packaging/AFI creation/image load or card reservation action is included.
Physical timing closure, image generation and actual on-board correctness/
accepted-output decode TPS remain separate milestones. The current instruction
is to prepare this plan and not start a costly competing physical build.

Receipts are preserved outside the Git checkout under this task's
`evidence/wideport-validation/lpn-pmp200-accounting-v1/` and
`lpn-pmp200-accounting-comparison.json`, and on the builder under
`/home/ubuntu/work/codex-wideport-20261021/results/`.
