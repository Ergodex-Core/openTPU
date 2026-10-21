# Wide-port 200 MHz qualification checkpoint

2026-10-21 UTC. This is a plan and evidence checkpoint; no new physical build or
hardware programming was performed. The successful 150 MHz image is preserved.

The isolated candidate is `codex/wide-port-200mhz-opt`. Production RTL last changed
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

Independent PMP0 prediction requires bank request/grant/replay state and unit
holds/backpressure represented inside the LPN, validated against cycle-level
events. Saved aggregate counters and 64-cycle buckets do not determine that
overlap. Do not fit a coefficient from these same elapsed times or inject all
instruction freezes into service time. A narrow cycle-level arbiter trace/test
is the next useful model experiment; it does not require repeating full models.

That first experiment is now implemented in `tools/wideport_tmem_resource.py`
and `sim/verilator/tb_tmem_replay_trace.sv`. The cycle-stepped reference resource
models the eight-port, eight-bank/two-subbank read-side arbitration, same-word
sharing, guest collapse, pending addresses, replay age and PMP0 landing-write
contention. The write pipeline helper checks IQ-to-pw's two core-edge delay.
PMP0's default side mask is 0xf0; PMP1's is 0x0c, so mode comparisons must account
for both the side split and write occupancy.

On EC2, 49 tests passed, including 768 RTL cycles in three cases: PMP0/default
side map, PMP1/default side map, PMP1/fixed0xf0 side map. Each cycle's loss,
freeze, cross/self-block masks and delayed writes matched exactly. The bench
uses the slice's MXU-only crossbar and other-port rotators, single- and eight-lane
requests, guest traffic, aged replays, same-word bypass and different-word
landing-write conflicts. In the controlled write-conflict window, PMP0 has 34
replay port-cycles and both PMP1 cases have zero. This is a resource-boundary
test, not an elapsed program-cycle or throughput result. It checks masks and
write delay, not returned operand data. The component is not yet wired into
`WpCompactOpenTPU`; the whole-model estimates/errors above are unchanged and
the uncalibrated/TPS suppression remains active.

The next integration requires per-beat unit state, not a delay on whole
commands. In the LPN, gather all units' current-cycle asks, arbitrate at one
deterministic cycle barrier, then advance only the relevant pipeline stages.
Keep MXU prefetch/compute independent from drain until its result FIFO fills;
keep VPU operand reads independent from write-buffer pops. A resource token
held for an entire instruction cannot express those overlaps.

The missing slice trace must use a documented pre-core-edge sampling point and
include the following, with initial pipeline/replay state if a bounded window
starts after reset:

- Raw `rq_en`, filtered `rq_t`, actual `r_en`, lane masks and enabled word
  addresses for every port; input granted `w_en/w_addr`, IQ registers, landing
  `u_tmem.pw_v/pw_a`, and configuration/side/rank/guest masks.
- `tm_lost/tm_frz`, `u_tmem.g_one.rpv/rpe/rpa`, replay age, write-conflict `wm`
  versus read blockers `hm`, group `gnt`, `rpl_hold/rpl_frz`, and guest blocking.
- VPU `rhold`, `g_wbuf.en_h/en_r/en_q/ren_r/wn/push/pop`, pipeline enable, and
  instruction start/done identity. The unit bench's immediate group freeze is
  a stimulus policy; it does not model the VPU's registered hold/room feedback.
- MXU `pop/mxu_pop`, `drain_go/t_gnt/d_pop`, `rf_n/rows_live/f_count/s_count`,
  result/operand pipeline valids, and DRAM A/B/write ready/grant events. Quant
  read stages, group hold and `q_awant/sw_rdy_i` are also needed to separate TMEM
  replay from external write-room stalls.

The smallest next integration experiment starts from
`test_wide_rowlanes_unpumped_tmem_under_stalls` (13 rows, KB9, seed7, native
stalls) paired under PMP0 and the core-clock PMP2 model, with a bounded trace.
Its current V_FILL has no operand reads: add a two-operand VOP on independent
TMEM buffers to exercise VPU B replay while row-lane MM/drain is active. Keep
ISA-exact TMEM/DRAM checks and assert the required conflict and queue events
actually occurred. Check the resource oracle against slice loss/freeze masks,
then use queue transitions to validate the next LPN unit model. No full Gemma or
Qwen rerun is necessary for this experiment. It has not been run yet.

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
