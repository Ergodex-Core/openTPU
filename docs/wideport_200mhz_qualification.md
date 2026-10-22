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

### Command-driven producer and FIFO-capacity follow-up

`validate_feedback_trace(..., predict_producers=True)` now reads the saved
eight-word instruction image and checks its assembled SHA256 (restoring the
trailing zero words omitted by `rtlsim._write_hex`). Two narrow producer models
replace captured events with command-generated state:

- `wideport_vadd_producer.py`: register-free V_ADD/B_FULL, SE/LANES8, no stream
  or experimental throttle. Its command queue, row/chunk addresses, enabled
  read/metadata pipeline and slot latency generate both read requests,
  `cw_words`, `done_i` and the write-buffer contents. Actual fp32 arithmetic is
  verified by the separate bit-exact ISA comparison, not computed here.
- `wideport_mxu_producer.py`: static RL4 int8, M1, KB1/KB9, CL8/CL16. Command
  KB and modeled pop advances generate row-ending tags and the free-running
  LROW+3 result-valid pipeline, including bubbles. Command/FIFO state generates
  active-head `first_row`, `c_unit`, `f_av`, `s_av`, `row_in`, `fpop` and capacity;
  command queue count and the delayed pop-command-cache load are checked too.
  Slice validation explicitly rejects a warm reset with live producer work.

Both saved mixed traces still match: 9,776 cycles, zero mismatches. No original
mixed or full-model RTL run was repeated for that comparison. The new bounded
`test_wideport_fifo_capacity.py` uses one 513-row/KB1 RL4 MM (129 row groups)
and a concurrent 2048-issue-cycle V_ADD that repeats a valid row. PMP2, native
NCH8, stall0/seed7/lat8 are fixed. It checks whole TMEM/DRAM against the ISA,
the entire bounded feedback trace, exact FIFO saturation, a ready row denied
solely by capacity during VPU write arbitration, and subsequent drain recovery.

| Capacity fixture | CL8 control | CL16 board configuration |
|---|---:|---:|
| FIFO capacity / observed maximum rows | 32 / 32 | 64 / 64 |
| RTL run cycles | 5,268 | 5,268 |
| Complete trace cycles | 5,304 | 5,304 |
| First ready-row capacity-block cycle | 2,336 | 2,376 |
| Result pushes while drain grant denied | 31 | 63 |
| ISA / command-driven cycle checks | pass | pass |

The EC2 focused regression passed 62 tests in 3.28s; three previously validated
standalone bank RTL cases were deselected. The two capacity fixtures reused
compiled RTL binaries. Earlier attempts are retained in `results/fifo-v1` and
`fifo-v2`: both passed ISA/cycle checks and RF32 saturation, but reached only
56/64 rows in the board case and correctly failed its capacity assertion. The
final program adds row groups so early drain activity cannot exhaust the offered
rows before saturation. These stimulus changes are not cycle correction factors.

This remains a trace-conditioned unit model. Its externally supplied signals are:
reset and accepted unit starts; MXU `c_done`, `pn` and `more` (sequencer release/
head state); inactive/load-edge prerequisite flags (unused while `more=0`);
MXU operand/scale arrivals `b_push`/`r_cpl`; drain epilogue `d_fin`/`c_asc`/`al_st`;
DMA/MXU/COLL/QUANT raw TMEM requests and writes; DRAM `q_want`/`write_ready`;
and the global collective grant. The small whole fixture therefore **cannot yet
be predicted independently**. In particular, its DMA and operand producers must
respond to the model's schedule, not replay captured arrival events under a new
PMP schedule. The smallest next step is a bounded native-memory issue/response
producer and explicit MXU release/head state, retaining these trace comparisons.

All four final trace comparisons total 20,384 cycles with zero mismatches.
`results/fifo-v3/final-producer-audit.json` records manifests, model/trace/log
digests, full-capacity cycles and recovery. Full-model LPN estimates/errors in
the first table remain unchanged; `calibrated=false` and `tok_s=null` remain
mandatory. No 200 MHz routed or on-card result is implied by these simulations.

## Deterministic native-memory milestone (2026-10-22)

The opt-in `+nat_deterministic=1` schedule in the simulation-only native memory
model uses a portable 32-bit hash keyed by seed, free-running core cycle,
channel and draw kind. It controls command/write readiness, performed-read/
write jitter and response bubbles. Default `$urandom` behavior remains selected
without that flag. This does not reconstruct the legacy simulator's global
random-call stream or model physical HBM/AXI timing. DDR3 timing mode is rejected
by the new predictor. Native data/masked-write contents remain separate from
its timing model and are checked by the fixture's full ISA comparison.

`wideport_native_memory.py`, `wideport_native_adapter.py` and
`wideport_mxu_issue.py` now couple issued MM commands through slice A/B request
registers, NCH8 read queues, APF32/APR16 scale prefetch/reuse, native memory,
B part joins, the slice B response register and the MXU scale-tag pipeline.
`wideport_memory_coupled.py` generates MM `b_push` and `r_cpl`; those observed
RTL signals are comparison targets only. Consumer admission and FIFO counts
use these generated arrivals. The prefix requires every MM issuer, operand
and scale entry, result pipeline and command to drain before it can pass.

The scoped adapter is read-only. The original PMP2 fixture allowed its VPU ST
to start before the last MM completed; its full-coverage guard correctly
failed despite matching through the first-write boundary. That trace is retained
in `native-mixed-v1`. The opt-in fixture now adds an explicit BAR before ST so
all five MMs lie in the modeled interval. The default fixture's program is
unchanged. This BAR is validation stimulus, not a production scheduling change.

| Deterministic mixed fixture | PMP0 | PMP2 |
|---|---:|---:|
| RTL run cycles, including validation BAR | 4,859 | 5,246 |
| Full slice trace cycles | 5,018 | 5,405 |
| Native boundary cycles through dump | 5,274 | 5,661 |
| Coupled command-to-arrival read-prefix cycles | 4,068 | 4,457 |
| Generated MM B entries / scale entries | 180 / 108 | 180 / 108 |
| MM A requests / B requests | 108 / 180 | 108 / 180 |
| Whole-TMEM/DRAM ISA / all-MM arrival checks | pass | pass |

Two standalone native tests additionally match 287 cycles with 640 accepted
commands, 512 read responses and 128 completed writes. The focused deterministic
suite passed 25 tests; the relevant default-mode regression passed 83 tests
(three unchanged standalone TMEM RTL cases deselected), including legacy mixed
ISA checks, model/calibration gates and board RF64 saturation. Corrupted native
response oracles are rejected. The saved-artifact audit also corrupts each
MM arrival oracle independently and requires a mismatch rather than feedback.

This is still not a whole-program predictor. External inputs are reset/A-flush,
accepted MM starts and release/head state (`more`, `pn`, `c_done`), drain epilogue,
TMEM drain grants, and DMA/loader requests/addresses. Adapter writes/SW and the
post-first-write interval are excluded from coupled prediction. The native-only
boundary comparison covers write-done timing from supplied native requests, not
independent generation of those write requests. No full Gemma/Qwen run was
repeated. Full LPN integration, calibration and TPS remain disabled.

Read-only license checks at 2026-10-22 10:39-10:40 UTC successfully sourced the
installed Vivado settings but found no configured `XILINXD_LICENSE_FILE` or
`LM_LICENSE_FILE` and no license at the three checked standard paths. The
installed lmutil's legacy ELF loader is absent; invoking it through the system
loader works, but `lmstat -a` exits 255 with "Cannot find license file" and no
feature/seat records. A bounded project/HDK script inspection found no additional
license-source configuration. This explains the current uncertainty, not prior
owner build licensing or global license availability. No server endpoint was
invented, no license checkout or configuration change occurred, and no physical
build started. Receipts: `license-status-v3.json` and its context companion.

## Bounded one/two-descriptor scheduling experiment (2026-10-22)

This is the first static ownership/prefetch experiment informed by the reviewed
Megakernels source at `7309cec801537b61fea3b50d7dfe454a6cde578e`:
https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e.
No GPU repository installation, runtime execution, numeric reordering or GPU
benchmark extrapolation is involved.

`tests/test_wideport_two_credit.py` compares the existing simulation-only
`+hz_uq=0:1:0:0:0` and `+hz_uq=0:2:0:0:0` dispatch caps. Both runs use the exact
same program, initial image, wait entries, compiled sources and parameters:
PMP0, NCH8, PORTB4, RL4 W8 M1, two N13/KB9 MMs, disjoint weight/scale pages and
output tiles, deterministic native stall50/seed7/lat120. The two MM inputs are
identical and both output tiles must match. A small V_ADD finishes before the
MM compute phase; this is not a transformer-layer or full-model benchmark.

The first BAR/NOP control was rejected: BAR intentionally allows early weight
prefetch in this RTL, so it did not serialize descriptor starts. Its traces and
failed assertion remain in `two-credit-v1`. The corrected control gates dispatch
until the previous instruction completes, without changing synthesized RTL.
This tests the benefit already provided by the existing two-descriptor design;
it does not introduce or prove a new production scheduling improvement.

| Short fixture result | One credit | Two credits |
|---|---:|---:|
| RTL run cycles | 1,464 | 1,198 |
| Second MM start, pre-edge trace cycle | 1,179 | 359 |
| First / second MM output committed | 1,174 / 1,562 | 1,256 / 1,293 |
| Last-old to first-new accepted native weight-read gap | 800 | -76 |
| Owner weight / scale native payload bytes, both MMs | 36,864 / 16,384 | 36,864 / 16,384 |
| Peak weight / scale FIFO entries | 36 / 36 | 72 / 72 |
| Peak result FIFO rows | 1 | 1 |
| Whole-TMEM/DRAM exact ISA and A/B match | pass | pass |

The complete fixture saves 266 cycles (18.17%) with no extra weight/scale
traffic and no observed ownership/dependency violations. Earlier prefetch also
delays the first MM's output by 82 cycles, so the analysis includes competition
with other traffic rather than subtracting a fitted handoff penalty. Signed
read gaps and phase gaps denote overlap when negative; they are not additive
idle-cycle budgets. Native port statistics cover boot and non-MM traffic too.

`tools/wideport_credit_budget.py` audits native channel-order read identities,
separate weight/scale arrivals, pop-order compute ownership, pipelined result
credits, drain, commits and conservative last-response page-release boundaries.
Actual buffer reuse/overwrite is not exercised. Exact dispatch timestamps,
explicit buffer release and accumulator availability remain uninstrumented;
per-owner collective completion is only a later-event candidate. Activation
readiness uses the fixture's sole preceding QACT completion.

Six focused tests passed in 2.90s on EC2. A saved-artifact audit independently
reconstructed both tiny ISA executions, checked original input/source hashes,
revalidated native timing, resource feedback and 2,889 coupled read-prefix
cycles, with no RTL rerun. Receipts are in `two-credit-v2/two-credit-2ds9puw7/`
including `comparison.json`, `credit-budget.json` and `final-credit-audit.json`.

This budget is an oracle analysis, not an independent sequencer or whole-program
LPN predictor. Calibration stays false and TPS null. Before chunk-level ready
edges or production scheduling changes, model command dispatch/release and
drain feedback independently, instrument missing ownership transitions, and
use a bounded dependent-producer/consumer fixture with exact operand/scale and
collective visibility. Neither this experiment nor nominal clock scaling proves
full-model accepted-output TPS, physical timing closure or FPGA execution.

## Ownership, accumulator stream and real page reuse gate (2026-10-22)

Verification-only trace additions now capture direct pre-edge dispatch PC/slot,
actual MM start and release identities, slot completion masks, operand/result
FIFO pointers, row first/last tags, column accumulator reset tags, output-pipe
busy and drained state. These additions change no synthesized design logic.
The checker binds starts/releases to `(slot, generation, expected MM PC)` and
rejects reuse of a live sequencer slot. It also checks released pop-head identity
and command-derived remaining advances.

`wideport_pipeline_ownership.py` independently advances reserved operand slots
and row-result credits. In the original two-MM coupled read-prefix check its
issue/arrival/pop/result pulses come from the generated memory/producer models;
observed arrivals remain comparison targets. In the write-reuse and saturation
audits those pulses are explicitly observed protocol inputs. Neither audit
independently predicts the sequencer, DMA or output epilogue schedule.

The DSP accumulators are continuously clocked per-column streams, rather than
per-MM banks. For the bounded RL4 W8 M1 KB1/KB9 shapes, the checker verifies
first/last ordering, pop -> m0p (1 cycle) -> m0 (2 cycles), each column reset
after its configured skew, row-valid at `2+LROW` and result FIFO push at
`3+LROW` (32 cycles with CL16/MCOLS8). Result-space reservation starts at the
first block and persists through pipeline latency/backpressure until a granted
drain. This is a bounded stream/credit audit, not unrestricted DSP hazard proof.

The real reuse fixture adds a third MM on WB0/SB0. It preloads distinct 64-byte
weight and scale patches to TMEM, completes the first two MMs at a validation
BAR, writes both patches through the ordinary DMA/native path, then reads the
reused page into a third output tile. Both patch acknowledgements must precede
that MM's start. The third result must differ from epoch zero and match the
exact ISA; all TMEM and DRAM must also match. Original and patched bytes form
the independent read-data oracle; no captured payload generates its expectation.

The old-epoch ledger uses native read counts generated by the independent
adapter/read-prefix model, including APF traffic, and closes before overwrite.
Each variant checks 1,152 owner read payload beats and both changed cache lines.
The third/new-epoch cache schedule and read-count closure are explicitly still
unmodeled. The ledger leaves that epoch unclosed and cannot authorize another
overwrite. This gate proves the conservative two-patch handoff, not earliest
page release, arbitrary page replacement or complete new-epoch reuse safety.

| Final bounded gate | Result |
|---|---|
| Relevant unit/negative/RTL regression | 71 passed in 131.98 s on EC2 |
| Original one/two-credit cycles | 1,464 / 1,198, unchanged |
| Three-MM write/reuse one/two-credit cycles | 2,183 / 1,868 |
| Reuse native traffic, each variant | 1,252 read beats and 5 write commands |
| Actual result-FIFO saturation | 64/64 rows; 129 rows produced and drained |
| Result slot reuse / output-backpressure cycles | 65 / 1,907 |
| Ready rows blocked at capacity | exercised; drain subsequently recovered |
| Five saved cases independently rechecked against ISA | exact full TMEM/DRAM |

Negative tests reject premature page overwrite, missing patch acknowledgements,
stale read epochs, premature FIFO-slot reuse, unavailable accumulators, duplicate
results and output drain without a grant. Six saved-oracle tests additionally
verify the positive trace and reject a corrupted accumulator reset, stale PC/slot
identity, premature output commit, grant-denied drain and old scale payload
after reuse. No assertion or numerical check was weakened.

The first-output 82-cycle penalty is visible before arithmetic: the second and
third prefix DMA completions shift from 587/726 to 669/808; BAR completion shifts
749 -> 831, QACT starts 752 -> 834 and completes 1092 -> 1174 (340 cycles in
both), MM release shifts 1095 -> 1177, and its first pop/result/drain/commit all
shift by 82. Thus the delay is already exposed in activation preparation and
release, while the measured MM compute/drain intervals remain unchanged. The
extra early requests alter shared memory phasing. This accounts for the observed
tradeoff without fitting a correction or treating the native read gap as an
additive penalty. Peak operand FIFO entries still grow from 36 to 72.

The first launch was stopped by SSH user-session teardown before RTL completion;
its partial log and traces are retained in `ownership-v1`. A transient system
job owned by ubuntu survived logout without changing login policy. The first
instrumented suite passed 16 tests (`ownership-v2`); final strengthened checks,
negative oracles and RF64 saturation passed in `ownership-v3`. The saved-artifact
audit reran only tiny ISA/model comparisons, never RTL. Receipts include
`ownership-v3/final-ownership-audit.json` and the matching logs/status.

The following write/reuse gate adds a command-generated single-chunk DMA ST
producer and a native adapter model for masked writes, independent command/data
handshakes, write completion and delayed A-cache/prefetch invalidation. The
three-MM program and arithmetic remain unchanged. Only simulation tracing was
added to RTL; synthesized RTL is unchanged by this gate.

Accepted DMA/MM starts remain explicit frontend inputs. The ST producer derives
its read/gather pipeline, request and completion from each ST instruction,
B-ready and generated write-idle. Native ready/read/write-completion timing
comes from the configured deterministic memory schedule. Observed request,
acknowledgement, cache-state and data fields are comparison oracles only. ST
source words come from an independent functional ISA oracle; this does not yet
model their dynamic TMEM value production. Initial DMA load/loader requests,
MXU release and drain epilogue, and TMEM drain grants remain external.

Both saved credit variants independently generate all five STs (including the
two 64-byte patches), then close every native command/read/write completion.
Each case checks 1,252 native payloads, 108 A-cache payloads and 482 joined B
chunks against independently sampled data. Old-epoch reads must retire before
page overwrite; new-epoch reads require both generated patch completions.
The new epoch generates and retires 288 weight beats and 32 scale beats, versus
288/128 for each initial page epoch. Final generated DRAM equals the exact ISA
result. The supported D128/NCH8/PORTB4/ALINE64/PMP0 configuration, program and
functional DRAM input are pinned before comparison.

The first instrumented case passed. A strengthened regression then exposed two
model bookkeeping errors: hashing boot-loader storage beyond the functional
DRAM input, and assigning multi-chunk read identities with the first chunk's
CHASH parity. Both were corrected structurally; failed logs/traces are retained.
No aggregate timing correction was introduced.

Final checks on the existing EC2 builder passed 84 focused/regression tests,
then 10 held-out-oracle checks. With native write-done lag increased from 4 to
31 cycles, the same RTL program and negative checks passed another 11 tests.
The default one/two-credit case remains 2,183/1,868 cycles; the delayed-ack case
is 2,310/2,004 cycles. All four cases have exact ISA TMEM/DRAM, identical input
program and memory, 1,252 read beats and five native write commands. The
post-run audit independently rechecks saved outputs and generated closure,
never relaunching RTL. Evidence is under `writes-v3`, `writes-delay31` and
`final-writes-audit.json`; preliminary `writes-v1/v2` and their failures remain
preserved. These are bounded simulated cycles, not decode TPS or clock closure.

The next finite gate is independent initial DMA load/loader request production,
followed by dependency-driven sequencer dispatch/release and output drain timing.
Keep calibration false and TPS null until those and other-unit dependencies are
qualified. No full Gemma/Qwen simulation, physical build, license configuration
or hardware action was performed in this gate.

## Physical target and prerequisites

The subsequent bounded input-loading gate generates the initial boot loader
and all five DMA LDs for the same three-MM program. Boot configuration is
pinned to the saved simulation command and the exact program/wait bytes in
DRAM. The loader's requests, IMEM rows and wait-sidecar rows advance on modeled
B-register grants and generated responses. The LD model represents its reserved
chunk buffer, synchronous read-first RAM, segment delivery, TMEM write register
and completion pipeline. Observed request/address/data/done fields are held-out
checks. The two page-patch ST payloads now come from generated LD data; the
three final MM-output ST payloads still come from the functional ISA oracle.

All four default/severe-memory credit cases independently generate ten boot
reads (five instruction rows and five sidecar rows), 40 LD chunk requests and
responses, 150 delivery segments, 1,200 TMEM words, and five LD done pulses.
Every request, enabled word/address/data, IMEM/sidecar row, grant ordering and
completion matches the trace. Mutations reject changed LD request/address/data,
early done, issued-word address, loader request/address/instruction bytes, and
sidecar row/data. Unit tests additionally cover denied grants, delayed responses,
partial segments, wrong response order, orphan responses and incomplete closure.

The final regression passed 114 tests in 14.26 s. A separate same-program run
with stall85%, read latency241, native command queue1 and write-done lag31
passed 22 tests in 28.56 s. Default cycles remain 2,183/1,868; severe-memory
cycles are 4,481/3,959. The LD producer sees 9/33 denied request-grant cycles
under that schedule, compared with zero at default. The initial loader has zero
denied-grant cycles in these RTL fixtures; its unit tests explicitly hold both
read-idle and grant low. All four cases have exact ISA TMEM/DRAM and identical
read and masked-write traffic. The original two-MM comparison remains separate
and unchanged at 1,464/1,198 cycles. A first model test rejected the saved HEX
format's omitted zero operand words; explicit instruction-count padding and the
existing full-program hash fixed that provenance check. Its failed log remains.

Receipts: `input-v1`, `input-delay`, `input-regression` and
`final-input-audit.json`. The saved-artifact audit reruns only tiny ISA/model
checks, not RTL. Remaining external inputs are the initial testbench boot start,
sys_rst/dma_rst/a_flush, accepted DMA/MM starts, MXU c_done/pn/more and drain
epilogue, TMEM drain grants, and functional final MM-output store words. The
next finite gate is dependency-driven sequencer dispatch/start/release for this
fixed program, then independent output-drain timing. Calibration remains false
and TPS null; this gate supplies no FPGA execution or 200MHz closure result.

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
