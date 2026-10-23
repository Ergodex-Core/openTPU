# Inferlab/OpenTPU context benchmark checkpoint — 2026-10-23

**No matched 1K/2K/4K FPGA timings are available yet.** The live Inferlab worker
is validated for 768 **total** tokens, including output. This task did not open
the F2 card, acquire its locks, pause serving, reload an image or change credentials.
The same-image software capacity candidate, exact workloads and measurement
harness are prepared and tested on the existing builder. Physical qualification
and performance measurement remain pending a coordinated ownership window.

## Comparison currently supported by evidence

The colleague's screenshot was materialized from Library
`libfile_0eadc4441a84819180919299e4d875eb`, version 0, and inspected as readable
pixels. Its SHA256 is
`c912d880f40940ea17768928aa9ff22fa7ef7701f963c1ee20de45d4d03abdba`.
The table matches the parent's transcription. The user confirms batch one.
These are colleague-reported **on-device** results on an independently written
F2 accelerator, not results independently reproduced here.

| Model / weights | Input tokens | Reported decode tok/s | Reported TTFT s |
|---|---:|---:|---:|
| Qwen3-0.6B INT4 | 1024 | 710.4 | 1.05 |
| Qwen3-0.6B INT4 | 2048 | 587.9 | 2.40 |
| Qwen3-0.6B INT4 | 4032 | 434.2 | 5.89 |
| Qwen3-0.6B INT8 | 1024 | 394.1 | 1.61 |
| Qwen3-0.6B INT8 | 2048 | 353.4 | 3.51 |
| Qwen3-0.6B INT8 | 4032 | 291.5 | 8.07 |
| Gemma3-1B INT4 | 1024 | 567.3 | 1.33 |
| Gemma3-1B INT4 | 2048 | 560.9 | 2.68 |
| Gemma3-1B INT4 | 4032 | 549.2 | 5.37 |
| Gemma3-1B INT8 | 1024 | 266.4 | 2.20 |
| Gemma3-1B INT8 | 2048 | 264.9 | 4.43 |
| Gemma3-1B INT8 | 4032 | 262.3 | 8.82 |

| Matched primary baseline | Input | OpenTPU TTFT | OpenTPU decode TPS | Colleague TTFT / TPS |
|---|---:|---|---|---|
| Gemma3-1B INT8, batch 1 | 1024 | Unmeasured | Unmeasured | 2.20 s / 266.4 |
| Gemma3-1B INT8, batch 1 | 2048 | Unmeasured | Unmeasured | 4.43 s / 264.9 |
| Gemma3-1B INT8, batch 1 | 4032 | Unmeasured | Unmeasured | 8.82 s / 262.3 |

The screenshot does not identify exact checkpoint/quantizer, tokenizer/template,
clock, quality results, repetitions or first-token timing scope. The user says
generated count need not block evaluation; this harness chooses a disclosed
64-token decode window. Different output windows still affect averaged KV-growth
cost, so the denominator and late-window rate remain explicit. Qwen3-0.6B here is
a text LM, not Qwen3-TTS. INT4 is a separate precision comparison.

Historical Inferlab browser evidence reports **0.67 s warm TTFT and 63.21 tok/s**
for the short sky prompt, with 25 visible output tokens. Its prompt is about 20
rendered tokens, with the final input held for device generation. It is not a
1K result. The cold cached browser run was 10.46 s TTFT. A longer 545-rendered-token
prompt (544 prefill + one held input) had 15.86 s browser TTFT, with 15.53 s
reported worker prefill-phase wall time. That phase includes FPGA execution and
host orchestration; it is not an isolated cycle measurement. None of those values is extrapolated to 1K/2K/4K and no
matched speedup ratio is calculated. See the retained
[historical verification](context_benchmark_20261023_evidence/historical/verification-v4.md) and
[raw bounded hardware receipts](context_benchmark_20261023_evidence/historical/cap768-hardware.jsonl).

## What prevents a direct run today

The serving lineage is `18fb978033799f0f21258f0e6e085c0360322d96`.
Its Gemma sliding cache has `min(capacity, 512+256)` slots. Above 768 total tokens,
it overwrites old positions, becomes stateful, and fails the unchanged strict
stateless-KV FPGA guard. All requested input lengths alone exceed the current
serving capacity; a 1024-token prompt plus 64 output tokens needs at least 1088
total slots. Raising just a worker/API limit or truncating input is invalid.

The isolated implementation is based on Nanbeige staging revision
`8864ef65267a5d45fd72e5fdc17f4b797095a687`, whose Gemma 768 preparation was already
compared byte-for-byte to the serving lineage by the Inferlab owner. This task
does not claim that historical offline migration check is physical migration
acceptance. See the retained [migration receipt](context_benchmark_20261023_evidence/historical/gemma-runtime-migration.json).

The opt-in `kv_full=True` setting retains KV at every context position and keeps
the exact 512-token attention window. It propagates through compile-worker and
cache layout reconstruction. The default remains unchanged; strict FPGA guards
are preserved. This is host/runtime storage support, with no RTL or AFI change.

## Builder results and additional 270M candidate

SSH used the existing trusted route `ubuntu@54.174.195.152`. Before execution,
IMDS verified `i-0ec8d33362b3100be`, the expected `opentpu-f2-ssm` instance profile,
and hostname `ip-172-31-89-205`. No ancestor or checkout `AGENTS.md`/`.agents`
instructions were present at the inspected locations. Existing
`/home/ubuntu/work/p1-venv` supplied Torch/Transformers; no package was installed.
All preparation/tests/ISA runs used the isolated builder directory
`/home/ubuntu/inferlab-context-bench-20261023`. The Mac handled inspection,
source editing, Git and transfers only.

Both models were already cached under `/home/ubuntu/models`, including the gated
Gemma3-270M weights. No new license agreement was accepted, token supplied or
checkpoint downloaded. The official [270M model card](https://huggingface.co/google/gemma-3-270m-it)
lists 32K architectural context. That architectural limit does not qualify this
runtime. Cached config, weights, tokenizer and template hashes are retained.

| Cached INT8 candidate | Layers / hidden / FFN | Full-KV capacity | Packed RAM image bytes | Largest generate program words |
|---|---|---:|---:|---:|
| Gemma3-1B | 26 / 1152 / 6912 | 4096 total | 1,612,115,456 | 7,456 |
| Gemma3-270M | 18 / 640 / 2048 | 4096 total | 657,016,320 | 5,352 |

Both have four query heads, one KV head, 256-dimensional heads, 512-token sliding
attention, global layers every sixth layer, tied embedding/head, GeGLU and the
same supported norm/RoPE conventions. Full INT8 packing and all 16 greedy
generation buckets compiled against the existing image profile's 4 GiB memory
and 32,768-word IMEM geometry. RAM chain writes were bounds-checked. No host
embedding, PLE or compute service was enabled. Preparation recorded **zero
physical-open attempts and zero model-forward attempts**.

Weightless layouts also fit at 4352 total tokens for the optional 4096-input case:
1B uses 1,628,632,576 bytes and 270M uses 668,814,848 bytes. All 17 decode buckets
compile. The 4352 case was not packed with real weights or physically validated.

The canonical 1024/2048/4032/4096 input cases were generated without truncation.
Rendered text re-encoded to the exact stored token IDs. Those case IDs happen to
match across these two cached tokenizers; model files and tokenizer hashes stay
separate. The main three cases all use one 4096-total layout; 4032+64 fills it.

Validation receipts:

* The main regression selection passed **58 tests**, with three skips: two
  optional relative-path real checkpoint probes and the absent optional GGUF
  package. The explicitly supplied real cached models were separately packed
  and simulated, so those probes do not mean these models were unavailable.
* An additional small complete numerical model traversed **4160 positions and
  all 17 buckets**, comparing old sliding storage to full storage bit-for-bit;
  it passed. The mixed sliding/global test also passed across the original
  768-slot ring boundary, rollback/reset and chained generation boundaries.
* Both actual cached models, INT8/greedy/full-KV/4096, produced eight tokens after
  “Say hello.” on the ISA simulator and reproduced identical raw IDs after reset.
  1B: “Hello there! How’s your day”; 270M: “Hello! How can I help you today”.
  Host token selection was forbidden and physical-open attempts stayed zero.
* The final **16 analyzer/observer tests** passed, covering the N−1 decode
  denominator, p95 convention, coalesced arrivals, EOS and fail-closed identity /
  physical-backend checks. Seven overlap the earlier suite; do not add these
  suite totals as unique tests.

ISA correctness and RAM packing are not hardware performance or long-context
native-checkpoint quality validation. The 270M model remains an additional
physically unaccepted candidate. Nanbeige remains unavailable for performance
reporting; its staged host preparation is not a physical acceptance receipt.

## Measurement protocol and remaining approval

The [runbook and tools](../tools/context-bench/README.md) freeze exact canonical
counts, batch one, greedy selection, fixed output target 64, full raw IDs,
reset/no-prefix reuse, warm residency and separate cold runs. The observer
accepts only an owner's already-qualified physical backend; it opens no card,
takes no locks and controls no service. The analyzer rejects ISA/CPU/estimate
receipts and mismatched identity/precision/context. Client TTFT is unavailable
until an actual client trace is attached; worker TTFT is reported separately.

Average decode rate uses `(N−1)/(t_last−t_first)`, excluding prefill and the
first-token interval. It also reports the final 32 observed IDs over 31
intervals as a later KV-growth window. Readback/arrival coalescing is disclosed;
these intervals are not cycle-level device token latency. Existing generation
cycles include the held final input and first pick and remain separately labeled.
Use at least 10 warm repetitions if the approved window allows; report median,
nearest-rank p95, run count, rejected/incomplete runs and quality checks. Honor
early EOS and disclose the shorter actual count.

The exact remaining gates are:

1. **Pending, not attempted:** parent/production owner approves exclusive card
   ownership. The Accelsmith second slot has not been released. No hardware
   action is authorized here yet. No physical operation has an uncertain outcome.
2. **Pending, not attempted:** owner integrates and qualifies this isolated
   extended-capacity runtime, source/model identity, worker/API budgets and cache
   namespace. The live worker still only accepts 512/768 and was not changed.
3. **Unqualified bound:** retain the user's maximum 30-second serving
   interruption. The complete transfer/run/cleanup/restore/reopen path, hard
   native-call deadlines and independent restoration watchdog lack a measured
   bound for these workloads. Do not promise that an unmeasured long prompt fits
   the outage. Obtain a released independent slot or qualify the complete bound
   before a shared-slot test.
4. **Pending:** physical long-context token/reset/cancel/quality acceptance and
   actual client/worker/device-cycle collection. Only then fill the empty
   matched-result table. No new AFI, image reload, credentials or paid resource
   is requested by the capacity candidate itself.

No automatic approval review denied an action in this task. A transient
execution-environment disconnect interrupted a local inspection command before
it started; execution resumed. An initial test command referenced a nonexistent
test filename and ran no tests; the corrected selection passed. All launched
builder jobs are terminal in the final checkpoint.

## What the gap does and does not establish

The old warm short TTFT versus cold TTFT shows a setup/residency cost. The
historical long prefill also has substantial measured worker prefill-phase time,
so removing browser or initialization overhead alone does not explain it. The
retained worker receipt does not separate hardware cycles from host orchestration
within that phase.
Neither observation identifies the cause of a matched long-context TPS gap,
because that OpenTPU measurement does not yet exist.

For an independent F2 RTL accelerator or OpenTPU optimization, collect matched
phase/cycle and trace evidence for matrix engine occupancy, HBM channel balance
and burst efficiency, weight reuse across prefill rows, global versus sliding
KV traffic, GQA reuse, normalization/activation costs, program/synchronization
stalls and LM-head/argmax time. Host launch/compilation/readback and network
delivery need separate clocks. The existing image has hardware generation and
chaining already; a device loop alone does not prove the higher reported TPS
comes from software. The screenshot cannot establish bandwidth as the sole
limit either. Exact INT8 format, clock, attention and quality comparisons come
before an architectural attribution.

An independent accelerator can be implemented as RTL on F2 using its FPGA
resources. That is an FPGA accelerator implementation, not fabrication of a
new silicon chip. This task starts no replacement RTL design or image change.

## Publication evidence

The original completed source commit is `f66a50a0fdae78c10ff5763eb0161a19eb53b6c7`. This publication adds its [unaltered text/JSON receipts](context_benchmark_20261023_evidence/README.md) and fixes the historical evidence links; runtime code is unchanged from that source. The original 182-path manifest includes 91 AppleDouble metadata paths. All 91 actual Python sources and eight changed source/test hashes match the committed source; metadata files are not imported. New preparation must generate a fresh source manifest/digest.
