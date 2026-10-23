# Matched-context FPGA benchmark

Status: preparation and ISA checks only. No 1K/2K/4K physical measurement has run.

These tools keep model preparation, numerical correctness, physical measurements,
client delivery and colleague reports distinguishable. Run preparation and tests
on the existing `opentpu-m8a-build` (`i-0ec8d33362b3100be`), using existing
dependencies. The Mac is the control and Git bridge.

## Frozen workload

| Case | Exact rendered input | Reserved output | Minimum total | Aligned total |
|---|---:|---:|---:|---:|
| 1K | 1024 | 64 | 1088 | 1280 |
| 2K | 2048 | 64 | 2112 | 2304 |
| Screenshot 4K | 4032 | 64 | 4096 | 4096 |
| Optional exact 4096 input | 4096 | 64 | 4160 | 4352 |

Use one **4096 total-token** packed layout for all three primary cases, so
layout/residency does not change with prompt length. The optional 4096-input
case needs a separate 4352 layout and must be reported separately.

Batch is one, as confirmed by the user for the colleague's results. Compare
Gemma3-1B INT8 weights and INT8 head first. Greedy selection, including the first
token, must run on the FPGA. Use the same image
`agfi-0001de1dd3df8e21e`, profile `f2-hbm:p4rl8a64c150`, build `0x417243f1`,
150 MHz, for every accepted model. Gemma3-270M is an additional candidate and
cannot stand in for the screenshot's 1B model. Qwen3-0.6B text LM is separate
from Qwen3-TTS. Nanbeige remains physically unaccepted.

`prepare.py` uses each cached canonical tokenizer and chat template. It **adds**
neutral text to reach each target exactly and verifies rendered text re-encodes
to identical IDs. It saves all messages, rendered text, token IDs, file hashes,
capacity limits and instruction hashes. It never truncates. The quality task is
an integer list intended to sustain a 64-token window. Respect real EOS. If EOS
occurs early, report the actual output count and an incomplete 64-token window;
never suppress EOS or fabricate/pad output. The colleague's output count is
not a gate for doing this experiment.

## Reproduce offline checks

From the isolated builder checkout, using existing packages:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
  /home/ubuntu/work/p1-venv/bin/python tools/context-bench/prepare.py \
  --pack --output results-new

env OTPU_BUILD_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  OTPU_IMAGE_CACHE=0 OTPU_PROG_CACHE=0 \
  /home/ubuntu/work/p1-venv/bin/python -m pytest -q \
  tests/test_gemma_full_kv.py tests/test_context_bench_metrics.py \
  tests/test_gemma3.py tests/test_fpga_only_chat.py

/home/ubuntu/work/p1-venv/bin/python tools/context-bench/isa_smoke.py \
  --output isa-smoke-new.json
```

`prepare.py` forbids physical transports and all forward calls. `isa_smoke.py`
permits the ISA simulator only, blocks physical transports and host token picks,
and checks reset/replay on real cached weights. ISA timing is never FPGA timing.
No package installation, model download, credential change, image build or image
reload is needed for these checks.

## Capacity change

The default Gemma ring has `min(capacity, 512+256)` slots. Above 768 total tokens,
old positions are overwritten, `Engine.stateful` becomes true, and the strict
FPGA-only serving guard refuses it. Simply raising the worker limit is invalid.

The isolated `kv_full=True` option stores sliding-layer KV at every context
position. `_slide_seq` still applies the checkpoint's exact 512-token sliding
window. No attention-mask, quantization, model-weight or RTL change is intended.
The option propagates through compiler reconstruction/cache choices. Default
callers retain the old layout. Existing strict guards remain intact.

Memory fit, successful compilation and ISA equality establish a candidate for
physical qualification; they do not qualify DMA, board execution, quality at
long prompts or a production interruption bound.

## Owner-run measurement, after approval

There is no transport opener, lock acquisition, image loader or service control
in this harness. `observe_owned.capture_owned_turn()` accepts only an already
qualified physical `BoardBackend` engine supplied by the production owner. It
records the raw trace and all selected token IDs, resets KV positions before
each request, forbids a host first-token pick, and records worker timings. The
owner supplies exact model/checkpoint/runtime identity, confirmed normal locks,
window ID and a stop callback. A stop callback alone is **not** a hard bound for
blocking native calls; the owner must qualify restoration/watchdog behavior.

The current worker only accepts 512/768 contexts and its serving metadata,
budget guard and lease/cache identity are pinned. This candidate requires an
isolated owner integration and physical acceptance before any public limit
changes. Do not patch production or bypass the strict guard to run a benchmark.
The observer does not yet provide the worker migration or end-to-end client
instrumentation; those are explicit integration gates.

1. Coordinate ownership first. Keep service interruption at most 30 seconds.
   The second F2 slot is still owned by Accelsmith and is not available. Unknown
   long-prompt duration cannot be used to promise a 30-second outage. Obtain a
   released independent slot or an owner-qualified complete interruption bound,
   including cleanup, same-owner Gemma restoration and reopening.
2. Freeze the source/model/tokenizer/template hashes from `preparation.json`.
   Prepare RAM images and programs before the ownership window. Re-probe the
   approved board identity under both normal locks; do not reload the image.
3. Physically qualify long-context boundaries, greedy token/reset parity and
   cancellation/restore. Check the integer-prefix task and retain raw text/IDs.
   A small task sanity check is not full native-checkpoint accuracy acceptance.
4. Measure at least 10 warm repetitions per length when the approved window
   supports them. Reset KV for every repetition while retaining weights/programs;
   do not reuse the prompt prefix. Preserve order/temperature/run IDs. Report
   medians and nearest-rank p95 with repetition count. Small-sample p95 is merely
   descriptive. Separate model-cold cached, model-cold uncached and warm resident
   runs. Cold restaging also needs approved ownership.
5. Instrument the actual client at submit and every observed token arrival, on
   its own monotonic clock. Record HTTP/gateway/worker phases. A browser receives
   text chunks that may contain several tokens: retain cumulative token indices
   and coalescing instead of pretending each chunk is one device token. The
   current worker delta events alone cannot establish per-token device latency.

## Timing definitions

* **End-to-end TTFT:** client submit to first visible generated token at that
  client, with queueing/import/preparation/upload/network/render scope stated.
* **Worker TTFT:** owner observer submit to first non-EOS token ID read back;
  this is neither browser TTFT nor a cycle-level first-token timestamp.
* **Device prefill:** summed prefill cycles / measured clock. Strict FPGA mode
  holds the final prompt token for the first device generate run; report the
  exact input total and prefill's `P-1` token count separately.
* **Average decode TPS:** `(N-1)/(t_last-t_first)` for N visible output IDs,
  excluding the first-token interval and prefill. Never use N in that numerator.
* **Late decode window:** last 32 observed IDs, 31 intervals divided by their
  elapsed time, plus median/p95 observed intervals. KV grows during the full
  window. Host polling can coalesce IDs, so these intervals are readback/arrival
  observations, not device token latencies. Cycle-level steady-state latency
  needs additional per-token cycle evidence, which this harness does not invent.
* **Generate device rate:** the existing cycle counter includes the held final
  input token and first pick; retain steps/cycles and label it separately. It
  cannot silently become a decode-only denominator.

`metrics.py` rejects non-FPGA evidence and mismatched model, precision, clock,
image, prompt hash, runtime digest, checkpoint hash, context and host-compute
contracts. Append owner records as JSONL; raw events stay attached to each run.
Cold receipts need a separate owner recorder including initialization phases.
If no client trace exists, leave end-to-end TTFT unavailable.

```bash
/home/ubuntu/work/p1-venv/bin/python tools/context-bench/metrics.py \
  --preparation results-new/preparation.json \
  --receipts owner-measured.jsonl --output measured-summary.json
```

To fill `identity.runtime_source_digest`, SHA256 the UTF-8 JSON of `sources` in
`preparation.json`, with sorted keys and compact `(',', ':')` separators. Model
and config hashes come from that model's `files` entries. Attach raw receipts,
source SHA, environment, all rejected/incomplete runs and output-quality checks.
