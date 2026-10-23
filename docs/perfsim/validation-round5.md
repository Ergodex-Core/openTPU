# Baseline WAITC and stream-engine validation

Round five adds the pinned compiler's **WAITC issue path and DMA DSTEP/STREAM engine**,
including finite buffers, VPU ownership and closed-loop Ramulator stalls. The original
RTL, compiler, LPN model, numerical gates and earlier receipts remain unchanged. This
closes three concrete baseline gaps; it does **not** establish the production board or
complete pretrained-model timing. The final pinned-Ramulator regression run passes **283 tests**.

Evidence is under `tools/perfsim/receipts/2026-10-19-round5/`. Its index hashes retained
observations and omitted regenerable images. New regression fixtures freeze actual RTL
observations, not simulator-generated expected timings. No checkpoint weights are included.

## Compiled production-path inventory

`audit_baseline.py` now inventories DSTEP and STREAM alongside existing scalar, DMA,
MXU, quantizer and VPU instructions. Pinned `opentpu/llm/qwen35.py` compiles wait entries
at lines 807–812, uses DSTEP at line 1690 onward, and uses STREAM for disjoint state updates
in batched/speculative paths around lines 1921–1995. `opentpu/language.py`'s `state_step`
supports DELTA, DELTA1, SCALE and DOT recurrences with constant, column or unit gates.
These compiler inputs are reused; no hardware or frontend source was rewritten.

WAITC takes the unchanged `opentpu.waits.wait_counts` table, per-unit distance counters and
a 31-entry VPU dependency mask. It has a two-stage frontend, registered readiness, tag-lifetime
backpressure and dynamic LOOP history. DMA/SE uses the actual eight-lane pipeline and finite
128-chunk input/output capacities, 64-chunk memory runs and explicit VPU drain/ownership.
Unsupported descriptors fail explicitly. Details are in [architecture.md](architecture.md).

The production target still requires **TMEM_IMPL=3 pumped/replay/guest timing, MXU_E=4,
DDR3/LiteDRAM shell/channel mapping and program loading/CHAIN/WAITW boundaries**. None of
those is implied by this round's replicated-TMEM, native DDR4/HBM2 comparisons. S>1/GATHER,
wide/T2 exports and other unsupported configuration fields remain rejected. Baseline compiler
feature flags are not a proof that the production target's physical topology is modeled.

## Fresh bounded RTL comparisons

All accepted comparisons require equal total cycles, every instruction dispatch/start/end,
every memory-port acceptance/completion with its address/size/direction, physical traffic,
ISA/RTL bitwise state agreement and an independent floating-point reference where supplied.
The memory controller is shared on both sides; compute request times are generated independently.
There is no timestamp forcing or averaged channel-rate substitution.

* 36 DSTEP/STREAM cases: D=32 and D=128; shapes 1×64, 7×64, 3×192, 32×256, 128×128 and
  256×256; DSTEP, zero-state DSTEP and GDN STREAM for every shape. All pass.
* 24 descriptor-mode cases: DELTA/DELTA1/SCALE/DOT × constant/column/unit gate × q output
  on/off, with A slot2, zero/nonzero state and disjoint source/destination. All nondegenerate
  final cases pass. The original three zero-state, no-output DOT checks are retained as
  undefined-relative-reference failures, not counted as numerical passes.
* Six HBM2 cases at a hypothetical 1 GHz core with eight outstanding transactions exercise
  read/gather/write backpressure. Four DDR4 cases exercise nonintegral 833 ps memory versus
  5000 ps core clocks. All pass.
* Three ordinary compiler programs with WAITC (vector/reduction, partial GEMM and complete
  tiny Qwen loop) pass. Six cold-launch hybrid steps also passed during development.
* Four persistent hybrid sequences, each comprising a two-token batched prefill and two
  teacher-forced decode launches, pass with DSTEP/STREAM on HBM2 and DDR4.

The hybrid workload is the unchanged Qwen3.5 frontend with a **complete random two-layer
model topology**: one DeltaNet layer with two 128×128 states, one gated-attention layer,
all norms/nonlinearities/MLPs, KV state and final vocabulary head. Hidden size128, FFN256,
vocabulary128; D128/MCOLS4/LANES8/SEQ_WC1, 64K-word TMEM and 32 MiB configured DRAM.
It is intentionally bounded and is not a pretrained billion-parameter model.

| Configuration | Prefill cycles | Decode cycles | Numeric verdict |
|---|---:|---|---|
| HBM2 DSTEP | 25352 | 16611, 16820 | all pass |
| HBM2 STREAM | 25392 | 16632, 16806 | all pass |
| DDR4 DSTEP | 25175 | 16681, 16824 | all pass |
| DDR4 STREAM | 25219 | 16770, 16970 | all pass |

At the stated 200 MHz core, HBM2 STREAM's device-only TTFT is **126.96 μs** and decode is
**11,956.72 tokens/s**, conditional on the two measured decode launches and their eight-cycle
launch gaps. These numbers describe only this tiny random topology and the program-resident
measurement boundary. They exclude loading, sampling, communication and host scheduling;
they are neither measured FPGA performance nor Gemma/Bonsai throughput. Machine-readable
per-stage cycles, bytes, transactions, stalls and utilization accompany the sequence report.

The persistent tests retain DRAM, TMEM, ACT RAM, KV/recurrent state and Ramulator time/open
rows/refresh. The strict instruction gate caught QACT's readiness occurring one edge before
sequencer completion. Registered stream input-count visibility and VPU drain ownership also
required explicit modeling. None of these fixes changes the golden RTL or measurement boundary.

## Independent int4 diagnosis

The two round-four random int4 failures remain failures of the original 5% float-model gate:
D32 **5.77272%** relative L2; D128 **7.94663%**. `diagnose_int4.py` regenerates the original
seeded float arrays, independently decodes nibbles/scale records, computes activation
quantization with NumPy, then computes the integer accumulation window and float32 epilogue.
It does not call `opentpu.quant` or `opentpu.fp32` for its expectation.

The independent QACT matches the ISA; the independent packed arithmetic matches **RTL and
ISA bit for bit**. No groups are dropped in these cases. The first representation difference
is before execution: original weight[0,0] is 0.0012643238296732306, encoded/dequantized weight
is zero. Weight relative-L2 loss is 6.6910% (D32) and 9.1544% (D128). This is quantization
quality in these two seeded cases, not evidence of a simulator, packing or RTL arithmetic bug.
It does not establish full-model int4 quality. Original references and failure receipts are
retained, and tests assert that the original float gate still fails.

## Measured host cost and exact batching

A cProfile sample identified memory tick/poll Python/FFI work as a major cost. The new optional
bridge call batches actual Ramulator ticks only until a completion or core boundary. Every
memory tick, refresh event, callback and RMW/ingress ordering remains intact. Python also
reuses bank counters, tracks the latest completion incrementally and bounds VPU-mask lookup.
No core cycle is skipped. Older bridges use scalar ticking; missing Ramulator never becomes
a simple-backend result.

The differential benchmark compares scalar and batched FFI on the **same final Python code**.
Trace-enabled runs match all memory events, instruction boundaries/stalls/work, statistics and
DRAM/TMEM hashes. Then three serial measurements per mode alternate order, with tracing off.
Median construction/run times, including ISA arithmetic and excluding backend construction,
input loading and serialization, on macOS ARM64/Python3.14.5/NumPy2.5.4:

| Workload | Scalar seconds | Batched seconds | Batched core cycles/host second |
|---|---:|---:|---:|
| Complete Gemma-shaped MLP up, 1×1152×6912 | 5.165 | 4.979 | 27,379 |
| Complete Bonsai-shaped MLP down, 1×6144×2048 | 7.866 | 7.560 | 28,403 |
| Tiny hybrid DSTEP two-token prefill | 1.625 | 1.568 | 16,172 |

This is only about 3.5–4% lower wall time from FFI batching. The third benchmark's historical
directory name contains `stream`, but its recorded configuration/opcodes are DSTEP; it is
reported as DSTEP here. No large speedup or full-model runtime is claimed. Earlier host
measurements use a different code revision and are not a controlled before/after comparison.
A true pretrained full-model run remains unperformed; no unbounded run was started.

## Reproduce

Build the pinned Ramulator dependency as described in [README.md](README.md), then set
`BRIDGE` to the produced `.dylib`/`.so`. Use fresh output paths. All inputs below are seeded;
no model download or credentials are required.

```sh
python tools/perfsim/stream_examples.py --out /tmp/streams --depth 128
python tools/perfsim/stream_examples.py --out /tmp/modes --depth 128 --modes
python tools/perfsim/build_rtl_ramulator.py --bundle /tmp/streams/dstep_128x128 --out /tmp/rtl-stream
python tools/perfsim/compare_native.py --bundle /tmp/streams/dstep_128x128 \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --executable /tmp/rtl-stream/obj/Vtb_perfsim --out /tmp/check-stream

python tools/perfsim/hybrid_example.py --out /tmp/hybrid --waitc --stream --chunk-prefill
python tools/perfsim/build_rtl_ramulator.py --bundle /tmp/hybrid/hybrid_0 \
  --persistent --out /tmp/rtl-hybrid
python tools/perfsim/compare_persistent.py --bundles /tmp/hybrid \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --executable /tmp/rtl-hybrid/obj/Vtb_perfsim --out /tmp/check-hybrid
python tools/perfsim/sequence.py --bundles /tmp/hybrid \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --out /tmp/hybrid-metrics
OTPU_RAMULATOR_BRIDGE="$BRIDGE" python -m pytest tests/perfsim -q
```

Omit `--stream` to compile DSTEP; omit `--chunk-prefill` for serial cold-launch input bundles.
Use `catalog.json` for all kernel cases and the supplied DDR4/stress profiles for the other
memory conditions. `benchmark_differential.py --bundles ... --profile ... --bridge ... --out ...`
runs the exact scalar/batched comparison and bounded host measurements. `diagnose_int4.py`
takes the original round-four `--bundle`, `--rtl-run` and a new `--out`; the corresponding
regression also embeds the frozen RTL output vector and original float reference.

## Preserved failures and limits

Initial persistent prefill extracted row0 logits although the compiler wrote requested row1;
the generator now records the correct row offset. Its original failed report is retained.
One nominal STREAM rerun omitted the CLI flag and actually compiled DSTEP; it is excluded
from STREAM evidence and replaced with a verified STREAM opcode/configuration run. Three
zero-state DOT references had zero norm; their unchanged relative/cosine gate is undefined,
so nonzero DOT cases were added. These are input/measurement bookkeeping corrections, not
relaxed gates. Source snapshots distinguish development observations from final regressions.

The full pretrained model, production pumped-TMEM/DDR3 shell, broader hardware parameter
space and physical Fmax/resource fit remain open. Existing LPN calibration and its five
configuration-specific aggregate errors are untouched and are not presented as cycle-exact
or broad validation. No PR was created, no branch was pushed, and no FPGA job was run.
