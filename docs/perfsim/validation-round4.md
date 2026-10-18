# Persistent execution and model-sized operators

This round adds **18 persistent RTL launches, 28 operator comparisons and two complete
projection comparisons** against the shared, closed-loop Ramulator controller. Every one
matches total cycles, every instruction dispatch/start/end, accepted memory addresses and
sizes, acceptance/completion cycles, and physical traffic. The pytest suite passes **200 tests**.
These are bounded validation results, not complete pretrained-model or production-board
validation. Earlier receipts and their failures remain unchanged.

Evidence is in `tools/perfsim/receipts/2026-10-19-round4/`. `index.json` records original file
hashes and retained paths. Large observed memory traces are gzip compressed. Input/output
images are hashed; synthetic inputs can be regenerated from the compiler and fixed seed.
No checkpoint-derived weight images or output tensors are committed. Regression fixtures
contain synthetic inputs and frozen RTL observations; replaying them is not a new RTL run.

## Persistent launches

One Verilator process now executes each sequence. DRAM, TMEM, ACT/scale RAM and Ramulator
open rows, refresh phase and global clock persist. Core registers, issue state and unit
pipelines reset between drained launches. Only declared host-input ranges are updated.
The comparator checks idle controller queues before switching stages and checks global
launch times independently. Program loading is outside the measured core boundary.

| Configuration | Per-launch RTL/model cycles | Explicit gap |
|---|---|---|
| HBM2, 200 MHz, tiny Qwen | 5748, 5758, 5715, 5752, 5763 | 8 cycles |
| DDR4, 200 MHz, tiny Qwen | 5783, 5781, 5744, 5746, 5748 | 8 cycles |
| HBM2, 1 GHz, eight outstanding transactions | 18319, 18687, 18721, 18711, 18477 | 264 cycles |
| HBM2, retained TMEM/ACT probe | 81, 52, 74 | 8 cycles |

Each five-launch sequence comprises two prompt-token steps and three teacher-forced decode
steps of the complete **tiny random two-layer Qwen topology**. All independent NumPy logits
pass the existing numerical gate. The separate state probe consumes retained ACT and TMEM
in later launches, including accumulation; RTL and the sequential ISA oracle agree bitwise.
It has no separate floating-point model reference.

The eight-cycle fixed launch gap is three dump/settle edges plus five core-reset edges.
Additional `--host-gap` cycles advance memory time, including refresh. No request timestamps
are forced. For the HBM2 tiny sequence, device TTFT is **57.57 microseconds**, and the three
decode launches plus their preceding gaps take **86.27 microseconds**: **34,774.55 tokens/s**
conditional on 200 MHz. These numbers describe only this tiny workload and testbench launch
contract; they exclude loading, sampling, host scheduling and communication. They are not
Gemma/Bonsai throughput or measured FPGA performance.

## Real model dimensions and separate numerical verdicts

Local Gemma 3 1B metadata specifies hidden size 1152, FFN 6912, 26 layers, four query heads,
one KV head and head dimension 256. Local Ternary Bonsai 2 2B metadata specifies hidden size
2048, FFN 6144, 24 layers, and 128-by-128 linear-attention state per head. No downloads or
credential access were used. The baseline compiler emits every measured program.

Both D=32 and D=128 use MCOLS=4, LANES=8, replicated/count-arbitrated TMEM with RPB=32,
WPB=1 and ordinary footprint issue. Each geometry has 14 measured operators: five projection
shards, four-token prefill Q shard, random int4 PAIR, two RMSNorm widths, full-width GELU and
SiLU, one local pretrained Gemma Q projection, synthetic ternary PAIR, and a full 128-by-128
DeltaNet state update/readout implemented with VPU RDOT/OUTER. Shards retain the complete K
dimension but use 64 output columns. The local pretrained projection uses all 1024 output
columns and synthetic activations; it reads only one BF16 tensor from an existing checkpoint.
It does not validate complete pretrained inference. Bonsai GGUF weights are not loaded.

All 28 timing comparisons pass. All ISA/RTL results agree bitwise. **26/28 independent
floating-point checks pass.** The random int4 PAIR case fails the unchanged 5% relative-L2
gate at D=32 (5.77%) and D=128 (7.95%); those negative results and nonzero exit status remain
part of the evidence. Synthetic ternary PAIR passes without assuming zero/sparsity speedups.
This numerical distinction is not hidden by an exact timing result.

Two additional complete D=128 projections pass the strict timing and independent numeric gates:

| Complete compiled operator | M × K × N | RTL/model cycles | Relative L2 |
|---|---|---:|---:|
| Gemma MLP up | 1 × 1152 × 6912 | 136314 | 0.898% |
| Bonsai MLP down | 1 × 6144 × 2048 | 214732 | 0.980% |

Long-K shapes exposed two simulator bugs. MXU FIFO occupancy is now reserved and released
per accepted/consumed chunk, across active commands, including int4 replay. DMA stores retain
a chunk until memory acceptance, and freeze the read pipeline when the finite output FIFO
fills. The latter previously gave an exact aggregate cycle count with an incorrect port
schedule. Strict port gates caught it. Original RTL, compiler, numerical gates and measurement
counters were not changed.

## Measured host execution cost

The following serial runs use the actual pinned Ramulator library, trace recording disabled,
Python 3.14.5/NumPy 2.5.4 on macOS ARM64. Wall time covers simulator construction and execution,
including ISA arithmetic; it excludes compilation, independent NumPy reference and JSON output.
Each entry is one measurement, not a confidence interval. Peak RSS is the process high-water
mark across the successive cases, not isolated per-operator allocation.

| Complete synthetic operator | D=32 cycles / host seconds | D=128 cycles / host seconds |
|---|---:|---:|
| Gemma Q | 40389 / 1.55 | 20366 / 0.84 |
| Gemma MLP up | 271183 / 10.30 | 136314 / 5.54 |
| Gemma MLP down | 272300 / 9.94 | 136172 / 5.31 |
| Bonsai MLP up | 428147 / 15.73 | 214965 / 8.40 |
| Bonsai MLP down | 428647 / 16.74 | 214732 / 8.27 |

Measured throughput is 24,161–27,388 simulated core cycles per host second. Final process peak
RSS is 720 MB (D=32) and 609 MB (D=128). All ten independent operator numerical checks pass.
Only the two complete projections identified above have additional full-operator RTL evidence
in this benchmark set; the benchmark tool deliberately labels its cycle outputs as predictions.

The long projections process roughly 1.4–1.5 million dense weights per host second at D=128.
A linear **host budgeting estimate**, applying that rate to approximately one billion Gemma
projection weights, is around 11–12 minutes per token for projection work alone. This is not
a full-model execution, latency prediction or token/s result; it excludes attention, additional
operators, compiler/image construction and memory scaling. Full-model peak memory was not
measured. Longer sequences and large vocabulary heads remain untested.

## Reproduce

Use the pinned dependencies and build procedure in [README](README.md). Output directories
must be new. Set `BRIDGE` to the built `.dylib` or `.so`; `GEMMA_CONFIG`, `BONSAI_CONFIG` and
optionally `GEMMA_WEIGHTS` refer to already local files. The retained catalogs contain the
configuration metadata and tensor hash used here. No network access is needed after the
pinned dependency source has been populated.

```sh
python tools/perfsim/examples.py --steps 5 --out build/perfsim/persistent-bundles
python tools/perfsim/build_rtl_ramulator.py \
  --bundle build/perfsim/persistent-bundles/compute --persistent --out /tmp/rtl-persistent
python tools/perfsim/compare_persistent.py --bundles build/perfsim/persistent-bundles \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --executable /tmp/rtl-persistent/obj/Vtb_perfsim --out /tmp/check-persistent

python tools/perfsim/model_shapes.py --gemma-config "$GEMMA_CONFIG" \
  --bonsai-config "$BONSAI_CONFIG" --gemma-weights "$GEMMA_WEIGHTS" \
  --depth 128 --out /tmp/shapes-d128
python tools/perfsim/build_rtl_ramulator.py --bundle /tmp/shapes-d128/gemma_q \
  --out /tmp/rtl-shapes-d128
python tools/perfsim/compare_native.py --bundle /tmp/shapes-d128/bonsai_deltanet_head128 \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --executable /tmp/rtl-shapes-d128/obj/Vtb_perfsim --out /tmp/check-delta
python tools/perfsim/model_shapes.py --gemma-config "$GEMMA_CONFIG" \
  --bonsai-config "$BONSAI_CONFIG" --depth 128 --full --out /tmp/full-shapes-d128
python tools/perfsim/benchmark_shapes.py --bundles /tmp/full-shapes-d128 \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json --bridge "$BRIDGE" \
  --out /tmp/host-d128
python tools/perfsim/audit_baseline.py --bundles /tmp/shapes-d128 \
  build/perfsim/persistent-bundles --out /tmp/baseline-audit.json
OTPU_RAMULATOR_BRIDGE="$BRIDGE" python -m pytest tests/perfsim -q
```

Repeat the operator generation/build/comparisons with `--depth 32`. Iterate `catalog.json`
for all operator cases. Random int4 cases intentionally return exit 1 on numerical failure;
do not disable that gate. Swap to the supplied DDR4/stress profiles for persistent sequences;
the stress receipt uses `--host-gap 256`. The audit command requires Python >=3.11 (`tomllib`).

## Remaining production and full-model limits

The pinned repository's actual `targets/ypcb-00338.toml` enables **SEQ_WC, DSTEP, STREAM,
TMEM_IMPL=3 with replay/guest/pumped dual-port timing, MXU_E=4, and DDR3/LiteDRAM channel
mapping**. Those are real production paths, not hypothetical extensions. They are not covered
by this ordinary-scoreboard/replicated-TMEM/shared-DDR4-or-HBM2 simulator. Unsupported opcodes
and configuration fields remain rejected. The audit exports the production target and its
hash alongside the implemented program inventory; S=1 fallback coverage is not board coverage.

The pinned baseline has no Gemma 3 or Bonsai GGUF frontend module. Newer local exports use
incompatible wait-count/streaming/T2 configurations and are not silently imported. A complete
pretrained run still needs compatible frontend integration (including correct Gemma attention,
Bonsai rotated/packed weights and recurrence), large-program/head validation, persistent KV
coverage at representative lengths, and the production microarchitecture/memory-shell paths.
No full Gemma/Bonsai TTFT or TPS is reported. Hardware Fmax, resources, power and fit still
require physical implementation. No synthesis, push, PR publication or paid resources occurred.
