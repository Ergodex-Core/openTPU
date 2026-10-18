# OpenTPU standalone performance simulator

**Experimental structural timing model, not a cycle-exact replacement for RTL.** It executes
OpenTPU's eight-word instructions and models their resource/dependency interactions. All
predictions are labelled `opentpu-perfsim-estimate`; the simple backend is explicitly `simple`.
There is no fallback from a missing/broken Ramulator installation. Existing RTL, compiler,
ISA golden model, FPGA workloads, `tools/hwloop/lpn` and its validation registry are unchanged.

This implementation is a starting point toward the requested fully validated simulator.
It does **not** establish pretrained full-model timing accuracy, hardware Fmax or physical fit.
See [architecture and limitations](architecture.md) and [current validation](validation-round3.md) and [round-two history](validation-round2.md) and [preserved initial evidence](validation.md).

## Quick start

From the repository root, with Python >=3.10 and NumPy (the tested versions are in
`tools/perfsim/dependencies.lock.json`):

```sh
python tools/perfsim/examples.py --out build/perfsim/bundles
python -m opentpu.perfsim --bundle build/perfsim/bundles/gemm \
  --profile tools/perfsim/configs/simple.json --out build/perfsim/simple-run --numeric-check
python -m pytest tests/perfsim -q
```

This compiles a partial GEMM and a **tiny random two-layer Qwen3 topology** with the real
OpenTPU compiler, all attention/norm/MLP/head paths included. No checkpoint is downloaded.
The tiny sequence is two serial prompt-token steps and one teacher-forced decode step, not
a pretrained billion-parameter model and not a benchmark of a batched serving system.

Build the actual Ramulator backend (CMake >=3.14, C++20; public source downloads require
network access, no system installation):

```sh
python tools/perfsim/build_ramulator.py --source build/perfsim/ramulator2 --fetch --jobs 4
python -m opentpu.perfsim --bundle build/perfsim/bundles/gemm \
  --profile tools/perfsim/configs/hbm2-f2-200mhz.json \
  --bridge build/perfsim/bridge/libopentpu_ramulator.dylib \
  --out build/perfsim/hbm-run --numeric-check --trace
OTPU_RAMULATOR_BRIDGE="$PWD/build/perfsim/bridge/libopentpu_ramulator.dylib" \
  python -m pytest tests/perfsim -q
```

Use `.so` instead of `.dylib` on Linux. `--bridge` overrides the example profile's macOS path.
Omit `--fetch` for an offline build after populating all exact commits in the lock file.
The build script refuses to discard edits in a checkout at the wrong revision. Use a new,
disposable source directory. On macOS it builds for the architecture of the running Python.

Run the full bounded sequence with persistent KV and Ramulator controller/row state:

```sh
python tools/perfsim/sequence.py --bundles build/perfsim/bundles \
  --profile tools/perfsim/configs/hbm2-f2-200mhz.json --out build/perfsim/hbm-sequence
python tools/perfsim/sequence.py --bundles build/perfsim/bundles \
  --profile tools/perfsim/configs/hbm2-asic-1ghz.json --out build/perfsim/asic-sequence
python tools/perfsim/validate.py --bundle build/perfsim/bundles/compute \
  --out build/perfsim/compute-rtl --rtl
```

Output directories must be empty: runs never overwrite prior evidence. `metrics.json` reports
core cycles, physical and logical memory bytes, transactions, stalls, unit busy fractions,
MXU advances/useful MAC fraction, per-instruction and labelled-region spans. These spans can
overlap and must not be added as wall time. `--trace` enables request-accept/completion JSONL;
it is off by default so a long run does not retain every memory event. `manifest.json` records
input, source, dependency and backend-library hashes. CLI failures write `failure.json`.

## Inputs and measurement boundaries

`input.json`, `prog_0.hex` and `dram_0.bin` use the LPN input-bundle convention: eight 32-bit
instruction words, one hex word per line, DRAM file in big-endian 32-bit words converted back
to byte memory on load. Trailing zero program words may be omitted only when the manifest
supplies the complete instruction count. Register arguments initialise R8..R15. Unknown
architecture fields (including wide-model-only `PORTB`, `ROWLANES` or `T2`) are rejected.
The implementation targets this branch's actual RTL/Config, not an invented wide-port target.

Optional `labels` maps static PCs to layer/region names. Without it, compiler comments name
regions. A label containing `{iteration}` expands to the innermost LOOP iteration (zero based); every command also records its loop context. The bounded Qwen example labels its two decoder layers separately from the final norm/head. Region spans can overlap because the sequencer overlaps layers; they are not additive wall time.

The boundary starts with the program resident and the core beginning the run; it ends after
HALT, outstanding commands and writes have drained. It excludes loading weights/programs,
PCIe/XDMA, host scheduling, sampling and network time. Standalone runs report TTFT/prefill/
decode/TPS as null because those concepts require workload boundaries. `sequence.py` declares
them explicitly, maintains KV across steps and reports **device-only conditional** decode TPS
at the stated clock. Changing core clock does not rescale HBM's time constants.

The 200 MHz profile is the user's requested F2 clock, **not verified achieved Fmax**; it is not
an F2 shell model. The 1 GHz profile is a hypothetical core-frequency projection with the same
memory and on-chip architecture. FPGA/ASIC area, routing, power and timing closure are outside
this simulator and need physical implementation.

## Research and licensing

The [SCALE-Sim v3 paper](https://arxiv.org/abs/2504.15377v2) and
[official implementation](https://github.com/scalesim-project/SCALE-Sim/tree/9f98c4371055a54c75209c2e02b640b897550532)
inform the modular compute/buffer/memory design and reporting. Its checked `README_ramulator.md`
describes demand-trace generation, Ramulator response replay and a second SCALE-Sim run. This
implementation instead makes accepted requests and completions feed back during execution;
it does not reuse SCALE-Sim's GEMM timing kernel, replay approximation or sparsity assumptions.
No SCALE-Sim source code is copied. Ternary/zero weights receive no timing speedup.

[Ramulator 2.0a](https://github.com/CMU-SAFARI/ramulator2/tree/be93be78055d922aa1d4d33e15bcc8f2b0c61a9d)
is an MIT-licensed external dependency, pinned with yaml-cpp/spdlog/argparse commits. The
small compatibility patch, build receipt and license are retained. See
`tools/perfsim/ramulator/NOTICE.md`. Simulator code follows this repository's Apache-2.0 license.

## Reproducing instruction-level timing comparisons

Use the unchanged RTL executable built by `tools/perfsim/validate.py --rtl`, or an existing `Vtb_top` with its `Vtb_top__verFiles.dat` build receipt. The corpus verifies the parameter set before running:

```sh
python tools/perfsim/timing_suite.py --executable /path/to/Vtb_top \
  --config-bundle build/perfsim/bundles/compute --out /tmp/opentpu-timing \
  --memory rtl_fixed --compiled
```

This writes raw RTL traces, actual input bundles, output hashes, simulator metrics, per-instruction comparisons and content-addressed manifests. Exit 2 means the exact trace/cycle gate failed; all 112 current fixed-memory cases pass. Exit 1 means a numeric or execution failure. The ±5% aggregate-cycle check is recorded separately and cannot override the exact gate. Golden RTL, its counters and measurement boundaries are unchanged. Frozen RTL observations in `tests/perfsim/fixtures` are replayed by ordinary pytest; they do not run Verilator and are not new RTL measurements.

On hosts with file-provider stalls in synced Documents folders, use a disposable **non-synced** directory for both the pinned Ramulator checkout and build, for example `--source /tmp/opentpu-ramulator --build /tmp/opentpu-bridge`. Upstream Ramulator places its library in its source tree, so moving only the build directory is insufficient. Populate all pinned dependencies (or explicitly use `--fetch`), then run the same build command. The round-two receipt records a successful offline build and repeat build on macOS ARM64, followed by tests against that exact library. No system installation is needed.

## Closed-loop RTL / Ramulator validation

The isolated harness derives a new top and testbench in its build directory. It preserves
original compute RTL, instruction traces, counter/reset/dump boundaries and golden files.
Only the external-memory boundary is replaced by a DPI adapter to the **same** Python
native-port controller and pinned Ramulator bridge used by the simulator. Each run generates
its own requests; no trace replay, arrival forcing or average-channel-rate calibration occurs.
These results validate the documented shared controller contract, not the FPGA AXI shell.

```sh
python tools/perfsim/build_rtl_ramulator.py --bundle build/perfsim/bundles/compute \
  --out /tmp/opentpu-rtl-native
python tools/perfsim/compare_native.py --bundle build/perfsim/bundles/tiny_qwen_0 \
  --profile tools/perfsim/configs/native-hbm2-200mhz.json \
  --bridge /path/to/libopentpu_ramulator.dylib \
  --executable /tmp/opentpu-rtl-native/obj/Vtb_perfsim --out /tmp/opentpu-native-check
```

`native-ddr4-200mhz.json` selects DDR4; `native-hbm2-stress-1ghz.json` limits outstanding
transactions to eight at a hypothetical 1 GHz core. `--micro-trace` records diagnostic unit
signals from the derived top. The strict gate checks every D/S/E event, total cycles, per-port
accepted-address/size/write sequence and acceptance/completion cycle, plus physical traffic.
Numerical checks have a separate verdict; available independent NumPy logits must pass too.
The harness explicitly rejects non-contiguous native masks and supports S=1, D<=128.

A standalone stage starts with a cold Ramulator controller. `sequence.py` instead preserves
controller and KV state across launches. Its persistent sequence timing remains a prediction
until compared with a correspondingly persistent RTL harness; summing cold-stage RTL times
is not a substitute. Use the native profiles explicitly for this registered-port contract.
