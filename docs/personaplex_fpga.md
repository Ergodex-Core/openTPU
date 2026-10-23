# PersonaPlex on burst200

Goal: full-duplex speech conversation with the PersonaPlex temporal transformer
running on the existing openTPU FPGA, served through a localhost SSM tunnel.

## Current implementation and validation boundary

The port accepts `aufklarer/PersonaPlex-7B-MLX-4bit` directly on Linux. The MLX
affine INT4 values are unpacked and requantized into openTPU's integer matrix
format. This is an additional quantization step, not byte-compatible MLX execution.
The actual checkpoint's config and tensor shapes are checked before conversion.

All 32 temporal transformer layers, their attention and INT8 KV updates, and the
final RMSNorm execute through openTPU kernels. Q/K projection rows are permuted
within each head to preserve PersonaPlex's adjacent-pair RoPE convention. Norms,
softmax and residual activations use the existing FP32 kernels. The engine accepts
summed text/audio embeddings and returns normalized hidden activations.

Mimi, embedding gathering, the text head, sampling, and the depth transformer
remain CPU components in this first implementation. NVIDIA's LMGen supplies
voice conditioning, stream delays, and token scheduling. The bridge removes its
original temporal transformer and has no host fallback for FPGA execution.
The depth transformer is substantial; it is not the approximately 50 MB component
shown in the model card's obsolete size table.

Tests exercise MLX affine unpacking and shape checks, a small multi-frame
transformer against an independent numerical reference using the actual openTPU
ISA simulator, the real BoardBackend API over simulated transport, attention
bucket boundaries, reset isolation, and the host/server bridge contract. Header
validation runs without reading tensor payloads. Offline startup tests ensure the
upstream server cannot implicitly download another LM. The production-size
temporal program compiles for the recorded burst200 configuration. These checks
do **not** establish production checkpoint quality, full upstream Moshi
compatibility, FPGA execution, or real-time speech.

The installed instance validation suite passed on 2026-10-21: **128 passed in
22.99 seconds, no skips**, using the F2 instance's CPU and Verilator 5.048
(SSM `da545afa-2e55-47c5-b642-6a21ecd75ddd`). This includes the genuine NVIDIA
Moshi integration test. The fixtures do not use FPGA hardware or production
checkpoint payloads. Cap1024 programs use 8,624–17,072 instruction words.

Run PersonaPlex validation in the installed environment:

```bash
source /home/ubuntu/personaplex-prep-20261021/venv/bin/activate
export PATH=/home/ubuntu/opt/verilator-5.048/bin:$PATH
cd /home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source
NO_TORCH_COMPILE=1 python -m pytest -q --tb=short tests/test_personaplex_*.py
```

Source and dependencies are installed on instance `i-0c78de2253f66519d` in
account `438374459156`, region `us-east-1`, under
`/home/ubuntu/personaplex-prep-20261021`. The source directory is
`personaplex-fpga-source/`; its editable installation followed verification of all
228 manifest hashes from the base archive and readiness overlay
(SSM `d98a40f8-eaf2-468c-9158-3c499448f36d`). A subsequent two-file reader/test fix
for U32 tensors with safetensors 0.4.5 was deployed separately
(SSM `baae7793-d2c2-48ae-b3dc-8b0cf143cbaf`). The historical source archives remain
unchanged and do not contain that later fix.

The initial source download returned HTTP 403. After explicit user approval,
policy `PersonaPlexSourceRead20261021c5bbd5e7` was applied to role
`opentpu-f2-ssm`, granting `s3:GetObject` on exactly the seven source chunks and
one readiness overlay. Source download and installation then completed.

The pinned MLX checkpoint downloaded successfully into the isolated `model/`
directory at revision `8c686347007de51d9017706e5543d6f1757ecf5d`.
The original NVIDIA audio assets initially returned HTTP 401, then downloaded
successfully from `nvidia/personaplex-7b-v1` revision
`fdaf4090a61cb315c138a1faee287ffd6c716309`
(SSM `acd356c3-f351-465e-816e-a08b5ea4dfba`). The `assets/` directory contains
the original codec/config, 18 extracted voice prompts, and the extracted web UI.
Offline preflight against the installed real checkpoint and assets, plus
`pip check`, passed (SSM `8f98eab3-59e7-469c-86f7-6274dbcf57fe`). This validates
headers, runtime schema, compiled capacity and asset structure.

Real CPU host payload checks also passed (SSM
`6ca25950-4308-4aaf-b6cc-51cb28a1e39e`): strict assignment of all 282 tensors as
FP32, native initial summed embeddings, SentencePiece, all 18 voice payloads,
and original Mimi streaming encode/decode of one 1,920-sample frame at 24 kHz.
The captured initial embedding is `[1, 4096]`, repeatable bit-for-bit. These checks
do not execute the temporal transformer, depth transformer or text head and do
not assess speech quality. The receipt is
`tools/data/personaplex/host-assets-validation.json`; the reusable tool is
`tools/personaplex_host_check.py`.

A full 32-layer CPU temporal reference completed for that one initial embedding
(SSM `88ec0231-42d5-4739-b75d-11fd2259fe1e`). It uses dequantized MLX affine INT4
weights and independent FP64 transformer arithmetic, starting at position zero
with empty KV. The finite `[1, 4096]` output is stored on the instance in
`temporal-reference-001/reference_hidden.npy`. No FPGA rounding, temporal quality
or hardware timing is validated by this reference. The reproducible tool is
`tools/personaplex_reference.py`; inputs must be 1–4 consecutive initial frames.

For a reproducible multi-frame input, `tools/personaplex_capture_voice.py`
extracts the first 1–4 stored embeddings from an existing local voice `.pt`
using `torch.load(weights_only=True, map_location="cpu")`. This follows the
upstream voice-prompt replay order. It saves FP32 `[frames,4096]` data and an
`input.json` sidecar with the source voice hash, shape/dtype, frame indices,
output hash, and position-zero/empty-KV contract. A new output directory is
required; non-finite values or FP32 overflow fail before output creation.
It does not load the LM, Mimi or another checkpoint.

Run the following on the existing EC2 environment from the source checkout,
after checking current jobs and at least 40 GiB of free memory and disk. Use
new output directories under an isolated validation root. Keep the process at
nice 19 and pinned to one available CPU (including its build workers), so it
does not compete with higher-priority wide-port work. These commands do not
access FPGA hardware or require the device reservation to be released:

```bash
nice -n 19 taskset -c "${PERSONAPLEX_CPU:?select an idle permitted CPU}" \
  python tools/personaplex_capture_voice.py --voice-file /path/to/voices/NATF2.pt \
  --frames 4 --output-dir /path/to/validation/capture
nice -n 19 taskset -c "$PERSONAPLEX_CPU" \
  python tools/personaplex_reference.py --model-dir /path/to/existing/model \
  --embeddings /path/to/validation/capture/input_embeddings.npy \
  --output-dir /path/to/validation/reference
nice -n 19 taskset -c "$PERSONAPLEX_CPU" \
  python tools/personaplex_isa_check.py --model-dir /path/to/existing/model \
  --embeddings /path/to/validation/capture/input_embeddings.npy \
  --reference-dir /path/to/validation/reference \
  --output-dir /path/to/validation/isa \
  --cache-dir /path/to/validation/quant-cache --cap 1024
```

Before retrying an interrupted run, inspect its receipt, progress log and live
processes; do not overwrite results or start duplicate jobs. The ISA checker
binds its input/output arrays to the completed reference by SHA-256, converts
all 32 temporal layers to openTPU INT4 weights with INT8 activations/KV, reports
aggregate and per-frame precision differences, and requires bit-exact output
after resetting and replaying the same frames. Numerical differences are
measurements, not an inferred quality acceptance threshold. Its CPU wall time
is not FPGA milliseconds/frame or end-to-end audio latency. Voice conditioning,
text/audio token generation, the depth transformer and Mimi still need separate
streaming integration and quality validation.

All actual checkpoint headers were read without tensor payloads and validated
locally against the installed NVIDIA model's genuine meta-device schema:
**all 32 temporal layers and all 282 host tensors passed**. Temporal data has
128 INT4/group64 matrices and 65 F16 norms; the host has 220 INT4 matrices and
62 F16 tensors. This snapshot needs zero copied tensors and zero attention
expansions. Expanded FP32 host parameters occupy 7,177,895,936 bytes. These are
metadata checks only, not numerical or speech-quality validation. The receipt is
`tools/data/personaplex/actual-header-validation.json`; the reproducible local
metadata-only validator is `build/personaplex/validate_actual_headers.py`.

An isolated public dependency installation completed successfully under
`/home/ubuntu/personaplex-prep-20261021` (SSM command
`ef1d43e3-b3ab-4282-85e3-74d4a7aae1b1`). `pip check` passed. The environment
contains NVIDIA revision `3428dfd95309a7f3c84fd93259ded0f810d1ff91`,
moshi-personaplex 0.1.0, sphn 0.1.12, NumPy 2.1.3, safetensors 0.4.5,
huggingface-hub 0.24.7, aiohttp 3.10.11 and sentencepiece 0.2.0.
This dependency preparation does not establish checkpoint or port compatibility.
Native imports of `LMModel`, `sphn`, PyTorch 2.4.1+cpu, aiohttp and sentencepiece
passed (SSM command `a458b1a7-b858-4f91-8d1c-9c80e23500ed`).
The shared card is reserved for the Nanbeige workload until explicitly released.
No PersonaPlex image load, device access, HBM write, hardware inference,
conversation service, or localhost tunnel has been performed. The 145.42 ms
estimate below remains a simulation extrapolation, not a hardware receipt.

The installed CLI qualifies the recorded burst200 configuration, 200 MHz clock,
reported memory geometry, runtime-argument/INT4 capabilities, calibration, idle/drained state,
and error status before backend construction. The probe disables calibration and
quiescing and is repeated immediately before device-memory writes. **47 focused
CLI tests passed**. The guard was delivered in the readiness overlay. It cannot
authenticate the AFI or establish ownership: legacy and integrated runtimes use
different lock names,
so explicit confirmation of Nanbeige's release remains mandatory.

The historical base archive SHA-256 is
`e3310ac387b87961ce8337420df9eac8e39779ad63bfc96568cc56de3b12d016`.
Its seven chunks, `part00.b64` through `part06.b64`, are under
`opentpu-personaplex/20261021/e3310ac3/` in
`ergodex-bonsai-on-chip-438374459156-us-east-1`.
The readiness source bundle is `build/personaplex/personaplex-fpga-readiness-source.tar.gz`
(SHA-256 `c5bbd5e73dbb4192bb30ac44d1a17fed05709aa32a3ed12cea3dd3245b587a58`).
Its overlay was downloaded from `opentpu-personaplex/20261021/c5bbd5e7/readiness-overlay.tar.gz.b64`
in the same approved bucket. Its decoded SHA-256 is
`fb679653e62fbd0c6a2146cb651e68c3c89199650d54c527a57da2124417ffc6`.
The original seven chunks plus this overlay reproduce the 228 manifest hashes
verified before the later U32 compatibility fix. The updated local snapshot is
`build/personaplex/personaplex-fpga-installed-source-20261021.tar.gz`, including
the compatibility fix, CPU validation tools and current receipts. Its hash and
file count are recorded in `build/personaplex/installed-source-package.json`.

## Capacity and latency constraints

Recorded burst200 config: 200 MHz, S=1, D=128, MCOLS=4, 4 GiB logical HBM,
32,768 instruction words. The profile is copied from `alex-burst200/sw/f2.toml`
to `tools/data/personaplex/burst200.toml`. Hardware runs use the device's probed
config rather than assuming the saved profile is the loaded image.

| Requested context | Image + program reservation | Fits 4 GiB |
|---|---:|---|
| 256 frames / 20.48 s | 3,586,068,480 bytes | yes |
| 512 frames / 40.96 s | 3,655,274,496 bytes | yes |
| 1,024 frames / 81.92 s | 3,793,690,624 bytes | yes |
| 3,000 frames / 240 s | 4,347,346,944 bytes | no |

Context includes voice and text prompt frames. The initial engine is bounded:
it raises at the limit and requires a new session; it does not silently discard
context or claim a sliding-window implementation. The default is 1,024 frames.

The temporal weight layout contains 3,516,416,000 bytes including scales/norms.
The packed matrix values alone occupy 3,288,334,336 bytes. At burst200's theoretical
25.6 GB/s weight-port maximum, reading just those values takes **128.45 ms**.
Each speech frame represents **80 ms**. This conservative lower bound excludes
scales/norms, KV traffic, activation arithmetic, host communication, codec and depth
transformer, so the unchanged burst200 image cannot sustain real-time operation
for this mapping. Do not infer frame rate from CPU/ISA simulation wall time.

The follow-on performance work is the wider weight port and larger exposed HBM
path, plus depth-transformer/codec acceleration and a measured streaming budget.
A wider port by itself does not prove end-to-end real-time performance.

## Verilator gate before FPGA execution

`tools/personaplex_verilator.py` runs the temporal kernels on the actual RTL with
the recorded burst200 compute geometry and microarchitecture. Only simulated DRAM
capacity is reduced to the fixture's needs. It compares all persistent DRAM bytes
with the ISA simulator and compares hidden outputs with an independent FP64
transformer reference. A mismatch raises and prevents a successful report.

```bash
source /home/ubuntu/personaplex-prep-20261021/venv/bin/activate
export PATH=/home/ubuntu/opt/verilator-5.048/bin:$PATH
cd /home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source
python tools/personaplex_verilator.py --mode tiny --output tiny.json
python tools/personaplex_verilator.py --mode layer --frames 1 \
  --bw 100 --lat 60 --stall 0 --output layer.json
python -m pytest -q tests/test_f2_rtl.py \
  -k 'native_protocol or memory_ordering or ocl_crossing'
```

On the F2 instance, the default Verilator 5.020 failed the tiny RTL test.
Selecting the installed Verilator 5.048 with the explicit `PATH` above passed
the tiny correctness/reset check in 138.65 seconds of host wall time. The later
complete PersonaPlex suite passed with that compiler selected. These CPU
simulation runs do not access the FPGA.

The tiny case uses four frames and a reset check across two layers, with 256-wide
hidden states, two 128-wide attention heads and FFN width 512. The layer case uses the production
hidden width 4096, 32 heads and FFN width 11264, but only **one layer with synthetic
nonzero weights**. Neither is a checkpoint-quality test. Passing these fixtures
does not replace real-checkpoint reference checks before full-checkpoint FPGA runs.

Reports separate simulated RTL cycles from simulator wall time, convert cycles
at 200 MHz, and record the effective RTL parameters and memory settings. The
memory model is a synthetic two-channel native interface, with configurable beat
service, request latency and stalls. It is **not** the AMD HBM controller, AWS
shell, PCIe host path, or a DRAM timing model such as Ramulator. The separate F2
bridge tests exercise the checked-in F2 bridge logic under stalls and clock
crossings; they do not turn the core's memory model into HBM simulation.

The production-shape command above specifies a 60-cycle/300-ns minimum request
latency, peak beat service and no random stalls. Any full-model estimate obtained
by multiplying a one-layer result must be labeled as extrapolated. It excludes
host audio, depth-transformer and codec execution.

Measured results on 2026-10-20, using Verilator 5.048:

| Case | Simulated cycles | Kernel time at 200 MHz | RTL versus ISA |
|---|---:|---:|---|
| Two small layers, each of four frames and reset | 12,738 per run | 0.06369 ms | All 1,036,800 model-memory bytes identical |
| One production-shape layer, first frame | 908,897 | 4.544485 ms | All 112,347,648 model-memory bytes identical |

The unquantized FP64-reference relative errors were 4.11% and 3.45%, respectively;
these characterize the synthetic INT4 fixtures, not production speech quality.
The production-shape run took 129.33 seconds of host wall time including fixture
creation, reference computation, quantization, first RTL build and simulation.
Host wall time is not FPGA inference latency. Seven separate local F2
native-protocol, memory-ordering and control-crossing tests also passed.

Multiplying the one-layer kernel time by 32 gives an **extrapolated 145.42 ms per
first-frame temporal step**, approximately 6.88 steps/second versus the required
12.5. This is not a measured full-model result: it repeats the final normalization
and omits depth, codec, host transfers and the actual HBM subsystem. Together with
the packed-weight bandwidth floor above, it confirms that the current mapping is
not ready to meet the real-time frame budget.

Full run metadata, effective RTL parameters, memory assumptions and test outputs
are recorded in `tools/data/personaplex/verilator-validation.json`.

## Earlier local preparation

Initial workstation tests used `../pi05-venv` and a partial
`../personaplex-venv` built from cached packages. Missing native audio dependencies
and incompatible cached versions prevented complete local Moshi validation.
The installed F2 environment now supplies the pinned NVIDIA dependencies;
its imports, `pip check`, and genuine Moshi fixture tests passed. Use that
environment and the paths below for deployment.

## Entry points

No model files are needed to inspect the compiled layout:

```bash
source /home/ubuntu/personaplex-prep-20261021/venv/bin/activate
cd /home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source
python tools/personaplex_fpga.py inspect
python tools/personaplex_fpga.py inspect --cap 3000
```

The second command reports `fits: false` and exits 2. The dry report never opens
the board or allocates an oversized model image.

Validate the installed runtime and local checkpoint before any device access:

```bash
python tools/personaplex_fpga.py preflight \
  --model-dir /home/ubuntu/personaplex-prep-20261021/model \
  --assets-dir /home/ubuntu/personaplex-prep-20261021/assets --cap 1024
```

Preflight checks required modules, checkpoint headers, all compiled attention
buckets, profile memory capacity, the installed NVIDIA model's exact host tensor
schema, and local service assets. It reports host FP32 parameter memory and the
exact allowed 8-to-16-step depth expansion. It does not unpack large weights,
load the codec, open the FPGA, or validate model quality. `serve` runs the same
check automatically before opening the device. Failed checks exit 2.

For a hardware numerical/latency check, prepare a `.npy` file of summed embedding
vectors captured from the same checkpoint and compare the output with its CPU
reference. Synthetic embeddings only test the temporal execution interface.
This hardware command has not run; it requires explicit confirmation of
Nanbeige's release and verification of the loaded FPGA image:

```bash
python tools/personaplex_fpga.py benchmark \
  --model-dir /home/ubuntu/personaplex-prep-20261021/model \
  --backend board --dev f2:0 --cap 1024 \
  --embeddings /home/ubuntu/personaplex-prep-20261021/host-validation-001/input_embeddings.npy \
  --output fpga_hidden.npy
```

Compare the resulting hidden vector against
`/home/ubuntu/personaplex-prep-20261021/temporal-reference-001/reference_hidden.npy`,
reporting the error introduced by the additional FPGA quantization. This is one
initial frame; it does not establish multi-frame checkpoint accuracy or speech quality.

The command validates finite float32-compatible embeddings and context limits
before device access, then probes the loaded image and validates memory capacity before unpacking
the checkpoint, executes frames, and reports temporal timings. It never loads an
AFI. Board access uses the integrated runtime's exclusive device lock. The legacy
runtime uses a different lock name, so this lock does not establish that the
other workload has released the card.

## Instance setup and conversation service

Provisioning, dependency installation, source installation, checkpoint/assets
downloads and offline preflight are complete. The existing isolated runtime is
under `/home/ubuntu/personaplex-prep-20261021`:

| Directory | Installed content |
|---|---|
| `venv/` | Validated Python environment, including NVIDIA's PersonaPlex Moshi |
| `personaplex-fpga-source/` | Editable openTPU source, readiness overlay and U32 compatibility fix |
| `upstream/` | NVIDIA source pinned to `3428dfd95309a7f3c84fd93259ded0f810d1ff91` |
| `model/` | Requested MLX checkpoint and SentencePiece tokenizer |
| `assets/` | Original Mimi/config, 18 voice prompts and browser UI |

Select the existing environment; no reinstall or model download is needed:

```bash
source /home/ubuntu/personaplex-prep-20261021/venv/bin/activate
cd /home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source
python -m pip check
```

NVIDIA's PersonaPlex version of Moshi is required; the similarly named stock
package is not interchangeable. Native audio imports passed in this environment.
The full original LM checkpoint is **not** downloaded by this wrapper: its
`--moshi-weight` is set to the requested MLX directory and the loader is replaced
with the strict host-only loader. Mimi's MLX file is not used by the CPU codec.
Record the upstream Git revision, model snapshot revision and environment freeze
with any hardware result.

The installed offline service bundle uses the original downloaded config and
codec, extracted upstream `voices.tgz` and `dist.tgz` contents, and the MLX
directory's SentencePiece tokenizer. Assets are separate from the source bundle.
Preflight validates structure; upstream loading still validates the codec, voice
files, tokenizer, and browser contents.

| Local path relative to the assets directory | Required content |
|---|---|
| `config.json` | Original upstream JSON config |
| `tokenizer-e351c8d8-checkpoint125.safetensors` | Original PyTorch Mimi codec |
| `voices/*.pt` | Named upstream voice prompts; choose an available voice in the UI |
| `dist/index.html` and companion files | Extracted upstream browser build |

`tokenizer_spm_32k_3.model` stays in the MLX model directory. The service resolves
the upstream config-counter request locally and rejects other Hub requests when
`--assets-dir` is set. It supplies every asset path explicitly to the upstream
server. Without this flag, upstream's normal online asset retrieval remains enabled.

```bash
source /home/ubuntu/personaplex-prep-20261021/venv/bin/activate
cd /home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source
python tools/personaplex_fpga.py preflight \
  --model-dir /home/ubuntu/personaplex-prep-20261021/model \
  --assets-dir /home/ubuntu/personaplex-prep-20261021/assets --cap 1024
```

The conversation service has not started. After ownership release, image
verification and the remaining hardware correctness checks, launch it with:

```bash
python tools/personaplex_fpga.py serve \
  --model-dir /home/ubuntu/personaplex-prep-20261021/model \
  --assets-dir /home/ubuntu/personaplex-prep-20261021/assets \
  --backend board --dev f2:0 --cap 1024 --port 8998 --threads 8
```

The listener is explicitly bound to `127.0.0.1`, including around upstream's
`web.run_app` call. `/api/fpga-status` reports which component runs on the FPGA,
temporal frame latency, and the fact that end-to-end real-time performance has
not been validated. `serve` refuses the ISA backend. Warmup and each new
conversation reset the FPGA temporal state through Moshi's streaming lifecycle.

No tunnel has been started. After the server is actually listening, run on the
local workstation:

```bash
bash tools/personaplex_tunnel.sh i-0c78de2253f66519d 8998 8998
```

This opens AWS SSM port forwarding to `http://localhost:8998`. Keep the tunnel
process open. There is no security-group change or public listener. Browser
microphone access is through localhost.

## Source bundle

Build a deterministic runtime source archive from this checkout:

```bash
python tools/package_personaplex.py --output build/personaplex-fpga-source.tar.gz
```

The archive contains the openTPU Python runtime and support files, RTL and Verilator
testbenches, device profiles, PersonaPlex source and tests, tunnel launcher, and this guide. It includes a
`SOURCE_MANIFEST.json` recording every source file's size/SHA-256, the base Git
commit, and the requested model revision. Model weights, credentials, Git metadata,
FPGA bitstreams and full synthesis flows are excluded. The archive supports RTL
simulation and software deployment against an already loaded compatible FPGA image.

The current instance source is already installed at
`/home/ubuntu/personaplex-prep-20261021/personaplex-fpga-source`. A fresh bundle
records the current checkout, including fixes deployed after the historical
base/overlay installation; those immutable archives are not the latest source.
The packaging command refuses to overwrite an existing output. Repackaging an
extracted bundle preserves its recorded base commit and hashes its current sources.

## Source references

- Requested checkpoint: https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit
- Native orchestration: https://github.com/NVIDIA/personaplex
- MLX conversion and key layout: https://github.com/soniqo/speech-swift
- Existing burst200 bandwidth measurements: `docs/wide_port.md`

Remaining acceptance gates are production temporal checkpoint conversion to
FPGA precision, full CPU/Moshi streaming integration, captured-hidden accuracy
on the actual FPGA, voice/text response
quality, measured frame timing, context handling, and a working localhost session.
