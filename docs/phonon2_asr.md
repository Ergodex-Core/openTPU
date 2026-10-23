# Phonon-2: FPGA-only ASR readiness

Status: **blocked; the complete learned subsampler and 24-layer encoder are
qualified in the shared-profile ISA simulator**. The latest
[encoder report](phonon2_encoder.md) binds the evidence to the shared Gemma
profile, with real-speech comparisons, masks/reset and an explicit program/weight
residency scheme. The previous [subsampler report](phonon2_subsampler.md) and
[Conformer/front-end report](phonon2_conformer.md)
contains the new golden vectors, policies, full-width block budgets and real-weight
receipt. The original projection receipt below retains its historical compiler binding.
There is no audio-to-transcript engine, FPGA transcript, latency result, or WER
result from this integration. No FPGA locks, image loads, registers, or running
inference jobs were touched. No synthesis was started.

## Pinned evidence

| Item | Revision / source |
|---|---|
| Model | `FermionResearch/Phonon-2`, `fb09063c9dc8e130d97b3a2191ed5aed6b71bb31` |
| Inherited NVIDIA configuration | `nvidia/parakeet-tdt-0.6b-v3`, `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |
| Transformers source inspected | `ea52711b4850c332be3f778396ce882ab84a9011` |
| Fermion runtime source inspected | `43b98dd54f13277715d9273b921792aba9aa94ce` |
| OpenTPU branch base | `model-infer-f2`, `e55992e4621f1930c57c9712dd96d5f1ec7c4e1c` |

The [official model config][model-config], [NVIDIA config][base-config],
[processor config][processor], [model graph][graph], [TDT generation][generation]
and [Fermion CUDA source][cuda] were read as text. Neither `trust_remote_code`
nor an upstream installer/model runtime was executed. This is **FastConformer
with a Token Duration Transducer**, not a Whisper adapter. A newer Transformers
source snapshot is not a claim of equivalence to the originally trained runtime.
Per-file URLs and SHA256 values are in [source provenance](phonon2/source_provenance.json).

The checkpoint was downloaded and inspected as data on the existing EC2 host.
Both archive and container hashes match the pinned config. No checkpoint weights
were downloaded to the Mac or added to Git. Static inventory is in
[checkpoint_inventory.json](phonon2/checkpoint_inventory.json).

## Architecture and data format

* **Parameters:** 627,008,134, excluding BatchNorm running means/variances and
  counters. The container has 627,057,310 total tensor elements in 723 records.
* **Frontend:** 16 kHz mono, preemphasis 0.97, 400-sample Hann window, 160-sample
  hop, 512-point STFT, 128 Slaney mel filters, natural log, and per-feature
  normalization over valid frames. This signal processing must run on device.
* **Learned subsampling:** three stride-2, kernel-3 Conv2D stages, 256 channels,
  padding 1, ReLU; the later two stages are depthwise plus pointwise convolutions.
  Time and mel frequency shrink 8x. Flattened `256*16=4096` features project to
  1024. Length masks apply after each convolution.
* **Encoder:** 24 noncausal FastConformer blocks, width 1024, 8 heads of 128,
  two 4096-wide SiLU feed-forward modules per block with half-weight residuals,
  five LayerNorms, relative-position attention with learned `u/v` biases and
  relative-key projection/shift, and a convolution module. That module has
  `1024 -> 2048` pointwise projection, GLU, same-padded kernel-9 depthwise
  convolution, inference BatchNorm, SiLU and `1024 -> 1024` projection. This is
  not causal LLM attention or RoPE.
* **Predictor:** token embedding `8193 x 640`; two LSTM layers, each with
  `2560 x 640` input/recurrent matrices and two 2560 biases; a `640 -> 640`
  projection. Its recurrent state must remain on device, with blank/nonblank
  state handling matching the selected reference.
* **Joint:** encoder `1024 -> 640`, addition with predictor output, ReLU,
  `640 -> 8198`. Select a token over 8193 classes and a duration separately over
  5 classes `[0,1,2,3,4]`. Blank is 8192; the base generation start is blank.
* **Tokenizer:** 8192 inherited BPE pieces; the inspected tokenizer JSON uses a
  Metaspace decoder. Blank is an additional model class. Preserve special-token
  filtering, Unicode bytes, punctuation, casing, leading-space handling and
  duration timestamps. No external language model is specified. The strict
  contract conservatively keeps token-to-text assembly on device too.

The [container reader][reader] specifies `fermion-five-value-parakeet-v1`:
an 8-byte little-endian JSON-header length, the JSON index, then record blobs.
The 163,515,201-byte `.bps.tar.zst` contains a 177,438,361-byte `model.fermion`
and three small JSON files. It is not safetensors, GGUF, or a Transformers
`from_pretrained` checkpoint. The archive's config names NeMo tokenizer assets
which are not present in the archive; the reference retrieves its processor and
tokenizer from the base repository.

| Record kind | Count | Elements | Encoding |
|---|---:|---:|---|
| five_value | 264 | 603,979,776 | Per-row base-3 sign codes, five trits/byte; high/low bit for each nonzero; FP16 low/high magnitudes per row |
| int6 | 35 | 22,655,360 | Four unsigned 6-bit codes per three bytes, minus 32; FP16 row scales |
| fp16 | 424 | 422,174 | Norms, biases, small convolutions and buffers |

Each encoder weight is `0`, `+/-lo[row]`, or `+/-hi[row]`. Arbitrary learned
`hi/lo` ratios cannot generally be represented exactly by one existing uniform
INT8/INT4 row scale. INT6 tables can be unpacked to signed INT8 plus an unchanged
row scale offline; their device layout/exporter is future work.

## Runtime and preprocessing ambiguities

The [published reference][reference] expands the container into stock
`ParakeetForTDT`, checks missing/unexpected keys, and uses the base
`ParakeetProcessor` and generation config. It uses FP32 by default or BF16.
The archive names `mlx-audio 0.4.6` plus two native 2-bit affine matrix multiplies
and BF16 activations. Fermion also publishes CPU and CUDA runtimes. The inspected
CPU Dockerfile pins `fermion-research==0.2.11` and `transformers==5.17.0`.
None is an FPGA backend or an acceptable production fallback here.

Before a full graph is accepted, resolve these explicitly:

1. The archive's NeMo config has `dither=1e-5`; the inspected Transformers and
   CUDA feature paths contain no dither. The follow-up qualification explicitly
   selects zero dither and includes feature golden vectors. MLX execution parity
   has not been established.
2. The inspected feature path uses a nonperiodic Hann window, centered constant
   padding, magnitude via `sqrt(real^2+imag^2)^2`, Slaney filters built in FP32,
   and `ln(mel + 2^-24)`. Do not silently replace these by a different FFT,
   mel convention or power calculation when comparing exact intermediate data.
3. Valid feature length is `floor(samples/160)`, although the centered STFT
   produces an extra frame. Normalization ignores padded frames, divides
   variance by `valid_frames-1`, and adds `1e-5` **after** square root. Short
   inputs with fewer than two valid frames are rejected by the qualification policy. A 30 s
   input has 3000 valid / 3001 stored mel frames and 375 valid / 376 stored
   encoder frames under the inspected unbucketed path.
4. The reference mixes channels on host and chooses librosa resampling, with a
   torchaudio fallback. Fermion also chunks long files at pauses. These are not
   admitted as host model preprocessing. The first proposed adapter accepts
   only canonical 16 kHz mono PCM, at most 30 s, with no resampling/chunking or
   hotwords. PCM-to-FP32 conversion is still an unimplemented device operation.
5. Fermion CUDA forces frame advance after ten zero-duration symbols; the
   inspected Transformers TDT update forces progress on blank plus zero duration
   but omits that per-frame ten-symbol guard. Both conventions must be tested
   and one pinned, including initial blank state, end-of-encoder/EOS handling,
   pad-token treatment and global token budget. Never take one argmax over all
   8198 outputs or collapse repeats as if this were CTC.
6. Our QACT uses INT8 activations, unlike the upstream dense FP32/BF16 or CUDA
   FP16 paths. Exact preservation of weight levels does not establish model
   accuracy. Full-encoder error, token agreement and WER remain gates.

## Implemented slice and resource budget

`opentpu.asr.phonon2_linear` independently decodes one bounded five-value record
and compiles two disjoint signed INT8 planes with native row scales:
`y = MM(QACT(x), sign_low, lo) + MM(QACT(x), sign_high, hi)`.
Unpacking/packing immutable weights happens offline. Activation quantization,
both matrix products and accumulation are device instructions. No dynamic
activation is transformed on host by this kernel. No RTL or ISA change is used.

The compiler checks single-slice use, ACT rows/blocks, DRAM and IMEM budgets;
TMEM fitting is checked by the existing compiler. It pads input columns with
zero weight codes to D alignment. It does not tile the whole encoder. The test
harness transfers synthetic activations only to qualify this isolated kernel;
the eventual model must keep intermediate activations on device.

| Resource | Evidence / budget |
|---|---|
| Saved F2 target | S=1, D=128, MCOLS=4, ACT blocks=128, ACT rows=16, TMEM=65,536 words, IMEM=32,768 words, logical DRAM=4 GiB |
| Full real `layer0.q_proj`, four input rows | 2,105,344 weight bytes, 2,138,112-byte image, 9 instructions / 72 instruction words |
| All five-value matrices as two INT8 planes | 1,211,498,496 bytes including FP32 row scales |
| INT6 tables as D=128-padded INT8 plus row scales | 25,795,996 bytes, proposed layout only |
| All FP16 records expanded to FP32 | 1,688,696 bytes, includes unused BatchNorm counters |
| Total proposed weight storage | 1,238,983,188 bytes before allocation alignment; graph scratch/programs/outputs additional |
| Dense reference parameter storage | 2,508,032,536 bytes FP32; 1,254,016,268 bytes FP16 |

The current loaded image was deliberately not queried. The Inferlab worker
knows burst200 and wideport150 profiles, but that is not fresh device identity
evidence. Qualification must read and check actual capabilities within a
parent-coordinated lock window. Memory arithmetic alone does not prove full
graph fit, bandwidth, timing closure or usable ASR latency.

## Remaining device gaps and next bounded plan

| Missing integration | Existing-ISA possibility / exact gap |
|---|---|
| Waveform frontend | No FFT/STFT/mel ASR kernels. FP32 add/mul/reductions/log2/rsqrt can express the math, but fixed-coefficient transforms, permutations, masks and normalization need lowering and validation. No native FFT or PCM16 conversion opcode. |
| Learned subsampling | The complete Conv2D/depthwise/pointwise/projection path and handoff through all 24 layers are shared-profile ISA-qualified with real speech. Device waveform frontend and physical runtime integration remain missing. |
| FastConformer | All 24 layers with tiled full-context attention, device mask handoff, one-slot residency and reset are shared-profile ISA-qualified. Physical loading/execution and larger-context numerical coverage remain gated. |
| LSTM predictor | No resident two-layer Phonon LSTM, embedding lookup or cache commit policy. Affines and sigmoid/tanh can use existing units; state/gate layout and gating must be compiled. |
| TDT and text output | ARGMAX, RLD, LOOP and HALT CHAIN exist; token/duration split, blank masking, zero-duration bounds, state updates, UTF-8 token assembly and termination are not compiled. |
| Five-value storage | No native trit/nonuniform-five-level MXU format is required: two disjoint INT8 planes preserve the exact stored levels on the shared image, at greater storage/bandwidth cost. |

No new image has yet been shown necessary for correctness. Frontend qualification
semantics and vectors are available in the follow-up reports. The learned
subsampler and tiled 24-layer graph are now qualified with real checkpoint
weights on `f2-hbm:p4rl8a64c150`; see the encoder report for precise scope.
Next implement the bounded device waveform frontend and test its handoff into
the subsampler. Resolve the documented TDT runtime disagreement, then add
INT6 export and resident LSTM/joint/TDT control and test blank/zero-duration/limit
cases. Compare features, encoder outputs,
decoder states and tokens against the selected reference. Only if an existing
instruction cannot implement a required operation or measured budgets fail,
report the exact shared-image capability gap. Model-specific image changes are
outside the user's shared-image requirement; no synthesis is authorized here.

Final acceptance needs a coordinated hardware window, both normal locks,
verified loaded-image capabilities, real audio-to-transcript evidence with
model/revision/input hashes, token/text and quality results, timings, and proof
that frontend, learned layers and decoding stay on FPGA. Never reload an image
or evict the resident Gemma worker to obtain this evidence without coordination.
The required shared image is `agfi-0001de1dd3df8e21e`, with at most 30 seconds
of serving interruption in any future coordinated acceptance window.

## Inferlab adapter contract

`model_status()` provides pinned architecture, blockers and explicit false
readiness flags. `inferlab_model_status()` supplies the existing Inferlab card
keys (`id=phonon2`, `kind=asr`, `capability=transcription`, `ready=false`).
Generate either with `python -m tools.phonon2_status [--inferlab]` from the repo.
Static exports accompany this document. No active Inferlab UI/serving file was
changed.

The reserved `transcribe(pcm_s16le, sample_rate_hz=16000, channels=1)` entry point
always raises `ASRNotReady`, code `ASR_FPGA_NOT_READY`, suggested HTTP 503, before
file/model/device access. Keep audio upload/run controls disabled. The existing
Inferlab generic unavailable-model guard may continue returning its 409
`MODEL_UNAVAILABLE`; do not route around it. A future dedicated transcription
route should use the ASR error while unavailable, not return empty text as success.
No CPU/GPU fallback or simulated transcript is exposed.

## License and verification

The [publisher's NOTICE][notice] and [weight license][weights-license] identify
**CC-BY-4.0** for weights and inherited architecture/tokenizer/feature config;
the [Fermion code license][code-license] is **Apache-2.0**. Preserve the NVIDIA
and Fermion attribution, license links and change notices when distributing
derived weights. The NOTICE also lists training-data attribution; no broader
legal conclusion is inferred here. No weights are redistributed by this change.

Validation and exact measured errors are in [validation.json](phonon2/validation.json).
Local tests cover malformed records, sign/bit/row order, nonuniform scales, all-zero
rows, padding, resource rejection, actual encoder projection shapes, dense CPU
math versus ISA, and fail-closed serving metadata. A full real 1024x1024 query
projection was additionally run on the existing EC2 CPU/ISA simulator with four
seeded activation rows. It is not an audio or hardware test. Existing autodecode
tests requiring Transformers are skipped because that dependency is absent.

[model-config]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/config.json
[base-config]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/config.json
[processor]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/processor_config.json
[graph]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/modeling_parakeet.py
[generation]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/generation_parakeet.py
[cuda]: https://github.com/fermionresearch/phonon/blob/43b98dd54f13277715d9273b921792aba9aa94ce/docker-phonon2/phonon2_cuda_engine.py
[reader]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/fermion_container.py
[reference]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/reference_transformers.py
[notice]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/NOTICE
[weights-license]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/LICENSE-WEIGHTS-CC-BY-4.0.txt
[code-license]: https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/LICENSE-CODE-Apache-2.0.txt
