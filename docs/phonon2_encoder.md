# Phonon-2 learned encoder on the shared FPGA profile

The learned 8x subsampler and all 24 full-width Conformer layers now compose in
the CPU ISA simulator using the exact shared profile `f2-hbm:p4rl8a64c150`.
This is learned-encoder evidence. Waveform processing is an offline oracle, the
predictor/TDT/text stages are absent, and the Inferlab endpoint remains blocked.
No FPGA, image reload, production worker, synthesis, or new resource was used.

## Source and image binding

The checkpoint is [FermionResearch/Phonon-2 at
fb09063c](https://huggingface.co/FermionResearch/Phonon-2/tree/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31),
container SHA256
`4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`.
The model remains FastConformer plus a two-layer LSTM token-duration transducer,
627,008,134 learned parameters. Weights are CC-BY-4.0; inspected code is
Apache-2.0. NVIDIA/Fermion attribution and the pinned [NOTICE](https://huggingface.co/FermionResearch/Phonon-2/blob/fb09063c9dc8e130d97b3a2191ed5aed6b71bb31/NOTICE)
remain required. [The architecture/source audit](phonon2_asr.md) and
[semantic policy](phonon2/semantic_policy.json) still govern this work.
Only checkpoint data and our native Torch audit equations executed on existing
EC2. No upstream model code was installed or imported, and no weights reached
the Mac or Git.

The required AGFI is `agfi-0001de1dd3df8e21e`. Its association with the shared
profile comes from the parent instruction; no live card identity was read.
The source declaration resolves to CFG hash `dd6b92a6`, D=128, MCOLS=8,
PORTB=4, ROWLANES=true, ALINE=64, T2=true, 150 MHz, 128 ACT blocks,
65,536 TMEM words, 32,768 IMEM words, SEQ_WC=true and 4 GiB device DRAM.

Qualification uses the common runtime source at
`df69c0a9d970ba4f78b8ac6e979b41dc69d4f181`, with every imported core/compiler
file bound by SHA256 in [shared_profile.json](phonon2/shared_profile.json).
The old branch's Config lacks the shared profile fields; `shared_profile_config`
fails with `ASR_SHARED_PROFILE_RUNTIME_REQUIRED` there. Merge these isolated ASR
files onto the common runtime, then revalidate if any bound compiler/ISA file
changes. The historical projection/Conformer/subsampler receipts retain their
original compiler bindings and have not been rewritten as shared-image evidence.

The lowering needs FP32 load/store and VOP reductions/activations, INT8 MM with
row scales and unit identity weights, QACT, wide-port PAIR/QUAD and duplicated
activation rows, loop/control instructions, and program plus wait-table loading.
It does not require packed T2 weights, grouped row-lane weights, STREAM, DSTEP,
GATHER, a new opcode, or a model-specific image. Physical capability and clock
verification against the deployed AGFI remain acceptance gates.

## Composition and precision

`phonon2_encoder.py` tiles each attention head across the whole stored context.
It preserves noncausal relative attention; it does not split the utterance into
independent short windows. Relative position indexing is `T-1-query+key` after
the source relative shift. Softmax and value reductions use FP32 VOPs.

Masks follow the pinned source: valid queries cannot attend to invalid keys;
padded queries have a uniform distribution over all stored keys after the
finite-min mask. Invalid GLU rows are zero before the nine-tap depthwise
convolution. Final padded rows remain nonzero. All 24 layers execute, with no
extra global encoder norm or input scaling. Layer scratch, convolution halos,
and reset reuse are explicit.

The original single QACT pass accumulated 2.0971% relative L2 error on the short
speech crop. A second device pass quantizes the residual
`x - dequant(QACT(x))`; identity INT8 MMs reconstruct the first pass on device.
The correction contribution is accumulated using the same immutable weights.
Five-value weights remain two exact disjoint INT8 planes with their stored row
scales; INT6 weights retain their exact decoded values. No activation
correction, learned computation, or inter-layer transformation happens on host.
The original validated component source files and receipts are unchanged.

## Actual phase and residency scheme

Each layer has 27 programs: scratch reset, FF1, QKV, relative keys, eight pairs
of score/value programs, attention residual, GLU, halo padding, depthwise/BN/SiLU,
convolution residual, FF2, and final norm. Programs fit IMEM individually.
One complete layer program does not need to fit IMEM.

All 24 immutable layer prefixes use the same addresses and instruction bytes
for a fixed shape. Before each prefix is staged, a device copy preserves the
previous output at the next layer's input. Only immutable constants are uploaded
between layers. The qualification's host reads are audit observations; they
never feed the next layer. A request has 682 dispatches: one graph reset, nine
subsampler programs, 24 device copies, and 24 × 27 layer programs.

The program/wait ABI reserves 163,840 bytes at DRAM address `0xfffd8000`, above
the graph. `ProgramSlotDevice` serializes each program plus aligned WAITC table,
stages these bytes, decodes from that slot, and executes them in the simulator.
A complete 24-layer short-speech trial matched direct loading bit for bit.
The physical Board/PCIe loader has not been exercised. Wait entries conservatively
wait for all prior unit/VPU dependencies; this establishes a correctness baseline
and makes no concurrency or throughput promise.

| Resource | 7.06-second speech execution | 30-second compile envelope |
|---|---:|---:|
| Mel stored / valid frames | 707 / 706 | 3001 / 3000 |
| Encoder stored / valid frames | 89 / 89 | 376 / 375 |
| Subsampler graph bytes | 46,165,632 | 180,715,008 |
| Reused layer-slot bytes | 56,851,456 | 77,369,600 |
| Layer immutable weights | 50,581,504 | 50,581,504 |
| Peak TMEM words | 38,924 / 65,536 | 38,924 / 65,536 |
| Largest phase words | 1384 / 32,768 | 1824 / 32,768 |
| Largest program + wait payload | 7016 bytes | 9120 bytes |

The graph and program slot are disjoint. Peak request graph memory is the
larger of the subsampler and layer slot, rather than their sum. Full-speech
constant staging totaled 1,236,133,888 bytes across the request. This bounded
scheme fits memory but has unmeasured physical upload/dispatch cost. The
simulator has a 4 GiB virtual DRAM allocation, touches only the graph region and
program slot, and peaked at 705,852 KiB resident RSS in the main EC2 audit.
Maximum-context values come from [compile/resource checks](phonon2/shared_capacity.json),
with official shapes and synthetic values; they are not a numerical test at
30 seconds.

## Validation and receipts

[encoder_validation.json](phonon2/encoder_validation.json) records all layers,
684 checkpoint record hashes, exact runtime/source hashes, masks, outputs,
budgets, wait checks, reset, and the program-slot trial. Repeated identical
checkpoint hash tables were deduplicated from the original receipts; original
receipt hashes and successful SSM command IDs are retained.

| Real speech trial | Final valid-frame relative L2 vs dense audit |
|---|---:|
| 0.56-second crop, single activation pass | 2.09714% |
| Same crop, device residual correction | 0.0090466% |
| Full 7.06-second recording, device residual correction | 0.0094929% |
| Crop after full-recording reset | 0.0090466%, bit-identical to fresh crop |

Each trial ran the complete learned subsampler followed by all 24 official
checkpoint layers. Every layer was finite and passed its mask checks. The
declared learned-encoder threshold is 3% relative L2 on valid frames; the
single-pass trial also passes that threshold. The two-pass path is the selected
integration candidate because its observed accumulated error is much smaller.
These metrics establish neither transcript quality nor WER.

The crop and full recording are pinned in [speech_fixture.json](phonon2/speech_fixture.json).
Their input features come from the offline nondithered Torch frontend oracle,
not a device waveform frontend. FP32 silence reductions and the documented
NeMo dither disagreement remain unresolved for final frontend acceptance.

Local regression: 155 passed, one skipped (Transformers is not installed),
covering the Phonon suites plus kernel/autodecode regressions on the bound shared
runtime. Updated metadata and the exact shared maximum-context compile check
also pass. Tests cover frame/tile boundaries, masked keys, padded queries,
convolution halos, reset reuse, all 24 value sets sharing addresses/code, and
serialized program/wait slot bounds.

Before the ASR commit is merged into the common runtime, reproduce with:

```sh
OTPU_TARGET=f2-hbm:p4rl8a64c150 python -B -m tools.phonon2_run_shared \
  --runtime-root /path/to/common-runtime-at-df69c0a9 \
  --container /existing/ec2/model.fermion \
  --pcm docs/phonon2/speech_1688_crop.pcm \
  --wave /existing/ec2/speech_1688_full.wav \
  --output-dir /tmp/phonon2-encoder-qualification
```

The runtime root must contain the manifest's core files and `source_manifest.json`.
Use the existing authorized EC2 environment for checkpoint tests. Do not execute
this CPU/ISA audit as an Inferlab serving backend.

## Remaining steps to full ASR

1. Compile device PCM16 conversion, preemphasis, 512-point STFT, Slaney mel,
   logarithm and valid-frame normalization against the retained waveform
   goldens, including silence, duration boundaries and reset. No native FFT or
   PCM conversion opcode exists; lowering is still missing, rather than proven
   impossible on existing arithmetic. Report a precise capability gap if a
   required step cannot be expressed safely on this shared image.
2. Implement the INT6 embedding, two-layer LSTM predictor/state commit,
   joint head, token/duration selection, blank/zero-duration control and UTF-8
   text output. Resolve the documented source disagreement over the ten-symbol
   forced-advance rule before selecting the production decoder policy.
3. Integrate the physical shared-image loader and complete graph; qualify
   larger-context numerics, decoder states/tokens and real audio-to-transcript
   output. Only a parent-coordinated lock window may use the card, with the
   same AGFI, no model-specific reload, and a maximum 30-second serving
   interruption. The current conservative schedule provides no timing guarantee
   for that window.

`model_status` and Inferlab card exports remain `ready=false`,
`hardware_validated=false`, and `audio_to_transcript_validated=false`.
There is no CPU/GPU fallback or production inference action in this increment.
