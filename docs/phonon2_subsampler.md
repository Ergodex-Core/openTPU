# Phonon-2 learned 8x subsampler qualification

The complete learned subsampler now compiles to the existing ISA: three stride-2
convolution stages, their masks and ReLUs, channel/frequency flattening, and the
4096→1024 projection. Real checkpoint tests passed on speech-derived features,
including a full 7.06-second recording. The short speech excerpt also passed a
device-to-device handoff into the existing complete first Conformer block.

This is CPU/ISA qualification. **Full FPGA ASR remains unavailable.** The waveform
frontend is still an offline test oracle, and no FPGA, image, lock, service,
synthesis job or new resource was used. The adapter continues raising
`ASR_FPGA_NOT_READY`; no CPU inference fallback is exposed.

## Source and exact checkpoint binding

Model `FermionResearch/Phonon-2` remains pinned at
`fb09063c9dc8e130d97b3a2191ed5aed6b71bb31`, with container SHA256
`4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`.
The [official graph][graph] defines:

1. Conv2D 1→256, kernel3/stride2/pad1, bias; time mask; ReLU.
2. Depthwise Conv2D 256 channels, kernel3/stride2/pad1, bias; time mask;
   pointwise Conv2D 256→256, bias; time mask; ReLU.
3. The same depthwise/pointwise sequence again.
4. Transpose `[N,C,T,F]` to `[N,T,C,F]`, flatten channel then frequency, and
   apply the biased Linear `[1024,4096]`. There is **no time mask after Linear**.

All twelve `encoder.subsampling.*` records are hash-checked and decoded as data.
Three INT6 matrices retain codes in [-32,31] and unchanged FP16 row scales; nine
FP16 arrays contain the three spatial filters and six biases. The decoder checks
record shapes, kinds, lengths, duplicates, finite values and total container
extent. It executes no upstream loader or `trust_remote_code`.

The numerical receipt uses source from published OpenTPU commit
`8815ac80226aa3a0d0b07d03344f61839580415e`, with new modules in an isolated
overlay. The previous projection and Conformer implementations and receipts are
retained byte-for-byte. Their original scope is unchanged.

## Device operations and interface

`opentpu/asr/phonon2_subsampler.py` accepts batch-one row-major FP32 features
`[stored_frames,128]`, with a static nonempty valid prefix and at most 3001 stored
frames. Output is `[ceil(stored_frames/8),1024]` plus the separately propagated
valid length. It does not accept waveform bytes or call a host feature extractor.

| Phase | Device work |
|---|---|
| Reset scratch | V_FILL/ST resets every halo, masked cell, padding tail and output on each invocation |
| Pad input | LD/ST copies features into a zero-bordered device buffer |
| First convolution | FP32 RDOT against the nine filter coefficients, bias and ReLU |
| Depthwise stages | Nine neighboring per-channel MUL/ADD operations and bias |
| Pointwise stages | QACT, native INT8 MM with original row scale, bias and ReLU |
| Flatten | On-device column copies transpose frequency/channel order |
| Projection | QACT, native INT8 MM and the original unmasked output bias |

Internal convolution buffers use time/frequency/channel order for contiguous
channel vectors. Device flattening explicitly restores channel/frequency order.
Immutable weights alone are unpacked/packed on host. Every learned convolution,
projection, activation, layout conversion and mask runs as device instructions.
Only the three matrix inputs use INT8 activation quantization; spatial filters
use FP32 vector arithmetic. No new opcode, optional stream feature or image is
required by this lowering.

Masks follow the source after **every** convolution, including pointwise layers.
The lowering fuses those masks by resetting outputs and computing valid time
rows. It preserves the source module's input behavior: it does not silently mask
the feature argument before the first convolution. Frontend-produced padded
feature frames are already zero. Finite intermediate values are required for
the mask-fusion equivalence.

Valid and stored time lengths propagate independently by ceil(length/2). For
the speech excerpt they are `56→28→14→7` and `57→29→15→8`. The last padded
encoder frame equals the final Linear bias, rather than zero; tests enforce this.
For the maximum shape they are `3000→1500→750→375` and
`3001→1501→751→376`.

Repeated executions keep simulator DRAM, TMEM and ACT state. Reset tests run a
longer valid prefix followed by a shorter one and require bitwise equality with
a fresh invocation. Fresh tests poison scratch with NaNs. Additional checks
ensure weights are unchanged and kernels do not write past their allocated image.

## Resources and evidence

| Saved F2 resource | Speech excerpt | Maximum compiled 30-second shape |
|---|---:|---:|
| Stored mel / encoder frames | 57 / 8 | 3001 / 376 |
| DRAM image | 8,086,016 bytes | 180,698,624 bytes |
| Aligned immutable weights | 4,368,384 bytes | 4,368,384 bytes |
| Peak live TMEM | 21,512 words | 21,512 words |
| Largest phase program | 360 words | 360 words |
| QACT depth | 32 blocks of 128 | 32 blocks of 128 |
| Program phases | 9 | 9 |

These fit the saved S=1, D=128, MCOLS=4, ACT_ROWS=16 declaration with 65,536 TMEM
words, 32,768 IMEM words and 4 GiB logical DRAM. The short test simulator uses
64 MiB. The maximum shape was compiled with a 256 MiB simulator budget; it was
not executed at 30 seconds. Current card identity, FPGA timing, bandwidth,
latency and complete-model memory fit remain unmeasured.

[subsampler_validation.json](phonon2/subsampler_validation.json) records checkpoint
and implementation hashes, phase budgets, per-stage comparisons, mask/reset
results, and the device handoff. Independent NumPy spatial equations and native
PyTorch `conv2d`/`linear` equations follow the inspected source. A second NumPy
reference includes ISA activation quantization to isolate lowering errors.

The 0.56-second speech excerpt passed with **1.32%** final relative L2 error
against the dense reference; its maximum per-stage relative error against the
activation-quantized reference was about **3.0e-7**. The excerpt's learned
subsampler plus first Conformer block passed at **1.39%** versus the dense chain.
The full recording, with 707 stored / 706 valid mel frames and 89 encoder frames,
passed at **1.52%** versus the dense subsampler reference.
Its largest per-stage relative error against the activation-quantized reference
was **2.49e-4**, within the explicit 1e-3 criterion. Comparisons are tolerance-based;
the full graph is not claimed to match bit-for-bit across FP32 reduction orders.
These are numerical slice results, not transcript agreement or WER measurements.
The receipt also includes the full recording and synthetic boundary cases.

The handoff runs a device copy from subsampler output to the first block's input
address before the host stages the block's immutable weight/position prefix.
The same Machine retains dynamic activations. There is no host transformation
or re-upload of intermediate values. This qualification staging demonstrates the
handoff; a resident complete-model allocation/driver is still required.

## Real speech fixture and reproduction

The [official PyTorch tutorial][audio-tutorial] identifies LibriSpeech test-other
utterance `1688-142285-0007`. Its [WAVE data][audio] is 451,898 bytes, mono 16 kHz
IEEE float32, SHA256
`67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8`.
LibriSpeech is published as [CC-BY-4.0][corpus]. Attribution, conversion and crop
details are in [speech_fixture.json](phonon2/speech_fixture.json).

The checked-in [PCM excerpt](phonon2/speech_1688_crop.pcm) covers samples
`[16000,24960)`, rounded/clipped to PCM16. The [compact golden artifact](phonon2/subsampler_speech_golden.npz)
contains exact EC2 reference features/mask and the seven valid dense subsampler
output rows. It excludes raw checkpoint tensors and the padded bias-only row.
Its [manifest](phonon2/subsampler_speech_golden.json) binds the hashes and licenses.
Only this small audio crop and derived vectors were transferred to the Mac;
model weights remain on the existing EC2 instance.

Frontend generation and WAVE decoding in the qualification tools are explicitly
offline CPU oracles. They preserve the previously selected zero-dither frontend
policy and duration guards. The serving adapter imports neither tool.

```sh
python -B -m tools.phonon2_qualify_subsampler \
  --container /path/to/existing/model.fermion \
  --pcm docs/phonon2/speech_1688_crop.pcm \
  --report /tmp/subsampler.json --golden /tmp/subsampler-golden.npz \
  --full-audio-wave /path/to/existing/1688-142285-0007.wav
python -B -m pytest -q tests/test_phonon2_subsampler.py \
  tests/test_phonon2_conformer.py tests/test_phonon2_asr.py \
  tests/test_kernels.py tests/test_autodecode.py
```

The qualification tools use the existing NumPy/PyTorch environment (Python 3.11+
for the retained Conformer audit helper). No dependency was installed.

## Remaining full-ASR work

Next implement the device waveform frontend: PCM conversion, preemphasis,
FFT512/STFT, mel/log and masked normalization, including the recorded low-variance
numerical issue. Then integrate its output into this subsampler. Full-context
tiled attention and all 24 Conformer blocks, LSTM predictor, joint head, the
explicitly selected TDT duration/liveness policy, token-to-text output and a
resident graph/driver remain missing. The existing Conformer slice accepts at
most eight encoder frames; do not split a long utterance into independent blocks
and claim equivalent noncausal attention.

Final acceptance still requires a parent-coordinated card window, normal locks,
verified loaded-image capabilities and real audio-to-transcript FPGA evidence
with all model computation on device. No such hardware action was attempted.

[graph]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/modeling_parakeet.py#L350
[audio-tutorial]: https://docs.pytorch.org/audio/2.2.0/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html
[audio]: https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav
[corpus]: https://www.openslr.org/12
