# Qwen3-TTS 0.6B CustomVoice: FPGA readiness

The exact requested model is **Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice**.
This change implements its **text projection compiler slice only**. Full
text-to-waveform inference is **blocked**, and the serving contract refuses it
before loading weights or opening a device. There is no CPU, GPU, or external
API fallback. Do not advertise this model as usable inference or reuse a Gemma,
PersonaPlex, or generic Qwen language-model route.

The work starts from OpenTPU `model-infer-f2` commit
`bbee9039ac65fa54bd3f9b29ceb4007e070a5d75` (including the publisher’s deployment logging fix).
The compiler/runtime source matches the tested `e55992e4` base. No RTL, image loading, synthesis,
active UI, or production serving files are changed. Gemma retains priority on
the shared 150 MHz wide-port image.

## Pinned primary evidence

- [Model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
  and [pinned config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/config.json).
  Model revision: `85e237c12c027371202489a0ec509ded67b5e4b5`.
- [Upstream source](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e),
  package version `qwen-tts 0.1.1`, with Transformers `4.57.3` and Accelerate `1.12.0`.
- [TTS model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py),
  [inference wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py),
  and [codec decoder implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py).
- [Codec config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/speech_tokenizer/config.json)
  and [text tokenizer config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/tokenizer_config.json).
- [Apache-2.0 source license](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE).
  The model card also declares Apache-2.0. The retained config fixture is from
  the Alibaba Qwen model release; preserve upstream attribution/license on
  redistribution. No weights or upstream executable Python are vendored.

`qwen3_tts_sources.json` records URLs, revisions, file hashes, and parameter
counts derived by reading only the two safetensors headers with HTTP Range.
No tensor payload was downloaded, no upstream source was imported, and no
unrecognized remote code was executed.

## Exact variant and controls

The config says `qwen3_tts`, `tts_model_size=0b6`,
`tts_model_type=custom_voice`, and `qwen3_tts_tokenizer_12hz`.
It is neither the 1.7B model nor the Base/VoiceDesign checkpoint.

The model card shows a style-instruction example, but the pinned wrapper's
`generate_custom_voice` explicitly clears `instruct` for `0b6`; the upstream
README's model table also omits instruction control for this variant. Therefore
the portal must omit style control, and the adapter **rejects nonempty
instructions** rather than silently ignoring them. Voice cloning, microphone
input, and free-form voice design are not CustomVoice capabilities.

Supported speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna,
and Sohee. Supported languages: Chinese, English, Japanese, Korean, German,
French, Russian, Portuguese, Spanish, Italian, plus Auto. The config assigns
Dylan and Eric Beijing/Sichuan dialect conditioning. Speaker names and language
names are case-insensitive at the adapter boundary.

The text frontend is the checkpoint's Qwen2 BPE tokenizer (`vocab.json`,
`merges.txt`, special tokens), not a phonemizer or a learned text encoder.
The reference wraps text in an assistant message and adds another assistant
prefix. Learned text embeddings have width 2048. The projection is two biased
affine layers, 2048→2048→1024, with SiLU between them. Special text embeddings,
speaker/language codec embeddings, and speech codes are assembled into talker
inputs by the upstream generation wrapper.

## Architecture and rates

| Component | Verified configuration | Device status |
|---|---|---|
| Text embedding | 151,936 × 2,048 | Device lookup/prompt assembly absent |
| Text projection | 2,048→2,048→1,024; both biases; SiLU | New INT8/fp32 ISA slice |
| Talker | 28 layers, hidden 1,024, MLP 3,072, 16 query/8 KV heads, head dimension 128; RMSNorm; Q/K norm; RoPE theta 1e6 | Dedicated adapter absent |
| Talker codec output | 3,072 logits, including codec control/speaker/language entries | Sampling/control absent |
| Code predictor | 5 layers, hidden 1,024, MLP 3,072, 16 query/8 KV heads, head dimension 128; 15 separate 2,048-logit heads and embeddings | Adapter and frame-local cache absent |
| RVQ reconstruction | 16 code groups; first semantic group plus 15 acoustic groups | Device gather/sum/projections absent |
| Codec pre-transformer | 8 layers, hidden 512, MLP 1,024, 16 query/16 KV heads, head dimension 64, sliding window 72; layer scales | Dedicated adapter absent |
| Waveform network | Causal conv, two ×2 transposed-conv/ConvNeXt blocks, then ×8/×5/×4/×3 upsampling; SnakeBeta and final clamp | Device implementation absent |

Talker predicts one primary code per audio frame; the code predictor generates
the remaining 15 sequentially, and their embeddings are summed with the primary
code embedding for the next talker input. Equal hidden widths make
`small_to_mtp_projection` an identity for this variant. The temporal and
within-frame prediction loops must remain distinct.

The audio is **24,000 Hz mono**, with **1,920 samples per codec frame**:
24,000 / 1,920 = **12.5 frames/s** (80 ms per frame). The marketing name is
“12Hz”; `position_id_per_seconds=13` is separate position metadata. Sixteen
codes per frame means 200 code IDs per second, not 200 waveform samples/s.
The packed generation defaults use sampling, temperature 0.9, top-k 50,
top-p 1.0, repetition penalty 1.05, and maximum 8,192 new talker tokens.
EOS/control tokens do not themselves constitute audio frames.

## Reference examples, not FPGA execution paths

1. The [model-card Python example](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice#sample-usage)
   loads the exact checkpoint with `device_map="cuda:0"`, BF16, and FlashAttention 2,
   then calls `generate_custom_voice`. Use its explicit model ID for reference
   work; omit `instruct` for 0.6B. The upstream
   [CustomVoice test example](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/examples/test_model_12hz_custom_voice.py)
   instead defaults to 1.7B and must not be mistaken for exact-variant evidence.
2. [Qwen's Gradio demo](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/cli/demo.py)
   supplies the speaker/language interaction pattern. The wrapper returns a
   complete waveform. `non_streaming_mode=False` simulates streaming text
   conditioning; it does not make that Python API yield live audio chunks.
3. [vLLM-Omni v0.18.0 streaming examples](https://docs.vllm.ai/projects/vllm-omni/en/v0.18.0/user_guide/examples/online_serving/qwen3_tts/)
   explicitly list 0.6B CustomVoice and demonstrate HTTP chunked PCM and a
   FastRTC/WebRTC player. They require an async-chunk serving configuration;
   streaming uses signed 16-bit PCM at 24 kHz mono. These are reference GPU/NPU
   serving/transport examples, not evidence of OpenTPU support. Older v0.14.0
   documentation explicitly lacked streaming, so the version matters.

None of these stacks is installed or launched by this change. The series-wide
97 ms latency claim is not an OpenTPU measurement or an accepted 0.6B FPGA target.
The decoder source's own batch chunking uses 300-frame chunks with 25 frames of
left context; that is not a one-frame real-time FPGA stream implementation.

## Memory and precision

Counts below come from tensor header shapes/dtypes, not the model name.
For the codec, these count stored scalar elements, including codebook buffers;
they are not a separate census of trainable parameters.

| Stored component | Parameters | Tensor bytes |
|---|---:|---:|
| TTS checkpoint, BF16 | 905,788,672 | 1,811,577,344 |
| Codec decoder, FP32 | 114,323,137 | 457,292,548 |
| Codec encoder, FP32, unused for fixed-speaker synthesis | 56,234,304 | 224,937,216 |
| Complete download, tensor data only | 1,076,346,113 | 2,493,807,108 |

The TTS file is 1,811,626,576 bytes and the codec file 682,293,092 bytes including
headers. The native stored TTS-plus-decoder weights are 2,268,869,892 bytes
(about 2.113 GiB). Actual runtime dtype conversion can change this footprint;
it is not an assertion about PyTorch peak VRAM. Text embeddings alone account
for 311,164,928 parameters; the code predictor accounts for 141,570,304.

At BF16, the talker KV payload is 114,688 bytes per cached position:
2 × 28 layers × 8 heads × 128 dimensions × 2 bytes. Thus 8,192 positions need
896 MiB and the configured maximum 32,768 positions need 3.5 GiB, before
workspace. Code-predictor KV is frame-local; waveform attention and convolution
state are additional. The historical F2 target exposes a 4 GiB address window,
but no claim is made about free/resident memory on the current shared image.
Native weights plus maximum-context KV alone exceed that window.

The new slice packs matrix weights into OpenTPU INT8, quantizes activations
with QACT, and uses the ISA's fp32 arithmetic for biases and SiLU. This changes
precision from the BF16 reference and is **not** a validated whole-model
quantization recipe. Full INT8/INT4 storage estimates must include scales,
alignment, codebooks, fp32 vectors, KV, and streaming buffers; no full image is
packed here. A full-precision fp32 deployment would be larger still.

## Implemented compiler boundary

`ProjectionSpec.from_config` accepts only the audited variant/projection.
`ProjectionSpec.compile` validates shape/alignment/finite values, packs the four
exact checkpoint tensors under `talker.text_projection`, and emits two MM stages
with on-device biases and SiLU. It accepts at most MCOLS rows per invocation.
Multi-slice output rows and biases are sharded, with a device gather of the
intermediate activations. The dimensions are independently testable at reduced
size; the real 2048→2048→1024 shape has a compile-only check.

No new opcode or RTL is used. The compiled program uses ordinary loads/stores,
INT8 MM, QACT, VOP arithmetic/exp2/reciprocal, and collectives. Input embeddings
and output readback belong to the **diagnostic harness**, not an authorized
production host compute path. There is no device text embedding lookup in this
slice and no claim that this subgraph produces speech.

## Full host-dependency audit

| Stage/dependency in reference | Current state | Required FPGA-only outcome |
|---|---|---|
| Qwen2 BPE, special-token parsing, request validation | Not wired here | Deterministic text frontend may run on host; no learned embedding math there |
| Embeddings, prompt mixing, speaker/language conditioning | Missing | Device gather/add of learned tensors |
| Text projection | ISA implemented/tested separately | Both affine layers, biases, SiLU on FPGA |
| Talker attention, Q/K RMSNorm, RoPE, SwiGLU, KV, head | Missing | Device-resident kernels/caches; offline constant RoPE tables are possible |
| Code predictor/15 heads/embedding accumulation | Missing | Device frame-local autoregressive loop |
| Temperature, repetition penalty, top-k/top-p, random draw, EOS | Missing | Device logits processing and RNG/control; host sampling is disallowed |
| RVQ lookup/sum and projection | Missing | Device code-to-latent reconstruction |
| Decoder transformer, sliding-window mask, layer scale | Missing | Device state/mask/math |
| Causal conv, depthwise conv, transposed conv, trimming | Missing | Device scheduling and ring buffers, no host im2col |
| ConvNeXt LayerNorm and exact GELU | Missing | Device normalization and accepted exact-function implementation |
| SnakeBeta: learned exp(alpha/beta), sin², reciprocal/add | Missing | Constants may be prepared offline; dynamic sine must execute on device |
| Clamp and waveform output | Missing | Final samples produced by FPGA |
| PyTorch/Transformers/Accelerate, FlashAttention | Reference only | No dependency in a deployed inference route |
| Librosa/torchaudio/SoX/ONNX Runtime | Upstream package dependencies | No runtime neural/audio transformation fallback; no audio encoder required here |
| SoundFile/container writing, transport, browser playback | Future host orchestration | May transport/package/play FPGA-computed samples |
| Offline checkpoint conversion and compilation | Host preparation | Explicit fixed provenance/precision; not hidden request-time inference |

`synthesize(TTSRequest(...))` always raises `FPGAOnlyUnavailable`. It does not
import a backend, access a checkpoint, allocate a card, or start a service.
The catalog's `enabled=false` must not be overridden just because the shared
card is healthy or the Gemma endpoint is warm.

## Exact gaps and smallest next plan

1. Validate checkpoint packing/projection error on EC2 with approved reference
   execution and a small selected tensor set; then implement device embedding
   lookup/prompt assembly. Toy tests establish compiler behavior, not audio quality.
2. Write dedicated talker and code-predictor adapters against independent source
   references. Existing Qwen primitives are candidates, not drop-in compatibility.
   Match the talker's interleaved MRoPE convention and frame-local predictor cache.
3. Lower causal/depthwise/transposed convolution using existing MM/vector
   instructions with device-resident history and padding. The pre-transformer's
   64-dimensional heads do not satisfy the existing Qwen adapter's D=128
   divisibility check; explicit padding/layout and mask logic are needed.
4. The current VPU has neither sine nor erf/exact GELU. First characterize
   approved polynomial/range-reduction lowerings and their audio error using
   existing arithmetic. If accurate/performance-bounded lowering is infeasible,
   propose a bounded sine/range-reduction primitive (and accepted GELU strategy),
   bit-exact ISA model and RTL tests before any synthesis. No RTL necessity or
   timing/area budget is asserted without that experiment. Device stochastic
   sampling/RNG also needs an explicit design; ARGMAX alone is not the reference.
5. Close a full host-compute audit, checkpoint/audio-quality tests and streaming
   state equivalence, then ask the parent to coordinate a shared-card validation
   window. Pin the current image identity/config; do not reload it or interrupt
   warm Gemma. No such window is consumed by this change.

## Portal integration handoff

Import `portal_metadata()` or consume `qwen3_tts_portal.json` as a catalog entry.
Use a text field, language selector, and speaker selector. Show readiness as
blocked with the specific remaining stages. Do not display style, voice cloning,
or an enabled synthesis/stream button. Keep the existing Gemma route independent.

For a future enabled implementation, the intended request is text + language +
speaker; output is mono PCM at 24 kHz, optionally framed for streaming only after
stream-state validation. A serving wrapper should translate
`FPGAOnlyUnavailable` into its normal unavailable-model response, without trying
another engine. This branch does not add an HTTP endpoint or edit Inferlab UI
files; the parent/UI/deployment workers own that integration.

## Validation receipt

On the existing EC2 host, the isolated test run completed **18 passed in 0.34s**
under Python 3.12.3 / NumPy 2.5.4 / pytest 9.1.1, using a single CPU thread.
SSM command: `1268b94d-ef94-4b39-a485-4e46ddae1ba4`.
The tests cover independent float64 reference comparison with a 2.5% relative
L2 error bound for INT8 synthetic fixtures, both biases, one/two slices,
D=32/128, one/four rows, deterministic ISA replay, native-shape compilation,
invalid inputs/configs, and the no-fallback serving gate.

The native one-row projection compiles to **16 instructions / 128 IMEM words**
against the target's 32,768-word budget and occupies **6,377,472 device bytes**
inside an 8 MiB test allocation. Its instructions are 3 LD, 2 QACT, 2 MM,
7 VOP, 1 ST, and 1 HALT. Successful compilation enforces TMEM constraints;
no separate measured TMEM peak is claimed. Footprint SSM command:
`68c47f4f-ddca-497f-ad66-541f6e4f3a29`.

These are synthetic reference/ISA and native-shape compile results, not RTL,
real-checkpoint, physical-FPGA, speech-quality, or latency validation. All tested
source hashes and the base relationship are retained in
`qwen3_tts_validation.json`. The only local execution was lightweight syntax,
metadata, Git, and source inspection; no Mac model tests or model downloads ran.

Requested next coordinated hardware action: parent reserves a projection-only
validation window on the current image, queries its actual configuration under
the shared-card protocol, then compares a synthetic projection against ISA
without reloading the image. This action is pending coordination, not denied,
and has not been attempted. Full TTS synthesis remains blocked by the stages
listed above even if that slice passes hardware validation.
