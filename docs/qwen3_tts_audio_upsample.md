# Qwen3-TTS two-stage upsampling and ConvNeXt

This separately reviewable increment follows resident replay commit `30841221`.
The resident path now includes both factor-2 causal transposed convolutions and
both ConvNeXt blocks after the eight-layer transformer. It emits **four rows of
1024 decoder features per codec frame**, not waveform samples. Software component
acceptance passes; full TTS, quality and physical readiness remain false.

## Exact source and arithmetic

The model remains `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, with Apache-2.0 source revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`. The [official tokenizer source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py)
and [pinned config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/speech_tokenizer/config.json)
define two upsampling ratios [2,2]. Each transposed convolution has kernel and
stride 2, so it has no inter-frame overlap or right trim. Its output feeds a
causal depthwise convolution (kernel 7, groups 1024), LayerNorm epsilon 1e-6,
1024-to-4096 affine, exact-erf-form GELU, 4096-to-1024 affine, learned gamma and
residual addition.

The checkpoint reader again verifies the full cached tokenizer file: 682,293,092
bytes, SHA-256 `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`,
plus tensor dtype/shape/extents. Twenty-two unique official F32 tensors back 24
physical parameter views. ConvTranspose source weights are [input,output,tap],
so each physical MX weight explicitly transposes one source tap. Depthwise
[channel,1,tap] weights are explicitly viewed as FP32 [tap,channel]. Biases,
normalization weights and gamma remain resident FP32; no shapes are inferred
from unrelated model paths.

The FP32-source-algebra contract is unchanged. All affine operations use the
existing two-term INT8 weight/activation correction with four device MM products.
The 4096-wide projection is processed in eight 512-column chunks on the native
profile, reusing a resident identity for device activation reconstruction.
Depthwise products, means/centered variances, normalization, biases, gamma,
residuals and state are FP32 device operations.

Source `nn.GELU()` requires erf. The new `gelu_erf_approx` explicitly approximates
that function using a five-coefficient erf polynomial with reciprocal, abs and
exp2 VOPs. It does not substitute tanh-GELU or call host erf during inference.
The independent references use Python `math.erf` for diagnostic exact-function
comparisons. A 4096-point [-16,16] grid plus tails measured maximum absolute
error **4.76837158203125e-7**, below the fixed 2e-6 bound. Component checkpoint
preactivations ranged from **-37.9632645 to 24.5342388**; a supplemental native-ISA
probe over **8192 points in [-40,32]**, plus +/-10000 and signed zeros, measured
the same maximum error. That probe passed in 0.114 seconds. These sampled-domain
receipts are not an exhaustive proof over all FP32 inputs or a quality claim.

## Resident state and chunk composition

`UpsampleChunkImage`/`UpsampleChunkFrameImage` append nine programs to each
decoded codec row. Stage 0 produces two rows; its ConvNeXt rows are consumed in
order by stage 1, which produces four final rows. Each ConvNeXt keeps six FP32
input rows. It stores the transposed-convolution input to the depthwise branch,
not the residual block's output. At chunk restart, row positions return to zero;
negative-history positions are zero and stale ring slots are never read before
overwrite. Full reset clears those rings as well.

The resident 300/25 controller dispatches these same programs during retained-ID
recomputation. Context outputs are overwritten/discarded and the restored current
IDs produce the four exposed rows. `desc.audio_upsampled` has shape `(4,1024)`;
the standalone diagnostic returns that shape. No learned features are uploaded
from the host in generated-code integration. The output is not PCM and cannot
be played or scaled as waveform audio.

## Builder acceptance and limits

All jobs used isolated `/home/ubuntu/qwen3-tts-upsample-20261023-01` on the existing
`opentpu-m8a-build`, Python 3.12.3/NumPy 2.5.4 from the existing venv, one BLAS/OMP
thread, nice 19 and bounded timeouts. No packages, upstream executable model
code, model downloads or new resources were introduced. The Mac was the source,
Git and small receipt-transport bridge. Control connectivity briefly dropped;
the existing outcomes were inspected after reconnection and no job was restarted.

`upsample-tests-01`: **4 passed in 13.81 seconds**. They cover independent exact-
erf source algebra, asymmetric transposed-convolution taps, causal depthwise
state, chunk/reset behavior, neural-generated resident IDs, native memory/program
limits and the GELU grid/tails. The parent replay increment separately passed
all 15 earlier regression/replay tests.

`upsample-acceptance-01`: **exit 0 in 204.71 seconds**. It evaluated nine retained
three-prompt prefix frames and a separately labeled 14-frame recorded-code
diagnostic with 5/3 chunks. Decoder and upsampling histories were poisoned at the
short boundaries, then recomputed correctly. Prefix reset replays and a seven-
frame replay crossing a short boundary were bitwise equal. References independently
evaluate the full source front, transformer and both upsampling blocks in FP32
and FP64. All branch boundaries are checked before gamma/residual addition, so a
small gamma does not hide projection errors. The bound remains 3.5% relative L2.

| Maximum per-row relative L2, percent | FP32 source | FP64 source |
| --- | ---: | ---: |
| Stage 0 transposed convolution | 0.012871002% | 0.012866544% |
| Stage 0 depthwise convolution | 0.003134169% | 0.003133513% |
| Stage 0 LayerNorm | 0.016321914% | 0.016311040% |
| Stage 0 first affine | 0.005981810% | 0.005978547% |
| Stage 0 GELU | 0.014044137% | 0.013977410% |
| Stage 0 second affine | 0.018716050% | 0.018720532% |
| Stage 0 residual output | 0.019643307% | 0.019650482% |
| Stage 1 transposed convolution | 0.020232377% | 0.020239211% |
| Stage 1 depthwise convolution | 0.007102580% | 0.007112266% |
| Stage 1 LayerNorm | 0.021901331% | 0.021898659% |
| Stage 1 first affine | 0.021459881% | 0.021463730% |
| Stage 1 GELU | 0.021469338% | 0.021431397% |
| Stage 1 second affine | 0.020758165% | 0.020717868% |
| Stage 1 residual output | 0.021162767% | 0.021134282% |

The actual main checkpoint additionally generated three `Hello.` frames twice,
with full reset between runs, using explicit 2/1 diagnostic chunks. All 48/48
codec IDs per run matched the preserved FP32 path, restored device IDs matched,
and upsampled features replayed bitwise. Maximum integrated output error was
**0.017288148%**. The retained nine-frame corpus contains five unique codec-ID
frames. Repeated/permuted diagnostic codes are not generated utterances or
speech-quality evidence.

The previous increment tested 602 real-weight decoder frames with public 300/25
boundaries. This increment's upsampling composition is tested at shorter
boundaries; **a real-checkpoint 300-frame boundary including upsampling remains
unvalidated**. The existing main path still has its separate 512-position limit
including prefill, and no complete native main generation through a public
300-frame boundary is claimed.

## Same-image resource qualification

The software target remains AGFI `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, with runtime
`30100afb075374381601a6e01d9f914a58dbec91`. Only this pinned target/runtime and the
historical register receipt support the qualification; no live card query or
opcode equivalence with another profile is assumed.

New programs compiled at decoder positions 0, 25 and 324 fit **3,088 IMEM words**
and **15,365 TMEM words** at worst, within 32,768/65,536. Standalone public-chunk
front/transformer/upsampling residency is **150,077,312 bytes**. Combined main
residency is **2,824,640,128 bytes**, an increment of 42,704,896 bytes, leaving
**1,470,327,168 bytes** of 4 GiB. The new stage adds nine programs per decoded
codec frame, including every recomputed context frame. Dispatch totals are 554
ordinary and 4,532 public-boundary programs including the existing main path.
These are software resource/dispatch counts, not measured hardware performance.
The remaining waveform weights/buffers/programs have not been qualified.

## Host audit, remaining gates and evidence

All implemented learned request-time math is in device programs: transposed and
depthwise convolution, LayerNorm, corrected affines, erf approximation, gamma,
residuals and causal state. Host work is offline verified parameter views/packing,
fixed constants, deterministic scheduling/reset, transport and explicit references.
No CPU/GPU/API fallback or serving backend was added. No RTL or synthesis was
needed for this stage.

Remaining model stages are the causal waveform convolution and four waveform
decoder blocks (rates 8/5/4/3), including SnakeBeta, followed by raw clamp/scaling.
Input-dependent sine still needs a validated device lowering; it cannot be
computed on the host under the FPGA-only contract. Full utterances, quality,
long-boundary upsampling and coordinated physical acceptance remain. Waveform,
quality, physical acceptance and `tts_enabled` stay false. The historical BF16
gate remains failed at 127/144 IDs, 6/9 frames and 7.51369% maximum conditional
error; no threshold or failed receipt was changed. Service interruption was zero
and the 30-second maximum remains in force. Hardware, image and production were
untouched.

The [evidence directory](qwen3_tts_audio_upsample_evidence/) preserves unmodified
receipts, every measured branch row, requests/results, source hashes, runner,
environment and supplemental GELU probe. Thirty-nine source/fixture hashes and
both supplemental code hashes match the review tree. The snapshot verifies 64
component files and 95 runtime files. Five extra AppleDouble metadata entries in
the start wildcard manifest were hashed only, never imported; original transport
archive hashes verify them separately. The original preparation record marked
the chunk commit pending; `resolved-component-dependency.json` now proves its
source matches immutable parent `30841221` without changing tested source.

```sh
/home/ubuntu/otpu-venv/bin/python -m pytest -q tests/test_qwen3_tts_audio_upsample.py
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_audio_upsample_acceptance.py --data /home/ubuntu/qwen3-tts-upsample-20261023-01/data --output /path/to/new-upsample-receipt.json --integrated
/home/ubuntu/otpu-venv/bin/python tools/qwen3_tts_gelu_domain_acceptance.py --output /path/to/new-gelu-domain-receipt.json
```

All saved jobs are terminal; inspect receipts before any rerun.
