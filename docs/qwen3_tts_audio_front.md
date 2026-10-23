# Qwen3-TTS decoder front: bounded software acceptance

The resident codec-frame path now feeds device programs for all 16 RVQ lookups,
usage normalization, semantic/acoustic projection and causal pre-convolution.
The new component passes its real-checkpoint FP32/FP64 reference checks. It does
not produce waveform samples. Full TTS remains disabled; the preserved overall
main-model gate remains failed, including its BF16 comparisons.

## Source and arithmetic contract

This is exactly `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, model revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, with reviewed Qwen3-TTS source revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`. The model and source are Apache-2.0.
Primary sources are the [pinned model files](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5),
[decoder config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/speech_tokenizer/config.json),
[decoder source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py),
and [source license](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE).
Official reference scripts target CPU/GPU libraries; they are source guidance,
not FPGA execution evidence. No upstream executable model code was imported.

Both public safetensors files were acquired and LFS-verified on the existing
`opentpu-m8a-build`, without downloading model weights to the Mac:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Main model | 1,811,626,576 | `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb` |
| Speech tokenizer, including decoder and unused encoder | 682,293,092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |

The tokenizer file has 496 F32 tensors and 170,557,441 elements. Its raw header
SHA-256 is `2ba4e8dcbacb8aca087f1f8cfd8b7a2ab144d32486e635afaf2b3f6918b8cb65`.
The reader validates the full digest, header, tensor types/shapes, contiguous
extents and finite values before supplying weights. No pickle or remote-code
loader is used. Acquisition and official metadata are preserved in the evidence
directory.

The [arithmetic contract](qwen3_tts_arithmetic_contract.md) targets pinned source
weights evaluated with FP32 algebra. BF16 main weights extend exactly to FP32;
decoder weights are already F32. This component keeps codebooks, usage counters,
residuals and history in FP32. Affines use the previously validated two-term
INT8 weight and activation representation: four MX products plus device FP32
reconstruction. It is approximate FP32 algebra, not bit-exact BF16 execution.

## Implemented boundary

Each generated frame has 16 in-range codec IDs. The first codebook is semantic;
the remaining 15 are acoustic. Each selected 256-vector is reconstructed as
`embedding_sum[id] / max(cluster_usage[id], 1e-5)` on the device. Acoustic vectors
accumulate in source order. Separate 256-to-512 projections produce a summed
512-vector. A three-tap causal 512-to-1024 convolution with bias consumes the
current vector and two previous vectors; two zero vectors implement left padding
after reset.

There are 38 physical parameter views backed by 36 unique official tensors. The
source-to-device map explicitly separates convolution taps and singleton kernel
dimensions. It does not infer compatibility with unrelated models.

`AudioFrontFrameImage` extends `TalkerPrecisionFrameImage`, reusing the same
resident `code_ids` descriptor. Three audio programs run before the existing
talker feedback programs. Generated IDs and learned features are not uploaded
from the host. `audio_rvq` and `audio_preconv` expose the boundary tensors for
diagnostic reads; neither is an audio waveform or a production backend.

Reset clears the two causal history vectors as well as the inherited caches.
The standalone diagnostic rejects invalid IDs before writes and latches an
execution failure until reset. Its CPU ISA simulator and independent references
are explicitly test tools; they are not connected as inference fallbacks.

## Verified results

All computation ran on the builder under isolated
`/home/ubuntu/qwen3-tts-waveform-20261023-01`; the Mac performed file/Git control
and receipt transport. No FPGA, synthesis, image or serving operation occurred.
No new instance, service, credentials, system PATH or shared environment change
was needed.

* `front-tests-01`: **4 passed in 8.43 s**. Tests cover source tensor views,
  independent FP64 algebra, zero/negative usage clamping, 35-frame causal history,
  chunk partition equivalence, poisoned-state reset, rejected IDs, injected
  execution failure, resident codec integration and native program/memory limits.
* `front-acceptance-01`: **exit 0 in 155.78 s**. The new component gate passed on
  75 frames: nine preserved three-prompt codec-prefix frames and 66 edge/seeded
  diagnostic frames. The 66 diagnostic frames are not speech-quality evidence.
* Integrated generation rebuilt the exact main checkpoint and decoded three
  generated `Hello.` frames, then repeated from reset. All **48/48 codec IDs**
  match the preserved FP32 path on each run; decoder outputs replay bitwise.
  The reference consumes diagnostic reads of the actual generated IDs only after
  device execution. It supplies no IDs, hidden states or arithmetic to that path.

Maximum relative L2 errors across the 75 component frames, shown as percentages:

| Boundary | Independent FP32 | Independent FP64 |
| --- | ---: | ---: |
| Semantic vector | 0.000008282% | 0.000007831% |
| Acoustic sum | 0.000011153% | 0.000013101% |
| Projected RVQ latent | 0.009753482% | 0.009753055% |
| Causal pre-convolution | 0.009112815% | 0.009103170% |

The unchanged component bound is 3.5%. The maximum integrated pre-convolution
error is 0.007362349%; maximum absolute error there is 0.0001863242. Integrated
reset replay and all standalone prefix reset replays are bitwise equal. These
features have no waveform scaling interpretation.

The upstream receipt at `883edbdd` remains unchanged: FP32 source-algebra
144/144 IDs, BF16 127/144 IDs and 6/9 complete frames; maximum conditional BF16
logit error 7.5136935% exceeds 3.5%. Overall `accepted=false` and all historical
failed receipts remain preserved. The new front receipt repeats those BF16
results and explicitly records `upstream_overall_gate_passed=false`,
`waveform_produced=false`, `raw_waveform_scaling_validated=false`,
`complete_utterances_validated=false` and `tts_enabled=false`.

## Shared-image resource qualification

The software target remains AGFI `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, using runtime
`30100afb075374381601a6e01d9f914a58dbec91`. No opcode equivalence with another
profile is assumed. Qualification here uses the pinned target and historical
image receipt, not a fresh hardware query.

| New program | Instructions | IMEM words | Peak TMEM words |
| --- | ---: | ---: | ---: |
| RVQ lookup/normalization/sum | 165 | 1,320 | 1,043 |
| Semantic/acoustic projections | 21 | 168 | 1,538 |
| Causal pre-convolution | 35 | 280 | 7,682 |

These fit IMEM 32,768 and TMEM 65,536 words. Standalone residency is 37,765,248
bytes. Combined residency is 2,713,376,640 bytes: an increment of 37,765,120 bytes
over the corrected main path, leaving 1,581,590,656 bytes of the 4 GiB address
space. Sharing the resident ID tensor accounts for the 128-byte difference from
the standalone allocation. The front adds three launches per frame to the
preserved 395-program main schedule. These are program/resource counts, not
measured hardware latency or throughput. Full decoder residency is not yet
established.

## Host-dependency audit and remaining work

Device programs perform every implemented learned request-time operation:
lookup, usage clamp/reciprocal, normalization, acoustic accumulation, corrected
affines, bias and causal history. Host operations are verified offline parameter
views/packing, identity constants, deterministic scheduling, state clearing,
diagnostic transport and independent test references. Existing host BPE/control
remain as documented in the main-model contract. No CPU/GPU/API fallback exists.

The next bounded stage is input projection plus the eight-layer decoder
transformer. It requires source-specific attention without talker Q/K RMS norms,
learned branch scales, exact 72-position sliding/causal masking and independent
decoder position/cache/reset handling. Those operations need implementation and
tests; this increment does not claim they work.

After that remain two factor-2 upsampling/ConvNeXt stages and the four waveform
blocks with rates 8, 5, 4 and 3. Causal and transposed/depthwise convolution,
ConvNeXt LayerNorm/GELU and SnakeBeta require audited ISA lowerings. The ISA has
no native sine or erf instruction; bounded sine/GELU approximations or table
lowerings must be validated before deciding whether a genuine RTL gap exists.
No new RTL or synthesis is requested by this result.

The pinned decoder emits 1,920 mono samples per codec frame at 24 kHz (12.5 frames
per second). Final output uses raw clamp to [-1, 1], without per-utterance
normalization. These are verified source semantics, not tested waveform output.
Official chunked decode recomputes with 25 left-context frames; it must be tested
separately from continuous 72-position transformer state. Raw sample scaling,
streaming chunks, complete utterances, quality and coordinated physical/serving
acceptance remain open. Serving interruption was zero; its maximum permitted
budget remains 30 seconds. No readiness enablement is justified.

## Reproduction and evidence

From a builder checkout containing the pinned runtime and this component:

```sh
/home/ubuntu/otpu-venv/bin/python -m pytest -q tests/test_qwen3_tts_audio_front.py
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_audio_front_acceptance.py --data /home/ubuntu/qwen3-tts-waveform-20261023-01/data --output /path/to/new-receipt.json --integrated
```

The saved jobs are terminal; inspect existing receipts before any rerun. The
evidence directory contains unmodified final receipts, per-frame metrics,
requests, process results, test log, official LFS metadata, acquisition/source
audit manifests and the post-run source inventory. Six acceptance-start code
hashes and per-row receipt hash match locally. All 49 component/fixture files
match the builder (one unused full-receipt copy has an explicit path alias), and
95 runtime files match Git revision `30100afb`. Source hashing is provenance,
not evidence of physical execution.
