# Resident Qwen3-TTS public decoder chunks

This increment closes the generated-code replay gap identified at `e0d68fea`.
It implements the public tokenizer's 300-frame chunks and 25-frame recomputation
through the decoder transformer. Codec IDs, context replay and neural features
remain resident in device memory. Continuous-mode diagnostics remain separate;
no waveform API or readiness flag is enabled.

## Source and device control

The exact model is `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, with Apache-2.0 Qwen source revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`. The [official tokenizer source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py)
calls `decoder.chunked_decode()` from its public `decode()` method. Each chunk
recomputes up to 25 preceding codec frames with fresh decoder state and positions,
then discards their outputs. The [previous transformer report](qwen3_tts_audio_transformer.md)
documents the source/configuration audit and separate continuous mode.

`ChunkedAudioFrameImage`/`ChunkedAudioFrameSession` add the following device work:

1. After each decoded frame, copy all 16 current IDs to a 25-row FP32 ring.
2. At zero-based frame 300, 600, etc., save the current generated IDs separately.
3. Zero the two pre-convolution history vectors and restart decoder positions.
4. Copy retained IDs in chronological order into the existing resident ID tensor
   and execute the decoder for context positions 0 through 24. Discard those
   feature outputs; do not run talker/predictor feedback for context replays.
5. Restore the current IDs, decode the new frame at position 25, record its IDs,
   and run the existing main-model feedback once.

Subsequent decoder positions advance to 324 before the next reset. Text/talker
position remains independent. The 72-slot KV rings are reset logically: each
visible slot is overwritten before use, and future stale slots remain masked.
The full reset also clears retained IDs, decoder state and inherited counters.
Execution failure latches until reset. The current ID ring is updated only after
successful decoder execution, preserving the oldest context until replay ends.

`DecoderChunkSpec` defaults to exactly 300/25. Smaller chunk/context settings are
explicit diagnostics, including context warm-up and zero context. The standalone
`step(ids)` accepts recorded test IDs; `step_resident()` instead consumes a
validated device producer without a host ID read/upload. The generated main path
uses the latter resident-ID architecture through its frame feedback dispatch.
RLD is not claimed to provide hardware bounds protection; producers must supply
the source-valid integer range, as enforced by the existing neural selection
policy and diagnostic fixture validation.

## Builder acceptance

All computation used the existing `opentpu-m8a-build`, isolated directory
`/home/ubuntu/qwen3-tts-resident-chunks-20261023-01`, existing Python/NumPy venv,
one BLAS/OMP thread, nice 19 and bounded jobs. No upstream executable model code,
new package, model download, instance, credential or network service was needed.
The cached tokenizer's full LFS digest and requested F32 tensor shapes were
verified again. The Mac performed source/Git and small receipt transport only.

* `chunk-tests-01`: **15 passed in 73.07 seconds**, including both earlier decoder
  suites. New tests check exact replay order, restored current IDs, warm-up when
  context exceeds early chunk history, zero context, failure/reset, generated
  main IDs/positions across shortened chunks, and native public boundary budgets.
* `chunk-acceptance-01`: **exit 0 in 1256.93 seconds**. A 602-frame diagnostic
  permutes retained real neural-prefix IDs and evaluates actual checkpoint
  decoder weights. It is not a newly generated utterance. The fixture is loaded
  once into a separate 38,528-byte device table; a diagnostic device-copy producer
  supplies current IDs. There are zero subsequent host current/replay ID uploads.
* Public boundaries at frames 300 and 600 replayed all **50 context frames**
  (800 individual codec IDs), and both current-ID restores matched exactly.
  Decoder caches/history were deliberately poisoned at each boundary. The input
  fixture table remained unchanged. Full reset replay of the first five frames
  was bitwise equal.
* The exact main checkpoint separately generated three `Hello.` frames twice,
  using explicit short 2/1 chunks to exercise neural-producer replay. All 48/48
  IDs per run matched the preserved FP32 path; current device IDs were restored
  and features replayed bitwise. This is not a native 300-frame main generation
  test. Its maximum output error was 0.013376946%.

| Maximum relative L2 over 602 emitted diagnostic frames | FP32 reference | FP64 reference |
| --- | ---: | ---: |
| Transformer output | 0.014718958% | 0.014722283% |
| Layer 7 output | 0.013089301% | 0.013101569% |

The unchanged acceptance bound is 3.5%. Each oracle recomputes the exact source
chunks independently. Source continuous and public chunked references differed
by up to **7.0068294%** in this corpus: 6.1179673% at frame 300 and 6.2234066% at
frame 600. Continuous output was not substituted. The first-chunk endpoint at
frame 299 matched exactly, providing a control for the mode comparison.

## Same-image resources and dependencies

The target remains AGFI `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, runtime
`30100afb075374381601a6e01d9f914a58dbec91`. Qualification uses the pinned target
and historical register receipt, not a live card query or another image's ISA.

The retained-code state adds **1,792 bytes** including alignment. Standalone
model residency is 107,372,416 bytes; combined main/front/transformer/replay
residency is **2,781,935,232 bytes**, leaving 1,513,032,064 bytes of 4 GiB.
Ordinary decoder dispatch uses 150 programs; a public boundary uses 3,903.
Including the preserved 395-program main schedule gives 545 ordinary and 4,298
boundary programs. Maximum tested program budgets are **1,928 IMEM words** and
**7,811 TMEM words**, within 32,768/65,536. These are resource/dispatch counts,
not FPGA latency or throughput measurements. The diagnostic fixture is not part
of the production model footprint.

The FP32-source-algebra arithmetic contract is unchanged. Host work is offline
packing/constants, deterministic frame/chunk scheduling, state-reset control,
transport and explicit test references. Replay copies, lookups and all neural
operations are device programs. No CPU/GPU/API fallback was introduced.

## Limits and evidence

The main path still has its qualified 512-position talker limit including text
prefill. Decoder chunking does not extend that limit. A complete native main
generation through a 300-frame boundary remains untested; the long default-mode
test and short actual-neural-producer test are separate evidence. Neither is a
speech-quality claim. Waveform stages, raw scaling, utterance quality and physical
acceptance remain incomplete. The historical main BF16 gate is still failed
(127/144 IDs, 6/9 frames, 7.51369% maximum conditional error); all failed receipts
are preserved. Full TTS remains disabled. Hardware/image/production were untouched
and service interruption was zero, against the unchanged 30-second maximum.

The [evidence directory](qwen3_tts_resident_chunks_evidence/) contains unmodified
receipts, all emitted-frame metrics, requests/results, runner and environment
provenance. Thirty-five source/fixture hashes match the review tree, 58 component
files and 95 runtime files match their snapshots, and the row digest matches.
The start manifest also caught three AppleDouble `._` metadata entries from Mac
tar transport. They were hashed only, never imported; their hashes match the
original overlay archives and are identified separately in local verification.
One unused copied main receipt has an explicit path alias. Source verification
does not imply physical execution.

```sh
/home/ubuntu/otpu-venv/bin/python -m pytest -q tests/test_qwen3_tts_audio_front.py tests/test_qwen3_tts_audio_transformer.py tests/test_qwen3_tts_audio_chunks.py
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_audio_chunks_acceptance.py --data /home/ubuntu/qwen3-tts-resident-chunks-20261023-01/data --output /path/to/new-chunk-receipt.json --integrated
```

Saved jobs are terminal. Inspect existing outcomes before any rerun; neither job
was duplicated when the control environment briefly disconnected.
