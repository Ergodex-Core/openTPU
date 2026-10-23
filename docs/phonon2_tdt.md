# Phonon-2 device LSTM, TDT and UTF-8 path

This increment preserves `cf6bb28e` and adds a device decoder after its complete
learned encoder, using the same `f2-hbm:p4rl8a64c150` profile, CFG `dd6b92a6`,
and parent-designated AGFI `agfi-0001de1dd3df8e21e`. It adds no opcode or image.
The waveform frontend and physical end-to-end acceptance remain missing.
Inferlab stays disabled, with no CPU/GPU serving fallback.

## Exact source policy

The checkpoint remains `FermionResearch/Phonon-2` revision
`fb09063c9dc8e130d97b3a2191ed5aed6b71bb31`, container SHA256
`4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`.
The [original source/license audit](phonon2_asr.md) and NVIDIA/Fermion attribution
apply: CC-BY-4.0 weights/tokenizer data and Apache-2.0 code. No checkpoint weights
were downloaded to the Mac or committed. The small vocabulary asset is an
ID-ordered extraction of the pinned tokenizer data with its source hash and
documented changes, rather than learned weights.

The selected decoder is the inspected [Transformers source at ea52711b](https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/generation_parakeet.py),
with explicit greedy generation settings. The [new semantic policy](phonon2/tdt_semantic_policy.json)
resolves the previously recorded decoder disagreement; the earlier frontend and
Conformer policies and receipts retain their original historical scope.
The numerical audit is eager FP32; it makes no equivalence claim for Fermion
CUDA's bfloat16 execution or an MLX/NeMo runtime.

* Start from token 8192 (blank), zero hidden/cell state, uninitialized predictor
  cache, and encoder frame zero. The embedding is 8193 × 640, followed by two
  640-unit LSTM layers, gate order input/forget/candidate/output, and a 640 × 640
  decoder projector. INT6 rows retain their exact decoded values/scales.
* Consume the initial blank once and commit its LSTM/cache output. Thereafter,
  a blank input reuses the cached output and state; a nonblank input updates both
  layers and the projector cache before joint selection. This is the inspected
  Transformers cache contract. Fermion CUDA instead commits a candidate state
  on a nonblank joint emission; intermediate raw states at blank boundaries
  therefore should not be compared as though the cache representations matched.
* Project the 1024-wide encoder output to 640. Joint logits are a biased INT6
  linear over ReLU(encoder projection + cached decoder projection), with 8198
  outputs. Select token and duration separately: first 8193 token classes, then
  five duration classes mapping to `[0,1,2,3,4]`. Ties choose the first index.
* Blank plus duration zero advances one frame. Nonblank plus duration zero emits
  a token and stays at the same frame. There is **no ten-symbol forced advance**.
  Fermion CUDA does force such an advance; adopting it would change this policy.
* Stop at the valid encoder length or configured EOS token 3. The default bound
  is `10 * stored_encoder_frames - 1` decoding steps, corresponding to the
  Transformers default maximum length including the initial blank. Exhausting
  this bound returns status 3 (incomplete/error), never a successful transcript
  and never an invented frame advance. Status 1 means encoder exhaustion;
  status 2 means EOS. Partial traces/text remain audit data on incomplete runs.

Trace rows are `[token_id, effective_duration, input_frame_index, emit_nonblank]`.
They describe joint decoding steps and omit the separate initial blank input.
Token frames/durations are exposed; word timestamps are not implemented.

All state, cache, frame, token/duration selection, active masking and termination
decisions are device operations. The host dispatches three static programs and
does not read tokens to drive the decoding loop.

## Text semantics and output I/O contract

The pinned tokenizer has a Metaspace decoder with replacement `▁`, prepend
scheme `always`, and no cleanup of tokenization spaces. TDT decoding does not
merge repeated IDs as CTC would. Configured special IDs 0 (UNK), 2 (PAD), 3
(EOS) and 8192 (blank) are omitted. Other control-looking pieces are retained,
matching this configuration rather than Fermion CUDA's broad special-piece
filter and final `.strip()`.

The inspected [Parakeet tokenizer](https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/tokenization_parakeet.py)
and backend post-initialization mark configured named special tokens, including
EOS, as special. [Metaspace's decoder](https://github.com/huggingface/tokenizers/blob/v0.22.1/tokenizers/src/pre_tokenizers/metaspace.rs)
removes replacement characters from the first retained token and replaces them
with spaces thereafter. An empty first piece still counts as that first token.
The device tracks this explicitly. The native installed Tokenizers comparison
covers every vocabulary entry in first and subsequent position, including
multilingual UTF-8, without importing upstream model code.

The device selects the final UTF-8 bytes from immutable tables and writes
32-byte scatter/gather segments plus their exact byte lengths. Each decoding
step owns one segment; blank/special steps have length zero. The host may copy
or coalesce these preselected byte ranges for ordinary response/file I/O. It
must not perform token lookup, Metaspace conversion, special filtering, trimming,
or any model computation. This representation avoids unsupported unaligned
single-byte DMA stores; a contiguous packed UTF-8 buffer on device is not
claimed. Immutable vocabulary preparation is independent of audio/tokens and is
equivalent to checkpoint preparation.

## Device implementation and integration

`opentpu/asr/phonon2_tdt.py` emits three programs: reset, valid-frame encoder
projection, and a bounded autonomous TDT loop. The loop uses existing RLD,
register-count LOOP, FP32 VOP/ARGMAX, QACT, INT8 MM and register-indexed LD/ST.
It needs at most three nested loops, below the existing four-level sequencer
limit. A device 0/1 loop skips cache recomputation on blank and all computation
after completion. All computed register indices carry simulator-checkable
bounds. Padded encoder frames are neither projected nor selected.

Affines use the encoder's device residual-QACT correction. Embedding lookup
loads an immutable FP32 expansion of the exact INT6 table; expansion is weight
decoding, independent of inference inputs. LSTM gate, cell and hidden arithmetic
execute on device. No native LSTM opcode, new image, GATHER or host token loop
is needed.

Encoder output is copied directly in device DRAM to decoder input before the
decoder's immutable prefix is staged. Decoder reset preserves that input while
clearing hidden/cell state, cache, trace, text segments, lengths and all control
words. The preserved encoder output SHA must match the previous encoder receipt
before its evidence is reused. Audit host reads never feed the decoder.

For the 30-second compile envelope (376 stored, 375 valid encoder frames), the
production-shaped decoder uses 37,360,640 graph bytes, a 34,636,672-byte immutable
prefix, 25,879 peak TMEM words, and at most 2,248 phase instruction words. Its
default step bound is 3759. The largest program plus wait table is 11,336 bytes
within the 160 KiB reserved slot. These are compile/resource checks with synthetic
values and official shapes, rather than a 30-second numerical result. Optional
audit-only state snapshots increase memory; serving does not need them. The
existing 160 KiB program/wait slot remains separate from graph memory.

## Reproducible common runtime

[tdt_runtime_dependency.json](phonon2/tdt_runtime_dependency.json) pins common
runtime commit `df69c0a9d970ba4f78b8ac6e979b41dc69d4f181`, all 17 imported runtime
source files, three relevant RTL source hashes, and the additional tokenizer
source URLs/hashes. The RTL inspection confirms register-fed loops, RLD
conversion and ARGMAX as source capabilities; it does not identify the live
card. The AGFI association remains parent-provided and physically unverified.

The preparer reads immutable local Git objects, verifies every hash before
writing a new source directory, and runs no hooks, setup code, network request
or FPGA operation:

```sh
python -B -m tools.phonon2_common_runtime \
  --prepare-from-git /existing/common-runtime-checkout \
  --destination /tmp/phonon2-common-runtime
python -B -m tools.phonon2_common_runtime \
  --verify-root /tmp/phonon2-common-runtime
python -B -m tools.phonon2_run_shared_tdt \
  --runtime-root /tmp/phonon2-common-runtime \
  --container /existing/ec2/model.fermion \
  --pcm docs/phonon2/speech_1688_crop.pcm \
  --wave /existing/ec2/speech_1688_full.wav \
  --output-dir /tmp/phonon2-tdt-qualification
```

Use an existing NumPy/Torch environment on the authorized EC2 host for model
qualification. Tokenizers is needed only for the exhaustive text-reference
test; no Transformers model package or remote-code execution is required.
Source extraction and tamper rejection were checked locally. Changed compiler,
ISA or runtime files require fresh qualification rather than reusing receipts.

## Validation and remaining gates

The shared-runtime regression passes 172 tests, with one skipped
because Transformers is absent. Tests include native built-in Torch LSTM parity,
per-step hidden/cell/cache comparisons, blank-cache reuse, blank duration-zero
progress, twelve consecutive nonblank zero-duration emissions with an incomplete
result, duration overshoot, EOS, masked encoder rows, repeated/empty/Unicode text
pieces, and bitwise reset after different recurrent history.

The real-checkpoint qualification on existing EC2 passed all three cases. The
full 7.06-second recording matches all 53 dense token/duration/frame decisions
and the exact UTF-8 text. Its maximum per-step hidden/cell relative L2 error is
`3.98615e-5` (0.003986%); maximum cache relative L2 error is `6.38976e-5`
(0.006390%). Both the short crop and full recording reproduce the
preserved encoder output hashes before device handoff. A short crop after the
full run reproduces its fresh hidden/cell/cache, trace, control state and text
bit-for-bit. The short crop has seven blank steps and empty text in both paths.
The transcript on the full recording is:

> I really was very much afraid of showing him how much shocked I was at some parts of what he said.

[tdt_validation.json](phonon2/tdt_validation.json) contains the source/runtime and
checkpoint record hashes, all traces/state comparisons, budgets, wait checks
and reset receipt. These are CPU ISA and native-Torch audit results, with an
offline waveform oracle. No physical inference result, WER, device frontend
parity or latency is claimed.

The real full-speech audit uses 46,667,648 decoder graph bytes and 2,376 maximum
phase words because it retains per-step state snapshots. It adds four dispatches
(device copy plus three programs) to the encoder's 682. The short case checks
1,884 instructions through the conservative wait dependency checker. The full
case executes the same wait-image format in the ISA model; its receipt records
that the wait checker was not separately run for that case. Physical scheduling
and throughput remain unmeasured.

Next compile the device PCM/preemphasis/STFT/mel/log/normalization path against
the retained waveform goldens, preserving duration, mask, dither and silence
policies. Final acceptance still requires coordinated card ownership, verified
shared-image capabilities, physical audio-to-transcript evidence and measured
scheduling within the user's 30-second maximum serving interruption. This work
does not access the FPGA, reload an image, synthesize RTL, change production or
enable Inferlab readiness.
