# Dense device decode and matched wall-time profiling

`otpu-chat --fpga-only --backend board` is an opt-in contract for the qualified dense Qwen3
and Gemma3 text paths. The default behavior is unchanged. It rejects missing resident run
arguments, device generation/chaining, input tables or generate programs, unsupported
sampling, host embedding/PLE rows, expert offload, host services, convolution warmup, stateful
models, MTP and vision. Qualification is deliberately limited to these tested model types;
it is not a promise about an arbitrary future image or a loaded FPGA's performance.

The final prompt token is left unfed by the prefill stage. The device generate loop feeds it,
computes the first output and picks its ID, then feeds subsequent IDs back on the device. No
prefill logits are copied back for a CPU first-token pick. Resume, EOS, repeated cached
prompts, and following conversation turns retain the existing output semantics. The device
cycle/step counts now include that final prompt token's inference and first output; the
`Turn.device_first` field identifies this scope. Prefill counts exclude the one token handed
to decode. Wall decode still means the intervals between non-EOS output tokens, so it has
`generated_tokens - 1` intervals, not the device step count (which may include EOS).

This is not yet a single device start for an entire inference request. The host still loads
the image/programs, tokenizes and renders the prompt, launches prefill chunks, manages saved
contexts, initializes the device loop and reads/detokenizes output. The device executes each
chunk's model math. Sampling also prepares a run's random uniforms and initial repetition
factors in one host setup operation; subsequent picks and penalty updates are on-device. Use
greedy decoding when auditing a path without host random-number generation. Prompt chaining
and device RNG would be separate architecture work; neither is claimed here.

## Opt-in phase evidence

`--decode-trace PATH` collects events in memory and writes JSONL when the chat exits, including
on an exception in the chat run. It does not serialize or flush a file per token. The Python
interfaces also accept `phase_trace=events.append` on `Chat` or `Engine.generate_card`.
Callbacks must not raise. Tracing adds overhead and is for attribution, not an uninstrumented
performance claim. Run with a bounded output length; traces can grow throughout a long REPL.

The events distinguish request/prompt preparation, prefill chunks, generation preparation,
program/chain readiness, state/marker writes, dispatch, IMEM reload/start, output readback,
text callbacks, observed HALT, observed write-idle, and the final run counters. Engine chain
preparation currently combines compilation/cache lookup with chain-table writes; it is not a
measurement of compilation alone. Readback timestamps are host observation times, never
individual device emission timestamps. Multiple token IDs may share one readback. The full
raw ID sequence includes EOS; the chat summary separately records non-EOS IDs, displayed
token count, wall intervals, device steps, raw cycles, clock and termination reason. Text
callbacks report character counts, including empty deltas, separately from token IDs.
Host fallback is explicitly marked by `decode_begin.device_loop=false`, `selection=host`
and host-step events. A streaming host sampler can do work during the device step, so the
final host-pick interval alone does not measure all its CPU work. MTP-specific V/E/D counters
are not added by this trace; use its existing counters and qualify that separate path first.

No trace changes the meaning of `WR_IDLE`: write completion remains required before final
results can be read safely. No polling wait or synchronization was removed.

## Cold versus warm experiment, once hardware use is authorized

1. Pin the host revision, checkpoint/quantization, image/build ID, clock, prompt IDs, KV
   start position, output budget, greedy/sampling settings and EOS behavior. Keep ownership
   and the card's exclusive lock intact. Do not compare different builds or numerator scopes.
2. Run a fresh session with an explicitly empty program-cache directory and collect phases.
   Record whether background prewarm completed; a short warmup in bucket 1 does not establish
   that a later prompt's bucket is warm. Identify cold compilation from preparation evidence,
   not from a guessed per-token CPU cost.
3. Warm every bucket needed for the identical request; reset KV/conversation state while
   retaining programs/cache, and repeat the same tokenized request. Include output readback,
   HALT and write settlement. Compare raw IDs and EOS before comparing times.
4. Match first/last IDs and the exact same intervals. Report submit-to-first-ID TTFT,
   request-to-completion wall time, first-to-last non-EOS readback intervals, per-run device
   cycles/steps, and first-to-last text delivery separately. Attribute setup, readback,
   detokenization/flush and final settlement from their own events.
5. Repeat without tracing to bound its overhead. A no-op text callback versus normal text
   output is a separate A/B with identical IDs and request setup. Do not infer a production
   gain merely because one callback or compilation phase is faster.

Qwen3 multi-slot engines now prewarm the ordinary slot-0 generation program only for the
exact supported Qwen3 batch layout. Tests rebuild that layout in the spawned prewarm process,
compare greedy and sampled program bytes for two buckets, require disk-cache reuse, and
compare ISA output across position 256. This removes a known eligibility gap in prewarming;
no measured FPGA speedup is attached to it.

Speculative MTP decoding remains a separate Qwen3.5 path with suitable MTP tensors. This
change does not add a Gemma3 drafter, speculative verification or a speculative TPS claim.
