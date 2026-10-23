# Checked PersonaPlex duplex residency candidate

This software-only candidate targets the existing shared image
`agfi-0001de1dd3df8e21e`, profile `f2-hbm:p4rl8a64c150`, architecture
`dd6b92a6`, using runtime revision
`18fb978033799f0f21258f0e6e085c0360322d96`. The builder verified D=128,
one slice, 32,768 instruction words, SEQ_WC and a 4,294,967,296-byte mapping.
No physical execution, full-speech readiness or waveform-quality acceptance
is established by this plan.

The old four-packet A-region proposal is invalid: the input/output codecs now
own part of that region. It must never be used to overwrite resident codec
weights, convolution histories or KV state. The earlier five-packet estimate
also fails the exact runtime descriptors: nine INT4 layers need 995,033,088
bytes, exceeding the actual 992,505,856-byte page window by 2,527,232 bytes.
The retained failed preflight receipts document those rejected candidates.

## Physical ownership

`opentpu/llm/personaplex_duplex_layout.py` constructs the complete existing
input/output codec and depth mapping, copies every prefix span unchanged,
then reserves disjoint LM tables, controls, scratch, temporal KV and programs.
All intervals are end-exclusive. `check_spans` rejects overlap, alignment,
capacity and duplicate-owner errors; temporal aliases must match base and size.

| Owner | Interval or exact size |
|---|---|
| Native delay ring and controls | Allocated holes below `0x10000` |
| Persistent temporal INT8 KV | `0x10000–0x02110000`, 34,603,008 B |
| Existing depth KV | Retained in `0x02110000–0x02510000` |
| Activation scratch | Checked holes in `0x02510000–0x03510000` |
| Programs including WAITC data | `0x04000000–0x05000000` |
| FP32 LM embedding tables, one zero row each | 1,061,715,968 B from `0x05000000` |
| Voice replay reservation | 65,536 B, ending `0x44498000` |
| Depth embeddings and audio heads | Existing addresses retained |
| INT8 text head codes and scales | `0x539B8000–0x5B7D1400` |
| Temporal static coefficients/norm/mask | Checked fixed spans from `0x5D831400` |
| All input/output codec histories, KV and weights | Existing named spans retained |
| Exclusive temporal immutable page lease | `1,960,284,160–0xB0000000` |
| Depth B weights | Existing named spans retained |

The 74 conservative program slots use 12,124,160 of 16,777,216 bytes, leaving
4,653,056 bytes. Slots comprise 60 established codec/depth programs, six
temporal pages, four scheduler programs, embedding gather, text head,
temporal-to-depth handoff and reserved voice replay. Voice replay reservation
does not establish server prompt parity.

## Immutable temporal packets

Each layer prefix comes from the pinned runtime's actual quantized descriptor:
INT4 110,559,232 B; INT8 207,945,728 B. KV tails are excluded from refills and
relocated to persistent P. The precision rule remains INT8 layers0–3 and31,
INT4 layers4–30.

| Packet | Layers | Bytes |
|---|---|---:|
| 0 | 0–3 INT8 | 831,782,912 |
| 1 | 4–11 INT4 | 884,473,856 |
| 2 | 12–19 INT4 | 884,473,856 |
| 3 | 20–27 INT4 | 884,473,856 |
| 4 | 28–30 INT4 | 331,677,696 |
| 5 | 31 INT8 | 207,945,728 |

Only one packet owns the page lease at a time. The largest leaves 108,032,000
bytes free. Full layer payload traffic is 4,024,827,904 B per temporal frame;
this is a packed byte budget, not a DMA measurement or performance result.
Final norm and RoPE tables are immutable initialization payloads. Residual
activations and all layer KV remain resident across pages. Final RMSNorm runs
once, after packet5.

Refill only after the prior program halts and transport drains. Check the
whole packet or each contiguous layer block against its lease before writing.
Active-run large H2C traffic is not qualified. Host transport may submit
predetermined programs and copy immutable prepacked weights; it must not
compute activations, embeddings, codec values, logits or token selection.

## Cold duplex routing and reset

Pinned LMGen uses 16 audio streams plus text, initial IDs `[32000,2048×16]`,
a four-column ring and delays `[0,0,1×7,0,1×7]`. Input RVQ IDs map to global
slots9–16. All 16 depth heads sample on device. Keep raw samples, selected
feedback and committed cache values distinct: a supplied target overrides
feedback and cache commitment while the raw sample remains observable.

Every raw PCM call advances the input codec. Cold call0 primes LMGen and skips
LM compute; call1 computes without PCM emission; call2 first emits. Output
Mimi reads committed, delay-aligned slots1–8, not current raw depth IDs.
Four raw calls are therefore required for two consecutive emitted PCM frames.
A six-call oracle covers ring wraparound. This cold slice omits voice/text
prefixes; native server prompt scheduling and saved voice-cache restoration
remain separate integration gates.

Per-frame depth initialization must preserve codec and temporal conversation
state. Conversation reset restarts all stage counters/histories once; native
LMGen logical reset retains cache bytes and clears their validity. Do not call
standalone Engine wrappers that upload overlapping images or read interstage
activations. The final integrated runner must use one backend and resident
seams throughout.

## Acceptance limits

The input codec and RoPE tables admit at most128 raw PCM calls in this cold
candidate, hence127 temporal frames and126 emitted frames. The inherited
native Mimi ring-position/all-mask defect is retained and remains a quality
and longer-context gate. ISA F_BOUND guards are diagnostic checks, not physical
RTL enforcement; host orchestration must also enforce admission and ordering.

Component correctness and a checked map do not establish complete checkpoint
conversation quality. Required acceptance includes actual-checkpoint raw PCM
through the whole input/temporal/depth/output chain for consecutive frames and
reset, accumulated PCM and LM errors, native delay/prompt parity, measured
halted transport, and coordinated physical execution on the unchanged shared
image. Full speech stays disabled until those gates pass.
