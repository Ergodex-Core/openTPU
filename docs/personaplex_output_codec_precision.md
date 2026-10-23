# Optional resident output-codec precision

This increment follows `bff8bbe1` and targets the unchanged upstream RVQ and
Mimi decoder precision gap. It is CPU/ISA qualification on the exact shared
profile, not physical inference or complete PersonaPlex. Defaults, previous
receipts and failed numerical results are retained.

## Source-faithful opt-in computation

`MimiRVQSpec(projection_precision="fp32")` preserves normalized codebooks,
lookup IDs and the original semantic/acoustic accumulation order. Its two
256-to-512 projections load FP32 coefficient rows and reduce the resident
branch vectors using existing VPU instructions. The branches and their sum
remain resident. The original INT8 projection path is unchanged by default.

`MimiDecoderSpec(precision="fp32")` selects FP32 projection matrices,
activations and explicit K/VT caches. LayerNorm, RoPE, layer scale, causal
upsampling and erf-form GELU approximation retain their prior source semantics.
Both tokens append to the native ring before either query attends, with the
same frozen masks and source `delta<=0` ring-position defect. Q/K use the same
offline adjacent-pair-to-half-split row permutation. The decoder frame counter
and slot/mask/RoPE selection are device-resident.

Attention reads at most 32 context rows at once. K tiles are [32,head_dim]
and V tiles [head_dim,32]; scores and probabilities use small resident scratch
arrays. Global max/sum softmax remains in device FP32 arithmetic. This avoids
retaining whole-context cache tiles in TMEM without changing the image or
moving model arithmetic to the host. FP32 caches use the actual head dimension
64 rather than quantized cache padding 128, and are initialized finite once at upload.
Logical reset retains masked old KV and resets the stream/overlap state as the
source does. Matrices stream coefficient rows through argument-free programs.

## Reproduction and declared gates

`tools/personaplex_output_codec_check.py --decoder-precision int8|fp32` consumes
the pinned two-frame generated audio-ID fixture. It executes only resident
RVQ → upsample/eight decoder layers → all-FP32 SEANet, with no LM/depth rerun.
The fixture IDs and completed-depth marker are initial diagnostic inputs;
there are no CPU activation/weight reads or writes between model components.
The native cached checkpoint modules are offline references only.

The checker requires the preserved baseline/arrays/ancestry, native contracts,
source hashes and shared target/runtime pins. It compares native accumulated
PCM bit-exactly to the old reference. Native fresh-stream replay, original
lookup sums, every resident layer/PCM boundary, raw decoder KV/control and
PCM persistent state are checked after reset. Progress is saved after each
program so a long CPU simulation remains inspectable.

The existing gates are unchanged and apply to aggregate AND each frame:

| Scope | Relative L2 | Raw RMS error | Raw peak absolute error |
|---|---:|---:|---:|
| SEANet on identical decoder inputs | ≤1e-4 | ≤1e-6 | ≤1e-5 |
| Accumulated RVQ/decoder/PCM | ≤1e-3 | ≤1e-5 | ≤1e-4 |

Raw PCM remains mono FP32 with no gain, clipping, normalization, resampling or
integer conversion. A gate pass is limited to this fixture, and cannot establish
perceptual quality, speech intelligibility or complete model acceptance.

## Numerical receipts

The corrected full-FP32 experiment completed two consecutive frames (3,840 raw
PCM samples) and their logical-reset replay. Both fixed numerical gates passed
in aggregate and for each frame:

| Scope | Relative L2 | Raw RMS error | Raw peak absolute error | SNR |
|---|---:|---:|---:|---:|
| RVQ projected latent | 2.15107662e-7 | 7.37694381e-7 | 3.81469727e-6 | — |
| Decoder on identical RVQ input | 4.24953138e-7 | 2.39626050e-7 | 4.76837158e-6 | — |
| SEANet on identical decoder input | 3.43101477e-6 | 1.11731688e-8 | 4.93600965e-8 | 109.29 dB |
| Accumulated RVQ → decoder → PCM | 3.61053741e-6 | 1.17577893e-8 | 5.91389835e-8 | 108.85 dB |

Accumulated relative L2 decreased from `0.0687144346001231` in the preserved
`bff8bbe1` receipt to `0.0000036105374127199477`. The quiet second frame has
relative L2 `1.1224604643557407e-5`, RMS error `8.557191399355841e-9`, peak
absolute error `4.260800778865814e-8`, SNR `98.99657893815466 dB`, and error RMS
`-161.35337508008217 dBFS`. These waveform metrics use full scale 1 without
normalization. No clipping or gain remedy was used.

Lookup branch sums, native accumulated reference replay, all captured outputs,
final raw decoder KV, decoder control and PCM persistent state replayed exactly
under their respective bit-equality checks. Intercomponent implicit host reads
and writes were zero. There are 50 saved diagnostic arrays. All 138 preserved
baseline/ancestry hashes remained unchanged. The checker hashes consumed payloads
and pinned metadata/source; it does not claim an independent whole-checkpoint
payload digest (`checkpoint_payload_hashes_verified` remains false).

The preferred builder passed 81 checks in 2.26 seconds and compiled both
production layouts. F2 CPU passed 149 aggregate regressions in 22.69 seconds;
the checkpoint/ISA checker took 552.0014738619793 seconds CPU wall time, and its
wrapper 576.5184267869918 seconds. Physical-open guards recorded zero attempts.
All 147 runtime/overlay hashes match the builder, F2 and frozen sources; manifest
SHA256 is `a1be571da8deb6db75a50caf80d959f6cc91ca8ea88a65dd552fddda40ddd601`.
These are simulator wall times, not FPGA frame latency or throughput.


The first RVQ-only experiment completed both frames and their reset replay,
then failed while reporting decoder-layer errors: the native helper returns
`[frame, layer, token, hidden]`, but the checker indexed it as layer-first.
Its saved partial receipt establishes RVQ relative L2 error of
`2.1510766219454524e-7` and unchanged-INT8 decoder error of
`0.01405383222797485` on identical RVQ inputs. All three reset checks passed.
That failed reporting run did not persist PCM arrays or an accumulated PCM
metric, so it cannot establish an accumulated gate result. The corrected checker
has a regression for this axis order and persists first-pass arrays/state before
reset or reporting. The failed receipt remains unchanged.

An independent 24-frame/48-token test uses context 40 padded to 64, exercising
two attention tiles, ring wrap, raw K/VT comparison and logical reset. Maximum
absolute output error against independent NumPy equations was
`2.7159492832495857e-7`; output and final raw KV replay were bit-exact after reset.
This is CPU/ISA evidence, not measured board behavior.

The source ring-position defect is retained explicitly rather than corrected
under a precision change. At the currently accepted API boundary `context=2`,
it can leave the first query entirely masked and produce a nonfinite softmax.
This boundary is not qualified. The checkpoint contract uses context 250; the
independent longer-context probe uses 40. A source-semantics correction or an
API rejection of the two-entry boundary is a separate compatibility decision.

## Shared-profile costs

Exact target `f2-hbm:p4rl8a64c150`, architecture hash `dd6b92a6`, image context
`agfi-0001de1dd3df8e21e`, runtime
`18fb978033799f0f21258f0e6e085c0360322d96`. Use the committed reproducible
builder and manifest; the old-runtime branch alone is insufficient.

FP32 RVQ occupies 17,842,176 bytes, with 17,825,792 weight bytes, and one 984-word
program. It adds 778,240 bytes versus INT8 projections. With the INT8 decoder
and all-FP32 SEANet, combined named occupancy is 1,734,545,892 bytes.

With FP32 decoder matrices/KV, decoder occupancy is 109,860,868 bytes,
weights 100,769,792 bytes, raw KV 8,388,608 bytes. Its upsample uses 128 words;
seven layer programs use 21,728 words each and the last 21,744. Combined named
occupancy is 1,814,090,212 bytes, adding 79,544,320 bytes over the RVQ-only
candidate and 80,322,560 bytes over `bff8bbe1`. Highest mapped extent remains
4,273,471,488 within 4 GiB; all 27 programs fit 32,768-word IMEM individually.
B slack remains 21,495,808 bytes. Decoder/SEANet state ends at 1,604,055,040,
below the existing `0x60000000` reservation. Temporal-weight residency remains a
separate paging/chaining gate.

Existing-image compilation required resolving explicit broadcast/reshape API
use and whole-context TMEM exhaustion. Independent tiny decoder equations
passed before the production-capacity failure; context tiling then cleared the
production compile. Those failed preflight receipts are preserved, not replaced
by later successes. No new opcode, RTL, synthesis or image is required by this
software layout. Hardware performance is unmeasured; additional FP32 memory,
VPU work and larger programs can increase latency substantially.

Preferred `opentpu-m8a-build` runs compatible tests/capacity with existing
NumPy dependencies. It lacks existing Torch/Moshi, so actual native checkpoint
comparison uses the existing F2 CPU environment/cache. All numerical execution
is on EC2, one thread and nice19 with affinity/headroom guards. Mac only handles
source/Git/SSH/AWS control and stdlib syntax/hash checks. No installs, new paid
resources, hardware/image/service changes or production mutation.

After the coherent output-codec numerical gate passes, the missing input Mimi
codec belongs in a separate increment, followed by complete scheduling
integration. Remaining full-speech gates also include temporal paging/resident
handoff, prompt/voice/delay state, streaming beyond 128 code frames, qualified
same-image BoardBackend/WAITC transport and coordinated physical end-to-end
acceptance. Full speech stays disabled and there is no serving CPU/GPU fallback.
