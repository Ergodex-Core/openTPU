# Optional FP32 initial and transposed SEANet convolutions

This is a software/ISA precision increment on the exact shared Gemma profile.
It is not physical FPGA or complete PersonaPlex inference. Defaults and earlier
INT8/head-only/residual-only receipts remain unchanged. It adds independent
`MimiPCMSpec.initial_precision` and `transpose_precision` choices, each `int8`
by default or explicitly `fp32`, alongside the existing residual/final options.

## Arithmetic and source streaming semantics

The initial K7 convolution retains its causal six-row history, including when
a two-row input frame is shorter than that history. Each coefficient row is
loaded as FP32 and reduced against the tap-major FP32 input window through
existing VPU multiply/reduction instructions. The initial convolution has no
ELU; subsequent source activations are unchanged.

Each ConvTranspose native tensor `[in,out,tap]` is packed into coefficient row
`tap*out_channels + channel` without quantization. The FPGA program loops over
those resident rows, computes a reduction against the activated input, adds
the channel bias and writes the raw tap/channel matrix. The original streaming
overlap follows afterward: the first row has no prior partial, later rows add
saved biased tail minus one bias, and the raw upper kernel half becomes the
next partial. A resident validity word selects that operation. No host branch
or tensor calculation is added. Reset clears validity and partial/history state.

Raw FP32 PCM has no clipping, gain, normalization, resampling or integer
conversion. Optional FP32 arithmetic can be slower than INT8 matrices; no
physical latency or real-time claim follows from compilation or CPU wall time.

## Declared numerical checks

The opt-in checker `tools/personaplex_pcm_precision_check.py --all-convolutions`
replays only the preserved INT8 baseline and the all-FP32 SEANet candidate.
The ordinary CLI still selects its original three precision candidates. This
avoids rerunning the completed residual-only experiment. Both use the saved
two-frame decoder outputs and existing cached codec; the LM is not rerun.

Before measurement, the new engineering acceptance was set to:

| Scope | Relative L2 | Raw RMS error | Raw peak absolute error |
|---|---:|---:|---:|
| SEANet on identical decoder inputs | ≤1e-4 | ≤1e-6 | ≤1e-5 |
| Accumulated native RVQ/decoder through PCM | ≤1e-3 | ≤1e-5 | ≤1e-4 |

Each bound must hold for both the aggregate and every frame. Missing relative
L2 for zero-reference signals fails this particular agreement check rather
than silently passing. Reset replay must be bit-exact across all 15 stage
outputs, PCM and persistent state, with no implicit interstage host transfers.
The checker fails if all-FP32 SEANet misses its local limits. The accumulated
codec check is reported separately: it includes upstream RVQ/transformer
operators unchanged by this increment and can still block output-codec
acceptance. Neither check establishes listening quality or distribution-level
model acceptance on this short fixture.

Checkpoint job `1fc2fdf2-c9ba-4b0d-92ea-0ed93a0bcd57` completed successfully.
The F2 aggregate suite passed 132 tests in 18.70 seconds, with zero physical
opens. The two-frame component checker took 457.5623 seconds CPU wall time,
478.0566 seconds including tests. These are diagnostic CPU times, not measured
FPGA milliseconds per frame or audio frames per second.

All-FP32 SEANet on identical decoder inputs achieved relative L2
3.5307980e-6, raw RMS error 1.1605849e-8 and peak absolute error 5.5879354e-8.
Its quieter second frame had relative L2 1.0363434e-5, RMS error 8.0282652e-9
and peak error 2.4912879e-8. Aggregate and both frames passed every declared
SEANet bound. Initial convolution local relative L2 is 7.55e-7; the four
transposed convolutions are 5.25e-7/1.85e-7/2.06e-7/1.11e-7. The largest
residual-block local error is 1.03e-6. This closes the isolated SEANet numerical
gate on the saved fixture, without claiming audio quality or full-codec acceptance.

Accumulated native RVQ-through-PCM relative L2 improved from the historical
INT8 0.1602663172 and previous residual-plus-head 0.0944946988 to
0.0687144346. RMS error is 0.0002237700, peak absolute error 0.0016133395,
SNR 23.2590 dB and error RMS -73.0040 dBFS. The quiet second frame is
0.0556949852 relative L2, RMS error 0.0000424596, peak error 0.0001877915,
SNR 25.0837 dB and error RMS -87.4405 dBFS. These still fail the separate
accumulated-codec gate. No PCM rescaling was used to improve agreement.

A new comparison between the two preserved native references found an
upstream relative L2 of 0.0687143629 (RMS error 0.0002237697). This is native
SEANet driven by ISA decoder inputs versus native RVQ/decoder inputs, so it
isolates the error entering SEANet. Its near equality with the final candidate
confirms the remaining accumulated error is upstream on this fixture.

Historical actual-checkpoint decoder receipts already quantify that gap:
RVQ latent relative L2 0.0153765324, eight-layer decoder output 0.0270071256
accumulated, and decoder-only 0.0139889176 on identical RVQ latents. These are
historical measurements, not freshly rerun inference. Source code still uses
INT8 semantic/acoustic RVQ projections and INT8 decoder matrices, activations
and KV. The smallest next output-codec slice is the two 256-to-512 RVQ
projections, followed by a separately isolated decoder precision check. The
full output-codec gate must remain open before the input-codec/scheduling work
can be presented as the next integration stage.

Both new candidate streams replayed all 15 stage outputs, PCM and persistent
state bit-exact after reset. INT8 matched historical stages, PCM and state
exactly; all-FP32 state intentionally differs but replays exactly within that
candidate. Native full/forced-stage fresh-stream replays were exact and 138
baseline hashes remained unchanged. Each component frame had a 4096-byte
fixture write before execution and no implicit host read/write among its 14
programs. Diagnostics followed execution.

The all-FP32 raw range was -0.0150311617 to 0.0188075267, RMS 0.0032870332;
zero of 3840 samples exceeded ±1. No clipping, gain or scaling was applied.
All 143 runtime/overlay source hashes match the preferred builder and actual
F2 run; manifest SHA256 is
`336439f43864bac47fa429b65b14d6d86aeb6bb7952cc2d9069efab3c7fe744f`.
Earlier receipts and the unchanged defaults are preserved.

## Resource costs and reproducibility

Exact target `f2-hbm:p4rl8a64c150`, hash `dd6b92a6`, image context
`agfi-0001de1dd3df8e21e`, runtime
`18fb978033799f0f21258f0e6e085c0360322d96`.
Use the committed reproducible runtime builder and dependency manifest;
103 runtime + 40 PersonaPlex files are verified. The previously published
old-runtime branch alone is insufficient.

All-FP32 SEANet occupies 66,963,864 bytes, including 58,950,656 weight bytes,
59,016,580 immutable payload bytes and unchanged 46,612 persistent reset bytes.
It adds 44,028,920 bytes over the INT8 layout and 41,904,128 bytes over the
previous residual-plus-final remedy. Its 14 program sizes in words are
368/424/408/432/472/408/432/472/408/432/472/408/432/304.
Combined named occupancy is 1,733,767,652 bytes. Highest aligned extent remains
4,273,471,488 within 4 GiB; all 27 programs individually fit 32,768-word IMEM;
B slack remains 21,495,808 bytes. No new opcode, RTL or image gap is demonstrated
for these compiled components. Simultaneous temporal-weight residency and
transport latency remain open.

Preferred `opentpu-m8a-build` executed the 64 compatible CPU/ISA checks,
including independent NumPy global-time convolution formulas, payload
orientation, short-frame history, four transpose stages, biased overlap,
reset and per-frame acceptance enforcement. The production layouts compiled.
The builder's existing environment lacks Torch/Moshi; the actual-checkpoint
native comparison used the existing F2 CPU environment and cached codec.
No dependency installation, new resource or Mac model download/execution.
All numerical jobs use one thread, affinity and nice 19, with physical opens
blocked. Mac serves only as source/Git/SSH/AWS control bridge.

A coherent output codec must clear its declared numerical gate before moving
to the missing input Mimi codec and full scheduling integration. Other
full-speech gates remain temporal paging/resident handoff, prompt/voice/delay
state, extension beyond 128 code frames, same-image transport and coordinated
physical end-to-end acceptance. Full speech stays disabled. No production,
card, image, service, synthesis or paid-resource changes occurred.
