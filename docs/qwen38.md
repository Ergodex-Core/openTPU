# Qwen3.8-27B-INT4 on openTPU (RedHatAI/Qwen3.8-27B-INT4)

openTPU runs Red Hat's 4-bit [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
on Qwen3.5's code (`opentpu/llm/qwen35.py`, [qwen35.md](qwen35.md)): its hybrid DeltaNet /
attention layers, DRAM image, resident decode and Engine, split over four slices of the board
configuration ([gemma4.md](gemma4.md) "Slices"; the v4 plan's P4). New is the checkpoint
reader for compressed-tensors' packed int4 (`opentpu/llm/ctensors.py`) and the plumbing a
27-billion-parameter model needs on the simulator (four 4 GiB images in host memory). No new
hardware op, no new ISA: the MX MXU's int4x weights, int8 activations.

```sh
hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir models/Qwen3.8-27B-INT4    # 19.45 GB
python3 -m pytest -q tests/test_ctensors.py                           # the packed-int4 reader
# greedy decode on the ISA simulator, 4 slices, int4 layers, int8 head, context 512
PYTHONPATH=. python3 -m opentpu.host.chat --model models/Qwen3.8-27B-INT4 --backend isa \
    --slices 4 --greedy --max-new 8 --cap 512 --wformat int4 --head-format int8 \
    --prompt "Why is the sky blue? Answer in one sentence."
# an openTPU Lens profile of one decode token (the prompt's last step), the same formats
PYTHONPATH=. python3 -m opentpu.lens record model --model models/Qwen3.8-27B-INT4 --isa \
    --board --slices 4 --mhz 200 --wformat int4 --head-format int8 \
    --prompt "Why is the sky blue? Answer in one sentence." \
    -o build/lens/qwen38-27b-int4-isa.otpuprof
PYTHONPATH=. python3 -m opentpu.lens summary build/lens/qwen38-27b-int4-isa.otpuprof --slice 0
```

`build/run_chat.sh` and `build/run_lens.sh` (not in the tree: see below) ran these with
`OTPU_BUILD_JOBS=8` and the process tree's memory sampled.

## The model

The checkpoint is `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`, a vision-language
model); only its language model is loaded, `model.language_model.*` as `model.*`, as for every
Qwen3.5 (`qwen3.Weights` / `LazyWeights` skip `model.visual.*`). Its text config
(`Spec.from_hf` reads `text_config`): 64 layers, hidden 5120, MLP 17408 (SwiGLU), vocabulary
248,320, untied LM head, 262,144 positions, RMSNorm eps 1e-6, RoPE theta 1e7 on a quarter of
the head (`partial_rotary_factor` 0.25: 64 of 256), mRoPE interleaved (1-D for text).

- **Layers.** Three of every four are Gated DeltaNet (`linear_attn.*`: 48 value heads of 128
  on 16 key heads of 128, a 4-tap convolution, `A_log` / `dt_bias`, `in_proj_a` / `in_proj_b`
  gates), every fourth is full attention (24 query heads of 256 on 4 KV heads, gated: q_proj's
  12288 outputs are 24 x (256 q + 256 gate)). 48 + 16 layers.
- **Not loaded.** The vision tower (`model.visual.*`, 333 tensors), the MTP layer
  (`model_mtp.safetensors`, 849 MB: the drafter, [mtp.md](mtp.md) runs one slice only), the
  KV cache's fp8 scales `k_scale` / `v_scale` (one each per attention layer, unused: the KV
  cache is the image's int8).

The checkpoint's 19,452,788,048 bytes are `model.safetensors` (18,603,387,656) and
`model_mtp.safetensors`, as its `model.safetensors.index.json` states.

## The compressed-tensors import (`opentpu/llm/ctensors.py`)

`config.json`'s `quantization_config` is compressed-tensors' (`quant_method`
"compressed-tensors", `format` "pack-quantized"), one config group over every `Linear`:
weights `int`, 4 bits, `symmetric`, strategy `group` of 128, `actorder` "static". Every
quantized Linear (the MLP's gate / up / down, attention's q / k / v / o, DeltaNet's
`in_proj_qkv` / `in_proj_z` / `out_proj`: 400 matrices) is three tensors:

| tensor | dtype, shape | holds |
|---|---|---|
| `X.weight_packed` | int32 `[N, K / 8]` | 8 int4 codes a word, code i in bits 4i .. 4i+3 (the least significant first), packed along the input dimension K |
| `X.weight_scale` | bf16 `[N, K / 128]` | one scale per group of 128 input columns of each output row |
| `X.weight_shape` | int64 `[2]` | `(N, K)` |

and `weight = code * scale[row, column // 128]`. The codes are **offset-binary**, not two's
complement: the nibble is `value + 8` (a -8 is 0, a +7 is 15). That is the library's
`pack_to_int32` / `unpack_from_int32` (compressed-tensors 0.19.0,
`src/compressed_tensors/compressors/pack_quantized/helpers.py`: `offset = 1 << (num_bits -
1)`, `value + offset`, element i at bit position `i * num_bits`, `(unpacked - offset)`), and
the checkpoint agrees: the nibble histogram of `layers.0.mlp.down_proj` peaks at 8 and is
symmetric about it (counts 242,817 at 0 ... 13,782,409 at 8 ... 886,596 at 15), which two's
complement would show as a peak at 0 with a second at 15. `actorder` "static" is the
library's alias of "weight": the quantization order was permuted, the stored weight and its
groups are in the original column order (no `weight_g_idx` tensors: `names()` refuses a
checkpoint with them, or with zero points). The not-quantized tensors are bf16 as usual: the
embedding table and LM head (248,320 x 5120 each), the norms, DeltaNet's `in_proj_a` /
`in_proj_b` / `A_log` / `dt_bias` / `conv1d` / `norm`.

The reader presents every packed matrix as `X.weight`, fp32, dequantized when read and not
kept (`qwen3.Weights` caches tensors of at most 16 MB and the last large one; `LazyWeights`
keeps none): the dequantized fp32 model is 97 GB, the image build quantizes one matrix at a
time (0.3 .. 0.9 s each to unpack, numpy). A 27B model in fp32 on the host (Hugging Face's
`AutoModelForCausalLM`) would be 109 GB and was not attempted; the sanity check is the
dequantized statistics against the bf16 tensors beside them: mean 0, standard deviation
0.009 .. 0.017 (the bf16 LM head's is 0.0138, `in_proj_a`'s 0.0168), the 99.9th percentile of
|w| 0.03 .. 0.06 (the LM head's 0.051), a few outlier columns to 1.4 (group scales up to
0.17), 15.5% exact zeros (the code 0). `tests/test_ctensors.py` packs synthetic codes with
the library's `pack_to_int32` (ported verbatim) and reads them back through `unpack`,
`Weights` and `LazyWeights`.

## The format: int4x, re-quantized (not an exact import)

The MXU's 4-bit formats ([mx_mxu.md](mx_mxu.md), [quant.md](quant.md), `opentpu/quant.py`)
scale a weight row by one fp32 `Sw`, and under it each K-block of 128 by a power-of-two
offset `2^-dw` (a byte) with, for int4x / fp4x6, a 6-bit multiplier `m` in 1 .. 63 per
sub-block of 32: a sub-block's scale is `Sw 2^-dw m`. The checkpoint's scale is one bf16 per
group of 128 = one K-block, so an exact import needs the block's four sub-block scales all
equal to the group's bf16 value: `s_g = Sw 2^-dw m`. bf16 has an 8-bit significand, `m` 6
bits, and `Sw` is one per row across 40 .. 136 groups whose scales are arbitrary bf16 values;
the ratios `s_g / s_h` of two groups of a row would have to be ratios of 6-bit integers times
powers of two. They are not: over 6 sampled matrices (1024 rows each), 0.01% of the blocks'
scales are representable (`build/probe_requant.py`; nvfp4's E4M3 per 16 has 3 mantissa bits,
and E2M1 elements cannot hold the codes 5 and 7; int8x's per-block scale is a power of two
under the row's). **The import is not exact**: the dequantized fp32 matrix is quantized
again into int4x by the usual quantizer (`quant.quantize_mxu`, the row's `Sw`, the block's
`dw`, the sub-block `m` picked for the least squared error).

The double quantization's error, measured on the same 6 matrices (1024 rows of
`layers.0.mlp.down_proj`, `layers.0.linear_attn.in_proj_qkv`, `layers.3.self_attn.q_proj` /
`o_proj`, `layers.31.mlp.gate_proj`, `layers.63.mlp.up_proj`) against the checkpoint's
values: relative RMS error 4.63% (4.58 .. 4.77% per matrix), the largest single error 0.021
(on a value of 0.27), 82.5% of the int4 codes unchanged (the quantizer picks a finer
sub-block scale where a sub-block's largest code is below 7, so codes move with their
scale). For comparison, quantizing a bf16 tensor (`in_proj_a`) to int4x directly is a 9.25%
relative RMS error: the checkpoint's own quantization noise against its bf16 original (not
available) is of that order, and the re-quantization adds about half of it in quadrature
(sqrt(9.3^2 + 4.6^2) = 10.4%). The run below decodes sensibly with it.

## Fitting four slices

`Spec.check` refused the model at first: "an inner dimension exceeds ACT RAM" (the MLP width
17408 against the board's 128 blocks x 128 = 16384). The MLP kernel (`opentpu/kernels/mlp.py`)
holds a slice's `ffn / S` columns and runs the down projection in chunks of them, so the ACT
RAM never holds the whole width; the check now takes `ffn // S` (4352 on four slices; the
compiler and the simulator still refuse any MM whose K exceeds ACT RAM). The other checks
pass as they are: 4 KV heads and 248,320 vocabulary rows divide by 4, 48 DeltaNet heads by 8
(pairs per slice), 5120 and 17408 by 512.

Per-slice image bytes (`Spec.image` on a 1 TiB DRAM configuration, S=4, KV capacity 512,
8 prefill rows, lookup tables; `build/probe_image.py`):

| weights | head | embedding table | per slice | fits 4 GiB |
|---|---|---|---|---|
| int8 | int8 | int8, on the card | 8,378,909,184 B = 7.80 GiB | no |
| int4 | int8 | int8, on the card | 5,255,660,032 B = 4.90 GiB | no |
| int4 | int4 | int8, on the card | 5,104,433,152 B = 4.75 GiB | no |
| **int4** | **int8** | **int8, on the host** | **3,944,573,696 B = 3.67 GiB** | **yes** |
| int4 | int4 | int8, on the host | 3,793,346,816 B = 3.53 GiB | yes |
| int4 | nvfp4 | int8, on the host | 3,803,776,256 B = 3.54 GiB | yes |
| fp4 | fp4 | int8, on the host | 3,793,346,816 B = 3.53 GiB | yes |
| nvfp4 | nvfp4 | int8, on the host | 4,067,624,192 B = 3.79 GiB | yes |

The int8 embedding table (`Spec.embed` "int8": the fp32 one is 5 GB) is whole on every slice
(1.27 GB) and pushes the 4-bit images past 4 GiB; on the host (`embed_host`: the host writes
each token's int8 row into a slot of the image, [offload.md](offload.md) 5.9) the int4 layers
with an int8 head fit at 3.67 GiB per slice, the chosen run. The Engine falls back to the
host table when the image does not fit the configuration's DRAM; `board.sim_config` (which
sizes the simulators' DRAM from a probe image) now probes that way too, so the DRAM is the
card's 4 GiB per slice rather than 8. The per-sequence state at capacity 512 is 42.6 MiB per
slice: the KV cache (16 attention layers x 1 KV head x 512 x 256 x K and V, int8: 4.2 MB), the
DeltaNet states (48 layers x 12 heads x 128 x 128 fp32: 37.7 MB) and the convolution rings.

## Running it in 32 GB

Four 4 GiB images are 16 GiB of host memory once the ISA simulator has them (it adopts the
built arrays and grows them in place, `IsaBackend(adopt=True)`). The build's peak on top of
that was cut where it was avoidable: the two 248K x 5120 bf16 tables convert to fp32 by rows
(a chunk at a time into one fp32 array, instead of the bf16 copy and the fp32 one at once),
and `Weights` drops its cached large table once an image with device-side inputs is built
(5 GB that the host would never read again). Lens's `record model --isa` used to copy every
slice's DRAM for the traced step and copy it again into the timing machine (three sets of
four 4 GiB arrays); the traced step is now timed on the live DRAM (`isasim.Machine(adopt=True)`:
the timing machine's slices take the arrays as they are and leave them as the step would,
being the same simulator). The quantizer's worker processes take small jobs (4M elements);
`OTPU_BUILD_JOBS=8` on the 12-core host shared with two RTL simulations.

## The run (what ran, what did not)

The host was shared with two other agents' RTL simulations the whole time (load average 60
.. 250 on 12 cores, 17 .. 28 GB of swap in use). The 64-layer run and a truncated one ran as
follows; the numbers of the truncated one are the measured ones.

**64 layers, 4 slices (`otpu-chat ... --slices 4 --wformat int4 --head-format int8 --cap
512`).** `Spec.from_hf` accepted the config (text_config) and `Spec.check` the configuration
after the ACT RAM fix. The image built in 2,001 s (the 400 matrices dequantized and
re-quantized in 8 worker processes, the head int8, the embedding table to the host), 3.67
GiB per slice, weights "int4, head int8". The sampler reported a physical footprint of 17.7
GB for the process once built, 29.4 GB at the build's peak (the four images plus the fp32
LM head and its quantization in flight: over the ~20 GB this host could spare; with the
other agents' memory the process went into swap). The prefill of the 23-token prompt
(`<|im_start|>user ... <|im_end|> <|im_start|>assistant <think> </think>`: otpu-chat renders
the template with `enable_thinking=False`) then ran swap-bound: 41 CPU-minutes over 2 h 12
min of wall time (6 .. 50% of one core, the process in uninterruptible page-fault waits, its
resident set 0.1 .. 2 GB of the 16 GiB of images), no token out. It was paused (SIGSTOP) to
run the truncated model below, then resumed once the host was quiet: the prefill's first
token came out ("The") and the device-side decode loop then hung at its first step,
`slice 1: WAITW at pc 19 never holds`. That was a bug of the embedding table on the host on
more than one slice: the Engine's row server (`qwen3.Image.row_server`, `offload.RowServer`
on a `BackendDram`) wrote the requested row and the served word into slice 0 only, while
every slice of a tensor-parallel decode posts the request in lockstep and waits on its own
served word (the MoE models that use it run on one slice). `BackendDram(slices=...)` now
writes every slice; `tests/test_qwen38.py` runs a 2-slice tiny Qwen3.5 with the table on
the host through `generate_card` and fails the same way without the fix. The run was
started again with it (its numbers below when it completes).
A 64-layer decode token costs the simulator about 6x the 8-layer one below (24.3 of 25.6
GMAC), i.e. ~3 .. 4 min of CPU per token when its images stay resident.

**8 layers (two units of three DeltaNet layers and one attention layer: `--layers 8`,
`Spec.truncated`), 4 slices, the same formats, capacity 512.** The image built in 187 s (0.73
GiB per slice: the layers 95 MB each, the int8 head 318 MB, the tables), peak 8.05 GiB of
resident memory for the process tree during the build (the LM head's conversion), 3 .. 5 GiB
while decoding. The prompt runs' programs did not fit for this plan (`prefill._probe`: "no
prompt run fits bucket 1", every run size refused by a TMEM / ACT RAM / IMEM limit; the
64-layer plan's fit, so an open point of the truncation), so the prefill took the row-batched
route (`--no-prompt-runs`, 8 rows a run, as Lens's Engine does). The greedy reply to "Why is
the sky blue? Answer in one sentence." was ` .  .  ,` (8 tokens, stopped at `--max-new 8`):
not the model's text, 8 of its 64 layers. otpu-chat's numbers: TTFT 843 s (the 23-token
prefill, 3 runs of 8 rows), decode 0.03 tok/s = 33 s per token on the loaded host (the
process had 20 .. 90% of one core); 1,325 s in all with the build.

## The Lens profile

`lens record model --isa --board --slices 4 --mhz 200 --layers 8 --wformat int4 --head-format
int8` traced the step at position 62 (the chat template's default rendering, 63 tokens with
its "Reasoning effort is set to xhigh" system line and an open `<think>`; the step's token
`'\n' -> ' '`), 1,724 s with the build and the 62-token prefill:
`build/lens/qwen38-27b-int4-8layers-isa.otpuprof` (not in the tree). `lens summary --slice s`
for every slice (the split is symmetric, the four slices' counts are identical):

| per slice, 8-layer token | measured |
|---|---|
| cycles (analytic: each instruction back to back for its nominal work) | 6,152,035 = 30.76 ms at 200 MHz, 46.1 ms at 133.33 MHz |
| instructions | 4,552 (982 MM, 596 QACT, 1,646 VOP, 154 GATHER, 122 LD, 92 ST, 72 DSTEP) |
| DRAM port B (weights streamed) | 5,794,532 chunks of 128 B = 741.7 MB (MM 5,706,816; DSTEP 73,728; LD 9,450; ST 4,538) |
| DRAM port A (activations) | 8,924,040 chunks |
| DRAM bound (max of the ports) | 8,924,040 cycles |
| roofline fraction as Lens reports it | 145.1% (bound / cycles: the bound is port A's count, the analytic MM time port B's; see below) |
| unit busy | MXU 92.8% (5,706,816), DMA 3.3%, VPU 2.2%, QUANT 1.1%, COLL 0.7% |
| device tok/s, 8-layer token | 32.5 at 200 MHz, 21.7 at 133.33 MHz |

Where the MXU cycles go (slice 0, by kernel source): the int8 LM head 2,483,200 (8 MMs: the
slice's 62,080 rows x 5120 = 317.8 MB, 40% of the token), the MLP gate/up 1,392,640 (272 MMs),
the DeltaNet projections (`qwen35.project`) 737,280 (78), the MLP down 696,320 (544),
attention 204,800 (10), the DeltaNet flush 184,320 (24); DSTEP 147,456 (72 DMA), the VPU's
sigmoid / conv / norms under 40,000 each. The "145%" is an artefact of the analytic model on
this split: port A's count (the activation chunks every MM re-reads: 2 per weight chunk for
the MLP and projections at MCOLS=4) exceeds port B's, and the analytic MM time counts port
B's chunks only; the RTL would show the real overlap. Record with `--axi` on the RTL for a
measurement (not run: Verilator was off limits on the shared host).

**The 64-layer token, extrapolated** (the head once, 16 units of four layers instead of 2;
every unit is the same instruction stream, lfm2.plan's hardware loop): cycles 2,491,600 (head
and its DMA) + 16 x 1,830,218 (a unit: 3,660,435 for two) = **31.8 M cycles per token per
slice = 159 ms at 200 MHz (6.3 tok/s), 238 ms at 133.33 MHz (4.2 tok/s)**; port B 318 MB +
16 x 211.8 MB = 3.71 GB of weights streamed per slice per token (the int4 image's layers and
the int8 head: 14.8 GB over the four slices); ~36,000 instructions per slice. The 64-layer
image's own profile was not recorded (the second 16 GiB run on this host: the memory above).

## Open points

- The 64-layer run's text and per-token time on the simulator (above): needs ~20 GB of
  resident memory for the four images for a few hours of one core.
- The prompt-run programs of the 8-layer truncation do not fit (`_probe`), the 64-layer
  ones do; `--no-prompt-runs` for the truncated model.
- A compressed-tensors checkpoint with `actorder` "group" (g_idx), zero points, or a
  non-group strategy is refused by `ctensors.scheme` / `names`; "channel" is read but untested.
- `models/Qwen3.8-27B-INT4` is a symlink in the working tree (ignored), not a MODELS name.
