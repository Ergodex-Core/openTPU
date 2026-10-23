# Qwen3-TTS: exact-checkpoint software acceptance

This follow-up validates the resident embedding/projection/talker-layer-0 slice
from `cca65cd5` using actual tensors from the exact official
**Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice** checkpoint. It changes no runtime kernel,
serving route, portal metadata, RTL, image, or synthesis configuration.

## Provenance and diagnostic scope

- [Pinned model revision](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5):
  `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Official `model.safetensors`: **1,811,626,576 bytes**;
  SHA-256 **`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`**.
  The whole file was verified against the official Hugging Face API's LFS hash,
  and the 49,224-byte header matches the previously pinned header hash.
- [Source revision](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e):
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`, `qwen-tts 0.1.1`, Apache-2.0
  model/source license. The earlier source/function hashes remain in
  `tests/fixtures/qwen3_tts_talker_contract.json`.

Materialization and execution use the existing EC2 instance and disk, under
`/home/ubuntu/qwen3-tts-checkpoint-20261023-01`. No weights or large arrays were
downloaded to the Mac or committed. No packages, credentials, services, or
compute resources were added. The fetch runs anonymously against the pinned
public data URL; no remote executable model code is downloaded or imported.
The runtime validation command makes no network or device transport calls.

`tools/qwen3_tts_checkpoint_acceptance.py` verifies the entire pinned file again
before reading tensors. It decodes BF16 to FP32 by exact bit extension and
packs all 17 real tensors used by the slice, including the full 151,936-row text
and 3,072-row codec embedding tables. Their 336,336,128 stored elements are not
substituted, selected down to a small vocabulary, or replaced with random
weights. The diagnostic explicitly permits a 2 GiB host packing budget.

The model computation under test is the unchanged MX INT8/FP32 ISA program.
CPU computation belongs to the explicit diagnostic simulator and independent
references. It is not a production backend or a host fallback. The full
text-to-waveform entry point stays gated.

## Reference definitions

The existing NumPy reference evaluates independently written FP64 algebra on
the exact decoded BF16 checkpoint values. The new, locally reviewed PyTorch
reference reproduces eager BF16 cast locations: RMS variance in FP32 followed
by the source's BF16 casts, BF16 linear/SiLU/residual operations, FP32 softmax
cast back to BF16, and equal-axis rotate-half RoPE. Neither reference imports
Qwen's model classes or its generation wrapper.

The BF16 reference uses CPU eager PyTorch, so it does not claim bitwise equality
with the official CUDA/FlashAttention example. FP64 results separate the
changed INT8/FP32 arithmetic from BF16 rounding. Each comparison reports relative
L2, maximum absolute error, RMS error/reference RMS, and cosine similarity.

The inherited 0.035 relative-L2 diagnostic threshold is kept unchanged and
reported explicitly. It was established for the prior synthetic tests; it is
not a calibrated speech-quality threshold. A numerical threshold failure exits
with status 2, while cache/masking/reset invariant failures exit with status 1.
The runner retains numeric results in both cases.

## Measured checkpoint results

**Passed:** 36 regression/reader/reference tests in 4.78 seconds, followed by the
checkpoint run completing in 64.91 seconds with diagnostic exit code 0. This is
CPU diagnostic elapsed time including verification, packing, references and
two ISA machines; it is not an FPGA latency or throughput measurement.

All 260 positions satisfy the unchanged 0.035 relative-L2 threshold at all four
stages, for both references. Maximum observed errors:

| Stage | ISA vs FP64 | ISA vs eager BF16 |
|---|---:|---:|
| Text projection | 0.7853% | 0.8062% |
| Assembled input | 0.7850% | 0.8016% |
| Attention residual | 2.3218% | 2.6163% |
| Layer output | **2.5820%** | **2.8676%** |

For layer output, mean/p95 error is 1.5211%/2.1008% against FP64 and
1.5978%/2.1722% against BF16. Minimum cosine similarity is 0.999667/0.999592;
maximum absolute hidden-component error is 0.052987/0.056096. Worst output
relative error occurs at position 20. The independent eager-BF16 output itself
differs from FP64 by up to 1.2636%; its rounding baseline is not subtracted
from the measured ISA errors.

The full real embedding tables, projection, layer-0 matrices/vectors, 512-token
KV, and fixed RoPE tables occupy **1,280,885,888 device bytes**. Across these
260 positions the maximum measured program/scratch usage is 3,432 IMEM words
and 15,162 peak live TMEM words. This is the existing f2 configuration; only
the diagnostic simulator's allocated DRAM extent is reduced to the image size.
The previously measured max-context descriptor budget remains separate.

No quantizer, threshold, kernel, or input was tuned after seeing checkpoint
results. Python 3.12.3, NumPy 2.5.4 and CPU PyTorch 2.14.1 were used, with one
numerical thread at low process priority. No skipped tests are counted as passed.

## Sequence, masking and reset checks

The sequence has 260 sequential positions, crossing the native 256-token
V-transpose/attention-block boundary. It starts with text/codec control,
language, speaker and boundary-ID probes, followed by deterministic random
in-range row pairs (seed 2306). These are **diagnostic aligned rows, not a valid
complete CustomVoice prompt or generated speech codes**. This tests real weights
and actual ID-addressed tables without claiming frontend/generation acceptance.

Two ISA machines process the same sequence. One starts with every future KV
byte poisoned with `0x3f` (finite FP32 scale values). All four visible stages
must remain bitwise identical at every position, including 255/256 and the
partial tail. KV-ablation at the last fixed position measures dependence on
the preceding device cache without changing the current input or position.

Afterward, the used layer-0 KV regions are explicitly zeroed and the position
restarts at zero. The first eight rows are compared bitwise with a fresh ISA
machine and numerically with a reset independent reference. This validates the
specified diagnostic reset; it does not add or claim a production/full-model
reset API.

There were **zero bitwise mismatches** across all four stages over 260 poisoned
future-cache comparisons. Reset clears 1,081,344 layer-0 KV bytes; all eight
replayed positions match a pristine machine bitwise. Reset output error against
FP64 is at most 2.1631%. Ablating the preceding KV at the final fixed position
changes layer output by 58.2162% relative L2, confirming that the checkpoint run
depends on cache history rather than merely repeating an independent row path.

## Greedy and readiness implications

Layer 0 returns hidden vectors. It does not emit codec logits. The other 27
talker layers, final norm/head, predictor, sampling and audio decoder remain
absent. Consequently neither small hidden-vector error nor successful masking
can establish greedy codec-token parity, EOS behavior, acoustic quality, or
end-to-end speech readiness. No early-exit codec-head proxy is substituted for
full-model greedy decoding, and stochastic generation defaults are unchanged.

Hardware validation remains pending the parent's coordinated shared-card
window. This work never acquires the card, reloads an image, interrupts serving,
or starts synthesis. Prior synthetic receipts remain separate in
`qwen3_tts_talker_validation.json`; this checkpoint-backed receipt is recorded
in `qwen3_tts_checkpoint_acceptance.json`.

## Reproduction and receipts

From the isolated EC2 source checkout, with the verified file/manifest already
present, the explicit diagnostic command is:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  /home/ubuntu/opentpu-venv/bin/python tools/qwen3_tts_checkpoint_acceptance.py \
  --checkpoint /home/ubuntu/qwen3-tts-checkpoint-20261023-01/model.safetensors \
  --manifest /home/ubuntu/qwen3-tts-checkpoint-20261023-01/checkpoint-manifest.json \
  --output /home/ubuntu/qwen3-tts-checkpoint-20261023-01/checkpoint-acceptance-01.json \
  --bf16-reference
```

The run used a 300-second subprocess timeout and a 32 GiB address-space ceiling.
Its source directory is `/home/ubuntu/qwen3-tts-checkpoint-20261023-01/source`.
As in the prior milestone, tested compiler/runtime code matches `e55992e4`;
the local publisher base additionally contains only the unrelated deployment
logging fix. The unchanged kernel/adapter hashes are recorded in the receipt.

- Materialization SSM command: `a28b58fc-9ac5-4b11-a584-688313042100`.
- Regression plus checkpoint acceptance SSM command:
  `30a0f4c9-3786-46c9-b56a-46c5a724a067` (Success; pytest exit 0; diagnostic exit 0).
- Receipt retrieval SSM command: `00da2162-f050-4d77-bcf3-7902daf41cc0`.
- The committed JSON is the exact unmodified 44,545-byte EC2 result:
  SHA-256 `778d42c3bb451c87d2f1f44e8630e669753f8c72505d627d20caad822275a77e`.
- Complete per-position measurements remain on EC2 as
  `checkpoint-acceptance-01.positions.json` (633,846 bytes), SHA-256
  `47bb495c1b6ba3d44f7270f70812886ac682cf0e338c4fecf4ea387b80d6913f`.
  The committed receipt includes aggregate results and eleven selected positions.

All five reported code hashes and the three uploaded-file hashes match the
local files committed for handoff. The prior `cca65cd5` kernel remains unchanged;
this follow-up adds only explicit diagnostic code, tests, documentation and receipts.
