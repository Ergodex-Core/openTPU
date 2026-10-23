# Qwen3-TTS public chunk boundary with upsampling

This acceptance increment preserves base `7652f552` and closes its stated
300-frame/25-frame recomputation gap for the resident decoder transformer and
both upsampling/ConvNeXt stages. It does not yet validate the waveform stage.

The model is the Apache-2.0 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, checkpoint
`85e237c12c027371202489a0ec509ded67b5e4b5`; the exact
[public decoder source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py)
uses chunks of 300 codec frames and 25 recomputed left-context frames. The
cached 682,293,092-byte tokenizer checkpoint was verified against SHA-256
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.

`public-upsample-01` completed on the existing `opentpu-m8a-build` in 626.24
seconds (626.04 harness seconds). It evaluated 301 recorded-prefix-code diagnostic
frames, crossing the native boundary once and recomputing all 25 context frames.
A single initial device fixture supplies the codes; there are zero host uploads
of current or replay IDs after that fixture. Decoder and upsampling states were
poisoned at frame 300 before the controller recomputed them. The fixture remained
unchanged. The first three frames replayed bitwise after full reset.

Maximum per-frame output relative L2 was **0.0180961614% against FP32 source
algebra** and **0.0181050788% against FP64 source algebra**, below the unchanged
3.5% bound. Source references compute each public chunk independently and discard
its 25 context outputs. This is actual checkpoint decoder validation with a
recorded-code diagnostic producer, not newly generated 301-frame speech or a
complete utterance. There is no quality claim.

The exact shared target remains AGFI `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, requiring
runtime `30100afb075374381601a6e01d9f914a58dbec91`. No card was queried or used.
Standalone resident model bytes remain 150,077,312; this diagnostic fixture adds
19,328 bytes. No program, compiler or RTL change is required by this acceptance
increment. The component checkout's older compiler is not a substitute for the
explicit runtime dependency.

The historical main-model BF16 comparison remains failed: 127/144 generated IDs,
6/9 complete frames and 7.5136935% maximum conditional predictor error. The
FP32-source-algebra contract does not imply bit-exact BF16 execution. All failed
receipts and the overall failed gate remain intact. Full TTS, waveform readiness,
quality and physical acceptance remain false; serving interruption was zero.

The [evidence directory](qwen3_tts_public_upsample_evidence/) contains original
requests, terminal results, log, all 301 comparison rows, environment, source
manifests and post-run verification. All 38 acceptance source/fixture hashes and
150 source snapshot hashes were verified: 86 runtime Python/JSON files against
Git runtime `30100afb`, and 64 component/fixture files against this review tree.
The builder-only historical receipt fixture aliases the preserved talker report.
No remote model code was executed; references and ISA simulation are authored
NumPy diagnostics on the builder. No model weights were downloaded to the Mac.

```sh
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_public_upsample_acceptance.py --data /home/ubuntu/qwen3-tts-public-upsample-20261023-01/data --output /path/to/new-public-upsample-receipt.json
```

Next validation is the waveform decoder with source-correct raw scaling, then
complete utterance/quality and coordinated same-image physical acceptance.
