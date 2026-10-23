# Projection revalidation for PR #39

The real 1024x1024 first-layer query projection **passed again** using source from
published commit `8815ac80226aa3a0d0b07d03344f61839580415e`, with four seeded FP32
activation rows on the existing EC2 CPU/ISA simulator. This is not FPGA or ASR
evidence. No device, locks, running service, model runtime or new resource was used.

[projection_validation_pr39.json](projection_validation_pr39.json) records the
checkpoint/record, input, program, output and every imported OpenTPU source hash.
All ten imported source hashes were independently compared with `git show` at
the published commit and matched. The check used an isolated copy of those source
files, not the active EC2 checkout. The model checkpoint remained on EC2.

* Maximum absolute error versus activation-quantized dense reference: **1.43218e-6**.
* Relative L2 error versus unquantized dense reference: **0.00753760** (0.754%).
* Nine instructions / 72 words; image 2,138,112 bytes, weights 2,105,344 bytes.
* `tests/test_phonon2_asr.py`: **35 passed** against the published checkout.

The original [validation.json](validation.json) is retained as historical evidence
for runtime `b4d6bb19c367c1a0005d08d682c8363d38f2e79a`. Its compiler and language
hashes differ from PR #39. It must not be relabeled as evidence for the newer
compiler. This new receipt supplies the requested revalidation.

The [exact source comparison](compiler_compatibility_pr39.json) shows:

* `compiler.py` adds device counters and derived address registers, propagates
  `run_seen` into Program, permits a destination offset on VOP, and extends
  `quantize_into` with column scales and a source offset.
* `language.py` exposes those `quantize_into` arguments and expands bounded
  runtime-token handling for multirow KV appends.
* `isa.py`, `isasim.py`, `fp32.py` and `quant.py` are byte-identical between the
  inspected historical source and the published source.

The bounded projection uses no counters, runtime arguments, KV appends or partial
`quantize_into`. Nevertheless, it was actually rerun; compatibility is not inferred
only from the diff. Separately, `git diff d3dfa4f9 8815ac80226a -- opentpu` is empty.
The only intervening Git changes are `tools/f2_web/deploy.py` and its tests.

Reproduce against an existing approved checkpoint:

```sh
python -B -m tools.phonon2_projection_check \
  --container /path/to/existing/model.fermion \
  --runtime-revision 8815ac80226aa3a0d0b07d03344f61839580415e \
  --report /tmp/phonon2-projection.json
```

The revision argument labels a run; source hashes must independently match that
Git revision before treating it as revision-specific evidence. The tool performs
no downloads, device access or remote-code imports. Its container SHA256 check
precedes parsing. This receipt work is separate from the new Conformer slice.
