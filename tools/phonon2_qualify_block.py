"""Qualify one full-width block against CPU references, using an existing checkpoint.

Explicit offline test tool; no device drivers, networking, model-code imports or
production inference. Keeps all simulated intermediates in one Machine's DRAM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from opentpu import Config, isa as I
from opentpu.asr.phonon2 import MODEL_ID, MODEL_REVISION
from opentpu.asr.phonon2_conformer import BlockShape, compile_conformer_slice
from opentpu.isasim import Machine
from tools.phonon2_reference import read_block, conformer_reference
from tools.phonon2_torch_audit import conformer_torch


def qualify(container, report_path, *, frames=8, valid=7):
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    weights, hashes = read_block(container)
    shape = BlockShape()
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=64 << 20)
    compiled = compile_conformer_slice(weights, frames, valid, cfg, shape)
    x = np.random.default_rng(20261023).standard_normal((frames, 1024)).astype(np.float32)
    image = compiled.image.copy()
    image[compiled.input_at:compiled.input_at + x.nbytes] = x.view(np.uint8).ravel()
    machine = Machine(cfg, [compiled.programs[0]], [image]).run()
    for program in compiled.programs[1:]:
        machine.load([program]).run()
    def read(at):
        return machine.slices[0].dram[at:at + x.nbytes].view(np.float32).reshape(x.shape).copy()
    actual = {name: read(at) for name, at in compiled.checkpoints.items()}
    quant = conformer_reference(weights, x, valid, shape, quantize_D=cfg.D)
    dense = conformer_reference(weights, x, valid, shape)
    pytorch = conformer_torch(weights, x, valid, shape)
    def compare(a, b):
        delta = a.astype(np.float64) - b.astype(np.float64)
        return {"max_abs_error": float(np.max(np.abs(delta))),
                "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-20))}
    stages = {name: compare(actual[name], quant[name]) for name in actual}
    dense_error = compare(actual["output"], dense["output"])
    torch_error = compare(dense["output"], pytorch)
    # Explicit engineering acceptance for this bounded numerical test, not WER.
    passed = (all(e["relative_l2_error"] < .001 for e in stages.values()) and
              dense_error["relative_l2_error"] < .03 and torch_error["relative_l2_error"] < 1e-5 and
              np.isfinite(actual["output"]).all() and np.array_equal(actual["glu"][valid:], np.zeros_like(actual["glu"][valid:])))
    root = Path(__file__).resolve().parents[1]
    names = ("opentpu/asr/phonon2_conformer.py", "opentpu/asr/phonon2_linear.py",
             "tools/phonon2_reference.py", "tools/phonon2_torch_audit.py", "tools/phonon2_qualify_block.py")
    import opentpu
    runtime = Path(opentpu.__file__).resolve().parent
    runtime_names = ("__init__.py", "compiler.py", "isasim.py", "isa.py", "language.py", "fp32.py", "quant.py",
                     "kernels/__init__.py", "kernels/lib.py", "kernels/vit.py", "kernels/attention.py",
                     "kernels/pieces.py", "kernels/mlp.py")
    report = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "layer": 0,
              "test": "complete 1024/4096, 8-head, kernel-9 Conformer eval block; synthetic FP32 input",
              "passed": bool(passed), "hardware_used": False, "audio_to_transcript_validated": False,
              "backend": "CPU ISA simulator + independent NumPy and own PyTorch CPU audit equations",
              "numpy": np.__version__, "torch": torch.__version__, "elapsed_seconds": time.monotonic() - started,
              "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
              "output_sha256": hashlib.sha256(actual["output"].tobytes()).hexdigest(),
              "source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
              "runtime_source_sha256": {n: hashlib.sha256((runtime / n).read_bytes()).hexdigest() for n in runtime_names},
              "checkpoint_record_sha256": hashes, "budget": compiled.budget(),
              "opcodes": sorted({I.OPNAMES[op.op] for program in compiled.programs for op in program}),
              "stage_errors_vs_quantized_numpy": stages, "final_vs_dense_numpy": dense_error,
              "dense_numpy_vs_torch": torch_error,
              "acceptance": {"per_stage_relative_l2_vs_quantized": .001, "final_relative_l2_vs_dense": .03,
                             "dense_numpy_relative_l2_vs_torch": 1e-5,
                             "scope": "one synthetic activation sequence, no transcript-quality conclusion"}}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = qualify(args.container, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
