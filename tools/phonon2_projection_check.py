"""CPU/ISA qualification of a real Phonon-2 projection; never opens an FPGA.

Uses only an existing pinned checkpoint and the current checkout's compiler.
No upstream loader/model code, downloads, GPU path or production ASR fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from opentpu import Config, fp32 as F
from opentpu.asr.phonon2 import MODEL_REVISION, CONTAINER_SHA256
from opentpu.asr.phonon2_linear import decode_five_value, compile_linear_slice
from opentpu.isasim import Machine


def qualify(container, runtime_revision):
    path = Path(container)
    name = "encoder.layers.0.self_attn.q_proj"
    blob = None
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CONTAINER_SHA256:
            raise ValueError("container SHA256 differs from the pinned checkpoint")
        f.seek(0)
        n = int.from_bytes(f.read(8), "little")
        if not 0 < n < 1 << 20:
            raise ValueError("invalid header length")
        header = json.loads(f.read(n))
        if header["format"] != "fermion-five-value-parakeet-v1":
            raise ValueError("unexpected container format")
        for record in header["index"]:
            if record["n"] == name:
                if record["k"] != "five_value" or record["shape"] != [1024, 1024]:
                    raise ValueError("unexpected projection kind or shape")
                blob = f.read(record["b"])
                break
            f.seek(record["b"], 1)
    if blob is None:
        raise ValueError("projection record absent")
    weights = decode_five_value(blob, (1024, 1024))
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=4 << 20)
    compiled = compile_linear_slice(weights, 4, cfg)
    x = np.random.default_rng(20261023).standard_normal((4, 1024)).astype(np.float32)
    image = compiled.image.copy()
    image[compiled.input_at:compiled.input_at + x.nbytes] = x.view(np.uint8).ravel()
    machine = Machine(cfg, [compiled.program], [image]).run()
    got = machine.slices[0].dram[compiled.output_at:compiled.output_at + x.nbytes].view(np.float32).reshape(4, 1024)
    q, scale, offsets = F.quantize_mx(x, 32)
    dequant = q.astype(np.float64).reshape(4, -1, 32) * np.ldexp(scale.astype(np.float64)[:, None], -offsets.astype(int))[:, :, None]
    reference = dequant.reshape(4, 1024) @ weights.dense().astype(np.float64).T
    dense = x.astype(np.float64) @ weights.dense().astype(np.float64).T
    passed = np.allclose(got, reference, rtol=3e-5, atol=2e-5)
    import opentpu
    runtime = Path(opentpu.__file__).resolve().parent.parent
    sources = {}
    for key, module in tuple(sys.modules.items()):
        if (key == "opentpu" or key.startswith("opentpu.")) and getattr(module, "__file__", None):
            source = Path(module.__file__).resolve()
            relative = source.relative_to(runtime).as_posix()  # reject any overlay outside this snapshot
            sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    encoded = np.asarray([op.encode() for op in compiled.program], dtype="<u4")
    return {"passed": bool(passed), "model_revision": MODEL_REVISION,
            "runtime_revision": runtime_revision, "runtime_source_sha256": sources,
            "qualification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_sha256": CONTAINER_SHA256, "record": name,
            "record_sha256": hashlib.sha256(blob).hexdigest(), "shape": [1024, 1024], "rows": 4,
            "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "program_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
            "output_sha256": hashlib.sha256(got.tobytes()).hexdigest(),
            "max_abs_error_vs_quantized_reference": float(np.max(np.abs(got - reference))),
            "relative_l2_error_vs_dense_reference": float(np.linalg.norm(got - dense) / np.linalg.norm(dense)),
            "budget": compiled.budget(), "backend": "CPU reference + ISA simulator",
            "numpy": np.__version__, "hardware_used": False, "transcription_tested": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = qualify(args.container, args.runtime_revision)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
