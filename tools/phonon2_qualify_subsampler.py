"""CPU/ISA qualification of real Phonon-2 subsampler weights and speech features.

Uses an existing checkpoint and a local, attributed PCM fixture. No downloads,
remote model code, FPGA access, GPU execution or serving fallback are provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from opentpu import Config, isa as I, language as ol
from opentpu.compiler import Tensor
from opentpu.asr.phonon2 import MODEL_ID, MODEL_REVISION
from opentpu.asr.phonon2_conformer import compile_conformer_slice, BlockShape
from opentpu.asr.phonon2_subsampler import compile_subsampler, SubsampleShape
from opentpu.isasim import Machine
from tools.phonon2_reference import read_block, conformer_reference
from tools.phonon2_torch_audit import frontend_torch, conformer_torch
from tools.phonon2_subsample_reference import read_subsampler, subsampler_reference, subsampler_torch, read_tutorial_audio


@ol.jit
def device_copy(x, out, frames: int):
    for t in ol.range(frames):
        ol.store(out[t, :], ol.load(x[t, :]))


def _offset(value):
    return int(value) if isinstance(value, (int, np.integer)) else value.static()


def snapshots(machine, compiled):
    return {name: np.ndarray(shape=spec["shape"], dtype="<f4", buffer=machine.slices[0].dram,
                            offset=_offset(spec["offset"]), strides=tuple(4 * s for s in spec["strides"])).copy()
            for name, spec in compiled.checkpoints.items()}


def execute(compiled, features, machine=None):
    # The audited Torch frontend returns a transposed NumPy view. Canonicalize
    # transfer byte order without changing values; kernels consume row-major FP32.
    features = np.ascontiguousarray(features, dtype=np.float32)
    if machine is None:
        image = compiled.image.copy()
        start = (compiled.input_at + features.nbytes + 127) // 128 * 128
        image[start:] = 255  # Poison every scratch word; device reset must remove it.
        machine = Machine(compiled.cfg, [compiled.programs[0]], [image])
    else:
        machine.load([compiled.programs[0]])
    memory = machine.slices[0].dram
    immutable_before = hashlib.sha256(memory[:compiled.input_at]).hexdigest()
    memory[compiled.input_at:compiled.input_at + features.nbytes] = features.view(np.uint8).ravel()
    machine.run()
    for program in compiled.programs[1:]:
        machine.load([program]).run()
    assert hashlib.sha256(memory[:compiled.input_at]).hexdigest() == immutable_before, "weights were overwritten"
    assert not memory[compiled.image.size:].any(), "kernel wrote beyond its allocated image"
    return machine


def compare(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"max_abs_error": float(np.max(np.abs(delta))),
            "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-20))}


def qualify(container, pcm_path, report_path, golden_path, *, full_audio_path=None):
    import torch
    torch.set_num_threads(1)
    start = time.monotonic()
    weights, hashes = read_subsampler(container)
    fixture = Path(pcm_path).read_bytes()
    pcm_sha = hashlib.sha256(fixture).hexdigest()
    if pcm_sha != "7f84e087aa28b5a33212af26a4273b4902761b2543afadc32c3236bf51ca3f74":
        raise ValueError("speech fixture differs from attributed pinned crop")
    speech = frontend_torch(fixture)  # Offline oracle only; never the serving path.
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=64 << 20)
    rng = np.random.default_rng(20261023)
    cases = [("tiny_odd", rng.normal(size=(3, 128)).astype(np.float32), 2),
             ("stride_boundary", rng.normal(size=(9, 128)).astype(np.float32), 8),
             ("speech_1688_crop", speech["features"], speech["valid_frames"])]
    full_wave_sha = None
    if full_audio_path is not None:
        full_wave_sha = hashlib.sha256(Path(full_audio_path).read_bytes()).hexdigest()
        if full_wave_sha != "67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8":
            raise ValueError("full speech WAVE differs from pinned tutorial recording")
        full_pcm = read_tutorial_audio(full_audio_path)
        full_features = frontend_torch(full_pcm.tobytes())
        cases.append(("speech_1688_full_utterance", full_features["features"], full_features["valid_frames"]))
    results, speech_state = [], None
    passed = True
    for name, x, valid in cases:
        compiled = compile_subsampler(weights, len(x), valid, cfg)
        machine = execute(compiled, x)
        actual = snapshots(machine, compiled)
        quant = subsampler_reference(weights, x, valid, quantize_D=cfg.D)
        dense = subsampler_reference(weights, x, valid)
        official_equations = subsampler_torch(weights, x, valid)
        stage_errors = {stage: compare(actual[stage], quant[stage]) for stage in actual}
        dense_error = compare(actual["output"], dense["output"])
        torch_errors = {stage: compare(dense[stage], official_equations[stage]) for stage in dense}
        mask_passed = True
        for stage, depth in (("conv0", 1), ("depthwise2", 2), ("pointwise3", 2), ("depthwise5", 3), ("pointwise6", 3)):
            mask_passed &= bool(np.array_equal(actual[stage][compiled.valid_lengths[depth]:], np.zeros_like(actual[stage][compiled.valid_lengths[depth]:])))
        padded = actual["output"][compiled.valid_lengths[3]:]
        bias_passed = bool(np.array_equal(padded, np.broadcast_to(weights["linear.bias"], padded.shape)))
        case_passed = (all(e["relative_l2_error"] < .001 for e in stage_errors.values()) and
                       dense_error["relative_l2_error"] < .03 and
                       all(e["relative_l2_error"] < 2e-5 for e in torch_errors.values()) and
                       mask_passed and bias_passed and np.isfinite(actual["output"]).all())
        passed &= bool(case_passed)
        results.append({"name": name, "passed": bool(case_passed), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                        "output_sha256": hashlib.sha256(actual["output"].tobytes()).hexdigest(),
                        "stage_errors_vs_quantized_numpy": stage_errors, "final_vs_dense_numpy": dense_error,
                        "dense_numpy_vs_torch": torch_errors, "mask_passed": mask_passed,
                        "padded_linear_bias_preserved": bias_passed, "budget": compiled.budget()})
        if name == "speech_1688_crop":
            speech_state = (compiled, machine, actual, quant, dense, official_equations)

    # Same DRAM/TMEM/ACT retained across clips, including a shorter valid prefix.
    full = compile_subsampler(weights, 9, 9, cfg)
    short = compile_subsampler(weights, 9, 1, cfg)
    reset_machine = execute(full, rng.normal(size=(9, 128)).astype(np.float32))
    x = rng.normal(size=(9, 128)).astype(np.float32)
    repeated = snapshots(execute(short, x, reset_machine), short)
    fresh = snapshots(execute(short, x), short)
    reset_passed = all(np.array_equal(repeated[name], fresh[name]) for name in fresh)
    passed &= reset_passed

    # A device-to-device handoff to the existing complete block. Host stages
    # immutable constants only, after the device copy has preserved activations
    # outside the weights' prefix. There is no host read/transform/re-upload.
    compiled, machine, actual, quant, dense, torch_sub = speech_state
    block_weights, block_hashes = read_block(container)
    block = compile_conformer_slice(block_weights, compiled.stored_lengths[3], compiled.valid_lengths[3], cfg)
    n = compiled.stored_lengths[3]
    source = Tensor(compiled.output_at, (n, 1024), (1024, 1))
    target = Tensor(block.input_at, (n, 1024), (1024, 1))
    copy_program = device_copy.trace(cfg, 0, dict(x=source, out=target, frames=n)).finish()
    machine.load([copy_program]).run()
    memory = machine.slices[0].dram
    handoff = memory[block.input_at:block.input_at + 4 * n * 1024].view(np.float32).reshape(n, 1024).copy()
    copy_passed = np.array_equal(handoff, actual["output"])
    # This prefix contains only packed weights and immutable sinusoidal constants.
    memory[:block.input_at] = block.image[:block.input_at]
    for program in block.programs:
        machine.load([program]).run()
    chain = memory[block.output_at:block.output_at + 4 * n * 1024].view(np.float32).reshape(n, 1024).copy()
    chain_quant = conformer_reference(block_weights, quant["output"], compiled.valid_lengths[3], quantize_D=cfg.D)["output"]
    chain_dense = conformer_torch(block_weights, torch_sub["output"], compiled.valid_lengths[3], BlockShape())
    chain_error, chain_dense_error = compare(chain, chain_quant), compare(chain, chain_dense)
    chain_passed = bool(copy_passed and chain_error["relative_l2_error"] < .005 and chain_dense_error["relative_l2_error"] < .08)
    passed &= chain_passed
    np.savez_compressed(golden_path, pcm_s16le=np.frombuffer(fixture, "<i2"), features=speech["features"],
                        feature_mask=speech["attention_mask"], subsampler_dense=torch_sub["output"],
                        subsampler_isa=actual["output"], first_block_dense=chain_dense, first_block_isa=chain)
    root = Path(__file__).resolve().parents[1]
    names = ("opentpu/asr/phonon2_int6.py", "opentpu/asr/phonon2_subsampler.py", "tools/phonon2_subsample_reference.py",
             "tools/phonon2_qualify_subsampler.py", "opentpu/asr/phonon2_conformer.py", "opentpu/asr/phonon2_linear.py",
             "tools/phonon2_reference.py", "tools/phonon2_torch_audit.py")
    import opentpu
    runtime = Path(opentpu.__file__).resolve().parent
    runtime_names = ("__init__.py", "compiler.py", "isasim.py", "isa.py", "language.py", "fp32.py", "quant.py",
                     "kernels/__init__.py", "kernels/lib.py", "kernels/vit.py", "kernels/attention.py", "kernels/pieces.py", "kernels/mlp.py")
    report = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "passed": bool(passed),
              "backend": "CPU ISA simulator + independent NumPy and own native PyTorch CPU audit equations",
              "hardware_used": False, "audio_to_transcript_validated": False,
              "frontend_device_implemented": False, "frontend_role": "Offline golden preparation only; fixture features supplied to the learned device subsampler",
              "numpy": np.__version__, "torch": torch.__version__, "elapsed_seconds": time.monotonic() - start,
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
              "runtime_source_sha256": {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in runtime_names},
              "checkpoint_record_sha256": hashes, "speech_pcm_sha256": pcm_sha, "full_wave_sha256": full_wave_sha,
              "cases": results,
              "reset_passed": reset_passed,
              "device_handoff_and_first_block": {"passed": chain_passed, "copy_passed": bool(copy_passed),
                                                 "copy_instructions": len(copy_program), "first_block_weights_sha256": block_hashes,
                                                 "final_vs_quantized_numpy": chain_error, "final_vs_dense_torch": chain_dense_error,
                                                 "host_activation_transform_or_reupload": False},
              "golden_sha256": hashlib.sha256(Path(golden_path).read_bytes()).hexdigest(), "golden_bytes": Path(golden_path).stat().st_size,
              "acceptance": {"per_stage_relative_l2_vs_quantized": .001, "subsampler_final_relative_l2_vs_dense": .03,
                             "dense_numpy_vs_torch_relative_l2": 2e-5, "chain_relative_l2_vs_quantized": .005,
                             "chain_relative_l2_vs_dense": .08, "scope": "Numerical slice qualification only; no WER or FPGA-performance conclusion"}}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--pcm", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--full-audio-wave", help="Optional existing pinned tutorial WAVE for full-utterance qualification")
    args = parser.parse_args()
    result = qualify(args.container, args.pcm, args.report, args.golden, full_audio_path=args.full_audio_wave)
    print(json.dumps({"passed": result["passed"], "report": args.report, "elapsed_seconds": result["elapsed_seconds"]}))
    raise SystemExit(0 if result["passed"] else 1)
