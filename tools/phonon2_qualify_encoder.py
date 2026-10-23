"""Complete learned encoder composition on the shared-profile CPU/ISA model.

No card access or serving fallback. The waveform frontend remains an explicitly
offline oracle. One immutable layer slot is streamed; activations never return
to the host to feed the next layer. Progress/receipts are saved after each layer.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from opentpu import isa as I, waits as W, target
from opentpu.compiler import Tensor
from opentpu.isasim import Machine
from opentpu.asr.phonon2 import MODEL_ID, MODEL_REVISION
from opentpu.asr.phonon2_encoder import (compile_encoder_layer, shared_profile_config, clear, copy_rows,
                                         scheduled_programs, SHARED_PROFILE, SHARED_AGFI, SHARED_ARCH_HASH)
from opentpu.asr.phonon2_subsampler import compile_subsampler
from opentpu.asr.phonon2_precise_subsampler import compile_precise_subsampler
from tools.phonon2_encoder_reference import Checkpoint, layer_torch
from tools.phonon2_subsample_reference import subsampler_torch, read_tutorial_audio
from tools.phonon2_torch_audit import frontend_torch


def error(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    delta, norm = a - b, max(np.linalg.norm(b), 1e-20)
    return {"relative_l2": float(np.linalg.norm(delta) / norm), "max_abs": float(np.max(np.abs(delta))),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * norm, 1e-20))}


def fingerprint(programs):
    return hashlib.sha256(b"".join(np.asarray([i.encode() for i in p], dtype="<u4").tobytes() for p in programs)).hexdigest()


class SimulatorDevice:
    def __init__(self, cfg):
        self.cfg, self.machine, self.high_water = cfg, None, 0
        self.dispatches, self.constant_bytes, self.wait_checked = 0, 0, 0

    def phase(self, program, check=False):
        if self.machine is None:
            self.machine = Machine(self.cfg, [program], [None])
        else:
            self.machine.load([program])
        if check:
            entries = np.full(len(program), W.DEFAULT, np.uint64)
            self.wait_checked += W.check_isasim(self.machine, [entries])
        else:
            self.machine.run(max_steps=50_000_000)
        self.dispatches += 1

    @property
    def memory(self):
        return self.machine.slices[0].dram

    def reset(self, footprint):
        self.high_water = max(self.high_water, footprint)
        desc = Tensor(0, (self.high_water // 4,), (1,))
        self.phase(clear.trace(self.cfg, 0, dict(scratch=desc, words=desc.shape[0])).finish())

    def constants(self, image, ranges):
        for a, b in ranges:
            self.memory[a:b] = image[a:b]
            self.constant_bytes += b - a

    def read(self, at, rows, width=1024):
        return self.memory[at:at + 4 * rows * width].view(np.float32).reshape(rows, width).copy()

    def copy(self, source, destination, rows, width=1024, check=False):
        x, out = Tensor(source, (rows, width), (width, 1)), Tensor(destination, (rows, width), (width, 1))
        self.phase(copy_rows.trace(self.cfg, 0, dict(x=x, out=out, rows=rows)).finish(), check=check)


def trial(checkpoint, device, features, valid, passes, name, progress, *, check_waits=False):
    start = time.monotonic()
    cfg, frames = device.cfg, len(features)
    weights, sub_hashes = checkpoint.subsampler()
    sub = (compile_subsampler if passes == 1 else compile_precise_subsampler)(weights, frames, valid, cfg)
    n, v = sub.stored_lengths[3], sub.valid_lengths[3]
    layer0, _ = checkpoint.layer(0)
    first = compile_encoder_layer(layer0, n, v, cfg, passes=passes)
    del layer0
    dispatch_at, bytes_at, checked_at = device.dispatches, device.constant_bytes, device.wait_checked
    device.reset(max(sub.image.size, first.image.size))
    device.constants(sub.image, getattr(sub, "constant_ranges", ((0, sub.input_at),)))
    x = np.ascontiguousarray(features, dtype=np.float32)
    device.memory[sub.input_at:sub.input_at + x.nbytes] = x.view(np.uint8).ravel()  # oracle input only
    for program in sub.programs:
        device.phase(program, check=check_waits)
    actual_sub = device.read(sub.output_at, n)
    dense_sub = subsampler_torch(weights, x, valid)["output"]
    dense_state, conditioned, quant_state = dense_sub.copy(), actual_sub.copy(), actual_sub.copy()
    sub_error = error(actual_sub[:v], dense_sub[:v])
    del weights
    records, previous = [], sub.output_at
    signature = fingerprint(first.programs)
    loadables = scheduled_programs(first)
    largest_loadable = max(a[2].nbytes for a in loadables)
    layer_hashes, mask_ok = {}, True
    for index in range(24):
        weights, hashes = checkpoint.layer(index)
        plan = first if index == 0 else compile_encoder_layer(weights, n, v, cfg, passes=passes)
        if fingerprint(plan.programs) != signature or plan.input_at != first.input_at:
            raise AssertionError("layer shape/layout/program changed; one-slot scheme invalid")
        # Preserve the live output before the host restages this layer's immutable prefix.
        device.copy(previous, plan.input_at, n, check=check_waits and index == 0)
        layer_input = device.read(plan.input_at, n)  # audit only, never re-uploaded
        device.constants(plan.image, ((0, plan.input_at),))
        for program in plan.programs:
            device.phase(program, check=check_waits and index == 0)
        actual = device.read(plan.output_at, n)
        dense_state = layer_torch(weights, dense_state, v)
        conditioned = layer_torch(weights, conditioned, v)
        quant_state = layer_torch(weights, quant_state, v, activation_passes=passes)
        local = layer_torch(weights, layer_input, v)
        glu = device.read(plan.checkpoints["glu"], n)
        probability = device.read(plan.attention_scores_at, n, n)
        masks = bool(np.array_equal(glu[v:], np.zeros_like(glu[v:])) and
                     np.array_equal(probability[:v, v:], np.zeros_like(probability[:v, v:])))
        if v < n:
            masks &= bool(np.allclose(probability[v:], 1 / n, rtol=2e-6, atol=1e-8))
        mask_ok &= masks
        record = {"layer": index, "local_vs_dense": error(actual[:v], local[:v]),
                  "accum_encoder_vs_dense_same_subsample_input": error(actual[:v], conditioned[:v]),
                  "accum_end_to_end_vs_dense": error(actual[:v], dense_state[:v]),
                  "accum_vs_quantized_oracle": error(actual[:v], quant_state[:v]),
                  "all_rows_vs_dense": error(actual, dense_state), "mask_passed": masks,
                  "finite": bool(np.isfinite(actual).all()),
                  "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        records.append(record)
        layer_hashes.update(hashes)
        Path(progress).write_text(json.dumps({"trial": name, "activation_passes": passes, "completed_layers": index + 1,
                                             "elapsed_seconds": time.monotonic() - start, "latest": record}, indent=2))
        previous = plan.output_at
        del weights, plan, layer_input, local, glu, probability
        gc.collect()
    actual = device.read(previous, n)
    final_error = records[-1]["accum_end_to_end_vs_dense"]
    accepted = bool(mask_ok and all(r["finite"] for r in records) and final_error["relative_l2"] <= .03)
    result = {"name": name, "activation_passes": passes, "numerical_accepted": accepted,
              "hardware_validated": False, "model_ready": False, "frame_counts": {"mel_stored": frames, "mel_valid": valid, "encoder_stored": n, "encoder_valid": v},
              "subsampler_vs_dense": sub_error, "layers": records, "all_masks_passed": mask_ok,
              "final_end_to_end_vs_dense": final_error, "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
              "budget": {"subsampler": sub.budget(), "layer_slot": first.budget(),
                         "slot_reused_across_layers": 24, "layer_program_sha256": signature,
                         "actual_dispatches": device.dispatches - dispatch_at,
                         "actual_constant_bytes_staged": device.constant_bytes - bytes_at,
                         "virtual_dram_bytes": cfg.DRAM_BYTES, "touched_graph_high_water_bytes": device.high_water,
                         "largest_program_and_wait_image_bytes": largest_loadable,
                         "wait_policy": "conservative full prior-unit/VPU dependency waits",
                         "wait_checker_instructions": device.wait_checked - checked_at,
                         "host_activation_transform_or_reupload": False},
              "checkpoint_record_sha256": {**sub_hashes, **layer_hashes}, "elapsed_seconds": time.monotonic() - start}
    return result, actual


def qualify(container, pcm, wave, output_dir, *, include_full=True):
    import torch
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = shared_profile_config()
    if f"{target.arch_hash(cfg):08x}" != SHARED_ARCH_HASH:
        raise RuntimeError("shared image CFG hash mismatch")
    fixture = Path(pcm).read_bytes()
    if hashlib.sha256(fixture).hexdigest() != "7f84e087aa28b5a33212af26a4273b4902761b2543afadc32c3236bf51ca3f74":
        raise ValueError("speech crop hash differs")
    checkpoint, device = Checkpoint(container), SimulatorDevice(cfg)
    short = frontend_torch(fixture)
    report = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "profile": SHARED_PROFILE,
              "agfi": SHARED_AGFI, "declared_cfg_hash": SHARED_ARCH_HASH, "config": dataclasses.asdict(cfg),
              "backend": "CPU/ISA and own native-Torch source audit", "hardware_used": False,
              "frontend_role": "Offline waveform oracle only; device PCM/FFT/mel path remains missing",
              "model_ready": False, "audio_to_transcript_validated": False,
              "numerical_acceptance": {"final_valid_relative_l2_vs_dense": .03, "scope": "learned encoder only; no transcript/WER or hardware acceptance"},
              "trials": []}
    def save():
        (output_dir / "encoder_report.json").write_text(json.dumps(report, indent=2))
    baseline, _ = trial(checkpoint, device, short["features"], short["valid_frames"], 1, "short_baseline", output_dir / "progress.json", check_waits=True)
    report["trials"].append(baseline); save()
    precise, fresh_short = trial(checkpoint, device, short["features"], short["valid_frames"], 2, "short_residual", output_dir / "progress.json", check_waits=True)
    report["trials"].append(precise); save()
    if include_full:
        raw = Path(wave).read_bytes()
        if hashlib.sha256(raw).hexdigest() != "67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8":
            raise ValueError("full speech wave hash differs")
        full = frontend_torch(read_tutorial_audio(wave).tobytes())
        long, _ = trial(checkpoint, device, full["features"], full["valid_frames"], 2, "full_speech_residual", output_dir / "progress.json")
        report["trials"].append(long); save()
    reset, reset_short = trial(checkpoint, device, short["features"], short["valid_frames"], 2, "short_after_reset", output_dir / "progress.json")
    report["trials"].append(reset)
    report["reset_passed"] = bool(np.array_equal(fresh_short, reset_short))
    report["numerical_accepted"] = bool(report["reset_passed"] and all(t["numerical_accepted"] for t in report["trials"] if t["activation_passes"] == 2))
    report["maximum_resident_rss_platform_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    root = Path(__file__).resolve().parents[1]
    names = ["opentpu/asr/phonon2_encoder.py", "opentpu/asr/phonon2_precision.py", "opentpu/asr/phonon2_precise_subsampler.py",
             "tools/phonon2_encoder_reference.py", "tools/phonon2_qualify_encoder.py", "opentpu/asr/phonon2_conformer.py",
             "opentpu/asr/phonon2_subsampler.py", "opentpu/asr/phonon2_linear.py", "opentpu/asr/phonon2_int6.py",
             "tools/phonon2_subsample_reference.py", "tools/phonon2_torch_audit.py"]
    report["source_sha256"] = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    import opentpu
    runtime = Path(opentpu.__file__).resolve().parents[1]
    report["runtime_manifest"] = json.loads((runtime / "source_manifest.json").read_text())
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--pcm", required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--short-only", action="store_true")
    a = parser.parse_args()
    r = qualify(a.container, a.pcm, a.wave, a.output_dir, include_full=not a.short_only)
    print(json.dumps({"numerical_accepted": r["numerical_accepted"], "reset_passed": r["reset_passed"]}))
