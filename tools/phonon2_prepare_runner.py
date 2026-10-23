"""Offline pinned checkpoint -> immutable PCM-length-specific device artifact.

Run on the existing EC2, before any card lease. No board import/open, model
inference, Torch reference invocation, network, installation or synthesis.
"""
import argparse
import hashlib
import json
from pathlib import Path


class Writer:
    def __init__(self, destination):
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "blobs").mkdir()
        self.blobs, self.actions = {}, []

    def blob(self, raw):
        raw = bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in self.blobs:
            (self.root / "blobs" / sha).write_bytes(raw)
            self.blobs[sha] = len(raw)
        return sha

    def constants(self, name, image, ranges):
        for i, (a, b) in enumerate(ranges):
            self.actions.append(dict(kind="write", name=f"{name}.constants.{i}", at=a,
                                     bytes=b-a, blob=self.blob(image[a:b].tobytes())))

    def program(self, name, program):
        import numpy as np
        from opentpu import waits as W
        if getattr(program, "run_args", None):
            raise ValueError("unbound program arguments")
        words = np.asarray([op.encode() for op in program], dtype="<u4").reshape(-1)
        entries = np.full(len(program), W.DEFAULT, dtype="<u8")
        if words.size > 32768 or W.loadable(words, entries, 128).nbytes > 163840:
            raise ValueError("program/wait image exceeds shared IMEM/slot")
        self.actions.append(dict(kind="program", name=name, words=int(words.size),
                                 code=self.blob(words.tobytes()), waits=self.blob(entries.tobytes())))

    def phases(self, name, plan):
        for i, program in enumerate(plan.programs):
            self.program(f"{name}.{i}", program)

    def move(self, name, cfg, source, destination, rows, width):
        from opentpu.compiler import Tensor
        from opentpu.asr.phonon2_waveform import move_rows
        reverse = source < destination < source + rows * width * 4
        x = Tensor(source, (rows, width), (width, 1))
        out = Tensor(destination, (rows, width), (width, 1))
        self.program(name, move_rows.trace(cfg, 0, dict(x=x, out=out, rows=rows, reverse=reverse)).finish())


def prepare(container, samples, destination, root):
    import opentpu
    from opentpu import target
    from opentpu.compiler import Tensor
    from opentpu.asr.phonon2_encoder import clear, compile_encoder_layer, shared_profile_config
    from opentpu.asr.phonon2_precise_subsampler import compile_precise_subsampler
    from opentpu.asr.phonon2_tdt import compile_decoder
    from opentpu.asr.phonon2_waveform import compile_waveform
    from opentpu.asr.phonon2_runner import Artifact, MODEL, REVISION, PROFILE, AGFI, CFG_HASH, PROGRAM_AT
    # These imports are audited DATA preparation only. Torch equations are never called.
    from tools.phonon2_encoder_reference import Checkpoint
    from tools.phonon2_tdt_reference import read_weights, load_text_asset
    from tools.phonon2_waveform_audit import coefficients
    root = Path(root)
    binding = json.loads((root / "docs/phonon2/runner_source_binding.json").read_text())
    runtime = Path(opentpu.__file__).resolve().parents[1]
    for name, sha in binding["runtime_files"].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != sha:
            raise ValueError("common runtime source differs: " + name)
    for name, sha in binding["compute_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("preserved compute source differs: " + name)
    cfg = shared_profile_config()
    if f"{target.arch_hash(cfg):08x}" != CFG_HASH:
        raise ValueError("shared configuration hash differs")
    front = compile_waveform(samples, cfg, coefficients(root))
    cp = Checkpoint(container)  # full pinned container SHA before selective data decode
    sw, _ = cp.subsampler()
    sub = compile_precise_subsampler(sw, front.frames, front.valid, cfg)
    del sw
    frames, valid = sub.stored_lengths[3], sub.valid_lengths[3]
    weights, _ = cp.layer(0)
    first = compile_encoder_layer(weights, frames, valid, cfg, passes=2)
    del weights
    weights, _ = read_weights(cp)
    table, _, _ = load_text_asset(root)
    decoder = compile_decoder(weights, table, frames, valid, cfg, passes=2, audit=False)
    del weights, table
    graph = max(front.image.size, sub.image.size, first.image.size, decoder.image.size)
    writer = Writer(destination)
    scratch = Tensor(0, (graph // 4,), (1,))
    writer.program("request_reset", clear.trace(cfg, 0, dict(scratch=scratch, words=graph // 4)).finish())
    writer.constants("waveform", front.image, ((0, front.input_at),))
    writer.actions.append(dict(kind="pcm", name="pcm_s16le"))
    writer.phases("waveform", front)
    writer.move("waveform_to_subsampler", cfg, front.output_at, sub.input_at, front.frames, 128)
    writer.constants("subsampler", sub.image, sub.constant_ranges)
    writer.phases("subsampler", sub)
    previous, capacity = sub.output_at, {"waveform": front.budget(), "subsampler": sub.budget(),
                                        "encoder_layer": first.budget(), "decoder": decoder.budget()}
    del sub
    signature = None
    for i in range(24):
        if i == 0:
            layer = first
        else:
            weights, _ = cp.layer(i)
            layer = compile_encoder_layer(weights, frames, valid, cfg, passes=2)
            del weights
        if layer.image.size != first.image.size or layer.input_at != first.input_at or layer.output_at != first.output_at:
            raise ValueError("one-layer residency layout changed")
        writer.move(f"encoder.{i}.handoff", cfg, previous, layer.input_at, frames, 1024)
        writer.constants(f"encoder.{i}", layer.image, ((0, layer.input_at),))
        before = len(writer.actions)
        writer.phases(f"encoder.{i}", layer)
        keys = [(a["code"], a["waits"]) for a in writer.actions[before:]]
        if signature is not None and signature != keys:
            raise ValueError("one-slot layer code changed with weights")
        signature = keys
        previous = layer.output_at
        del layer
    writer.move("encoder_to_tdt", cfg, previous, decoder.input_at, frames, 1024)
    writer.constants("tdt", decoder.image, ((0, decoder.input_at),))
    writer.phases("tdt", decoder)
    manifest = dict(schema=1, model=MODEL, revision=REVISION, profile=PROFILE, agfi=AGFI,
                    cfg_hash=CFG_HASH, binding=binding, samples=samples, pcm_bytes=2*samples,
                    raw_padded_bytes=front.raw_padded_bytes, input_at=front.input_at,
                    frames=frames, valid=valid, max_steps=decoder.max_steps,
                    output={k: decoder.checkpoints[k]["offset"] for k in ("state", "text", "text_lengths")},
                    graph_bytes=int(graph), program_at=PROGRAM_AT, capacity=capacity,
                    actions=writer.actions, blobs=writer.blobs, ready=False,
                    host_audio_or_model_compute=False, offline_preparation_only=True)
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (writer.root / "manifest.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    Artifact(writer.root, sha)
    return {"manifest_sha256": sha, "samples": samples, "frames": frames, "valid": valid,
            "dispatches": sum(a["kind"] == "program" for a in writer.actions),
            "unique_blob_bytes": sum(writer.blobs.values()), "graph_bytes": int(graph),
            "program_slot_bytes": 163840, "hardware_used": False, "ready": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", required=True); p.add_argument("--samples", type=int, required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args()
    print(json.dumps(prepare(a.container, a.samples, a.output_dir, a.root), indent=2))
