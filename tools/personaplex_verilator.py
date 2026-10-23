#!/usr/bin/env python3
"""PersonaPlex temporal kernels on real RTL, with synthetic weights.

The tiny case checks multiple streaming frames and reset. The layer case keeps
the production hidden/FFN/head dimensions, but runs ONE synthetic layer. Neither
case measures the 32-layer checkpoint, host audio pipeline, or AWS HBM hardware.

    python tools/personaplex_verilator.py --mode tiny --output tiny.json
    python tools/personaplex_verilator.py --mode layer --output layer.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opentpu import rtlsim, target  # noqa: E402
from opentpu.llm.personaplex_temporal import Engine, Image, Spec  # noqa: E402
from opentpu.llm.qwen3 import IsaBackend  # noqa: E402
from opentpu.llm.rtl_backend import RtlBackend  # noqa: E402

PROFILE = ROOT / "tools/data/personaplex/burst200.toml"


def check_environment():
    """Do not let unrelated simulator experiments override the reported run."""
    names = {"OTPU_UARCH", "OTPU_MXU_E", "OTPU_QUANT_XD", "OTPU_VPU_CL",
             "OTPU_SEQ_WC", "OTPU_ULANES", "OTPU_SIM_PLUSARGS", "OTPU_WAITS"}
    active = sorted(k for k, v in os.environ.items()
                    if v and (k in names or k.startswith("OTPU_TMEM_")))
    if active:
        raise ValueError("unset simulator overrides for this fixed-profile check: " +
                         ", ".join(active))


def synthetic_model(spec, frames, seed=192):
    """Bounded, nonzero deterministic weights; no checkpoint or torch dependency."""
    rng = np.random.default_rng(seed)
    h, f = spec.hidden, spec.ffn
    shapes = {"wq": (h, h), "wk": (h, h), "wv": (h, h), "wo": (h, h),
              "wg": (f, h), "wu": (f, h), "wd": (h, f)}
    layers = []
    for _ in range(spec.layers):
        weights = {}
        for name, shape in shapes.items():
            a = rng.standard_normal(shape, dtype=np.float32)
            a *= (1.0 if name in ("wq", "wk") else 0.5) / np.sqrt(shape[1])
            weights[name] = a
        weights.update(norm1=rng.uniform(.8, 1.2, h).astype(np.float32),
                       norm2=rng.uniform(.8, 1.2, h).astype(np.float32))
        layers.append(weights)
    final_norm = rng.uniform(.8, 1.2, h).astype(np.float32)
    inputs = rng.standard_normal((frames, h), dtype=np.float32)
    return layers, final_norm, inputs


def reference(spec, layers, final_norm, embeddings):
    """Independent float64 causal attention, adjacent-pair RoPE and SwiGLU.

    No production kernel, quantizer, or RoPE helper is used here. The difference
    from the fixed-point path includes deliberate weight/activation/KV rounding.
    """
    x = np.asarray(embeddings, np.float64)
    count, hidden = x.shape
    width = spec.head_dim
    angle = np.arange(count)[:, None, None] * spec.rope_base ** (
        -np.arange(0, width, 2) / width)
    cosine, sine = np.cos(angle), np.sin(angle)

    def norm(a, gain):
        return a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + spec.eps) * gain

    def rotate(a):
        out = np.empty_like(a)
        even, odd = a[..., ::2], a[..., 1::2]
        out[..., ::2] = even * cosine - odd * sine
        out[..., 1::2] = odd * cosine + even * sine
        return out

    causal = np.where(np.arange(count)[None, :] <= np.arange(count)[:, None], 0., -np.inf)
    for weights in layers:
        normalized = norm(x, weights["norm1"])
        q = rotate((normalized @ weights["wq"].T).reshape(count, spec.n_heads, width))
        k = rotate((normalized @ weights["wk"].T).reshape(count, spec.n_heads, width))
        v = (normalized @ weights["wv"].T).reshape(count, spec.n_heads, width)
        attention = np.empty_like(v)
        for head in range(spec.n_heads):
            scores = q[:, head] @ k[:, head].T / np.sqrt(width) + causal
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            attention[:, head] = probability @ v[:, head]
        x = x + attention.reshape(count, hidden) @ weights["wo"].T
        normalized = norm(x, weights["norm2"])
        gate = normalized @ weights["wg"].T
        up = normalized @ weights["wu"].T
        x = x + (gate / (1 + np.exp(-gate)) * up) @ weights["wd"].T
    return norm(x, final_norm)


def simulation_config(spec, cap):
    """Only external memory capacity shrinks; compute geometry stays burst200's."""
    profile = target.load(str(PROFILE))
    cfg = target.config(profile)
    image = Image(spec, cfg, cap=cap)
    capacity = max(2 << 20, 1 << (image.required_bytes - 1).bit_length())
    cfg = replace(cfg, DRAM_BYTES=capacity)
    uarch = target.sim_uarch(profile)
    # sim_uarch intentionally exports only a subset. State the other timing
    # knobs explicitly instead of relying on the selected default board.
    for name in ("MXU_IMPL", "MXU_CL", "MXU_E", "VPU_CL", "ULANES"):
        uarch[name] = profile.uarch[name]
    uarch["COLL_GATHER"] = 0
    return profile, cfg, uarch


class ComparingRtlBackend(RtlBackend):
    """The existing RTL backend plus explicit memory knobs and an ISA oracle."""

    def __init__(self, cfg, images, *, uarch, bw, lat, stall, seed):
        super().__init__(cfg, images, uarch=uarch, axi=True, boot=True, stall=stall)
        self.oracle = IsaBackend(cfg, images)
        self.bw, self.lat, self.seed = bw, lat, seed
        self.checks = []

    def write(self, s, addr, data):
        super().write(s, addr, data)
        self.oracle.write(s, addr, data)

    def run(self, programs, args=None):
        self.oracle.run(programs, args=args)
        start = time.perf_counter()
        drams, _, stats = rtlsim.run(
            self.cfg, programs, self.drams, max_cycles=1 << 40,
            uarch=self.uarch, axi=True, boot=True, stall=self.stall,
            bw=self.bw, lat=self.lat, seed=self.seed, ldc=0, args=args, plusargs=[])
        elapsed = time.perf_counter() - start
        sizes = [len(d) for d in self.drams]
        self.drams = [d[:n].copy() for d, n in zip(drams, sizes)]
        for index, actual in enumerate(self.drams):
            expected = self.oracle.machine.slices[index].dram[:len(actual)]
            if not np.array_equal(actual, expected):
                first = int(np.argmax(actual != expected))
                raise AssertionError(f"RTL differs from ISA in slice {index} at byte {first}")
        self.checks.append(dict(cycles=int(stats["cycles"]),
                                simulator_wall_seconds=elapsed,
                                compared_dram_bytes=sum(sizes),
                                rtl_isa_bit_exact=True))
        return stats


def run_case(mode="tiny", *, frames=None, cap=256, bw=100, lat=20, stall=20, seed=192):
    if mode not in ("tiny", "layer"):
        raise ValueError("mode must be tiny or layer")
    frames = (4 if mode == "tiny" else 1) if frames is None else frames
    if not 1 <= frames <= cap <= 3000:
        raise ValueError("frames and cap must satisfy 1 <= frames <= cap <= 3000")
    if not 1 <= bw <= 100 or not 0 <= stall < 100 or lat < 0 or seed < 0:
        raise ValueError("bw must be 1..100, stall 0..99, and lat/seed nonnegative")
    check_environment()
    if shutil.which("verilator") is None:
        raise RuntimeError("Verilator is required; this check never falls back to ISA-only")
    spec = (Spec(hidden=256, layers=2, n_heads=2, ffn=512) if mode == "tiny"
            else replace(Spec(), layers=1))
    profile, cfg, uarch = simulation_config(spec, cap)
    started = time.perf_counter()
    layers, norm, inputs = synthetic_model(spec, frames, seed)
    expected = reference(spec, layers, norm, inputs)

    def factory(config, images):
        return ComparingRtlBackend(config, images, uarch=uarch,
                                   bw=bw, lat=lat, stall=stall, seed=seed)

    with Engine(spec, layers, norm, cfg=cfg, cap=cap, backend=factory) as engine:
        build_seconds = time.perf_counter() - started
        outputs = np.stack([engine.step(x) for x in inputs])
        error = float(np.linalg.norm(outputs - expected) / np.linalg.norm(expected))
        if not np.isfinite(error) or error >= 0.10:
            raise AssertionError(f"INT4 RTL vs float64 relative error {error:.6f} exceeds 0.10")
        reset_exact = None
        if mode == "tiny":
            engine.reset()
            reset_exact = bool(np.array_equal(engine.step(inputs[0]).view(np.uint32),
                                             outputs[0].view(np.uint32)))
            if not reset_exact:
                raise AssertionError("reset result differs from the fresh first frame")
        checks = engine.backend.checks
        for check in checks:
            check["milliseconds_at_200mhz"] = check["cycles"] / 200_000
        return dict(
            mode=mode, model="synthetic PersonaPlex temporal", spec=asdict(spec),
            checkpoint_used=False, full_model_measured=False, fpga_used=False,
            verilator=rtlsim.verilator_version(), target=profile.name,
            target_core_mhz=200.0, simulated_config=asdict(cfg), uarch=uarch,
            effective_rtl_parameters=rtlsim.top_params(cfg, 20, uarch, True),
            memory_model=dict(name="otpu_native_mem", bw_percent=bw,
                              minimum_latency_cycles=lat, stall_percent=stall,
                              seed=seed, boot_from_dram=True,
                              note="Synthetic two-channel native memory with latency jitter; not the AMD HBM controller or AWS shell."),
            frames=frames, cap=cap, image_bytes=engine.image.nbytes,
            weight_code_bytes=engine.image.weight_code_bytes,
            fixture_reference_and_image_build_seconds=build_seconds,
            float64_relative_error=error, float64_relative_error_limit=0.10,
            reset_bit_exact=reset_exact, runs=checks,
            output_sha256=hashlib.sha256(outputs.tobytes()).hexdigest(),
            total_wall_seconds=time.perf_counter() - started,
            timing_scope="RTL temporal kernel cycles only, excluding program boot and post-halt drain; excludes host audio, depformer, codec, PCIe and HBM effects. Simulator wall time includes build (on cache miss), file IO, boot, and drain. Last tiny run is reset verification.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiny", "layer"), default="tiny")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--bw", type=int, default=100)
    parser.add_argument("--lat", type=int, default=20)
    parser.add_argument("--stall", type=int, default=20)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_case(args.mode, frames=args.frames, cap=args.cap, bw=args.bw,
                      lat=args.lat, stall=args.stall, seed=args.seed)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
