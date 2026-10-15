"""Run Qwen3 or Gemma 3 on the board and compare its actual logits with CPU goldens.

  python tools/fpga_inference_check.py --model qwen3 --dev f2:0 --compare-isa --json qwen3.json
  python tools/fpga_inference_check.py --model gemma3 --dev f2:0 --compare-isa --json gemma3.json

The board is the default backend; it never falls back to a simulator. HF fp32 measures
model quality; --compare-isa separately checks bit-exact device execution with the
same quantized weights and architecture. Both consume the board's chosen history.
No checkpoints are downloaded. --inspect compiles the layout without opening a device.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def checkpoint_info(path):
    files = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix in (".safetensors", ".gguf"):
            files.append(dict(name=p.name, bytes=p.stat().st_size,
                              resolved=str(p.resolve())))
    return dict(path=str(path.resolve()),
                config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                weights=files)


def compare_logits(device, golden):
    a, b = np.asarray(device, np.float32), np.asarray(golden, np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("logit shape mismatch or nonfinite logits")
    af, bf = a.astype(np.float64), b.astype(np.float64)
    norm = np.linalg.norm(af) * np.linalg.norm(bf)
    return dict(argmax=int(b.argmax()), match=int(a.argmax()) == int(b.argmax()),
                cosine=float(af @ bf / norm) if norm else None,
                max_abs=float(np.abs(af - bf).max()),
                unequal_words=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))))


def compare_logit_rows(device, golden):
    """Reject incomplete oracle histories instead of silently truncating a zip."""
    if np.shape(device) != np.shape(golden):
        raise ValueError(f"logit batch shape mismatch: device {np.shape(device)}, "
                         f"golden {np.shape(golden)}")
    return [compare_logits(d, g) for d, g in zip(device, golden)]


def collect(engine, prompt, steps, stops, progress=None):
    """Run just the device, timing engine calls independently of CPU golden evaluation."""
    t0 = time.monotonic()
    for token in prompt:
        logits = engine.step(token)
    prefill_seconds = time.monotonic() - t0
    generated, outputs, decode_seconds = [], [], []
    prefill_runs = len(engine.stats)
    cycles = lambda runs: sum((s or {}).get("cycles", 0) or 0 for s in runs)

    def timing():
        return dict(prompt_engine_seconds=prefill_seconds,
                    decode_engine_seconds=list(decode_seconds),
                    prompt_cycles=cycles(engine.stats[:prefill_runs]),
                    decode_cycles=cycles(engine.stats[prefill_runs:]),
                    decode_runs=len(decode_seconds))

    for i in range(steps):
        if not np.isfinite(logits).all():
            raise ValueError("device returned nonfinite logits")
        token = int(logits.argmax())
        generated.append(token)
        outputs.append(logits.copy())
        if progress:
            progress(generated, np.asarray(outputs), timing())
        if token in stops or i + 1 == steps:
            break
        t0 = time.monotonic()
        logits = engine.step(token)
        decode_seconds.append(time.monotonic() - t0)
    return generated, np.asarray(outputs), timing()


class BoardSession:
    """Own the transport even if Engine fails before constructing its backend."""

    def __init__(self, dev, model):
        from opentpu.host.board import Board, device_config, open_transport
        self.transport = open_transport(dev)
        self.backend = None
        self.model = model
        try:
            self.info = Board(self.transport, lock=False).info()
            self.cfg = device_config(self.info)
        except BaseException:
            self.close()
            raise

    def factory(self, cfg, images):
        from opentpu.host.board import BoardBackend
        self.backend = BoardBackend(cfg, images, transport=self.transport, model=self.model,
                                    own_transport=True)
        return self.backend

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
        finally:
            self.transport.close()


def hf_logits(path, tokens):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
    with torch.inference_mode():
        return model(torch.tensor([tokens]), use_cache=False).logits[0].float().numpy()


def main(argv=None):
    from tokenizers import Tokenizer
    from opentpu import isa, target
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=("board", "isa"), default="board")
    ap.add_argument("--dev", help="f2:0, f2:<PCI BDF>, or XDMA prefix; default OTPU_DEVICE")
    ap.add_argument("--target", default="f2", help="layout for --inspect or --backend isa")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--wformat", choices=("int8", "int4", "fp4", "t2"), default="int8")
    ap.add_argument("--head-format", choices=("int8", "int4", "fp4", "t2"), default="int8")
    ap.add_argument("--rotate", action="store_true",
                    help="Ternary-Bonsai 2: keep the latent ternary weights and rotate on the "
                         "device (docs/ternary.md 4; implied by --wformat t2)")
    ap.add_argument("--golden", choices=("hf", "numpy"), default="hf")
    ap.add_argument("--compare-isa", action="store_true")
    ap.add_argument("--prompt", default="Write one sentence about the Moon.")
    ap.add_argument("--chat", action="store_true", help="apply the local chat template")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--stop-token", type=int, action="append", default=[])
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--json", help="result file, also saved before golden comparisons")
    ap.add_argument("--save-logits", help="NPZ checkpointed after device tokens and each oracle")
    a = ap.parse_args(argv)
    if a.steps < 1 or a.cap < 1:
        ap.error("--steps and --cap must be positive")
    path = model_dir(a.model)
    cfg_json = json.loads((path / "config.json").read_text())
    family = cfg_json.get("model_type")
    if family not in ("qwen3", "gemma3", "gemma3_text", "qwen3_5", "qwen3_5_text"):
        ap.error("this probe supports Qwen3, Qwen3.5 (Ternary-Bonsai 2 GGUF) and Gemma 3 text "
                 "checkpoints")
    spec = load_spec(path)
    from opentpu.llm import bonsai2
    rotated = bonsai2.is_bonsai2(path) and (a.rotate or a.wformat == "t2")
    if rotated:
        spec = bonsai2.spec(path, ternary=True)
    meta = checkpoint_info(path)
    image_kw = dict(wformat=a.wformat, head_format=a.head_format, lookup=True, rows=1)
    cfg = target.config(target.load(a.target))
    if a.inspect:
        img = spec.image(cfg, a.cap, **image_kw)
        programs, _ = img.compile_decode(1, getattr(spec, "conv_k", 1) - 1)  # (Qwen3.5: p >= K - 1)
        meta.update(image_bytes=img.nbytes, architecture=asdict(cfg),
                    program_words=[int(isa.assemble(p).size) for p in programs],
                    wformat=a.wformat, head_format=a.head_format, capacity=a.cap)
        print(json.dumps(meta, indent=2))
        if a.json:
            Path(a.json).write_text(json.dumps(meta, indent=2) + "\n")
        return 0
    if not meta["weights"]:
        ap.error(f"no checkpoint weights in {path}; download them separately")
    if a.golden == "hf" and not list(path.glob("*.safetensors")):
        ap.error("the HF golden requires an HF safetensors checkpoint, not GGUF")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    if a.chat:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        prompt = tok.apply_chat_template([dict(role="user", content=a.prompt)], tokenize=True,
                                        add_generation_prompt=True, return_dict=False,
                                        enable_thinking=False)
    else:
        prompt = tokenizer.encode(a.prompt, add_special_tokens=True).ids
    if not prompt or len(prompt) + a.steps - 1 > a.cap:
        ap.error("empty prompt or prompt plus generated history exceeds --cap")
    versions = {}
    for package in ("numpy", "torch", "transformers", "safetensors", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(checkpoint=meta, backend=a.backend, requested_device=a.dev, device=a.dev,
                  board_info=None, architecture=asdict(cfg), wformat=a.wformat,
                  head_format=a.head_format, capacity=a.cap, requested_steps=a.steps,
                  stop_tokens=a.stop_token, chat=a.chat, versions=versions, prompt=prompt,
                  generated=[], generated_text="", golden=a.golden,
                  comparison_complete=False, phase="preparing")
    arrays = dict(prompt=np.asarray(prompt, np.int64))

    def save():
        # Replace only complete files. A failed oracle leaves the most recent successful
        # device prefix (and, when complete, the ISA comparison) available for diagnosis.
        if a.save_logits:
            p = Path(a.save_logits)
            temp = p.with_name(p.name + ".tmp")
            with temp.open("wb") as f:
                np.savez(f, **arrays)
            temp.replace(p)
        if a.json:
            p = Path(a.json)
            temp = p.with_name(p.name + ".tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(p)

    started = time.monotonic()
    save()
    session = None
    try:
        result["phase"] = "loading_weights"
        save()
        W = bonsai2.Weights(path, latent=True, ffn=spec.ffn) if rotated else load_weights(path)
        backend = "isa"
        try:
            if a.backend == "board":
                result["phase"] = "opening_device"
                save()
                session = BoardSession(a.dev, path.name)
                backend, cfg = session.factory, session.cfg
                result.update(board_info=session.info, device=session.transport.dev)
            else:
                img = spec.image(cfg, a.cap, **image_kw)
                cfg = replace(cfg, DRAM_BYTES=-(-img.nbytes // 4096) * 4096)
            result.update(architecture=asdict(cfg), phase="building_uploading")
            save()                 # device identity survives an image build/upload failure
            engine_kw = dict(cap=a.cap, cfg=cfg, wformat=a.wformat, head_format=a.head_format,
                             resident=True, rows=1, pipeline=False)
            if family in ("qwen3", "qwen3_5", "qwen3_5_text"):
                engine_kw["embed_host"] = rotated    # (rotated: the model's rows on the host)
            t0 = time.monotonic()
            engine = Engine(spec, W, backend=backend, **engine_kw)
            result.update(image_bytes=engine.image.nbytes,
                          build_upload_seconds=time.monotonic() - t0, phase="device_inference")
            save()

            def progress(tokens, logits, timing):
                result.update(generated=list(tokens), timing=timing,
                              generated_text=tokenizer.decode(tokens, skip_special_tokens=False))
                arrays.update(device=logits, generated=np.asarray(tokens, np.int64))
                save()

            generated, device, _ = collect(engine, prompt, a.steps,
                                          set(spec.eos) | set(a.stop_token), progress)
        finally:
            if session is not None:
                session.close()
        del engine
        gc.collect()
        result["phase"] = "device_complete"
        save()
        print(result["generated_text"], flush=True)
        history = prompt + generated[:-1]
        if a.compare_isa:
            # Only memory capacity changes. The architecture, quantization and image layout
            # stay identical, with all embeddings resident in both engines.
            result["phase"] = "isa_comparison"
            save()
            isa_cfg = replace(cfg, DRAM_BYTES=-(-result["image_bytes"] // 4096) * 4096)
            sim = Engine(spec, W, backend="isa", **dict(engine_kw, cfg=isa_cfg))
            logits = [sim.step(t) for t in history]
            arrays["isa"] = np.asarray(logits[len(prompt) - 1:])
            result["isa_comparison"] = compare_logit_rows(device, arrays["isa"])
            result["isa_bit_exact"] = all(r["unequal_words"] == 0 for r in result["isa_comparison"])
            del sim, logits
            gc.collect()
            save()
        result["phase"] = "golden_comparison"
        save()
        if a.golden == "hf":
            del W                  # release cached fp32 matrices before HF loads its own copy
            gc.collect()
            golden = hf_logits(path, history)
        else:
            from opentpu.llm import gemma3, qwen3, qwen35
            ref = {"qwen3": qwen3, "qwen3_5": qwen35, "qwen3_5_text": qwen35}.get(family, gemma3)
            if rotated:             # the model's own weights: the rotation unfolded, tied
                W, spec = bonsai2.Weights(path), bonsai2.spec(path)
            golden = ref.reference_logits(spec, W, history)
        arrays["golden"] = golden[len(prompt) - 1:]
        result["golden_comparison"] = compare_logit_rows(device, arrays["golden"])
        result["golden_argmax_matches"] = sum(r["match"] for r in result["golden_comparison"])
        result.update(comparison_complete=True, phase="complete",
                      total_seconds=time.monotonic() - started)
        save()
    except BaseException as exc:
        result.update(failed_phase=result["phase"], phase="failed", comparison_complete=False,
                      error=dict(type=type(exc).__name__, message=str(exc)),
                      total_seconds=time.monotonic() - started)
        try:
            save()
        except OSError as report_error:
            print(f"could not save failure report: {report_error}", file=sys.stderr)
        raise
    print(json.dumps({k: result[k] for k in ("backend", "image_bytes", "generated",
                                           "golden_argmax_matches", "total_seconds")}), flush=True)
    if a.compare_isa:
        print(f"ISA bit-exact: {result['isa_bit_exact']}", flush=True)
    # A quality disagreement with an unquantized model is reported, not a hardware failure.
    return 2 if a.compare_isa and not result["isa_bit_exact"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
