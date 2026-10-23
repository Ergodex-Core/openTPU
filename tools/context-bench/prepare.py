#!/usr/bin/env python3
"""Offline Gemma benchmark cases, capacity and compilation. NEVER opens hardware."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def exact_case(tok, target):
    # Add content until the canonical template has the requested number of tokens.
    # Never slice token IDs, cut text, drop a template token or truncate a request.
    prefix = ("This is a token-length benchmark. The marker is ORCHID-731. "
              "Ignore the neutral context and follow the final instruction.\n")
    suffix = ("\nNow list consecutive integers starting at 1, separated by commas. "
              "Continue through 200. Give only the list, without a preamble.")
    filler = ""
    def encoded():
        messages = [{"role": "user", "content": prefix + filler + suffix}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return messages, [int(x) for x in ids]
    messages, ids = encoded()
    while len(ids) < target:
        for unit in (" This is neutral context." * 32, " This is neutral context.",
                     " x", " a", " 1", " ."):
            before = filler
            filler += unit
            candidate, got = encoded()
            if len(ids) < len(got) <= target:
                messages, ids = candidate, got
                break
            filler = before
        else:
            raise ValueError(f"Cannot construct exact canonical length {target}; at {len(ids)}")
    assert len(ids) == target
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    check = tok(rendered, add_special_tokens=False)["input_ids"]
    assert list(check) == ids
    return {"prompt_tokens": target, "generated_token_target": 64, "batch": 1,
            "messages": messages, "rendered_text": rendered, "input_ids": ids,
            "input_ids_sha256": digest(json.dumps(ids, separators=(",", ":")).encode()),
            "rendered_sha256": digest(rendered.encode()), "truncated": False,
            "quality_task": "consecutive_integer_list_1_to_200",
            "minimum_total_capacity": target + 64,
            "aligned_capacity": ((target + 64 + 255) // 256) * 256}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-root", type=Path, default=Path("/home/ubuntu/models"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pack", action="store_true", help="Pack real cached weights into RAM; no forward")
    a = p.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Preparation/tests belong on opentpu-m8a-build, never the Mac")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for k in list(os.environ):
        if k.startswith("OTPU_"):
            del os.environ[k]
    os.environ.update(OTPU_BUILD_JOBS="1", OTPU_IMAGE_CACHE="0", OTPU_PROG_CACHE="0",
                     OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                     TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np
    from transformers import AutoTokenizer
    from opentpu import waits as W
    from opentpu.host import board, f2
    from opentpu.llm import load_spec, qwen3 as Q, prefill as PF
    from opentpu.target import config, arch_hash

    counts = {"physical_open_attempts": 0, "forward_attempts": 0}
    def physical(*args, **kwargs):
        counts["physical_open_attempts"] += 1
        raise RuntimeError("PHYSICAL_ACCESS_FORBIDDEN")
    def forward(*args, **kwargs):
        counts["forward_attempts"] += 1
        raise RuntimeError("MODEL_FORWARD_FORBIDDEN")
    board.open_transport = physical
    f2.F2Transport.__init__ = physical
    board.SimTransport.__init__ = physical
    Q.IsaBackend.__init__ = forward
    for name in ["step", "step_rows", "prefill", "prefill_chunks", "generate_card", "generate"]:
        if hasattr(Q.Engine, name):
            setattr(Q.Engine, name, forward)

    class RAM:
        args = generates = chains = True
        streams = fills = runs_words = False
        host = None
        def __init__(self, cfg, images):
            self.cfg, self.images, self.slots = cfg, images, {}
            self.waitc = cfg.SEQ_WC
        def attach(self, eng):
            self.engine = eng
        def prepare(self, programs):
            for program in programs:
                words = Q.worker_words(self.engine.image, program)
                self.slots[id(program)] = W.loadable(words, words.waits, self.cfg.D)
        def slot_words(self, program):
            return self.slots[id(program)]
        def write(self, sid, at, data):
            data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            assert 0 <= sid < len(self.images)
            assert 0 <= at <= at + len(data) <= len(self.images[sid])
            self.images[sid][at:at + len(data)] = data
        read = run = run_generate = start = wait = forward

    cfg = config("f2-hbm:p4rl8a64c150")
    assert arch_hash(cfg) == 3714814630
    source = {}
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "opentpu").rglob("*.py")):
        source[str(path.relative_to(root))] = file_hash(path)
    results = {"evidence_kind": "offline_preparation_NOT_FPGA_performance",
               "hardware_executed": False, "production_mutated": False,
               "host": platform.node(), "python": sys.version, "numpy": np.__version__,
               "profile": "f2-hbm:p4rl8a64c150", "architecture_hash": arch_hash(cfg),
               "same_image_required": "agfi-0001de1dd3df8e21e",
               "config": dataclasses.asdict(cfg), "sources": source, "models": {}}
    for name in ["gemma-3-1b-it", "gemma-3-270m-it"]:
        model = a.models_root / name
        spec = load_spec(model)
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        item = {"spec": dataclasses.asdict(spec), "files": {}, "cases": [], "layouts": []}
        for path in sorted(model.iterdir()):
            if path.is_file() and path.suffix in (".json", ".jinja", ".model", ".safetensors"):
                item["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for target in [1024, 2048, 4032, 4096]:
            case = exact_case(tok, target)
            case["model"] = name
            case["case_id"] = f"{name}-p{target}-g64-b1-greedy"
            item["cases"].append(case)
        for cap in [768, 1280, 2304, 4096, 4352]:
            for full in [False, True]:
                img = spec.image(cfg, cap, rows=8, wformat="int8", head_format="int8",
                                 lookup=True, kv_full=full)
                programs = []
                for bucket in range(1, cap // 256 + 1):
                    progs, args = img.compile_decode(bucket, (bucket - 1) * 256)
                    words = Q.worker_words(img, progs[0], args)
                    programs.append({"bucket": bucket, "words": len(words),
                                     "sha256": digest(memoryview(words)),
                                     "fits_imem": len(words) <= cfg.IMEM_WORDS})
                item["layouts"].append({"capacity": cap, "kv_full": full, "bytes": img.nbytes,
                    "fits_memory": img.nbytes < cfg.DRAM_BYTES, "ring_tokens": img.ring,
                    "attention_window_tokens": spec.window, "stateless_kv": not img.state_regions(),
                    "head_format": img.head_format, "weights_format": img.wformat,
                    "lookup": bool(img.lookup), "host_ple": img.ple_host,
                    "generation_buckets": cap // 256, "decode_programs": programs})
        if a.pack:
            # One common 4096 TOTAL-token layout for all primary cases, with 4032+64
            # exactly fitting. 4096 INPUT is a separate 4352-capacity candidate.
            t0 = time.perf_counter()
            eng = Q.Engine(spec, Q.load_weights(model), cap=4096, cfg=cfg, backend=RAM,
                           resident=True, pipeline=False, rows=8, snapshots=0,
                           wformat="int8", head_format="int8", kv_full=True, prog_cache=False)
            pack_s = time.perf_counter() - t0
            raw = [{"bytes": len(x), "sha256": digest(memoryview(x))} for x in eng.backend.images]
            assert not eng.stateful and eng.device_inputs and eng.resident and eng.can_generate
            # These are instructions and RAM writes only, never model execution.
            generated = []
            for bucket in range(1, 17):
                progs = eng._generate_prog(bucket)
                for part in progs if isinstance(progs, tuple) else (progs,):
                    words = Q.worker_words(eng.image, part[0])
                    generated.append({"bucket": bucket, "words": len(words),
                                      "sha256": digest(memoryview(words)),
                                      "fits_imem": len(words) <= cfg.IMEM_WORDS})
            eng._generate_chain(1, 16)
            prompt = []
            for bucket in range(1, 17):
                programs, state = PF.programs(eng, (bucket - 1) * 256, 1, "P", tile=False)
                words = Q.worker_words(eng.image, programs[0])
                prompt.append({"bucket": bucket, "words": len(words),
                               "sha256": digest(memoryview(words)), "state_words": state})
            item["packed"] = {"capacity": 4096, "packing_seconds_NOT_inference": pack_s,
                              "raw_images": raw, "generate_programs": generated,
                              "prompt_programs": prompt, "stateless_kv": not eng.stateful,
                              "device_inputs": eng.device_inputs, "can_generate": eng.can_generate,
                              "host_service": eng.backend.host is not None,
                              "host_embeddings": bool(getattr(eng.image, "embed_host", False)),
                              "host_ple": bool(eng.image.ple_host)}
            del eng
        results["models"][name] = item
        (a.output / f"{name}.json").write_text(json.dumps(item, indent=2) + "\n")
        print(json.dumps({"model": name, "complete": True, "hardware_executed": False}), flush=True)
    assert counts == {"physical_open_attempts": 0, "forward_attempts": 0}
    results["guards"] = counts
    (a.output / "preparation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
