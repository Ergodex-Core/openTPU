#!/usr/bin/env python3
"""Serving benchmark on the FPGA: each model through otpu-serve's board backend, measured from the
client's side of its streamed chat completions.

    python tools/f2_bench.py --dev f2:0 --json bench.json [--models qwen3-0.6b,gemma3-1b,...]

Per model: otpu-serve is started (weights uploaded, programs compiled: the load time), one warm-up
request, then for each prompt (a short question and a long passage to summarize) a greedy
completion of --max-tokens: TTFT (request to first streamed token: the prompt's prefill on the
card plus the first decode step), prefill tokens / s (prompt tokens / TTFT), decode tokens / s
(tokens after the first over the time after it), and otpu-serve's own /v1/stats. Inference runs on
the card only: the resident programs and the card's sampler (greedy), no host sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "f2_web"))
from gateway import MODELS, SERVE_PORT  # noqa: E402

SHORT = "Write one sentence about the Moon."
LONG = ("Summarize the following passage in two sentences.\n\n"
        "The Field-Programmable Gate Array, or FPGA, is an integrated circuit designed to be "
        "configured after manufacturing. Unlike a processor, which executes a fixed instruction "
        "set, an FPGA is a sea of configurable logic blocks, block memories and arithmetic slices "
        "joined by a programmable routing fabric. Engineers describe the circuit they want in a "
        "hardware description language such as Verilog or VHDL, and vendor tools synthesize, "
        "place and route that description onto the device, producing a bitstream that sets every "
        "switch and lookup table. Because the hardware itself is reconfigured, an FPGA can run "
        "many operations in parallel, with deterministic timing and without the overhead of "
        "fetching and decoding instructions. Cloud providers now rent FPGAs by the hour: an AWS "
        "F2 instance carries Virtex UltraScale+ devices with high-bandwidth memory, and a "
        "developer loads a pre-built image, called an AFI, onto a slot in seconds. Machine "
        "learning accelerators are a natural fit, since inference is dominated by matrix "
        "multiplication and memory bandwidth, both of which an FPGA design can tailor to a model. "
        "The trade-offs are clock speed, which is far below that of a CPU or GPU, and the long "
        "build times of the vendor tools, which can take many hours for a large design.")


def stream(model: str, prompt: str, max_tokens: int) -> dict:
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
           "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
           "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{SERVE_PORT}/v1/chat/completions",
                               data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n, text, usage = 0, [], None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n += 1
                    text.append(piece)
    t1 = time.perf_counter()
    ptoks = usage["prompt_tokens"] if usage else None
    ctoks = usage["completion_tokens"] if usage else n
    ttft = (first - t0) if first else None
    dec = (ctoks - 1) / (last - first) if first and last and last > first and ctoks > 1 else None
    return {"prompt_tokens": ptoks, "completion_tokens": ctoks, "ttft_s": ttft,
            "prefill_tok_s": (ptoks / ttft) if ptoks and ttft else None,
            "decode_tok_s": dec, "total_s": t1 - t0, "text": "".join(text)}


def get(path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}{path}", timeout=30) as r:
        return json.loads(r.read())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default=os.environ.get("OTPU_DEVICE", "f2:0"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args(argv)
    results = json.loads(a.json.read_text()) if a.json.exists() else {}
    for name in a.models.split(","):
        label, args = MODELS[name]
        log = open(ROOT / "build" / f"f2-bench-{name}.log", "w")
        cmd = [sys.executable, "-m", "opentpu.serve.api", *args, "--backend", "board",
               "--dev", a.dev, "--host", "127.0.0.1", "--port", str(SERVE_PORT),
               "--cap", str(a.cap), "--served-model-name", name]
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if p.poll() is not None:
                    raise RuntimeError(f"{name}: otpu-serve exited {p.returncode}")
                try:
                    get("/health")
                    break
                except OSError:
                    time.sleep(1)
            load = time.perf_counter() - t0
            stream(name, "Hi.", 4)                                      # warm-up
            res = {"label": label, "load_s": load, "args": args}
            for key, prompt in (("short", SHORT), ("long", LONG)):
                res[key] = stream(name, prompt, a.max_tokens)
                print(name, key, {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in res[key].items() if k != "text"},
                      repr(res[key]["text"][:120]), flush=True)
            try:
                res["server_stats"] = get("/v1/stats")
            except OSError:
                pass
            results[name] = res
            a.json.write_text(json.dumps(results, indent=1) + "\n")
        finally:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"saved {a.json}")


if __name__ == "__main__":
    main()
