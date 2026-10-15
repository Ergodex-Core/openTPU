#!/usr/bin/env python3
"""Where a served request's time to first token goes, on a host without a card: otpu-serve in this
process (--backend board-isa: the card's driver over the ISA simulator, or isa), f2_bench.py's
requests (a warm-up, a short and a long prompt), and per request the host's functions timed
between the request and its first token (compiles, wait entries, program loads, the KV store's
saves and restores, the bytes the host moved), the simulator's time taken off: host = TTFT - sim.

    python tools/ttft_probe.py --models qwen3-0.6b --backend board-isa --json probe.json

`sim` is the simulator's wall time (a card's device time is its cycles / CORE_KHZ: not known
here); `pcie_read_mb` / `pcie_write_mb` the bytes a card's PCIe would carry (the F2 small shell's
BAR4 has no DMA engine: its reads are uncached MMIO), `pcie_read_bg_mb` those read by the KV
store's background thread (off the request's path; "bg:" events are that thread's).
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "f2_web"))

EVENTS: list = []                      # (name, t0, t1, thread name)
_lock = threading.Lock()


def timed(obj, attr: str, name: str | None = None) -> None:
    f = getattr(obj, attr)
    if getattr(f, "_probe", False):
        return
    nm = name or f"{getattr(obj, '__name__', type(obj).__name__)}.{attr}"

    @functools.wraps(f)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            t1 = time.perf_counter()
            with _lock:
                EVENTS.append((nm, t0, t1, threading.current_thread().name))
    w._probe = True
    setattr(obj, attr, w)


def instrument(backend: str) -> None:
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as W
    from opentpu.host import board as B
    from opentpu.llm import batch as BA
    from opentpu.llm import generate as G
    from opentpu.llm import prefill as PF
    from opentpu.llm import qwen3 as Q
    from opentpu.llm import snapshot as SN
    from opentpu.llm import tiled as TL
    from opentpu.serve import batching as BW
    from opentpu.serve import kvio as KV
    from opentpu.serve import scheduler as SC
    from opentpu.serve import store as ST
    for obj, attr, nm in [
            (BA, "compile_step", "compile.batch_step"), (G, "compile_bucket", "compile.gen"),
            (PF, "programs", "prefill.programs"), (TL, "compile_tile_run", "compile.tile_run"),
            (TL, "compile_tile", "compile.tile"),
            (Q.Image, "compile_prompt_run", "compile.prompt_run"),
            (Q.Image, "compile_rows", "compile.rows"), (Q.Image, "compile_step", "compile.step"),
            (Q.Image, "compile_decode", "compile.decode"),
            (W, "entries", "waits.entries"), (W, "entries_many", "waits.entries_many"),
            (SN, "copy_programs", "kv.copy_programs"),
            (KV.ContextIO, "_read", "kv.read"), (KV.ContextIO, "_write", "kv.write"),
            (KV.ContextIO, "_copy", "kv.copy"), (KV.ContextIO, "copy_from", "kv.copy_from"),
            (KV.ContextIO, "read_state", "kv.read_state"),
            (KV.ContextIO, "write_state", "kv.write_state"),
            (ST.PrefixCache, "save", "store.save"), (ST.PrefixCache, "restore", "store.restore"),
            (PC, "_load", "progcache.disk_load"),
            (B.Board, "read", "board.read"), (B.Board, "write", "board.write"),
            (B.Board, "load_program", "board.load_program"),
            (B.BoardBackend, "start", "backend.start"), (B.BoardBackend, "wait", "backend.wait"),
            (B.BoardBackend, "run_generate", "backend.run_generate"),
            (Q.Engine, "generate_card", "engine.generate_card"),
            (Q.Engine, "prefill_chunks", "engine.prefill_chunks"),
            (Q.Engine, "_generate_prog", "engine.generate_prog"),
            (Q.Engine, "_generate_chain", "engine.generate_chain"),
            (BA.Steps, "_get", "steps.get_program"), (BA.Steps, "_go", "steps.run"),
            (BW.BatchWorker, "_bind", "worker.bind"),
            (BW.BatchWorker, "_prefill_run", "worker.feed_run"),
            (BW.BatchWorker, "_card_loop", "worker.card_loop"),
            (BW.BatchWorker, "_save_slot", "worker.save_slot"),
            (SC.Worker, "_switch_to", "worker.switch"), (SC.Worker, "_prefill", "worker.prefill"),
            (SC.Worker, "_decode", "worker.decode"),
            (SC.Worker, "_save_device", "worker.save_device")]:
        timed(obj, attr, nm)
    from opentpu.host import isaboard as IB
    timed(IB.IsaTransport, "_run", "sim.run")
    rd = IB.IsaTransport.mem_read

    def mem_read(self, ch, off, n, out=None):       # the bytes read off the worker's path
        if threading.current_thread().name.startswith("otpu-kv-read"):
            self.bytes_read_bg = getattr(self, "bytes_read_bg", 0) + n
        return rd(self, ch, off, n, out)
    IB.IsaTransport.mem_read = mem_read
    timed(IB.IsaTransport, "_load", "sim.load")
    timed(Q.IsaBackend, "run", "sim.run")
    _ = I


def sums(t0: float, t1: float) -> dict:
    """Seconds of each event name inside [t0, t1] (clipped), the worker threads' and others'."""
    out = collections.defaultdict(float)
    n = collections.Counter()
    with _lock:
        evs = list(EVENTS)
    for nm, a, b, th in evs:
        if th.startswith("otpu-kv-read"):       # (the KV store's background reads)
            nm = "bg:" + nm
        a, b = max(a, t0), min(b, t1)
        if b > a:
            out[nm] += b - a
            n[nm] += 1
    return {k: (round(v, 4), n[k]) for k, v in sorted(out.items())}


def stream(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
           "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
           "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n, text, usage = 0, [], None
    with urllib.request.urlopen(r, timeout=36000) as resp:
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
    return {"t0": t0, "first": first, "last": last, "t1": t1, "text": "".join(text),
            "prompt_tokens": usage["prompt_tokens"] if usage else None,
            "completion_tokens": usage["completion_tokens"] if usage else n}


def get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as r:
        return json.loads(r.read())


def transport():
    """The board-isa transport of the running server, or None."""
    import gc

    from opentpu.host.isaboard import IsaTransport
    for o in gc.get_objects():
        if isinstance(o, IsaTransport):
            return o
    return None


def main(argv=None) -> None:
    from f2_bench import LONG, SHORT
    from gateway import LOCAL_MODELS, MODELS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="qwen3-0.6b")
    ap.add_argument("--backend", default="board-isa", choices=["isa", "board-isa"])
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--prompts", default="warm,short,long,short2",
                    help="warm (Hi., 4 tokens), short, long, short2 (short again, other words)")
    ap.add_argument("--extra", default="", help="more otpu-serve arguments (space-separated)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    instrument(a.backend)
    from opentpu.serve import api
    models = {**MODELS, **LOCAL_MODELS}
    texts = {"warm": ("Hi.", 4), "short": (SHORT, a.max_tokens), "long": (LONG, a.max_tokens),
             "short2": ("Write one sentence about the Sun.", a.max_tokens)}
    results = json.loads(a.json.read_text()) if a.json and a.json.exists() else {}
    for name in a.models.split(","):
        label, args = models[name]
        argv_s = [*args, "--backend", a.backend, "--host", "127.0.0.1", "--port", str(a.port),
                  "--cap", str(a.cap), "--served-model-name", name, *a.extra.split()]
        httpd = {}
        orig = api.serve

        def grab(*x, **k):
            h = orig(*x, **k)
            httpd["h"] = h
            return h
        api.serve = grab
        t0 = time.perf_counter()
        th = threading.Thread(target=api.main, args=(argv_s,), daemon=True, name="otpu-serve")
        th.start()
        while True:
            try:
                get(a.port, "/health")
                break
            except OSError:
                if not th.is_alive():
                    raise SystemExit(f"{name}: otpu-serve ended")
                time.sleep(0.5)
        load = time.perf_counter() - t0
        tr = transport()
        res = {"label": label, "backend": a.backend, "load_s": round(load, 2),
               "load_events": sums(t0, time.perf_counter())}
        print(f"{name}: loaded in {load:.1f} s", flush=True)
        for key in a.prompts.split(","):
            prompt, mt = texts[key]
            c0 = {**tr.counters(), "bg": getattr(tr, "bytes_read_bg", 0)} if tr else None
            r = stream(a.port, name, prompt, mt)
            c1 = {**tr.counters(), "bg": getattr(tr, "bytes_read_bg", 0)} if tr else None
            ttft = r["first"] - r["t0"]
            ev = sums(r["t0"], r["first"])
            sim = ev.get("sim.run", (0, 0))[0] + ev.get("sim.load", (0, 0))[0]
            one = {"prompt_tokens": r["prompt_tokens"],
                   "completion_tokens": r["completion_tokens"], "ttft_s": round(ttft, 3),
                   "sim_s": round(sim, 3), "host_s": round(ttft - sim, 3),
                   "total_s": round(r["t1"] - r["t0"], 3), "events": ev,
                   "after_first": sums(r["first"], r["t1"]), "text": r["text"][:200]}
            if c0 is not None:
                one["pcie_read_mb"] = round((c1["bytes_read"] - c0["bytes_read"]) / 2**20, 2)
                one["pcie_read_bg_mb"] = round((c1["bg"] - c0["bg"]) / 2**20, 2)
                one["pcie_write_mb"] = round((c1["bytes_written"] - c0["bytes_written"]) / 2**20,
                                             2)
                one["sim_runs"] = c1["sim_runs"] - c0["sim_runs"]
            res[key] = one
            print(name, key, json.dumps({k: v for k, v in one.items()
                                         if k not in ("events", "after_first")}), flush=True)
            for k, v in ev.items():
                print(f"    {k:28s} {v[0]:9.3f} s  x{v[1]}", flush=True)
        try:
            res["server_stats"] = get(a.port, "/v1/stats")
        except OSError:
            pass
        results[name] = res
        if a.json:
            a.json.write_text(json.dumps(results, indent=1) + "\n")
        h = httpd.get("h")
        if h is not None:
            h.shutdown()
        th.join(timeout=600)
        api.serve = orig
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
