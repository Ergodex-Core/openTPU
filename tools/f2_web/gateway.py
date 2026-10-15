#!/usr/bin/env python3
"""The F2 demo's gateway: one otpu-serve on the slot at a time, the model chosen per request.

    python tools/f2_web/gateway.py --dev f2:0 --port 8080        (on the F2 host)

The web page (tools/f2_web/lambda: a Lambda web function in the host's VPC) calls it:

  GET  /api/models      the models, the one loaded, whether a switch is in progress
  POST /api/select      {"model": NAME}: stop the running otpu-serve, start NAME's (weights
                        uploaded, programs compiled; ~10-40 s), answer once it is healthy
  POST /api/chat        {"model", "messages", "max_tokens", "temperature"}: selects the model if
                        needed, then streams otpu-serve's chat completion (SSE) through
  GET  /health

Inference runs on the FPGA only: otpu-serve's board backend with its resident programs (the
host compiles token ids into the run and computes nothing of the model) and the card's sampler,
so every request's top_k is at most 64 (above it the host would sample; docs/host.md). One
request at a time reaches the card (otpu-serve queues; the switch lock keeps a switch from
interleaving with a stream).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# name -> (label, otpu-serve arguments). The Bonsai models' weights are ternary (one scale per
# 128), which int4 holds; Bonsai-4B's int8 image (4.0 GB) is over the slice's 4 GiB besides.
MODELS = {
    "qwen3-0.6b": ("Qwen3-0.6B (int8)", ["--model", "qwen3", "--wformat", "int8",
                                         "--head-format", "int8"]),
    "gemma3-1b": ("Gemma 3 1B (int8)", ["--model", "gemma3", "--wformat", "int8",
                                       "--head-format", "int8"]),
    "bonsai-1.7b": ("Ternary-Bonsai-1.7B (int4)", ["--model", "models/Ternary-Bonsai-1.7B",
                                                  "--wformat", "int4", "--head-format", "int4"]),
    "bonsai-4b": ("Ternary-Bonsai-4B (int4, int8 head)", ["--model", "models/Ternary-Bonsai-4B",
                                                          "--wformat", "int4",
                                                          "--head-format", "int8"]),
}
TOP_K_CARD = 64                         # the card's sampler: no host sampling above
MAX_TOKENS = 512
SERVE_PORT = 8000


class Server:
    def __init__(self, dev: str, python: str, cap: int):
        self.dev, self.python, self.cap = dev, python, cap
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.state = "idle"             # idle | loading | ready | error
        self.error = ""
        self.lock = threading.Lock()    # held across a switch and across each stream

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)       # otpu-serve stops the card's run, unlocks
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc, self.model = None, None

    def select(self, name: str, timeout: float = 600.0) -> None:
        """Make `name` the loaded model (the caller holds self.lock)."""
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}")
        if self.model == name and self.proc and self.proc.poll() is None:
            return
        self._stop()
        self.state, self.error = "loading", ""
        log = open(ROOT / "build" / f"f2-web-{name}.log", "ab")
        cmd = [self.python, "-m", "opentpu.serve.api", *MODELS[name][1], "--backend", "board",
               "--dev", self.dev, "--host", "127.0.0.1", "--port", str(SERVE_PORT),
               "--cap", str(self.cap), "--served-model-name", name]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                self.state, self.error = "error", f"otpu-serve exited {self.proc.returncode}"
                raise RuntimeError(self.error)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/health", timeout=2):
                    self.model, self.state = name, "ready"
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        self._stop()
        self.state, self.error = "error", "otpu-serve did not become healthy"
        raise RuntimeError(self.error)


def make_handler(srv: Server):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):     # one line a request on stderr
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def _json(self, code: int, obj) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True})
            if self.path == "/api/models":
                return self._json(200, {"models": [{"id": k, "label": v[0]}
                                                   for k, v in MODELS.items()],
                                        "loaded": srv.model, "state": srv.state,
                                        "error": srv.error})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad JSON"})
            if self.path == "/api/select":
                try:
                    with srv.lock:
                        srv.select(body.get("model", ""))
                except (ValueError, RuntimeError) as e:
                    return self._json(400 if isinstance(e, ValueError) else 503, {"error": str(e)})
                return self._json(200, {"loaded": srv.model})
            if self.path == "/api/chat":
                return self._chat(body)
            self._json(404, {"error": "not found"})

        def _chat(self, body: dict) -> None:
            name, msgs = body.get("model", ""), body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return self._json(400, {"error": "messages required"})
            req = {"model": name, "messages": msgs, "stream": True,
                   "stream_options": {"include_usage": True},
                   "max_tokens": max(1, min(int(body.get("max_tokens") or 256), MAX_TOKENS)),
                   "temperature": float(body.get("temperature", 0.0)),
                   "top_k": max(1, min(int(body.get("top_k") or TOP_K_CARD), TOP_K_CARD)),
                   "chat_template_kwargs": {"enable_thinking": False}}
            with srv.lock:
                try:
                    srv.select(name)
                except (ValueError, RuntimeError) as e:
                    return self._json(400 if isinstance(e, ValueError) else 503, {"error": str(e)})
                up = urllib.request.Request(f"http://127.0.0.1:{SERVE_PORT}/v1/chat/completions",
                                            data=json.dumps(req).encode(),
                                            headers={"Content-Type": "application/json"})
                try:
                    resp = urllib.request.urlopen(up, timeout=600)
                except urllib.error.HTTPError as e:
                    return self._json(e.code, {"error": e.read().decode(errors="replace")[:500]})
                except (urllib.error.URLError, OSError) as e:
                    return self._json(503, {"error": str(e)})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass                         # the client went away; otpu-serve finishes
                finally:
                    resp.close()
    return H


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default=os.environ.get("OTPU_DEVICE", "f2:0"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cap", type=int, default=2048, help="otpu-serve's context, tokens")
    ap.add_argument("--preload", choices=list(MODELS), help="load this model at startup")
    a = ap.parse_args(argv)
    srv = Server(a.dev, sys.executable, a.cap)
    if a.preload:
        with srv.lock:
            srv.select(a.preload)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv))
    signal.signal(signal.SIGTERM, lambda *_: (srv._stop(), os._exit(0)))
    print(f"f2 gateway on http://{a.host}:{a.port} ({a.dev})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
