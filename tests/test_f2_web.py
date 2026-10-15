"""tools/f2_web: the gateway's chat stream and request limits against a fake otpu-serve, and the
Lambda web function's passcode and proxying (Node, when installed)."""
import http.client
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("f2_gateway", ROOT / "tools/f2_web/gateway.py")
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServe(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        FakeServe.seen.append(json.loads(self.rfile.read(n)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("Hello", " FPGA"):
            ev = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def gateway(monkeypatch):
    up = ThreadingHTTPServer(("127.0.0.1", 0), FakeServe)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    monkeypatch.setattr(gw, "SERVE_PORT", up.server_address[1])
    srv = gw.Server("f2:0", "python3", 2048)

    def select(name, timeout=0):
        if name not in gw.MODELS:
            raise ValueError(f"unknown model {name!r}")
        srv.model, srv.state = name, "ready"
    srv.select = select
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gw.make_handler(srv))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    FakeServe.seen.clear()
    yield httpd.server_address[1]
    httpd.shutdown()
    up.shutdown()


def _post(port, path, obj, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(obj), {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    return r.status, r.read().decode()


def test_gateway_streams_and_limits(gateway):
    st, body = _post(gateway, "/api/chat", {"model": "bonsai-1.7b", "top_k": 500, "max_tokens": 9999,
                                            "messages": [{"role": "user", "content": "hi"}]})
    assert st == 200 and "Hello" in body and " FPGA" in body and "[DONE]" in body
    req = FakeServe.seen[-1]
    assert req["top_k"] == gw.TOP_K_CARD and req["max_tokens"] == gw.MAX_TOKENS
    assert req["stream"] and req["chat_template_kwargs"] == {"enable_thinking": False}
    st, body = _post(gateway, "/api/chat", {"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    assert st == 400
    c = http.client.HTTPConnection("127.0.0.1", gateway, timeout=10)
    c.request("GET", "/api/models")
    d = json.loads(c.getresponse().read())
    assert [m["id"] for m in d["models"]] == list(gw.MODELS) and d["loaded"] == "bonsai-1.7b"


def test_every_model_keeps_inference_on_the_card():
    for name, (_, args) in gw.MODELS.items():
        assert "--wformat" in args and "--per-position" not in args and "--top-k" not in args, name


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_web_function_passcode_and_proxy(gateway):
    port = _free_port()
    env = dict(os.environ, GATEWAY_URL=f"http://127.0.0.1:{gateway}", PASSCODE="s3cret",
               AWS_LAMBDA_HTTP_ENDPOINT=f"127.0.0.1:{port}")
    p = subprocess.Popen(["node", str(ROOT / "tools/f2_web/lambda/index.js")], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/")
        r = c.getresponse()
        assert r.status == 200 and b"openTPU on an AWS F2 FPGA" in r.read()
        msg = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "hi"}]}
        assert _post(port, "/api/chat", msg)[0] == 401
        assert _post(port, "/api/chat", msg, {"x-passcode": "wrong!"})[0] == 401
        st, body = _post(port, "/api/chat", msg, {"x-passcode": "s3cret"})
        assert st == 200 and "Hello" in body and "[DONE]" in body
    finally:
        p.terminate()
        p.wait(5)
