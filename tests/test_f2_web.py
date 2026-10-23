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


def test_gateway_rejects_bad_fields(gateway):
    msgs = [{"role": "user", "content": "hi"}]
    for bad in ({"max_tokens": "abc"}, {"temperature": "hot"}, {"top_k": [1]}):
        st, body = _post(gateway, "/api/chat", {"model": "gemma3-1b", "messages": msgs, **bad})
        assert st == 400 and "numbers" in body, bad
    for raw in (b"[]", b'"x"', b"not json"):
        c = http.client.HTTPConnection("127.0.0.1", gateway, timeout=10)
        c.request("POST", "/api/chat", raw, {"Content-Type": "application/json"})
        assert c.getresponse().status == 400, raw
    assert not FakeServe.seen
    st, _ = _post(gateway, "/api/chat", {"model": "gemma3-1b", "messages": msgs, "temperature": None})
    assert st == 200 and FakeServe.seen[-1]["temperature"] == 0.0


def test_gateway_backend_reaches_otpu_serve(monkeypatch, tmp_path):
    seen = {}

    class Proc:
        returncode = 1

        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd

        def poll(self):
            return 1
    monkeypatch.setattr(gw, "ROOT", tmp_path)          # (build/ made for the log)
    monkeypatch.setattr(gw.subprocess, "Popen", Proc)
    for backend in ("board", "isa"):
        with pytest.raises(RuntimeError):
            gw.Server("f2:0", "python3", 2048, backend).select("gemma3-1b")
        cmd = seen["cmd"]
        assert cmd[cmd.index("--backend") + 1] == backend
        assert cmd.count("--backend") == 1 and cmd.index("--backend") > cmd.index("--model")


def test_gateway_models_filter(monkeypatch):
    class Httpd:
        def __init__(self, addr, handler):
            pass

        def serve_forever(self):
            pass
    monkeypatch.setattr(gw, "ThreadingHTTPServer", Httpd)
    monkeypatch.setattr(gw, "MODELS", dict(gw.MODELS))
    monkeypatch.setattr(gw.signal, "signal", lambda *a: None)
    gw.main(["--backend", "isa", "--models", "gemma3-1b,functiongemma-270m", "--port", "0"])
    assert list(gw.MODELS) == ["gemma3-1b", "functiongemma-270m"]
    assert gw.MODELS["functiongemma-270m"] == gw.LOCAL_MODELS["functiongemma-270m"]
    for argv in (["--models", "nope"], ["--models", "gemma3-1b", "--preload", "qwen3-0.6b"],
                 ["--backend", "board-sim"]):
        with pytest.raises(SystemExit):
            gw.main(argv)


def test_page_decode_rate_skips_a_burst():
    """The ISA simulator hands a run's tokens over at once: the rate is first to last token,
    none when they came together; the run's status is set after the model list's."""
    page = (ROOT / "tools/f2_web/lambda/public/index.html").read_text()
    assert "tLast - tFirst >= 50" in page and "(tLast - tFirst) / 1000" in page
    assert page.index("await loadModels();") < page.index("status(`Done on the FPGA")


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


def _arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


@pytest.mark.parametrize("name", list(gw.MODELS))
def test_every_card_model_fits_its_slots(name, capsys):
    """Each card model's image, with otpu-serve's KV slots at the gateway's context, its batch
    area and the program area, fits the f2 slice's 4 GiB (tools/format_agree.py fit: the
    checkpoint's config only; Gemma 3 runs one sequence)."""
    args = gw.MODELS[name][1]
    model = _arg(args, "--model")
    path = ROOT / "models" / {"qwen3": "Qwen3-0.6B", "gemma3": "gemma-3-1b-it"}.get(model, "") \
        if not model.startswith("models/") else ROOT / model
    if not (path / "config.json").exists():
        pytest.skip(f"no checkpoint config in {path}")
    spec_ = importlib.util.spec_from_file_location("format_agree", ROOT / "tools/format_agree.py")
    fa = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(fa)
    slots = int(_arg(args, "--slots") or 8)
    if json.loads((path / "config.json").read_text()).get("model_type", "").startswith("gemma"):
        slots = 1
    fa.main(["fit", str(path), "--wformat", _arg(args, "--wformat"), "--head-format",
             _arg(args, "--head-format"), "--cap", "2048", "--slots", str(slots)])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["fits"], name
