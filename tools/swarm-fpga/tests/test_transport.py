"""Contract tests use a fake local worker; they do not validate F2 hardware."""

import json
import socket
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from swarmfpga.backend import F2Backend, OPS, PROTOCOL, ReferenceBackend
from swarmfpga.graph import Graph


class Worker:
    def __init__(self, reply=None, hello=None):
        self.directory = tempfile.TemporaryDirectory(prefix="sfpga-", dir="/tmp")
        self.path = str(Path(self.directory.name) / "worker.sock")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen()
        self.server.settimeout(0.05)
        self.stopped = threading.Event()
        self.requests = []
        self.reply = reply
        self.hello = hello or dict(ok=True, protocol=PROTOCOL, device="f2", dtypes=["float32", "int64"], ops=sorted(OPS))
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()
        assert not self.thread.is_alive()

    def serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            with connection:
                data = bytearray()
                while b"\n" not in data:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data.extend(chunk)
                request = json.loads(data)
                self.requests.append(request)
                if request["op"] == "hello":
                    response = self.hello
                elif self.reply:
                    response = self.reply(request)
                else:
                    feeds = {name: np.asarray(t["data"], dtype=t["dtype"]).reshape(t["shape"])
                             for name, t in request["feeds"].items()}
                    outputs = ReferenceBackend().run(request["program"], feeds)
                    response = {"ok": True, "outputs": {
                        name: {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}
                        for name, a in outputs.items()}}
                connection.sendall(json.dumps(response).encode() + b"\n")


def program():
    graph = Graph()
    x = graph.input("x", (2, 2))
    return graph.export({"y": x * 2 + 1}), {"x": np.arange(4, dtype=np.float32).reshape(2, 2)}


def test_f2_protocol_forwards_graph_and_validates_outputs():
    graph, feeds = program()
    with Worker() as worker:
        result = F2Backend(worker.path).run(graph, feeds)
        np.testing.assert_array_equal(result["y"], feeds["x"] * 2 + 1)
        assert worker.requests[1]["program"] == graph
        assert worker.requests[1]["protocol"] == PROTOCOL


@pytest.mark.parametrize("change", [{"device": "cpu"}, {"protocol": "other"}, {"dtypes": ["float32"]}])
def test_f2_rejects_incompatible_worker(change):
    hello = dict(ok=True, protocol=PROTOCOL, device="f2", dtypes=["float32", "int64"], ops=sorted(OPS))
    with Worker(hello={**hello, **change}) as worker:
        with pytest.raises(ValueError):
            F2Backend(worker.path)


def test_f2_rejects_missing_kernel_before_submission():
    hello = dict(ok=True, protocol=PROTOCOL, device="f2", dtypes=["float32", "int64"], ops=["input"])
    with Worker(hello=hello) as worker:
        with pytest.raises(ValueError, match="lacks operations"):
            F2Backend(worker.path).run(*program())
        assert len(worker.requests) == 1


@pytest.mark.parametrize("outputs", [
    {},
    {"y": {"dtype": "float32", "shape": [4], "data": [0, 1, 2, 3]}},
    {"y": {"dtype": "float32", "shape": [2, 2], "data": [0, 1, 2, float("nan")]}},
    {"y": {"dtype": "float64", "shape": [2, 2], "data": [0, 1, 2, 3]}},
    {"y": {"dtype": "int64", "shape": [2, 2], "data": [0, 1, 2, 3]}},
])
def test_f2_rejects_invalid_worker_output(outputs):
    with Worker(reply=lambda _: {"ok": True, "outputs": outputs}) as worker:
        with pytest.raises(ValueError):
            F2Backend(worker.path).run(*program())


def test_f2_has_no_cpu_fallback_for_missing_worker():
    with tempfile.TemporaryDirectory(prefix="sfpga-", dir="/tmp") as directory:
        with pytest.raises(OSError):
            F2Backend(str(Path(directory) / "missing.sock"))


def test_f2_bounds_requests_and_replies():
    with Worker(reply=lambda _: {"ok": True, "padding": "x" * 5000}) as worker:
        backend = F2Backend(worker.path, max_bytes=2000)
        with pytest.raises(ValueError, match="reply exceeds"):
            backend.run(*program())
        backend.max_bytes = 100
        with pytest.raises(ValueError, match="request exceeds"):
            backend.run(*program())


def test_reference_rejects_invalid_inputs():
    graph, _ = program()
    with pytest.raises(ValueError, match="shape"):
        ReferenceBackend().run(graph, {"x": np.zeros((4,))})
    with pytest.raises(ValueError, match="Non-finite"):
        ReferenceBackend().run(graph, {"x": np.full((2, 2), np.nan)})
