"""Reference execution and a strict, optional F2 worker integration boundary.

Wire protocol: one UTF-8 JSON line per Unix-socket connection. Requests contain
``protocol`` and ``op`` (``hello`` or ``run``). Replies must contain ``ok: true``.
Hello replies also contain ``protocol``, ``device: f2``, ``dtypes`` and ``ops``.
Run requests contain ``program`` and ``feeds``; replies contain ``outputs``.
Wire tensors are {dtype, shape, data}, with flat, row-major data. The external
worker owns FPGA execution; this module provides no native OpenTPU lowering.
"""

import json
import socket

import numpy as np

PROTOCOL = "swarmfpga.tensor.v1"
DTYPES = {"float32": np.dtype("float32"), "int64": np.dtype("int64")}
OPS = frozenset("input const add sub mul div neg matmul sum max reshape transpose "
                "exp log rsqrt tanh broadcast_to gather scatter_add concat slice slice_grad".split())


def _array(value, dtype, shape):
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    array = np.asarray(value)
    if dtype == "int64" and array.dtype.kind not in "iu":
        raise ValueError("Integer tensors require integer values")
    if array.shape != tuple(shape):
        raise ValueError(f"Tensor shape {array.shape} does not match {tuple(shape)}")
    with np.errstate(over="raise", invalid="raise"):
        array = array.astype(DTYPES[dtype], copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite tensor values")
    return array


def _validate(program):
    nodes = {}
    for node in program["nodes"]:
        if node["id"] in nodes or node["op"] not in OPS:
            raise ValueError("Duplicate node ID or unsupported operation")
        if any(i not in nodes for i in node["inputs"]):
            raise ValueError("Graph must be in dependency order")
        if any(type(d) is not int or d < 0 for d in node["shape"]):
            raise ValueError("Invalid tensor shape")
        nodes[node["id"]] = node
    if not program["outputs"] or any(i not in nodes for i in program["outputs"].values()):
        raise ValueError("Invalid graph outputs")
    return nodes


def _inputs(program, feeds):
    result = {}
    for node in program["nodes"]:
        if node["op"] == "input":
            attrs = node["attrs"]
            name = attrs["name"]
            result[name] = _array(feeds[name], attrs.get("dtype", "float32"), node["shape"])
    return result


def _dtypes(program):
    result = {}
    preserves_integer = {"add", "sub", "mul", "neg", "matmul", "sum", "max", "reshape",
                         "transpose", "broadcast_to", "gather", "concat", "slice"}
    for node in program["nodes"]:
        integer = node["op"] in preserves_integer and all(result[i] == "int64" for i in node["inputs"])
        result[node["id"]] = node["attrs"].get("dtype", "int64" if integer else "float32")
    return result


def _slice(attrs, rank):
    index = [slice(None)] * rank
    index[attrs["axis"]] = slice(attrs["start"], attrs["stop"])
    return tuple(index)


def _scatter(grad, indices, attrs):
    result = np.zeros(attrs["target_shape"], dtype=np.float32)
    axis = attrs["axis"]
    if np.any(indices < 0) or np.any(indices >= result.shape[axis]):
        raise ValueError("Gather indices outside tensor bounds")
    coordinates = np.indices(grad.shape, sparse=True)
    index = list(coordinates[:axis])
    index.append(indices.reshape((1,) * axis + indices.shape + (1,) * (result.ndim - axis - 1)))
    index.extend(coordinates[axis + indices.ndim:])
    np.add.at(result, tuple(index), grad)
    return result


class ReferenceBackend:
    """Explicit CPU correctness backend. Never selected as an F2 fallback."""

    device = "reference"

    def run(self, program, feeds):
        _validate(program)
        feeds = _inputs(program, feeds)
        values = {}
        simple = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": np.divide, "neg": np.negative, "matmul": np.matmul,
                  "exp": np.exp, "log": np.log, "rsqrt": lambda x: 1 / np.sqrt(x), "tanh": np.tanh}
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for node in program["nodes"]:
                op, attrs = node["op"], node["attrs"]
                args = [values[i] for i in node["inputs"]]
                if op == "input":
                    value = feeds[attrs["name"]]
                elif op == "const":
                    value = _array(attrs["value"], attrs.get("dtype", "float32"), node["shape"])
                elif op in simple:
                    value = simple[op](*args)
                elif op in ("sum", "max"):
                    value = getattr(np, op)(args[0], axis=tuple(attrs["axis"]), keepdims=attrs["keepdims"])
                elif op == "reshape":
                    value = args[0].reshape(attrs["target_shape"])
                elif op == "transpose":
                    value = args[0].transpose(attrs["axes"])
                elif op == "broadcast_to":
                    value = np.broadcast_to(args[0], attrs["target_shape"])
                elif op == "gather":
                    if np.any(args[1] < 0) or np.any(args[1] >= args[0].shape[attrs["axis"]]):
                        raise ValueError("Gather indices outside tensor bounds")
                    value = np.take(*args, axis=attrs["axis"])
                elif op == "scatter_add":
                    value = _scatter(*args, attrs)
                elif op == "concat":
                    value = np.concatenate(args, axis=attrs["axis"])
                elif op == "slice":
                    value = args[0][_slice(attrs, args[0].ndim)]
                elif op == "slice_grad":
                    value = np.zeros(attrs["target_shape"], dtype=np.float32)
                    value[_slice(attrs, value.ndim)] = args[0]
                values[node["id"]] = _array(value, "int64" if np.asarray(value).dtype.kind in "iu" else "float32", node["shape"])
        return {name: values[i] for name, i in program["outputs"].items()}


class F2Backend:
    """Submit the entire numerical graph to an explicitly configured F2 worker.

    This is a transport contract, not an implemented FPGA kernel library. Missing
    workers or capabilities raise errors. No host-side model or optimizer runs.
    The simple v1 protocol transfers supplied feeds per call; it is not a
    persistent device-buffer or zero-copy API. Message size is explicitly bounded.
    """

    device = "f2"

    def __init__(self, socket_path, timeout=30, max_bytes=16 * 1024 * 1024):
        self.socket_path, self.timeout, self.max_bytes = socket_path, timeout, max_bytes
        hello = self._rpc({"op": "hello"})
        if hello.get("protocol") != PROTOCOL or hello.get("device") != "f2":
            raise ValueError("Worker must identify as an F2 worker with the exact protocol")
        if not set(DTYPES).issubset(hello.get("dtypes", [])):
            raise ValueError("Worker must support float32 and int64 tensors")
        self.ops = set(hello.get("ops", []))

    def _rpc(self, request):
        wire = json.dumps({"protocol": PROTOCOL, **request}, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(wire) > self.max_bytes:
            raise ValueError("F2 request exceeds configured message-size limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(self.timeout)
            stream.connect(self.socket_path)
            stream.sendall(wire)
            data = bytearray()
            while b"\n" not in data:
                chunk = stream.recv(min(65536, self.max_bytes + 1 - len(data)))
                if not chunk:
                    raise ConnectionError("F2 worker closed without a complete reply")
                data.extend(chunk)
                if len(data) > self.max_bytes:
                    raise ValueError("F2 reply exceeds configured message-size limit")
        response = json.loads(data.split(b"\n", 1)[0])
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"F2 worker error: {response}")
        return response

    def run(self, program, feeds):
        nodes = _validate(program)
        dtypes = _dtypes(program)
        missing = {node["op"] for node in program["nodes"]} - self.ops
        if missing:
            raise ValueError(f"F2 worker lacks operations: {sorted(missing)}")
        inputs = _inputs(program, feeds)
        if sum(a.nbytes for a in inputs.values()) > self.max_bytes:
            raise ValueError("F2 tensors exceed configured message-size limit before encoding")
        encoded = {name: {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}
                   for name, a in inputs.items()}
        outputs = self._rpc({"op": "run", "program": program, "feeds": encoded}).get("outputs", {})
        if set(outputs) != set(program["outputs"]):
            raise ValueError("F2 worker returned incorrect output names")
        result = {}
        for name, spec in outputs.items():
            if spec["dtype"] != dtypes[program["outputs"][name]]:
                raise ValueError("F2 worker returned an incorrect output dtype")
            if tuple(spec["shape"]) != tuple(nodes[program["outputs"][name]]["shape"]):
                raise ValueError("F2 worker returned an incorrect output shape")
            value = np.asarray(spec["data"]).reshape(spec["shape"])
            result[name] = _array(value, spec["dtype"], spec["shape"])
        return result
