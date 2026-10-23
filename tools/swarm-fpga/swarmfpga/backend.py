"""Explicit CPU reference execution; F2 uses the native OpenTPU backend."""
import numpy as np

DTYPES = {"float32": np.dtype("float32"), "int64": np.dtype("int64")}
OPS = frozenset("input const add sub mul div neg matmul matmul_int8 sum max reshape transpose "
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
        if node["attrs"].get("compute_dtype", "fp32") not in ("fp32", "bf16", "fp16"):
            raise ValueError("unsupported compute dtype")
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


def _round(value, dtype):
    if dtype not in ("fp32", "bf16", "fp16"):
        raise ValueError("unsupported compute dtype")
    x = np.asarray(value)
    if dtype == "fp32" or x.dtype.kind != "f":
        return x
    limit = 2.0 ** 110 if dtype == "bf16" else 65504.0
    if not np.isfinite(x).all() or np.any(np.abs(x) > limit):
        raise FloatingPointError(f"{dtype} emulation range exceeded")
    if dtype == "fp16":
        return x.astype(np.float16).astype(np.float32)
    x = np.where(np.abs(x) < np.finfo(np.float32).tiny, 0, x).astype(np.float32)
    bits = x.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def _int8(x, w):
    from opentpu.quant import fake_act, fake_w
    batch = np.broadcast_shapes(x.shape[:-2], w.shape[:-2])
    left = np.broadcast_to(x, batch + x.shape[-2:])
    right = np.broadcast_to(w, batch + w.shape[-2:])
    result = np.empty(batch + (x.shape[-2], w.shape[-1]), np.float32)
    pad = (-x.shape[-1]) % 128
    for index in np.ndindex(batch):
        a = np.pad(left[index], ((0, 0), (0, pad)))
        b = np.pad(right[index].T, ((0, 0), (0, pad)))
        result[index] = fake_act(a, D=128) @ fake_w(b, "int8", D=128).T
    return result


class ReferenceBackend:
    """Explicit CPU correctness backend. Never selected as an F2 fallback."""

    device = "reference"

    def run(self, program, feeds):
        _validate(program)
        feeds = _inputs(program, feeds)
        values = {}
        simple = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": np.divide, "neg": np.negative, "matmul": np.matmul, "matmul_int8": _int8,
                  "exp": np.exp, "log": np.log, "rsqrt": lambda x: 1 / np.sqrt(x), "tanh": np.tanh}
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for node in program["nodes"]:
                op, attrs = node["op"], node["attrs"]
                dtype = attrs.get("compute_dtype", "fp32")
                args = [_round(values[i], dtype) for i in node["inputs"]]
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
                value = _round(value, dtype)
                values[node["id"]] = _array(value, "int64" if np.asarray(value).dtype.kind in "iu" else "float32", node["shape"])
        return {name: values[i] for name, i in program["outputs"].items()}
