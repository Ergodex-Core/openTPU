"""Explicit tensor programs and symbolic first-order differentiation.

Building a graph performs shape arithmetic only. Backends execute every numerical
operation, including gradient accumulation and parameter updates.
"""
from itertools import zip_longest
from math import prod


def _shape(shape):
    shape = tuple(shape)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
        raise ValueError(f"invalid shape: {shape}")
    return shape


def _broadcast(a, b):
    result = []
    for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"cannot broadcast {a} and {b}")
        result.append(y if x == 1 else x)
    return tuple(reversed(result))


def _constant_shape(value):
    if not isinstance(value, (list, tuple)):
        return ()
    shapes = [_constant_shape(v) for v in value]
    if shapes and any(s != shapes[0] for s in shapes):
        raise ValueError("constant must be rectangular")
    return (len(value),) + (shapes[0] if shapes else ())


class Graph:
    def __init__(self):
        self.nodes = []

    def node(self, op, inputs=(), shape=(), **attrs):
        if any(t.graph is not self for t in inputs):
            raise ValueError("tensors belong to different graphs")
        node = dict(id=len(self.nodes), op=op, inputs=[t.id for t in inputs],
                    attrs=attrs, shape=list(_shape(shape)))
        self.nodes.append(node)
        return Tensor(self, node["id"])

    def input(self, name, shape, dtype="float32"):
        if not name or any(n["op"] == "input" and n["attrs"]["name"] == name for n in self.nodes):
            raise ValueError(f"duplicate or empty input name: {name}")
        return self.node("input", shape=shape, name=name, dtype=dtype)

    def const(self, value, dtype="float32"):
        value = value.tolist() if hasattr(value, "tolist") else value
        return self.node("const", shape=_constant_shape(value), value=value, dtype=dtype)

    def export(self, outputs):
        if not outputs or any(t.graph is not self for t in outputs.values()):
            raise ValueError("outputs must be nonempty and belong to this graph")
        return {"nodes": list(self.nodes), "outputs": {name: t.id for name, t in outputs.items()}}

    def grad(self, loss, params):
        if loss.graph is not self or loss.shape:
            raise ValueError("loss must be a scalar in this graph")
        if any(p.graph is not self for p in params.values()):
            raise ValueError("parameters belong to another graph")
        gradients = {loss.id: self.const(1.0)}
        for n in reversed(list(self.nodes[:loss.id + 1])):
            if n["id"] not in gradients or n["op"] in ("input", "const", "max"):
                continue
            g, y = gradients[n["id"]], Tensor(self, n["id"])
            xs = [Tensor(self, i) for i in n["inputs"]]
            a, op = n["attrs"], n["op"]
            x = xs[0]
            if op == "add": parts = [g, g]
            elif op == "sub": parts = [g, -g]
            elif op == "mul": parts = [g * xs[1], g * x]
            elif op == "div": parts = [g / xs[1], -g * x / (xs[1] * xs[1])]
            elif op == "neg": parts = [-g]
            elif op == "matmul":
                parts = [g @ xs[1].swapaxes(-1, -2), x.swapaxes(-1, -2) @ g]
            elif op == "reshape": parts = [g.reshape(x.shape)]
            elif op == "transpose": parts = [g.transpose(tuple(a["axes"].index(i) for i in range(len(x.shape))))]
            elif op == "exp": parts = [g * y]
            elif op == "log": parts = [g / x]
            elif op == "rsqrt": parts = [-0.5 * g * y * y * y]
            elif op == "tanh": parts = [g * (1.0 - y * y)]
            elif op == "broadcast_to": parts = [g]
            elif op == "sum":
                shape = tuple(1 if i in a["axis"] else d for i, d in enumerate(x.shape))
                parts = [g.reshape(shape).broadcast_to(x.shape)]
            elif op == "gather":
                parts = [self.node("scatter_add", (g, xs[1]), x.shape, axis=a["axis"], target_shape=list(x.shape)), None]
            elif op == "concat":
                split = x.shape[a["axis"]]
                parts = [g.slice(a["axis"], 0, split), g.slice(a["axis"], split, y.shape[a["axis"]])]
            elif op == "slice":
                parts = [self.node("slice_grad", (g,), x.shape, **a, target_shape=list(x.shape))]
            else:
                raise ValueError(f"no first-order gradient rule for {op}")
            for source, part in zip(xs, parts):
                if part is None:
                    continue
                part = part.reduce_to(source.shape)
                gradients[source.id] = gradients[source.id] + part if source.id in gradients else part
        return {name: gradients[p.id] if p.id in gradients else self.const(0.0).broadcast_to(p.shape)
                for name, p in params.items()}


class Tensor:
    def __init__(self, graph, node_id):
        self.graph, self.id = graph, node_id

    @property
    def shape(self):
        return tuple(self.graph.nodes[self.id]["shape"])

    def _tensor(self, other):
        return other if isinstance(other, Tensor) else self.graph.const(other)

    def _binary(self, op, other):
        other = self._tensor(other)
        return self.graph.node(op, (self, other), _broadcast(self.shape, other.shape))

    def __add__(self, other): return self._binary("add", other)
    def __sub__(self, other): return self._binary("sub", other)
    def __mul__(self, other): return self._binary("mul", other)
    def __truediv__(self, other): return self._binary("div", other)
    def __neg__(self): return self.graph.node("neg", (self,), self.shape)
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return self._tensor(other) - self
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return self._tensor(other) / self

    def __matmul__(self, other):
        other = self._tensor(other)
        if len(self.shape) < 2 or len(other.shape) < 2 or self.shape[-1] != other.shape[-2]:
            raise ValueError(f"matmul shape mismatch: {self.shape}, {other.shape}")
        shape = _broadcast(self.shape[:-2], other.shape[:-2]) + (self.shape[-2], other.shape[-1])
        return self.graph.node("matmul", (self, other), shape)

    def reshape(self, *shape):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        if shape.count(-1) == 1:
            known = prod(d for d in shape if d != -1)
            if not known or prod(self.shape) % known:
                raise ValueError("cannot infer reshape dimension")
            shape = tuple(prod(self.shape) // known if d == -1 else d for d in shape)
        if prod(shape) != prod(self.shape):
            raise ValueError("reshape changes tensor size")
        return self.graph.node("reshape", (self,), shape, target_shape=list(shape))

    def transpose(self, *axes):
        axes = tuple(axes[0]) if len(axes) == 1 and isinstance(axes[0], (list, tuple)) else axes
        axes = axes or tuple(reversed(range(len(self.shape))))
        axes = tuple(i + len(self.shape) if i < 0 else i for i in axes)
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError("transpose requires a permutation of axes")
        return self.graph.node("transpose", (self,), tuple(self.shape[i] for i in axes), axes=list(axes))

    @property
    def T(self): return self.transpose()

    def swapaxes(self, a, b):
        axes = list(range(len(self.shape)))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(axes)

    def _axes(self, axis):
        axes = tuple(range(len(self.shape))) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        axes = tuple(a + len(self.shape) if a < 0 else a for a in axes)
        if len(set(axes)) != len(axes) or any(a < 0 or a >= len(self.shape) for a in axes):
            raise ValueError("invalid reduction axis")
        return axes

    def _reduce(self, op, axis, keepdims):
        axes = self._axes(axis)
        shape = tuple(1 if i in axes else d for i, d in enumerate(self.shape)) if keepdims else tuple(
            d for i, d in enumerate(self.shape) if i not in axes)
        return self.graph.node(op, (self,), shape, axis=list(axes), keepdims=bool(keepdims))

    def sum(self, axis=None, keepdims=False): return self._reduce("sum", axis, keepdims)
    def max(self, axis=None, keepdims=True): return self._reduce("max", axis, keepdims)
    def mean(self, axis=None, keepdims=False): return self.sum(axis, keepdims) / prod(self.shape[i] for i in self._axes(axis))
    def exp(self): return self.graph.node("exp", (self,), self.shape)
    def log(self): return self.graph.node("log", (self,), self.shape)
    def rsqrt(self): return self.graph.node("rsqrt", (self,), self.shape)
    def tanh(self): return self.graph.node("tanh", (self,), self.shape)

    def broadcast_to(self, shape):
        shape = _shape(shape)
        if _broadcast(self.shape, shape) != shape:
            raise ValueError("invalid broadcast target")
        return self.graph.node("broadcast_to", (self,), shape, target_shape=list(shape))

    def reduce_to(self, shape):
        result = self
        if len(result.shape) > len(shape):
            result = result.sum(tuple(range(len(result.shape) - len(shape))))
        axes = tuple(i for i, d in enumerate(shape) if d == 1 and result.shape[i] != 1)
        return result.sum(axes, keepdims=True) if axes else result

    def gather(self, indices, axis=0):
        indices = self._tensor(indices)
        axis = self._axes(axis)[0]
        shape = self.shape[:axis] + indices.shape + self.shape[axis + 1:]
        return self.graph.node("gather", (self, indices), shape, axis=axis)

    def concat(self, other, axis=0):
        axis = self._axes(axis)[0]
        if len(self.shape) != len(other.shape) or any(x != y for i, (x, y) in enumerate(zip(self.shape, other.shape)) if i != axis):
            raise ValueError("concat shapes differ outside concatenated axis")
        shape = list(self.shape)
        shape[axis] += other.shape[axis]
        return self.graph.node("concat", (self, other), shape, axis=axis)

    def slice(self, axis, start, stop):
        axis = self._axes(axis)[0]
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start <= stop <= self.shape[axis]:
            raise ValueError("slice requires 0 <= start <= stop <= dimension")
        shape = list(self.shape)
        shape[axis] = stop - start
        return self.graph.node("slice", (self,), shape, axis=axis, start=start, stop=stop)
