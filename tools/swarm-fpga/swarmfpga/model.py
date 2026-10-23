"""One causal decoder: learned positions, RMSNorm, attention and SwiGLU."""
import numpy as np
from .graph import Graph


def shapes(c):
    result = {"embedding": (c.vocab_size, c.width), "position": (c.context, c.width),
              "norm": (c.width,), "head": (c.width, c.vocab_size)}
    for i in range(c.layers):
        for name, shape in {"norm1": (c.width,), "qkv": (c.width, 3 * c.width),
                            "proj": (c.width, c.width), "norm2": (c.width,),
                            "up": (c.width, c.ff_width), "gate": (c.width, c.ff_width),
                            "down": (c.ff_width, c.width)}.items():
            result[f"layer{i}.{name}"] = shape
    return result


def initialize(c):
    rng = np.random.default_rng(c.seed)
    return {name: (np.ones(shape) if "norm" in name else rng.normal(0, 0.02, shape)).astype(np.float32)
            for name, shape in shapes(c).items()}


def build(c, batch, length, cached=0, training=False):
    if batch < 1 or length < 1 or cached < 0 or cached + length > c.context:
        raise ValueError("invalid sequence dimensions")
    g = Graph()
    p = {name: g.input(name, shape) for name, shape in shapes(c).items()}
    ids = g.input("tokens", (batch, length), dtype="int64")
    x = p["embedding"].gather(ids) + p["position"].slice(0, cached, cached + length)
    outputs = {}
    dim = c.width // c.heads
    def norm(x, weight):
        return x * ((x * x).mean(-1, True) + 1e-5).rsqrt() * weight
    for i in range(c.layers):
        prefix = f"layer{i}."
        qkv = norm(x, p[prefix + "norm1"]) @ p[prefix + "qkv"]
        q, k, v = [qkv.slice(2, j * c.width, (j + 1) * c.width)
                   .reshape(batch, length, c.heads, dim).transpose(0, 2, 1, 3) for j in range(3)]
        if cached:
            k = g.input(f"cache{i}.k", (batch, c.heads, cached, dim)).concat(k, axis=2)
            v = g.input(f"cache{i}.v", (batch, c.heads, cached, dim)).concat(v, axis=2)
        mask = [[0.0 if col <= cached + row else -1e9 for col in range(cached + length)]
                for row in range(length)]
        scores = (q @ k.transpose(0, 1, 3, 2)) / (dim ** 0.5) + g.const(mask)
        e = (scores - scores.max(-1, True)).exp()
        attention = (e / e.sum(-1, True)) @ v
        x = x + attention.transpose(0, 2, 1, 3).reshape(batch, length, c.width) @ p[prefix + "proj"]
        h = norm(x, p[prefix + "norm2"])
        gate = h @ p[prefix + "gate"]
        # tanh form of sigmoid avoids overflow for large negative gate values.
        x = x + ((h @ p[prefix + "up"]) * gate * (1 + (gate * 0.5).tanh()) * 0.5) @ p[prefix + "down"]
        outputs.update({f"cache{i}.k": k, f"cache{i}.v": v})
    logits = norm(x, p["norm"]) @ p["head"]
    if training:
        labels = g.input("labels", (batch, length, c.vocab_size))
        shifted = logits - logits.max(-1, True)
        loss = -(labels * (shifted - shifted.exp().sum(-1, True).log())).sum() / (batch * length)
        grads = g.grad(loss, p)
        outputs = {"loss": loss, **{name: value - c.learning_rate * grads[name] for name, value in p.items()}}
    else:
        outputs["logits"] = logits
    return g.export(outputs)
