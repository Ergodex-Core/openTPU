"""Explicit state, token data, reproducible SGD and incremental generation."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid
import numpy as np
from .backend import ReferenceBackend
from .config import Config
from .model import build, initialize, shapes


class SwarmModel:
    def __init__(self, config, backend=None):
        self._owns_backend = backend is None
        self.config = config if isinstance(config, Config) else Config.from_dict(config)
        self.weights = initialize(self.config)
        if backend is None and self.config.backend == "f2" and self.config.image:
            from .opentpu import OpenTPUBackend
            backend = OpenTPUBackend(image=self.config.image)
        self.backend = backend if backend is not None else ReferenceBackend()
        if self.backend.device != self.config.backend:
            raise ValueError("backend must match the explicitly selected device")
        self.step, self.cursor = 0, 0
        self.dataset_digest = None
        self._identity, self._programs = uuid.uuid4().hex, {}

    def close(self):
        if self._owns_backend and hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def _tokens(self, tokens):
        ids = np.asarray(tokens)
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype.kind not in "iu":
            raise ValueError("tokens must be a nonempty integer [batch, sequence] array")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token ID outside vocabulary")
        return ids.astype(np.int64, copy=False)

    def _run(self, ids, cached=0, training=False, extra=None):
        key = (*ids.shape, cached, training)
        if key not in self._programs:
            self._programs[key] = build(self.config, *ids.shape, cached, training)
        return self.backend.run(self._programs[key], {**self.weights, "tokens": ids, **(extra or {})})

    def logits(self, tokens, cache=None):
        ids = self._tokens(tokens)
        if cache is not None and (cache["owner"] != self._identity or cache["version"] != self.step):
            raise ValueError("cache belongs to another model or stale weights")
        expected = {f"cache{i}.{kind}" for i in range(self.config.layers) for kind in ("k", "v")}
        if cache is not None and (set(cache["tensors"]) != expected or type(cache["length"]) is not int or cache["length"] < 1):
            raise ValueError("invalid cache tensors or length")
        cached = cache["length"] if cache else 0
        result = self._run(ids, cached, extra=cache["tensors"] if cache else None)
        logits = result.pop("logits")
        return logits, {"owner": self._identity, "version": self.step, "length": cached + ids.shape[1], "tensors": result}

    def train_step(self, tokens, targets):
        ids, target = self._tokens(tokens), self._tokens(targets)
        if target.shape != ids.shape:
            raise ValueError("targets and tokens must have the same shape")
        labels = np.zeros((*target.shape, self.config.vocab_size), np.float32)
        np.put_along_axis(labels, target[..., None], 1.0, axis=-1)
        started = time.perf_counter()
        result = self._run(ids, training=True, extra={"labels": labels})
        loss = float(result.pop("loss"))
        self.weights = result
        self.step += 1
        return {"loss": loss, "step": self.step, "wall_seconds": time.perf_counter() - started,
                "tokens": ids.size, "device": self.backend.device}

    def train(self, dataset, steps=None):
        count = self.config.steps if steps is None else steps
        if type(count) is not int or count < 0:
            raise ValueError("steps must be a nonnegative integer")
        data = self._tokens(dataset)
        if data.shape[1] != self.config.sequence_length + 1:
            raise ValueError("dataset rows must have sequence_length + 1 tokens")
        digest = hashlib.sha256(str(data.shape).encode() + data.tobytes()).hexdigest()
        if self.dataset_digest is not None and digest != self.dataset_digest:
            raise ValueError("training dataset contents or ordering changed")
        history = []
        for _ in range(count):
            indices = (np.arange(self.config.batch_size) + self.cursor) % len(data)
            batch = data[indices]
            history.append(self.train_step(batch[:, :-1], batch[:, 1:]))
            self.dataset_digest = digest
            self.cursor = (self.cursor + self.config.batch_size) % len(data)
        return history

    def generate(self, tokens, max_new_tokens=8):
        ids = self._tokens(tokens)
        if type(max_new_tokens) is not int or max_new_tokens < 0 or ids.shape[1] + max_new_tokens > self.config.context:
            raise ValueError("generation must fit context and use a nonnegative token count")
        result, current, cache = ids.copy(), ids, None
        for _ in range(max_new_tokens):
            logits, cache = self.logits(current, cache)
            current = logits[:, -1].argmax(-1).reshape(-1, 1).astype(np.int64)
            result = np.concatenate((result, current), axis=1)
        return result

    def save(self, path):
        path = Path(path)
        meta = {"format": 1, "config": self.config.to_dict(), "step": self.step, "cursor": self.cursor,
                "dataset_sha256": self.dataset_digest}
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                name = stream.name
                np.savez(stream, metadata=np.array(json.dumps(meta)), **self.weights)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path, backend=None, expected_config=None):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta.get("format") != 1 or any(type(meta.get(k)) is not int or meta[k] < 0 for k in ("step", "cursor")):
                raise ValueError("invalid checkpoint metadata")
            saved = dict(meta["config"])
            if saved.pop("socket", ""):
                raise ValueError("socket execution is no longer supported")
            config, weights = Config(**saved), {}
            if expected_config is not None and config != expected_config:
                raise ValueError("resume configuration differs from checkpoint")
            expected = shapes(config)
            if set(archive.files) != set(expected) | {"metadata"}:
                raise ValueError("checkpoint parameter names differ")
            for name, shape in expected.items():
                a = archive[name]
                if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f"invalid checkpoint parameter: {name}")
                weights[name] = a.copy()
            digest = meta.get("dataset_sha256")
            if digest is not None and (not isinstance(digest, str) or
                    len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid dataset digest")
            result = cls(config, backend)
            result.weights, result.dataset_digest = weights, digest
            result.step, result.cursor = meta["step"], meta["cursor"]
            return result
