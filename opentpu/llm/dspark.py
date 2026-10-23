"""Portable FP32 DSpark reference, not an FPGA execution backend.

The Qwen3 DSpark contract is from Nanbeige/vllm ``nanbeige42`` at
62f6de733d7ae63b759329993bc209e67afdf431 (qwen3_dspark.py and qwen3_dflash.py).
Target auxiliary states project to context K/V; query slots run a noncausal
decoder; a sequential low-rank Markov head conditions each proposal on the
previous proposed *target* token. The target remains responsible for verifying
every token. This implementation supports greedy fixed-length drafting only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Spec:
    hidden: int
    layers: int
    n_q: int
    n_kv: int
    head_dim: int
    ffn: int
    vocab: int
    draft_vocab: int
    target_hidden: int
    target_layers: tuple[int, ...]
    mask_token: int
    markov_rank: int
    eps: float = 1e-6
    theta: float = 1e6
    logit_scale: float = 1.0
    neox_rope: bool = True
    confidence_head: bool = False
    confidence_with_markov: bool = False
    topk: int | None = None
    block_size: int = 7

    @property
    def capture_layers(self) -> tuple[int, ...]:
        """Pre-layer target taps, as vLLM's eagle3_utils converts checkpoint IDs.

        Checkpoint target_layer_ids name layer outputs; the target capture hook
        runs before a layer and therefore takes these indices plus one.
        """
        return tuple(i + 1 for i in self.target_layers)

    def __post_init__(self):
        for name in ("hidden", "layers", "n_q", "n_kv", "head_dim", "ffn", "vocab",
                     "draft_vocab", "target_hidden", "markov_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"DSpark {name} must be positive")
        if self.n_q % self.n_kv or self.head_dim % 2:
            raise ValueError("DSpark needs integral GQA groups and even head_dim")
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("DSpark needs explicit, distinct target_layer_ids")
        if min(self.target_layers) < 0 or not 0 <= self.mask_token < self.vocab:
            raise ValueError("Invalid DSpark target layer or mask token")
        if self.eps <= 0 or self.theta <= 0 or not np.isfinite(self.logit_scale):
            raise ValueError("Invalid DSpark numerical configuration")
        if self.topk is not None and not 1 <= self.topk <= self.draft_vocab:
            raise ValueError("DSpark topk must be within draft vocabulary")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("DSpark block_size must be a positive integer")

    @classmethod
    def from_hf(cls, path, *, target_layer_ids: Sequence[int] | None = None,
                target_hidden_size: int | None = None, neox_rope: bool = True):
        """Read an actual draft config. Never guess auxiliary layer IDs.

        ``neox_rope`` must match the target (the checkpoint need not store it).
        A caller may supply layer IDs only when resolved from the official
        runtime's target-specific configuration.
        """
        c = json.loads((Path(path) / "config.json").read_text())
        d = dict(c.get("eagle_config") or {})
        d.update(c.get("dflash_config") or {})
        if c.get("model_type", "qwen3") != "qwen3":
            raise ValueError("DSpark reference supports the Qwen3 backbone only")
        if "markov_rank" not in c and "markov_rank" not in d:
            raise ValueError("Not a Qwen3 DSpark config: missing markov_rank")
        if c.get("markov_head_type", d.get("markov_head_type", "vanilla")) != "vanilla":
            raise ValueError("DSpark reference supports the vanilla low-rank Markov head only")
        if not c.get("sample_from_anchor", True):
            raise ValueError("DSpark reference supports sample_from_anchor=True only")
        if c.get("attention_bias") or c.get("mlp_bias"):
            raise ValueError("DSpark reference does not implement projection biases")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("DSpark reference requires SiLU")
        if not d.get("use_aux_hidden_state", c.get("use_aux_hidden_state", True)):
            raise ValueError("DSpark reference requires auxiliary target states")
        if (c.get("is_causal", d.get("causal", False)) or d.get("use_swa")
                or any(t != "full_attention" for t in (c.get("layer_types") or []))
                or d.get("attention_sink_bias") or c.get("add_swa_attention_sink_bias")):
            raise ValueError("DSpark reference requires full noncausal attention without sinks")
        rp = dict(c.get("rope_scaling") or {})
        rp.update(c.get("rope_parameters") or {})
        if rp.get("rope_type", rp.get("type", "default")) != "default":
            raise ValueError("DSpark reference requires default RoPE")
        if c.get("quantization_config"):
            raise ValueError("DSpark reference needs floating-point draft weights")
        ids = target_layer_ids
        if ids is None:
            ids = d.get("target_layer_ids", c.get("target_layer_ids"))
        if ids is None:
            raise ValueError("Missing explicit DSpark target_layer_ids")
        mask = d.get("mask_token_id", c.get("mask_token_id"))
        if mask is None:
            raise ValueError("Missing DSpark mask_token_id")
        hidden = int(c["hidden_size"])
        return cls(hidden=hidden, layers=int(c["num_hidden_layers"]),
                   n_q=int(c["num_attention_heads"]), n_kv=int(c["num_key_value_heads"]),
                   head_dim=int(c.get("head_dim") or hidden // c["num_attention_heads"]),
                   ffn=int(c["intermediate_size"]), vocab=int(c["vocab_size"]),
                   draft_vocab=int(c.get("draft_vocab_size") or c["vocab_size"]),
                   target_hidden=int(target_hidden_size or c.get("target_hidden_size") or hidden),
                   target_layers=tuple(int(i) for i in ids), mask_token=int(mask),
                   markov_rank=int(c.get("markov_rank", d.get("markov_rank"))), eps=float(c["rms_norm_eps"]),
                   theta=float(rp.get("rope_theta") or c.get("rope_theta") or 1e6),
                   logit_scale=float(c.get("logit_scale", 1.0)), neox_rope=neox_rope,
                   confidence_head=bool(c.get("enable_confidence_head", d.get("enable_confidence_head", False))),
                   confidence_with_markov=bool(c.get("confidence_head_with_markov", d.get("confidence_head_with_markov", False))),
                   topk=c.get("dspark_draft_topk"),
                   block_size=c.get("block_size", 7))


def _rms(x, gain, eps):
    return x * (np.float32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                                      + np.float32(eps))) * gain


def _rope(x, positions, theta, neox):
    d = x.shape[-1]
    freq = np.power(np.float32(theta), -np.arange(0, d, 2, dtype=np.float32) / d)
    angles = np.asarray(positions, dtype=np.float32)[:, None, None] * freq[None, None, :]
    co, si = np.cos(angles), np.sin(angles)
    a, b = (np.split(x, 2, axis=-1) if neox else (x[..., ::2], x[..., 1::2]))
    u, v = a * co - b * si, b * co + a * si
    if neox:
        return np.concatenate((u, v), axis=-1)
    out = np.empty_like(x)
    out[..., ::2], out[..., 1::2] = u, v
    return out


def _attention(q, k, v):
    """Noncausal grouped-query attention: every query sees every context/query key."""
    group = q.shape[1] // k.shape[1]
    # Keep the KV-head axis rather than physically repeating the cache G times.
    qr = q.reshape(q.shape[0], k.shape[1], group, q.shape[-1])
    scores = np.einsum("qhgd,khd->qhgk", qr, k) * np.float32(q.shape[-1] ** -.5)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("qhgk,khd->qhgd", probs, v).reshape(q.shape)


@dataclass(frozen=True)
class Context:
    """Projected target states only; speculative query K/V are never committed here."""
    positions: np.ndarray
    keys: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]

    def append(self, other: "Context") -> "Context":
        if len(self.keys) != len(other.keys):
            raise ValueError("DSpark context layer counts differ")
        if self.positions.size and other.positions.size and self.positions[-1] >= other.positions[0]:
            raise ValueError("DSpark context append must strictly advance positions")
        return Context(np.concatenate((self.positions, other.positions)),
                       tuple(np.concatenate((a, b)) for a, b in zip(self.keys, other.keys)),
                       tuple(np.concatenate((a, b)) for a, b in zip(self.values, other.values)))

    def truncate(self, position: int) -> "Context":
        """Discard states at or after a rejected position."""
        n = int(np.searchsorted(self.positions, position))
        return Context(self.positions[:n].copy(), tuple(x[:n].copy() for x in self.keys),
                       tuple(x[:n].copy() for x in self.values))


@dataclass(frozen=True)
class Proposal:
    tokens: np.ndarray
    logits: np.ndarray  # Draft vocabulary, after Markov bias (before any temperature).
    hidden: np.ndarray
    confidence: np.ndarray | None


class Reference:
    """Full floating-point DSpark Qwen3 draft with checkpoint-backed weight mappings.

    Weight names may have the HF checkpoint's unprefixed form or ``model.``.
    Embedding and LM head may be omitted only when compatible target tensors are
    explicitly supplied. This is a correctness oracle; it reports no FPGA TPS.
    """
    backend = "numpy-fp32-reference"

    def __init__(self, spec: Spec, weights: Mapping, *, target_weights: Mapping | None = None,
                 target_vocab_size: int | None = None, cache_weights: bool = True):
        self.spec, self.weights = spec, weights
        # Keep converted tensors on the host during inference. Lazy checkpoint
        # mappings otherwise repeatedly reload the large shared embedding/head.
        self.cache_weights, self._cache = cache_weights, {}
        self.target_weights = target_weights if target_weights is not None else {}
        self.target_vocab = target_vocab_size if target_vocab_size is not None else spec.vocab
        if not 0 < self.target_vocab <= spec.vocab:
            raise ValueError("DSpark input vocabulary must cover the target vocabulary")
        self.d2t = np.arange(spec.draft_vocab, dtype=np.int64)
        if spec.draft_vocab != self.target_vocab:
            offset = self._raw("d2t", optional=True)
            if offset is None:
                offset = self._raw("draft_id_to_target_id", optional=True)
            if offset is None:
                raise ValueError("Reduced-vocabulary DSpark requires d2t offsets")
            if not np.issubdtype(offset.dtype, np.integer) or offset.shape != self.d2t.shape:
                raise ValueError("DSpark d2t must be a draft-vocabulary integer vector")
            self.d2t += offset
        if (np.any(self.d2t < 0) or np.any(self.d2t >= self.target_vocab)
                or len(np.unique(self.d2t)) != spec.draft_vocab):
            raise ValueError("DSpark d2t contains invalid or duplicate target IDs")
        # Validate the shared-weight contract before a costly model forward.
        self._weight("embed_tokens.weight", (spec.vocab, spec.hidden), shared=True)
        self._weight("lm_head.weight", (spec.draft_vocab, spec.hidden), shared=True)

    def _raw(self, name, *, optional=False):
        for key in (name, "model." + name):
            if key in self.weights:
                return np.asarray(self.weights[key])
        if optional:
            return None
        raise ValueError(f"Missing DSpark tensor {name}")

    def _weight(self, name, shape, *, shared=False):
        if name in self._cache:
            return self._cache[name]
        raw = self._raw(name, optional=shared)
        if raw is None and shared:
            key = "lm_head.weight" if name == "lm_head.weight" else "model." + name
            if key not in self.target_weights:
                raise ValueError(f"DSpark {name} omitted without an explicit compatible target tensor")
            raw = np.asarray(self.target_weights[key])
        if raw.shape != shape:
            raise ValueError(f"DSpark {name}: shape {raw.shape}, expected {shape}")
        if not np.issubdtype(raw.dtype, np.floating):
            raise ValueError(f"DSpark {name} must be dequantized floating point")
        value = raw.astype(np.float32, copy=False)
        if not np.isfinite(np.sum(value, dtype=np.float64)):
            raise ValueError(f"DSpark {name} contains nonfinite values")
        if self.cache_weights:
            self._cache[name] = value
        return value

    def _layer(self, layer, name, shape):
        return self._weight(f"layers.{layer}.{name}.weight", shape)

    def project_context(self, auxiliary_states, positions) -> Context:
        """Project concatenated selected target states, ordered as target_layers.

        auxiliary_states: [tokens, len(target_layers), target_hidden] or its
        [tokens, len(target_layers)*target_hidden] flattening. These must be real
        target states, including correct logical-layer indexing across loops.
        """
        s = self.spec
        p = np.asarray(positions, dtype=np.int64)
        if p.ndim != 1 or np.any(p < 0) or np.any(np.diff(p) <= 0):
            raise ValueError("DSpark context positions must be nonnegative and strictly increasing")
        x = np.asarray(auxiliary_states, dtype=np.float32)
        expected = (p.size, len(s.target_layers) * s.target_hidden)
        if x.shape == (p.size, len(s.target_layers), s.target_hidden):
            x = x.reshape(expected)
        if x.shape != expected:
            raise ValueError(f"DSpark auxiliary state shape {x.shape}, expected {expected}")
        if not np.isfinite(np.sum(x, dtype=np.float64)):
            raise ValueError("DSpark auxiliary states contain nonfinite values")
        x = x @ self._weight("fc.weight", (s.hidden, expected[1])).T
        x = _rms(x, self._weight("hidden_norm.weight", (s.hidden,)), s.eps)
        keys, values = [], []
        for layer in range(s.layers):
            k = x @ self._layer(layer, "self_attn.k_proj", (s.n_kv * s.head_dim, s.hidden)).T
            v = x @ self._layer(layer, "self_attn.v_proj", (s.n_kv * s.head_dim, s.hidden)).T
            k = _rms(k.reshape(-1, s.n_kv, s.head_dim),
                     self._layer(layer, "self_attn.k_norm", (s.head_dim,)), s.eps)
            keys.append(_rope(k, p, s.theta, s.neox_rope))
            values.append(v.reshape(-1, s.n_kv, s.head_dim))
        return Context(p.copy(), tuple(keys), tuple(values))

    def backbone(self, token_ids, positions, context: Context):
        s = self.spec
        ids, p = np.asarray(token_ids, dtype=np.int64), np.asarray(positions, dtype=np.int64)
        if (ids.ndim != 1 or ids.size == 0 or ids.shape != p.shape
                or np.any(ids < 0) or np.any(ids >= s.vocab) or np.any(p < 0)
                or np.any(np.diff(p) != 1)):
            raise ValueError("Invalid DSpark query IDs or consecutive positions")
        if context.positions.size and context.positions[-1] >= p[0]:
            raise ValueError("DSpark target context must precede the anchor")
        if len(context.keys) != s.layers or len(context.values) != s.layers:
            raise ValueError("DSpark context has the wrong layer count")
        x = self._weight("embed_tokens.weight", (s.vocab, s.hidden), shared=True)[ids].copy()
        for layer in range(s.layers):
            a = _rms(x, self._layer(layer, "input_layernorm", (s.hidden,)), s.eps)
            q = a @ self._layer(layer, "self_attn.q_proj", (s.n_q * s.head_dim, s.hidden)).T
            k = a @ self._layer(layer, "self_attn.k_proj", (s.n_kv * s.head_dim, s.hidden)).T
            v = a @ self._layer(layer, "self_attn.v_proj", (s.n_kv * s.head_dim, s.hidden)).T
            q = _rms(q.reshape(-1, s.n_q, s.head_dim),
                     self._layer(layer, "self_attn.q_norm", (s.head_dim,)), s.eps)
            k = _rms(k.reshape(-1, s.n_kv, s.head_dim),
                     self._layer(layer, "self_attn.k_norm", (s.head_dim,)), s.eps)
            q, k = (_rope(t, p, s.theta, s.neox_rope) for t in (q, k))
            kv_shape = (context.positions.size, s.n_kv, s.head_dim)
            if context.keys[layer].shape != kv_shape or context.values[layer].shape != kv_shape:
                raise ValueError("DSpark context K/V shape does not match the spec")
            keys = np.concatenate((context.keys[layer], k))
            vals = np.concatenate((context.values[layer], v.reshape(-1, s.n_kv, s.head_dim)))
            out = _attention(q, keys, vals).reshape(-1, s.n_q * s.head_dim)
            x += out @ self._layer(layer, "self_attn.o_proj", (s.hidden, s.n_q * s.head_dim)).T
            a = _rms(x, self._layer(layer, "post_attention_layernorm", (s.hidden,)), s.eps)
            gate = a @ self._layer(layer, "mlp.gate_proj", (s.ffn, s.hidden)).T
            up = a @ self._layer(layer, "mlp.up_proj", (s.ffn, s.hidden)).T
            # Stable sigmoid, including very negative gates.
            sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
            x += (gate * sigmoid * up) @ self._layer(layer, "mlp.down_proj", (s.hidden, s.ffn)).T
        return _rms(x, self._weight("norm.weight", (s.hidden,)), s.eps)

    def draft(self, anchor_token: int, anchor_position: int, context: Context,
              num_tokens: int | None = None, *, query_tokens: int | None = None) -> Proposal:
        """Predict a prefix while retaining the requested noncausal query block.

        With no sizes, use the checkpoint block size for both. An explicit
        ``num_tokens`` without ``query_tokens`` retains the shorter-block API.
        A seven-query, three-prediction call computes all seven backbone states
        but applies the output/Markov heads only to the first three. Later
        predictions cannot affect this prefix, unlike later backbone queries.
        """
        s = self.spec
        if num_tokens is None:
            num_tokens = s.block_size
        if query_tokens is None:
            query_tokens = num_tokens
        if (not 0 <= anchor_token < self.target_vocab or type(num_tokens) is not int
                or type(query_tokens) is not int or num_tokens <= 0 or query_tokens < num_tokens):
            raise ValueError("Invalid DSpark anchor token or block size")
        ids = np.full(query_tokens, s.mask_token, dtype=np.int64)
        ids[0] = anchor_token
        p = np.arange(anchor_position, anchor_position + query_tokens, dtype=np.int64)
        hidden = self.backbone(ids, p, context)[:num_tokens]
        base = hidden @ self._weight("lm_head.weight", (s.draft_vocab, s.hidden), shared=True).T
        base *= np.float32(s.logit_scale)
        if not np.all(np.isfinite(base)):
            raise ValueError("DSpark backbone produced nonfinite logits")
        w1 = self._weight("markov_head.markov_w1.weight", (s.vocab, s.markov_rank))
        w2 = self._weight("markov_head.markov_w2.weight", (s.draft_vocab, s.markov_rank))
        logits = np.empty_like(base)
        tokens = np.empty(num_tokens, dtype=np.int64)
        confidence = np.empty(num_tokens, dtype=np.float32) if s.confidence_head else None
        prev = anchor_token
        for i in range(num_tokens):
            emb = w1[prev]
            if s.topk is None:
                logits[i] = base[i] + np.float32(s.logit_scale) * (w2 @ emb)
            else:
                # Official top-k path truncates base logits BEFORE Markov correction.
                ix = np.argsort(-base[i], kind="stable")[:s.topk]
                logits[i].fill(-np.inf)
                logits[i, ix] = base[i, ix] + np.float32(s.logit_scale) * (w2[ix] @ emb)
            if (np.any(np.isnan(logits[i])) or np.any(np.isposinf(logits[i]))
                    or not np.any(np.isfinite(logits[i]))):
                raise ValueError("DSpark Markov head produced invalid logits")
            tokens[i] = self.d2t[int(np.argmax(logits[i]))]
            if confidence is not None:
                inp = np.concatenate((hidden[i], emb)) if s.confidence_with_markov else hidden[i]
                w = self._weight("confidence_head.proj.weight", (1, inp.size))
                b = self._weight("confidence_head.proj.bias", (1,))
                z = (w @ inp + b)[0]
                confidence[i] = np.exp(-np.logaddexp(np.float32(0), -z))
            prev = int(tokens[i])
        return Proposal(tokens, logits, hidden, confidence)


@dataclass(frozen=True)
class Verification:
    accepted: int
    tokens: tuple[int, ...]


def verify_greedy(draft_tokens, target_logits, *, max_tokens: int | None = None,
                  eos_tokens: Sequence[int] = ()) -> Verification:
    """Greedy prefix acceptance plus the first correction or all-accepted bonus.

    Target rows must be obtained on [anchor, *draft_tokens], in that order.
    The caller must truncate target/draft caches to the committed prefix before
    another iteration. No stochastic or sampled-distribution claim is made.
    """
    draft, logits = np.asarray(draft_tokens), np.asarray(target_logits)
    if draft.ndim != 1 or logits.ndim != 2 or logits.shape[0] != draft.size + 1:
        raise ValueError("Target verification needs one logits row per anchor and draft token")
    if (not np.issubdtype(draft.dtype, np.integer) or np.any(draft < 0)
            or np.any(draft >= logits.shape[1]) or not np.all(np.isfinite(logits))):
        raise ValueError("Invalid draft tokens or target logits")
    limit = draft.size + 1 if max_tokens is None else int(max_tokens)
    if limit < 0:
        raise ValueError("max_tokens must be nonnegative")
    chosen = np.argmax(logits, axis=-1)
    accepted, emitted = 0, []
    for i, token in enumerate(chosen):
        if len(emitted) >= limit:
            break
        token = int(token)
        emitted.append(token)
        match = i < draft.size and token == draft[i]
        if match:
            accepted += 1
        if token in eos_tokens or not match:
            break
    return Verification(accepted, tuple(emitted))
