"""A single resident Qwen3-TTS talker block, not a speech-generation engine.

All learned inference math in this slice executes in ISA programs. The host
packs weights/constants offline, supplies two integer token IDs, and chooses
the sequential position. No model download, device opener, or host fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import quant as Q
from ..compiler import Affine, KVDesc, QTensor, Tensor, k_scale_bytes
from ..isasim import Machine
from ..kernels.tts_talker import talker_block
from .qwen3_tts import ProjectionSpec


@dataclass(frozen=True)
class TalkerSpec:
    text_vocab: int = 151936
    codec_vocab: int = 3072
    text_hidden: int = 2048
    hidden: int = 1024
    n_q: int = 16
    n_kv: int = 8
    head_dim: int = 128
    ffn: int = 3072
    eps: float = 1e-6
    theta: float = 1e6

    @classmethod
    def from_config(cls, config):
        ProjectionSpec.from_config(config)
        c = config["talker_config"]
        expected = {"text_vocab_size": 151936, "vocab_size": 3072,
                    "num_attention_heads": 16, "num_key_value_heads": 8,
                    "head_dim": 128, "intermediate_size": 3072,
                    "num_hidden_layers": 28, "attention_bias": False,
                    "use_sliding_window": False, "sliding_window": None,
                    "rms_norm_eps": 1e-6, "rope_theta": 1e6,
                    "max_position_embeddings": 32768}
        if any(c.get(k) != value for k, value in expected.items()):
            raise ValueError("unsupported talker block configuration")
        r = c.get("rope_scaling", {})
        if (r.get("rope_type") != "default" or r.get("interleaved") is not True
                or r.get("mrope_section") != [24, 20, 20]):
            raise ValueError("unsupported talker rotary configuration")
        return cls()

    def check(self, cfg, capacity, attention_block):
        dims = (self.text_vocab, self.codec_vocab, self.text_hidden, self.hidden,
                self.n_q, self.n_kv, self.head_dim, self.ffn)
        if any(type(n) is not int or n <= 0 for n in dims):
            raise ValueError("positive integer model dimensions required")
        if cfg.S != 1:
            raise ValueError("this bounded talker slice supports one FPGA slice")
        if self.n_q % self.n_kv or self.n_q // self.n_kv > cfg.MCOLS:
            raise ValueError("query/KV grouping does not fit MCOLS")
        if any(n % cfg.D for n in (self.text_hidden, self.hidden, self.head_dim, self.ffn)):
            raise ValueError("dimensions must align to the MXU depth")
        if self.head_dim % 2 or max(self.text_hidden, self.hidden, self.ffn,
                                   self.n_q * self.head_dim) > cfg.D * cfg.ACT_BLOCKS:
            raise ValueError("invalid rotary dimensions or ACT RAM capacity")
        if (type(capacity) is not int or not 0 < capacity <= 32768
                or capacity % cfg.D):
            raise ValueError("capacity must be a positive multiple of D, at most 32768")
        if (type(attention_block) is not int or attention_block <= 0
                or attention_block % cfg.D or capacity % attention_block):
            raise ValueError("attention block must be D-aligned and divide capacity")
        if max(self.text_vocab, self.codec_vocab) >= 2**24:
            raise ValueError("token IDs must be exactly representable as fp32 integers")
        if not (math.isfinite(self.eps) and self.eps > 0 and
                math.isfinite(self.theta) and self.theta > 0):
            raise ValueError("finite positive RMS epsilon and rotary theta required")


def _tensor(address, shape):
    strides, stride = [], 1
    for n in reversed(shape):
        strides.append(stride)
        stride *= n
    return Tensor(Affine(address), tuple(shape), tuple(reversed(strides)))


class TalkerBlockImage:
    """Descriptor-only native layout; build() separately packs provided tensors.

    FP32 embedding tables/norms/biases/RoPE; INT8 affine weights, activations,
    and KV. There is no automatic embedding offload if residency fails.
    """
    def __init__(self, spec, cfg, capacity=256, attention_block=None):
        attention_block = min(256, capacity) if attention_block is None else attention_block
        spec.check(cfg, capacity, attention_block)
        self.spec, self.cfg, self.capacity = spec, cfg, capacity
        self.attention_block = attention_block
        self.nbytes, self.regions, self.parameters = 0, {}, {}

        def alloc(name, size):
            address = self.nbytes
            self.regions[name] = (address, size)
            self.nbytes = ((address + size + 127) // 128) * 128
            return address

        def tensor(name, shape, checkpoint=None):
            t = _tensor(alloc(name, 4 * math.prod(shape)), shape)
            if checkpoint is not None:
                self.parameters[checkpoint] = ("fp32", t, shape)
            return t

        def weight(name, shape, checkpoint):
            n, k = shape
            data = alloc(name + ".data", n * Q.row_bytes(k, "int8", cfg.D))
            scale = alloc(name + ".scales", n * Q.scale_bytes(k, "int8", cfg.D))
            t = QTensor.weight(data, scale, n, k, "int8", cfg.D)
            self.parameters[checkpoint] = ("int8", t, shape)
            return t

        h, t, d, f = spec.hidden, spec.text_hidden, spec.head_dim, spec.ffn
        m = SimpleNamespace(spec=spec)
        m.ids = tensor("ids", (2,))
        m.text_scratch = tensor("text_scratch", (1, t))
        m.projected = tensor("projected", (1, h))
        m.assembled = tensor("assembled", (1, h))
        m.attention_residual = tensor("attention_residual", (1, h))
        m.output = tensor("output", (1, h))
        m.text_embedding = tensor("text_embedding", (spec.text_vocab, t),
                                  "talker.model.text_embedding.weight")
        m.codec_embedding = tensor("codec_embedding", (spec.codec_vocab, h),
                                   "talker.model.codec_embedding.weight")
        p = "talker.text_projection."
        m.fc1 = weight("fc1", (t, t), p + "linear_fc1.weight")
        m.bias1 = tensor("bias1", (t,), p + "linear_fc1.bias")
        m.fc2 = weight("fc2", (h, t), p + "linear_fc2.weight")
        m.bias2 = tensor("bias2", (h,), p + "linear_fc2.bias")
        p = "talker.model.layers.0."
        m.input_norm = tensor("input_norm", (h,), p + "input_layernorm.weight")
        m.post_norm = tensor("post_norm", (h,), p + "post_attention_layernorm.weight")
        m.q_norm = tensor("q_norm", (d,), p + "self_attn.q_norm.weight")
        m.k_norm = tensor("k_norm", (d,), p + "self_attn.k_norm.weight")
        for name, shape, suffix in (
                ("wq", (spec.n_q * d, h), "self_attn.q_proj.weight"),
                ("wk", (spec.n_kv * d, h), "self_attn.k_proj.weight"),
                ("wv", (spec.n_kv * d, h), "self_attn.v_proj.weight"),
                ("wo", (h, spec.n_q * d), "self_attn.o_proj.weight"),
                ("gate", (f, h), "mlp.gate_proj.weight"),
                ("up", (f, h), "mlp.up_proj.weight"),
                ("down", (h, f), "mlp.down_proj.weight")):
            setattr(m, name, weight(name, shape, p + suffix))
        m.cosine = tensor("cosine", (capacity, d // 2))
        m.sine = tensor("sine", (capacity, d // 2))
        heads = {}
        for head in range(spec.n_kv):
            heads[head] = {key: alloc(f"kv.{head}.{key}", size) for key, size in (
                ("k", capacity * d), ("ks", capacity * k_scale_bytes(d, cfg.D)),
                ("vt", capacity * d), ("vs", capacity * 4))}
        m.kv = KVDesc(heads, capacity, d, cfg.D, 1, 0)
        self.desc = m
        if self.nbytes > min(cfg.DRAM_BYTES, 2**32):
            raise MemoryError(f"resident TTS block needs {self.nbytes} device bytes; no offload")

    def build(self, weights, *, max_bytes=256 << 20):
        """Offline conversion of supplied arrays. Defaults to a bounded test allocation."""
        if self.nbytes > max_bytes:
            raise MemoryError("image exceeds explicit host packing budget; use descriptor-only compile")
        image = np.zeros(self.nbytes, np.uint8)

        def put(address, data):
            data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            image[address:address + data.size] = data

        for name, (fmt, desc, shape) in self.parameters.items():
            a = np.asarray(weights[name], np.float32)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"{name}: expected finite {shape} tensor")
            a = F.ftz(a)
            if fmt == "fp32":
                put(desc.base.const, a)
            else:
                quantized, scales = Q.quantize_mxu(a, fmt, self.cfg.D)
                put(desc.data.const, quantized)
                put(desc.scale.const, scales)
        # Fixed tables are prepared once, not computed by the host per request.
        d = self.spec.head_dim
        inv = np.float32(1.0) / np.power(np.float32(self.spec.theta),
                                         np.arange(0, d, 2, dtype=np.float32) / d)
        angles = np.arange(self.capacity, dtype=np.float32)[:, None] * inv[None, :]
        put(self.desc.cosine.base.const, np.cos(angles).astype(np.float32))
        put(self.desc.sine.base.const, np.sin(angles).astype(np.float32))
        return image

    def compile(self, position):
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError("position outside resident KV capacity")
        b = talker_block.trace(self.cfg, 0, {"m": self.desc, "position": position,
                                            "attention_block": self.attention_block})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise MemoryError("TTS block program exceeds instruction memory; split required")
        budget = {"position": position, "instructions": len(program),
                  "imem_words": len(program) * 8, "tmem_peak_words": b.tmem_peak,
                  "dram_bytes": self.nbytes}
        return [program], budget

    def token_bytes(self, text_id, codec_id):
        # RLD's optional BOUND is not enforced by hardware. Reject invalid host
        # tokens before DMA; do not claim a hardware memory-protection mechanism.
        for name, token, count in (("text", text_id, self.spec.text_vocab),
                                   ("codec", codec_id, self.spec.codec_vocab)):
            if (isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral)
                    or not 0 <= token < count):
                raise ValueError(f"{name} token ID outside [0, {count})")
        return np.asarray([text_id, codec_id], np.float32).view(np.uint8)


class ISASession:
    """Explicit diagnostic simulator, never a production or fallback backend.

    Only 8 token-ID bytes are written between steps. All embedding data, learned
    arithmetic, and KV updates belong to the ISA machine. No weights are retained
    by this harness after construction.
    """
    def __init__(self, image, weights):
        self.image, self.position = image, 0
        packed = image.build(weights)
        self.machine = Machine(image.cfg, [[I.halt()]], [packed])

    def step(self, text_id, codec_id):
        token_bytes = self.image.token_bytes(text_id, codec_id)
        programs, _ = self.image.compile(self.position)
        address = self.image.desc.ids.base.const
        self.machine.slices[0].dram[address:address + 8] = token_bytes
        self.machine.load(programs).run()
        self.position += 1
        address = self.image.desc.output.base.const
        return self.machine.slices[0].dram[address:address + 4 * self.image.spec.hidden].view(np.float32).copy()
