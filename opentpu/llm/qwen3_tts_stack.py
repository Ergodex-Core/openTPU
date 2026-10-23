"""Resident multi-layer talker with codec logits; explicit diagnostic ISA session.

Host orchestration launches sequential programs. Hidden vectors, weight tables,
KV, final normalization, codec head and greedy selection stay in device memory.
This does not implement the CustomVoice prompt, predictor, sampler or decoder.
"""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import quant as Q
from ..compiler import KVDesc, QTensor, k_scale_bytes
from ..isasim import Machine
from ..kernels.tts_stack import talker_layer, codec_logits
from .qwen3_tts_talker import TalkerBlockImage, _tensor


class TalkerStackImage(TalkerBlockImage):
    def __init__(self, spec, cfg, capacity=256, attention_block=None, *, layers=28,
                 audio_vocab=2048, eos=2150):
        if type(layers) is not int or not 1 <= layers <= 28:
            raise ValueError("talker layers must be 1..28")
        if not (type(audio_vocab) is int and type(eos) is int
                and 0 < audio_vocab <= eos < spec.codec_vocab):
            raise ValueError("invalid fixed codec suppression range")
        super().__init__(spec, cfg, capacity, attention_block)
        self.layer_count, self.audio_vocab, self.eos = layers, audio_vocab, eos
        self.layers = [self.desc]

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

        # Relocate only the layer's own tensors/KV. Embeddings, RoPE, hidden and
        # diagnostic scratch descriptors are shared; no repeated text projection.
        originals = list(self.parameters.items())
        attrs = {id(getattr(self.desc, n)): n for n in
                 ("input_norm", "post_norm", "q_norm", "k_norm", "wq", "wk", "wv", "wo", "gate", "up", "down")}
        for index in range(1, layers):
            layer = SimpleNamespace(**vars(self.desc))
            for key, (fmt, desc, shape) in originals:
                if ".layers.0." not in key:
                    continue
                attr = attrs[id(desc)]
                checkpoint = key.replace(".layers.0.", f".layers.{index}.")
                name = f"layer.{index}.{attr}"
                setattr(layer, attr, tensor(name, shape, checkpoint) if fmt == "fp32"
                        else weight(name, shape, checkpoint))
            heads = {}
            for head in range(spec.n_kv):
                heads[head] = {key: alloc(f"layer.{index}.kv.{head}.{key}", size) for key, size in (
                    ("k", capacity * spec.head_dim),
                    ("ks", capacity * k_scale_bytes(spec.head_dim, cfg.D)),
                    ("vt", capacity * spec.head_dim), ("vs", capacity * 4))}
            layer.kv = KVDesc(heads, capacity, spec.head_dim, cfg.D, 1, 0)
            self.layers.append(layer)
        m = self.desc
        m.final_norm = tensor("final_norm", (spec.hidden,), "talker.model.norm.weight")
        m.codec_head = weight("codec_head", (spec.codec_vocab, spec.hidden), "talker.codec_head.weight")
        m.normalized = tensor("normalized", (1, spec.hidden))
        m.logits = tensor("logits", (1, spec.codec_vocab))
        m.greedy_raw = tensor("greedy_raw", (2,))
        m.greedy_masked = tensor("greedy_masked", (2,))
        if self.nbytes > min(cfg.DRAM_BYTES, 2**32):
            raise MemoryError(f"resident talker stack needs {self.nbytes} device bytes; no offload")

    def compile(self, position):
        raise NotImplementedError("use sequential compile_parts; parts are not parallel slices")

    def compile_parts(self, position):
        programs, budget = TalkerBlockImage.compile(self, position)
        parts = [("layer.0", programs, budget)]
        for index, layer in enumerate(self.layers[1:], 1):
            b = talker_layer.trace(self.cfg, 0, {"m": layer, "position": position,
                                               "attention_block": self.attention_block})
            parts.append((f"layer.{index}", [b.finish()], {"tmem_peak_words": b.tmem_peak}))
        b = codec_logits.trace(self.cfg, 0, {"m": self.desc, "audio_vocab": self.audio_vocab, "eos": self.eos})
        parts.append(("codec_head", [b.finish()], {"tmem_peak_words": b.tmem_peak}))
        for _, programs, budget in parts:
            budget.update(position=position, instructions=len(programs[0]),
                          imem_words=8 * len(programs[0]), dram_bytes=self.nbytes)
            if budget["imem_words"] > self.cfg.IMEM_WORDS:
                raise MemoryError("talker part exceeds instruction memory")
        return parts

    def cache_regions(self):
        return {n: v for n, v in self.regions.items() if n.startswith("kv.") or ".kv." in n}


class StackISASession:
    """CPU diagnostic simulator; no device opener or production fallback."""
    def __init__(self, image, packed):
        self.image, self.position = image, 0
        self.machine = Machine(replace(image.cfg, DRAM_BYTES=image.nbytes), [[I.halt()]], [packed])

    def read(self, name):
        t = getattr(self.image.desc, name)
        n = 4 * math.prod(t.shape)
        return self.machine.slices[0].dram[t.base.const:t.base.const + n].view(np.float32).copy().reshape(-1)

    def step(self, text_id, codec_id, observe=None):
        encoded = self.image.token_bytes(text_id, codec_id)
        parts = self.image.compile_parts(self.position)
        address = self.image.desc.ids.base.const
        self.machine.slices[0].dram[address:address + 8] = encoded
        for name, programs, _ in parts:
            self.machine.load(programs).run()
            if observe is not None:
                observe(name, self)
        self.position += 1
        return self.read("logits"), int(self.read("greedy_raw")[1]), int(self.read("greedy_masked")[1])

    def reset(self):
        for address, size in self.image.cache_regions().values():
            self.machine.slices[0].dram[address:address + size] = 0
        self.position = 0
