"""Resident Mimi RVQ-to-latent ISA stage; no decoder, PCM or physical backend.

Frozen centroids are normalized offline from the source checkpoint. Inference
gathers the first eight generated IDs, sums the two source branches separately,
projects with INT8 weights by default and adds the FP32 results entirely in
device programs. An optional FP32 coefficient-row path avoids projection
quantization while retaining resident source sums and outputs.
"""
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import CompileError, QTensor, Tensor, trace_fit
from .personaplex_audio_heads import _checked
from .personaplex_embeddings import _row


@dataclass(frozen=True)
class MimiRVQSpec:
    codebooks: int = 8
    vocab: int = 2048
    dim: int = 256
    latent: int = 512
    projection_precision: str = "int8"

    def __post_init__(self):
        for name in ("codebooks", "vocab", "dim", "latent"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below 2^24")
        if self.codebooks < 2:
            raise ValueError("Mimi split RVQ requires semantic and acoustic codebooks")
        if self.projection_precision not in ("int8", "fp32"):
            raise ValueError("projection_precision must be int8 or fp32")


def normalize_codebooks(embedding_sums, cluster_usages):
    """Offline frozen-table preparation, never invoked by an inference step."""
    sums = np.asarray(embedding_sums, np.float32)
    usage = np.asarray(cluster_usages, np.float32)
    if sums.ndim != 3 or usage.shape != sums.shape[:2]:
        raise ValueError("expected [codebooks,vocab,dim] sums and [codebooks,vocab] usages")
    if not np.isfinite(sums).all() or not np.isfinite(usage).all():
        raise ValueError("non-finite source RVQ statistics")
    with np.errstate(over="ignore", invalid="ignore"):
        value = sums / np.maximum(usage, np.float32(1e-5))[..., None]
    return _checked(value, sums.shape, "normalized RVQ centroids")


def _isa_exact_guard(value, expected, label):
    """Simulator guards only: F_BOUND is not enforced by current physical RTL."""
    b = ol.current()
    difference = ol.abs(value - expected)
    r = b.scratch()
    b.emit(I.rld(r, difference.base, raw=True, bound=(0, 0), comment=label))
    b.unscratch(r)


@ol.jit
def _rvq_latent(m):
    _isa_exact_guard(ol.load(m.counter), float(m.expected_steps), "completed depth pass before Mimi")
    codes = ol.load(m.codes)
    values = []
    for index, table in enumerate(m.tables):
        token = codes[index:index + 1]
        # Besides bounding integer lookup offsets, verify that conversion to a
        # row index did not truncate a fractional or non-finite resident value.
        expected_id = _row(m.identity, token)
        _isa_exact_guard(token, expected_id, "integral resident Mimi ID")
        values.append(_row(table, token))
    # Native ResidualVectorQuantization.decode initializes each branch at +0
    # and adds codebooks in order. There is no LM-style -1/zero prefix row.
    semantic = ol.zeros((1, m.dim)) + values[0]
    acoustic = ol.zeros((1, m.dim))
    for value in values[1:]:
        acoustic = acoustic + value
    ol.store(m.semantic, semantic)
    ol.store(m.acoustic, acoustic)
    if m.projection_precision == "fp32":
        # Stream one checkpoint coefficient row per output channel. The VPU
        # multiplies and reduces resident FP32 values without QACT or MXU MM.
        for channel in ol.range(m.latent):
            coefficient = ol.load(Tensor(m.projection_fp_data[0] + channel * (4 * m.dim),
                                          (m.dim,), (1,)))
            reduced = ol.sum(semantic * coefficient[None, :], axis=1)
            ol.store(m.semantic_projected[:, channel:channel + 1], reduced[:, None])
        for channel in ol.range(m.latent):
            coefficient = ol.load(Tensor(m.projection_fp_data[1] + channel * (4 * m.dim),
                                          (m.dim,), (1,)))
            reduced = ol.sum(acoustic * coefficient[None, :], axis=1)
            ol.store(m.acoustic_projected[:, channel:channel + 1], reduced[:, None])
        first = ol.load(m.semantic_projected)
        rest = ol.load(m.acoustic_projected)
    else:
        first = ol.dot(ol.quantize(semantic), m.projections[0])
        rest = ol.dot(ol.quantize(acoustic), m.projections[1])
        ol.store(m.semantic_projected, first)
        ol.store(m.acoustic_projected, rest)
    ol.store(m.output, first + rest)


class MimiRVQImage:
    """Sparse resident allocation inside the candidate P-region guard."""
    def __init__(self, spec, cfg, *, audio_ids_at=64, counter_at=0, expected_steps=16,
                 base=0x5D850000, limit=0x60000000):
        if cfg.S != 1 or spec.dim % cfg.D:
            raise ValueError("Mimi RVQ requires one slice and codebook width divisible by D")
        if type(expected_steps) is not int or not spec.codebooks <= expected_steps < 1 << 24:
            raise ValueError("Mimi needs a complete depth pass with enough resident IDs")
        for name, value in (("base", base), ("limit", limit), ("audio_ids_at", audio_ids_at),
                            ("counter_at", counter_at)):
            if type(value) is not int or value < 0 or value % 4:
                raise ValueError(f"{name} must be an aligned nonnegative integer")
        if not base < limit <= min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError("Mimi allocation is outside mapped unsigned32 capacity")
        if audio_ids_at + 4 * spec.codebooks > cfg.DRAM_BYTES or counter_at + 4 > cfg.DRAM_BYTES:
            raise MemoryError("Mimi input descriptors exceed mapped DRAM")
        self.spec, self.cfg = spec, cfg
        self.audio_ids_at, self.counter_at, self.expected_steps = audio_ids_at, counter_at, expected_steps
        self.spans, self.payload_hashes, self.programs = [], {}, None
        at = base

        def alloc(name, size, alignment=128, kind="state"):
            nonlocal at
            start = -(-at // alignment) * alignment
            if start + size > limit:
                raise MemoryError(f"Mimi {name} exceeds reserved persistent region")
            self.spans.append({"name": "mimi_" + name, "base": start, "bytes": size,
                               "alignment": alignment, "kind": kind, "region": "mimi"})
            at = start + size
            return start

        self.io = {name: alloc(name, 4 * width) for name, width in
                   (("semantic", spec.dim), ("acoustic", spec.dim),
                    ("semantic_projected", spec.latent), ("acoustic_projected", spec.latent),
                    ("output", spec.latent))}
        self.identity_at = alloc("identity", 4 * spec.vocab, kind="immutable")
        self.tables_at = alloc("tables", 4 * spec.codebooks * spec.vocab * spec.dim, kind="immutable")
        if spec.projection_precision == "fp32":
            self.code_bytes = self.scale_bytes = 0
            self.projection_offsets = [alloc(f"projection{index}_fp32",
                                             4 * spec.latent * spec.dim,
                                             kind="immutable") for index in range(2)]
        else:
            self.code_bytes = spec.latent * Q.row_bytes(spec.dim, "int8", cfg.D)
            self.scale_bytes = spec.latent * Q.scale_bytes(spec.dim, "int8", cfg.D)
            self.projection_offsets = [(alloc(f"projection{index}_codes", self.code_bytes,
                                              max(128, cfg.D), "immutable"),
                                        alloc(f"projection{index}_scales", self.scale_bytes,
                                              128, "immutable")) for index in range(2)]
        self.required_bytes = -(-at // 4096) * 4096
        self.weight_bytes = (4 * spec.codebooks * spec.vocab * spec.dim
                             + (8 * spec.latent * spec.dim if spec.projection_precision == "fp32"
                                else 2 * (self.code_bytes + self.scale_bytes)))
        if self.required_bytes > limit:
            raise MemoryError("aligned Mimi allocation exceeds reserved persistent region")
        external = [(audio_ids_at, audio_ids_at + 4 * spec.codebooks), (counter_at, counter_at + 4)]
        if any(p["base"] < end and start < p["base"] + p["bytes"]
               for p in self.spans for start, end in external):
            raise ValueError("Mimi payload/state overlaps resident input IDs or counter")

    def descriptors(self):
        s = self.spec
        return SimpleNamespace(
            counter=Tensor(self.counter_at, (1,), (1,)), expected_steps=self.expected_steps,
            codes=Tensor(self.audio_ids_at, (s.codebooks,), (1,)), dim=s.dim,
            latent=s.latent, projection_precision=s.projection_precision,
            identity=Tensor(self.identity_at, (s.vocab, 1), (1, 1)),
            tables=[Tensor(self.tables_at + 4 * i * s.vocab * s.dim,
                           (s.vocab, s.dim), (s.dim, 1)) for i in range(s.codebooks)],
            projections=([QTensor.weight(da, sa, s.latent, s.dim, "int8", self.cfg.D)
                          for da, sa in self.projection_offsets]
                         if s.projection_precision == "int8" else None),
            projection_fp_data=(self.projection_offsets
                                if s.projection_precision == "fp32" else None),
            **{name: Tensor(at, (1, s.dim if name in ("semantic", "acoustic") else s.latent),
                            (s.dim if name in ("semantic", "acoustic") else s.latent, 1))
               for name, at in self.io.items()})

    def compile(self):
        if self.programs is None:
            b = trace_fit(_rvq_latent, self.cfg, 0, lambda: {"m": self.descriptors()})
            p = b.finish()
            if p.run_args or len(p) * 8 > self.cfg.IMEM_WORDS:
                raise CompileError("Mimi RVQ has host arguments or exceeds instruction capacity")
            self.programs = [p]
        return self.programs

    def budget(self):
        return {"required_bytes": self.required_bytes,
                "occupied_bytes": sum(p["bytes"] for p in self.spans),
                "payload_bytes": sum(p["bytes"] for p in self.spans if p["kind"] == "immutable"),
                "weight_bytes": self.weight_bytes,
                "program_words": [len(p) * 8 for p in self.compile()],
                "spans": self.spans, "payload_sha256": self.payload_hashes,
                "range_guard_scope": "ISA-only F_BOUND; no physical range-protection claim"}

    def upload(self, backend, tables, projections):
        self.compile()  # layout/program preflight precedes all lazy source reads
        tables = tables() if callable(tables) else tables
        projections = projections() if callable(projections) else projections
        s = self.spec
        tables = _checked(tables, (s.codebooks, s.vocab, s.dim), "Mimi centroids")
        projections = _checked(projections, (2, s.latent, s.dim), "Mimi projections")
        def put(name, address, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            self.payload_hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, address, raw)
        put("identity", self.identity_at, np.arange(s.vocab, dtype=np.float32))
        put("tables", self.tables_at, tables)
        if s.projection_precision == "fp32":
            for index, address in enumerate(self.projection_offsets):
                put(f"projection{index}_fp32", address, projections[index])
        else:
            for index, (da, sa) in enumerate(self.projection_offsets):
                batch = QC.Batch()
                batch.submit(projections[index], "int8", self.cfg.D,
                             lambda q, scales, index=index, da=da, sa=sa:
                             (put(f"projection{index}_codes", da, q),
                              put(f"projection{index}_scales", sa, scales)))
                batch.wait()
