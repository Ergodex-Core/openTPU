"""Resident, layer-range temporal pages with persistent KV and residual state.

This is a software ISA candidate. A page's immutable matrix payload may be
replaced only after the preceding page program has halted. No activation, KV,
position, RoPE coefficient, or mask is computed by the host between pages.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import waits, target
from ..compiler import CompileError, KVDesc, Tensor, trace_fit
from ..kernels.attention import Bucket
from ..kernels.lib import rmsnorm
from . import personaplex_temporal as T
from . import qwen3 as G


DEFAULT_PAGES = ((0, 4, "int8"), (4, 12, "int4"), (12, 20, "int4"),
                 (20, 28, "int4"), (28, 31, "int4"), (31, 32, "int8"))


def _align(n, alignment=128):
    return -(-n // alignment) * alignment


def _exact_guard(value, expected, label):
    """ISA simulator guard; F_BOUND is not enforced by current physical RTL."""
    b = ol.current()
    difference = ol.abs(value - float(expected))
    r = b.scratch()
    b.emit(I.rld(r, difference.base, raw=True, bound=(0, 0), comment=label))
    b.unscratch(r)


class _ResidentFrame:
    """Load exactly one owned frame word using the pinned compiler contract.

    Runtime 18fb9780's generic RunWords reads an entire generation state. The
    paged image owns a single frame word, so use a bounded local initializer.
    """
    def __init__(self, m, pos):
        self.m, self.pos = m, pos

    def __enter__(self):
        b = ol.current()
        self.state = ol.load(self.m.frame)
        self.after = b.stack[-1][-1]
        b.run_words = {"tpos": self.state.base}
        self.pos.words = dict(b.run_words)
        return self

    def __exit__(self, *exc):
        b = ol.current()
        if exc[0] is None:
            body = b.stack[-1]
            index = next(j for j, instruction in enumerate(body)
                         if instruction is self.after) + 1
            initializers = []
            for k, (variable, coefficient) in enumerate(b.run_args):
                if variable is not self.pos.tpos:
                    raise CompileError("unexpected resident temporal argument")
                scale = int(coefficient)
                endpoint = scale * (self.m.frame_limit - 1)
                # Negative affine coefficients wrap in the unsigned address
                # register. Their set contains zero and high wrapped values,
                # so a single narrow unsigned interval cannot describe it.
                initializers.append(I.rld(15 - k, self.state.base, mul=scale,
                    bound=(0, endpoint) if scale >= 0 else None,
                    comment=f"resident {scale}*tpos"))
            body[index:index] = initializers
        b.run_words = None
        return False


@ol.jit
def _page_step(m, pos):
    with _ResidentFrame(m, pos):
        b = ol.current()
        # The frame is read from DRAM by RunWords. All address multipliers are
        # initialized from that resident word, never from run() arguments.
        r = b.scratch()
        b.emit(I.rld(r, pos.words["tpos"], bound=(0, m.frame_limit - 1),
                     comment="temporal frame within admitted context"))
        b.unscratch(r)
        page = ol.load(m.page)
        _exact_guard(page, m.page_index, "temporal page source order")
        x = ol.load(m.x)
        c = ol.load(m.cos_t[pos.pos, :])
        s = ol.load(m.sin_t[pos.pos, :])
        for lw in G._layers(m):
            lw.mlp_loop = True
            x.set(T._attention(x, lw, c, s, pos, m.spec, m.block))
            x.set(G._mlp(x, lw, m.spec))
        if m.last_page:
            ol.store(m.x, rmsnorm(x, ol.load(m.g_final), m.spec.eps))
            ol.store(m.frame, ol.load(m.frame) + 1.0)
            ol.store(m.page, ol.zeros((1,)))
        else:
            ol.store(m.x, x)
            ol.store(m.page, page + 1.0)


@dataclass(frozen=True)
class _Page:
    start: int
    end: int
    format: str
    weight_bytes: int
    layout: T.Image
    key: tuple
    kv_offsets: tuple


class TemporalPagedImage:
    """Six-page production layout, or a contiguous explicit tiny layer plan.

    ``x_at`` is the resident residual [hidden]. ``kv_at`` owns every layer's
    cache and must be disjoint from all prefix spans. The other explicit state
    addresses are disjoint immutable/control spans. The page span is one
    physical overlay, sized for the largest packet. ``pack_layer`` is an
    offline builder; ``upload_page`` accepts already packed immutable bytes.
    """

    def __init__(self, spec: T.Spec, cfg, *, cap=128, pages=None,
                 kv_at=0x10000, x_at, frame_at, page_at, cos_at, sin_at,
                 zmask_at, final_norm_at, weight_page_base,
                 weight_page_limit=0xB0000000, program_base=0x04000000,
                 program_limit=0x05000000, prefix_spans=()):
        if cfg.S != 1 or cfg.D > 256 or 256 % cfg.D:
            raise ValueError("paged temporal requires one slice and a dividing attention D")
        if type(cap) is not int or not 0 < cap <= min(spec.context, 128):
            raise ValueError("paged temporal context must be within 1..128 frames")
        plan = tuple(DEFAULT_PAGES if pages is None else pages)
        if not plan or plan[0][0] != 0 or plan[-1][1] != spec.layers:
            raise ValueError("temporal pages must cover every layer exactly once")
        for i, (start, end, fmt) in enumerate(plan):
            if (type(start) is not int or type(end) is not int or
                    start < 0 or not start < end <= spec.layers or
                    fmt not in ("int4", "int8") or
                    (i and start != plan[i - 1][1])):
                raise ValueError("temporal pages must be contiguous INT4/INT8 layer ranges")
        self.spec, self.cfg, self.frame_limit, self.plan = spec, cfg, cap, plan
        self.cap = _align(cap, cfg.D)
        self.block = cfg.D  # cap<=128: exactly one attention bucket.
        self.io = dict(x=x_at, frame=frame_at, page=page_at, cos=cos_at,
                       sin=sin_at, zmask=zmask_at, final_norm=final_norm_at)
        self.kv_at, self.weight_page_base = kv_at, weight_page_base
        self.weight_page_limit = weight_page_limit
        self.prefix_spans = list(prefix_spans)
        self.pages = []
        for start, end, fmt in plan:
            # Existing one-layer image supplies the exact quantized matrix
            # prefix and offset map. Its KV tail is never uploaded into a page.
            local_spec = replace(spec, layers=1)
            layout = T.Image(local_spec, cfg, cap=cap, wformat=fmt)
            key = layout.lf[0]
            offsets = layout.bofs[key]["kvs"][0]
            kv_first = offsets[0]["k"]
            kv_last = offsets[-1]["vs"] + 4 * layout.cap
            kv_bytes = kv_last - kv_first
            if self.pages and kv_bytes != self.kv_layer_bytes:
                raise ValueError("page formats unexpectedly change KV layout")
            self.kv_layer_bytes = kv_bytes
            self.pages.append(_Page(start, end, fmt, (end - start) * kv_first,
                                    layout, key, tuple(offsets)))
        self.kv_bytes = spec.layers * self.kv_layer_bytes
        self.page_bytes = max(page.weight_bytes for page in self.pages)
        production = (spec.hidden, spec.layers, spec.n_heads, spec.ffn, cap) == \
            (4096, 32, 32, 11264, 128)
        if production:
            if kv_at != 0x10000 or self.kv_bytes != 34603008:
                raise MemoryError("production temporal KV must occupy fixed P reservation")
            if target.arch_hash(cfg) == 0xdd6b92a6 and plan == DEFAULT_PAGES and [p.weight_bytes for p in self.pages] != \
                    [831782912, 884473856, 884473856, 884473856, 331677696, 207945728]:
                raise MemoryError("temporal packet bytes changed from pinned page plan")
            if (program_base, program_limit) != (0x04000000, 0x05000000):
                raise ValueError("production temporal programs must use guarded IMEM region")
        if (type(weight_page_base) is not int or type(weight_page_limit) is not int or
                weight_page_base % 128 or weight_page_limit % 128 or
                not 0 <= weight_page_base < weight_page_limit <= min(cfg.DRAM_BYTES, 1 << 32) or
                weight_page_base + self.page_bytes > weight_page_limit):
            raise MemoryError("largest temporal weight page exceeds its reserved overlay")
        rh = spec.head_dim // 2
        self.spans = [
            dict(name="temporal_kv", base=kv_at, bytes=self.kv_bytes, kind="state", region="temporal"),
            dict(name="temporal_x", base=x_at, bytes=4 * spec.hidden, kind="state", region="temporal"),
            dict(name="temporal_frame", base=frame_at, bytes=4, kind="state", region="temporal"),
            dict(name="temporal_page", base=page_at, bytes=4, kind="state", region="temporal"),
            dict(name="temporal_cos", base=cos_at, bytes=4 * self.cap * rh,
                 kind="immutable", region="temporal"),
            dict(name="temporal_sin", base=sin_at, bytes=4 * self.cap * rh,
                 kind="immutable", region="temporal"),
            dict(name="temporal_zmask", base=zmask_at, bytes=4 * (self.cap + self.block),
                 kind="immutable", region="temporal"),
            dict(name="temporal_final_norm", base=final_norm_at, bytes=4 * spec.hidden,
                 kind="immutable", region="temporal"),
            dict(name="temporal_weight_page", base=weight_page_base, bytes=self.page_bytes,
                 kind="overlay", region="temporal_weights")]
        for span in self.spans:
            span["alignment"] = 128
        slot_bytes = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        if (type(program_base) is not int or type(program_limit) is not int or
                program_base % 128 or program_limit % 128 or
                not 0 <= program_base < program_limit <= min(cfg.DRAM_BYTES, 1 << 32)):
            raise MemoryError("invalid temporal program reservation")
        program_end = max([program_base] + [s["base"] + s["bytes"]
                                          for s in self.prefix_spans
                                          if s["region"] == "program"])
        program_end = _align(program_end)
        self.program_slots = []
        for i in range(len(self.pages)):
            if program_end + slot_bytes > program_limit:
                raise MemoryError("temporal page programs exceed guarded IMEM region")
            self.program_slots.append(program_end)
            self.spans.append(dict(name=f"temporal_program{i}", base=program_end,
                                   bytes=slot_bytes, kind="program", region="program",
                                   alignment=128))
            program_end += slot_bytes
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        for span in self.spans:
            if (type(span["base"]) is not int or span["base"] % 128 or
                    span["bytes"] <= 0 or span["base"] + span["bytes"] > capacity):
                raise MemoryError("invalid temporal span: " + span["name"])
        ordered = sorted(self.prefix_spans + self.spans, key=lambda s: s["base"])
        for left, right in zip(ordered, ordered[1:]):
            if left["base"] + left["bytes"] > right["base"]:
                raise MemoryError("temporal resident spans overlap: " + left["name"] +
                                  "/" + right["name"])
        self.programs = {}

    def _descriptors(self, page_index):
        page = self.pages[page_index]
        layout = page.layout
        # Rebase the existing matrix descriptors onto packed weight-only
        # blocks. The layer's original KV offsets are replaced below.
        stride = page.weight_bytes // (page.end - page.start)
        layout.loc = {i: (self.weight_page_base, stride, i, 0)
                      for i in range(page.end - page.start)}
        layout.runs = [(0, (page.key,), page.end - page.start)]
        m = layout.descriptors(0)
        original_layer = m.layer
        kv0 = page.kv_offsets[0]["k"]

        def layer(local_index, iteration=None):
            lw = original_layer(local_index, iteration)
            at = self.kv_at + (page.start + local_index) * self.kv_layer_bytes
            if iteration is not None:
                at = at + iteration * self.kv_layer_bytes
            heads = {h: {name: at + offset - kv0 for name, offset in fields.items()}
                     for h, fields in enumerate(page.kv_offsets)}
            lw.kv = KVDesc(heads, self.cap, self.spec.head_dim, self.cfg.D, 1, 0)
            lw.kvs = [lw.kv]
            return lw

        m.layer = layer
        m.runs = layout.runs
        m.spec = spec = self.spec._core()
        m.x = Tensor(self.io["x"], (1, spec.hidden), (spec.hidden, 1))
        m.cos_t = Tensor(self.io["cos"], (self.cap, spec.head_dim // 2),
                         (spec.head_dim // 2, 1))
        m.sin_t = Tensor(self.io["sin"], (self.cap, spec.head_dim // 2),
                         (spec.head_dim // 2, 1))
        m.g_final = Tensor(self.io["final_norm"], (spec.hidden,), (1,))
        m.frame = Tensor(self.io["frame"], (1,), (1,))
        m.page = Tensor(self.io["page"], (1,), (1,))
        m.frame_limit, m.page_index = self.frame_limit, page_index
        m.last_page, m.block = page_index == len(self.pages) - 1, self.block
        return m

    def compile_page(self, page_index):
        if type(page_index) is not int or not 0 <= page_index < len(self.pages):
            raise ValueError("invalid temporal page index")
        if page_index not in self.programs:
            pos = G.RunRows(1, self.block, 0, self.io["zmask"], self.cap, 1)
            pos.tpos.bound = self.frame_limit
            pos.words_src = (Tensor(self.io["frame"], (1,), (1,)), {"tpos": 0})
            m = self._descriptors(page_index)
            program = trace_fit(_page_step, self.cfg, 0,
                                lambda: {"m": m, "pos": pos}).finish()
            if any(var is not pos.tpos for var, _ in program.run_args):
                raise CompileError("temporal page unexpectedly needs host arguments")
            # RunWords inserted RLD initializers for every coefficient above.
            # Clear metadata so backend.run(program) cannot request host args.
            program.run_args.clear()
            if len(program) * 8 > self.cfg.IMEM_WORDS:
                raise CompileError("temporal page exceeds IMEM")
            self.programs[page_index] = program
        return self.programs[page_index]

    def compile(self):
        return [self.compile_page(i) for i in range(len(self.pages))]

    def pack_layer(self, page_index, layer_weights):
        """Offline quantization of one layer; returns its immutable page block."""
        page = self.pages[page_index]
        layout = page.layout
        # The descriptor rebase affects compile only. Restore the one-layer
        # build mapping before using the established PersonaPlex packer.
        layout.loc = {0: (layout.layer0, layout.bsize[page.key], 0, 0)}
        layout.runs = [(0, (page.key,), 1)]
        full = layout.build([layer_weights], np.ones(self.spec.hidden, np.float32))[0]
        stride = page.weight_bytes // (page.end - page.start)
        return full[layout.layer0:layout.layer0 + stride].copy()

    def upload_page(self, backend, page_index, layer_payloads):
        """Refill immutable matrices after the preceding page has halted."""
        page = self.pages[page_index]
        payloads = list(layer_payloads)
        if len(payloads) != page.end - page.start:
            raise ValueError("temporal page has the wrong number of layer blocks")
        stride = page.weight_bytes // len(payloads)
        blocks = []
        for payload in payloads:
            data = np.asarray(payload)
            if data.dtype != np.uint8 or data.ndim != 1 or data.size != stride:
                raise ValueError("temporal page layer block has wrong byte layout")
            blocks.append(np.ascontiguousarray(data))
        for index, block in enumerate(blocks):
            backend.write(0, self.weight_page_base + index * stride, block)

    def run_frame(self, backend, page_payloads):
        """Synchronous refill/run sequence; each run drains before the next refill."""
        packets = list(page_payloads)
        if len(packets) != len(self.pages):
            raise ValueError("temporal frame requires one immutable packet per page")
        stats = []
        for index, payloads in enumerate(packets):
            self.upload_page(backend, index, payloads)
            stats.append(backend.run([self.compile_page(index)]))
        return stats

    def upload_static(self, backend, final_norm):
        """Write immutable RoPE/mask/norm tables once, before any page runs."""
        norm = T._checked(final_norm, (self.spec.hidden,), "temporal final norm")
        core = self.spec._core()
        cos, sin = zip(*(G.rope_tables(core, p) for p in range(self.cap)))
        backend.write(0, self.io["cos"], np.stack(cos).astype(np.float32))
        backend.write(0, self.io["sin"], np.stack(sin).astype(np.float32))
        mask = np.concatenate((np.full(self.cap, np.inf, np.float32),
                               np.full(self.block, -np.inf, np.float32)))
        backend.write(0, self.io["zmask"], mask)
        backend.write(0, self.io["final_norm"], norm)
        self.reset(backend)

    def reset(self, backend):
        """Logical reset masks retained KV by restarting the frame position."""
        backend.write(0, self.io["frame"], np.zeros(1, np.float32))
        backend.write(0, self.io["page"], np.zeros(1, np.float32))

    def budget(self):
        return dict(kv_bytes=self.kv_bytes, kv_at=self.kv_at,
                    page_bytes=[p.weight_bytes for p in self.pages],
                    overlay_bytes=self.page_bytes, weight_page_base=self.weight_page_base,
                    weight_page_limit=self.weight_page_limit,
                    overlay_slack_bytes=self.weight_page_limit -
                        (self.weight_page_base + self.page_bytes),
                    program_slots=self.program_slots,
                    program_words=[len(p) * 8 for p in self.compile()],
                    spans=self.spans, prefix_spans=self.prefix_spans,
                    combined_occupied_bytes=sum(s["bytes"] for s in
                                                self.prefix_spans + self.spans),
                    frame_limit=self.frame_limit, physical_cap=self.cap)
