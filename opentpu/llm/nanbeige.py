"""Nanbeige looped decoder on the existing openTPU integer kernels.

Physical layers share weights across loops, but each logical layer has independent
KV.  The image uses openTPU W8A8 arithmetic, including an INT8 output head; it is
not bit-identical to the published weight-only INT8/BF16 target.  Checkpoint
quality must therefore be measured separately from ISA/RTL correctness.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import language as ol
from ..compiler import Affine, DevVar, KVDesc, Tensor, CompileError, arg_words, current, trace_fit
from ..kernels.lib import rmsnorm
from . import qwen3 as Q
from . import generate as G


@dataclass(frozen=True)
class Spec(Q.Spec):
    """``layers`` is the physical count; ``logical_layers`` includes the loops."""
    loops: int = 2
    skip_loop_final_norm: bool = False
    qk_norm: bool = False
    qk_smooth: bool = False
    tied: bool = False
    compact_head: bool = True  # No standard head duplicate when row-lane MMs own the head.
    capture_layers: tuple = ()

    @property
    def logical_layers(self):
        return self.layers * self.loops

    @staticmethod
    def from_hf(model_dir):
        path = Path(model_dir)
        c = json.loads((path / "config.json").read_text())
        if c.get("model_type") != "nanbeige":
            raise ValueError("expected model_type nanbeige")
        for key in ("attention_bias", "mlp_bias", "use_sliding_window", "qk_norm"):
            if c.get(key):
                raise ValueError(f"Nanbeige {key} is not supported")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("Nanbeige requires silu")
        from . import rope_parameters
        rp = rope_parameters(c, path)
        eos = c.get("eos_token_id")
        if (path / "generation_config.json").exists():
            eos = json.loads((path / "generation_config.json").read_text()).get(
                "eos_token_id", eos)
        return Spec(hidden=c["hidden_size"], layers=c["num_hidden_layers"],
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=c["intermediate_size"], vocab=c["vocab_size"],
                    eps=c.get("rms_norm_eps", 1e-5),
                    theta=rp.get("rope_theta", c.get("rope_theta", 10000.0)),
                    tied=c.get("tie_word_embeddings", False), bos=c.get("bos_token_id"),
                    eos=tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,)),
                    loops=c.get("num_loops", 1),
                    skip_loop_final_norm=c.get("skip_loop_final_norm", False),
                    ctx=c.get("max_position_embeddings", 0))

    def check(self, cfg):
        if self.loops < 1 or self.layers < 1:
            raise ValueError("Nanbeige needs positive physical layers and loops")
        if self.qk_norm or self.nope or self.rotary:
            raise ValueError("Nanbeige supports full RoPE without QK norm")
        if len(set(self.capture_layers)) != len(self.capture_layers) or any(
                not isinstance(i, int) or not 0 <= i < self.logical_layers
                for i in self.capture_layers):
            raise ValueError("capture_layers must contain unique logical layer indices")
        # Qwen3's conservative front-end restriction predates its attention
        # kernels' explicit MCOLS-wide query splitting. Only lift that check.
        Q.Spec.check(self, replace(cfg, MCOLS=max(cfg.MCOLS, self.n_q // self.n_kv)))

    def image(self, cfg, cap, batch=1, rows=1, wformat="int8", head_format=None,
              lookup=False, embed_host=None, formats=None, spec_rows=0, base=0):
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup,
                     embed_host, formats, spec_rows, base)


class Image(Q.Image):
    """One physical weight copy, one cache per logical layer."""

    def __init__(self, spec, cfg, cap, batch=1, rows=1, wformat="int8", head_format=None,
                 lookup=False, embed_host=None, formats=None, spec_rows=0, base=0):
        if spec_rows:
            raise ValueError("use Nanbeige compile_rows for verification; the generic "
                             "SpecDecoder does not implement Nanbeige/DSpark")
        if base:
            raise ValueError("the pinned wide-port image API requires base=0")
        super().__init__(spec, cfg, cap, batch=batch, rows=rows, wformat=wformat,
                         head_format=head_format, lookup=lookup,
                         embed_host=embed_host, formats=formats)
        self.physical_nbytes = self.nbytes
        self.extra_kv = {}
        b = Q._Bump(self.nbytes)
        # Additional loops use a fixed cache stride, independent of the physical
        # weight-layout stride. Descriptors can walk both with one loop counter.
        lb = Q._Bump()
        local = [[{"k": lb.alloc(cap * spec.head_dim),
                   "ks": lb.alloc(cap * Q.k_scale_bytes(spec.head_dim, cfg.D)),
                   "vt": lb.alloc(spec.head_dim * cap), "vs": lb.alloc(4 * cap)}
                  for _ in range(self.nkv_loc)] for _ in range(batch)]
        self.extra_kv_stride = (lb.next + 4095) // 4096 * 4096
        self.extra_kv_local = local
        for loop in range(1, spec.loops):
            self.extra_kv[loop] = b.alloc(spec.layers * self.extra_kv_stride)
        # A tiny enable/offset table preserves the physical-layer hardware loop
        # while writing only requested DSpark taps. No capture cost at baseline.
        self.capture_slots = {li: slot for slot, li in enumerate(spec.capture_layers)}
        self.capture = b.alloc(len(self.capture_slots) * rows * spec.hidden * 4) \
            if self.capture_slots else None
        self.capture_control = b.alloc(spec.logical_layers * 8) if self.capture_slots else None
        # Runtime positions with original host-fed embeddings need only masks;
        # no vocabulary or RoPE lookup tables are stored on the device.
        self.host_mask_block = min(Q.ATTN_BLOCK, cap)
        self.host_zmask = b.alloc(4 * (cap + self.host_mask_block)) if not self.lookup else None
        # Contiguous additive tiles let multi-row attention load all query masks
        # at once. Compact row broadcasts cost enough instructions to overflow
        # burst200 IMEM for the production four-row verifier.
        self.host_amask = b.alloc(Q.amask_bytes(self.host_mask_block, cfg.MCOLS)) \
            if not self.lookup and rows > 1 else None
        self.kv_bytes *= spec.loops
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"Nanbeige shared-weight image needs {self.nbytes} bytes; "
                              f"device exposes {cfg.DRAM_BYTES}")

    def build(self, W):
        images = super().build(W)
        if self.capture_slots:
            records = np.zeros((self.spec.logical_layers, 2), np.float32)
            for li, slot in self.capture_slots.items():
                records[li] = (1, slot * self.rows * self.spec.hidden * 4)
            data = records.view(np.uint8).reshape(-1)
            for image in images:
                image[self.capture_control:self.capture_control + data.size] = data
        if self.host_zmask is not None:
            data = np.concatenate((np.full(self.cap, np.inf, np.float32),
                                   np.full(self.host_mask_block, -np.inf, np.float32)))
            data = data.view(np.uint8)
            for image in images:
                image[self.host_zmask:self.host_zmask + data.size] = data
        if self.host_amask is not None:
            data = Q.amask_table(self.host_mask_block, self.cfg.MCOLS).view(np.uint8).reshape(-1)
            for image in images:
                image[self.host_amask:self.host_amask + data.size] = data
        return images

    def descriptors(self, sid):
        m = super().descriptors(sid)
        original = m.layer

        def loop_layer(li, it=None, loop=0):
            lw = original(li, it)
            _, _, old_it, _ = self.loc[li]
            index = Affine.of(li) if it is None else Affine.of(li - old_it) + it
            logical = loop * self.spec.layers + index
            lw.capture = self.capture
            lw.capture_control = Tensor(self.capture_control + logical * 8, (2,), (1,)) \
                if self.capture_slots else None
            if loop:
                # loc's repeat index is relative to this run's first physical
                # layer; convert it back to an absolute physical layer index.
                off = self.extra_kv[loop] + index * self.extra_kv_stride
                lw.kvs = [KVDesc({sid + j * self.cfg.S:
                                  {k: off + v for k, v in h.items()}
                                  for j, h in enumerate(heads)}, self.cap,
                                 self.spec.head_dim, self.cfg.D, self.cfg.S, sid)
                          for heads in self.extra_kv_local]
                lw.kv = lw.kvs[0]
            return lw

        m.loop_layer = loop_layer
        return m

    def compile_step(self, pos, block=Q.ATTN_BLOCK, tok=None):
        return [trace_fit(nanbeige_step, self.cfg, s,
                          lambda s=s: {"m": Q.step_descriptors(self, s), "pos": pos,
                                       "block": block, **Q._tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_decode(self, blocks, lo, block=Q.ATTN_BLOCK):
        return Q.compile_decode(self, nanbeige_step, blocks, lo, block)

    def _host_position(self, blocks, block, rows=1):
        if self.lookup or self.host_zmask is None:
            raise ValueError("runtime host inputs require an image without lookup tables")
        if not 0 < block <= self.host_mask_block:
            raise ValueError("attention block exceeds the host mask table")
        if rows > 1 and block != self.host_mask_block:
            raise ValueError("runtime host rows require the allocated mask tile width")
        lo = (blocks - 1) * block
        if blocks < 1 or lo >= self.cap:
            raise ValueError("runtime host bucket is outside the KV cache")
        if rows == 1:
            return Q.RunPos(blocks, block, lo, self.host_zmask, self.cap)
        return Q.RunRows(blocks, block, lo, self.host_zmask, self.cap, rows,
                          amask=(self.host_amask, 4 * self.cfg.MCOLS * (block + 1)))

    def compile_host_step(self, blocks, block=Q.ATTN_BLOCK):
        """One host-input program reused throughout an attention bucket."""
        block = min(block, self.host_mask_block)
        pos = self._host_position(blocks, block)
        builders = [trace_fit(nanbeige_step, self.cfg, sid,
                    lambda sid=sid: {"m": Q.step_descriptors(self, sid), "pos": pos,
                                     "block": block, "host_inputs": True})
                    for sid in range(self.cfg.S)]
        return _runtime_programs(builders, self.cfg)

    def compile_host_rows(self, blocks, rows, logit_rows, block=Q.ATTN_BLOCK,
                          tiled=False):
        """Consecutive host-input rows, cached by bucket, row count and output rows."""
        if not 1 <= rows <= self.rows:
            raise ValueError("runtime host row count exceeds the image")
        block = min(block, self.host_mask_block)
        pos = self._host_position(blocks, block, rows)
        if rows == 1:
            # Keep the row kernel's output placement and optional skipped head.
            pos = Q.RunRows(blocks, block, (blocks - 1) * block, self.host_zmask,
                            self.cap, rows)
        if not tiled:
            try:
                builders = [trace_fit(nanbeige_rows, self.cfg, sid,
                            lambda sid=sid: {"m": self.descriptors(sid), "rows": pos,
                                             "logit_rows": list(logit_rows), "block": block,
                                             "host_inputs": True})
                            for sid in range(self.cfg.S)]
                return _runtime_programs(builders, self.cfg)
            except CompileError:
                pass
        from . import tiled as TL
        TL.check(self, rows)

        def trace(kw, fit):
            return [nanbeige_tiled_rows.trace(self.cfg, 0,
                    {"m": self.descriptors(0), "rows": pos,
                     "logit_rows": list(logit_rows), "block": block, **kw}, fit=fit)]

        programs, args = TL._first_fit(trace, TL.variants("qwen3", rows, self.cfg.MCOLS),
                                        self.cfg, levels=False)
        if any(len(p) * 8 > self.cfg.IMEM_WORDS for p in programs):
            raise CompileError("Nanbeige runtime host verification exceeds IMEM")
        return programs, args

    def compile_generate(self, blocks, lo, block=Q.ATTN_BLOCK, chain=True,
                         samp=None, debug=False, part=None):
        return G.compile_generate(self, nanbeige_step, blocks, lo, block,
                                  chain, samp, debug, part)

    def compile_rows(self, rows, logit_rows, block=Q.ATTN_BLOCK, tokens=None):
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows exceeds image capacity {self.rows}")
        try:
            progs = [trace_fit(nanbeige_rows, self.cfg, s,
                              lambda s=s: {"m": self.descriptors(s), "rows": list(rows),
                                           "logit_rows": list(logit_rows), "block": block,
                                           **Q._tokens_arg(self, tokens, rows)}).finish()
                    for s in range(self.cfg.S)]
            if any(len(p) * 8 > self.cfg.IMEM_WORDS for p in progs):
                raise CompileError("Nanbeige untiled verification exceeds IMEM")
            return progs
        except CompileError:
            return self.compile_tiled_rows(rows, logit_rows, block, tokens)

    def compile_tiled_rows(self, rows, logit_rows, block=Q.ATTN_BLOCK, tokens=None):
        from . import tiled as TL
        if self.cfg.S != 1 or not rows or list(rows) != [(0, rows[0][1] + r)
                                                        for r in range(len(rows))]:
            raise CompileError("Nanbeige tiled verify needs consecutive sequence-0 rows")
        TL.check(self, len(rows))
        def trace(kw, fit):
            return [nanbeige_tiled_rows.trace(self.cfg, 0,
                      {"m": self.descriptors(0), "rows": list(rows),
                       "logit_rows": list(logit_rows), "block": block,
                       **Q._tokens_arg(self, tokens, rows), **kw}, fit=fit)]
        progs, _ = TL._first_fit(trace, TL.variants("qwen3", len(rows), self.cfg.MCOLS),
                                self.cfg, levels=False)
        if any(len(p) * 8 > self.cfg.IMEM_WORDS for p in progs):
            raise CompileError("Nanbeige tiled verification exceeds IMEM")
        return progs

    def compile_prompt_run(self, blocks, R, kind, block=Q.ATTN_BLOCK):
        if not self.lookup or kind not in ("P", "L"):
            raise ValueError("a prompt run needs lookup tables and kind P or L")
        pos = Q.RunRows(blocks, block, (blocks - 1) * block, self.lookup["zmask"],
                        self.cap, R, 0, Q._amask(self.lookup))
        bs = [trace_fit(nanbeige_prompt_run, self.cfg, s,
                        lambda s=s: {"m": self.descriptors(s), "pos": pos,
                                     "R": R, "kind": kind, "block": block})
              for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)


def _runtime_programs(builders, cfg):
    programs = [b.finish() for b in builders]
    if any(len(p) * 8 > cfg.IMEM_WORDS for p in programs):
        raise CompileError("Nanbeige runtime host program exceeds IMEM")
    if any(b.run_args != builders[0].run_args for b in builders):
        raise CompileError("Nanbeige slices take different runtime arguments")
    return programs, list(builders[0].run_args)


def _layers(m, loop):
    for first, unit, reps in m.runs:
        if reps == 1:
            for e in range(len(unit)):
                yield m.loop_layer(first + e, loop=loop)
        else:
            for it in ol.range(reps):
                for e in range(len(unit)):
                    yield m.loop_layer(first + e, it, loop)


def _capture(lw, x):
    """Store a selected layer input without unrolling the physical-layer loop."""
    if lw.capture_control is None:
        return
    b = current()
    control = ol.load(lw.capture_control)
    gate, offset = b.scratch(), b.scratch()
    b.rld(gate, control[0:1], comment="capture selected layer")
    b.rld(offset, control[1:2], comment="capture slot byte offset")
    loop = b.begin_loop(0, rcount=gate)
    b.unscratch(gate)
    out = Tensor(Affine.of(lw.capture) + DevVar("capture slot", offset),
                 x.shape, (x.cols, 1))
    ol.store(out, x)
    b.end_loop(loop)
    b.unscratch(offset)


def _norm_rows_inplace(x, gamma, eps):
    """Bound temporary norm tiles when a wide verify occupies most of TMEM."""
    g = ol.load(gamma)
    width = 1 if x.rows > 4 else x.rows
    for start in range(0, x.rows, width):
        rows = x[start:min(start + width, x.rows), :]
        rows.set(rmsnorm(rows, g, eps))


def _head_rows(x, m, logit_rows):
    if x.rows <= 4:
        return Q._lm_head_rows(x, m, m.spec, logit_rows)
    if not logit_rows:
        return
    a, end = logit_rows[0], logit_rows[-1] + 1
    selected = x[a:end, :]
    _norm_rows_inplace(selected, m.g_final, m.spec.eps)
    xs = ol.quantize(selected)
    chunk = Q.head_rows_chunk(end - a)
    sinks = getattr(m, "lm_sinks", None)
    for first in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - first)
        col = ol.program_id() * m.v_loc + first
        values = ol.dot(xs, m.head[first:first + n, :])
        if sinks is None:
            ol.store(m.logitsr[a:end, col:col + n], values)
        else:
            for row, sink in enumerate(sinks):
                sink(values[row:row + 1, :], col)
        del values


@ol.jit
def nanbeige_step(m, pos, block=Q.ATTN_BLOCK, tok=None, host_inputs=False):
    Q.fill_logits(m)
    x, c, s = (ol.load(m.x), ol.load(m.cos), ol.load(m.sin)) if host_inputs \
        else Q._inputs(m, pos, tok)
    for loop in range(m.spec.loops):
        for lw in _layers(m, loop):
            _capture(lw, x)
            x.set(Q._attention(x, lw, c, s, pos, m.spec, block))
            x.set(Q._mlp(x, lw, m.spec))
        if loop + 1 < m.spec.loops and not m.spec.skip_loop_final_norm:
            x.set(rmsnorm(x, ol.load(m.g_final), m.spec.eps))
    Q._lm_head(x, m, m.spec)


@ol.jit
def nanbeige_rows(m, rows, logit_rows, block=Q.ATTN_BLOCK, tokens=None, host_inputs=False):
    x, c, s = (ol.load(m.xr[:rows.R, :]), ol.load(m.cosr[:rows.R, :]),
               ol.load(m.sinr[:rows.R, :])) if host_inputs else Q._inputs_rows(m, rows, tokens)
    for loop in range(m.spec.loops):
        for lw in _layers(m, loop):
            _capture(lw, x)
            x.set(Q._attention_rows(x, lw, c, s, rows, m.spec, block))
            x.set(Q._mlp(x, lw, m.spec))
        if loop + 1 < m.spec.loops and not m.spec.skip_loop_final_norm:
            x.set(rmsnorm(x, ol.load(m.g_final), m.spec.eps))
    Q._lm_head_rows(x, m, m.spec, logit_rows)


@ol.jit
def nanbeige_tiled_rows(m, rows, logit_rows, block=Q.ATTN_BLOCK, tokens=None,
                        spill=False, depth=Q.ATTN_DEPTH, whole=True, oscr=True):
    from . import tiled as TL
    T, p0 = (rows.R, rows) if isinstance(rows, Q.RunRows) else (len(rows), rows[0][1])
    if tokens is None:
        # Long-lived RoPE rows at the bottom of TMEM leave one contiguous span
        # after the residual is spilled. Allocating x first strands the RoPE
        # rows halfway through TMEM and prevents the 8x6144 attention output.
        c, s = ol.load(m.cosr[:T, :]), ol.load(m.sinr[:T, :])
        x = ol.load(m.xr[:T, :])
    else:
        x, c, s = Q._inputs_rows(m, rows, tokens)
    res = TL.Res(x, m.xr[:T, :] if spill else None)
    del x
    parts, sub = TL._widths(m, T)
    for loop in range(m.spec.loops):
        for lw in _layers(m, loop):
            _capture(lw, res.x)
            TL._attention_tile(res, lw, c, s, p0, T, m.spec, block, parts,
                               depth=depth, whole=whole, oscr=TL._oscr(m, oscr), loop=0)
            TL._mlp_tile(res.x, lw, m.spec, parts, sub(lw))
        if loop + 1 < m.spec.loops and not m.spec.skip_loop_final_norm:
            _norm_rows_inplace(res.x, m.g_final, m.spec.eps)
    _head_rows(res.x, m, logit_rows)


@ol.jit
def nanbeige_prompt_run(m, pos, R, kind, block=Q.ATTN_BLOCK):
    with Q.RunWords(m, pos):
        nanbeige_rows.fn(m, pos, [R - 1] if kind == "L" else [], block)


class Engine(Q.Engine):
    """Explicitly exclude optimizers which hard-code another model's forward."""
    def __init__(self, spec, W, runtime_host=True, **kwargs):
        if kwargs.get("layer_major") or kwargs.get("spec_rows"):
            raise ValueError("Nanbeige layer-major/generic speculative engine unsupported")
        kwargs["tile_runs"] = False
        super().__init__(spec, W, **kwargs)
        self.runtime_host = bool(runtime_host and not self.device_inputs and self.batch == 1
                                 and getattr(self.backend, "args", False)
                                 and self.block >= self.image.host_mask_block)
        self._host_block = min(self.block, self.image.host_mask_block)
        self._host_programs = {}
        self._host_compile_errors = {}
        if self.runtime_host:
            # Runtime programs are cached below. Per-position compile prefetch and
            # Qwen's asynchronous stream-fill path are not used by this execution.
            self.pipeline = False
            self.stream_logits = False
            self.image.stream_fill = False

    def _host_program(self, position, rows=1, logit_rows=None):
        bucket = position // self._host_block + 1
        key = (bucket, rows, None if logit_rows is None else tuple(logit_rows))
        if key in self._host_compile_errors:
            raise CompileError(self._host_compile_errors[key])
        hit = key in self._host_programs
        if not hit:
            def compile():
                if logit_rows is None:
                    return self.image.compile_host_step(bucket, self._host_block)
                return self.image.compile_host_rows(bucket, rows, logit_rows, self._host_block)
            try:
                programs, arguments = self.cached(("nanbeige-host-v2",) + key, compile)
            except CompileError as exc:
                # A deterministic capacity failure must not retrace at every
                # prompt chunk before the caller selects a smaller row count.
                self._host_compile_errors[key] = str(exc)
                raise
            prepare = getattr(self.backend, "prepare", None)
            if prepare is not None:
                prepare(programs)
            self._host_programs[key] = programs, arguments
        return (*self._host_programs[key], hit)

    def _host_inputs(self, tokens, positions):
        io = self.image.io
        x = Q.F.ftz(self.embed[[int(t) for t in tokens]].astype(np.float32))
        for sid in range(self.cfg.S):
            self.backend.write(sid, io["x"], x)
            self.backend.write(sid, io["cos"], self._rope[0][positions])
            self.backend.write(sid, io["sin"], self._rope[1][positions])

    def _host_logits(self, rows):
        image, S, V = self.image, self.cfg.S, self.spec.vocab
        vloc = image.v_loc
        return np.asarray([np.concatenate([
            self.backend.read(sid, image.io["logits"] + 4 * (row * V + sid * vloc),
                              4 * vloc).view(np.float32) for sid in range(S)])
                           for row in rows], np.float32).reshape(len(rows), V)

    def step(self, token, on_start=None, sink=None):
        if not self.runtime_host:
            return super().step(token, on_start=on_start, sink=sink)
        if self.pos >= self.cap:
            raise RuntimeError("KV cache full")
        programs, arguments, hit = self._host_program(self.pos)
        self._host_inputs([token], [self.pos])
        words = arg_words(arguments, Q.RunPos.values(int(token), self.pos, block=self._host_block))
        if sink is not None:
            sink.begin(self.spec.vocab)
        start = getattr(self.backend, "start", None)
        if start is None:
            if on_start is not None:
                on_start()
            stats = self.backend.run(programs, args=words)
        else:
            start(programs, args=words)
            if on_start is not None:
                on_start()
            stats = self.backend.wait()
        stats.update(runtime_host=True, host_program_cache_hit=hit)
        self.stats.append(stats)
        logits = self._host_logits([0])[0]
        if sink is not None:
            sink.feed(0, logits)
        self.pos += 1
        return logits

    def run_rows(self, rows, tokens, logit_rows):
        rows, tokens, logit_rows = list(rows), list(tokens), list(logit_rows)
        if not self.runtime_host or not rows or rows != [
                (0, rows[0][1] + r) for r in range(len(rows))] or \
                rows[0][1] // self._host_block != rows[-1][1] // self._host_block:
            return super().run_rows(rows, tokens, logit_rows)
        if len(rows) != len(tokens):
            raise ValueError("runtime host tokens and rows have different lengths")
        if any(p < 0 or p >= self.cap for _, p in rows):
            raise RuntimeError("KV cache full")
        self._drain()
        programs, arguments, hit = self._host_program(rows[0][1], len(rows), logit_rows)
        self._host_inputs(tokens, [p for _, p in rows])
        words = arg_words(arguments, Q.RunRows.values(tokens, rows[0][1], block=self._host_block))
        start = getattr(self.backend, "start", None)
        if start is None:
            stats = self.backend.run(programs, args=words)
        else:
            start(programs, args=words)
            stats = self.backend.wait()
        stats.update(rows=len(rows), runtime_host=True, host_program_cache_hit=hit)
        self.stats.append(stats)
        return self._host_logits(logit_rows)

    def read_hidden(self, layer_ids=(2, 11, 22, 33, 41), rows=1):
        """Latest run's pre-layer residuals, shape [rows, taps, hidden].

        The caller retains only committed token positions across verify runs.
        Padded/stale rows are never implicitly treated as newly computed tokens.
        """
        if not 1 <= rows <= self.image.rows:
            raise ValueError("hidden row count exceeds image capacity")
        if not layer_ids or any(i not in self.image.capture_slots for i in layer_ids):
            raise ValueError("requested layer was not enabled in capture_layers")
        values = []
        for i in layer_ids:
            at = self.image.capture + self.image.capture_slots[i] * self.image.rows * self.spec.hidden * 4
            raw = self.backend.read(0, at, rows * self.spec.hidden * 4)
            values.append(np.asarray(raw).view(np.float32).reshape(rows, self.spec.hidden))
        return np.stack(values, axis=1)


def load_weights(path):
    from .nanbeige_weights import load_nanbeige_weights
    return load_nanbeige_weights(path)


def reference_logits(spec, W, tokens):
    """Independent unquantized FP64 sequence reference, including loop norms."""
    ts = list(tokens)
    T, d, group = len(ts), spec.head_dim, spec.n_q // spec.n_kv
    if not T:
        return np.empty((0, spec.vocab))
    x = np.asarray(W["model.embed_tokens.weight"][ts], np.float64)
    freq = spec.theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angles = np.arange(T)[:, None] * freq[None, :]
    c, s = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]

    def norm(v, gamma):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + spec.eps) * gamma

    def rot(v):
        a, b = np.split(v, 2, axis=-1)
        return np.concatenate((a * c - b * s, b * c + a * s), axis=-1)

    mask = np.triu(np.full((T, T), -np.inf), 1)
    for loop in range(spec.loops):
        for li in range(spec.layers):
            p = f"model.layers.{li}."
            h = norm(x, W[p + "input_layernorm.weight"])
            q = rot((h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, spec.n_q, d))
            k = rot((h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, spec.n_kv, d))
            v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, spec.n_kv, d)
            out = np.empty((T, spec.n_q, d))
            for head in range(spec.n_q):
                scores = q[:, head] @ k[:, head // group].T / math.sqrt(d) + mask
                prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
                out[:, head] = (prob / prob.sum(axis=-1, keepdims=True)) @ v[:, head // group]
            x += out.reshape(T, -1) @ W[p + "self_attn.o_proj.weight"].T
            h = norm(x, W[p + "post_attention_layernorm.weight"])
            gate = h @ W[p + "mlp.gate_proj.weight"].T
            up = h @ W[p + "mlp.up_proj.weight"].T
            x += (gate / (1 + np.exp(-gate)) * up) @ W[p + "mlp.down_proj.weight"].T
        if loop + 1 < spec.loops and not spec.skip_loop_final_norm:
            x = norm(x, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return norm(x, W["model.norm.weight"]) @ head.T
