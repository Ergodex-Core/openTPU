"""Nanbeige target-only inference with no host model computation.

The same device programs may be exercised explicitly by the ISA test backend.
There is no CPU fallback. Checkpoint packing and constant-table construction are
offline setup; embeddings, RoPE application, all 44 logical layers, final norm,
head, greedy selection (including the first token), feedback and EOS run on device.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral
import os
import time

from .. import waits
from ..compiler import CompileError
from . import generate as G
from . import prefill as PF
from .nanbeige import Engine, Spec


PRECISION = {
    "target_execution": "openTPU MX W8A8, INT8 head, INT8 device embedding gather; FP32 residual/norm/RoPE/softmax",
    "source_equivalent": False,
    "embedding_change": "source BF16 embeddings are additionally quantized to MX INT8 during image preparation",
    "checkpoint_quality_validated": False,
}


def strict_spec(spec, embedding_format="int8"):
    if not isinstance(spec, Spec):
        raise TypeError("strict Nanbeige requires a Nanbeige Spec")
    if embedding_format not in ("int8", "f32"):
        raise ValueError("embedding_format must be int8 or f32")
    if spec.formats or os.environ.get("OTPU_FORMATS", "").strip():
        raise ValueError("strict Nanbeige requires uniform INT8 weight/head formats")
    # Captures belong to the speculative draft; the strict target needs none.
    return replace(spec, embed=embedding_format, capture_layers=())


def request_tokens(prompt, max_new_tokens, spec, cap):
    """Reject lossy conversions and oversized requests before state or device writes."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    try:
        tokens = list(prompt)
    except TypeError as exc:
        raise ValueError("prompt must contain integer token IDs") from exc
    if not tokens or any(isinstance(t, bool) or not isinstance(t, Integral) or
                         not 0 <= t < spec.vocab for t in tokens):
        raise ValueError("prompt must contain nonempty integer token IDs within the vocabulary")
    if len(tokens) + max(int(max_new_tokens) - 1, 0) > cap:
        raise ValueError("prompt plus requested generation exceeds capacity")
    return [int(t) for t in tokens], int(max_new_tokens)


def plan(spec, cfg, cap=128, embedding_format="int8", block=256):
    """Metadata and compiler preflight only; no checkpoint tensors or transport."""
    spec = strict_spec(spec, embedding_format)
    if cfg.S != 1:
        raise ValueError("strict Nanbeige currently requires one slice")
    if type(cap) is not int or cap < cfg.D or cap % cfg.D:
        raise ValueError("capacity must be a positive multiple of D")
    if block != 256:
        raise ValueError("strict Nanbeige uses the image's 256-token attention buckets")
    if len(spec.eos) > G.N_STOP or any(type(t) is not int or not 0 <= t < spec.vocab for t in spec.eos):
        raise ValueError("EOS IDs must fit the device stop table and vocabulary")
    # Calculate the full requirement before rejecting, including the program area.
    image = spec.image(replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, 1 << 40)), cap,
                       rows=1, lookup=True, embed_host=False,
                       wformat="int8", head_format="int8", formats="")
    required = (image.nbytes + 4095) // 4096 * 4096 + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    report = {"embedding_format": embedding_format, "image_bytes": image.nbytes,
              "required_bytes": required, "dram_bytes": cfg.DRAM_BYTES,
              "fits_dram": required <= cfg.DRAM_BYTES, "cap": cap, "rows": 1,
              "host_embedding": False, "logical_layers": spec.logical_layers,
              "physical_layers": spec.layers,
              "head_storage": "row_lanes_only" if image.head is None else
                              "standard_and_row_lanes" if image.head_rl else "standard",
              "programs": []}
    if not report["fits_dram"]:
        raise MemoryError(f"strict Nanbeige {embedding_format} embedding needs {required} bytes; "
                          f"device has {cfg.DRAM_BYTES}; host offload is forbidden")
    for bucket in range(1, G.buckets(cap, block) + 1):
        prefix, _ = image.compile_prompt_run(bucket, 1, "P", block)
        if not G.fits(image, prefix):
            raise CompileError(f"strict prefill bucket {bucket} exceeds IMEM")
        generated = G.compile_bucket(image, bucket, (bucket - 1) * block, block, chain=True)
        parts = generated if isinstance(generated, tuple) else (generated,)
        if not all(G.fits(image, p) for p in parts):
            raise CompileError(f"strict generation bucket {bucket} exceeds IMEM")
        report["programs"].append({"bucket": bucket,
            "prefill_words": [len(p) * 8 for p in prefix],
            "generate_words": [[len(p) * 8 for p in part] for part in parts],
            "split_generate": isinstance(generated, tuple)})
    report["fits_imem"] = True
    return report


class StrictEngine(Engine):
    """A deliberately small adapter: one sequence, one-row prefill, greedy only.

    Production callers supply a qualified board factory. ``backend='isa'`` is
    explicit validation only and never selected after a hardware failure.
    """
    def __init__(self, spec, weights, *, cfg, backend, cap=128,
                 embedding_format="int8", prog_cache=False):
        self.strict_plan = plan(spec, cfg, cap, embedding_format)
        super().__init__(strict_spec(spec, embedding_format), weights, cfg=cfg,
                         backend=backend, cap=cap, rows=1, runtime_host=False,
                         resident=True, embed_host=False, pipeline=False,
                         prompt_runs=False, wformat="int8", head_format="int8",
                         prog_cache=prog_cache)
        self.stream_logits = False
        self.image.stream_fill = False
        self.embed = None  # A host lookup must never become an implicit fallback.
        if not self.image.embed_host:
            self._host_rows = None  # Q.Image.host_rows would only return an empty list.
        self._check_strict()

    def _check_strict(self):
        if not self.can_generate or not getattr(self.backend, "chains", False):
            raise RuntimeError("strict generation requires device ARGMAX/RLD/HALT CHAIN")
        if not self.device_inputs or self.image.embed_host or self.runtime_host or self._conv_lo != 0:
            raise RuntimeError("strict generation forbids host inputs or initial CPU token selection")
        if self.server is not None or self.row_server is not None or self._host_rows is not None:
            raise RuntimeError("strict generation forbids host model servers")
        if self.gen_debug or self._rope is not None or self.embed is not None:
            raise RuntimeError("strict generation forbids host logits/embedding/RoPE paths")
        if self.image.spec.embed not in ("f32", "int8") or not self.image.lookup:
            raise RuntimeError("strict generation requires a device embedding table")

    def generate_strict(self, prompt, max_new_tokens=16, on_token=None, stop=None):
        tokens, budget = request_tokens(prompt, max_new_tokens, self.spec, self.cap)
        self._check_strict()
        started = time.perf_counter()
        first = len(self.stats)
        self.reset()
        if not budget:
            return {"tokens": [], "stop_reason": "length", "generated_tokens": 0,
                    "prefill_seconds": 0.0, "generation_seconds": 0.0,
                    "total_seconds": time.perf_counter() - started,
                    "prefill_runs": 0, "generation_runs": 0, "raw_runs": []}
        # Compile all needed programs before any prompt write. Prefix programs
        # omit the LM head. The last prompt token enters the device greedy loop,
        # so the first output token is also selected on FPGA.
        prefix = [PF.programs(self, p, 1, "P", tile=False) for p in range(len(tokens) - 1)]
        for bucket in range((len(tokens) - 1) // self.block + 1,
                            (len(tokens) + budget - 2) // self.block + 2):
            self._generate_prog(bucket)
        PF.write_tokens(self, 0, tokens)
        for p, entry in enumerate(prefix):
            if stop is not None and stop():
                return self._result([], "cancelled", started, time.perf_counter(), first, len(self.stats))
            stat = PF.run(self, entry, p, 1)
            stat["strict_stage"] = "prefill"
            self.stats.append(stat)
            self.pos += 1
        split = len(self.stats)
        generation_started = time.perf_counter()
        out = self.generate_card(tokens[-1], budget, on_token=on_token, stop=stop)
        if any(type(t) is not int or not 0 <= t < self.spec.vocab for t in out):
            raise RuntimeError("device returned an invalid token ID")
        reason = "eos" if out and out[-1] in self.spec.eos else \
                 "length" if len(out) == budget else "cancelled" if stop is not None and stop() else "incomplete"
        if reason == "incomplete":
            raise RuntimeError("device stopped before EOS or the requested budget")
        return self._result(out, reason, started, generation_started, first, split)

    def _result(self, out, reason, started, generation_started, first, split):
        ended = time.perf_counter()
        return {"tokens": out, "stop_reason": reason, "generated_tokens": len(out),
                "prefill_seconds": generation_started - started,
                "generation_seconds": ended - generation_started,
                "total_seconds": ended - started, "prefill_runs": split - first,
                "generation_runs": len(self.stats) - split,
                "raw_runs": list(self.stats[first:])}

    def generate(self, prompt, max_new=32, sampler=None, on_token=None):
        if sampler is not None:
            raise ValueError("strict Nanbeige supports FPGA greedy selection only")
        return self.generate_strict(prompt, max_new, on_token=on_token)["tokens"]

    def generate_card(self, tok, n, stop_ids=None, on_token=None, stop=None,
                      sampling=None, context=(), rng=None):
        self._check_strict()
        if sampling is not None or rng is not None or context or stop_ids is not None:
            raise ValueError("strict Nanbeige permits only greedy generation with model EOS IDs")
        request_tokens([tok], n, self.spec, self.cap - self.pos)
        return super().generate_card(tok, n, on_token=on_token, stop=stop)

    def step(self, *args, **kwargs):
        raise RuntimeError("strict adapter does not expose host logits; use generate_strict")

    prefill = step
    run_rows = step
    prefill_chunks = step
    step_batch = step
    generate_batch = step
