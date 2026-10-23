"""Resident Mimi input RVQ search; native checkpoint qualification is separately gated.

The semantic and acoustic branches independently project one input token to
256 dimensions. Each source-order codebook selects the closest shared frozen
centroid, then subtracts it from only its own branch residual. Direct FP32
squared distances can round differently from native torch.cdist; exact native
ID agreement and distance margins require separate qualification.
"""
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np

from .. import language as ol
from .. import waits
from ..compiler import CompileError, Tensor, trace_fit
from .generate import Greedy
from .personaplex_audio_heads import _checked
from .personaplex_embeddings import _row
from .personaplex_mimi import _isa_exact_guard


@dataclass(frozen=True)
class MimiInputRVQSpec:
    hidden: int = 512
    dim: int = 256
    vocab: int = 2048
    codebooks: int = 8

    def __post_init__(self):
        for name in ("hidden", "dim", "vocab", "codebooks"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below 2^24")
        if self.codebooks < 2:
            raise ValueError("Input RVQ requires semantic and acoustic codebooks")


@ol.jit
def _input_rvq_init(m):
    x = ol.load(m.input)
    for branch in range(2):
        output = m.semantic_projected if branch == 0 else m.acoustic_projected
        for channel in ol.range(m.dim):
            coefficient = ol.load(Tensor(m.projection_data[branch] + channel * (4 * m.hidden),
                                          (m.hidden,), (1,)))
            reduced = ol.sum(x * coefficient[None, :], axis=1)
            ol.store(output[:, channel:channel + 1], reduced[:, None])
    ol.store(m.semantic_residual, ol.load(m.semantic_projected))
    ol.store(m.acoustic_residual, ol.load(m.acoustic_projected))
    ol.store(m.output_ids, ol.zeros((m.codebooks,)))
    ol.store(m.counter, ol.zeros((1,)))


@ol.jit
def _input_rvq_search(m):
    counter = ol.load(m.counter)
    _isa_exact_guard(counter, float(m.codebook_index), "input RVQ source-order codebook")
    residual = ol.load(m.residual)
    ol.store(m.residual_before[m.codebook_index:m.codebook_index + 1, :], residual)
    greedy = Greedy(ol.current(), m.vocab, m.chunk)
    for lo in range(0, m.vocab, m.chunk):
        hi = min(lo + m.chunk, m.vocab)
        centroids = ol.load(m.table[lo:hi, :])
        delta = centroids - residual[0, :][None, :]
        distance = ol.sum(delta * delta, axis=1).reshape(1, hi - lo)
        ol.store(m.distances[m.codebook_index:m.codebook_index + 1, lo:hi], distance)
        greedy(distance * -1.0, lo)
    token = greedy.token()  # ARGMAX of -distance picks the first minimum ID.
    ol.store(m.output_ids[m.codebook_index:m.codebook_index + 1], token)
    chosen = _row(m.table, token)
    ol.store(m.selected_centroids[m.codebook_index:m.codebook_index + 1, :], chosen)
    updated = residual - chosen
    ol.store(m.residual, updated)
    ol.store(m.residual_after[m.codebook_index:m.codebook_index + 1, :], updated)
    ol.store(m.counter, counter + 1.0)


class MimiInputRVQImage:
    """One projection program and eight ordered resident codebook programs."""

    def __init__(self, spec, cfg, *, input_at, tables_at, state_base, weight_base,
                 state_limit=0x60000000, weight_limit=0xB0000000):
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        if cfg.S != 1 or spec.dim % cfg.D or spec.hidden % cfg.D:
            raise ValueError("Input RVQ needs one slice and D-aligned vectors")
        if type(input_at) is not int or input_at < 0 or input_at % 128:
            raise ValueError("Input RVQ input must be a nonnegative aligned address")
        if type(tables_at) is not int or tables_at < 0 or tables_at % 128:
            raise ValueError("Input RVQ shared tables must be a nonnegative aligned address")
        if input_at + 4 * spec.hidden > capacity or tables_at + 4 * spec.codebooks * spec.vocab * spec.dim > capacity:
            raise MemoryError("Input RVQ input or shared tables exceed mapped capacity")
        for lo, hi in ((state_base, state_limit), (weight_base, weight_limit)):
            if type(lo) is not int or type(hi) is not int or lo % 128 or not 0 <= lo < hi <= capacity:
                raise MemoryError("Invalid input RVQ reservation")
        if state_base < weight_limit and weight_base < state_limit:
            raise ValueError("Input RVQ state and weights overlap")
        self.spec, self.cfg = spec, cfg
        self.input_at, self.tables_at = input_at, tables_at
        self.spans, self.payload_hashes, self.programs = [], {}, None
        cursor = {"input_rvq_state": state_base, "input_rvq_weights": weight_base}
        limits = {"input_rvq_state": state_limit, "input_rvq_weights": weight_limit}

        def alloc(name, size, region="input_rvq_state", kind="state"):
            at = -(-cursor[region] // 128) * 128
            if size <= 0 or at + size > limits[region]:
                raise MemoryError(f"Input RVQ {name} exceeds {region}")
            for source, start, length in (("input", input_at, 4 * spec.hidden),
                                          ("shared tables", tables_at,
                                           4 * spec.codebooks * spec.vocab * spec.dim)):
                if at < start + length and start < at + size:
                    raise ValueError(f"Input RVQ {name} overlaps {source}")
            self.spans.append(dict(name="input_rvq_" + name, base=at, bytes=size,
                                   region=region, kind=kind, alignment=128))
            cursor[region] = at + size
            return at

        self.io = dict(
            semantic_projected=alloc("semantic_projected", 4 * spec.dim),
            acoustic_projected=alloc("acoustic_projected", 4 * spec.dim),
            semantic_residual=alloc("semantic_residual", 4 * spec.dim),
            acoustic_residual=alloc("acoustic_residual", 4 * spec.dim),
            residual_before=alloc("residual_before", 4 * spec.codebooks * spec.dim),
            selected_centroids=alloc("selected_centroids", 4 * spec.codebooks * spec.dim),
            residual_after=alloc("residual_after", 4 * spec.codebooks * spec.dim),
            distances=alloc("distances", 4 * spec.codebooks * spec.vocab),
            output_ids=alloc("output_ids", 4 * spec.codebooks),
            counter=alloc("counter", 4))
        self.io["ids"] = self.io["output_ids"]
        self.projection_data = [alloc(f"projection{branch}_fp32", 4 * spec.dim * spec.hidden,
                                      "input_rvq_weights", "immutable") for branch in range(2)]
        self.reset_regions = [(span["base"], span["bytes"]) for span in self.spans
                              if span["region"] == "input_rvq_state"]
        self.state_end = -(-cursor["input_rvq_state"] // 4096) * 4096
        self.weight_end = -(-cursor["input_rvq_weights"] // 4096) * 4096
        if self.state_end > state_limit or self.weight_end > weight_limit:
            raise MemoryError("Input RVQ aligned reservation exceeds capacity")
        self.required_bytes = max(self.state_end, self.weight_end)
        self.weight_bytes = 8 * spec.dim * spec.hidden

    @classmethod
    def after_downsample(cls, ds, spec=None):
        """Append state/weights/programs, referring to the existing mimi_tables."""
        if not hasattr(ds, "prefix_spans"):
            raise ValueError("Use a downsample appended to the resident input path")
        spec = MimiInputRVQSpec() if spec is None else spec
        prefix = list(ds.prefix_spans) + list(ds.spans)
        tables = [span for span in prefix if span["name"] == "mimi_tables"]
        expected = 4 * spec.codebooks * spec.vocab * spec.dim
        if len(tables) != 1 or tables[0]["bytes"] != expected or tables[0]["kind"] != "immutable":
            raise ValueError("Input RVQ requires the exact existing frozen mimi_tables span")
        if ds.spec.hidden != spec.hidden or ds.io["output"] % 128:
            raise ValueError("Input RVQ needs an aligned one-row downsample output")
        state_limit = ds.input_state_limit
        image = cls(spec, ds.cfg, input_at=ds.io["output"], tables_at=tables[0]["base"],
                    state_base=ds.state_end, state_limit=state_limit,
                    weight_base=ds.weight_end)
        image.prefix_spans = prefix
        image.input_state_limit = state_limit
        slot_bytes = waits.area(ds.cfg.IMEM_WORDS, ds.cfg.SEQ_WC)
        next_slot = max(span["base"] + span["bytes"] for span in prefix
                        if span["region"] == "program")
        image.program_slots = []
        for index in range(1 + spec.codebooks):
            if next_slot + slot_bytes > 0x05000000:
                raise MemoryError("Combined input RVQ instructions exceed program region")
            image.program_slots.append(next_slot)
            image.spans.append(dict(name=f"input_rvq_program{index}", base=next_slot,
                                    bytes=slot_bytes, region="program", kind="program", alignment=128))
            next_slot += slot_bytes
        ordered = sorted(prefix + image.spans, key=lambda span: span["base"])
        for first, second in zip(ordered, ordered[1:]):
            if first["base"] + first["bytes"] > second["base"]:
                raise MemoryError("Input RVQ overlaps resident span: " + first["name"] +
                                  "/" + second["name"])
        return image

    def descriptors(self):
        s = self.spec
        row = lambda address, width: Tensor(address, (1, width), (width, 1))
        return SimpleNamespace(
            input=row(self.input_at, s.hidden), hidden=s.hidden, dim=s.dim,
            vocab=s.vocab, codebooks=s.codebooks, chunk=min(32, s.vocab),
            projection_data=self.projection_data,
            semantic_projected=row(self.io["semantic_projected"], s.dim),
            acoustic_projected=row(self.io["acoustic_projected"], s.dim),
            semantic_residual=row(self.io["semantic_residual"], s.dim),
            acoustic_residual=row(self.io["acoustic_residual"], s.dim),
            residual_before=Tensor(self.io["residual_before"],
                                   (s.codebooks, s.dim), (s.dim, 1)),
            selected_centroids=Tensor(self.io["selected_centroids"],
                                      (s.codebooks, s.dim), (s.dim, 1)),
            residual_after=Tensor(self.io["residual_after"],
                                  (s.codebooks, s.dim), (s.dim, 1)),
            distances=Tensor(self.io["distances"], (s.codebooks, s.vocab), (s.vocab, 1)),
            output_ids=Tensor(self.io["output_ids"], (s.codebooks,), (1,)),
            counter=Tensor(self.io["counter"], (1,), (1,)),
            tables=[Tensor(self.tables_at + 4 * index * s.vocab * s.dim,
                           (s.vocab, s.dim), (s.dim, 1)) for index in range(s.codebooks)])

    def compile(self):
        if self.programs is None:
            m = self.descriptors()
            programs = [trace_fit(_input_rvq_init, self.cfg, 0, lambda: {"m": m}).finish()]
            for index, table in enumerate(m.tables):
                branch = m.semantic_residual if index == 0 else m.acoustic_residual
                step = SimpleNamespace(counter=m.counter, codebook_index=index,
                    residual=branch, table=table, distances=m.distances,
                    residual_before=m.residual_before,
                    selected_centroids=m.selected_centroids,
                    residual_after=m.residual_after,
                    output_ids=m.output_ids, vocab=m.vocab, chunk=m.chunk)
                programs.append(trace_fit(_input_rvq_search, self.cfg, 0,
                                           lambda step=step: {"m": step}).finish())
            if any(program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS for program in programs):
                raise CompileError("Input RVQ requires host arguments or exceeds IMEM")
            self.programs = programs
        return self.programs

    def upload(self, backend, input_proj_weights):
        """Upload only two raw FP32 projection matrices; tables have one owner."""
        self.compile()
        value = input_proj_weights() if callable(input_proj_weights) else input_proj_weights
        value = _checked(value, (2, self.spec.dim, self.spec.hidden), "Mimi input projections")
        for branch, address in enumerate(self.projection_data):
            raw = np.ascontiguousarray(value[branch]).view(np.uint8).reshape(-1)
            self.payload_hashes[f"projection{branch}_fp32"] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, address, raw)
        self.reset(backend)

    def reset(self, backend):
        for address, size in self.reset_regions:
            backend.write(0, address, np.zeros(size, np.uint8))

    def budget(self):
        result = dict(required_bytes=self.required_bytes, state_end=self.state_end,
            weight_end=self.weight_end, occupied_bytes=sum(span["bytes"] for span in self.spans),
            weight_bytes=self.weight_bytes,
            payload_bytes=sum(span["bytes"] for span in self.spans if span["kind"] == "immutable"),
            persistent_reset_bytes=sum(size for _, size in self.reset_regions),
            program_words=[len(program) * 8 for program in self.compile()],
            spans=self.spans, payload_sha256=self.payload_hashes,
            shared_tables_at=self.tables_at,
            shared_tables_bytes=4 * self.spec.codebooks * self.spec.vocab * self.spec.dim,
            distance_policy="Resident FP32 direct squared distance; native cdist ID/margin qualification pending",
            selection_policy="ARGMAX(-distance), first minimum ID across ordered 32-row chunks")
        if hasattr(self, "prefix_spans"):
            joint = self.prefix_spans + self.spans
            result.update(combined_occupied_bytes=sum(span["bytes"] for span in joint),
                combined_required_bytes=-(-max(span["base"] + span["bytes"]
                                           for span in joint) // 4096) * 4096,
                program_slots=self.program_slots,
                input_state_slack_bytes=self.input_state_limit - self.state_end,
                input_weight_slack_bytes=0xB0000000 - self.weight_end)
        return result
