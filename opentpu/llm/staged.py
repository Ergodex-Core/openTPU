"""Prepare resident greedy models in host RAM without opening a device or running math.

There is deliberately no board factory here. A future, explicitly authorized
owner supplies an already-uploaded backend. Preparation is not FPGA acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np

from .. import waits
from . import generate as G, prefill as PF
from .qwen3 import Engine, worker_words


@dataclass(frozen=True)
class Case:
    name: str
    prompt: tuple[int, ...]
    expected: tuple[int, ...]

    def check(self, spec, cap):
        from .nanbeige_strict import request_tokens
        request_tokens(self.prompt, len(self.expected), spec, cap)
        if not self.name or len(self.expected) < 2 or any(
                type(t) is not int or not 0 <= t < spec.vocab for t in self.expected):
            raise ValueError("cases require a name and at least two exact expected token IDs")


class RamBackend:
    """Image buffers and preassembled programs only. Execution always raises."""
    args = generates = chains = True
    host = None
    streams = fills = runs_words = False

    def __init__(self, cfg, images):
        if cfg.S != 1:
            raise ValueError("staging currently requires one slice")
        self.cfg, self.images = cfg, tuple(images)
        self.waitc = cfg.SEQ_WC
        self.engine = None
        self.programs, self.slots = {}, {}
        self.sealed = False

    def attach(self, engine):
        self.engine = engine

    def prepare(self, programs):
        if id(programs) in self.programs:
            if self.programs[id(programs)][0] is not programs:
                raise RuntimeError("program identity changed")
            return
        if self.sealed:
            raise RuntimeError("program was not prepared before the outage")
        if len(programs) != 1:
            raise ValueError("expected one slice's programs")
        prog = programs[0]
        words = worker_words(self.engine.image, prog)
        words.flags.writeable = False
        if words.waits is not None:
            words.waits.flags.writeable = False
        self.programs[id(programs)] = (programs, words)
        slot = waits.loadable(words, words.waits, self.cfg.D) if self.waitc else words
        slot.flags.writeable = False
        self.slots[id(prog)] = (prog, slot)

    def slot_words(self, prog):
        got = self.slots.get(id(prog))
        if got is None or got[0] is not prog:
            raise RuntimeError("chain program was not prepared before the outage")
        return got[1]

    def write(self, sid, at, data):
        if self.sealed:
            raise RuntimeError("sealed RAM preparation cannot be modified")
        v = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if sid != 0 or at < 0 or at + len(v) > len(self.images[sid]):
            raise ValueError("staged image write exceeds its buffer")
        self.images[sid][at:at + len(v)] = v

    def read(self, *args, **kwargs):
        raise RuntimeError("RAM preparation cannot execute inference or read model outputs")

    run = run_generate = read


class _Execution:
    """Use prepared Words on a board and original programs in explicit ISA tests."""
    def __init__(self, ram, backend):
        self.ram, self.raw = ram, backend

    def __getattr__(self, name):
        if name == "run_generate":
            runner = getattr(self.raw, name)  # absence preserves the ISA-only run path
            return lambda programs, *a, **kw: runner(self._program(programs), *a, **kw)
        return getattr(self.raw, name)

    def _program(self, programs):
        got = self.ram.programs.get(id(programs))
        if got is None or got[0] is not programs:
            raise RuntimeError("inference attempted an unprepared program")
        return got[1] if getattr(self.raw, "runs_words", False) else programs

    def prepare(self, programs):
        self._program(programs)

    def slot_words(self, prog):
        return self.ram.slot_words(prog)

    def run(self, programs, **kwargs):
        return self.raw.run(self._program(programs), **kwargs)

    def write(self, sid, at, data):
        image = self.ram.engine.image
        if at in {image.io[k] for k in ("x", "cos", "sin", "rope", "pe") if k in image.io}:
            raise RuntimeError("host model inputs are forbidden")
        self.raw.write(sid, at, data)

    def read(self, sid, at, count):
        out = self.ram.engine.image.lookup["gen"]["out"]
        if sid != 0 or not out <= at <= at + count <= out + 4 * (self.ram.engine.cap + 1):
            raise RuntimeError("only selected token IDs may be read")
        return self.raw.read(sid, at, count)


class PreparedModel:
    """Sealed image/program buffers. The owner controls upload and backend lifetime."""
    def __init__(self, name, engine, cases, provenance, started):
        self.name, self.engine, self.ram = name, engine, engine.backend
        if not isinstance(self.ram, RamBackend):
            raise TypeError("preparation must use RamBackend")
        self.cases = tuple(cases)
        if not self.cases or len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("case names must be nonempty and unique")
        if engine.pipeline or engine.server is not None or engine.row_server is not None or \
                not engine.resident or not engine.device_inputs or \
                getattr(engine.image, "embed_host", False) or \
                engine._conv_lo or engine._rope is not None or not engine.can_generate:
            raise ValueError("staging requires a resident device-only greedy model")
        host_rows = engine._host_rows
        if host_rows is not None and host_rows([0]):
            raise ValueError("host row service is forbidden")
        engine._host_rows = None
        engine.embed = None
        engine.stream_logits = engine.image.stream_fill = False
        for case in self.cases:
            case.check(engine.spec, engine.cap)
            for p in range(len(case.prompt) - 1):
                PF.programs(engine, p, 1, "P", tile=False)
        for bucket in range(1, G.buckets(engine.cap, engine.block) + 1):
            engine._generate_prog(bucket)
        engine._generate_chain(1, G.buckets(engine.cap, engine.block))
        # Hash once before admission gating; arrays stay read-only and resident in RAM.
        hashes = [hashlib.sha256(memoryview(a)).hexdigest() for a in self.ram.images]
        for image in self.ram.images:
            image.flags.writeable = False
        self.ram.sealed = True
        self.manifest = dict(name=name, config=asdict(engine.cfg), cap=engine.cap,
            image_bytes=[len(a) for a in self.ram.images], image_sha256=hashes,
            cases=[asdict(c) for c in self.cases], provenance=provenance,
            prepared_programs=len(self.ram.programs),
            program_words=[len(w) for _, w in self.ram.programs.values()],
            preparation_seconds=time.monotonic() - started,
            numpy_version=np.__version__,
            hardware_executed=False, model_forward_executed=False,
            host_images_read_only=True)
        self.identity = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()
        self.active = False

    @property
    def images(self):
        return self.ram.images

    def activate(self, backend):
        """Owner calls this only after uploading these exact buffers under normal locks.

        This method does not acquire, upload, stop, close, or release a backend.
        The owner must enforce its deadline around those operations.
        """
        if self.active or not self.ram.sealed or backend.cfg != self.engine.cfg:
            raise ValueError("prepared model is active, unsealed, or configuration mismatched")
        if any(not getattr(backend, k, False) for k in ("args", "generates", "chains")) or \
                getattr(backend, "host", None) is not None:
            raise ValueError("backend lacks the device-only execution contract")
        if bool(getattr(backend, "waitc", False)) != bool(self.ram.waitc) and \
                getattr(backend, "runs_words", False):
            raise ValueError("prepared wait-entry ABI differs from the backend")
        self.engine.backend = _Execution(self.ram, backend)
        self.engine.reset()
        self.engine.stats.clear()
        self.active = True
        try:
            if hasattr(backend, "attach"):
                backend.attach(self.engine)
        except BaseException:
            self.engine.backend = self.ram
            self.active = False
            raise

    def detach(self):
        """Owner must close/release its backend separately; keep staged RAM intact."""
        self.engine.backend = self.ram
        self.engine.reset()
        self.active = False

    def run_case(self, name, *, stop=None):
        if not self.active:
            raise RuntimeError("owner has not activated an uploaded backend")
        case = next((c for c in self.cases if c.name == name), None)
        if case is None:
            raise ValueError("only a prepared acceptance case may run")
        eng = self.engine
        if hasattr(eng, "generate_strict"):
            return eng.generate_strict(case.prompt, len(case.expected), stop=stop)
        eng.reset()
        first = len(eng.stats)
        PF.write_tokens(eng, 0, case.prompt)
        for p in range(len(case.prompt) - 1):
            if stop is not None and stop():
                raise TimeoutError("restoration verification cancelled before completion")
            eng.stats.append(PF.run(eng, PF.programs(eng, p, 1, "P", tile=False), p, 1))
            eng.pos += 1
        tokens = eng.generate_card(case.prompt[-1], len(case.expected), stop=stop)
        return dict(tokens=tokens, raw_runs=list(eng.stats[first:]))


def prepare_nanbeige(spec, weights, cfg, cases, *, cap=128, provenance=None):
    from .nanbeige_strict import StrictEngine
    start = time.monotonic()
    engine = StrictEngine(spec, weights, cfg=cfg, backend=RamBackend, cap=cap)
    return PreparedModel("nanbeige", engine, cases, provenance or {}, start)


def prepare_gemma(spec, weights, cfg, cases, *, cap=768, provenance=None):
    from .gemma3 import Spec as GemmaSpec
    if not isinstance(spec, GemmaSpec):
        raise TypeError("restore preparation requires a Gemma 3 spec")
    start = time.monotonic()
    engine = Engine(spec, weights, cfg=cfg, backend=RamBackend, cap=cap,
                    resident=True, pipeline=False, snapshots=0,
                    wformat="int8", head_format="int8", prog_cache=False)
    return PreparedModel("gemma", engine, cases, provenance or {}, start)
