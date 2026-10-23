"""RAM-only construction plus explicit tiny-model ISA execution via a mock device."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from opentpu import isa
from opentpu.llm.nanbeige import Spec
from opentpu.llm.nanbeige_strict import StrictEngine
from opentpu.llm.qwen3 import Engine, IsaBackend
from opentpu.llm.staged import Case, RamBackend, prepare_nanbeige, prepare_gemma
from opentpu.target import config, load
from test_nanbeige import tiny_weights
from test_row_lanes import _tiny_gemma3


class WordsDevice:
    """Fake board transport: decode the staged words in the explicit ISA simulator."""
    args = generates = chains = runs_words = True
    host = None
    def __init__(self, cfg, images):
        self.cfg, self.waitc = cfg, cfg.SEQ_WC
        self.sim = IsaBackend(cfg, [a.copy() for a in images], adopt=True)
        self.words_seen = []
        self.closed = False
    def attach(self, engine):
        self.engine = engine
    def write(self, *a):
        self.sim.write(*a)
    def read(self, *a):
        return self.sim.read(*a)
    def run(self, words, **kw):
        assert isinstance(words, np.ndarray) and not words.flags.writeable
        assert words.waits is not None
        self.words_seen.append(words)
        instructions = [isa.Instr.decode(words[i:i + 8]) for i in range(0, len(words), 8)]
        return self.sim.run([instructions], **kw)
    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cfg = replace(config(load("f2-hbm:p4rl8a64c150")), DRAM_BYTES=16 << 20)
    nspec = Spec(128, 1, 6, 1, 128, 256, 128, eos=())
    nw = tiny_weights(nspec)
    oracle = StrictEngine(nspec, nw, cfg=cfg, backend="isa")
    prompt = (3, 7)
    expected = tuple(oracle.generate_strict(prompt, 2)["tokens"])
    n = prepare_nanbeige(nspec, nw, cfg,
        [Case("raw", prompt, expected), Case("raw_reset_replay", prompt, expected)])
    gw, gspec = _tiny_gemma3(tmp_path_factory)
    gspec = replace(gspec, eos=())
    oracle = Engine(gspec, gw, cfg=cfg, cap=256, backend="isa", resident=True,
                    pipeline=False, wformat="int8", head_format="int8")
    prompt = (3, 7, 9)
    token = int(np.argmax(oracle.prefill(prompt)))
    expected = (token, int(np.argmax(oracle.step(token))))
    g = prepare_gemma(gspec, gw, cfg, [Case("verify", prompt, expected)], cap=256)
    return n, g


def test_ram_preparation_retains_read_only_images_words_and_no_forward_result(staged):
    for prepared in staged:
        assert isinstance(prepared.engine.backend, RamBackend)
        assert prepared.ram.sealed and not prepared.active
        assert not prepared.manifest["hardware_executed"]
        assert not prepared.manifest["model_forward_executed"]
        assert prepared.engine._host_rows is None and prepared.engine.embed is None
        assert prepared.engine._rope is None and prepared.engine.stats == []
        for image, digest in zip(prepared.images, prepared.manifest["image_sha256"]):
            assert not image.flags.writeable
            assert hashlib.sha256(memoryview(image)).hexdigest() == digest
        assert prepared.ram.programs and all(not w.flags.writeable for _, w in
                                             prepared.ram.programs.values())
        with pytest.raises(RuntimeError, match="not activated"):
            prepared.run_case(prepared.cases[0].name)
        with pytest.raises(RuntimeError, match="cannot execute"):
            prepared.ram.run([])
        with pytest.raises(RuntimeError, match="cannot be modified"):
            prepared.ram.write(0, 0, np.zeros(1, np.uint8))


@pytest.mark.parametrize("model", [0, 1])
def test_prepared_model_executes_exactly_and_resets_without_runtime_compile(staged, model, monkeypatch):
    prepared = staged[model]
    before = prepared.manifest["image_sha256"]
    device = WordsDevice(prepared.engine.cfg, prepared.images)
    monkeypatch.setattr(prepared.engine, "cached", lambda *a: pytest.fail("compiled during outage"))
    prepared.activate(device)
    try:
        for case in (*prepared.cases, prepared.cases[0]):
            result = prepared.run_case(case.name)
            assert tuple(result["tokens"]) == case.expected
        assert device.words_seen
        assert len({id(w) for w in device.words_seen}) <= len(prepared.ram.programs)
        with pytest.raises(RuntimeError, match="only selected token"):
            prepared.engine.backend.read(0, prepared.engine.image.io["logits"], 4)
        with pytest.raises(RuntimeError, match="host model inputs"):
            prepared.engine.backend.write(0, prepared.engine.image.io["x"], np.zeros(1))
        with pytest.raises(RuntimeError, match="unprepared program"):
            prepared.engine.backend.run([[]])
        with pytest.raises(ValueError, match="prepared acceptance case"):
            prepared.run_case("new-prompt")
    finally:
        prepared.detach()
    assert not device.closed  # only the owner may close/release its hardware backend
    assert isinstance(prepared.engine.backend, RamBackend) and not prepared.active
    assert [hashlib.sha256(memoryview(a)).hexdigest() for a in prepared.images] == before


def test_mismatched_backend_fails_before_attachment_or_model_execution(staged):
    prepared = staged[0]
    device = WordsDevice(prepared.engine.cfg, prepared.images)
    device.cfg = replace(device.cfg, MCOLS=4)
    with pytest.raises(ValueError, match="configuration mismatched"):
        prepared.activate(device)
    device.cfg = prepared.engine.cfg
    device.waitc = not device.waitc
    with pytest.raises(ValueError, match="wait-entry ABI"):
        prepared.activate(device)
    device.waitc = prepared.ram.waitc
    device.host = lambda: None
    with pytest.raises(ValueError, match="device-only"):
        prepared.activate(device)
    assert not prepared.active and not device.words_seen


def test_sealed_program_registry_rejects_late_preparation(staged):
    with pytest.raises(RuntimeError, match="not prepared before"):
        staged[0].ram.prepare([[]])


def test_attachment_failure_rolls_back_without_closing_an_owner_backend(staged):
    prepared = staged[0]
    device = WordsDevice(prepared.engine.cfg, prepared.images)
    def fail(engine):
        raise RuntimeError("injected attachment failure")
    device.attach = fail
    with pytest.raises(RuntimeError, match="attachment failure"):
        prepared.activate(device)
    assert not prepared.active and prepared.engine.backend is prepared.ram
    assert not device.closed and not device.words_seen


def test_case_rejects_lossy_or_insufficient_acceptance_tokens(staged):
    prepared = staged[0]
    for case in (Case("", (1,), (2, 3)), Case("raw", (True,), (2, 3)),
                 Case("raw", (1,), (2,)), Case("raw", (1,), (2, True))):
        with pytest.raises(ValueError):
            case.check(prepared.engine.spec, prepared.engine.cap)
