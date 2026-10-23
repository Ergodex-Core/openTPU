"""Bounded PersonaPlex depth-chain integration on EC2's ISA simulator.

The standalone conditioning, depth-layer, and audio-head engines provide an
independent device-executed composition. These tests need neither a checkpoint
nor physical FPGA. Do not run model arithmetic on the development Mac.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm.personaplex_audio_heads import AudioHeadEngine
from opentpu.llm.personaplex_depth_input import DepthInputEngine, DepthInputSpec
from opentpu.llm.personaplex_depth_layer import DepthLayerEngine, DepthLayerSpec
from opentpu.llm.personaplex_depth_chain import DepthChainEngine, DepthChainImage, DepthChainSpec


NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def fixture(*, steps=4, layers=2, policy="native"):
    spec = DepthChainSpec(hidden=64, depth=64, n_heads=2, ffn=128,
                          steps=steps, layers=layers, text_vocab=7,
                          audio_vocab=5, head_vocab=4, cache_policy=policy)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 32)
    rng = np.random.default_rng(6431)
    projections = rng.normal(0, 0.12, (steps, spec.depth, spec.hidden)).astype(np.float32)
    embeddings = (rng.normal(0, 0.6, (spec.text_vocab, spec.depth)).astype(np.float32),) + tuple(
        rng.normal(0, 0.6, (spec.audio_vocab, spec.depth)).astype(np.float32)
        for _ in range(steps - 1))
    weights = []
    for _ in range(layers):
        banks = []
        for _ in range(steps):
            bank = {name: rng.normal(0, 0.025, (spec.depth, spec.depth)).astype(np.float32)
                    for name in NAMES[:4]}
            bank.update(wg=rng.normal(0, 0.025, (spec.ffn, spec.depth)).astype(np.float32),
                        wu=rng.normal(0, 0.025, (spec.ffn, spec.depth)).astype(np.float32),
                        wd=rng.normal(0, 0.025, (spec.depth, spec.ffn)).astype(np.float32))
            banks.append(bank)
        weights.append({"norm1": rng.uniform(0.8, 1.2, spec.depth).astype(np.float32),
                        "norm2": rng.uniform(0.8, 1.2, spec.depth).astype(np.float32),
                        "steps": banks})
    heads = rng.normal(0, 0.15, (steps, spec.head_vocab, spec.depth)).astype(np.float32)
    hidden = rng.normal(0, 1.5, spec.hidden).astype(np.float32)
    return spec, cfg, projections, embeddings, weights, heads, hidden


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def reference_engines(spec, cfg, projections, embeddings, layers, heads):
    """The three existing standalone implementations, without a chain adapter."""
    input_spec = DepthInputSpec(hidden=spec.hidden, depth=spec.depth, codebooks=spec.steps,
                                text_vocab=spec.text_vocab, audio_vocab=spec.audio_vocab)
    layer_spec = DepthLayerSpec(hidden=spec.depth, n_heads=spec.n_heads, ffn=spec.ffn,
                                steps=spec.steps, layers=spec.layers,
                                cache_policy=spec.cache_policy)
    compact_cfg = replace(cfg, DRAM_BYTES=1 << 27)
    return (DepthInputEngine(projections, embeddings, spec=input_spec, cfg=compact_cfg),
            DepthLayerEngine(layers, spec=layer_spec, cfg=compact_cfg),
            AudioHeadEngine(heads, cfg=compact_cfg, hidden=spec.depth,
                            codebooks=spec.steps, vocab=spec.head_vocab))


def test_chain_matches_independent_device_composition_at_every_boundary():
    spec, cfg, projections, embeddings, layers, heads, hidden = fixture()
    from contextlib import ExitStack

    with ExitStack() as stack:
        chain = stack.enter_context(DepthChainEngine(projections, embeddings, layers, heads,
                                                     spec=spec, cfg=cfg))
        conditioner, transformer, audio = [stack.enter_context(eng) for eng in
                                             reference_engines(spec, cfg, projections,
                                                               embeddings, layers, heads)]
        text_id = spec.text_vocab - 1  # last valid text row, beside the padding row
        assert chain.begin_frame(hidden, text_id) is None
        assert chain.read_counter() == 0
        assert chain.read_previous_id() == text_id
        with pytest.raises(RuntimeError):
            chain.read_ids()
        expected_ids = []
        previous_id = text_id
        for step in range(spec.steps):
            conditioner.step(hidden, step, previous_id)
            transformer.step(conditioner.read_output())
            token_id = audio.step(transformer.read_output(), step)
            expected_ids.append(token_id)
            assert chain.step() is None
            assert chain.read_counter() == step + 1
            assert chain.read_previous_id() == token_id
            np.testing.assert_array_equal(bits(chain.read_projected()),
                                          bits(conditioner.read_projected()))
            np.testing.assert_array_equal(bits(chain.read_conditioned()),
                                          bits(conditioner.read_output()))
            np.testing.assert_array_equal(bits(chain.read_output()),
                                          bits(transformer.read_output()))
            np.testing.assert_array_equal(bits(chain.read_logits()),
                                          bits(audio.read_logits()))
            for layer in range(spec.layers):
                np.testing.assert_array_equal(chain.read_cache(layer),
                                              transformer.read_cache(layer))
            previous_id = token_id
        np.testing.assert_array_equal(chain.read_ids(), np.asarray(expected_ids, np.int64))
        with pytest.raises(RuntimeError, match="full"):
            chain.step()


def test_chain_keeps_generated_ids_on_device_and_selects_each_audio_table():
    spec, cfg, projections, embeddings, layers, heads, hidden = fixture()
    projections.fill(0)
    for table in embeddings:
        table.fill(0)
    for layer in layers:
        layer["norm1"].fill(1)
        layer["norm2"].fill(1)
        for bank in layer["steps"]:
            for weight in bank.values():
                weight.fill(0)
    heads.fill(0)
    hidden.fill(0)
    # The generated 2,0,3 are the only rows that produce the next basis vector.
    # In particular, previous ID zero must select a real audio row, not padding.
    embeddings[0][spec.text_vocab - 1, 0] = 8
    embeddings[1][2, 1] = 8
    embeddings[2][0, 2] = 8
    embeddings[3][3, 3] = 8
    expected = (2, 0, 3, 1)
    for step, token in enumerate(expected):
        heads[step, token, step] = 4
    with DepthChainEngine(projections, embeddings, layers, heads,
                          spec=spec, cfg=cfg) as chain:
        assert chain.run_frame(hidden, spec.text_vocab - 1) is None
        np.testing.assert_array_equal(chain.read_ids(), expected)
        assert chain.read_counter() == spec.steps
        np.testing.assert_array_equal(chain.read_conditioned()[3:4], [8.0])
        chain.reset()
        with pytest.raises(RuntimeError):
            chain.read_ids()
        chain.run_frame(hidden, spec.text_vocab - 1)
        np.testing.assert_array_equal(chain.read_ids(), expected)
        chain.begin_frame(hidden, -1)  # leading zero row of the text table
        chain.step()
        np.testing.assert_array_equal(chain.read_conditioned(),
                                      np.zeros(spec.depth, np.float32))
        assert chain.read_previous_id() == 0


def test_one_frame_input_upload_and_three_fixed_programs_per_step(monkeypatch):
    spec, cfg, projections, embeddings, layers, heads, hidden = fixture()
    with DepthChainEngine(projections, embeddings, layers, heads,
                          spec=spec, cfg=cfg) as chain:
        # Construction has already consumed every resident source.
        projections.fill(np.nan)
        for table in embeddings:
            table.fill(np.nan)
        for layer in layers:
            layer["norm1"].fill(np.nan)
            layer["norm2"].fill(np.nan)
            for bank in layer["steps"]:
                for matrix in bank.values():
                    matrix.fill(np.nan)
        heads.fill(np.nan)
        read, write, run = chain.backend.read, chain.backend.write, chain.backend.run
        reads, writes, runs = [], [], []

        def traced_read(*args, **kwargs):
            reads.append((args, kwargs))
            return read(*args, **kwargs)

        def traced_write(s, address, value):
            writes.append((s, address, np.asarray(value).copy()))
            return write(s, address, value)

        def traced_run(program, *args, **kwargs):
            assert not args and kwargs.get("args") is None
            runs.append(program)
            return run(program, *args, **kwargs)

        monkeypatch.setattr(chain.backend, "read", traced_read)
        monkeypatch.setattr(chain.backend, "write", traced_write)
        monkeypatch.setattr(chain.backend, "run", traced_run)
        chain.begin_frame(hidden, spec.text_vocab - 1)
        begin_writes = writes[:]
        assert len(begin_writes) == 2
        assert {addr for _, addr, _ in begin_writes} == {
            chain.image.io["hidden"], chain.image.io["control"]}
        assert sum(v.nbytes == 4 * spec.hidden for _, _, v in begin_writes) == 1
        assert sum(v.nbytes == 8 for _, _, v in begin_writes) == 1
        assert not reads
        for _ in range(spec.steps):
            chain.step()
            assert len(writes) == len(begin_writes)
            assert not reads
        assert len(runs) == 3 * spec.steps
        assert all(len(program) == 1 for program in runs)
        assert all(runs[3 * i + j][0] is runs[j][0]
                   for i in range(spec.steps) for j in range(3))
        assert len({id(program[0]) for program in runs[:3]}) == 3
        assert all(not program[0].run_args for program in runs[:3])
        assert len(chain.read_ids()) == spec.steps
        assert len(reads) == 1
        assert reads[0][0] == (0, chain.image.io["audio_ids"], 4 * spec.steps)


def unread():
    raise AssertionError("payload was accessed before layout/program preflight")


def test_layout_and_compile_preflight_precede_lazy_payload(monkeypatch):
    spec, cfg, *_ = fixture()
    layout = DepthChainImage(spec, cfg)
    budget = layout.budget()
    json.dumps(budget)
    assert budget["mapped_bytes"] <= 1 << 32
    assert budget["occupied_bytes"] <= budget["mapped_bytes"]
    spans = sorted(budget["spans"], key=lambda span: span["base"])
    assert {"B", "heads", "embeddings", "program", "kv", "scratch", "control"} <= {
        span["region"] for span in spans}
    assert all(0 <= span["base"] < span["base"] + span["bytes"] <= 1 << 32
               for span in spans)
    assert all(left["base"] + left["bytes"] <= right["base"]
               for left, right in zip(spans, spans[1:]))
    assert layout.required_bytes <= cfg.DRAM_BYTES
    assert layout.required_bytes <= 1 << 32
    programs = layout.compile()
    assert len(programs) == 3 and all(not p.run_args for p in programs)
    with pytest.raises(MemoryError):
        DepthChainEngine(unread, unread, unread, unread, spec=spec,
                         cfg=replace(cfg, DRAM_BYTES=layout.required_bytes - 1))

    def fail(self):
        raise CompileError("synthetic depth-chain compile failure")

    monkeypatch.setattr(DepthChainImage, "compile", fail)
    with pytest.raises(CompileError, match="compile failure"):
        DepthChainEngine(unread, unread, unread, unread, spec=spec, cfg=cfg)


def test_lazy_sources_are_loaded_once_after_all_three_programs_compile(monkeypatch):
    spec, cfg, projections, embeddings, layers, heads, hidden = fixture()
    actual_compile = DepthChainImage.compile
    events = []

    def compile_traced(self):
        programs = actual_compile(self)
        assert len(programs) == 3
        events.append("compile")
        return programs

    def source(name, value):
        def load():
            assert events and events[0] == "compile"
            events.append(name)
            return value
        return load

    monkeypatch.setattr(DepthChainImage, "compile", compile_traced)
    with DepthChainEngine(source("projections", projections),
                          source("embeddings", embeddings),
                          source("layers", layers), source("heads", heads),
                          spec=spec, cfg=cfg) as chain:
        chain.run_frame(hidden, spec.text_vocab - 1)
        chain.reset()
        chain.run_frame(hidden, spec.text_vocab - 1)
    assert events[0] == "compile"
    assert sorted(events[1:]) == ["embeddings", "heads", "layers", "projections"]


def test_invalid_frame_does_not_write_and_failed_run_latches_until_reset(monkeypatch):
    spec, cfg, projections, embeddings, layers, heads, hidden = fixture()
    with DepthChainEngine(projections, embeddings, layers, heads,
                          spec=spec, cfg=cfg) as chain:
        write = chain.backend.write
        monkeypatch.setattr(chain.backend, "write", lambda *a: pytest.fail("unexpected write"))
        for bad in (-2, spec.text_vocab, True, 1.5):
            with pytest.raises(ValueError):
                chain.begin_frame(hidden, bad)
        for bad in (hidden[None, :], np.full(spec.hidden, np.nan),
                    np.full(spec.hidden, 1e300)):
            with np.errstate(over="ignore"), pytest.raises(ValueError):
                chain.begin_frame(bad, 0)
        monkeypatch.setattr(chain.backend, "write", write)
        chain.begin_frame(hidden, 0)
        run = chain.backend.run

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic chain execution failure")

        monkeypatch.setattr(chain.backend, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic chain"):
            chain.step()
        monkeypatch.setattr(chain.backend, "run", run)
        with pytest.raises(RuntimeError):
            chain.step()
        with pytest.raises(RuntimeError):
            chain.read_ids()
        chain.reset()
        chain.run_frame(hidden, 0)
        assert len(chain.read_ids()) == spec.steps


def test_unqualified_board_is_refused_before_payload_or_device_access(monkeypatch):
    from opentpu.host import board

    spec, cfg, *_ = fixture()
    monkeypatch.setattr(board, "open_transport", unread)
    monkeypatch.setattr(board, "BoardBackend", unread)
    with pytest.raises(ValueError, match="board|isa|qualified"):
        DepthChainEngine(unread, unread, unread, unread, spec=spec, cfg=cfg,
                         backend="board")
    with pytest.raises(ValueError, match="backend|isa"):
        DepthChainEngine(unread, unread, unread, unread, spec=spec, cfg=cfg,
                         backend=lambda cfg, images: unread())
