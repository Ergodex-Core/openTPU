"""Bounded resident Mimi RVQ lookup/projection on EC2's ISA simulator.

These are synthetic component tests, not codec, PCM or physical-board claims.
Only the EC2 runner executes the model arithmetic below.
"""
import json

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_depth_chain import DepthChainEngine, DepthChainSpec
from opentpu.llm.personaplex_mimi import MimiRVQImage, MimiRVQSpec, normalize_codebooks


NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def fixture():
    chain_spec = DepthChainSpec(hidden=32, depth=32, n_heads=1, ffn=64,
                                steps=4, layers=1, text_vocab=7,
                                audio_vocab=5, head_vocab=4)
    mimi_spec = MimiRVQSpec(codebooks=3, vocab=4, dim=32, latent=64)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 32)
    projections = np.zeros((4, 32, 32), np.float32)
    embeddings = (np.zeros((7, 32), np.float32),) + tuple(
        np.zeros((5, 32), np.float32) for _ in range(3))
    weights = []
    for _ in range(chain_spec.layers):
        banks = []
        for _ in range(chain_spec.steps):
            bank = {name: np.zeros((32, 32), np.float32) for name in NAMES[:4]}
            bank.update(wg=np.zeros((64, 32), np.float32),
                        wu=np.zeros((64, 32), np.float32),
                        wd=np.zeros((32, 64), np.float32))
            banks.append(bank)
        weights.append({"norm1": np.ones(32, np.float32),
                        "norm2": np.ones(32, np.float32), "steps": banks})
    heads = np.zeros((4, 4, 32), np.float32)
    # Greedy sequence 2,0,3,1; codebook 1's ID zero is a real centroid row.
    embeddings[0][6, 0] = 8
    embeddings[1][2, 1] = 8
    embeddings[2][0, 2] = 8
    embeddings[3][3, 3] = 8
    for step, token in enumerate((2, 0, 3, 1)):
        heads[step, token, step] = 4
    rng = np.random.default_rng(9073)
    tables = rng.normal(0, 0.7, (3, 4, 32)).astype(np.float32)
    mimi_projections = rng.normal(0, 0.2, (2, 64, 32)).astype(np.float32)
    hidden = np.zeros(chain_spec.hidden, np.float32)
    return (chain_spec, mimi_spec, cfg, projections, embeddings, weights,
            heads, tables, mimi_projections, hidden)


def engine_args(parts):
    chain_spec, mimi_spec, cfg, projections, embeddings, layers, heads, tables, matrix, _ = parts
    return (projections, embeddings, layers, heads), {
        "spec": chain_spec, "cfg": cfg, "mimi_tables": tables,
        "mimi_projections": matrix, "mimi_spec": mimi_spec}


def reference_projection(value, matrix, width):
    activation = Q.fake_act(value, width)
    weight = Q.fake_w(matrix, "int8", width)
    offsets, window, D, transposed = weight.mx
    assert D == width and not transposed
    return Q.mx_matmul(activation, np.asarray(weight), offsets, window, width)


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def test_offline_centroid_normalization_uses_source_clamp_and_rejects_bad_statistics():
    sums = np.array([
        [[2e-5, -3e-5], [4e-5, 7e-5], [6.0, -2.0]],
        [[0.0, 1e-5], [8.0, -4.0], [3.0, 5.0]],
    ], np.float32)
    usages = np.array([[0.0, 5e-6, 2.0], [1e-5, 4.0, 8.0]], np.float32)
    before_sums, before_usages = sums.copy(), usages.copy()
    expected = sums / np.maximum(usages, np.float32(1e-5))[..., None]
    actual = normalize_codebooks(sums, usages)
    assert actual.dtype == np.float32 and actual.shape == sums.shape
    np.testing.assert_array_equal(bits(actual), bits(expected))
    np.testing.assert_allclose(actual[0, 0], [2.0, -3.0], rtol=1e-6)
    np.testing.assert_array_equal(bits(sums), bits(before_sums))
    np.testing.assert_array_equal(bits(usages), bits(before_usages))
    for bad_sums, bad_usages in (
            (sums[0], usages), (sums, usages[:, :2]),
            (np.where(np.arange(sums.size).reshape(sums.shape) == 0, np.nan, sums), usages),
            (sums, np.where(np.arange(usages.size).reshape(usages.shape) == 0,
                            np.inf, usages))):
        with pytest.raises(ValueError):
            normalize_codebooks(bad_sums, bad_usages)


def test_resident_ids_select_source_order_centroids_and_quantized_projections():
    parts = fixture()
    chain_spec, mimi_spec, cfg, _, _, _, _, tables, matrix, hidden = parts
    args, kwargs = engine_args(parts)
    with DepthChainEngine(*args, **kwargs) as chain:
        chain.run_frame(hidden, chain_spec.text_vocab - 1)
        with pytest.raises(RuntimeError):
            chain.read_mimi()
        assert chain.run_mimi() is None
        ids = chain.read_ids()
        np.testing.assert_array_equal(ids, (2, 0, 3, 1))
        semantic = np.zeros(mimi_spec.dim, np.float32)
        acoustic = np.zeros(mimi_spec.dim, np.float32)
        semantic = np.add(semantic, tables[0, ids[0]], dtype=np.float32)
        for codebook in range(1, mimi_spec.codebooks):
            acoustic = np.add(acoustic, tables[codebook, ids[codebook]], dtype=np.float32)
        np.testing.assert_array_equal(bits(chain.read_mimi("semantic")), bits(semantic))
        np.testing.assert_array_equal(bits(chain.read_mimi("acoustic")), bits(acoustic))
        s_projected = chain.read_mimi("semantic_projected")
        a_projected = chain.read_mimi("acoustic_projected")
        np.testing.assert_allclose(s_projected,
                                   reference_projection(semantic, matrix[0], cfg.D),
                                   rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(a_projected,
                                   reference_projection(acoustic, matrix[1], cfg.D),
                                   rtol=2e-5, atol=2e-5)
        expected = np.add(s_projected, a_projected, dtype=np.float32)
        np.testing.assert_array_equal(bits(chain.read_mimi()), bits(expected))
        assert chain.read_mimi().shape == (mimi_spec.latent,)


def test_mimi_run_has_one_fixed_program_no_feedback_transfer_and_resident_sources(monkeypatch):
    parts = fixture()
    chain_spec, mimi_spec, _, _, embeddings, layers, heads, tables, matrix, hidden = parts
    args, kwargs = engine_args(parts)
    with DepthChainEngine(*args, **kwargs) as chain:
        chain.run_frame(hidden, chain_spec.text_vocab - 1)
        # These objects were consumed during offline packing. A Mimi run must
        # never revisit host centroids, matrices, or source depth/head weights.
        tables.fill(np.nan)
        matrix.fill(np.nan)
        for table in embeddings:
            table.fill(np.nan)
        for layer in layers:
            for bank in layer["steps"]:
                for value in bank.values():
                    value.fill(np.nan)
        heads.fill(np.nan)
        read, write, run = chain.backend.read, chain.backend.write, chain.backend.run
        reads, writes, programs = [], [], []

        def traced_read(*args):
            reads.append(args)
            return read(*args)

        def traced_write(s, address, value):
            writes.append((s, address, np.asarray(value).copy()))
            return write(s, address, value)

        def traced_run(program, *args, **kwargs):
            assert not args and kwargs.get("args") is None
            programs.append(program)
            return run(program, *args, **kwargs)

        monkeypatch.setattr(chain.backend, "read", traced_read)
        monkeypatch.setattr(chain.backend, "write", traced_write)
        monkeypatch.setattr(chain.backend, "run", traced_run)
        assert chain.run_mimi() is None
        assert not reads and not writes
        assert len(programs) == 1 and len(programs[0]) == 1
        assert not programs[0][0].run_args
        assert np.isfinite(chain.read_mimi()).all()
        assert len(reads) == 1 and reads[0] == (0, chain.image.mimi.io["output"],
                                                4 * mimi_spec.latent)


def test_mimi_readiness_reset_replay_and_execution_failure_latch(monkeypatch):
    parts = fixture()
    chain_spec, _, _, _, _, _, _, _, _, hidden = parts
    args, kwargs = engine_args(parts)
    with DepthChainEngine(*args, **kwargs) as chain:
        with pytest.raises(RuntimeError):
            chain.run_mimi()
        chain.begin_frame(hidden, chain_spec.text_vocab - 1)
        chain.step()
        with pytest.raises(RuntimeError):
            chain.run_mimi()
        for _ in range(chain_spec.steps - 1):
            chain.step()
        chain.run_mimi()
        first = chain.read_mimi().copy()
        chain.begin_frame(hidden, chain_spec.text_vocab - 1)
        with pytest.raises(RuntimeError):
            chain.read_mimi()
        for _ in range(chain_spec.steps):
            chain.step()
        chain.run_mimi()
        np.testing.assert_array_equal(bits(chain.read_mimi()), bits(first))
        chain.reset()
        with pytest.raises(RuntimeError):
            chain.read_mimi()
        chain.run_frame(hidden, chain_spec.text_vocab - 1)
        run = chain.backend.run

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic Mimi execution failure")

        monkeypatch.setattr(chain.backend, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic Mimi"):
            chain.run_mimi()
        monkeypatch.setattr(chain.backend, "run", run)
        with pytest.raises(RuntimeError):
            chain.run_mimi()
        with pytest.raises(RuntimeError):
            chain.read_mimi()
        chain.reset()
        chain.run_frame(hidden, chain_spec.text_vocab - 1)
        chain.run_mimi()
        np.testing.assert_array_equal(bits(chain.read_mimi()), bits(first))


def test_standalone_rvq_reads_only_first_eight_of_sixteen_ids_and_guards_vocab():
    _, _, cfg, *_ = fixture()
    spec = MimiRVQSpec(codebooks=8, vocab=5, dim=32, latent=64)
    rng = np.random.default_rng(1907)
    tables = rng.normal(0, 0.4, (8, 5, 32)).astype(np.float32)
    matrix = rng.normal(0, 0.15, (2, 64, 32)).astype(np.float32)
    image = MimiRVQImage(spec, cfg, audio_ids_at=64, counter_at=0, expected_steps=16)
    programs = image.compile()
    assert len(programs) == 1 and not programs[0].run_args
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, tables, matrix)
    backend.write(0, 0, np.array([16], np.float32))
    chosen = np.array([4, 0, 3, 1, 2, 4, 0, 3], np.float32)
    with_tail = np.concatenate((chosen, np.full(8, np.nan, np.float32)))
    backend.write(0, 64, with_tail)
    backend.run(programs)
    first = backend.read(0, image.io["output"], 4 * spec.latent).view(np.float32).copy()
    assert np.isfinite(first).all()
    backend.write(0, 64, np.concatenate((chosen, np.full(8, 1e6, np.float32))))
    backend.run(programs)
    second = backend.read(0, image.io["output"], 4 * spec.latent).view(np.float32).copy()
    np.testing.assert_array_equal(bits(first), bits(second))
    for bad in (spec.vocab, 1.5, np.nan, np.inf):
        with_tail[0] = bad
        backend.write(0, 64, with_tail)
        with pytest.raises(RuntimeError, match="bound|outside"):
            backend.run(programs)


def unread():
    raise AssertionError("Mimi payload read before layout/program preflight")


def test_mimi_layout_budget_and_compile_precede_lazy_payload(monkeypatch):
    parts = fixture()
    _, spec, cfg, *_ = parts
    layout = MimiRVQImage(spec, cfg, audio_ids_at=64, counter_at=0, expected_steps=16)
    budget = layout.budget()
    json.dumps(budget)
    assert budget["required_bytes"] <= cfg.DRAM_BYTES
    assert len(layout.compile()) == 1
    with pytest.raises(MemoryError):
        MimiRVQImage(spec, cfg, audio_ids_at=64, counter_at=0, expected_steps=16,
                     limit=0x5D850000 + 64)

    def fail(self):
        raise CompileError("synthetic Mimi program capacity failure")

    monkeypatch.setattr(MimiRVQImage, "compile", fail)
    chain_spec, _, _, projections, embeddings, layers, heads, _, _, _ = parts
    with pytest.raises(CompileError, match="Mimi program capacity"):
        DepthChainEngine(projections, embeddings, layers, heads,
                         spec=chain_spec, cfg=cfg,
                         mimi_tables=unread, mimi_projections=unread,
                         mimi_spec=spec)


@pytest.mark.parametrize("omit", ["tables", "projections"])
def test_mimi_requires_both_payloads_before_any_source_read(omit):
    parts = fixture()
    chain_spec, spec, cfg, *_ = parts
    options = {"spec": chain_spec, "cfg": cfg, "mimi_spec": spec,
               "mimi_tables": unread, "mimi_projections": unread}
    options["mimi_" + omit] = None
    with pytest.raises(ValueError):
        DepthChainEngine(unread, unread, unread, unread, **options)


@pytest.mark.parametrize("bad", ["table_shape", "table_nan", "matrix_shape", "matrix_inf"])
def test_bad_mimi_payload_is_rejected(bad):
    _, spec, cfg, _, _, _, _, tables, matrix, _ = fixture()
    if bad == "table_shape":
        tables = tables[:-1]
    elif bad == "table_nan":
        tables[1, 0, 2] = np.nan
    elif bad == "matrix_shape":
        matrix = matrix[:, :-1]
    else:
        matrix[1, 3, 4] = np.inf
    image = MimiRVQImage(spec, cfg, audio_ids_at=64, counter_at=0, expected_steps=16)

    class DummyBackend:
        def write(self, *args):
            return None

    with pytest.raises(ValueError):
        image.upload(DummyBackend(), tables, matrix)
