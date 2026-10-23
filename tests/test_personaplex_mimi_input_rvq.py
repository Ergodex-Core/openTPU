"""Synthetic source-equation/ISA checks for resident Mimi input RVQ.

Run on the CPU ISA builder only. Native torch.cdist IDs and margins require a
separate pinned-checkpoint qualification before integration.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_mimi_input_rvq import MimiInputRVQImage, MimiInputRVQSpec


def _case():
    spec = MimiInputRVQSpec(hidden=32, dim=32, vocab=130, codebooks=4)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 25)
    image = MimiInputRVQImage(spec, cfg, input_at=0x1000, tables_at=0x10000,
        state_base=0x30000, state_limit=0x100000,
        weight_base=0x100000, weight_limit=0x2000000)
    x = np.arange(1, 33, dtype=np.float32) / 32
    frames = np.stack((x, x * np.float32(1.1))).astype(np.float32)
    weights = np.zeros((2, 32, 32), np.float32)
    weights[0] = np.eye(32, dtype=np.float32)
    weights[1] = np.eye(32, dtype=np.float32) * np.float32(2.0)
    tables = np.full((4, 130, 32), 100.0, np.float32)
    tables[0, 5] = x
    tables[0, 70] = x  # Exact tie across 32-row chunks: first ID wins.
    tables[1, 66] = x * np.float32(0.5)
    tables[2, 3] = x * np.float32(0.75)
    tables[3, 129] = x * np.float32(0.75)
    return spec, cfg, image, frames, weights, tables


def _reference(x, weights, tables):
    projections = [weights[branch] @ x for branch in range(2)]
    semantic, acoustic = (value.copy() for value in projections)
    ids, distances, before, selected, after = [], [], [], [], []
    for codebook, table in enumerate(tables):
        residual = semantic if codebook == 0 else acoustic
        before.append(residual.copy())
        scores = np.sum((table - residual[None, :]) ** 2, axis=1,
                        dtype=np.float32)
        token = int(np.argmin(scores))
        ids.append(token)
        distances.append(scores)
        selected.append(table[token].copy())
        if codebook == 0:
            semantic = np.subtract(semantic, table[token], dtype=np.float32)
        else:
            acoustic = np.subtract(acoustic, table[token], dtype=np.float32)
        after.append((semantic if codebook == 0 else acoustic).copy())
    return (np.asarray(ids, np.float32), projections, semantic, acoustic,
            distances, np.stack(before), np.stack(selected), np.stack(after))


def _read(backend, address, count):
    return backend.read(0, address, 4 * count).view(np.float32).copy()


def test_source_order_residuals_first_tie_and_resident_reset(monkeypatch):
    spec, cfg, image, frames, weights, tables = _case()
    programs = image.compile()
    assert len(programs) == 1 + spec.codebooks
    assert all(not program.run_args for program in programs)
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    backend.write(0, image.tables_at, tables)
    image.upload(backend, weights)
    expected = [_reference(frame, weights, tables) for frame in frames]
    assert tuple(expected[0][0].astype(int)) == (5, 66, 3, 129)
    # After upload only resident copies can affect the device result.
    weights.fill(np.nan)
    tables.fill(np.nan)

    def stream():
        observations = []
        for frame, source in enumerate(frames):
            backend.write(0, image.input_at, source.reshape(1, -1))
            read, write = backend.read, backend.write
            transfers = {"reads": 0, "writes": 0}

            def counted_read(*args):
                transfers["reads"] += 1
                return read(*args)

            def counted_write(*args):
                transfers["writes"] += 1
                return write(*args)

            with monkeypatch.context() as patch:
                patch.setattr(backend, "read", counted_read)
                patch.setattr(backend, "write", counted_write)
                for program in programs:
                    backend.run([program])
            assert transfers == {"reads": 0, "writes": 0}
            ids = _read(backend, image.io["output_ids"], spec.codebooks)
            semantic = _read(backend, image.io["semantic_residual"], spec.dim)
            acoustic = _read(backend, image.io["acoustic_residual"], spec.dim)
            projected = (_read(backend, image.io["semantic_projected"], spec.dim),
                         _read(backend, image.io["acoustic_projected"], spec.dim))
            assert _read(backend, image.io["counter"], 1)[0] == spec.codebooks
            (ref_ids, ref_projected, ref_semantic, ref_acoustic, ref_distances,
             ref_before, ref_selected, ref_after) = expected[frame]
            np.testing.assert_array_equal(ids, ref_ids)
            for got, reference in zip(projected, ref_projected):
                np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(semantic, ref_semantic, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(acoustic, ref_acoustic, rtol=1e-5, atol=1e-5)
            measured_distances = _read(backend, image.io["distances"],
                                       spec.codebooks * spec.vocab).reshape(
                                           spec.codebooks, spec.vocab)
            for codebook, reference in enumerate(ref_distances):
                np.testing.assert_allclose(measured_distances[codebook], reference,
                                           rtol=1e-5, atol=1e-4)
            for name, reference in (("residual_before", ref_before),
                                    ("selected_centroids", ref_selected),
                                    ("residual_after", ref_after)):
                measured = _read(backend, image.io[name], spec.codebooks * spec.dim).reshape(
                    spec.codebooks, spec.dim)
                np.testing.assert_allclose(measured, reference, rtol=1e-5, atol=1e-5)
            observations.append((ids, semantic, acoustic))
        return observations

    first = stream()
    image.reset(backend)
    for name in ("residual_before", "selected_centroids", "residual_after"):
        assert not np.any(_read(backend, image.io[name], spec.codebooks * spec.dim))
    second = stream()
    for before, after in zip(first, second):
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
    # A search program cannot skip the resident source-order counter.
    backend.run([programs[0]])
    with pytest.raises(RuntimeError, match="bound|outside"):
        backend.run([programs[2]])


def test_shared_table_span_is_exact_and_not_reallocated():
    spec, cfg, _, _, _, _ = _case()
    # Layout validation uses the production address ceiling; only the separate
    # 32 MiB ISA fixture allocates DRAM, so this test never reserves 4 GiB.
    cfg = replace(cfg, DRAM_BYTES=1 << 32)
    shared = dict(name="mimi_tables", base=0x10000,
                  bytes=4 * spec.codebooks * spec.vocab * spec.dim,
                  region="mimi", kind="immutable", alignment=128)
    ds = SimpleNamespace(cfg=cfg, spec=SimpleNamespace(hidden=spec.hidden),
        io={"output": 0x1000}, state_end=0x30000, input_state_limit=0x100000,
        weight_end=0x100000, prefix_spans=[shared],
        spans=[dict(name="downsample_program", base=0x200000, bytes=4096,
                    region="program", kind="program", alignment=128)])
    image = MimiInputRVQImage.after_downsample(ds, spec)
    assert image.tables_at == shared["base"]
    assert len(image.program_slots) == 1 + spec.codebooks
    assert all(span["name"] != "mimi_tables" for span in image.spans)
    assert image.budget()["shared_tables_bytes"] == shared["bytes"]
    broken = dict(shared, bytes=shared["bytes"] - 4)
    ds.prefix_spans = [broken]
    with pytest.raises(ValueError, match="exact existing frozen mimi_tables"):
        MimiInputRVQImage.after_downsample(ds, spec)


@pytest.mark.parametrize("field,value", [
    ("hidden", 0), ("dim", 0), ("vocab", 0), ("codebooks", 1),
    ("hidden", True), ("vocab", 1 << 24)])
def test_bad_spec_rejected(field, value):
    values = dict(hidden=32, dim=32, vocab=130, codebooks=4)
    values[field] = value
    with pytest.raises(ValueError):
        MimiInputRVQSpec(**values)
