"""Optional resident RVQ projection precision on the ISA simulator.

The builder runs these NumPy/source-equation checks. Local Mac verification is
AST-only and must not import the model or NumPy.
"""
import dataclasses
import hashlib

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_mimi import MimiRVQImage, MimiRVQSpec


def _config():
    return Config(S=1, D=64, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                  TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                  DRAM_BYTES=1 << 23)


def _image(spec, cfg):
    return MimiRVQImage(spec, cfg, audio_ids_at=64, counter_at=0,
                        expected_steps=16, base=1 << 20, limit=1 << 23)


def _read(backend, image, name):
    width = image.spec.dim if name in ("semantic", "acoustic") else image.spec.latent
    return backend.read(0, image.io[name], 4 * width).view(np.float32).copy()


def _source_sums(tables, ids):
    semantic = np.add(np.zeros(tables.shape[-1], np.float32),
                      tables[0, ids[0]], dtype=np.float32)
    acoustic = np.zeros(tables.shape[-1], np.float32)
    for codebook in range(1, len(ids)):
        acoustic = np.add(acoustic, tables[codebook, ids[codebook]], dtype=np.float32)
    return semantic, acoustic


def test_fp32_256_to_512_resident_projection_matches_source_equations(monkeypatch):
    spec = MimiRVQSpec(codebooks=8, vocab=5, dim=256, latent=512,
                       projection_precision="fp32")
    cfg = _config()
    image = _image(spec, cfg)
    rng = np.random.default_rng(6731)
    tables = rng.normal(0, 0.35, (8, 5, 256)).astype(np.float32)
    weights = rng.normal(0, 0.08, (2, 512, 256)).astype(np.float32)
    ids = np.array([4, 0, 3, 1, 2, 4, 0, 3], np.int32)
    semantic, acoustic = _source_sums(tables, ids)
    expected_semantic = weights[0] @ semantic
    expected_acoustic = weights[1] @ acoustic

    programs = image.compile()
    assert len(programs) == 1 and not programs[0].run_args
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, tables, weights)
    tables.fill(np.nan)
    weights.fill(np.nan)
    backend.write(0, image.counter_at, np.array([16], np.float32))
    backend.write(0, image.audio_ids_at,
                  np.concatenate((ids.astype(np.float32), np.full(8, np.nan, np.float32))))

    # The completed depth pass and the RVQ program exchange only resident DRAM.
    read, write = backend.read, backend.write
    transfers = []

    def traced_read(*args):
        transfers.append(("read", args))
        return read(*args)

    def traced_write(*args):
        transfers.append(("write", args))
        return write(*args)

    monkeypatch.setattr(backend, "read", traced_read)
    monkeypatch.setattr(backend, "write", traced_write)
    backend.run(programs)
    assert transfers == []
    monkeypatch.setattr(backend, "read", read)
    monkeypatch.setattr(backend, "write", write)

    np.testing.assert_array_equal(_read(backend, image, "semantic"), semantic)
    np.testing.assert_array_equal(_read(backend, image, "acoustic"), acoustic)
    first = _read(backend, image, "semantic_projected")
    rest = _read(backend, image, "acoustic_projected")
    np.testing.assert_allclose(first, expected_semantic, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(rest, expected_acoustic, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(_read(backend, image, "output"),
                                  np.add(first, rest, dtype=np.float32))
    assert np.isfinite(first).all() and np.isfinite(rest).all()

    # Clearing only resident state and replaying the same IDs is deterministic.
    initial = _read(backend, image, "output")
    for address in image.io.values():
        width = spec.dim if address in (image.io["semantic"], image.io["acoustic"]) else spec.latent
        backend.write(0, address, np.zeros(width, np.float32))
    backend.run(programs)
    np.testing.assert_array_equal(_read(backend, image, "output"), initial)

    # Range and completion guards still apply to the FP32 path.
    backend.write(0, image.counter_at, np.array([15], np.float32))
    with pytest.raises(RuntimeError, match="bound|outside"):
        backend.run(programs)
    backend.write(0, image.counter_at, np.array([16], np.float32))
    for bad in (float(spec.vocab), 1.5, np.nan, np.inf):
        invalid = ids.astype(np.float32)
        invalid[0] = bad
        backend.write(0, image.audio_ids_at, invalid)
        with pytest.raises(RuntimeError, match="bound|outside"):
            backend.run(programs)


def test_precision_modes_keep_int8_default_layout_and_upload_contract():
    cfg = _config()
    default = MimiRVQSpec(codebooks=3, vocab=5, dim=64, latent=128)
    explicit = dataclasses.replace(default, projection_precision="int8")
    legacy = _image(default, cfg)
    selected = _image(explicit, cfg)
    assert default.projection_precision == "int8"
    assert legacy.spans == selected.spans
    assert legacy.projection_offsets == selected.projection_offsets
    assert legacy.weight_bytes == selected.weight_bytes
    assert [p["name"] for p in legacy.spans if "projection" in p["name"]] == [
        "mimi_projection0_codes", "mimi_projection0_scales",
        "mimi_projection1_codes", "mimi_projection1_scales"]
    assert len(legacy.compile()) == len(selected.compile()) == 1

    rng = np.random.default_rng(911)
    tables = rng.normal(0, 0.3, (3, 5, 64)).astype(np.float32)
    weights = rng.normal(0, 0.1, (2, 128, 64)).astype(np.float32)
    default_backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    explicit_backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    legacy.upload(default_backend, tables, weights)
    selected.upload(explicit_backend, tables, weights)
    assert legacy.payload_hashes == selected.payload_hashes
    for span in legacy.spans:
        if span["kind"] == "immutable":
            at, size = span["base"], span["bytes"]
            np.testing.assert_array_equal(default_backend.read(0, at, size),
                                          explicit_backend.read(0, at, size))

    fp32 = _image(dataclasses.replace(default, projection_precision="fp32"), cfg)
    assert [p["name"] for p in fp32.spans if "projection" in p["name"]] == [
        "mimi_projection0_fp32", "mimi_projection1_fp32"]
    assert fp32.weight_bytes == 4 * 3 * 5 * 64 + 8 * 128 * 64
    assert fp32.budget()["payload_bytes"] == sum(
        p["bytes"] for p in fp32.spans if p["kind"] == "immutable")
    fp_backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    fp32.upload(fp_backend, tables, weights)
    for index, at in enumerate(fp32.projection_offsets):
        payload = np.ascontiguousarray(weights[index]).view(np.uint8).reshape(-1)
        assert fp32.payload_hashes[f"projection{index}_fp32"] == hashlib.sha256(
            memoryview(payload)).hexdigest()
        np.testing.assert_array_equal(fp_backend.read(0, at, payload.size), payload)


@pytest.mark.parametrize("precision", [None, "f32", "INT8", 8])
def test_invalid_rvq_projection_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="projection_precision"):
        MimiRVQSpec(projection_precision=precision)
