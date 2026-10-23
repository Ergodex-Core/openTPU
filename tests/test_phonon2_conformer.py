"""Complete block CPU/ISA checks. No remote code, model downloads or device use."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.asr.phonon2_conformer import BlockShape, compile_conformer_slice, weight_shapes
from opentpu.asr.phonon2_linear import FiveValue
from opentpu.isasim import Machine
from tools.phonon2_reference import conformer_reference, frontend_reference


def synthetic_weights(shape, seed=17):
    rng = np.random.default_rng(seed)
    weights = {}
    for name, (kind, dims) in weight_shapes(shape).items():
        if kind == "five_value":
            low = np.float16(.3 / np.sqrt(dims[1]))
            weights[name] = FiveValue(rng.integers(-1, 2, dims, dtype=np.int8), rng.random(dims) > .5,
                                     np.full(dims[0], low), np.full(dims[0], np.float16(3 * low)))
        elif name.endswith("running_var"):
            weights[name] = rng.uniform(.25, 1.5, dims).astype(np.float32)
        elif "norm" in name and name.endswith("weight"):
            weights[name] = rng.uniform(.7, 1.2, dims).astype(np.float32)
        else:
            weights[name] = rng.normal(0, .15, dims).astype(np.float32)
    return weights


def simulate(compiled, x):
    image = compiled.image.copy()
    image[compiled.input_at:compiled.input_at + x.nbytes] = x.view(np.uint8).ravel()
    machine = Machine(compiled.cfg, [compiled.programs[0]], [image]).run()
    for program in compiled.programs[1:]:
        # No readback or host activation work between phases.
        machine.load([program]).run()
    return {name: machine.slices[0].dram[at:at + x.nbytes].view(np.float32).reshape(x.shape).copy()
            for name, at in compiled.checkpoints.items()}


@pytest.mark.parametrize("frames,valid", [(1, 1), (4, 4), (5, 3), (8, 7)])
def test_complete_block_all_stages_against_independent_cpu_reference(frames, valid):
    shape = BlockShape(128, 256, 4)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=2 << 20)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(82).normal(size=(frames, shape.hidden)).astype(np.float32)
    # Padded hidden rows deliberately nonzero: upstream does not mask residuals.
    compiled = compile_conformer_slice(weights, frames, valid, cfg, shape)
    got = simulate(compiled, x)
    quant = conformer_reference(weights, x, valid, shape, quantize_D=cfg.D)
    dense = conformer_reference(weights, x, valid, shape)
    for name in got:
        np.testing.assert_allclose(got[name], quant[name], rtol=5e-4, atol=1e-4, err_msg=name)
    error = np.linalg.norm(got["output"] - dense["output"]) / np.linalg.norm(dense["output"])
    assert error < .03  # Qualification tolerance, not a transcript-quality claim.
    np.testing.assert_array_equal(got["glu"][valid:], 0)
    if valid < frames:
        assert np.linalg.norm(got["output"][valid:]) > 1
        # Finite-min masked padded queries have uniform attention over stored rows.
        np.testing.assert_allclose(got["attention"][valid:], np.tile(got["v"].mean(0), (frames - valid, 1)), atol=1e-5)
    assert not compiled.image.flags.writeable
    assert compiled.budget()["max_instruction_words"] <= cfg.IMEM_WORDS
    assert compiled.budget()["max_tmem_live_words"] <= cfg.TMEM_WORDS
    assert {op.op for p in compiled.programs for op in p} <= {I.LD, I.ST, I.QACT, I.MM, I.VOP, I.LOOP, I.LI, I.ADDI, I.HALT}


@pytest.mark.parametrize("frames,valid", [(0, 0), (4, 0), (4, 5), (9, 9), (True, 1)])
def test_frame_guard(frames, valid):
    with pytest.raises(ValueError, match="frames"):
        compile_conformer_slice({}, frames, valid, Config())


@pytest.mark.parametrize("change", [{"DRAM_BYTES": 128}, {"IMEM_WORDS": 8}, {"ACT_BLOCKS": 1}, {"S": 2}])
def test_block_budget_fails_closed(change):
    shape = BlockShape(128, 256, 4)
    with pytest.raises(ValueError):
        compile_conformer_slice(synthetic_weights(shape), 4, 3, dataclasses.replace(Config(D=128), **change), shape)


def test_weight_structure_and_batchnorm_variance_guards():
    shape = BlockShape(128, 256, 4)
    weights = synthetic_weights(shape)
    weights["conv.norm.running_var"][0] = -1
    with pytest.raises(ValueError, match="variance"):
        compile_conformer_slice(weights, 4, 3, Config(D=128), shape)
    weights.pop("self_attn.bias_u")
    with pytest.raises(ValueError, match="keys"):
        compile_conformer_slice(weights, 4, 3, Config(D=128), shape)


def test_actual_full_width_block_fits_saved_f2_limits():
    shape = BlockShape()
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=64 << 20)
    compiled = compile_conformer_slice(synthetic_weights(shape), 8, 7, cfg)
    budget = compiled.budget()
    assert len(compiled.programs) == 17
    assert budget["dram_image_bytes"] == 51097600
    assert budget["max_instruction_words"] <= cfg.IMEM_WORDS
    assert budget["max_tmem_live_words"] <= cfg.TMEM_WORDS
    assert budget["max_projection_act_blocks"] == 32


@pytest.mark.parametrize("samples", [0, 1, 319, 480001])
def test_frontend_sample_duration_guard(samples):
    with pytest.raises(ValueError, match="samples"):
        frontend_reference(bytes(2 * samples))


def test_frontend_mask_extra_frame_silence_and_policy():
    for samples in (320, 321, 1600, 1601):
        got = frontend_reference(bytes(2 * samples))
        assert got["stored_frames"] == samples // 160 + 1
        assert got["valid_frames"] == samples // 160
        # Unbiased variance + tiny epsilon amplifies FP32 reduction rounding on
        # silence. Preserve the pinned equations; do not force these rows to zero.
        assert np.isfinite(got["features"]).all()
        np.testing.assert_array_equal(got["features"][-1], 0)
        assert got["attention_mask"][-1] == 0
    for kwargs in ({"dither": 1e-5}, {"sample_rate": 44100}, {"channels": 2}):
        with pytest.raises(ValueError):
            frontend_reference(bytes(640), **kwargs)
    with pytest.raises(ValueError):
        frontend_reference(bytes(641))


def test_frontend_golden_vectors_have_exact_masks_and_audited_stage_tolerances():
    root = Path(__file__).resolve().parents[1] / "docs/phonon2"
    manifest = json.loads((root / "frontend_vectors.json").read_text())
    path = root / "frontend_vectors.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["vector_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        for record in manifest["records"]:
            name = record["name"]
            pcm = arrays[name + ".pcm_s16le"].astype("<i2").tobytes()
            assert hashlib.sha256(pcm).hexdigest() == record["pcm_sha256"]
            got = frontend_reference(pcm)
            np.testing.assert_array_equal(got["attention_mask"], arrays[name + ".attention_mask"])
            np.testing.assert_array_equal(got["preemphasis"], arrays[name + ".preemphasis"])
            np.testing.assert_allclose(got["log_mel"], arrays[name + ".log_mel"], rtol=0, atol=3e-4)
            # Silence is retained as a numerical stress vector, not hidden by a
            # wide global tolerance. Its reduction-order discrepancy is reported.
            if name != "silence_100ms":
                np.testing.assert_allclose(got["features"], arrays[name + ".features"], rtol=0, atol=2e-4)
            else:
                assert record["numpy_max_abs_error"]["features"] > .1
            np.testing.assert_array_equal(got["features"][~got["attention_mask"]], 0)


def test_frontend_max_duration_retains_extra_stored_frame():
    # Do not normalize silence to zero by policy; only enforce length and masks.
    got = frontend_reference(bytes(2 * 480000))
    assert got["stored_frames"] == 3001 and got["valid_frames"] == 3000
    assert got["attention_mask"].sum() == 3000
    assert np.isfinite(got["features"]).all()


def test_optional_installed_torch_checks_independent_dense_block_equations():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from tools.phonon2_torch_audit import conformer_torch
    shape = BlockShape(128, 256, 4)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(9).normal(size=(5, 128)).astype(np.float32)
    a = conformer_torch(weights, x, 3, shape)
    b = conformer_reference(weights, x, 3, shape)["output"]
    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
