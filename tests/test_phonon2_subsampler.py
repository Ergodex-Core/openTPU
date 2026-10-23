"""Complete learned subsampler tests; no device or remote runtime is required."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.asr.phonon2_int6 import Int6Weight, decode_int6
from opentpu.asr.phonon2_subsampler import SubsampleShape, compile_subsampler, lengths, weight_shapes
from opentpu.isasim import Machine
from tools.phonon2_subsample_reference import subsampler_reference, subsampler_torch


def synthetic_weights(shape=SubsampleShape(8, 16), seed=20261023):
    rng = np.random.default_rng(seed)
    result = {}
    for name, (kind, dims) in weight_shapes(shape).items():
        if kind == "int6":
            columns = int(np.prod(dims[1:]))
            result[name] = Int6Weight(rng.integers(-32, 32, (dims[0], columns), dtype=np.int8),
                                     np.full(dims[0], np.float16(.05 / np.sqrt(columns))))
        else:
            result[name] = rng.normal(0, .2, dims).astype(np.float16).astype(np.float32)
    return result


def snapshot(machine, compiled):
    memory = machine.slices[0].dram
    return {name: np.ndarray(shape=spec["shape"], dtype="<f4", buffer=memory, offset=int(spec["offset"]) if isinstance(spec["offset"], int) else spec["offset"].static(),
                            strides=tuple(4 * s for s in spec["strides"])).copy()
            for name, spec in compiled.checkpoints.items()}


def run(compiled, features, *, machine=None, poison=False):
    if machine is None:
        image = compiled.image.copy()
        if poison:
            start = (compiled.input_at + features.nbytes + 127) // 128 * 128
            image[start:] = 255  # NaN halos/tails must be overwritten by device reset.
        machine = Machine(compiled.cfg, [compiled.programs[0]], [image])
    else:
        machine.load([compiled.programs[0]])
    machine.slices[0].dram[compiled.input_at:compiled.input_at + features.nbytes] = features.view(np.uint8).ravel()
    machine.run()
    for p in compiled.programs[1:]:
        machine.load([p]).run()
    return snapshot(machine, compiled), machine


def test_int6_external_golden_little_endian_sign_bias_and_row_scales():
    # Four six-bit values [0,31,32,63], followed by FP16 [.5,.25].
    decoded = decode_int6(bytes.fromhex("c007fe00380034"), (2, 2))
    np.testing.assert_array_equal(decoded.q, [[-32, -1], [0, 31]])
    np.testing.assert_array_equal(decoded.dense(), [[-16, -.5], [0, 7.75]])
    assert not decoded.q.flags.writeable and not decoded.scale.flags.writeable


def test_int6_packing_crosses_rows_and_ignores_global_final_padding():
    decoded = decode_int6(bytes.fromhex("c00f065f08fc00380034"), (2, 3))
    np.testing.assert_array_equal(decoded.q, [[-32, 31, 0], [-31, -1, 1]])


@pytest.mark.parametrize("blob", [b"", bytes(6), bytes(8)])
def test_int6_record_extent_guards(blob):
    with pytest.raises(ValueError, match="length"):
        decode_int6(blob, (2, 2))


@pytest.mark.parametrize("q,scale", [(32, .5), (-33, .5), (0, .1), (0, np.nan), (0, -1), (0, np.inf)])
def test_int6_invalid_codes_scales_are_rejected(q, scale):
    with pytest.raises(ValueError):
        Int6Weight(np.full((2, 2), q), np.full(2, scale))


@pytest.mark.parametrize("stored,valid", [(1, 1), (2, 1), (3, 2), (7, 6), (8, 7), (9, 8), (17, 16), (57, 56)])
def test_complete_subsampler_boundaries_masks_bias_and_flatten(stored, valid):
    shape = SubsampleShape(8, 16)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16, DRAM_BYTES=4 << 20,
                 TMEM_WORDS=65536, IMEM_WORDS=32768)
    weights = synthetic_weights(shape)
    # Nonzero masked input deliberately tests exactly what the official module
    # does: it masks convolution outputs, not its feature argument in advance.
    x = np.random.default_rng(42).normal(size=(stored, 128)).astype(np.float32)
    compiled = compile_subsampler(weights, stored, valid, cfg, shape)
    got, _ = run(compiled, x, poison=True)
    expected = subsampler_reference(weights, x, valid, shape, quantize_D=cfg.D)
    for name in got:
        np.testing.assert_allclose(got[name], expected[name], rtol=5e-4, atol=2e-5, err_msg=name)
    for name, stage in (("conv0", 1), ("depthwise2", 2), ("pointwise3", 2), ("depthwise5", 3), ("pointwise6", 3)):
        np.testing.assert_array_equal(got[name][compiled.valid_lengths[stage]:], 0)
    tail = got["output"][compiled.valid_lengths[3]:]
    if len(tail):
        np.testing.assert_array_equal(tail, np.broadcast_to(weights["linear.bias"], tail.shape))
        assert np.linalg.norm(tail) > 0
    np.testing.assert_array_equal(got["flat"][:, :shape.channels * 16], got["pointwise6"].transpose(0, 2, 1).reshape(len(got["flat"]), -1))
    assert {op.op for p in compiled.programs for op in p} <= {I.NOP, I.LD, I.ST, I.VOP, I.MM, I.QACT, I.LOOP, I.LI, I.ADDI, I.HALT}
    assert not compiled.image.flags.writeable


def test_repeated_execution_resets_halos_masked_cells_and_outputs():
    shape = SubsampleShape(8, 16)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20)
    weights = synthetic_weights(shape)
    long = compile_subsampler(weights, 17, 17, cfg, shape)
    short = compile_subsampler(weights, 17, 1, cfg, shape)
    rng = np.random.default_rng(53)
    _, machine = run(long, rng.normal(size=(17, 128)).astype(np.float32))
    x = rng.normal(size=(17, 128)).astype(np.float32)
    again, _ = run(short, x, machine=machine)
    fresh, _ = run(short, x, poison=True)
    for name in fresh:
        np.testing.assert_array_equal(again[name], fresh[name], err_msg=name)


def test_qualification_transfer_accepts_transposed_feature_view():
    from tools.phonon2_qualify_subsampler import execute, snapshots
    shape = SubsampleShape(8, 16)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(91).normal(size=(128, 9)).astype(np.float32).T
    assert not x.flags.c_contiguous
    compiled = compile_subsampler(weights, 9, 8, cfg, shape)
    actual = snapshots(execute(compiled, x), compiled)
    expected = subsampler_reference(weights, x, 8, shape, quantize_D=cfg.D)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=5e-4, atol=2e-5, err_msg=name)


def test_odd_channels_outputs_and_padded_act_projection_tails():
    shape = SubsampleShape(3, 7)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(64).normal(size=(9, 128)).astype(np.float32)
    compiled = compile_subsampler(weights, 9, 8, cfg, shape)
    actual, _ = run(compiled, x, poison=True)
    expected = subsampler_reference(weights, x, 8, shape, quantize_D=32)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=5e-4, atol=2e-5, err_msg=name)
    np.testing.assert_array_equal(actual["flat"][:, 48:], 0)


def test_official_torch_layout_and_mask_order_agree_with_independent_numpy():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    shape = SubsampleShape(8, 16)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(55).normal(size=(17, 128)).astype(np.float32)
    a, b = subsampler_torch(weights, x, 16, shape), subsampler_reference(weights, x, 16, shape)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=3e-5, atol=2e-6, err_msg=name)


@pytest.mark.parametrize("frames,valid", [(0, 0), (2, 0), (2, 3), (3002, 3001), (True, 1)])
def test_length_guards(frames, valid):
    with pytest.raises(ValueError):
        lengths(frames, valid)


@pytest.mark.parametrize("change", [{"DRAM_BYTES": 128}, {"IMEM_WORDS": 8}, {"ACT_BLOCKS": 1}, {"S": 2}])
def test_resource_guards(change):
    shape = SubsampleShape(8, 16)
    with pytest.raises(ValueError):
        compile_subsampler(synthetic_weights(shape), 9, 8, dataclasses.replace(Config(D=32), **change), shape)


def test_full_30_second_dimensions_compile_with_existing_f2_limits():
    shape = SubsampleShape()
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=256 << 20)
    compiled = compile_subsampler(synthetic_weights(shape), 3001, 3000, cfg)
    assert compiled.stored_lengths == (3001, 1501, 751, 376)
    assert compiled.valid_lengths == (3000, 1500, 750, 375)
    assert compiled.budget()["max_instruction_words"] <= 32768
    assert compiled.budget()["max_tmem_live_words"] <= 65536
    assert len(compiled.programs) == 9


def test_attributed_real_speech_golden_masks_shapes_and_frontend_parity():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from tools.phonon2_torch_audit import frontend_torch
    root = Path(__file__).resolve().parents[1] / "docs/phonon2"
    manifest = json.loads((root / "subsampler_speech_golden.json").read_text())
    provenance = json.loads((root / "speech_fixture.json").read_text())
    pcm = (root / "speech_1688_crop.pcm").read_bytes()
    assert hashlib.sha256(pcm).hexdigest() == provenance["pcm_sha256"]
    assert provenance["license"] == "CC-BY-4.0" and provenance["qualification_only"]
    path = root / "subsampler_speech_golden.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    got = frontend_torch(pcm)
    with np.load(path, allow_pickle=False) as golden:
        assert golden["features"].shape == (57, 128)
        assert golden["subsampler_dense_valid"].shape == (7, 1024)
        assert np.isfinite(golden["subsampler_dense_valid"]).all()
        np.testing.assert_array_equal(got["attention_mask"], golden["feature_mask"])
        np.testing.assert_allclose(got["features"], golden["features"], rtol=0, atol=2e-4)
        np.testing.assert_array_equal(golden["features"][-1], 0)
