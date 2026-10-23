"""Bounded format/ISA qualification; no weights, remote runtime, or FPGA required."""
import dataclasses
import json

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I
from opentpu.asr.phonon2 import ASRNotReady, MODEL_ID, MODEL_REVISION, inferlab_model_status, model_status, transcribe
from opentpu.asr.phonon2_linear import FiveValue, compile_linear_slice, decode_five_value
from opentpu.isasim import Machine


GOLDEN = bytes.fromhex("417ac47bc60200340038003a003e")


def test_container_record_sign_order_row_padding_and_nonzero_bit_order():
    # Two rows of seven columns: trit padding is per row, high bits span rows.
    w = decode_five_value(GOLDEN, (2, 7))
    np.testing.assert_array_equal(w.dense(), [[.25, -.75, 0, .75, -.25, .25, 0],
                                            [0, .5, -1.5, 0, 1.5, -.5, 1.5]])
    assert not w.sign.flags.writeable and not w.low_scale.flags.writeable


@pytest.mark.parametrize("blob", [b"", GOLDEN[:-1], GOLDEN + b"\0", b"\xff" + GOLDEN[1:]])
def test_bad_record_lengths_and_trits(blob):
    with pytest.raises(ValueError):
        decode_five_value(blob, (2, 7))


@pytest.mark.parametrize("shape", [(0, 7), (-1, 7), (True, 7), (2, 7.0), (2,), (1 << 30, 2)])
def test_bad_shapes_fail_before_allocation(shape):
    with pytest.raises(ValueError, match="shape"):
        decode_five_value(GOLDEN, shape)


@pytest.mark.parametrize("scale", [np.inf, np.nan, -1.0, 0.1, 1e10])
def test_nonfinite_negative_or_requantized_scales_rejected(scale):
    with pytest.raises(ValueError, match="scales"):
        FiveValue(np.ones((1, 3)), np.zeros((1, 3), bool), np.array([scale]), np.array([1.0]))


def test_zero_record_and_zero_magnitude():
    w = decode_five_value(bytes([121, 121]) + bytes(4), (1, 7))
    np.testing.assert_array_equal(w.dense(), np.zeros((1, 7)))


def run_isa(weight, x, cfg):
    p = compile_linear_slice(weight, len(x), cfg)
    a = np.zeros((len(x), p.padded_inputs), np.float32)
    a[:, :p.inputs] = x
    image = p.image.copy()
    image[p.input_at:p.input_at + a.nbytes] = a.view(np.uint8).ravel()
    m = Machine(cfg, [p.program], [image]).run()
    y = m.slices[0].dram[p.output_at:p.output_at + 4 * len(x) * p.outputs].view(np.float32)
    return y.reshape(len(x), p.outputs), p, a


@pytest.mark.parametrize("rows,inputs,outputs,D", [(1, 7, 2, 128), (4, 129, 17, 128),
                                                   (4, 1024, 64, 128), (4, 4096, 32, 128),
                                                   (3, 65, 31, 32)])
def test_two_plane_projection_against_independent_dense_math(rows, inputs, outputs, D):
    rng = np.random.default_rng(inputs)
    sign = rng.integers(-1, 2, (outputs, inputs), dtype=np.int8)
    high = rng.integers(0, 2, sign.shape).astype(bool)
    low = rng.uniform(.01, .2, outputs).astype(np.float16)
    upper = rng.uniform(.23, .51, outputs).astype(np.float16)
    w = FiveValue(sign, high, low, upper)
    x = rng.standard_normal((rows, inputs)).astype(np.float32)
    # Exercise QACT offsets and an all-zero row, including padded activation tails.
    x[:, :min(32, inputs)] *= .01
    if rows > 1:
        x[-1] = 0
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=128, DRAM_BYTES=4 << 20)
    got, p, padded = run_isa(w, x, cfg)
    q, scale, offsets = F.quantize_mx(padded, D // 4)
    deq = q.astype(np.float64).reshape(rows, -1, D // 4)
    deq *= np.ldexp(scale.astype(np.float64)[:, None], -offsets.astype(int))[:, :, None]
    expected = deq.reshape(rows, -1)[:, :inputs] @ w.dense().astype(np.float64).T
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=2e-5)
    dense = x.astype(np.float64) @ w.dense().astype(np.float64).T
    # This tolerance measures activation quantization; it is not a WER claim.
    assert np.linalg.norm(got - dense) / max(np.linalg.norm(dense), 1e-20) < .02
    assert [op.op for op in p.program].count(I.QACT) == 1
    mms = [op for op in p.program if op.op == I.MM]
    assert len(mms) == 2 and all(op.flags & I.F_RSC for op in mms)
    assert mms[1].flags & I.F_ACC
    assert set(op.op for op in p.program) <= {I.LD, I.ST, I.QACT, I.MM, I.HALT}
    assert not p.image.flags.writeable


def test_weight_levels_are_preserved_in_device_image_without_requantization():
    w = decode_five_value(GOLDEN, (2, 7))
    cfg = Config(S=1, D=128)
    p = compile_linear_slice(w, 1, cfg)
    # Inspect the bytes, independently of the CPU dense expansion.
    reconstructed = np.zeros((2, 128), np.float64)
    # Input (512 B), two planes (256 B each), then two row scale arrays (128 B aligned).
    for at, scale_at in ((512, 1024), (768, 1152)):
        q = p.image[at:at + 256].view(np.int8).reshape(2, 128)
        sc = p.image[scale_at:scale_at + 8].view('<f4')
        reconstructed += q * sc[:, None]
    np.testing.assert_array_equal(reconstructed[:, :7], w.dense())
    np.testing.assert_array_equal(reconstructed[:, 7:], 0)


@pytest.mark.parametrize("outputs,inputs", [(4096, 1024), (1024, 4096), (1024, 1024), (2048, 1024)])
def test_actual_encoder_projection_shapes_compile_in_saved_f2_budget(outputs, inputs):
    w = FiveValue(np.zeros((outputs, inputs), np.int8), np.zeros((outputs, inputs), bool),
                  np.zeros(outputs), np.zeros(outputs))
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=16 << 20)
    p = compile_linear_slice(w, 4, cfg)
    assert p.weight_bytes == 2 * outputs * inputs + 8 * outputs
    assert p.budget()["instruction_words"] <= 32768


@pytest.mark.parametrize("changes,rows", [({"S": 2}, 1), ({}, 0), ({}, 5),
                                        ({"DRAM_BYTES": 128}, 1), ({"ACT_BLOCKS": 0}, 1),
                                        ({"IMEM_WORDS": 8}, 1)])
def test_resource_limits_are_checked(changes, rows):
    cfg = dataclasses.replace(Config(S=1, D=128, MCOLS=4), **changes)
    with pytest.raises(ValueError):
        compile_linear_slice(decode_five_value(GOLDEN, (2, 7)), rows, cfg)


def test_status_never_claims_audio_inference_or_uses_a_fallback(monkeypatch):
    import builtins
    status = model_status()
    assert status["id"] == MODEL_ID and status["revision"] == MODEL_REVISION
    assert status["status"] == "blocked" and status["readiness"] == "learned_encoder_isa_only"
    assert status["shared_image"]["profile"] == "f2-hbm:p4rl8a64c150"
    assert status["shared_image"]["hardware_identity_verified"] is False
    assert status["qualification"]["conformer_blocks"] == 24
    assert all(status[k] is False for k in ("ready", "available", "cpu_gpu_fallback",
                                           "hardware_validated", "audio_to_transcript_validated",
                                           "host_model_compute_allowed"))
    assert len(status["blockers"]) >= 8
    json.dumps(status)
    status["blockers"].clear()
    assert model_status()["blockers"]
    def forbidden(*a, **kw):
        raise AssertionError("not-ready adapter attempted file or runtime access")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(builtins, "__import__", forbidden)
    with pytest.raises(ASRNotReady) as error:
        transcribe(b"\0\0" * 16000)
    assert error.value.code == "ASR_FPGA_NOT_READY" and error.value.http_status == 503


def test_inferlab_card_is_explicitly_blocked():
    card = inferlab_model_status()
    assert card['id'] == 'phonon2' and card['model_id'] == MODEL_ID
    assert card['kind'] == 'asr' and card['capability'] == 'transcription'
    assert card['ready'] is False and card['hardware_validated'] is False
    assert card['cpu_gpu_fallback'] is False
    assert card['error_code'] == 'ASR_FPGA_NOT_READY'
