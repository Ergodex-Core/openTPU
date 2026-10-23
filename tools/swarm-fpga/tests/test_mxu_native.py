"""INT8 forward execution uses on-card quantizers and actual MXU instructions."""
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("opentpu", reason="optional OpenTPU ISA simulator required")
from opentpu import isa as I, quant as Q
from test_native import ISABackend


def golden(left, right, depth=128):
    padding = -left.shape[1] % depth
    a = np.pad(left, ((0, 0), (0, padding)))
    w = np.pad(right.T, ((0, 0), (0, padding)))
    return np.asarray(Q.fake_act(a, depth) @ Q.fake_w(w, "int8", depth).T, np.float32)


@pytest.mark.parametrize("shape", [(3, 7, 5), (6, 128, 7), (2, 257, 4)])
def test_forward_uses_dynamic_qst_qact_and_int8_mxu(shape):
    m, k, n = shape
    rng = np.random.default_rng(732 + k)
    left = rng.normal(0, 0.3, (m, k)).astype(np.float32)
    right = rng.normal(0, 0.2, (k, n)).astype(np.float32)
    left[0] = 0
    right[:, 0] = 0
    expected = golden(left, right)
    backend, instructions = ISABackend(), []
    original_run = backend.board.run
    def record(programs):
        instructions.extend(ins for program in programs for ins in program)
        return original_run(programs)
    backend.board.run = record
    with patch.object(Q, "quantize_mxu", side_effect=AssertionError("host weight quantization")):
        actual = backend._launch("matmul_int8", left, right, (m, n))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=3e-5)
    assert {I.QST, I.QACT, I.MM}.issubset({ins.op for ins in instructions})
    assert all((ins.flags >> I.WF_SHIFT) & 3 == I.W8 for ins in instructions if ins.op == I.MM)
    assert backend.mxu_instructions == sum(ins.op == I.MM for ins in instructions)
    assert all(backend.instruction_counts[name] >= 1 for name in ("MM", "QACT", "QST"))
    assert backend.op_counts["matmul_int8"] == 1


def test_weight_updates_are_requantized_and_padding_does_not_leak():
    rng = np.random.default_rng(98)
    left = rng.normal(size=(2, 33)).astype(np.float32)
    right = rng.normal(size=(33, 3)).astype(np.float32)
    backend = ISABackend()
    first = backend._launch("matmul_int8", left, right, (2, 3))
    right[:, 1] *= -0.25
    second = backend._launch("matmul_int8", left, right, (2, 3))
    np.testing.assert_allclose(second, golden(left, right), atol=2e-6, rtol=3e-5)
    assert not np.array_equal(first, second)
    # Reusing the staging area for smaller logical K must quantize fresh zero padding.
    narrow = backend._launch("matmul_int8", left[:, :3], right[:3], (2, 3))
    np.testing.assert_allclose(narrow, golden(left[:, :3], right[:3]), atol=2e-6, rtol=3e-5)


def test_mixed_scale_groups_follow_mxu_drop_window():
    rng = np.random.default_rng(194)
    left = rng.normal(0, 0.3, (2, 256)).astype(np.float32)
    right = rng.normal(0, 0.2, (256, 4)).astype(np.float32)
    left[:, 32:128] *= np.float32(2 ** -24)
    right[128:] *= np.float32(2 ** -20)
    expected = golden(left, right)
    actual = ISABackend()._launch("matmul_int8", left, right, (2, 4))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=3e-5)


def test_int8_padding_is_included_in_native_word_budget():
    backend = ISABackend()
    with pytest.raises(ValueError, match="8192"):
        backend._launch("matmul_int8", np.ones((64, 1), np.float32),
                        np.ones((1, 8), np.float32), (64, 8))
    assert backend.board.launches == 0
