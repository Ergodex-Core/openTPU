"""Small installed-native regressions for the bounded depth-chain checker.

These tests run on the CPU EC2 environment with the pinned moshi source. They
do not require a checkpoint or a physical FPGA.
"""
from __future__ import annotations

import numpy as np
import pytest

from opentpu.llm.personaplex_depth_chain import DepthChainSpec
from opentpu.llm.personaplex_depth_layer import DepthLayerSpec
from tools.personaplex_depth_chain_check import (
    exact_bytes, make_native_input, native_chain_pass, native_stack,
    payload_sha256, preflight,
)


def _tiny_chain():
    """Zero transformer banks leave each selected source embedding intact."""
    spec = DepthChainSpec(hidden=32, depth=32, n_heads=2, ffn=64, steps=4,
                          layers=2, text_vocab=7, audio_vocab=5, head_vocab=4)
    projections = np.zeros((spec.steps, spec.depth, spec.hidden), np.float32)
    tables = [np.zeros((spec.text_vocab, spec.depth), np.float32)]
    tables.extend(np.zeros((spec.audio_vocab, spec.depth), np.float32)
                  for _ in range(spec.steps - 1))
    tables[0][6, 0] = 1
    tables[1][2, 1] = 1
    tables[2][0, 2] = 1
    tables[3][3, 3] = 1
    # A different teacher-forced first ID changes the next embedding and logits.
    for table in tables[1:]:
        table[1, 4] = 1
    heads = np.zeros((spec.steps, spec.head_vocab, spec.depth), np.float32)
    for step, token in enumerate((2, 0, 3, 1)):
        heads[step, token, step] = 1
    def layer():
        def bank():
            return {"wq": np.zeros((32, 32), np.float32),
                    "wk": np.zeros((32, 32), np.float32),
                    "wv": np.zeros((32, 32), np.float32),
                    "wo": np.zeros((32, 32), np.float32),
                    "wg": np.zeros((64, 32), np.float32),
                    "wu": np.zeros((64, 32), np.float32),
                    "wd": np.zeros((32, 64), np.float32)}
        return {"norm1": np.ones(32, np.float32), "norm2": np.ones(32, np.float32),
                "steps": [bank() for _ in range(spec.steps)]}
    return spec, projections, tuple(tables), heads, [layer() for _ in range(spec.layers)]


def test_native_greedy_feedback_and_teacher_forcing_use_distinct_previous_ids():
    spec, projections, tables, heads, layers = _tiny_chain()
    source, linears = make_native_input(projections, tables, spec)
    native = native_stack(layers, DepthLayerSpec(hidden=spec.depth, n_heads=spec.n_heads,
                                                 ffn=spec.ffn, steps=spec.steps,
                                                 layers=spec.layers))
    hidden = np.zeros(spec.hidden, np.float32)
    greedy = native_chain_pass(native, source, linears, heads, hidden, 6)
    np.testing.assert_array_equal(greedy["ids"], [2, 0, 3, 1])
    expected_positions = [[0, -1, -1, -1], [0, 1, -1, -1],
                          [0, 1, 2, -1], [4, 1, 2, 3]]
    assert greedy["cache_positions"] == [expected_positions] * spec.layers
    np.testing.assert_allclose(greedy["conditioned"], greedy["depth"], atol=0, rtol=0)

    forced_ids = np.array([1, 1, 1, 1], np.int32)
    forced = native_chain_pass(native, source, linears, heads, hidden, 6,
                               forced_ids=forced_ids)
    assert forced["cache_positions"] == greedy["cache_positions"]
    np.testing.assert_array_equal(forced["conditioned"][0], greedy["conditioned"][0])
    assert not np.array_equal(forced["conditioned"][1], greedy["conditioned"][1])
    assert not np.array_equal(forced["logits"][1], greedy["logits"][1])
    assert int(np.argmax(forced["logits"][1])) == int(forced["ids"][1])
    with pytest.raises(ValueError, match="Teacher-forced ID count"):
        native_chain_pass(native, source, linears, heads, hidden, 6,
                          forced_ids=forced_ids[:-1])


def test_existing_output_directory_is_rejected_before_baseline_reads(tmp_path):
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must be new"):
        preflight(tmp_path / "model", tmp_path / "token", tmp_path / "conditioning",
                  tmp_path / "stack", output)


def test_payload_hash_includes_shape_name_dtype_and_exact_bytes_checks_shape():
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    original = payload_sha256([("matrix", matrix)])
    assert original["raw_bytes"] == matrix.nbytes
    assert original["sha256"] == payload_sha256([("matrix", matrix.copy())])["sha256"]
    assert original["sha256"] != payload_sha256([("matrix", matrix.reshape(3, 2))])["sha256"]
    assert original["sha256"] != payload_sha256([("other", matrix)])["sha256"]
    assert original["sha256"] != payload_sha256([("matrix", matrix.astype(np.float64))])["sha256"]
    exact_bytes(matrix, matrix.copy(), "same")
    with pytest.raises(AssertionError, match="raw array bytes differ"):
        exact_bytes(matrix, matrix.reshape(3, 2), "shape")
    with pytest.raises(AssertionError, match="raw array bytes differ"):
        exact_bytes(matrix, matrix.astype(np.int32), "dtype")
    with pytest.raises(ValueError, match="Invalid source payload"):
        payload_sha256([("bad", np.array([np.nan], np.float32))])


@pytest.mark.parametrize("frame", [-1, 4, True, 1.5])
def test_frame_selection_is_bounded_before_baseline_reads(tmp_path, frame):
    with pytest.raises(ValueError, match="Frame index"):
        preflight(tmp_path/"model", tmp_path/"token", tmp_path/"conditioning",
                  tmp_path/"stack", tmp_path/"out", frame_index=frame)
    assert not (tmp_path/"out").exists()
