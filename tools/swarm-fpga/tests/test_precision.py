"""Precision policies and the explicitly declared full-precision STE surrogate.

Quantized rounding is discrete: finite differences through it are not an oracle
for its chosen surrogate gradient. Native MXU opcode tests live separately.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from swarmfpga.backend import ReferenceBackend
from swarmfpga.config import Config
from swarmfpga.graph import Graph
from swarmfpga.model import build
from swarmfpga.runtime import SwarmModel
from test_framework import configuration, cross_entropy


def rounded(value, precision):
    """Independent floating-point spacing oracle, without bit manipulation."""
    x = np.asarray(value, np.float64)
    if precision == "fp16":
        return x.astype(np.float16).astype(np.float32)
    x = np.where(np.abs(x) < 2.0 ** -126, 0.0, x)
    _, exponent = np.frexp(x)
    quantum = np.maximum(np.ldexp(1.0, exponent - 8), 2.0 ** -133)
    return (np.rint(x / quantum) * quantum).astype(np.float32)


def mixed_config(backward="fp32"):
    return Config.from_dict(configuration(forward_dtype="int8", backward_dtype=backward))


def test_int8_forward_quantizes_both_operands_and_handles_zero_rows():
    pytest.importorskip("opentpu")
    x = np.array([[1.0, 0.12345, -0.4567], [0, 0, 0]], np.float32)
    w = np.array([[0.987654, 0], [0.55555, 0], [-0.112233, 0]], np.float32)
    def quantized_rows(rows):
        scale = np.max(np.abs(rows), axis=-1, keepdims=True) / np.float32(127)
        inverse = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
        return np.clip(np.rint(rows * inverse), -127, 127) * scale
    expected = quantized_rows(x) @ quantized_rows(w.T).T
    graph = Graph()
    a, b = graph.input("x", x.shape), graph.input("w", w.shape)
    y = graph.node("matmul_int8", (a, b), (2, 2))
    actual = ReferenceBackend().run(graph.export({"y": y}), {"x": x, "w": w})["y"]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=3e-5)
    assert np.max(np.abs(actual - x @ w)) > 1e-4
    np.testing.assert_array_equal(actual[1], 0)
    np.testing.assert_array_equal(actual[:, 1], 0)


def test_int8_backward_is_the_documented_original_operand_surrogate():
    pytest.importorskip("opentpu")
    x = np.array([[1.0, 0.12345, -0.4567], [0.31, -0.21, 0.73]], np.float32)
    w = np.array([[0.987654, -0.81], [0.55555, 0.29], [-0.112233, 0.39]], np.float32)
    upstream = np.array([[0.3, -0.2], [0.7, 0.1]], np.float32)
    graph = Graph()
    a, b = graph.input("x", x.shape), graph.input("w", w.shape)
    y = graph.node("matmul_int8", (a, b), (2, 2))
    gradients = graph.grad((y * graph.const(upstream)).sum(), {"x": a, "w": b})
    actual = ReferenceBackend().run(graph.export(gradients), {"x": x, "w": w})
    np.testing.assert_allclose(actual["x"], upstream @ w.T, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual["w"], x.T @ upstream, atol=1e-7, rtol=1e-6)
    assert sum(node["op"] == "matmul_int8" for node in graph.nodes) == 1
    assert sum(node["op"] == "matmul" for node in graph.nodes) == 2


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_backward_matmul_rounds_operands_and_result_with_fp32_accumulation(precision):
    x = np.array([[0.123456, 1.23456], [-0.34567, 0.45678]], np.float32)
    w = np.array([[0.98765, -0.76543], [0.54321, 0.32109]], np.float32)
    graph = Graph()
    a, b = graph.input("x", x.shape), graph.input("w", w.shape)
    y = graph.node("matmul", (a, b), (2, 2), compute_dtype=precision)
    expected = rounded(rounded(x, precision) @ rounded(w, precision), precision)
    actual = ReferenceBackend().run(graph.export({"y": y}), {"x": x, "w": w})["y"]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("precision,values,expected", [
    ("bf16", [1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, 2.0 ** -134, 3 * 2.0 ** -134],
     [1, 1 + 1 / 64, -1, 0, 0]),
    ("fp16", [1 + 1 / 2048, 1 + 3 / 2048, -1 - 1 / 2048, 2.0 ** -25, 3 * 2.0 ** -25],
     [1, 1 + 1 / 512, -1, 0, 2.0 ** -23]),
])
def test_rounding_is_ties_to_even_with_documented_subnormal_policy(precision, values, expected):
    from swarmfpga.backend import _round
    actual = _round(np.array(values, np.float32), precision)
    np.testing.assert_array_equal(actual, np.array(expected, np.float32))


@pytest.mark.parametrize("precision,value", [("fp16", 65520.0), ("bf16", 2.0 ** 111)])
def test_backward_rounding_overflow_fails_instead_of_silently_saturating(precision, value):
    graph = Graph()
    x = graph.input("x", (1,))
    y = graph.node("add", (x, graph.const([0.0])), (1,), compute_dtype=precision)
    with pytest.raises((ValueError, FloatingPointError)):
        ReferenceBackend().run(graph.export({"y": y}), {"x": np.array([value], np.float32)})


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_fp32_master_updates_preserve_values_that_low_precision_weights_would_lose(precision):
    graph = Graph()
    weight = graph.input("weight", ())
    gradient = graph.node("add", (graph.const(1.0), graph.const(0.0)), (), compute_dtype=precision)
    update = weight - np.float32(1e-4) * gradient
    result = ReferenceBackend().run(graph.export({"weight": update}), {"weight": np.array(1, np.float32)})
    assert result["weight"].dtype == np.float32
    assert float(result["weight"]) < 1
    assert float(rounded(result["weight"], precision)) == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_precision_annotations_stop_before_master_parameter_updates(precision):
    config = mixed_config(precision)
    program = build(config, 2, 3, training=True)
    nodes = {node["id"]: node for node in program["nodes"]}
    forward = [node for node in nodes.values() if node["op"] == "matmul_int8"]
    assert len(forward) == 6
    assert all("compute_dtype" not in node["attrs"] for node in forward)
    loss_id = program["outputs"]["loss"]
    attention = [node for node in nodes.values() if node["id"] < loss_id and node["op"] == "matmul"]
    assert len(attention) == 2  # Q@K.T and P@V retain FP32 to preserve causal scaling.
    for name, node_id in program["outputs"].items():
        assert "compute_dtype" not in nodes[node_id]["attrs"], name
    if precision != "fp32":
        backward = [node for node in nodes.values() if node["attrs"].get("compute_dtype") == precision]
        assert backward
        assert any(node["op"] == "matmul" for node in backward)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_quantized_forward_training_reduces_loss_and_resumes_exactly(precision, tmp_path):
    pytest.importorskip("opentpu")
    config = mixed_config(precision)
    rows = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], np.int64)
    tokens, targets = rows[:, :-1], rows[:, 1:]
    model = SwarmModel(config)
    initial = cross_entropy(model.logits(tokens)[0], targets)
    model.train(rows, steps=12)
    final = cross_entropy(model.logits(tokens)[0], targets)
    assert final < initial * 0.95
    checkpoint = tmp_path / "state.npz"
    model.save(checkpoint)
    restored = SwarmModel.load(checkpoint)
    assert restored.config == config
    model.train(rows, steps=2)
    restored.train(rows, steps=2)
    for name, value in model.weights.items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(restored.weights[name], value)


@pytest.mark.parametrize("layers", [1, 2])
def test_int8_projections_preserve_causality_and_chunked_cache_equivalence(layers):
    pytest.importorskip("opentpu")
    model = SwarmModel(replace(mixed_config(), layers=layers))
    for name, weight in model.weights.items():
        if "norm" not in name:
            weight *= 5
    tokens = np.array([[0, 1, 2, 3, 4], [4, 2, 3, 1, 0]], np.int64)
    changed = tokens.copy()
    changed[:, 2:] = np.array([[4, 0, 1], [0, 4, 2]])
    full, _ = model.logits(tokens)
    other, _ = model.logits(changed)
    np.testing.assert_allclose(full[:, :2], other[:, :2], atol=2e-6, rtol=2e-5)
    prefix, cache = model.logits(tokens[:, :2])
    middle, cache = model.logits(tokens[:, 2:4], cache)
    tail, _ = model.logits(tokens[:, 4:], cache)
    np.testing.assert_allclose(np.concatenate((prefix, middle, tail), axis=1), full,
                               atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_native_precision_policy_is_preserved_without_acquiring_hardware(precision):
    config = replace(mixed_config(precision), backend="f2", image="burst200")
    # Constructor injection checks policy plumbing only, not numerical execution.
    backend = SimpleNamespace(device="f2")
    model = SwarmModel(config, backend=backend)
    assert model.backend is backend
    assert model.config.forward_dtype == "int8"
    assert model.config.backward_dtype == precision
    assert all(value.dtype == np.float32 for value in model.weights.values())


@pytest.mark.parametrize("precision,limit", [("fp16", 65504.0), ("bf16", 2.0 ** 110)])
def test_emulated_precision_bounds_accept_limit_and_reject_next_float(precision, limit):
    from swarmfpga.backend import _round
    value = np.array([limit], np.float32)
    np.testing.assert_array_equal(_round(value, precision), value)
    with pytest.raises((ValueError, FloatingPointError)):
        _round(np.nextafter(value, np.float32(np.inf)), precision)


@pytest.mark.parametrize("name,value", [("forward_dtype", "bf16"), ("forward_dtype", "int4"),
                                        ("backward_dtype", "int8"), ("backward_dtype", "float16")])
def test_invalid_precision_policies_fail_early(name, value):
    with pytest.raises(ValueError):
        Config.from_dict(configuration(**{name: value}))
