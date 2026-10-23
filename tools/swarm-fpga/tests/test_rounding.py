"""Independent reduced-precision rounding checks using the ISA, never hardware.

Results are FP32 carriers of rounded values; this is not native 16-bit arithmetic.
BF16 follows the VPU's input/output flush-to-zero rule; zero signs may canonicalize.
"""
import numpy as np
import pytest

pytest.importorskip("opentpu", reason="optional OpenTPU ISA dependency unavailable")

from swarmfpga.graph import Graph
from test_native import ISABackend


def rounded_reference(values, kind):
    x = np.array(values, dtype=np.float32, copy=True)
    if kind == "fp16":
        return x.astype(np.float16).astype(np.float32)
    x[np.abs(x) < np.finfo(np.float32).tiny] = 0
    bits = x.view(np.uint32)
    bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    result = bits.view(np.float32)
    result[np.abs(result) < np.finfo(np.float32).tiny] = 0
    return result


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_native_rounding_ties_bounds_and_single_pass_counterexamples(kind):
    common = [0, -0., 1., -1., 1.99999, -1.99999, 16383.2099609375,
              -32767.912109375, 32765.763671875, 2.980224245163754e-8,
              -4.701973199393757e-38, np.finfo(np.float32).tiny,
              np.nextafter(np.float32(0), np.float32(1))]
    half_ulp = 2.0 ** (-8 if kind == "bf16" else -11)
    ties = np.array([1 + half_ulp, 1 + 3 * half_ulp, 1 + 5 * half_ulp], np.float32)
    around = np.concatenate([np.nextafter(ties, np.float32(-np.inf)), ties,
                             np.nextafter(ties, np.float32(np.inf))])
    bound = np.float32(2.0 ** 110 if kind == "bf16" else 65504)
    values = np.concatenate([np.array(common, np.float32), around, -around, [bound, -bound]])
    actual = ISABackend()._launch("round_" + kind, values)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, rounded_reference(values, kind))


def test_native_fp16_preserves_subnormals_and_rounds_underflow_ties():
    ties = np.array([2.0 ** -25, 3 * 2.0 ** -25, 2.0 ** -14 - 2.0 ** -25], np.float32)
    positive = np.concatenate([np.nextafter(ties, np.float32(0)), ties,
                               np.nextafter(ties, np.float32(np.inf)), [2.0 ** -24]])
    values = np.concatenate([positive, -positive]).astype(np.float32)
    actual = ISABackend()._launch("round_fp16", values)
    np.testing.assert_array_equal(actual, rounded_reference(values, "fp16"))
    assert actual[-1] == -(2.0 ** -24)


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_native_rounding_random_exponent_coverage(kind):
    rng = np.random.default_rng(219)
    exponents = np.arange(1, 237, dtype=np.uint32) if kind == "bf16" else np.arange(101, 143, dtype=np.uint32)
    bits = (np.repeat(exponents, 5) << 23) | rng.integers(0, 1 << 23, len(exponents) * 5, dtype=np.uint32)
    positive = bits.view(np.float32)
    if kind == "fp16":
        positive = positive[positive <= 65504]
    values = np.concatenate([positive, -positive])
    actual = ISABackend()._launch("round_" + kind, values)
    np.testing.assert_array_equal(actual, rounded_reference(values, kind))


@pytest.mark.parametrize("kind,bound", [("bf16", 2.0 ** 110), ("fp16", 65504)])
@pytest.mark.parametrize("case", ["positive_overflow", "negative_overflow", "nan", "inf", "negative_inf"])
def test_native_rounding_rejects_unsupported_values_before_launch(kind, bound, case):
    above = np.nextafter(np.float32(bound), np.float32(np.inf))
    value = {"positive_overflow": above, "negative_overflow": -above,
             "nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[case]
    backend = ISABackend()
    with pytest.raises(FloatingPointError):
        backend._launch("round_" + kind, np.array([value], np.float32))
    assert backend.board.launches == 0


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_native_marked_node_rounds_operands_and_result(kind):
    graph = Graph()
    left, right = graph.input("left", (4,)), graph.input("right", (4,))
    output = left + right
    graph.nodes[output.id]["attrs"]["compute_dtype"] = kind
    values = {"left": np.array([1.0038, -1.99999, .0000513, 127.343], np.float32),
              "right": np.array([.0042, .00027, .0000112, -.0372], np.float32)}
    expected = rounded_reference(rounded_reference(values["left"], kind) +
                                 rounded_reference(values["right"], kind), kind)
    backend = ISABackend()
    actual = backend.run(graph.export({"sum": output}), values)["sum"]
    np.testing.assert_array_equal(actual, expected)
    assert backend.op_counts["round_" + kind] == 3


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_native_marked_gather_leaves_integer_indices_unchanged(kind):
    graph = Graph()
    weights = graph.input("weights", (5, 2))
    ids = graph.input("ids", (3,), dtype="int64")
    output = weights.gather(ids)
    graph.nodes[output.id]["attrs"]["compute_dtype"] = kind
    values = np.arange(10, dtype=np.float32).reshape(5, 2) * .137
    backend = ISABackend()
    actual = backend.run(graph.export({"gathered": output}),
                         {"weights": values, "ids": np.array([4, 1, 4], np.int64)})["gathered"]
    np.testing.assert_array_equal(actual, rounded_reference(values[[4, 1, 4]], kind))
    assert backend.op_counts["round_" + kind] == 2
