"""Optional native-lowering checks using only the OpenTPU ISA simulator.

Set PYTHONPATH to an existing OpenTPU checkout. No native board constructor,
device lock, transport, bitstream loading, or physical FPGA is used here.
"""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("opentpu", reason="optional OpenTPU dependency is not installed")

from opentpu.isasim import Machine
from opentpu.target import config as target_config
from swarmfpga.backend import ReferenceBackend
from swarmfpga.config import Config
from swarmfpga.graph import Graph
from swarmfpga.opentpu import OpenTPUBackend
from swarmfpga.runtime import SwarmModel
from test_framework import configuration


class ISABoard:
    """Board read/write/run interface backed solely by in-memory ISA execution."""

    def __init__(self, cfg):
        assert cfg.S == 1
        self.cfg = cfg
        self.memory = np.zeros(cfg.DRAM_BYTES, np.uint8)
        self.launches = 0

    def write(self, chip, address, data):
        assert chip == 0
        self.memory[address:address + data.size] = data

    def run(self, programs):
        machine = Machine(self.cfg, programs, [self.memory]).run()
        self.memory = machine.slices[0].dram
        self.launches += 1
        # ISA simulation validates values; it does not measure device cycles.
        return {"cycles": 0}

    def read(self, chip, address, count):
        assert chip == 0
        return self.memory[address:address + count].copy()


class ISABackend(OpenTPUBackend):
    def __init__(self):
        # Deliberately bypass OpenTPUBackend.__init__: that is hardware-only.
        self.cfg = replace(target_config("f2"), DRAM_BYTES=1 << 20)
        self.board = ISABoard(self.cfg)
        self.kernel_count, self.device_cycles, self.op_counts = 0, 0, {}


def assert_backend_parity(graph, outputs, feeds, atol=2e-5, rtol=2e-4):
    program = graph.export(outputs)
    expected = ReferenceBackend().run(program, feeds)
    backend = ISABackend()
    actual = backend.run(program, feeds)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, atol=atol, rtol=rtol, err_msg=name)
    return backend


def test_native_arithmetic_and_reductions_match_reference():
    graph = Graph()
    x = graph.input("x", (2, 3))
    y = graph.input("y", (3,))
    outputs = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y,
               "neg": -x, "exp": x.exp(), "log": x.log(), "rsqrt": x.rsqrt(),
               "tanh": x.tanh(), "sum": x.sum(0), "maximum": x.max(1, True)}
    backend = assert_backend_parity(graph, outputs,
        {"x": np.array([[0.2, 0.5, 1.1], [0.7, 0.4, 0.8]], np.float32),
         "y": np.array([0.4, 0.8, 1.2], np.float32)})
    assert backend.board.launches >= len(outputs)
    assert backend.kernel_count == backend.board.launches
    assert sum(backend.op_counts.values()) == backend.kernel_count


def test_native_broadcast_matmul_and_gradients_match_reference():
    graph = Graph()
    x = graph.input("x", (2, 1, 2, 3))
    w = graph.input("w", (1, 2, 3, 2))
    output = x @ w
    loss = output.tanh().sum()
    gradients = graph.grad(loss, {"x": x, "w": w})
    rng = np.random.default_rng(24)
    assert_backend_parity(graph, {"output": output, "loss": loss, **gradients},
        {"x": rng.normal(0, 0.2, x.shape).astype(np.float32),
         "w": rng.normal(0, 0.2, w.shape).astype(np.float32)})


@pytest.mark.parametrize("axis,shape,indices", [
    (0, (5, 3), [[1, 1, 4], [0, 1, 4]]),
    (1, (2, 4, 3), [[1, 1], [2, 1]]),
])
def test_native_gather_repeated_indices_accumulate(axis, shape, indices):
    graph = Graph()
    w = graph.input("w", shape)
    ids = graph.input("ids", np.shape(indices), dtype="int64")
    gathered = w.gather(ids, axis=axis)
    loss = (gathered * gathered).sum()
    gradients = graph.grad(loss, {"w": w})
    assert_backend_parity(graph, {"gathered": gathered, **gradients},
        {"w": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.03,
         "ids": np.array(indices, np.int64)})


@pytest.mark.parametrize("forward,backward", [
    ("fp32", "fp32"), ("int8", "fp32"), ("int8", "bf16"), ("int8", "fp16"),
])
def test_native_full_decoder_logits_cache_and_sgd_match_reference(forward, backward):
    base = configuration(forward_dtype=forward, backward_dtype=backward)
    native_config = {**base, "backend": {"type": "f2", "image": "burst200"}}
    native = SwarmModel(Config.from_dict(native_config), backend=ISABackend())
    reference = SwarmModel(Config.from_dict(base))
    tokens = np.array([[0, 1, 2], [2, 0, 1]], np.int64)
    targets = np.array([[1, 2, 0], [0, 1, 2]], np.int64)
    expected, _ = reference.logits(tokens)
    actual, _ = native.logits(tokens)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=5e-4)
    prefix, cache = native.logits(tokens[:, :2])
    final, _ = native.logits(tokens[:, 2:], cache=cache)
    np.testing.assert_allclose(np.concatenate([prefix, final], axis=1), expected,
                               atol=3e-5, rtol=5e-4)
    expected_step = reference.train_step(tokens, targets)
    actual_step = native.train_step(tokens, targets)
    np.testing.assert_allclose(actual_step["loss"], expected_step["loss"], atol=3e-5, rtol=5e-4)
    for name, value in reference.weights.items():
        np.testing.assert_allclose(native.weights[name], value, atol=3e-5, rtol=5e-4,
                                   err_msg=f"updated parameter {name}")
    assert native.backend.board.launches > 100
    if forward == "int8":
        assert all(native.backend.instruction_counts[op] > 0 for op in ("MM", "QST", "QACT"))
    if backward != "fp32":
        assert native.backend.op_counts[f"round_{backward}"] > 0


def test_native_launch_enforces_bounded_kernel_size():
    backend = ISABackend()
    with pytest.raises(ValueError, match="8192"):
        backend._launch("add", np.ones(3000, np.float32), np.ones(3000, np.float32))
    assert backend.board.launches == 0
