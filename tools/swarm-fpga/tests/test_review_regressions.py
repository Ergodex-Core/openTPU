"""Failure recovery, bounded label allocation and executable graph regressions."""
import json
import sys
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from swarmfpga import Config, SwarmModel
from swarmfpga.backend import ReferenceBackend
from swarmfpga.graph import Graph
from swarmfpga.model import build
from test_framework import configuration


DATA = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], np.int64)


def assert_same_state(actual, expected):
    assert (actual.step, actual.cursor, actual.dataset_digest) == (
        expected.step, expected.cursor, expected.dataset_digest)
    for name, value in expected.weights.items():
        np.testing.assert_array_equal(actual.weights[name], value, err_msg=name)


@pytest.mark.parametrize("steps", [-1, True, 1.5, 0])
def test_rejected_or_zero_step_training_does_not_bind_dataset(steps):
    model, baseline = SwarmModel(configuration()), SwarmModel(configuration())
    if steps == 0 and type(steps) is int:
        assert model.train(DATA, steps=steps) == []
    else:
        with pytest.raises(ValueError, match="steps must be a nonnegative integer"):
            model.train(DATA, steps=steps)
    assert_same_state(model, baseline)
    changed = DATA[::-1].copy()
    model.train(changed, steps=1)
    baseline.train(changed, steps=1)
    assert_same_state(model, baseline)


def test_first_backend_failure_allows_retry_with_another_dataset():
    model, baseline = SwarmModel(configuration()), SwarmModel(configuration())
    with patch.object(model.backend, "run", side_effect=RuntimeError("execution failed")):
        with pytest.raises(RuntimeError, match="execution failed"):
            model.train(DATA, steps=1)
    assert_same_state(model, baseline)
    changed = DATA[::-1].copy()
    model.train(changed, steps=1)
    baseline.train(changed, steps=1)
    assert_same_state(model, baseline)


def test_failure_after_committed_step_preserves_resumable_progress(tmp_path):
    class FailsSecondRun(ReferenceBackend):
        calls = 0

        def run(self, program, feeds):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second step failed")
            return super().run(program, feeds)

    interrupted = SwarmModel(configuration(), backend=FailsSecondRun())
    expected = SwarmModel(configuration())
    expected.train(DATA, steps=1)
    with pytest.raises(RuntimeError, match="second step failed"):
        interrupted.train(DATA, steps=3)
    assert_same_state(interrupted, expected)
    with pytest.raises(ValueError, match="dataset contents or ordering changed"):
        interrupted.train(DATA[::-1], steps=1)
    checkpoint = tmp_path / "partial.npz"
    interrupted.save(checkpoint)
    restored = SwarmModel.load(checkpoint)
    assert_same_state(restored, expected)
    restored.train(DATA, steps=2)
    expected.train(DATA, steps=2)
    assert_same_state(restored, expected)


def test_label_preparation_scales_with_batch_tokens_and_vocabulary():
    class CaptureLabels(ReferenceBackend):
        labels = None

        def run(self, program, feeds):
            self.labels = feeds["labels"]
            return {"loss": np.array(0, np.float32), **{
                name: value for name, value in feeds.items() if name not in ("tokens", "labels")}}

    config = configuration()
    config["model"]["vocab_size"] = 4096
    backend = CaptureLabels()
    model = SwarmModel(config, backend=backend)
    tokens = np.array([[0, 4095, 2]], np.int64)
    targets = np.array([[4095, 2, 0]], np.int64)
    tracemalloc.start()
    try:
        model.train_step(tokens, targets)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The former vocabulary identity would allocate 64 MiB. Allow generous
    # graph-building overhead while ruling out that quadratic allocation.
    assert peak < 16 * 1024 * 1024
    assert backend.labels.shape == (1, 3, 4096)
    assert backend.labels.dtype == np.float32
    np.testing.assert_array_equal(backend.labels.sum(-1), np.ones((1, 3)))
    np.testing.assert_array_equal(backend.labels.argmax(-1), targets)


def test_weight_initialization_failure_never_acquires_native_device():
    config = configuration()
    config["backend"] = {"type": "f2", "image": "burst200"}
    constructor = Mock()
    module = SimpleNamespace(OpenTPUBackend=constructor)
    with patch.dict(sys.modules, {"swarmfpga.opentpu": module}), \
            patch("swarmfpga.runtime.initialize", side_effect=MemoryError("initialization failed")):
        with pytest.raises(MemoryError, match="initialization failed"):
            SwarmModel(config)
    constructor.assert_not_called()


def test_mismatched_resume_config_never_acquires_native_device(tmp_path):
    config = configuration()
    path = tmp_path / "state.npz"
    SwarmModel(config).save(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(values["metadata"]))
    metadata["config"].update(backend="f2", image="burst200")
    values["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **values)
    constructor = Mock()
    with patch.dict(sys.modules, {"swarmfpga.opentpu": SimpleNamespace(OpenTPUBackend=constructor)}):
        with pytest.raises(ValueError, match="resume configuration differs"):
            SwarmModel.load(path, expected_config=Config.from_dict(config))
    constructor.assert_not_called()


def test_atomic_checkpoint_failure_preserves_previous_state(tmp_path):
    model = SwarmModel(configuration())
    path = tmp_path / "state.npz"
    model.save(path)
    original = path.read_bytes()
    model.train(DATA, steps=1)
    with patch("swarmfpga.runtime.os.replace", side_effect=OSError("publish failed")):
        with pytest.raises(OSError, match="publish failed"):
            model.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert_same_state(SwarmModel.load(path), SwarmModel(configuration()))


def test_exported_gradient_skips_unused_out_of_range_fp16_work():
    graph = Graph()
    x = graph.input("x", ())
    loss = x * 0.0
    start = len(graph.nodes)
    gradient = graph.grad(loss, {"x": x})
    for node in graph.nodes[start:]:
        node["attrs"]["compute_dtype"] = "fp16"
    output = ReferenceBackend().run(graph.export(gradient), {"x": np.array(70000, np.float32)})
    np.testing.assert_array_equal(output["x"], np.array(0, np.float32))


@pytest.mark.parametrize("forward,backward", [("fp32", "fp32"), ("int8", "bf16"), ("int8", "fp16")])
def test_training_export_omits_unused_nodes_and_preserves_native_results(forward, backward):
    pytest.importorskip("opentpu")
    from test_native import ISABackend

    config = Config.from_dict(configuration(forward_dtype=forward, backward_dtype=backward))
    # Retain a complete graph as an independent before/after execution oracle.
    original_export = Graph.export
    captured = {}

    def capture(graph, outputs):
        captured["nodes"] = list(graph.nodes)
        return original_export(graph, outputs)

    with patch.object(Graph, "export", capture):
        pruned = build(config, 2, 3, training=True)
    complete = {**pruned, "nodes": captured["nodes"]}
    model = SwarmModel(config)
    targets = DATA[:2, 1:]
    labels = np.zeros((*targets.shape, config.vocab_size), np.float32)
    np.put_along_axis(labels, targets[..., None], 1.0, axis=-1)
    feeds = {**model.weights, "tokens": DATA[:2, :-1], "labels": labels}
    before, after = ISABackend(), ISABackend()
    expected, actual = before.run(complete, feeds), after.run(pruned, feeds)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert len(pruned["nodes"]) < len(complete["nodes"])
    assert 0 < after.kernel_count < before.kernel_count
