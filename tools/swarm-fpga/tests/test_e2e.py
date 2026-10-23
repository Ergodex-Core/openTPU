"""User-facing CLI workflows in fresh processes, with persisted state between them.

CLI tests select the reference device explicitly. Native lifecycle tests inject
an in-memory ISA board; no physical board, locks, credentials or network are used.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from swarmfpga import SwarmModel
from test_framework import configuration


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POLICIES = [("fp32", "fp32"), ("int8", "fp32"), ("int8", "bf16"), ("int8", "fp16")]


def cli(directory, *arguments, success=True):
    """Run outside the source directory so no cwd import or file assumptions leak in."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), str(PACKAGE_ROOT.parents[1])])
    result = subprocess.run([sys.executable, "-m", "swarmfpga", *map(str, arguments)],
                            cwd=directory, env=environment, capture_output=True, text=True, timeout=30)
    if success:
        assert result.returncode == 0, result.stderr
        assert result.stderr == "", result.stderr
    else:
        assert result.returncode != 0, result.stdout
    return result


def write_inputs(directory, forward="fp32", backward="fp32"):
    config = configuration(forward_dtype=forward, backward_dtype=backward)
    data = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2],
                     [3, 1, 3, 1], [4, 2, 4, 2]], np.int64)
    config_path, dataset_path = directory / "model config.yaml", directory / "token rows.npy"
    config_path.write_text(yaml.safe_dump(config))
    np.save(dataset_path, data)
    return config, data, config_path, dataset_path


def train(directory, config, data, checkpoint, steps, resume=None):
    arguments = ["train", config.name, data.name, checkpoint.name, "--steps", str(steps)]
    if resume is not None:
        arguments.extend(["--resume", resume.name])
    return [json.loads(line) for line in cli(directory, *arguments).stdout.splitlines()]


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        weights = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    return metadata, weights


@pytest.mark.parametrize("forward,backward", POLICIES)
def test_cli_training_resume_inspection_and_cached_generation(forward, backward, tmp_path):
    if forward == "int8":
        pytest.importorskip("opentpu")
    config, data, config_path, data_path = write_inputs(tmp_path, forward, backward)
    # Paths intentionally contain spaces and no .npz extension. Every command
    # runs in a separate interpreter, including the resumed training process.
    full, partial, resumed = [tmp_path / name for name in ("full state", "partial state", "resumed state")]
    full_metrics = train(tmp_path, config_path, data_path, full, 7)
    partial_metrics = train(tmp_path, config_path, data_path, partial, 3)
    assert partial.is_file()
    partial_bytes = partial.read_bytes()
    resumed_metrics = train(tmp_path, config_path, data_path, resumed, 4, resume=partial)
    assert partial.read_bytes() == partial_bytes

    full_meta, full_weights = read_checkpoint(full)
    resumed_meta, resumed_weights = read_checkpoint(resumed)
    assert resumed_meta == full_meta
    assert full_meta["step"] == 7
    assert full_meta["cursor"] == (7 * config["trainer"]["batch_size"]) % len(data)
    assert full_meta["config"]["forward_dtype"] == forward
    assert full_meta["config"]["backward_dtype"] == backward
    assert len(full_weights) > 5
    for name, value in full_weights.items():
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(resumed_weights[name], value, err_msg=name)

    expected_metrics = full_metrics[:-1]
    replayed_metrics = partial_metrics[:-1] + resumed_metrics[:-1]
    assert len(expected_metrics) == len(replayed_metrics) == 7
    for expected, actual in zip(expected_metrics, replayed_metrics):
        assert actual["step"] == expected["step"]
        assert actual["loss"] == expected["loss"]
        assert actual["tokens"] == 6
        assert actual["device"] == "reference"
        assert actual["wall_seconds"] >= 0
    assert expected_metrics[-1]["loss"] < expected_metrics[0]["loss"]
    for records, checkpoint in ((full_metrics, full), (partial_metrics, partial), (resumed_metrics, resumed)):
        assert records[-1]["checkpoint"] == checkpoint.name
        assert records[-1]["dataset_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
        assert records[-1]["numpy"]
        assert records[-1]["python"]

    inspected = json.loads(cli(tmp_path, "inspect", config_path.name).stdout)
    assert inspected["config"] == full_meta["config"]
    assert inspected["nodes"] > 100
    assert "matmul" in inspected["required_operations"]
    assert ("matmul_int8" in inspected["required_operations"]) == (forward == "int8")

    generated = json.loads(cli(tmp_path, "generate", resumed.name, "--tokens", "0,1",
                               "--max-new-tokens", "3").stdout)
    # The CLI uses the KV cache. Compare against an independent greedy loop
    # that recalculates every full prefix, after reading the persisted model.
    expected_tokens = np.array([[0, 1]], np.int64)
    with SwarmModel.load(full) as model:
        for _ in range(3):
            logits, _ = model.logits(expected_tokens)
            expected_tokens = np.concatenate((expected_tokens, logits[:, -1].argmax(-1)[:, None]), axis=1)
    assert generated["device"] == "reference"
    np.testing.assert_array_equal(generated["tokens"], expected_tokens)
    unchanged = json.loads(cli(tmp_path, "generate", resumed.name, "--tokens", "0,1",
                               "--max-new-tokens", "0").stdout)
    assert unchanged["tokens"] == [[0, 1]]


@pytest.mark.parametrize("mutation,error", [
    ("row_order", "training dataset contents or ordering changed"),
    ("token_content", "training dataset contents or ordering changed"),
    ("config", "resume configuration differs from checkpoint"),
])
def test_cli_failed_resume_preserves_existing_checkpoint(mutation, error, tmp_path):
    config, data, config_path, data_path = write_inputs(tmp_path)
    checkpoint = tmp_path / "state"
    train(tmp_path, config_path, data_path, checkpoint, 2)
    original = checkpoint.read_bytes()
    if mutation == "row_order":
        np.save(data_path, data[::-1])
    elif mutation == "token_content":
        data[0, 0] = 4
        np.save(data_path, data)
    else:
        config["trainer"]["learning_rate"] *= 2
        config_path.write_text(yaml.safe_dump(config))
    # Use the same checkpoint as input/output: failure must leave it readable
    # and byte-for-byte intact rather than publish a partial resumed state.
    result = cli(tmp_path, "train", config_path.name, data_path.name, checkpoint.name,
                 "--steps", "1", "--resume", checkpoint.name, success=False)
    assert error in result.stderr
    assert result.stdout == ""
    assert checkpoint.read_bytes() == original
    metadata, weights = read_checkpoint(checkpoint)
    assert metadata["step"] == 2
    assert weights


@pytest.mark.parametrize("invalid_data,error", [
    (np.array([[0.0, 1.0, 2.0, 0.0]], np.float32), "integer"),
    (np.array([[0, 1, 2]], np.int64), "sequence_length + 1"),
    (np.array([[0, 1, 2, 5]], np.int64), "outside vocabulary"),
])
def test_cli_invalid_training_data_does_not_publish_checkpoint(invalid_data, error, tmp_path):
    _, _, config_path, data_path = write_inputs(tmp_path)
    np.save(data_path, invalid_data)
    checkpoint = tmp_path / "invalid state"
    result = cli(tmp_path, "train", config_path.name, data_path.name, checkpoint.name,
                 "--steps", "1", success=False)
    assert error in result.stderr
    assert result.stdout == ""
    assert not checkpoint.exists()


def test_cli_generation_rejects_context_overflow_without_changing_checkpoint(tmp_path):
    _, _, config_path, data_path = write_inputs(tmp_path)
    checkpoint = tmp_path / "state"
    train(tmp_path, config_path, data_path, checkpoint, 1)
    original = checkpoint.read_bytes()
    result = cli(tmp_path, "generate", checkpoint.name, "--tokens", "0,1",
                 "--max-new-tokens", "7", success=False)
    assert "generation must fit context" in result.stderr
    assert result.stdout == ""
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("forward,backward", POLICIES)
def test_isa_training_checkpoint_resume_and_cached_generation(forward, backward, tmp_path):
    pytest.importorskip("opentpu")
    from unittest.mock import patch
    from test_native import ISABackend

    reference_config = configuration(forward_dtype=forward, backward_dtype=backward)
    native_config = {**reference_config, "backend": {"type": "f2", "image": "burst200"}}
    # A two-row batch traverses three rows: the second step wraps around, so
    # saving after two steps must restore cursor=1 rather than restart at zero.
    rows = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], np.int64)
    checkpoint = tmp_path / "ISA checkpoint"
    with patch("swarmfpga.opentpu.OpenTPUBackend.__init__",
               side_effect=AssertionError("physical backend constructor must never run")):
        full = SwarmModel(native_config, backend=ISABackend())
        partial = SwarmModel(native_config, backend=ISABackend())
        reference = SwarmModel(reference_config)
        expected_metrics = reference.train(rows, steps=3)
        full_metrics = full.train(rows, steps=3)
        split_metrics = partial.train(rows, steps=2)
        assert partial.cursor == 1
        partial.save(checkpoint)
        original_checkpoint = checkpoint.read_bytes()
        restored = SwarmModel.load(checkpoint, backend=ISABackend())
        assert restored.config == partial.config
        assert restored.step == partial.step == 2
        assert restored.cursor == 1
        assert restored.dataset_digest == partial.dataset_digest
        for name, value in partial.weights.items():
            np.testing.assert_array_equal(restored.weights[name], value, err_msg=name)
        split_metrics.extend(restored.train(rows, steps=1))
        assert restored.step == full.step == 3
        assert restored.cursor == full.cursor == 0
        assert restored.dataset_digest == full.dataset_digest == reference.dataset_digest
        assert checkpoint.read_bytes() == original_checkpoint
        for name, value in full.weights.items():
            assert value.dtype == np.float32
            assert np.isfinite(value).all()
            np.testing.assert_array_equal(restored.weights[name], value, err_msg=f"exact replay: {name}")
            np.testing.assert_allclose(value, reference.weights[name], atol=3e-5, rtol=5e-4,
                                       err_msg=f"reference parity: {name}")
        assert [metric["step"] for metric in split_metrics] == [1, 2, 3]
        np.testing.assert_array_equal([metric["loss"] for metric in split_metrics],
                                      [metric["loss"] for metric in full_metrics])
        np.testing.assert_allclose([metric["loss"] for metric in full_metrics],
                                   [metric["loss"] for metric in expected_metrics], atol=3e-5, rtol=5e-4)
        assert full_metrics[-1]["loss"] < full_metrics[0]["loss"]
        assert all(metric["device"] == "f2" and metric["tokens"] == 6 for metric in full_metrics)

        # Full-prefix greedy decoding deliberately avoids the KV cache used by
        # generate. Compare both uninterrupted and restored native models with it.
        prompt = np.array([[0, 1], [1, 2]], np.int64)
        expected_tokens = prompt.copy()
        for _ in range(3):
            logits, _ = reference.logits(expected_tokens)
            expected_tokens = np.concatenate((expected_tokens, logits[:, -1].argmax(-1)[:, None]), axis=1)
        np.testing.assert_array_equal(full.generate(prompt, 3), expected_tokens)
        np.testing.assert_array_equal(restored.generate(prompt, 3), expected_tokens)
        np.testing.assert_allclose(restored.logits(expected_tokens)[0], reference.logits(expected_tokens)[0],
                                   atol=3e-5, rtol=5e-4)
        for model in (full, partial, restored):
            backend = model.backend
            assert backend.board.launches == backend.kernel_count > 100
            assert backend.device_cycles == 0  # simulator counts are not device measurements
            if forward == "int8":
                assert all(backend.instruction_counts[op] > 0 for op in ("MM", "QST", "QACT"))
            if backward != "fp32":
                assert backend.op_counts[f"round_{backward}"] > 0
