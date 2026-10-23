"""End-to-end model tests against independent loss and replay calculations.

All numerical execution here uses the explicitly selected reference backend.
These tests are not evidence that a native F2 implementation exists or works.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from swarmfpga.backend import ReferenceBackend
from swarmfpga.config import Config
from swarmfpga.runtime import SwarmModel


def configuration(**trainer):
    return {
        "model": {"type": "causal_lm", "vocab_size": 5, "context": 8,
                  "width": 4, "layers": 1, "heads": 2, "ff_width": 8},
        "trainer": {"learning_rate": 0.05, "batch_size": 2,
                    "sequence_length": 3, "steps": 30, "seed": 17, **trainer},
        "backend": {"type": "reference"},
    }


def cross_entropy(logits, targets):
    """Independent float64 logsumexp; no framework loss/autograd is used."""
    logits = np.asarray(logits, np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    selected = np.take_along_axis(log_probs, np.asarray(targets)[..., None], axis=-1)
    return -float(selected.mean())


class FrameworkTest(unittest.TestCase):
    def make_model(self, **trainer):
        return SwarmModel(Config.from_dict(configuration(**trainer)))

    def setUp(self):
        self.tokens = np.array([[0, 1, 2], [2, 0, 1]], np.int64)
        self.targets = np.array([[1, 2, 0], [0, 1, 2]], np.int64)

    def loss(self, model):
        logits, _ = model.logits(self.tokens)
        return cross_entropy(logits, self.targets)

    def test_seed_reproduces_initialization(self):
        a, b, other = self.make_model(), self.make_model(), self.make_model(seed=18)
        self.assertEqual(set(a.weights), set(b.weights))
        for name in a.weights:
            np.testing.assert_array_equal(a.weights[name], b.weights[name])
        self.assertTrue(any(not np.array_equal(a.weights[name], other.weights[name])
                            for name in a.weights))

    def test_full_decoder_sgd_gradient_matches_finite_difference(self):
        model = self.make_model()
        # Increase non-norm weights to make attention/MLP gradient checks
        # sensitive to errors that tiny initialization could otherwise hide.
        for name, value in model.weights.items():
            if "norm" not in name:
                value *= 5
        before = {name: value.copy() for name, value in model.weights.items()}
        expected_loss = self.loss(model)
        numerical = {}
        epsilon = 1e-3
        # Check entries in every parameter family, including embeddings/norms.
        for name, weight in model.weights.items():
            indices = list(dict.fromkeys([0, weight.size // 2, weight.size - 1]))
            for flat_index in indices:
                index = np.unravel_index(flat_index, weight.shape)
                original = float(weight[index])
                weight[index] = original + epsilon
                positive = self.loss(model)
                weight[index] = original - epsilon
                negative = self.loss(model)
                weight[index] = original
                numerical[name, index] = (positive - negative) / (2 * epsilon)
        result = model.train_step(self.tokens, self.targets)
        self.assertAlmostEqual(result["loss"], expected_loss, places=5)
        self.assertEqual(result["step"], 1)
        for (name, index), expected in numerical.items():
            actual = float(before[name][index] - model.weights[name][index]) / 0.05
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=5e-3,
                                       err_msg=f"Decoder gradient {name}{index}")

    def test_training_decreases_independently_computed_loss(self):
        model = self.make_model()
        initial = self.loss(model)
        for _ in range(30):
            result = model.train_step(self.tokens, self.targets)
            self.assertTrue(np.isfinite(result["loss"]))
        self.assertLess(self.loss(model), 0.9 * initial)

    def test_checkpoint_resumes_exact_parameters_and_data_cursor(self):
        data = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2],
                         [3, 1, 3, 1], [4, 2, 4, 2]], np.int64)
        uninterrupted, split = self.make_model(), self.make_model()
        uninterrupted.train(data, steps=7)
        split.train(data, steps=3)
        with tempfile.TemporaryDirectory(prefix="swarm-checkpoint-") as directory:
            path = Path(directory) / "state.npz"
            split.save(path)
            restored = SwarmModel.load(path)
            self.assertEqual(restored.step, split.step)
            self.assertEqual(restored.cursor, split.cursor)
            self.assertEqual(restored.config, split.config)
            self.assertEqual(restored.dataset_digest, split.dataset_digest)
            for name in split.weights:
                np.testing.assert_array_equal(restored.weights[name], split.weights[name])
            restored.train(data, steps=4)
        self.assertEqual(restored.step, uninterrupted.step)
        self.assertEqual(restored.cursor, uninterrupted.cursor)
        for name in uninterrupted.weights:
            np.testing.assert_array_equal(restored.weights[name], uninterrupted.weights[name])
        np.testing.assert_array_equal(restored.logits(self.tokens)[0],
                                      uninterrupted.logits(self.tokens)[0])

    def test_resumed_training_rejects_changed_dataset_or_row_order(self):
        data = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], np.int64)
        model = self.make_model()
        model.train(data, steps=1)
        changed = data.copy()
        changed[0, 0] = 4
        with tempfile.TemporaryDirectory(prefix="swarm-dataset-") as directory:
            path = Path(directory) / "state.npz"
            model.save(path)
            for dataset in (data[::-1], changed):
                with self.subTest(dataset=dataset):
                    restored = SwarmModel.load(path)
                    before = {name: value.copy() for name, value in restored.weights.items()}
                    with self.assertRaises(ValueError):
                        restored.train(dataset, steps=1)
                    self.assertEqual(restored.step, model.step)
                    self.assertEqual(restored.cursor, model.cursor)
                    for name, value in before.items():
                        np.testing.assert_array_equal(restored.weights[name], value)

    def test_cached_decode_matches_full_prefix_with_batch_and_multiple_heads(self):
        for layers in (1, 2):
            with self.subTest(layers=layers):
                config = configuration()
                config["model"]["layers"] = layers
                model = SwarmModel(Config.from_dict(config))
                tokens = np.array([[0, 1, 2, 3, 4], [4, 2, 3, 1, 0]], np.int64)
                full, _ = model.logits(tokens)
                prefix, cache = model.logits(tokens[:, :2])
                second, cache = model.logits(tokens[:, 2:4], cache=cache)
                last, _ = model.logits(tokens[:, 4:], cache=cache)
                np.testing.assert_allclose(np.concatenate([prefix, second, last], axis=1), full,
                                           atol=2e-6, rtol=2e-5)

    def test_causality_prevents_future_tokens_changing_prefix_logits(self):
        model = self.make_model()
        first, _ = model.logits(np.array([[0, 1, 2, 3, 4]], np.int64))
        changed, _ = model.logits(np.array([[0, 1, 4, 2, 3]], np.int64))
        np.testing.assert_allclose(first[:, :2], changed[:, :2], atol=1e-6, rtol=1e-6)

    def test_training_invalidates_existing_cache(self):
        model = self.make_model()
        _, cache = model.logits(self.tokens[:, :2])
        model.train_step(self.tokens, self.targets)
        with self.assertRaises(ValueError):
            model.logits(self.tokens[:, 2:], cache=cache)

    def test_cache_from_another_model_is_rejected(self):
        model, other = self.make_model(), self.make_model()
        _, cache = model.logits(self.tokens[:, :2])
        with self.assertRaises(ValueError):
            other.logits(self.tokens[:, 2:], cache=cache)

    def test_cache_cannot_override_weights_or_token_inputs(self):
        model = self.make_model()
        _, cache = model.logits(self.tokens[:, :2])
        for name, value in [("head", np.zeros_like(model.weights["head"])),
                            ("tokens", np.zeros_like(self.tokens[:, 2:]))]:
            with self.subTest(name=name):
                corrupted = {**cache, "tensors": {**cache["tensors"], name: value}}
                # owner/version remain valid, so the tensor whitelist must reject.
                with self.assertRaises(ValueError):
                    model.logits(self.tokens[:, 2:], cache=corrupted)

    def test_cache_length_must_be_a_positive_integer(self):
        model = self.make_model()
        _, cache = model.logits(self.tokens[:, :2])
        for length in (0, -1, True, 2.0):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    model.logits(self.tokens[:, 2:], cache={**cache, "length": length})

    def test_checkpoint_rejects_invalid_state(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory(prefix="swarm-invalid-checkpoint-") as directory:
            path = Path(directory) / "state.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                saved = {name: archive[name].copy() for name in archive.files}
            name = next(iter(model.weights))
            bad_states = []
            metadata = json.loads(str(saved["metadata"]))
            metadata["step"] = -1
            bad_states.append({**saved, "metadata": np.array(json.dumps(metadata))})
            for digest in ("z" * 64, "0" * 63, 123):
                metadata = json.loads(str(saved["metadata"]))
                metadata["dataset_sha256"] = digest
                bad_states.append({**saved, "metadata": np.array(json.dumps(metadata))})
            bad_states.append({key: value for key, value in saved.items() if key != name})
            bad_states.append({**saved, name: saved[name].astype(np.float64)})
            bad_states.append({**saved, name: np.full(saved[name].shape, np.nan, np.float32)})
            bad_states.append({**saved, name: saved[name].reshape(-1)})
            for state in bad_states:
                with self.subTest(state=list(state)):
                    np.savez(path, **state)
                    with self.assertRaises(ValueError):
                        SwarmModel.load(path)

    def test_generate_matches_greedy_full_prefix_loop(self):
        model = self.make_model()
        prompt = self.tokens[:, :2]
        expected = prompt.copy()
        for _ in range(3):
            logits, _ = model.logits(expected)
            expected = np.concatenate([expected, logits[:, -1].argmax(-1)[:, None]], axis=1)
        np.testing.assert_array_equal(model.generate(prompt, max_new_tokens=3), expected)

    def test_legacy_fp32_checkpoint_with_empty_socket_still_loads(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                values = {name: archive[name].copy() for name in archive.files}
            metadata = json.loads(str(values["metadata"]))
            metadata["config"].pop("forward_dtype")
            metadata["config"].pop("backward_dtype")
            metadata["config"]["socket"] = ""
            np.savez(path, **{**values, "metadata": np.array(json.dumps(metadata))})
            restored = SwarmModel.load(path)
            np.testing.assert_array_equal(restored.logits(self.tokens)[0], model.logits(self.tokens)[0])

    def test_tokens_are_validated_without_silent_coercion(self):
        model = self.make_model()
        for tokens in [self.tokens.astype(np.float32), np.array([[5]]),
                       np.array([[-1]]), np.zeros((1, 9), np.int64)]:
            with self.subTest(tokens=tokens):
                with self.assertRaises((ValueError, TypeError)):
                    model.logits(tokens)

    def test_strict_configuration_rejects_unknown_and_inconsistent_fields(self):
        invalid = []
        for section in [None, "model", "trainer", "backend"]:
            value = configuration()
            (value if section is None else value[section])["typo"] = 1
            invalid.append(value)
        for section, key, value in [("model", "type", "bert"), ("model", "heads", 3),
                                     ("model", "layers", 0), ("model", "width", True),
                                     ("trainer", "learning_rate", -0.1),
                                     ("trainer", "learning_rate", float("nan")),
                                     ("trainer", "sequence_length", 9),
                                     ("trainer", "batch_size", 0),
                                     ("backend", "type", "gpu")]:
            config = configuration()
            config[section][key] = value
            invalid.append(config)
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises((ValueError, TypeError)):
                    Config.from_dict(config)

    def test_f2_unavailable_device_never_uses_reference_fallback(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        config = configuration()
        config["backend"] = {"type": "f2", "image": "burst200"}
        unavailable = SimpleNamespace(OpenTPUBackend=Mock(side_effect=OSError("device unavailable")))
        with patch.dict(sys.modules, {"swarmfpga.opentpu": unavailable}):
            with patch.object(ReferenceBackend, "run", side_effect=AssertionError("CPU fallback")):
                with self.assertRaises(OSError):
                    model = SwarmModel(Config.from_dict(config))
                    model.logits(self.tokens)

    def test_f2_cannot_be_overridden_with_reference_backend(self):
        config = configuration()
        config["backend"] = {"type": "f2", "image": "burst200"}
        with self.assertRaises(ValueError):
            SwarmModel(Config.from_dict(config), backend=ReferenceBackend())

    def test_native_backend_selector_is_strict_without_loading_hardware(self):
        config = configuration()
        config["backend"] = {"type": "f2", "image": "burst200"}
        selected = Config.from_dict(config)
        self.assertEqual(selected.backend, "f2")
        self.assertEqual(selected.image, "burst200")
        invalid_selectors = [
            {"type": "f2"},
            {"type": "f2", "image": ""},
            {"type": "f2", "image": "unrecognized-image"},
            {"type": "f2", "image": None},
            {"type": "f2", "image": "burst200", "socket": "/unused-worker.sock"},
            {"type": "reference", "image": "burst200"},
            {"type": "reference", "socket": "/unused-worker.sock"},
        ]
        for selector in invalid_selectors:
            with self.subTest(selector=selector):
                config["backend"] = selector
                with self.assertRaises(ValueError):
                    Config.from_dict(config)


if __name__ == "__main__":
    unittest.main()
