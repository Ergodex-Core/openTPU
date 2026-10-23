"""Independent numerical checks: no accelerator or network is required."""

import unittest

import numpy as np

from swarmfpga.backend import ReferenceBackend
from swarmfpga.graph import Graph


class GradientsTest(unittest.TestCase):
    def setUp(self):
        self.backend = ReferenceBackend()
        self.rng = np.random.default_rng(12)

    def random(self, shape):
        return (self.rng.normal(size=shape) * 0.3).astype(np.float32)

    def assert_gradients(self, graph, loss, params, feeds, atol=3e-3):
        gradients = graph.grad(loss, params)
        program = graph.export({"loss": loss, **gradients})
        actual = self.backend.run(program, feeds)
        for name, tensor in params.items():
            numerical = np.zeros(tensor.shape, dtype=np.float32)
            for index in np.ndindex(tensor.shape):
                plus = {k: np.array(v, copy=True) for k, v in feeds.items()}
                minus = {k: np.array(v, copy=True) for k, v in feeds.items()}
                plus[name][index] += 1e-3
                minus[name][index] -= 1e-3
                a = self.backend.run(program, plus)["loss"]
                b = self.backend.run(program, minus)["loss"]
                numerical[index] = (float(a) - float(b)) / 2e-3
            self.assertEqual(actual[name].shape, tensor.shape)
            np.testing.assert_allclose(actual[name], numerical, atol=atol, rtol=1e-2,
                                       err_msg=f"Incorrect gradient for {name}")

    def test_broadcast_arithmetic(self):
        graph = Graph()
        x, bias, scale = [graph.input(n, s) for n, s in
                          [("x", (2, 3)), ("bias", (3,)), ("scale", (2, 1))]]
        y = (x + bias) * scale
        loss = (y * y).mean()
        self.assert_gradients(graph, loss, {"x": x, "bias": bias, "scale": scale},
                              {"x": self.random(x.shape), "bias": self.random(bias.shape),
                               "scale": self.random(scale.shape)})

    def test_matmul_broadcasts_batch_and_head_dimensions(self):
        graph = Graph()
        x = graph.input("x", (2, 1, 2, 3))
        weight = graph.input("weight", (1, 3, 3, 2))
        loss = (x @ weight).tanh().sum()
        self.assert_gradients(graph, loss, {"x": x, "weight": weight},
                              {"x": self.random(x.shape), "weight": self.random(weight.shape)})

    def test_embedding_repeated_indices_accumulate(self):
        graph = Graph()
        weight = graph.input("weight", (5, 3))
        ids = graph.input("ids", (2, 3), dtype="int64")
        loss = weight.gather(ids).sum()
        grad = graph.grad(loss, {"weight": weight})["weight"]
        result = self.backend.run(graph.export({"grad": grad}),
                                  {"weight": self.random(weight.shape),
                                   "ids": np.array([[1, 1, 4], [0, 1, 4]])})
        expected = np.array([1, 3, 0, 0, 2], np.float32)[:, None] * np.ones((1, 3))
        np.testing.assert_array_equal(result["grad"], expected)

    def test_causal_attention_gradients_and_output(self):
        graph = Graph()
        q, k, v = [graph.input(n, (1, 2, 3, 2)) for n in ["q", "k", "v"]]
        mask = np.where(np.tril(np.ones((3, 3))), 0.0, -1e4).astype(np.float32)
        scores = (q @ k.transpose((0, 1, 3, 2))) / np.sqrt(2.0) + graph.const(mask.tolist())
        exps = (scores - scores.max(-1, keepdims=True)).exp()
        probability = exps / exps.sum(-1, keepdims=True)
        output = probability @ v
        feeds = {n: self.random(t.shape) for n, t in {"q": q, "k": k, "v": v}.items()}
        self.assert_gradients(graph, output.tanh().sum(), {"q": q, "k": k, "v": v}, feeds)
        actual = self.backend.run(graph.export({"output": output}), feeds)["output"]
        score_np = feeds["q"] @ np.swapaxes(feeds["k"], -1, -2) / np.sqrt(2.0) + mask
        probs_np = np.exp(score_np - np.max(score_np, axis=-1, keepdims=True))
        probs_np /= probs_np.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, probs_np @ feeds["v"], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual[..., 0, :], feeds["v"][..., 0, :], atol=1e-6)

    def test_rmsnorm_gradients(self):
        graph = Graph()
        x = graph.input("x", (2, 3))
        scale = graph.input("scale", (3,))
        normalized = x * ((x * x).mean(-1, keepdims=True) + 1e-3).rsqrt() * scale
        self.assert_gradients(graph, normalized.tanh().sum(), {"x": x, "scale": scale},
                              {"x": self.random(x.shape), "scale": self.random(scale.shape)})

    def test_log_softmax_loss_gradients(self):
        graph = Graph()
        logits = graph.input("logits", (2, 4))
        shifted = logits - logits.max(-1, keepdims=True)
        log_probs = shifted - shifted.exp().sum(-1, keepdims=True).log()
        target = graph.const([[0, 1, 0, 0], [0, 0, 0, 1]])
        loss = -(log_probs * target).sum() / 2
        self.assert_gradients(graph, loss, {"logits": logits},
                              {"logits": self.random(logits.shape)})

    def test_slice_concat_and_transpose_gradients(self):
        graph = Graph()
        x = graph.input("x", (2, 4))
        left = x.slice(1, 0, 3)
        right = x.slice(1, 1, 4)
        loss = left.concat(right, axis=1).transpose().reshape((12,)).tanh().sum()
        self.assert_gradients(graph, loss, {"x": x}, {"x": self.random(x.shape)})

    def test_scalar_broadcast_gradient(self):
        graph = Graph()
        x = graph.input("x", (2, 3))
        scalar = graph.input("scalar", ())
        self.assert_gradients(graph, ((x / scalar) + scalar).sum(), {"x": x, "scalar": scalar},
                              {"x": self.random(x.shape), "scalar": np.array(1.5, np.float32)})

    def test_gather_nonzero_axis_repeated_indices(self):
        graph = Graph()
        weight = graph.input("weight", (2, 4, 3))
        ids = graph.input("ids", (2,), dtype="int64")
        loss = weight.gather(ids, axis=1).sum()
        gradient = graph.grad(loss, {"weight": weight})["weight"]
        actual = self.backend.run(graph.export({"gradient": gradient}),
                                  {"weight": self.random(weight.shape), "ids": np.array([2, 2])})
        expected = np.zeros(weight.shape, np.float32)
        expected[:, 2, :] = 2
        np.testing.assert_array_equal(actual["gradient"], expected)

    def test_disconnected_parameter_has_zero_gradient(self):
        graph = Graph()
        x = graph.input("x", (2, 3))
        unused = graph.input("unused", (4,))
        gradient = graph.grad(x.sum(), {"unused": unused})["unused"]
        actual = self.backend.run(graph.export({"gradient": gradient}),
                                  {"x": self.random(x.shape), "unused": self.random(unused.shape)})
        np.testing.assert_array_equal(actual["gradient"], np.zeros(unused.shape, np.float32))


if __name__ == "__main__":
    unittest.main()
