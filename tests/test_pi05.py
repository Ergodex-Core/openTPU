"""pi0.5 (opentpu/llm/pi05.py, docs/pi05.md) on a tiny random-weight model: the numpy
reference's pieces against lerobot's definitions, and the card's phases on the ISA simulator
(int8 weights and operands) against the reference."""
from __future__ import annotations

import numpy as np
import pytest

from opentpu.llm import pi05 as P

TINY = P.Spec(v_hidden=256, v_heads=4, v_ffn=300, v_layers=2, image=56, hidden=256, layers=2,
              ffn=1024, e_hidden=256, e_layers=2, e_ffn=512, n_q=2, head_dim=256, chunk=10,
              steps=3)


def tiny_weights(s: P.Spec, rng) -> dict:
    """Random tensors under the checkpoint's names, scaled like a trained model's (unit-ish
    activations through every layer)."""
    W = {}

    def lin(name, n, k, bias=True, scale=1.0):
        W[name + ".weight"] = (scale * rng.standard_normal((n, k)) / np.sqrt(k)).astype(
            np.float32)
        if bias:
            W[name + ".bias"] = (0.1 * rng.standard_normal(n)).astype(np.float32)

    def vec(name, n, mu=0.0, sd=0.1):
        W[name] = (mu + sd * rng.standard_normal(n)).astype(np.float32)
    C, F = s.v_hidden, s.v_ffn
    for i in range(s.v_layers):
        p = f"{P.TOWER}encoder.layers.{i}."
        for n in "qkv":
            lin(p + f"self_attn.{n}_proj", C, C)
        lin(p + "self_attn.out_proj", C, C, scale=0.5)
        lin(p + "mlp.fc1", F, C)
        lin(p + "mlp.fc2", C, F, scale=0.5)
        for n in ("layer_norm1", "layer_norm2"):
            vec(p + n + ".weight", C, 1.0)
            vec(p + n + ".bias", C)
    k = s.patch
    W[P.TOWER + "embeddings.patch_embedding.weight"] = \
        (rng.standard_normal((C, 3, k, k)) / 24).astype(np.float32)
    vec(P.TOWER + "embeddings.patch_embedding.bias", C)
    W[P.TOWER + "embeddings.position_embedding.weight"] = \
        (0.5 * rng.standard_normal((s.n_img, C))).astype(np.float32)
    vec(P.TOWER + "post_layernorm.weight", C, 1.0)
    vec(P.TOWER + "post_layernorm.bias", C)
    lin(P.PG + "multi_modal_projector.linear", s.hidden, C)
    d = s.head_dim
    for pre, H, Fx, L in ((P.LLM, s.hidden, s.ffn, s.layers),
                          (P.EXP, s.e_hidden, s.e_ffn, s.e_layers)):
        for i in range(L):
            p = f"{pre}layers.{i}."
            lin(p + "self_attn.q_proj", s.n_q * d, H, False)
            lin(p + "self_attn.k_proj", d, H, False)
            lin(p + "self_attn.v_proj", d, H, False)
            lin(p + "self_attn.o_proj", H, s.n_q * d, False, 0.5)
            lin(p + "mlp.gate_proj", Fx, H, False)
            lin(p + "mlp.up_proj", Fx, H, False)
            lin(p + "mlp.down_proj", H, Fx, False, 0.5)
            if pre == P.LLM:
                vec(p + "input_layernorm.weight", H)
                vec(p + "post_attention_layernorm.weight", H)
            else:
                for n in ("input_layernorm", "post_attention_layernorm"):
                    lin(p + n + ".dense", 3 * H, H, scale=0.3)
    lin(P.EXP + "norm.dense", 3 * s.e_hidden, s.e_hidden, scale=0.3)
    W[P.LLM + "embed_tokens.weight"] = rng.standard_normal((64, s.hidden)).astype(
        np.float32) * np.float32(s.hidden ** -0.5)
    E = s.e_hidden
    lin(P.TOP + "action_in_proj", E, s.action_dim)
    lin(P.TOP + "action_out_proj", s.action_dim, E)
    lin(P.TOP + "time_mlp_in", E, E)
    lin(P.TOP + "time_mlp_out", E, E)
    return W


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    W = tiny_weights(TINY, rng)
    imgs = [rng.uniform(-1, 1, (3, TINY.image, TINY.image)).astype(np.float32)
            for _ in range(TINY.cameras)]
    tokens = np.array([2, 5, 9, 33, 17, 4, 0, 0])
    mask = tokens != 0
    noise = rng.standard_normal((TINY.chunk, TINY.action_dim)).astype(np.float32)
    return W, imgs, tokens, mask, noise


@pytest.fixture(scope="module")
def card(tiny):
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import IsaBackend
    W = tiny[0]
    cfg0 = board_config()
    image = P.Image(TINY, cfg0, np_max=64)
    cfg = board_config(DRAM_BYTES=image.end)
    image.cfg = cfg
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)

    def put(addr, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    image.build(W, put)
    return image, IsaBackend(cfg, [img])


def test_time_embedding_is_lerobots():
    torch = pytest.importorskip("torch")
    from math import pi
    s = P.Spec()
    for t in P.times(s):
        frac = torch.linspace(0.0, 1.0, s.e_hidden // 2, dtype=torch.float64)
        period = s.min_period * (s.max_period / s.min_period) ** frac
        x = (1.0 / period * 2 * pi)[None, :] * torch.tensor([t], dtype=torch.float32)[:, None]
        ref = torch.cat([torch.sin(x), torch.cos(x)], 1).float()[0].numpy()
        assert np.array_equal(P.time_embedding(s, t), ref)


def test_patches_are_the_convolution():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    img = rng.standard_normal((3, 56, 56)).astype(np.float32)
    w = rng.standard_normal((8, 3, 14, 14)).astype(np.float32)
    ref = torch.nn.functional.conv2d(torch.from_numpy(img[None]), torch.from_numpy(w),
                                     stride=14)[0].reshape(8, -1).T.numpy()
    got = P.patches(img) @ w.reshape(8, -1).T
    assert np.abs(got - ref).max() < 1e-4


def test_the_expert_step_program_fits_imem(card):
    image, _ = card
    progs = image.programs(14)
    for name, (prog, _) in progs.items():
        assert len(prog) <= image.cfg.IMEM_WORDS // 8, (name, len(prog))


def _cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def test_the_chunk_on_the_card_is_the_reference(tiny, card):
    """The whole tiny policy on the ISA simulator (vision, prefix, 3 Euler steps, int8):
    the projector rows, and the actions, close to the fp32 reference."""
    W, imgs, tokens, mask, noise = tiny
    image, be = card
    prompt = P.prompt_rows(TINY, W, tokens, mask)
    Np = TINY.cameras * TINY.n_img + len(prompt)
    progs = image.programs(Np)
    cap = {}
    ref = P.reference_actions(TINY, W, imgs, tokens, mask, noise, capture=cap)
    P.run_chunk(image, be, progs, imgs, prompt, noise, stop="vision")
    vis = be.read(0, image.work["PX"], 4 * 2 * TINY.n_img * TINY.hidden).view(
        np.float32).reshape(2 * TINY.n_img, TINY.hidden)
    rv = np.concatenate([cap["vis0"], cap["vis1"]])
    assert _cos(vis, rv).min() > 0.995
    got = P.run_chunk(image, be, progs, imgs, prompt, noise)
    err = np.abs(got - ref).max()
    assert err < 0.05 * np.abs(ref).max(), err
