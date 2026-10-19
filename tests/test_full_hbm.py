"""v5 (docs/full_hbm.md): Gemma 3 sharded over S slices on the ISA simulator -- the logits of
a decode step on S = 2 and 4 slices against one slice's (the same integer dot products in the
same K order; the fp32 sums of the MLP's chunks in a different order, so allclose, and the same
token), on a tiny random model and on the real Gemma-3-1B."""
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import Config, Machine
from opentpu.llm import gemma3 as G3
from opentpu.llm import gemma4 as G
from opentpu.llm import load_spec, model_dir

D = 128


def tiny_spec(ffn: int = 640) -> G3.Spec:
    return G3.Spec(hidden=256, kinds=(G.SLIDE, G.FULL, G.SLIDE), kv_src=(0, 1, 2),
                   ffn=(ffn,) * 3, n_q=4, n_kv=1, head_dim=D, global_head_dim=D,
                   vocab=320, ple_dim=0, window=256, global_rot=D // 2, eps=1e-6,
                   softcap=None, attn_scale=D ** -0.5)


def close(a, b):
    """The S > 1 logits against one slice's: the same integers in the same K order except
    where the MLP's activation chunks are quantized with another per-chunk scale (a
    different chunking), so int8-level agreement: 1% of the logits' range, the same token."""
    np.testing.assert_allclose(b, a, rtol=0.05, atol=0.01 * float(np.abs(a).max()))
    assert int(np.argmax(a)) == int(np.argmax(b))
    cos = float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    assert cos > 0.9999, cos


def tiny_weights(spec: G3.Spec, seed: int = 7) -> dict:
    r = np.random.default_rng(seed)
    H, d = spec.hidden, spec.head_dim

    def mat(n, k, s=0.05):
        return (r.standard_normal((n, k)) * s).astype(np.float32)

    W = {"model.embed_tokens.weight": mat(spec.vocab, H, 1.0),
         "model.norm.weight": (r.standard_normal(H) * 0.1).astype(np.float32)}
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        for n in G._norms(0).values():
            W[p + n + ".weight"] = (r.standard_normal(H) * 0.1).astype(np.float32)
        W[p + "self_attn.q_norm.weight"] = (r.standard_normal(d) * 0.1).astype(np.float32)
        W[p + "self_attn.k_norm.weight"] = (r.standard_normal(d) * 0.1).astype(np.float32)
        W[p + "self_attn.q_proj.weight"] = mat(spec.n_q * d, H)
        W[p + "self_attn.k_proj.weight"] = mat(spec.n_kv * d, H)
        W[p + "self_attn.v_proj.weight"] = mat(spec.n_kv * d, H)
        W[p + "self_attn.o_proj.weight"] = mat(H, spec.n_q * d)
        W[p + "mlp.gate_proj.weight"] = mat(spec.ffn[i], H)
        W[p + "mlp.up_proj.weight"] = mat(spec.ffn[i], H)
        W[p + "mlp.down_proj.weight"] = mat(H, spec.ffn[i])
    return W


def v5_config(S: int, **kw) -> Config:
    """The v5 slice's ISA configuration (PORTB 4, row lanes, MCOLS 4, ALINE 64) on S slices."""
    return Config(S=S, D=D, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16, PAIR=True, PORTB=4,
                  ROWLANES=True, ALINE=64, DRAM_BYTES=1 << 24, **kw)


def run_steps(spec, W, cfg: Config, tokens: list, cap: int = 256, **ikw) -> list:
    """The logits of decode steps at positions 0 .. len(tokens) - 1 on cfg.S slices (one
    machine: the KV cache carries over), each [vocab] fp32."""
    img = spec.image(cfg, cap, **ikw)
    drams = img.build(W)
    assert len(drams) == cfg.S
    S, vl, io = cfg.S, img.v_loc, img.io
    m = None
    out = []
    for pos, tok in enumerate(tokens):
        progs = img.compile_step(pos)
        assert len(progs) == S
        for addr, v in img.host_inputs([tok], [pos]):
            b = np.ascontiguousarray(v, np.float32).view(np.uint8).ravel()
            for s in range(S):
                dr = drams[s] if m is None else m.slices[s].dram
                dr[addr:addr + b.size] = b
        m = Machine(cfg, progs, drams) if m is None else m.load(progs)
        m.run()
        lg = np.concatenate([m.slices[s].dram[io["logits"] + 4 * s * vl:
                                              io["logits"] + 4 * (s + 1) * vl].view(np.float32)
                             for s in range(S)])[:spec.vocab]
        out.append(lg.copy())
    return out


def test_tiny_gemma3_two_slices_bit_exact():
    """ffn 512: the MLP's chunks are 256 wide on one slice and on two (one chunk a slice,
    gathered into the same K order), so every integer and every fp32 sum is the same: the
    logits of five decode steps bit for bit."""
    spec = tiny_spec(512)
    W = tiny_weights(spec)
    toks = [2, 18, 72, 30, 5]
    one = run_steps(spec, W, v5_config(1), toks)
    two = run_steps(spec, W, v5_config(2), toks)
    ref = G.reference_logits(spec, W, toks)
    for p, (a, b) in enumerate(zip(one, two)):
        assert np.isfinite(a).all()
        assert np.array_equal(a, b), p
        cos = float((a * ref[p]).sum() / np.linalg.norm(a) / np.linalg.norm(ref[p]))
        assert cos > 0.98, (p, cos)


@pytest.mark.parametrize("S", [2, 4])
def test_tiny_gemma3_sliced_matches_one_slice(S):
    spec = tiny_spec()
    W = tiny_weights(spec)
    toks = [2, 18, 72, 30, 5]
    one = run_steps(spec, W, v5_config(1), toks)
    many = run_steps(spec, W, v5_config(S), toks)
    for a, b in zip(one, many):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        close(a, b)


def test_tiny_gemma3_sliced_plain_port():
    """The same on the one-chunk port without row lanes (S = 4: no row-lane layouts)."""
    spec = tiny_spec()
    W = tiny_weights(spec, 11)
    toks = [2, 44, 9]
    cfg1 = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, DRAM_BYTES=1 << 24)
    cfg4 = dataclasses.replace(cfg1, S=4)
    one, many = run_steps(spec, W, cfg1, toks), run_steps(spec, W, cfg4, toks)
    for a, b in zip(one, many):
        close(a, b)


def test_sharding_rules():
    spec = tiny_spec()
    sh = G._sharding(spec, 4)
    assert sh == {"nq_loc": 1, "kv_loc": 32, "h_loc": 64, "v_loc": 80}
    sh3 = G._sharding(spec, 3)                  # 4 heads and 128 K rows do not split 3 ways
    assert sh3["nq_loc"] == 4 and sh3["kv_loc"] == 128 and sh3["v_loc"] == 112
    assert G._ffn_pad(6912, "int8", D, 4) == 7168 and G._ffn_pad(6912, "int8", D, 1) == 6912
    assert G._ffn_pad(6912, "int8", D, 2) == 6912 and G._ffn_pad(6912, "int8", D, 3) == 6912
    with pytest.raises(ValueError):
        dataclasses.replace(spec, hidden=200).check(Config(S=4, D=D, MCOLS=4, ACT_BLOCKS=16))


@pytest.mark.slow
def test_gemma3_1b_sliced_step():
    """The real Gemma-3-1B: one decode step at position 9 on 4 slices against one (the
    receipt's configuration: int8, PORTB 4, row lanes, MCOLS 4, ALINE 64)."""
    path = model_dir("gemma3")
    if not (Path(path) / "config.json").exists():
        pytest.skip("models/gemma-3-1b-it is not here")
    from opentpu.llm.qwen3 import load_weights
    spec = load_spec(path)
    W = load_weights(path)
    toks = [2, 791, 792, 793, 794, 795, 796, 797, 798, 799]
    S = int(os.environ.get("OTPU_TEST_SLICES", "4"))
    kw = dict(wformat="int8", head_format="int8")
    one = run_steps(spec, W, dataclasses.replace(v5_config(1), ACT_BLOCKS=128, ACT_ROWS=8,
                                                 DRAM_BYTES=1 << 31), toks[:1], **kw)
    many = run_steps(spec, W, dataclasses.replace(v5_config(S), ACT_BLOCKS=128, ACT_ROWS=8,
                                                  DRAM_BYTES=1 << 31), toks[:1], **kw)
    close(one[0], many[0])
