"""Ternary-Bonsai 2's GGUF side (opentpu/llm/gguf_file.py, opentpu/llm/bonsai2.py; docs/ternary.md
4): the prism ternary packings decode to their codes (a tiny GGUF written here, the packings
as ggml-quants.c's reference quantizers write them), the Hadamard matrix and the activation
transform, and llama.cpp's tiled value-head order undone."""
import struct

import numpy as np
import pytest

from opentpu.llm import bonsai2 as B2
from opentpu.llm import gguf_file as G


def _pq2(q, d):
    """PQ2_0 blocks of codes q [N, K] in {-1, 0, 1} and scales d [N, K/128]: fp16 d, then the
    2-bit codes q + 1 (element j at byte j/4, bits 2 (j % 4))."""
    N, K = q.shape
    c = (q + 1).astype(np.uint8).reshape(N, K // 128, 32, 4)
    qs = c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6
    dd = d.astype(np.float16).view(np.uint8).reshape(N, K // 128, 2)
    return np.concatenate([dd, qs], -1).reshape(-1)


def _ptq1(q, d):
    """PTQ1_0 blocks as quantize_row_ptq1_0_ref writes them: 24 bytes of 5 trits (stages of 16
    then 8 bytes, a stage's elements trit-major), 2 bytes of 4 trits, fp16 d."""
    N, K = q.shape
    out = []
    for blk, dv in zip((q + 1).reshape(-1, 128), d.reshape(-1)):
        x, qs = [int(t) for t in blk], []
        for c in (16, 8):
            for m in range(c):
                v = 0
                for n in range(5):
                    v = v * 3 + x[m + n * c]
                qs.append((v * 256 + 242) // 243)
            x = x[5 * c:]
        qh = []
        for h in range(2):
            v = 0
            for m in range(4):
                v = v * 3 + x[h + m * 2]
            qh.append((v * 3 * 256 + 242) // 243)
        out.append(np.array(qs + qh, np.uint8))
        out.append(np.array([dv], np.float16).view(np.uint8))
    return np.concatenate(out)


def _write_gguf(path, tensors, meta):
    """A GGUF v3 of tensors {name: (type, shape numpy order, raw bytes)} and string / int meta."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        head += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                        else struct.pack("<II", 4, v))
    off, blobs = 0, []
    for name, (typ, shape, raw) in tensors.items():
        head += s(name) + struct.pack("<I", len(shape)) + \
            b"".join(struct.pack("<Q", n) for n in reversed(shape)) + struct.pack("<IQ", typ, off)
        pad = (-len(raw)) % 32
        blobs.append(raw.tobytes() + b"\0" * pad)
        off += len(raw) + pad
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + b"".join(blobs))


def test_ternary_packings_decode_to_their_codes(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.integers(-1, 2, (6, 384)).astype(np.int8)
    d = rng.uniform(0.005, 0.02, (6, 3)).astype(np.float16).astype(np.float32)
    f = rng.standard_normal((3, 64)).astype(np.float32)
    p = tmp_path / "t.gguf"
    _write_gguf(p, {"a": (G.PQ2_0, (6, 384), _pq2(q, d)),
                    "b": (G.PTQ1_0, (6, 384), _ptq1(q, d)),
                    "c": (G.F32, (3, 64), f)}, {"general.architecture": "qwen35"})
    g = G.GGUF(p)
    assert g.meta["general.architecture"] == "qwen35"
    for n in ("a", "b"):
        qq, ss, gs = g[n].ternary()
        assert gs == 128 and np.array_equal(qq, q) and np.array_equal(ss, d)
        want = (q.reshape(6, 3, 128) * d[..., None]).reshape(6, 384)
        assert np.array_equal(g[n].dequant(), want.astype(np.float32))
        assert np.array_equal(g[n].ternary(rows=[4, 1])[0], q[[4, 1]])
    assert np.array_equal(g["c"].array(), f)


def test_hadamard_is_the_normalized_sylvester_matrix():
    H = B2.hadamard(8)
    assert np.allclose(H @ H.T, np.eye(8)) and np.allclose(H, H.T)
    S = np.array([[1.0]])
    for _ in range(3):
        S = np.block([[S, S], [S, -S]])
    assert np.allclose(H, S / np.sqrt(8))


def test_unfold_is_the_activation_transform():
    """W_f R x = unfold(W_f) x: the folded weight times the rotated input is the model's."""
    rng = np.random.default_rng(1)
    w = B2.Weights.__new__(B2.Weights)
    w.block, w.H = 64, B2.hadamard(64).astype(np.float32)
    w._signs = {256: np.where(rng.random(256) < 0.5, -1.0, 1.0).astype(np.float32)}
    Wf = rng.standard_normal((5, 256)).astype(np.float32)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    assert np.allclose(w.rotate(x) @ Wf.T, x @ w.unfold(Wf).T, atol=1e-4)


def test_untile_inverts_llama_cpp_reorder():
    nk, rep, hd = 3, 2, 4
    grouped = np.arange(nk * rep * hd * 2).reshape(nk * rep * hd, 2)
    # convert_hf_to_gguf's _reorder_v_heads: [nk, rep, hd] -> [rep, nk, hd]
    tiled = grouped.reshape(nk, rep, hd, 2).swapaxes(0, 1).reshape(-1, 2)
    assert np.array_equal(B2.untile(tiled, nk, rep, hd), grouped)
    assert np.array_equal(B2.untile(tiled.T, nk, rep, hd, axis=1), grouped.T)


@pytest.mark.parametrize("fmt", ["int4", "t2"])
@pytest.mark.parametrize("B,K", [(512, 1024), (1024, 2048), (4096, 4096)])
def test_rotate_on_the_isa_is_the_transform(fmt, B, K):
    """kernels.hadamard.rotate (x * s', an MM by H_512, butterflies for B > 512) against
    H_B (s * x) in float64: within the MM's int8 activations (QACT per row of 512)."""
    import dataclasses

    from opentpu import Config, language as ol
    from opentpu.kernels.hadamard import BASE, Rotation, rotate
    from opentpu.runtime import Input, Output, Weight, launch

    rng = np.random.default_rng(B + K)
    M = 2
    x = rng.standard_normal((M, K)).astype(np.float32)
    s = np.where(rng.random(K) < 0.5, -1.0, 1.0).astype(np.float32)
    H0 = np.where(B2.hadamard(BASE) > 0, 1.0, -1.0).astype(np.float32)

    @ol.jit
    def k(x, h, sg, out):
        rot = Rotation(block=B, base=BASE, H=h, signs={K: sg.base}, onall=None)
        ol.store(out, rotate(ol.load(x), rot))

    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=64, ACT_ROWS=16, TMEM_WORDS=1 << 16,
                 PAIR=True, T2=True)
    got = launch(k, cfg, x=Input(x), h=Weight(H0, fmt=fmt), sg=Input(s * B ** -0.5),
                 out=Output((M, K))).outputs["out"]
    want = ((x * s).reshape(M, K // B, B) @ B2.hadamard(B)).reshape(M, K)
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max()
