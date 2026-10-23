"""t2 weights (docs/ternary.md): 2-bit elements c - 1, four per byte, int4's scale words after a
16-byte header; MM QUAD (column reuse x4, ACT rows j + c M from QACT DUP4). Here the format's
quantizer (exact on ternary blocks, the bytes of llama.cpp's PQ2_0), the ISA simulator's MM
against the format's own math, PAIR and QUAD changing no bit, the encoding and its checks."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu.isasim import Machine, SimError

WA, SA, XA, OUT = 16384, 49152, 1 << 17, 8192


def _ternary(rng, N, K, D=128):
    """A ternary matrix with one scale per block of D: (W f32 [N, K], codes, scales)."""
    q = rng.integers(-1, 2, (N, K)).astype(np.int8)
    s = rng.uniform(0.005, 0.02, (N, K // D)).astype(np.float32)
    return (q.reshape(N, -1, D) * s[..., None]).reshape(N, K).astype(np.float32), q, s


def _setup(W, x, fmt="t2", D=128):
    wrows, srows = Q.quantize_mxu(W, fmt, D)
    assert WA + wrows.size <= SA and SA + srows.size <= XA
    dram = np.zeros(1 << 19, np.uint8)
    dram[XA:XA + x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + wrows.size] = wrows.reshape(-1)
    dram[SA:SA + srows.size] = srows.reshape(-1)
    return wrows, srows, dram


def _run(cfg, dram, prog):
    cfg = dataclasses.replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, dram.size))
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


def test_t2_quantizer_is_exact_on_ternary_codes_and_packs_pq2_0():
    rng = np.random.default_rng(1)
    W, q, s = _ternary(rng, 64, 1024)
    rows, srows = Q.quantize_mxu(W, "t2")
    assert rows.shape == (64, Q.row_bytes(1024, "t2")) == (64, 256)
    assert srows.shape[1] == Q.scale_bytes(1024, "t2") == 16 + 16 * 2
    assert np.array_equal(Q.elements(rows, "t2", 1024), q)
    # PQ2_0's qs: element j of a block at byte j / 4, bits 2 (j % 4), code q + 1
    assert np.array_equal(rows, Q.pack2((q + 1).astype(np.uint8)))
    deq = Q.dequantize_mxu(rows, srows, "t2", 1024)
    assert np.abs(deq - W).max() <= np.abs(W).max() / 64     # a 6-bit multiplier per block


def test_unpack2_pack2_roundtrip():
    c = np.random.default_rng(2).integers(0, 4, (3, 64)).astype(np.uint8)
    assert np.array_equal(Q.unpack2(Q.pack2(c)), c)


def test_encoding_is_wf1_with_rs_bits():
    ins = I.mm(WA, SA, OUT, 4, 8, 256, 4, 1, 0, 48, wf=I.WT2, pair=True, quad=True)
    assert (ins.flags >> I.WF_SHIFT) & 3 == I.W4I and ins.w[4] == 256 | I.RS_T2 | I.RS_QUAD
    assert I.mm_format(ins) == (I.WT2, True, True)
    assert I.mm_format(I.mm(WA, SA, OUT, 4, 8, 512, 4, 1, 0, 48, wf=I.W4I)) == \
        (I.W4I, False, False)
    with pytest.raises(ValueError):
        I.mm(WA, SA, OUT, 4, 8, 256, 4, 1, 0, 48, wf=I.WT2, quad=True)        # QUAD needs PAIR
    with pytest.raises(ValueError):
        I.mm(WA, SA, OUT, 4, 8, 256, 4, 1, 0, 48, wf=I.WT2, pair=True, quad=True, k0=2)


@pytest.mark.parametrize("KB", [1, 3, 4, 8])
def test_t2_mm_matches_the_format_math(KB):
    """MM's integer sum with t2 elements is the docs' loop: c = sum act * w per group, B = m <<
    (W - dw - da), y = (i2f(acc) * Sw) * (Sa * 2^-W); against quant.mx_matmul on the format's
    own decoded weights."""
    D, N, M = 128, 9, 2
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, T2=True)
    rng = np.random.default_rng(10 + KB)
    W, _, _ = _ternary(rng, N, KB * D)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    wrows, srows, dram = _setup(W, x)
    s = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D),
                         I.mm(WA, SA, OUT, N, KB, wrows.shape[1], N, M, 0, srows.shape[1],
                              wf=I.WT2)])
    got = s.tget(OUT + np.arange(M * N)).reshape(M, N).astype(np.float64)
    ref = Q.fake_act(x) @ Q.fake_w(W, "t2", D, (wrows, srows)).T
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


@pytest.mark.parametrize("KB", [1, 4, 5, 8])
@pytest.mark.parametrize("mode", ["pair", "quad"])
def test_pair_and_quad_change_no_bit(KB, mode):
    D, N, M = 128, 7, 1
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, T2=True)
    rng = np.random.default_rng(100 + KB)
    W, _, _ = _ternary(rng, N, KB * D)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    wrows, srows, dram = _setup(W, x)
    rs, srs = wrows.shape[1], srows.shape[1]
    plain = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=I.WT2)])
    quad = mode == "quad"
    fast = _run(cfg, dram, [I.ld(XA, 0, x.size),
                            I.qact(0, M, 0, KB, KB * D, dup=not quad, dup4=quad),
                            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=I.WT2, pair=True,
                                 quad=quad)])
    a, b = (r.tget(OUT + np.arange(M * N)) for r in (plain, fast))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_quad_takes_block_k_from_row_j_plus_k_mod_4_times_m():
    """ACT rows c M + j hold x times a sign per c: QUAD's output is the plain MM of x with block
    k's sign that of row (k mod 4) M + j, bit for bit."""
    D, N, M, KB = 128, 5, 1, 8
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, T2=True)
    rng = np.random.default_rng(7)
    W, _, _ = _ternary(rng, N, KB * D)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    sg = np.array([1, -1, -1, 1], np.float32)
    blk = sg[(np.arange(KB * D) // D) % 4]
    rows = np.concatenate([x * sg[c] for c in range(4)] + [x * blk])
    wrows, srows, dram = _setup(W, rows)
    rs, srs = wrows.shape[1], srows.shape[1]
    got = _run(cfg, dram, [I.ld(XA, 0, rows.size), I.qact(0, 4 * M, 0, KB, KB * D),
                           I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=I.WT2, pair=True,
                                quad=True)]).tget(OUT + np.arange(N))
    want = _run(cfg, dram, [I.ld(XA, 0, rows.size),
                            I.qact(4 * M * KB * D, M, 0, KB, KB * D),
                            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=I.WT2)]
                ).tget(OUT + np.arange(N))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_t2_needs_the_capability_and_quad_fits_mcols():
    D, N, KB = 128, 3, 4
    rng = np.random.default_rng(3)
    W, _, _ = _ternary(rng, N, KB * D)
    x = rng.standard_normal((2, KB * D)).astype(np.float32)
    wrows, srows, dram = _setup(W, x)
    rs, srs = wrows.shape[1], srows.shape[1]
    no = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True)
    with pytest.raises(SimError):
        _run(no, dram, [I.ld(XA, 0, x.size), I.qact(0, 1, 0, KB, KB * D),
                        I.mm(WA, SA, OUT, N, KB, rs, N, 1, 0, srs, wf=I.WT2)])
    yes = dataclasses.replace(no, T2=True)
    _run(yes, dram, [I.ld(XA, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup4=True)])
    with pytest.raises(SimError):            # 4 x 2 rows > MCOLS
        _run(yes, dram, [I.ld(XA, 0, x.size), I.qact(0, 2, 0, KB, KB * D, dup4=True)])


def test_column_parts_of_whole_chunks_are_the_whole_rows():
    rng = np.random.default_rng(4)
    W, _, _ = _ternary(rng, 8, 2048)
    rows, srows = Q.quantize_mxu(W, "t2")
    whole = Q.dequantize_mxu(rows, srows, "t2", 2048)
    for C in (512, 1024):
        for j, (pr, ps) in enumerate(Q.column_parts(rows, srows, "t2", 2048, C)):
            assert pr.shape[1] == Q.row_bytes(C, "t2") and ps.shape[1] == Q.scale_bytes(C, "t2")
            assert np.array_equal(Q.dequantize_mxu(pr, ps, "t2", C), whole[:, j * C:(j + 1) * C])
    with pytest.raises(ValueError):          # a part starts on a whole chunk of 4 blocks
        Q.column_parts(rows, srows, "t2", 2048, 256)
