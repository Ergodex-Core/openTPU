"""The packed MXU (otpu_mxu PACK, docs/mxu_packing.md 6 / 7) against the ISA simulator, bit for
bit: a PACK build computes int8 (W8) MMs two weight rows a pass (2 MCOLS logical columns a
cycle) and the other formats unpacked, one product a DSP with its lane select; nothing an MM
writes changes. At MCOLS 4 and 8 physical with ACT_ROWS up to 64: the random programs of
tests/test_rtl.py (int8 half the MMs, int4 / fp4 / NVFP4 with PAIR, UNIT, RSC, ACC, RMAX,
ASCALE, replay at M > MCOLS, odd and even N, KB 1..4, mixed formats back to back through the
MXU's two heads), directed W8 MMs (every mode, k0, RSC, odd N, a replayed pair, KB 1 passes back
to back: the result FIFO's two rows a pass), tests/test_quant_rtl.py's 4-bit and NVFP4 MM cases
and tests/test_t2_rtl.py's mixed-format program on a PACK build, the ACT RAM read counts of a
packed command, and -128 in the weights: exact where docs/mxu_packing.md 3.4 says (a -128 in
the pair's first row; a -128 in the second beside a first-row weight >= 0), and the one
pattern the quantizer never produces pinned as inexact (-128 in the second row beside a
negative first-row weight at the same position: the 27-bit pre-adder wraps, the second row's
product reads as +128's)."""
import functools

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu.isasim import Machine

import test_quant_rtl as QR
import test_rtl as TR
import test_t2_rtl as T2
from test_quant_rtl import _both, _put, _xrows

X, WA, SA, OUT, AL = 0, 0x40000, 0x80000, 0x8000, 0xB0000


def _check(cfg, prog, imgs):
    """The program on both; DRAM and TMEM identical."""
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [i.copy() for i in imgs])
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    return m, st


@pytest.mark.parametrize("mcols,rows,seed", [(4, 4, 0), (4, 8, 1), (8, 8, 2), (4, 16, 3),
                                              (8, 16, 4), (4, 64, 5), (8, 64, 6)])
def test_fuzz_pack(have_verilator, mcols, rows, seed):
    """tests/test_rtl.py's random programs on a PACK build: int8 MMs (packed) and 4-bit /
    NVFP4 MMs with PAIR (unpacked, one lane) with UNIT / RSC / ACC / RMAX / ASCALE, M up to
    ACT_ROWS (replayed pairs), N 1..23 (odd: the last row a pair alone), KB 1..4, VOPs, QST and
    loads between them; the results are the ISA's."""
    rng = np.random.default_rng(5000 + 10 * seed + rows)
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows, PACK=True)
    prog = TR._random_program(rng, cfg, n_ops=60)
    mms = [i for i in prog if i.op == I.MM]
    assert any(I.mm_format(i)[0] == I.W8 for i in mms)
    _check(cfg, prog, TR._images(rng, 1))
    if rows > mcols:                                  # (the seeds cover replay: say so)
        print(f"replayed MMs: {sum((i.w[5] >> 16) & 0xFF > mcols for i in mms)} of {len(mms)}")


# wf, D, mcols, pair, rsc, k0, E: the quant tests' 4-bit and NVFP4 cases (PAIR, RSC, k0, the
# epilogue lanes); unpacked on a PACK build
@pytest.mark.parametrize("case", [c for c in QR.MM_CASES if c[0] != I.W8])
def test_other_formats_on_a_pack_build(have_verilator, monkeypatch, case):
    monkeypatch.setattr(QR, "Config", functools.partial(Config, PACK=True))
    QR.test_mm_mx_rtl(True, *case)


@pytest.mark.parametrize("mcols,rows,wf,D", [(2, 6, I.W4F, 32), (4, 12, I.WNV, 32),
                                             (4, 16, I.WNV, 128)])
def test_other_formats_replay_on_a_pack_build(have_verilator, monkeypatch, mcols, rows, wf, D):
    monkeypatch.setattr(QR, "Config", functools.partial(Config, PACK=True))
    QR.test_mm_mx_replay_rtl(True, mcols, rows, wf, D)


def test_mixed_formats_on_a_pack_build(have_verilator):
    """tests/test_t2_rtl.py's mixed program on a PACK + T2 build, M = 1 at MCOLS 4: QACT DUP4,
    then int4 PAIR, t2 PAIR, t2 QUAD, t2, int8 (packed: its own weight rows in the quantizer's
    range; the t2 test's random bytes read as int8 hold -128 beside negative weights, the
    inexact pattern), fp4 PAIR, int8 with ACC and RMAX, t2 QUAD with ACC and a plain t2 MM of M
    = 4 back to back: packed and unpacked commands alternate through the MXU's two heads."""
    D, KB, N = 32, 6, 13
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=True, T2=True, PACK=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(31)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, 1, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = KB * D
    dram[WA:WA + N * rs] = rng.integers(0, 256, N * rs, dtype=np.uint8)
    W8A = WA + 0x8000                                   # the int8 rows, [-127, 127]
    _put(dram, W8A, _w8(rng, N, KB, D))
    srs = T2._t2_scales(rng, dram, N, KB + 4, big=False)
    for n in range(N):
        dram[SA + n * srs + 4:SA + n * srs + 16] = rng.integers(0, 12, 12, dtype=np.uint8)
    mm = dict(sa=WA, ssa=SA, n=N, kb=KB, rs=rs, ors=N, ab=0, srs=srs)
    m8 = dict(mm, sa=W8A)
    prog = [I.ld(X, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup4=True),
            I.mm(out=OUT, m=1, wf=I.W4I, pair=True, **mm),
            I.mm(out=OUT + 64, m=1, wf=I.WT2, pair=True, **mm),
            I.mm(out=OUT + 128, m=1, wf=I.WT2, pair=True, quad=True, **mm),
            I.mm(out=OUT + 192, m=1, wf=I.WT2, k0=1, **mm),
            I.mm(out=OUT + 256, m=1, wf=I.W8, **m8),
            I.mm(out=OUT + 320, m=1, wf=I.W4F, pair=True, k0=2, **mm),
            I.mm(out=OUT + 256, m=1, wf=I.W8, k0=2, acc=True, rmax=True, **m8),
            I.mm(out=OUT + 128, m=1, wf=I.WT2, pair=True, quad=True, k0=4, acc=True, **mm),
            I.mm(out=OUT + 384, m=4, wf=I.WT2, **mm),
            I.mm(out=OUT + 448, m=4, wf=I.W8, unit=True, **m8),
            I.halt()]
    _both(cfg, prog, dram)


def _w8(rng, N, KB, D, lo=-127, hi=127):
    """N int8 weight rows of KB blocks in [lo, hi] (the quantizer's range by default)."""
    return rng.integers(lo, hi + 1, (N, KB * D)).astype(np.int8)


def _w8_scales(rng, N, KBr, big=True):
    """int8 MX scale rows at SA: Sw, a dw byte a block (random offsets, most small), padded
    to 32 bytes. Returns srs."""
    srs = 4 + 4 * -(-KBr // 4)
    srs = (srs + 31) // 32 * 32
    rows = rng.integers(0, 256, (N, srs), dtype=np.uint8)
    sw = (rng.standard_normal(N) * 0.01).astype(np.float32)
    dw = np.where(rng.random((N, KBr)) < 0.8, rng.integers(0, 6, (N, KBr)),
                  rng.integers(0, 20 if big else 8, (N, KBr)))
    rows[:, :4] = sw[:, None].view(np.uint8)
    rows[:, 4:4 + KBr] = dw
    return rows, srs


def _w8_image(rng, cfg, M, N, KB, k0=0, extra=1, wa=None):
    """A DRAM image with M activation rows (x at X), N int8 weight rows at WA (rs = (KB +
    extra) D) and their scale rows at SA; returns (dram, rs, srs)."""
    D = cfg.D
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, M, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = (KB + extra) * D
    w = _w8(rng, N, KB, D) if wa is None else wa
    dram[WA:WA + N * rs] = rng.integers(0, 256, N * rs, dtype=np.uint8)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + KB * D] = w[n].view(np.uint8)
    srows, srs = _w8_scales(rng, N, k0 + KB)
    _put(dram, SA, srows)
    return dram, rs, srs, x


@pytest.mark.parametrize("mcols,M,N,KB,k0", [(4, 4, 11, 3, 1), (4, 1, 2, 1, 0), (4, 3, 1, 5, 2),
                                             (8, 8, 16, 4, 0), (8, 5, 7, 2, 3), (4, 2, 23, 1, 0)])
def test_w8_modes_pack(have_verilator, mcols, M, N, KB, k0):
    """int8 MMs on a PACK build in every mode: MX scales at k0, UNIT with RMAX, ACC with RMAX
    onto the first, UNIT + ACC + ASCALE, RSC; N odd (the last row a pair alone) and even, N 1
    (one row), KB 1 (one advance a pass: two result rows a cycle pair, the pop's bubble)."""
    D = 32
    cfg = Config(S=1, D=D, MCOLS=mcols, PACK=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(100 * N + 10 * KB + M + k0 + mcols)
    dram, rs, srs, x = _w8_image(rng, cfg, M, N, KB, k0)
    alpha = rng.uniform(-1.5, 1.5, M).astype(np.float32)
    _put(dram, AL, alpha)
    rsc = (rng.standard_normal(N) * 0.05).astype(np.float32)
    _put(dram, SA + 0x1000, rsc)
    mm = dict(sa=WA, n=N, kb=KB, rs=rs, ors=N + 1, m=M, ab=1, srs=srs)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M), I.qact(0, M, 1, KB, KB * D),
            I.mm(ssa=SA, out=OUT, k0=k0, **mm),
            I.mm(ssa=SA, out=OUT + 4096, **mm, unit=True, rmax=True),
            I.mm(ssa=SA, out=OUT, k0=k0, **mm, acc=True, rmax=True),
            I.mm(ssa=0, out=OUT + 4096, **mm, unit=True, acc=True, ascale=0x7000),
            I.mm(ssa=SA + 0x1000, out=OUT + 8192, **{**mm, "srs": 4}, rsc=True, rmax=True),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("mcols,rows,N,KB", [(4, 8, 9, 4), (4, 16, 6, 3), (8, 64, 5, 2),
                                             (4, 64, 4, 1)])
def test_w8_replay_pack(have_verilator, mcols, rows, N, KB):
    """M > MCOLS on a PACK build: each pair of int8 rows is streamed once and consumed for
    every group of MCOLS rows (its two chunks a slot of the two-bank FIFO); the drain steps
    +1 to the pair's second row and gs - 1 to the next group; ACT_ROWS up to 64."""
    D = 32
    cfg = Config(S=1, D=D, MCOLS=mcols, ACT_ROWS=rows, PACK=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(300 + rows + N)
    dram, rs, srs, x = _w8_image(rng, cfg, rows, N, KB, 1, extra=0)
    prog = [I.ld(X, 0, x.size), I.qact(0, rows, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, k0=1),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, acc=True),
            I.mm(WA, SA, OUT + 4096, N, KB, rs, N, rows, 0, srs, unit=True),
            I.halt()]
    _both(cfg, prog, dram)


def test_kb1_passes_back_to_back_pack(have_verilator):
    """KB 1 int8 MMs of many rows (K = D: a pass an advance; the attention scores' shape) on
    a PACK build, a 4-bit MM of KB 1 and an odd-N int8 MM behind them (the MXU's two heads: the
    pop head moves on only between commands the pipeline treats alike), then a wide one."""
    D, mcols = 32, 4
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=True, PACK=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(11)
    N = 40
    dram, rs, srs, x = _w8_image(rng, cfg, mcols, N, 1)
    w4 = I.W4I
    rows4 = TR._mx_scale_rows(w4, N)
    _put(dram, SA + 0x4000, rows4)
    dram[WA + 0x8000:WA + 0x8000 + N * D] = rng.integers(0, 256, N * D, dtype=np.uint8)
    prog = [I.ld(X, 0, x.size), I.qact(0, mcols, 0, 1, D),
            I.mm(WA, SA, OUT, N, 1, rs, N, mcols, 0, srs),
            I.mm(WA, SA, OUT + 1024, N, 1, rs, N, mcols, 0, srs, unit=True, rmax=True),
            I.mm(WA + 0x8000, SA + 0x4000, OUT + 2048, N, 1, D, N, mcols, 0, 64, wf=w4),
            I.mm(WA, SA, OUT + 3072, N - 1, 1, rs, N, mcols, 0, srs),
            I.mm(WA, SA, OUT + 1024, N, 1, rs, N, mcols, 0, srs, acc=True, rmax=True),
            I.mm(WA, SA, OUT, N, 1, rs, N, 2, 0, srs, acc=True),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("mcols,rows", [(4, 8), (8, 16)])
@pytest.mark.parametrize("M,N", [(1, 20), (3, 7), (9, 20)])
def test_act_reads_follow_the_packed_mm_rows(have_verilator, mcols, rows, M, N):
    """A packed int8 MM reads the ACT RAM once a pass (ceil(N / 2) x KB x groups pops, half
    an unpacked MM's), the columns of its output rows only (test_rtl's act_reads at PACK); the
    results are the ISA's."""
    if M > rows:
        pytest.skip("M <= ACT_ROWS")
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=rows, PACK=True)
    D, KB = cfg.D, 3
    prog = [I.ld(TR.DATA, 0, 4096), I.qact(0, M, 0, KB, KB * D),
            I.mm(TR.INT8, TR.MX_BASE[I.W8], 4096, N, KB, KB * D, N, M, 0, TR.MX_PITCH[I.W8]),
            I.halt()]
    img = TR._images(np.random.default_rng(7), 1)[0]
    _, st = _check(cfg, prog, [img])
    a = st["act_reads"][0]
    G = -(-M // mcols)
    cols = mcols if G > 1 else min(mcols, M)
    last = M - (G - 1) * mcols
    per = -(-N // 2) * KB                               # a group's passes (pops)
    assert a["cols"] == mcols
    assert a["any"] == (1 << cols) - 1
    assert a["pop"] == per * G
    assert a["use"] == per * ((G - 1) * mcols + last)
    assert a["rd"] % cols == 0 and a["use"] <= a["rd"] <= a["cyc"] * cols


def test_minus_128_weights_pack(have_verilator):
    """-128 in int8 weights on a PACK build (docs/mxu_packing.md 3.4; quant.py never writes
    one, a host might). Exact: every weight of a pair's first row -128 (the low lane: |L| <=
    16 x 128 x 127 < 2^18); -128 in the second row beside first-row weights >= 0 (the pre-adder
    in range); then the pattern that is not: -128 in the second row beside a negative first-row
    weight at the same position wraps the 27-bit pre-adder (wa + wb 2^19 < -2^26) and the
    second row's product reads as (+128) x act. Pinned: with every position so (w_a = -1, w_b
    = -128, UNIT, the activations -127: x = -1.0 rows), the second row's outputs are the ISA's
    negated exactly, the first row's and every other row's the ISA's."""
    D, mcols, KB = 32, 4, 2
    cfg = Config(S=1, D=D, MCOLS=mcols, PACK=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(128)
    N = 6
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, mcols, KB * D, D // 4, 3.0)
    x[1] = -1.0                                         # row 1 quantizes to -127 everywhere
    _put(dram, X, x)
    rs = KB * D
    # rows 0 / 1: the first row all -128, the second random; rows 2 / 3: the second -128 at
    # random positions beside a first row >= 0; rows 4 / 5: the inexact pattern
    w = _w8(rng, N, KB, D)
    w[0] = -128
    w[2] = rng.integers(0, 128, KB * D)
    w[3][rng.random(KB * D) < 0.5] = -128
    w[4] = -1
    w[5] = -128
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + KB * D] = w[n].view(np.uint8)
    srows, srs = _w8_scales(rng, N, KB, big=False)
    _put(dram, SA, srows)
    exact = [I.ld(X, 0, x.size), I.qact(0, mcols, 0, KB, KB * D),
             I.mm(WA, SA, OUT, 4, KB, rs, 8, mcols, 0, srs),
             I.mm(WA, SA, OUT + 64, 4, KB, rs, 8, mcols, 0, srs, unit=True, rmax=True),
             I.halt()]
    _both(cfg, exact, dram)
    bad = [I.ld(X, 0, x.size), I.qact(0, mcols, 0, KB, KB * D),
           I.mm(WA, SA, OUT, N, KB, rs, 8, mcols, 0, srs, unit=True),
           I.halt()]
    m = Machine(cfg, [bad], [dram.copy()]).run()
    _, tmems, _ = rtlsim.run(cfg, [bad], [dram.copy()])
    ref, got = m.slices[0].tmem.copy(), tmems[0]
    idx = OUT + 8 * np.arange(mcols)[:, None] + np.arange(N)[None, :]
    # the ISA's row 5: every product (-128) x act; the RTL's (+128) x act, whatever act: the
    # integer negated, so the fp32 output negated exactly (the sign bit)
    flip = ref[idx[:, 5]] ^ 0x80000000
    assert np.all(ref[idx[:, 5]] & 0x7FFFFFFF)                 # (nonzero outputs)
    assert np.array_equal(got[idx[:, 5]], flip), (got[idx[:, 5]], ref[idx[:, 5]])
    ref[idx[:, 5]] = flip
    assert np.array_equal(got, ref)
