"""t2 weights, MM QUAD and QACT DUP4 on the RTL (a T2 build, otpu_mxu / otpu_actram /
otpu_quant T2; docs/ternary.md section 6) against the ISA simulator, bit for bit: plain t2
(four advances a chunk), PAIR (two), QUAD (one, four columns a row), KB not a multiple of 4, k0
offsets, M 1 and 2, UNIT / ACC / RMAX / ASCALE, a replayed t2 row (M > MCOLS), random reserved
header words and padding (never read), and QUAD's row choice with distinct ACT rows. Then the
other formats' MMs on the T2 build (the t2 terms constant 0 for them) and a program mixing all
formats, PAIR and QUAD back to back (the MXU's two heads). Each runs the same program on both
and compares DRAM and TMEM."""
import dataclasses
import functools

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine, SimError

import test_quant_rtl as QR
from test_quant_rtl import _both, _put, _xrows

X, WA, SA, OUT, AL = 0, 0x40000, 0x80000, 0x8000, 0xB0000


def _t2_weights(rng, dram, N, KB, D, extra=1):
    """Random 2-bit codes [N, KB D] packed at WA, rows rs = (ceil(KB/4) + extra) D apart, the
    rest of each row's chunks random bytes. Returns rs."""
    rs = (-(-KB // 4) + extra) * D
    dram[WA:WA + N * rs] = rng.integers(0, 256, N * rs, dtype=np.uint8)
    data = Q.pack2(rng.integers(0, 4, (N, KB * D)))
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    return rs


def _t2_scales(rng, dram, N, KBr, big=True):
    """t2 scale rows at SA: Sw, three reserved words (random), a word per block (dw, four 6-bit
    m), padding to whole 16-byte groups (random). Returns srs."""
    srs = 16 + 16 * -(-KBr // 4)
    rows = rng.integers(0, 256, (N, srs), dtype=np.uint8)
    sw = (rng.standard_normal(N) * 0.01).astype(np.float32)
    dw = rng.integers(0, 20 if big else 4, (N, KBr))
    m = rng.integers(0, 64, (N, KBr, 4))
    word = (dw | sum(m[:, :, b] << (8 + 6 * b) for b in range(4))).astype(np.uint32)
    rows[:, :4] = sw[:, None].view(np.uint8)
    rows[:, 16:16 + 4 * KBr] = word.view(np.uint8).reshape(N, -1)
    _put(dram, SA, rows)
    return srs


# mode, KB, k0, M, MCOLS
T2_CASES = [
    ("plain", 5, 1, 4, 4), ("plain", 3, 0, 2, 8), ("plain", 8, 3, 1, 4),
    ("pair", 5, 2, 2, 4), ("pair", 2, 0, 1, 4), ("pair", 7, 4, 4, 8),
    ("quad", 5, 4, 1, 4), ("quad", 8, 0, 1, 4), ("quad", 1, 0, 1, 4),
    ("quad", 6, 4, 2, 8), ("quad", 3, 0, 2, 8), ("quad", 9, 8, 1, 8),
]


@pytest.mark.parametrize("mode,KB,k0,M,mcols", T2_CASES)
def test_t2_mm_rtl(have_verilator, mode, KB, k0, M, mcols):
    """QACT (DUP for PAIR, DUP4 for QUAD), then t2 MMs over its rows: with scales, UNIT with
    RMAX, ACC with RMAX onto the first, UNIT with ACC and ASCALE."""
    D = 32
    pair, quad = mode != "plain", mode == "quad"
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=True, T2=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(1000 * KB + 100 * k0 + 10 * M + mcols + 7 * pair + 3 * quad)
    N = 11
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, M, KB * D, D // 4)
    _put(dram, X, x)
    rs = _t2_weights(rng, dram, N, KB, D)
    srs = _t2_scales(rng, dram, N, k0 + KB)
    alpha = rng.uniform(-1.5, 1.5, M).astype(np.float32)
    _put(dram, AL, alpha)
    mm = dict(sa=WA, n=N, kb=KB, rs=rs, ors=N + 1, m=M, ab=1, srs=srs, wf=I.WT2, pair=pair,
              quad=quad, k0=k0)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M),
            I.qact(0, M, 1, KB, KB * D, dup=pair and not quad, dup4=quad),
            I.mm(ssa=SA, out=OUT, **mm),
            I.mm(ssa=SA, out=OUT + 4096, **mm, unit=True, rmax=True),
            I.mm(ssa=SA, out=OUT, **mm, acc=True, rmax=True),
            I.mm(ssa=0, out=OUT + 4096, **mm, unit=True, acc=True, ascale=0x7000),
            I.halt()]
    _both(cfg, prog, dram)


def test_t2_mm_replay_rtl(have_verilator):
    """t2 with M > MCOLS: each streamed chunk (four blocks) and its scale entry serve both
    groups of MCOLS rows."""
    D, KB, N, rows = 32, 6, 9, 8
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=rows, T2=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(77)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, rows, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = _t2_weights(rng, dram, N, KB, D, extra=0)
    srs = _t2_scales(rng, dram, N, KB + 3, big=False)
    prog = [I.ld(X, 0, x.size), I.qact(0, rows, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, wf=I.WT2, k0=3),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, wf=I.WT2, acc=True),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("M,mcols", [(1, 4), (2, 8)])
def test_quad_row_choice_rtl(have_verilator, M, mcols):
    """QUAD over 4M distinct ACT rows (one QACT; rows c M + j = x_j times a sign per c, so every
    row has x_j's Sa and offsets): block k comes from row j + (k mod 4) M, as the ISA's."""
    D, KB, N = 32, 7, 6
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=True, T2=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(5 + M)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, M, KB * D, D // 4, 3.0)
    sg = np.array([1, -1, -1, 1], np.float32)
    rows = np.concatenate([x * sg[c] for c in range(4)])
    _put(dram, X, rows)
    rs = _t2_weights(rng, dram, N, KB, D)
    srs = _t2_scales(rng, dram, N, KB, big=False)
    prog = [I.ld(X, 0, rows.size), I.qact(0, 4 * M, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=I.WT2, pair=True, quad=True),
            I.halt()]
    _both(cfg, prog, dram)


def test_t2_mixed_formats_rtl(have_verilator):
    """One program, M = 1 at MCOLS 4: QACT DUP4 (rows 0..3 the same), then int4 PAIR, t2 PAIR,
    t2 QUAD, t2, int8, fp4 PAIR, t2 QUAD with ACC and a plain t2 MM of M = 4 over the four
    copies, back to back (consecutive PAIR MMs of one M, and QUAD after QUAD, share the
    pipeline: the MXU's two heads). The weight and scale bytes are the same random bytes read as
    each format."""
    D, KB, N = 32, 6, 13
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=True, T2=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(31)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, 1, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = KB * D                                     # wide enough for int8's rows
    dram[WA:WA + N * rs] = rng.integers(0, 256, N * rs, dtype=np.uint8)
    srs = _t2_scales(rng, dram, N, KB + 4, big=False)
    # every format's view of the same scale rows: int4's words from byte 8, int8's bytes from 4
    for n in range(N):
        dram[SA + n * srs + 4:SA + n * srs + 16] = rng.integers(0, 12, 12, dtype=np.uint8)
    mm = dict(sa=WA, ssa=SA, n=N, kb=KB, rs=rs, ors=N, ab=0, srs=srs)
    prog = [I.ld(X, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup4=True),
            I.mm(out=OUT, m=1, wf=I.W4I, pair=True, **mm),
            I.mm(out=OUT + 64, m=1, wf=I.WT2, pair=True, **mm),
            I.mm(out=OUT + 128, m=1, wf=I.WT2, pair=True, quad=True, **mm),
            I.mm(out=OUT + 192, m=1, wf=I.WT2, k0=1, **mm),
            I.mm(out=OUT + 256, m=1, wf=I.W8, **mm),
            I.mm(out=OUT + 320, m=1, wf=I.W4F, pair=True, k0=2, **mm),
            I.mm(out=OUT + 128, m=1, wf=I.WT2, pair=True, quad=True, k0=4, acc=True, **mm),
            I.mm(out=OUT + 384, m=4, wf=I.WT2, **mm),
            I.halt()]
    _both(cfg, prog, dram)


@pytest.mark.parametrize("case", [c for c in QR.MM_CASES if c[1] == 32])
def test_other_formats_on_a_t2_build_rtl(have_verilator, monkeypatch, case):
    """tests/test_quant_rtl.py's MM cases at D = 32 (int8, int4, fp4, NVFP4; PAIR, RSC, k0) on
    a T2 build: the t2 terms are constant 0 for them, every bit as without T2."""
    monkeypatch.setattr(QR, "Config", functools.partial(Config, T2=True))
    QR.test_mm_mx_rtl(True, *case)


def test_t2_rules():
    """RSC does not take t2; QUAD needs PAIR, k0 % 4 == 0 and 4 M <= MCOLS, and on a one-chunk
    port B only t2 (an int4 QUAD streams two chunks a cycle: Config.PORTB 2, docs/wide_port.md;
    the simulator refuses the rate); DUP4 needs T2 (or a wide port) and 4 rows <= MCOLS (the
    RTL's simulation checks stop on the same: otpu_mxu)."""
    with pytest.raises(ValueError):
        I.mm(WA, SA, OUT, 4, 4, 32, 4, 1, 0, 32, wf=I.WT2, rsc=True)
    with pytest.raises(ValueError):                 # QUAD without PAIR
        I.mm(WA, SA, OUT, 4, 4, 32, 4, 1, 0, 32, wf=I.W4I, quad=True)
    cfg = Config(S=1, D=32, MCOLS=4, PAIR=True, T2=True, DRAM_BYTES=1 << 20)
    assert cfg.PORTB == 1
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[SA:SA + 4] = np.array([1.0], np.float32).view(np.uint8)
    x = np.ones((2, 128), np.float32)
    _put(dram, X, x)
    base = [I.ld(X, 0, x.size), I.qact(0, 2, 0, 4, 128, dup=True)]
    with pytest.raises(SimError):                   # 4 M > MCOLS
        Machine(cfg, [base + [I.mm(WA, SA, OUT, 1, 4, 32, 1, 2, 0, 32, wf=I.WT2, pair=True,
                                   quad=True), I.halt()]], [dram.copy()]).run()
    with pytest.raises(SimError):                   # DUP4 of 2 rows: 8 > MCOLS
        Machine(cfg, [[I.ld(X, 0, x.size), I.qact(0, 2, 0, 4, 128, dup4=True), I.halt()]],
                [dram.copy()]).run()
    with pytest.raises(SimError):                   # without T2
        Machine(dataclasses.replace(cfg, T2=False),
                [base + [I.mm(WA, SA, OUT, 1, 4, 32, 1, 1, 0, 32, wf=I.WT2), I.halt()]],
                [dram.copy()]).run()
    with pytest.raises(SimError, match="PORTB 1"):  # int4 QUAD: two chunks a cycle
        Machine(cfg, [base + [I.mm(WA, SA, OUT, 1, 4, 32, 1, 1, 0, 32, wf=I.W4I, pair=True,
                                   quad=True), I.halt()]], [dram.copy()]).run()


@pytest.mark.parametrize("mode", ["quad", "pair4"])
def test_quad_takes_a_chunk_a_cycle_rtl(have_verilator, mode):
    """One-row decode: QUAD consumes a D-byte chunk (four t2 blocks) a cycle, its four scale
    words one port A line request a chunk; int4 PAIR, the reference, a chunk (two blocks) a
    cycle. Measured as the cycles 64 more weight rows add (the behavioural DRAM)."""
    D, KB = 32, 16
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=True, T2=True, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(9)
    dram = rng.integers(0, 256, cfg.DRAM_BYTES, dtype=np.uint8)
    x = rng.standard_normal((1, KB * D)).astype(np.float32)
    _put(dram, X, x)
    quad = mode == "quad"
    bpc = 4 if quad else 2
    rs = -(-KB // bpc) * D
    srs = 16 + 16 * -(-KB // 4) if quad else 8 + 4 * KB
    cyc, areq = [], []
    for N in (64, 128):
        mm = I.mm(WA, SA, OUT, N, KB, rs, 1, 1, 0, srs, wf=I.WT2 if quad else I.W4I, pair=True,
                  quad=quad)
        prog = [I.ld(X, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup=not quad, dup4=quad), mm,
                I.halt()]
        _, _, st = rtlsim.run(cfg, [prog], [dram.copy()])
        cyc.append(st["cycles"])
        areq.append(st["mxu_scale"][0]["areq"])
    chunks = 64 * KB // bpc
    assert cyc[1] - cyc[0] == chunks, cyc
    assert areq[1] - areq[0] == chunks, areq
