"""A wide port B on the RTL (otpu_mxu / otpu_slice / otpu_dram / otpu_native_dram PORTB;
docs/wide_port.md "RTL") against the ISA simulator, bit for bit: MM column reuse for every
weight format at more than a chunk a cycle (int8 PAIR at PORTB 2, int8 QUAD at 4, int4 / fp4 /
NVFP4 QUAD at 2, t2 QUAD with T2 and a wide port), and the narrower reuses on a wide build. KB 1
to 9 (a row's last advance short of the reuse), k0 offsets, M 1 and 2, UNIT / ACC / RMAX /
ASCALE, scale rows at alignments that put a 4-bit / NVFP4 QUAD advance's scale data across two
port-A lines, a replayed row (M > MCOLS) on a wide build, QUAD's row choice on distinct ACT rows,
a program of every format and reuse back to back, and a compiled kernel. Each case runs on the
behavioural DRAM (one response of up to PORTB chunks) and on the board's memory path (the native
adapter on OTPU_NCH 4 / 8 channels with the native memory model's random stalls: a wide read on
PORTB pairs a cycle, in two parts where it needs a pair twice). Then the throughput: a one-row
int8 MM streams PORTB chunks a cycle."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

from test_mx import _scale_rows
from test_quant_rtl import _put, _xrows
from test_t2_rtl import _t2_scales, _t2_weights

X, WA, SA, OUT, AL = 0, 0x40000, 0x80000, 0x8000, 0xB0000
D = 128                                 # the board's memory path needs D = 128
MEMS = ["dram", "nch"]                  # the behavioural DRAM; the native adapter and model


def _run_both(cfg, prog, dram, mem, monkeypatch, nch=None):
    """The ISA simulator and the RTL on `mem` ("dram": otpu_dram; "nch": otpu_native_dram on
    `nch` channels, default 2 PORTB, at least 4); DRAM and TMEM compared. Returns the RTL's
    stats."""
    m = Machine(cfg, [prog], [dram.copy()]).run()
    if mem == "nch":
        monkeypatch.setenv("OTPU_NCH", str(nch or max(4, 2 * cfg.PORTB)))
    drams, tmems, st = rtlsim.run(cfg, [prog], [dram.copy()], axi=mem == "nch")
    bad = np.nonzero(tmems[0] != m.slices[0].tmem)[0]
    assert len(bad) == 0, (f"{len(bad)} TMEM words differ, first {bad[:6]}: RTL "
                           f"{[hex(int(v)) for v in tmems[0][bad[:6]]]} ISA "
                           f"{[hex(int(v)) for v in m.slices[0].tmem[bad[:6]]]}")
    bad = np.nonzero(drams[0] != m.slices[0].dram)[0]
    assert len(bad) == 0, f"{len(bad)} DRAM bytes differ, first at {bad[:6]}"
    return st


def _weights(rng, dram, fmt, N, KBr):
    """Random weight rows of KBr blocks at WA (int8 with -128, 4-bit nibbles, t2 codes): the
    row stride rs."""
    if fmt == "t2":
        return _t2_weights(rng, dram, N, KBr, D)
    if fmt == "int8":
        wel = rng.integers(-128, 128, (N, KBr * D))
        wel[:, ::7] = -128
        data = wel.astype(np.int8).view(np.uint8)
    else:
        data = Q.pack4(rng.integers(0, 16, (N, KBr * D)))
    rs = -(-data.shape[1] // D) * D
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    return rs


def _scales(rng, dram, fmt, N, KBr, sa):
    """Random scale rows at `sa` (offsets past the shift window too): srs."""
    if fmt == "t2":
        return _t2_scales(rng, dram, N, KBr)
    rows, *_ = _scale_rows(rng, Q.mxu_wf(fmt), N, KBr, False, True)
    _put(dram, sa, rows)
    return rows.shape[1]


# fmt, mode, PORTB, KB, k0, M, MCOLS, sa offset (a 4-bit / NVFP4 QUAD advance's scale bytes
# across a line at 8 / 16)
CASES = [
    ("int8", "pair", 2, 1, 0, 1, 4, 0), ("int8", "pair", 2, 5, 2, 1, 4, 0),
    ("int8", "pair", 2, 9, 0, 2, 4, 0),
    ("int8", "quad", 4, 1, 0, 1, 4, 0), ("int8", "quad", 4, 6, 4, 1, 4, 0),
    ("int8", "quad", 4, 9, 0, 1, 4, 0), ("int8", "quad", 4, 7, 4, 2, 8, 0),
    ("int4", "quad", 2, 3, 0, 1, 4, 0), ("int4", "quad", 2, 6, 4, 1, 4, 8),
    ("int4", "quad", 2, 9, 8, 2, 8, 8),
    ("fp4", "quad", 2, 5, 0, 1, 4, 8), ("fp4", "quad", 2, 8, 4, 1, 4, 0),
    ("nvfp4", "quad", 2, 5, 4, 1, 4, 0), ("nvfp4", "quad", 2, 9, 0, 1, 4, 16),
    ("int4", "pair", 2, 5, 2, 1, 4, 0),          # (one chunk a cycle on a wide build)
    ("int8", "pair", 4, 6, 2, 1, 4, 0),          # (two chunks a cycle on a port of four)
    ("fp4", "quad", 4, 7, 4, 1, 4, 8),
    ("int8", "plain", 2, 4, 1, 4, 4, 0),         # (no reuse on a wide build)
    ("t2", "quad", 2, 5, 4, 1, 4, 0), ("t2", "quad", 2, 9, 0, 1, 4, 0),
]


@pytest.mark.parametrize("mem", MEMS)
@pytest.mark.parametrize("fmt,mode,portb,KB,k0,M,mcols,soff", CASES)
def test_wide_mm_rtl(have_verilator, monkeypatch, mem, fmt, mode, portb, KB, k0, M, mcols, soff):
    """QACT (DUP for PAIR, DUP4 for QUAD), then MMs over its rows: with scales, UNIT with RMAX,
    ACC with RMAX onto the first, UNIT with ACC and ASCALE."""
    pair, quad = mode != "plain", mode == "quad"
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=True, T2=fmt == "t2", PORTB=portb,
                 DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(1000 * KB + 100 * k0 + 10 * M + mcols + 7 * pair + 3 * quad +
                                portb + soff + len(fmt))
    N = 11
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, M, KB * D, D // 4)
    _put(dram, X, x)
    rs = _weights(rng, dram, fmt, N, KB)
    srs = _scales(rng, dram, fmt, N, k0 + KB, SA + soff)
    alpha = rng.uniform(-1.5, 1.5, M).astype(np.float32)
    _put(dram, AL, alpha)
    mm = dict(sa=WA, n=N, kb=KB, rs=rs, ors=N + 1, m=M, ab=1, srs=srs, wf=Q.mxu_wf(fmt),
              pair=pair, quad=quad, k0=k0)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M),
            I.qact(0, M, 1, KB, KB * D, dup=pair and not quad, dup4=quad),
            I.mm(ssa=SA + soff, out=OUT, **mm),
            I.mm(ssa=SA + soff, out=OUT + 4096, **mm, unit=True, rmax=True),
            I.mm(ssa=SA + soff, out=OUT, **mm, acc=True, rmax=True),
            I.mm(ssa=0, out=OUT + 4096, **mm, unit=True, acc=True, ascale=0x7000),
            I.halt()]
    st = _run_both(cfg, prog, dram, mem, monkeypatch)
    if mem == "nch" and I.chunks_per_cycle(Q.mxu_wf(fmt), pair, quad) > 1 and \
            KB > I.blocks_per_chunk(Q.mxu_wf(fmt)):     # (a row of one chunk: one-chunk reads)
        assert st["native_b"][0]["wide"] > 0, st["native_b"]


@pytest.mark.parametrize("mem", MEMS)
def test_wide_mm_replay_rtl(have_verilator, monkeypatch, mem):
    """M > MCOLS on a wide build (each entry one chunk, replayed for both groups), then int8
    QUAD over the same weights from a DUP4 operand."""
    KB, N, rows = 5, 9, 8
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=rows, PAIR=True, PORTB=4, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(77)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, rows, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = _weights(rng, dram, "int8", N, KB)
    srs = _scales(rng, dram, "int8", N, KB + 3, SA)
    prog = [I.ld(X, 0, x.size), I.qact(0, rows, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, k0=3),
            I.mm(WA, SA, OUT, N, KB, rs, N + 3, rows, 0, srs, acc=True),
            I.qact(0, 1, 0, KB, KB * D, dup4=True),
            I.mm(WA, SA, OUT + 2048, N, KB, rs, N, 1, 0, srs, pair=True, quad=True),
            I.halt()]
    _run_both(cfg, prog, dram, mem, monkeypatch)


@pytest.mark.parametrize("M,mcols", [(1, 4), (2, 8)])
def test_wide_quad_row_choice_rtl(have_verilator, monkeypatch, M, mcols):
    """int8 QUAD over 4M distinct ACT rows (rows c M + j = x_j times a sign per c): block k comes
    from row j + (k mod 4) M, as the ISA's."""
    KB, N = 7, 6
    cfg = Config(S=1, D=D, MCOLS=mcols, PAIR=True, PORTB=4, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(5 + M)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, M, KB * D, D // 4, 3.0)
    sg = np.array([1, -1, -1, 1], np.float32)
    _put(dram, X, np.concatenate([x * sg[c] for c in range(4)]))
    rs = _weights(rng, dram, "int8", N, KB)
    srs = _scales(rng, dram, "int8", N, KB, SA)
    prog = [I.ld(X, 0, 4 * x.size), I.qact(0, 4 * M, 0, KB, KB * D),
            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, pair=True, quad=True),
            I.halt()]
    _run_both(cfg, prog, dram, "dram", monkeypatch)


@pytest.mark.parametrize("mem,nch", [("dram", None), ("nch", 8), ("nch", 16)])
def test_wide_mixed_formats_rtl(have_verilator, monkeypatch, mem, nch):
    """One program at PORTB 4 with T2, M = 1, MCOLS 4: QACT DUP4, then int8 QUAD, int8 PAIR,
    int4 QUAD, fp4 PAIR, NVFP4 QUAD, t2 QUAD, plain int8, int4 and NVFP4, int8 QUAD with ACC,
    a plain int8 MM of M = 4 over the four copies, int8 and fp4 QUAD with RSC, back to back (the MXU's two heads share
    the pipeline between QUAD MMs of one M whatever their port rates). The weight and scale
    bytes are the same random bytes read as each format; the scale rows' stride puts some
    4-bit / NVFP4 QUAD advances' scale data across two lines."""
    KB, N = 6, 13
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=True, T2=True, PORTB=4, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(31)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, 1, KB * D, D // 4, 3.0)
    _put(dram, X, x)
    rs = KB * D                                     # wide enough for int8's rows
    dram[WA:WA + N * rs] = rng.integers(0, 256, N * rs, dtype=np.uint8)
    srs = 16 + 16 * -(-(KB + 4) // 2) + 16          # every format's (NVFP4's the longest)
    rows = rng.integers(0, 12, (N, srs), dtype=np.uint8)
    rows[:, :4] = (rng.standard_normal((N, 1)) * 0.01).astype(np.float32).view(np.uint8)
    _put(dram, SA, rows)
    mm = dict(sa=WA, ssa=SA, n=N, kb=KB, rs=rs, ors=N, ab=0, srs=srs)
    prog = [I.ld(X, 0, x.size), I.qact(0, 1, 0, KB, KB * D, dup4=True),
            I.mm(out=OUT, m=1, wf=I.W8, pair=True, quad=True, **mm),
            I.mm(out=OUT + 64, m=1, wf=I.W8, pair=True, k0=2, **mm),
            I.mm(out=OUT + 128, m=1, wf=I.W4I, pair=True, quad=True, **mm),
            I.mm(out=OUT + 192, m=1, wf=I.W4F, pair=True, k0=2, **mm),
            I.mm(out=OUT + 256, m=1, wf=I.WNV, pair=True, quad=True, k0=4, **mm),
            I.mm(out=OUT + 320, m=1, wf=I.WT2, pair=True, quad=True, **mm),
            I.mm(out=OUT + 384, m=1, wf=I.W8, k0=1, **mm),
            I.mm(out=OUT + 448, m=1, wf=I.W4I, k0=1, **mm),
            I.mm(out=OUT + 512, m=1, wf=I.WNV, **mm),
            I.mm(out=OUT, m=1, wf=I.W8, pair=True, quad=True, k0=4, acc=True, **mm),
            I.mm(out=OUT + 576, m=4, wf=I.W8, **mm),
            I.mm(out=OUT + 640, m=1, wf=I.W8, pair=True, quad=True, rsc=True, **mm),
            I.mm(out=OUT + 704, m=1, wf=I.W4F, pair=True, quad=True, rsc=True, k0=4, **mm),
            I.halt()]
    _run_both(cfg, prog, dram, mem, monkeypatch, nch)


@pytest.mark.parametrize("portb,mem", [(2, "dram"), (4, "dram"), (2, "nch"), (4, "nch")])
def test_wide_compiled_kernel_rtl(have_verilator, monkeypatch, portb, mem):
    """A compiled one-row int8 dot (decode's MM: the compiler's QACT DUP4 and MM PAIR / QUAD at
    PORTB 2 / 4, tests/test_wide_port.py) on the RTL, bit for bit the ISA's."""
    from opentpu.runtime import Input, Output, Weight, compile_kernel
    from test_wide_port import k_dot
    N, K = 24, 9 * D
    rng = np.random.default_rng(3 + portb)
    x = rng.standard_normal((1, K)).astype(np.float32)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=portb, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(k_dot, cfg, x=Input(x), w=Weight(W), out=Output((1, N)))
    mms = [i for i in compiled.programs[0] if i.op == I.MM]
    assert mms and all(I.mm_format(i)[2 if portb == 4 else 1] for i in mms)
    _run_both(cfg, list(compiled.programs[0]), np.asarray(images[0], np.uint8), mem, monkeypatch)


@pytest.mark.parametrize("mem", ["dram", "nch"])
def test_wide_port_throughput_rtl(have_verilator, monkeypatch, mem):
    """One-row int8 decode (N = 256 rows of KB = 16 blocks): plain at PORTB 1, PAIR at 2, QUAD
    at 4 stream 1, 2, 4 chunks a cycle; measured as the cycles 256 more rows add (behavioural
    DRAM: exactly N KB / PORTB; the native path on 8 channels with the model's 20% stalls: each
    channel then needs PORTB / 4 of its commands, so PORTB 4 runs at the stalls' rate)."""
    KB = 16
    got = {}
    for portb in (1, 2, 4):
        cfg = Config(S=1, D=D, MCOLS=4, PAIR=True, PORTB=portb, DRAM_BYTES=1 << 21)
        rng = np.random.default_rng(9)
        dram = rng.integers(0, 256, cfg.DRAM_BYTES, dtype=np.uint8)
        x = rng.standard_normal((1, KB * D)).astype(np.float32)
        _put(dram, X, x)
        srs = 4 + KB
        quad, pair = portb == 4, portb >= 2
        cyc = []
        for N in (256, 512):
            mm = I.mm(0x20000, 0x1C0000, OUT, N, KB, KB * D, 1, 1, 0, srs, pair=pair, quad=quad)
            prog = [I.ld(X, 0, x.size),
                    I.qact(0, 1, 0, KB, KB * D, dup=pair and not quad, dup4=quad), mm, I.halt()]
            if mem == "nch":
                monkeypatch.setenv("OTPU_NCH", "8")
            _, _, st = rtlsim.run(cfg, [prog], [dram.copy()], axi=mem == "nch")
            cyc.append(st["cycles"])
        got[portb] = cyc[1] - cyc[0]
    print(f"throughput ({mem}): cycles for 256 rows x {KB} chunks: {got}")
    if mem == "dram":
        assert got == {1: 256 * KB, 2: 256 * KB // 2, 4: 256 * KB // 4}, got
    else:
        assert got[2] < 0.6 * got[1] and got[4] < 0.4 * got[1], got
