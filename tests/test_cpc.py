"""The wide port, Config.CPC 2 (docs/v4/proposal-1-wide-port.md, docs/v4/p1-status.md): the RTL
at two chunks a cycle against the bit-exact ISA simulator on the kernels (the behavioural DRAM
and the board's memory path), on MX MMs in every weight format (int8 pairs; the half pair of an
odd KB; a pair's two scale bytes straddling a 32-byte line at an odd k0, in the middle and at a
command's last advance; RSC, UNIT, ACC, RMAX, ASCALE; replay over 16 ACT rows; 4-bit, PAIR and
NVFP4 at one chunk an advance; int8 and 4-bit commands back to back), and the analytic timing
(N x ceil(KB / CPC) an int8 MM; the flat model of Lens's candidate C)."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q, rtlsim
from opentpu.isasim import board_config
from opentpu.kernels import attention_decode, mlp
from opentpu.profile import _describe
from opentpu.runtime import collect, compile_kernel, run

from conftest import assert_same_state, rel
from test_kernels import attn_args, mlp_args
from test_mx import _scale_rows
from test_quant_rtl import AL, F32_CORNERS, OUT, SA, WA, X, _both, _put, _xrows


def _cfg2(**kw):
    """The card's configuration at CPC 2 (the f2 target's arch), the DRAM cut for a kernel."""
    return board_config(CPC=2, DRAM_BYTES=1 << 24, **kw)


def _uarch2():
    """The board's micro-architecture with the f2 target's chunk FIFO (512 entries of 2)."""
    return dict(rtlsim.BOARD_UARCH, FIFO_DEPTH=512)


@pytest.mark.parametrize("axi", [False, True], ids=["fixed", "axi"])
@pytest.mark.parametrize("kernel", ["mlp", "attn"])
def test_kernels_cpc2(have_verilator, kernel, axi):
    """MLP decode at KB 9 (hidden 1152: every row a half pair) and KB 36, and flash attention
    at d = 128 (KB 1 rows: half pairs alone): the RTL at CPC 2 equals the ISA simulator, DRAM
    and TMEM, on the behavioural DRAM and on the native memory path (random stalls, boot)."""
    cfg = _cfg2()
    if kernel == "mlp":
        args, want = mlp_args(np.random.default_rng(2), M=1, H=1152, Fd=4608)
        k = mlp
    else:
        args, want = attn_args(np.random.default_rng(3), Hq=8, Hkv=2, d=128, T=200, cap=256,
                               block=128)
        k = attention_decode
    comp, imgs = compile_kernel(k, cfg, **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    drams, tmems, st = rtlsim.run(cfg, comp.programs, [i.copy() for i in imgs],
                                  uarch=_uarch2(), axi=axi, boot=axi)
    rr = collect(comp, drams, tmems, st)
    assert_same_state(ri, rr)
    assert rel(rr.outputs["out"], want) < 0.03


# wf, KB, k0, pair, rsc, rows (ACT rows: M, replay above MCOLS 4), E
MM_CASES = [
    (I.W8, 9, 0, False, False, 1, 1),      # decode: M 1, half pairs
    (I.W8, 8, 0, False, False, 4, 4),      # whole pairs, M 4
    (I.W8, 3, 1, False, False, 4, 2),      # odd k0 (the pair's bytes at odd offsets)
    (I.W8, 2, 27, False, False, 4, 1),     # bytes 31 / 32: straddles, at the command's last
                                           # advance too (row 24's: 24 x 36 % 32 == 0)
    (I.W8, 5, 27, False, False, 2, 2),     # straddles mid-row (c 0), the half pair after
    (I.W8, 9, 0, False, True, 4, 4),       # RSC: Sw alone, pairs
    (I.W8, 4, 1, False, False, 12, 4),     # replay over three groups of 4 rows
    (I.W8, 9, 0, False, False, 16, 4),     # replay, half pairs, the FIFO's rows
    (I.W4I, 3, 1, False, False, 4, 2),     # 4-bit: one chunk an advance
    (I.W4F, 3, 2, True, False, 2, 1),      # PAIR
    (I.WNV, 3, 0, False, False, 4, 2),
    (I.WNV, 3, 2, True, False, 2, 2),
]


def _mm_prog(rng, cfg, wf, KB, k0, pair, rsc, rows):
    """QACT of `rows` rows, then MMs over them as test_quant_rtl's: with scales, UNIT + RMAX,
    ACC + RMAX, UNIT + ACC + ASCALE; Sw corners in rows 16.. ."""
    D = cfg.D
    M = rows
    N = 16 + (0 if rsc else len(F32_CORNERS))
    x = _xrows(rng, M, KB * D, D // 4)
    KBr = k0 + KB
    if wf == I.W8:
        wel = rng.integers(-128, 128, (N, KB * D))
        wel[:, ::7] = -128
        data = wel.astype(np.int8).view(np.uint8)
    else:
        data = Q.pack4(rng.integers(0, 16, (N, KB * D)))
    rs = -(-data.shape[1] // D) * D
    srows, *_ = _scale_rows(rng, wf, N, KBr, rsc, True)
    srows = srows.copy()
    if not rsc:
        sw = srows[:, :4].copy().view(np.uint32)
        sw[16:, 0] = F32_CORNERS
        srows[:, :4] = sw.view(np.uint8)
    srs = srows.shape[1]
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    _put(dram, SA, srows)
    alpha = rng.uniform(-1.5, 1.5, M).astype(np.float32)
    _put(dram, AL, alpha)
    mm = dict(sa=WA, n=N, kb=KB, rs=rs, ors=N + 1, m=M, ab=1, srs=srs, wf=wf, pair=pair, k0=k0,
              rsc=rsc)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M), I.qact(0, M, 1, KB, KB * D, dup=pair),
            I.mm(ssa=SA, out=OUT, **mm),
            I.mm(ssa=SA, out=OUT + 4096, **mm, unit=True, rmax=rows <= cfg.MCOLS),
            I.mm(ssa=SA, out=OUT, **mm, acc=True, rmax=rows <= cfg.MCOLS),
            I.mm(ssa=0, out=OUT + 4096, **mm, unit=True, acc=True,
                 ascale=0x7000 if rows <= cfg.MCOLS else None),
            I.halt()]
    return prog, dram


@pytest.mark.parametrize("wf,KB,k0,pair,rsc,rows,E", MM_CASES)
def test_mm_mx_cpc2(have_verilator, wf, KB, k0, pair, rsc, rows, E):
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, PAIR=pair, CPC=2, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(1000 + 100 * wf + KB + 7 * k0 + 3 * pair + 5 * rsc + rows)
    prog, dram = _mm_prog(rng, cfg, wf, KB, k0, pair, rsc, rows)
    _both(cfg, prog, dram, uarch={"MXU_E": E, "FIFO_DEPTH": 64})


def test_mm_formats_back_to_back_cpc2(have_verilator):
    """An int8 MM (pairs), a 4-bit one (one chunk an advance) and an int8 one again, queued:
    the pop head only moves early between commands the pipeline treats alike."""
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=4, CPC=2, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(77)
    D, KB, M, N = 128, 5, 4, 20
    x = _xrows(rng, M, KB * D, D // 4)
    d8 = rng.integers(-128, 128, (N, KB * D)).astype(np.int8).view(np.uint8)
    d4 = Q.pack4(rng.integers(0, 16, (N, KB * D)))
    rs8, rs4 = KB * D, -(-d4.shape[1] // D) * D
    s8, *_ = _scale_rows(rng, I.W8, N, KB, False, True)
    s4, *_ = _scale_rows(rng, I.W4F, N, KB, False, True)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    W4, S4 = WA + 0x20000, SA + 0x8000
    for n in range(N):
        dram[WA + n * rs8:WA + n * rs8 + KB * D] = d8[n]
        dram[W4 + n * rs4:W4 + n * rs4 + d4.shape[1]] = d4[n]
    _put(dram, SA, s8)
    _put(dram, S4, s4)
    mm8 = dict(sa=WA, ssa=SA, n=N, kb=KB, rs=rs8, ors=N + 1, m=M, ab=0, srs=s8.shape[1])
    mm4 = dict(sa=W4, ssa=S4, n=N, kb=KB, rs=rs4, ors=N + 1, m=M, ab=0, srs=s4.shape[1], wf=I.W4F)
    prog = [I.ld(X, 0, x.size), I.qact(0, M, 0, KB, KB * D),
            I.mm(out=OUT, **mm8), I.mm(out=OUT, acc=True, **mm4), I.mm(out=OUT, acc=True, **mm8),
            I.mm(out=OUT + 4096, **mm4), I.mm(out=OUT + 4096, acc=True, rmax=True, **mm8),
            I.halt()]
    _both(cfg, prog, dram, uarch={"FIFO_DEPTH": 64})


def test_describe_follows_cpc():
    """An int8 MM's nominal cycles: N x KB at CPC 1, N x ceil(KB / 2) at CPC 2 (the RTL's half
    pairs), ceil(N x KB / 2) in the flat model; its port-B transfers stay N x KB; 4-bit MMs are
    one chunk an advance at either."""
    ins8 = I.mm(0, 0, 0, 1000, 9, 1152, 1, 1, 0, 16)
    ins4 = I.mm(0, 0, 0, 1000, 9, 1152, 1, 1, 0, 48, wf=I.W4F)
    c1, c2 = Config(S=1, D=128), Config(S=1, D=128, CPC=2)
    assert _describe(ins8, c1)[2:4] == (9000, 9000)
    assert _describe(ins8, c2)[2:4] == (5000, 9000)
    assert _describe(ins8, c2, flat=True)[2:4] == (4500, 9000)
    assert _describe(ins4, c1)[2] == _describe(ins4, c2)[2] == 9000
    assert replace(c1, CPC=2) == c2
