"""A wide port B (Config.PORTB, docs/wide_port.md): MM column reuse for every weight format, as
many D-byte chunks a cycle as the port takes: int8 PAIR (2 chunks) and QUAD (4), 4-bit QUAD
(2). As with 4-bit PAIR the sums are one exact integer per output, so reuse changes no bit;
here the ISA simulator's row choice, the PORTB limit, QACT DUP4 and the compiler's choice."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu import language as ol
from opentpu.isasim import Machine, SimError
from opentpu.runtime import Input, Output, Weight, launch

WA, SA, XA, OUT = 16384, 81920, 1 << 18, 8192


def _setup(rng, fmt, KB, N, D, x):
    W = (rng.standard_normal((N, KB * D)) * 0.1).astype(np.float32)
    wrows, srows = Q.quantize_mxu(W, fmt, D)
    dram = np.zeros(1 << 19, np.uint8)
    dram[XA:XA + x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + wrows.size] = wrows.reshape(-1)
    dram[SA:SA + srows.size] = srows.reshape(-1)
    return wrows.shape[1], srows.shape[1], dram


def _run(cfg, dram, prog):
    cfg = dataclasses.replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, dram.size))
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


@pytest.mark.parametrize("fmt,mode,portb", [("int8", "pair", 2), ("int8", "quad", 4),
                                            ("int4", "quad", 2), ("fp4", "quad", 2)])
@pytest.mark.parametrize("KB", [1, 4, 5, 8])
def test_reuse_changes_no_bit(fmt, mode, portb, KB):
    D, N, M = 128, 7, 1
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=portb)
    rng = np.random.default_rng(10 * KB + portb)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    rs, srs, dram = _setup(rng, fmt, KB, N, D, x)
    wf = Q.mxu_wf(fmt)
    plain = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=wf)])
    quad = mode == "quad"
    fast = _run(cfg, dram, [I.ld(XA, 0, x.size),
                            I.qact(0, M, 0, KB, KB * D, dup=not quad, dup4=quad),
                            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=wf, pair=True,
                                 quad=quad)])
    a, b = (r.tget(OUT + np.arange(M * N)) for r in (plain, fast))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("fmt,mode,need", [("int8", "pair", 2), ("int8", "quad", 4),
                                           ("int4", "quad", 2)])
def test_reuse_beyond_the_port_is_refused(fmt, mode, need):
    D, N, M, KB = 128, 3, 1, 4
    rng = np.random.default_rng(1)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    rs, srs, dram = _setup(rng, fmt, KB, N, D, x)
    quad = mode == "quad"
    prog = [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D, dup=not quad, dup4=quad),
            I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, wf=Q.mxu_wf(fmt), pair=True, quad=quad)]
    with pytest.raises(SimError, match="PORTB|DUP4"):
        _run(Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=need // 2), dram, prog)
    _run(Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=need), dram, prog)


@ol.jit
def k_dot(x, w, out):
    ol.store(out, ol.dot(ol.load(x), w))


@pytest.mark.parametrize("portb,want", [(1, ""), (2, " pair"), (4, " quad")])
def test_the_compiler_takes_the_widest_int8_reuse_the_port_allows(portb, want):
    """A one-row int8 MM (decode): plain at PORTB 1, PAIR at 2, QUAD at 4; the same output."""
    D, N, K = 128, 16, 8 * 128
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, K)).astype(np.float32)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=portb)
    from opentpu.runtime import compile_kernel
    compiled, _ = compile_kernel(k_dot, cfg, x=Input(x), w=Weight(W), out=Output((1, N)))
    mms = [i for i in compiled.programs[0] if i.op == I.MM]
    assert mms and all(i.comment.endswith(want) or (want == "" and not
                       i.comment.endswith((" pair", " quad"))) for i in mms)
    got = launch(k_dot, cfg, x=Input(x), w=Weight(W), out=Output((1, N))).outputs["out"]
    base = launch(k_dot, dataclasses.replace(cfg, PORTB=1), x=Input(x), w=Weight(W),
                  out=Output((1, N))).outputs["out"]
    assert np.array_equal(got.view(np.uint32), base.view(np.uint32))
