"""Row lanes (MM RL, docs/row_lanes.md) on the ISA simulator and in the compiler: an RL MM over
a matrix in row-lane groups (quant.rl_rows / rl_scales) gives bit for bit the plain MM's values
over the same matrix in standard rows (the mode changes cycles, not values), for every format
and L, with UNIT / ACC / RMAX / ASCALE / RSC, N not a multiple of L; the simulator refuses what
the ISA forbids; the compiler's one-row dot over an RL weight emits MM RL with the operand's
copies (QACT DUP / DUP4 / DUP8); a tiny Gemma 3 and Qwen3 decode step, and Gemma-3-1B's, give
the same logits with and without Config.ROWLANES; the tally counts the mode's cycles."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I, quant as Q
from opentpu.isasim import Config, Machine, SimError

D = 32
X, WA, SA, WB, SB, OUT, AL = 0, 0x8000, 0x20000, 0x30000, 0x50000, 0x6000, 0x7000
FMTS = ["int8", "int4", "fp4", "nvfp4", "t2"]
REAL = Path(__file__).resolve().parent.parent / "models" / "gemma-3-1b-it"


def _put(dram, addr, a):
    v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[addr:addr + v.size] = v


def _run(cfg, prog, dram):
    return Machine(cfg, [prog], [dram.copy()]).run().slices[0].tmem


def _matrix(rng, fmt, N, K):
    """A random matrix quantized (quant.quantize_mx), with values spread over the formats'
    ranges (every block scale offset exercised)."""
    W = rng.standard_normal((N, K)).astype(np.float32)
    W *= 2.0 ** rng.integers(-6, 3, (N, K // (D // 4))).repeat(D // 4, axis=1)
    return Q.quantize_mx(W, fmt, D)


def _prog(cfg, fmt, rows, srows, N, KB, k0, L, x, mode, M=1):
    """QACT (the copies an RL MM of L lanes needs; M > 1: the M rows, no copies), then the MM
    pair: plain over the standard rows at WA / SA and, over the groups at WB / SB (the same k0:
    the rows hold k0 + KB blocks; the groups at cfg.ALINE lines), RL at M = 1 or the group walk
    at M > 1 (srs bit 0), each in mode plain / unit+rmax / acc+rmax / ascale / rsc."""
    wf = Q.mxu_wf(fmt)
    bpc = I.blocks_per_chunk(wf)
    copies = L * bpc if M == 1 else 1
    K = (k0 + KB) * D
    line = cfg.ALINE
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    _put(dram, X, x)
    rs, srs = Q.row_bytes(K, fmt, D), Q.scale_bytes(K, fmt, D)
    _put(dram, WA, rows)
    _put(dram, SA, srows)
    rsg, srsg = Q.rl_row_stride(K, fmt, L, D), Q.rl_scale_bytes(K, fmt, L, D, line)
    _put(dram, WB, Q.rl_rows(rows, L, D))
    _put(dram, SB, Q.rl_scales(srows, fmt, K, L, D, line))
    rsc = mode == "rsc"
    if rsc:                     # RSC: the rows' Sw alone (standard RSC rows = the RL groups)
        sw = np.ascontiguousarray(srows[:, :4])
        _put(dram, SA, sw)
        _put(dram, SB, sw)
        srs, srsg = 4, 4 * L
    alpha = np.linspace(1.25, -0.75, M).astype(np.float32)
    _put(dram, AL, alpha)
    kw = dict(n=N, kb=KB, ors=N + 1, m=M, ab=0, k0=0 if rsc else k0, wf=wf, rsc=rsc)
    if mode == "unit":
        kw.update(unit=True, rmax=True)
    elif mode == "acc":
        kw.update(acc=True, rmax=True)
    elif mode == "ascale":
        kw.update(unit=True, acc=True, ascale=0x7000)
    g = dict(rl=L, walk=M > 1)
    sao = k0 * D // bpc
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, M),
            I.qact(0, M, 0, KB, KB * D, dup=copies == 2, dup4=copies == 4, dup8=copies == 8)]
    if mode in ("acc", "ascale"):           # an accumulator to add to: the plain MM's own
        prog += [I.mm(WA + sao, SA, OUT, rs=rs, srs=srs, **{**kw, "acc": False, "rmax": False,
                                                              "unit": False, "ascale": None}),
                 I.mm(WB + (k0 // bpc) * L * D, SB, OUT + 2048, rs=rsg, srs=srsg,
                      **{**kw, "acc": False, "rmax": False, "unit": False, "ascale": None}, **g)]
    prog += [I.mm(WA + sao, SA, OUT, rs=rs, srs=srs, **kw),
             I.mm(WB + (k0 // bpc) * L * D, SB, OUT + 2048, rs=rsg, srs=srsg, **kw, **g),
             I.halt()]
    return prog, dram


CASES = [  # fmt, L, KB, k0, N, mode
    ("int8", 2, 5, 0, 7, "plain"), ("int8", 4, 9, 0, 13, "plain"), ("int8", 4, 3, 2, 8, "unit"),
    ("int8", 4, 6, 1, 9, "acc"), ("int8", 4, 4, 0, 6, "ascale"), ("int8", 8, 9, 0, 17, "plain"),
    ("int8", 8, 2, 0, 3, "rsc"), ("int8", 2, 1, 3, 5, "rsc"),
    ("int4", 2, 5, 0, 7, "plain"), ("int4", 4, 9, 2, 10, "plain"), ("int4", 4, 4, 0, 4, "acc"),
    ("int4", 2, 7, 4, 9, "unit"), ("int4", 4, 3, 0, 5, "ascale"), ("int4", 2, 6, 2, 8, "rsc"),
    ("fp4", 2, 5, 0, 6, "plain"), ("fp4", 4, 8, 0, 11, "acc"),
    ("nvfp4", 2, 5, 2, 7, "plain"), ("nvfp4", 2, 4, 0, 4, "unit"),
    ("t2", 2, 9, 0, 7, "plain"), ("t2", 2, 7, 4, 5, "acc"),
]


@pytest.mark.parametrize("aline", [32, 64])
@pytest.mark.parametrize("fmt,L,KB,k0,N,mode", CASES)
def test_row_lanes_change_no_bit(fmt, L, KB, k0, N, mode, aline):
    """The RL MM over the groups writes the plain MM's words: y, the RMAX, ACC's sums and
    ASCALE's rescale, over L lanes with N not a multiple of L; the groups at 32- and at
    64-byte port-A lines (Config.ALINE)."""
    bpc = I.blocks_per_chunk(Q.mxu_wf(fmt))
    cfg = Config(S=1, D=D, MCOLS=max(8, L * bpc), ACT_BLOCKS=32, PAIR=True, T2=True, PORTB=8,
                 ROWLANES=True, DRAM_BYTES=1 << 20, ALINE=aline)
    rng = np.random.default_rng(100 * KB + 10 * k0 + N + L + len(fmt) + len(mode))
    rows, srows = _matrix(rng, fmt, N, (k0 + KB) * D)
    x = (rng.standard_normal((1, KB * D)) * 3).astype(np.float32)
    prog, dram = _prog(cfg, fmt, rows, srows, N, KB, k0, L, x, mode)
    t = _run(cfg, prog, dram)
    a, b = t[OUT:OUT + 2 * (N + 1)], t[OUT + 2048:OUT + 2048 + 2 * (N + 1)]
    assert np.any(a != 0)
    np.testing.assert_array_equal(a, b)


WALK_CASES = [  # fmt, L, KB, k0, N, M, mode: the group walk (srs bit 0) at M rows
    ("int8", 4, 9, 0, 13, 8, "plain"), ("int8", 4, 3, 2, 8, 2, "unit"), ("int8", 2, 6, 1, 9, 3, "acc"),
    ("int8", 8, 4, 0, 6, 8, "ascale"), ("int8", 4, 9, 0, 17, 16, "plain"), ("int8", 8, 2, 0, 3, 5, "rsc"),
    ("int4", 4, 9, 2, 10, 8, "plain"), ("int4", 2, 5, 0, 7, 2, "acc"), ("int4", 4, 4, 0, 4, 16, "plain"),
    ("int4", 2, 6, 2, 8, 4, "rsc"), ("int4", 4, 3, 0, 5, 3, "ascale"), ("int4", 4, 5, 0, 7, 7, "unit"),
    ("fp4", 4, 8, 0, 11, 8, "plain"), ("fp4", 2, 5, 0, 6, 16, "plain"),
    ("nvfp4", 2, 5, 2, 7, 8, "plain"), ("nvfp4", 2, 4, 0, 4, 2, "unit"),
    ("t2", 2, 9, 0, 7, 8, "plain"), ("t2", 2, 7, 4, 5, 6, "acc"), ("t2", 2, 7, 0, 9, 16, "plain"),
]


@pytest.mark.parametrize("aline", [32, 64])
@pytest.mark.parametrize("fmt,L,KB,k0,N,M,mode", WALK_CASES)
def test_group_walk_change_no_bit(fmt, L, KB, k0, N, M, mode, aline):
    """A plain MM of M rows over the groups (the group walk, srs bit 0 with rs bits 3:2:
    prefill over a decode image) writes the plain MM's words over the standard rows: y for
    every row, the RMAX, ACC's sums and ASCALE's rescale; M up to ACT_ROWS 16 (replayed rows
    over MCOLS 8), N not a multiple of L, k0 offsets, every format; at either port-A line."""
    cfg = Config(S=1, D=D, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=32, PAIR=True, T2=True, PORTB=4,
                 ROWLANES=True, DRAM_BYTES=1 << 20, ALINE=aline)
    rng = np.random.default_rng(100 * KB + 10 * k0 + N + L + M + len(fmt) + len(mode))
    rows, srows = _matrix(rng, fmt, N, (k0 + KB) * D)
    x = (rng.standard_normal((M, KB * D)) * 3).astype(np.float32)
    prog, dram = _prog(cfg, fmt, rows, srows, N, KB, k0, L, x, mode, M=M)
    assert I.mm_walk(prog[-2]) == L and not I.mm_lanes(prog[-2]) and I.mm_layout(prog[-2]) == L
    t = _run(cfg, prog, dram)
    n = (M + 1) * (N + 1)
    a, b = t[OUT:OUT + n], t[OUT + 2048:OUT + 2048 + n]
    assert np.any(a != 0)
    np.testing.assert_array_equal(a, b)


def test_the_simulator_refuses_what_the_isa_forbids():
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=16, PAIR=True, PORTB=4, ROWLANES=True,
                 DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(3)
    rows, srows = _matrix(rng, "int8", 8, 2 * D)
    x = rng.standard_normal((1, 2 * D)).astype(np.float32)
    prog, dram = _prog(cfg, "int8", rows, srows, 8, 2, 0, 4, x, "plain")
    _run(cfg, prog, dram)
    with pytest.raises(SimError, match="ROWLANES"):
        _run(dataclasses.replace(cfg, ROWLANES=False), prog, dram)
    with pytest.raises(SimError, match="PORTB"):
        _run(dataclasses.replace(cfg, PORTB=2), prog, dram)
    prog8, _ = _prog(dataclasses.replace(cfg, MCOLS=8, PORTB=8), "int4", *_matrix(
        rng, "int4", 8, 2 * D), 8, 2, 0, 4, x, "plain")
    with pytest.raises(SimError, match="columns|ACT RAM"):   # int4 L 4: 8 columns
        _run(dataclasses.replace(cfg, PORTB=8), prog8, dram)
    with pytest.raises(ValueError, match="PAIR"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 1, 0, 64, pair=True, rl=4)
    with pytest.raises(ValueError, match="M = 1"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 2, 0, 64, rl=4)
    with pytest.raises(ValueError, match="blocks a chunk"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 1, 0, 64, wf=I.W4I, k0=1, rl=4)
    # the group walk (srs bit 0): no PAIR, needs rl, not a stride that is no multiple of 4
    I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 2, 0, 64, rl=4, walk=True)
    with pytest.raises(ValueError, match="PAIR"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 2, 0, 64, pair=True, rl=4, walk=True)
    with pytest.raises(ValueError, match="needs rl"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 2, 0, 64, walk=True)
    with pytest.raises(ValueError, match="multiple of 4"):
        I.mm(WB, SB, OUT, 8, 2, 4 * 2 * D, 9, 2, 0, 65, rl=4, walk=True)
    wprog, _ = _prog(cfg, "int8", rows, srows, 8, 2, 0, 4, np.tile(x, (2, 1)), "plain", M=2)
    _run(cfg, wprog, dram)
    with pytest.raises(SimError, match="ROWLANES"):
        _run(dataclasses.replace(cfg, ROWLANES=False), wprog, dram)
    # a scale group at 32 bytes on a configuration whose port-A line is 64
    with pytest.raises(SimError, match="ALINE"):
        _run(dataclasses.replace(cfg, ALINE=64), prog, dram)
    with pytest.raises(SimError, match="DUP8"):
        _run(dataclasses.replace(cfg, ROWLANES=False, MCOLS=8), [
            I.ld(X, 0, x.size), I.qact(0, 1, 0, 2, 2 * D, dup8=True), I.halt()], dram)


@pytest.mark.parametrize("fmt,portb,mcols,want", [("int8", 2, 4, 2), ("int8", 4, 4, 4),
                                                  ("int8", 4, 8, 4), ("int4", 2, 4, 2),
                                                  ("int4", 4, 8, 4), ("int4", 4, 4, 2)])
def test_the_compiler_streams_an_rl_weight_with_mm_rl(fmt, portb, mcols, want):
    """A one-row dot over a Weight(rl=True): the widest L the Config allows (quant.rl_lanes),
    an MM RL with the operand in L x blocks-a-chunk copies, the standard compile's values."""
    from opentpu.runtime import Input, Output, Weight, compile_kernel, launch
    from test_wide_port import k_dot
    N, K = 24, 9 * D
    rng = np.random.default_rng(5 + portb + mcols)
    x = rng.standard_normal((1, K)).astype(np.float32)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=mcols, ACT_BLOCKS=16, PAIR=True, PORTB=portb, ROWLANES=True,
                 DRAM_BYTES=1 << 20)
    comp, _ = compile_kernel(k_dot, cfg, x=Input(x), w=Weight(W, fmt=fmt, rl=True),
                             out=Output((1, N)))
    mms = [i for i in comp.programs[0] if i.op == I.MM]
    assert [I.mm_lanes(i) for i in mms] == [want] * len(mms)
    qa = [i for i in comp.programs[0] if i.op == I.QACT]
    copies = want * I.blocks_per_chunk(Q.mxu_wf(fmt))
    assert all(bool(i.flags & I.F_DUP8) == (copies == 8 or (mcols >= 8 and copies >= 2))
               or bool(i.flags & (I.F_DUP4 | I.F_DUP)) for i in qa)
    got = launch(k_dot, cfg, x=Input(x), w=Weight(W, fmt=fmt, rl=True),
                 out=Output((1, N))).outputs["out"]
    ref = launch(k_dot, dataclasses.replace(cfg, ROWLANES=False), x=Input(x),
                 w=Weight(W, fmt=fmt), out=Output((1, N))).outputs["out"]
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("fmt,portb,mcols,M", [("int8", 4, 4, 4), ("int8", 4, 8, 16),
                                                ("int4", 4, 8, 8), ("int4", 2, 4, 3)])
def test_the_compiler_walks_an_rl_weight_at_more_rows(fmt, portb, mcols, M):
    """A dot of M > 1 rows over a Weight(rl=True) (prefill over a decode image): MMs that walk
    the groups (srs bit 0, no PAIR / QUAD), the standard compile's values bit for bit; M past
    ACT_ROWS 8 in chunks (the walk flag on every chunk, a one-row tail's too)."""
    from opentpu.runtime import Input, Output, Weight, compile_kernel, launch
    from test_wide_port import k_dot
    N, K = 24, 9 * D
    rng = np.random.default_rng(5 + portb + mcols + M)
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=mcols, ACT_ROWS=8, ACT_BLOCKS=32, PAIR=True, PORTB=portb,
                 ROWLANES=True, DRAM_BYTES=1 << 20)
    comp, _ = compile_kernel(k_dot, cfg, x=Input(x), w=Weight(W, fmt=fmt, rl=True),
                             out=Output((M, N)))
    mms = [i for i in comp.programs[0] if i.op == I.MM]
    L = Q.rl_lanes(fmt, portb, mcols)
    assert mms and all(I.mm_walk(i) == L and not I.mm_lanes(i) for i in mms)
    assert all(not (i.flags & I.F_PAIR) for i in mms)
    assert sorted((i.w[5] >> 16) & 0xFF for i in mms) == sorted(
        min(8, M - m0) for m0 in range(0, M, 8))
    got = launch(k_dot, cfg, x=Input(x), w=Weight(W, fmt=fmt, rl=True),
                 out=Output((M, N))).outputs["out"]
    ref = launch(k_dot, dataclasses.replace(cfg, ROWLANES=False), x=Input(x),
                 w=Weight(W, fmt=fmt), out=Output((M, N))).outputs["out"]
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))


def _random_weights(spec, names: dict, seed: int) -> dict:
    """Random weights of the given shapes: the norms near 1, the projections scaled."""
    rng = np.random.default_rng(seed)
    W = {}
    for n, shape in names.items():
        if "norm" in n:
            W[n] = (1 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
        else:
            W[n] = (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
    return W


def _tiny_gemma3(tmp_path_factory):
    """A tiny Gemma 3 (3 layers, two sliding and a global) with random weights."""
    import json
    from opentpu.llm import gemma3 as G
    H, d, F, V, nq, nkv = 128, 128, 256, 256, 2, 1
    hc = dict(model_type="gemma3_text", hidden_size=H, num_hidden_layers=3,
              num_attention_heads=nq, num_key_value_heads=nkv, head_dim=d,
              intermediate_size=F, vocab_size=V,
              layer_types=["sliding_attention", "sliding_attention", "full_attention"],
              sliding_window=256, query_pre_attn_scalar=128, max_position_embeddings=1024,
              rope_theta=1e6, rope_local_base_freq=1e4, rms_norm_eps=1e-6,
              tie_word_embeddings=True, bos_token_id=2, eos_token_id=[1, 106])
    dd = tmp_path_factory.mktemp("gemma3")
    (dd / "config.json").write_text(json.dumps(hc))
    spec = G.Spec.from_hf(dd)
    names = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,)}
    for i in range(3):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            names[p + n + ".weight"] = (H,)
        names[p + "self_attn.q_norm.weight"] = (d,)
        names[p + "self_attn.k_norm.weight"] = (d,)
        names[p + "self_attn.q_proj.weight"] = (nq * d, H)
        names[p + "self_attn.k_proj.weight"] = (nkv * d, H)
        names[p + "self_attn.v_proj.weight"] = (nkv * d, H)
        names[p + "self_attn.o_proj.weight"] = (H, nq * d)
        names[p + "mlp.gate_proj.weight"] = (F, H)
        names[p + "mlp.up_proj.weight"] = (F, H)
        names[p + "mlp.down_proj.weight"] = (H, F)
    return _random_weights(spec, names, 31), spec


def _tiny_qwen3():
    """A tiny Qwen3 (2 layers, GQA 4 / 2, tied head) with random weights."""
    from opentpu.llm.qwen3 import Spec
    H, d, F, V, nq, nkv = 256, 128, 512, 1000, 4, 2
    spec = Spec(H, 2, nq, nkv, d, F, V)
    names = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,)}
    for i in range(2):
        p = f"model.layers.{i}."
        names[p + "input_layernorm.weight"] = (H,)
        names[p + "post_attention_layernorm.weight"] = (H,)
        names[p + "self_attn.q_norm.weight"] = (d,)
        names[p + "self_attn.k_norm.weight"] = (d,)
        names[p + "self_attn.q_proj.weight"] = (nq * d, H)
        names[p + "self_attn.k_proj.weight"] = (nkv * d, H)
        names[p + "self_attn.v_proj.weight"] = (nkv * d, H)
        names[p + "self_attn.o_proj.weight"] = (H, nq * d)
        names[p + "mlp.gate_proj.weight"] = (F, H)
        names[p + "mlp.up_proj.weight"] = (F, H)
        names[p + "mlp.down_proj.weight"] = (H, F)
    return _random_weights(spec, names, 7), spec


@pytest.mark.parametrize("model", ["gemma3", "qwen3"])
@pytest.mark.parametrize("fmt,portb,mcols", [("int8", 4, 4), ("int8", 2, 4), ("int4", 4, 8),
                                             ("int4", 2, 4)])
def test_tiny_decode_is_bit_identical_with_row_lanes(tmp_path_factory, model, fmt, portb, mcols):
    """A decode step's logits on the ISA simulator with Config.ROWLANES (the image in row-lane
    groups, the compiler's RL MMs) are the standard step's, bit for bit; the step has RL MMs."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny_gemma3(tmp_path_factory) if model == "gemma3" else _tiny_qwen3()
    base = board_config(DRAM_BYTES=1 << 24, MCOLS=mcols, PORTB=portb, PAIR=True)
    toks = [2, 18, 72, 30]
    out = {}
    for rl in (False, True):
        cfg = dataclasses.replace(base, ROWLANES=rl)
        eng = Engine(spec, W, cap=512, cfg=cfg, wformat=fmt, head_format=fmt, rows=1)
        out[rl] = np.array([eng.step(t) for t in toks]).view(np.uint32)
        progs = eng.image.compile_step(3)
        mms = [i for i in progs[0] if i.op == I.MM]
        lanes = sorted({I.mm_lanes(i) for i in mms})
        assert (lanes != [0]) == rl, lanes
        if rl:
            assert max(lanes) == Q.rl_lanes(fmt, portb, mcols)
    np.testing.assert_array_equal(out[True], out[False])


@pytest.mark.parametrize("model", ["gemma3", "qwen3"])
@pytest.mark.parametrize("fmt,portb,mcols,aline", [("int8", 4, 4, 32), ("int4", 4, 8, 64),
                                                    ("int4", 2, 4, 32)])
def test_tiny_prefill_is_bit_identical_with_row_lanes(tmp_path_factory, model, fmt, portb,
                                                      mcols, aline):
    """A prompt's prefill (runs of rows > 1: the compiler's group walk over the row-lane
    image) on the ISA simulator with Config.ROWLANES gives the standard image's logits, bit
    for bit; the prompt programs walk the groups (no RL lanes at M > 1), the decode step after
    it is the same too."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny_gemma3(tmp_path_factory) if model == "gemma3" else _tiny_qwen3()
    base = board_config(DRAM_BYTES=1 << 24, MCOLS=mcols, PORTB=portb, PAIR=True, ALINE=aline)
    toks = [2, 18, 72, 30, 5, 10, 63, 125, 7, 99, 3, 41, 8]
    out = {}
    for rl in (False, True):
        cfg = dataclasses.replace(base, ROWLANES=rl)
        eng = Engine(spec, W, cap=512, cfg=cfg, wformat=fmt, head_format=fmt, rows=4)
        out[rl] = np.stack([eng.prefill(toks), eng.step(17)]).view(np.uint32)
        progs = eng.image.compile_rows([(0, p) for p in range(4)], [3])
        mms = [i for i in progs[0] if i.op == I.MM]
        walks = [I.mm_walk(i) for i in mms if I.mm_walk(i)]
        assert (len(walks) > 0) == rl and not any(I.mm_lanes(i) for i in mms if
                                                   ((i.w[5] >> 16) & 0xFF) > 1)
        if rl:
            assert max(walks) == Q.rl_lanes(fmt, portb, mcols)
    np.testing.assert_array_equal(out[True], out[False])


@pytest.mark.skipif(not REAL.exists(), reason="models/gemma-3-1b-it is not here")
@pytest.mark.parametrize("fmt,portb,mcols,aline", [("int4", 4, 8, 64)])
def test_gemma3_1b_prefill_is_bit_identical_with_row_lanes(fmt, portb, mcols, aline):
    """Gemma-3-1B's prefill of a 32-token prompt in runs of 8 rows (the real weights, 26
    layers and the head) over the row-lane image (the group walk) and over the standard one:
    the same logits bit for bit, and the decode token after it."""
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, load_weights
    spec, W = load_spec(REAL), load_weights(REAL)
    toks = [2, 651, 6996, 576, 573, 4645, 603, 12767, 235269, 578, 573, 2134, 576, 573,
            12767, 603, 573, 1546, 235265, 714, 2134, 603, 476, 1426, 235269, 578, 573,
            12767, 603, 1426, 575, 573]
    out = {}
    for rl in (False, True):
        cfg = board_config(DRAM_BYTES=1 << 31, MCOLS=mcols, PORTB=portb, PAIR=True, ROWLANES=rl,
                           ALINE=aline)
        eng = Engine(spec, W, cap=256, cfg=cfg, wformat=fmt, head_format=fmt, rows=8)
        out[rl] = np.stack([eng.prefill(toks), eng.step(573)]).view(np.uint32)
    np.testing.assert_array_equal(out[True], out[False])


@pytest.mark.skipif(not REAL.exists(), reason="models/gemma-3-1b-it is not here")
@pytest.mark.parametrize("fmt,portb,mcols", [("int8", 4, 4), ("int4", 4, 8)])
def test_gemma3_1b_decode_token_is_bit_identical_with_row_lanes(fmt, portb, mcols):
    """Gemma-3-1B's decode token (the real weights, the full 26 layers and the head) with and
    without the mode: the same logits bit for bit; with it, the projections' and the head's
    MMs are RL (the attention's K / V MMs stay PAIR / QUAD)."""
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, load_weights
    spec, W = load_spec(REAL), load_weights(REAL)
    out = {}
    for rl in (False, True):
        cfg = board_config(DRAM_BYTES=1 << 31, MCOLS=mcols, PORTB=portb, PAIR=True, ROWLANES=rl)
        eng = Engine(spec, W, cap=256, cfg=cfg, wformat=fmt, head_format=fmt, rows=1)
        out[rl] = np.asarray(eng.step(2)).view(np.uint32)
        progs = eng.image.compile_step(0)
        mms = [i for i in progs[0] if i.op == I.MM]
        n_rl = sum(1 for i in mms if I.mm_lanes(i))
        assert (n_rl > 0) == rl and (not rl or n_rl >= 0.8 * len(mms))
    np.testing.assert_array_equal(out[True], out[False])


def test_the_tally_counts_the_mode(capsys):
    """tools/tally_decode.py --rowlanes on the tiny shapes: RL MMs in the kinds, fewer MXU
    advances than the standard step's at PORTB 4."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import tally_decode as T
    from opentpu.isasim import board_config
    W, spec = _tiny_qwen3()
    std = T.tally(spec, board_config(PORTB=4, PAIR=True, MCOLS=4, DRAM_BYTES=1 << 40), 9,
                  200.0, "int8", "int8")
    rl = T.tally(spec, board_config(PORTB=4, PAIR=True, MCOLS=4, ROWLANES=True,
                                    DRAM_BYTES=1 << 40), 9, 200.0, "int8", "int8")
    assert any(k.endswith("rl4") for k in rl["kinds"]) and not any(
        "rl" in k for k in std["kinds"])
    assert rl["mm_adv"] < std["mm_adv"]
    assert rl["mm_adv"] == std["rl_adv"] and rl["porta_lines"] == std["rl_lines"]
