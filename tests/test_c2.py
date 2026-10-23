"""C2 milestone 1: independent packed reads/arithmetic versus canonical ISA MM."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from opentpu import c2, isa as I, lens, quant as Q
from opentpu.isasim import Config, SimError, Slice


def make_case(kb=9, *, n=7, n0=4, k_begin=3, scale_k0=2, ab=2, acc=False, seed=7):
    rng = np.random.default_rng(seed)
    full_kb, rows = k_begin + kb + 1, n0 + n + 1
    rs = (full_kb + 1) * c2.D                 # nontrivial canonical row padding
    srs = ((4 + scale_k0 + full_kb + 3) // 4) * 4 + 4
    sa = 256
    ssa = ((sa + rows * rs + 127) // 128) * 128
    matrix = c2.Matrix(rows, full_kb, sa, ssa, rs, srs, scale_k0)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=ab + kb + 1,
                 DRAM_BYTES=ssa + rows * srs + 128, TMEM_WORDS=2048)
    dram = np.full(cfg.DRAM_BYTES, 0xA5, np.uint8)
    weights = rng.integers(-127, 128, (rows, full_kb * 128), dtype=np.int8)
    weights[n0] = 0
    weights[n0 + 1, ::2] = 127
    weights[n0 + 1, 1::2] = -127
    sw = np.ldexp((np.arange(rows) + 1).astype(np.float32) / np.float32(17), -9)
    if n0 + 2 < rows:
        sw[n0 + 2] = np.float32(0)
    for row in range(rows):
        dram[sa + row * rs:sa + row * rs + weights.shape[1]] = weights[row].view(np.uint8)
        base = ssa + row * srs
        dram[base:base + 4] = sw[row:row + 1].view(np.uint8)
        # Include the shift boundary, and unrelated prefix/suffix offsets.
        dram[base + 4:base + 4 + scale_k0 + full_kb] = rng.integers(0, 22, scale_k0 + full_kb)
    sl = Slice(cfg, 0, [], dram)
    sl.act[:] = rng.integers(-127, 128, sl.act.shape, dtype=np.int8)
    sl.adoff[:] = rng.integers(0, 20, sl.adoff.shape, dtype=np.uint8)
    sl.ascale[:] = np.float32(0.03125)
    sl.tput(np.arange(cfg.TMEM_WORDS), rng.standard_normal(cfg.TMEM_WORDS).astype(np.float32))
    # Exercise runtime address resolution as well as n0/k_begin/scale origin/ab.
    sl.R[1], sl.R[2], sl.R[3] = 128, 4, 5
    ins = I.mm(sa + n0 * rs + k_begin * 128 - sl.R[1], ssa + n0 * srs - sl.R[2],
               41 - sl.R[3], n, kb, rs, n + 3, 1, ab, srs, acc=acc,
               k0=scale_k0 + k_begin, ra=1, rb=2, rc=3)
    packed = c2.pack(dram, matrix, bases=tuple(128 * pc for pc in range(c2.PCS)))
    return sl, ins, packed, n0, k_begin


@pytest.mark.parametrize("kb", [1, 9, 54, 255])
@pytest.mark.parametrize("acc", [False, True])
def test_packed_reference_matches_isa_entire_state(kb, acc):
    sl, ins, packed, n0, start = make_case(kb, acc=acc)
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    trace = c2.Trace()
    got = c2.execute(sl, ins, packed, n0=n0, k_begin=start, trace=trace)
    assert got.operation == "MM_ROWS"
    assert np.array_equal(sl.tmem, expected.tmem)
    assert np.array_equal(sl.dram, expected.dram)
    assert np.array_equal(sl.act, expected.act)
    assert np.array_equal(sl.ascale.view(np.uint32), expected.ascale.view(np.uint32))
    counts = trace.counts()
    assert counts["advance_slots"] == 2 * kb
    assert counts["weight_chunks"] == 7 * kb
    assert counts["products"] == 7 * kb * 128
    assert counts["scale_payload_bytes"] == 7 * (kb + 4)
    assert counts["column_utilization"] == 7 / 8
    commits = [e for e in trace.events if e["op"] == "MM_ROWS.commit"]
    assert [addr for e in commits for addr in e["outputs"]] == list(range(41, 48))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9])
def test_tail_mask_never_reads_padding(n):
    sl, ins, packed, n0, start = make_case(9, n=n)
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    # Poison physical padding; it must never become a logical read/product.
    banks = tuple(bank.copy() for bank in packed.banks)
    valid = [np.zeros(len(bank), bool) for bank in banks]
    for row in range(packed.matrix.rows):
        for block in range(packed.matrix.kb):
            pc, addr = packed.address(row, block)
            off = addr - packed.bases[pc]
            valid[pc][off:off + 128] = True
    for pc in range(16):
        banks[pc][~valid[pc]] = 127
    packed = replace(packed, banks=banks)
    trace = c2.Trace()
    c2.execute(sl, ins, packed, n0=n0, k_begin=start, trace=trace)
    assert np.array_equal(sl.tmem, expected.tmem)
    reads = [tuple(read) for e in trace.events for read in e.get("reads", [])]
    want = [packed.address(row, block) for row in range(n0, n0 + n)
            for block in range(start, start + 9)]
    assert sorted(reads) == sorted(want)
    with pytest.raises(ValueError, match="padding"):
        packed.read(packed.matrix.rows, 0)
    with pytest.raises(ValueError, match="padding"):
        packed.read(0, packed.matrix.kb)


def test_packing_copies_every_byte_scale_and_uses_full_pitch():
    sl, ins, packed, n0, start = make_case()
    m = packed.matrix
    for row in range(m.rows):
        for k in range(m.kb):
            want = sl.dram[m.sa + row * m.rs + k * 128:][:128].view(np.int8)
            assert np.array_equal(packed.read(row, k), want)
        assert np.array_equal(packed.scales[row], sl.dram[m.ssa + row * m.srs:][:m.srs])
    desc = packed.descriptor()
    assert desc["version"] == 1 and len(desc["content_sha256"]) == 64
    assert desc == c2.pack(sl.dram, m, bases=packed.bases).descriptor()
    with pytest.raises(ValueError):
        packed.banks[0][0] = 1
    with pytest.raises(ValueError):
        packed.scales[0, 0] = 1
    mutated = sl.dram.copy()
    mutated[m.ssa] ^= 1
    assert desc["content_sha256"] != c2.pack(mutated, m, bases=packed.bases).descriptor()["content_sha256"]


def test_pc_capacity_overlap_and_bad_descriptors_are_rejected():
    sl, ins, packed, n0, start = make_case()
    for kw in ({"bases": (1,) * 16}, {"bases": (0,) * 15}, {"capacity": packed.bank_bytes - 1},
               {"occupied": [(0, 0, 1)]}, {"occupied": [(16, 0, 1)]}):
        with pytest.raises(ValueError):
            c2.pack(sl.dram, packed.matrix, **kw)
    # Adjacent intervals and equal offsets in different PCs do not overlap.
    c2.pack(sl.dram, packed.matrix, occupied=[(0, packed.bank_bytes, packed.bank_bytes + 128)])
    c2.pack(sl.dram, packed.matrix, occupied=[(0, 1, 1)])
    for broken in (replace(packed, version=2), replace(packed, layout="unknown"),
                   replace(packed, banks=packed.banks[:15]),
                   replace(packed, scales=packed.scales[:-1])):
        with pytest.raises(ValueError):
            c2.execute(sl, ins, broken, n0=n0, k_begin=start)
    with pytest.raises(ValueError, match="only canonical MM"):
        c2.execute(sl, I.Instr(0xFE), packed)


@pytest.mark.parametrize("kw,reason", [
    ({"capability": False}, "capability"), ({"static": False}, "static"),
    ({"purpose": "kv"}, "KV/gather"), ({"purpose": "gather"}, "KV/gather"),
    ({"n0": 1}, "unaligned"), ({"k_begin": 999}, "out-of-range"),
])
def test_excluded_requests_execute_original_mm(kw, reason):
    sl, ins, packed, n0, start = make_case()
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    args = {"n0": n0, "k_begin": start, **kw}
    trace = c2.Trace()
    result = c2.execute(sl, ins, packed, trace=trace, **args)
    assert result.operation == "MM" and reason in result.reason
    assert np.array_equal(sl.tmem, expected.tmem)
    assert trace.counts()["fallbacks"] == 1 and trace.counts()["weight_bytes"] == 0
    with pytest.raises(ValueError, match="unmodeled"):
        c2.lens_data(trace, sl.cfg)


@pytest.mark.parametrize("mode", ["m2", "m3", "prefill", "fp4", "int4", "nvfp4", "pair", "rmax", "ascale", "unit", "rsc", "missing"])
def test_format_and_flag_fallback_is_bit_exact(mode):
    rng = np.random.default_rng(20)
    cfg = Config(D=128, MCOLS=4, ACT_BLOCKS=3, DRAM_BYTES=8192, TMEM_WORDS=256, PAIR=True)
    fmt = mode if mode in ("fp4", "int4", "nvfp4") else "fp4" if mode == "pair" else "int8"
    data, scales = Q.quantize_mxu(rng.standard_normal((5, 256)).astype(np.float32), fmt)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    rs = ((data.shape[1] + 127) // 128) * 128
    for row in range(5):
        dram[128 + row * rs:128 + row * rs + data.shape[1]] = data[row]
    dram[4096:4096 + scales.size] = scales.reshape(-1)
    sl = Slice(cfg, 0, [], dram)
    sl.act[:] = rng.integers(-127, 128, sl.act.shape, dtype=np.int8)
    sl.ascale[:] = np.float32(0.01)
    sl.tput(np.arange(256), rng.standard_normal(256).astype(np.float32))
    m = {"m2": 2, "m3": 3, "prefill": 4}.get(mode, 1)
    ins = I.mm(128, 4096 if mode != "ascale" else 100, 0, 5, 2, rs, 8, m, 0, scales.shape[1],
               wf=I.WFORMATS[fmt], pair=mode == "pair", rmax=mode == "rmax",
               unit=mode in ("unit", "ascale"), acc=mode == "ascale", ascale=100 if mode == "ascale" else None,
               rsc=mode == "rsc")
    # A valid int8 sidecar still cannot select a four-bit or otherwise excluded command.
    _, _, packed, _, _ = make_case()
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    result = c2.execute(sl, ins, None if mode == "missing" else packed)
    assert result.operation == "MM"
    assert np.array_equal(sl.tmem, expected.tmem)


def test_shift_boundary_and_fp32_ftz_contract():
    sl, ins, packed, n0, start = make_case(1, n=3, n0=0, k_begin=0, scale_k0=0, ab=0)
    m = packed.matrix
    sl.act[0, :128] = 127
    sl.adoff[0, 0] = [0, 1, 16, 17]           # sh=0 survives; sh=-1 drops
    sw_bits = [0x00000001, 0x80000000, 0x00800000]  # subnormal, -0, minimum normal
    for row, bits in enumerate(sw_bits):
        sl.dram[m.sa + row * m.rs:][:128] = 127
        sl.dram[m.ssa + row * m.srs:][:4] = np.array([bits], np.uint32).view(np.uint8)
        sl.dram[m.ssa + row * m.srs + 4] = 0
    sl.ascale[0, 0] = np.float32(0.5)
    packed = c2.pack(sl.dram, m)
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    trace = c2.Trace()
    c2.execute(sl, ins, packed, trace=trace)
    assert np.array_equal(sl.tmem, expected.tmem)
    want = 32 * 127 * 127 * ((1 << 16) + (1 << 15) + 1)
    assert trace.events[-1]["accumulator"] == [want] * 3


def test_bad_activation_scale_or_output_fails_before_write():
    sl, ins, packed, n0, start = make_case()
    before = sl.tmem.copy()
    sl.ascale[0, 3] *= 2
    with pytest.raises(SimError, match="different Sa"):
        c2.execute(sl, ins, packed, n0=n0, k_begin=start)
    assert np.array_equal(sl.tmem, before)
    sl.ascale[:] = np.float32(0.03125)
    ins.w[2] = sl.cfg.TMEM_WORDS
    with pytest.raises(SimError):
        c2.execute(sl, ins, packed, n0=n0, k_begin=start)
    assert np.array_equal(sl.tmem, before)


def test_wrong_scale_origin_or_layout_explicitly_falls_back():
    sl, ins, packed, n0, start = make_case()
    # Same weight bytes, different scale k0: do not silently use the sidecar's dw.
    ins.w[3] += 1 << 24
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    result = c2.execute(sl, ins, packed, n0=n0, k_begin=start)
    assert result.operation == "MM" and "scale origin" in result.reason
    assert np.array_equal(sl.tmem, expected.tmem)


def test_lens_round_trip_contains_executed_reads_and_model_label(tmp_path):
    sl, ins, packed, n0, start = make_case()
    trace = c2.Trace()
    c2.execute(sl, ins, packed, n0=n0, k_begin=start, trace=trace)
    data = c2.lens_data(trace, sl.cfg)
    path = lens.save([data], tmp_path / "c2.otpuprof")
    doc = lens.load(path)
    assert doc["profiles"][0]["c2"]["counts"] == trace.counts()
    summary = lens.render_summary(doc["profiles"][0])
    assert "NOT RTL" in summary and "NOT RTL cycles" in summary
    assert "MM_ROWS.advance" in summary and "MM_ROWS.commit" in summary
    assert "serialized-functional-events-v1" in lens.render(doc)
    assert data["peak_macs"] == 512 and data["config"]["C"] == 1


def test_repository_quantizer_and_qact_feed_packed_path():
    rng = np.random.default_rng(31)
    weights, scales = Q.quantize_mxu(rng.standard_normal((7, 1152)).astype(np.float32), "int8")
    x = rng.standard_normal((1, 1152)).astype(np.float32)
    cfg = Config(D=128, MCOLS=4, ACT_BLOCKS=12, DRAM_BYTES=32768, TMEM_WORDS=2048)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[8192:8192 + weights.size] = weights.reshape(-1)
    dram[24576:24576 + scales.size] = scales.reshape(-1)
    sl = Slice(cfg, 0, [], dram)
    sl.tput(np.arange(x.size), x.reshape(-1))
    sl.execute(I.qact(0, 1, 2, 9, 1152))
    packed = c2.pack(sl.dram, c2.Matrix(7, 9, 8192, 24576, 1152, scales.shape[1]))
    ins = I.mm(8192, 24576, 1500, 7, 9, 1152, 7, 1, 2, scales.shape[1])
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    assert c2.execute(sl, ins, packed).operation == "MM_ROWS"
    assert np.array_equal(sl.tmem, expected.tmem)
