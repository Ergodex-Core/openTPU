"""C1's sidecar and independent arithmetic against actual canonical ISA execution."""
from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu.isasim import Slice
from opentpu import c1_rows as C


def fixture(rows=19, blocks=9, engines=16, seed=4):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(rows, blocks * 128)).astype(np.float32)
    weight *= np.exp(rng.uniform(-7, 7, (rows, blocks))).repeat(128, axis=1)
    w, s = Q.quantize_mxu(weight, "int8", 128)
    image = C.pack(w, s, engines=engines,
                   bases=tuple((i + 1) * 4096 for i in range(engines)))
    return rng, w, s, image


def states(w, s, *, seed=9):
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128,
                 TMEM_WORDS=65536, DRAM_BYTES=2 << 20)
    wa, sa = 128, 1 << 20
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[wa:wa + w.size] = w.reshape(-1)
    dram[sa:sa + s.size] = s.reshape(-1)
    pair = [Slice(cfg, 0, [], dram) for _ in range(2)]
    rng = np.random.default_rng(seed)
    # Include legal groups that contribute and groups dropped by a negative shift.
    a = rng.integers(-127, 128, pair[0].act.shape, dtype=np.int8)
    off = rng.integers(0, 21, pair[0].adoff.shape, dtype=np.uint8)
    initial = rng.normal(size=cfg.TMEM_WORDS).astype(np.float32).view(np.uint32)
    for st in pair:
        st.act[:] = a
        st.adoff[:] = off
        st.ascale[:] = np.float32(0.037)
        st.tmem[:] = initial
    return pair, wa, sa


def golden(st, c, w, s, wa, sa):
    st.execute(I.mm(wa + c.n0 * w.shape[1] + c.k0 * 128,
                    sa + c.n0 * s.shape[1], c.out, c.n, c.kb,
                    w.shape[1], c.n, c.m, c.ab, s.shape[1],
                    acc=bool(c.flags & I.F_ACC), k0=c.k0))


@pytest.mark.parametrize("rows,engines", [(1, 16), (9, 16), (19, 16), (35, 32)])
@pytest.mark.parametrize("blocks", [1, 6, 9, 54])
def test_sidecar_exact_bytes_tails_and_locations(rows, engines, blocks, tmp_path):
    _, w, s, image = fixture(rows, blocks, engines)
    # Canonical padding also survives; inputs may be strided views.
    s[:, 4 + blocks:] = 0xA5
    image = C.pack(w[:, ::-1][:, ::-1], s, engines=engines,
                   pc_ids=tuple(reversed(range(engines))), bases=image.descriptor.bases)
    d = image.descriptor
    assert C.Descriptor.from_dict(json.loads(json.dumps(d.to_dict()))) == d
    for n in range(rows):
        rw, rs = image.row(n)
        assert np.array_equal(rw, w[n]) and np.array_equal(rs, s[n])
        assert d.location(n) == (d.pc_ids[n % engines],
                                 d.bases[n % engines] + n // engines * d.pitch)
        assert not rw.flags.writeable
    assert sum(map(len, image.payloads)) == rows * d.pitch
    loaded = C.PackedRows.load(image.save(tmp_path / "weights.c1rows"))
    assert loaded == image
    with pytest.raises(ValueError, match="row out of bounds"):
        image.row(rows)


@pytest.mark.parametrize("kb,k0,ab", [(1, 0, 0), (6, 3, 7), (9, 1, 3), (54, 2, 8)])
@pytest.mark.parametrize("engines,n0,n", [(1, 0, 19), (16, 3, 13), (32, 18, 1)])
@pytest.mark.parametrize("acc", [False, True])
def test_reference_matches_canonical_isa(kb, k0, ab, engines, n0, n, acc):
    _, w, s, image = fixture(19, k0 + kb + 1, engines)
    (ref, canonical), wa, sa = states(w, s)
    c = C.Command(n0, n, k0, kb, ab, 500, flags=I.F_ACC if acc else 0)
    actual = C.execute(image, c, ref)
    golden(canonical, c, w, s, wa, sa)
    assert np.array_equal(actual.view(np.uint32), canonical.tmem[c.out:c.out + n])
    # No tail writes or changes elsewhere in TMEM/DRAM/ACT are allowed.
    assert np.array_equal(ref.tmem, canonical.tmem)
    assert np.array_equal(ref.dram, canonical.dram)
    assert np.array_equal(ref.act, canonical.act)


def test_chunked_accumulation_preserves_isa_rounding():
    _, w, s, image = fixture(17, 54)
    (ref, canonical), wa, sa = states(w, s)
    for k0 in range(0, 54, 6):
        c = C.Command(0, 17, k0, 6, k0 + 2, 123, flags=I.F_ACC if k0 else 0)
        C.execute(image, c, ref)
        golden(canonical, c, w, s, wa, sa)
        assert np.array_equal(ref.tmem, canonical.tmem)


@pytest.mark.parametrize("sa_bits", [0, 0x80000000, 1, 0x00800000, 0x3F800000,
                                      0x7F800000, 0x7FC12345])
def test_epilogue_special_bits_and_acc(sa_bits):
    _, w, s, _ = fixture(7, 1)
    sw_bits = np.array([0, 0x80000000, 1, 0x00800000, 0x3F800000,
                        0x7F800000, 0x7FC12345], np.uint32)
    s[:, :4] = sw_bits[:, None].view(np.uint8)
    image = C.pack(w, s)
    (ref, canonical), wa, sa = states(w, s)
    for st in (ref, canonical):
        st.ascale[:] = np.array(sa_bits, np.uint32).view(np.float32)
        st.tmem[100:107] = sw_bits
    c = C.Command(0, 7, 0, 1, 2, 100, flags=I.F_ACC)
    with np.errstate(invalid="ignore", over="ignore"):
        C.execute(image, c, ref)
        golden(canonical, c, w, s, wa, sa)
    assert np.array_equal(ref.tmem, canonical.tmem)


@pytest.mark.parametrize("change", [dict(m=4), dict(immutable=False), dict(flags=I.F_UNIT),
                                     dict(flags=I.F_RSC), dict(flags=I.F_RMAX),
                                     dict(flags=I.F_ASCALE | I.F_ACC), dict(flags=I.F_PAIR),
                                     dict(flags=I.W4F << I.WF_SHIFT)])
def test_ineligible_commands_require_fallback_without_writes(change):
    _, w, s, image = fixture()
    (ref, _), _, _ = states(w, s)
    c = replace(C.Command(0, 9, 0, 1, 0, 40), **change)
    before = ref.tmem.copy()
    assert C.fallback_reason(ref.cfg, c)
    with pytest.raises(C.FallbackRequired):
        C.execute(image, c, ref)
    assert np.array_equal(ref.tmem, before)


@pytest.mark.parametrize("change", [dict(n0=-1), dict(n0=18, n=2), dict(n=0),
                                     dict(k0=8, kb=2), dict(ab=127, kb=2),
                                     dict(out=65535, n=2), dict(kb=0)])
def test_invalid_ranges_do_not_commit(change):
    _, w, s, image = fixture()
    (ref, _), _, _ = states(w, s)
    before = ref.tmem.copy()
    with pytest.raises(ValueError):
        C.execute(image, replace(C.Command(0, 1, 0, 1, 0, 30), **change), ref)
    assert np.array_equal(ref.tmem, before)


def test_bad_scale_snapshot_and_descriptor_rejected():
    _, w, s, image = fixture()
    (ref, _), _, _ = states(w, s)
    ref.ascale[0, 1] *= 2
    with pytest.raises(ValueError, match="different Sa"):
        C.execute(image, C.Command(0, 1, 0, 2, 0, 0), ref)
    for change in (dict(version=2), dict(pc_ids=(0,) * 16), dict(bases=(1,) * 16),
                   dict(limits=(0,) * 16), dict(blocks=0), dict(rows=0)):
        with pytest.raises(ValueError):
            replace(image.descriptor, **change)


def test_corrupt_sidecar_rejected(tmp_path):
    _, _, _, image = fixture()
    path = image.save(tmp_path / "good.c1rows")
    broken = tmp_path / "broken.c1rows"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(broken, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "pc0.bin":
                data = bytes([data[0] ^ 1]) + data[1:]
            dst.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        C.PackedRows.load(broken)


def test_real_qact_snapshot_matches_isa():
    rng, w, s, image = fixture(9, 9)
    (ref, canonical), wa, sa = states(w, s)
    x = rng.normal(size=9 * 128).astype(np.float32)
    for st in (ref, canonical):
        st.tput(np.arange(x.size), x)
        st.execute(I.qact(0, 1, 3, 9, x.size))
    c = C.Command(0, 9, 0, 9, 3, 5000)
    C.execute(image, c, ref)
    golden(canonical, c, w, s, wa, sa)
    assert np.array_equal(ref.tmem, canonical.tmem)


@pytest.mark.parametrize("sign", [-1, 1])
def test_long_row_requires_wide_integer_accumulator(sign):
    w = np.full((3, 54 * 128), sign * 127, np.int8).view(np.uint8)
    s = Q.pack_scales("int8", np.full(3, 0.01, np.float32),
                      dw=np.zeros((3, 54), np.uint8))
    image = C.pack(w, s)
    (ref, canonical), wa, sa = states(w, s)
    for st in (ref, canonical):
        st.act[:] = 127
        st.adoff[:] = 0
    c = C.Command(0, 3, 0, 54, 0, 100)
    C.execute(image, c, ref)
    golden(canonical, c, w, s, wa, sa)
    assert np.array_equal(ref.tmem, canonical.tmem)


@pytest.mark.parametrize("engines,rows,expected", [(1, 1, 112), (1, 2, 207), (2, 2, 113)])
def test_analytic_event_policy_known_small_schedule(engines, rows, expected):
    from opentpu.c1_profile import profile
    _, w, s, image = fixture(rows, 1, engines)
    (state, _), _, _ = states(w, s)
    d = profile(image.descriptor, C.Command(0, rows, 0, 1, 0, 10), state.cfg)
    assert d["cycles"] == expected
    assert d["kind"] == "analytic-c1"
    assert d["macs"] == rows * 128
    assert d["bytes"] == rows * (128 + 32)
    assert d["slices"][0]["buckets"] == {}  # do not invent measured RTL counters


def test_lens_roundtrip_credit_policy_and_delayed_pc(tmp_path):
    from opentpu import lens
    from opentpu.c1_profile import profile
    _, w, s, image = fixture(35, 9, 16)
    (state, _), _, _ = states(w, s)
    command = C.Command(3, 31, 2, 6, 4, 100)
    normal = profile(image.descriptor, command, state.cfg)
    slow = profile(image.descriptor, command, state.cfg, extra_pc_latency={3: 160})
    assert slow["cycles"] > normal["cycles"]
    for d in (normal, slow):
        events = d["c1_model"]["events"]
        assert [e["row"] for e in events] == list(range(3, 34))
        assert all(a["retired"] <= b["retire_start"] for a, b in zip(events, events[1:]))
        last_retire = {}
        for e in events:
            assert last_retire.get(e["pc"], 0) <= e["dispatch"]
            last_retire[e["pc"]] = e["retired"]
        assert d["cycles"] == events[-1]["retired"]
        assert lens.summary_data(d)["causes"] is None
        assert "analytic-c1" in lens.render_summary(d)
    path = lens.save([normal, slow], tmp_path / "c1.otpuprof")
    document = lens.load(path)
    assert document["profiles"][0]["cycles"] == normal["cycles"]
    html = lens.render(document)
    assert "/*__DATA__*/" not in html and "one-row-credit analytic policy" in html
    assert "NOT RTL or F2 hardware timing" in html
