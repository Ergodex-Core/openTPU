"""The packed MXU column (rtl/mxu/otpu_mxu_pk_col.sv, docs/mxu_packing.md): two int8 products a
DSP on the CL 16 cascade, the lanes split at the chain end, two column ends. Bit-exact against
numpy (the ISA's exact integer: acc = sum_k sum_g bm[g] * sum_i act[i] * w[i]) over the int8
corners (all +-127, every sign combination of the two lanes at +-258,064, the fix-up at L = -1 /
0 / +1, alternating patterns), random int8 rows of up to 128 blocks (K = 16384, the pi0.5 MLPs)
with the MX shifts, bubbles between rows, one chain and a whole column, the weights in front of
the DSP or in its registers; and -128 in the operands, where only all-(-128) breaks the lane."""
import re

import numpy as np
import pytest

from opentpu import rtlsim

D, CLS, NG = 128, 16, 8
LANE_MAX = CLS * 127 * 127                       # 258,064 < 2^18: the lane bound
SOURCES = [rtlsim.RTL / "vpu/otpu_fp.sv", rtlsim.RTL / "vpu/otpu_fpipe.sv",
           rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.RTL / "mxu/otpu_mxu.sv",
           rtlsim.RTL / "tech/generic/otpu_tech_dsp.sv", rtlsim.RTL / "mxu/otpu_mxu_pk_col.sv",
           rtlsim.TB / "tb_mxu_pk.sv"]


def _build(nch=NG, pack=True, wreg=False):
    return rtlsim.build("tb_mxu_pk", SOURCES, {"NCH": nch, "PACK": int(pack), "WREG": int(wreg)})


def _hex(a):
    """A vector's hex as the testbench reads it: element N-1 first."""
    return bytes(reversed(np.asarray(a, np.int8).view(np.uint8).tobytes())).hex()


def _bmhex(bm):
    return f"{sum(int(b) << (17 * g) for g, b in enumerate(bm)):034x}"


def _run(exe, lines, tmp_path):
    """lines: (v, first, last, act, wa, wb, bma, bmb) per cycle. Returns the rows' (acc_a, acc_b)
    as int64 [rows, 2]."""
    f = tmp_path / "vec.txt"
    with open(f, "w") as fh:
        for v, first, last, act, wa, wb, bma, bmb in lines:
            fh.write(f"{v:x} {first:x} {last:x} {_hex(act)} {_hex(wa)} {_hex(wb)} {_bmhex(bma)} "
                     f"{_bmhex(bmb)}\n")
    r = rtlsim.run_sim([str(exe), f"+vec={f}"], timeout=600)
    out = r.stdout + r.stderr
    assert re.search(r"PKTEST cycles=\d+ rows=\d+", out), out[-2000:]
    rows = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"ROW \d+ ([0-9a-f]+) ([0-9a-f]+)", out)]
    acc = np.array(rows, np.int64).reshape(-1, 2)
    return np.where(acc >= 1 << 47, acc - (1 << 48), acc)


def _ref(rows, nch):
    """Each row's two exact integers (python ints: no overflow anywhere), and the extremes of
    the chain lanes seen (max |L|, max |H|)."""
    out, lmax, hmax = [], 0, 0
    for blocks in rows:
        a = b = 0
        for act, wa, wb, bma, bmb in blocks:
            for g in range(nch):
                s = slice(g * CLS, (g + 1) * CLS)
                L = int(np.dot(act[s].astype(np.int64), wa[s].astype(np.int64)))
                H = int(np.dot(act[s].astype(np.int64), wb[s].astype(np.int64)))
                lmax, hmax = max(lmax, abs(L)), max(hmax, abs(H))
                a += int(bma[g]) * L
                b += int(bmb[g]) * H
        out.append((a, b))
    return np.array(out, np.int64).reshape(-1, 2), lmax, hmax


def _lines(rows, rng=None, bubbles=0):
    """The cycles of the rows (v, first, last with each block), with up to `bubbles` cycles
    without a block between rows (random operands and bm: the column must ignore them)."""
    lines = []
    for blocks in rows:
        if rng is not None and bubbles:
            for _ in range(int(rng.integers(0, bubbles + 1))):
                lines.append((0, int(rng.integers(0, 2)), int(rng.integers(0, 2)),
                              rng.integers(-128, 128, D).astype(np.int8),
                              rng.integers(-128, 128, D).astype(np.int8),
                              rng.integers(-128, 128, D).astype(np.int8),
                              rng.integers(0, 1 << 17, NG), rng.integers(0, 1 << 17, NG)))
        for k, (act, wa, wb, bma, bmb) in enumerate(blocks):
            lines.append((1, int(k == 0), int(k == len(blocks) - 1), act, wa, wb, bma, bmb))
    return lines


def _full(v):
    return np.full(D, v, np.int8)


def _row(act, wa, wb, bm=1, kb=1, bmb=None):
    bma = np.full(NG, bm, np.int64)
    bmb = bma if bmb is None else np.full(NG, bmb, np.int64)
    return [(act, wa, wb, bma, bmb)] * kb


def _check(exe, rows, nch, tmp_path, rng=None, bubbles=0):
    ref, lmax, hmax = _ref(rows, nch)
    got = _run(exe, _lines(rows, rng, bubbles), tmp_path)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    bad = np.nonzero((got != ref).any(axis=1))[0]
    assert len(bad) == 0, (f"{len(bad)} of {len(ref)} rows differ, first row {bad[0]}: RTL "
                           f"{got[bad[0]].tolist()} ref {ref[bad[0]].tolist()}")
    return ref, lmax, hmax


@pytest.mark.parametrize("nch", [1, NG])
@pytest.mark.parametrize("wreg", [False, True])
def test_packed_corners(have_verilator, nch, wreg, tmp_path):
    """The int8 corners: every sign combination of act, wa, wb at +-127 (both lanes at
    +-258,064, the bound), the same under B = 2^16 (the int8 shift at dw = da = 0) and over 128
    blocks (K = 16384: 1.7e13 in the 48-bit accumulator), the fix-up where a lane is -1 / 0 / +1
    (a negative low lane borrows from the high one), L = -1 beside H near its extreme, and
    alternating patterns."""
    exe = _build(nch, True, wreg)
    rows = []
    for a in (127, -127):
        for x in (127, -127):
            for y in (127, -127):
                rows.append(_row(_full(a), _full(x), _full(y)))
                rows.append(_row(_full(a), _full(x), _full(y), bm=1 << 16))
                rows.append(_row(_full(a), _full(x), _full(y), bm=1 << 16, kb=128))
    one = _full(1)
    small = {-1: [1, -1, -1], 0: [1, -1], 1: [1]}
    for l in (-1, 0, 1):
        for h in (-1, 0, 1):
            wa, wb = np.zeros(D, np.int8), np.zeros(D, np.int8)
            for g in range(NG):
                wa[g * CLS:g * CLS + len(small[l])] = small[l]
                wb[g * CLS:g * CLS + len(small[h])] = small[h]
            rows.append(_row(one, wa, wb))
            rows.append(_row(one, wa, wb, bm=1 << 16, kb=3))
    # L = -1 (or +1) with H at 127 * 15 * 127 + 127 = 242,062 (and its negative): the activation
    # 127 at fifteen positions of each chain and 1 at the last
    act = np.tile(np.array([127] * 15 + [1], np.int8), NG)
    for l in (-1, 1):
        for h in (127, -127):
            wa = np.tile(np.array([0] * 15 + [l], np.int8), NG)
            rows.append(_row(act, wa, _full(h)))
            rows.append(_row(act, _full(h), wa))
    # alternating signs: in phase (+258,064), out of phase (-258,064), one against a constant (0),
    # 7 against 9 (-32,258), and the chains alternating
    alt = np.tile(np.array([127, -127], np.int8), D // 2)
    rows.append(_row(alt, alt, -alt))
    rows.append(_row(alt, _full(127), _full(-127)))
    seven = np.tile(np.array([127] * 7 + [-127] * 9, np.int8), NG)
    rows.append(_row(_full(127), seven, -seven))
    chains = np.repeat(np.array([127, -127] * (NG // 2), np.int8), CLS)
    rows.append(_row(_full(127), chains, -chains, bm=1 << 16, kb=128))
    ref, lmax, hmax = _check(exe, rows, nch, tmp_path)
    assert lmax == LANE_MAX and hmax == LANE_MAX
    assert np.abs(ref).max() == 128 * nch * (1 << 16) * LANE_MAX
    print(f"packed corners nch {nch} wreg {int(wreg)}: {len(rows)} rows exact, lanes to "
          f"+-{lmax}, |acc| to {np.abs(ref).max()}")


@pytest.mark.parametrize("nch,wreg,seed", [(NG, False, 0), (NG, True, 1), (1, False, 2)])
def test_packed_random(have_verilator, nch, wreg, seed, tmp_path):
    """Random int8 rows in [-127, 127]: 1 to 128 blocks (the pi0.5 MLPs' K = 16384), each
    group's B a random MX shift (0: a dropped group; 2^sh, sh <= 16, as int8x; m << sh with a
    6-bit m as the 4-bit formats, under 2^17), the two lanes' B independent (two weight rows
    have their own offsets), and random bubbles between rows."""
    rng = np.random.default_rng(100 + seed)
    exe = _build(nch, True, wreg)

    def bm():
        kind = rng.random(NG)
        sh = rng.integers(0, 17, NG)
        m = np.where(kind < 0.7, 1, rng.integers(1, 64, NG))
        sh = np.where(m > 1, np.minimum(sh, 10), sh)
        return np.where(kind < 0.1, 0, m << sh).astype(np.int64)

    rows = []
    for _ in range(160):
        kb = int(rng.choice([1, 2, 3, 8, 17, 64, 128], p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]))
        blocks = []
        for _ in range(kb):
            blocks.append((rng.integers(-127, 128, D).astype(np.int8),
                           rng.integers(-127, 128, D).astype(np.int8),
                           rng.integers(-127, 128, D).astype(np.int8), bm(), bm()))
        rows.append(blocks)
    ref, lmax, hmax = _check(exe, rows, nch, tmp_path, rng, bubbles=3)
    nb = sum(len(r) for r in rows)
    print(f"packed random nch {nch} wreg {int(wreg)} seed {seed}: {len(rows)} rows, {nb} blocks "
          f"exact; lanes to {lmax} / {hmax}, |acc| to {np.abs(ref).max():.3g}")


def test_minus_128(have_verilator, tmp_path):
    """-128 in the operands (quant.py's int8 never holds it; a weight byte from outside it
    could). Exact: -128 activations against weights in [-127, 127] (|L| <= 16 * 128 * 127 =
    260,096 < 2^18), a -128 weight in the low lane (wa) against anything in [-127, 127], a -128
    weight in the high lane (wb) beside a low-lane weight >= 0, and fifteen (-128)^2 products
    with one -128 * -127 (262,016, the last sum under the bound). Two patterns break, pinned
    here: a chain of sixteen (-128)^2 products (+262,144 = 2^18) overflows the lane (the low
    lane reads -262,144, the high lane one more: the borrow); and wb = -128 beside a negative wa
    overflows the 27-bit pre-adder (wa + wb * 2^19 < -2^26), which wraps the high lane's
    product by 256 * act. Neither needs a hardware change: both need an operand of -128 the
    quantizer never produces."""
    exe = _build(NG, True, False)
    rng = np.random.default_rng(7)
    rows = [_row(_full(-128), _full(127), _full(-127)), _row(_full(-128), _full(-127), _full(-127)),
            _row(_full(-128), _full(-127), _full(127), bm=1 << 16, kb=128),
            _row(_full(-127), _full(-128), _full(-127)), _row(_full(127), _full(-128), _full(127)),
            _row(_full(-127), _full(-128), _full(127), bm=1 << 16, kb=128),
            _row(_full(-127), _full(0), _full(-128)), _row(_full(-127), _full(127), _full(-128)),
            _row(_full(127), _full(127), _full(-128), bm=1 << 16, kb=128)]
    for _ in range(12):
        kb = int(rng.integers(1, 9))
        rows.append(_row(_full(-128), rng.integers(-127, 128, D).astype(np.int8),
                         rng.integers(-127, 128, D).astype(np.int8), bm=1 << int(rng.integers(0, 17)),
                         kb=kb))
        rows.append(_row(rng.integers(-127, 128, D).astype(np.int8), _full(-128),
                         rng.integers(-127, 128, D).astype(np.int8), kb=kb))
        rows.append(_row(rng.integers(-127, 128, D).astype(np.int8),
                         rng.integers(0, 128, D).astype(np.int8), _full(-128), kb=kb))
    edge = np.tile(np.array([-128] * 15 + [-127], np.int8), NG)
    rows.append(_row(_full(-128), edge, _full(-127)))
    ref, lmax, hmax = _check(exe, rows, NG, tmp_path)
    assert lmax == 262016 and hmax == 260096
    print(f"-128: {len(rows)} rows exact with one operand -128, lanes to {lmax} / {hmax}")
    # the failing patterns: the lane bound (sixteen (-128)^2 in the low lane), and the
    # pre-adder's range (wb = -128 beside wa < 0)
    bad = [_row(_full(-128), _full(-128), _full(-127)), _row(_full(-127), _full(-1), _full(-128))]
    refb, _, _ = _ref(bad, NG)
    got = _run(exe, _lines(bad), tmp_path)
    assert refb[0].tolist() == [NG * 262144, NG * 260096]
    assert got[0].tolist() == [NG * (262144 - (1 << 19)), NG * (260096 + 1)], got[0]
    assert refb[1].tolist() == [NG * 16 * 127, NG * 260096]
    assert got[1].tolist() == [NG * 16 * 127, NG * (260096 - 16 * 127 * 256)], got[1]
    print(f"-128: the failing rows read {got.tolist()} for {refb.tolist()}")


def test_unpacked_column(have_verilator, tmp_path):
    """PACK 0 (the out-of-context comparison column: otpu_pe, one column end) computes row a's
    integer; acc_b is 0."""
    exe = _build(NG, False, True)
    rng = np.random.default_rng(3)
    rows = [_row(_full(127), _full(-127), _full(5), bm=1 << 16, kb=128)]
    for _ in range(12):
        rows.append([(rng.integers(-127, 128, D).astype(np.int8),
                      rng.integers(-127, 128, D).astype(np.int8),
                      rng.integers(-127, 128, D).astype(np.int8),
                      1 << rng.integers(0, 17, NG), 1 << rng.integers(0, 17, NG))
                     for _ in range(int(rng.integers(1, 20)))])
    ref, _, _ = _ref(rows, NG)
    got = _run(exe, _lines(rows, rng, 2), tmp_path)
    assert np.array_equal(got[:, 0], ref[:, 0])
    assert np.all(got[:, 1] == 0)
