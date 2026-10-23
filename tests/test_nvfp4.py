"""W4A16 proposal conformance, exact-rational oracle and legacy compatibility."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import random

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import nvfp4 as N
from opentpu.mxu_job import Buffer, W4A16Job, PROFILE_DIGESTS


def test_format_tables_and_canonical_packing():
    values = [Fraction(x, 2) for x in (0, 1, 2, 3, 4, 6, 8, 12)]
    assert [N.e2m1_value(c) for c in range(16)] == values + [-x for x in values]
    assert N.e4m3_value(0) == 0 and N.e4m3_value(1) == Fraction(1, 512)
    assert N.e4m3_value(0x38) == 1 and N.e4m3_value(0x7e) == 448
    assert all(N.e4m3_value(c) < N.e4m3_value(c + 1) for c in range(126))
    for c in range(127, 256):
        with pytest.raises(N.NumericalFault):
            N.e4m3_value(c)
    assert N.pack_codes(range(16)) == bytes.fromhex("1032547698badcfe")
    assert N.pack_codes([1, 2, 3]) == bytes.fromhex("2103")
    assert N.unpack_codes(bytes.fromhex("21f3"), 3) == [1, 2, 3]  # high tail ignored


def test_rne_ties_underflow_carry_signed_zero_overflow():
    assert N.round32(1 + N.pow2(-24)) == 0x3f800000
    assert N.round32(1 + 3 * N.pow2(-24)) == 0x3f800002
    assert N.round32(2 - N.pow2(-24)) == 0x40000000
    assert N.round32(N.pow2(-150)) == 0
    assert N.round32(3 * N.pow2(-150)) == 2
    assert N.round32(N.pow2(-126) - N.pow2(-150)) == 0x00800000
    assert N.round32(N.pow2(-126) - N.pow2(-150), ftz=True) == 0x00800000
    assert N.round32(N.pow2(-149), ftz=True) == 0
    assert N.round32(-N.pow2(-151)) == 0x80000000
    assert N.mul32(0x80000000, 0x3f800000) == 0x80000000
    assert N.add32(0x80000000, 0x80000000) == 0x80000000
    assert N.add32(0x80000000, 0) == 0
    assert N.add32(0x3f800000, 0xbf800000) == 0
    with pytest.raises(N.NumericalFault):
        N.round32(N.pow2(128) - N.pow2(103))
    for bits in (0x7f800000, 0xff800000, 0x7fc00000):
        with pytest.raises(N.NumericalFault):
            N.mul32(bits, 0)


def test_rational_primitives_against_independent_ieee_host():
    rng = np.random.default_rng(20261022)
    a = rng.integers(0, 1 << 32, 4000, dtype=np.uint32)
    b = rng.integers(0, 1 << 32, 4000, dtype=np.uint32)
    finite = ((a >> 23) & 255 != 255) & ((b >> 23) & 255 != 255)
    a, b = a[finite], b[finite]
    for op, ref in ((N.add32, np.add), (N.mul32, np.multiply)):
        with np.errstate(all="ignore"):
            want = ref(a.view(np.float32), b.view(np.float32), dtype=np.float32).view(np.uint32)
        for x, y, w in zip(a, b, want):
            if int(w) & 0x7f800000 == 0x7f800000:
                with pytest.raises(N.NumericalFault):
                    op(int(x), int(y))
            else:
                assert op(int(x), int(y)) == int(w)


def test_strict_order_scales_tail_and_subnormal_profiles():
    # Exact alignment is still distinct from separately rounded FP32 tree additions.
    a, c = [0x4c00, 0x3f80, 0xcc00, 0x3f80] + [0] * 12, [2] * 16
    assert N.strict_dot16(a, c) == 0
    fast = N.aligned_dot16(a, c)
    assert not fast.discarded and fast.value == 2
    # Two K16 blocks, explicit block scaling and one final global multiply.
    a = [0x3f80] * 17  # 17 ones
    assert N.strict_row(a, N.pack_codes([2] * 17), [0x38, 0x40], 0x3f000000) == 0x41100000
    assert N.strict_row([1], b"\x02", [0x38], 0x3f800000) == 0x00010000
    assert N.strict_row([1], b"\x02", [0x38], 0x3f800000, profile=N.STRICT_FTZ) == 0
    assert N.strict_dot16([0x8000] * 16, [2] * 16) == 0x80000000
    # Overflow is detected before a compensating or zero scale, never hidden.
    with pytest.raises(N.NumericalFault, match="overflow"):
        N.strict_row([0x7f7f], b"\x07", [0], 0)
    with pytest.raises(N.NumericalFault):
        N.strict_row([0x7f80], b"\x00", [0], 0)


@pytest.mark.parametrize("ftz", [False, True])
def test_strict_rows_against_independent_numpy_tree(ftz):
    rng = random.Random(932)
    def flush(x):
        x = np.float32(x)
        bits = int(x.view(np.uint32))
        return np.uint32(bits & 0x80000000).view(np.float32) if ftz and not bits & 0x7f800000 else x
    def op(f, a, b):
        return flush(f(flush(a), flush(b)))
    for case in range(100):
        k = 1 + case % 49
        a = [((rng.randrange(100, 150) if i % 7 else 0) << 7) |
             rng.randrange(128) | (rng.randrange(2) << 15) for i in range(k)]
        c = [rng.randrange(16) for _ in range(k)]
        scales = [rng.randrange(127) for _ in range((k + 15) // 16)]
        acc = np.float32(0)
        for b in range(len(scales)):
            p = []
            for i in range(16):
                j = 16*b+i
                x = np.uint32(a[j] << 16).view(np.float32) if j < k else np.float32(0)
                code = c[j] if j < k else 0
                q = np.float32([0, .5, 1, 1.5, 2, 3, 4, 6][code & 7])
                if code & 8:
                    q = -q
                p.append(op(np.multiply, x, q))
            while len(p) > 1:
                p = [op(np.add, p[i], p[i+1]) for i in range(0, len(p), 2)]
            e, m = scales[b] >> 3, scales[b] & 7
            s = np.float32((m if e == 0 else m+8) * 2.0 ** (max(e, 1)-10))
            acc = op(np.add, acc, op(np.multiply, p[0], s))
        want = op(np.multiply, acc, np.uint32(0x3f012345).view(np.float32))
        got = N.strict_row(a, N.pack_codes(c), scales, 0x3f012345,
                           profile=N.STRICT_FTZ if ftz else N.STRICT_GRADUAL)
        assert got == int(want.view(np.uint32))


@pytest.mark.parametrize("f", [16, 24])
def test_alignment_bound_width_and_discarded(f):
    rng = random.Random(100 + f)
    for case in range(1200):
        # Every finite BF16 exponent, signs, subnormals and both zero encodings.
        a = [(rng.randrange(2) << 15) | (rng.randrange(255) << 7) | rng.randrange(128)
             for _ in range(16)]
        if case % 3 == 0:
            e = rng.randrange(1, 255 - f)
            a = [(x & 0x807f) | ((e + rng.randrange(f + 1)) << 7) for x in a]
        c = [rng.randrange(16) for _ in range(16)]
        mask = rng.randrange(1 << 16)
        got = N.aligned_dot16(a, c, fraction_bits=f, mask=mask)
        exact = sum((N.fp32_value(x << 16) * N.e2m1_value(w)
                     for i, (x, w) in enumerate(zip(a, c)) if mask >> i & 1), Fraction())
        assert abs(got.value - exact) <= 48 * N.pow2(got.e_star - f)
        assert -(1 << (f + 16)) <= got.integer < 1 << (f + 16)
        if not got.discarded:
            assert got.value == exact
    # Tie even in both directions and negative values; gap F+8 rounds 128/256 to 0.
    a = [0x7e80, ((253 - f - 8) << 7), ((253 - f - 8) << 7) | 1] + [0] * 13
    assert N.aligned_dot16(a, [2] * 16, fraction_bits=f).discarded


def job():
    return W4A16Job(Buffer(1 << 40, 68, 34), Buffer(0x1000, 27, 9),
                    Buffer(0x2000, 6, 2), Buffer(0x3000, 24, 12),
                    tokens=2, outputs=3, k=17, global_bits=0x3f800000)


def test_job_snapshot_roundtrip_and_existing_isa_unchanged():
    j = job()
    d = j.to_mapping()
    assert W4A16Job.from_mapping(d) == j
    d["a"]["address"] = 0
    assert j.a.address == 1 << 40
    with pytest.raises(FrozenInstanceError):
        j.epoch = 1
    assert I.MM == 0x20 and I.WNV == 3 and "MX_LAUNCH" not in I.OPNAMES.values()
    assert I.Instr.decode(I.mm(0, 0, 0, 1, 1, 128, 1, 1, 0, 32, wf=I.WNV).encode()).op == I.MM


@pytest.mark.parametrize("change", [
    {"abi_major": 2}, {"abi_minor": 1}, {"abi_major": True}, {"flags": 1}, {"reserved": 1},
    {"k": 0}, {"tokens": -1}, {"outputs": 1 << 32}, {"profile": N.FAST_F24},
    {"profile_digest": "0" * 64}, {"transform": "RHT64"}, {"scale_direction": "divide"},
    {"output_format": "BF16"}, {"post_operation": "bias"}, {"generation": 1 << 64},
    {"global_bits": 0x7f800000}, {"global_bits": 0x80000000}, {"dependencies": [1]},
    {"a": Buffer(1 << 64, 68, 34)}, {"a": Buffer((1 << 64) - 2, 68, 34)},
    {"a": Buffer(1, 68, 34)}, {"a": Buffer(0, 68, 32)}, {"scales": Buffer(0x2000, 5, 2)},
    {"w": Buffer(0x1000, 27, 8)}, {"y": Buffer(0x1000, 24, 12)},
])
def test_job_rejects_unsupported_or_unsafe_layout(change):
    with pytest.raises(ValueError):
        replace(job(), **change)


def test_parser_rejects_unknown_missing_fields_and_accepts_distinct_ftz_tuple():
    d = job().to_mapping()
    for bad in (dict(d, mystery=1), dict(d, a={"address": 0}), {}, dict(d, dependencies="1"),
                {k: v for k, v in d.items() if k != "abi_major"}):
        with pytest.raises(ValueError):
            W4A16Job.from_mapping(bad)
    assert replace(job(), profile=N.STRICT_FTZ,
                   profile_digest=PROFILE_DIGESTS[N.STRICT_FTZ]).profile == N.STRICT_FTZ
