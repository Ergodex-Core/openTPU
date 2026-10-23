"""Separate converter, scale, sum and final-global rounding boundaries."""
from fractions import Fraction
import random

import pytest

from opentpu import nvfp4 as N


def test_conversion_sign_width_ties_and_range():
    for sign in (-1, 1):
        assert N.convert_subtotal(sign * ((1 << 24)+1), 25) == N.round32(sign * (1 << 24))
        assert N.convert_subtotal(sign * ((1 << 24)+3), 25) == N.round32(sign * ((1 << 24)+4))
        assert N.convert_subtotal(sign, -125) == (0x80000000 if sign < 0 else 0)
        assert N.convert_subtotal(sign * 3, -125) == (0x80000002 if sign < 0 else 2)
    assert N.convert_subtotal(-(1 << 40), 25) == N.round32(-(1 << 40))
    assert N.convert_subtotal(0, -133) == 0
    for exponent in (-256, -134, 121, 255):
        with pytest.raises(N.NumericalFault):
            N.convert_subtotal(0, exponent)
    with pytest.raises(N.NumericalFault, match="overflow"):
        N.convert_subtotal((1 << 40)-1, 120)


def test_epilogue_separate_rounding_and_no_compensating_scale_shortcut():
    # A rounded integer subtotal followed by scale must differ from a fused convert/scale.
    z = (1 << 24)+1
    dot = N.AlignedDot(z, 25, False, 24)
    got = N.accumulate_subtotals([dot], [0x3c], 0x3f800000, profile=N.FAST_F24)
    fused = N.round32(z * Fraction(3, 2))
    assert got == N.round32((1 << 24) * Fraction(3, 2)) and got != fused
    # Sequential accumulation loses the one; changing block order changes its answer.
    big, one, neg = N.round32(1 << 25), 0x3f800000, N.round32(-(1 << 25))
    assert N.accumulate_subtotals([big, one, neg], [0x38]*3, 0x3f800000) == 0
    assert N.accumulate_subtotals([big, neg, one], [0x38]*3, 0x3f800000) == one
    with pytest.raises(N.NumericalFault):
        N.accumulate_subtotals([N.AlignedDot((1 << 40)-1, 120, False, 24)], [0], 0,
                               profile=N.FAST_F24)
    with pytest.raises(N.NumericalFault, match="fallback"):
        N.accumulate_subtotals([N.AlignedDot(0, 0, True, 24)], [0], 0, profile=N.FAST_F24)


def test_subnormal_rounding_before_ftz_and_global_last():
    assert N.mul32(0x00800000, 0x3f7fffff, ftz=True) == 0x00800000
    assert N.add32(0x007fffff, 1) == 0x00800000
    assert N.add32(0x007fffff, 1, ftz=True) == 0
    assert N.accumulate_subtotals([0x00800000], [0x30], 0x40000000) == 0x00800000
    assert N.accumulate_subtotals([0x00800000], [0x30], 0x40000000,
                                 profile=N.STRICT_FTZ) == 0


@pytest.mark.parametrize("profile", [N.STRICT_GRADUAL, N.STRICT_FTZ])
def test_epilogue_matches_strict_row(profile):
    rng = random.Random(118)
    ftz = profile == N.STRICT_FTZ
    for k in range(1, 66):
        a = [(rng.randrange(110, 140) << 7) | rng.randrange(128) |
             (rng.randrange(2) << 15) for _ in range(k)]
        c = [rng.randrange(16) for _ in range(k)]
        s = [rng.randrange(127) for _ in range((k+15)//16)]
        dots = []
        for i in range(0, k, 16):
            n = min(16, k-i)
            dots.append(N.strict_dot16(a[i:i+n]+[0]*(16-n), c[i:i+n]+[0]*(16-n), ftz=ftz))
        assert N.accumulate_subtotals(dots,s,0x3ef12345,profile=profile) == N.strict_row(
            a,N.pack_codes(c),s,0x3ef12345,profile=profile)
