"""W4A16 proposal v1 reference, independent of MM's int8/MX arithmetic.

All operands/results are integer bit patterns. Fraction arithmetic supplies exact
binary intermediates; host float, BLAS and the existing FTZ fp32 model are unused.
This is a small conformance oracle, not an inference backend.
"""
from dataclasses import dataclass
from fractions import Fraction

STRICT_GRADUAL = "nvfp4-w4a16-tree16-rne-gradual-v1"
STRICT_FTZ = "nvfp4-w4a16-tree16-rne-daz-ftz-v1"
FAST_F24 = "nvfp4-w4a16-align-f24-rne-gradual-v1"
D2 = (0, 1, 2, 3, 4, 6, 8, 12)  # twice the E2M1 magnitude


class NumericalFault(ValueError):
    """Nonfinite operand or overflow at a declared FP32 rounding boundary."""


def _uint(x, bits):
    if type(x) is not int or not 0 <= x < 1 << bits:
        raise ValueError(f"expected unsigned {bits}-bit integer, got {x!r}")
    return x


def pow2(e):
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def fp32_value(bits):
    _uint(bits, 32)
    e, m = (bits >> 23) & 255, bits & 0x7fffff
    if e == 255:
        raise NumericalFault("nonfinite FP32 operand")
    v = (m if e == 0 else (1 << 23) | m) * pow2(-149 if e == 0 else e - 150)
    return -v if bits >> 31 else v


def _rne(x):
    q, r = divmod(x.numerator, x.denominator)
    return q + (2 * r > x.denominator or (2 * r == x.denominator and q & 1))


def round32(value, *, negative_zero=False, ftz=False):
    """Exact rational to binary32 RNE, with explicit overflow failure."""
    v = Fraction(value)
    sign = v < 0 or (not v and negative_zero)
    v = abs(v)
    if not v:
        return int(sign) << 31
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if v < pow2(e):
        e -= 1
    if e < -126:
        m = _rne(v / pow2(-149))
        bits = m  # rounding may produce the smallest normal, 0x00800000
        if ftz and bits < 1 << 23:
            bits = 0
    else:
        m = _rne(v / pow2(e - 23))
        if m == 1 << 24:
            e, m = e + 1, m >> 1
        if e > 127:
            raise NumericalFault("FP32 overflow at rounding boundary")
        bits = ((e + 127) << 23) | (m - (1 << 23))
    return (int(sign) << 31) | bits


def _operand(bits, ftz):
    fp32_value(bits)  # validate before any zero shortcut
    return bits & 0x80000000 if ftz and not bits & 0x7f800000 else bits


def add32(a, b, *, ftz=False):
    a, b = _operand(a, ftz), _operand(b, ftz)
    return round32(fp32_value(a) + fp32_value(b),
                   negative_zero=bool(a >> 31 and b >> 31), ftz=ftz)


def mul32(a, b, *, ftz=False):
    a, b = _operand(a, ftz), _operand(b, ftz)
    return round32(fp32_value(a) * fp32_value(b),
                   negative_zero=bool((a ^ b) >> 31), ftz=ftz)


def e2m1_value(code):
    _uint(code, 4)
    return Fraction((-1 if code & 8 else 1) * D2[code & 7], 2)


def e4m3_value(code):
    _uint(code, 8)
    if code > 0x7e:
        raise NumericalFault("E4M3 scale must be unsigned finite (0x00..0x7e)")
    e, m = code >> 3, code & 7
    return (m if e == 0 else 8 + m) * pow2(-9 if e == 0 else e - 10)


def pack_codes(codes):
    """Even K in low nibble; odd K in high; unused high nibble is zero."""
    c = [_uint(x, 4) for x in codes]
    return bytes(c[i] | ((c[i + 1] if i + 1 < len(c) else 0) << 4)
                 for i in range(0, len(c), 2))


def unpack_codes(data, k):
    if type(k) is not int or k <= 0 or len(data) != (k + 1) // 2:
        raise ValueError("packed row length does not match K")
    return [(data[i // 2] >> (4 * (i % 2))) & 15 for i in range(k)]


def strict_dot16(activations, codes, *, ftz=False):
    if len(activations) != 16 or len(codes) != 16:
        raise ValueError("strict_dot16 requires exactly 16 elements")
    p = []
    for a, c in zip(activations, codes):
        _uint(a, 16)
        q = round32(e2m1_value(c), negative_zero=bool(c & 8))
        p.append(mul32(a << 16, q, ftz=ftz))
    while len(p) > 1:
        p = [add32(p[i], p[i + 1], ftz=ftz) for i in range(0, len(p), 2)]
    return p[0]


def strict_row(activations, packed_weights, scales, global_bits, *, profile=STRICT_GRADUAL):
    """One output: tree16 -> block scale -> ordered add -> global factor.

    Missing tail elements are +0. Each operation rounds separately. No fused
    multiply-add, implicit bias or BF16 cast. Zero scale does not bypass the dot.
    """
    if profile not in (STRICT_GRADUAL, STRICT_FTZ):
        raise ValueError("unsupported strict profile")
    ftz = profile == STRICT_FTZ
    k = len(activations)
    codes = unpack_codes(packed_weights, k)
    if len(scales) != (k + 15) // 16:
        raise ValueError("one scale required per K16 group")
    if fp32_value(global_bits) < 0 or global_bits >> 31:
        raise NumericalFault("global scale must be unsigned finite FP32")
    acc = 0
    for i in range(0, k, 16):
        n = min(16, k - i)
        dot = strict_dot16(list(activations[i:i + n]) + [0] * (16 - n),
                           codes[i:i + n] + [0] * (16 - n), ftz=ftz)
        scale = round32(e4m3_value(scales[i // 16]))
        acc = add32(acc, mul32(dot, scale, ftz=ftz), ftz=ftz)
    return mul32(acc, global_bits, ftz=ftz)


@dataclass(frozen=True)
class AlignedDot:
    integer: int
    e_star: int
    discarded: bool
    fraction_bits: int

    @property
    def value(self):
        return self.integer * pow2(self.e_star - self.fraction_bits - 1)


def aligned_dot16(activations, codes, *, fraction_bits=24, mask=0xffff):
    """Fast profile subtotal only. Discarded bits require explicit fallback/policy.

    Even when discarded=False this is NOT the strict FP32 pairwise tree.
    Software also supports F16 experiments; the first RTL is fixed F24.
    """
    if len(activations) != 16 or len(codes) != 16 or fraction_bits not in (16, 24):
        raise ValueError("expected K16 and F16 or F24")
    _uint(mask, 16)
    decoded = []
    for i, (a, c) in enumerate(zip(activations, codes)):
        _uint(a, 16)
        _uint(c, 4)
        if not mask >> i & 1:
            decoded.append((0, -133, c))
            continue
        e, m = (a >> 7) & 255, a & 127
        if e == 255:
            raise NumericalFault("nonfinite BF16 activation")
        decoded.append(((-1 if a >> 15 else 1) * (m if e == 0 else 128 + m),
                        -133 if e == 0 else e - 134, c))
    top = max((e for m, e, _ in decoded if m), default=-133)
    total, discarded = 0, False
    for m, e, c in decoded:
        exact = m * pow2(fraction_bits + e - top)
        aligned = _rne(abs(exact)) * (-1 if m < 0 else 1)
        discarded |= exact != aligned
        total += aligned * D2[c & 7] * (-1 if c & 8 else 1)
    assert -(1 << (fraction_bits + 16)) <= total < 1 << (fraction_bits + 16)
    return AlignedDot(total, top, discarded, fraction_bits)


def convert_subtotal(integer, e_star):
    """F24 signed41 subtotal to FP32 RNE, BEFORE the scale multiplication."""
    if type(integer) is not int or not -(1 << 40) <= integer < 1 << 40:
        raise ValueError("subtotal outside signed41")
    if type(e_star) is not int or not -133 <= e_star <= 120:
        raise NumericalFault("e_star outside finite BF16 decode range")
    return round32(integer * pow2(e_star - 25))


def accumulate_subtotals(subtotals, scales, global_bits, *, profile=STRICT_GRADUAL):
    """Ordered, separately rounded epilogue for one already-produced row.

    Strict profiles consume FP32 tree subtotal bits; FAST_F24 consumes AlignedDot
    records. Fast discarded alignment is rejected until a separate error policy
    is approved. This does not turn an aligned subtotal into the strict tree.
    """
    if profile not in (STRICT_GRADUAL, STRICT_FTZ, FAST_F24):
        raise ValueError("unsupported epilogue profile")
    if not subtotals or len(subtotals) != len(scales):
        raise ValueError("nonempty matching subtotals/scales required")
    if global_bits >> 31 or fp32_value(global_bits) < 0:
        raise NumericalFault("global scale must be unsigned finite FP32")
    ftz = profile == STRICT_FTZ
    acc = 0
    for dot, scale in zip(subtotals, scales):
        if profile == FAST_F24:
            if not isinstance(dot, AlignedDot) or dot.fraction_bits != 24:
                raise ValueError("F24 epilogue needs F24 AlignedDot")
            if dot.discarded:
                raise NumericalFault("discarded alignment requires strict fallback")
            dot = convert_subtotal(dot.integer, dot.e_star)
        else:
            fp32_value(dot)
        term = mul32(dot, round32(e4m3_value(scale)), ftz=ftz)
        acc = add32(acc, term, ftz=ftz)
    return mul32(acc, global_bits, ftz=ftz)
