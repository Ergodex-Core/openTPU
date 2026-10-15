"""The activation side of a Hadamard-folded checkpoint (Ternary-Bonsai 2, docs/ternary.md 4).

Its projections are stored as W_f = W R^T in a rotated input basis, R x = H_B (s * x): s a +-1
vector per input width, H_B the normalized Sylvester Walsh-Hadamard matrix of B (512 for
Bonsai-2-2B and 4B, 4096 for 9B, 1024 for 27B), block-diagonal over the input. The image keeps
the ternary W_f (t2) and the kernels rotate each projection's input first:

* x * s' (a VOP; s' = s * B^-1/2, so every Hadamard factor below has the integers +-1);
* the rows viewed as K / B0 rows of B0 = min(B, 512) times H_B0 (one MM, H_B0 a UNIT t2 or
  int4 matrix of B0 rows: 64 KB a transform);
* for B > B0, H_B = H_r (x) H_B0 with r = B / B0 (Sylvester: entry (r, c)'s sign is the parity
  of r & c, so the high and low index bits factor): log2(r) butterfly stages of adds and
  subtracts across each group of r consecutive B0-rows (VOPs), y_hi = sum_c H_r[hi, c] z_c.

The output rows laid end to end are R x. H_B is block-diagonal, so a column part of whole
blocks of B (a W_down chunk, an out_proj head group) rotates on its own with its part of s'.
"""
from __future__ import annotations

from types import SimpleNamespace

from .. import language as ol
from ..compiler import Tensor

BASE = 512          # the MM's Hadamard block (B0): larger blocks add butterfly stages


class Rotation(SimpleNamespace):
    """An image's rotation: `block` (B), `base` (B0), `H` (QTensor [B0, B0], UNIT), and
    `signs` {K: the DRAM address of s' for inputs of width K}."""

    def sign(self, K: int, c0=0, n: int | None = None) -> Tensor:
        """s' of width K, elements c0 .. c0 + n (c0 may be a loop expression)."""
        if K not in self.signs:
            raise ol.CompileError(f"no rotation signs for an input of width {K} "
                                  f"(the image has {sorted(self.signs)})")
        t = Tensor(self.signs[K], (K,), (1,))
        n = K if n is None else n
        return t if n == K else t[c0:c0 + n]


def rotate(v, rot: Rotation | None, K: int | None = None, c0=0):
    """R v over the columns of v [M, n] (fp32 tile): v * s'[c0 .. c0 + n], each row's blocks
    of B0 times H_B0, then the butterflies of H_r over each block of B. K: the full input width
    whose signs apply (default n: the whole input); c0: v's first column within it (a whole
    number of blocks of B). None: v itself."""
    if rot is None:
        return v
    B, B0 = rot.block, rot.base
    M, n = v.rows, v.cols
    K = n if K is None else K
    if n % B:
        raise ol.CompileError(f"rotate: {n} columns are not whole blocks of {B}")
    y = v * ol.load(rot.sign(K, c0, n))[None, :]
    z = ol.empty([M * n])                   # contiguous: its rows laid end to end are R v
    ol.dot(y.reshape(M * n // B0, B0), rot.H, out=z.reshape(M * n // B0, B0))
    r = B // B0
    if r > 1:                               # H_r across each block's r rows of B0: in place
        g = z.reshape(M * n // B, B)        # [blocks, r B0]: column part i is row i of each
        h = 1
        while h < r:
            for i in range(r):
                if i & h:
                    continue
                a, b = g[:, i * B0:(i + 1) * B0], g[:, (i + h) * B0:(i + h + 1) * B0]
                t = a + b
                b.set(a - b)
                a.set(t)
            h *= 2
    return z.reshape(M, n)
