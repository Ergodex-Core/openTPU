"""Device-only residual activation quantization using existing QACT/MM/VOP.

The identity MM reconstructs QACT's first activation copy one D-block at a
time. A second QACT quantizes x - dequant(QACT(x)); its weighted contribution
is accumulated. Immutable weights keep their exact stored values.
"""
from .. import language as ol
from ..compiler import Stationary


def _primary_block(st, k):
    """Read just the original ACT copy, also on DUP/DUP4/DUP8 profiles.

    The compiler's blocks() conservatively rejects duplicated operands. This
    explicit view deliberately disables copy reuse and uses the first mc rows;
    no QACT bytes/scales move, and the parent retains ACT allocation ownership.
    """
    if len(st.chunks) != 1 or not 0 <= k < st.KB or st.qranges is not None:
        raise ValueError("residual precision requires a single whole-QACT row group")
    ab, mc = st.chunks[0]
    D = st.src.b.cfg.D
    src = st.src[:, k * D:(k + 1) * D] if len(st.src.shape) == 2 else st.src[k * D:(k + 1) * D]
    view = Stationary(src, [(ab + k, mc)], 1, list(st.owners))
    view.loop, view.parent = getattr(st, "loop", None), st
    return view


def linear(x, weights, identity=None, passes=1):
    if passes not in (1, 2):
        raise ValueError("activation passes must be 1 or 2")
    weights = weights if isinstance(weights, tuple) else (weights,)
    q = ol.quantize(x)
    y = ol.dot(q, weights[0])
    for weight in weights[1:]:
        ol.dot(q, weight, acc=y)
    if passes == 2:
        if identity is None:
            raise ValueError("residual quantization requires the immutable D-by-D identity")
        reconstructed = ol.empty(x.shape)  # odd row stride for MXU output banks
        D = ol.block_size()
        for k in range(x.cols // D):
            view = _primary_block(q, k)
            ol.dot(view, identity, out=reconstructed[:, k * D:(k + 1) * D])
            del view
        residual = x - reconstructed
        del reconstructed, q
        correction = ol.quantize(residual)
        for weight in weights:
            ol.dot(correction, weight, acc=y)
    return y
