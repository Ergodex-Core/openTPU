"""RhinoVLA V3 (opentpu/llm/rhinovla.py, docs/rhinovla.md) on a tiny random-weight model: the
host side's pieces against the HF / repo definitions, and the card's phases on the ISA
simulator (int8 weights and operands) against the numpy fp32 reference. The reference itself
is checked against the repo's PyTorch model by tools/rhinovla_reference.py + rhinovla_check.py
--reference (docs/rhinovla.md 2), not here (the repo is not a dependency of the tests)."""
from __future__ import annotations

import numpy as np
import pytest

from opentpu.llm import rhinovla as RV

TINY = RV.TINY
# the processor's token ids of the v2 prompt for TINY's 4-token views (tools/
# rhinovla_reference.py --tiny saw them): 9 text, 4 pads, 9 text, 4 pads, 9 text, 4 pads, 14 text
IDS = ([151644, 872, 198, 71333, 13138, 91, 16509, 60, 151652] + [RV.IMAGE_PAD] * 4
       + [151653, 198, 58, 10661, 9579, 91, 16509, 60, 151652] + [RV.IMAGE_PAD] * 4
       + [151653, 198, 58, 10661, 9859, 91, 16509, 60, 151652] + [RV.IMAGE_PAD] * 4
       + [151653, 198, 628, 279, 3691, 19212, 389, 279, 11968, 151645, 198, 151644, 77091, 198])


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    W = RV.random_weights(TINY, rng)
    imgs = [rng.uniform(-1, 1, (3, TINY.image, TINY.image)).astype(np.float32)
            for _ in range(TINY.views)]
    lay = RV.layout(TINY, IDS)
    state = rng.normal(0, 0.5, TINY.state_dim).astype(np.float32)
    noise = rng.standard_normal((TINY.chunk, TINY.action_dim)).astype(np.float32)
    amask = np.ones(TINY.action_dim, np.float32)
    amask[40:] = 0                      # 40 live slots, as a robot's mapping would set
    return W, imgs, lay, state, noise, amask


@pytest.fixture(scope="module")
def card(tiny):
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import IsaBackend
    W = tiny[0]
    image = RV.Image(TINY, board_config(), np_max=64)
    cfg = board_config(DRAM_BYTES=image.end)
    image.cfg = cfg
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)

    def put(addr, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    image.build(W, put)
    return image, IsaBackend(cfg, [img])


def test_layout_is_get_rope_index():
    """The mRoPE positions of the tiny prompt: text tokens count up, a view's 4 tokens take
    the view's start on T and start + row / column on H / W, the next text token starts
    merge-grid (2) later; the suffix starts at max + 1 (rope delta = max + 1 - L)."""
    lay = RV.layout(TINY, IDS)
    assert lay.L == len(IDS) == 53 and lay.starts == [9, 22, 35]
    assert lay.pos[:, 8].tolist() == [8, 8, 8]
    assert lay.pos[:, 9:13].tolist() == [[9, 9, 9, 9], [9, 9, 10, 10], [9, 10, 9, 10]]
    assert lay.pos[:, 13].tolist() == [11, 11, 11]
    assert lay.suffix_pos == lay.pos.max() + 1 == lay.L - 6


def test_patches_are_the_conv3d_in_window_order():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    img = rng.standard_normal((3, 64, 64)).astype(np.float32)
    w = rng.standard_normal((8, 3, 2, 16, 16)).astype(np.float32)
    x = torch.from_numpy(np.stack([img, img], 1)[None])           # [1, 3, 2, 64, 64]
    ref = torch.nn.functional.conv3d(x, torch.from_numpy(w), stride=(2, 16, 16))[0, :, 0]
    ref = ref.reshape(8, 4, 4).numpy()                              # [C, gh, gw]
    wp = w.astype(np.float64).sum(2).reshape(8, -1).astype(np.float32)
    got = RV.patches(img, 16, 2) @ wp.T                             # window order
    rc = RV.window_order(4, 2)
    assert np.abs(got - ref[:, rc[:, 0], rc[:, 1]].T).max() < 1e-3


def test_vit_rope_table_is_hf():
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    d = TINY.v_head_dim
    rot = Qwen3VLVisionRotaryEmbedding(d // 2)
    tab = rot(4)
    rc = torch.from_numpy(RV.window_order(4, 2))
    emb = tab[rc].flatten(1)
    cos, sin = RV.vit_rope_table(TINY)
    assert np.abs(cos - emb.cos().numpy()).max() < 1e-6 and np.abs(sin - emb.sin().numpy()).max() < 1e-6


def test_the_programs_fit_imem(card, tiny):
    image, _ = card
    lay = tiny[2]
    for name, (prog, _) in image.programs(lay).items():
        assert len(prog) <= image.cfg.IMEM_WORDS // 8, (name, len(prog))


def _cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def test_the_chunk_on_the_card_is_the_reference(tiny, card):
    """The whole tiny policy on the ISA simulator (3 views, the causal prefix with deepstack,
    3 Euler steps, int8): the merged image rows, and the actions, close to the fp32
    reference; the dead slots exactly 0."""
    W, imgs, lay, state, noise, amask = tiny
    image, be = card
    host = RV.Host(TINY, W, lay, state, None, amask)
    progs = image.programs(lay)
    cap = {}
    ref = RV.reference_actions(TINY, W, imgs, lay, state, noise, None, amask, capture=cap)
    RV.run_chunk(image, be, progs, imgs, host, noise, stop="vision")
    vis = RV.read_prefix_rows(image, be, lay)
    rv = np.concatenate([cap[f"vis{v}"] for v in range(TINY.views)])
    assert _cos(vis, rv).min() > 0.995
    got = RV.run_chunk(image, be, progs, imgs, host, noise)
    assert np.all(got[:, 40:] == 0)
    err = np.abs(got - ref)[:, :40].max()
    rng_ = ref[:, :40].max() - ref[:, :40].min()
    assert err < 0.05 * rng_, (err, rng_)
