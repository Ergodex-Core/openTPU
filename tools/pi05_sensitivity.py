"""Which of the card's quantization points cost pi0.5's actions accuracy (docs/pi05.md 4):
the fp32 reference with opentpu.llm.pi05.Emulated at some of them, against the exact
reference, on tools/pi05_reference.py's observation. The tower's rows are lerobot's own
(exact): the prefix and the expert are what is measured.

    python tools/pi05_sensitivity.py --ckpt DIR --ref ref.npz [--json out.json]
        [--variants all,w8a16,...]

Variants (prefix numerics / expert numerics):
  w8a8        weights int8 (MX), activations int8 (QACT), attention int8: the card
  w8a8_io32   the card, action_in_proj / action_out_proj exact (fp32 on the host / VPU)
  w8a16       weights int8, activations and attention exact ("W8A16": wider activations)
  prefix_only the prefix the card's, the expert exact
  expert_only the expert the card's, the prefix exact
  kv_only     only the attention's operands int8, everywhere
  prefix_a_*  the prefix's weights int8 and its activations int8 into q / k / v, o, gate /
              up or down only (the expert exact)
Each prints the actions' error against the exact reference: max |err| and RMS over the 50
x 7 real dims (model space: normalized), and the unnormalized max per dim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.llm import pi05 as P  # noqa: E402

VARIANTS = {
    "w8a8": (("w", "a", "kv"), True, ("w", "a", "kv")),
    "w8a8_io32": (("w", "a", "kv"), False, ("w", "a", "kv")),
    "w8a16": (("w",), True, ("w",)),
    "prefix_only": (("w", "a", "kv"), True, ()),
    "expert_only": ((), True, ("w", "a", "kv")),
    "kv_only": (("kv",), True, ("kv",)),
    # the prefix's int8 weights with its activations int8 into some matrices only (the
    # expert exact): which of the 2B's operands cost the most
    "prefix_a_qkv": (("w", "a"), True, (), ("q_proj", "k_proj", "v_proj")),
    "prefix_a_o": (("w", "a"), True, (), ("o_proj",)),
    "prefix_a_gateup": (("w", "a"), True, (), ("gate_proj", "up_proj")),
    "prefix_a_down": (("w", "a"), True, (), ("down_proj",)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--json")
    args = ap.parse_args()
    spec = P.Spec.from_ckpt(args.ckpt)
    W = P.Weights(args.ckpt)
    r = np.load(args.ref)
    vis = [r["vis_feat0"], r["vis_feat1"]]
    common = dict(images=None, tokens=r["tokens"], mask=r["token_mask"], noise=r["noise"],
                  vis=vis)
    exact = P.reference_actions(spec, W, **common)
    sd, A = r["action_std"], spec.out_dim
    out = {}
    for name in args.variants.split(","):
        pp, io, ep, *only = VARIANTS[name]
        t = time.time()
        got = P.reference_actions(spec, W, nq=P.Emulated(pp, io=io, a_only=only[0] if only
                                                         else None),
                                  enq=P.Emulated(ep, io=io), **common)
        e = (got - exact)[:, :A]
        out[name] = {"max_abs": float(np.abs(e).max()), "rms": float(np.sqrt((e ** 2).mean())),
                     "rel": float(np.abs(e).max() / np.abs(exact[:, :A]).max()),
                     "unnorm_max_per_dim": (np.abs(e) * sd).max(0).round(5).tolist(),
                     "seconds": round(time.time() - t, 1)}
        print(name, out[name], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
