"""pi0.5 against lerobot's fp32 tensors (tools/pi05_reference.py's .npz): the numpy reference
(--reference) or the card on the ISA simulator (--isa), whole or cut (docs/pi05.md).

    python tools/pi05_check.py --ckpt DIR --ref ref.npz --reference
    python tools/pi05_check.py --ckpt DIR --ref ref.npz --isa [--vision N] [--prefix N]
        [--expert N] [--steps S] [--json out.json]

--reference prints the max relative error (max |a - b| / max |b|) of every stage of
opentpu.llm.pi05's fp32 reference against lerobot's: camera 0's tower stages, the projector
rows, the prefix embedding, each layer's K / V, each step's condition and velocity, the
actions. --isa: see run_isa below.
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


def rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def check_reference(spec, W, r) -> dict:
    out = {}
    t = time.time()
    cap = {}
    v0 = P.reference_vision(spec, W, r["img0"], capture=cap)
    out["vision"] = {"embed": rel(cap["embed"], r["vis_embed0"]),
                     **{f"layer{i}": rel(cap[f"layer{i}"], r[f"vis_layer0_{i}"])
                        for i in (0, 13, 26)},
                     "post": rel(cap["post"], r["vis_post0"]), "proj": rel(v0, r["vis_feat0"])}
    pad = r["prefix_pad"]
    cap = {}
    acts = P.reference_actions(spec, W, [r["img0"], r["img1"]], r["tokens"], r["token_mask"],
                               r["noise"], capture=cap)
    out["prefix_rows"] = rel(cap["prefix"], r["prefix_embs"][pad])
    out["kv"] = {i: (rel(cap["kv"][i][0], r[f"prefix_k{i}"][pad]),
                     rel(cap["kv"][i][1], r[f"prefix_v{i}"][pad])) for i in range(spec.layers)}
    out["steps"] = {s: (rel(cap[f"cond{s}"], r[f"adarms_cond{s}"]), rel(cap[f"v{s}"], r[f"v{s}"]))
                    for s in range(spec.steps)}
    out["actions_rel"] = rel(acts, r["actions_model"])
    out["actions_abs"] = float(np.abs(acts - r["actions_model"]).max())
    out["seconds"] = time.time() - t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    spec = P.Spec.from_ckpt(args.ckpt)
    W = P.Weights(args.ckpt)
    r = np.load(args.ref)
    res = {}
    if args.reference:
        res["reference"] = check_reference(spec, W, r)
    print(json.dumps(res, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
