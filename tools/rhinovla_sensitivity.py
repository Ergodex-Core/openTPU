"""Which of the card's quantization points cost RhinoVLA's actions accuracy, and the precision
map (docs/rhinovla.md 4; the design's Phase 7 exit): the fp32 reference with
opentpu.llm.rhinovla.Emulated at some of its points, against the exact reference, on
tools/rhinovla_reference.py's observation; pi05_sensitivity's method with the groups of all
three parts and the promotion procedure of docs/designs/vla-1hz-packed-slice.md.

    python tools/rhinovla_sensitivity.py --ref ref.npz --ckpt rhinovla_pretrain.ckpt
        --assets DIR [--config yaml] [--variants all,w8a16,...] [--map] [--bound 0.02]
        [--json out.json]
    python tools/rhinovla_sensitivity.py --ref ref.npz --tiny [--map]       (a smoke run)

The error of a variant: max |a - a_ref| over the live slots of the 30 x 96 chunk divided by
the exact chunk's range (max - min over the live slots), the design's metric.

Variants: w8a8 (every weight int8 MX, every MM activation int8 as QACT makes it, the
attention's q / K / V / P int8: the card as it runs today), w8a16 (weights int8, activations
and attention exact), backbone_w8a16 (the text model W8A16, the tower and the expert W8A8:
the paper's precision), kv_only (the int8 attention operands alone), <part>_only (one part
W8A8, the rest exact), and the sweep: one matrix group at W8A8 with everything else at W8A16
(weights int8 everywhere; the group's activations int8), the group's attributed error.

--map: the precision map. Start at W8A8 everywhere; while the chunk is above --bound, promote
to W8A16 the group with the largest attributed error not yet promoted, the backbone's groups
before the tower's and the expert's (the design's order), re-measure; print the map, its MACs
at W8A16 (per part and in all) and the MXU seconds it implies at the f2-vla point (--mxu:
the tally's vision, prefix, 10-step expert MXU seconds; a W8A16 MM runs at one product a DSP,
half the packed rate: its seconds double).
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

from opentpu.llm import rhinovla as RV  # noqa: E402

# the matrix groups: name -> (part kind, the names' substrings (a tuple: all of them))
GROUPS = {
    "v_patch": ("v", ("wp",)),
    "v_qkv": ("v", ("blocks", "attn.qkv")),
    "v_proj": ("v", ("blocks", "attn.proj")),
    "v_fc1": ("v", ("blocks", "mlp.linear_fc1")),
    "v_fc2": ("v", ("blocks", "mlp.linear_fc2")),
    "v_merger": ("v", ("merger",)),
    "p_qkv": ("p", ("language_model", "q_proj"), ("language_model", "k_proj"),
              ("language_model", "v_proj")),
    "p_o": ("p", ("language_model", "o_proj")),
    "p_gateup": ("p", ("language_model", "gate_proj"), ("language_model", "up_proj")),
    "p_down": ("p", ("language_model", "down_proj")),
    "e_qkv": ("e", ("action_expert", "q_proj"), ("action_expert", "k_proj"),
              ("action_expert", "v_proj")),
    "e_o": ("e", ("action_expert", "o_proj")),
    "e_gateup": ("e", ("action_expert", "gate_proj"), ("action_expert", "up_proj")),
    "e_down": ("e", ("action_expert", "down_proj")),
}
KV_GROUPS = {"v_kv": "v", "p_kv": "p", "e_kv": "e"}
ORDER = ("p", "v", "e")                 # the promotion order among equal errors


def group_macs(spec: RV.Spec, L: int) -> dict:
    """MACs of each group a chunk (the tally's shapes; the attention's operands under the
    *_kv groups), to weigh the map."""
    N, C, F = spec.n_patch * spec.views, spec.v_hidden, spec.v_ffn
    M = spec.n_tok * spec.views
    H, d = spec.hidden, spec.head_dim
    g = {"v_patch": N * C * 3 * spec.patch ** 2, "v_qkv": N * 3 * C * C * spec.v_layers,
         "v_proj": N * C * C * spec.v_layers, "v_fc1": N * C * F * spec.v_layers,
         "v_fc2": N * C * F * spec.v_layers,
         "v_merger": M * (16 * C * C + 4 * C * H) * (1 + len(spec.deepstack)),
         "v_kv": 2 * N * spec.n_patch * C * spec.v_layers,
         "p_qkv": L * H * (spec.e_q + 2 * spec.e_kv) * spec.layers,
         "p_o": L * H * spec.e_q * (spec.layers - 1),
         "p_gateup": L * H * 2 * spec.ffn * (spec.layers - 1),
         "p_down": L * H * spec.ffn * (spec.layers - 1),
         "p_kv": 2 * L * L * spec.e_q * (spec.layers - 1) // 2}
    T, E, Fe, S = spec.rows, spec.e_hidden, spec.e_ffn, spec.steps
    g.update({"e_qkv": S * T * E * (spec.e_q + 2 * spec.e_kv) * spec.e_layers,
              "e_o": S * T * E * spec.e_q * spec.e_layers,
              "e_gateup": S * T * E * 2 * Fe * spec.e_layers,
              "e_down": S * T * E * Fe * spec.e_layers,
              "e_kv": S * 2 * T * (L + T) * spec.e_q * spec.e_layers})
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt")
    ap.add_argument("--assets")
    ap.add_argument("--config")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--bound", type=float, default=0.02)
    ap.add_argument("--mxu", default="0.163,0.218,0.059",
                    help="the f2-vla tally's MXU seconds: vision, prefix, the 10 expert steps")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.tiny:
        spec = RV.TINY
        W = RV.random_weights(spec, np.random.default_rng(args.seed))
    else:
        if not (args.ckpt and args.assets):
            raise SystemExit("--tiny, or --ckpt and --assets")
        spec = RV.Spec.from_config(args.assets, args.config)
        W = RV.Weights(args.ckpt)
    r = np.load(args.ref)
    imgs = [r[f"img{v}"] for v in range(spec.views)]
    lay = RV.layout(spec, r["ids"])
    common = dict(images=imgs, lay=lay, state=r["state"], noise=r["noise"],
                  state_mask=r["state_mask"], action_mask=r["action_mask"])
    live = np.asarray(r["action_mask"], bool)
    t0 = time.time()
    exact = RV.reference_actions(spec, W, **common)
    rng_ = float(exact[:, live].max() - exact[:, live].min())
    print(f"exact reference in {time.time() - t0:.0f} s; the live slots' range {rng_:.3f}")
    out = {"range": rng_}

    def measure(name, nq, enq=None):
        t = time.time()
        got = RV.reference_actions(spec, W, nq=nq, enq=enq, **common)
        e = (got - exact)[:, live]
        res = {"max_abs": float(np.abs(e).max()), "over_range": float(np.abs(e).max() / rng_),
               "rms": float(np.sqrt((e ** 2).mean())), "seconds": round(time.time() - t, 1)}
        out[name] = res
        print(f"{name:18s} max |err| {res['max_abs']:.4f}  {100 * res['over_range']:.2f}% of the "
              f"range  rms {res['rms']:.4f}  ({res['seconds']} s)", flush=True)
        return res["over_range"]

    def group_nq(groups, kv_groups=()):
        """W8 everywhere; A8 in `groups`' matrices and the attention operands of kv_groups."""
        subs = [s for g in groups for s in GROUPS[g][1:]]
        kinds = tuple(KV_GROUPS[k] for k in kv_groups)
        nq = RV.Emulated(("w", "a") + (("kv",) if kinds else ()), a_only=subs or [("<none>",)])
        if kinds:
            class _Kv(RV.Emulated):
                def attn(self, q, k, v, scale, mask=None, kind=""):
                    if kind not in kinds:
                        return RV._attn(q, k, v, scale, mask)
                    return super().attn(q, k, v, scale, mask, kind)
            nq = _Kv(("w", "a", "kv"), a_only=subs or [("<none>",)])
        return nq

    variants = {
        "w8a8": lambda: RV.Emulated(),
        "w8a16": lambda: RV.Emulated(("w",)),
        "backbone_w8a16": lambda: group_nq([g for g in GROUPS if g[0] != "p"],
                                           [k for k in KV_GROUPS if k[0] != "p"]),
        "kv_only": lambda: RV.Emulated(("kv",)),
        "vision_only": lambda: RV.Emulated(kinds=("v",)),
        "prefix_only": lambda: RV.Emulated(kinds=("p",)),
        "expert_only": lambda: RV.Emulated(kinds=("e",)),
    }
    for g in GROUPS:
        variants["a_" + g] = (lambda g=g: group_nq([g]))
    for k in KV_GROUPS:
        variants["a_" + k] = (lambda k=k: group_nq([], [k]))
    names = list(variants) if args.variants == "all" else args.variants.split(",")
    if args.map:
        names = list(dict.fromkeys(names + ["w8a8", "w8a16"] + ["a_" + g for g in GROUPS]
                                   + ["a_" + k for k in KV_GROUPS]))
    for name in names:
        measure(name, variants[name]())

    if args.map:
        macs = group_macs(spec, lay.L)
        total = sum(macs.values())
        allg = list(GROUPS) + list(KV_GROUPS)
        attributed = {g: out["a_" + g]["over_range"] for g in allg}
        promoted = []
        err = out["w8a8"]["over_range"]
        print(f"\nprecision map: W8A8 everywhere {100 * err:.2f}% of the range, the bound "
              f"{100 * args.bound:.1f}%")
        while err > args.bound:
            left = [g for g in allg if g not in promoted]
            if not left:
                break
            # the design's order: the largest attributed error first, the backbone's groups
            # first among errors within 10% of each other
            top = max(attributed[g] for g in left)
            near = [g for g in left if attributed[g] >= 0.9 * top]
            nxt = min(near, key=lambda g: (ORDER.index(g[0]), -attributed[g]))
            promoted.append(nxt)
            keep = [g for g in GROUPS if g not in promoted]
            keepkv = [k for k in KV_GROUPS if k not in promoted]
            err = measure("map_" + "+".join(promoted), group_nq(keep, keepkv))
        pm = sum(macs[g] for g in promoted)
        mxu = [float(x) for x in args.mxu.split(",")]
        parts = {}
        for p, sec in zip("vpe", mxu):
            pp = sum(macs[g] for g in macs if g[0] == p)
            pr = sum(macs[g] for g in promoted if g[0] == p)
            parts[p] = {"macs": pp, "promoted": pr, "mxu_s": sec, "mxu_s_map": sec * (1 + pr / pp)}
        out["map"] = {"promoted": promoted, "error_over_range": err,
                      "w8a16_mac_fraction": pm / total, "parts": parts,
                      "mxu_s_map": sum(v["mxu_s_map"] for v in parts.values())}
        print(f"W8A16: {promoted} ({100 * pm / total:.0f}% of the MACs); the chunk "
              f"{100 * err:.2f}% of the range; MXU at the f2-vla point "
              f"{sum(mxu):.3f} s -> {out['map']['mxu_s_map']:.3f} s "
              + " ".join(f"{p} {v['mxu_s']:.3f}->{v['mxu_s_map']:.3f}" for p, v in parts.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
