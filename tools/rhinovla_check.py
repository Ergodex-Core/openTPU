"""RhinoVLA against the repo's fp32 tensors (tools/rhinovla_reference.py's .npz): the numpy
reference (--reference) or the card on the ISA simulator (--isa), whole or cut
(docs/rhinovla.md).

    python tools/rhinovla_check.py --ref ref.npz --tiny --reference
    python tools/rhinovla_check.py --ref ref.npz --tiny --isa [--emulated]
    python tools/rhinovla_check.py --ref ref.npz --ckpt rhinovla_pretrain.ckpt --assets DIR
        [--config yaml] --reference --isa [--vision N] [--prefix N] [--expert N] [--steps S]
        [--json out.json]

--reference prints the max relative error (max |a - b| / max |b|) of every stage of
opentpu.llm.rhinovla's fp32 reference against the repo's: the tower's embedding and blocks,
the deepstack and merged rows, every text layer's K / V and rows, every step's suffix
embedding, condition and velocity, the actions.

--isa builds the card's image (opentpu.llm.rhinovla.Image: int8 weights) of the model cut to
--vision / --prefix / --expert layers (default all; a cut prefix keeps offset + expert
layers), runs one action chunk on the ISA simulator (Config: the target's, OTPU_TARGET; S=1)
through llm/rhinovla.run_chunk, and compares it with the fp32 numpy reference of the same cut
model on the same inputs: every view's merged rows (cosine, relative error), every step's
velocity, the actions (max abs, relative to the chunk's range, per slot), and with no cut the
repo's own actions too; --emulated also measures the card against the quantization
emulation (Emulated: the card's int8 points in float64).
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


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def load(args):
    r = np.load(args.ref)
    if args.tiny:
        spec = RV.TINY
        W = RV.random_weights(spec, np.random.default_rng(args.seed))
    else:
        spec = RV.Spec.from_config(args.assets, args.config)
        W = RV.Weights(args.ckpt)
    return spec, W, r


def inputs(spec, r):
    imgs = [r[f"img{v}"] for v in range(spec.views)]
    lay = RV.layout(spec, r["ids"])
    return imgs, lay


def check_reference(spec, W, r) -> dict:
    out = {}
    t = time.time()
    imgs, lay = inputs(spec, r)
    N, M = spec.n_patch, spec.n_tok
    if not np.array_equal(lay.pos, r["prefix_pos"]):
        out["prefix_pos"] = "MISMATCH"
    out["rope_delta"] = int(lay.suffix_pos - lay.L) == int(r["rope_delta"])
    tb = (RV.pos_table(spec, W), *RV.vit_rope_table(spec))
    vis = []
    for v, im in enumerate(imgs):
        cap = {}
        vis.append(RV.reference_vision(spec, W, im, capture=cap, tables=tb))
        if v == 0:
            out["vision"] = {"embed": rel(cap["embed"], r["vis_embed"][:N]),
                             **{f"layer{i}": rel(cap[f"layer{i}"], r[f"vis_layer{i}"][:N])
                                for i in sorted({0, spec.v_layers // 2, spec.v_layers - 1})}}
    out["vision"]["merged"] = rel(np.concatenate([a for a, _ in vis]), r["vis_merged"])
    for k in range(len(spec.deepstack)):
        out["vision"][f"deep{k}"] = rel(np.concatenate([d[k] for _, d in vis]), r[f"vis_deep{k}"])
    cap = {}
    acts = RV.reference_actions(spec, W, imgs, lay, r["state"], r["noise"], r["state_mask"],
                                r["action_mask"], capture=cap, vis=vis)
    out["kv"] = {i: (rel(cap["kv"][i][0], r[f"prefix_k{i}"]), rel(cap["kv"][i][1], r[f"prefix_v{i}"]))
                 for i in range(spec.layers)}
    kvmax = max(max(a, b) for a, b in out["kv"].values())
    out["kv_max"] = kvmax
    out["steps"] = {s: {"cond": rel(RV.time_embedding(spec, float(r["times"][s])), r[f"cond{s}"]),
                        "x": rel(cap[f"x{s}"], r[f"x{s}"]), "v": rel(cap[f"v{s}"], r[f"v{s}"])}
                    for s in range(spec.steps)}
    out["times"] = [float(x) for x in RV.times(spec)] == [float(x) for x in r["times"]]
    out["actions_rel"] = rel(acts, r["actions"])
    out["actions_abs"] = float(np.abs(acts - r["actions"]).max())
    out["seconds"] = time.time() - t
    return out


def run_isa(spec, W, r, cut: dict, emulated: bool = False, log=print) -> dict:
    import dataclasses
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import IsaBackend
    ne = cut.get("expert") or spec.e_layers
    npre = cut.get("prefix") or spec.layers
    off = min(spec.offset, max(npre - ne, 0))
    if off + ne > npre:
        raise SystemExit(f"the expert's {ne} layers need offset {off} + {ne} <= {npre} prefix layers")
    cs = dataclasses.replace(spec, v_layers=cut.get("vision") or spec.v_layers, layers=npre,
                             e_layers=ne, offset=off, steps=cut.get("steps") or spec.steps,
                             deepstack=tuple(d for d in spec.deepstack
                                             if d < (cut.get("vision") or spec.v_layers)))
    res = {"cut": {"vision": cs.v_layers, "prefix": cs.layers, "expert": cs.e_layers,
                   "offset": cs.offset, "steps": cs.steps}}
    imgs, lay = inputs(cs, r)
    t0 = time.time()
    image = RV.Image(cs, board_config(), np_max=max(256, lay.L))
    cfg = board_config(DRAM_BYTES=image.end)
    image.cfg = cfg
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)

    def put(addr, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    image.build(W, put, log=lambda m: None)
    res["build_s"] = time.time() - t0
    res["image_bytes"] = int(image.end)
    log(f"image {image.end / 2**30:.2f} GiB built in {res['build_s']:.0f} s")
    be = IsaBackend(cfg, [img], adopt=True)
    del img
    host = RV.Host(cs, W, lay, r["state"], r["state_mask"], r["action_mask"])
    progs = image.programs(lay)
    res["program_instructions"] = {k: len(p) for k, (p, _) in progs.items()}
    t1 = time.time()
    RV.run_chunk(image, be, progs, imgs, host, r["noise"], stop="vision")
    res["vision_s"] = time.time() - t1
    log(f"vision on the ISA simulator {res['vision_s']:.0f} s")
    cap = {}
    t2 = time.time()
    ref = RV.reference_actions(cs, W, imgs, lay, r["state"], r["noise"], r["state_mask"],
                               r["action_mask"], capture=cap)
    res["reference_s"] = time.time() - t2
    got_vis = RV.read_prefix_rows(image, be, lay)
    rv = np.concatenate([cap[f"vis{v}"] for v in range(cs.views)])
    cosv = (got_vis * rv).sum(-1) / np.linalg.norm(got_vis, axis=-1) / np.linalg.norm(rv, axis=-1)
    res["vision"] = {"cos_min": float(cosv.min()), "rel": rel(got_vis, rv)}
    log(f"vision rows: cosine >= {cosv.min():.5f}, rel {res['vision']['rel']:.4f}")
    steps = {}

    def after(name, args):
        if name != "e_step":
            return
        s = args["step"]
        v = RV.read_xt(image, be, "v")
        steps[s] = {"v_rel": rel(v, cap[f"v{s}"]), "v_abs": float(np.abs(v - cap[f"v{s}"]).max()),
                    "t": time.time() - t3}
        log(f"step {s}: {steps[s]}")
    t3 = time.time()
    got = RV.run_chunk(image, be, progs, imgs, host, r["noise"], after=after)
    res["chunk_s"] = time.time() - t3
    res["steps"] = steps
    live = np.asarray(r["action_mask"], bool)
    err = (got - ref)[:, live]
    rng_ = float(ref[:, live].max() - ref[:, live].min())
    res["actions"] = {"max_abs": float(np.abs(err).max()), "rel": rel(got[:, live], ref[:, live]),
                      "range": rng_, "max_abs_over_range": float(np.abs(err).max() / rng_),
                      "per_dim_max_abs": np.abs(err).max(0).tolist(),
                      "per_dim_rms": np.sqrt((err ** 2).mean(0)).tolist()}
    if not any(v for k, v in cut.items() if k != "offset"):
        e2 = (got - r["actions"])[:, live]
        res["vs_repo"] = {"max_abs": float(np.abs(e2).max()),
                          "max_abs_over_range": float(np.abs(e2).max() / rng_)}
    if emulated:
        em = RV.reference_actions(cs, W, imgs, lay, r["state"], r["noise"], r["state_mask"],
                                  r["action_mask"], nq=RV.Emulated())
        e3 = (got - em)[:, live]
        e4 = (em - ref)[:, live]
        res["vs_emulated"] = {"max_abs": float(np.abs(e3).max()),
                              "max_abs_over_range": float(np.abs(e3).max() / rng_),
                              "emulated_vs_exact_over_range": float(np.abs(e4).max() / rng_)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt")
    ap.add_argument("--assets", help="the Qwen3-VL config directory (the repo's assets)")
    ap.add_argument("--config", help="the training yaml")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--isa", action="store_true")
    ap.add_argument("--emulated", action="store_true")
    ap.add_argument("--vision", type=int)
    ap.add_argument("--prefix", type=int)
    ap.add_argument("--expert", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--json")
    args = ap.parse_args()
    if not args.tiny and not (args.ckpt and args.assets):
        raise SystemExit("--tiny, or --ckpt and --assets")
    spec, W, r = load(args)
    res = {}
    if args.reference:
        res["reference"] = check_reference(spec, W, r)
    if args.isa:
        res["isa"] = run_isa(spec, W, r, {"vision": args.vision, "prefix": args.prefix,
                                          "expert": args.expert, "steps": args.steps},
                             args.emulated)
    print(json.dumps(res, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
