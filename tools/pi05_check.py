"""pi0.5 against lerobot's fp32 tensors (tools/pi05_reference.py's .npz): the numpy reference
(--reference) or the card on the ISA simulator (--isa), whole or cut (docs/pi05.md).

    python tools/pi05_check.py --ckpt DIR --ref ref.npz --reference
    python tools/pi05_check.py --ckpt DIR --ref ref.npz --isa [--vision N] [--prefix N]
        [--expert N] [--steps S] [--json out.json]

--reference prints the max relative error (max |a - b| / max |b|) of every stage of
opentpu.llm.pi05's fp32 reference against lerobot's: camera 0's tower stages, the projector
rows, the prefix embedding, each layer's K / V, each step's condition and velocity, the
actions.

--isa builds the card's image (opentpu.llm.pi05.Image: int8 weights) of the model cut to
--vision / --prefix / --expert layers (default all; the expert needs no more layers than the
prefix), runs one action chunk on the ISA simulator (Config: the target's, OTPU_TARGET; S=1)
through llm/pi05.run_chunk, and compares it with the fp32 numpy reference of the same cut
model (on the same inputs: the .npz's pixels, tokens, noise): every camera's projector rows
(cosine, relative error), every step's velocity and x_t, the final actions (max abs, max
relative, per action dim; the first 7 also unnormalized), and with no cut lerobot's own
actions too.
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


def run_isa(spec, W, r, cut: dict, log=print) -> dict:
    import dataclasses
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import IsaBackend
    cs = dataclasses.replace(spec, v_layers=cut.get("vision") or spec.v_layers,
                             layers=cut.get("prefix") or spec.layers,
                             e_layers=cut.get("expert") or spec.e_layers,
                             steps=cut.get("steps") or spec.steps)
    if cs.e_layers > cs.layers:
        raise SystemExit("the expert needs as many prefix layers as its own")
    res = {"cut": {"vision": cs.v_layers, "prefix": cs.layers, "expert": cs.e_layers,
                   "steps": cs.steps}}
    t0 = time.time()
    image = P.Image(cs, board_config())
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
    imgs = [r["img0"], r["img1"]]
    prompt = P.prompt_rows(cs, W, r["tokens"], r["token_mask"])
    Np = cs.cameras * cs.n_img + len(prompt)
    progs = image.programs(Np)
    t1 = time.time()
    P.run_chunk(image, be, progs, imgs, prompt, r["noise"], stop="vision")
    res["vision_s"] = time.time() - t1
    log(f"vision on the ISA simulator {res['vision_s']:.0f} s")
    cap = {}
    t2 = time.time()
    ref = P.reference_actions(cs, W, imgs, r["tokens"], r["token_mask"], r["noise"], capture=cap)
    res["reference_s"] = time.time() - t2
    n = cs.n_img
    vis = be.read(0, image.work["PX"], 4 * 2 * n * cs.hidden).view(np.float32).reshape(
        2 * n, cs.hidden)
    rv = np.concatenate([cap["vis0"], cap["vis1"]])
    cosv = (vis * rv).sum(-1) / np.linalg.norm(vis, axis=-1) / np.linalg.norm(rv, axis=-1)
    res["vision"] = {"cos_min": float(cosv.min()), "rel": rel(vis, rv)}
    steps = {}

    def after(name, args):
        if name != "e_step":
            return
        s = args["step"]
        v = P.read_xt(image, be, "v")
        steps[s] = {"v_rel": rel(v, cap[f"v{s}"]), "v_abs": float(np.abs(v - cap[f"v{s}"]).max()),
                    "t": time.time() - t3}
        log(f"step {s}: {steps[s]}")
    t3 = time.time()
    got = P.run_chunk(image, be, progs, imgs, prompt, r["noise"], after=after,
                      log=lambda nm, a: None)
    res["chunk_s"] = time.time() - t3
    res["steps"] = steps
    err = got - ref
    A = cs.out_dim
    res["actions"] = {"max_abs": float(np.abs(err).max()), "rel": rel(got, ref),
                      "max_abs_7": float(np.abs(err[:, :A]).max()),
                      "per_dim_max_abs": np.abs(err[:, :A]).max(0).tolist(),
                      "per_dim_rms": np.sqrt((err[:, :A] ** 2).mean(0)).tolist(),
                      "ref_per_dim_absmax": np.abs(ref[:, :A]).max(0).tolist()}
    sd = r["action_std"]
    res["actions"]["unnorm_per_dim_max_abs"] = (np.abs(err[:, :A]) * sd).max(0).tolist()
    if not any(cut.values()):
        e2 = got - r["actions_model"]
        res["vs_lerobot"] = {"max_abs": float(np.abs(e2).max()),
                             "rel": rel(got, r["actions_model"]),
                             "per_dim_max_abs": np.abs(e2[:, :A]).max(0).tolist(),
                             "unnorm_per_dim_max_abs": (np.abs(e2[:, :A]) * sd).max(0).tolist()}
    np.save(Path(args_out_dir()) / f"isa_actions_{cs.v_layers}_{cs.layers}_{cs.e_layers}.npy",
            got)
    return res


_OUT = ["."]


def args_out_dir() -> str:
    return _OUT[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--isa", action="store_true")
    ap.add_argument("--vision", type=int)
    ap.add_argument("--prefix", type=int)
    ap.add_argument("--expert", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.json:
        _OUT[0] = str(Path(args.json).parent)
    spec = P.Spec.from_ckpt(args.ckpt)
    W = P.Weights(args.ckpt)
    r = np.load(args.ref)
    res = {}
    if args.reference:
        res["reference"] = check_reference(spec, W, r)
    if args.isa:
        res["isa"] = run_isa(spec, W, r, {"vision": args.vision, "prefix": args.prefix,
                                          "expert": args.expert, "steps": args.steps})
    print(json.dumps(res, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
