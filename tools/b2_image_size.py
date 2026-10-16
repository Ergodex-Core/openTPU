"""Size a Ternary-Bonsai 2 image per slice, tensor class by tensor class, without building it
(docs/ternary.md 7): the ternary matrices in the weight format (data and scale rows), the LM
head, the norms and small vectors, the DeltaNet windows and states, the KV cache for --cap
tokens, the rotation's H and sign vectors, the device-side tables (embedding slot or table,
RoPE, masks, the generate loop's area) and the MTP drafter; against the slice's DRAM.

    python tools/b2_image_size.py MODEL_DIR [--target f2-hbm] [--wformat t2] [--head-format t2]
                                  [--cap 2048] [--embed-host] [--mtp] [--json OUT]

The classes are read off the image's own layout (Image.lofs / io / lookup offsets, each
allocation's size the gap to the next), so they add up to Image.nbytes exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu import target  # noqa: E402
from opentpu.llm import bonsai2  # noqa: E402
from opentpu.llm.qwen35 import ATTN, LIN  # noqa: E402

NORMS = {"g_in", "g_post", "alog", "dtb", "gn", "qn", "kn", "eb", "gf", "hs"}


def _flat(d: dict, prefix: str = "") -> list:
    """(offset, name) of every allocation in a layout dict: tuples are (data, scale) pairs,
    lists repeat, dicts (the KV heads) nest."""
    out = []
    for k, v in d.items():
        n = prefix + k
        if isinstance(v, tuple):
            out.append((v[0], n + ".data"))
            if len(v) > 1 and isinstance(v[1], int):
                out.append((v[1], n + ".scale"))
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if isinstance(e, dict):
                    out += _flat(e, f"{n}{i}.")
                elif isinstance(e, tuple):
                    out += [(e[0], f"{n}{i}.data"), (e[1], f"{n}{i}.scale")]
                else:
                    out.append((e, f"{n}{i}"))
        elif isinstance(v, int):
            out.append((v, n))
    return out


def _sizes(allocs: list, end: int) -> dict:
    """name -> bytes, each allocation up to the next (sorted by offset), the last to `end`."""
    a = sorted(allocs)
    return {n: (a[i + 1][0] if i + 1 < len(a) else end) - o for i, (o, n) in enumerate(a)}


def _cls(name: str) -> str:
    base = name.split(".")[0].rstrip("0123456789")
    if base in NORMS:
        return "norms"
    if base == "kv":
        return "kv cache"
    if base in ("cv",):
        return "deltanet windows"
    if base == "state":
        return "deltanet state"
    if base == "groups":
        return "groups"
    if name.endswith(".scale"):
        return "weights: scales"
    return "weights: data"


def layer_block(img, kind, g) -> dict:
    """Bytes per class of one layer block of kind `kind` in formats group g."""
    lay = img.layouts[g]
    sz = _sizes(_flat(lay.lofs[kind]), lay.size[kind])
    out = {}
    for n, b in sz.items():
        out[_cls(n)] = out.get(_cls(n), 0) + b
    if "groups" in out:         # group-major DeltaNet: a group's pairs (wh, cv, state), wout, eb
        del out["groups"]
        ngroups = img.nl // img.og
        ps = _sizes(_flat(lay.pofs), lay.PS)
        for n, b in ps.items():
            c = _cls(n)
            out[c] = out.get(c, 0) + b * (img.og // 2) * ngroups
        gs = _sizes(_flat(lay.gofs) + [(0, "pairs")], lay.GS)
        for n, b in gs.items():
            if n == "pairs":
                continue
            c = _cls(n)
            out[c] = out.get(c, 0) + b * ngroups
    pad = lay.size[kind] - sum(out.values())
    out["padding"] = pad
    return out


def size(model_dir, tgt: str = "f2-hbm", wformat: str = "t2", head_format: str | None = None,
         cap: int = 2048, embed_host: bool = True, mtp: bool = False) -> dict:
    cfg = target.config(target.load(tgt))
    big = replace(cfg, DRAM_BYTES=1 << 40)
    spec = bonsai2.spec(model_dir, ternary=True, slices=cfg.S)
    if mtp:
        spec = replace(spec, mtp=True)
    hf = head_format or wformat
    img = spec.image(big, cap, wformat=wformat, head_format=hf, lookup=True, rows=1,
                     embed_host=embed_host)
    tot = {}

    def add(c, b):
        tot[c] = tot.get(c, 0) + int(b)
    # I/O rows
    io = _sizes(_flat(img.io), img.layer0)
    for n, b in io.items():
        add("i/o rows (x, rope, logits, hid)", b)
    # layers
    per_kind = {}
    for li, (kind, g) in enumerate(img.keys):
        blk = layer_block(img, kind, g)
        name = ("mtp layer" if li == spec.layers else "attention" if kind == ATTN else "deltanet")
        per_kind.setdefault(name, {"layers": 0})
        per_kind[name]["layers"] += 1
        for c, b in blk.items():
            per_kind[name][c] = per_kind[name].get(c, 0) + b
            add(f"{c}" if c != "padding" else "padding (4 KiB blocks)", b)
    # the head
    from opentpu import quant as Q
    D, H = cfg.D, spec.hidden
    add("lm head: data", img.v_loc * Q.row_bytes(H, img.head_format, D))
    add("lm head: scales", img.v_loc * Q.scale_bytes(H, img.head_format, D))
    # the rotation
    if img.rot is not None:
        r = img.rot
        add("rotation: H (t2)", r.base * Q.row_bytes(r.base, r.fmt, D))
        add("rotation: signs", sum(4 * k for k in r.signs))
        add("rotation: onrows", 4 * img.dnrows * img.nl * spec.lin_dv)
    # the MTP drafter
    if spec.mtp:
        mo = img.mtpo
        add("mtp: norms", 3 * 4 * H)
        add("mtp: fc", H * (Q.row_bytes(2 * H, img.mtp_fc, D) + Q.scale_bytes(2 * H, img.mtp_fc, D)))
        add("mtp: draft head (fp4, 32K ids)",
            img.nd * (Q.row_bytes(H, "fp4", D) + Q.scale_bytes(H, "fp4", D)))
    # the lookup tables
    lk = img.lookup
    if "embed_q" in lk:
        if lk.get("host"):
            add("tables: embedding slot (host table)", lk["rows"] * lk["rec"] + 4 * 16)
        elif lk.get("own"):
            add("tables: int8 embedding table", spec.vocab * H
                + spec.vocab * Q.gather_scale_bytes(H, "int8", D))
        else:
            add("tables: head gather scales", spec.vocab * Q.gather_scale_bytes(H, "int8", D))
    if "embed" in lk:
        add("tables: fp32 embedding table", 4 * spec.vocab * H)
    add("tables: rope cos / sin", 2 * 4 * cap * lk["half"])
    add("tables: zmask", 4 * (cap + lk["block"]))
    acc = sum(tot.values())
    add("tables: one-hot, masks, generate loop, mtp loop (the rest)", img.nbytes - acc)
    return {"model": Path(model_dir).name, "target": tgt, "S": cfg.S, "wformat": wformat,
            "head_format": img.head_format, "cap": cap, "embed_host": bool(img.embed_host),
            "mtp": bool(spec.mtp), "rot": spec.rot, "nbytes": int(img.nbytes),
            "dram_bytes": int(cfg.DRAM_BYTES), "classes": tot, "per_kind": per_kind,
            "fits": img.nbytes <= cfg.DRAM_BYTES,
            "headroom": int(cfg.DRAM_BYTES - img.nbytes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--target", default="f2-hbm")
    ap.add_argument("--wformat", default="t2")
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--embed-table", action="store_true",
                    help="the int8 embedding table in the image (default: a host-fed slot)")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = size(a.model, a.target, a.wformat, a.head_format, a.cap, not a.embed_table, a.mtp)
    MiB = 2 ** 20
    print(f"{r['model']} on {r['target']} (S {r['S']}): {r['wformat']} weights, "
          f"{r['head_format']} head, cap {r['cap']}, rotation block {r['rot']}, "
          f"embed_host {r['embed_host']}, mtp {r['mtp']}")
    for c, b in sorted(r["classes"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:48s} {b / MiB:10.1f} MiB")
    print(f"  {'total (Image.nbytes)':48s} {r['nbytes'] / MiB:10.1f} MiB  = "
          f"{r['nbytes'] / 2**30:.3f} GiB of {r['dram_bytes'] / 2**30:.0f} GiB a slice: "
          f"{'fits' if r['fits'] else 'DOES NOT FIT'}, headroom {r['headroom'] / MiB:.0f} MiB")
    for k, v in r["per_kind"].items():
        n = v["layers"]
        print(f"  per {k} layer (x{n}): " + ", ".join(
            f"{c} {b / n / MiB:.2f} MiB" for c, b in v.items() if c != "layers"))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1) + "\n")


if __name__ == "__main__":
    main()
