"""One decode token of a Ternary-Bonsai 2 model cut to its first N layers (the LM head whole) on
the RTL simulator: the board's memory path (otpu_native_dram on OTPU_NCH channels a slice, the
native memory model), on every slice of the target. Prints the cycles, each slice's port-B
chunks, the roofline and its efficiency, and a wide port's reads (docs/wide_port.md).

    OTPU_NCH=8 python tools/perf_bonsai2.py MODEL_DIR f2-hbm [--layers 4] [--wformat t2]
        [--rotate] [--portb 2] [--mcols 8] [--rowlanes]

MODEL_DIR holds the GGUF beside the Qwen3.5 config and tokenizer (docs/ternary.md 4). --wformat
t2 --rotate is the t2 decode with the rotation on the device (MM QUAD); --wformat int4 the
unfolded int4 weights (QUAD at --portb 2). --portb N (Config.PORTB, with PAIR) builds the RTL
with a port B of N chunks a cycle and compiles the step with the reuse it allows; the target's
channels must be at least 2 N (f2-hbm's 8). --mcols 8 --rowlanes is the row-lane mode
(docs/row_lanes.md): t2 RL 2 at PORTB 2 / MCOLS 8, two rows' chunks a cycle into eight
columns. (b2-scripts/perf_b2.py, with the port.)
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("target")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--wformat", default="t2")
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--portb", type=int, default=None, choices=[1, 2, 4],
                    help="chunks a cycle on port B (Config.PORTB; sets PAIR); default the target's")
    ap.add_argument("--mcols", type=int, default=None,
                    help="MXU columns (Config.MCOLS; default the target's)")
    ap.add_argument("--rowlanes", action="store_true",
                    help="row lanes (Config.ROWLANES, docs/row_lanes.md): the image in row-lane "
                         "groups, MM RL of the widest L for --portb / --mcols (t2 L 2 at PORTB 2 "
                         "/ MCOLS 8)")
    ap.add_argument("--lat", type=int, default=60, help="read latency, core cycles")
    ap.add_argument("--bw", type=int, default=100)
    ap.add_argument("--mhz", type=float, default=200.0, help="core clock (tokens/s)")
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ["OTPU_TARGET"] = a.target            # (rtlsim's board uarch: the target's)
    from opentpu import rtlsim, target
    from opentpu.llm import bonsai2
    from opentpu.llm.qwen3 import rope_tables
    from opentpu.profile import parse
    t = target.load(a.target)
    cfg0 = target.config(t)
    if a.portb:
        cfg0 = dataclasses.replace(cfg0, PORTB=a.portb, PAIR=True)
    if a.mcols:
        cfg0 = dataclasses.replace(cfg0, MCOLS=a.mcols)
    if a.rowlanes:
        cfg0 = dataclasses.replace(cfg0, ROWLANES=True)
    S = cfg0.S
    spec = bonsai2.spec(a.model, ternary=a.rotate, slices=S)
    spec = dataclasses.replace(spec, kinds=spec.kinds[:a.layers])
    W = bonsai2.Weights(a.model, latent=a.rotate, ffn=spec.ffn)
    hf = a.head_format or a.wformat
    cap = 256
    need = spec.image(dataclasses.replace(cfg0, DRAM_BYTES=1 << 40), cap, wformat=a.wformat,
                      head_format=hf).nbytes
    cfg = dataclasses.replace(cfg0, DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    img = spec.image(cfg, cap, wformat=a.wformat, head_format=hf)
    drams = img.build(W)
    emb = np.asarray(W["model.embed_tokens.weight"][791], np.float32)[None]
    c, s = rope_tables(spec, a.pos)
    for d in drams:
        for addr, v in ((img.io["x"], emb), (img.io["cos"], c[None]), (img.io["sin"], s[None])):
            b = np.ascontiguousarray(v, np.float32).view(np.uint8).ravel()
            d[addr:addr + b.size] = b
    progs = img.compile_step(a.pos)
    t0 = time.time()
    _, _, st = rtlsim.run(cfg, progs, drams, trace=True, uarch=rtlsim.BOARD_UARCH, axi=True,
                          boot=True, bw=a.bw, lat=a.lat, max_cycles=1 << 40)
    wall = time.time() - t0
    p = parse(st["trace"], cfg, progs, os.path.basename(a.model))
    p.cycles = st["cycles"]
    rl = p.roofline()
    portb = cfg.PORTB
    out = {"model": os.path.basename(a.model), "target": a.target, "S": S,
           "nch": os.environ.get("OTPU_NCH"), "layers": a.layers, "wformat": a.wformat,
           "rotate": a.rotate, "portb": portb, "mcols": cfg.MCOLS,
           "rowlanes": bool(getattr(cfg, "ROWLANES", False)), "cycles": p.cycles,
           "sim_s": round(wall),
           "slices": [{"portb": ps["portb"], "porta": ps["porta_cycles"], "mxu": ps["mxu"]}
                      for ps in rl["per_slice"]]}
    # port B streams its chunks at PORTB a cycle; port A and the MXU at their own rates
    ideal = max(max(ps["portb"] * 100 / a.bw / portb, ps["porta_cycles"], ps["mxu"])
                for ps in rl["per_slice"])
    out["roofline"], out["efficiency"] = round(ideal), round(ideal / p.cycles, 3)
    out["tok_s"] = round(a.mhz * 1e6 / p.cycles, 2)
    out["native_b"] = st.get("native_b", [])
    # the token's phases (DeltaNet / attention / MLP / LM head: perf_qwen's names, by the
    # kernel source of each instruction; Profile.phases' running-max rule), the per-unit busy
    # cycles: what tools/wideport_lpn.py calibrates its LPN estimate against, phase by phase
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from perf_qwen import _phase
    out["phases"] = p.phases(lambda r: _phase(progs[0][r.pc]))
    out["busy"] = p.unit_busy(0)
    out["lat"] = a.lat
    print(out, flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
