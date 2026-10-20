"""P2's Gemma 3 1B decode token (one slice, MXU_C lock-step chunk lanes; docs/v4/p2-f2.md) as a
Latency Petri Net: the VLA track's LPN (tools/pi05_lpn.py on the vla branch, its lpnlang net:
expand / cost / Net, used as it is) driven by this tree's resident decode program with the m4
slice's constants, the method of P4's tools/p4_lpn.py (branch p4-f2, the P4 agent's), and the
chunk lanes added to the net's MM cost: lane l streams rows l, l + C, ... of every MM, so an
MM's port-B stream is ceil(N / C) x KB chunk-cycles (C chunks a cycle; opentpu.profile._describe
with C), its pops ceil(N / C) x KB x ceil(M / MCOLS) advances, its drain ceil(N / C) x
ceil(M / MCOLS) x ceil(MCOLS / (E / C)) steps (each lane drains its row on E / C epilogue lanes:
otpu_mxu_lanes). The DMA's chunks stay on lane 0's port B, one a cycle, ahead of the stream (the
lanes wait for lane 0 in lock step). --vla runs pi05_lpn's own chunk (the f2-vla config) with the
same lane patch at --mxu-c (1: the VLA track's number as it is).

Labels: every number here is `lpn-estimate`; the RTL cycles it is compared with are `measured`
(Verilator, tools/p2_gemma3_token.py: the memory model and latency named with them).

    python tools/p2_lpn.py --vla-tree DIR --lpn DIR/lpn/language/src [--mxu-c 4] [--lat 30]
        [--rtl-cycles N] [--mhz 150] [--json out]
    OTPU_TARGET=f2-vla OTPU_MCOLS=32 OTPU_LANES=64 OTPU_PACK=1 python tools/p2_lpn.py --vla \\
        --vla-tree DIR --lpn DIR/lpn/language/src --ckpt PI05_CONFIG_DIR [--mxu-c 4] [--json out]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABEL = "lpn-estimate"


def load_net(vla_tree: Path, lpn_src: Path, own_opentpu: bool = True):
    """The vla branch's pi05_lpn module (its lpnlang net). own_opentpu: this tree's opentpu
    underneath (the Gemma token's programs); else the vla tree's (its own chunk)."""
    sys.path.insert(0, str(ROOT if own_opentpu else vla_tree))
    from opentpu import isa as I
    for name in ("F_T",):                    # (the vla ISA's QST transposed rows: none here)
        if not hasattr(I, name):
            setattr(I, name, 0)
    if not hasattr(np, "Inf"):
        np.Inf = np.inf                      # lpnlang's simulator is NumPy 1.x's
    sys.path.insert(0, str(lpn_src))
    spec = importlib.util.spec_from_file_location("pi05_lpn", vla_tree / "tools" / "pi05_lpn.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pi05_lpn"] = m
    spec.loader.exec_module(m)
    return m


def add_lanes(m, C: int) -> None:
    """The net's MM cost with C lock-step chunk lanes (see the top); C 1 leaves it as it is."""
    if C == 1:
        return
    I = m.I
    cost0 = m.cost

    def cost(d, cfg, kn, E):
        cost0(d, cfg, kn, E)
        if d.ins.op != I.MM:
            return
        w = d.ins.w
        N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[5] >> 16) & 0xFF
        wf, pair, quad = I.mm_format(d.ins)
        pack = bool(getattr(cfg, "PACK", False)) and wf == I.W8 and not pair and not quad
        G = -(-M // cfg.MCOLS)
        rows = -(-N // 2) if pack else N
        kb = max(KB, 2) if pack else KB
        lane_rows = -(-rows // C)
        el = max(1, E // C)                  # a lane's epilogue lanes
        d.pb = -(-d.pb * lane_rows // max(1, rows))     # the stream's chunk-cycles
        d.pops = lane_rows * kb * G
        d.drain = lane_rows * G * (2 if pack else 1) * -(-cfg.MCOLS // el)
        d.cyc = max(d.pops, d.drain) + kn.l_mm
    m.cost = cost


def knobs(m, cfg, lat: int):
    """The m4 slice's constants (p4_lpn.py's: pi05_lpn.Knobs' fields, the fitted write-path
    ones kept): L_MM = CL 16 + MCOLS + 5 + 9 + 10, the LD round trip at --lat + 28."""
    return m.Knobs(l_mm=16 + cfg.MCOLS + 5 + 9 + 10, l_ld=lat + 28, l_st=lat + 20,
                   fifo=1024, win=16, vpu_cl=2, sum_w=min(16, cfg.LANES))


def capture(model: str, prompt: str, wformat: str):
    """The resident decode step after the chat prompt (the gate's and Lens's token, position 14
    for the default prompt): the prompt but its last token prefilled on the ISA simulator, the
    last token's step run there with its programs and run arguments captured."""
    import transformers
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_dir(model)
    spec = load_spec(path)
    cfg = sim_config(spec, 256, board_config(PAIR=True, DSTEP=True, STREAM=True, S=1),
                     lookup=True, wformat=wformat)
    eng = Engine(spec, load_weights(path), cap=256, cfg=cfg, wformat=wformat, resident=True)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    eng.prefill(ids[:-1])
    got = {}
    run0 = eng.backend.run

    def run(programs, args=None):
        got["programs"], got["args"] = programs, args
        return run0(programs, args)
    eng.backend.run = run
    logits = eng.step(ids[-1])
    return cfg, got["programs"], got["args"], len(ids) - 1, int(np.argmax(logits))


def gemma(a, m) -> dict:
    from opentpu.isasim import ARG0
    t0 = time.time()
    cfg, programs, args, pos, nxt = capture(a.model, a.prompt, a.wformat)
    print(f"captured the step at pos {pos} (next token {nxt}): {len(programs[0])} instructions, "
          f"{time.time() - t0:.0f} s")
    kn = knobs(m, cfg, a.lat)
    E = min(4, cfg.MCOLS)
    regs = {ARG0 + k: int(v) for k, v in enumerate(args or [])}
    dyn = m.expand(programs[0], cfg, kn, E, regs)
    net = m.Net(dyn, kn)
    t1 = time.time()
    cyc = int(net.run())
    busy = {m.UNAMES[u]: int(c) for u, c in sorted(net.vbusy.items())}
    out = {"label": LABEL, "mxu_c": a.mxu_c, "lat": a.lat, "pos": pos, "next_token": nxt,
           "cycles": cyc, "instructions": len(dyn), "busy": busy,
           "knobs": {k: getattr(kn, k) for k in vars(kn)},
           "cfg": {"MCOLS": cfg.MCOLS, "LANES": cfg.LANES, "E": E}}
    line = (f"token ({LABEL}, MXU_C {a.mxu_c}, --lat {a.lat}): {cyc:,} cycles = "
            f"{cyc / a.mhz / 1e3:.2f} ms, {a.mhz * 1e6 / cyc:.1f} tok/s at {a.mhz:g} MHz; busy {busy} "
            f"({time.time() - t1:.0f} s)")
    if a.rtl_cycles:
        out["rtl_cycles"], out["error"] = a.rtl_cycles, cyc / a.rtl_cycles - 1
        line += f"; the RTL (measured) {a.rtl_cycles:,}: {out['error']:+.1%}"
    print(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-tree", type=Path, required=True, help="a checkout of the vla branch")
    ap.add_argument("--lpn", type=Path, required=True, help="dslab-epfl/lpn's language/src")
    ap.add_argument("--mxu-c", type=int, default=4, help="the MXU's lock-step chunk lanes")
    ap.add_argument("--vla", action="store_true", help="pi05_lpn's own chunk (OTPU_TARGET=f2-vla ...)")
    ap.add_argument("--ckpt", help="--vla: the pi0.5 checkpoint (its config.json)")
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--lat", type=int, default=30, help="the memory's read latency (OTPU_LAT)")
    ap.add_argument("--rtl-cycles", type=int, default=None, help="the measured RTL token")
    ap.add_argument("--mhz", type=float, default=150.0)
    ap.add_argument("--json")
    a, rest = ap.parse_known_args()
    m = load_net(a.vla_tree.resolve(), a.lpn.resolve(), own_opentpu=not a.vla)
    add_lanes(m, a.mxu_c)
    if a.vla:
        # pi05_lpn's main with its own arguments (the full chunk, lpnlang), the lanes patched in
        sys.argv = [str(a.vla_tree / "tools" / "pi05_lpn.py"), "--ckpt", a.ckpt, "--lpn", str(a.lpn),
                    "--backend", "lpnlang", "--full"] + (["--json", a.json] if a.json else []) + rest
        print(f"pi05_lpn ({LABEL}) with MXU_C {a.mxu_c}: {' '.join(sys.argv[1:])}")
        m.main()
        return
    out = gemma(a, m)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
