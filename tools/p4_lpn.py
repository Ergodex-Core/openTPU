"""P4's Gemma 3 1B decode token, slice by slice, as a Latency Petri Net: the VLA track's LPN
(tools/pi05_lpn.py on the vla branch, its lpnlang net: expand / cost / Net, the model of
docs/pi05.md 5.5 on that branch, used as it is) driven by P4's per-slice programs (S = 4,
every matrix split by output rows, the inputs all-gathered: docs/gemma4.md "Slices") with the
m4 slice's constants instead of f2-vla's. Labels: every number here is `lpn-estimate`; the RTL
cycles it is compared with are `measured` (the S = 4 gate or the Lens RTL run, Verilator).

    python tools/p4_lpn.py --vla-tree DIR --lpn DIR/lpn/language/src [--slices 4]
        [--prompt "Why is the sky blue?"] [--lat 30] [--rtl-cycles N] [--mhz 175] [--json out]
        [--coll-beat A --coll-fixed B]

What the net is (pi05_lpn's header has the whole description): the sequencer's one instruction
a cycle into a 16-slot window with the scoreboard's RAW / WAR / WAW dependencies (opentpu.waits),
the DMA, the quantizer and the collective unit each one instruction at a time, the VPU's issue
and completion pipelines, the MXU with two commands held (the next stream while the current
drains), port B one chunk a cycle with the DMA first, the result FIFO, the drain at E lanes a
step, the drain-first TMEM write arbitration. The m4 slice's constants: MCOLS 4, MXU_E 4,
LANES 8, VPU_CL 2 and the sums at 8 a cycle, unpacked MMs, L_MM = CL 16 + MCOLS 4 + 5 + 9 + 10,
the LD round trip at --lat (30: the HBM-like reference, the gate's OTPU_LAT) + 28. The
collective: a GATHER is the unit's nominal S x rows x ceil(cols / 8) cycles (profile._describe)
on one slice; the net has one slice, so the rendezvous skew and the hub's crossing latency are
not in it (the RTL's "waiting on collectives" line is what they cost). --coll-beat A /
--coll-fixed B put the board's hub in: a GATHER of n beats costs A n + B, a BAR B (the
collective through otpu_coll_hub: A ~9 with one read in flight, ~1 pipelined; B its crossings,
rendezvous and ack; both fitted to the RTL's GATHERs, the hub in the simulation top, otpu_top
COLL_HUB: a model of the hub, not of the slices' skew, which the net still cannot have)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402  (this tree's opentpu first: the net reads its ISA)
from opentpu.isasim import ARG0  # noqa: E402

LABEL = "lpn-estimate"


def load_net(vla_tree: Path, lpn_src: Path):
    """The vla branch's pi05_lpn module (its lpnlang net), this tree's opentpu underneath. The
    vla branch's ISA has QST's transposed rows (F_T), which this tree's has not: no instruction
    here carries the flag, so it reads as 0 (every QST one pass)."""
    from opentpu import isa as I
    for name in ("F_T",):
        if not hasattr(I, name):
            setattr(I, name, 0)
    if not hasattr(np, "Inf"):
        np.Inf = np.inf                     # lpnlang's simulator is NumPy 1.x's (pi05_lpn's alias)
    sys.path.insert(0, str(lpn_src))
    spec = importlib.util.spec_from_file_location("pi05_lpn", vla_tree / "tools" / "pi05_lpn.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pi05_lpn"] = m             # (its dataclasses look their module up)
    spec.loader.exec_module(m)
    return m


def knobs(m, cfg, lat: int):
    """The m4 slice's constants (pi05_lpn.Knobs' fields; the fitted write-path ones kept)."""
    return m.Knobs(l_mm=16 + cfg.MCOLS + 5 + 9 + 10, l_ld=lat + 28, l_st=lat + 20,
                   fifo=1024, win=16, vpu_cl=2, sum_w=min(16, cfg.LANES))


def capture(model: str, prompt: str, slices: int, wformat: str):
    """The resident decode step after the chat prompt (the gate's and Lens's token): the
    prompt but its last token prefilled on the ISA simulator, the last token's step run there
    with its programs and run arguments captured. (cfg, programs, args, pos, next token)."""
    import transformers
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_dir(model)
    spec = load_spec(path)
    cfg = sim_config(spec, 256, board_config(PAIR=True, DSTEP=True, STREAM=True, S=slices),
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-tree", type=Path, required=True, help="a checkout of the vla branch")
    ap.add_argument("--lpn", type=Path, required=True, help="dslab-epfl/lpn's language/src")
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--lat", type=int, default=30, help="the memory's read latency (OTPU_LAT)")
    ap.add_argument("--rtl-cycles", type=int, default=None, help="the measured RTL token")
    ap.add_argument("--mhz", type=float, default=175.0)
    ap.add_argument("--coll-beat", type=float, default=1.0,
                    help="cycles a GATHER beat costs (1: the unit's nominal rate)")
    ap.add_argument("--coll-fixed", type=float, default=0.0,
                    help="cycles a collective's crossings, rendezvous and ack add")
    ap.add_argument("--json")
    a = ap.parse_args()
    m = load_net(a.vla_tree.resolve(), a.lpn.resolve())
    t0 = time.time()
    cfg, programs, args, pos, nxt = capture(a.model, a.prompt, a.slices, a.wformat)
    print(f"captured the step at pos {pos} (next token {nxt}): S = {cfg.S}, "
          f"{[len(p) for p in programs]} instructions a program, {time.time() - t0:.0f} s")
    kn = knobs(m, cfg, a.lat)
    E = min(4, cfg.MCOLS)
    regs = {ARG0 + k: int(v) for k, v in enumerate(args or [])}
    out = {"label": LABEL, "slices": [], "knobs": {k: getattr(kn, k) for k in vars(kn)},
           "cfg": {"S": cfg.S, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES, "E": E}, "pos": pos,
           "next_token": nxt}
    out["coll"] = {"beat": a.coll_beat, "fixed": a.coll_fixed}
    for s, prog in enumerate(programs):
        dyn = m.expand(prog, cfg, kn, E, regs)      # the step's run arguments in R8..R15
        for d in dyn:                               # the board's hub (--coll-beat / --coll-fixed)
            if d.unit == m.U_COLL and (a.coll_beat != 1 or a.coll_fixed):
                beat = a.coll_beat if d.ins.op == I.GATHER else 1.0
                d.cyc = int(round(beat * d.cyc + a.coll_fixed))
        net = m.Net(dyn, kn)
        t1 = time.time()
        cyc, n, wall = net.run(), len(dyn), time.time() - t1
        busy = {m.UNAMES[u]: int(c) for u, c in sorted(net.vbusy.items())}
        print(f"slice {s}: {LABEL} {cyc:,} cycles, {n} unit instructions, busy {busy} ({wall:.0f} s)")
        out["slices"].append({"cycles": int(cyc), "instructions": n, "busy": busy})
    tok = max(sl["cycles"] for sl in out["slices"])
    out["token_cycles"] = tok
    line = f"token ({LABEL}, the slowest slice; a GATHER beat {a.coll_beat:g} cycles, a collective " \
           f"+{a.coll_fixed:g}): {tok:,} cycles = {tok / a.mhz / 1e3:.2f} ms, " \
           f"{a.mhz * 1e6 / tok:.1f} tok/s at {a.mhz:g} MHz"
    if a.rtl_cycles:
        err = tok / a.rtl_cycles - 1
        out["rtl_cycles"], out["error"] = a.rtl_cycles, err
        line += f"; the RTL (measured) {a.rtl_cycles:,}: {err:+.1%}"
    print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
