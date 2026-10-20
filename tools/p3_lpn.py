"""v4 P3 (docs/v4/proposal-3-eight-lanes-packed.md) as Latency Petri Nets: the VLA track's LPN
(tools/pi05_lpn.py on the vla branch, its lpnlang net: expand / cost / Net, docs/pi05.md 5.5 on
that branch, used as it is; P4's tools/p4_lpn.py method) with P3's MXU in its cost function:
C lock-step chunk lanes (otpu_mxu_lanes), each streaming every C-th row of an MM:

  * port B: the MM's chunks move C a cycle (lane 0 on port B, lanes 1 .. C-1 on their lane
    ports; the DMA's chunks stay one a cycle on lane 0's port), so its stream takes ceil(pb / C);
  * the consume: ceil(N / C) rows a lane, KB advances a row a group (the lanes in lock step);
  * the drain: a lane's rows at EPL = max(E / C, 1) epilogue lanes, ceil(M_g / EPL) steps a row
    group (the RTL's drain takes min(run, M) results a step: at M = 1 one step a row);
  * each lane's FIFO_DEPTH chunks = FIFO_DEPTH cycles of its stream ahead of the consumer;
  * the row pipeline: packed (MXU_PACK) at MCOLS 4 has the unpacked LROW (25), so L_MM is
    P4's m4 constant CL 16 + MCOLS 4 + 5 + LLAT 9 + 10 either way.

Two runs:

  gemma  the Gemma 3 1B int8 decode token: the resident decode step at the chat prompt's last
         position (14, "Why is the sky blue?"; the RTL receipts' token), its programs captured
         from this tree's ISA simulator, the net at P3's MXU and the m4 slice's constants (P4's:
         LANES 8, VPU_CL 2, the sums at 8 a cycle, the LD round trip --lat + 28).
  vla    pi0.5's action chunk (pi05_lpn's own flow, --full: vision, prefix, expert) on the vla
         tree's f2-vla target, once as it is (the track's baseline) and once with P3's MXU in
         place of f2-vla's (C lanes of MCOLS --vla-mcols, the weight rows unpacked: P3 packs two
         stationary rows a DSP, which saves DSPs and no cycles).

Labels: every number here is `lpn-estimate`; the RTL cycles printed beside it are `measured`.

    python tools/p3_lpn.py gemma --vla-tree DIR --lpn DIR/language/src [--lanes 8] [--lat 30] \\
        [--model DIR] [--rtl-cycles N] [--mhz 200] [--json out.json]
    python tools/p3_lpn.py vla --vla-tree DIR --lpn DIR/language/src --ckpt PI05_CONFIG_DIR \\
        [--lanes 8] [--vla-mcols 4] [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "lpn-estimate"


def load_net(vla_tree: Path, lpn_src: Path, own_opentpu: bool):
    """The vla branch's pi05_lpn module (its lpnlang net). own_opentpu: this tree's opentpu
    underneath (the Gemma programs); else the vla tree's (pi0.5's)."""
    import numpy as np
    sys.path.insert(0, str(ROOT if own_opentpu else vla_tree))
    sys.path.insert(1, str(vla_tree / "tools"))
    from opentpu import isa as I
    for name in ("F_T",):                   # the vla ISA's QST T flag: none of ours carries it
        if not hasattr(I, name):
            setattr(I, name, 0)
    if not hasattr(I, "mm_format"):         # the vla ISA's (format, PAIR, QUAD): ours has no
        def mm_format(ins):                 # t2 / QUAD (flags[5:4] the format, F_PAIR)
            return (ins.flags >> 4) & 3, bool(ins.flags & I.F_PAIR), False
        I.mm_format = mm_format
    if not hasattr(np, "Inf"):
        np.Inf = np.inf                     # lpnlang's simulator is NumPy 1.x's
    sys.path.insert(0, str(lpn_src))
    spec = importlib.util.spec_from_file_location("pi05_lpn", vla_tree / "tools" / "pi05_lpn.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pi05_lpn"] = m             # (its dataclasses look their module up)
    spec.loader.exec_module(m)
    return m


def lanes_cost(m, C: int, E: int):
    """pi05_lpn.cost with P3's MXU: C lock-step chunk lanes (the module docstring)."""
    base = m.cost
    I = m.I

    def cost(d, cfg, kn, E_=E):
        base(d, cfg, kn, E_)
        if d.ins.op != I.MM or C == 1:
            return
        w = d.ins.w
        N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[5] >> 16) & 0xFF
        G = -(-M // cfg.MCOLS)
        rows = -(-N // C)                                   # a lane's rows (lane 0's: the most)
        epl = max(E_ // C, 1)
        last = M - (G - 1) * cfg.MCOLS
        steps = (G - 1) * -(-cfg.MCOLS // epl) + -(-last // epl)
        d.pb = -(-d.pb // C)
        d.pops = rows * KB * G
        d.drain = rows * steps
        d.tail = G * KB
        d.cyc = max(d.pops, d.drain) + d.lat
    return cost


def capture(model: str, prompt: str, wformat: str):
    """The resident decode step after the chat prompt (the RTL receipts' token): the prompt but
    its last token prefilled on the ISA simulator, the last token's step run there with its
    programs and run arguments captured. (cfg, programs, args, pos, next token)."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "tools"))
    from p3_gemma3_token import chat_ids
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_dir(model)
    spec = load_spec(path)
    ids, _, _ = chat_ids(path, prompt)
    cap = max(256, -(-(len(ids) + 1) // 256) * 256)
    cfg = sim_config(spec, cap, board_config(PAIR=True, DSTEP=True, STREAM=True), lookup=True,
                     wformat=wformat)
    eng = Engine(spec, load_weights(path), cap=cap, cfg=cfg, wformat=wformat, resident=True)
    eng.prefill(ids[:-1])
    got = {}
    run0 = eng.backend.run

    def run(programs, args=None):
        got["programs"], got["args"] = programs, args
        return run0(programs, args)
    eng.backend.run = run
    logits = eng.step(ids[-1])
    return eng.cfg, got["programs"], got["args"], len(ids) - 1, int(np.argmax(logits))


def run_gemma(a) -> dict:
    t0 = time.time()
    m = load_net(a.vla_tree.resolve(), a.lpn.resolve(), own_opentpu=True)
    cfg, progs, args, pos, nxt = capture(a.model, a.prompt, a.wformat)
    print(f"captured the step at pos {pos} (next token {nxt}): {len(progs[0])} instructions, "
          f"{time.time() - t0:.0f} s", flush=True)
    E = 4                                               # the card's MXU_E (targets/ypcb-00338)
    kn = m.Knobs(l_mm=16 + cfg.MCOLS + 5 + 9 + 10, l_ld=a.lat + 28, l_st=a.lat + 20,
                 fifo=a.fifo // 2, win=16, vpu_cl=2, sum_w=min(16, cfg.LANES))
    m.cost = lanes_cost(m, a.lanes, E)
    rec = {"label": LABEL, "model": str(a.model), "pos": pos, "next_token": nxt, "lanes": a.lanes,
           "lat": a.lat, "mhz": a.mhz, "knobs": {k: getattr(kn, k) for k in
                                                ("l_mm", "l_ld", "l_st", "fifo", "win", "vpu_cl", "sum_w")}}
    from opentpu.isasim import ARG0
    regs = {ARG0 + k: int(v) for k, v in enumerate(args if args is not None else [])}
    t1 = time.time()
    dyn = m.expand(progs[0], cfg, kn, E, regs)          # the step's run arguments in R8..R15
    net = m.Net(dyn, kn)
    cyc = net.run()
    busy = {m.UNAMES[u]: int(c) for u, c in sorted(net.vbusy.items())}
    rec.update(cycles=int(cyc), instructions=len(dyn), busy=busy, ms=cyc / a.mhz / 1e3,
               tok_s=a.mhz * 1e6 / cyc, net_s=time.time() - t1)
    line = (f"token ({LABEL}, {a.lanes} lanes): {cyc:,} cycles = {cyc / a.mhz / 1e3:.2f} ms, "
            f"{a.mhz * 1e6 / cyc:.1f} tok/s at {a.mhz:g} MHz ({len(dyn)} unit instructions, "
            f"{time.time() - t1:.0f} s)")
    if a.rtl_cycles:
        err = (cyc - a.rtl_cycles) / a.rtl_cycles
        rec.update(rtl_cycles=a.rtl_cycles, error=err)
        line += f"; the RTL (measured) {a.rtl_cycles:,}: {100 * err:+.1f}%"
    print(line)
    return rec


def run_vla(a) -> dict:
    """pi05_lpn's main twice in subprocesses (each imports the vla tree's opentpu with its own
    target environment): f2-vla as it is, then with P3's MXU."""
    out = {"label": LABEL}
    base_env = dict(os.environ, OTPU_TARGET="f2-vla", OTPU_LANES="64", PYTHONPATH=str(a.vla_tree))
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "_vla_inner", "--vla-tree",
           str(a.vla_tree), "--lpn", str(a.lpn), "--ckpt", str(a.ckpt), "--mhz", str(a.mhz)]
    for name, env, extra in (
            ("f2-vla", dict(base_env, OTPU_MCOLS="32", OTPU_PACK="1"), ["--lanes", "1"]),
            (f"f2-vla with P3's MXU ({a.lanes} lanes x MCOLS {a.vla_mcols})",
             dict(base_env, OTPU_MCOLS=str(a.vla_mcols), OTPU_PACK="0"),
             ["--lanes", str(a.lanes), "--lmm", str(16 + a.vla_mcols + 5 + 9 + 10)])):
        print(f"== {name}", flush=True)
        js = Path(a.work) / f"vla-{len(out)}.json"
        r = subprocess.run(cmd + extra + ["--json", str(js)], env=env, cwd=str(a.vla_tree),
                           text=True, capture_output=True)
        print(r.stdout[-6000:], end="")
        if r.returncode:
            print(r.stderr[-4000:])
            out[name] = {"error": r.stderr[-2000:]}
            continue
        out[name] = json.loads(js.read_text())
    return out


def vla_inner(argv) -> int:
    """(the vla tree's opentpu on the path) pi05_lpn.main with P3's MXU cost at --lanes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla-tree", type=Path)
    ap.add_argument("--lpn", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--mhz", default="200")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--lmm")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    m = load_net(a.vla_tree.resolve(), a.lpn.resolve(), own_opentpu=False)
    from opentpu.isasim import board_config
    E = m.target_E(board_config())
    if a.lanes > 1:
        m.cost = lanes_cost(m, a.lanes, E)
    sys.argv = ["pi05_lpn.py", "--ckpt", a.ckpt, "--backend", "lpnlang", "--full", "--lpn", str(a.lpn),
                "--mhz", a.mhz, "--json", a.json] + (["--lmm", a.lmm] if a.lmm else [])
    m.main()
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_vla_inner":
        return vla_inner(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", choices=("gemma", "vla"))
    ap.add_argument("--vla-tree", type=Path, required=True, help="a checkout of the vla branch")
    ap.add_argument("--lpn", type=Path, required=True, help="dslab-epfl/lpn's language/src")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--lat", type=int, default=30, help="the memory's read latency (OTPU_LAT)")
    ap.add_argument("--fifo", type=int, default=1024, help="a lane's chunk FIFO (FIFO_DEPTH)")
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--rtl-cycles", type=int, default=0, help="the measured RTL cycles to grade by")
    ap.add_argument("--ckpt", help="pi0.5's checkpoint directory (its config.json)")
    ap.add_argument("--vla-mcols", type=int, default=4)
    ap.add_argument("--work", default="/tmp/p3_lpn")
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    rec = run_gemma(a) if a.run == "gemma" else run_vla(a)
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
