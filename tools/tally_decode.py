"""The cost of one decode token on the ISA (opentpu.profile's nominal unit cycles and port
transfers per instruction), compiled for a Config with a wide port B: the weight bytes a
token, the chunks, the MXU's advances, the port-bound and MXU-bound cycles, what an 8-way
column reuse (OCT: not in the ISA, docs/wide_port.md "PORTB 8") would take, and the row-lane
mode (MM RL, docs/row_lanes.md: each of L = PORTB lanes streams its own weight row). No weights
are loaded: the step is compiled for the model's shapes and run on the ISA simulator over a
zeroed image (the loops over the layer kinds unroll as they run).

    OTPU_TARGET=f2-hbm python tools/tally_decode.py --model gemma3 --wformat int8 \\
        --head-format int8 --portb 1 2 4 [--mcols 4 8] [--layers 0] [--mhz 200] [--rowlanes]

Without --rowlanes the row-lane columns are computed from the standard program's MMs (every
scaled, non-RSC MM of one row would be an RL MM of the configuration's widest L, quant.rl_lanes);
with it the step is compiled for Config.ROWLANES (the image in row-lane groups, the compiler's
RL MMs) and tallied as it is.
"""
import argparse
import dataclasses
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentpu import isa as I  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.profile import _describe  # noqa: E402

FMT = ["int8", "int4", "fp4", "nvfp4", "t2"]


def tally(spec, cfg0, pos: int, mhz: float, wf: str, hf: str) -> dict:
    """The step compiled for `cfg0` (its PORTB, MCOLS, ROWLANES), then run on the ISA simulator
    over a zeroed image of the model's size (the step's loops over the layer kinds unroll as
    they execute; the weights' values do not change the tally), every executed instruction
    costed. `rl_*`: the same step with every eligible MM as a row-lane MM (the standard program:
    an estimate; a ROWLANES program: what it is)."""
    import numpy as np
    from opentpu.isasim import Machine, Slice
    cap = 256 * (pos // 256 + 1)
    need = spec.image(cfg0, cap, wformat=wf, head_format=hf, rows=1).nbytes
    cfg = dataclasses.replace(cfg0, DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    img = spec.image(cfg, cap, wformat=wf, head_format=hf, rows=1)
    progs = img.compile_step(pos)
    D = cfg.D
    out = {"portb": cfg.PORTB, "mcols": cfg.MCOLS, "chunks": 0, "mm_adv": 0, "mm_adv8": 0,
           "porta": 0, "porta_lines": 0, "other": 0, "kinds": Counter(), "mm_bytes": 0,
           "instructions": 0,
           # the row-lane step: chunks, MXU advances, port-A lines, the MM kinds
           "rl_chunks": 0, "rl_adv": 0, "rl_lines": 0, "rl_kinds": Counter(), "rl_bytes": 0}

    def cost(ins):
        name, detail, cyc, pb, pa = _describe(ins, cfg)
        out["instructions"] += 1
        if ins.op == I.MM:
            wfm, pair, quad = I.mm_format(ins)
            L = I.mm_lanes(ins) if hasattr(I, "mm_lanes") else 0
            N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
            unit, rsc = bool(ins.flags & I.F_UNIT), bool(ins.flags & I.F_RSC)
            fmt = FMT[wfm]
            bpc = I.blocks_per_chunk(wfm)
            KBc = -(-KB // bpc)
            rep = -(-M // cfg.MCOLS)
            out["chunks"] += pb
            out["mm_adv"] += cyc * rep
            # OCT: eight blocks an advance (int8: 8 chunks a cycle, 4-bit: 4, t2: 2)
            out["mm_adv8"] += N * -(-KB // 8) * rep
            out["porta"] += pa
            out["mm_bytes"] += pb * D + pa * 4
            r = " quad" if quad else " pair" if pair else f" rl{L}" if L else ""
            out["kinds"][fmt + r] += 1
            # port A: a line an advance (PAIR / QUAD: the reuse group's data in one line; RL:
            # the lanes' data in one line, the group's Sw line when not the first advance's)
            if L:
                lines = pa
            elif unit:
                lines = 0
            else:
                lines = N * -(-KB // (4 if quad else 2 if pair else 1))
            out["porta_lines"] += lines
            # the row-lane step: an RL MM as it is; else every scaled, non-RSC MM of one row
            # becomes an RL MM of the configuration's widest L for its format
            Lr = L or (Q.rl_lanes(fmt, cfg.PORTB, cfg.MCOLS) if M == 1 and not unit and not rsc
                       else 0)
            if Lr:
                G = -(-N // Lr)
                adv = G * KBc
                line = getattr(cfg, "ALINE", 32)
                sw_alone = Q.rl_sw_alone(fmt, Lr, line)
                out["rl_chunks"] += G * Lr * KBc
                out["rl_adv"] += adv
                out["rl_lines"] += adv + (G if sw_alone else 0)
                out["rl_bytes"] += G * Lr * KBc * D + G * Q.rl_scale_bytes(KB * D, fmt, Lr, D,
                                                                          line)
                out["rl_kinds"][f"{fmt} rl{Lr}"] += 1
            else:
                out["rl_chunks"] += pb
                out["rl_adv"] += cyc * rep
                out["rl_lines"] += lines
                out["rl_bytes"] += pb * D + pa * 4
                out["rl_kinds"][fmt + r] += 1
        else:
            out["other"] += cyc
            out["chunks"] += pb                     # the DMA's chunks (LD / ST, DSTEP)
            out["rl_chunks"] += pb

    orig = Slice.execute

    def ex(self, ins):
        cost(ins)
        return orig(self, ins)
    Slice.execute = ex
    try:
        Machine(cfg, progs, [np.zeros(need, np.uint8)]).run()
    finally:
        Slice.execute = orig
    out["bytes"] = out["chunks"] * D + out["porta"] * 4
    out["portb_cycles"] = -(-out["chunks"] // cfg.PORTB)
    # the port never slower than the MXU's advances; the other units' nominal cycles added as
    # the serial part they mostly are at M = 1 (the RTL runs put the MMs at 93-99% of a token)
    out["bound"] = max(out["portb_cycles"], out["mm_adv"], out["porta_lines"]) + out["other"]
    out["tok_s"] = mhz * 1e6 / out["bound"]
    # with OCT: the MXU at eight blocks an advance, the port at the chunks a cycle that needs
    # (int8 8, 4-bit 4: PORTB 8 / 4)
    out["bound8"] = max(out["mm_adv8"], -(-out["chunks"] // 8)) + out["other"]
    out["tok_s8"] = mhz * 1e6 / out["bound8"]
    # the row-lane step: its chunks at PORTB a cycle, its advances, its port-A lines
    out["rl_portb_cycles"] = -(-out["rl_chunks"] // cfg.PORTB)
    out["rl_bound"] = max(out["rl_portb_cycles"], out["rl_adv"], out["rl_lines"]) + out["other"]
    out["rl_tok_s"] = mhz * 1e6 / out["rl_bound"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--wformat", default="int8", choices=["int8", "int4", "fp4"])
    ap.add_argument("--head-format", default=None, choices=["int8", "int4", "fp4"])
    ap.add_argument("--portb", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--mcols", type=int, nargs="+", default=None, help="MXU columns (default "
                    "the target's)")
    ap.add_argument("--layers", type=int, default=0, help="0: all")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--rowlanes", action="store_true",
                    help="compile the step for Config.ROWLANES (the mode's own MMs)")
    a = ap.parse_args()
    spec = load_spec(model_dir(a.model))
    if a.layers:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]}
                                            if hasattr(spec, "kinds") else {"layers": a.layers}))
    hf = a.head_format or a.wformat
    print(f"{a.model} {a.wformat} weights, {hf} head, {spec.layers} layers, "
          f"OTPU_TARGET={os.environ.get('OTPU_TARGET', 'default')}, {a.mhz:g} MHz"
          f"{' (compiled with ROWLANES)' if a.rowlanes else ''}")
    print("PORTB | MCOLS | MM kinds | weight+scale bytes a token | port-B chunks | port-B cycles "
          "(chunks / PORTB) | MXU advances | port-A lines | other units | bound cycles | tok/s | "
          "with OCT (8 blocks an advance): MXU advances, bound, tok/s | row lanes (MM RL, L = "
          "quant.rl_lanes): MM kinds, bytes, chunks, port-B cycles, MXU advances, port-A lines, "
          "bound, tok/s")
    for pb in a.portb:
        for mc in (a.mcols or [None]):
            kw = dict(PORTB=pb, PAIR=True, DRAM_BYTES=1 << 40)
            if mc:
                kw["MCOLS"] = mc
            if a.rowlanes:
                kw["ROWLANES"] = True
            cfg = board_config(**kw)
            t = tally(spec, cfg, a.pos, a.mhz, a.wformat, hf)
            print(f"  ({t['instructions']:,} instructions executed)", file=sys.stderr)
            kinds = ", ".join(f"{k} x{n}" for k, n in sorted(t["kinds"].items()))
            rk = ", ".join(f"{k} x{n}" for k, n in sorted(t["rl_kinds"].items()))
            print(f"{pb} | {t['mcols']} | {kinds} | {t['bytes']:,} | {t['chunks']:,} | "
                  f"{t['portb_cycles']:,} | {t['mm_adv']:,} | {t['porta_lines']:,} | "
                  f"{t['other']:,} | {t['bound']:,} | {t['tok_s']:.1f} | "
                  f"{t['mm_adv8']:,}, {t['bound8']:,}, {t['tok_s8']:.1f} | "
                  f"{rk} | {t['rl_bytes']:,} | {t['rl_chunks']:,} | {t['rl_portb_cycles']:,} | "
                  f"{t['rl_adv']:,} | {t['rl_lines']:,} | {t['rl_bound']:,} | {t['rl_tok_s']:.1f}")


if __name__ == "__main__":
    main()
