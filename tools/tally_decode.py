"""The cost of one decode token on the ISA (opentpu.profile's nominal unit cycles and port
transfers per instruction), compiled for a Config with a wide port B: the weight bytes a
token, the chunks, the MXU's advances, the port-bound and MXU-bound cycles, and what an
8-way column reuse (OCT: not in the ISA, docs/wide_port.md "PORTB 8") would take. No weights
are loaded: the step is compiled for the model's shapes and run on the ISA simulator over a
zeroed image (the loops over the layer kinds unroll as they run).

    OTPU_TARGET=f2-hbm python tools/tally_decode.py --model gemma3 --wformat int8 \\
        --head-format int8 --portb 1 2 4 [--layers 0] [--mhz 200]
"""
import argparse
import dataclasses
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentpu import isa as I  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.profile import _describe  # noqa: E402


def tally(spec, cfg0, pos: int, mhz: float, wf: str, hf: str) -> dict:
    """The step compiled for `cfg0` (its PORTB), then run on the ISA simulator over a zeroed
    image of the model's size (the step's loops over the layer kinds unroll as they execute;
    the weights' values do not change the tally), every executed instruction costed."""
    import numpy as np
    from opentpu.isasim import Machine, Slice
    cap = 256 * (pos // 256 + 1)
    need = spec.image(cfg0, cap, wformat=wf, head_format=hf, rows=1).nbytes
    cfg = dataclasses.replace(cfg0, DRAM_BYTES=1 << max(20, (need - 1).bit_length()))
    img = spec.image(cfg, cap, wformat=wf, head_format=hf, rows=1)
    progs = img.compile_step(pos)
    D = cfg.D
    out = {"portb": cfg.PORTB, "chunks": 0, "mm_adv": 0, "mm_adv8": 0, "mm_lanes": 0,
           "porta": 0, "other": 0, "kinds": Counter(), "mm_bytes": 0, "instructions": 0}
    L = cfg.PORTB                                   # row lanes: a weight row a column group

    def cost(ins):
        name, detail, cyc, pb, pa = _describe(ins, cfg)
        out["instructions"] += 1
        if ins.op == I.MM:
            wfm, pair, quad = I.mm_format(ins)
            N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
            chunks = N * -(-KB // I.blocks_per_chunk(wfm))
            rep = -(-M // cfg.MCOLS)
            out["chunks"] += chunks
            out["mm_adv"] += cyc * rep
            # OCT: eight blocks an advance (int8: 8 chunks a cycle, 4-bit: 4, t2: 2)
            out["mm_adv8"] += N * -(-KB // 8) * rep
            # row lanes (docs/wide_port.md "Future improvements"): PORTB lanes each streaming
            # its own row's chunks, a lane's columns taking a chunk's blocks a cycle: no
            # rounding of KB to the reuse, only of N to the lanes
            out["mm_lanes"] += -(-N // L) * -(-KB // I.blocks_per_chunk(wfm)) * rep
            out["porta"] += pa
            out["mm_bytes"] += chunks * D + pa * 4
            k = (["int8", "int4", "fp4", "nvfp4", "t2"][wfm] +
                 (" quad" if quad else " pair" if pair else ""))
            out["kinds"][k] += 1
        else:
            out["other"] += cyc
            out["chunks"] += pb                     # the DMA's chunks (LD / ST, DSTEP)

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
    out["bound"] = max(out["portb_cycles"], out["mm_adv"]) + out["other"]
    out["tok_s"] = mhz * 1e6 / out["bound"]
    # with OCT: the MXU at eight blocks an advance, the port at the chunks a cycle that needs
    # (int8 8, 4-bit 4: PORTB 8 / 4)
    out["bound8"] = max(out["mm_adv8"], -(-out["chunks"] // 8)) + out["other"]
    out["tok_s8"] = mhz * 1e6 / out["bound8"]
    out["boundl"] = max(out["mm_lanes"], out["portb_cycles"]) + out["other"]
    out["tok_sl"] = mhz * 1e6 / out["boundl"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--wformat", default="int8", choices=["int8", "int4", "fp4"])
    ap.add_argument("--head-format", default=None, choices=["int8", "int4", "fp4"])
    ap.add_argument("--portb", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--layers", type=int, default=0, help="0: all")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--mhz", type=float, default=200.0)
    a = ap.parse_args()
    spec = load_spec(model_dir(a.model))
    if a.layers:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]}
                                            if hasattr(spec, "kinds") else {"layers": a.layers}))
    hf = a.head_format or a.wformat
    print(f"{a.model} {a.wformat} weights, {hf} head, {spec.layers} layers, "
          f"OTPU_TARGET={os.environ.get('OTPU_TARGET', 'default')}, {a.mhz:g} MHz")
    print("PORTB | MM kinds | weight+scale bytes a token | port-B chunks | port-B cycles "
          "(chunks / PORTB) | MXU advances | other units | bound cycles | tok/s | "
          "with OCT (8 blocks an advance): MXU advances, bound, tok/s | with PORTB row lanes: "
          "MXU cycles, bound, tok/s")
    for pb in a.portb:
        cfg = board_config(PORTB=pb, PAIR=True, DRAM_BYTES=1 << 40)
        t = tally(spec, cfg, a.pos, a.mhz, a.wformat, hf)
        print(f"  ({t['instructions']:,} instructions executed)", file=sys.stderr)
        kinds = ", ".join(f"{k} x{n}" for k, n in sorted(t["kinds"].items()))
        print(f"{pb} | {kinds} | {t['bytes']:,} | {t['chunks']:,} | {t['portb_cycles']:,} | "
              f"{t['mm_adv']:,} | {t['other']:,} | {t['bound']:,} | {t['tok_s']:.1f} | "
              f"{t['mm_adv8']:,}, {t['bound8']:,}, {t['tok_s8']:.1f} | "
              f"{t['mm_lanes']:,}, {t['boundl']:,}, {t['tok_sl']:.1f}")


if __name__ == "__main__":
    main()
