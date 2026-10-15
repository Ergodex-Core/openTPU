"""Gemma 3 / Gemma 4 tiled prompt-run cycles by phase (RTL co-simulation, docs/tiled_prefill.md):
one P (or L) run of T rows at the start of bucket --blocks over the model's first --layers
layers, on the target's configuration (OTPU_TARGET; ACT_ROWS 16), the memory a port of --bw
percent of the core's D bytes a cycle (25: 32 B/cycle, the f2 image's DDR bridge) and --lat
cycles' latency.

    python3 tools/perf_gemma_prefill.py --model gemma3 [--layers 6] [--blocks 1] [--rows 12]
        [--kind P] [--bw 25] [--lat 60] [--wformat int8] [--head-format int8] [--compile-only]

One JSON line a run: cycles, cycles a row, the whole model's estimate (the layers' cycles
scaled to every layer), MXU / VPU / DMA busy, port-B chunks, and per phase (the innermost
kernel function of an instruction's source that names one: gather, attn-proj, kv, attn-core,
attn-out, mlp, norm, head) cycles and units' busy.
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

from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.profile import parse  # noqa: E402

PHASES = [("_lm_head_rows", "head"), ("_gathered_out", "gather"), ("_pos_rows", "gather"),
          ("_scores", "attn-core"), ("_finish", "attn-core"), ("_tile_blocks", "attn-core"),
          ("_attend_heads", "attn-core"), ("_heads_loop", "attn-core"),
          ("kv_append", "kv"), ("_add_norm", "norm"), ("_add_parts", "out-proj"),
          ("_mlp_tile", "mlp"), ("_gemma_attention_tile", "attn-proj"),
          ("_gemma_layer_tile", "attn-proj")]


def phase(ins) -> str:
    fns = [fn for _, _, fn in ins.src]
    for f, name in PHASES:
        if f in fns:
            if name == "out-proj" and "_mlp_tile" in fns:
                return "mlp"
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--rows", default="12")
    ap.add_argument("--kind", default="P", choices=["P", "L"])
    ap.add_argument("--bw", type=int, default=25)
    ap.add_argument("--lat", type=int, default=60)
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--narrow", action="store_true",
                    help="the attention's tiles of at most MCOLS query rows (tiled.WIDE off)")
    ap.add_argument("--dump", help="write every instruction's record (JSON lines) here")
    a = ap.parse_args()
    from opentpu.llm import tiled
    if a.narrow:
        tiled.WIDE = False
    spec0 = load_spec(model_dir(a.model))
    spec = spec0.truncated(range(a.layers)) if a.layers else spec0
    L = spec0.layers
    for T in [int(x) for x in a.rows.split(",")]:
        kw = dict(ACT_ROWS=16)
        ikw = dict(rows=16, wformat=a.wformat, head_format=a.head_format, lookup=True)
        probe = spec.image(board_config(DRAM_BYTES=1 << 40, **kw), a.cap, **ikw)
        cfg = board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()), **kw)
        img = spec.image(cfg, a.cap, **ikw)
        rec = dict(model=a.model, layers=spec.layers, blocks=a.blocks, T=T, kind=a.kind,
                   bw=a.bw, lat=a.lat, wformat=a.wformat)
        t = time.time()
        try:
            progs = tiled.compile_tile_run(img, a.blocks, T, a.kind)[0]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            print(json.dumps(rec), flush=True)
            continue
        rec["instructions"] = max(len(p) for p in progs)
        rec["imem_fits"] = rec["instructions"] * 8 <= cfg.IMEM_WORDS
        rec["compile_s"] = round(time.time() - t, 1)
        if a.compile_only:
            print(json.dumps(rec), flush=True)
            continue
        dram = np.zeros(img.nbytes, np.uint8)
        t = time.time()
        _, _, st = rtlsim.run(cfg, progs, [dram], trace=True, uarch=rtlsim.BOARD_UARCH, axi=True,
                              boot=True, stall=0, max_cycles=1 << 40, ldc=0, bw=a.bw,
                              lat=a.lat, plusargs=[])
        rec["sim_s"] = round(time.time() - t)
        p = parse(st["trace"], cfg, progs, "gemma")
        p.cycles = st["cycles"]
        ps = p.roofline()["per_slice"][0]
        ph = p.phases(lambda r: phase(progs[0][r.pc]))
        lay = sum(v["cycles"] for k, v in ph.items() if k not in ("gather", "head"))
        rest = p.cycles - lay
        rec.update(cycles=p.cycles, per_row=round(p.cycles / T), portb=ps["portb"],
                   mxu_blocks=ps["mxu"],
                   busy={k: v for k, v in p.unit_busy(0).items() if v},
                   full_model_per_row=round((rest + lay * L / spec.layers) / T),
                   phases={k: dict(cycles=v["cycles"], portb=v["portb"], VPU=v["VPU"],
                                   MXU=v["MXU"]) for k, v in sorted(ph.items())})
        print(json.dumps(rec), flush=True)
        if a.dump:
            with open(a.dump, "w") as f:
                for r in sorted(p.slice_recs(0), key=lambda r: r.idx):
                    ins = progs[0][r.pc]
                    f.write(json.dumps([r.idx, r.pc, r.name, r.detail, r.dispatch, r.ready,
                                        r.start, r.release, r.end, phase(ins),
                                        [f"{fn}:{ln}" for _, ln, fn in ins.src[:3]]]) + "\n")


if __name__ == "__main__":
    main()
