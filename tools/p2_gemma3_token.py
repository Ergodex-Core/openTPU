"""v4 candidate P2 (docs/v4/proposal-2-four-lanes.md): one real Gemma 3 1B decode token on the
Verilator RTL with MXU_C chunk lanes against the ISA simulator, bit for bit.

The prompt (opentpu.lens's default) is prefilled and the next token's step run on the ISA
simulator (the resident decode program, as otpu-chat runs), then the same step on the RTL
through the board's memory path (the native adapter, the boot loader; lanes 1 .. C-1 on the
memory model's lane ports) from the same DRAM image: the argmax tokens, the logits and the
whole DRAM image (the KV cache the token wrote) are compared, and the traced run is saved as
a Lens profile (`lens summary` it).

    OTPU_LAT=30 PYTHONPATH=. python tools/p2_gemma3_token.py --mxu-c 4 --stall 0 \\
        -o build/lens/p2-gemma3-c4-stall0.otpuprof

The uarch is the card's (rtlsim.BOARD_UARCH: OTPU_TARGET's, ypcb-00338 m4) with MXU_C; the
memory model's stall percent (--stall) and latency (OTPU_LAT) as `lens record --axi`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu import lens, rtlsim                                 # noqa: E402
from opentpu.profile import parse                                 # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--prompt", default=lens.DEFAULT_PROMPT)
    ap.add_argument("--mxu-c", type=int, default=4)
    ap.add_argument("--stall", type=int, default=0, help="memory model stall percent (+axi_stall)")
    ap.add_argument("--mhz", type=float, default=200.0, help="the clock the times are quoted at")
    ap.add_argument("-o", "--out", help="the Lens profile to write")
    a = ap.parse_args(argv)

    import transformers
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import formats as FM
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    from opentpu.llm.rtl_backend import RtlBackend

    path = model_dir(a.model)
    spec = load_spec(path)
    wformat = FM.auto(spec, "auto")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    pos = len(ids) - 1
    cap = max(256, -(-(pos + 2) // 256) * 256)
    cfg = sim_config(spec, cap, board_config(PAIR=True, DSTEP=True, STREAM=True), lookup=True,
                     wformat=wformat)
    uarch = {**rtlsim.BOARD_UARCH, "MXU_C": a.mxu_c}
    print(f"{path.name}: {len(ids)} prompt tokens, step at pos {pos}, {wformat}, uarch MXU_C "
          f"{a.mxu_c}, memory stall {a.stall}%, latency {rtlsim.MEMORY['LAT']}", flush=True)

    eng = Engine(spec, load_weights(path), cap=cap, cfg=cfg, wformat=wformat, resident=True)
    t0 = time.time()
    eng.prefill(ids[:-1])
    print(f"prefill on the ISA simulator: {time.time() - t0:.0f} s", flush=True)
    isa = eng.backend
    n = eng.image.nbytes
    drams0 = [s.dram[:n].copy() for s in isa.machine.slices]

    class Traced(RtlBackend):
        def run(self, programs, args=None):
            drams, _, st = rtlsim.run(self.cfg, programs, self.drams, self.dram_lat,
                                      max_cycles=1 << 40, uarch=self.uarch, axi=True, boot=True,
                                      stall=self.stall, args=args, trace=True)
            self.drams = [d[:k].copy() for d, k in zip(drams, [len(d) for d in self.drams])]
            self.last = (programs, st)
            return st

    t = ids[-1]
    t0 = time.time()
    want = eng.step(t)
    print(f"ISA step: {time.time() - t0:.0f} s, argmax {int(np.argmax(want))} "
          f"{tok.decode([int(np.argmax(want))])!r}", flush=True)
    isa_dram = isa.machine.slices[0].dram[:n].copy()
    rtl = Traced(eng.cfg, drams0, uarch=uarch, axi=True, boot=True, stall=a.stall)
    eng.backend, eng.pos = rtl, eng.pos - 1
    t0 = time.time()
    got = eng.step(t)
    wall = time.time() - t0
    programs, st = rtl.last
    cyc = st["cycles"]
    same_logits = bool(np.array_equal(want.view(np.uint32), got.view(np.uint32)))
    same_dram = bool(np.array_equal(isa_dram, rtl.drams[0][:n]))
    print(f"RTL step: {wall:.0f} s, {cyc:,} cycles ({cyc / wall:.0f} cycles/s), argmax "
          f"{int(np.argmax(got))} {tok.decode([int(np.argmax(got))])!r}", flush=True)
    print(f"logits bit-exact: {same_logits}; DRAM image (KV cache) bit-exact: {same_dram}")
    if not same_logits:
        d = np.nonzero(want.view(np.uint32) != got.view(np.uint32))[0]
        print(f"  {len(d)} logits differ, first at {d[:8]}: ISA {want[d[:4]]} RTL {got[d[:4]]}")
    name = (f"{path.name} token at pos {pos} (board config, MXU_C {a.mxu_c}, stall {a.stall}%, "
            f"lat {rtlsim.MEMORY['LAT']}): {tok.decode([t])!r} -> "
            f"{tok.decode([int(np.argmax(got))])!r}")
    p = parse(st["trace"], eng.cfg, programs, name, C=a.mxu_c)
    p.cycles = cyc
    d = lens._mem(lens.to_data(p, uarch), eng.cfg, True)
    d["clock_mhz"] = a.mhz
    d["notes"].append({"level": "info", "text":
                       f"ISA simulator comparison: logits {'bit-exact' if same_logits else 'DIFFER'}, "
                       f"DRAM image {'bit-exact' if same_dram else 'DIFFERS'}."})
    rl = d["roofline"]
    print(f"roofline bound {rl['bound']:,} cycles: {100 * rl['efficiency']:.1f}%; "
          f"MACs/cycle {d['macs'] / cyc:.0f} of {d['peak_macs']}; "
          f"{cyc / a.mhz / 1e3:.1f} ms = {a.mhz * 1e6 / cyc:.1f} tok/s at {a.mhz:g} MHz")
    causes = lens.port_b_causes(d)
    if causes:
        tot = sum(causes.values()) or cyc
        print("port B: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in causes.items()
                                     if v > 0))
    if a.out:
        out = lens.save([d], a.out)
        print(f"wrote {out}")
    summary = {"cycles": cyc, "bound": rl["bound"], "efficiency": rl["efficiency"],
               "macs_per_cycle": d["macs"] / cyc, "same_logits": same_logits,
               "same_dram": same_dram, "argmax_isa": int(np.argmax(want)),
               "argmax_rtl": int(np.argmax(got)), "causes": causes, "wall_s": wall,
               "stall": a.stall, "lat": rtlsim.MEMORY["LAT"], "mxu_c": a.mxu_c}
    print(json.dumps(summary, default=float))
    return 0 if same_logits and same_dram else 1


if __name__ == "__main__":
    sys.exit(main())
