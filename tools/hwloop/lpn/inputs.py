#!/usr/bin/env python3
"""The LPN model's input bundle from a frozen openTPU export: the decode token's program and
DRAM image exactly as tools/perf_qwen.py hands them to the RTL testbench, written as
prog_<slice>.hex / dram_<slice>.bin (rtlsim._write_hex's format, big-endian words for $fread)
and input.json (LPNDialect examples/opentpu/workload.py's manifest: the pinned revision, the
Config, the arguments, the instruction counts), plus workload.json: what was compiled, the
ISA simulator's result hashes (the numerical reference the LPN frontend is checked against),
the per-opcode tally and the DRAM bytes from the footprints, and the ISA features upstream's
timing backend rejects if the program uses any.

Runs under the venv python with the export tree on sys.path (the tree's own compiler and ISA
simulator; not this worktree's):

    python tools/hwloop/lpn/inputs.py --tree EXPORT --out DIR --revision PIN_SHA \
        [--model gemma3 --layers 2 --pos 9 --wformat int8 --head-format int8 --portb 1 --mcols N]

The environment (OTPU_TARGET, OTPU_NCH, ...) is the run's, as run-rtl's.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

# the fields of upstream's Config at the pin (opentpu/isasim.py f237dab2); anything else the
# export's Config carries (PORTB, T2, ...) is recorded in workload.json and left out of input.json
PIN_CONFIG_FIELDS = ("S", "D", "MCOLS", "ACT_BLOCKS", "TMEM_WORDS", "DRAM_BYTES", "IMEM_WORDS", "LANES",
                     "PAIR", "DSTEP", "STREAM", "ACT_ROWS", "SEQ_WC")


def sha256_bytes(b) -> str:
    return hashlib.sha256(memoryview(b)).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="the frozen export (git archive), the compiler and ISA simulator used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", required=True, help="the openTPU commit the LPN model is pinned to (input.json's revision)")
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--layers", default="2")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--portb", type=int, default=None)
    ap.add_argument("--mcols", type=int, default=None)
    ap.add_argument("--no-isa", action="store_true", help="skip the ISA simulator run (no reference hashes, no tally)")
    a = ap.parse_args(argv)

    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree))
    import numpy as np
    from opentpu import isa as I
    from opentpu import waits as W
    from opentpu.isasim import Machine, board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import load_weights, rope_tables
    import opentpu
    if not Path(opentpu.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"opentpu imported from {opentpu.__file__}, not from {tree}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    # --- the program and image, as tools/perf_qwen.py's main builds them (the decode step branch)
    path = model_dir(a.model)
    spec = load_spec(path)
    layers = [int(x) for x in a.layers.split(",")]
    if hasattr(spec, "truncated") and layers != [0]:
        spec = spec.truncated(layers if len(layers) > 1 else range(layers[0]))
    elif len(layers) > 1:
        raise SystemExit("--layers: a list of layers needs a Gemma 4 checkpoint")
    elif layers[0]:
        spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:layers[0]]} if hasattr(spec, "kinds")
                                            else {"layers": layers[0]}))
    R = 1
    cap = 256 * ((a.pos + R - 1) // 256 + 1)
    Wt = load_weights(path)
    wkw = dict(wformat=a.wformat, head_format=a.head_format, rows=R)
    mk = {"MCOLS": a.mcols} if a.mcols else {}
    if a.portb:
        mk.update(PORTB=a.portb, PAIR=True)          # perf_qwen --portb: column reuse on, as the f2 bitstreams
    need = spec.image(board_config(DRAM_BYTES=1 << 40, **mk), cap, **wkw).nbytes
    cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()), **mk)
    img = spec.image(cfg, cap, **wkw)
    dram = img.build(Wt)[0]
    rows = [(0, a.pos + r) for r in range(R)]
    if hasattr(img, "host_inputs"):
        parts = img.host_inputs([791 + r for r in range(R)], [p for _, p in rows])
    else:
        emb = np.stack([np.asarray(Wt["model.embed_tokens.weight"][791 + r], np.float32) for r in range(R)])
        tabs = [rope_tables(spec, p) for _, p in rows]
        c, s = np.stack([t[0] for t in tabs]), np.stack([t[1] for t in tabs])
        parts = [(img.io[k], v) for k, v in (("x", emb), ("cos", c), ("sin", s))]
    for addr, v in parts:
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).ravel()
        dram[addr:addr + b.size] = b
    for addr, v in getattr(img, "host_rows", lambda t: [])([791 + r for r in range(R)]):
        b = np.ascontiguousarray(v).view(np.uint8).ravel()
        dram[addr:addr + b.size] = b
    progs = img.compile_step(a.pos)
    if 8 * len(progs[0]) > cfg.IMEM_WORDS:
        cfg = dataclasses.replace(cfg, IMEM_WORDS=1 << (8 * len(progs[0]) - 1).bit_length())
    build_s = time.monotonic() - t0

    # --- the bundle (rtlsim._write_hex's truncation; $fread's big-endian words)
    full = dataclasses.asdict(cfg)
    pin_cfg = {k: full[k] for k in PIN_CONFIG_FIELDS if k in full}
    dropped = {k: v for k, v in full.items() if k not in PIN_CONFIG_FIELDS}
    forced = {}
    if pin_cfg.get("SEQ_WC"):
        # the programs do not change with SEQ_WC (the host loads the wait entries); upstream's
        # model requires the range sequencer, so the manifest says so
        forced["SEQ_WC"] = (pin_cfg["SEQ_WC"], False)
        pin_cfg["SEQ_WC"] = False
    hashes = {}
    for s in range(cfg.S):
        words = np.asarray(I.assemble(progs[s]), "<u4")
        if len(words) > cfg.IMEM_WORDS:
            raise SystemExit("program does not fit IMEM")
        nz = np.nonzero(words)[0]
        n = int(nz[-1]) + 1 if len(nz) else 1
        (out / f"prog_{s}.hex").write_text("\n".join(f"{int(w):08x}" for w in words[:n]) + "\n")
        im = np.asarray(dram if s == 0 else img.build(Wt)[s], np.uint8)
        im = np.concatenate([im, np.zeros(-len(im) % 4, np.uint8)])
        im.view("<u4").astype(">u4").tofile(out / f"dram_{s}.bin")
        for f in (f"prog_{s}.hex", f"dram_{s}.bin"):
            hashes[f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    manifest = {"revision": a.revision, "config": pin_cfg, "args": [], "instruction_counts": [len(p) for p in progs]}
    (out / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- the program's static tally and the ISA features upstream rejects
    names = I.OPNAMES
    ops: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for ins in progs[0]:
        nm = names.get(ins.op, f"op{ins.op:#x}")
        ops[nm] = ops.get(nm, 0) + 1
        if nm in ("WAITW", "DSTEP", "STREAM", "WAITC") or (nm == "HALT" and ins.flags & getattr(I, "F_CHAIN", 1)):
            key = "HALT CHAIN" if nm == "HALT" else nm
            rejected[key] = rejected.get(key, 0) + 1
    work = {"tree": str(tree), "model": a.model, "checkpoint": str(path), "layers": a.layers, "pos": a.pos,
            "wformat": a.wformat, "head_format": a.head_format, "portb": a.portb, "mcols": a.mcols,
            "config_full": full, "config_pin_fields": pin_cfg, "config_dropped_fields": dropped, "config_forced": forced,
            "instructions_static": len(progs[0]), "opcodes_static": ops, "rejected_features_static": rejected,
            "image_bytes": int(len(dram)), "dram_bytes_config": cfg.DRAM_BYTES, "hashes": hashes,
            "build_s": build_s, "compiler": "the export tree's opentpu.llm compile_step (perf_qwen's decode-step path)"}

    # --- the ISA simulator: the numerical reference (its DRAM / TMEM hashes) and the dynamic tally
    if not a.no_isa:
        t1 = time.monotonic()
        m = Machine(cfg, [progs[0]], [dram.copy()], None)
        dyn: dict[str, int] = {}
        bytes_rd = bytes_wr = 0
        chunks = 0
        n_cmd = 0
        SP_DRAM = getattr(W, "SP_DRAM", 0)

        def record(self):
            nonlocal bytes_rd, bytes_wr, chunks, n_cmd
            ins = self.prog[self.pc]
            regs = [(self.reg(r),) * 2 for r in range(16)]
            fp = W.footprint(ins, regs, cfg.D, cfg.S)
            nm = names.get(ins.op, f"op{ins.op:#x}")
            dyn[nm] = dyn.get(nm, 0) + 1
            n_cmd += 1
            for sp, lo, hi in fp.rd:
                if sp == SP_DRAM:
                    bytes_rd += hi - lo
            for sp, lo, hi in fp.wr:
                if sp == SP_DRAM:
                    bytes_wr += hi - lo
            if ins.op == I.MM:
                w = [v[0] for v in W.resolve(ins, regs)]
                n, kb = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF
                chunks += n * kb
            original()

        from types import MethodType
        sl = m.slices[0]
        original = sl.step
        sl.step = MethodType(record, sl)
        m.run(max_steps=1 << 40)
        work["isa"] = {"wall_s": time.monotonic() - t1, "dynamic_commands": n_cmd, "opcodes_dynamic": dyn,
                       "dram_read_bytes_footprint": bytes_rd, "dram_write_bytes_footprint": bytes_wr,
                       "mm_weight_chunks": chunks, "mm_weight_bytes": chunks * cfg.D,
                       "dram_sha256": sha256_bytes(np.ascontiguousarray(sl.dram, np.uint8)),
                       "tmem_sha256": sha256_bytes(np.ascontiguousarray(sl.tmem, np.uint32)),
                       "note": "the export's ISA simulator (opentpu.isasim.Machine): the numerical reference; "
                               "footprint bytes are the instructions' DRAM ranges (weights, scales, KV, I/O), "
                               "not beats"}
        np.ascontiguousarray(sl.tmem, np.uint32).tofile(out / "isa_tmem_0.bin")
    (out / "workload.json").write_text(json.dumps(work, indent=2, default=str) + "\n")
    print(json.dumps({k: work[k] for k in ("instructions_static", "opcodes_static", "rejected_features_static", "image_bytes",
                                            "config_pin_fields", "config_dropped_fields", "config_forced", "build_s")}
                     | ({"isa": {k: v for k, v in work["isa"].items() if k != "opcodes_dynamic"}} if "isa" in work else {}),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
