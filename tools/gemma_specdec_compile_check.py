"""Configuration-only fit gate for the measured wide-port Gemma prompt-lookup loop.

Reads no weights and runs no simulated or physical inference.  A successful
result proves compiler/image capacity only, never timing closure or speedup.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from opentpu.isasim import board_config
    from opentpu.llm.gemma3 import Spec
    from opentpu.llm.gemma_specdec import E, V, KINDS, compile_gen
    from opentpu.llm import generate as G

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    spec = Spec.from_hf(args.model)
    cfg = board_config(DRAM_BYTES=4 << 30, MCOLS=8, PORTB=4, PAIR=True,
                       DSTEP=True, STREAM=True, T2=True, ROWLANES=True,
                       ALINE=64, ACT_ROWS=16, SEQ_WC=True)
    baseline = spec.image(cfg, args.cap, rows=2, lookup=True,
                          wformat="int8", head_format="int8")
    image = spec.image(cfg, args.cap, rows=2, lookup=True,
                       wformat="int8", head_format="int8", gemma_prompt_lookup=True)
    result = dict(scope="configuration-only; no weights, inference, hardware or TPS",
                  config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                  spec=asdict(spec), cfg=asdict(cfg), cap=args.cap,
                  image_bytes=image.nbytes, extra_bytes=image.nbytes - baseline.nbytes,
                  instruction_capacity=cfg.IMEM_WORDS // 8, programs=[])
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    ok = True
    for bucket in range(1, G.buckets(args.cap, image.block) + 1):
        for kind in (V, E):
            row = dict(bucket=bucket, kind=KINDS[kind])
            try:
                programs = compile_gen(image, bucket, kind)
                row.update(instructions=[len(p) for p in programs], fits=G.fits(image, programs))
            except Exception as error:
                row.update(fits=False, error_type=type(error).__name__, error=str(error))
            result["programs"].append(row)
            ok &= row["fits"]
            args.json.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    result["all_fit"] = bool(ok)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
