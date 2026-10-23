"""Configuration-only FunctionGemma/Gemma image budget and compiler check.

No weights, inference or hardware. Independent images both start at address zero;
their summed size is a budget, not a working co-resident layout.
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
    from opentpu import waits
    from opentpu.isasim import board_config
    from opentpu.llm import generate as G
    from opentpu.llm.gemma3 import Spec

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = board_config(DRAM_BYTES=4 << 30, MCOLS=8, PORTB=4, PAIR=True,
                       DSTEP=True, STREAM=True, T2=True, ROWLANES=True,
                       ALINE=64, ACT_ROWS=16, SEQ_WC=True)
    rows, cap = 8, 512
    result = dict(scope=__doc__.strip(), cfg=asdict(cfg), cap=cap, rows=rows, models={})
    images = {}
    for name, path in (("target", args.target), ("draft", args.draft)):
        spec = Spec.from_hf(path)
        image = spec.image(cfg, cap, rows=rows, lookup=True, wformat="int8",
                           head_format="int8", gemma_prompt_lookup=(name == "target"))
        images[name] = image
        record = dict(config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                      spec=asdict(spec), image_bytes=image.nbytes, wformat="int8", head_format="int8")
        result["models"][name] = record
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        decode, _ = image.compile_decode(1, 0)
        generate = G.compile_bucket(image, 1, 0, image.block, chain=True)
        record.update(decode_instructions=[len(p) for p in decode],
                      generate_part_instructions=[[len(p) for p in part]
                          for part in (generate if isinstance(generate, tuple) else (generate,))],
                      generate_fits=all(G.fits(image, part) for part in
                          (generate if isinstance(generate, tuple) else (generate,))))
        print(json.dumps(dict(model=name, **record)), flush=True)
    image_sum = sum(image.nbytes for image in images.values())
    # A conservative budget reserves two transient program areas, MiB alignment
    # for each image and the existing host's optional 256 MiB KV staging area.
    reserve = 2 * waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC) + 2 * (1 << 20) + (256 << 20)
    result.update(image_sum_bytes=image_sum, conservative_extra_reserve_bytes=reserve,
                  budget_with_reserve_bytes=image_sum + reserve,
                  budget_fits=image_sum + reserve <= cfg.DRAM_BYTES,
                  co_resident_ready=False,
                  blockers=["Gemma Image/Engine has no relocation/base argument: independently compiled images alias",
                            "No target/draft device controller, ID policy, KV synchronization or rollback qualification",
                            "No acceptance, latency, traffic or throughput measurement for FunctionGemma drafts"])
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("image_sum_bytes", "budget_with_reserve_bytes",
                                           "budget_fits", "co_resident_ready")}), flush=True)


if __name__ == "__main__":
    main()
