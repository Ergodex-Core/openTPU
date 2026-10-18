"""Validate C3's reference against canonical ISA and export both through openTPU Lens.

    PYTHONPATH=. python tools/c3_head_check.py --output build/lens/c3

Default: seeded synthetic head, real Gemma 3 hidden dimension, reduced vocabulary.
--input accepts an npz of canonical uint8 `weights`, uint8 `scales`, fp32 `normalized`,
and the separately supplied uint8 `embedding_weights` and `embedding_scales`.
No model conversion, runtime dispatch, RTL simulation or FPGA performance is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from opentpu import lens, quant as Q
from opentpu.isasim import Machine
from opentpu.llm import vocab_engine as C


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=1152)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("build/lens/c3"))
    a = p.parse_args(argv)
    if a.input:
        with np.load(a.input, allow_pickle=False) as z:
            w, s, x, ew, es = (z[n] for n in ("weights", "scales", "normalized",
                                               "embedding_weights", "embedding_scales"))
        label = f"input {a.input.name}"
    else:
        C.Layout(vocab=a.vocab, hidden=a.hidden)  # bounds checked before allocating
        rng = np.random.default_rng(a.seed)
        source = (rng.standard_normal((a.vocab, a.hidden)) * 0.02).astype(np.float32)
        w, s = Q.quantize_mxu(source, "int8", 128)
        ew, es = w, s
        x = rng.standard_normal(a.hidden).astype(np.float32)
        label = f"synthetic seed {a.seed}, {a.vocab} rows x {a.hidden}"
    layout = C.Layout(vocab=w.shape[0], hidden=w.shape[1])
    sidecar = C.Sidecar.build(layout, w, s, embedding_weights=ew, embedding_scales=es)
    sidecar.verify_tied(w, s, ew, es)
    cfg, prog, dram, out = C.canonical_program(layout, w, s, x)
    machine = Machine(cfg, [prog], [dram.copy()]).run()
    expected = machine.slices[0].dram[out:out + 4 * layout.vocab].view(np.uint32)
    run = C.execute(sidecar, C.Activation.quantize(x))
    np.testing.assert_array_equal(run.logits_bits, expected)
    canonical = lens.isa_data(f"Canonical ISA head ({label}); analytic cycles", cfg, [prog], [dram])
    canonical["clock_mhz"] = 200.0
    # Stock Lens's generic ISA note calls it a bound. Be precise for this comparison.
    canonical["notes"] = [{"level": "info", "text": "Actual canonical ISA head program executed. "
                            "Displayed cycles are stock Lens nominal instruction work, not measured RTL. "
                            "This profile includes QACT and full-logit stores; C3 starts after QACT."}]
    c3 = C.lens_profile(run)
    c3["name"] += f" ({label})"
    report = {"workload": label, "vocab": layout.vocab, "hidden": layout.hidden,
              "logits_bit_exact": True, "winner": {"bits": run.winner.bits, "token_id": run.winner.token_id},
              "canonical_source_sha256": sidecar.source_sha256,
              "sidecar_bytes": sum(map(len, sidecar.pcs)),
              "reference_seconds": run.reference_seconds,
              "timing_status": "reference CPU wall time plus explicit analytic service accounting; no RTL/HBM timing",
              "model_assumptions": c3["c3"]["assumptions"]}
    a.output.mkdir(parents=True, exist_ok=True)
    path = lens.save([canonical, c3], a.output / "c3-head-reference.otpuprof")
    (a.output / "c3-head-reference.html").write_text(lens.render(lens.load(path)))
    (a.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "sidecar-descriptor.json").write_text(json.dumps(sidecar.descriptor(), indent=2) + "\n")
    (a.output / "summary.txt").write_text("\n\n".join(lens.render_summary(x) for x in (canonical, c3)) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Lens profile: {path}")


if __name__ == "__main__":
    main()
