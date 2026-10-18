"""Execute isolated C3 RTL, compare every logit to canonical ISA, export openTPU Lens.

Synthetic row service only: one header slot plus KB 128-byte blocks per row;
initial latency and independent random input stalls, no AXI/HBM controller.
200 MHz is a visualization assumption, not achieved timing or F2 performance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from opentpu import lens, quant as Q, rtlsim
from opentpu.isasim import Machine
from opentpu.llm import vocab_engine as C, vocab_rtl as R


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("build/lens/c3-rtl"))
    args = parser.parse_args(argv)
    if args.input:
        with np.load(args.input, allow_pickle=False) as data:
            w, s, x, ew, es = (data[key] for key in ("weights", "scales", "normalized",
                                                   "embedding_weights", "embedding_scales"))
        workload = f"input {args.input.name}"
    else:
        C.Layout(vocab=args.vocab, hidden=args.hidden)
        if args.vocab > 8192:
            raise ValueError("standalone harness is bounded to 8192 vocabulary rows")
        rng = np.random.default_rng(args.seed)
        w, s = Q.quantize_mxu((rng.standard_normal((args.vocab, args.hidden))*0.02).astype(np.float32), "int8", 128)
        x = rng.standard_normal(args.hidden).astype(np.float32)
        ew, es = w, s
        workload = f"synthetic seed {args.seed}, {args.vocab} x {args.hidden}"
    layout = C.Layout(vocab=w.shape[0], hidden=w.shape[1])
    sidecar = C.Sidecar.build(layout, w, s, embedding_weights=ew, embedding_scales=es)
    activation = C.Activation.quantize(x)
    cfg, prog, dram, out = C.canonical_program(layout, w, s, x)
    canonical = Machine(cfg, [prog], [dram.copy()]).run()
    expected = canonical.slices[0].dram[out:out+4*layout.vocab].view(np.uint32)
    reference = C.execute(sidecar, activation)
    np.testing.assert_array_equal(reference.logits_bits, expected)
    scenarios = [("one engine, unstalled service", dict(engines=1, latency=30)),
                 ("four engines, unstalled service", dict(engines=4, latency=30)),
                 ("four engines, 20% service stalls", dict(engines=4, latency=30, stalls=20)),
                 ("four engines, service + diagnostic stalls", dict(engines=4, latency=30, stalls=20, output_stalls=35))]
    args.output.mkdir(parents=True, exist_ok=True)
    profiles, runs = [], []
    for index, (label, options) in enumerate(scenarios):
        sim = R.run(sidecar, activation, seed=args.seed, **options)
        np.testing.assert_array_equal(sim.logits_bits[0], expected)
        assert sim.winners[0] == reference.winner
        profile = R.lens_profile(sim)
        profile["name"] = f"C3 RTL: {label} ({workload}); synthetic memory"
        profiles.append(profile)
        trace = args.output / f"events-{index}.json.gz"
        with gzip.open(trace, "wt") as stream:
            json.dump(sim.events, stream, separators=(",", ":"))
        runs.append({"scenario": label, "cycles": profile["cycles"], "logits_bit_exact": True,
                     "trace": trace.name, **profile["c3"]})
        print(f"{label}: {profile['cycles']} actual RTL cycles, every logit exact", flush=True)
    model = C.lens_profile(reference)
    model["name"] = f"C3 legacy serial accounting model ({workload}); NOT RTL"
    profiles.append(model)
    profile_path = lens.save(profiles, args.output / "c3-head-rtl.otpuprof")
    (args.output / "c3-head-rtl.html").write_text(lens.render(lens.load(profile_path)))
    (args.output / "summary.txt").write_text("\n\n".join(lens.render_summary(p) for p in profiles)+"\n")
    np.savez_compressed(args.output / "input.npz", weights=w, scales=s, normalized=x,
                        embedding_weights=ew, embedding_scales=es)
    report = {"workload": workload, "vocab": layout.vocab, "hidden": layout.hidden,
              "timing_status": "actual isolated RTL cycles under synthetic row service; no HBM, board, F2 timing or whole-token claim",
              "display_clock_mhz_assumed": 200, "verilator": rtlsim.verilator_version(),
              "source_sha256": sidecar.source_sha256,
              "rtl_source_sha256": {p: hashlib.sha256((rtlsim.ROOT/p).read_bytes()).hexdigest() for p in R.SOURCES},
              "runs": runs, "model_cycles_not_comparable_speedup": model["cycles"],
              "limitations": ["Production HBM request queues, response reassembly and address generation are not RTL yet.",
                              "The adapter must flush old responses on reset; this ordered block stream has no generation tag.",
                              "Standalone diagnostic stream is consumed by the testbench; there are no TMEM stores.",
                              "No command capability is advertised to greedy generation; fallback guards are unchanged.",
                              "No full checkpoint, full-token simulation, placement, achieved clock or F2 run."]}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Lens profile: {profile_path}")


if __name__ == "__main__":
    main()
