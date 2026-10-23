#!/usr/bin/env python3
"""Measure the standalone C1 RTL kernel and export openTPU Lens profiles.

All runs execute RTL and compare every output bit against canonical ISA simulation.
Memory parameters are assumptions; cycles are observed handshakes, not baseline scaling.
"""
import argparse
import json
from pathlib import Path

from opentpu import c1_rtl as R, lens


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("build/lens/c1/rtl"))
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    case=R.workload(n=257,kb=9)
    profiles,reports=[],[]
    for slots in (1,2,4,8):
        # Hold service rate fixed while changing row latency; then hold latency
        # fixed while changing per-PC bandwidth. PC0 delay is a separate case.
        for latency,interval,delay in ((20,1,0),(80,1,0),(160,1,0),(80,2,0),(80,4,0),(80,1,160)):
            label=f"s{slots}-lat{latency}-beat{interval}-pc0delay{delay}"
            events,summary=R.run(case,args.output/label,slots=slots,latency=latency,
                                 beat_interval=interval,pc0_delay=delay)
            profiles.append(R.profile(case,events,summary,
                name=f"C1 E16 S{slots}; N257 KB9; LAT{latency} beat/{interval} PC0+{delay}"))
            reports.append({"case":label,**summary})
            print(f"{label}: {summary['cycles']} cycles; {summary['outputs_bit_exact']} exact outputs",flush=True)
    # Include real reset/backpressure events in a separate profile, never combine
    # recovery cycles with the steady command matrix above.
    events,summary=R.run(case,args.output/"reset-backpressure",slots=4,latency=80,
                         pc0_delay=160,request_percent=63,output_percent=47,
                         interleave=True,reset_after=25,seed=29)
    profiles.append(R.profile(case,events,summary,name="C1 E16 S4; reset recovery, randomized flow"))
    reports.append({"case":"reset-backpressure",**summary})
    bundle=lens.save(profiles,args.output/"c1-rtl.otpuprof")
    (args.output/"c1-rtl.html").write_text(lens.render(lens.load(bundle)))
    (args.output/"validation.json").write_text(json.dumps({"kind":"rtl-kernel-with-abstract-memory",
        "rows":257,"kb":9,"reports":reports},indent=2)+"\n")
    print(f"Lens: {bundle}")


if __name__=="__main__":
    main()
