#!/usr/bin/env python3
"""Produce Lens evidence from native benchmark RTL; never a physical-FPGA profile."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from opentpu import c2_hbm_rtl as R, lens, rtlsim
from opentpu.c2_f2 import make_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('build/lens/c2-native'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ['rtl/c2/otpu_c2_hbm_kernel.sv', 'rtl/mxu/otpu_c2_mxu.sv',
             'rtl/mxu/otpu_mxu.sv', 'rtl/vpu/otpu_fp.sv', 'rtl/vpu/otpu_fpipe.sv',
             'rtl/top/otpu_pkg.sv', 'rtl/tech/generic/otpu_tech_dsp.sv',
             'sim/verilator/tb_c2_hbm.sv', 'opentpu/c2_hbm_rtl.py', 'opentpu/c2_f2.py']
    source = dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=rtlsim.ROOT,text=True).strip(),
                  verilator=rtlsim.verilator_version(),
                  files={p:hashlib.sha256((rtlsim.ROOT/p).read_bytes()).hexdigest() for p in paths})
    cases = [(257,9,10,0), (257,9,30,0), (257,9,100,0), (257,9,30,150),
             (31,65,100,200)]
    profiles, reports = [], []
    for rows,kb,latency,delay in cases:
        fixture = make_fixture(rows,kb,seed=20261019)
        previous = None
        for mode in (0,1):
            name=f'c2-native-{["legacy","decode"][mode]}-n{rows}-kb{kb}-lat{latency}-pc0delay{delay}'
            service=dict(latency=latency,stall=20,delay_pc0=delay,seed=921)
            evidence=R.run(fixture,mode,out_dir=args.out_dir/name,**service)
            if previous is not None:
                assert evidence['bits']==previous['bits'] and evidence['integers']==previous['integers']
            previous=evidence
            profiles.append(R.lens_data(fixture,mode,evidence,name,source,service))
            reports.append(dict(name=name,cycles=evidence['cycles'],counters=evidence['counters'],
                                mode=mode,rows=rows,kb=kb,fixture_sha256=fixture.digest(),service=service))
            print(name,evidence['cycles'],flush=True)
    target=lens.save(profiles,args.out_dir/'c2-native-rtl.otpuprof')
    (args.out_dir/'c2-native-rtl.html').write_text(lens.render(profiles,title='C2 native kernel RTL — synthetic memory service'))
    (args.out_dir/'c2-native-rtl.json').write_text(json.dumps(dict(source=source,
        evidence_kind='RTL simulation with synthetic native service; not FPGA/HBM hardware',cases=reports),indent=2)+'\n')
    print(target)


if __name__=='__main__':
    main()
