#!/usr/bin/env python3
"""Run seeded native-adapter tests or the Vivado memory-fabric test."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--fabric', action='store_true', help='Use Vivado and generated memory/PCIS IP')
    ap.add_argument('--ip-run', type=Path)
    ap.add_argument('--pcis-run', type=Path)
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--iverilog', default='iverilog')
    ap.add_argument('--vvp', default='vvp')
    ap.add_argument('--iverilog-base', type=Path, help='Optional -B directory for an unpacked Icarus install')
    a = ap.parse_args()
    out = a.output.resolve()
    if out.exists():
        ap.error('Select a fresh output directory')
    if a.fabric and (a.ip_run is None or a.pcis_run is None):
        ap.error('--fabric requires --ip-run and --pcis-run')
    out.mkdir(parents=True)
    if a.fabric:
        rtl = out / 'rtl'
        rtl.mkdir()
        for p in [ROOT / 'tests/f2_burst200/tb_f2_mem.sv', ROOT / 'tests/f2_burst200/axi_hbm_model.sv',
                  ROOT / 'rtl/boards/f2/burst200/otpu_native_axi.sv', ROOT / 'rtl/boards/f2/burst200/otpu_hbm_geometry.sv']:
            shutil.copy2(p, rtl / p.name)
        cmd = [a.vivado, '-mode', 'batch', '-source',
               str(ROOT / 'boards/f2/burst200/scripts/sim_f2_mem.tcl'), '-tclargs',
               str(out), str(a.ip_run.resolve()), str(a.pcis_run.resolve())]
        with (out / 'console.log').open('w') as log:
            subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        if 'PASS' not in (out / 'console.log').read_text():
            raise SystemExit('Fabric simulation did not report PASS')
        return
    results = []
    for burst, outstanding in [(1, 8), (16, 8), (16, 32), (16, 128)]:
        exe = out / f'b{burst}_o{outstanding}'
        cmd = [a.iverilog, '-g2012', '-s', 'tb_native_axi',
               f'-Ptb_native_axi.BURST={burst}', f'-Ptb_native_axi.OUT={outstanding}',
               '-o', str(exe), str(ROOT / 'rtl/boards/f2/burst200/otpu_native_axi.sv'),
               str(ROOT / 'tests/f2_burst200/tb_native_axi.sv')]
        if a.iverilog_base:
            cmd[1:1] = ['-B', str(a.iverilog_base.resolve())]
        with (out / f'compile_b{burst}_o{outstanding}.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        for seed in [1, 42, 2026, 4700]:
            logpath = out / f'b{burst}_o{outstanding}_s{seed}.log'
            with logpath.open('w') as log:
                subprocess.run([a.vvp, str(exe), f'+seed={seed}'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            if 'PASS' not in logpath.read_text():
                raise SystemExit(f'No PASS in {logpath}')
            results.append(dict(burst=burst, outstanding=outstanding, seed=seed, passed=True))
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} native-adapter trials')


if __name__ == '__main__':
    main()
