#!/usr/bin/env python3
"""Bounded C1 E16/S4 synthesis for the F2 part; no shell or hardware operations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ['rtl/vpu/otpu_fp.sv', 'rtl/vpu/otpu_fpipe.sv',
           'rtl/c1/otpu_c1_row_engine.sv', 'rtl/c1/otpu_c1_cluster.sv',
           'rtl/c1/otpu_c1_epilogue.sv', 'rtl/c1/otpu_c1_top.sv']
PART = 'xcvu47p-fsvh2892-2-e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records(root, names):
    result = {}
    for name in names:
        p = root / name
        if p.is_symlink() or not p.is_file():
            raise RuntimeError(f'Expected regular input: {p}')
        result[name] = dict(bytes=p.stat().st_size, sha256=sha(p))
    return result


def tcl(root):
    # Controlled paths are Tcl braced; reject braces rather than interpolate them.
    if any(c in str(root) for c in '{}\\\n\r'):
        raise ValueError('Unsupported source path')
    reads = '\n'.join(f'read_verilog -sv {{{root / n}}}' for n in SOURCES)
    return f'''set_param general.maxThreads 2
create_project -in_memory -part {PART}
{reads}
set fp [open clocks.xdc w]
puts $fp {{create_clock -name core_clk -period 5.000 [get_ports clk]}}
close $fp
read_xdc clocks.xdc
set_property USED_IN {{synthesis implementation OUT_OF_CONTEXT}} [get_files clocks.xdc]
synth_design -mode out_of_context -top otpu_c1_top -part {PART} -generic ENGINES=16 -generic SLOTS=4 -flatten_hierarchy rebuilt
set blackboxes [get_cells -hier -filter {{IS_BLACKBOX == 1}}]
if {{[llength $blackboxes] != 0}} {{error "Unexpected black boxes: $blackboxes"}}
write_checkpoint post_synth.dcp
report_utilization -hierarchical -file synth_utilization.rpt
report_timing_summary -report_unconstrained -file synth_timing.rpt
report_drc -file synth_drc.rpt
set out [open complete.txt {{WRONLY CREAT EXCL}}]
puts $out C1_E16_S4_OOC_SYNTH_COMPLETE_NOT_SHELL_QUALIFIED
close $out
close_project
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--vivado', default='vivado')
    a = p.parse_args()
    if sha(a.source_manifest) != a.source_manifest_sha256:
        raise RuntimeError('Source manifest digest mismatch')
    source = json.loads(a.source_manifest.read_text())
    if source.get('kind') != 'c1-ooc-source' or not re.fullmatch('[0-9a-f]{40}', source.get('revision', '')):
        raise RuntimeError('Invalid source identity')
    wanted = records(ROOT, SOURCES + ['boards/c1_f2_alex/ooc.py'])
    if wanted != source['files']:
        raise RuntimeError('Source input bytes differ')
    version = subprocess.check_output([a.vivado, '-version'], text=True)
    if not re.search(r'\bvivado\s+v?2025\.2\b', version, re.I):
        raise RuntimeError('Vivado 2025.2 required')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'build.tcl').write_text(tcl(ROOT))
    (a.out / 'version.txt').write_text(version)
    inputs = dict(source=source, source_manifest_sha256=a.source_manifest_sha256,
                  script_sha256=sha(a.out / 'build.tcl'), part=PART, engines=16, slots=4,
                  target_core_mhz=200, max_threads=2, shell_linked=False)
    (a.out / 'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    cmd = [a.vivado, '-mode', 'batch', '-source', str((a.out / 'build.tcl').resolve())]
    with (a.out / 'console.log').open('x') as log:
        result = subprocess.run(cmd, cwd=a.out, stdout=log, stderr=subprocess.STDOUT)
    (a.out / 'exit.json').write_text(json.dumps(dict(returncode=result.returncode))+'\n')
    if wanted != records(ROOT, SOURCES + ['boards/c1_f2_alex/ooc.py']):
        raise RuntimeError('Source changed during synthesis')
    if result.returncode:
        raise RuntimeError('Synthesis failed; retained console.log and exit.json')
    if (a.out / 'complete.txt').read_text().strip() != 'C1_E16_S4_OOC_SYNTH_COMPLETE_NOT_SHELL_QUALIFIED':
        raise RuntimeError('Missing synthesis completion marker')
    outputs = records(a.out, ['post_synth.dcp','synth_utilization.rpt','synth_timing.rpt','synth_drc.rpt',
                             'console.log','complete.txt','build.tcl','version.txt','exit.json'])
    receipt = dict(schema=1, kind='c1-e16-s4-ooc-synthesis', inputs_sha256=sha(a.out/'inputs.json'),
                   outputs=outputs, shell_linked=False, route_qualified=False, hardware_measured=False)
    (a.out / 'synth-complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('C1 E16/S4 OOC synthesis complete; shell integration and routing remain')


if __name__ == '__main__':
    main()
