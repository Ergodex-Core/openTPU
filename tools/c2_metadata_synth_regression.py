#!/usr/bin/env python3
"""C2 metadata FIFO synthesis regression (Vivado 2025.2, explicitly invoked).

Run after sourcing Vivado settings64.sh:
  python tools/c2_metadata_synth_regression.py --out build/metadata-synth

The same command with --legacy-struct is a negative control: on Vivado2025.2
it must fail after synthesis although RTL simulation passes. --emit-only creates
source-linked inputs without requiring Vivado. Nothing runs on an FPGA.

This reduces the production metadata path, not the complete MXU. Arithmetic is
replaced by observed payload inputs; the testbench drives the KB1 command
schedule. The faithful wrapper ties acc/old_valid0 and leaves metadata addresses
unused, matching kernel observability. It retains packed result_t dimensions,
two command banks, delays25/12, FIFO pointers, backpressure, mask/tag retirement,
and count/done. 32 commands independently check44 rows and result-write targets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MXU = 'rtl/mxu/otpu_c2_mxu.sv'
DELAY = 'rtl/vpu/otpu_fpipe.sv'
TEMPLATE = 'sim/vivado/c2_metadata_repro.sv.in'
TB = 'sim/vivado/tb_c2_metadata.sv'
PART = 'xcvu47p-fsvh2892-2-e'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def one(pattern, text, name):
    hits = re.findall(pattern, text, re.S)
    if len(hits) != 1:
        raise ValueError(f'Expected one production {name}; found {len(hits)}')
    return hits[0]


def extract(mxu, delay):
    """Fail on source drift we cannot represent; never silently use old copies."""
    normalized = re.sub(r'\s+', '', mxu)
    if ('parameterintD=128,C=4,DEPTH=256,RF=8' not in normalized or
            'localparamintCLS=D/8,LROW=CLS+C+5,EP=12;' not in normalized):
        raise ValueError('Regression requires the production C4/RF8/25+12 contract')
    types = one(r'(  typedef struct packed \{.*?\} meta_t;\s*typedef struct packed \{.*?\} result_t;)', mxu, 'metadata/result types')
    declaration = one(r'logic \[FW-1:0\] rh, rt;\s*(.*?)\s*wire accept=', mxu, 'FIFO declaration')
    if not re.search(r'\bresults\s*\[RF\]', declaration) or not re.search(r'\bhead\b', declaration):
        raise ValueError('Production FIFO declaration is not represented')
    return {
        'TYPES': types,
        'FIFO_DECLARATION': declaration,
        'FIFO_READ': one(r'(assign head\s*=[^;]+;)', mxu, 'FIFO read'),
        'FIFO_WRITE': one(r'(\bresults\[rt\]\s*<=[^;]+;)', mxu, 'FIFO write'),
        'META_DELAY': one(r'((?:\(\*[^;]*?\*\)\s*)?otpu_delay\s*#\([^;]*?\)\s+u_meta\s*\([^;]*?;)', mxu, 'row metadata delay'),
        'EMETA_DELAY': one(r'((?:\(\*[^;]*?\*\)\s*)?otpu_delay\s*#\([^;]*?\)\s+u_emeta\s*\([^;]*?;)', mxu, 'epilogue metadata delay'),
        'DELAY_MODULE': one(r'(module otpu_delay\b.*?\nendmodule)', delay, 'delay module'),
    }


def emit(out, *, root=ROOT, legacy=False):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Output directory must be fresh')
    raw = {n: (root / n).read_bytes() for n in (MXU, DELAY, TEMPLATE, TB)}
    pieces = extract(raw[MXU].decode(), raw[DELAY].decode())
    if legacy:
        pieces.update(FIFO_DECLARATION='result_t results[RF], head;',
                      FIFO_READ='assign head=results[rh];',
                      FIFO_WRITE="results[rt]<='{meta:tail_meta,x:integers,sw:tail_sw,sa:tail_sa};")
    design = raw[TEMPLATE].decode()
    for key, value in pieces.items():
        if key == 'DELAY_MODULE':
            continue
        if design.count('@' + key + '@') != 1:
            raise ValueError('Missing or repeated template insertion: ' + key)
        design = design.replace('@' + key + '@', value)
    if re.search(r'@[A-Z_]+@', design):
        raise ValueError('Unfilled source template')
    out.mkdir(parents=True)
    (out / 'repro.sv').write_text(design)
    (out / 'delay.sv').write_text(pieces['DELAY_MODULE'] + '\n')
    (out / 'tb.sv').write_bytes(raw[TB])
    (out / 'synth.tcl').write_text(f'''if {{[version -short] ne "2025.2"}} {{error "Vivado 2025.2 required"}}
set_param general.maxThreads 4
read_verilog -sv delay.sv
read_verilog -sv repro.sv
synth_design -mode out_of_context -top metadata_faithful -part {PART} -flatten_hierarchy rebuilt -keep_equivalent_registers
write_verilog -mode funcsim synthesized.v
write_checkpoint synthesized.dcp
report_utilization -hierarchical -file utilization.rpt
puts "METADATA_SYNTHESIS_COMPLETE"
close_design
''')
    record = {'kind': 'c2-metadata-synthesis-regression-inputs', 'variant': 'legacy-struct' if legacy else 'production',
              'sources': {n: {'sha256': digest(b), 'bytes': len(b)} for n, b in raw.items()},
              'excerpts': pieces, 'part': PART, 'hardware_actions': False}
    record['generated'] = {p.name: {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
                           for p in out.iterdir() if p.is_file()}
    (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def check_simulation(text):
    rows = re.findall(r'^METADATA_FAITHFUL_POST_SYNTH_RESULT commands=32 count=(\d+) received=(\d+) failures=(\d+) done=(\d+)\s*$', text, re.M)
    if rows != [('44', '32', '0', '1')] or re.search(r'^\s*(?:Fatal:|FATAL:|ERROR:|ERROR\s+\[)', text, re.M):
        raise ValueError('Post-synthesis metadata/count/done regression failed; inspect simulate.log')


def run(out, vivado, *, timeout=540):
    """Only generated functional netlist enters XSim, never original design RTL."""
    out = Path(out).resolve()
    binary = Path(vivado).resolve()
    bindir = binary.parent
    glbl = bindir.parent / 'data/verilog/src/glbl.v'
    if not glbl.is_file():
        raise ValueError('Cannot find glbl.v beside the selected Vivado installation')
    inputs = json.loads((out / 'inputs.json').read_text())
    started = time.monotonic()
    phases, failure = [], None
    commands = [
        ('synth', [str(binary), '-mode', 'batch', '-source', 'synth.tcl']),
        ('compile', [str(bindir/'xvlog'), '--sv', 'synthesized.v', 'tb.sv', str(glbl)]),
        ('elaborate', [str(bindir/'xelab'), '-mt', '4', '-L', 'unisims_ver', '-L', 'secureip', 'tb', 'glbl', '-s', 'metadata_gate_sim']),
        ('simulate', [str(bindir/'xsim'), 'metadata_gate_sim', '-runall']),
    ]
    try:
        for phase, argv in commands:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Synthesis regression timeout')
            with (out / (phase + '.log')).open('wb') as log:
                result = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            phases.append({'phase': phase, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError(f'{phase} failed ({result.returncode}); inspect {phase}.log')
        check_simulation((out / 'simulate.log').read_text())
    except Exception as exc:
        failure = str(exc)
    for name, row in inputs['generated'].items():
        if digest((out / name).read_bytes()) != row['sha256']:
            failure = 'Regression input changed: ' + name
    receipt = {'kind': 'c2-metadata-synthesis-regression', 'passed': failure is None,
               'variant': inputs['variant'], 'failure': failure, 'phases': phases,
               'inputs_sha256': digest((out/'inputs.json').read_bytes()), 'hardware_actions': False,
               'files': {p.name: {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
                         for p in out.iterdir() if p.is_file()}}
    (out/'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if failure:
        raise RuntimeError(failure)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vivado', help='Vivado executable (defaults to PATH; source settings64.sh first)')
    parser.add_argument('--emit-only', action='store_true', help='Generate inputs without synthesis or simulation')
    parser.add_argument('--legacy-struct', action='store_true', help='Reproduce the old broken FIFO representation; expected failure on2025.2')
    args = parser.parse_args()
    vivado = args.vivado or shutil.which('vivado')
    if not args.emit_only and not vivado:
        parser.error('Vivado is unavailable; this is an explicit optional regression, or use --emit-only')
    emit(args.out, legacy=args.legacy_struct)
    if not args.emit_only:
        run(args.out, vivado)
        print('PASS: synthesized C2 metadata,32 commands,44 rows,completion')
    else:
        print(args.out.resolve())


if __name__ == '__main__':
    main()
