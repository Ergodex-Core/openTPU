#!/usr/bin/env python3
"""Run a pinned, matched PMP1/PMP0 Gemma RTL comparison on the existing EC2 builder.

This measures simulated cycles, not FPGA timing closure or hardware throughput.
Run the unpumped TMEM correctness regressions first. This script never builds an AFI,
programs hardware, or starts cloud resources. Existing result directories are preserved.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, record):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def checkpoint_manifest(path):
    files = sorted({*path.glob('*.json'), *path.glob('*.safetensors'), *path.glob('*.gguf')})
    if not (path / 'config.json').is_file() or not any(
            p.suffix in ('.safetensors', '.gguf') for p in files):
        raise RuntimeError(f'{path}: expected a config and checkpoint weights')
    manifest = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                digest.update(chunk)
        manifest[file.name] = dict(size=file.stat().st_size, sha256=digest.hexdigest())
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected', required=True, help='Exact clean source commit')
    ap.add_argument('--out', type=Path, required=True, help='New result directory')
    ap.add_argument('--lat', type=int, required=True, help='Same modeled core-cycle latency for both modes')
    ap.add_argument('--layers', choices=['4', '0'], default='4', help='0: whole 26-layer model plus head')
    a = ap.parse_args()
    if sys.platform != 'linux' or not Path('/home/ubuntu').is_dir():
        ap.error('Run on the existing EC2 builder, never on the Mac')
    if a.lat < 0:
        ap.error('--lat must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / 'models/gemma-3-1b-it').resolve(strict=True)
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if rev != a.expected or subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise SystemExit('Expected an exact, clean source revision')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / 'status.json'
    record = dict(revision=rev, started_at=now(), state='running', runs=[],
                  checkpoint=str(checkpoint),
                  scope='Gemma3 int8 weights/head, pos9, RTL cycles; no host time or physical closure')
    save(receipt, record)
    # Remove inherited experimental OTPU overrides so PMP is the only A/B change.
    env = {k: v for k, v in os.environ.items() if not k.startswith('OTPU_')}
    env.update(PYTHONUNBUFFERED='1', OTPU_TARGET='f2-hbm:p4rl8a64', OTPU_NCH='8',
               OTPU_NAXI='1', OTPU_NAXI_XR='3', OTPU_NAXI_BURST='8',
               OTPU_NAXI_IDLE='1', OTPU_NAXI_WAIT='1', OTPU_NAXI_OT='32')
    common = ['--model', str(checkpoint), '--layers', a.layers, '--pos', '9',
              '--wformat', 'int8', '--head-format', 'int8', '--mhz', '200',
              '--portb', '4', '--rowlanes', '--mcols', '8', '--aline', '64',
              '--lat', str(a.lat), '--check']
    launcher = '''
import json,runpy,sys
from opentpu import rtlsim
pmp=int(sys.argv.pop(1))
rtlsim.VFLAGS[2]='4'
rtlsim.BOARD_UARCH['TMEM_PMP']=pmp
original=rtlsim.run
def checked_run(*args, **kwargs):
    p=rtlsim.top_params(args[0], uarch=kwargs.get('uarch'), axi=1)
    required={'S':1,'PORTB':4,'MCOLS':8,'ALINE':64,'LANES':8,
              'TMEM_IMPL':3,'TMEM_NSB':2,'TMEM_SBP':3,'TMEM_RPL':1,
              'TMEM_TDP':1,'TMEM_GST':1,'TMEM_PMP':pmp,
              'NCH':8,'NAXI':1,'NAXI_XR':3,'NAXI_BURST':8,
              'NAXI_IDLE':1,'NAXI_WAIT':1,'NAXI_OT':32}
    for k,v in required.items():
        if p.get(k)!=v: raise RuntimeError(f'{k}: expected {v}, found {p.get(k)}')
    print('WIDEPORT_TMEM_PARAMETERS '+json.dumps(p,sort_keys=True),flush=True)
    return original(*args, **kwargs)
rtlsim.run=checked_run
sys.argv=['tools/perf_qwen.py']+sys.argv[1:]
runpy.run_path('tools/perf_qwen.py',run_name='__main__')
'''
    try:
        record['checkpoint_manifest'] = checkpoint_manifest(checkpoint)
        save(receipt, record)
        for pmp in [1, 0]:
            if checkpoint_manifest(checkpoint) != record['checkpoint_manifest']:
                raise RuntimeError('Checkpoint changed between runs')
            name = f'pmp{pmp}'
            log = out / (name + '.log')
            stat = out / (name + '-tmemstat.log')
            numbers = out / (name + '.jsonl')
            cmd = [sys.executable, '-c', launcher, str(pmp), *common,
                   '--json', str(numbers), '--dump', str(out / (name + '-lens.json'))]
            run_env = dict(env, OTPU_TMEMSTAT_LOG=str(stat))
            run = dict(pmp=pmp, started_at=now(), command=cmd,
                       environment={k: v for k, v in run_env.items() if k.startswith('OTPU_')})
            record['runs'].append(run)
            save(receipt, record)
            with log.open('w') as stream:
                # Drain the pipe to EOF as well as waiting for the model process:
                # multiprocessing's resource tracker can write after its parent exits.
                with subprocess.Popen(cmd, cwd=root, env=run_env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as process:
                    for chunk in iter(lambda: process.stdout.read1(65536), b''):
                        stream.write(chunk.decode(errors='replace'))
                        stream.flush()
                    rc = process.wait()
            data = log.read_bytes()
            run.update(returncode=rc, finished_at=now(),
                       log_sha256=hashlib.sha256(data).hexdigest(),
                       bit_exact_isa=b'bit-exact vs ISA: True' in data)
            if rc or not run['bit_exact_isa']:
                raise RuntimeError(f'{name} failed; inspect {log}')
            params = re.findall(r'^WIDEPORT_TMEM_PARAMETERS (.+)$', data.decode(), re.M)
            rows = [json.loads(line) for line in numbers.read_text().splitlines() if line.strip()]
            if len(params) != 1 or len(rows) != 1 or not stat.is_file() or not stat.stat().st_size:
                raise RuntimeError(f'{name}: incomplete parameter, cycle, or TMEM receipt')
            run.update(parameters=json.loads(params[0]), cycles=rows[0]['cycles'],
                       tmem_counters=stat.read_text().splitlines())
            save(receipt, record)
        if checkpoint_manifest(checkpoint) != record['checkpoint_manifest']:
            raise RuntimeError('Checkpoint changed during the comparison')
        first, second = record['runs']
        p1, p0 = (dict(r['parameters']) for r in (first, second))
        p1.pop('TMEM_PMP')
        p0.pop('TMEM_PMP')
        if p1 != p0:
            raise RuntimeError('A/B changed parameters besides TMEM_PMP')
        record.update(state='passed', pmp0_cycle_change_pct=100*(second['cycles']/first['cycles']-1),
                      pmp0_equal_clock_throughput_factor=first['cycles']/second['cycles'])
    except BaseException as error:
        record.update(state='failed', error=str(error))
        raise
    finally:
        record['finished_at'] = now()
        save(receipt, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
