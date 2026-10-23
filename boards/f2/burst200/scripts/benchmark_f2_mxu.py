#!/usr/bin/env python3
"""Direct MXU weight-stream bandwidth, avoiding LD/ST's 32 B/cycle TMEM limit.

Writes a small test image at logical address zero. Run between model sessions.
Uses the loaded image without changing its RTL, clock, or AFI.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import dataclasses
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OTPU_RUN_DIR', str(ROOT / 'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET', str(ROOT / 'targets/f2-burst200.toml'))
import numpy as np
from opentpu import language as ol, quant as Q
from opentpu.isasim import Machine
from opentpu.host.board import Board, device_config
from opentpu.host import regs as R
from opentpu.llm.qwen3 import _qdesc, _tdesc
from opentpu.host.f2_transport import F2Transport


@ol.jit
def weight_stream(m, n: int, rows: int, width: int):
    x = ol.quantize(ol.load(m.x))
    y = ol.empty([1, rows])
    for i in range(n):
        ol.dot(x, m.w[i * rows:(i + 1) * rows, :], out=y)
    ol.store(m.out, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--smoke', required=True, type=Path)
    a = ap.parse_args()
    if json.loads(a.smoke.read_text()).get('pass') is not True:
        raise SystemExit('Successful hardware smoke required')
    if a.output.exists():
        raise SystemExit('Choose a new result path')
    tr = F2Transport()
    board = Board(transport=tr)
    result = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol': 'Direct one-row MXU dot products; 8 MiB weights, three repetitions; full ISA output comparison',
              'trials': []}
    try:
        info = board.info()
        if info['running']:
            raise RuntimeError('Accelerator is already running')
        result['device'] = info
        cfg = device_config(info)
        for wf in ('int8', 'fp4'):
            for width in (1024, 4096):
                rows = 1024
                rowbytes = Q.row_bytes(width, wf, cfg.D)
                n = (8 << 20) // (rows * rowbytes)
                rng = np.random.default_rng(60 + width)
                w = rng.standard_normal((rows, width)).astype(np.float32)
                q, sc = Q.quantize_mxu(w, wf, cfg.D)
                q = np.ascontiguousarray(q).view(np.uint8).ravel()
                sc = np.ascontiguousarray(sc).view(np.uint8).ravel()
                sa = n * q.size
                xa = sa + n * sc.size
                out = xa + width * 4
                img = np.zeros(out + rows * 4, np.uint8)
                img[:sa] = np.tile(q, n)
                img[sa:xa] = np.tile(sc, n)
                img[xa:out] = rng.standard_normal(width).astype(np.float32).view(np.uint8)
                desc = SimpleNamespace(w=_qdesc(0, sa, n * rows, width, cfg.D, wf),
                                       x=_tdesc(xa, (1, width)), out=_tdesc(out, (1, rows)))
                prog = weight_stream.trace(cfg, 0, {'m': desc, 'n': n, 'rows': rows,
                                                   'width': width}).finish()
                simcfg = dataclasses.replace(cfg, DRAM_BYTES=(len(img) + 4095) // 4096 * 4096)
                sim = Machine(simcfg, [prog], [img]).run()
                expected = sim.slices[0].dram[out:out + rows * 4].copy()
                board.write(0, img)
                board.load((len(img) + 4095) // 4096 * 4096, prog)
                for rep in range(3):
                    before = board.snapshot()
                    stats = board.run(timeout=30)
                    after = board.snapshot()
                    got = board.read(out, rows * 4)
                    np.testing.assert_array_equal(got, expected)
                    status = tr.reg_read(R.R_STATUS)
                    if status & (R.ST_AXI_ERR | R.ST_ERROR):
                        raise RuntimeError(f'Hardware error {status:#x}')
                    delta = {k: int(after[k] - before[k]) for k in before if k != 'snaps'}
                    sec = stats['cycles'] / (info['core_khz'] * 1000)
                    row = {'format': wf, 'width': width, 'rows': rows, 'matrices': n,
                           'repetition': rep, 'weight_and_scale_bytes': sa + n * sc.size,
                           'cycles': stats['cycles'], 'counters': delta,
                           'counter_GBps': (delta['DRAM_RD'] + delta['DRAM_WR']) * 64 / sec / 1e9,
                           'useful_weight_GBps': (sa + n * sc.size) / sec / 1e9,
                           'output_bit_exact': True, 'run': stats}
                    result['trials'].append(row)
                    print(wf, width, rep, f"{row['counter_GBps']:.4f} GB/s",
                          f"MAC/busy={delta['MXU_MAC']}/{delta['MXU_BUSY']}", flush=True)
        result['pass'] = True
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        board.close()
        tr.close()


if __name__ == '__main__':
    main()
