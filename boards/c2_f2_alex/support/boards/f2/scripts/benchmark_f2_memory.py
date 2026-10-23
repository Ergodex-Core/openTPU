#!/usr/bin/env python3
"""Measure and verify CPU-to-HBM transfers before uploading model weights.

Writes test data at 64 MiB in each native channel window. Requires smoke PASS.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OTPU_RUN_DIR', str(ROOT / 'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET', str(ROOT / 'targets/f2.toml'))
import numpy as np
from opentpu.host.f2_transport import F2Transport
from opentpu.host.board import Board


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if json.loads(args.smoke.read_text()).get('pass') is not True:
        raise SystemExit('A successful hardware smoke report is required')
    tr = F2Transport()
    board = None
    try:
        board = Board(transport=tr)
        rng = np.random.default_rng(0xF247)
        results = []
        for channel in (0, 1):
            for size in (151936 * 4, 4 << 20):
                data = rng.integers(0, 256, size, dtype=np.uint8)
                writes, reads = [], []
                for _ in range(3):
                    start = time.perf_counter()
                    tr.mem_write(channel, 64 << 20, data)
                    writes.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    got = tr.mem_read(channel, 64 << 20, size)
                    reads.append(time.perf_counter() - start)
                    np.testing.assert_array_equal(got, data)
                results.append(dict(channel=channel, bytes=size, write_seconds=writes,
                                    read_seconds=reads,
                                    median_write_MBps=size/statistics.median(writes)/1e6,
                                    median_read_MBps=size/statistics.median(reads)/1e6))
        result = dict(pass_=True, device=board.info(), write_combining=tr.mem_wc is not None,
                      read_mode=tr.read_mode, read_threads=tr.read_threads, transfers=results)
        result['pass'] = result.pop('pass_')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(results, indent=2))
    finally:
        if board is not None:
            board.close()
        tr.close()


if __name__ == '__main__':
    main()
