"""Opt-in exhaustive significand checks for bounded FP32-emulated rounding.

This evaluates the published bias formulas with OpenTPU's FP32 arithmetic model,
not the FPGA. Actual compiler/ISA lowering has separate tests/test_rounding.py.
Run from tools/swarm-fpga:
  PYTHONPATH=../.. python hardware/check_rounding.py --exhaustive --output rounding.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from opentpu import fp32 as F


def emulate(values, kind):
    if kind == "bf16":
        bias = F.mul(values, 65536.)
        first = F.sub(F.add(values, bias), bias)
        bias = F.mul(first, 65536.)
        return F.sub(F.add(values, bias), bias)
    positive = np.maximum(values, np.float32(0))
    negative = np.minimum(values, np.float32(0))
    qp, qn = positive, negative
    for _ in range(2):
        bp = np.maximum(F.mul(qp, 8192.), np.float32(.5))
        bn = np.minimum(F.mul(qn, 8192.), np.float32(-.5))
        qp = F.sub(F.add(positive, bp), bp)
        qn = F.sub(F.add(negative, bn), bn)
    return F.add(qp, qn)


def reference(values, kind):
    if kind == "fp16":
        return values.astype(np.float16).astype(np.float32)
    flushed = values.copy()
    flushed[np.abs(flushed) < np.finfo(np.float32).tiny] = 0
    bits = flushed.view(np.uint32)
    result = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    result[np.abs(result) < np.finfo(np.float32).tiny] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start_time = time.perf_counter()
    exponents = {"bf16": [1, 127, 236], "fp16": [102, 112, 113, 127, 141, 142]}
    counts = {"bf16": 0, "fp16": 0}
    for start in range(0, 1 << 23, 1 << 19):
        mantissa = np.arange(start, start + (1 << 19), dtype=np.uint32)
        for kind, fields in exponents.items():
            for exponent in fields:
                positive = ((np.uint32(exponent) << 23) | mantissa).view(np.float32)
                if kind == "fp16":
                    positive = positive[positive <= 65504]
                for sign in [1, -1]:
                    values = positive * sign
                    actual, expected = emulate(values, kind), reference(values, kind)
                    np.testing.assert_array_equal(actual, expected)
                    counts[kind] += len(values)
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "FP32 arithmetic-model emulation; exhaustive significands in selected exponent bins; no FPGA",
              "comparisons": counts, "total_comparisons": sum(counts.values()),
              "exponent_fields": exponents, "mantissas_per_exponent": 1 << 23,
              "signs": [1, -1], "bf16_absolute_bound": 2.0 ** 110, "fp16_absolute_bound": 65504,
              "semantics": "RNE values; BF16 flushes FP32 subnormals; zero signs may canonicalize",
              "numpy": np.__version__, "elapsed_seconds": time.perf_counter() - start_time,
              "reproduce": "PYTHONPATH=../.. python hardware/check_rounding.py --exhaustive --output rounding.json"}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
