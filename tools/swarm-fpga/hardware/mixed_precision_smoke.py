"""Bounded native MXU training proof; run only on an allocated burst200 slot."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from opentpu.host import regs as R
from swarmfpga import Config, SwarmModel
from swarmfpga.backend import _int8
from swarmfpga.opentpu import OpenTPUBackend


def main():
    config = Config(vocab_size=5, context=6, width=4, layers=1, heads=2, ff_width=8,
                    learning_rate=0.05, batch_size=1, sequence_length=3, seed=17,
                    forward_dtype="int8")
    tokens, targets = np.array([[0, 1, 0]], np.int64), np.array([[1, 0, 2]], np.int64)
    root = Path(__file__).parents[1]
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "scope": "tiny mixed-precision training correctness",
               "forward": "INT8 learned projections using QST/QACT/MM; FP32 attention and reductions",
               "backward": "on-card BF16/FP16 rounding emulated with FP32 VPU arithmetic, accumulation and storage",
               "native_16bit_arithmetic": False, "source_sha256": {
                   p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "swarmfpga").glob("*.py"))},
               "modes": {}, "microbenchmark": {}, "tolerance": {"atol": 2e-6, "rtol": 2e-4}}
    with OpenTPUBackend() as backend:
        receipt["device"] = backend.board.info
        for precision in ("bf16", "fp16"):
            c = replace(config, backward_dtype=precision)
            device = SwarmModel(replace(c, backend="f2", image="burst200"), backend)
            reference = SwarmModel(c)
            before = dict(backend.instruction_counts)
            checks, steps = {}, []
            def compare(name, a, b):
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-4, err_msg=name)
                checks[name] = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            compare("initial_logits", device.logits(tokens)[0], reference.logits(tokens)[0])
            for index in range(3):
                expected = reference.train_step(tokens, targets)
                actual = device.train_step(tokens, targets)
                compare(f"loss{index}", actual["loss"], expected["loss"])
                for name, weight in device.weights.items():
                    compare(f"step{index}.{name}", weight, reference.weights[name])
                steps.append(actual)
            if steps[-1]["loss"] >= steps[0]["loss"]:
                raise AssertionError("loss did not decrease")
            full, _ = device.logits(tokens)
            prefix, cache = device.logits(tokens[:, :2])
            tail, _ = device.logits(tokens[:, 2:], cache)
            compare("cached_logits", np.concatenate((prefix, tail), axis=1), full)
            compare("trained_logits", full, reference.logits(tokens)[0])
            counts = {op: count - before.get(op, 0) for op, count in backend.instruction_counts.items()}
            if not all(counts.get(op, 0) > 0 for op in ("MM", "QACT", "QST")):
                raise AssertionError("forward did not use the MXU and both hardware quantizers")
            receipt["modes"][precision] = {"config": c.to_dict(), "steps": steps,
                "max_absolute_errors": checks, "instructions": counts}
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(4, 128)).astype(np.float32), rng.normal(size=(128, 32)).astype(np.float32)
        for op, expected in (("matmul", a @ b), ("matmul_int8", _int8(a, b))):
            samples = []
            for _ in range(3):
                cycles = backend.device_cycles
                started = time.perf_counter()
                result = backend._launch(op, a, b, (4, 32))
                elapsed = time.perf_counter() - started
                np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-4)
                samples.append({"device_cycles": backend.device_cycles - cycles,
                                "wall_seconds_compile_dma_execute_read": elapsed})
            receipt["microbenchmark"][op] = {"shape": [4, 128, 32], "samples": samples}
        receipt["kernel_count"] = backend.kernel_count
        receipt["device_cycles"] = backend.device_cycles
        receipt["instructions"] = backend.instruction_counts
        receipt["operations"] = backend.op_counts
        status = backend.board.board.info()
        if status["running"] or status["status"] & (R.ST_ERROR | R.ST_AXI_ERR | R.ST_WAIT_TO | R.ST_ECC_DED):
            raise AssertionError("device not healthy and idle")
        receipt["final_status"] = status
    receipt["passed"] = True
    print(json.dumps(receipt, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
