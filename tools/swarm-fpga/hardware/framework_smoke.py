"""Bounded end-to-end Transformer correctness test on the already-loaded image.

Run only on the F2 host with the matching OpenTPU source on PYTHONPATH. The native
backend exclusively owns the card; this script never loads an image or installs
anything. Host reference computations are independent comparison work only.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from swarmfpga import Config, SwarmModel
from swarmfpga.opentpu import OpenTPUBackend
from opentpu.host import regs as R


def main():
    config = Config(vocab_size=5, context=6, width=4, layers=1, heads=2,
                    ff_width=8, learning_rate=0.05, batch_size=1,
                    sequence_length=3, seed=17, steps=2)
    tokens = np.array([[0, 1, 0]], dtype=np.int64)  # Repeated ID tests embedding accumulation.
    targets = np.array([[1, 0, 2]], dtype=np.int64)
    reference = SwarmModel(config)
    receipt = {"scope": "tiny causal Transformer full inference, gradients, SGD and cached decoding",
               "utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
               "python": platform.python_version(), "numpy": np.__version__,
               "host_role": "layout, data preparation, comparisons, and greedy selection",
               "tolerance": {"atol": 2e-6, "rtol": 2e-4}, "checks": {}, "steps": []}
    receipt["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((Path(__file__).parents[1] / "swarmfpga").glob("*.py"))}
    def compare(name, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-4, err_msg=name)
        receipt["checks"][name] = {"max_absolute_error": float(np.max(np.abs(actual - expected))),
                                  "shape": list(np.shape(actual))}
    started = time.perf_counter()
    with OpenTPUBackend(image="burst200") as backend:
        device = SwarmModel(replace(config, backend="f2", image="burst200"), backend=backend)
        receipt["device"] = backend.board.info
        actual, _ = device.logits(tokens)
        expected, _ = reference.logits(tokens)
        compare("initial_logits", actual, expected)
        for step in range(2):
            expected_metrics = reference.train_step(tokens, targets)
            actual_metrics = device.train_step(tokens, targets)
            compare(f"loss{step}", np.array(actual_metrics["loss"]), np.array(expected_metrics["loss"]))
            for name in device.weights:
                compare(f"step{step}.{name}", device.weights[name], reference.weights[name])
            receipt["steps"].append({"f2": actual_metrics, "reference": expected_metrics})
        if receipt["steps"][1]["f2"]["loss"] >= receipt["steps"][0]["f2"]["loss"]:
            raise AssertionError("training did not lower loss")
        full, _ = device.logits(tokens)
        compare("trained_logits", full, reference.logits(tokens)[0])
        prefix, cache = device.logits(tokens[:, :2])
        tail, _ = device.logits(tokens[:, 2:], cache)
        compare("cached_logits", np.concatenate((prefix, tail), axis=1), full)
        receipt["last_kernel_stats"] = backend.board.last[1]
        receipt["kernel_count"] = backend.kernel_count
        receipt["device_cycles"] = backend.device_cycles
        receipt["numerical_operation_counts"] = backend.op_counts
        receipt["final_status"] = backend.board.board.info()
        status = receipt["final_status"]
        if status["running"] or status["status"] & (R.ST_ERROR | R.ST_AXI_ERR | R.ST_WAIT_TO | R.ST_ECC_DED):
            raise AssertionError("device did not finish healthy and idle")
        if any(status[key] != receipt["device"][key] for key in ("build_id", "core_khz", "cfg")):
            raise AssertionError("device identity changed")
    receipt["wall_seconds_including_compile_dma_reference"] = time.perf_counter() - started
    receipt["hardware_image_changed"] = False
    receipt["passed"] = True
    print(json.dumps(receipt, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))


if __name__ == "__main__":
    main()
