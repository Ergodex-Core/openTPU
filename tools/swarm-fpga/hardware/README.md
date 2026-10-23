# burst200 correctness evidence

## Mixed precision: ISA verified, physical run blocked

[`mixed_precision_smoke.py`](mixed_precision_smoke.py) is the bounded physical test for INT8 learned projections with BF16/FP16-rounded backward computation. Both reduced modes use FP32 VPU arithmetic and storage; neither claims native 16-bit acceleration. The harness checks three SGD steps per mode, every updated parameter, cached decoding, actual QST/QACT/MM counts and a separate 4×128×32 matmul comparison.

The new physical run **did not start**. A fresh check at 2026-10-23 08:58 UTC found both device locks held by a resident worker and a different loaded image. No owner was displaced and no image was loaded. See [`mixed-precision-hardware-status.json`](mixed-precision-hardware-status.json) for the exact blocked target and image identities. This is an ownership/image blocker, not an approval denial or an uncertain execution.

Current local coverage includes 127 tests and 42 subtests, including 15 CLI/ISA end-to-end cases and full-decoder ISA parity for four validated precision policies. The [validation manifest](mixed-precision-isa-validation.json) records commands and source hashes. The [graph-pruning receipt](isa-graph-pruning-validation.json) records fewer ISA launches with bit-exact results for the tiny training configuration; it is not a hardware timing result. [`check_rounding.py`](check_rounding.py) separately checks the two-pass FP32 rounding model; its [receipt](rounding-emulation-verification.json) records 150,978,562 comparisons over selected exponent bins. These are simulator/arithmetic-model results, not physical measurements.

When slot 0 is available with a qualified image, run from `tools/swarm-fpga`:

```sh
OTPU_LOCK_WAIT=0 PYTHONPATH=.:../.. python hardware/mixed_precision_smoke.py > mixed-receipt.json
```

## Historical FP32 physical run

[`framework_smoke.py`](framework_smoke.py) checks the complete tiny Transformer on an already-loaded F2 image. [`framework-receipt.json`](framework-receipt.json) records the numerical comparisons, source and dependency hashes, operation counts, device identity and timings from 2026-10-23 at 07:56 UTC. Infrastructure identifiers, remote paths and unrelated process metadata have been omitted from the public receipt.

The test used vocabulary 5, context 6, width 4, one layer, two heads, feedforward width 8, batch 1 and three input tokens. It checked full inference, two backward/SGD steps and cached decoding against the independent NumPy backend. Repeated input token IDs exercise embedding-gradient accumulation.

| Check | Result |
| --- | --- |
| Loaded image | `agfi-0d06cb48544a02106` |
| Build / clock | `7e42a626` / 200 MHz |
| Numerical tolerance | absolute `2e-6`, relative `2e-4` |
| Loss across two SGD steps | 1.64580953 → 1.57363200 |
| Maximum parameter error | `7.45e-9` |
| Trained-logit maximum error | `1.12e-8` |
| Cached versus full logits | exactly equal |
| FPGA numerical kernels | 634, including 100 matrix multiplications |
| Sum of device cycles | 179,068, equivalent to 0.89534 ms |
| Training-call wall times | 156.21 ms / 137.91 ms |
| Whole correctness harness | 0.61733 s including compilation, DMA and CPU references |

The image stayed unchanged; final hardware was healthy and idle, and device locks were released. No services were stopped or images reloaded. These timings describe a tiny correctness test, not model-serving throughput or useful language quality. Intermediate tensors currently cross PCIe for each numerical operation.

The receipt is historical evidence for the original FP32 source hashes it contains. The package has since added lifecycle cleanup and mixed-precision numerical paths, covered by local tests. The historical receipt must not be attributed to the current mixed-precision code. No new physical run is claimed for those changes.

To reproduce on a suitably configured F2 host, from `tools/swarm-fpga`, with the matching OpenTPU branch and dependencies available:

```sh
OTPU_LOCK_WAIT=0 PYTHONPATH=.:../.. python hardware/framework_smoke.py > receipt.json
```

This command writes a bounded payload and runs kernels on slot 0. The native adapter acquires nonblocking exclusive locks, checks build, architecture, clock, idle/health state and existing initialized-memory marker, and refuses a mismatched or occupied device. It never loads an image. Run only when the device is allocated to this test.
