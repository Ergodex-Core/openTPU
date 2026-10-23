# Physical burst200 correctness test

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

The receipt is historical evidence for the source hashes it contains. Publication adds CLI context-manager cleanup and rejects native slots other than `f2:0`; those changes have local regression coverage. Model math, autodiff, numerical lowering and the hardware harness are unchanged. The current F2 branch is covered by the local ISA tests; no new physical run is claimed for the publication cleanup.

To reproduce on a suitably configured F2 host, from `tools/swarm-fpga`, with the matching OpenTPU branch and dependencies available:

```sh
OTPU_LOCK_WAIT=0 PYTHONPATH=.:../.. python hardware/framework_smoke.py > receipt.json
```

This command writes a bounded payload and runs kernels on slot 0. The native adapter acquires nonblocking exclusive locks, checks build, architecture, clock, idle/health state and existing initialized-memory marker, and refuses a mismatched or occupied device. It never loads an image. Run only when the device is allocated to this test.
