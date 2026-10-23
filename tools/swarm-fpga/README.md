# Swarm FPGA

A new, small decoder framework with a Ludwig-inspired configuration interface and seqax-inspired explicit math and state. The implementation is original; neither project is a dependency. AutoGen/AgentChat can orchestrate requests above this layer, but does not schedule FPGA kernels or implement training.

The current model is a **tiny causal Transformer**, suitable for validating execution and gradients. It implements learned position embeddings, multihead causal attention, RMSNorm, SwiGLU, full-parameter SGD, and incremental KV-cache decoding. Its original FP32 forward/backward path passed on physical `burst200`. The new INT8-forward and rounded BF16/FP16-backward modes pass ISA tests; their physical test is blocked by an occupied slot and changed image. This is an execution prototype, not a pretrained Gemma or speech-model implementation.

## Interface

```python
import numpy as np
from swarmfpga import SwarmModel, load_config

model = SwarmModel(load_config("examples/tiny.yaml"))
# Pretokenized rows: each contains sequence_length + 1 token IDs.
rows = np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0],
                 [1, 2, 3, 0, 1, 2, 3, 0, 1]], dtype=np.int64)
history = model.train(rows, steps=20)
model.save("state.npz")
model = SwarmModel.load("state.npz")
tokens = model.generate([[0, 1]], max_new_tokens=8)
```

Configuration uses `input_features`, `output_features`, `model`, `trainer`, and `backend`; unknown fields fail early. The feature schema intentionally supports only token IDs. There is no generic feature registry, tokenizer, dataset downloader, or hidden backend selection.

From this directory (`tools/swarm-fpga`), with Python 3.10+, NumPy and PyYAML available:

```sh
python -m swarmfpga inspect examples/tiny.yaml
python -m swarmfpga train examples/tiny.yaml rows.npy state.npz --steps 20
python -m swarmfpga train examples/tiny.yaml rows.npy state.npz --resume state.npz --steps 20
python -m swarmfpga generate state.npz --tokens 0,1 --max-new-tokens 8
PYTHONPATH=../.. python -m pytest -q
```

No packages are installed by these commands. A fresh environment can install the declared dependencies through its normal package-management process. Tests also require pytest. `PYTHONPATH=../..` uses this repository's OpenTPU source for ISA tests. Native execution requires the F2 support on `model-infer-f2`; this package has its own project configuration and is not included by the root OpenTPU package installation.

## Execution and state

`model.py` constructs a shaped tensor program. `graph.py` differentiates it symbolically; forward computation, cross-entropy, gradients, gradient accumulation, and SGD parameter updates are all backend operations. The host initializes parameters, prepares token data/one-hot labels, manages state and selects greedy output tokens. First-order gradients only; `max` stops gradients and is used for numerically stable softmax shifts.

The explicit `reference` backend executes NumPy for correctness. Selecting `f2` requires `image: burst200` and directly uses OpenTPU's compiler and board driver. It never falls back to CPU arithmetic. The earlier prototype socket transport was removed to keep the native framework under 1,000 lines. FP32 checkpoints containing an empty legacy socket field still load.

For native execution on the existing F2 host, put the matching OpenTPU source tree and this project on `PYTHONPATH`, set `OTPU_LOCK_WAIT=0`, and use `examples/burst200.yaml`:

```python
rows = np.array([[0, 1, 0, 2]], dtype=np.int64)
with SwarmModel(load_config("examples/burst200.yaml")) as model:
    history = model.train(rows, steps=2)
    result = model.generate([[0, 1]], max_new_tokens=2)
```

The native example expects rows of four token IDs, in the range 0–4. The `with` block releases device ownership. The adapter takes exclusive ownership without waiting and checks the architecture, build `7e42a626`, 200 MHz clock, idle/healthy status and initialized memory before writing. It does not load FPGA images. The tested image was `agfi-0d06cb48544a02106`; receipts record the live identity and dependency hashes. Its exposed memory window is distinct from the physical F2 memory capacity.

`opentpu.py` lowers FP32 math to VPU/RDOT instructions and INT8 learned projections to QST/QACT/MM. Host operations are shape/layout copies, padding, token indexing, zero initialization, parameter initialization and greedy token selection. The bring-up implementation launches separate kernels and transfers intermediates per numerical operation. It bounds each launch to 8,192 input/output tensor words, including K padding to 128, and reserves a 512 KiB staging area inside a 1 MiB allocation. It is deliberately limited to tiny correctness workloads; it does not tile large matrices or keep model state resident on the card.

## Mixed precision

`examples/burst200.yaml` selects this policy:

```yaml
trainer:
  forward_dtype: int8
  backward_dtype: bf16  # or fp16 / fp32
```

INT8 forward applies to the learned QKV, attention output, gate/up/down and LM-head projections. Every native launch quantizes both updated weights and activations on the FPGA using QST and QACT, then uses the MXU's MM instruction. Weight requantization is not a hidden CPU operation. Attention QK/PV matmuls, normalization, softmax and loss stay FP32. In particular, quantizing V across all sequence positions would allow future values to change past quantization scales; keeping attention FP32 preserves causality and cached/full-prefix equivalence.

The INT8 matmul uses an explicit original-operand surrogate gradient: for `Y = INT8_MM(X,W)` and upstream `G`, use `dX = G @ W.T` and `dW = X.T @ G`, then apply the selected backward precision. This is not a derivative of discrete rounding or an exact fake-quantization derivative using dequantized operands. The tests verify this declared surrogate directly, alongside training loss reduction.

| Backward setting | Current implementation |
| --- | --- |
| `fp32` | FP32 operands, arithmetic, accumulation and results |
| `bf16` | BF16-rounded inputs/results at each backward graph operation; FP32 arithmetic, accumulation and storage |
| `fp16` | FP16-rounded inputs/results at each backward graph operation; FP32 arithmetic, accumulation and storage |

Burst200 has no native BF16/FP16 tensor arithmetic. The two reduced-precision modes emulate round-to-nearest-even entirely on its FP32 VPU using two bias passes; the host only checks bounds. They provide numerical experiments, **not native 16-bit throughput or packed-memory savings**, and add kernel launches. BF16 emulation accepts finite `abs(x) <= 2**110` and flushes FP32 subnormals; FP16 accepts finite `abs(x) <= 65504` and retains FP16 subnormals. Zero signs may canonicalize. Out-of-range intermediates fail instead of saturating; automatic loss scaling is not implemented. Master weights and SGD updates remain FP32, outside backward rounding.

The reference INT8 path uses this repository's `opentpu.quant` oracle, including MXU block-scale/drop rules. Put the repository root on `PYTHONPATH` for those runs. FP32 reference execution remains usable without OpenTPU installed.

Checkpoints are atomic NPZ files without pickle. They contain configuration, FP32 parameters, SGD step, sequential dataset cursor and a dataset digest. Initialization is seeded; subsequent training is deterministic for the same backend and dataset. There is no dropout, random sampling, momentum or Adam state. Resume checks dataset contents and ordering; checkpoints do not embed the dataset. KV caches belong to one model instance/version and become invalid after training. Programs are cached by batch, sequence length, prior cache length and mode.

## Validation and limits

Tests cover finite-difference FP32 primitive/decoder gradients, the declared INT8 surrogate, decreasing cross-entropy, exact precision-aware checkpoint resume, mixed-mode causality and cache equivalence, strict input/configuration checks, unsupported slots, resource cleanup and no F2-to-CPU fallback.

Local verification passed **98 tests and 42 subtests**. Native ISA tests require the matching OpenTPU source on `PYTHONPATH`; without it, they skip. The four validated policies (`fp32/fp32`, `int8/fp32`, `int8/bf16`, `int8/fp16`) have full-decoder forward/backward/cache comparisons. Instruction tests prove QST, QACT and MM execute, and reject CPU quantization on the native path. An optional arithmetic-model check passed **150,978,562 rounding comparisons** across exhaustive significands in selected exponent bins, not every possible FP32 input.

[`hardware/`](hardware/README.md) contains the original FP32 physical receipt, new mixed-mode harness and current blocked status. The historical two-step FPGA loss **1.64581 → 1.57363** applies only to the original FP32 source hashes. It is not evidence that the new mixed modes ran physically, nor a throughput benchmark.

The framework contains **928 physical lines across `swarmfpga/*.py`**, including blanks, comments, the CLI, autodiff and native adapter. Tests, examples and documentation are separate; NumPy, PyYAML and the existing OpenTPU compiler/driver/RTL are dependencies outside this count. This is a small framework built on that toolchain, not a complete FPGA toolchain under 1,000 lines. Reproduce the count with `wc -l swarmfpga/*.py`.

The MVP has one model family, bounded mixed-precision modes, SGD and greedy generation. It has no pretrained checkpoint conversion, native 16-bit floating MACs, multi-device pipeline, resource allocator, session service, preemption, continuous batching or multi-model scheduling. A future Swarm OS should put those services above the numerical runtime and enforce image identity and exclusive device ownership. Separate per-user token latency from aggregate throughput in any performance report; current correctness runs are not performance benchmarks.

## Inspiration and primary sources

- [Ludwig](https://github.com/ludwig-ai/ludwig): declarative feature/model/trainer interface; its Torch/Ray stack is not copied.
- [seqax](https://github.com/MatX-inc/seqax): explicit decoder math and training state; its JAX autodiff/compiler stack is not included.
- [AutoGen GraphFlow](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/graph-flow.html): optional application-level workflow orchestration, separate from device scheduling.
- [Int21 SwarmOS](https://int21.ai/#swarmos): architectural inspiration; this implementation does not claim to reproduce its unpublished internals.
- [AWS F2](https://aws.amazon.com/ec2/instance-types/f2/): physical device specifications do not determine the memory exposed by a particular OpenTPU image.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): [rooflines](https://jax-ml.github.io/scaling-book/roofline/), [TPU architecture](https://jax-ml.github.io/scaling-book/tpus/) and [training](https://jax-ml.github.io/scaling-book/training/) motivate measuring bytes moved, reuse and both backward matmuls. Google's native BF16 MXU is not a capability of this OpenTPU image. FP32 carriers here still occupy four bytes; device residency, less padding and reuse of quantized operands are the next performance work.
