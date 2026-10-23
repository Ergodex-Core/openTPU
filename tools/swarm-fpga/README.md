# Swarm FPGA

A new, small decoder framework with a Ludwig-inspired configuration interface and seqax-inspired explicit math and state. The implementation is original; neither project is a dependency. AutoGen/AgentChat can orchestrate requests above this layer, but does not schedule FPGA kernels or implement training.

The current model is a **tiny causal Transformer**, suitable for validating execution and gradients. It implements learned position embeddings, multihead causal attention, RMSNorm, SwiGLU, full-parameter SGD, and incremental KV-cache decoding. Both its forward and backward numerical computation have passed on the physical OpenTPU `burst200` F2 image. This is an execution prototype, not a pretrained Gemma or speech-model implementation.

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

The explicit `reference` backend executes NumPy for correctness. Selecting `f2` requires either `image: burst200` for native execution or an explicit worker socket. Neither path falls back to NumPy. The socket adapter remains an optional integration contract; the tested native adapter directly uses OpenTPU's compiler and board driver.

For native execution on the existing F2 host, put the matching OpenTPU source tree and this project on `PYTHONPATH`, set `OTPU_LOCK_WAIT=0`, and use `examples/burst200.yaml`:

```python
rows = np.array([[0, 1, 0, 2]], dtype=np.int64)
with SwarmModel(load_config("examples/burst200.yaml")) as model:
    history = model.train(rows, steps=2)
    result = model.generate([[0, 1]], max_new_tokens=2)
```

The native example expects rows of four token IDs, in the range 0–4. The `with` block releases device ownership. The adapter takes exclusive ownership without waiting and checks the architecture, build `7e42a626`, 200 MHz clock, idle/healthy status and initialized memory before writing. It does not load FPGA images. The tested image was `agfi-0d06cb48544a02106`; receipts record the live identity and dependency hashes. Its exposed memory window is distinct from the physical F2 memory capacity.

`opentpu.py` lowers every arithmetic operation, reduction, FP32 matrix multiplication and gradient accumulation to FPGA VPU/RDOT instructions. Host operations are shape/layout copies, token indexing, zero initialization, parameter initialization and greedy token selection. The bring-up implementation launches separate kernels and transfers intermediates per numerical operation. It bounds each launch to 8,192 tensor words and reserves a 512 KiB staging area inside a 1 MiB allocation. It is deliberately limited to tiny correctness workloads; it does not tile large matrices or keep model state resident on the card.

The v1 socket contract (`swarmfpga.tensor.v1`) sends one newline-delimited JSON request per Unix socket connection. `hello` checks protocol, F2 device identity, operations and dtypes. `run` submits the complete program and feeds, and returns shaped tensors. This simple path uploads weights and downloads updated weights/cache on every call; it does not implement resident weights, efficient DMA, or continuous batching. It is suitable for correctness bring-up, not throughput claims. The default message cap is 16 MiB.

Checkpoints are atomic NPZ files without pickle. They contain configuration, FP32 parameters, SGD step, sequential dataset cursor and a dataset digest. Initialization is seeded; subsequent training is deterministic for the same backend and dataset. There is no dropout, random sampling, momentum or Adam state. Resume checks dataset contents and ordering; checkpoints do not embed the dataset. KV caches belong to one model instance/version and become invalid after training. Programs are cached by batch, sequence length, prior cache length and mode.

## Validation and limits

Tests cover finite-difference primitive and decoder gradients, decreasing independent cross-entropy, exact checkpoint resume, causality, cached versus full-prefix decoding, strict input/configuration checks, and malformed/missing F2 workers. Socket fixtures are explicitly simulated workers and are not hardware results.

Local verification passed **52 tests and 42 subtests**, including native OpenTPU ISA parity and CLI ownership cleanup. The six native ISA tests require the matching upstream source on `PYTHONPATH`; without it, they skip. The physical receipt and reproduction harness are in [`hardware/`](hardware/README.md). The Transformer hardware test uses one layer, two heads, width 4, vocabulary 5 and three-token sequences; it checks every updated parameter, repeated-token embedding accumulation, logits and cache equivalence against the reference. Its two FPGA SGD steps lowered cross-entropy from **1.64581 to 1.57363**. These are correctness measurements on a toy model. The receipt preserves the original source hashes; the CLI cleanup and explicit slot-0 restriction were added afterward and tested locally. The numerical implementation is unchanged.

The framework contains **929 physical lines across `swarmfpga/*.py`**, including blanks, comments, the CLI, autodiff and native adapter. Tests, examples and documentation are separate; NumPy, PyYAML and the existing OpenTPU compiler/driver/RTL are dependencies outside this count. This is a small framework built on that toolchain, not a complete FPGA toolchain under 1,000 lines. Reproduce the count with `wc -l swarmfpga/*.py`.

The MVP has one model family, fixed FP32 arithmetic, SGD and greedy generation. It has no quantization, pretrained checkpoint conversion, multi-device pipeline, resource allocator, session service, preemption, continuous batching or multi-model scheduling. A future Swarm OS should put those services above the numerical runtime and enforce image identity and exclusive device ownership. Separate per-user token latency from aggregate throughput in any performance report; current correctness runs are not performance benchmarks.

## Inspiration and primary sources

- [Ludwig](https://github.com/ludwig-ai/ludwig): declarative feature/model/trainer interface; its Torch/Ray stack is not copied.
- [seqax](https://github.com/MatX-inc/seqax): explicit decoder math and training state; its JAX autodiff/compiler stack is not included.
- [AutoGen GraphFlow](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/graph-flow.html): optional application-level workflow orchestration, separate from device scheduling.
- [Int21 SwarmOS](https://int21.ai/#swarmos): architectural inspiration; this implementation does not claim to reproduce its unpublished internals.
- [AWS F2](https://aws.amazon.com/ec2/instance-types/f2/): physical device specifications do not determine the memory exposed by a particular OpenTPU image.
