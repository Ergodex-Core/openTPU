# Learning TPU systems with the JAX Scaling Book

Start with [How to Scale Your Model](https://jax-ml.github.io/scaling-book/),
by Jacob Austin and collaborators at Google DeepMind. This is an original OpenTPU
reading guide, not a copy of the book or evidence of FPGA performance.

The reference revision is
[`173fb0851d7e654872f0327aff2e17607cf24106`](https://github.com/jax-ml/scaling-book/tree/173fb0851d7e654872f0327aff2e17607cf24106)
(upstream commit date: 2026-10-19; reviewed/downloaded: 2026-10-23 UTC).
The public website can change after this revision.

## Reading route

Read the introduction, then work through the following groups. Write down shapes,
data types, bytes, operations, and the resource that limits each example before
looking at its answer.

| Read | Question to answer | OpenTPU connection |
| --- | --- | --- |
| [1: Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | Which lower bound dominates: arithmetic, memory traffic, or communication? | [Lens](../lens.md), [dataflow ledger](../dataflow.md) |
| [2: TPU architecture](https://jax-ml.github.io/scaling-book/tpus/) | How do the matrix unit, vector unit, local memory, HBM, and interconnect feed one another? | [ISA](../isa.md), [matrix unit](../mx_mxu.md), [targets](../targets.md) |
| [3: Sharding](https://jax-ml.github.io/scaling-book/sharding/) | Which device owns each matrix dimension, and where must partial results meet? | [Compiler](../compiler.md), [dataflow](../dataflow.md) |
| [4: Transformer arithmetic](https://jax-ml.github.io/scaling-book/transformers/) | How much work and storage does each layer require? | [Quantization](../quant.md), [formats](../formats.md) |
| [5: Training](https://jax-ml.github.io/scaling-book/training/), [6: worked training example](https://jax-ml.github.io/scaling-book/applied-training/) | How do gradients, optimizer state, saved activations, and collectives change the budget? | Use as a design exercise; this guide establishes no OpenTPU training implementation. |
| [7: Inference](https://jax-ml.github.io/scaling-book/inference/), [8: worked serving example](https://jax-ml.github.io/scaling-book/applied-inference/) | How do prefill, decode, batching, and KV state change latency and throughput? | [Prefill](../prefill.md), [serving](../serve.md), [offload](../offload.md) |
| [9: Profiling](https://jax-ml.github.io/scaling-book/profiling/), [10: JAX](https://jax-ml.github.io/scaling-book/jax-stuff/) | Does a trace support the proposed bottleneck and array layout? | [Observability](../observability.md), [Lens](../lens.md) |

Finish with [11: further reading](https://jax-ml.github.io/scaling-book/conclusion/).
[12: GPUs](https://jax-ml.github.io/scaling-book/gpus/) is a useful comparison after
the TPU path. Programming examples are reading material; downloading the book
does not run them or install JAX, Ruby, Jekyll, or any toolchain.

## A worksheet for each workload

For a phase that performs `O` operations and moves `D` bytes through a particular
memory interface, define arithmetic intensity as `I = O / D`. A first estimate is
`t_compute = O / P` and `t_memory = D / B`, where `P` is the relevant arithmetic
rate and `B` is sustainable bandwidth. With ideal overlap, time is at least their
maximum. Serialized phases add their times; overlap, launch overhead, dependencies,
padding, and utilization must be checked in a trace. Capacity is a separate
constraint: a fast schedule is unusable if its live tensors do not fit.

For `A[M,K] @ W[K,N]`, the usual dense convention counts approximately `2*M*K*N`
operations (a multiply and an add separately). State whether the hardware rate is
MAC/s, integer OP/s, or floating-point FLOP/s, and which precision it supports.
Count actual transfers, including scale blocks, padding, rereads, and outputs;
logical tensor size alone can understate bus traffic. Build separate budgets for
external memory, on-chip memory, the host link, and any collective link.

During training, budget parameters, gradients, optimizer state, activations, and
temporary buffers independently. Compare data parallelism, FSDP, tensor
parallelism, and pipeline parallelism using both memory saved and communication
introduced. Rematerialization exchanges stored activations for repeated work;
gradient accumulation changes scheduling and effective batch size. These are
learning exercises, not claims that OpenTPU currently supports those schedules.

During inference, separate prefill from decode. Larger prefill matrices can reuse
weights across prompt positions; a small decode batch often exposes weight
bandwidth and host overhead. KV reads grow with context, while live KV storage
grows with both context and concurrent requests. For a simple dense attention
model, estimate KV storage as `2*L*batch*context*kv_heads*head_dim*bytes_per_element`,
then adjust for the actual model, quantization metadata, sharing, and padding.
Compare time to first token, time per output token, and aggregate throughput under
the same request mix. A throughput improvement need not reduce request latency.

For sharding, draw the layout before and after each operation. Splitting an output
dimension and splitting a reduction dimension need different communication.
Record collective payload, participant count, topology, startup latency, and
contention; dividing bytes by a link's advertised rate is only an optimistic
bound. A TPU mesh and OpenTPU slices are not interchangeable interconnects.

## Applying the lessons to OpenTPU and F2

The OpenTPU baseline for this guide is
[`f237dab2`](https://github.com/Ergodex-Core/openTPU/tree/f237dab2f196855a59a083b1454786df8f01291f).
Its README reports YPCB-00338 card measurements; those are **not F2 measurements**.
This guide contains no new hardware run, F2 result, or claim of TPU compatibility.
F2 work can live on separate branches; record the exact F2 branch and revision
before comparing a result with this baseline.

For a future F2 experiment, fill in the following from that implementation and
its own evidence:

| Item | Evidence needed |
| --- | --- |
| Arithmetic ceiling | Matrix/vector configuration, precision, operations per cycle, implemented clock, and utilization |
| Memory ceiling | Addressable capacity and measured bandwidth through the active memory path, including channel mapping and access pattern |
| Host overhead | Transfers, program loading, dispatch, synchronization, sampling, and device versus wall time |
| Communication | Slice ownership, fan-out/fan-in layout, actual link topology, payload, and latency |
| Workload identity | Model/checkpoint, weight and KV formats, batch, context, prompt/output lengths, and correctness criterion |
| Result provenance | Source SHA, bitstream/AFI identity where applicable, target configuration, raw counters/trace, and whether measured, simulated, or estimated |

Use [Lens](../lens.md) to distinguish nominal ISA work from RTL cycles and hardware
counters. In particular, its ISA timeline is analytic, not measured timing.
Use the [dataflow ledger](../dataflow.md) to explain bytes and the
[target description](../targets.md) to identify configuration-dependent limits.
Never substitute the book's TPU HBM bandwidth, interconnect rates, clock, or
floating-point throughput into an FPGA result. Likewise, a quantized FPGA MAC
rate is not a BF16 TPU FLOP rate.

Three useful exercises: (1) predict the limiting resource for one prefill tile
and one decode step; (2) sweep batch/context on paper to find when KV traffic or
capacity dominates; (3) compare output-sharded and reduction-sharded matmuls,
including the communication needed before the next layer. Label the answers
**estimates** until a matching implementation and trace verify them.

## Download for local study

Full public redistribution has unresolved third-party figure permissions. See
the [license review](jax-scaling-book-licensing.md); the book and images are not
vendored here. The original download tool retrieves the pinned public upstream
archive for local reading and copies only documentation and referenced PNG/GIF
assets. It preserves upstream source, attribution, and the upstream license.

From the repository root, choose an output location **outside every Git checkout**:

```sh
python3 tools/learning/download_scaling_book.py --output ../jax-scaling-book-173fb085
python3 tools/learning/download_scaling_book.py --output ../jax-scaling-book-173fb085 --verify
```

The parent directory must exist. Open the generated `index.md` in a Markdown
reader. `readable/` has local chapter/image paths and expanded figure/answer
markup; `source/` preserves exact upstream bytes. TeX remains TeX, citation keys
link to the live chapter, and three interactive plots remain online links. This
is an offline **source reading bundle**, not a fully rendered mirror of the site.
The index lists these limitations. The tool does not run downloaded code.

An existing official archive can be used without network access:

```sh
python3 tools/learning/download_scaling_book.py --archive /path/to/book.tar.gz --output ../jax-scaling-book-173fb085
```

Python needs a working TLS certificate store for the network download. On a Mac
whose standalone Python cannot find its CA bundle, use the trusted system bundle
for that command (without disabling verification or changing system settings):

```sh
SSL_CERT_FILE=/etc/ssl/cert.pem python3 tools/learning/download_scaling_book.py --output ../jax-scaling-book-173fb085
```

Every selected file must match the checked-in SHA-256 manifest. The tool refuses
an existing output directory, unsafe archive paths, links, excessive sizes, or
unexpected selected-file contents. Verification checks saved hashes, local file
links, missing assets, and the generated file inventory. Keep the output local;
the download mechanism does not grant additional redistribution rights.

## Updating the reference

1. Resolve a new full commit SHA from the [official repository](https://github.com/jax-ml/scaling-book).
   Record its commit date and the review date; do not replace the pin with `main`.
2. Review its LICENSE, README, figure credits, JAX-derived material, and bundled
   asset notices again. Resolve third-party permissions before proposing any
   public mirror. Preserve each applicable license and attribution.
3. Download that revision for review without running its build scripts. Update
   `tools/learning/scaling_book_manifest.json` with the new SHA, dates, chapter
   titles, and sizes/SHA-256 hashes of the explicit selected files. Review any
   new image type, iframe, or template construct before changing the downloader.
4. Run the download into a fresh directory, run `--verify`, inspect representative
   chapters and figure links, and run the downloader's unit tests. Review byte
   totals and omitted online dependencies. Update this guide and the license
   review in the same PR. Never stage the downloaded book or archive.
