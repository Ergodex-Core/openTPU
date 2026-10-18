# Validation status

This is an experimental implementation. **The timing acceptance gate fails.** Do not use
its predicted cycles or tokens/second as measured OpenTPU performance. The existing LPN
model and validation registry are preserved; their reported 0.8–4.2% aggregate errors apply
only to the five configurations in that registry and do not validate this simulator.

## Local evidence

The implementation was exercised with Python 3.14.5, NumPy 2.5.4, pytest 9.1.1,
Apple Clang 17, CMake 4.4.3 and Verilator 5.048 on macOS ARM64. Ramulator 2.0a and
its dependencies were built from the commits in `tools/perfsim/dependencies.lock.json`.

* 31 focused tests passed, including two optional real-Ramulator tests using HBM2 and DDR4.
  They cover dependency ordering, scalar loops, finite queues, contention, backpressure,
  completion IDs, read/modify/write serialization, partial tiles, five weight formats,
  nonlinear/reduction operations, unsupported inputs and persistent memory clocks.
* A compiler-generated 3×64×19 partial GEMM ran through real HBM2 and matched sequential
  ISA arithmetic. This is a functional comparison; its timing has not been validated.
* A complete **tiny random two-layer Qwen3 topology** ran two serial prompt steps and one
  teacher-forced decode step with persistent KV and Ramulator controller state. It includes
  attention, normalization, MLP and head paths. It is not a pretrained full-size model.
  Every stage matched the ISA simulator bit-for-bit. An independent NumPy float forward
  gave relative L2 errors of 2.0373%, 2.0884% and 2.1099%, and cosines above 0.99978,
  satisfying the declared quantization-tolerance gates of 5% and 0.995 respectively.
  The ISA comparison shares functional arithmetic with the simulator and is not by itself
  an independent numerical reference.
* All three tiny-model programs also ran on the unchanged Verilator RTL with fixed-latency
  behavioral memory. DRAM and TMEM matched the ISA reference bit-for-bit, and the output
  logits passed the independent NumPy comparison. The measured counts were 5,716, 5,718
  and 5,720 cycles. These are per-program runs from declared input/KV snapshots, not a
  continuous RTL serving run. Their memory conditions differ from Ramulator, so no timing
  error or acceptance result is assigned to those comparisons.

Retained compact receipts are under `tools/perfsim/receipts/2026-10-19/`. Their manifests
hash inputs and simulator/evaluator source; upstream compiler/ISA/RTL are identified by
the base Git revision, not a claim that every upstream file received a separate hash.
Absolute paths in historical receipts identify the execution environment. Regenerate
bundles using the recorded seed and `tools/perfsim/examples.py`.

## Predicted bounded-workload results

Both profiles hold the HBM2 controller and memory clock at 1 ns. The 200 MHz core clock
is user supplied; 1 GHz is a hypothetical ASIC clock. Neither is verified physical Fmax.

| Core clock | Prompt step cycles | Decode cycles | Device-only prefill / TTFT | Decode time |
|---|---:|---:|---:|---:|
| 200 MHz | 5,817 + 5,766 | 5,796 | 57.915 µs | 28.980 µs |
| 1 GHz | 8,270 + 8,410 | 8,433 | 16.680 µs | 8.433 µs |

These are **unvalidated simulator predictions**, including the conditional tiny-workload
TPS in the JSON summaries. They exclude host transfer, queuing, sampling and networking.
Prefill here means two serial token programs, not a batched matrix prefill implementation.
At 200 MHz, physical read bytes per stage were 79,712 / 79,776 / 79,840, with 4,982 /
4,986 / 4,990 read transactions; each stage used 2,368 physical write bytes in 148 writes.
The pinned Ramulator HBM2 organization has a 16-byte transaction; see the architecture
document before interpreting its bandwidth as any physical board's bandwidth.

## Measured RTL comparison: failed timing gate

The unchanged `tb_top` Verilator testbench executed the same compiled FILL(67), MUL(67),
RSUM(67), HALT program with idle external memory. Its unmodified result counter reported
**107 cycles**, while this simulator predicted **77 cycles**: **−28.04% error**, failing
the predeclared ±5% gate. The raw RTL stdout is retained. The RTL trace shows completion
at cycles 22, 46 and 105, followed by HALT at 107. No boundary adjustment or fitted
constant was used to hide this discrepancy. This mismatch requires further implementation
of frontend/startup, vector-pipeline and reduction scheduling details, followed by a
broader held-out kernel suite; fitting one constant to this example is insufficient.
The RTL DRAM/TMEM outputs and simulator state matched the sequential ISA reference.
Validation scripts return exit status 1 for numeric failure and 2 for a failed timing gate,
after writing the evidence. A non-comparable timing run has a null gate, not a passing gate.

The cached executable was built locally with D=32, MCOLS=4, LANES=8, WIN=16,
RPB=32, WPB=1, FIFO_DEPTH=1024, DRAM_LAT=8 and AXI=0. The cached-validator checks
the full relevant parameter set in the Verilator build receipt. Reusing it does not prove
freshness against a new source checkout; the receipt records that limitation explicitly.
Active-memory RTL runs with this behavioral DRAM are not timing-comparable to Ramulator.

## Remaining work and limitations

* Full pretrained-model timing validation remains absent. No checkpoint was downloaded.
* A shared closed-loop Ramulator interface for RTL, or identical controlled memory stimuli
  on both sides, is still needed before comparing active-memory cycle counts.
* VPU/quantizer overlap, exact arbitration, MXU drain/epilogue scheduling and startup timing
  are structural approximations. The failed timing gate is evidence of this limitation.
* Multi-slice execution, streaming/WAITC/DSTEP configurations and unsupported opcodes fail
  explicitly. This is not complete coverage of every OpenTPU configuration.
* Quantized byte stores use conservative read/modify/write transactions. Physical combining
  and the FPGA shell/interconnect are not modeled. Controller/memory divided clocks are
  unsupported; core and memory clocks are independently specified.
* Hardware resource fit, routing, power and Fmax require physical implementation. None was
  performed. No FPGA workload, golden test, measurement boundary or LPN result was changed.

Several redundant local validation attempts encountered stalled file reads or macOS
`Operation canceled` errors. Their incomplete output directories are not successful
receipts. The reported successful sequence runs and test results are separately retained.
