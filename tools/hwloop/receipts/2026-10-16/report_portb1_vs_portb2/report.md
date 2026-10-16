# openTPU Gemma-3-1B int8 decode token, 2 layers + head, f2-hbm NCH 8: PORTB 1 baseline vs PORTB 2 candidate

Generated 2026-10-16T14:30:20.137419+00:00; workload `gemma3-1b-int8`.

Labels: **measured** = a receipt (board, or an RTL simulation as stated); **calibrated estimate** = fitted to receipts, error stated; **unvalidated scenario** = no receipt checks it; **waiting** = evidence does not exist yet.

## RTL (Verilator) receipts

| run | label | layers | cycles | core MHz | ms/token (device) | tok/s (device, cut-down model) | DRAM bytes | GB/s achieved | port B util | roofline eff | MXU starved cyc | MXU blocked % | max outstanding reads (est) | sim wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | measured | 2 | 2.798e+06 | 200 | 13.99 | 71.47 | 3.622e+08 | 25.88 | 0.9962 | 0.996 | 17 | 0.3 | 203 | 2190 |
| candidate | measured | 2 | 1.644e+06 | 200 | 8.218 | 121.7 | 3.622e+08 | 44.07 | 1.696 | 0.935 | 1324 | 5.7 | 307 | 1782 |

`baseline`: `cd ~/ergodex-03/hwloop-scratch/opentpu-a9acb2b0 && OTPU_TARGET=f2-hbm OTPU_NCH=8 OMP_NUM_THREADS=8 ~/ergodex-03/bonsai2-venv/bin/python tools/perf_qwen.py --model gemma3 --layers 2 --wformat int8 --head-format int8 --lat 60 --mhz 200 --portb 1 --json $out/perf.jsonl --plus +nat_trace=$out/nat_trace.txt`

Phases (baseline): other 2368741 cyc, 312486912 B, VPU 552; MLP 376542 cyc, 49287168 B, VPU 26469; attention 53193 cyc, 6177016 B, VPU 6774

`candidate`: `cd ~/ergodex-03/hwloop-scratch/opentpu-a9acb2b0 && OTPU_TARGET=f2-hbm OTPU_NCH=8 OMP_NUM_THREADS=8 ~/ergodex-03/bonsai2-venv/bin/python tools/perf_qwen.py --model gemma3 --layers 2 --wformat int8 --head-format int8 --lat 60 --mhz 200 --portb 2 --json $out/perf.jsonl --plus +nat_trace=$out/nat_trace.txt`

Phases (candidate): other 1392851 cyc, 312486912 B, VPU 552; MLP 218258 cyc, 49287168 B, VPU 26584; attention 32484 cyc, 6177016 B, VPU 6768

**Candidate vs baseline (RTL, measured):** cycles x0.5873 (speedup 1.703x); DRAM bytes x1.000. RTL cycles, same workload, same memory model: the candidate change's own effect in simulation

## VisualSim

- baseline `f2-hbm-nch8-rtl`: status **PREPARED_NOT_EXECUTED** (waiting); not executed: pass --execute when no Architect GUI or batch run holds the seat
- candidate `f2-hbm-nch8-rtl`: status **PREPARED_NOT_EXECUTED** (waiting); not executed: --execute not given

## Board receipt

- `board:f2-hbm-burst200` (measured, 2026-10-16): decode_tok_s_short 22.3, decode_tok_s_long 21.7, prefill_serving_tok_s [3.7, 4.5], ttft_s_16tok 4.3, ttft_s_268tok 59.6, weights_bytes_per_token_approx 1000000000
  pin: `{"agfi": "agfi-0d06cb48544a02106", "afi": "afi-03339c4cef05e79d8", "s3": "s3://ergodex-bonsai-on-chip-438374459156-us-east-1/opentpu-alex/opentpu_burst200_20261015", "rtl_source": "not in the repository (Alex's DCP: aws-fpga cl_mem_hbm_wrapper + 2x otpu_native_axi + f2_mem BD)", "software_rev": "ternary-bonsai c52ba1113888432a3ed30a6a69e7f54ffcf4e63c (docs/ternary.md 5, docs/wide_port.md 'Baseline on the board')"}`; tool: tools/f2_bench.py via otpu-serve (card sampler, greedy, 128 tokens max); raw JSON receipt not in this repository (lives on the F2 host)

## Calibration

- RTL device tok/s 71.47 vs board decode tok/s 22.3; not comparable: the RTL run covers layers=2 of the model: its tok/s is for the cut-down model and is not the board's whole-model number; compare bytes/token and port utilization, or run --layers 0

## Waiting

- VisualSim baseline run (seat / execution)
- VisualSim candidate run
- nightly FPGA build + matched board bench of the candidate (needs approval)

## Metric definitions

- `model_load_s`: model loading and initialization (weights upload, program compile): serving receipts only
- `host_overhead`: host / runtime overhead: the difference between wall and device decode rates
- `ttft_s`: client TTFT: request to first streamed token (prefill on the device + the first decode step + host)
- `prefill_serving_tok_s`: prompt tokens / client TTFT: a serving metric, not device prefill throughput
- `device_tok_s`: decode tokens/s from device cycles only (one token's cycles at the core clock; RTL: no host)
- `decode_tok_s`: tokens after the first / their decode interval (client side)
