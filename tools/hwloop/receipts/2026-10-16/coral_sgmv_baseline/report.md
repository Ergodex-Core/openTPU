# hwloop comparison (Coral + SGMV)

Generated 2026-10-16T13:39:58.465882+00:00; workload `gemma3-1b-int8`.

Labels: **measured** = a receipt (board, or the Verilator testbench as stated); **calibrated estimate** = uses a receipt-derived value, error stated; **unvalidated scenario** = no receipt checks it; **waiting** = evidence does not exist yet.

## Engine (Verilator) receipts

| run | label | commands | engine cycles | MHz | engine ms/token | weight bytes | GB/s achieved | streaming eff | overall eff | controller ms (measured split) | token tok/s est (serialized) | est label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | measured | 313 | 5526826 | 250 | 22.11 | 1013383168 | 45.98 | 0.943 | 0.718 | 35 | 17.51 | calibrated estimate |

`baseline`: `make -C sim/engine build/L4/Vsgmv_top JOBS=6 PY=~/ergodex-03/bonsai2-venv/bin/python && ~/ergodex-03/bonsai2-venv/bin/python sim/engine/gen_perf.py sim/engine/build/perf --only perf_token && sim/engine/build/L4/Vsgmv_top sim/engine/build/perf/perf_token`; HBM testbench model {'lat_min': 100, 'lat_max': 100, 'gap_pct': 0, 'max_outstanding': 32, 'dtcm_cycles_per_beat': 10}

## Board receipt

- `f2-engine-2026_10_15` (5/5 bit-exact): decode 15.81 tok/s device, prefill 19.0 tok/s (serving), TTFT [0.94, 1.34] s; 62.7 ms/token = engine 27.7 + controller 35.0 (README split). pin `{"agfi": "agfi-037d530ca13047e52", "afi": "afi-087be4144a5ab5287", "build_tag": "2026_10_15-035732", "commit": "d2227ac5fa6c06e8549742b2eb1701bfea456cca", "coral_rtl": "pinned upstream commit, scripts/gen_coral_rtl.sh (third_party/coral_rtl/README.md)", "routed": "WNS +0.036 ns, Vivado 2025.2 (docs/F2_BUILD_NOTES.md)", "logs": "results/f2_runs/gemma-3-1b-it_engine_lane-pcs-15-12.log, qwen3-0.6b_engine_lane-pcs-15-12.log"}`

## Calibration (consistency, not prediction)

- engine Verilator 22.11 ms vs board engine share 27.7 ms: -20.2%
- serialized estimate 17.51 tok/s vs board 15.81 tok/s: +10.8% (calibrated estimate). the estimate's controller part is the README's measured split, so this is a consistency check of the engine's Verilator time against the board's engine share, not an independent prediction

## VisualSim

- baseline `f2-engine-2026_10_15`: **PREPARED_NOT_EXECUTED** (waiting) not executed: --execute not given

## Waiting

- VisualSim baseline run (seat / execution)
- nightly F2 build + matched board run of the candidate (needs approval)
- a counted firmware work JSON (CG_COUNT_OPS build) to replace the README split with per-kernel controller time

## Metric definitions

- `decode_tok_s_device`: the firmware's STEP timing on the card (includes the first step)
- `prefill_tok_s`: P / wall of P-1 FEED calls: serving-side
- `ttft_s`: from immediately before FEED to the first token
- `engine_ms_per_token`: Verilator cycles of the engine command stream at the engine clock, testbench memory model
- `token_ms_est_serialized`: engine time + the measured controller share, serialized as the firmware does
