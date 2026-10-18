# P2 "four lock-step chunk lanes": status of M1-M2 and the correctness gate

Branch `v4-p2-four-lanes` (off `tpubench` 016e953a). Proposal: [proposal-2-four-lanes.md](proposal-2-four-lanes.md);
plan: [README.md](README.md) ("Decision"). This note says what was built, what was measured and how,
and what remains (M3-M7). Status (2026-10-18): **M1 and M2 done and gated**; M3-M7 not started.

## What was done

### M1: the configuration key (`MXU_C`)

`MXU_C` is an optional `[uarch]` key (default 1): the MXU's lock-step chunk lanes
(`opentpu/target.py` `MXU_OPT_KEYS`; stated in every `targets/*.toml`; docs/targets.md "MXU_C").
It is timing-only: lane l streams rows l, l + C, ... of every MM, each row is summed in one lane
in today's order, RMAX folds the lanes' maxima under the keys' total order, so the ISA, the
images, the programs and the CFG_HASH do not change. The card's `c4` variant
(`OTPU_TARGET=ypcb-00338:c4`) is m4 at MXU_C 4; `OTPU_MXU_C=n` overrides a simulation's.

- `target.sim_uarch` carries it to the Verilator builds (`rtlsim.BOARD_UARCH` -> tb_top /
  otpu_top / otpu_slice `MXU_C`), `target.rtl_params` to the board / ASIC tops;
  `rtlsim.chunk_lanes(uarch)` reads it.
- Analytic timing: `profile._describe(ins, cfg, C)` times an MM at ceil(N / C) x KB chunk-cycles
  (its transfers unchanged), `Profile.roofline` divides the ports' transfers by C,
  `lens.isa_data` without `--candidate C=` takes the uarch's C, `to_data` reports `config.C` and
  `peak_macs` / `peak_bytes` x C. A configuration with MXU_C 4 profiles exactly as
  `--candidate C=4` (tests/test_lens.py `test_isa_timing_takes_the_uarchs_chunk_lanes`).
- Check (proposal M1, "lens record --isa on Gemma 3"): `OTPU_TARGET=ypcb-00338:c4 python -m
  opentpu.lens record model --model gemma3 --isa --mhz 200 -o build/lens/p2-gemma3-isa-c4.otpuprof`
  gives **2,403,474 cycles, 81.7% of the 1,962,714-cycle bound** (12.0 ms at 200 MHz), the
  README's "C=4, 8 lanes" row to the cycle (the proposal's 2.19 M is at 16 vector lanes: M3).

### M2: the RTL

`rtl/mxu/otpu_mxu.sv`:

- `otpu_mxu` grew lock-step hooks, all inert at C = 1: parameters `C` (the drain's runs keep the
  C lanes' words of a step in distinct banks: `drain_run` with C), `RSTEP` (the output address
  step from one streamed row to the next: C), `RMAX_WR` (lane 0 writes RMAX, folding the other
  lanes' keys `mk_x`), `ALPHA_EXT` (ASCALE factors from lane 0); ports `pop_ok` / `pop_want` /
  `pop_pend` / `pop_ord` (the lock step), `fin_ok` / `fin_o` (the lanes complete together),
  `mk_o` / `mxh_o` / `mk_x` / `mxh_x` (the RMAX fold), `alpha_o` / `al_rdy_o` / `alpha_i` /
  `al_rdy_i`.
- `otpu_mxu_lanes` (the same file): C copies of `otpu_mxu`, each with the MM as its own rows
  (row base + l x rs, strides x C, ceil((N - l) / C) rows, output base + l; the scale rows the
  same way), **one ACT RAM read** (lane 0's addresses and enables) broadcast to the lanes, which
  pop block k of rows n .. n + C - 1 in the same cycle (`pop_ok`: every lane with a pop left in
  lane 0's command must want to pop; a lane with fewer rows, N not a multiple of C, waits for
  lane 0 at the next command), **per row group credits** (each lane's result FIFO fills in the
  same cycles and the drains step together under the one TMEM grant), the lanes' results as **C
  consecutive TMEM words** (lane l's drain slots at l x E/C + k of the MXU's TMEM lanes; E / C
  epilogue lanes a lane, 1 at the card's E 4), the RMAX fold, ASCALE from lane 0. Lane 0 streams
  on the slice's ports A / B (with the DMA, as today); lanes 1 .. C-1 on read-only **lane ports**
  of their own (`lb_*` chunks, `la_*` scale lines) through registers like port B's. The slice
  sees one MXU (`rdy` the lanes' AND, `done` lane 0's, the profiling counters lane 0's; `bm` /
  `am` count the cycles any lane asked).
- `rtl/top/otpu_slice.sv`: `MXU_C`, the lane ports (`rd_idle_i` covers their reads),
  `otpu_mxu_lanes` in place of `otpu_mxu`; `rtl/shell/otpu_board.sv` and
  `rtl/asic/otpu_asic_top.sv` tie the lane inputs off (MXU_C 1).
- Memory models at C chunks a cycle: `rtl/mem/otpu_dram.sv` (the behavioural DRAM: lane ports
  at LAT), `sim/verilator/otpu_native_mem.sv` (the board model's memory behind the native
  adapter: lane ports performed at the request, data back after `+axi_lat` + 0..7 cycles of
  jitter with `+axi_stall` bubbles, like a channel's reads; no `+axi_bw` or DDR3 bank model on
  them: a stand-in for an HBM pseudo-channel group a lane, the real adapter is M4; `l_idle`
  into `rd_idle`). `rtl/top/otpu_top.sv` wires them (`AXI 2`, the co-simulated LiteDRAM, is
  MXU_C 1 only). The chunk FIFO is per lane (each `otpu_mxu` has its own DEPTH-chunk FIFO).
- `tools/p2_gemma3_token.py`: one real Gemma 3 1B token on the RTL at MXU_C against the ISA
  simulator (logits and the DRAM image bit for bit), saved as a Lens profile.

### Correctness at MXU_C = 1: bit-exact and cycle-exact to today

Eight RTL runs (mlp and attention kernels, two random programs, an MM-replay program, three
board-memory-path programs incl. one on the card's uarch) were recorded before the RTL change
and after it: identical cycle counts and state hashes (build/p2/baseline_c1.json vs
after_c1.json: mlp 7,508, attn 2,275, fuzz 2,424 / 1,345, replay 3,468, board 2,445 / 5,116 /
5,778 cycles). Test files run at MXU_C 1 (all pass; the list in "Tests run" below).

### Correctness at MXU_C = 4: the gate

All bit-exact against the ISA simulator (TMEM, DRAM, logits):

- (a) kernels and MXU unit tests: `tests/test_rtl.py -k chunk_lanes` (12 tests, 11 min): random
  programs at C 2 / 3 / 4 (MMs of N 1..23, not multiples of C: lanes with fewer rows; RMAX, ACC,
  ASCALE, PAIR, int4 / fp4 / NVFP4, RSC), MM replay (M > MCOLS) at C 4, the board memory path
  (hazard and random programs, stalls 0 / 40 / 60 %, latency 20 / 120) at C 4, the card's uarch
  (one pumped TMEM copy, WAITC) at C 4, and a decode MM of 256 rows x 8 chunks taking < 0.45x
  its C = 1 cycles at C 4. The `mlp` / `attention` kernels run at C 1 in test_rtl.py as before
  (`test_mlp_rtl`, `test_attention_*_rtl`); at C 4 the Gemma 3 token below covers them.
- (b) the tiny random Gemma 3 (tests/test_gemma3.py `test_token_on_board_rtl_is_bit_exact[1|4]`,
  --runslow: a token at position 600, past the sliding window, resident decode, the board memory
  path with the boot loader): logits and the whole DRAM image bit-exact at MXU_C 1 and 4 (3:46).
- (c) the real Gemma 3 1B decode token (`tools/p2_gemma3_token.py --mxu-c 4`, the prompt "Why is
  the sky blue?", step at position 14, '\n' -> 'Okay', the same token as the ISA simulator's and
  docs/gemma3.md's): **argmax 19058 on both, all 262,144 logits bit-exact, the DRAM image (the KV
  cache the token wrote) bit-exact**, at memory stalls 0, 5 and 20 % (the three runs below).

## Measurements (Gemma 3 1B, one decode token, position 14, the card's m4 arch, 200 MHz)

The memory is the board model's (the native adapter in front of `otpu_native_mem`, lanes 1-3 on
its lane ports), latency 30 (`OTPU_LAT=30`), 128 reads in flight a lane. "Stall 0" is the
HBM-like reference the coordinator set (no handshake withheld); 5 % stands for HBM jitter, 20 % is
the DDR3 model's default. The DRAM bound at 4 chunks a cycle is 1,962,714 cycles (7,850,855 / 4).

| run | cycles | of the bound | MXU not consuming | MACs/cycle | ms at 200 MHz | tok/s |
|:--|--:|--:|--:|--:|--:|--:|
| C = 1, stall 20 (the README's trace, `today-rtl-200mhz`) | 10,279,178 | 76.4 % | 18.2 % | 99 of 512 | 51.4 | 19.5 |
| C = 1, stall 5 (coordinator's `today-rtl-200mhz-stall5`) | 8,651,201 | 90.7 % | 5.1 % | 117 | 43.3 | 23.1 |
| C = 1, stall 0, LAT 30 (coordinator's reference) | 8,221,777 | 95.5 % | 1.4 % | 123 | 41.1 | 24.3 |
| ISA model, MXU_C 4 (8 vector lanes; = `--candidate C=4`) | 2,403,474 | 81.7 % | - | - | 12.0 | 83.2 |
| **C = 4, stall 0, LAT 30** (`p2-gemma3-c4-stall0.otpuprof`) | **2,292,283** | **85.6 %** | **0.9 %** | **442 of 2048** | **11.5** | **87.2** |
| C = 4, stall 5, LAT 30 (`p2-gemma3-c4-stall5.otpuprof`) | 2,407,648 | 81.5 % | 0.9 % | 421 | 12.0 | 83.1 |
| C = 4, stall 20, LAT 30 (`p2-gemma3-c4-stall20.otpuprof`) | 3,013,560 | 65.1 % | 2.0 % | 336 | 15.1 | 66.4 |
| C = 4, stall 0, Lens's recorder (`p2-gemma3.otpuprof`, the task's command) | 2,292,246 | 85.6 % | 0.9 % | 442 | 11.5 | 87.3 |

`lens summary build/lens/p2-gemma3-c4-stall0.otpuprof` (the stall-0 run), where the cycles went:

```
  2,292,283 cycles   11.46 ms at 200 MHz   5350 instructions
  Roofline 85.6% of DRAM-bound minimum (bound 1,962,714 cycles)   MACs/cycle 442 (peak 2048)
  DRAM port B busy 93.2%  (MXU 2121402, DMA 14588)        [cycles in which any lane streamed]
  92.7%  DRAM port B streaming        0.1%  MXU starved (chunk FIFO empty)
   0.9%  MXU not consuming             0.7%  TMEM bank arbitration
   1.3%  MXU computing, not streaming  2.0%  waiting on VPU work   2.1%  waiting on quantizer work
  Unit utilisation: MXU 99.7%, VPU 19.9%, QUANT 5.3%, DMA 3.1%
  TMEM arbitration losses: MXU drain 9.4% (214,421 cycles; absorbed: the drain is 4 results a
  cycle at M = 1 against VPU-first)
```

Reading it, against the proposal's section 5 and M0's split (docs/v4/m0-gating-split.md):

- **The stream scales by 3.59x** (8.22 M -> 2.29 M cycles at the same memory): 4 lanes at 4
  chunks a cycle, the RTL 5 % under the 8-lane ISA model (2.40 M) and 4.6 % over the proposal's
  2.19 M model at 16 lanes. The 85.6 % is above the README's "~74 tok/s if today's 0.81x ratio
  holds" because M0 was right: the 18.2 % was the memory model, not the MXU.
- **Lane deskew**: port B counts a cycle busy when any lane streams: 2,121,402 such cycles for
  1,959,392 chunk-cycles of work, so **162 k cycles (7.1 % of the token)** have fewer than 4
  lanes streaming: a lane's chunks arriving before the others' at an MM's start, the MM's end
  (each lane's last row), and the lanes' separate scale-line arrivals. This is the largest P2
  specific loss and the item for M4's adapter and a lead on the slowest lane (section "What
  remains").
- **Per-row-group gating** ("MXU not consuming", `pf_block` of lane 0 incl. waits for other
  lanes' chunks): 0.9 % = 20.6 k cycles over 240 k row groups, **0.09 cycles a group** (M0's C = 1
  figure 0.24 a row at stall 0); the result-FIFO credits and the "zero-bubble" lead were not
  needed for it (M0: not a performance item). The MXU's start / drain per command ("computing,
  not streaming") 1.3 %, 30 k cycles = 33 a command (123 k at C = 1: 130 a command).
- **The vector work is not widened (no M3)**: the VPU is in flight 19.9 % of the token (4.5 %
  at C = 1: the same 0.46 M cycles against a 4x shorter run) and the quantizer 5.3 %; what the
  FIFO cannot hide is **2.0 % (VPU) + 2.1 % (quantizer) = 4.1 % of the cycles exposed, 94 k
  cycles**, in the QACTs before the attention and MLP matmuls (gemma4.py:1634 / 1695,
  attention.py:195: 116 k cycles of MXU waits named by Lens, overlapping). M3's 16 lanes / CL 4
  would halve the VPU's 0.46 M and the quantizer's 0.12 M; the exposed part would fall to
  ~2 % (the proposal's 0.15 M expected -> ~0.05 M).
- **TMEM**: the drain lost 9.4 % of the cycles to arbitration (VPU-first), absorbed by the result
  FIFO (0.7 % reaches port B); at M3's 16 banks the 4 consecutive words of a row group spread
  over more banks.
- **Memory jitter** (stall 5: 2.41 M, 81.5 %; stall 20: 3.01 M, 65.1 %): with the lanes in
  lock step the group runs at the slowest lane's rate, and lane 0's path is the card's adapter
  over the two-channel model whose peak equals the port's (M0's 0.76 chunks a cycle at 20 %
  stalls), so at 20 % the token is 1.31x the stall-0 one (C = 1: 1.25x) with port B "busy"
  95.3 % of the cycles (some lane streaming) but the lanes deskewed most of the time. On the
  F2 each lane has 4.5x its demand in HBM bandwidth (proposal section 2); the stall-0 run is
  the reference, the stall-5 one the jitter allowance (-5 %).
- **Against the proposal's section 5**: expected 2.47 M (79 tok/s), pessimistic 2.84 M (69): the
  RTL at the HBM-like memory is 2.29 M = 87 tok/s before M3 and before the software items
  (software-optimizations.md: the card loop, row folding), with the lane deskew (7 %) and the
  exposed vector work (4 %) the two items left on the P2 side.
- Simulation speed: 4.6-7.6 k cycles/s at C = 4 (Verilator, one thread; 4 MXU arrays): a token
  in 6-10 minutes, about the C = 1 token's wall time.

## Tests run

At MXU_C 1 (the suite as it was, plus the new tests at C 1): `tests/test_target.py` (23),
`tests/test_lens.py` (ISA parts, then the whole file with its RTL fixture), `tests/test_kernels.py`,
`tests/test_mx.py`, `tests/test_pair.py`, `tests/test_isa.py`, `tests/test_config_matrix.py` (111
pass, the one pre-existing failure above), `tests/test_fp_rtl.py`, `tests/test_quant_rtl.py`,
`tests/test_stream_rtl.py`, `tests/test_perf.py` (with test_lens: 67 passed, 1 pre-existing
xfail, 70 min), `tests/test_rtl.py` (188 passed, 1 skipped, 81 min; one run hit a full disk
during a Verilator build and passed alone), the 8 cycle-probe runs (identical cycles and
hashes); at MXU_C 4: `tests/test_rtl.py -k chunk_lanes` (12), `tests/test_gemma3.py -k
token_on_board --runslow` (C 1 and 4), the three real-token runs. Each Verilator build is
~0.4 GB under build/verilator: clear it between suites on a full disk.

## What remains (M3-M7)

- **M3** VPU 16 lanes / VPU_CL 4, quantizer 16 + XD 512, TMEM 128K x 16 banks in URAM, ACT_ROWS 16:
  not started. Without it the vector work is not widened: see "exposed" above.
- **M4** `otpu_hbm_dram`: the 32-PC HBM adapter (async FIFOs, group muxes, the 4 x 4 lane
  crossbar), ports A / SW / DMA through it: not started. The lane ports of M2 are the memory
  models' stand-in (per lane a request a cycle at a fixed latency with jitter); the adapter must
  present the same lane-port protocol to the slice (`lb_*` / `la_*`: a request taken with its
  grant, data in order).
- **M5** `targets/f2.toml`, the F2 top and flow; **M6** the AFI; **M7** bring-up: not started.
- Open items from M2 worth a look before or with M4: (1) **the lane deskew, 7 % of the token**:
  the lanes' chunks arrive independently (lane 0 through the adapter, the others through their
  ports), so a row group waits for its last lane at every MM start and the lanes idle one by one
  at the end; the adapter of M4 should issue a row group's four chunk requests together (one
  512 B burst a group, proposal section 2) and a Lens counter per lane ("lanes streaming this
  cycle", 0-4) would measure it directly; (2) the lane ports are read-only and separate from port
  B's register / arbitration (the DMA shares lane 0's port B only); (3) `pf_block` at C > 1
  includes cycles lane 0 waited for another lane's chunks (M0's split by cause would separate
  them); (4) a lane with fewer rows than lane 0 idles for a row at every MM whose N is not a
  multiple of C (none of Gemma 3's); (5) the drain's 9.4 % TMEM arbitration loss is absorbed
  today but is the first thing the result FIFO's margin meets at M3's shorter rows.
- Pre-existing on the base (tpubench 016e953a, fails identically on an export of it, nothing P2
  touches): `tests/test_config_matrix.py::test_a_decode_step_is_placed_best_fit_where_next_fit_fails`
  (Qwen3.5's compiler placement at a 16K-word TMEM: "DID NOT RAISE CompileError").

## Commands to reproduce

```sh
# environment: this worktree's tree with the main tree's interpreter
PY="PYTHONPATH=. /Users/rachittibrewal/ergodex-04/tpubench/.venv/bin/python"

# M1: the ISA analytic profile at MXU_C 4 (the candidate C=4 model)
OTPU_TARGET=ypcb-00338:c4 $PY -m opentpu.lens record model --model gemma3 --isa --mhz 200 \
    -o build/lens/p2-gemma3-isa-c4.otpuprof
$PY -m opentpu.lens summary build/lens/p2-gemma3-isa-c4.otpuprof

# M2 at MXU_C 1: the suite's MXU / RTL / profiler / Lens tests
$PY -m pytest -q tests/test_rtl.py tests/test_mx.py tests/test_kernels.py tests/test_pair.py \
    tests/test_isa.py tests/test_fp_rtl.py tests/test_quant_rtl.py tests/test_stream_rtl.py \
    tests/test_perf.py tests/test_lens.py tests/test_target.py tests/test_config_matrix.py
# M2 at MXU_C 4: the chunk-lane tests, the tiny Gemma 3 token on the RTL
$PY -m pytest -q tests/test_rtl.py -k chunk_lanes
$PY -m pytest -q tests/test_gemma3.py -k token_on_board --runslow

# the gate's real token: logits against the ISA simulator, the Lens profile (HBM-like memory:
# no handshake withheld, latency 30; then 5% and the default 20% of stalls)
OTPU_LAT=30 $PY tools/p2_gemma3_token.py --mxu-c 4 --stall 0 -o build/lens/p2-gemma3-c4-stall0.otpuprof
OTPU_LAT=30 $PY tools/p2_gemma3_token.py --mxu-c 4 --stall 5 -o build/lens/p2-gemma3-c4-stall5.otpuprof
OTPU_LAT=30 $PY tools/p2_gemma3_token.py --mxu-c 4 --stall 20 -o build/lens/p2-gemma3-c4-stall20.otpuprof
# the same token through Lens's own recorder (the task's command), the uarch at MXU_C 4
OTPU_TARGET=ypcb-00338:c4 OTPU_LAT=30 $PY -m opentpu.lens record model --model gemma3 --axi --stall 0 \
    --mhz 200 -o build/lens/p2-gemma3.otpuprof
$PY -m opentpu.lens summary build/lens/p2-gemma3.otpuprof
```
