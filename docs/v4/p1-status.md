# P1 "wide port": milestones 1-3, status

Branch `v4-p1-wide-port` (off `tpubench`), 2026-10-18. The proposal is
[proposal-1-wide-port.md](proposal-1-wide-port.md); this is what of it is built, what it
measures, and what remains. Every number below is from a run in this tree (section 5 reproduces
them); nothing is projected.

## 1. What was built

**M1, the configuration key.** `Config.CPC` (chunks per cycle, 1 or 2: the D-byte chunks DRAM port
B delivers and the MXU consumes a cycle; `opentpu/isasim.py`) is an arch key stated in every
target's `[arch]` (`targets/*.toml`, CPC = 1) and in the new `targets/f2.toml` at CPC 2 (one slice
as the card's: D 128, MCOLS 4, 8 lanes, 64K TMEM words, the card's pumped TMEM pick; FIFO_DEPTH 512
entries of 2 chunks, 128 KiB as the card; 200 MHz, the clock the F2 synthesis closed).
`OTPU_CPC=2` sets it in `board_config()` as `OTPU_MCOLS` does. It is in the arch hash only above
1, as SEQ_WC is only when on, so every one-chunk build's CFG_HASH stays (`tests/test_target.py`'s
pinned hashes hold). `rtl_params` carries it; `rtlsim.top_params` passes it to the simulation
top. The ISA simulator is unchanged (bit-exact, no timing). The analytic timing
(`opentpu/profile.py _describe`): an int8 MM is `N x ceil(KB / CPC)` cycles (the RTL's half
pairs), or `ceil(N x KB / CPC)` with `flat=True`, which is what Lens's `--candidate C=n` uses (the
proposal's milestone-4 stream): one function for both, so a CPC 2 Config profiles like
`--candidate C=2` up to the half pairs of odd KB rows (section 2). The roofline gains the DRAM
bound beside the bound (`roofline.dram`: port-B chunks over CPC; `lens summary` prints it when
the MXU's advances are above it); `peak_macs` and `peak_bytes` follow CPC. Documented in
[../targets.md](../targets.md) and [../isa.md](../isa.md) (MM).

**M2, the RTL** (`rtl/mxu/otpu_mxu.sv`, `rtl/top/otpu_slice.sv`, `rtl/top/otpu_top.sv`,
`rtl/mem/otpu_dram.sv`, `rtl/mem/otpu_native_dram.sv`, `sim/verilator/tb_top.sv`,
`rtl/shell/otpu_board.sv`, `rtl/asic/otpu_asic_top.sv`), parameter `CPC`:

- The MXU's chunk FIFO holds entries of CPC chunks (FIFO_DEPTH counts entries). An int8 MM takes
  a pair an advance: `ceil(KB/2)` advances a row, the chunk address stepping 2D. A second group
  of MCOLS physical columns sits in lock step with the first: NCOL = CPC x MCOLS columns, the
  weight stream hopping on through group 1's columns as through group 0's (the pre-adder's
  "high" stream carries the pair's second chunk, as it carries PAIR's high half), group 0 on
  chunk 2c with ACT block ab + 2c, group 1 on chunk 2c + 1 with block ab + 2c + 1. Group 1's
  columns read a second ACT RAM copy (`otpu_slice`: CPC copies written alike; group 1 reads
  `act_blk2` as PAIR's high rows do), so each column end accumulates its own blocks with the
  block's own `da`; the groups' integers are added before the result FIFO as PAIR's partner
  columns are: the same exact sum, so bit-exact by construction. LROW grows by MCOLS (the
  deskew), RF stays its rule (64 at CPC 2).
- An odd KB's last advance is a half pair: a narrow port-B request (`b_wide` low, one chunk;
  no read past the row), group 1's multipliers 0 (PAIR's "+0 for the missing block"), group 1's
  columns reading the even block again (outside the MM's ACT range nothing is read).
- The scale entry of an advance holds the pair's two block bytes from one line ({Sw, dw of 2c},
  {dw of 2c + 1}: PAIR's two halves), one port-A request an advance (half the requests of CPC 1).
  When the two bytes straddle a 32-byte line (an odd k0: the bytes sit at odd offsets) the
  second comes from a request of its own the cycle after (`i_sx`, a cycle without a chunk, as
  the row's Sw request `i_sws`). Gemma 3's kernels have even k0 everywhere, so no straddles.
- The pop head moves early to the next command only between commands of one kind (`w2`, as
  PAIR and M already gate it); 4-bit MMs (PAIR or not) stay one chunk an advance at any CPC
  (group 1 idle; narrow requests). The proposal's "each group takes a PAIR" for 4-bit formats
  is not built (section 4).
- Port B: `b_wide` beside the request (the MXU's pairs; the DMA's loads, stores and the loader
  read one chunk and take the low chunk of `b_rdata`, which is CPC chunks wide). The behavioural
  `otpu_dram` returns CPC chunks from the address. `otpu_native_dram`, the card's adapter and
  the simulation's memory path (`--axi`), and the native memory model behind it
  (`otpu_native_mem`) run NCH = 2 x CPC channels: chunk m on channel pair m mod CPC (its halves
  on the pair's two channels, hashed by the chunk's parity as today) at channel beat m / CPC,
  so a wide read is one beat on each of the four channels (any pair alignment: the two chunks
  are on different pairs), a narrow read one beat on its chunk's pair, and the merge takes an
  order entry's channel set. The first version of this milestone issued a wide read as two
  beats on each of two channels; the token of section 2.1 then took 8.17 M cycles, CPC 1's
  figure, with "MXU not consuming" 44%: two 64-byte channels deliver 128 B a cycle, half of a
  CPC 2 port's request, and the MXU consumed each pair as it landed (milestone 0's "the memory
  below the port's rate", exactly). The simulation's memory must deliver 256 B a cycle to
  measure the MXU at CPC 2, which the four channels do; the second chunk's parity is computed
  in the adapter (a simulation stand-in: the HBM adapter of milestone 5 keeps a pair on one
  pseudo-channel and sees none of this).
- At CPC 1 every module is what it was: the four baseline runs (the MLP at D 32; the board's
  MLP on the behavioural DRAM and on the native path with the boot loader; flash attention on
  the native path) give the same cycle counts before and after (7,508 / 99,846 / 130,124 /
  3,254: `build/p1/baseline.py`), and the test suite at CPC 1 passes (section 3).

**M3, correctness at CPC 2** (`tests/test_cpc.py`, `tests/test_gemma3.py`): section 3.

**Also:** `lens record model --check` runs the traced step on the ISA simulator too and compares
the RTL's DRAM (the logits, the caches) byte for byte, the result in the profile's name and
notes; `rtlsim` gives the simulation the image's extent of the DRAM and dumps that extent
(`+dump_words`), so a 1 GiB image in a 2 GiB DRAM writes 1 GiB files, not 2 (the build host's
disk was full).

## 2. Measurements

### 2.1 Gemma 3 1B, one decode token, the RTL at CPC 2

The same token as [../gemma3.md](../gemma3.md) "Lens" and
[m0-gating-split.md](m0-gating-split.md) (position 14 after the chat template's 15-token
prompt, int8 weights and head, the resident decode program, 5,350 instructions, 922 MMs,
721,408 streamed rows, 7.81 M chunks), on the Verilator board model of `targets/f2.toml` (CPC 2,
FIFO 512 x 2 chunks, 200 MHz) through the native memory path (`--axi`), at three settings of the
memory model: `--stall 0` with `OTPU_LAT=30` (a memory that never withholds the port's
handshakes, the coordinator's HBM-like reference), `--stall 5` (HBM jitter) and the default
stall 20 / LAT 20 (the model fitted to the card's DDR3-1066, whose peak equals a CPC 1 port's
rate and is half a CPC 2 port's). The `--check` column: the RTL's DRAM after the step against
the ISA simulator's (the logits and the caches, the whole image), and the next token.

| memory model | cycles | ms at 200 MHz | tok/s at 200 MHz | of the DRAM bound (3,925,428) | of the roofline (the MXU's advances with half pairs, 4,249,632) | MXU not consuming (`mb`) | MACs / cycle | against CPC 1 | bit-exact |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| stall 0, LAT 30 (HBM-like) | **4,639,029** | **23.20** | **43.1** | 84.6% | 91.6% | 2.1% (98 k cycles, 0.14 a row) | 218 | 1.77x (8,221,777) | yes: the whole DRAM, next token 'Okay' |
| stall 5, LAT 30 | 4,707,847 | 23.54 | 42.5 | 83.4% | 90.3% | 3.1% (146 k, 0.20 a row) | 215 | 1.84x (8,651,201) | (the stall-0 run's check) |
| stall 20, LAT 20 (the card's DDR3) | 5,409,476 | 27.05 | 37.0 | 72.6% | 78.6% | 12.8% (692 k, 0.96 a row) | 187 | 1.90x (10,279,178) | |
| CPC 1, stall 0, LAT 30 (m0's reference) | 8,221,777 | 41.1 | 24.3 | 95.5% of 7.85 M | the same | 2.1% (171 k, 0.24 a row) | 99 | | |

The rest of the stall-0 run (port B's cycles): streaming 91.9%; the chunk FIFO starved 1.2%
(56 k cycles: the 30-cycle latency at each MM's start); "MXU computing, not streaming" 1.5%
(67 k: the fill and drain of the 922 MMs); waiting on the VPU 1.5% and on the quantizer 1.5%
(139 k: the norm -> QACT -> MM chains, whose share doubled with the stream rate as the
proposal's section 2 expected); TMEM arbitration of the stream 0.2%. The MXU drain lost 230 k
cycles (5.0% of the run) to TMEM bank arbitration, the same 225-230 k as at CPC 1 at twice the
share; it does not reach port B (the result-FIFO credits absorb it: milestone 0) but it is the
first per-row item that will. The VPU is in flight 9.9% of the run, the quantizer 3.0%, the DMA
1.5%.

The ISA simulator's analytic timing of the same token (units serialized, no RTL overlap and no
RTL loss): at CPC 2 (`OTPU_TARGET=f2 --isa`) 4,693,714 cycles with the half pairs (the RTL at
stall 0 is 1.2% under it, as CPC 1's 8.22 M is 1% under its 8.28 M); the README's `--candidate
C=2,MHZ=200` flat model 4,362,834 (90.0% of the DRAM bound). The half pairs are the difference
between the DRAM bound and the MXU's: 324,204 port cycles (7.0% of the token), Gemma 3's 661,504
KB-9 rows at half a cycle each, the proposal's milestone-4 item.

**Reading.** On a memory that keeps the port's rate the CPC 2 RTL takes 4.64 M cycles for a
token, 1.77x fewer than the CPC 1 RTL on the same memory (8.22 M) and 2.2x fewer than the card's
trace (10.28 M at stall 20), 43 tok/s at 200 MHz against the proposal's "realistic 38-41" and
its 51 tok/s roofline: 84.6% of the DRAM bound, 91.6% of what this milestone's half pairs
allow. The per-row MXU-side cost did not double its share: `mb` is 0.14 cycles a streamed row
at CPC 2 against 0.24 at CPC 1 (milestone 0's stall-0 split: rows late at the model's hiccups
and the scale entries after their chunks), because at CPC 2 a port-A line request serves a
pair and the half of the rows whose scale line arrives late is the same half. What did double
in share is everything per instruction (the VPU and quantizer waits, 3.0%) and the drain's
arbitration loss (5.0%). At the DDR3-fitted stall 20 the token is 5.41 M cycles: the two
channel pairs still deliver 1.9x the card's bytes, but the memory is 0.79 of a CPC 2 port's rate
and `mb` is 12.8% (0.96 a row), milestone 0's "memory below the port's rate" again. The F2's
HBM at 11% utilisation is the stall-0 / stall-5 case.

### 2.2 Milestone 0's findings, as they apply here

[m0-gating-split.md](m0-gating-split.md) split "MXU not consuming" by cause on the CPC 1 RTL: at
the DDR3 model's 20% stalls 97% of it is "row incomplete" (the memory delivering 0.76 chunks a
cycle under the port's 1), credits 0.1%, scales 2.6%; at stall 0 / LAT 30 the item is 2.1% of
the cycles (0.24 a row) = rows 57% / scales 40% (68.6 k cycles, all in the layers) / credits 3%.
So for P1: the RF 32 -> 128 credit widening (proposal section 2 (a)) is not a performance item
(kept as insurance: RF is 64 at CPC 2 by the existing rule, LROW + 7 > 32); the two-row lead (b)
cannot pop rows whose chunks have not arrived; the scale-line prefetch a row ahead (c) is the
one MXU-side item, and its share grows at CPC 2 unless port A keeps pace. What this milestone
does for port A: an int8 MM makes one line request a pair (half of CPC 1's), so port A's
request rate per cycle is unchanged by the doubled chunk rate; the row's Sw request of its own
(`i_sws`, when the Sw is in neither neighbour's line) still costs a port-B cycle, now worth two
chunks. Measured (section 2.1, stall 0): `mb` 0.14 cycles a row at CPC 2 against 0.24 at CPC 1,
"MXU computing, not streaming" 0.09 a row against 0.17: the per-row fixed cost went down, not
up, and the Sw prefetch (milestone 2) is worth at most the 2.1% `mb` holds. The cause split by
`mbc` / `mbr` / `mbs` is milestone 0's branch; this tree does not have it, so the 2.1% is not
split here.

## 3. Tests run

At CPC 1 (must be unchanged, bit-exact and cycle-exact; `PYTHONPATH=. python -m pytest -q
<file>`, the card's target): `tests/test_isa.py` 14 passed, `test_kernels.py` 41,
`test_mx.py` 32, `test_pair.py` 55, `test_target.py` 23, `test_config_matrix.py` 43 passed and
1 failed (`test_a_decode_step_is_placed_best_fit_where_next_fit_fails`, Qwen3.5 at a 16K-word
TMEM: "DID NOT RAISE CompileError"; it fails the same way on the unmodified `tpubench` tree, so
it is not this branch's; the new f2 target is in the matrix and passes its kernels and models),
`test_lens.py` 16, `test_quant_rtl.py` 30, `test_stream_rtl.py` 6, `test_waitw_rtl.py` 11,
`test_perf.py` 6 passed and 1 xfailed, `test_board.py` 156, `test_rtl.py` 176 passed and 1
skipped, `test_gemma3.py` 7 passed (its two slow RTL tokens below), `test_tech.py` 1 (the
xilinx7 cells against the generic ones), `test_gemma4.py -k token_on_board_rtl --runslow` 3
passed (the RTL token through the board's memory path). The cycle counts of the four baseline
runs are unchanged (section 1). `test_quant_rtl.py` and `test_board.py` first failed with
"No space left on device" (the host's disk, full from outside this tree) and were re-run.

At CPC 2 (`tests/test_cpc.py`, `tests/test_gemma3.py -k token_on_rtl --runslow`): the MLP decode
kernel (hidden 1152: KB 9 rows, every one a half pair) and flash attention (d 128: KB 1 rows,
half pairs alone) on the behavioural DRAM and on the native memory path (random stalls, the
boot loader); MX MMs at D 128 / MCOLS 4 in every weight format: int8 pairs at M 1 and M 4, odd k0,
the pair's bytes straddling a line mid-row and at a command's last advance (k0 27), RSC, replay
over 12 and 16 ACT rows (half pairs through the FIFO's row replay), int4, fp4 PAIR, NVFP4 with and
without PAIR, each with UNIT, ACC, RMAX and ASCALE epilogues and Sw corners; int8 and 4-bit
commands queued back to back; the analytic timing. The tiny Gemma 3 with odd block counts
(hidden 384, intermediate 640: KB 3 / 5 rows) decoded at position 600 on the RTL through the
board's memory path at CPC 1 and CPC 2: logits and the whole DRAM image equal to the ISA
simulator's, bit for bit (2 passed, 221 s). The real Gemma 3 1B token: section 2.1's check
column.

## 4. What remains

- **The row-gating fix** (proposal milestone 2), by milestone 0's finding: the scale-line
  prefetch a row ahead (port A), not the credits; measure `mbs` at CPC 2 first (section 2.1).
- **Flat pairs** (proposal milestone 4): the issuer streaming a matrix as 256-byte pairs across
  row boundaries, a per-group row-total FIFO pairing the totals, 256-byte alignment of the
  weight matrices in the image (`_Bump.alloc`, Gemma 4's Image) and the ISA simulator's
  alignment check. Today's half pairs cost a port cycle per odd-KB row: Gemma 3's KB-9 rows
  stream in 5 advances for 4.5 chunks' worth, 0.33 M cycles a token (661,504 KB-9 rows x 0.5),
  the difference between the DRAM bound (3.93 M) and the MXU's bound (4.24 M).
- **4-bit formats at two chunks an advance** ("each group takes a PAIR", four blocks an
  advance): not built; fp4 / int4 / nvfp4 MMs run at one chunk an advance at CPC 2 (the card's
  rate), bit-exact. Gemma 3 1B is int8 throughout, so its token does not use them.
- **Port A at CPC 2**: the Sw-only request (`i_sws`) a row ahead (milestone 2's prefetch), and
  the scale FIFO's entries filled two from one line as the proposal's section 9 says (today one
  entry a request, a pair's bytes in it).
- **The HBM adapter, the F2 shell and flow, the host's channel map, the CFG field**: done on
  Alex's burst200 fabric instead of a 32-PC `otpu_hbm_dram` (section 6): the slice's four native
  channels each on an `otpu_native_axi` and its own HBM port, `otpu_board` at CPC 2, CAPS2 [3:1]
  = CPC, the host's CPC 2 map. What section 6 leaves: the AFI and the card (the coordinator's),
  Lens on the card (milestone 7), the 32-PC adapter (only if a later point needs more than four
  HBM ports a slice).
- The native adapter's wide path at CPC 2 is no longer a stand-in: it is the F2 build's
  adapter (four channels, the second chunk's parity computed in it). At CPC 1 (two channels)
  its B-queue and order entries carry constant-zero `wide` / pair / swap bits that synthesis
  removes, and the slice's `b_wide` output is constant zero: the card's build is otherwise what
  it was, and its simulations are cycle-exact to before.

## 5. Reproducing

```sh
# the baseline cycle counts at CPC 1 (before / after: the same numbers)
PYTHONPATH=. python build/p1/baseline.py build/p1/after.json

# correctness at CPC 2
PYTHONPATH=. python -m pytest -q tests/test_cpc.py
PYTHONPATH=. python -m pytest -q --runslow tests/test_gemma3.py -k token_on_rtl

# the real Gemma 3 1B token on the RTL at CPC 2 (targets/f2.toml), the HBM-like memory, with
# the ISA comparison; then the jittery and the DDR3-fitted memory models
OTPU_TARGET=f2 OTPU_LAT=30 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 \
    --axi --stall 0 --check -o build/lens/p1-gemma3.otpuprof
OTPU_TARGET=f2 OTPU_LAT=30 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 \
    --axi --stall 5 -o build/lens/p1-gemma3-stall5.otpuprof
OTPU_TARGET=f2 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 \
    --axi -o build/lens/p1-gemma3-stall20.otpuprof
OTPU_TARGET=f2 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --isa \
    -o build/lens/p1-gemma3-isa.otpuprof
PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --isa --candidate C=2,MHZ=200 \
    -o build/lens/p1-gemma3-cand.otpuprof                       # the README's flat model
PYTHONPATH=. python -m opentpu.lens summary build/lens/p1-gemma3.otpuprof
```

The profiles are `build/lens/p1-gemma3{,-stall5,-stall20,-isa,-cand}.otpuprof` on the build
host (not committed); `build/p1/followup.sh` is the chain that produced the follow-up runs and
the regression log.

The build host's `models/gemma-3-1b-it` is a symlink to the checkpoint. Each RTL run of the
token is 10-25 minutes on the Verilator board model (the Python side holds the fp32 weights,
the ISA machine's DRAM and the RTL's: about 10 GB).

## 6. On the AWS F2: P1 on Alex's burst200 fabric (2026-10-20)

Branch `p1-f2` (this tree: `v4-p1-wide-port` bdc32a03 plus the commits below), built and run on
the m8a.16xlarge build box (Verilator 5.048, Vivado 2025.2, AWS HDK 2.3.0 small shell
0x10212415). Everything here is simulation and timing; nothing has run on the card yet.

### 6.1 The decision: the burst200 fabric, not a new 32-PC adapter

Two proven F2 paths could carry the slice. **(a) Alex's burst200 fabric** (`boards/f2/burst200`,
imported byte-identical from `e04/tpubench` 23f3f77d, commit 135cdda6): the only openTPU F2
design that has decoded on the card (build 6: one slice at 200 MHz on two HBM ports, 22.2 int8
device tok/s; build 11: three slices at 175 MHz on six ports, 48.3 int8 device tok/s, text
identical to build 6; docs/v4/f2-burst200.md, f2-burst200-s3.md). Each native channel of
`otpu_native_dram` (one command a 64-byte beat) is one `otpu_native_axi` (16-beat read
bursts, single-beat writes, a direction barrier) on its own 2 GiB HBM window and port, two
windows and the host per `f2_mem` SmartConnect (3 SI x 2 MI); P1's base descends from Alex's
7e42a626, so the native interface is the one his adapter was written for. **(b) The Alex-shell
flow** (`wide-port-alexshell`: a 1x1 SmartConnect per channel, NCH configurable, full build
and route-qualification tooling): never decoded on a card; its PORTB 4 build is stuck at
-0.064 ns.

(a) is taken. P1 at CPC 2 needs 256 B a cycle: its adapter already runs 2 CPC = 4 native
channels (milestone 2), so on burst200 the slice is four adapters on four HBM ports (0, 4, 8,
12: four distinct HBM memory channels) in two `f2_mem` groups, the host reaching both through
an `f2_host` crossbar: exactly the per-slice structure of the three-slice build, with P1's one
slice owning two groups. No new memory adapter: the slot-reserved reorder FIFO and the 32-PC
interleave of milestone 5 are not needed while four ports carry the port (4 x 64 B x 200 MHz =
51.2 GB/s at the native ports; an HBM port is 32 B x 450 MHz = 14.4 GB/s raw, and build 6
measured 89% of the native port rate on the weight stream).

### 6.2 What was built

| piece | files |
|:--|:--|
| the board at CPC 2: NCH = 2 CPC native channels a slice, CFG_DRAM four channels of 1 GiB, CAPS2 [3:1] = CPC (the wide-port line's PORTB field) | `rtl/shell/otpu_board.sv`, `rtl/shell/otpu_ctrl.sv` |
| the target's F2 geometry (channels_per_slice 4, channel_bytes 1 GiB, hbm_ports 0 / 4 / 8 / 12, rtl_board f2-burst200-p1, floorplan_p1.tcl, BUILD_ID 0x50310002) | `targets/f2.toml`, `opentpu/target.py` (rtl_board) |
| the CL top, generated: one otpu_board at CPC 2, four otpu_native_axi (BASE_ADDR j x 2 GiB), `f2_mem_s0` (ports 0 / 4) and `f2_mem_s1` (8 / 12), `f2_host` 1 x 2 and a register slice per group, SLR0's own reset synchronizers and pipes | `boards/f2/burst200/scripts/generate_f2_top.py` (grouped mode; one slice of two channels still gives Alex's top byte for byte), `rtl/boards/f2/burst200-p1/cl_opentpu.sv` |
| the flow on groups (otpu_G in geometry.tcl), the floorplan (the board, the adapters and both SmartConnects in SLR0, each SmartConnect in a grandchild by its ports, the crossbar in SLR1) | `build.py`, `create_f2_ip.tcl`, `build_f2.tcl`, `constraints/floorplan_p1.tcl` |
| build 6's two cycle-exact timing fixes in the core (the TMEM read requests replicated per 16 loads; the TMEM's rank select a register) | `rtl/dma/otpu_dma.sv`, `rtl/mxu/otpu_mxu.sv`, `rtl/vpu/otpu_vpu.sv`, `rtl/mem/otpu_tmem.sv` (3275eee5) |
| the host: the CPC 2 channel map (chunk m on pair m mod CPC, no pair hash; beat_at / split / join / Board.write / read / scrub), device_config's CPC from CAPS2 (CFG_HASH has it), the F2 transport's windows (each channel in its own 2 GiB window), `--dev f2:<slot>` for chat and hwlens | `opentpu/host/board.py`, `regs.py`, `f2.py`, `chat.py`, `hwlens.py` |
| the F2 memory path in simulation: per channel otpu_native_axi and an AXI slave model of the SmartConnect / HBM side (outstanding limits, stalls, B after the write is performed, AXI-obligation checks) in front of the native memory model: `otpu_top` AXI 3 (`OTPU_F2MEM=1`, or a run's `axi=3`), `tb_board` MEM_NATIVE 3 (the board model's transport: `SimTransport(native="f2")`) | `sim/verilator/otpu_axi_native.sv`, `rtl/top/otpu_top.sv`, `sim/verilator/tb_board.sv`, `opentpu/rtlsim.py` |
| P1's LPN (the wide-port-lpn tool with CPC as the port credit, int8 advances of CPC chunks, P1's unhashed pairs) and the packaging review | `tools/wideport_lpn.py`, `boards/f2/burst200/scripts/review_gates.py` |
| tests | `tests/test_p1_f2.py` (15), `tests/test_cpc.py` (the F2 path), `tests/test_gemma3.py` (cpc2-f2) |

### 6.3 Gates (Verilator, on the box)

Receipts are the logs under `/home/ubuntu/opt/logs/p1-*.log` (each opens with the commit and
the exact command) and the profiles under `/home/ubuntu/lens-runs/p1/`.

| gate | commit | result | log |
|:--|:--|:--|:--|
| tests/test_p1_f2.py (target, flow plan, generated top and floorplan, Alex's top regenerated byte for byte, CAPS2, the host map against the adapter's, the F2 windows, device_config; the MLP on the F2 path at CPC 1; the board model at CPC 2 reports CPC 2 / four 1 GiB channels; the tiny Gemma 3 on the board model through the host's map and the F2 path, three tokens bit-exact, DRAM equal) | 3328f88f / 17cab46d | 12 + 3 passed | p1-f2tests1.log |
| tests/test_cpc.py + tests/test_p1_f2.py (kernels at CPC 2 on the fixed, native and F2 paths; the 13 MX MM cases) | 3275eee5 | 36 passed | p1-cpcb.log |
| tests/test_gemma3.py -k token_on_rtl --runslow (the odd-block tiny model at position 600: cpc1, cpc2, cpc2-f2) | 17cab46d | 3 passed | p1-g3tiny.log |
| Gemma 3 1B int8 (weights and head), one decode token at position 14 after the 15-token chat prompt, the resident program, whole model (26 layers + head), the F2 path never withholding a handshake (stall 0, LAT 30), `--check`: the whole DRAM image against the ISA simulator | 3275eee5 | **bit-exact, next token 'Okay', 4,838,409 cycles** (17cab46d: the same count) | p1-g3f2s0b.log, p1-g3f2s0.log |
| the same token, per-position program, tools/perf_qwen.py `--check` (the LPN's calibration receipt) | 3275eee5 | bit-exact, 4,623,290 cycles | p1-pqf2pp.log |
| the CPC 1 suites (test_board, test_rtl, test_isa, test_kernels, test_mx, test_pair, test_target, test_config_matrix, test_lens, test_quant_rtl, test_stream_rtl, test_waitw_rtl, test_perf, test_tech, test_gemma3), `-n 4` | 93064ad1 | 598 passed, 9 failed, 15 skipped, 1 xfailed: the 8 `test_model[*-qwen35]` failures are the venv's transformers 4.57.6 (no `Qwen3_5TextConfig`), the ninth is section 3's pre-existing `test_a_decode_step_is_placed_best_fit_where_next_fit_fails` | p1-reg.log |
| test_config_matrix + test_gemma3 + test_gemma4 under transformers 5.20.2 | 93064ad1 | 85 passed, 2 failed, 9 skipped: both failures (`test_a_decode_step_is_placed_best_fit_where_next_fit_fails`, `test_resident_gathers_are_bit_exact[fp4]`: a compiler TMEM-exhausted error) fail the same way on bdc32a03 with the same venv | p1-reg5.log |
| test_gemma4 -k token_on_board_rtl --runslow (the RTL token through the board's memory path, CPC 1) | 93064ad1 | 3 passed | p1-g4board.log |
| the fixes of 6.5 (registered `sw_rdy`, the regenerated top): test_cpc + test_p1_f2; the tiny tokens; the whole-model F2 check | 57584b2b | 36 passed; 3 passed; bit-exact, 4,838,409 cycles (the same count) | p1-cpcc.log, p1-g3tinyc.log, p1-g3f2s0c.log |

Throughput, all **RTL whole-model simulation at 200 MHz, not FPGA inference**: on the F2 path
41.3 tok/s (4,838,409 cycles, resident, stall 0 / LAT 30); with each channel capped at 89% of a
beat a cycle (build 6's measured port utilization; `OTPU_BW=89`) 40.9 tok/s (4,887,567 cycles,
p1-g3f2bw89.log); per-position program 43.3 tok/s. The same token on the native path
(section 2.1) was 4,639,029 cycles (43.1 tok/s): the F2 path costs 4.3%, the adapter's 16-beat
read coalescing showing as rows whose chunks arrive later (Lens: "MXU not consuming" 4.4%
against 2.1%) and longer QACT waits (2.9% against 1.5%).

### 6.4 Lens and LPN

Lens (`lens summary /home/ubuntu/lens-runs/p1/p1f2b-gemma3-f2-stall0.otpuprof`): 87.8% of the
roofline (4,249,632: the MXU's advances with the half pairs), port B streaming 88.1%, chunk
FIFO starved 0.8%, MXU not consuming 4.4%, MXU computing not streaming 2.0%, waiting on the
quantizer 2.9% and the VPU 1.4%, TMEM arbitration 0.3%; the drain lost 231,076 cycles to TMEM
bank arbitration. The four channels' read beats are balanced (3.99 M, 3.99 M, 4.10 M, 4.10 M).

LPN, all **lpn-estimate** (tools/wideport_lpn.py on LPNDialect b06f42d, the hwloop wide model
ac048e4a, built on the box with conda-forge's LLVM / MLIR 20): the whole model at position 14,
per-position program, against the F2-path receipt: **4,412,249 cycles, -4.6%** (MLP -3.9%, LM
head -1.7%, attention -16.3%; mean |error| 7.3%, under the 10% gate: calibrated), 45.3 tok/s
at 200 MHz against the RTL's 43.3; at the F2 path's adapter latency (32 cycles over --lat
instead of the native path's 14) 4,415,543 (-4.5%) (`/home/ubuntu/opt/logs/p1-wplpn2.log`,
`/home/ubuntu/lens-runs/p1/p1-wplpn-perpos-al{14,32}.json`). The VLA's pi0.5 LPN
(openTPU-vla e896ab4f tools/pi05_lpn.py at its f2-vla target, which P1's change does not touch):
the whole chunk 1.477 s / 0.68 Hz (lpnlang) and 1.414 s / 0.71 Hz (dialect) at 200 MHz, the
cut phases within -17.9..+5.5% of their Verilator cycles (`/home/ubuntu/opt/logs/p1-pi05lpn.log`,
`/home/ubuntu/lens-runs/p1/pi05-lpn-{phases,full}.json`).

### 6.5 Vivado (burst200 flow, m8a, Vivado 2025.2; per-clock tables from report_timing_summary)

Clocks: the core 200 MHz (clk_out1_f2_core_clk, 5.000 ns) and the pumped TMEM's 400 MHz
(clk_out2, 2.500 ns) from one clk_wiz on clk_main_a0 (M 24.000, D 5, O 6.000 / 3, VCO 1200 MHz:
exactly 200 / 400, the link stage's VERIFIED_CORE_CLOCK 5.000 / 2.500 ns; no AVAL-168), the HBM
side's 450 MHz (clk_out1_f2_hbm_clk), the shell's clk_main_a0 250 MHz.

**Synthesis (b1, 93064ad1)**: CL 184,062 LUTs (29,940 LUTRAM), 195,575 FFs, 322 RAMB36, 1,275 DSP,
no URAM; the board 137,834 LUTs / 318 RAMB36 / 1,275 DSP (the MXU 1,122 DSP, the two ACT RAM
copies 68 RAMB36 each), the CPC 2 adapter 23,649 LUTs (11,320 LUTRAM), each SmartConnect 16,615
LUTs / 28,094 FFs (`.../p1-93064ad1/b1/synth_link/synth_utilization.rpt`). pblock_CL_SLR0 after
opt_design: LUT 33.5%, LUTRAM 23.4%, FF 21.9%, RAMB36 54.5% (327 / 600), DSP 50.1% (1,275 /
2,544) (`slr0_utilization.rpt`).

**b1** placed in 23 min, then its post_place.dcp write failed: the box's root disk filled at 17:02
UTC (the coordinator grew it to 678 GB). **b2** (the same netlist, `--from-netlist b1`):

| stage | WNS / TNS (failing) | core 200 | TMEM 400 | 400 -> 200 / 200 -> 400 | HBM 450 | host 250 | report |
|:--|:--|--:|--:|--:|--:|--:|:--|
| placed | -0.741 / -423.0 (3,160) | -0.741 (2,874) | -0.132 | -0.356 / -0.330 | +0.033 | +0.613 | b2/implement/place_timing.rpt |
| phys_opt (estimate) | +0.005 / 0 | | | | | | b2/implement/console.log |
| routed | -0.037 / -2.922 (190) | -0.031 (150) | -0.026 (7) | -0.029 / -0.037 | +0.014 | +0.164 | b2/implement/route_timing.rpt |
| refine Explore, AggressiveExplore | -0.027, -0.006 / -0.062 (20) | -0.006 (17) | +0.018 | -0.005 / met | +0.014 | +0.164 | b2/refine/route_timing.rpt |
| recover (4 post-route phys_opt passes) | -0.006 | | | | | | b2/recover/progress.txt |

b2's placed worst paths (`p1-b7ae926d/b2-place-analysis/worst200.rpt`): the board's SLR0 reset
through otpu_axil_iso's combinational readies into the OCL converter in SLR1 (-0.741, 5.07 ns of
route); the pumped TMEM's read data into the DMA / VPU (-0.356); the adapter's SW queue count
through `sw_rdy` into the quantizer's grant (its pipeline enable, 7,941 loads; -0.344); the
adapter's hazard buckets and port-A run address (up to 11 levels); its write count into the
status synchronizer in SLR1 (-0.361). phys_opt replicated the reset and the enables away; the
refined state's last 20 endpoints were the SW write's channel hash into the queue count, the
TMEM's reads into the DMA / VPU, the sequencer's issue registers and the hazard buckets.

The fixes, each gated in Verilator (6.3): the control path through `f2_ocl_pipe` in SLR0 and the
status synchronizer in SLR0 (3d982d6e), `sw_rdy` from a register (d3373cdf), the SW channel
from the slice's registered parity `sw_par` (2f7ddc9d). b3 = 57584b2b (the first two), b4 =
2f7ddc9d (all three).

**b3 (57584b2b: the OCL register slice, the status synchronizer in SLR0, `sw_rdy` registered):
timing met, all six gates pass.** Run `/home/ubuntu/opt/hwloop-nightly/p1-57584b2b/b3`:

| clock | placed (implement/place_timing.rpt) | routed (implement/route_timing.rpt) | final: refine Explore, audit (audit/timing.rpt) |
|:--|--:|--:|--:|
| design WNS / TNS, WHS | -0.482 / -3.604 (78), -0.502 (line 144) | +0.001 / 0, +0.004 (line 183) | **+0.001 / 0 (0 of 956,783), WHS +0.004** (line 183) |
| core 200 MHz (clk_out1_f2_core_clk) | -0.006 (18; line 185) | +0.001 (line 224) | **+0.001**, WHS +0.004 (line 224) |
| TMEM 400 MHz (clk_out2_f2_core_clk) | -0.118 (32; line 186) | +0.009 (line 225) | **+0.009**, WHS +0.101 (line 225) |
| 400 -> 200 / 200 -> 400 | -0.005 / -0.064 (lines 201-202) | +0.010 / +0.007 (lines 240-241) | **+0.010 / +0.007** (lines 240-241) |
| HBM 450 MHz (clk_out1_f2_hbm_clk) | +0.067 (line 183) | +0.006 (line 222) | **+0.006**, WHS +0.010 (line 222) |
| host 250 MHz (clk_main_a0) | +0.377 (line 184) | +0.158 (line 223) | **+0.158**, WHS +0.010 (line 223) |

The audit (`audit/`, `scripts/gate_summary.sh`, `scripts/review_gates.py` -> `audit/review.json`):
setup, hold and pulse width met ("All user specified timing constraints are met"); 0 nets
with routing errors (400,472 fully routed); DRC 0 errors, 0 critical warnings (835 warnings:
DPIP-2 / DPOP-3 / DPOP-4 DSP pipelining, BUFC-1, FLBO-1 x3 the SmartConnect and crossbar
grandchild pblocks as in build 11, RTSTAT-10 x1; advisories AVAL-155 / 156, HDPRA-57; no
AVAL-168); CDC 9 criticals, all False Path from the shell's hidden reset flops into the CL's
reset synchronizers, 0 outside an exception (builds 6 and 11's profile); bus skew 93 rows, 0
violated; check_timing 0 unconstrained internal endpoints, the shell's 35 no_clock pins and one
hidden shell pulse-width pin. The margins are thin (+0.001 ns setup on the core clock; build 6
shipped at +0.035, build 11 at +0.010).

**b4 (2f7ddc9d: b3 plus `sw_par`)** placed at -0.471 and routed at -0.614 ns, refine -0.546: its
placement landed worse (not packaged; `p1-2f7ddc9d/b4`).

### 6.6 The package and the card (for the coordinator)

`package_f2.py` on b3's audited checkpoint (local only: not uploaded, no AFI):
`/home/ubuntu/opt/hwloop-nightly/p1-57584b2b/b3/package/cl/build/checkpoints/otpu_p1_cpc2_57584b2b.Developer_CL.tar`
(181,831,680 B, SHA-256 eb5f7b1dee62898f19fbcc66eaba05491cfc953326a32fc923cdda5e93801cd6);
its manifest: dcp_hash 52d12bfd8c040e06dfc6c4f5e696a46798c0496b0b8e417231d3cf704ed7fd17 (the
reviewed `refine/post_route.dcp`), shell 0x10212415, HDK 2.3.0, Vivado v2025.2, PCI
1D0F:F001 / FEDD:1D51, clock recipes A1 / B2 / C0 / H2. The review JSON is the agent's reading
of the six gates; the sign-off is the coordinator's. The image's RTL is 57584b2b's; the branch's
later 2f7ddc9d (`sw_par`, gated in Verilator, its build b4 not closed) is not in it, and changes
nothing the host sees (the same CFG_HASH).

Bring-up: the host must be this branch's (its CPC 2 channel map and windows: four channels of
1 GiB at BAR4 0 / 2 / 4 / 6 GiB). An older host refuses the image: tpubench's f2.py rejects the
CFG_DRAM geometry (four 1 GiB channels) and its device_config's CPC 1 hash is not CFG_HASH
0x54839966. Checks in order: CFG_PLATFORM 3, CAPS2 [3:1] = 2, CFG_DRAM 4 channels of 1 GiB and
4 GiB a slice, CFG_HASH 54839966, BUILD_ID 0x50310002, CORE_KHZ 200000. Then
`OTPU_DEVICE=f2:<slot> python -m opentpu.host.chat --backend board --dev f2:<slot> --model gemma3
--greedy --prompt "Why is the sky blue? Answer in two sentences."` and `python -m
opentpu.host.hwlens record --dev f2:<slot> --model models/gemma-3-1b-it -o p1-hw.otpuprof`. The
image load writes 4 GiB of zeros once (scrub) and the 1.05 GB image.

What the card should show, as predictions from the RTL (not measured): about 41 device tok/s
at 200 MHz (4.84-4.89 M cycles a token on the F2 path, resident, position 14), if the four
ports deliver what build 6's two did (89% of the native port rate).

### 6.7 On the card (2026-10-21): the image hangs

The setup:

- **AFI:** afi-067e5810d0bb33969 / agfi-0097f20d1a4bfbe85 on opentpu-f2's slot 0.
- **Host:** this branch at 9950fd5d.
- **Coordinator:** card-run.sh under the canonical lock, with the baseline restored afterwards.

The identity is as built:

- CORE_KHZ 200000, BUILD_ID 0x50310002
- CAPS2 0x5 (CPC 2)
- CFG_DRAM 0x001e0420 (4 channels of 1 GiB, 4 GiB a slice)
- CFG_PLATFORM 3, CFG_HASH 54839966
- STATUS 0x68 (calibrated, idle)

The image never finishes a program:

- **otpu-chat** ("Why is the sky blue? Answer in two sentences.", int8 / int8 head):
  - It loaded in 14.5 s (image built in 7.9 s).
  - The first prefill program never halted. `board.wait` timed out after about 10 minutes: `TimeoutError: register 0xc = 0xe8, waiting for 0x1/0x1`, so STATUS had RUN set, HALTED clear, and no ERROR.
- **hwlens record**, started after it on the same image:
  - The program load (762 words from DRAM 0x3ed94000) never completed.
  - Two read-only snapshots 3 s apart gave identical values: STATUS 0x6c (LOADING), ICOUNT 0, and frozen counters (CYCLES 0x1bf08f32af, B_RD 0x1147, A_RD 0xe50, no writes).
  - AXI_ERR stayed clear, and SCRATCH = DRAM_INIT, so the scrub had finished.
  - It was stopped after 4 minutes.
- **What it means:** a DRAM read issued on the four-channel CPC 2 map gets no response on the card, both during a run and in the program loader. The Verilator gates (6.3) cannot show this: their memory always answers.
- **Not yet localized.** Candidates:
  - a port or window of hbm_ports 0 / 4 / 8 / 12 that does not answer through the HBM switch
  - the per-channel SmartConnect / ID path
  - the channel select of an address
- **Next on the card:** a register-level probe, i.e. single reads of each channel's window through BAR4 and a one-instruction program per channel.

Log: opentpu-f2 /home/ubuntu/opt/coord-bench/card-p1-57584b2b.log and bench-p1-f2/build/chat-p1-57584b2b.txt. The ISA text reference is on opentpu-m8a-build at /home/ubuntu/opt/coord-isa/isa-ref-p1-twosentences.txt.
