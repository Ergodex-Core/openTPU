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
- **The HBM adapter** (`otpu_hbm_dram`: 32 pseudo-channels, a pair per 8-beat burst on one PC,
  the slot-reserved reorder FIFO, the async crossings; proposal milestone 5), the F2 shell
  (`rtl/boards/f2`, `boards/f2`), the host's channel map, `make synthcheck` on the xilinxus
  cells (`tests/test_tech.py` on DSP48E2 for the second column group), the bitstream and the
  AFI (milestone 6), Lens on the F2 (milestone 7).
- **The device's configuration register**: CPC is in `rtl_params` and the arch hash but no
  CFG_* field reports it yet (`opentpu/host/board.py device_config` builds a CPC 1 Config); an
  F2 bitstream needs the field (proposal section 3).
- The native adapter's wide path is a simulation stand-in (four channels, the second chunk's
  parity computed in the adapter); at CPC 1 (two channels) its B-queue and order entries carry
  constant-zero `wide` / pair / swap bits that synthesis removes, and the slice's `b_wide`
  output is constant zero: the card's build is otherwise what it was, and its simulations are
  cycle-exact to before. The card's netlist has not been re-run through Vivado in this
  milestone.

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
