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
  the simulation's memory path (`--axi`), issues a wide read as two consecutive beats a channel
  from one B-queue entry (`b_iss`, two response slots reserved at the first), and the merge pops
  two beats a channel and returns both chunks, each with its own CHASH swap (the second chunk's
  parity computed in the adapter: a simulation stand-in; the HBM adapter of milestone 5 keeps a
  pair on one pseudo-channel). The native memory model (`otpu_native_mem`) is beat-level and
  unchanged.
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

| memory model | cycles | ms at 200 MHz | tok/s at 200 MHz | DRAM bound (3.93 M) | roofline (the MXU's half pairs, 4.24 M) | MXU not consuming (`mb`) | MACs / cycle | bit-exact |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| stall 0, LAT 30 (HBM-like) | TBD | | | | | | | |
| stall 5, LAT 30 | TBD | | | | | | | |
| stall 20, LAT 20 (the card's DDR3) | TBD | | | | | | | |
| CPC 1, stall 0, LAT 30 (m0's reference) | 8,221,777 | 41.1 | 24.3 | 7.85 M (95.5%) | | 2.1% | 99 | |

The ISA simulator's analytic timing of the same token (units serialized, no RTL overlap and no
RTL loss): at CPC 2 (`OTPU_TARGET=f2 --isa`) TBD cycles (its half pairs: 4.24 M of MM work);
the README's `--candidate C=2,MHZ=200` flat model 4.36 M.

TBD: the reading.

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
chunks. The per-row fixed cost (0.41 cycles a row at stall 0 at CPC 1: 0.24 `mb` + 0.17 "MXU
computing, not streaming") is per advance stream here, not per chunk: section 2.1's `mb` and
"computing, not streaming" rows measure it at CPC 2.

## 3. Tests run

At CPC 1 (must be unchanged, bit-exact and cycle-exact): TBD (the list of files and results).

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
- The native adapter's wide path is a simulation stand-in (two beats a channel per pair, the
  second chunk's parity computed in the adapter); at CPC 1 its B-queue and order entries carry
  a constant-zero `wide` bit (and the order entry a constant swap bit) that synthesis removes,
  and the slice's `b_wide` output is constant zero: the card's build is otherwise what it was,
  and its simulations are cycle-exact to before. The card's netlist has not been re-run through
  Vivado in this milestone.

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
PYTHONPATH=. python -m opentpu.lens summary build/lens/p1-gemma3.otpuprof
```

The build host's `models/gemma-3-1b-it` is a symlink to the checkpoint. Each RTL run of the
token is 10-25 minutes on the Verilator board model (the Python side holds the fp32 weights,
the ISA machine's DRAM and the RTL's: about 10 GB).
