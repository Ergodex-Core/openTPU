# openTPU on the F2's HBM (f2-hbm, f2-hbm16)

The F2 slot's xcvu47p has two 8 GiB HBM2 stacks: 32 pseudo-channels of 512 MiB, each with its
own AXI3 port (256 bits at 450 MHz, 14.4 GB/s). The DDR build (targets/f2.toml, docs/f2.md)
runs one slice at 100 MHz on the slot's DIMM through one 512-bit sh_ddr port; Alex's f2-hbm200
image (agfi-0d06cb48544a02106, no source here) runs one slice at 200 MHz on two HBM ports. This
build puts **NCH channels per slice on NCH HBM ports**, each channel on its own pseudo-channel,
and runs S slices (S x NCH <= 32) at 200 MHz with the TMEM pumped at 400 MHz.

| target | S | NCH | ports | DRAM per slice (channels) | what programs address |
|---|---|---|---|---|---|
| `f2-hbm` | 1 | 8 | 0-7 | 4 GiB | 4 GiB |
| `f2-hbm:s2` | 2 | 8 | 0-15 | 4 GiB | 4 GiB |
| `f2-hbm:s4` | 4 | 8 | 0-31 | 4 GiB | 4 GiB |
| `f2-hbm16` | 1 | 16 | 0-15 | 8 GiB (CFG_DRAM 33) | 4 GiB (see "Past 4 GiB a slice") |

Every one has the f2 target's arch (the YPCB card's m4: CFG_HASH 4ffd1e17 at S = 1, the same
programs), platform id 3 (opentpu/host/f2.py), core 200 MHz, TMEM_PMP 1.

## Architecture

```
            core clock (200 MHz; clk_f 400)                              | 450 MHz (vendor IP)        | HBM
 slice s -- otpu_native_dram (NCH) --+-- ch 0: otpu_native_axi --+-- otpu_f2_hbm_port --> cl_axi_sc_1x1 --> port NCH s     (PC NCH s)
                                     +-- ch 1: otpu_native_axi --+-- otpu_f2_hbm_port --> cl_axi_sc_1x1 --> port NCH s + 1
                                     ...                   host's |
 PCIS (aclk 250 MHz) -- otpu_f2_hbm_host ==afifo==> one 64 B beat at a time, per port ----+ (master 1 of every port)
 OCL  (aclk) -- otpu_axil_cdc (held while PCIS writes are in flight) -- otpu_ctrl (core)
```

Since round 5 ([v4/f2-round5.md](v4/f2-round5.md)) nothing of openTPU's runs on the HBM's 450
MHz clock: every block of the fabric is at the core clock, and the HDK's `cl_axi_sc_1x1`
SmartConnect block design, one per port, does the crossing into 450 MHz, the 512 -> 256 width
conversion and the AXI4 -> AXI3 conversion into the HBM wrapper's port (as the HDK's
cl_mem_perf and cl_dram_hbm_dma use it). Alex's burst200 image, the only openTPU F2 image that
closed 200 MHz, has this structure on two channels.

- `rtl/mem/otpu_native_dram.sv` (the slice's DRAM adapter) takes `NCH` channels (a power of two,
  2 to 32). At NCH = 2 it is the two-channel adapter in every bit and every cycle (below).
- `rtl/shell/otpu_board.sv` takes `NCH`, `CH_LOG2`, `DRAM_LOG2` (from `opentpu.target.rtl_params`:
  `channels_per_slice`, `channel_bytes`, the arch's `DRAM_BYTES` or the platform's `slice_bytes`)
  instead of hardcoding 2 / 31 / 32; slice s is on channels NCH s .. NCH s + NCH - 1, CFG_DRAM
  reports them, CALIB0 / CALIB1 are the even / odd channels' ready.
- `rtl/boards/f2/otpu_native_axi.sv`: one channel's native port as an AXI4 master of 512 bits
  at the core clock (below).
- `rtl/boards/f2/otpu_f2_hbm_port.sv`: a port's AXI4 slave side (the SmartConnect's) shared by
  its channel's master (ID 0) and the host (ID 1): AW round-robin with W locked to the granted
  AW to WLAST, AR round-robin, R and B by ID.
- `rtl/boards/f2/otpu_f2_hbm_host.sv`: PCIS (BAR4 offset = HBM address) onto the port that owns
  the address.
- `rtl/boards/f2/otpu_f2_hbm_mem.sv` (the fabric), `otpu_f2_hbm_sys.sv` (otpu_board + fabric +
  OCL), `boards/f2/design/cl_opentpu_hbm.sv` (the CL: clocks, resets, a `cl_axi_sc_1x1_wrapper`
  per port, the HDK's `cl_mem_hbm_wrapper` with `NUM_OF_AXI_PORTS = S x NCH`, `AXI4_INTERFACE =
  0`, the unused-DDR template for the DIMM's pins).

### Address map

Logical (what a slice's programs and `Board.write / read` use): byte address `a` of slice s, chunk
`m = a / 128` (word address bits 31:5), beat `b = a / 64 % 2` in it. With `NP = NCH / 2` pairs:

- pair `p = (m mod NP) ^ fold(m / NP)` (CHASH; `fold(u)`: the XOR of u's log2(NP)-bit groups),
- channel `c = 2p + (b ^ parity(m))`, channel beat `u = m / NP`,
- HBM port and pseudo-channel `P = NCH s + c`, HBM byte `P x 512 MiB + u x 64 + a mod 64`
  (= the PCIS BAR4 offset the host writes).

At NCH = 2 this is the DDR boards' map (`c = b ^ parity(m)`, `u = m`). Every bridge's access stays
in its own pseudo-channel: the cl_hbm IP has the global address switch on (USER_SWITCH_ENABLE_00
/ 01 TRUE, 34-bit addresses: any port reaches all 16 GiB, through the lateral switch with added
latency and shared bandwidth, and across the stacks at a fraction of it), and this build never
uses it -- each channel's traffic, and the host's to that channel, go through the port that owns
the pseudo-channel. `tb_f2hbm` checks it (an AW / AR of a bridge outside its 512 MiB fails).

**Why a chunk on a pair, rotating over the pairs.** A chunk is 128 bytes (D = 128: port B's unit,
one a core cycle), two 64-byte beats; a 64-byte beat is two of the HBM port's 32-byte AXI beats,
its smallest efficient access. Splitting a chunk over more than two channels would mean 32-byte
native beats (one AXI beat each: half the commands' efficiency, twice the order checks and
tags). Keeping a chunk on two channels and rotating successive chunks over the NCH / 2 pairs
spreads a stream evenly (every channel one beat in NCH / 2 chunks: at NCH = 8 a channel carries
3.2 GB/s of the slice's 25.6), keeps the B merge a two-channel join (the tag FIFO holds the
pair), keeps each bridge a plain one-beat-per-command master, and gives each channel's
consecutive beats to consecutive chunks of its pair, so a channel's A run (APF beats) and a
stream's beats on it are runs of channel beats that the bridge turns into bursts. The fold
spreads power-of-two strides of chunks (a transposed V's column: one byte per cache row) over all
pairs, as CHASH's parity spreads them over a pair's two halves.

At NCH > 2 port B's request passes a register in the adapter (BREG): the pair's fold (27 address
bits deep) would otherwise run at the end of the slice's grant chain into every queue write.
b_rdy reserves room for it (QD - 1, OD - 1). One cycle on port B's reads against the HBM's
hundreds of nanoseconds.

### The channel bridge (otpu_native_axi)

Alex's burst200 adapter, taken as it closed and ran on his card (the channel's size a
parameter, the read return registered). Commands and write data enter two 16-entry LUT RAM
queues; **reads of adjacent beats are coalesced** into one AXI4 burst of up to 16 x 64 B (1 KiB;
the burst goes when it is full, when the head command does not continue it, or after 4 idle
cycles), **writes are single-beat**; a **direction barrier** keeps the native port's
read-after-write contract (otpu_native_dram: a read after a write to its beat sees it) whatever
the port reorders between its read and write channels: a read is collected only while no
earlier write is unanswered (its B means the HBM holds it), a write goes only while no earlier
read is outstanding. All reads use ID 0 and return in order; a lengths FIFO checks RLAST; 128
beats of read credit bound the return storage the adapter reserves (the response is never
stalled). The SmartConnect behind the port splits a 1 KiB burst into two AXI3 bursts of 16 x 32
B. The barrier is coarser than the per-beat hazard tables of the retired `otpu_f2_hbm_nch`: a
direction change drains the outstanding reads (~hundreds of ns), which Alex's measured 22 GB/s
on the Gemma decode includes.

Little's law: from issue to data in the core clock ~300-400 ns (the SmartConnect's entry and
exit pipelines and crossing, the HBM's latency under load). 128 beats in flight per channel
sustain 128 x 64 B / 400 ns = 20 GB/s per channel, above the port's 12.8 GB/s and six times a
stream's share at NCH = 8 (3.2 GB/s).

### The host path (otpu_f2_hbm_host)

PCIS writes: each 64-byte W beat crosses with its address (afifo, aclk -> the core clock) and
goes to the port of its address (`addr[33:29]`) as one single-beat AXI4 write of the whole 512
bits with the PCIS beat's strobes (a narrow PCIS write is a beat with the lanes it covers; the
address goes out 64-byte aligned, so the SmartConnect's downsizer sends two aligned 32-byte
beats), one at a time (a beat per ~5 core cycles: 2.5 GB/s, above BAR4 write combining's ~2
GB/s). The host's B for a burst goes back when the host write count (every port's ID-1 B's,
gray-coded back to aclk) reaches the burst's last beat; `pcis_wbusy` (an AW / W in its skid
buffer, link or queue, a B not given back: registers only) holds the host's reads and the OCL's
writes (a RUN after data finds the data in the HBM) as on the DDR build. The PCIS side: AW, W
and AR enter two-entry skid buffers (`otpu_skidbuf`), B and R leave through output registers,
and an `otpu_link` per channel (a registered take, a registered stage, a registered credit
pulse back) joins those registers, which the floorplan puts beside the shell, to the queues in
SLR0: +2 aclk cycles each way, every net across the boundary a flip-flop to a flip-flop
([v4/f2-timing.md](v4/f2-timing.md) "Round 3"). PCIS reads: a beat at a time crosses as a
single-beat read on its port, RQ = 16 in flight, one port at a time (a read to another port
waits for the last port's reads: one ID's reads come back in order on a port, not across
ports), so the data returns in PCIS order. Addresses past the last used port: writes dropped,
reads return zeros. The host transport maps BAR4 for S x NCH x 512 MiB.

### Clocks and CDC

| clock | source | used by |
|---|---|---|
| clk_main_a0 250 MHz (aclk) | shell | OCL, PCIS, the HBM wrapper's 250 MHz side |
| clk 200 MHz, clk_f 400 MHz | `u_mmcm` from clk_main_a0: VCO 1200, /6, /3 | the slices, the fabric (masters, arbiters, the host path's back half), the SmartConnects' `aclk_250` pins |
| hclk 450 MHz | `u_hmmcm` from clk_hbm_ref (100 MHz): VCO 900, /2 | the HBM AXI ports, the SmartConnects' `aclk_450` sides (vendor IP only; the wrapper's reset synchronizer `hbm_reset_n` is the one openTPU register on it) |
| clk_hbm_ref 100 MHz | shell | the HBM's reference and APB clock |

`boards/f2/build/constraints/cl_timing_user_hbm.tcl` (applied after synthesis, in place of the
DDR build's): max delay, datapath only, between every pair of {core, main, hclk} at the
destination clock's period (a crossing's word is read only after its pointer or count has gone
through the destination's two synchronizer flip-flops; the SmartConnects' own, more specific
constraints take precedence on their crossings); bus skew on the host path's three otpu_afifos'
gray pointers and on its write count's gray bus at the smallest period; a check that no openTPU
cell is clocked by hclk (`OTPU_HBM_NO_HCLK_RTL`), failing closed; the MMCMs' lock false-pathed
to reset pins only. otpu_clk2x.tcl (the pumped TMEM's multicycles) as on the DDR build. The
clocks are not grouped asynchronously.

## Bandwidth

| | per port | per slice, NCH = 8 | per slice, NCH = 16 | 32 ports |
|---|---|---|---|---|
| HBM peak (256 b x 450 MHz) | 14.4 GB/s | 115 GB/s | 230 GB/s | 460 GB/s |
| native port and SmartConnect slave limit (64 B per core cycle per channel) | 12.8 GB/s | 102 GB/s | 205 GB/s | |
| master's reads in flight (128 x 64 B / ~400 ns) | ~20 GB/s | | | |
| what the slice consumes at 200 MHz: port B, a 128 B chunk a cycle | | 25.6 GB/s | 25.6 GB/s | |

A slice streams at most 25.6 GB/s (port B; port A's scale lines are a small fraction, held in
runs); at NCH = 8 that is 3.2 GB/s per channel, 22% of a port: the HBM side has 4.5x headroom,
so a slice's DMA is limited by the core's port width, not memory, and HBM efficiency (bank
conflicts, refresh, the 30-40% loss of random 64-byte accesses) has room to spare. The DDR
build, for comparison: one DIMM port at 16 GB/s shared by both channels, a 100 MHz core asking
12.8. Four slices consume 102 GB/s of the 460.

## Past 4 GiB a slice

Everything below the slice handles more than 4 GiB per slice: the adapter's chunk index is word
address bits 31:5 at NCH > 2 (16 GiB), the channel beat fits 25 bits up to NCH = 32, the bridges'
AXI addresses are 34 bits, otpu_board takes DRAM_LOG2 up to 34 (CFG_DRAM), the host's map
(`beat_at`, `split`, `join`, `Board.write / read`) takes logical addresses past 4 GiB (tested at
NCH 8 / 16 with addresses above 4 GiB: `tb_native_nch` HIGH, `test_f2_host`). f2-hbm16's slice has
8 GiB in its channels (CFG_DRAM 33; `regs.cfg` reports `DRAM_BYTES` 4 GiB and `slice_bytes` 8).

**The slice itself addresses 4 GiB**: DRAM addresses are byte addresses in 32-bit registers
(docs/isa.md: "DRAM: byte addressed"), and the slice turns them into 32-bit word addresses by a
shift, so bits 33:32 never exist:

- `rtl/dma/otpu_dma.sv`: `ic <= su_src >> 2` (loads, line 773), `ds_wb / ds_wa <= su_dst >> 2`
  (stores, 780-781), `a = cmd.w1 >> 2` (720);
- `rtl/mxu/otpu_mxu.sv:401-403`: `b_addr = chunk_addr >> 2`, `a_addr = scale_addr >> 2` (and the
  row scale address): the weight stream and its scales;
- `rtl/vpu/otpu_quant.sv:634, 662`: QST's scale and data addresses;
- `rtl/top/otpu_slice.sv:361`: the loader (`ld_addr`, PROG_ADDR, a 32-bit register) and HALT's
  chain address R[ra];
- `opentpu/target.py` refuses DRAM_BYTES over 4 GiB, and the ISA simulator's DRAM is the
  Config's DRAM_BYTES.

The smallest fix is to make every DRAM operand a **word address** (byte address / 4) in the ISA:
every DRAM access is word aligned already (chunks and rows 128-byte, scales and QST words 4-byte),
so the registers lose nothing and reach 16 GiB. In the RTL it is the removal of those six shifts
(the registers then hold the word address the ports take); in the toolchain the compiler's and
isasim's DRAM address arithmetic in units of 4 bytes, the loader's PROG_ADDR and the host's
program and chain addresses divided by 4, target.py's limit at 16 GiB, and DRAM_BYTES 8 GiB for
f2-hbm16. (A per-slice window register, the reserved CFG_WIN offsets 0xA0 + 4s, does not serve a
27B model whose one run streams all of its 7.3 GiB.) That ISA change is not made here (isa.py,
isasim.py, compiler.py and rtl/mxu are other work's). Until it is, f2-hbm16 runs programs of up
to 4 GiB interleaved over all 16 channels (the low 256 MiB of each), and the 27B's t2 image
(~7.3 GiB, one slice: its 1024-wide Hadamard blocks against the FFN's 17408) fits only after that
change, or split over slices (f2-hbm:s2, 3.65 GiB a slice) once its FFN is padded.

## Tests

| test | what | result |
|---|---|---|
| `tests/test_native_nch.py` (`tb_native_nch`) | NCH = 2: every output of the adapter equals the two-channel adapter's (git 4e9616a4) in every cycle, random traffic on all three ports, random native stalls, latencies, a_flush; CHASH on and off | pass |
| same | NCH 2/4/8/16 (CHASH on, and 8 off): reads of unwritten beats find init(logical beat) under the host's map, B reads see B writes, A runs / reuse right, every written byte where the map puts it; NCH 8 / 16 with port B above 4 GiB; a testbench map without the fold fails | pass |
| `tests/test_rtl.py` with `OTPU_NCH=8` (otpu_native_mem on 8 channels) | board memory path stress / fuzz, native memory path, logits fill, DRAM range to 4 GiB, axi_* (scale runs, SW gather / RMW, write bursts, partial beats beside DMA writes), port A host writes: 58 tests | pass |
| `OTPU_NCH=4 OTPU_AXI=1 OTPU_BOOT=1` | MLP / attention decode / two-slice fuzz / scoreboard stress through the adapter: 14 tests | pass |
| `OTPU_NCH=16` | the same 58 as at 8 (rtlsim's per-channel statistics now parse two-digit channels) | pass |
| no `OTPU_NCH` (NCH = 2) | the same 58, on the generalized adapter, top and model | pass |
| `tests/test_f2_rtl.py::test_f2_hbm_fabric` (`tb_f2hbm`) | 4 / 8 / 16 / 32 ports, three clock ratios, through `tb_f2hbm_sc` (a behavioural stand-in per port for the HDK's SmartConnect, which has no Verilator model: the crossing, 512 -> 256, the AXI3 split, in-order responses) onto an HBM model that reorders reads against writes: every channel's native contract; every port's traffic inside its pseudo-channel, INCR within 4 KiB, runs of reads as bursts; host writes / reads on every port in PCIS order, B / R IDs, zeros past the last port | pass |
| `test_f2_hbm_fabric_detects_missing_hazards` | HAZ = 0 (no direction barrier in the masters) returns stale data | pass |
| `tests/test_f2_host.py` | S x NCH geometry (1x8, 2x8, 1x16, 4x8): bases, Board.write / read round trip with unaligned edges, every byte at beat_at's place; beat_at is the testbench's (RTL's) inverse | pass |
| `tests/test_f2_build.py` | the HBM snapshot (top, scripts, constraints, wrapper, geometry, parameters, sources) for f2-hbm, :s2, :s4, f2-hbm16; the DDR snapshot without HBM files | pass |
| existing F2 tests (DDR: tb_f2nch_protocol, tb_f2mem, tb_f2axil, build, host) | unchanged | pass |

Commands (in the tree, `/Users/.../bonsai2-venv/bin/python -m pytest`; pyproject's pythonpath
makes `import opentpu` the tree's):

```
python -m pytest -q tests/test_native_nch.py tests/test_f2_rtl.py tests/test_f2_host.py tests/test_f2_build.py
OTPU_NCH=8 python -m pytest -q tests/test_rtl.py -k "board_memory_path or native_memory_path or axi_ or porta or logits_fill or dram_range"
OTPU_NCH=4 OTPU_AXI=1 OTPU_BOOT=1 python -m pytest -q tests/test_rtl.py -k "mlp_rtl or attention_decode_rtl or fuzz_two_slices or scoreboard_stress_two"
```

## Build

```
python boards/f2/build.py --hdk /home/ubuntu/aws-fpga --target f2-hbm16 --out RUN --flow SynthCL
python boards/f2/build.py --hdk /home/ubuntu/aws-fpga --target f2-hbm16 --out RUN2 --flow BuildAll --package
```

build.py checks the HDK, the cl_hbm IP (both stacks, 450 MHz, switch on, generated) and the
cl_axi_sc_1x1 block design with its wrapper (`check_hbm_sc`: the interfaces' protocols, widths,
ID widths and clock pins as the CL instantiates them), copies the design and build directories,
puts `cl_opentpu_hbm.sv`, `synth_cl_opentpu_hbm.tcl` and `cl_timing_user_hbm.tcl` in the DDR
files' places, appends the chosen floorplan (`--floorplan five | slr0 | none`) to
`cl_pnr_user.tcl`, copies the HDK's `cl_mem_hbm_wrapper.sv`, writes `opentpu_geometry.vh` (S,
NCH) and `opentpu_params.vh` (otpu_f2_hbm_sys's parameters), and runs the HDK's build_all.tcl
with the directives (`--place-directive`, `--phys-directive`, `--route-directive`, `--retime`;
`--recommended` = Alex's ExtraNetDelay_high / AggressiveExplore / Explore / retime) and the clock
recipes A1 B2 C0 H2 (H2: the HBM recipe the manifest records, as Alex's image). `--package`
produces the Developer_CL tar for `aws ec2 create-fpga-image`. The synthesis script reads the
block design and its wrapper from the HDK's `HDK_BD_SRC_DIR` / `HDK_BD_GEN_DIR` as cl_mem_perf's
does ([v4/f2-round5.md](v4/f2-round5.md)).

### Synthesis (f2-hbm16: S = 1, NCH = 16; f2-hbm: S = 1, NCH = 8)

`build.py --target f2-hbm16 --flow SynthCL` (and `--target f2-hbm`) on the build instance (Vivado
2025.2, HDK 2.3.0, synth_design out of context, 6 threads, nice 10; runs in
`/home/ubuntu/opt/f2hbm-build/runs/`): elaboration and synthesis with 0 errors and 0 critical
warnings, the openTPU CDC constraints applied without a warning (f2-hbm16: 35 crossing FIFOs and
17 write counters bus-skewed; the pumped TMEM's 16 RAMs' multicycles), the HDK footer passed
(`OTPU_STAGE_SYNTHESIS_PASSED`), post_synth.dcp written. 18-50 minutes a run (two at once).

| at 4bef3758 | f2-hbm16 (1 x 16) | f2-hbm (1 x 8) | of xcvu47p |
|---|---|---|---|
| CLB LUTs | 241,392 (18.5%) | 174,919 (13.4%) | 1,303,680 |
| of them LUT RAM | 78,644 | 48,051 | 600,960 |
| registers | 182,318 (7.0%) | 137,674 (5.3%) | 2,607,360 |
| RAMB36 | 261.5 (13.0%) | 261.5 | 2,016 |
| DSP48E2 | 727 (8.1%) | 727 | 9,024 |
| HBM AXI ports | 32 (16 used) | 32 (8 used) | 32 |
| MMCM | 2 | 2 | 12 |
| setup WNS, post-synthesis estimate | +0.126 ns (all clocks) | +0.154 ns (all clocks) | |

By block (f2-hbm16): the slice 99K LUTs (as in Alex's image), its adapter 79K (35K LUT RAM: 16
channels' B / A / SW FIFOs and queues; 39K at NCH = 8), a bridge 3.3K (16: 53K), the host path
2K, a port arbiter 43, the HBM wrapper 2.1K. Alex's S = 1, 2-channel CL was 131K LUTs. S = 4 x 8
would be ~4 x 145K + the wrapper, ~45% of the device, before floorplanning.

Timing, post-synthesis (no placement, so an estimate): the core 200 MHz +1.10 ns, clk_main
+2.49, the 450 MHz domain +0.126 ns, every crossing within its bound; the pumped TMEM pair's
hold (-0.26 ns) and hclk's (-0.095) are pre-placement artefacts (the DDR build shows the first).
Three runs got there: the first (f278ff5f) had the 450 MHz domain at -0.913 ns, 21,973 endpoints,
every path in the host path (the crossings' gray compare and LUT RAM read through the port decode
into the ports' arbitration and back, 10 levels); with its head write, head read and R beat in
registers (89bbb6bc) -0.509; with the host's AR valid and B count from registers, the arbiter's W
readies from its own state and the bridges' R registered (6fac2d06) -0.189, 20,418 endpoints, all
in the bridges' burst builder (the head's 25-bit compare with the open burst's next beat into the
held beat's 600 enables and the W FIFO's); with that compare made as the head loads and the W
FIFO's tail written every cycle (4bef3758) every clock meets setup.

## Remaining steps to a full build

1. Place and route (`--flow BuildAll --package` above, hours): builds 1-4 failed timing in the
   fabric's 450 MHz RTL ([v4/f2-timing.md](v4/f2-timing.md)); round 5 moved every openTPU block
   to the core clock behind the HDK's SmartConnects ([v4/f2-round5.md](v4/f2-round5.md): the
   build command with Alex's tuning). What remains at 200 MHz is the slice's own (the TMEM pump's
   paths, the MXU's request registers: the TMEM box of the five-pblock floorplan) and the
   adapter's NCH-wide B merge; the SmartConnects' size per port is read from the first
   synthesis (for 16-32 ports the SLR0 box overflows and the slices spill into SLR1 / SLR2, the
   SmartConnects staying by the ports).
2. On the card: the vLED (bit 0 MMCMs locked, bit 1 HBM ready), CFG_DRAM / CFG_SLICES, a BAR4
   write / read of every port, `otpu-diag --mem`, then the token-exact checks (f2 docs).
3. The word-address ISA change above, for an 8 GiB slice (the 27B in one slice).
