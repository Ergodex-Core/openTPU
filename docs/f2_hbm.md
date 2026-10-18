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
| `f2-hbm:p2` | 1 | 8 | 0-7 | 4 GiB | 4 GiB; port B two chunks a cycle (PORTB 2, docs/wide_port.md) |
| `f2-hbm16:p4` | 1 | 16 | 0-15 | 8 GiB | 4 GiB; port B four chunks a cycle (PORTB 4) |

Every one has the f2 target's arch (the YPCB card's m4: CFG_HASH 4ffd1e17 at S = 1, the same
programs), platform id 3 (opentpu/host/f2.py), core 200 MHz, TMEM_PMP 1.

## Architecture

```
            core clock (200 MHz; clk_f 400)            |  hclk (450 MHz)              | HBM
 slice s -- otpu_native_dram (NCH) --+-- ch 0: otpu_f2_hbm_nch ==afifo==> bursts -- port NCH s     (PC NCH s)
                                     +-- ch 1: otpu_f2_hbm_nch ==afifo==> bursts -- port NCH s + 1
                                     ...                                     |          ^
                                                                otpu_f2_hbm_port (2 masters, ID 0/1)
 PCIS (aclk 250 MHz) -- otpu_f2_hbm_host ==afifo==> one 64 B beat at a time ----+ (master 1 of every port)
 OCL  (aclk) -- otpu_axil_cdc (held while PCIS writes are in flight) -- otpu_ctrl (core)
```

- `rtl/mem/otpu_native_dram.sv` (the slice's DRAM adapter) takes `NCH` channels (a power of two,
  2 to 32). At NCH = 2 it is the two-channel adapter in every bit and every cycle (below).
- `rtl/shell/otpu_board.sv` takes `NCH`, `CH_LOG2`, `DRAM_LOG2` (from `opentpu.target.rtl_params`:
  `channels_per_slice`, `channel_bytes`, the arch's `DRAM_BYTES` or the platform's `slice_bytes`)
  instead of hardcoding 2 / 31 / 32; slice s is on channels NCH s .. NCH s + NCH - 1, CFG_DRAM
  reports them, CALIB0 / CALIB1 are the even / odd channels' ready.
- `rtl/boards/f2/otpu_f2_hbm_nch.sv`: one channel's native port onto one HBM port (below).
- `rtl/boards/f2/otpu_f2_hbm_port.sv`: a port shared by its bridge (ID 0) and the host (ID 1):
  AW round-robin with W locked to the granted AW to WLAST, AR round-robin, R and B by ID.
- `rtl/boards/f2/otpu_f2_hbm_host.sv`: PCIS (BAR4 offset = HBM address) onto the port that owns
  the address.
- `rtl/boards/f2/otpu_f2_hbm_mem.sv` (the fabric; `NXR` register stages each way on the native
  ports, the SLR crossings below), `otpu_axil_xr.sv` (the control path across the same SLRs),
  `otpu_f2_hbm_sys.sv` (otpu_board + fabric + OCL, `NXR` = 3), `boards/f2/design/cl_opentpu_hbm.sv`
  (the CL: clocks, resets, the HDK's `cl_mem_hbm_wrapper` with `NUM_OF_AXI_PORTS = S x NCH`,
  `AXI4_INTERFACE = 0`, the unused-DDR template for the DIMM's pins).

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

### The channel bridge (otpu_f2_hbm_nch)

Core side (200 MHz): the native port's commands and write data into two 4-entry FIFOs (readies
from flip-flops), the command FIFO's head into a register (a core cycle on a command's way),
then in command order through the **order check** into an asynchronous FIFO (`otpu_afifo`, 16 x
602 bits: we, beat, mask, data; its ready and its RAM's write enables flip-flops, the tail slot
written every cycle there is room and the issue moving only the pointer: `RWR`, `WT`) to hclk:

- a read waits while a write to its beat is in flight (issued, not yet counted in n_wdone),
- a write waits while a read of its beat is in flight (issued, data not yet returned),

so accesses to one beat reach the HBM one at a time in command order whatever the AXI port
reorders (its read and write channels are independent; the controller reorders for open pages),
which is the native port's contract (otpu_native_dram: a read after a write to its beat sees it).
This is otpu_f2_nch's rule, moved to the core clock: 16 write and 64 read entries compared at 200
MHz instead of at 450. Reads in flight are at most RD = 64 (the return FIFO's depth), so R is
never stalled; the HBM's B count steps by one an hclk cycle toward the bursts' beat totals and
crosses back gray-coded (n_wdone).

HBM side (450 MHz): the crossing's head is held in a register (its LUT RAM read off the compare;
the beat's 576 enables take the load's one net, `k_ld`, replicated by synthesis per 64 of them,
docs/wide_port.md "Round 4 RTL"), then **joins the open burst** when it continues it (same direction, next beat, below BMAX = 8
native beats = 16 AXI beats, AXI3's longest, not across a BMAX boundary so never across 4 KiB),
else the open burst closes into a 2-deep AR or AW queue and the head opens the next. An open
burst also closes after GATHER = 4 hclk cycles (~9 ns) without a command: the core sends at most
one command per 2.25 hclk cycles, so the wait lets a stream's beats join. Write data waits in a
32-entry W FIFO until its burst's AW is accepted; W follows in AW order. R beats pair into native
beats (low half first) and cross back (64 x 512 bits).

Little's law: from issue to data in the core clock ~300-400 ns (the crossings, the gather, the
HBM's latency under load). 64 reads in flight per channel sustain 64 x 64 B / 400 ns = 10 GB/s
per channel, three times a stream's share at NCH = 8 (3.2 GB/s) and 1.6 times at NCH = 4; one
channel alone at the core's full rate (12.8 GB/s) would need ~80.

### The host path (otpu_f2_hbm_host)

PCIS side (aclk, 250 MHz): every output to the shell is a flip-flop. AW, W and AR enter
two-entry skid buffers (`otpu_skid`: the ready a register, the beat the logic sees from
the buffer's output register), B and R leave through output registers (`otpu_f2_hbm_oreg`:
valid and data loaded when the shell took the last beat or none is held), so no path runs from
a shell register through the block and back in the 4 ns cycle (round 3's placement ran the
shell's PCIS register in SLR1 into the readies' LUTs in SLR0 and back at -5.01 ns,
docs/wide_port.md "Round 4 RTL"). An AR leaves its buffer for the read queue only while no
write is in flight (`pcis_wbusy`, below, counts the buffers' beats): the order rule as before,
its gate moved from the shell handshake to the buffer's exit. The instances are `u_pcis_*`, held
beside the shell by the floorplan's pb_otpu_pcis (SLR1). Since round 5 `pcis_wbusy` is registers
only (the buffers' occupancy, the queue, the unanswered B's; not the shell's live valids), and
the OCL side (otpu_axil_cdc) takes its AW, W and AR through skid buffers of its own, its state
machine starting a write only while `!hold`: the OCL-after-PCIS order holds a cycle later than
before, with no CL output on clk_main_a0 a function of a CL input (docs/wide_port.md "Round 5
RTL"). Since round 6 the PCIS registers and u_host's queues are joined by `otpu_link`s
(rtl/shell/otpu_link.sv: a credit link whose every crossing net is a flip-flop to a flip-flop
both ways; `u_pcis_*_tx / _rx` on the registers' side, `u_xs_*` the stage on u_host's), two
aclk cycles a beat each way, 4 beats in flight a channel (docs/wide_port.md "Round 6 RTL").

PCIS writes: each 64-byte W beat crosses with its address (afifo, aclk -> hclk; the crossing
written as the bridges', `RWR`, `WT`) and goes to the port of its address (`addr[33:29]`) as one
two-beat AXI3 burst, one at a time (~7 hclk cycles a beat through the ports' registers: 4 GB/s,
above BAR4 write combining's ~2 GB/s). The head beat's AW, W and AR go to every port on shared
buses with a valid per port, and **each port registers them** (`otpu_f2_hbm_hreg` in
otpu_f2_hbm_mem: a one-entry register per channel, its ready its own emptiness, so no arbiter
or HBM ready reaches the host path in a cycle; the port's R data and B registered back the
same way), so every net between the host block and a port is a register's cycle long (the
first placed build, docs/wide_port.md "Placement", ran the head's data and its half select into
every port's W register over 6.9 ns of route at 450 MHz). The host's B for a burst goes back when the
host write count (every port's ID-1 B's, gray-coded back to aclk) reaches the burst's last beat;
`pcis_wbusy` (an AW / W in a buffer or taken, a B not given back) holds the host's reads and the
OCL's writes (a RUN after data finds the data in the HBM) as on the DDR build. PCIS reads: a beat at a
time crosses as a two-beat read on its port, RQ = 16 in flight, one port at a time (a read to
another port waits for the last port's reads: one ID's reads come back in order on a port, not
across ports), so the data returns in PCIS order. Addresses past the last used port: writes
dropped, reads return zeros. The host transport maps BAR4 for S x NCH x 512 MiB.

### Clocks and CDC

| clock | source | used by |
|---|---|---|
| clk_main_a0 250 MHz (aclk) | shell | OCL, PCIS, the HBM wrapper's 250 MHz side |
| clk 200 MHz, clk_f 400 MHz | `u_mmcm` from clk_main_a0: VCO 1200, /6, /3 | the slices, the bridges' core side |
| hclk 450 MHz | `u_hmmcm` from clk_hbm_ref (100 MHz): VCO 900, /2 | the HBM AXI ports, the bridges' and host's HBM side |
| clk_hbm_ref 100 MHz | shell | the HBM's reference and APB clock |

`boards/f2/build/constraints/cl_timing_user_hbm.tcl` (applied after synthesis, in place of the
DDR build's): max delay, datapath only, between every pair of {core, main, hclk}, each direction
at the destination clock's period (a crossing's word is read only after its pointer, request or
count has passed the destination's two synchronizer flip-flops; the first placed build bounded
both directions by the smaller period of the pair, 2.2 ns on the HBM side's data into the core
across an SLR); bus skew on every otpu_afifo's gray pointers and on every write count's gray bus
(bridges and host path); the MMCMs' lock false-pathed to reset pins only.

Resets: the CL's three synchronizers (`core_reset_n`, `hbm_reset_n`, `bridge_reset_n`: ASYNC_REG
flip-flops synthesis does not replicate) each drive one register in otpu_board (`rst_q`,
max_fanout 128: the control block, the loader and port registers, the adapter's ~2,200 counters
at NCH 8) and in otpu_f2_hbm_mem (`rst_q` / `hrst_q`, max_fanout 4), and every fabric module
registers its own again (otpu_f2_hbm_nch both sides, _port, _hreg, _host both sides), so no
reset net is longer than a register's cycle and the logic leaves reset two or three cycles
after the synchronizer (nothing can ask anything in them: the control registers' first access
is cycles away, the slices are in core_rst until RUN); the board's copy of `core_reset_n`
comes through the three crossing stages below, so the board enters and leaves reset three
cycles after the fabric (idle both times). otpu_clk2x.tcl (the pumped TMEM's multicycles) as
on the DDR build: its multicycle is `-from` the core clock `-through` every `otpu_pump_bank`'s
read pins (`ra`, `rb`, `rena`, `renb`) `-to` clk_f, so every core register that reaches a pumped
bank's address (the VPU's `ta_addr`, the MXU's `t_raddr`, the slice's `rl_pq` through the
collective's port) gets the one core period, whatever the start point. The clocks are not
grouped asynchronously.

### Floorplan and SLR crossings

The VU47P has three SLRs; the HBM stacks and their AXI ports are below SLR0, the HDK's small
shell gives the CL all of SLR2, 23 clock regions of SLR1 and 20 of SLR0 (`pblock_CL`,
hdk/common/shell_stable/build/constraints/small_shell_level_1_fp_cl.xdc). Left to the placer
(docs/wide_port.md "Placement": two placed p4rl8 builds at -4.9 and -4.6 ns) one slice of 226K
LUTs was spread over two SLRs with 7,693 unregistered crossings and every wide bus of the
design (the host path's into the ports, the adapter's readies and the TMEM's grants into the
units) ran 6-8 ns of route. `boards/f2/build/constraints/cl_pnr_user_hbm.tcl`
(appended by build.py to cl_pnr_user.tcl for an HBM target, which build_level_1_cl.tcl reads
unmanaged before the HDK's opt_design, after the HDK's own constraints: a managed XDC drops
Tcl, docs/wide_port.md "Placement, round 3") pins the design by whole clock regions cut down
to the sites `pblock_CL` holds, every pblock a child of `pblock_CL` with `CONTAIN_ROUTING` off:

| pblock | hierarchy (WRAPPER/CL/u_sys/...) | clock regions | LUT budget | used |
|---|---|---|---|---|
| `pb_otpu_hbm` (SLR0) | `u_mem` (8 bridges, host registers, arbiters, the host path), `u_ocl`, the SLR0 stage of every crossing chain | X0Y0:X3Y3 + X4Y0:X7Y0 (20) | ~272K | ~31K |
| `pb_otpu_mid` (SLR1) | the middle stage of every chain | X0Y4:X3Y7 (16) | ~217K | 0 LUT, ~18K FF |
| `pb_otpu_board` (SLR2) | `u_board` (the slice, its adapter, the control block, the trace), the SLR2 stages | X0Y8:X7Y11 (32) | ~435K | 226K (52%) |
| `pb_otpu_tmem` (in `pb_otpu_board`) | `u_board/u_slice/u_tmem`, `u_vpu` | X2Y9:X5Y10 (8) | ~109K | 48K |

Everything between SLR0 and SLR2 is a chain of three `otpu_xr` register stages, one per SLR,
so each SLR crossing runs from a flip-flop to a flip-flop: the native ports (otpu_f2_hbm_mem
`NXR` = 3: down, per channel, the adapter's takes with the command and the 512-bit beat; up,
the bridge's readies, read data, write count and error; 9,088 nets; the bridge's native FIFOs
grow from 4 to 16 entries and keep 6 beats of room below their ready's threshold, since the
adapter sees a ready three cycles late and its beat arrives three later: `XL`), the control
path (`otpu_axil_xr` between the OCL's clock crossing and the board: AXI4-Lite as pulses, which
otpu_axil_cdc's one-access-at-a-time master allows) and the board's reset and HBM-ready. A port
B request costs 6 core cycles more from the adapter to the HBM and back (30 ns on the HBM's
~400; the adapter keeps 64 reads a channel in flight). The three-SLR split was chosen over
SLR1 for the board (72% of its 23 clock regions: too full for 200 MHz routing) and over the
adapter in SLR1 with the slice in SLR2 (the adapter's readies into the units and its 4,096-bit
port B data are a same-cycle handshake that cannot cross an SLR without a redesign).

## Bandwidth

| | per port | per slice, NCH = 8 | per slice, NCH = 16 | 32 ports |
|---|---|---|---|---|
| HBM peak (256 b x 450 MHz) | 14.4 GB/s | 115 GB/s | 230 GB/s | 460 GB/s |
| native port limit (64 B per core cycle per channel) | 12.8 GB/s | 102 GB/s | 205 GB/s | |
| bridge reads in flight (64 x 64 B / ~400 ns) | ~10 GB/s | ~80 GB/s | ~160 GB/s | |
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
| `tests/test_f2_rtl.py::test_f2_hbm_fabric` (`tb_f2hbm`) | 4 / 8 / 16 / 32 ports, three clock ratios, the native ports through the build's three crossing stages each way (NXR 3) and wired (0): every channel's native contract on an HBM model that reorders across IDs, runs transactions in any per-ID order and interleaves two IDs' read beats; bridges inside their pseudo-channels, bursts formed (~18% of transactions multi-beat under the mixed traffic), INCR within 4 KiB; host writes / reads on every port in PCIS order, B / R IDs, zeros past the last port | pass |
| `test_f2_ocl_crossing` (`tb_f2axil`) | the OCL's clock crossing alone (NXR 0) and through `otpu_axil_xr`'s three stages (3): 300 write / read pairs each, backpressure both sides, hold, errors, the other windows | pass |
| `tests/test_f2_build.py::test_hbm_floorplan_pblocks_are_one_slr_each_and_fail_closed` | the floorplan file under a Tcl mock: four pblocks under `pblock_CL`, whole clock regions one SLR each cut down to the parent's sites, stage k of every chain in SLR k, routing not contained; a missing hierarchy stops it | pass |
| `test_link_wrapper_reads_the_placement_tcl_unmanaged_before_opt_design` | build_level_1_cl.tcl under a mock HDK sequence: cl_pnr_user.tcl read `-unmanaged` after the HDK's managed user XDC and before its opt_design, once; no opt_design or a Tcl error stops the build | pass |
| `test_f2_hbm_fabric_detects_missing_hazards` | HAZ = 0 returns stale data | pass |
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

build.py checks the HDK and the cl_hbm IP (both stacks, 450 MHz, switch on, generated), copies the
design and build directories, puts `cl_opentpu_hbm.sv`, `synth_cl_opentpu_hbm.tcl` and
`cl_timing_user_hbm.tcl` in the DDR files' places, copies the HDK's `cl_mem_hbm_wrapper.sv`,
writes `opentpu_geometry.vh` (S, NCH) and `opentpu_params.vh` (otpu_f2_hbm_sys's parameters), and
runs the HDK's build_all.tcl with the clock recipes A1 B2 C0 H2 (H2: the HBM recipe the
manifest records, as Alex's image). `--package` produces the Developer_CL tar for
`aws ec2 create-fpga-image`.

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

1. Place and route (`--flow BuildAll --package` above, hours) with the floorplan above (an S = 2
   build needs a second board pblock: SLR1's 23 clock regions hold a PORTB 1 slice of ~150K LUTs
   at 50%, with its own crossing chains). Expected timing risks: the 450 MHz
   side of 16-32 bridges and port arbiters along the HBM's ports in SLR0 (AXI register slices
   before the ports if needed); the core clock at 200 MHz with the adapter's NCH-wide B merge (a 4:1 or
   8:1 mux after the B FIFO's LUT RAM read) and the order check's 64-entry compare.
2. On the card: the vLED (bit 0 MMCMs locked, bit 1 HBM ready), CFG_DRAM / CFG_SLICES, a BAR4
   write / read of every port, `otpu-diag --mem`, then the token-exact checks (f2 docs).
3. The word-address ISA change above, for an 8 GiB slice (the 27B in one slice).
