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
| `f2-hbm:p4rl8a64c150` | 1 | 8 | 0-7 | 4 GiB | 4 GiB; the p4rl8a64 slice at a 150 MHz core (TMEM 300; the variant's `core_mhz`, docs/wide_port.md "Alex's shell at 150 MHz") |

The host's PCIS has its own HBM port after the channels' (port S x NCH): S x NCH <= 31, so
`f2-hbm:s4` (32 channels) is refused since the rework below. build.py takes a 200, 175 or 150 MHz
core (its HBM_CORE_MMCM: VCO 1200 /6 /3, PFD 50 VCO 1050 /6 /3, VCO 1200 /8 /4) into the CL's
u_mmcm through opentpu_geometry.vh, the block design's core_clk and CORE_KHZ.

Every one has the f2 target's arch (the YPCB card's m4: CFG_HASH 4ffd1e17 at S = 1, the same
programs), core 200 MHz, TMEM_PMP 1. Their platform ids are f2-hbm's 5 and f2-hbm16's 6, not f2's 3:
opentpu/host/f2.py takes every F2 target's (PLATFORMS). Before that change, it took only 3 and
refused 417243f1 on the card (2026-10-21).

## Architecture

Since 2026-10-19 (branch wide-port-alexshell) the memory path is **Alex's burst200 HBM shell**
(ergodex-openTPU, branch `alex-burst200`, commit **06c169cb**: the design his F2 image closed
timing with, WNS +0.032 ns, and ran Gemma-3-1B int8 at 22.3 tok/s on two channels), scaled from
his two channels to NCH: per channel his native-to-AXI adapter at the core clock and a
SmartConnect for the clock and width conversion, so **no logic of ours runs at 450 MHz** (rounds
3-8 of the placed builds held our own 450 MHz domain, the bridges, arbiters and host path: 117K
endpoints, at -2.2 ns for four rounds; Alex's 450 MHz logic is the SmartConnects and the HDK
wrapper, docs/wide_port.md "Alex's shell").

```
  SLR2 (core clk)                     SLR1      SLR0, core clk                            | 450 MHz (hclk)                 | HBM
 slice -- otpu_native_dram (NCH) --+ xr - xr - xr -- u_naxi 0 -- rs512 -- sc_c0 (1 x 1, clock + width) -- rs256 -- port 0 (PC 0)
                                   + xr - xr - xr -- u_naxi 1 -- rs512 -- sc_c1 ...                     -- rs256 -- port 1 (PC 1)
                                   ...                                                                            ...
 PCIS (aclk 250, SLR1) -- otpu_f2_pcis_gate (skids; AR held behind writes) ==SLR==> rs512 -- cl_axi_sc_1x1 (AWS's) -- rs256 -- port NCH
                                                                     (the global address switch: port NCH reaches every PC)
 OCL  (aclk) -- otpu_axil_cdc (held while PCIS writes are unanswered) -- otpu_axil_xr (3 stages) -- otpu_ctrl
```
(rs512 = the HDK's prebuilt axi_register_slice, 512-bit AXI4; rs256 = its cl_axi3_256b_reg_slice,
256-bit AXI3; the shape of AWS's cl_dram_hbm_dma, hdk/cl/examples/cl_dram_hbm_dma/design/
cl_hbm_axi4.sv. Until the first placed build, cd16e195, each channel had a 2 x 1 SmartConnect
shared with the host's branch and the host a 1 x NCH SmartConnect demux: "Since the first placed
build" below.)

- `rtl/mem/otpu_native_dram.sv` (the slice's DRAM adapter) takes `NCH` channels (a power of two,
  2 to 32). At NCH = 2 it is the two-channel adapter in every bit and every cycle (below).
- `rtl/shell/otpu_board.sv` takes `NCH`, `CH_LOG2`, `DRAM_LOG2` (from `opentpu.target.rtl_params`:
  `channels_per_slice`, `channel_bytes`, the arch's `DRAM_BYTES` or the platform's `slice_bytes`)
  instead of hardcoding 2 / 31 / 32; slice s is on channels NCH s .. NCH s + NCH - 1, CFG_DRAM
  reports them, CALIB0 / CALIB1 are the even / odd channels' ready.
- `rtl/boards/f2/otpu_native_axi.sv` (Alex's, changed: "The memory path" below): one channel's
  native port onto an AXI4 512-bit master at the core clock, BASE p x 512 MiB.
- `rtl/boards/f2/otpu_f2_hbm_axi.sv` (`otpu_f2_hbm_sys` u_mem): the `NXR` = 3 SLR chain on every
  native port, the channels' `u_naxi`, and `otpu_f2_pcis_gate.sv` (`u_pcis`: the PCIS onto the
  fabric's host branch through skid buffers, the read-after-write and OCL order).
- `rtl/boards/f2/otpu_hbm_geometry.sv` (Alex's, unchanged): 32-byte aligned HBM addresses after
  each channel's SmartConnect (the HBM requires them; SmartConnect keeps a narrow single beat's
  address).
- `otpu_axil_xr.sv` (the control path across the SLRs), `otpu_f2_hbm_sys.sv` (otpu_board + u_mem +
  OCL: the channels' AXI4 masters `ch_*` and the host branch `h_*` are its ports),
  `boards/f2/design/cl_opentpu_hbm.sv` (the CL: clocks, resets, the SmartConnect block design
  `u_sc` instantiated from `otpu_hbm_sc_inst.vh`, `otpu_hbm_geometry` per port, the HDK's
  `cl_mem_hbm_wrapper` with `NUM_OF_AXI_PORTS = S x NCH`, `AXI4_INTERFACE = 0`, the unused-DDR
  template for the DIMM's pins).
- The block design: `boards/f2/build/scripts/create_hbm_sc.tcl` (Alex's create_f2_ip.tcl the
  model), generated on the build host by build.py before synthesis (below, "Build").
- Out of the build since the rework (their files kept: `tb_f2hbm`,
  tests/test_f2_rtl.py::test_f2_hbm_fabric, still tests them; no target uses them):
  `otpu_f2_hbm_nch.sv` (the channel bridge: core-side order check, afifo, hclk burst builder),
  `otpu_f2_hbm_port.sv` (a port's two-master arbiter), `otpu_f2_hbm_host.sv` (the 450 MHz host
  path), `otpu_f2_hbm_mem.sv` (their fabric). Their description is in git at 5d0a5f9a
  (docs/f2_hbm.md "The channel bridge", "The host path").

### Address map

Logical (what a slice's programs and `Board.write / read` use): byte address `a` of slice s, chunk
`m = a / 128` (word address bits 31:5), beat `b = a / 64 % 2` in it. With `NP = NCH / 2` pairs:

- pair `p = (m mod NP) ^ fold(m / NP)` (CHASH; `fold(u)`: the XOR of u's log2(NP)-bit groups),
- channel `c = 2p + (b ^ parity(m))`, channel beat `u = m / NP`,
- HBM port and pseudo-channel `P = NCH s + c`, HBM byte `P x 512 MiB + u x 64 + a mod 64`
  (= the PCIS BAR4 offset the host writes).

At NCH = 2 this is the DDR boards' map (`c = b ^ parity(m)`, `u = m`). Every channel's access stays
in its own pseudo-channel: the cl_hbm IP has the global address switch on (USER_SWITCH_ENABLE_00
/ 01 TRUE, 34-bit addresses: any port reaches all 16 GiB, through the lateral switch with added
latency and shared bandwidth, and across the stacks at a fraction of it), and this build never
uses it -- each channel's traffic, and the host's to that channel, go through the port that owns
the pseudo-channel (each channel's SmartConnect reaches only its own port, at p x 512 MiB;
otpu_native_axi masks a beat into its window and flags one outside it). `tb_f2hbm_axi` checks
it (an AW / AR of a channel outside its 512 MiB fails).

**Why a chunk on a pair, rotating over the pairs.** A chunk is 128 bytes (D = 128: port B's unit,
one a core cycle), two 64-byte beats; a 64-byte beat is two of the HBM port's 32-byte AXI beats,
its smallest efficient access. Splitting a chunk over more than two channels would mean 32-byte
native beats (one AXI beat each: half the commands' efficiency, twice the order checks and
tags). Keeping a chunk on two channels and rotating successive chunks over the NCH / 2 pairs
spreads a stream evenly (every channel one beat in NCH / 2 chunks: at NCH = 8 a channel carries
3.2 GB/s of the slice's 25.6), keeps the B merge a two-channel join (the tag FIFO holds the
pair), keeps each channel a plain one-beat-per-command master, and gives each channel's
consecutive beats to consecutive chunks of its pair, so a channel's A run (APF beats) and a
stream's beats on it are runs of channel beats that otpu_native_axi's adjacency test (the
next beat, within 4 KiB) can turn into bursts ("The memory path" below). The fold
spreads power-of-two strides of chunks (a transposed V's column: one byte per cache row) over all
pairs, as CHASH's parity spreads them over a pair's two halves.

At NCH > 2 port B's request passes a register in the adapter (BREG): the pair's fold (27 address
bits deep) would otherwise run at the end of the slice's grant chain into every queue write.
b_rdy reserves room for it (QD - 1, OD - 1). One cycle on port B's reads against the HBM's
hundreds of nanoseconds.

### Since the first placed build: AWS's chain, the host on its own port (2026-10-20)

The first F2 build of this path (cd16e195, f2-hbm:p4rl8a64 at 200 MHz) placed at WNS -1.609 ns:
clk_main_a0 -1.609 (5,034 endpoints, all inside the host's 1 x 8 SmartConnect: 21,625 LUTs /
46,078 FFs, 2,091-bit register-slice vectors, 4.7 of 5.2 ns route), hclk -1.350 (17,226
endpoints in the 2 x 1 channel SmartConnects' 450 MHz side, each 8,548 LUTs / 16,207 FFs, 3.2 of
3.5 ns route), the core clock -1.078 (the MXU, 10 levels), clk -> clk_f -0.756 and clk_f -> clk
-0.793 (the pumped TMEM). The rework, after AWS's own HBM designs:

- **The chain of AWS's cl_dram_hbm_dma** per channel: u_naxi (core clock) -> `axi_register_slice`
  (the HDK's prebuilt, 512-bit AXI4, g_ch[c].u_rs512) -> a 1 x 1 SmartConnect (sc_c<c>, the
  clock and width converter: create_hbm_sc.tcl copies every setting of the HDK's
  cl_axi_sc_1x1.bd, its cell's CONFIG and its AXI4 / AXI3 ports', but the slave side's clock,
  the core's) -> `cl_axi3_256b_reg_slice` (the HDK's, at 450 MHz, g_ch[c].u_rs256) ->
  cl_mem_hbm_wrapper port c. otpu_hbm_geometry moved before the 450 MHz slice, so every AXI
  input of the wrapper comes from that slice's registers (AWS cl_mem_perf's q1_* stage). No
  crossbar, no 2-input SmartConnect anywhere.
- **The host on its own HBM port** (port NCH, the next after the channels; f2-hbm:s4's 32
  channels leave none and are refused): otpu_f2_pcis_gate -> axi_register_slice (both in the
  shell's PCIS SLR, SLR1; the SLR crossing runs flop to flop from the slice's registers to the
  SmartConnect's entry slice in SLR0) -> AWS's cl_axi_sc_1x1 as shipped (S00 at the shell's 250
  MHz, M00 AXI3 256 b at 450, beside port NCH) ->
  cl_axi3_256b_reg_slice -> port NCH. The cl_hbm IP's global address switch is on
  (USER_SWITCH_ENABLE_00 / 01 TRUE, build.py checks it), so port NCH reaches every channel's
  pseudo-channel at the switch's latency: BAR4 offset = HBM address, as before.
- **Host writes before a run's reads** (different HBM ports, no AXI order between them): the
  run starts with an OCL write (RUN), and otpu_axil_cdc holds every OCL write while
  `pcis_wbusy` is high: from a PCIS write's AW until its B is given back to the shell. The B
  comes back through cl_axi_sc_1x1 from the HBM controller of the pseudo-channel that holds
  the address (through the switch), sent once the write is in that controller; the run's first
  read of the address reaches the same controller (each pseudo-channel has one) later, after
  RUN's register write, the board's control path and the slice's first DMA command, and the
  controller orders a read behind an earlier write to its address. So a run's reads see every
  host write whose B came back before RUN, which pcis_wbusy makes all of them. The host's own
  reads after its writes are held in the gate as before (same port, in order anyway). If the
  card ever shows otherwise, a host-side fence (a BAR4 read of the last written address before
  RUN: Alex's transport does it) closes it without RTL.
- **Reads in flight.** cl_axi_sc_1x1's SmartConnect accepts 32 transactions a direction (its
  S00 / M00; the AXI3 port says 64): the per-channel cap. otpu_native_axi's READ_BURST is now 8
  (512 B: the 16-beat AXI3 burst AWS cl_mem_perf measured 426 GB/s read with) so 32 bursts are
  16 KB in flight a channel against 14.4 GB/s x 280 ns = 4 KB, and READ_WAIT is 1 again (a burst
  waits READ_IDLE = 1 cycle for its next command): under the 32-transaction cap the receipt
  measured waiting 8-beat bursts best (1,027,452 cycles at 150 MHz; 4-beat bursts from what is
  queued 1,030,204; 8-beat from what is queued 1,108,236: the queue's 10 usable slots stall
  behind a held burst), against 947,522 with no cap (docs/wide_port.md "Alex's shell at 150 MHz").
  create_hbm_sc.tcl fails closed unless every sc_c<c>/S00_AXI takes >= 32 each way and 8-beat
  bursts after validation, and logs the values (OTPU_HBM_SC lines). The receipt models the cap
  (OTPU_NAXI_OT, default 32): docs/wide_port.md "Alex's shell at 150 MHz".
- **Placement**: pb_otpu_hbm takes AWS's SLR0 outline (cl_mem_perf's pblock_CL_SLR0: X0Y0:X3Y3,
  X4Y0:X7Y0, X4Y3, X5Y1:X5Y3, X6Y1 and its four SLICE ranges); inside it a box per clock-region
  column of HBM ports (pb_otpu_hbmx<x>, that column's Y0:Y1) holds the SmartConnects and the
  450 MHz slices of its ports, the host's SmartConnect and 450 MHz slice in its port's column
  (its 512-bit slice with the gate in pb_otpu_pcis); each port's column
  from its BLI_HBM_AXI_INTF_X<p>Y0 site (a warning line and four ports a clock region if the
  device names it otherwise). The shell's PCIS: the floorplan finds the cell driving
  sh_cl_dma_pcis_awvalid in the linked design and fails closed unless it is in pb_otpu_pcis's
  SLR (SLR1 on the VU47P: round 3's placed report put that register at SLICE_X176Y399; the
  Shell Interface Specification's table says TOP). pb_otpu_pcis (X5Y5:X5Y6) is beside that
  register, outside AWS's PCIS-crossbar box (X0Y4:X4Y7) but in its pblock_CL_SLR1 (X5Y4:X5Y7).
- **Timing hygiene** (the Shell Interface Specification): `shreg_extract = "no"` on otpu_xr's
  stages and otpu_skidbuf's registers; each new domain's reset registered again from its
  synchronizer with max_fanout (sc_aresetn core, sc_host_aresetn 250, sc_hbm_aresetn 450);
  report_design_analysis -logic_level_distribution after synthesis and after routing, and
  -congestion after routing (reports, not gates).
- **The HBM monitor APB** (an unconnected monitor interface fails AFI creation): both racks'
  `hbm_apb_preset_n_<0/1>` straight into cl_mem_hbm_wrapper's `i_hbm_apb_preset_n_<0/1>`
  (MON_APB_<0/1>_PRESET_N inside it), every `hbm_apb_*_<0/1>` output from its `o_hbm_apb_*`,
  apb_clk = clk_hbm_ref, as cl_mem_perf; build.py checks the cl_hbm IP's USER_MON_APB_EN is
  TRUE (and the switch), tests/test_f2_build.py that the CL connects all 22 signals.
- **The crossings** (none a plain two-flop path at 450 MHz): core <-> clk_main_a0 in
  otpu_axil_cdc (the OCL: held words, toggles through two-flop synchronizers) and the board's
  LED / HBM-ready levels, bounded by cl_timing_user_hbm.tcl's datapath-only max delays at the
  destination's period; core -> hclk and hclk -> core only inside each sc_c<c> (SmartConnect's
  clock converter, its own scoped constraints); clk_main_a0 <-> hclk only inside cl_axi_sc_1x1
  and the HDK wrapper (their own); the resets through the async-assert synchronizers (their
  linked exception, cl_pnr_user.tcl, takes hbm_reset_n); the MMCMs' LOCKED false-pathed to
  reset pins. No otpu_afifo in the build.

### The memory path (Alex's shell, since 2026-10-19)

From Alex's burst200 branch (ergodex-openTPU `alex-burst200`, commit 06c169cb on 7e42a626):
`otpu_native_axi` (his native-to-AXI adapter, changed as listed below), `otpu_hbm_geometry`
(unchanged), and the SmartConnect fabric of his `boards/f2/scripts/create_f2_ip.tcl` (one
SmartConnect `memory_sc`: S_HOST 512 b @ 250 + S_N0 / S_N1 512 b @ 200 onto M_H0 / M_H4, AXI3
256 b @ 450; his interface properties), restructured per channel in
`boards/f2/build/scripts/create_hbm_sc.tcl` (block design `otpu_hbm_sc`):

- per channel p: `sc_c<p>`, a SmartConnect with two slaves, S00 = S_N<p> (u_naxi p, @ core_clk
  200 MHz) and S01 = the host's branch (@ host_clk 250 MHz), one master M00 = M_H<p> (AXI3, 256
  bits, @ hbm_clk 450 MHz), NUM_CLKS 3: the clock crossing, the 512 -> 256 width conversion and
  the AXI4 -> AXI3 split are SmartConnect's. S_N<p> reaches M_H<p> only, at p x 512 MiB.
- the host: `sc_host`, a 1 x NPORT SmartConnect @ 250 MHz (NPORT > 16: a 1 x NPORT / 16 top over
  1 x 16 groups), M_H<p> at p x 512 MiB from S_HOST (64-bit addresses: BAR4 offset = HBM
  address). Addresses past the last channel get DECERR (the rounds-1-8 host path dropped such
  writes and returned zeros; the host transport maps BAR4 for exactly S x NCH x 512 MiB).
- `otpu_hbm_geometry` after every M_H<p> (32-byte aligned addresses: a narrow single-beat host
  access keeps its byte address through SmartConnect), then `cl_mem_hbm_wrapper` port p
  (`AXI4_INTERFACE 0`; the wrapper fixes AxSIZE 5). HBM IDs: the wrapper's M_H<p> ID ports are
  connected if the generated wrapper has them, else tied to 0 (Alex's had none: SmartConnect's
  single-threaded master), by the instance `create_hbm_sc.tcl` writes from the wrapper's own
  port list (`hbm_sc_inst.tcl`, fail closed on any input it cannot map, any width other than
  the CL's, any missing port).
- the block design, its OOC checkpoints and its wrapper are generated on the build host into
  the run directory (`cl_opentpu/ip`); no AMD IP product is committed.

`otpu_native_axi`, as built (u_naxi; Alex's file with these changes, marked in its header):

- BASE_ADDR aligned to the window `CH_LOG2` (29: a 512 MiB pseudo-channel; his 2 GiB is the
  default), fail closed at elaboration; a command whose beat is outside the window raises n_err
  and is masked into it (never into a neighbour's).
- `XL` = 2 NXR = 6: the command and write queues (16 entries) keep 6 slots below their readies
  and push every take that arrives (the adapter sees the ready three cycles late across the SLR
  chain and its take lands three later), as the rounds-1-8 bridges' `XL`.
- an AR register: a closed burst moves there and the next one gathers while it waits for
  ARREADY (his single pending burst stalled the queue for the AR's cycle: 16 commands in 17
  cycles; now a command a cycle, `tb_native_axi`'s stream phase: 512 in 513).
- `READ_WAIT` 0 (the build's; 1 is his): a burst takes only commands already queued and closes
  when the queue runs dry, so no read waits for a command still to come. `READ_BURST` 4,
  `READ_IDLE` 1 (his 16 / 4). Why: with 64 reads in flight a channel (otpu_board NRD at NCH > 2)
  the decode token is latency-bound, and a waiting burst holds its first beat's credit until its
  last is taken (a stream at a command a cycle: B - 1 cycles on every beat's round trip). The
  receipts (docs/wide_port.md "Alex's shell"): his waiting 16-beat bursts cost the int8 token
  +38%, waiting 4-beat bursts +23%, against +12.3% with READ_WAIT 0 at the same latency.
- `ORDER` (tests only: 0 removes the direction barriers).

Kept from his design: reads coalesced by the adjacency test (the queue head's beat = the open
burst's next beat, the same 4 KiB page), at most 16 native beats (32 AXI3 beats: SmartConnect
splits them), never across 4 KiB, bounded by MAX_OUT = 128 reads in flight (the adapter keeps
64); writes single beats (AW and W, one at a time); the direction barriers (a read waits until
every write has its B, a write until every read has its data), which make the native contract
hold whatever the HBM reorders between its read and write channels (the rounds-1-8 bridges
compared beat addresses instead: finer, a read waited only for writes to its beat); all IDs 0,
R and B always ready (the adapter reserves room for its reads), n_wdone counting B's, the error
checks (RRESP, BRESP, RLAST against an independent length queue).

Against our address map: a channel's consecutive beats are consecutive channel beats (the
address map above: a stream's chunks rotate over the pairs, each channel's beat index u = m / NP
steps by one), so the adjacency test does coalesce them when they are queued together. In the
decode receipt (READ_WAIT 0) the channels see a beat at a time (the adapter's credits, not a
backlog, pace them): channel 0 had 806,113 single-beat bursts of 808,137, 1,903 of four, 121 of
two or three. Under tb_f2hbm_axi's mixed traffic (random back-pressure: the queue backs up) 14%
of the reads go as bursts of 2-4 beats. A
64-byte read is two 32-byte HBM beats, the HBM port's smallest efficient access (the rounds-1-8
bridges sent the same single-beat commands when nothing else was waiting); the card's bandwidth
with them is the first thing to measure (below).

The host's path, `otpu_f2_pcis_gate` (u_mem/u_pcis, aclk): every PCIS channel through a
two-entry skid buffer (otpu_skidbuf: AW / W / AR towards the fabric, B / R towards the shell),
every output on either side a register. A write is pending from its AW's take until its B is
given back to the shell, or while write data waits in its buffer: an AR leaves its buffer only
while none is pending (registers only, so a write taken in the cycle before is seen), so the
host's read after its write returns the written data whichever channels the two use (the
SmartConnects do not order a read against a write); `pcis_wbusy` (that state registered) holds
the OCL's writes (otpu_axil_cdc: a write starts only from an AW three cycles old, `hold`
registered again), so a RUN after the data finds it in the HBM. The SmartConnect returns the
HBM's B. Alex's transport flushes posted writes with a read instead; ours keeps the rule in the
CL as before, so opentpu/host does not change.

Latency (the decode token's lever): the native port's round trip changes by the SmartConnect's
clock and width conversion and u_naxi's queue against the deleted bridges' core-side FIFO,
afifo, burst builder, arbiter and R pairing. Estimated (not measured: SmartConnect is encrypted,
and Alex's 23.7 GB/s at 128 reads in flight on 2 channels bounds his round trip only below ~690
ns): u_naxi 3 cycles for an isolated read (its queue and AR register, in the RTL), SmartConnect
~17 core cycles (PG247: its SI / MI register slices are on by default; an asynchronous clock
converter costs 5 or more destination cycles each way, as PG085 states for the AXI4-Stream
converters; two 256-bit beats packed into one 512-bit beat), the bridges ~10 core cycles: **+9
core cycles** at the native port, the SLR chain (6 cycles, in both) aside. The receipts run the
native model at --lat 69 (60 + 9) and the build's otpu_native_axi itself in the loop
(OTPU_NAXI=1).

### Clocks and CDC

| clock | source | used by |
|---|---|---|
| clk_main_a0 250 MHz (aclk) | shell | OCL, PCIS (otpu_f2_pcis_gate), the host's SmartConnect and the channels' S01, the HBM wrapper's 250 MHz side |
| clk 200 MHz, clk_f 400 MHz | `u_mmcm` from clk_main_a0: VCO 1200, /6, /3 | the slices, u_naxi, the channels' SmartConnect S00 |
| hclk 450 MHz | `u_hmmcm` from clk_hbm_ref (100 MHz): VCO 900, /2 | the SmartConnects' masters and the HBM AXI ports only (no openTPU logic) |
| clk_hbm_ref 100 MHz | shell | the HBM's reference and APB clock |

`boards/f2/build/constraints/cl_timing_user_hbm.tcl` (applied after synthesis, in place of the
DDR build's): max delay, datapath only, between the core and clk_main_a0, each direction at the
destination clock's period (otpu_axil_cdc's held words and the synchronized levels); the HBM
clock meets the others only inside the SmartConnects (their own scoped CDC constraints) and the
HDK wrapper (xpm_cdc), so no openTPU bound names it: the rounds-1-8 hclk bounds and the bus
skews on the bridges' and host path's afifo gray pointers and gray write counts went with them.
It fails closed if u_sc is not in the netlist; the MMCMs' lock false-pathed to reset pins only.

Resets: the CL's three synchronizers (`core_reset_n`, `hbm_reset_n`, `bridge_reset_n`: ASYNC_REG
flip-flops synthesis does not replicate) each drive one register in otpu_board (`rst_q`,
max_fanout 128: the control block, the loader and port registers, the adapter's ~2,200 counters
at NCH 8) and in otpu_f2_hbm_axi (`rst_q`, max_fanout 4; the gate registers aclk's again), the
SmartConnects' aresetn from a register on `core_reset_n` (the channels', core_clk) and on
`bridge_reset_n` (the host's, aclk), the HBM wrapper's from `hbm_reset_n`, so no openTPU
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
| `pb_otpu_hbm` (SLR0) | `u_mem/g_port[*].u_naxi` (8), `WRAPPER/CL/u_sc` (the SmartConnects), the SLR0 stage of every crossing chain | X0Y0:X3Y3 + X4Y0:X7Y0 + X4Y3 (21) | ~286K | (rounds 1-8: ~31K) |
| `pb_otpu_pcis` (SLR1) | `u_mem/u_pcis` (the PCIS gate), `u_ocl` (the OCL crossing), beside the shell | X5Y5:X5Y6 (2) | | |
| `pb_otpu_mid` (SLR1) | the middle stage of every chain | X0Y4:X3Y7 (16) | ~217K | 0 LUT, ~18K FF |
| `pb_otpu_board` (SLR2) | `u_board` (the slice, its adapter, the control block, the trace), the SLR2 stages | X0Y8:X7Y11 (32) | ~435K | 226K (52%) |
| `pb_otpu_tmem` (in `pb_otpu_board`) | `u_board/u_slice/u_tmem`, `u_vpu` | X2Y9:X5Y10 (8) | ~109K | 48K |

Everything between SLR0 and SLR2 is a chain of three `otpu_xr` register stages, one per SLR,
so each SLR crossing runs from a flip-flop to a flip-flop: the native ports (otpu_f2_hbm_axi
`NXR` = 3: down, per channel, the adapter's takes with the command and the 512-bit beat; up,
u_naxi's readies, read data, write count and error; 9,088 nets; u_naxi's queues have 16
entries and keep 6 below their ready's threshold, since the adapter sees a ready three cycles
late and its beat arrives three later: `XL`), the control
path (`otpu_axil_xr` between the OCL's clock crossing and the board: AXI4-Lite as pulses, which
otpu_axil_cdc's one-access-at-a-time master allows) and the board's reset and HBM-ready. A port
B request costs 6 core cycles more from the adapter to the HBM and back (30 ns on the HBM's
~400; the adapter keeps 64 reads a channel in flight). The three-SLR split was chosen over
SLR1 for the board (72% of its 23 clock regions: too full for 200 MHz routing) and over the
adapter in SLR1 with the slice in SLR2 (the adapter's readies into the units and its 4,096-bit
port B data are a same-cycle handshake that cannot cross an SLR without a redesign).

Since the alex-shell rework the HBM side holds u_naxi and the SmartConnects instead of the
bridges (Alex's floorplan.tcl put his whole design, board included, in one SLR0 child of
pblock_CL with the PCIS register slice and OCL converter in an SLR1 child; its SLR0 regions
include X4Y3, X5Y1:X5Y3, X6Y1 and site columns beside the shell, of which X4Y3 is taken here:
round 6's pb_otpu_xs held it and went with otpu_f2_hbm_host); his SLR split is the reference for
the HBM side, the three-SLR chain stays ours (his 131K-LUT board fit SLR0; our 226K slice does
not).

## Bandwidth

| | per port | per slice, NCH = 8 | per slice, NCH = 16 | 32 ports |
|---|---|---|---|---|
| HBM peak (256 b x 450 MHz) | 14.4 GB/s | 115 GB/s | 230 GB/s | 460 GB/s |
| native port limit (64 B per core cycle per channel) | 12.8 GB/s | 102 GB/s | 205 GB/s | |
| the adapter's reads in flight (64 x 64 B / ~400 ns) | ~10 GB/s | ~80 GB/s | ~160 GB/s | |
| what the slice consumes at 200 MHz: port B, a 128 B chunk a cycle | | 25.6 GB/s | 25.6 GB/s | |

A slice streams at most 25.6 GB/s (port B; port A's scale lines are a small fraction, held in
runs); at NCH = 8 that is 3.2 GB/s per channel, 22% of a port: the HBM side has 4.5x headroom,
so a slice's DMA is limited by the core's port width, not memory, and HBM efficiency (bank
conflicts, refresh, the 30-40% loss of random 64-byte accesses) has room to spare. The DDR
build, for comparison: one DIMM port at 16 GB/s shared by both channels, a 100 MHz core asking
12.8. Four slices consume 102 GB/s of the 460.

## Past 4 GiB a slice

Everything below the slice handles more than 4 GiB per slice: the adapter's chunk index is word
address bits 31:5 at NCH > 2 (16 GiB), the channel beat fits 25 bits up to NCH = 32, u_naxi's
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
| `tests/test_f2_rtl.py::test_native_axi` (`tb_native_axi`, Alex's ported) | otpu_native_axi alone, 9 configurations (his four; XL 6 / CH_LOG2 29 behind three register stages each way, waiting bursts of 16 and 4; READ_WAIT 0 at 4 / 16 beats, 128 / 64 credits), 4 seeds each: random AXI stalls, data before / after / with commands, saturated credits and queue, 4 KiB and window boundaries, idle flushes, injected RRESP / RLAST errors, reset, nothing lost behind the late readies, a stream at a command a cycle, a command outside the window flagged | pass |
| `test_f2_hbm_axi` (`tb_f2hbm_axi`) | the build's memory side (otpu_f2_hbm_axi: chain, 4 / 8 / 16 / 32 u_naxi, the PCIS gate) on a SmartConnect + HBM model (crossing latency, in order per slave, a write's service up to 4x a read's so reads overtake writes, two HBM beats a 512-bit beat, the host's demux and its order across ports), NXR 0 / 3, bursts 4 (no wait) and 16 (waiting), three clock ratios: the native contract, channels inside their 512 MiB, INCR within 4 KiB and 16 beats, bursts formed; host writes land, reads issued before their writes' B's return them (the gate holds them), B / R in order with IDs; pcis_wbusy high while a host write is unanswered | pass |
| `test_f2_hbm_axi_detects_missing_order` | ORDER 0 (no direction barriers, no read hold): stale reads on the channels and on the host | pass |
| `tests/test_f2_build.py::test_hbm_sc_instance_maps_every_wrapper_port_and_fails_closed`, `test_hbm_sc_block_design_is_per_channel` | hbm_sc_inst.tcl on a Vivado-style wrapper (with and without HBM-side IDs; an unknown input, a wrong width, a missing port stop it); create_hbm_sc.tcl under a Vivado mock at 8 and 32 channels (per channel a 2 x 1 SmartConnect on three clocks, the host's demux, every M_H<p> at p x 512 MiB from its channel and the host) | pass |
| `tests/test_f2_rtl.py::test_f2_hbm_fabric` (`tb_f2hbm`; the rounds-1-8 path, no longer built) | 4 / 8 / 16 / 32 ports, three clock ratios, the native ports through the build's three crossing stages each way (NXR 3) and wired (0): every channel's native contract on an HBM model that reorders across IDs, runs transactions in any per-ID order and interleaves two IDs' read beats; bridges inside their pseudo-channels, bursts formed (~18% of transactions multi-beat under the mixed traffic), INCR within 4 KiB; host writes / reads on every port in PCIS order, B / R IDs, zeros past the last port | pass |
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

build.py checks the HDK and the cl_hbm IP (both stacks, 450 MHz, switch on, generated), generates
the SmartConnect fabric first (BuildAll / SynthCL: `vivado -mode batch -source
cl_opentpu/build/scripts/create_hbm_sc.tcl -tclargs cl_opentpu/ip NPORT PART`, its log
hbm_sc.log / hbm_sc_console.log; recorded in invocation.json as `ip_command`), copies the
design and build directories, puts `cl_opentpu_hbm.sv`, `synth_cl_opentpu_hbm.tcl` and
`cl_timing_user_hbm.tcl` in the DDR files' places, copies the HDK's `cl_mem_hbm_wrapper.sv`,
writes `opentpu_geometry.vh` (S, NCH) and `opentpu_params.vh` (otpu_f2_hbm_sys's parameters), and
runs the HDK's build_all.tcl with the clock recipes A1 B2 C0 H2 (H2: the HBM recipe the
manifest records, as Alex's image). `--package` produces the Developer_CL tar for
`aws ec2 create-fpga-image`.

### Synthesis (f2-hbm16: S = 1, NCH = 16; f2-hbm: S = 1, NCH = 8; the rounds-1-8 path)

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
   at 50%, with its own crossing chains). Since the alex-shell rework, check first on the build
   host: create_hbm_sc.tcl's `validate_bd_design` and its `OTPU_HBM_SC` / `OTPU_HBM_SC_DONE`
   lines (the instance written from the wrapper: any port it cannot map stops it, with the
   port's name), synthesis reading the block design (u_sc in the netlist,
   `OTPU_HBM_CDC ... (hbm crossings: SmartConnect's own)`), the floorplan's
   `OTPU_FLOORPLAN ports=8` / `pblocks=5`, then report_cdc on the SmartConnects' crossings.
   Expected timing risks: the core clock at 200 MHz with the adapter's NCH-wide B merge (a 4:1
   or 8:1 mux after the B FIFO's LUT RAM read); u_naxi's AW address (its queue's LUT RAM read
   into SmartConnect's input, Alex's structure); the PCIS gate's SLR1 -> SLR0 hop into sc_host at
   250 MHz (Alex's register slice made the same hop).
   Route qualification (validate_route.tcl, also route_opt.py --validate-only on a routed
   checkpoint) knows the HBM CL by WRAPPER/CL/u_sc: the methodology classifier
   (check_scoped_clocks.tcl) names the CL it finds (fail closed: the DDR4 MMCM pin and UI clock
   absent, the fabric present) and reviews two TIMING-54 context findings, cl_timing_user_hbm.tcl's
   pair as write_xdc renders it, `set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to
   [get_clocks -of_objects [get_pins u_clk/O]] <core period>` (6.667 at 150 MHz, 5.000 at 200)
   and the reverse at 4.000, with the main -> core path bound checked at the core period; the 35
   inherited shell findings stay the exact match against the shell checkpoint; anything else in
   TIMING-54 stops it. No DIMM sentinel on the HBM CL (route_opt.py passes none_hbm; the drivers
   require the fabric in the checkpoint).
2. On the card: the vLED (bit 0 MMCMs locked, bit 1 HBM ready), CFG_DRAM / CFG_SLICES, a BAR4
   write / read of every port, `otpu-diag --mem`, then the token-exact checks (f2 docs), and the
   streaming bandwidth per channel with single-beat (64-byte) reads (READ_WAIT 0's decode
   pattern) against Alex's 23.7 GB/s on two channels.
3. The word-address ISA change above, for an 8 GiB slice (the 27B in one slice).
