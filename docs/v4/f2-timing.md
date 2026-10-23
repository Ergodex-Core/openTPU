# f2-hbm timing: the first routed build's violations and their fixes

The `f2-hbm` target ([f2_hbm.md](../f2_hbm.md): one slice on 8 HBM pseudo-channels, core 200
MHz, the pumped TMEM at 400 MHz, the HBM AXI ports at 450 MHz, the shell's clk_main_a0 at 250
MHz) was built on the F2 instance for the first time as build id 61d07199 (Vivado 2025.2,
`cl_opentpu.61d07199.post_place_timing.rpt`; the post-route report follows). The report's
worst paths hold 43 violated endpoints in three groups and two small ones. Branch
`v4-f2-timing` (on tpubench's 1113137e) fixes them by RTL pipelining, replication and
constraints only: nothing in the compute core's numerics moves, every change is bit-exact in
Verilator, and the bridges' changes are cycle-exact. Where the rhino fork's `wide-port` branch
had fought the same paths on its p4rl8 build (its docs/wide_port.md "Placement" rounds 1-7),
the fix is its, cherry-picked or rewritten for S = 1 and noted as such.

## The paths

| group | clock | worst path | slack | data path | what it is |
|---|---|---|---|---|---|
| 1 | hclk_raw 450 MHz (2.222 ns) | `u_sys/u_mem/u_host/ws_reg[0]/C` -> `u_sys/u_mem/g_port[4].u_port/m_wdata_reg[169]/D` | **-4.03** | 6.201 ns: logic 0.205 (2 LUT levels), route 5.996 | the host path's shared write bus into every port's W register |
| 2 | clk_raw 200 MHz (5 ns) | `u_sys/u_mem/g_port[0].u_nch/n_wdone_reg[3]/C` -> `u_sys/u_board/u_slice/w_n_reg[24]/R` | **-2.01** | 6.544: logic 0.660, route 5.884 | the bridge's write count into the slice's `wr_idle` chain |
| 3 | clk_raw -> clk_f_raw (the pumped TMEM; requirement 5.000 ns, two fast periods) | `u_slice/u_dma/t_re_q_reg[5]/C` -> `u_slice/u_tmem/g_one.g_bank[0].g_sub[1].g_pmp.u_pb/g_b.mem_reg_0/ADDRBWRADDR[3]` | **-1.0** | 4.67: logic 0.608 (2 levels), route 4.065 | the DMA's TMEM read request into a pumped bank's address |
| 4a | `**async_default**` | `u_hbm/PIPE_HBM_AXI_RST/pipe_reg[3][0]` -> the HBM IP's `AXI_xx_ARESET_N` | -0.33 | | the HDK wrapper's reset pipe into the HBM stacks |
| 4b | clk_main_a0 250 MHz | hidden shell cells | -0.29 / -0.24 | | the shell's own |

Every one of them is 90-97% route: fan-out, bus width and distance, not logic depth. The
build's placement directive is `SSI_SpreadLogic_high` (build.py's first tclarg since the
flow's first run), which spreads one slice of ~150K LUTs over the VU47P's three SLRs on
purpose; the 4-6 ns routes on 2-level paths are its signature (rhino's base f2-hbm build at
96a90c2f, the same directive, placed at -4.1 ns on the same host path). See "The next build".

## 1. hclk: the host's write bus into the ports (-4.03 ns)

**What it was.** otpu_f2_hbm_host holds the PCIS beat it is sending in registers (`hw_d`, 512
bits: two AXI beats; `hw_s`, the strobes; `ws`, which half is on the bus) and showed the half
`ws` selects on one 256 + 32-bit bus to all 8 ports' arbiters (`p_wdata = ws == 2 ?
hw_d[511:256] : hw_d[255:0]`), whose W registers took it on their own select (`m_wdata <=
s_wdata[wsel]`, otpu_f2_hbm_port). The arbiters sit at the HBM's AXI ports along the bottom
of SLR0, the host block wherever the placer left it: each `hw_d` bit had 8 loads across the
die behind a LUT, `ws` 8 x 288 LUT selects (replicated into `ws_reg[0]` and its copies), 2
levels and 6.0 ns of route at 2.2 ns. The read return was the same in the other direction:
the 8 ports' R data (from the HBM's pins through the arbiters) AND-ORed into `rq_d` in one
cycle.

**The fix** (commit 45ec6579; rhino's d024f05c and 24422567, the valids this branch's):

- **A register per port for the host's master**: `otpu_f2_hbm_hreg` (in otpu_f2_hbm_mem.sv,
  `g_port[p].u_hreg`), a one-entry register for AW, W and AR whose ready is its own emptiness
  (`h_awready = !aw_v`, a flip-flop: no path from the arbiter's or the HBM's readies back to
  the host in the cycle), shown to the arbiter until taken; the port's R and B registered the
  same way back (`h_rvalid / h_rdata`, `h_bvalid`, a cycle late; R needs no ready, the host's
  return FIFO has room for every read in flight and its count moves on the registered beat).
- **The W beat shown is a register** (`hw_wd`, `hw_ws`, `hw_wl`): the low half loads with the
  head, the high half replaces it as the low one is taken (`ws` 1 -> 2), so `ws` selects only
  this register's load, here, and each data bit goes from one flip-flop to the 8 ports'
  registers with nothing between.
- **The per-port AW and W valids are registers** (`pv_aw`, `pv_w`: `pv_aw` set to the head's
  port as it loads, cleared as the port takes the AW; `pv_w` set as the AW is taken and again
  as the low half is taken, cleared as the port takes the beat). So both directions of the
  handshake are flip-flop -> one LUT -> flip-flop: the hreg's `aw_v <= aw_v ? !m_awready :
  h_awvalid` on the way in, the host's `hw_aw = |(pv_aw & p_awready)` on the way back. A
  simulation assertion holds the valids to the state machine (an AW valid only in state 0, a W
  valid only in 1 / 2).
- The resets through registers of their own: `rst_q` / `hrst_q` in otpu_f2_hbm_mem
  (`max_fanout 4`, replicated toward the modules) and again in every module (nch, port, hreg,
  host): three flip-flops from the top's synchronizers to any reset pin, every net a
  register's cycle long (rhino's round 1 had `hbm_reset_n_reg` as the source of 120 of the 450
  MHz domain's worst 600 paths, ~1,250 pins from one unreplicated ASYNC_REG flip-flop).

**Why bit-exact.** The host path is not on a token: it carries the weights' load and a run's
inputs before the run and reads after it. Its protocol is unchanged (one beat at a time, in
order; `pcis_wbusy` holds the reads and the OCL's writes as before; the B count per beat), only
the latency to a port and back grows by two hclk cycles, and the write rate goes from a beat
per ~4 hclk cycles to ~7 (4 GB/s against BAR4 write combining's ~2). tb_f2hbm (tests/test_f2_rtl.py,
the fabric on a reordering, interleaving HBM model with random traffic on 4 / 8 / 16 / 32
ports and the host's writes, reads, B's and R's checked in order with their IDs) passes; with
HAZ 0 it still fails as it must.

**Expected recovery.** The path becomes a flip-flop to flip-flop net of 8 loads (a copy of
`hw_wd` per port's register) with no logic, and the handshake a single LUT each way: the 6.0 ns
of route are what the placer must now cover in one cycle from the host block to the farthest
port. Rhino's rounds 3-7 never had this path back among the hclk domain's worst after these
two changes (their worst became the bridge's internal cones at -2.0 to -2.25 ns, below); with
the directive change (the next build) the host block sits in SLR0 beside its ports. If the
post-route report still shows the host block -> hreg nets failing, the next step is one more
stage per port group (an otpu_xr-style register between the host block and the hregs,
`max_fanout 2` on `hw_wd`: the copies' D inputs then carry the distance on a full-period,
one-LUT path), not logic.

## 2. clk: the bridge's write count into the slice's wr_idle chain (-2.01 ns)

**What it was.** otpu_f2_hbm_nch's `n_wdone` (the HBM's write beats answered, counted on hclk,
gray-coded across, decoded into a clk register) goes to otpu_native_dram's `iss_all` (`iss_w[c]
== n_wdone[c]` over the 8 channels, 128 bits), into `wr_idle`, otpu_slice's `wr_idle_i`,
`halted`, and the profile window's end `w_end`, which is the synchronous reset of `w_n`, the
cycles-of-the-window counter. 0.66 ns of logic (the compare and the chain) and 5.9 ns of
route: the count's register sits with its gray synchronizer (`bg1`, `bg2`), and the CDC
constraint bound `bgray -> bg1` by the smaller period of the pair (2.2 ns datapath only, plus
the bus skew), which pulled `n_wdone_reg` to the HBM port while the adapter and the slice are
wherever the spread put them.

**The fix** (45ec6579 and 9ef46073):

- **A register stage on `n_wdone` and `n_err`** between the bridges and the adapter, in
  otpu_f2_hbm_mem (`b_wdone -> n_wdone`, clk, reset to 0 with the bridge's count). Its D net
  carries the distance from the HBM port (flip-flop to flip-flop, a full core period, no
  logic); the compare and the chain then start from a flip-flop the placer puts by the adapter.
- **The CDC bound by the destination period** (cl_timing_user_hbm.tcl; rhino's cf4ec4e5):
  hclk -> clk 5 ns, clk -> hclk 2.222, main -> hclk 2.222, hclk -> main 4, core <-> main 4 /
  5 (it was the smaller of the pair in both directions). A crossing's word (an otpu_afifo's
  LUT RAM read, a gray count, a held request) is read only after its pointer or count has gone
  through the destination's two synchronizer flip-flops, so a destination period on the data
  is the conservative bound; the old bound held the bridges' 512-bit return FIFOs and the
  counts' gray buses into the core at 2.2 ns across the die. The bus-skew bounds (every afifo's
  gray pointers, every write count's gray bus) stay at the smallest period.

**Why bit-exact.** A write is counted one core cycle later than before and never earlier, and
`wr_idle` is `iss_all && (nothing accepted and not yet gone out)`: its other terms (`wq_n`,
`wacc_q`, `gv`, the registered B request) cover the window from a write's accept to its issue
with no help from the count, so a later count can only hold `wr_idle` low a cycle longer,
never raise it while a write is in flight. The adapter's `iss_w[c] <= n_wdone[c]` in reset
samples the stage, which is 0 in reset as the bridge's count is (both sides' resets are the
same event, nothing in flight). The credit accounting is untouched: the bridge's `wret /
wn` retire against its own `n_wdone`, the adapter's `iss_w` against the stage, and the two
differ by one cycle's worth of counting at most, which the 16-bit modular compare absorbs.
tests/test_native_nch.py (the adapter at NCH 2 / 4 / 8 / 16, NCH 2 cycle-identical to the
two-channel adapter) and tb_f2hbm (`n_wdone` reaches the writes' count at idle, every write in
the HBM after it) pass.

**Expected recovery.** The reported path splits into a no-logic flip-flop -> flip-flop net
(the stage's D) and the chain from a register beside the adapter: the chain's 0.66 ns of logic
plus a local route, +2 ns and more of slack on both halves. On the token (the Gate 4 recipe,
below) the extra cycle is invisible: tb_top builds the board on otpu_native_mem's channel
model, not the fabric; on the card every `wr_idle` the slice waits for (a QST's completion,
HALT CHAIN, a read after a write) comes 5 ns later.

## 3. clk -> clk_f: the DMA's TMEM request into a pumped bank's address (-1.0 ns)

**What it was.** The pumped TMEM (TMEM_IMPL 3, TMEM_PMP 1, [tmem_pump.md](../tmem_pump.md):
one copy of true dual-port RAMs on clk_f, both ports reading at the core edge, the write
landing at the fast edge between) gives a read request two fast periods from the units'
request registers to the RAMs' pins (otpu_clk2x.tcl's multicycle; the report's 5.000 ns
requirement confirms it is in effect on this path, unlike rhino's round 4). The DMA's
`t_re_q` is release r2's input stage (the request from flip-flops, `adv` folded into its next
value), and from that one register the enable of lane 5 reaches TMEM's lane rotation, its bank
compares and the 16 pumped sub-banks' address AND-OR multiplexers (2 LUT levels into
`ADDRBWRADDR`) over 4.07 ns of route: one copy driving loads spread over the slice, which the
directive spread over SLRs.

**The options**, as the card ([tmem_pump.md](../tmem_pump.md) "UltraScale+: out of context")
and rhino list them:

- a register on the DMA's address path in the TMEM: not taken. A piece's read request is
  combinational from the units' request registers and the RAM samples it one core period
  later; a clk register at the bank would move the sampling to the next even edge, a core
  cycle on every TMEM read for every unit, which every unit's data pipeline and the replay
  protocol assume (rhino's round 7 reached the same conclusion). Not local, not cycle-exact.
- the uarch options: P3 in UltraRAM meets 200 / 400 MHz out of context (+0.180 ns, 48 URAM288
  in place of 192 RAMB36) and would free block RAM, but it is a RAM style change with its own
  synthesis risk and does not address the route, which is the problem here (the block RAM
  variant meets +0.195 out of context at these clocks: the arithmetic closes when the request
  is near its RAM).
- a pblock keeping the slice compact: the floorplan's (below).
- **replication** (commit ea4ac41e, taken): `(* max_fanout = 16 *)` on the DMA's `t_re_q` and
  `t_raddr` (now `t_rq`), the VPU's `ta_addr` / `tb_addr` (`ta_q` / `tb_q`) and the MXU's
  `t_raddr` (`t_rq`), rhino's 24422567 for the VPU and MXU with the DMA's added: synthesis
  keeps a copy of each request register per 16 loads, each beside them, and the distance moves
  to the copies' D inputs, a full core period behind one LUT (`adv_nx ? ... : 0`).

**Why bit-exact.** Attributes are not simulated and replication does not change a netlist's
function: cycle-exact and value-exact by construction. The output ports are driven from the
attributed registers through assigns; the DMA's `adv` prediction assertion still checks
`t_re_q` every cycle.

**Expected recovery.** The reported 4.07 ns of route becomes a copy's local route to its 16
loads (~1-1.5 ns): +1.5 ns on this path, and the same for the VPU's and MXU's request paths,
which rhino's round 2 placed at -2.4 to -3.0 ns on the free placement and which are the next
of this kind. The pblock the card's notes recommend (the TMEM beside the VPU, rhino's
`pb_otpu_tmem`) is the floorplan's step if these stay negative after the directive change.

## 4. The small ones

- **`**async_default**` -0.33 ns, `u_hbm/PIPE_HBM_AXI_RST/pipe_reg[3][0]` -> the HBM IP's
  `AXI_xx_ARESET_N`**: the HDK's cl_mem_hbm_wrapper (copied by build.py from
  hdk/cl/examples/cl_mem_perf, not ours) takes our `i_rst_450m_n = !hrst` (hbm_reset_n's third
  synchronizer flip-flop) through a `lib_pipe` of 4 stages on hclk, and the last stage drives
  the 8 ports' `ARESET_N` pins, which the IP uses as asynchronous resets inside (hence the
  recovery check). The driver is one flip-flop into every port's reset tree. Nothing in our RTL
  reaches it; the routed-stage phys_opt (AggressiveExplore, the flow's) replicates high-fanout
  register drivers and is the likely fix. If it persists post-route: build.py copies the
  wrapper, so a patched copy with a 5th stage per port (or `max_fanout` on the last) is a
  small flow change; a false path on the recovery check is defensible (the reset is held for
  the MMCM lock and three synchronizer cycles and released while every port is idle, and the
  IP's AXI interfaces re-synchronize nothing until a transaction comes) but was not set here.
- **clk_main_a0 -0.29 / -0.24 ns on hidden shell cells**: the shell's own encrypted cells both
  ends, 4 ns paths; route and phys_opt usually recover these. If they persist and the
  endpoints are ours (otpu_f2_hbm_host's `h_*ready`, otpu_axil_cdc's), rhino's skid buffers
  on the PCIS and OCL sides (`otpu_skidbuf`, its 7d3da502 / 1197a113) make every shell-facing
  output a flip-flop; not needed at this slack.

## Clocks and CDC after the change

The three domains' crossings ([f2_hbm.md](../f2_hbm.md) "Clocks and CDC") are the same set:
per channel the bridge's command afifo (clk -> hclk, 602 bits) and return afifo (hclk -> clk,
512), its write count (hclk -> clk, gray) and error (a level); the host path's three afifos
(aclk <-> hclk) and its write count (gray); the OCL crossing (aclk <-> clk, held words); the
resets (synchronized levels). None of the new registers crosses a domain: the hregs are hclk
only, the write-count stage is clk only and sits after the bridge's synchronizer (the
constraint's `$ch/bgray_reg* -> $ch/bg1_reg*` bus skew finds the same cells), the reset
registers are synchronous copies of synchronized levels, the replicated request registers are
clk and their loads keep the pump's multicycle (the RAM pins are the same). The constraint
file's object counts (3 + 2 x NPORT afifos, NPORT + 1 write counters, fail closed) are
unchanged; what changed is the max-delay value per direction, above.

## The next build

What the coordinator's next run should take, in order of expected effect:

1. **`--place-directive Default`** (or `SSI_BalanceSLLs`): build.py's new option; the default
   stays `SSI_SpreadLogic_high` so nothing changes without asking. One slice of ~150K LUTs and
   ~30K of fabric fits one SLR with room; spreading it is what put 4-6 ns of route on 2-level
   paths in every domain. Rhino's round 2 (`SSI_SpreadLogic_low`) still split the slice over
   two SLRs; their round 3 with pblocks took the core clocks from -4.05 to -1.42 ns.
2. **A floorplan**, if 1 leaves the slice split: rhino's cl_pnr_user_hbm.tcl (the fabric
   `u_sys/u_mem` in SLR0 by the HBM's ports, `u_sys/u_board` in SLR2, the TMEM beside the VPU
   in a child pblock), which needs its flow fix (the HDK reads `small_shell_cl_pnr_user.xdc`
   as a managed XDC and drops every Tcl line: rhino's 76a8f878 moves the Tcl to
   `cl_pnr_user.tcl`, read unmanaged inside an `opt_design` wrapper in build_level_1_cl.tcl)
   and, with the board two SLRs from the fabric, its SLR crossing register chains on the
   native ports (`otpu_xr`, `NXR` 3, the bridges' FIFOs 4 -> 16 with the ready's threshold
   lowered: 24422567; +6 core cycles of memory latency, which cost rhino 6-7% of the token on
   the native model at --lat 66). None of it is in this branch. One thing to check in
   61d07199's vivado.log meanwhile: `CRITICAL WARNING: [Designutils 20-1307]` on the read of
   `small_shell_cl_pnr_user.xdc`. Rhino found the HDK reads that file as a managed XDC and
   drops its Tcl, so the linked-reset exception runs on no pins (the synchronizers' CLR
   recovery paths then stay timed: safe, only pessimistic, and a place where small violations
   can appear); the same flow fix would carry the exception.
3. **The bridge at 450 MHz**: after the host path, rhino's hclk domain stayed 2.0-2.25 ns
   short in otpu_f2_hbm_nch's own cones (the head's load enables, the W count: both taken
   here; then the R-return FIFO's pointer into its LUT RAM write enables, the head's `k_nx`
   compares, the open burst's updates) with ~25-30K failing endpoints across three rounds. Each
   cone can be brought to 3 LUT levels by hand (rhino's list in docs/wide_port.md "Round 5
   RTL" 7); the honest fallback is hclk at 300 MHz on 16 pseudo-channels (f2-hbm16: 16 x 256
   bits x 300 MHz = 153 GB/s raw, 6x what one slice streams), which fits 5-6 level cones with
   1.5 ns of route. The post-route report of this branch's build decides.

## Round 3: build 2 (a2af0904, `--place-directive Default`) and its fixes

Build 2 (this branch at a2af0904, placed with `Default`) went from WNS -3.66 / TNS -68,500 to
WNS -2.19 / TNS -25,900 after phys_opt (WHS -0.75 there), still failing; the post-place
report's worst paths (50 endpoints, 10 each on the first two):

| group | worst path | slack | data path | what it is |
|---|---|---|---|---|
| hclk_raw | `u_host/hr_v_reg/C` -> `u_host/u_rt/mem_reg_0_15_434_447/RAMA/WE` | -2.60 | 4.451: logic 0.166 (2 levels), route 4.285 | the host's 512-bit read-return FIFO (LUT RAM, 64 SLICEMs): its write enable from one register through `r_zero` into every cell |
| clk_raw | `g_port[2].u_nch/cf_h_reg[0]/C` -> `u_nch/u_xq/mem_reg_0_15_518_531/RAMA/WE` | -2.42 | 7.119: logic 0.869 (8 levels), route 6.250 | the bridge's command FIFO head (a LUT RAM read by `cf_h`) through the 80 hazard compares and the issue into the 602-bit crossing's write enables |
| clk_raw -> clk_f_raw (req 5.000) | `u_dma/t_re_q_reg[2]/C` -> `g_bank[5].g_sub[0].g_pmp.u_pb/g_b.mem_reg_2/ADDRBWRADDR[11]` | -1.41 (-1.34, -1.32 siblings) | 4.899: logic 0.501 (8 levels), route 4.398 | the DMA's request through TMEM's lane rotation, rank select and address AND-OR into a bank's pins: the replication did not help, a lane's enable has one load per bank and every bank is elsewhere |
| clk_main_a0 | a hidden shell path | -1.03 | 4.652: logic 0.347 (3 levels), route 4.305 | our side of the PCIS / OCL handshake placed far from the shell's interface registers |
| `**async_default**` | `u_hbm/PIPE_HBM_AXI_RST/pipe_reg[3][0]` -> `HBM_SNGLBLI_INTF_AXI<15..11>_INST/ARESET_N` | -0.44 .. -0.89 | route 2.3-2.8 | one reset pipe register into the AXI resets of both stacks' 32 ports |

**RTL** (bit-exact; the bridges' and host path's changes are not on a token's numerics; the
suites and the token below):

1. **The wide crossing FIFOs written at their tail every cycle** (`otpu_afifo` `WT` with `RWR`,
   rhino's e165f4f8): the RAM's write enable is a registered copy of the room (`wput_q`,
   `max_fanout 32`, as the address copy `wa_q` already was), the tail slot takes whatever
   `wdata` shows every cycle there is room, and `wvalid` moves only the pointer; the slot at
   the write pointer is never live while the room is up (the room is against the read pointer
   as synchronized, which only grows), so the write is safe. On the host's read-return FIFO
   `u_rt` (the -2.60 path: `hr_v` now reaches 5 pointer flip-flops, not 64 SLICEMs' WE pins),
   the host's PCIS W crossing `u_xw`, the bridge's command crossing `u_xq` (the -2.42 path's
   end) and its read-return FIFO `u_rq` (the same 512-bit shape at 450 MHz, the next of its
   kind). The writers that never see these FIFOs full (`u_rt`: RQ beats in flight, `u_rq`: RD
   reads in flight) are unaffected by the registered ready's one cycle of conservatism.
2. **The bridge's command head in a register** (`h_v / h_we / h_a`, rhino's e165f4f8): the
   hazard compares and the issue start from flip-flops, not from the FIFO's LUT RAM read by
   `cf_h`; with WT the issue reaches only the crossing's pointer. The -2.42 path's 8 levels
   become the compares and the issue alone (no RAM read in front, no 600 enables behind). A
   command issues one core cycle later than before (5 ns on the HBM's ~400); the order check,
   the request order and the adapter are untouched.
3. **The host's PCIS side behind registers** (rhino's rounds 4-7: 7d3da502, 1197a113, 98aa0368,
   round 7; `rtl/shell/otpu_skidbuf.sv`, `otpu_link.sv`, `otpu_f2_hbm_oreg`, the one-bit stage
   `otpu_f2_hbm_xr`): AW, W and AR enter two-entry skid buffers whose ready is a flip-flop
   (`u_pcis_aw / _w / _ar`), B and R leave through output registers (`u_pcis_b / _r`), and
   between those registers (for the floorplan: beside the shell, SLR1) and the queues (SLR0)
   runs an `otpu_link` per channel (the sender registers the take with its beat, the receiver
   registers them again and queues them, the credit comes back as a registered pulse), so
   every net across the boundary is a flip-flop to a flip-flop in both directions. The order
   rule (`pcis_wbusy` holds the reads and the OCL's writes) is formed from registers only: the
   queues, the unanswered B's and the write-pending bit of the registers' side, registered
   twice on its way (`u_pcis_wb`, `u_xs_wb`); the OCL crossing (`otpu_axil_cdc`) takes its
   AW / W / AR through skid buffers of its own and starts a write only from a beat two cycles
   old, so a PCIS write shown with an OCL write is counted first (tb_f2axil checks the start,
   not the handshake, against the hold). Cost: +2 aclk cycles on a host access's way in and
   +2 on its response's way out; nothing on the token. The structural test
   `test_pcis_registers_only_drive_the_link_stage` keeps the registers' side from reaching
   the queues' logic any other way.

**Constraints only (no RTL)**:

4. **The floorplan** (`boards/f2/build/constraints/cl_pnr_user_hbm.tcl`, appended by build.py to
   `cl_pnr_user.tcl` for an HBM target): four pblocks under the HDK's `pblock_CL`.
   `pb_otpu_slr0` (SLR0's 20 CL clock regions, ~272K LUTs): the board and the HBM fabric
   together (175K, 64%), beside the HBM's AXI ports under SLR0, so no slice <-> fabric net
   crosses an SLR. `pb_otpu_tmem` (inside it, X0Y2:X3Y3, 8 regions): the TMEM with the VPU,
   the DMA and the quantizer (66K, 61%; 82 of the box's ~170 RAMB36), the units whose request
   registers reach the pumped banks in two fast periods: the -1.41 path's 4.4 ns of route
   becomes a box of 2 x 4 regions; the MXU wraps around it. `pb_otpu_pcis` (SLR1, X5Y5:X5Y6,
   beside the shell's PCIS / OCL column): the PCIS skids, output registers, link ends and
   `u_ocl`, so the shell -> register hop (the -1.03 path) stays in one SLR; `pb_otpu_xs` (SLR0,
   X4Y3): the links' SLR0 ends, so the SLR1 -> SLR0 hop is flip-flop to flip-flop and the
   stage -> queue paths stay in SLR0. Every pblock is cut down to the sites its parent holds
   (`otpu_contain`: the HDK's `pblock_CL` is not whole clock regions and opt_design's HDPR-23
   DRC rejects a child outside it), `CONTAIN_ROUTING` off, the cell queries failing closed.
   The flow fix that makes it apply (rhino's 76a8f878): the HDK reads
   `small_shell_cl_pnr_user.xdc` as a managed XDC and drops its Tcl (so the linked-reset
   exception had run on no pins in builds 1 and 2), so openTPU's placement Tcl is
   `cl_pnr_user.tcl`, read unmanaged by our `build_level_1_cl.tcl` inside a wrapped
   `opt_design` (otpu_clk2x.tcl's multicycles re-read with it on the linked design), the
   managed XDC holding comments only; tests/test_f2_build.py runs the floorplan and the
   wrapper under Tcl mocks (containment, cell lists, order, failing closed).
   What is not boxed: the MXU (wrapping the TMEM box: its `t_raddr` reaches the banks from
   outside, the same path family; if it shows, the box grows or the MXU's request registers
   get a box of their own beside it), the adapter and the bridges (free in SLR0 with the HBM
   ports), the top's MMCMs and synchronizers.
5. **The HBM AXI reset pipe replicated at synthesis** (`cl_synth_user_hbm.xdc`, read by the HBM
   synthesis script after `cl_synth_user.xdc`): `MAX_FANOUT 8` on the net of the wrapper's
   `PIPE_HBM_AXI_RST/pipe_reg[3]*/Q`, the HDK's own idiom for a reset driver (cl_mem_perf's
   `cl_synth_user.xdc` puts `MAX_FANOUT 50` on its `sync_rst_n_reg`'s net), so synthesis keeps
   a copy of the last stage per 8 of the IP's 32 `ARESET_N` loads and the placer can put each
   beside its ports (the recovery check's 2.3-2.8 ns of route become a quarter). The wrapper
   itself is the HDK's, copied by build.py, untouched; cl_mem_perf's constraints have no
   timing exception on this reset (its `cl_timing_user.xdc` carries multicycles on its own
   `PIPE_*_HBM` registers and false paths on its FIFOs only), so none is set here.

Not changed, and why: the DMA -> TMEM address stage (item 3's other option). A piece's read
request is combinational from the units' request registers and the RAM samples it at the
even edge one core period later; a clk register at the TMEM would move the sampling to the
next even edge, a core cycle on every TMEM read for every unit (the lanes take the word the
cycle after the grant; the replay buffers, the DMA's stream fill and the quantizer's three
reads assume it), so the VPU's, MXU's and quantizer's request registers would all move with
it: a slice-wide protocol change, not a local one. The uarch options stay as in round 1 (P3 in
block RAM is tmem_pump.md's pick for this part at 200 / 400 MHz; the arithmetic closes when
the request is near its RAM, which the box is for). The WHS -0.75 after phys_opt was not
detailed in the report; the pump's write stage (`pk_*`, one fast period) and the LUT RAM
crossings are the usual sources, and the route stage's hold fixing is what to read first.

## Round 4: build 3 (8b2fd3a0, the floorplan applied) and its fixes

Build 3 routed at WNS -1.398 / TNS -29,031, WHS +0.001, every constraint applied
(`OTPU_LINKED_RESET_EXCEPTION_PASSED endpoints=12`, `OTPU_FLOORPLAN_APPLIED pblocks=4`), and
failed qualification on timing. The post-route report's worst paths (40 endpoints):

| group | worst path | slack | data path | what it is |
|---|---|---|---|---|
| hclk_raw | `u_host/pv_w_reg[6]_replica` -> `u_host/hw_wd_reg[30]/CE` | -1.40 | 3.484: logic 0.361 (5 levels), route 3.123 | the shown-half register's enable: the W take (8 ports' AND-OR) and the half select into 288 clock enables |
| hclk_raw | `u_host/hw_wd_reg[31]` -> `g_port[0].u_hreg/m_wdata_reg[31]/D` | -1.39 | 3.507: logic 0.081 (0 levels), route 3.426 | a bare wire: the host block sits at SLR0's top by its PCIS link stages (pb_otpu_xs), the ports' registers along the HBM below |
| clk_raw | `u_mem/n_wdone_reg[7][10]` -> `u_slice/c_n_reg[24]/CE` | -0.99 | 5.557: 0.758 (9 levels), route 4.799 | the write-count stage through the adapter's compare, wr_idle and the slice's halted into a counter's enable |
| clk_raw | `g_port[6].u_nch/h_a0_reg[16]` -> `u_nch/ra_reg[56][12]/CE` | -0.99 | 5.610: 0.702 (7), route 4.908 | inside a bridge: the head's address through the hazard compares and the issue into the read-address array's enables, the bridge's core side stretched over the SLR0 box |
| clk_raw | `u_board/u_mem/wh_r_reg[4][3]` -> `u_mem/whb_reg[4][36][2]/CE` | -0.98 | 5.752: 0.582 (7), route 5.170 | the adapter's SW-queue bucket hash into the hazard counters' enables |
| clk_raw -> clk_f_raw | `u_mxu/t_rq_reg[1][2]` -> bank 6's `ADDRBWRADDR[13]`; `u_tmem/g_one.rpv_reg[6]_inv` -> bank 1's `ADDRBWRADDR[12]`; siblings | -0.47 .. -0.39 | 4.72-4.80: 0.67-0.78 (6-7), route 4.0 | the MXU's request registers (and Vivado's replicas) outside the TMEM box; the replay valid through the priority select |

**RTL** (bit-exact; cycle-exact except where said):

1. **The host sends the whole 64-byte beat as one 512-bit W** (otpu_f2_hbm_host; the port's
   register `otpu_f2_hbm_hreg` sends it as its two AXI beats, `w_hi` the second, one LUT of
   select into the arbiter's register beside it). The beat's registers (`hw_d`, `hw_s`) load
   on the head's load alone (`hw_ld = xw_v && !hw_v`, replicated per 64 enables): the shown
   half, its select and the 5-level enable cone are gone; the state machine is AW then W.
2. **A link of `HNS` register stages per port and channel** (`otpu_f2_hbm_hlink` in
   otpu_f2_hbm_mem: `otpu_link`'s sender at the host block, HNS = 2 stages forward for the
   beat and back for the pop pulse (`otpu_f2_hbm_xr`, placed along the way), `otpu_link`'s
   receiver with its queue at the port; B and R back through HNS stages), so the 3.4 ns of
   wire from the host block to a port is three wires of a cycle each; a beat crosses in HNS +
   2 cycles, DEPTH = 4 beats per credit round trip (the host sends one per ~7). The host's
   readies are the senders' credit flip-flops beside it. Not on the token.
3. **The adapter's write-done compare a register** (`otpu_native_dram` `WDQ`, set by
   otpu_f2_hbm_sys through otpu_board; 0 elsewhere, so the DDR boards, tb_top and the
   two-channel equivalence test are untouched): `iss_all_q` compares each channel's count after
   this cycle's issue with the channel's count as it is now, so it is the combinational
   compare one cycle later and never earlier (a write accepted later is covered by `wq_n /
   wacc_q` until its issue, its issue by the count's lag), and `wr_idle` is one LUT level from
   registers. The 9-level path becomes the stage -> a local compare -> a flip-flop, then the
   flip-flop -> wr_idle -> the slice. tb_native_nch runs it at NCH 8 (`WDQ 1`); on the card
   every wr_idle the slice waits for comes one more core cycle later.
4. **The adapter's fill-read hazard check as a register** (rhino's a1efd022, cherry-picked
   clean): `w_blk` kept equal to `wnz[c][wh_r[c]]` from three candidate live bits selected early,
   so the bucket hash no longer runs a 64:1 select in front of the take and the counters'
   enables (the `wh_r -> whb` family); cycle-exact (the register mirrors the select, checked
   every cycle in simulation; the NCH 2 equivalence test passes).
5. **The TMEM's priority mask a register** (otpu_tmem `bf`): computed a cycle early from the
   replay valids' next values (`rpv <= lost`) and the age's next value (`age_nx`), so the rank
   multiplexer into the pumped banks' addresses (`win = rq && !(rq & bf)`) starts from a
   flip-flop instead of `rpv` through the four-way select (the `rpv_reg_inv` path loses that
   level and Vivado's inverter). Cycle-exact: a simulation assertion compares the register
   with the select every cycle; the slice suites that exercise replays pass.

**Constraints only (no RTL)**:

6. **`pb_otpu_fab`** (a child of pb_otpu_slr0, X0Y0:X3Y0, SLR0's bottom row by the HBM's AXI
   ports): the 8 bridges, the host registers, the port arbiters and the links' receiving ends
   (30K LUTs, 56%), each bridge compact beside its port: the 4.9 ns inside a bridge and the 4.8
   ns from the write count's stage come from the bridges' core sides stretched up toward the
   adapter. The adapter (39K, LUT RAM heavy) is not boxed: the placer puts it between the
   bridges below and the slice above; the host block stays free by its PCIS stages (the links
   carry the distance).
7. **The MXU's request registers in the TMEM box** (`u_mxu/t_rq_reg*`, `t_ren_reg*`: the leaf
   cells and Vivado's `_rep` replicas, not the MXU), beside the banks they address: the -0.47
   family. Their D inputs come from the MXU's request logic over a full core period.
8. **Post-route recovery**: yes, run boards/f2/route_opt.py (phys_opt AggressiveExplore on the
   routed checkpoint) on build 4 if its WNS is within ~-0.5 ns: the clk -> clk_f family is
   route-bound at that scale and the recovery passes move exactly such paths; it is not worth
   it below -1 ns, where a floorplan or RTL change is needed.

## Verification

Verilator 5.048, one run at a time, from the worktree root (the tree's `.venv`):

- `tests/test_f2_rtl.py`: 13 passed (tb_f2hbm on 4 / 8 / 16 / 32 ports with HAZ 0 caught,
  the DDR build's otpu_f2_nch / otpu_f2_mem protocol and ordering, the OCL crossing).
- `tests/test_native_nch.py`: 10 passed (the adapter at NCH 2 / 4 / 8 / 16 and above 4 GiB; NCH
  2 cycle-identical to the two-channel adapter 4e9616a4).
- `tests/test_f2_build.py` and `tests/test_f2_scoped_clocks.py`: 149 passed (the HBM snapshot
  takes cl_timing_user_hbm.tcl byte for byte; the scoped-clock classifier).
- **Gate 4** ([f2-port-status.md](f2-port-status.md)): one Gemma 3 decode token through the
  NCH 8 board model (`OTPU_NCH=8 OTPU_AXI=1 PYTHONPATH=. python /tmp/f2port_rtl_token.py
  <tree>`: the prompt "Why is the sky blue?" prefilled on the ISA simulator, the step at its
  last token on the RTL and on the ISA simulator from the same resident program and DRAM):
  **bit-exact**: both pick token 19058 `Okay`, `logits identical: True, max |diff| 0.0, DRAM
  after the step identical: True`; the ISA step 27 s, the RTL step 568 s; **8,029,592 cycles,
  the number before this branch** (the register stages are in the fabric and the attributes
  are replication, so the board model's cycles are the same by construction; the channel
  statistics and `mxu_block` counters match f2-port-status.md's to the count). The fabric's
  extra cycle per write count and the host path's two show only on the card.

Round 3 (b3c27ee6 the RTL, 7c708acf the constraints and flow): tests/test_f2_rtl.py 14 passed
(the fabric at 4 / 8 / 16 / 32 ports with HAZ 0 caught, the DDR protocol and ordering, the OCL
crossing at both clock ratios, the structural link-stage test), tests/test_native_nch.py 10
passed, tests/test_f2_build.py + test_f2_scoped_clocks.py + test_target.py 177 passed (the
floorplan and the link wrapper under Tcl mocks); the Gate 4 token bit-exact again: token
19058 `Okay`, logits identical, DRAM identical, 8,029,592 cycles (the board model's sources
are unchanged: the WT mode is off on otpu_mem_ch's FIFOs, the fabric is not in tb_top).

Round 4 (566a30f8 the RTL, d61ffcee the floorplan): tests/test_f2_rtl.py 14 passed,
tests/test_native_nch.py 11 passed (NCH 2 cycle-identical to the two-channel adapter at WDQ 0,
NCH 8 at WDQ 1), tests/test_rtl.py -k "replay or act_reads or attention or lane_count" 27
passed, 1 skipped (the replays through the registered priority mask, its assertion every
cycle), tests/test_f2_build.py + test_f2_scoped_clocks.py + test_target.py 177 passed; the Gate 4
token bit-exact: token 19058 `Okay`, logits identical, DRAM identical, 8,029,592 cycles
(cycle-exact: the mask and the hazard registers mirror their selects, WDQ is the HBM board's
only, the fabric is not in tb_top).

The commits: 45ec6579 (the fabric: host, mem, port, nch), ea4ac41e and 3686f79e (the units'
request registers; the DMA's is `t_ra_q`, `t_rq` being its port-B request register), 9ef46073
(the CDC bound, `--place-directive`, docs/f2_hbm.md), and this document.

## Round 5

Structural: every openTPU block of the fabric moved to the core clock and the HDK's
cl_axi_sc_1x1 SmartConnect put in front of every HBM port for the 450 MHz side, the way Alex's
burst200 image did it: [f2-round5.md](f2-round5.md) (the comparison of the two fabrics, the
plan, the tuning and floorplan options, the build command on opentpu-f2).
