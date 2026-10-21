// The memory of a slice whose MXU runs C lock-step chunk lanes (otpu_mxu_lanes, MXU_C = C;
// docs/v4/proposal-2-four-lanes.md, docs/v4/p2-f2.md "M4"): C copies of the slice's DRAM image,
// copy l on its own channel pair (native channels 2 l and 2 l + 1) behind its own two-channel
// adapter (otpu_native_dram, unchanged: the adapter the burst200 F2 image ran Gemma with, the one
// P2's M2 gate ran lane 0 through). Lane l reads its chunks (lb_*) and scale lines (la_*) from
// copy l on copy l's adapter's ports B and A; lane 0 is the slice's own ports A and B (with the
// DMA) on copy 0. Every write goes to every copy: port SW (the quantizer's QST stores), port B's
// writes (the DMA's ST) and port A's (word writes), each taken only in a cycle every copy's
// adapter takes it, so the copies hold the same image at every point the slice can observe.
//
// Why copies, not one image over all 2 C channels (the proposal's 32-PC adapter with a 4 x 4
// chunk crossbar): the lanes read rows n + l of every MM at arbitrary row strides, so on one
// shared image their chunks collide on channels and come back out of order, which needs a
// crossbar, per-lane reorder buffers and per-channel per-source response FIFOs (the LUT-RAM that
// congested the wide-port build's adapter, docs/wide_port.md). On a copy per lane each lane has
// its own channels and a plain in-order adapter: no crossbar, no reordering, timing local to the
// lane, and each channel streams its lane's consecutive chunks (bursts form). The costs: C x the
// image in memory (Gemma 3 1B ~1.05 GB a copy: 4.2 GB of the F2's 16 GB), the host writes every
// copy (opentpu/host/board.py `copies`), and a write occupies every copy's channels.
//
// Ordering is the adapter's (otpu_native_dram "Ordering"), per copy: a write taken by all copies
// in one cycle enters each copy's queues there; wr_idle is every copy's (a read after wr_idle,
// on whichever copy, sees the write); rd_idle every copy's (lane reads included). a_flush drops
// every copy's reused beat and runs. The host's writes reach every copy through the host's own
// port (the memory model: each copy reads the same poke files).
//
// Readies depend only on registered state (the adapters' queue counts) and the slice's port
// registers (b_we, a_we: requests already registered, otpu_slice "port B's registers"): a write's
// readiness is the AND of every copy's, a read's its copy's; a lane's read waits while its
// copy's port is taking a broadcast write.
//
// C > 1 (eight copies spread over the SLR: P3's placed build, docs/v4/p3-status.md): every ready
// is a register of the copies' "room for two" (otpu_native_dram sw_rdy2 / b_rdy2 / a_rdy2: a
// port's queues grow by one request a cycle at most, so room for two now is room for one next
// cycle), and a request reaches a copy only when that ready says it is taken; wr_idle and rd_idle
// are registers too (every copy idle and no request of the kind this cycle: exact, or a cycle
// late). No copy's counts then run into the slice's port registers, the DMA, the lanes' issuers
// or the copies' queue enables in the cycle they change. C 1: the adapter's own, cycle for cycle.
module otpu_lane_mem #(
  parameter int D = 128,
  parameter int C = 4,                 // lanes = copies
  parameter bit CHASH = 1'b1,
  parameter int QD = 16,
  parameter int WQD = 32,
  parameter int WGATHER = 4,
  parameter int RD = 128,
  parameter int AD = 64,
  parameter int APF = 32,
  // the adapters' A runs take forward skips (otpu_native_dram APS): each lane's scale rows are C
  // rows apart (P3, docs/v4/p3-status.md "lane 0's scale beats")
  parameter bit APS = (C > 1),
  parameter bit SWFILL = 1'b1,         // SW partial beats with a fill read (otpu_native_dram)
  // the copies' requests through a register in each adapter (otpu_native_dram IREG; C > 1 only):
  // the copies an SLR from the slice (P3's F2 floorplan, lane_memory_slr 1)
  parameter bit IREG = 1'b0
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              a_flush,
  // the slice's ports (lane 0's reads, everyone's writes), as otpu_native_dram's
  output logic              a_rdy_x,
  input  logic              a_req_x,
  input  logic              a_we_x,
  input  logic [31:0]       a_addr_x,
  input  logic [31:0]       a_wdata_x,
  input  logic [3:0]        a_be_x,
  output logic              a_rvalid,
  output logic [255:0]      a_rdata,
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [31:0]       sw_addr,
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [31:0]       b_addr,
  input  logic              b_par,
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              wr_idle,
  output logic              rd_idle,
  // the lanes' read ports (otpu_slice lb_* / la_*; [0] unused: lane 0 is ports A / B)
  input  logic [C-1:0]          lb_req,
  input  logic [C-1:0][31:0]    lb_addr,
  output logic [C-1:0]          lb_rdy,
  output logic [C-1:0]          lb_rvalid,
  output logic [C-1:0][D*8-1:0] lb_rdata,
  input  logic [C-1:0]          la_req,
  input  logic [C-1:0][31:0]    la_addr,
  output logic [C-1:0]          la_rdy,
  output logic [C-1:0]          la_rvalid,
  output logic [C-1:0][255:0]   la_rdata,
  // native channels: copy l on [2 l] (half 0) and [2 l + 1]
  output logic [2*C-1:0]        n_cvalid,
  input  logic [2*C-1:0]        n_cready,
  output logic [2*C-1:0]        n_cwe,
  output logic [2*C-1:0][24:0]  n_caddr,
  output logic [2*C-1:0]        n_wvalid,
  input  logic [2*C-1:0]        n_wready,
  output logic [2*C-1:0][511:0] n_wdata,
  output logic [2*C-1:0][63:0]  n_wmask,
  input  logic [2*C-1:0]        n_rvalid,
  input  logic [2*C-1:0][511:0] n_rdata,
  input  logic [2*C-1:0][15:0]  n_wdone
);
  // (C 1: one copy, the slice's ports on one adapter, cycle for cycle the adapter alone: otpu_board
  // keeps its memory's name, u_mem, for every build)
  initial if (C < 1 || C > 16) $fatal(1, "otpu_lane_mem: C must be 1 to 16 (a copy per lane)");

  // each copy's adapter ports
  logic [C-1:0]          ca_rdy, ca_req, ca_we, ca_rvalid;
  logic [C-1:0][31:0]    ca_addr, ca_wdata;
  logic [C-1:0][3:0]     ca_be;
  logic [C-1:0][255:0]   ca_rdata;
  logic [C-1:0]          cs_rdy, cs_rdy2, cs_req;
  logic [C-1:0]          ca_rdy2, cb_rdy2;
  logic [C-1:0]          cb_rdy, cb_req, cb_tag, cb_we, cb_par, cb_rvalid, cb_rtag;
  logic [C-1:0][31:0]    cb_addr;
  logic [C-1:0][D*8-1:0] cb_rdata;
  logic [C-1:0]          c_wr_idle, c_rd_idle;

  // a broadcast write goes when every copy takes it; a read is its copy's alone
  wire a_wr = a_req_x && a_we_x;
  wire b_wr = b_req && b_we;
  // each copy's port readies as the slice and the lanes see them: C 1, the adapter's; C > 1, the
  // registers of room for two (above)
  logic [C-1:0] ca_ok, cb_ok;
  logic         a_all, b_all;
  if (C > 1) begin : g_rq
    logic [C-1:0] ca_q, cb_q;
    logic         a_all_q, b_all_q, sw_q, wi_q, ri_q;
    always_ff @(posedge clk) begin
      ca_q    <= rst ? '0 : ca_rdy2;
      cb_q    <= rst ? '0 : cb_rdy2;
      a_all_q <= !rst && (&ca_rdy2);
      b_all_q <= !rst && (&cb_rdy2);
      sw_q    <= !rst && (&cs_rdy2);
      // idle: every copy idle now and no request of the kind this cycle (a request taken now
      // makes its copies busy next cycle; one not taken is still the slice's): idle next cycle
      wi_q    <= !rst && (&c_wr_idle) && !sw_req && !b_wr && !a_wr;
      ri_q    <= !rst && (&c_rd_idle) && !(a_req_x && !a_we_x) && !(b_req && !b_we) &&
                 !(|lb_req) && !(|la_req);
    end
    assign ca_ok = ca_q;
    assign cb_ok = cb_q;
    assign a_all = a_all_q;
    assign b_all = b_all_q;
    assign sw_rdy  = sw_q;
    assign wr_idle = wi_q;
    assign rd_idle = ri_q;
  end else begin : g_r1
    assign ca_ok = ca_rdy;
    assign cb_ok = cb_rdy;
    assign a_all = &ca_rdy;
    assign b_all = &cb_rdy;
    assign sw_rdy  = &cs_rdy;
    assign wr_idle = &c_wr_idle;
    assign rd_idle = &c_rd_idle;
  end
  assign a_rdy_x = a_we_x ? a_all : ca_ok[0];
  assign b_rdy   = b_we ? b_all : cb_ok[0];

  for (genvar l = 0; l < C; l++) begin : g_cp
    if (l == 0) begin : g_s
      // copy 0: the slice's ports as they are (a write waits for the other copies; C > 1, a read
      // for the registered ready, so the copy takes exactly what the slice sees taken)
      assign ca_req[0]   = a_req_x && (a_we_x ? a_all : (C == 1 || ca_ok[0]));
      assign ca_we[0]    = a_we_x;
      assign ca_addr[0]  = a_addr_x;
      assign cb_req[0]   = b_req && (b_we ? b_all : (C == 1 || cb_ok[0]));
      assign cb_tag[0]   = b_tag;
      assign cb_we[0]    = b_we;
      assign cb_addr[0]  = b_addr;
      assign cb_par[0]   = b_par;
      assign lb_rdy[0] = 1'b0; assign lb_rvalid[0] = 1'b0; assign lb_rdata[0] = '0;
      assign la_rdy[0] = 1'b0; assign la_rvalid[0] = 1'b0; assign la_rdata[0] = '0;
    end else begin : g_l
      // copy l: lane l's reads, unless a broadcast write takes the port this cycle (a read only
      // with the copy's ready as the lane sees it)
      assign ca_req[l]   = a_wr ? a_all : (la_req[l] && ca_ok[l]);
      assign ca_we[l]    = a_wr;
      assign ca_addr[l]  = a_wr ? a_addr_x : la_addr[l];
      assign cb_req[l]   = b_wr ? b_all : (lb_req[l] && cb_ok[l]);
      assign cb_tag[l]   = b_wr && b_tag;
      assign cb_we[l]    = b_wr;
      assign cb_addr[l]  = b_wr ? b_addr : lb_addr[l];
      // (lb_addr is the slice's lane register: its 27-bit parity starts at a flip-flop)
      assign cb_par[l]   = b_wr ? b_par : ^lb_addr[l][31:5];
      assign la_rdy[l]    = ca_ok[l] && !a_wr;
      assign lb_rdy[l]    = cb_ok[l] && !b_wr;
      assign la_rvalid[l] = ca_rvalid[l];
      assign la_rdata[l]  = ca_rdata[l];
      assign lb_rvalid[l] = cb_rvalid[l];
      assign lb_rdata[l]  = cb_rdata[l];
    end
    assign ca_wdata[l] = a_wdata_x;
    assign ca_be[l]    = a_be_x;
    assign cs_req[l]   = sw_req && sw_rdy;
    otpu_native_dram #(.D(D), .QD(QD), .WQD(WQD), .CHASH(CHASH), .WGATHER(WGATHER), .RD(RD),
                       .AD(AD), .APF(APF), .APS(APS), .SWFILL(SWFILL), .IREG(IREG && C > 1)) u_adapt (
      .clk, .rst, .a_flush,
      .a_rdy_x(ca_rdy[l]), .a_req_x(ca_req[l]), .a_we_x(ca_we[l]), .a_addr_x(ca_addr[l]),
      .a_wdata_x(ca_wdata[l]), .a_be_x(ca_be[l]), .a_rvalid(ca_rvalid[l]), .a_rdata(ca_rdata[l]),
      .sw_rdy(cs_rdy[l]), .sw_rdy2(cs_rdy2[l]), .b_rdy2(cb_rdy2[l]), .a_rdy2(ca_rdy2[l]),
      .sw_req(cs_req[l]), .sw_addr, .sw_wdata, .sw_be,
      .b_rdy(cb_rdy[l]), .b_req(cb_req[l]), .b_tag(cb_tag[l]), .b_we(cb_we[l]), .b_wmask,
      .b_wdata, .b_addr(cb_addr[l]), .b_par(cb_par[l]), .b_rvalid(cb_rvalid[l]),
      .b_rtag(cb_rtag[l]), .b_rdata(cb_rdata[l]), .wr_idle(c_wr_idle[l]), .rd_idle(c_rd_idle[l]),
      .n_cvalid(n_cvalid[2*l +: 2]), .n_cready(n_cready[2*l +: 2]), .n_cwe(n_cwe[2*l +: 2]),
      .n_caddr(n_caddr[2*l +: 2]), .n_wvalid(n_wvalid[2*l +: 2]), .n_wready(n_wready[2*l +: 2]),
      .n_wdata(n_wdata[2*l +: 2]), .n_wmask(n_wmask[2*l +: 2]), .n_rvalid(n_rvalid[2*l +: 2]),
      .n_rdata(n_rdata[2*l +: 2]), .n_wdone(n_wdone[2*l +: 2]));
  end
  assign a_rvalid = ca_rvalid[0];
  assign a_rdata  = ca_rdata[0];
  assign b_rvalid = cb_rvalid[0];
  assign b_rtag   = cb_rtag[0];
  assign b_rdata  = cb_rdata[0];

`ifndef SYNTHESIS
  // C > 1: what the slice and the lanes see taken, every copy takes (a registered ready implies
  // the copy's own); the idles are never early
  always_ff @(posedge clk)
    if (!rst && C > 1) begin
      for (int l = 0; l < C; l++) begin
        if (ca_req[l] && !ca_rdy[l]) $fatal(1, "otpu_lane_mem: copy %0d's A request without room", l);
        if (cb_req[l] && !cb_rdy[l]) $fatal(1, "otpu_lane_mem: copy %0d's B request without room", l);
        if (cs_req[l] && !cs_rdy[l]) $fatal(1, "otpu_lane_mem: copy %0d's SW request without room", l);
      end
      if (wr_idle && !(&c_wr_idle)) $fatal(1, "otpu_lane_mem: wr_idle with a copy's write pending");
      if (rd_idle && !(&c_rd_idle)) $fatal(1, "otpu_lane_mem: rd_idle with a copy's read in flight");
    end
  // a lane copy answers only its lane's reads: a write never returns data, and lane reads are
  // never tagged (the DMA's loader reads are copy 0's)
  always_ff @(posedge clk)
    for (int l = 1; l < C; l++)
      if (!rst && cb_rvalid[l] && cb_rtag[l])
        $fatal(1, "otpu_lane_mem: copy %0d returned a tagged B read", l);
`endif
endmodule
