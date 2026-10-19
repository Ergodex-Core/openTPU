// The F2 HBM build's memory side with a 450 MHz engine per port (v5, docs/superpowers/specs/
// 2026-10-20-opentpu-design-v5.md section 3; replaces otpu_f2_hbm_axi's otpu_native_axi +
// SmartConnect per channel for the engine targets; otpu_f2_hbm_axi stays for the others):
// NPORT = S x NCH channels' native ports, each across the NXR-stage SLR chain (otpu_xr, one
// stage per SLR) into its otpu_hbm_ch (SLR0, by the port), whose AXI3 256-bit master at hclk
// is the CL's port p (through the HDK's 450 MHz register slice); the host's PCIS through
// otpu_f2_pcis_gate to the fabric's host branch (h_*, aclk: AWS's cl_axi_sc_1x1 as shipped,
// onto its own port NPORT) or, HOST_MUX, that chain's AXI3 master (hs_*, hclk) through
// otpu_hbm_mux2 onto port NPORT - 1 with the channel's engine (S x NCH = 32: no port left).
//
// The chains (the SLL budget, spec section 6): down, per channel, {take, we, beat} (27 bits);
// up {cready, wready, rvalid, rdata, wdone-pulse, err} (517); and per slice one write bus
// {take, channel, data, mask} (577 + log2 NCH): the slice-side arbiter takes one channel's
// write beat a cycle (round robin among the channels whose engine shows room; the adapter's
// n_wready[c] is the grant), SLR0 demuxes it into the channel's engine. Writes are rare in
// decode (the KV append's rows, QST's gathered beats), so one beat a cycle a slice costs
// nothing measurable and saves 8 x 576 bits a slice on every boundary crossed. The engine's
// FIFOs keep 2 NXR entries below their readies (the adapter sees a ready NXR cycles late and
// its take lands NXR later), the write count comes back as a pulse and is counted here. No
// reset reaches a chain stage: whatever is in one flushes in NXR cycles while both sides hold
// reset.
module otpu_f2_hbm_eng #(
  parameter int NPORT      = 8,         // S x NCH: slice s's channel c on port NCH s + c
  parameter int S          = 1,
  parameter int NCH        = 8,
  parameter int NXR        = 3,         // SLR crossing stages each way (0: wired)
  parameter int CH_LOG2    = 29,        // a port's window: a pseudo-channel's 512 MiB
  parameter int BURST      = 8,         // otpu_hbm_ch's knobs
  parameter int IDLE       = 2,
  parameter int OT         = 64,
  parameter int RDEPTH     = 256,
  parameter bit HOST_MUX   = 1'b0,      // the host's AXI3 (hs_*) muxed onto port NPORT - 1
  parameter bit ORDER      = 1'b1       // the direction barriers (0: tests only)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    hclk,
  input  logic                    hrst,
  input  logic                    aclk,
  input  logic                    arst,
  // ---- the channels' native ports (core clock, the slices' side)
  input  logic [NPORT-1:0]        n_cvalid,
  output logic [NPORT-1:0]        n_cready,
  input  logic [NPORT-1:0]        n_cwe,
  input  logic [NPORT-1:0][24:0]  n_caddr,
  input  logic [NPORT-1:0]        n_wvalid,
  output logic [NPORT-1:0]        n_wready,
  input  logic [NPORT-1:0][511:0] n_wdata,
  input  logic [NPORT-1:0][63:0]  n_wmask,
  output logic [NPORT-1:0]        n_rvalid,
  output logic [NPORT-1:0][511:0] n_rdata,
  output logic [NPORT-1:0][15:0]  n_wdone,
  output logic [NPORT-1:0]        n_err,
  // ---- the shell's PCIS (AppPF BAR4), aclk
  input  logic [15:0]   pcis_awid,
  input  logic [63:0]   pcis_awaddr,
  input  logic [7:0]    pcis_awlen,
  input  logic [2:0]    pcis_awsize,
  input  logic [1:0]    pcis_awburst,
  input  logic          pcis_awvalid,
  output logic          pcis_awready,
  input  logic [511:0]  pcis_wdata,
  input  logic [63:0]   pcis_wstrb,
  input  logic          pcis_wlast,
  input  logic          pcis_wvalid,
  output logic          pcis_wready,
  output logic [15:0]   pcis_bid,
  output logic [1:0]    pcis_bresp,
  output logic          pcis_bvalid,
  input  logic          pcis_bready,
  input  logic [15:0]   pcis_arid,
  input  logic [63:0]   pcis_araddr,
  input  logic [7:0]    pcis_arlen,
  input  logic [2:0]    pcis_arsize,
  input  logic [1:0]    pcis_arburst,
  input  logic          pcis_arvalid,
  output logic          pcis_arready,
  output logic [15:0]   pcis_rid,
  output logic [511:0]  pcis_rdata,
  output logic [1:0]    pcis_rresp,
  output logic          pcis_rlast,
  output logic          pcis_rvalid,
  input  logic          pcis_rready,
  output logic          pcis_wbusy,
  // ---- the host branch towards cl_axi_sc_1x1 (aclk; AXI4 512 bits)
  output logic [15:0]   h_awid,
  output logic [63:0]   h_awaddr,
  output logic [7:0]    h_awlen,
  output logic [2:0]    h_awsize,
  output logic [1:0]    h_awburst,
  output logic          h_awvalid,
  input  logic          h_awready,
  output logic [511:0]  h_wdata,
  output logic [63:0]   h_wstrb,
  output logic          h_wlast,
  output logic          h_wvalid,
  input  logic          h_wready,
  input  logic [15:0]   h_bid,
  input  logic [1:0]    h_bresp,
  input  logic          h_bvalid,
  output logic          h_bready,
  output logic [15:0]   h_arid,
  output logic [63:0]   h_araddr,
  output logic [7:0]    h_arlen,
  output logic [2:0]    h_arsize,
  output logic [1:0]    h_arburst,
  output logic          h_arvalid,
  input  logic          h_arready,
  input  logic [15:0]   h_rid,
  input  logic [511:0]  h_rdata,
  input  logic [1:0]    h_rresp,
  input  logic          h_rlast,
  input  logic          h_rvalid,
  output logic          h_rready,
  // ---- HOST_MUX: the host chain's AXI3 master (hclk), onto port NPORT - 1
  input  logic [5:0]    hs_awid,
  input  logic [33:0]   hs_awaddr,
  input  logic [3:0]    hs_awlen,
  input  logic [2:0]    hs_awsize,
  input  logic [1:0]    hs_awburst,
  input  logic          hs_awvalid,
  output logic          hs_awready,
  input  logic [5:0]    hs_wid,
  input  logic [255:0]  hs_wdata,
  input  logic [31:0]   hs_wstrb,
  input  logic          hs_wlast,
  input  logic          hs_wvalid,
  output logic          hs_wready,
  output logic [5:0]    hs_bid,
  output logic [1:0]    hs_bresp,
  output logic          hs_bvalid,
  input  logic          hs_bready,
  input  logic [5:0]    hs_arid,
  input  logic [33:0]   hs_araddr,
  input  logic [3:0]    hs_arlen,
  input  logic [2:0]    hs_arsize,
  input  logic [1:0]    hs_arburst,
  input  logic          hs_arvalid,
  output logic          hs_arready,
  output logic [5:0]    hs_rid,
  output logic [255:0]  hs_rdata,
  output logic [1:0]    hs_rresp,
  output logic          hs_rlast,
  output logic          hs_rvalid,
  input  logic          hs_rready,
  // ---- the ports (hclk): AXI3 256-bit masters, one per HBM port p < NPORT
  output logic [NPORT-1:0][5:0]    p_awid,
  output logic [NPORT-1:0][33:0]   p_awaddr,
  output logic [NPORT-1:0][3:0]    p_awlen,
  output logic [NPORT-1:0][2:0]    p_awsize,
  output logic [NPORT-1:0][1:0]    p_awburst,
  output logic [NPORT-1:0]         p_awvalid,
  input  logic [NPORT-1:0]         p_awready,
  output logic [NPORT-1:0][5:0]    p_wid,
  output logic [NPORT-1:0][255:0]  p_wdata,
  output logic [NPORT-1:0][31:0]   p_wstrb,
  output logic [NPORT-1:0]         p_wlast,
  output logic [NPORT-1:0]         p_wvalid,
  input  logic [NPORT-1:0]         p_wready,
  input  logic [NPORT-1:0][5:0]    p_bid,
  input  logic [NPORT-1:0][1:0]    p_bresp,
  input  logic [NPORT-1:0]         p_bvalid,
  output logic [NPORT-1:0]         p_bready,
  output logic [NPORT-1:0][5:0]    p_arid,
  output logic [NPORT-1:0][33:0]   p_araddr,
  output logic [NPORT-1:0][3:0]    p_arlen,
  output logic [NPORT-1:0][2:0]    p_arsize,
  output logic [NPORT-1:0][1:0]    p_arburst,
  output logic [NPORT-1:0]         p_arvalid,
  input  logic [NPORT-1:0]         p_arready,
  input  logic [NPORT-1:0][5:0]    p_rid,
  input  logic [NPORT-1:0][255:0]  p_rdata,
  input  logic [NPORT-1:0][1:0]    p_rresp,
  input  logic [NPORT-1:0]         p_rlast,
  input  logic [NPORT-1:0]         p_rvalid,
  output logic [NPORT-1:0]         p_rready
);
  localparam int CW = (NCH > 1) ? $clog2(NCH) : 1;
  localparam int DW = 1 + 1 + 25;                   // down: take, we, beat
  localparam int UW = 1 + 1 + 1 + 512 + 1 + 1;      // up: cready, wready, rvalid, data, wdp, err
  localparam int BW = 1 + CW + 512 + 64;            // the write bus: take, channel, data, mask
  initial if (NPORT != S * NCH || NPORT < 1 || NPORT > 32 || NPORT * (64'd1 << CH_LOG2) > (64'd1 << 34))
    $fatal(1, "otpu_f2_hbm_eng: NPORT = S x NCH, at most 32 channels within the HBM's 16 GiB");
  // the resets into registers here, replicated toward the engines
  (* max_fanout = 4 *) logic rst_q, hrst_q;
  always_ff @(posedge clk) rst_q <= rst;
  always_ff @(posedge hclk) hrst_q <= hrst;

  // ---- the engines' native sides (SLR0), e_*
  logic [NPORT-1:0]        e_cvalid, e_cready, e_cwe, e_wvalid, e_wready, e_rvalid, e_wdp, e_err;
  logic [NPORT-1:0][24:0]  e_caddr;
  logic [NPORT-1:0][511:0] e_rdata;
  logic [511:0]            e_wdata [S];
  logic [63:0]             e_wmask [S];
  // ---- the slices' side after the chains, x_*
  logic [NPORT-1:0]        x_cready, x_wready, x_rvalid, x_wdp, x_err;
  logic [NPORT-1:0][511:0] x_rdata;
  // the per-channel chains: the adapters' words (dn_a, up_x) and the engines' (dn_e, up_e)
  logic [NPORT-1:0][DW-1:0] dn_a, dn_e;
  logic [NPORT-1:0][UW-1:0] up_e, up_x;
  for (genvar p = 0; p < NPORT; p++) begin : g_ch
    assign dn_a[p] = {n_cvalid[p] && n_cready[p], n_cwe[p], n_caddr[p]};
    assign {e_cvalid[p], e_cwe[p], e_caddr[p]} = dn_e[p];
    assign up_e[p] = {e_cready[p], e_wready[p], e_rvalid[p], e_rdata[p], e_wdp[p], e_err[p]};
    assign {x_cready[p], x_wready[p], x_rvalid[p], x_rdata[p], x_wdp[p], x_err[p]} = up_x[p];
    assign n_cready[p] = x_cready[p];
    assign n_rvalid[p] = x_rvalid[p];
    assign n_rdata[p] = x_rdata[p];
    assign n_err[p] = x_err[p];
    // the write count, from the engine's pulses
    always_ff @(posedge clk)
      if (rst_q) n_wdone[p] <= '0;
      else if (x_wdp[p]) n_wdone[p] <= n_wdone[p] + 1'b1;
  end
  if (NXR == 0) begin : g_wire
    assign dn_e = dn_a;
    assign up_x = up_e;
  end else begin : g_xr
    logic [NXR:0][NPORT-1:0][DW-1:0] dn;     // dn[NXR] the adapters', dn[0] the engines'
    logic [NXR:0][NPORT-1:0][UW-1:0] up;     // up[0] the engines', up[NXR] the adapters'
    assign dn[NXR] = dn_a;
    assign dn_e = dn[0];
    assign up[0] = up_e;
    assign up_x = up[NXR];
    for (genvar k = 0; k < NXR; k++) begin : g_x    // k: the stage's SLR (0 the engines')
      otpu_xr #(.W(NPORT * DW)) u_dn (.clk, .rst(1'b0), .d(dn[k + 1]), .q(dn[k]));
      otpu_xr #(.W(NPORT * UW)) u_up (.clk, .rst(1'b0), .d(up[k]), .q(up[k + 1]));
    end
  end

  // ---- the write bus per slice: the arbiter on the slices' side, the chain, the demux
  logic [S-1:0][BW-1:0] wb_a, wb_e;          // the arbiters' words, the demuxes'
  for (genvar s = 0; s < S; s++) begin : g_wb
    logic [NCH-1:0] req, gnt;
    logic [CW-1:0]  ptr, gidx;
    logic [511:0]   gdata;
    logic [63:0]    gmask;
    always_comb begin
      logic found;
      found = 1'b0; gnt = '0; gidx = '0; gdata = '0; gmask = '0;
      for (int c = 0; c < NCH; c++) req[c] = n_wvalid[NCH * s + c] && x_wready[NCH * s + c];
      // round robin from ptr: the first requester at or after it
      for (int i = 0; i < NCH; i++) begin
        int c;
        c = (int'(ptr) + i) % NCH;
        if (!found && req[c]) begin
          found = 1'b1; gnt[c] = 1'b1; gidx = CW'(c);
        end
      end
      for (int c = 0; c < NCH; c++)
        if (gnt[c]) begin gdata = n_wdata[NCH * s + c]; gmask = n_wmask[NCH * s + c]; end
    end
    assign n_wready[NCH * s +: NCH] = gnt;
    always_ff @(posedge clk)
      if (rst_q) ptr <= '0;
      else if (|gnt) ptr <= gidx + 1'b1;
    assign wb_a[s] = {|gnt, gidx, gdata, gmask};
    // SLR0: the beat into its channel's engine
    logic          b_take;
    logic [CW-1:0] b_ch;
    assign {b_take, b_ch, e_wdata[s], e_wmask[s]} = wb_e[s];
    for (genvar c = 0; c < NCH; c++) begin : g_dx
      assign e_wvalid[NCH * s + c] = b_take && (b_ch == CW'(c));
    end
  end
  if (NXR == 0) begin : g_wirew
    assign wb_e = wb_a;
  end else begin : g_xw
    logic [NXR:0][S-1:0][BW-1:0] wb;         // wb[NXR] the arbiters', wb[0] the demuxes'
    assign wb[NXR] = wb_a;
    assign wb_e = wb[0];
    for (genvar k = 0; k < NXR; k++) begin : g_x
      otpu_xr #(.W(S * BW)) u_wb (.clk, .rst(1'b0), .d(wb[k + 1]), .q(wb[k]));
    end
  end

  // ---- the engines, and their AXI3 masters (port NPORT - 1 through the host mux if HOST_MUX)
  logic [5:0]    c_awid, c_wid, c_arid;
  logic [33:0]   c_awaddr, c_araddr;
  logic [3:0]    c_awlen, c_arlen;
  logic [2:0]    c_awsize, c_arsize;
  logic [1:0]    c_awburst, c_arburst;
  logic          c_awvalid, c_awready, c_wvalid, c_wready, c_wlast, c_bvalid, c_bready;
  logic          c_arvalid, c_arready, c_rvalid, c_rready, c_rlast;
  logic [255:0]  c_wdata, c_rdata;
  logic [31:0]   c_wstrb;
  logic [5:0]    c_bid, c_rid;
  logic [1:0]    c_bresp, c_rresp;
  for (genvar p = 0; p < NPORT; p++) begin : g_port
    localparam bit MX = HOST_MUX && (p == NPORT - 1);
    localparam int s = p / NCH;
    logic [5:0]    m_awid, m_wid, m_arid, m_bid, m_rid;
    logic [33:0]   m_awaddr, m_araddr;
    logic [3:0]    m_awlen, m_arlen;
    logic [2:0]    m_awsize, m_arsize;
    logic [1:0]    m_awburst, m_arburst, m_bresp, m_rresp;
    logic          m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
    logic          m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
    logic [255:0]  m_wdata, m_rdata;
    logic [31:0]   m_wstrb;
    otpu_hbm_ch #(.BASE_ADDR(34'(p) << CH_LOG2), .CH_LOG2(CH_LOG2), .XL(2 * NXR), .BURST(BURST),
                  .IDLE(IDLE), .OT(OT), .RDEPTH(RDEPTH), .ID(6'd0), .ORDER(ORDER)) u_eng (
      .clk, .rst(rst_q),
      .n_cvalid(e_cvalid[p]), .n_cready(e_cready[p]), .n_cwe(e_cwe[p]), .n_caddr(e_caddr[p]),
      .n_wvalid(e_wvalid[p]), .n_wready(e_wready[p]), .n_wdata(e_wdata[s]), .n_wmask(e_wmask[s]),
      .n_rvalid(e_rvalid[p]), .n_rdata(e_rdata[p]), .n_wdone(), .n_wdp(e_wdp[p]), .n_err(e_err[p]),
      .hclk, .hrst(hrst_q),
      .m_awid, .m_awaddr, .m_awlen, .m_awsize, .m_awburst, .m_awvalid, .m_awready,
      .m_wid, .m_wdata, .m_wstrb, .m_wlast, .m_wvalid, .m_wready,
      .m_bid, .m_bresp, .m_bvalid, .m_bready,
      .m_arid, .m_araddr, .m_arlen, .m_arsize, .m_arburst, .m_arvalid, .m_arready,
      .m_rid, .m_rdata, .m_rresp, .m_rlast, .m_rvalid, .m_rready);
    if (MX) begin : g_mx
      assign c_awid = m_awid; assign c_awaddr = m_awaddr; assign c_awlen = m_awlen;
      assign c_awsize = m_awsize; assign c_awburst = m_awburst; assign c_awvalid = m_awvalid;
      assign m_awready = c_awready;
      assign c_wid = m_wid; assign c_wdata = m_wdata; assign c_wstrb = m_wstrb;
      assign c_wlast = m_wlast; assign c_wvalid = m_wvalid; assign m_wready = c_wready;
      assign m_bid = c_bid; assign m_bresp = c_bresp; assign m_bvalid = c_bvalid;
      assign c_bready = m_bready;
      assign c_arid = m_arid; assign c_araddr = m_araddr; assign c_arlen = m_arlen;
      assign c_arsize = m_arsize; assign c_arburst = m_arburst; assign c_arvalid = m_arvalid;
      assign m_arready = c_arready;
      assign m_rid = c_rid; assign m_rdata = c_rdata; assign m_rresp = c_rresp;
      assign m_rlast = c_rlast; assign m_rvalid = c_rvalid; assign c_rready = m_rready;
    end else begin : g_direct
      assign p_awid[p] = m_awid; assign p_awaddr[p] = m_awaddr; assign p_awlen[p] = m_awlen;
      assign p_awsize[p] = m_awsize; assign p_awburst[p] = m_awburst; assign p_awvalid[p] = m_awvalid;
      assign m_awready = p_awready[p];
      assign p_wid[p] = m_wid; assign p_wdata[p] = m_wdata; assign p_wstrb[p] = m_wstrb;
      assign p_wlast[p] = m_wlast; assign p_wvalid[p] = m_wvalid; assign m_wready = p_wready[p];
      assign m_bid = p_bid[p]; assign m_bresp = p_bresp[p]; assign m_bvalid = p_bvalid[p];
      assign p_bready[p] = m_bready;
      assign p_arid[p] = m_arid; assign p_araddr[p] = m_araddr; assign p_arlen[p] = m_arlen;
      assign p_arsize[p] = m_arsize; assign p_arburst[p] = m_arburst; assign p_arvalid[p] = m_arvalid;
      assign m_arready = p_arready[p];
      assign m_rid = p_rid[p]; assign m_rdata = p_rdata[p]; assign m_rresp = p_rresp[p];
      assign m_rlast = p_rlast[p]; assign m_rvalid = p_rvalid[p]; assign p_rready[p] = m_rready;
    end
  end
  if (HOST_MUX) begin : g_mux
    localparam int P = NPORT - 1;
    otpu_hbm_mux2 u_mux (
      .hclk, .hrst(hrst_q),
      .s0_awid(c_awid), .s0_awaddr(c_awaddr), .s0_awlen(c_awlen), .s0_awsize(c_awsize),
      .s0_awburst(c_awburst), .s0_awvalid(c_awvalid), .s0_awready(c_awready),
      .s0_wid(c_wid), .s0_wdata(c_wdata), .s0_wstrb(c_wstrb), .s0_wlast(c_wlast),
      .s0_wvalid(c_wvalid), .s0_wready(c_wready),
      .s0_bid(c_bid), .s0_bresp(c_bresp), .s0_bvalid(c_bvalid), .s0_bready(c_bready),
      .s0_arid(c_arid), .s0_araddr(c_araddr), .s0_arlen(c_arlen), .s0_arsize(c_arsize),
      .s0_arburst(c_arburst), .s0_arvalid(c_arvalid), .s0_arready(c_arready),
      .s0_rid(c_rid), .s0_rdata(c_rdata), .s0_rresp(c_rresp), .s0_rlast(c_rlast),
      .s0_rvalid(c_rvalid), .s0_rready(c_rready),
      .s1_awid(hs_awid), .s1_awaddr(hs_awaddr), .s1_awlen(hs_awlen), .s1_awsize(hs_awsize),
      .s1_awburst(hs_awburst), .s1_awvalid(hs_awvalid), .s1_awready(hs_awready),
      .s1_wid(hs_wid), .s1_wdata(hs_wdata), .s1_wstrb(hs_wstrb), .s1_wlast(hs_wlast),
      .s1_wvalid(hs_wvalid), .s1_wready(hs_wready),
      .s1_bid(hs_bid), .s1_bresp(hs_bresp), .s1_bvalid(hs_bvalid), .s1_bready(hs_bready),
      .s1_arid(hs_arid), .s1_araddr(hs_araddr), .s1_arlen(hs_arlen), .s1_arsize(hs_arsize),
      .s1_arburst(hs_arburst), .s1_arvalid(hs_arvalid), .s1_arready(hs_arready),
      .s1_rid(hs_rid), .s1_rdata(hs_rdata), .s1_rresp(hs_rresp), .s1_rlast(hs_rlast),
      .s1_rvalid(hs_rvalid), .s1_rready(hs_rready),
      .m_awid(p_awid[P]), .m_awaddr(p_awaddr[P]), .m_awlen(p_awlen[P]), .m_awsize(p_awsize[P]),
      .m_awburst(p_awburst[P]), .m_awvalid(p_awvalid[P]), .m_awready(p_awready[P]),
      .m_wid(p_wid[P]), .m_wdata(p_wdata[P]), .m_wstrb(p_wstrb[P]), .m_wlast(p_wlast[P]),
      .m_wvalid(p_wvalid[P]), .m_wready(p_wready[P]),
      .m_bid(p_bid[P]), .m_bresp(p_bresp[P]), .m_bvalid(p_bvalid[P]), .m_bready(p_bready[P]),
      .m_arid(p_arid[P]), .m_araddr(p_araddr[P]), .m_arlen(p_arlen[P]), .m_arsize(p_arsize[P]),
      .m_arburst(p_arburst[P]), .m_arvalid(p_arvalid[P]), .m_arready(p_arready[P]),
      .m_rid(p_rid[P]), .m_rdata(p_rdata[P]), .m_rresp(p_rresp[P]), .m_rlast(p_rlast[P]),
      .m_rvalid(p_rvalid[P]), .m_rready(p_rready[P]));
  end else begin : g_nomux
    assign hs_awready = 1'b0; assign hs_wready = 1'b0; assign hs_arready = 1'b0;
    assign hs_bid = '0; assign hs_bresp = '0; assign hs_bvalid = 1'b0;
    assign hs_rid = '0; assign hs_rdata = '0; assign hs_rresp = '0; assign hs_rlast = 1'b0;
    assign hs_rvalid = 1'b0;
    assign c_awready = 1'b0; assign c_wready = 1'b0; assign c_arready = 1'b0;
    assign c_bid = '0; assign c_bresp = '0; assign c_bvalid = 1'b0;
    assign c_rid = '0; assign c_rdata = '0; assign c_rresp = '0; assign c_rlast = 1'b0;
    assign c_rvalid = 1'b0;
  end

  // ---- the host's PCIS onto the fabric's host branch (as otpu_f2_hbm_axi)
  otpu_f2_pcis_gate #(.HOLD(ORDER)) u_pcis (
    .aclk, .arst,
    .s_awid(pcis_awid), .s_awaddr(pcis_awaddr), .s_awlen(pcis_awlen), .s_awsize(pcis_awsize),
    .s_awburst(pcis_awburst), .s_awvalid(pcis_awvalid), .s_awready(pcis_awready),
    .s_wdata(pcis_wdata), .s_wstrb(pcis_wstrb), .s_wlast(pcis_wlast), .s_wvalid(pcis_wvalid),
    .s_wready(pcis_wready), .s_bid(pcis_bid), .s_bresp(pcis_bresp), .s_bvalid(pcis_bvalid),
    .s_bready(pcis_bready), .s_arid(pcis_arid), .s_araddr(pcis_araddr), .s_arlen(pcis_arlen),
    .s_arsize(pcis_arsize), .s_arburst(pcis_arburst), .s_arvalid(pcis_arvalid),
    .s_arready(pcis_arready), .s_rid(pcis_rid), .s_rdata(pcis_rdata), .s_rresp(pcis_rresp),
    .s_rlast(pcis_rlast), .s_rvalid(pcis_rvalid), .s_rready(pcis_rready),
    .m_awid(h_awid), .m_awaddr(h_awaddr), .m_awlen(h_awlen), .m_awsize(h_awsize),
    .m_awburst(h_awburst), .m_awvalid(h_awvalid), .m_awready(h_awready),
    .m_wdata(h_wdata), .m_wstrb(h_wstrb), .m_wlast(h_wlast), .m_wvalid(h_wvalid),
    .m_wready(h_wready), .m_bid(h_bid), .m_bresp(h_bresp), .m_bvalid(h_bvalid),
    .m_bready(h_bready), .m_arid(h_arid), .m_araddr(h_araddr), .m_arlen(h_arlen),
    .m_arsize(h_arsize), .m_arburst(h_arburst), .m_arvalid(h_arvalid), .m_arready(h_arready),
    .m_rid(h_rid), .m_rdata(h_rdata), .m_rresp(h_rresp), .m_rlast(h_rlast),
    .m_rvalid(h_rvalid), .m_rready(h_rready), .pcis_wbusy);
endmodule
