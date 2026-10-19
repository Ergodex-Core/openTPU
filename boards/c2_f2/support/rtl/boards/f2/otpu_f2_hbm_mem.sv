// The F2 HBM build's memory fabric (otpu_f2_hbm_sys; docs/f2_hbm.md, docs/v4/f2-round5.md): NPORT
// channels' native ports (core clock), each through its AXI4 master (otpu_native_axi, BASE
// p x 512 MiB: Alex's burst200 adapter) onto HBM port p's AXI4 slave side, shared there with
// the host's PCIS (otpu_f2_hbm_host, one master per port) by an otpu_f2_hbm_port, everything at
// the core clock. The ports' slave sides are the HDK's cl_axi_sc_1x1 SmartConnects in the CL
// top (cl_opentpu_hbm.sv), one per port, which cross into the HBM's 450 MHz, convert 512 -> 256
// bits and AXI4 -> AXI3 into the HBM wrapper's ports: no openTPU logic runs on the HBM clock.
// tb_f2hbm tests the fabric on a behavioural stand-in for the SmartConnects and an HBM model
// (tests/test_f2_rtl.py).
//
// Toward the slice (docs/v4/f2-timing.md): the masters' write counts and error bits go to the
// adapter through a register stage here (n_wdone, n_err), a core period of wire from the
// masters along the HBM's ports to the adapter's compare by the slice. A write is counted a
// core cycle later; never earlier, so every wr_idle the slice sees is true.
module otpu_f2_hbm_mem #(
  parameter int NPORT   = 8,            // HBM ports used: slice s's channel c on port NCH s + c
  parameter int CH_LOG2 = 29,           // a port's pseudo-channel: 512 MiB
  parameter bit HAZ     = 1'b1          // the masters' direction barrier (0: tests only)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    aclk,
  input  logic                    arst,
  // ---- the channels' native ports (core clock)
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
  output logic                    pcis_wbusy,
  // ---- the ports' AXI4 masters 0 .. S NCH - 1 (core clock; 512 bits, 34-bit addresses,
  //      INCR bursts of 64-byte beats, size 6, IDs in bit 0): the SmartConnects' slave sides
  output logic [NPORT-1:0][15:0]  m_awid,
  output logic [NPORT-1:0][33:0]  m_awaddr,
  output logic [NPORT-1:0][7:0]   m_awlen,
  output logic [NPORT-1:0]        m_awvalid,
  input  logic [NPORT-1:0]        m_awready,
  output logic [NPORT-1:0][511:0] m_wdata,
  output logic [NPORT-1:0][63:0]  m_wstrb,
  output logic [NPORT-1:0]        m_wlast,
  output logic [NPORT-1:0]        m_wvalid,
  input  logic [NPORT-1:0]        m_wready,
  input  logic [NPORT-1:0][15:0]  m_bid,
  input  logic [NPORT-1:0][1:0]   m_bresp,
  input  logic [NPORT-1:0]        m_bvalid,
  output logic [NPORT-1:0]        m_bready,
  output logic [NPORT-1:0][15:0]  m_arid,
  output logic [NPORT-1:0][33:0]  m_araddr,
  output logic [NPORT-1:0][7:0]   m_arlen,
  output logic [NPORT-1:0]        m_arvalid,
  input  logic [NPORT-1:0]        m_arready,
  input  logic [NPORT-1:0][15:0]  m_rid,
  input  logic [NPORT-1:0][511:0] m_rdata,
  input  logic [NPORT-1:0][1:0]   m_rresp,
  input  logic [NPORT-1:0]        m_rlast,
  input  logic [NPORT-1:0]        m_rvalid,
  output logic [NPORT-1:0]        m_rready
);
  // the resets' tree: the top's synchronizers into a register here, replicated by synthesis
  // (max_fanout) toward the modules, each of which registers its own again (otpu_f2_hbm_port,
  // otpu_f2_hbm_host; otpu_native_axi takes this one): three flip-flops from the synchronizer
  // to any reset pin, every net a register's cycle long
  (* max_fanout = 4 *) logic rst_q;
  always_ff @(posedge clk) rst_q <= rst;
  // the masters' write counts and errors, registered toward the adapter (see the top)
  logic [NPORT-1:0][15:0] b_wdone;
  logic [NPORT-1:0]       b_err;
  always_ff @(posedge clk)
    if (rst_q) begin n_wdone <= '0; n_err <= '0; end
    else begin n_wdone <= b_wdone; n_err <= b_err; end

  // the ports' masters: [p][0] the channel's, [p][1] the host's
  logic [NPORT-1:0][1:0]          s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid;
  logic [NPORT-1:0][1:0]          s_bready, s_arvalid, s_arready, s_rvalid, s_rready;
  logic [NPORT-1:0][1:0][33:0]    s_awaddr, s_araddr;
  logic [NPORT-1:0][1:0][7:0]     s_awlen, s_arlen;
  logic [NPORT-1:0][1:0][511:0]   s_wdata;
  logic [NPORT-1:0][1:0][63:0]    s_wstrb;
  logic [NPORT-1:0][1:0]          s_bresp, s_rresp;
  logic [NPORT-1:0][511:0]        s_rdata;
  logic [NPORT-1:0]               s_rlast;
  // the host's masters (otpu_f2_hbm_host): valids per port, the rest shared
  logic [NPORT-1:0] hp_awvalid, hp_awready, hp_wvalid, hp_wready, hp_bvalid, hp_arvalid;
  logic [NPORT-1:0] hp_arready, hp_rvalid;
  logic [33:0]      hp_awaddr, hp_araddr;
  logic [511:0]     hp_wdata;
  logic [63:0]      hp_wstrb;
  logic [NPORT-1:0][511:0] hp_rdata;
  for (genvar p = 0; p < NPORT; p++) begin : g_port
    otpu_native_axi #(.BASE_ADDR(34'(p) << CH_LOG2), .CH_LOG2(CH_LOG2), .BARRIER(HAZ)) u_axi (
      .clk, .rst(rst_q),
      .n_cvalid(n_cvalid[p]), .n_cready(n_cready[p]), .n_cwe(n_cwe[p]), .n_caddr(n_caddr[p]),
      .n_wvalid(n_wvalid[p]), .n_wready(n_wready[p]), .n_wdata(n_wdata[p]),
      .n_wmask(n_wmask[p]), .n_rvalid(n_rvalid[p]), .n_rdata(n_rdata[p]),
      .n_wdone(b_wdone[p]), .n_err(b_err[p]),
      .m_awaddr(s_awaddr[p][0]), .m_awvalid(s_awvalid[p][0]), .m_awready(s_awready[p][0]),
      .m_wdata(s_wdata[p][0]), .m_wstrb(s_wstrb[p][0]), .m_wvalid(s_wvalid[p][0]),
      .m_wready(s_wready[p][0]), .m_bvalid(s_bvalid[p][0]), .m_bresp(s_bresp[p]),
      .m_bready(s_bready[p][0]), .m_araddr(s_araddr[p][0]), .m_arlen(s_arlen[p][0]),
      .m_arvalid(s_arvalid[p][0]), .m_arready(s_arready[p][0]), .m_rdata(s_rdata[p]),
      .m_rvalid(s_rvalid[p][0]), .m_rlast(s_rlast[p]), .m_rresp(s_rresp[p]),
      .m_rready(s_rready[p][0]));
    assign s_awlen[p][0] = 8'd0;          // writes single-beat
    assign s_wlast[p][0] = 1'b1;
    // the host's master on this port: single-beat accesses, ready for its responses
    assign s_awvalid[p][1] = hp_awvalid[p];
    assign hp_awready[p] = s_awready[p][1];
    assign s_awaddr[p][1] = hp_awaddr;
    assign s_awlen[p][1] = 8'd0;
    assign s_wvalid[p][1] = hp_wvalid[p];
    assign hp_wready[p] = s_wready[p][1];
    assign s_wdata[p][1] = hp_wdata;
    assign s_wstrb[p][1] = hp_wstrb;
    assign s_wlast[p][1] = 1'b1;
    assign hp_bvalid[p] = s_bvalid[p][1];
    assign s_bready[p][1] = 1'b1;
    assign s_arvalid[p][1] = hp_arvalid[p];
    assign hp_arready[p] = s_arready[p][1];
    assign s_araddr[p][1] = hp_araddr;
    assign s_arlen[p][1] = 8'd0;
    assign hp_rvalid[p] = s_rvalid[p][1];
    assign hp_rdata[p] = s_rdata[p];
    assign s_rready[p][1] = 1'b1;
    otpu_f2_hbm_port u_port (
      .clk, .rst(rst_q),
      .s_awvalid(s_awvalid[p]), .s_awready(s_awready[p]), .s_awaddr(s_awaddr[p]),
      .s_awlen(s_awlen[p]), .s_wvalid(s_wvalid[p]), .s_wready(s_wready[p]),
      .s_wdata(s_wdata[p]), .s_wstrb(s_wstrb[p]), .s_wlast(s_wlast[p]), .s_bvalid(s_bvalid[p]),
      .s_bready(s_bready[p]), .s_bresp(s_bresp[p]), .s_arvalid(s_arvalid[p]),
      .s_arready(s_arready[p]), .s_araddr(s_araddr[p]), .s_arlen(s_arlen[p]),
      .s_rvalid(s_rvalid[p]), .s_rready(s_rready[p]), .s_rdata(s_rdata[p]),
      .s_rresp(s_rresp[p]), .s_rlast(s_rlast[p]),
      .m_awid(m_awid[p]), .m_awaddr(m_awaddr[p]), .m_awlen(m_awlen[p]),
      .m_awvalid(m_awvalid[p]), .m_awready(m_awready[p]), .m_wdata(m_wdata[p]),
      .m_wstrb(m_wstrb[p]), .m_wlast(m_wlast[p]), .m_wvalid(m_wvalid[p]),
      .m_wready(m_wready[p]), .m_bid(m_bid[p]), .m_bresp(m_bresp[p]),
      .m_bvalid(m_bvalid[p]), .m_bready(m_bready[p]), .m_arid(m_arid[p]),
      .m_araddr(m_araddr[p]), .m_arlen(m_arlen[p]), .m_arvalid(m_arvalid[p]),
      .m_arready(m_arready[p]), .m_rid(m_rid[p]), .m_rdata(m_rdata[p]),
      .m_rresp(m_rresp[p]), .m_rlast(m_rlast[p]), .m_rvalid(m_rvalid[p]),
      .m_rready(m_rready[p]));
  end

  // ---- the host's PCIS onto every port
  otpu_f2_hbm_host #(.NPORT(NPORT)) u_host (
    .aclk, .arst,
    .h_awid(pcis_awid), .h_awaddr(pcis_awaddr), .h_awlen(pcis_awlen), .h_awvalid(pcis_awvalid),
    .h_awready(pcis_awready), .h_wdata(pcis_wdata), .h_wstrb(pcis_wstrb), .h_wlast(pcis_wlast),
    .h_wvalid(pcis_wvalid), .h_wready(pcis_wready), .h_bid(pcis_bid), .h_bresp(pcis_bresp),
    .h_bvalid(pcis_bvalid), .h_bready(pcis_bready),
    .h_arid(pcis_arid), .h_araddr(pcis_araddr), .h_arlen(pcis_arlen), .h_arvalid(pcis_arvalid),
    .h_arready(pcis_arready), .h_rid(pcis_rid), .h_rdata(pcis_rdata), .h_rresp(pcis_rresp),
    .h_rlast(pcis_rlast), .h_rvalid(pcis_rvalid), .h_rready(pcis_rready), .pcis_wbusy,
    .clk, .rst(rst_q),
    .p_awvalid(hp_awvalid), .p_awready(hp_awready), .p_awaddr(hp_awaddr),
    .p_wvalid(hp_wvalid), .p_wready(hp_wready), .p_wdata(hp_wdata), .p_wstrb(hp_wstrb),
    .p_bvalid(hp_bvalid), .p_arvalid(hp_arvalid), .p_arready(hp_arready),
    .p_araddr(hp_araddr), .p_rvalid(hp_rvalid), .p_rdata(hp_rdata));
endmodule
