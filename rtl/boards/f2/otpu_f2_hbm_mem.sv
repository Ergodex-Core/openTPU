// The F2 HBM build's memory fabric (otpu_f2_hbm_sys; docs/f2_hbm.md): NPORT channels' native
// ports (core clock), each through its bridge (otpu_f2_hbm_nch, BASE p x 512 MiB) onto HBM AXI
// port p, shared there with the host's PCIS (otpu_f2_hbm_host, one master per port, registered
// at the port by an otpu_f2_hbm_hreg, below) by an otpu_f2_hbm_port. tb_f2hbm tests it on an
// HBM model (tests/test_f2_rtl.py).
//
// Toward the slice (docs/v4/f2-timing.md): the bridges' write counts and error bits go to the
// adapter through a register stage here (n_wdone, n_err), so the count's flip-flops, which the
// crossing bound ties to their gray synchronizer at the HBM port, do not feed the adapter's
// compare and the slice's wr_idle chain across the die in one core cycle (61d07199 placed
// g_port[0].u_nch/n_wdone_reg -> u_slice/w_n_reg/R at 5.9 of 6.5 ns of route). A write is
// counted a core cycle later; never earlier, so every wr_idle the slice sees is true.
module otpu_f2_hbm_mem #(
  parameter int NPORT = 8,              // HBM ports used: slice s's channel c on port NCH s + c
  parameter bit HAZ   = 1'b1,           // the bridges' same-beat order (0: tests only)
  parameter int GATHER = 4              // the bridges' burst gather (hclk cycles)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    aclk,
  input  logic                    arst,
  input  logic                    hclk,
  input  logic                    hrst,
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
  // ---- the HBM's AXI3 ports 0 .. S NCH - 1 (hclk; cl_mem_hbm_wrapper, AXI4_INTERFACE 0)
  output logic [NPORT-1:0][5:0]   hbm_awid,
  output logic [NPORT-1:0][33:0]  hbm_awaddr,
  output logic [NPORT-1:0][3:0]   hbm_awlen,
  output logic [NPORT-1:0]        hbm_awvalid,
  input  logic [NPORT-1:0]        hbm_awready,
  output logic [NPORT-1:0][255:0] hbm_wdata,
  output logic [NPORT-1:0][31:0]  hbm_wstrb,
  output logic [NPORT-1:0]        hbm_wlast,
  output logic [NPORT-1:0]        hbm_wvalid,
  input  logic [NPORT-1:0]        hbm_wready,
  input  logic [NPORT-1:0][5:0]   hbm_bid,
  input  logic [NPORT-1:0][1:0]   hbm_bresp,
  input  logic [NPORT-1:0]        hbm_bvalid,
  output logic [NPORT-1:0]        hbm_bready,
  output logic [NPORT-1:0][5:0]   hbm_arid,
  output logic [NPORT-1:0][33:0]  hbm_araddr,
  output logic [NPORT-1:0][3:0]   hbm_arlen,
  output logic [NPORT-1:0]        hbm_arvalid,
  input  logic [NPORT-1:0]        hbm_arready,
  input  logic [NPORT-1:0][5:0]   hbm_rid,
  input  logic [NPORT-1:0][255:0] hbm_rdata,
  input  logic [NPORT-1:0][1:0]   hbm_rresp,
  input  logic [NPORT-1:0]        hbm_rlast,
  input  logic [NPORT-1:0]        hbm_rvalid,
  output logic [NPORT-1:0]        hbm_rready
);
  // the resets' tree: the top's synchronizers (one flip-flop each, hclk's reaching 8 bridges, 8
  // host registers, 8 arbiters and the host path along the HBM's ports at 450 MHz) into a
  // register here, replicated by synthesis (max_fanout) toward the modules, each of which
  // registers its own again (otpu_f2_hbm_nch, _port, _hreg, _host): three flip-flops from the
  // synchronizer to any reset pin, every net a register's cycle long
  (* max_fanout = 4 *) logic rst_q, hrst_q;
  always_ff @(posedge clk) rst_q <= rst;
  always_ff @(posedge hclk) hrst_q <= hrst;
  // the bridges' write counts and errors, registered toward the adapter (see the top)
  logic [NPORT-1:0][15:0] b_wdone;
  logic [NPORT-1:0]       b_err;
  always_ff @(posedge clk)
    if (rst_q) begin n_wdone <= '0; n_err <= '0; end
    else begin n_wdone <= b_wdone; n_err <= b_err; end

  // the ports' masters: [p][0] the bridge, [p][1] the host
  logic [NPORT-1:0][1:0]          s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid;
  logic [NPORT-1:0][1:0]          s_bready, s_arvalid, s_arready, s_rvalid, s_rready;
  logic [NPORT-1:0][1:0][33:0]    s_awaddr, s_araddr;
  logic [NPORT-1:0][1:0][3:0]     s_awlen, s_arlen;
  logic [NPORT-1:0][1:0][255:0]   s_wdata;
  logic [NPORT-1:0][1:0][31:0]    s_wstrb;
  logic [NPORT-1:0][1:0]          s_bresp, s_rresp;
  logic [NPORT-1:0][255:0]        s_rdata;
  logic [NPORT-1:0]               s_rlast;
  // the host's masters (otpu_f2_hbm_host): valids per port, the rest shared
  logic [NPORT-1:0] hp_awvalid, hp_awready, hp_wvalid, hp_wready, hp_bvalid, hp_arvalid;
  logic [NPORT-1:0] hp_arready, hp_rvalid;
  logic [33:0]      hp_awaddr, hp_araddr;
  logic [255:0]     hp_wdata;
  logic [31:0]      hp_wstrb;
  logic             hp_wlast;
  logic [NPORT-1:0][255:0] hp_rdata;
  for (genvar p = 0; p < NPORT; p++) begin : g_port
    otpu_f2_hbm_nch #(.BASE(34'(p) << 29), .HAZ(HAZ), .GATHER(GATHER)) u_nch (
      .clk, .rst(rst_q),
      .n_cvalid(n_cvalid[p]), .n_cready(n_cready[p]), .n_cwe(n_cwe[p]), .n_caddr(n_caddr[p]),
      .n_wvalid(n_wvalid[p]), .n_wready(n_wready[p]), .n_wdata(n_wdata[p]),
      .n_wmask(n_wmask[p]), .n_rvalid(n_rvalid[p]), .n_rdata(n_rdata[p]),
      .n_wdone(b_wdone[p]), .n_err(b_err[p]),
      .hclk, .hrst(hrst_q),
      .m_awvalid(s_awvalid[p][0]), .m_awready(s_awready[p][0]), .m_awaddr(s_awaddr[p][0]),
      .m_awlen(s_awlen[p][0]), .m_wvalid(s_wvalid[p][0]), .m_wready(s_wready[p][0]),
      .m_wdata(s_wdata[p][0]), .m_wstrb(s_wstrb[p][0]), .m_wlast(s_wlast[p][0]),
      .m_bvalid(s_bvalid[p][0]), .m_bready(s_bready[p][0]), .m_bresp(s_bresp[p]),
      .m_arvalid(s_arvalid[p][0]), .m_arready(s_arready[p][0]), .m_araddr(s_araddr[p][0]),
      .m_arlen(s_arlen[p][0]), .m_rvalid(s_rvalid[p][0]), .m_rready(s_rready[p][0]),
      .m_rdata(s_rdata[p]), .m_rresp(s_rresp[p]), .m_rlast(s_rlast[p]));
    // the host's master on this port, registered here: two-beat bursts, ready for its responses
    assign s_awlen[p][1] = 4'd1;
    assign s_arlen[p][1] = 4'd1;
    assign s_bready[p][1] = 1'b1;
    assign s_rready[p][1] = 1'b1;
    otpu_f2_hbm_hreg u_hreg (
      .hclk, .hrst(hrst_q),
      .h_awvalid(hp_awvalid[p]), .h_awready(hp_awready[p]), .h_awaddr(hp_awaddr),
      .h_wvalid(hp_wvalid[p]), .h_wready(hp_wready[p]), .h_wdata(hp_wdata), .h_wstrb(hp_wstrb),
      .h_wlast(hp_wlast), .h_bvalid(hp_bvalid[p]),
      .h_arvalid(hp_arvalid[p]), .h_arready(hp_arready[p]), .h_araddr(hp_araddr),
      .h_rvalid(hp_rvalid[p]), .h_rdata(hp_rdata[p]),
      .m_awvalid(s_awvalid[p][1]), .m_awready(s_awready[p][1]), .m_awaddr(s_awaddr[p][1]),
      .m_wvalid(s_wvalid[p][1]), .m_wready(s_wready[p][1]), .m_wdata(s_wdata[p][1]),
      .m_wstrb(s_wstrb[p][1]), .m_wlast(s_wlast[p][1]), .m_bvalid(s_bvalid[p][1]),
      .m_arvalid(s_arvalid[p][1]), .m_arready(s_arready[p][1]), .m_araddr(s_araddr[p][1]),
      .m_rvalid(s_rvalid[p][1]), .m_rdata(s_rdata[p]));
    otpu_f2_hbm_port u_port (
      .hclk, .hrst(hrst_q),
      .s_awvalid(s_awvalid[p]), .s_awready(s_awready[p]), .s_awaddr(s_awaddr[p]),
      .s_awlen(s_awlen[p]), .s_wvalid(s_wvalid[p]), .s_wready(s_wready[p]),
      .s_wdata(s_wdata[p]), .s_wstrb(s_wstrb[p]), .s_wlast(s_wlast[p]), .s_bvalid(s_bvalid[p]),
      .s_bready(s_bready[p]), .s_bresp(s_bresp[p]), .s_arvalid(s_arvalid[p]),
      .s_arready(s_arready[p]), .s_araddr(s_araddr[p]), .s_arlen(s_arlen[p]),
      .s_rvalid(s_rvalid[p]), .s_rready(s_rready[p]), .s_rdata(s_rdata[p]),
      .s_rresp(s_rresp[p]), .s_rlast(s_rlast[p]),
      .m_awid(hbm_awid[p]), .m_awaddr(hbm_awaddr[p]), .m_awlen(hbm_awlen[p]),
      .m_awvalid(hbm_awvalid[p]), .m_awready(hbm_awready[p]), .m_wdata(hbm_wdata[p]),
      .m_wstrb(hbm_wstrb[p]), .m_wlast(hbm_wlast[p]), .m_wvalid(hbm_wvalid[p]),
      .m_wready(hbm_wready[p]), .m_bid(hbm_bid[p]), .m_bresp(hbm_bresp[p]),
      .m_bvalid(hbm_bvalid[p]), .m_bready(hbm_bready[p]), .m_arid(hbm_arid[p]),
      .m_araddr(hbm_araddr[p]), .m_arlen(hbm_arlen[p]), .m_arvalid(hbm_arvalid[p]),
      .m_arready(hbm_arready[p]), .m_rid(hbm_rid[p]), .m_rdata(hbm_rdata[p]),
      .m_rresp(hbm_rresp[p]), .m_rlast(hbm_rlast[p]), .m_rvalid(hbm_rvalid[p]),
      .m_rready(hbm_rready[p]));
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
    .hclk, .hrst(hrst_q),
    .p_awvalid(hp_awvalid), .p_awready(hp_awready), .p_awaddr(hp_awaddr),
    .p_wvalid(hp_wvalid), .p_wready(hp_wready), .p_wdata(hp_wdata), .p_wstrb(hp_wstrb),
    .p_wlast(hp_wlast), .p_bvalid(hp_bvalid), .p_arvalid(hp_arvalid), .p_arready(hp_arready),
    .p_araddr(hp_araddr), .p_rvalid(hp_rvalid), .p_rdata(hp_rdata));

endmodule

// The host's master on one port, registered at the port (otpu_f2_hbm_mem g_port u_hreg). The
// host path (otpu_f2_hbm_host) shows its head beat's AW, W and AR to every port on shared buses
// (34 + 290 + 34 bits, a valid per port) and takes the ports' R data (256 bits each) and B back:
// with 8 ports along the HBM, the 61d07199 placement ran the head's data and strobe registers
// and the replicas of its W-half select into every arbiter's W register over 6.0 ns of route
// (docs/v4/f2-timing.md). Here each channel is a one-entry register: AW, W and AR taken from
// the host when empty (the ready is the register's own emptiness, a flip-flop: no path from the
// arbiter's or the HBM's readies back to the host in the cycle; at most a beat every other
// cycle, and the host sends one per ~7), shown to the arbiter until taken; the port's R and B
// registered the same way back to the host (a cycle late; R needs no ready: the host's return
// FIFO has room for every read in flight, its count moves on the registered beat). So every
// net between the host block and a port starts and ends at a flip-flop, with the cycle for
// the route. The reset through a register of its own, as the port's.
module otpu_f2_hbm_hreg (
  input  logic          hclk,
  input  logic          hrst,
  // ---- the host (otpu_f2_hbm_host p_*: this port's valids, the shared buses)
  input  logic          h_awvalid,
  output logic          h_awready,
  input  logic [33:0]   h_awaddr,
  input  logic          h_wvalid,
  output logic          h_wready,
  input  logic [255:0]  h_wdata,
  input  logic [31:0]   h_wstrb,
  input  logic          h_wlast,
  output logic          h_bvalid,
  input  logic          h_arvalid,
  output logic          h_arready,
  input  logic [33:0]   h_araddr,
  output logic          h_rvalid,
  output logic [255:0]  h_rdata,
  // ---- the port arbiter (otpu_f2_hbm_port master 1)
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [33:0]   m_awaddr,
  output logic          m_wvalid,
  input  logic          m_wready,
  output logic [255:0]  m_wdata,
  output logic [31:0]   m_wstrb,
  output logic          m_wlast,
  input  logic          m_bvalid,
  output logic          m_arvalid,
  input  logic          m_arready,
  output logic [33:0]   m_araddr,
  input  logic          m_rvalid,
  input  logic [255:0]  m_rdata
);
  logic hrst_q;
  always_ff @(posedge hclk) hrst_q <= hrst;
  // the registers load whenever empty (their enables are their own valids, not the host's
  // decode), so each holds the host's beat of the cycle its valid rose
  (* max_fanout = 64 *) logic aw_v, w_v, ar_v;
  assign h_awready = !aw_v;
  assign h_wready  = !w_v;
  assign h_arready = !ar_v;
  assign m_awvalid = aw_v;
  assign m_wvalid  = w_v;
  assign m_arvalid = ar_v;
  always_ff @(posedge hclk) begin
    if (!aw_v) m_awaddr <= h_awaddr;
    if (!w_v) begin m_wdata <= h_wdata; m_wstrb <= h_wstrb; m_wlast <= h_wlast; end
    if (!ar_v) m_araddr <= h_araddr;
    h_rdata <= m_rdata;
    if (hrst_q) begin
      aw_v <= 1'b0; w_v <= 1'b0; ar_v <= 1'b0; h_rvalid <= 1'b0; h_bvalid <= 1'b0;
    end else begin
      aw_v <= aw_v ? !m_awready : h_awvalid;
      w_v  <= w_v ? !m_wready : h_wvalid;
      ar_v <= ar_v ? !m_arready : h_arvalid;
      h_rvalid <= m_rvalid;
      h_bvalid <= m_bvalid;
    end
  end
endmodule
