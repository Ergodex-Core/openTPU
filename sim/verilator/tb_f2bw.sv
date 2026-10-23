// Read throughput of the F2 slot's memory path (rtl/boards/f2: otpu_f2_nch x 2 in front of
// otpu_f2_mem) against its reads in flight: both native channels stream reads (one command a core
// cycle each when the bridge takes it, consecutive beats), and a DIMM model (aclk) answers every AR
// LAT aclk cycles after it is taken, one R beat an aclk cycle at most (the 512-bit port), at most
// BW percent of the beats a cycle (the controller's efficiency). Prints the read beats the core got
// a core cycle over the run's second half (both channels: x 64 bytes = bytes a core cycle) and the
// reads in flight per channel on average: with a long LAT the rate is ROD x 2 beats per round trip
// (Little's law), with a short one the DIMM's BW or the core's 2 beats a cycle.
// Parameters: ROD, RD (the bridge's), CORE_HALF / AXI_HALF (clock half periods). Plusargs: +lat=
// (aclk cycles, default 100), +bw= (percent, default 100), +cycles= (core cycles, default 20000).
// No checking beyond the bridges' own assertions: tb_f2mem checks the data (tests/test_f2_rtl.py).
module tb_f2bw;
  parameter int ROD = 32, RD = 64;
  parameter int CORE_HALF = 5, AXI_HALF = 2;
  localparam logic [63:0] CHB = 64'h8000_0000;

  logic clk = 1'b0, aclk = 1'b0;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;
  logic rst = 1'b1, arst = 1'b1;
  int lat = 100, bw = 100, cycles = 20000;
  initial begin
    void'($value$plusargs("lat=%d", lat));
    void'($value$plusargs("bw=%d", bw));
    void'($value$plusargs("cycles=%d", cycles));
  end

  logic [1:0]        n_cvalid, n_cready, n_wready, n_rvalid, n_err;
  logic [1:0][24:0]  n_caddr;
  logic [1:0][511:0] n_rdata;
  logic [1:0][15:0]  n_wdone;
  logic [1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
  logic [1:0]        m_arvalid, m_arready, m_rvalid, m_rready;
  logic [1:0][63:0]  m_awaddr, m_araddr, m_wstrb;
  logic [1:0][511:0] m_wdata;
  logic [511:0]      m_rdata;
  logic [1:0]        m_bresp, m_rresp;
  for (genvar c = 0; c < 2; c++) begin : g_ch
    otpu_f2_nch #(.BASE(CHB * c), .RD(RD), .ROD(ROD)) u_nch (
      .clk, .rst,
      .n_cvalid(n_cvalid[c]), .n_cready(n_cready[c]), .n_cwe(1'b0), .n_caddr(n_caddr[c]),
      .n_wvalid(1'b0), .n_wready(n_wready[c]), .n_wdata('0),
      .n_wmask('0), .n_rvalid(n_rvalid[c]), .n_rdata(n_rdata[c]),
      .n_wdone(n_wdone[c]), .n_err(n_err[c]),
      .aclk, .arst,
      .m_awvalid(m_awvalid[c]), .m_awready(m_awready[c]), .m_awaddr(m_awaddr[c]),
      .m_wvalid(m_wvalid[c]), .m_wready(m_wready[c]), .m_wdata(m_wdata[c]),
      .m_wstrb(m_wstrb[c]), .m_bvalid(m_bvalid[c]), .m_bready(m_bready[c]), .m_bresp(m_bresp),
      .m_arvalid(m_arvalid[c]), .m_arready(m_arready[c]), .m_araddr(m_araddr[c]),
      .m_rvalid(m_rvalid[c]), .m_rready(m_rready[c]), .m_rdata(m_rdata), .m_rresp(m_rresp));
  end
  logic [15:0]  h_bid, h_rid;
  logic         h_awready, h_wready, h_bvalid, h_arready, h_rvalid, h_rlast, pcis_wbusy;
  logic [511:0] h_rdata;
  logic [1:0]   h_bresp, h_rresp;
  logic [15:0]  d_awid, d_arid, d_rid;
  logic [63:0]  d_awaddr, d_araddr, d_wstrb;
  logic [7:0]   d_awlen, d_arlen;
  logic [2:0]   d_awsize, d_arsize;
  logic [1:0]   d_awburst, d_arburst;
  logic         d_awvalid, d_wvalid, d_wlast, d_arvalid, d_rvalid;
  logic [511:0] d_wdata;
  otpu_f2_mem #(.HQ(4)) u_mem (
    .aclk, .arst,
    .c_awvalid(m_awvalid), .c_awready(m_awready), .c_awaddr(m_awaddr),
    .c_wvalid(m_wvalid), .c_wready(m_wready), .c_wdata(m_wdata), .c_wstrb(m_wstrb),
    .c_bvalid(m_bvalid), .c_bready(m_bready), .c_bresp(m_bresp),
    .c_arvalid(m_arvalid), .c_arready(m_arready), .c_araddr(m_araddr),
    .c_rvalid(m_rvalid), .c_rready(m_rready), .c_rdata(m_rdata), .c_rresp(m_rresp),
    .h_awid('0), .h_awaddr('0), .h_awlen('0), .h_awsize(3'd6), .h_awburst(2'b01),
    .h_awvalid(1'b0), .h_awready, .h_wdata('0), .h_wstrb('0), .h_wlast(1'b0), .h_wvalid(1'b0),
    .h_wready, .h_bid, .h_bresp, .h_bvalid, .h_bready(1'b1),
    .h_arid('0), .h_araddr('0), .h_arlen('0), .h_arsize(3'd6), .h_arburst(2'b01),
    .h_arvalid(1'b0), .h_arready, .h_rid, .h_rdata, .h_rresp, .h_rlast, .h_rvalid,
    .h_rready(1'b1), .pcis_wbusy,
    .d_awid, .d_awaddr, .d_awlen, .d_awsize, .d_awburst, .d_awvalid, .d_awready(1'b1),
    .d_wdata, .d_wstrb, .d_wlast, .d_wvalid, .d_wready(1'b1), .d_bid('0), .d_bresp('0),
    .d_bvalid(1'b0), .d_bready(),
    .d_arid, .d_araddr, .d_arlen, .d_arsize, .d_arburst, .d_arvalid, .d_arready(1'b1),
    .d_rid, .d_rdata('0), .d_rresp('0), .d_rlast(1'b1), .d_rvalid, .d_rready());

  // ---- the DIMM model: an AR's beat is due LAT aclk cycles after it is taken; one beat a cycle,
  // BW percent of the cycles (a credit), in AR order
  typedef struct { logic [15:0] id; longint due; } ar_t;
  ar_t    arq[$];
  longint now = 0;
  int     cr = 0;
  logic   rv;
  logic [15:0] rid;
  assign d_rvalid = rv;
  assign d_rid = rid;
  always_ff @(posedge aclk) begin
    now <= now + 1;
    rv <= 1'b0;
    if (!arst) begin
      if (d_arvalid) arq.push_back('{d_arid, now + longint'(lat)});
      cr = (cr + bw > 200) ? 200 : cr + bw;
      if (arq.size() != 0 && arq[0].due <= now && cr >= 100) begin
        cr = cr - 100;
        rv <= 1'b1;
        rid <= arq[0].id;
        void'(arq.pop_front());
      end
    end
  end

  // ---- the native masters: a read of the next beat whenever the bridge takes one
  longint cyc = 0, got = 0, inflight = 0;
  logic [1:0][24:0] na;
  assign n_cvalid = {2{!rst}};
  assign n_caddr = na;
  always_ff @(posedge clk) begin
    cyc <= cyc + 1;
    if (rst) na <= '0;
    else for (int c = 0; c < 2; c++) if (n_cready[c]) na[c] <= na[c] + 1'b1;
    if (cyc >= cycles / 2) begin
      got <= got + longint'(n_rvalid[0]) + longint'(n_rvalid[1]);
      inflight <= inflight + longint'(g_ch[0].u_nch.rn) + longint'(g_ch[1].u_nch.rn);
    end
    if (cyc == cycles) begin
      automatic real n = real'(cycles - cycles / 2);
      $display("F2BW rod=%0d rd=%0d lat=%0d bw=%0d core_half=%0d axi_half=%0d: %.3f beats/core cycle = %.1f B/cycle, reads in flight per channel %.1f",
               ROD, RD, lat, bw, CORE_HALF, AXI_HALF, real'(got) / n, 64.0 * real'(got) / n,
               real'(inflight) / n / 2.0);
      $finish;
    end
  end
  initial begin
    repeat (8) @(posedge clk);
    rst = 1'b0;
    arst = 1'b0;
  end
endmodule
