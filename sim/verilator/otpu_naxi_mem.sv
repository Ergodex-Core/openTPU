// The F2 HBM build's memory path in the decode receipt (rtlsim OTPU_NAXI=1; otpu_top NAXI): per
// channel the adapter's native port across NXR register stages each way (the SLR chain, as
// otpu_f2_hbm_axi g_xr) into the build's otpu_native_axi (XL = 2 NXR, READ_BURST, READ_IDLE: its
// coalescing, direction barriers and queue as built), whose AXI4 master is turned back into
// native commands for otpu_native_mem (the receipts' memory model: LAT cycles from a command
// performed to its data, the whole of the SmartConnect's and the HBM's latency). The AXI side
// here: AR / AW / W taken when the command queue has room, a burst of n beats as n native
// reads in order, a write once its AW and W are both in; R beats from otpu_native_mem in order
// with RLAST at each burst's end (no back-pressure, as the SmartConnect towards otpu_native_axi,
// whose R is always ready); a B per write counted in n_wdone. The model performs at most a
// command a cycle per channel: the HBM port's 256 bits at 450 MHz (14.4 GB/s) are above a
// native beat a core cycle (12.8), so the width conversion costs no bandwidth here. The shim's
// queue adds one cycle (a command pushed is shown the next); the receipt's LAT takes it in.
// Same images, pokes and dump as otpu_native_mem (it is the model behind the shim).
module otpu_naxi_mem #(
  parameter int WORDS = 1 << 18,
  parameter int LAT   = 20,
  parameter int SID   = 0,
  parameter int NCH   = 2,
  parameter int NXR   = 0,               // SLR stages each way (the build's 3)
  parameter int READ_BURST = 4,
  parameter int READ_IDLE  = 1
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  run_rst,
  input  logic [NCH-1:0]            n_cvalid,
  output logic [NCH-1:0]            n_cready,
  input  logic [NCH-1:0]            n_cwe,
  input  logic [NCH-1:0][24:0]      n_caddr,
  input  logic [NCH-1:0]            n_wvalid,
  output logic [NCH-1:0]            n_wready,
  input  logic [NCH-1:0][511:0]     n_wdata,
  input  logic [NCH-1:0][63:0]      n_wmask,
  output logic [NCH-1:0]            n_rvalid,
  output logic [NCH-1:0][511:0]     n_rdata,
  output logic [NCH-1:0][15:0]      n_wdone,
  input  logic                  dump
);
  // the model's native ports (m_*), behind the shims
  logic [NCH-1:0]        m_cvalid, m_cready, m_cwe, m_wvalid, m_wready, m_rvalid;
  logic [NCH-1:0][24:0]  m_caddr;
  logic [NCH-1:0][511:0] m_wdata, m_rdata;
  logic [NCH-1:0][63:0]  m_wmask;
  logic [NCH-1:0][15:0]  m_wdone;
  otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .SID(SID), .NCH(NCH)) u_mem (
    .clk, .rst, .run_rst,
    .n_cvalid(m_cvalid), .n_cready(m_cready), .n_cwe(m_cwe), .n_caddr(m_caddr),
    .n_wvalid(m_wvalid), .n_wready(m_wready), .n_wdata(m_wdata), .n_wmask(m_wmask),
    .n_rvalid(m_rvalid), .n_rdata(m_rdata), .n_wdone(m_wdone), .dump);
  int nburst [NCH], nbeat [NCH], hist [NCH][17];
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    // ---- the SLR chain: down the takes with their payloads, up the readies and responses
    localparam int DW = 1 + 1 + 25 + 1 + 512 + 64, UW = 1 + 1 + 1 + 512 + 16;
    logic [NXR:0][DW-1:0] dn;
    logic [NXR:0][UW-1:0] up;
    logic b_cvalid, b_cready, b_cwe, b_wvalid, b_wready, b_rvalid, b_err;
    logic [24:0] b_caddr;
    logic [511:0] b_wdata, b_rdata;
    logic [63:0] b_wmask;
    logic [15:0] b_wdone;
    assign dn[NXR] = {n_cvalid[c] && n_cready[c], n_cwe[c], n_caddr[c], n_wvalid[c] && n_wready[c],
                      n_wdata[c], n_wmask[c]};
    assign {b_cvalid, b_cwe, b_caddr, b_wvalid, b_wdata, b_wmask} = dn[0];
    assign up[0] = {b_cready, b_wready, b_rvalid, b_rdata, b_wdone};
    assign {n_cready[c], n_wready[c], n_rvalid[c], n_rdata[c], n_wdone[c]} = up[NXR];
    for (genvar k = 0; k < NXR; k++) begin : g_x
      always_ff @(posedge clk) begin dn[k] <= dn[k + 1]; up[k + 1] <= up[k]; end
    end
    // ---- the build's channel adapter
    logic [33:0] awaddr, araddr;
    logic [7:0] arlen;
    logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rlast, rready;
    logic [511:0] wdata;
    logic [63:0] wstrb;
    otpu_native_axi #(.BASE_ADDR(34'd0), .CH_LOG2(31), .QD(16), .XL(2 * NXR), .MAX_OUT(128),
                      .READ_BURST(READ_BURST), .READ_IDLE(READ_IDLE)) u_naxi (
      .clk, .rst,
      .n_cvalid(b_cvalid), .n_cready(b_cready), .n_cwe(b_cwe), .n_caddr(b_caddr),
      .n_wvalid(b_wvalid), .n_wready(b_wready), .n_wdata(b_wdata), .n_wmask(b_wmask),
      .n_rvalid(b_rvalid), .n_rdata(b_rdata), .n_wdone(b_wdone), .n_err(b_err),
      .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awready), .m_wdata(wdata), .m_wstrb(wstrb),
      .m_wvalid(wvalid), .m_wready(wready), .m_bvalid(bvalid), .m_bresp(2'b00), .m_bready(bready),
      .m_araddr(araddr), .m_arlen(arlen), .m_arvalid(arvalid), .m_arready(arready),
      .m_rdata(m_rdata[c]), .m_rvalid(rvalid), .m_rlast(rlast), .m_rresp(2'b00), .m_rready(rready));
    always @(posedge clk) if (!rst && b_err) $fatal(1, "otpu_naxi_mem ch%0d: otpu_native_axi n_err", c);
    // ---- the shim: AXI back to native commands
    typedef struct packed { logic we; logic [24:0] a; } nc_t;
    nc_t          q [$];                     // native commands, in AXI arrival order
    logic [575:0] wq [$];
    int           lens [$];
    int           rleft = 0;
    logic         aw_p, w_p;                 // an AW / a W beat held for its pair
    logic [24:0]  aw_a;
    logic [575:0] w_d;
    logic [15:0]  wd_seen;
    int           bpend = 0;
    assign arready = q.size() < 32;
    assign awready = !aw_p && q.size() < 32;
    assign wready = !w_p;
    assign m_cvalid[c] = q.size() != 0;
    assign m_cwe[c] = q.size() ? q[0].we : 1'b0;
    assign m_caddr[c] = q.size() ? q[0].a : '0;
    assign m_wvalid[c] = wq.size() != 0;
    assign {m_wmask[c], m_wdata[c]} = wq.size() ? wq[0] : '0;
    assign rvalid = m_rvalid[c];
    assign rlast = rleft == 1 || (rleft == 0 && lens.size() && lens[0] == 1);
    assign bvalid = bpend != 0;
    always @(posedge clk) begin
      if (rst) begin
        q.delete(); wq.delete(); lens.delete(); rleft = 0; aw_p <= 1'b0; w_p <= 1'b0;
        wd_seen <= '0; bpend = 0;
      end else begin
        if (rvalid) begin
          if (rleft == 0) begin rleft = lens[0]; void'(lens.pop_front()); end
          rleft = rleft - 1;
        end
        if (bvalid && bready) bpend = bpend - 1;
        bpend = bpend + int'(16'(m_wdone[c] - wd_seen));
        wd_seen <= m_wdone[c];
        if (m_cvalid[c] && m_cready[c]) void'(q.pop_front());
        if (m_wvalid[c] && m_wready[c]) void'(wq.pop_front());
        if (arvalid && arready) begin
          lens.push_back(int'(arlen) + 1);
          for (int i = 0; i <= int'(arlen); i++) q.push_back({1'b0, 25'(araddr[33:6] + 28'(i))});
          nburst[c]++; nbeat[c] += int'(arlen) + 1; hist[c][arlen + 1]++;
        end
        if (awvalid && awready) begin aw_p <= 1'b1; aw_a <= 25'(awaddr[33:6]); end
        if (wvalid && wready) begin w_p <= 1'b1; w_d <= {wstrb, wdata}; end
        if (aw_p && w_p) begin
          q.push_back({1'b1, aw_a}); wq.push_back(w_d); aw_p <= 1'b0; w_p <= 1'b0;
        end
      end
    end
  end
  always @(posedge clk) if (dump)
    for (int c = 0; c < NCH; c++) begin
      $write("NAXI ch%0d bursts=%0d beats=%0d lengths", c, nburst[c], nbeat[c]);
      for (int i = 1; i <= 16; i++) if (hist[c][i]) $write(" %0d:%0d", i, hist[c][i]);
      $display("");
    end
endmodule
