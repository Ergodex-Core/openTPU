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
// native beat a core cycle (12.8), so the width conversion costs no bandwidth here. OT: at most
// that many AXI transactions in flight a direction (the 1 x 1 SmartConnect's acceptance, 32 in
// AWS's cl_axi_sc_1x1: a burst counts once, so READ_BURST is what makes the 64 reads a channel
// fit). The shim's queue adds one cycle (a command pushed is shown the next); the receipt's LAT
// takes it in.
// Same images, pokes and dump as otpu_native_mem (it is the model behind the shim). P2
// (docs/v4/p2-f2.md): the model is the two-channel one (NCH 2), and COPY > 0 one of the lane
// memory's image copies (otpu_native_mem COPY; the NAXI lines then "NAXI cp<COPY> ch<c>").
module otpu_naxi_mem #(
  parameter int WORDS = 1 << 18,
  parameter int LAT   = 20,
  parameter int SID   = 0,
  parameter int NCH   = 2,
  parameter int NXR   = 0,               // SLR stages each way (the build's 3)
  parameter int READ_BURST = 4,
  parameter int READ_IDLE  = 1,
  parameter bit READ_WAIT  = 1'b0,
  parameter int OT         = 32,         // AXI transactions in flight a direction (the 1 x 1
                                         //   SmartConnect's S00 / M00 acceptance: AWS's
                                         //   cl_axi_sc_1x1 takes 32; 0: no limit)
  parameter int COPY       = 0,          // the lane memory's copy (otpu_native_mem COPY)
  parameter bit HASH       = 1'b0        // the image's hash at the dump (otpu_native_mem HASH)
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
  initial if (NCH != 2) $fatal(1, "otpu_naxi_mem: the model is two channels (NCH 2)");
  otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .SID(SID), .COPY(COPY), .HASH(HASH)) u_mem (
    .clk, .rst, .run_rst,
    .n_cvalid(m_cvalid), .n_cready(m_cready), .n_cwe(m_cwe), .n_caddr(m_caddr),
    .n_wvalid(m_wvalid), .n_wready(m_wready), .n_wdata(m_wdata), .n_wmask(m_wmask),
    .n_rvalid(m_rvalid), .n_rdata(m_rdata), .n_wdone(m_wdone),
    .lb_req(1'b0), .lb_addr('0), .lb_rdy(), .lb_rvalid(), .lb_rdata(),
    .la_req(1'b0), .la_addr('0), .la_rdy(), .la_rvalid(), .la_rdata(), .l_idle(), .dump);
  int nburst [NCH], nbeat [NCH], hist [NCH][17];
  // otpu_native_axi's direction barriers, per channel: cycles a write at the queue's head waits
  // for the reads in flight to drain (wwait), cycles a read at the head waits for the writes'
  // responses (rwait), and write episodes (a write taking the head after reads: switches)
  longint wwait [NCH], rwait [NCH], nswitch [NCH];
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
                      .READ_BURST(READ_BURST), .READ_IDLE(READ_IDLE), .READ_WAIT(READ_WAIT)) u_naxi (
      .clk, .rst,
      .n_cvalid(b_cvalid), .n_cready(b_cready), .n_cwe(b_cwe), .n_caddr(b_caddr),
      .n_wvalid(b_wvalid), .n_wready(b_wready), .n_wdata(b_wdata), .n_wmask(b_wmask),
      .n_rvalid(b_rvalid), .n_rdata(b_rdata), .n_wdone(b_wdone), .n_err(b_err),
      .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awready), .m_wdata(wdata), .m_wstrb(wstrb),
      .m_wvalid(wvalid), .m_wready(wready), .m_bvalid(bvalid), .m_bresp(2'b00), .m_bready(bready),
      .m_araddr(araddr), .m_arlen(arlen), .m_arvalid(arvalid), .m_arready(arready),
      .m_rdata(m_rdata[c]), .m_rvalid(rvalid), .m_rlast(rlast), .m_rresp(2'b00), .m_rready(rready));
    always @(posedge clk) if (!rst && b_err) $fatal(1, "otpu_naxi_mem ch%0d: otpu_native_axi n_err", c);
    logic was_w;
    always @(posedge clk)
      if (rst) begin wwait[c] <= 0; rwait[c] <= 0; nswitch[c] <= 0; was_w <= 1'b0; end
      else begin
        if (u_naxi.cn != 0 && u_naxi.head_we && u_naxi.reads != 0) wwait[c] <= wwait[c] + 1;
        if (u_naxi.cn != 0 && !u_naxi.head_we && u_naxi.writes != 0) rwait[c] <= rwait[c] + 1;
        if (u_naxi.cn != 0) begin
          if (u_naxi.head_we && !was_w) nswitch[c] <= nswitch[c] + 1;
          was_w <= u_naxi.head_we;
        end
      end
    // ---- the shim: AXI back to native commands (circular buffers, every state change a
    // non-blocking update: the model and otpu_native_axi sample it at the same edges)
    localparam int QN = 64;
    logic [25:0]  q [QN];                    // native commands {we, beat}, in AXI arrival order
    logic [575:0] wq [16];
    logic [4:0]   lens [128];                // the bursts' beats, in AR order
    logic [6:0]   qh, qt, lh, lt;
    logic [3:0]   wh, wt;
    logic [7:0]   qn;
    logic [4:0]   wn;
    logic [4:0]   rleft;                     // the current burst's beats still to come back
    logic         aw_p, w_p;                 // an AW / a W beat held for its pair
    logic [24:0]  aw_a;
    logic [575:0] w_d;
    logic [15:0]  wd_seen;
    logic [15:0]  bpend;
    logic [7:0]   rd_ot, wr_ot;              // bursts / writes accepted, not yet answered
    wire [4:0] nar = 5'(arlen) + 5'd1;
    assign arready = qn <= 8'(QN - 17) && (OT == 0 || rd_ot < 8'(OT));
    assign awready = !aw_p && qn <= 8'(QN - 17) && (OT == 0 || wr_ot < 8'(OT));
    assign wready = !w_p && wn < 5'd15;
    assign m_cvalid[c] = qn != 0;
    assign {m_cwe[c], m_caddr[c]} = q[qh[5:0]];
    assign m_wvalid[c] = wn != 0;
    assign {m_wmask[c], m_wdata[c]} = wq[wh];
    assign rvalid = m_rvalid[c];
    assign rlast = rleft == 5'd1 || (rleft == 5'd0 && lens[lh] == 5'd1);
    assign bvalid = bpend != 0;
    wire c_pop = m_cvalid[c] && m_cready[c];
    wire w_pop = m_wvalid[c] && m_wready[c];
    wire ar_tk = arvalid && arready;
    wire pair = aw_p && w_p;
    always @(posedge clk) begin
      if (rst) begin
        qh <= '0; qt <= '0; qn <= '0; wh <= '0; wt <= '0; wn <= '0; lh <= '0; lt <= '0;
        rleft <= '0; aw_p <= 1'b0; w_p <= 1'b0; wd_seen <= '0; bpend <= '0;
        rd_ot <= '0; wr_ot <= '0;
      end else begin
        rd_ot <= rd_ot + 8'(ar_tk) - 8'(rvalid && rlast);
        wr_ot <= wr_ot + 8'(awvalid && awready) - 8'(bvalid && bready);
        if (rvalid) begin
          if (rleft == 0) begin rleft <= lens[lh] - 5'd1; lh <= lh + 1'b1; end
          else rleft <= rleft - 5'd1;
        end
        bpend <= bpend - 16'(bvalid && bready) + (m_wdone[c] - wd_seen);
        wd_seen <= m_wdone[c];
        // commands in: a burst's beats (AR) or a write pair, one source a cycle (AR first)
        if (ar_tk) begin
          for (int i = 0; i < 16; i++)
            if (i < int'(nar)) q[6'(qt + 7'(i))] <= {1'b0, 25'(araddr[33:6] + 28'(i))};
          qt <= qt + 7'(nar);
          lens[lt] <= nar; lt <= lt + 1'b1;
          nburst[c] <= nburst[c] + 1; nbeat[c] <= nbeat[c] + int'(nar); hist[c][nar] <= hist[c][nar] + 1;
        end else if (pair) begin
          q[qt[5:0]] <= {1'b1, aw_a}; qt <= qt + 1'b1;
          wq[wt] <= w_d; wt <= wt + 1'b1;
          aw_p <= 1'b0; w_p <= 1'b0;
        end
        qn <= qn + (ar_tk ? 8'(nar) : pair ? 8'd1 : 8'd0) - 8'(c_pop);
        wn <= wn + 5'(!ar_tk && pair) - 5'(w_pop);
        if (c_pop) qh <= qh + 1'b1;
        if (w_pop) wh <= wh + 1'b1;
        if (awvalid && awready) begin aw_p <= 1'b1; aw_a <= 25'(awaddr[33:6]); end
        if (wvalid && wready) begin w_p <= 1'b1; w_d <= {wstrb, wdata}; end
      end
    end
  end
  always @(posedge clk) if (dump)
    for (int c = 0; c < NCH; c++) begin
      if (COPY == 0) $write("NAXI ch%0d bursts=%0d beats=%0d lengths", c, nburst[c], nbeat[c]);
      else $write("NAXI cp%0d ch%0d bursts=%0d beats=%0d lengths", COPY, c, nburst[c], nbeat[c]);
      for (int i = 1; i <= 16; i++) if (hist[c][i]) $write(" %0d:%0d", i, hist[c][i]);
      $display(" wwait=%0d rwait=%0d switches=%0d", wwait[c], rwait[c], nswitch[c]);
    end
endmodule
