// One memory channel of a slice (otpu_native_dram's native port, core clock) onto one HBM AXI
// port (the cl_hbm IP's AXI3 slave through cl_mem_hbm_wrapper with AXI4_INTERFACE 0: 256-bit
// data, 4-bit lengths, 34-bit addresses, hclk = 450 MHz), at BASE + beat x 64: the F2 HBM build's
// counterpart of otpu_f2_nch (docs/f2_hbm.md). The port's pseudo-channel is the channel's own
// 512 MiB (BASE = its port x 512 MiB), so no access crosses the HBM's switch.
//
// Core side (clk). Commands and write data are taken into two small FIFOs (their readies are
// flip-flops: the native port's command select stays off this module's logic) and leave them in
// command order, one a cycle, through the order check below into an asynchronous FIFO to hclk
// (otpu_afifo: the command and its beat, 602 bits). The order check is otpu_f2_nch's, moved to
// the core clock where its compares are cheap (at 450 MHz a 64-entry compare would not close):
//   - a read waits while a write to its beat is in flight (issued, not yet counted in n_wdone),
//   - a write waits while a read of its beat is in flight (issued, its data not yet returned),
// so accesses to one beat reach the HBM one at a time, in command order, whatever the AXI port
// reorders (its read and write channels are independent; the controller reorders to open pages),
// and accesses to different beats overlap freely. The tables retire in order: reads at n_rvalid
// (single AXI ID: data in order), writes as n_wdone advances. Reads in flight (issued, data not
// returned) are at most RD, the return FIFO's depth, so the HBM's R channel is never stalled.
//
// HBM side (hclk). The head command is held in a register, then joins the open burst when it
// continues it (same direction, the next beat, the burst below BMAX native beats and not past a
// BMAX-beat boundary, so no burst crosses 4 KiB); else the open burst closes into the AR or AW
// queue and the head opens the next. An open burst also closes after GATHER hclk cycles without a
// command (the core sends at most one per core cycle, 2.25 hclk cycles: a stream's beats arrive
// spaced, and the wait lets them join), so a streamed run of beats on this channel goes out as
// bursts of up to BMAX x 64 bytes (16 AXI beats, AXI3's longest) while a lone beat waits only
// GATHER cycles (~9 ns of the HBM's ~150-300 ns read latency). A native beat is two AXI beats
// (low 256 bits first). A write's data waits in the W FIFO from its command's pop until its
// burst's AW is accepted; the W beats then follow in AW order (the port arbiter, otpu_f2_hbm_port,
// locks W to the master whose AW it granted). B responses add their bursts' beats to the write
// count, which steps by one an hclk cycle toward it and crosses back gray-coded (n_wdone). R
// beats pair into native beats and cross back through the return FIFO.
//
// Little's law: the core takes at most one beat a cycle per channel (12.8 GB/s at 200 MHz), the
// port gives 14.4 GB/s at 450 MHz; at ~400 ns from issue to data back in the core clock (the two
// crossings, the bursts' gather, the HBM's latency under load) a channel needs ~80 reads in
// flight to run at the core's full rate and ~20 at a stream's share (1 / (NCH / 2) of 25.6 GB/s
// on NCH = 8): RD = 64 reads covers 10 GB/s per channel, more than a stream's 3.2.
//
// n_err: a response other than OKAY (sticky until rst). Resets: rst (core) and hrst (hclk) are
// the same event, each side's synchronized; nothing is in flight while the accelerator is idle.
module otpu_f2_hbm_nch #(
  parameter logic [33:0] BASE = 34'h0,  // the channel's first byte in the HBM
  parameter int CQ     = 16,            // the crossing's command FIFO, beats
  parameter int RD     = 64,            // reads in flight (the return FIFO's depth)
  parameter int WOD    = 16,            // writes in flight
  parameter int BMAX   = 8,             // native beats per burst (AXI3: at most 8 = 16 x 32 B)
  parameter int GATHER = 4,             // hclk cycles an open burst waits for its next beat
  parameter bit HAZ    = 1'b1           // 0: no same-beat waits (tests only: tb_f2hbm fails)
) (
  // ---- the native port (core clock)
  input  logic          clk,
  input  logic          rst,
  input  logic          n_cvalid,
  output logic          n_cready,
  input  logic          n_cwe,
  input  logic [24:0]   n_caddr,
  input  logic          n_wvalid,
  output logic          n_wready,
  input  logic [511:0]  n_wdata,
  input  logic [63:0]   n_wmask,
  output logic          n_rvalid,
  output logic [511:0]  n_rdata,
  output logic [15:0]   n_wdone,
  output logic          n_err,
  // ---- AXI3 master (hclk): INCR bursts of 32-byte beats, one ID (otpu_f2_hbm_port gives it)
  input  logic          hclk,
  input  logic          hrst,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [33:0]   m_awaddr,
  output logic [3:0]    m_awlen,
  output logic          m_wvalid,
  input  logic          m_wready,
  output logic [255:0]  m_wdata,
  output logic [31:0]   m_wstrb,
  output logic          m_wlast,
  input  logic          m_bvalid,
  output logic          m_bready,
  input  logic [1:0]    m_bresp,
  output logic          m_arvalid,
  input  logic          m_arready,
  output logic [33:0]   m_araddr,
  output logic [3:0]    m_arlen,
  input  logic          m_rvalid,
  output logic          m_rready,
  input  logic [255:0]  m_rdata,
  input  logic [1:0]    m_rresp,
  input  logic          m_rlast
);
  localparam int RW = $clog2(RD), WW = $clog2(WOD), BW = $clog2(BMAX);
  initial if (RD < 4 || WOD < 2 || RD != (1 << RW) || WOD != (1 << WW) || BMAX < 1 || BMAX > 8 ||
              BMAX != (1 << BW) || CQ < 4)
    $fatal(1, "otpu_f2_hbm_nch: RD, WOD, BMAX powers of two (BMAX <= 8), CQ >= 4");

  // ================================================================== core side
  // ---- the native port into two FIFOs of 4 (readies from flip-flops: room for two more)
  logic [25:0]  cf [4];
  logic [575:0] df [4];
  logic [2:0]   cf_n, df_n;
  logic [1:0]   cf_h, df_h;
  logic         c_pop, d_pop;
  wire c_take = n_cvalid && n_cready;
  wire d_take = n_wvalid && n_wready;
  always_ff @(posedge clk) begin
    if (rst) begin
      cf_n <= '0; df_n <= '0; cf_h <= '0; df_h <= '0; n_cready <= 1'b0; n_wready <= 1'b0;
    end else begin
      if (c_take) cf[2'(cf_h + cf_n)] <= {n_cwe, n_caddr};
      if (d_take) df[2'(df_h + df_n)] <= {n_wmask, n_wdata};
      cf_n <= cf_n + 3'(c_take) - 3'(c_pop);
      df_n <= df_n + 3'(d_take) - 3'(d_pop);
      if (c_pop) cf_h <= cf_h + 1'b1;
      if (d_pop) df_h <= df_h + 1'b1;
      // room for this cycle's take and the next's: at most 2 after this cycle's changes
      n_cready <= (cf_n + 3'(c_take) - 3'(c_pop)) <= 3'd2;
      n_wready <= (df_n + 3'(d_take) - 3'(d_pop)) <= 3'd2;
    end
  end
  wire        h_we = cf[cf_h][25];
  wire [24:0] h_a  = cf[cf_h][24:0];

  // ---- in flight: the beats of the issued writes (to n_wdone) and reads (to n_rvalid)
  logic [WOD-1:0][24:0] wa;
  logic [WOD-1:0]       wv;
  logic [WW:0]          wn;             // writes issued, not retired
  logic [WW-1:0]        wt, wr_;        // the next write's slot; the oldest's
  logic [RD-1:0][24:0]  ra;
  logic [RD-1:0]        rv;
  logic [RW:0]          rn;
  logic [RW-1:0]        rt, rh;
  logic [15:0]          wdone_s, wret;  // n_wdone as crossed; writes retired (counts mod 2^16)
  logic                 hz_w, hz_r;
  always_comb begin
    hz_w = 1'b0;
    for (int k = 0; k < WOD; k++) hz_w |= HAZ && wv[k] && wa[k] == h_a;
    hz_r = 1'b0;
    for (int k = 0; k < RD; k++) hz_r |= HAZ && rv[k] && ra[k] == h_a;
  end

  // ---- issue: the head command into the crossing (a write with its data)
  logic        x_ready;
  wire iss_w = cf_n != 0 && h_we && df_n != 0 && x_ready && !wn[WW] && !hz_r;
  wire iss_r = cf_n != 0 && !h_we && x_ready && !rn[RW] && !hz_w;
  assign c_pop = iss_w || iss_r;
  assign d_pop = iss_w;
  logic        x_v, x_we;
  logic [24:0] x_a;
  logic [575:0] x_d;
  logic        xr_v;
  logic [511:0] xr_d;
  logic x_pop;
  otpu_afifo #(.W(602), .DEPTH(CQ)) u_xq (
    .wclk(clk), .wrst(rst), .wvalid(c_pop), .wready(x_ready), .wdata({h_we, h_a, df[df_h]}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst), .rvalid(x_v), .rready(x_pop), .rdata({x_we, x_a, x_d}));

  always_ff @(posedge clk) begin
    if (iss_w) wa[wt] <= h_a;
    if (iss_r) ra[rt] <= h_a;
    if (rst) begin
      wv <= '0; wn <= '0; wt <= '0; wr_ <= '0; wret <= wdone_s;
      rv <= '0; rn <= '0; rt <= '0; rh <= '0;
    end else begin
      // a write retires as the count passes it (one a cycle: the count may step by more)
      logic wrt;
      wrt = wret != wdone_s;
      if (iss_w) begin wv[wt] <= 1'b1; wt <= wt + 1'b1; end
      if (wrt) begin wv[wr_] <= 1'b0; wr_ <= wr_ + 1'b1; wret <= wret + 1'b1; end
      wn <= wn + (WW + 1)'(iss_w) - (WW + 1)'(wrt);
      if (iss_r) begin rv[rt] <= 1'b1; rt <= rt + 1'b1; end
      if (n_rvalid) begin rv[rh] <= 1'b0; rh <= rh + 1'b1; end
      rn <= rn + (RW + 1)'(iss_r) - (RW + 1)'(n_rvalid);
    end
  end

  // ================================================================== HBM side
  // ---- the crossing's head, registered (its LUT RAM read stays off the burst compare)
  logic        k_v, k_we;
  logic [24:0] k_a;
  logic [575:0] k_d;
  logic        k_pop;                  // the held command joins or opens a burst
  assign x_pop = x_v && (!k_v || k_pop);
  always_ff @(posedge hclk) begin
    if (hrst) k_v <= 1'b0;
    else if (!k_v || k_pop) begin
      k_v <= x_v;
      if (x_v) begin k_we <= x_we; k_a <= x_a; k_d <= x_d; end
    end
  end

  // ---- the open burst; the AR / AW queues (2 bursts each), the W FIFO (beats of queued bursts)
  logic        o_v, o_we;
  logic [24:0] o_a, o_nx;               // its first beat; the beat that would continue it
  logic [BW:0] o_len;                   // native beats
  logic [3:0]  o_age;
  typedef struct packed { logic [24:0] a; logic [BW:0] len; } bq_t;
  bq_t         arq [2], awq [2];
  logic [1:0]  arq_n, awq_n;
  logic        arq_h, awq_h;
  localparam int WD = 4 * BMAX;         // W FIFO: the open burst and three queued or sending
  localparam int WDW = $clog2(WD);
  logic [575:0] wf [WD];
  logic [WDW:0] wf_n;
  logic [WDW-1:0] wf_h;
  // the head continues the open burst: same direction, the next beat, room, no BMAX boundary
  wire cont = o_v && k_we == o_we && k_a == o_nx && o_len != (BW + 1)'(BMAX) &&
              (BW == 0 ? 1'b0 : (k_a % BMAX) != 0);
  // close the open burst: full, the head does not continue it, or it waited GATHER cycles
  wire q_room = o_we ? awq_n != 2'd2 : arq_n != 2'd2;
  wire close = o_v && q_room && (o_len == (BW + 1)'(BMAX) || (k_v && !cont) ||
                                 (!k_v && o_age >= 4'(GATHER)));
  wire w_room = wf_n <= (WDW + 1)'(WD - 1);
  // the head joins the open burst, or opens one where none is or the open one closes now
  assign k_pop = k_v && (!k_we || w_room) && (cont || !o_v || close);
  wire k_join = k_pop && cont;
  wire k_open = k_pop && !cont;

  // ---- AW / AR / W channels
  logic aw_tk, ar_tk, w_tk, wb_v, b_tk, r_tk;
  logic [BW:0] wb_len [4];              // bursts whose AW went: their beats, in order (W, B)
  logic [2:0]  wb_n, bb_n;
  logic [1:0]  wb_h, bb_h, wb_t;
  logic [BW+1:0] w_cnt;                 // AXI beats of the sending burst gone
  assign m_awvalid = awq_n != 0 && wb_n != 3'd4 && bb_n != 3'd4;
  assign m_awaddr  = BASE + {awq[awq_h].a, 6'd0};
  assign m_awlen   = 4'(2 * awq[awq_h].len - 1);
  assign m_arvalid = arq_n != 0;
  assign m_araddr  = BASE + {arq[arq_h].a, 6'd0};
  assign m_arlen   = 4'(2 * arq[arq_h].len - 1);
  assign aw_tk = m_awvalid && m_awready;
  assign ar_tk = m_arvalid && m_arready;
  // W: the oldest sent AW's burst, from the W FIFO's head (two AXI beats per native beat)
  assign wb_v = wb_n != 0 && wf_n != 0;
  wire [575:0] wh = wf[wf_h];
  assign m_wvalid = wb_v;
  assign m_wdata  = w_cnt[0] ? wh[511:256] : wh[255:0];
  assign m_wstrb  = w_cnt[0] ? wh[575:544] : wh[543:512];
  assign m_wlast  = w_cnt == (BW + 2)'(2 * wb_len[wb_h] - 1);
  assign w_tk = m_wvalid && m_wready;
  assign m_bready = 1'b1;
  assign m_rready = 1'b1;                // the return FIFO has room for every read in flight
  assign b_tk = m_bvalid;
  assign r_tk = m_rvalid;

  // bb: AW gone, B not back (the write count's beats); wb: AW gone, W not all sent
  logic [BW:0] bb_len [4];
  logic [15:0] btarget, bcnt, bgray;
  logic        aerr;
  always_ff @(posedge hclk) begin
    if (k_pop && k_we) wf[WDW'(wf_h + wf_n)] <= k_d;
    if (aw_tk) begin wb_len[wb_t] <= awq[awq_h].len; bb_len[2'(bb_h + bb_n)] <= awq[awq_h].len; end
    if (hrst) begin
      o_v <= 1'b0; o_age <= '0; arq_n <= '0; awq_n <= '0; arq_h <= 1'b0; awq_h <= 1'b0;
      wf_n <= '0; wf_h <= '0; wb_n <= '0; bb_n <= '0; wb_h <= '0; bb_h <= '0; wb_t <= '0;
      w_cnt <= '0; btarget <= '0; bcnt <= '0; bgray <= '0; aerr <= 1'b0;
    end else begin
      // the open burst
      if (close) begin
        if (o_we) awq[1'(awq_h + awq_n[0])] <= '{a: o_a, len: o_len};
        else      arq[1'(arq_h + arq_n[0])] <= '{a: o_a, len: o_len};
      end
      if (k_open) begin
        o_v <= 1'b1; o_we <= k_we; o_a <= k_a; o_nx <= k_a + 1'b1; o_len <= (BW + 1)'(1);
        o_age <= '0;
      end else if (k_join) begin
        o_nx <= o_nx + 1'b1; o_len <= o_len + 1'b1; o_age <= '0;
      end else begin
        if (close) o_v <= 1'b0;
        if (o_age != '1) o_age <= o_age + 1'b1;
      end
      awq_n <= awq_n + 2'(close && o_we) - 2'(aw_tk);
      arq_n <= arq_n + 2'(close && !o_we) - 2'(ar_tk);
      if (aw_tk) awq_h <= !awq_h;
      if (ar_tk) arq_h <= !arq_h;
      // W
      wf_n <= wf_n + (WDW + 1)'(k_pop && k_we) - (WDW + 1)'(w_tk && w_cnt[0]);
      if (w_tk && w_cnt[0]) wf_h <= wf_h + 1'b1;
      if (w_tk) w_cnt <= m_wlast ? '0 : w_cnt + 1'b1;
      if (aw_tk) wb_t <= wb_t + 1'b1;
      if (w_tk && m_wlast) wb_h <= wb_h + 1'b1;
      wb_n <= wb_n + 3'(aw_tk) - 3'(w_tk && m_wlast);
      // B: the burst's beats into the count, which steps by one a cycle (gray: one bit a step)
      if (b_tk) begin btarget <= btarget + 16'(bb_len[bb_h]); bb_h <= bb_h + 1'b1; end
      bb_n <= bb_n + 3'(aw_tk) - 3'(b_tk);
      if (bcnt != btarget) bcnt <= bcnt + 1'b1;
      bgray <= bcnt ^ (bcnt >> 1);
      if ((b_tk && m_bresp != 2'b00) || (r_tk && m_rresp != 2'b00)) aerr <= 1'b1;
    end
  end

  // ---- R: registered (the HBM's R off the return FIFO's write enables: the 89bbb6bc synthesis,
  // -0.21 ns at 450 MHz), then two AXI beats a native beat, back to the core
  logic         r_half, rr_v;
  logic [255:0] r_lo, rr_d;
  always_ff @(posedge hclk) begin
    rr_d <= m_rdata;
    if (hrst) begin r_half <= 1'b0; rr_v <= 1'b0; end
    else begin
      rr_v <= r_tk;
      if (rr_v) begin
        r_half <= !r_half;
        if (!r_half) r_lo <= rr_d;
      end
    end
  end
  otpu_afifo #(.W(512), .DEPTH(RD)) u_rq (
    .wclk(hclk), .wrst(hrst), .wvalid(rr_v && r_half), .wready(), .wdata({rr_d, r_lo}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(clk), .rrst(rst), .rvalid(xr_v), .rready(1'b1), .rdata(xr_d));
  assign n_rvalid = xr_v;
  assign n_rdata = xr_d;

  // ---- the write count and errors into the core clock
  (* ASYNC_REG = "TRUE" *) logic [15:0] bg1, bg2;
  (* ASYNC_REG = "TRUE" *) logic        e1, e2;
  logic [15:0] bbin;
  always_comb for (int i = 0; i < 16; i++) bbin[i] = ^(bg2 >> i);
  always_ff @(posedge clk) begin
    if (rst) begin
      bg1 <= '0; bg2 <= '0; e1 <= 1'b0; e2 <= 1'b0; n_wdone <= '0; n_err <= 1'b0;
    end else begin
      bg1 <= bgray; bg2 <= bg1; e1 <= aerr; e2 <= e1;
      n_wdone <= bbin;
      n_err <= n_err || e2;
    end
  end
  assign wdone_s = n_wdone;

`ifndef SYNTHESIS
  always_ff @(posedge hclk) if (!hrst) begin
    if (b_tk && bb_n == 0) $fatal(1, "otpu_f2_hbm_nch: B with no write in flight");
    if (k_pop && k_we && !w_room) $fatal(1, "otpu_f2_hbm_nch: W FIFO overflow");
  end
  always_ff @(posedge clk) if (!rst && n_rvalid && rn == 0)
    $fatal(1, "otpu_f2_hbm_nch: read data with no read in flight");
`endif
endmodule
