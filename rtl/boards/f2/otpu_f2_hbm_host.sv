// The host's PCIS (AppPF BAR4: AXI4, 512 bits, 64-byte beats, its own IDs; aclk = clk_main_a0,
// 250 MHz) onto the HBM ports' host masters (otpu_f2_hbm_port master 1; hclk = 450 MHz): BAR4
// offset = HBM byte address, so port p (pseudo-channel p, bytes p x 512 MiB ..) holds slice s's
// channel c at p = NCH s + c (opentpu/host/f2.py bases, channel_map). docs/f2_hbm.md.
//
// Writes: every 64-byte W beat crosses into hclk with its address (afifo), then goes to the port
// of its address as one AXI3 burst of two 32-byte beats (AW, then W low and high halves): one
// beat at a time, in order (host traffic is the weights' load and a run's inputs: ~2 GB/s at most
// through BAR4 write combining; a beat per ~7 hclk cycles through the ports' registers, below,
// is 4 GB/s). The host's B for a burst goes back once the write count (every port's B for the
// host, crossed back gray-coded) reaches the burst's last beat. Beats to a port past the last
// (NPORT) are dropped (counted done).
//
// The ports' side (p_*): the head beat's AW, W and AR go to every port on shared buses with a
// valid per port, and each port registers them (otpu_f2_hbm_mem's otpu_f2_hbm_hreg: a one-entry
// register per channel whose ready is its own emptiness), as it registers its R and B back, so
// nothing here reaches a port's arbiter or an HBM pin in the cycle, and a port's distance from
// this block is a register's cycle. The resets through registers of their own (arst_q, hrst_q:
// otpu_f2_hbm_nch).
//
// Order (otpu_f2_mem's): pcis_wbusy (an AW or W shown or taken, a burst's B not yet given back)
// holds the host's reads, so a read after a write returns the written data whichever ports the
// two use, and holds the OCL's writes (otpu_axil_cdc), so a RUN after the data finds it in the
// HBM. Reads: each beat of an AR crosses as a read of its 64 bytes (two AXI beats), at most RQ in
// flight (the return FIFO's depth); one port at a time (a read to another port waits until the
// last port's reads have returned: one ID's reads come back in order on one port, not across
// ports), so the data comes back in the host's order. A read past NPORT returns zeros.
module otpu_f2_hbm_host #(
  parameter int NPORT = 8,              // HBM ports in use (S x NCH), at most 32
  parameter int HQ    = 4,              // host bursts outstanding per direction
  parameter int CQ    = 16,             // the crossings' FIFOs, beats
  parameter int RQ    = 16              // host read beats in flight (the return FIFO)
) (
  input  logic                  aclk,
  input  logic                  arst,
  // ---- PCIS (AXI4 slave, aclk)
  input  logic [15:0]           h_awid,
  input  logic [63:0]           h_awaddr,
  input  logic [7:0]            h_awlen,
  input  logic                  h_awvalid,
  output logic                  h_awready,
  input  logic [511:0]          h_wdata,
  input  logic [63:0]           h_wstrb,
  input  logic                  h_wlast,
  input  logic                  h_wvalid,
  output logic                  h_wready,
  output logic [15:0]           h_bid,
  output logic [1:0]            h_bresp,
  output logic                  h_bvalid,
  input  logic                  h_bready,
  input  logic [15:0]           h_arid,
  input  logic [63:0]           h_araddr,
  input  logic [7:0]            h_arlen,
  input  logic                  h_arvalid,
  output logic                  h_arready,
  output logic [15:0]           h_rid,
  output logic [511:0]          h_rdata,
  output logic [1:0]            h_rresp,
  output logic                  h_rlast,
  output logic                  h_rvalid,
  input  logic                  h_rready,
  output logic                  pcis_wbusy,
  // ---- the ports' host masters (hclk): valids per port, the rest shared
  input  logic                  hclk,
  input  logic                  hrst,
  output logic [NPORT-1:0]      p_awvalid,
  input  logic [NPORT-1:0]      p_awready,
  output logic [33:0]           p_awaddr,
  output logic [NPORT-1:0]      p_wvalid,
  input  logic [NPORT-1:0]      p_wready,
  output logic [255:0]          p_wdata,
  output logic [31:0]           p_wstrb,
  output logic                  p_wlast,
  input  logic [NPORT-1:0]      p_bvalid,
  output logic [NPORT-1:0]      p_arvalid,
  input  logic [NPORT-1:0]      p_arready,
  output logic [33:0]           p_araddr,
  input  logic [NPORT-1:0]      p_rvalid,
  input  logic [NPORT-1:0][255:0] p_rdata
);
  localparam int HW = $clog2(HQ), PW = $clog2(32), RW = $clog2(RQ);
  initial if (HQ < 2 || HQ != (1 << HW) || NPORT < 1 || NPORT > 32 || RQ < 4 || RQ != (1 << RW))
    $fatal(1, "otpu_f2_hbm_host: HQ, RQ powers of two, 1 <= NPORT <= 32");
  assign h_bresp = 2'b00;
  assign h_rresp = 2'b00;
  logic arst_q, hrst_q;
  always_ff @(posedge aclk) arst_q <= arst;
  always_ff @(posedge hclk) hrst_q <= hrst;

  // ================================================================== aclk: writes
  typedef struct packed { logic [15:0] id; logic [27:0] a; } aw_t;
  aw_t          awq [HQ];
  logic [HW:0]  aw_n;
  logic [HW-1:0] aw_h;
  logic [7:0]   w_beat;                 // the head burst's beats gone
  typedef struct packed { logic [15:0] id; logic [15:0] cum; } bq_t;
  bq_t          bq [HQ];
  logic [HW:0]  b_n;
  logic [HW-1:0] b_h;
  logic [15:0]  wpushed, wdone_a;       // beats crossed; beats the HBM answered (crossed back)
  logic         xw_ready;
  assign h_awready = !aw_n[HW];
  wire aw_tk = h_awvalid && h_awready;
  assign h_wready = aw_n != 0 && xw_ready && !b_n[HW];
  wire w_tk = h_wvalid && h_wready;
  wire [27:0] w_a = awq[aw_h].a + 28'(w_beat);
  // a burst's B once the count has passed its last beat (counts mod 2^16, a few in flight)
  wire [15:0] b_gap = wdone_a - bq[b_h].cum;
  assign h_bvalid = b_n != 0 && !b_gap[15];
  assign h_bid = bq[b_h].id;
  wire b_tk = h_bvalid && h_bready;
  assign pcis_wbusy = aw_n != 0 || b_n != 0 || h_awvalid || h_wvalid;
  always_ff @(posedge aclk) begin
    if (aw_tk) awq[HW'(aw_h + aw_n)] <= '{id: h_awid, a: h_awaddr[33:6]};
    if (w_tk && h_wlast) bq[HW'(b_h + b_n)] <= '{id: awq[aw_h].id, cum: wpushed + 1'b1};
    if (arst_q) begin
      aw_n <= '0; aw_h <= '0; w_beat <= '0; b_n <= '0; b_h <= '0; wpushed <= '0;
    end else begin
      aw_n <= aw_n + (HW + 1)'(aw_tk) - (HW + 1)'(w_tk && h_wlast);
      if (w_tk) begin
        wpushed <= wpushed + 1'b1;
        w_beat <= h_wlast ? '0 : w_beat + 1'b1;
        if (h_wlast) aw_h <= aw_h + 1'b1;
      end
      b_n <= b_n + (HW + 1)'(w_tk && h_wlast) - (HW + 1)'(b_tk);
      if (b_tk) b_h <= b_h + 1'b1;
    end
  end

  // ================================================================== aclk: reads
  typedef struct packed { logic [15:0] id; logic [27:0] a; logic [7:0] len; } ar_t;
  ar_t          arq [HQ];
  logic [HW:0]  ar_n;
  logic [HW-1:0] ar_h;
  logic [7:0]   r_beat;                 // the head AR's beats crossed
  typedef struct packed { logic [15:0] id; logic [7:0] len; } rq_t;
  rq_t          rq [HQ];
  logic [HW:0]  ri_n;
  logic [HW-1:0] ri_h;
  logic [7:0]   o_beat;                 // the head burst's beats given to the host
  logic [RW:0]  r_out;                  // read beats crossed, not yet given to the host
  logic         xr_ready, rr_v;
  logic [511:0] rr_d;
  assign h_arready = !ar_n[HW] && !pcis_wbusy;
  wire ar_tk = h_arvalid && h_arready;
  wire r_iss = ar_n != 0 && xr_ready && !r_out[RW] && !ri_n[HW];
  wire r_last = r_iss && r_beat == arq[ar_h].len;
  assign h_rvalid = rr_v && ri_n != 0;
  assign h_rdata = rr_d;
  assign h_rid = rq[ri_h].id;
  assign h_rlast = o_beat == rq[ri_h].len;
  wire r_tk = h_rvalid && h_rready;
  always_ff @(posedge aclk) begin
    if (ar_tk) arq[HW'(ar_h + ar_n)] <= '{id: h_arid, a: h_araddr[33:6], len: h_arlen};
    if (r_last) rq[HW'(ri_h + ri_n)] <= '{id: arq[ar_h].id, len: arq[ar_h].len};
    if (arst_q) begin
      ar_n <= '0; ar_h <= '0; r_beat <= '0; ri_n <= '0; ri_h <= '0; o_beat <= '0; r_out <= '0;
    end else begin
      ar_n <= ar_n + (HW + 1)'(ar_tk) - (HW + 1)'(r_last);
      if (r_iss) r_beat <= r_last ? '0 : r_beat + 1'b1;
      if (r_last) ar_h <= ar_h + 1'b1;
      ri_n <= ri_n + (HW + 1)'(r_last) - (HW + 1)'(r_tk && h_rlast);
      if (r_tk) o_beat <= h_rlast ? '0 : o_beat + 1'b1;
      if (r_tk && h_rlast) ri_h <= ri_h + 1'b1;
      r_out <= r_out + (RW + 1)'(r_iss) - (RW + 1)'(r_tk);
    end
  end

  // ================================================================== the crossings
  logic         xw_v, xw_pop, xr_v, xr_pop, rt_push;
  logic [27:0]  xw_a, xr_a;
  logic [511:0] xw_d, rt_d;
  logic [63:0]  xw_s;
  // (the W crossing: RWR, WT: its ready and the RAM's 604 write enables are flip-flops; the tail
  // takes the PCIS beat every cycle there is room, the take moves the pointer)
  otpu_afifo #(.W(604), .DEPTH(CQ), .RWR(1'b1), .WT(1'b1)) u_xw (
    .wclk(aclk), .wrst(arst_q), .wvalid(w_tk), .wready(xw_ready), .wdata({w_a, h_wstrb, h_wdata}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst_q), .rvalid(xw_v), .rready(xw_pop), .rdata({xw_a, xw_s, xw_d}));
  otpu_afifo #(.W(28), .DEPTH(CQ)) u_xr (
    .wclk(aclk), .wrst(arst_q), .wvalid(r_iss), .wready(xr_ready),
    .wdata(arq[ar_h].a + 28'(r_beat)), .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst_q), .rvalid(xr_v), .rready(xr_pop), .rdata(xr_a));
  // (r_out <= RQ beats in flight: the return FIFO never overflows)
  otpu_afifo #(.W(512), .DEPTH(RQ)) u_rt (
    .wclk(hclk), .wrst(hrst_q), .wvalid(rt_push), .wready(), .wdata(rt_d),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(aclk), .rrst(arst_q), .rvalid(rr_v), .rready(r_tk), .rdata(rr_d));

  // ================================================================== hclk: writes
  // The head beat in registers (hw_*: its port one-hot, or none past the last port), so the
  // ports' valids are flip-flops and the crossing's LUT RAM read and gray compare stay off the
  // ports' arbitration (the f278ff5f synthesis: -0.9 ns at 450 MHz without them)
  logic [1:0]  ws;                      // 0: AW, 1: W low, 2: W high (last)
  logic        hw_v, hw_null;
  logic [27:0] hw_a;
  logic [511:0] hw_d;
  logic [63:0] hw_s;
  logic [NPORT-1:0] hw_sel;
  assign p_awvalid = (hw_v && ws == 2'd0) ? hw_sel : '0;
  assign p_wvalid = (hw_v && ws != 2'd0) ? hw_sel : '0;
  assign p_awaddr = {hw_a, 6'd0};
  assign p_wdata = ws == 2'd2 ? hw_d[511:256] : hw_d[255:0];
  assign p_wstrb = ws == 2'd2 ? hw_s[63:32] : hw_s[31:0];
  assign p_wlast = ws == 2'd2;
  wire hw_aw = |(p_awvalid & p_awready);
  wire hw_w = |(p_wvalid & p_wready);
  wire hw_end = hw_v && (hw_null || (ws == 2'd2 && hw_w));
  assign xw_pop = xw_v && !hw_v;
  logic [15:0] btarget, bcnt, bgray;
  logic [5:0]  bsum;
  logic [NPORT-1:0] b_q;                // the ports' host B's, registered (off the HBM's pins)
  always_ff @(posedge hclk) b_q <= hrst_q ? '0 : p_bvalid;
  always_comb begin
    bsum = '0;
    for (int p = 0; p < NPORT; p++) bsum += 6'(b_q[p]);
  end
  always_ff @(posedge hclk) begin
    if (xw_pop) begin
      hw_a <= xw_a; hw_d <= xw_d; hw_s <= xw_s;
      hw_null <= 32'(xw_a[27:23]) >= NPORT;
      for (int p = 0; p < NPORT; p++) hw_sel[p] <= xw_a[27:23] == 5'(p);
    end
    if (hrst_q) begin
      ws <= '0; hw_v <= 1'b0; btarget <= '0; bcnt <= '0; bgray <= '0;
    end else begin
      hw_v <= xw_pop || (hw_v && !hw_end);
      if (hw_v && !hw_null) begin
        if (ws == 2'd0 && hw_aw) ws <= 2'd1;
        else if (ws != 2'd0 && hw_w) ws <= ws == 2'd2 ? 2'd0 : 2'd2;
      end
      // a B per beat (the ports' host B's, bready 1); a dropped beat counts at once
      btarget <= btarget + 16'(bsum) + 16'(hw_v && hw_null);
      if (bcnt != btarget) bcnt <= bcnt + 1'b1;
      bgray <= bcnt ^ (bcnt >> 1);
    end
  end

  // ================================================================== hclk: reads
  // the head read in registers (hr_*) as the head write; the port's R beats registered (rq_*)
  // before they pair and cross
  logic [PW-1:0] cur;                   // the port the reads in flight are on
  logic [NPORT-1:0] cur_sel;
  logic [RW:0]   h_out;                 // reads issued to it, not yet back (or in rq)
  logic          r_hi;
  logic [255:0]  r_lo;
  logic          hr_v, hr_null;
  logic [27:0]   hr_a;
  logic [PW-1:0] hr_port;
  logic [NPORT-1:0] hr_sel;
  logic          rq_v;
  logic [255:0]  rq_d;
  // the head read may go once its port is the reads' in flight or none is in flight; its AR
  // valid is a register (ar_sel, set the cycle after: the 89bbb6bc synthesis had cur -> the
  // compare -> the ports' grants -> back into cur at -0.51 ns)
  logic [NPORT-1:0] ar_sel;
  wire r_ok = hr_v && (h_out == 0 || hr_port == cur);
  assign p_arvalid = ar_sel;
  assign p_araddr = {hr_a, 6'd0};
  wire hr_ar = |(ar_sel & p_arready);
  wire r_zero = r_ok && hr_null && h_out == 0;        // a read past the last port: zeros
  assign xr_pop = xr_v && !hr_v;
  assign rt_push = (rq_v && r_hi) || r_zero;
  assign rt_d = r_zero ? '0 : {rq_d, r_lo};
  always_ff @(posedge hclk) begin
    if (xr_pop) begin
      hr_a <= xr_a; hr_port <= xr_a[27:23];
      hr_null <= 32'(xr_a[27:23]) >= NPORT;
      for (int p = 0; p < NPORT; p++) hr_sel[p] <= xr_a[27:23] == 5'(p);
    end
    rq_d <= '0;
    for (int p = 0; p < NPORT; p++) if (cur_sel[p]) rq_d <= p_rdata[p];
    if (hrst_q) begin
      cur <= '0; cur_sel <= '0; h_out <= '0; r_hi <= 1'b0; hr_v <= 1'b0; rq_v <= 1'b0;
      ar_sel <= '0;
    end else begin
      hr_v <= xr_pop || (hr_v && !hr_ar && !r_zero);
      // shown from the cycle after it may go until taken (h_out counts it from the take)
      ar_sel <= hr_ar ? '0 : (ar_sel != '0) ? ar_sel : (r_ok && !hr_null) ? hr_sel : '0;
      if (hr_ar) begin cur <= hr_port; cur_sel <= hr_sel; end
      // the port's R (one ID, one port: in order), a cycle late
      rq_v <= |(p_rvalid & cur_sel) && h_out != 0;
      h_out <= h_out + (RW + 1)'(hr_ar) - (RW + 1)'(rq_v && r_hi);
      if (rq_v) begin r_hi <= !r_hi; if (!r_hi) r_lo <= rq_d; end
    end
  end

  // ================================================================== the write count, into aclk
  (* ASYNC_REG = "TRUE" *) logic [15:0] bg1, bg2;
  always_ff @(posedge aclk) begin
    if (arst_q) begin bg1 <= '0; bg2 <= '0; wdone_a <= '0; end
    else begin
      bg1 <= bgray; bg2 <= bg1;
      for (int i = 0; i < 16; i++) wdone_a[i] <= ^(bg2 >> i);
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge hclk) if (!hrst_q) begin
    if (|(p_rvalid & ~(NPORT'(1) << cur))) $fatal(1, "otpu_f2_hbm_host: R from a port not read");
    if (|(p_rvalid & cur_sel) && h_out == 0) $fatal(1, "otpu_f2_hbm_host: R with no read in flight");
  end
`endif
endmodule
