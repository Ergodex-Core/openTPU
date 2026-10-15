// The host's PCIS (AppPF BAR4: AXI4, 512 bits, 64-byte beats, its own IDs; aclk = clk_main_a0,
// 250 MHz) onto the HBM ports' host masters (otpu_f2_hbm_port master 1; hclk = 450 MHz): BAR4
// offset = HBM byte address, so port p (pseudo-channel p, bytes p x 512 MiB ..) holds slice s's
// channel c at p = NCH s + c (opentpu/host/f2.py bases, channel_map). docs/f2_hbm.md.
//
// Writes: every 64-byte W beat crosses into hclk with its address (afifo), then goes to the port
// of its address as one AXI3 burst of two 32-byte beats (AW, then W low and high halves): one
// beat at a time, in order (host traffic is the weights' load and a run's inputs: ~2 GB/s at most
// through BAR4 write combining, a beat per ~4 hclk cycles is 7 GB/s). The host's B for a burst
// goes back once the write count (every port's B for the host, crossed back gray-coded) reaches
// the burst's last beat. Beats to a port past the last (NPORT) are dropped (counted done).
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
    if (arst) begin
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
    if (arst) begin
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
  otpu_afifo #(.W(604), .DEPTH(CQ)) u_xw (
    .wclk(aclk), .wrst(arst), .wvalid(w_tk), .wready(xw_ready), .wdata({w_a, h_wstrb, h_wdata}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst), .rvalid(xw_v), .rready(xw_pop), .rdata({xw_a, xw_s, xw_d}));
  otpu_afifo #(.W(28), .DEPTH(CQ)) u_xr (
    .wclk(aclk), .wrst(arst), .wvalid(r_iss), .wready(xr_ready),
    .wdata(arq[ar_h].a + 28'(r_beat)), .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst), .rvalid(xr_v), .rready(xr_pop), .rdata(xr_a));
  // (r_out <= RQ beats in flight: the return FIFO never overflows)
  otpu_afifo #(.W(512), .DEPTH(RQ)) u_rt (
    .wclk(hclk), .wrst(hrst), .wvalid(rt_push), .wready(), .wdata(rt_d),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(aclk), .rrst(arst), .rvalid(rr_v), .rready(r_tk), .rdata(rr_d));

  // ================================================================== hclk: writes
  logic [1:0]  ws;                      // 0: AW, 1: W low, 2: W high (last)
  wire [PW-1:0] w_port = xw_a[27:23];
  wire w_null = 32'(w_port) >= NPORT;
  always_comb begin
    p_awvalid = '0; p_wvalid = '0;
    if (xw_v && !w_null) begin
      if (ws == 2'd0) p_awvalid[w_port] = 1'b1;
      else p_wvalid[w_port] = 1'b1;
    end
  end
  assign p_awaddr = {xw_a, 6'd0};
  assign p_wdata = ws == 2'd2 ? xw_d[511:256] : xw_d[255:0];
  assign p_wstrb = ws == 2'd2 ? xw_s[63:32] : xw_s[31:0];
  assign p_wlast = ws == 2'd2;
  wire hw_aw = |(p_awvalid & p_awready);
  wire hw_w = |(p_wvalid & p_wready);
  assign xw_pop = xw_v && (w_null || (ws == 2'd2 && hw_w));
  logic [15:0] btarget, bcnt, bgray;
  logic [5:0]  bsum;
  always_comb begin
    bsum = '0;
    for (int p = 0; p < NPORT; p++) bsum += 6'(p_bvalid[p]);
  end
  always_ff @(posedge hclk) begin
    if (hrst) begin
      ws <= '0; btarget <= '0; bcnt <= '0; bgray <= '0;
    end else begin
      if (xw_v && !w_null) begin
        if (ws == 2'd0 && hw_aw) ws <= 2'd1;
        else if (ws != 2'd0 && hw_w) ws <= ws == 2'd2 ? 2'd0 : 2'd2;
      end
      // a B per beat (the ports' host B's, bready 1); a dropped beat counts at once
      btarget <= btarget + 16'(bsum) + 16'(xw_v && w_null);
      if (bcnt != btarget) bcnt <= bcnt + 1'b1;
      bgray <= bcnt ^ (bcnt >> 1);
    end
  end

  // ================================================================== hclk: reads
  logic [PW-1:0] cur;                   // the port the reads in flight are on
  logic [RW:0]   h_out;                 // reads issued to it, not yet back
  logic          r_hi;
  logic [255:0]  r_lo;
  wire [PW-1:0] r_port = xr_a[27:23];
  wire r_null = 32'(r_port) >= NPORT;
  wire r_ok = xr_v && (h_out == 0 || r_port == cur);
  always_comb begin
    p_arvalid = '0;
    if (r_ok && !r_null) p_arvalid[r_port] = 1'b1;
  end
  assign p_araddr = {xr_a, 6'd0};
  wire hr_ar = |(p_arvalid & p_arready);
  // the port's R: two beats a host beat (in order: one ID, one port)
  wire hr_r = p_rvalid[cur] && h_out != 0;
  assign xr_pop = hr_ar || (r_ok && r_null && h_out == 0);
  assign rt_push = (hr_r && r_hi) || (r_ok && r_null && h_out == 0);
  assign rt_d = (r_ok && r_null && h_out == 0) ? '0 : {p_rdata[cur], r_lo};
  always_ff @(posedge hclk) begin
    if (hrst) begin
      cur <= '0; h_out <= '0; r_hi <= 1'b0;
    end else begin
      if (hr_ar) cur <= r_port;
      h_out <= h_out + (RW + 1)'(hr_ar) - (RW + 1)'(hr_r && r_hi);
      if (hr_r) begin r_hi <= !r_hi; if (!r_hi) r_lo <= p_rdata[cur]; end
    end
  end

  // ================================================================== the write count, into aclk
  (* ASYNC_REG = "TRUE" *) logic [15:0] bg1, bg2;
  always_ff @(posedge aclk) begin
    if (arst) begin bg1 <= '0; bg2 <= '0; wdone_a <= '0; end
    else begin
      bg1 <= bgray; bg2 <= bg1;
      for (int i = 0; i < 16; i++) wdone_a[i] <= ^(bg2 >> i);
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge hclk) if (!hrst) begin
    if (|(p_rvalid & ~(NPORT'(1) << cur))) $fatal(1, "otpu_f2_hbm_host: R from a port not read");
    if (hr_r && h_out == 0) $fatal(1, "otpu_f2_hbm_host: R with no read in flight");
  end
`endif
endmodule
