// Dual-clock FIFO of the F2 HBM engine (otpu_hbm_ch; docs/superpowers/specs/2026-10-20-opentpu-
// design-v5.md section 3): gray-coded pointers, each synchronized by two flip-flops into the
// other clock, a simple dual-port RAM written in wclk and read in rclk into one output register
// (first-word fall-through: rdata is the head while rvalid is high; a pop lets the next word
// in behind it the same cycle, so a consumer that keeps up takes a word a cycle). STYLE
// "block" makes the RAM a block RAM whose read enable is the fetch and whose output register
// is rdata (the engine's 512-bit read beats, 256 deep); "distributed" a LUT RAM (the commands,
// write beats and B tokens). wready is a flip-flop: the room after this cycle's write against
// the read pointer as last synchronized (never optimistic), less SLACK entries, for a writer
// that sees it late (the SLR chain: the engine's native side, as otpu_native_axi's XL); with
// TAKE the write is a take already made (pushed whatever wready says, the SLACK holds it).
// wcount / rcount: the occupancy each side sees (stale on the safe side: wcount high, rcount
// low). Every pointer step is from a precomputed register (wbin_n, rbin_n: no adder on the
// pop or push path), the not-empty flag is registered a cycle ahead (ne_q): the read side's
// fetch is one LUT level from flip-flops, the write side's wready one level deep after the
// synchronizer's subtract.
// OTPU_XPM: the same ports on AMD's xpm_fifo_async (the production drop-in; Verilator does not
// see it). DEPTH a power of two >= 4; both resets held together at start.
module otpu_hbm_afifo #(
  parameter int W = 32,
  parameter int DEPTH = 16,
  parameter int SLACK = 0,             // wready drops this many entries before full
  parameter bit TAKE = 1'b0,           // wvalid is a take: pushed regardless of wready
  parameter string STYLE = "distributed"
) (
  input  logic               wclk,
  input  logic               wrst,
  input  logic               wvalid,
  output logic               wready,
  input  logic [W-1:0]       wdata,
  output logic [$clog2(DEPTH):0] wcount,
  input  logic               rclk,
  input  logic               rrst,
  output logic               rvalid,
  input  logic               rready,
  output logic [W-1:0]       rdata,
  output logic [$clog2(DEPTH):0] rcount
);
  localparam int AW = $clog2(DEPTH);
  initial if (DEPTH != (1 << AW) || DEPTH < 4) $fatal(1, "otpu_hbm_afifo: DEPTH a power of two >= 4");
  initial if (SLACK < 0 || SLACK > DEPTH - 2) $fatal(1, "otpu_hbm_afifo: SLACK leaves two entries");

  function automatic logic [AW:0] gray(input logic [AW:0] b);
    return b ^ (b >> 1);
  endfunction
  function automatic logic [AW:0] g2b(input logic [AW:0] g);     // a prefix XOR, a tree per bit
    logic [AW:0] b;
    for (int i = 0; i <= AW; i++) b[i] = ^(g >> i);
    return b;
  endfunction

`ifdef OTPU_XPM
  logic full, empty, prog_full;
  xpm_fifo_async #(
    .FIFO_MEMORY_TYPE(STYLE), .FIFO_WRITE_DEPTH(DEPTH), .WRITE_DATA_WIDTH(W),
    .READ_DATA_WIDTH(W), .READ_MODE("fwft"), .FIFO_READ_LATENCY(0),
    .PROG_FULL_THRESH(DEPTH - SLACK - 1), .USE_ADV_FEATURES("0707"),
    .WR_DATA_COUNT_WIDTH(AW + 1), .RD_DATA_COUNT_WIDTH(AW + 1), .CDC_SYNC_STAGES(2)) u_xpm (
    .rst(wrst), .wr_clk(wclk), .wr_en(TAKE ? wvalid : wvalid && wready), .din(wdata),
    .full(full), .prog_full(prog_full), .wr_data_count(wcount),
    .rd_clk(rclk), .rd_en(rvalid && rready), .dout(rdata), .empty(empty),
    .rd_data_count(rcount), .sleep(1'b0), .injectsbiterr(1'b0), .injectdbiterr(1'b0),
    .wr_rst_busy(), .rd_rst_busy(), .almost_full(), .almost_empty(), .wr_ack(), .overflow(),
    .underflow(), .data_valid(), .prog_empty(), .sbiterr(), .dbiterr());
  assign wready = !prog_full;
  assign rvalid = !empty;
`else
  // ---- the RAM
  logic [W-1:0] mem [DEPTH];
  // ---- write side
  logic [AW:0] wbin, wgray, wbin_n, wgray_n;
  (* ASYNC_REG = "TRUE" *) logic [AW:0] rgray_w1, rgray_w2;
  logic [AW:0] rbin_w;
  logic        wrdy;
  wire         wput = TAKE ? wvalid : (wvalid && wready);
  assign rbin_w = g2b(rgray_w2);
  assign wcount = wbin - rbin_w;
  assign wready = wrdy;
  always_ff @(posedge wclk) begin
    if (wrst) begin
      wbin <= '0; wgray <= '0; wbin_n <= (AW + 1)'(1); wgray_n <= gray((AW + 1)'(1));
      rgray_w1 <= '0; rgray_w2 <= '0; wrdy <= 1'b0;
    end else begin
      rgray_w1 <= rgray; rgray_w2 <= rgray_w1;
      if (wput) begin
        wbin <= wbin_n; wgray <= wgray_n;
        wbin_n <= wbin_n + 1'b1; wgray_n <= gray(wbin_n + 1'b1);
      end
      wrdy <= ((wput ? wbin_n : wbin) - rbin_w) < (AW + 1)'(DEPTH - SLACK);
    end
  end
  if (STYLE == "block") begin : g_blk
    (* ram_style = "block" *) logic [W-1:0] memb [DEPTH];
    always_ff @(posedge wclk) if (wput) memb[wbin[AW-1:0]] <= wdata;
    // (the read below)
  end
  (* ram_style = "distributed" *) logic [W-1:0] memd [DEPTH];
  if (STYLE != "block") begin : g_dst
    always_ff @(posedge wclk) if (wput) memd[wbin[AW-1:0]] <= wdata;
  end
  // ---- read side: the output register (ov, od), the not-empty flag a cycle ahead
  logic [AW:0] rbin, rgray, rbin_n, rgray_n;
  (* ASYNC_REG = "TRUE" *) logic [AW:0] wgray_r1, wgray_r2;
  logic        ne_q, ov;
  logic [W-1:0] od;
  wire fetch = ne_q && (!ov || rready);
  assign rvalid = ov;
  assign rdata = od;
  assign rcount = (g2b(wgray_r2) - rbin) + (AW + 1)'(ov);
  always_ff @(posedge rclk) begin
    if (rrst) begin
      rbin <= '0; rgray <= '0; rbin_n <= (AW + 1)'(1); rgray_n <= gray((AW + 1)'(1));
      wgray_r1 <= '0; wgray_r2 <= '0; ne_q <= 1'b0; ov <= 1'b0;
    end else begin
      wgray_r1 <= wgray; wgray_r2 <= wgray_r1;
      // the flag for next cycle: the pointer after this fetch against the write pointer as it
      // will be synchronized then (wgray_r1 is wgray_r2's next value)
      ne_q <= (fetch ? rgray_n : rgray) != wgray_r1;
      if (fetch) begin
        rbin <= rbin_n; rgray <= rgray_n;
        rbin_n <= rbin_n + 1'b1; rgray_n <= gray(rbin_n + 1'b1);
        ov <= 1'b1;
      end else if (ov && rready) ov <= 1'b0;
    end
  end
  if (STYLE == "block") begin : g_rblk
    always_ff @(posedge rclk) if (fetch) od <= g_blk.memb[rbin[AW-1:0]];
  end else begin : g_rdst
    always_ff @(posedge rclk) if (fetch) od <= memd[rbin[AW-1:0]];
  end
`ifndef SYNTHESIS
  always @(posedge wclk) if (!wrst && wput && wcount == (AW + 1)'(DEPTH))
    $fatal(1, "otpu_hbm_afifo: overflow");
  always @(posedge rclk) if (!rrst && fetch && rgray == wgray_r2)
    $fatal(1, "otpu_hbm_afifo: underflow");
`endif
`endif
endmodule
