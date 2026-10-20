// The engine's read-return path (otpu_hbm_ch; docs/full_hbm.md "The engines' memories"): the
// HBM port's 256-bit R beats (hclk, from the engine's R register) into a synchronous FIFO in
// UltraRAM (4 x URAM288 a port: 256 bits wide, UDEPTH deep; both of a URAM's ports run on one
// clock, so it cannot cross by itself), read out at a beat a cycle into pairs (two AXI beats =
// one 512-bit native beat) and pushed into a small distributed-RAM asynchronous FIFO
// (otpu_hbm_afifo, AQD entries) to the core clock, where the native port takes a beat a cycle.
// No block RAM: SLR0 holds slice 0's RAMB36 and the HBM ports' bottom row has few RAM columns
// (the v5 ImplCL rounds 1-3 stopped in placement on the engines' 7 RAMB36 each, 224 in all);
// a URAM read a cycle (1.125 native beats a core cycle) is the AXI side's rate, so the path
// adds no limit of its own. The room the engine reserves is counted here in native beats:
// `count` = the beats held anywhere in the path (the URAM's entries, the read pipeline's, a
// pair's first beat, the crossing FIFO's input register and its entries as its write side
// sees them: stale high), a cycle old; the engine keeps count + its beats in flight under
// RDEPTH less a margin (otpu_hbm_ch beat_room), so UDEPTH >= 2 RDEPTH never overflows and
// the crossing FIFO is fed only while it has room for every pair in the pipeline (aq_room).
// Every register is in hclk but the crossing FIFO's read side; the input register and the
// URAM's own two output registers let the URAMs sit a clock region away from the port's
// engine (the floorplan boxes the engine's logic by its port and lets this path float with
// its URAMs). Latency, hclk cycles from the R register to the crossing FIFO: +5 over a single
// asynchronous FIFO (the input register, the write, the read a cycle after it is counted, the
// URAM's two output registers, the crossing FIFO's input register: the receipt model's LAT_UP
// 7, was 5).
module otpu_hbm_rpath #(
  parameter int RDEPTH = 256,          // native beats the engine may hold here or have in flight
  parameter int UDEPTH = 4096,         // the URAM FIFO's entries (AXI beats): a URAM288's depth
  parameter int AQD = 32               // the crossing FIFO's entries (native beats)
) (
  input  logic          hclk,
  input  logic          hrst,
  input  logic          r_valid,       // an R beat, registered by the engine
  input  logic [255:0]  r_data,
  output logic [$clog2(RDEPTH):0] count,   // native beats held in the path, a cycle old
  input  logic          clk,
  input  logic          rst,
  output logic          n_rvalid,
  output logic [511:0]  n_rdata
);
  localparam int UW = $clog2(UDEPTH);
  localparam int AW = $clog2(AQD);
  localparam int RW = $clog2(RDEPTH + 1);
  initial begin
    if (UDEPTH != (1 << UW) || UDEPTH < 2 * RDEPTH + 16) $fatal(1, "otpu_hbm_rpath: UDEPTH a power of two >= 2 RDEPTH + 16");
    if (AQD != (1 << AW) || AQD < 16) $fatal(1, "otpu_hbm_rpath: AQD a power of two >= 16");
    if (RDEPTH < 16) $fatal(1, "otpu_hbm_rpath: RDEPTH >= 16");
  end
  // ---- the input register (by the URAM; the engine's R register is by the port)
  logic         i_v;
  logic [255:0] i_d;
  always_ff @(posedge hclk) begin
    i_v <= r_valid && !hrst;
    i_d <= r_data;
  end
  // ---- the URAM FIFO: an entry written at wp the cycle after it is registered, counted at
  // the write, read at rp (a cycle after at the earliest: never the address being written)
  (* ram_style = "ultra" *) logic [255:0] mem [UDEPTH];
  logic [UW-1:0] wp, rp;
  logic [UW:0]   cnt;                   // entries held
  logic          ne;                    // cnt != 0, for this cycle (from the next count)
  logic          aq_room;               // the crossing FIFO has room for the pipeline's pairs
  logic          v1, v2;                // a read in the URAM's output register, in its pipeline register
  logic [255:0]  q1, q2;
  wire           issue = ne && aq_room;
  wire [UW:0]    cnt_n = cnt + (UW + 1)'(i_v) - (UW + 1)'(issue);
  always_ff @(posedge hclk) if (i_v) mem[wp] <= i_d;
  always_ff @(posedge hclk) begin       // the URAM's output register and its pipeline register
    q1 <= mem[rp];
    q2 <= q1;
  end
  // ---- the pairs: the first beat held, the second pushes both through a register
  logic          tog;                   // the next beat is a pair's second
  logic [255:0]  lo;
  logic          aq_wv;
  logic [511:0]  aq_wd;
  logic [AW:0]   aq_wcount;
  always_ff @(posedge hclk) begin
    if (hrst) begin
      wp <= '0; rp <= '0; cnt <= '0; ne <= 1'b0; aq_room <= 1'b0; v1 <= 1'b0; v2 <= 1'b0;
      tog <= 1'b0; aq_wv <= 1'b0;
    end else begin
      if (i_v) wp <= wp + 1'b1;
      if (issue) rp <= rp + 1'b1;
      cnt <= cnt_n;
      ne <= cnt_n != '0;
      // pairs not yet in aq_wcount: the output registers (one pair), lo, aq_wd (one each), and
      // one more read while this flag is a cycle old: at most four beyond the count
      aq_room <= aq_wcount < (AW + 1)'(AQD - 8);
      v1 <= issue; v2 <= v1;
      if (v2) begin
        tog <= !tog;
        if (!tog) lo <= q2;
      end
      aq_wv <= v2 && tog;
      if (v2 && tog) aq_wd <= {q2, lo};
    end
  end
  otpu_hbm_afifo #(.W(512), .DEPTH(AQD), .SLACK(0), .TAKE(1'b1), .STYLE("distributed")) u_rq (
    .wclk(hclk), .wrst(hrst), .wvalid(aq_wv), .wready(), .wdata(aq_wd), .wcount(aq_wcount),
    .rclk(clk), .rrst(rst), .rvalid(n_rvalid), .rready(1'b1), .rdata(n_rdata), .rcount());
  // ---- the beats held (native: AXI beats rounded up), from this cycle's registers: the
  // URAM's entries, the output registers', lo, the input register's pair, the crossing
  // FIFO's entries as its write side counts them (never low)
  wire [UW+1:0] held = (UW + 2)'(cnt) + (UW + 2)'(v1) + (UW + 2)'(v2) + (UW + 2)'(tog) +
                       (UW + 2)'({aq_wv, 1'b0}) + (UW + 2)'({aq_wcount, 1'b0}) + (UW + 2)'(1);
  always_ff @(posedge hclk)
    if (hrst) count <= '0;
    else count <= RW'(held >> 1);
`ifndef SYNTHESIS
  always @(posedge hclk) if (!hrst) begin
    if (i_v && cnt == (UW + 1)'(UDEPTH)) $fatal(1, "otpu_hbm_rpath: the URAM FIFO overflowed");
    if (issue && cnt == '0) $fatal(1, "otpu_hbm_rpath: a read of an empty URAM FIFO");
    if ((held >> 1) > (UW + 2)'(RDEPTH + 8)) $fatal(1, "otpu_hbm_rpath: %0d native beats held, over RDEPTH %0d (the engine's reservation)", held >> 1, RDEPTH);
  end
`endif
endmodule
