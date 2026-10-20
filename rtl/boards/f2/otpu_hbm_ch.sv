// One HBM pseudo-channel's engine (docs/superpowers/specs/2026-10-20-opentpu-design-v5.md
// section 3; the shape of AWS's cl_mem_perf kernel): the slice's native channel port (one
// command per 64-byte beat, write beats in their own stream, read beats back in command order
// with no back-pressure, n_wdone counting completed writes; docs/f2_hbm.md "The native port")
// at the core clock, onto the channel's AXI3 256-bit port at hclk (450 MHz), with nothing at
// hclk but a command FIFO popped into a registered burst generator, the read data into an
// asynchronous FIFO, and counters. Replaces otpu_native_axi + a SmartConnect per channel.
//
// Core clock (SLR0, behind the SLR chain: n_cvalid / n_wvalid are takes made against a ready
// seen XL / 2 cycles earlier, the FIFOs keep XL entries of room below their readies):
//   - the command's `cont` bit: it follows the previous command's beat (last + 1), both reads,
//     and is not the first of an aligned BURST-beat group, so a burst never crosses a 512 B
//     group (never 4 KiB) and the 450 MHz side compares no addresses;
//   - commands {cont, we, beat} into a 32-entry FIFO, write beats {mask, data} into a 16-entry
//     one; read beats out of the read path (otpu_hbm_rpath: a URAM FIFO at hclk and a 32-entry
//     crossing FIFO; taken every cycle: the engine holds or has in flight at most RDEPTH beats,
//     the adapter reserved the room, RD + AD + WQD <= RDEPTH); a B token per completed write (n_wdone++,
//     n_wdp a pulse for the chain); n_err sticky (an hclk error level, or a beat outside the
//     window: masked into it, as otpu_native_axi).
// hclk, every state element a register, at most three LUT levels on any path:
//   - the open read burst {beat, len}: the head extends it while `cont` and len < BURST, else
//     it closes into the AR register (ARADDR = BASE + beat x 64, ARLEN 2 len - 1, 32-byte beats,
//     INCR, ID) and the head starts the next; an idle timer (IDLE cycles with an empty queue)
//     closes a burst the stream stopped behind; a closed burst waits for the AR register
//     (ob_cl) while the queue holds. ARVALID is gated by the outstanding count (OT bursts: the
//     flag only drops as this AR fires) and a read is only started while the read FIFO has
//     room for its beats (beat_room, a registered margin).
//   - writes: AW (AWLEN 1: two 32-byte beats) and W's two beats with the mask's halves as
//     strobes, in command order, one write in flight through the registers (the next pops
//     once its AW and both W beats are taken); B always taken, counted.
//   - direction barriers (ORDER): a read starts only while no write lacks its B, a write only
//     while no read beat is outstanding, so the native order (a read after a write to its beat
//     returns it) holds whatever the HBM reorders between its channels.
//   - R always ready: the 256-bit beats registered into the read path, which assembles the
//     pairs (one 512-bit native beat) behind its URAM FIFO; RRESP / BRESP and RLAST (against a
//     length queue in AR order) into the error level.
module otpu_hbm_ch #(
  parameter logic [33:0] BASE_ADDR = 34'd0,
  parameter int CH_LOG2 = 29,          // the window's bytes, log2 (BASE_ADDR aligned to it)
  parameter int XL = 6,                // FIFO entries kept below the readies (2 NXR)
  parameter int BURST = 8,             // native beats a read burst at most (8: 512 B, AXI3's 16)
  parameter int IDLE = 2,              // hclk cycles a burst waits with an empty queue
  parameter int OT = 64,               // read bursts in flight at most
  parameter int RDEPTH = 256,          // read beats the read FIFO holds
  parameter logic [5:0] ID = 6'd0,
  parameter bit ORDER = 1'b1,          // the direction barriers (0: tests only)
  parameter int CQD = 32,              // the command FIFO
  parameter int WQD = 16               // the write-beat FIFO
) (
  input  logic          clk,
  input  logic          rst,
  // ---- the native port (core clock)
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
  output logic          n_wdp,        // a write completed (n_wdone stepped) this cycle
  output logic          n_err,
  // ---- the HBM port (hclk): AXI3, 256 bits
  input  logic          hclk,
  input  logic          hrst,
  output logic [5:0]    m_awid,
  output logic [33:0]   m_awaddr,
  output logic [3:0]    m_awlen,
  output logic [2:0]    m_awsize,
  output logic [1:0]    m_awburst,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [5:0]    m_wid,
  output logic [255:0]  m_wdata,
  output logic [31:0]   m_wstrb,
  output logic          m_wlast,
  output logic          m_wvalid,
  input  logic          m_wready,
  input  logic [5:0]    m_bid,
  input  logic [1:0]    m_bresp,
  input  logic          m_bvalid,
  output logic          m_bready,
  output logic [5:0]    m_arid,
  output logic [33:0]   m_araddr,
  output logic [3:0]    m_arlen,
  output logic [2:0]    m_arsize,
  output logic [1:0]    m_arburst,
  output logic          m_arvalid,
  input  logic          m_arready,
  input  logic [5:0]    m_rid,
  input  logic [255:0]  m_rdata,
  input  logic [1:0]    m_rresp,
  input  logic          m_rlast,
  input  logic          m_rvalid,
  output logic          m_rready
);
  localparam int AB = (CH_LOG2 - 6 < 25) ? CH_LOG2 - 6 : 25;   // beat bits inside the window
  localparam logic [33:0] WMASK = (34'd1 << CH_LOG2) - 34'd1;
  localparam int BW = (BURST > 1) ? $clog2(BURST) : 1;
  localparam int IW = (IDLE > 1) ? $clog2(IDLE) : 1;
  localparam int OW = $clog2(OT + 1);
  localparam int RW = $clog2(RDEPTH + 1);
  initial begin
    if (BURST < 1 || BURST > 8 || BURST != (1 << $clog2(BURST))) $fatal(1, "otpu_hbm_ch: BURST 1, 2, 4 or 8");
    if (IDLE < 1) $fatal(1, "otpu_hbm_ch: IDLE >= 1");
    if (OT < 1 || OT > 256) $fatal(1, "otpu_hbm_ch: OT 1..256");
    if (CH_LOG2 < 12 || CH_LOG2 > 31) $fatal(1, "otpu_hbm_ch: CH_LOG2 12..31");
    if ((BASE_ADDR & WMASK) != 0) $fatal(1, "otpu_hbm_ch: BASE_ADDR aligned to the window");
    if (XL > CQD - 4 || XL > WQD - 4) $fatal(1, "otpu_hbm_ch: XL leaves four entries");
  end
  function automatic logic [33:0] axaddr(input logic [24:0] a);
    return BASE_ADDR | ((34'(a) << 6) & WMASK);
  endfunction

  // ================================================================ core clock: the front
  // the cont bit against the last command pushed (a register of its beat + 1)
  logic        last_v, last_we;
  logic [24:0] next_beat;
  wire         pushc = n_cvalid && (XL > 0 || n_cready);
  wire         pushw = n_wvalid && (XL > 0 || n_wready);
  wire         cont  = (BURST > 1) && !n_cwe && last_v && !last_we && (n_caddr == next_beat) &&
                       (n_caddr[BW-1:0] != '0);
  wire         outside = AB < 25 && (n_caddr >> AB) != 0;
  always_ff @(posedge clk) begin
    if (rst) begin
      last_v <= 1'b0; last_we <= 1'b0; next_beat <= '0;
    end else if (pushc) begin
      last_v <= 1'b1; last_we <= n_cwe; next_beat <= n_caddr + 1'b1;
    end
  end
  // the command and write-beat FIFOs (core -> hclk)
  logic        cq_rvalid, cq_pop, h_cont, h_we;
  logic [24:0] h_beat;
  logic [$clog2(CQD):0] cq_wcount;
  otpu_hbm_afifo #(.W(27), .DEPTH(CQD), .SLACK(XL), .TAKE(1'b1), .STYLE("distributed")) u_cq (
    .wclk(clk), .wrst(rst), .wvalid(pushc), .wready(n_cready), .wdata({cont, n_cwe, n_caddr}),
    .wcount(cq_wcount),
    .rclk(hclk), .rrst(hrst), .rvalid(cq_rvalid), .rready(cq_pop), .rdata({h_cont, h_we, h_beat}),
    .rcount());
  logic         wq_rvalid, wq_pop;
  logic [511:0] h_data;
  logic [63:0]  h_mask;
  logic [$clog2(WQD):0] wq_wcount;
  otpu_hbm_afifo #(.W(576), .DEPTH(WQD), .SLACK(XL), .TAKE(1'b1), .STYLE("distributed")) u_wq (
    .wclk(clk), .wrst(rst), .wvalid(pushw), .wready(n_wready), .wdata({n_wmask, n_wdata}),
    .wcount(wq_wcount),
    .rclk(hclk), .rrst(hrst), .rvalid(wq_rvalid), .rready(wq_pop), .rdata({h_mask, h_data}),
    .rcount());
  // (the read beats: the read path u_rp below, hclk -> core) and the B tokens
  logic bq_wvalid, bq_rvalid;
  otpu_hbm_afifo #(.W(1), .DEPTH(32), .SLACK(0), .TAKE(1'b1), .STYLE("distributed")) u_bq (
    .wclk(hclk), .wrst(hrst), .wvalid(bq_wvalid), .wready(), .wdata(1'b1), .wcount(),
    .rclk(clk), .rrst(rst), .rvalid(bq_rvalid), .rready(1'b1), .rdata(), .rcount());
  // n_wdone, n_err (the hclk error level through two flip-flops)
  logic herr;
  (* ASYNC_REG = "TRUE" *) logic herr_s1, herr_s2;
  always_ff @(posedge clk) begin
    herr_s1 <= herr; herr_s2 <= herr_s1;
    if (rst) begin
      n_wdone <= '0; n_wdp <= 1'b0; n_err <= 1'b0;
    end else begin
      n_wdp <= bq_rvalid;
      if (bq_rvalid) n_wdone <= n_wdone + 1'b1;
      if (herr_s2 || (pushc && outside)) n_err <= 1'b1;
    end
  end

  // ================================================================ hclk: the generator
  // the open burst, the AR register, the write in flight, the counters and their flags
  logic         ob_v, ob_cl, ob_full;
  logic         ob_open;                 // ob_v && !ob_cl && !ob_full (one flag: the head's extend)
  logic [24:0]  ob_beat;
  logic [3:0]   ob_len;
  logic [2:0]   ob_lm1;                  // len - 1: the AR's ARLEN is {len - 1, 1}
  logic [IW-1:0] idle;
  logic         idle_to;
  logic         ar_v;
  logic [OW-1:0] rd_ot;                  // bursts issued, their last beat not yet back
  logic         ot_room;                 // rd_ot < OT
  logic [RW-1:0] rd_res;                 // read beats popped, not yet pushed into the read FIFO
  logic [RW-1:0] rqc_q, rdr_q;           // the read path's count and rd_res, a cycle old
  logic         beat_room;               // the read path's room covers them and eight more
  logic [4:0]   wr_pend;                 // writes popped, their B not yet back
  logic         no_wr, no_rd;
  logic         aw_v, w_v, w_hi, w_busy; // w_busy: aw_v || w_v, one flag
  logic [511:0] w_d;
  logic [63:0]  w_m;
  logic         hrst_q;
  always_ff @(posedge hclk) hrst_q <= hrst;
  // the head and what it may do this cycle
  wire h_rd  = cq_rvalid && !h_we;
  wire h_wr  = cq_rvalid && h_we;
  wire ext   = h_rd && h_cont && ob_open && beat_room && !idle_to;         // one LUT level
  // (not past the timer's close: the burst closes with its length as it was, the command starts
  // the next one)
  wire start = h_rd && !ob_v && beat_room && (no_wr || !ORDER);             // one
  // (the head cannot extend: a write, not cont, or no room; ob_full / the timer close too)
  wire close = ob_v && !ob_cl &&
               (ob_full || idle_to || (cq_rvalid && (h_we || !h_cont || !beat_room)));   // two
  wire ar_fire = m_arvalid && m_arready;
  wire ar_free = !ar_v || ar_fire;
  wire load_ar = (close || ob_cl) && ar_free;                                // three
  wire wgo   = h_wr && wq_rvalid && !ob_v && !ar_v && !w_busy && (no_rd || !ORDER);   // two
  assign cq_pop = ext || start || wgo;
  assign wq_pop = wgo;
  wire aw_fire = m_awvalid && m_awready;
  wire w_fire  = m_wvalid && m_wready;
  wire b_fire  = m_bvalid && m_bready;
  // R: taken into registers; the read path (otpu_hbm_rpath: the URAM FIFO, the pairs, the
  // crossing FIFO to the core clock) counts the native beats it holds
  logic         r_v, r_last, r_hi;
  logic [1:0]   r_resp;
  logic [255:0] r_d;
  wire          r_push = r_v && r_hi;   // a pair's second beat: its native beat is in the path
  logic [RW-1:0] rp_count;
  otpu_hbm_rpath #(.RDEPTH(RDEPTH)) u_rp (
    .hclk, .hrst, .r_valid(r_v), .r_data(r_d), .count(rp_count),
    .clk, .rst, .n_rvalid, .n_rdata);
  // the lengths in AR order (RLAST's check: the beat count of the burst being returned)
  (* ram_style = "distributed" *) logic [3:0] lens [256];
  logic [7:0]   lp, lr;
  logic [3:0]   rem;
  logic         rem_z;                   // rem == 0: the next beat starts a burst
  wire [3:0]    r_len = lens[lr];
  wire          r_last_x = rem_z ? (r_len == 4'd0) : (rem == 4'd1);
  wire          r_end = r_v && r_last_x;
  // the AXI outputs: registers
  assign m_awid = ID; assign m_wid = ID; assign m_arid = ID;
  assign m_awsize = 3'b101; assign m_arsize = 3'b101;
  assign m_awburst = 2'b01; assign m_arburst = 2'b01;
  assign m_awlen = 4'd1;
  assign m_awvalid = aw_v;
  assign m_wvalid = w_v;
  assign m_wlast = w_hi;
  assign m_arvalid = ar_v && ot_room;
  assign m_rready = !hrst_q;
  assign m_bready = !hrst_q;
  always_ff @(posedge hclk) begin
    if (hrst) begin
      ob_v <= 1'b0; ob_cl <= 1'b0; ob_full <= 1'b0; ob_open <= 1'b0; ob_len <= '0; ob_lm1 <= '0;
      ob_beat <= '0; rqc_q <= '0; rdr_q <= '0; w_busy <= 1'b0;
      idle <= '0; idle_to <= 1'b0;
      ar_v <= 1'b0; m_araddr <= '0; m_arlen <= '0;
      rd_ot <= '0; ot_room <= 1'b1; rd_res <= '0; beat_room <= 1'b0;
      wr_pend <= '0; no_wr <= 1'b1; no_rd <= 1'b1;
      aw_v <= 1'b0; w_v <= 1'b0; w_hi <= 1'b0; m_awaddr <= '0; m_wdata <= '0; m_wstrb <= '0;
      r_v <= 1'b0; r_hi <= 1'b0; r_last <= 1'b0; r_resp <= '0;
      lp <= '0; lr <= '0; rem <= '0; rem_z <= 1'b1;
      bq_wvalid <= 1'b0; herr <= 1'b0;
    end else begin
      // ---- the open burst
      if (start) begin
        ob_v <= 1'b1; ob_cl <= 1'b0; ob_beat <= h_beat; ob_len <= 4'd1; ob_lm1 <= '0;
        ob_full <= (BURST == 1); ob_open <= (BURST != 1); idle <= '0;
      end else if (ext) begin
        ob_len <= ob_len + 1'b1; ob_lm1 <= ob_len[2:0];
        ob_full <= (ob_len == 4'(BURST - 1)); ob_open <= (ob_len != 4'(BURST - 1)); idle <= '0;
      end
      if (close && !ar_free) begin ob_cl <= 1'b1; ob_open <= 1'b0; end
      if (load_ar) begin
        ob_v <= 1'b0; ob_cl <= 1'b0; ob_open <= 1'b0; idle <= '0;
      end
      // the idle timer: counts while a burst is open and nothing is queued
      if (ob_v && !ob_cl && !cq_rvalid && !ext) begin
        if (idle != IW'(IDLE - 1)) idle <= idle + 1'b1;
      end else if (!ob_v) idle <= '0;
      idle_to <= ob_v && !ob_cl && !cq_rvalid && (idle == IW'(IDLE - 1));
      // ---- the AR register
      if (load_ar) begin
        ar_v <= 1'b1; m_araddr <= axaddr(ob_beat); m_arlen <= {ob_lm1, 1'b1};
      end else if (ar_fire) ar_v <= 1'b0;
      if (ar_fire) begin lens[lp] <= m_arlen; lp <= lp + 1'b1; end
      // ---- the counters and their flags (from their next values)
      rd_ot <= rd_ot + OW'(ar_fire) - OW'(r_end);
      ot_room <= (rd_ot + OW'(ar_fire) - OW'(r_end)) < OW'(OT);
      rd_res <= rd_res + RW'(ext || start) - RW'(r_push);
      // (the room flag from copies a cycle old: a native beat leaves rd_res up to four cycles
      // before the path counts it (its R register, the path's input register and count
      // register, this copy: at most two beats at a beat a cycle on R) and the pops between
      // this flag and the pop it gates add two more; the margin is eight)
      rqc_q <= rp_count; rdr_q <= rd_res;
      beat_room <= (rqc_q + rdr_q) < RW'(RDEPTH - 8);
      wr_pend <= wr_pend + 5'(wgo) - 5'(b_fire);
      no_wr <= (wr_pend + 5'(wgo) - 5'(b_fire)) == 5'd0;
      no_rd <= ((rd_res + RW'(ext || start) - RW'(r_push)) == '0) &&
               ((rd_ot + OW'(ar_fire) - OW'(r_end)) == '0);
      // ---- the write in flight
      if (wgo) begin
        aw_v <= 1'b1; m_awaddr <= axaddr(h_beat);
        w_v <= 1'b1; w_hi <= 1'b0; w_d <= h_data; w_m <= h_mask; w_busy <= 1'b1;
        m_wdata <= h_data[255:0]; m_wstrb <= h_mask[31:0];
      end else begin
        if (aw_fire) aw_v <= 1'b0;
        if (w_fire) begin
          if (!w_hi) begin w_hi <= 1'b1; m_wdata <= w_d[511:256]; m_wstrb <= w_m[63:32]; end
          else w_v <= 1'b0;
        end
        w_busy <= (aw_v && !aw_fire) || (w_v && !(w_fire && w_hi));
      end
      bq_wvalid <= b_fire;
      // ---- R
      r_v <= m_rvalid && m_rready; r_d <= m_rdata; r_last <= m_rlast; r_resp <= m_rresp;
      if (r_v) begin
        r_hi <= !r_hi;
        if (rem_z) begin rem <= r_len; rem_z <= (r_len == 4'd0); end
        else begin rem <= rem - 1'b1; rem_z <= (rem == 4'd1); end
        if (r_last_x) lr <= lr + 1'b1;
      end
      // ---- errors (sticky)
      if (r_v && (r_resp != 2'b00 || r_last != r_last_x || (rem_z && lp == lr))) herr <= 1'b1;
      if (b_fire && (m_bresp != 2'b00 || wr_pend == 5'd0)) herr <= 1'b1;
    end
  end
`ifndef SYNTHESIS
  always @(posedge hclk) if (!hrst) begin
    if (ext && start) $fatal(1, "otpu_hbm_ch: ext and start together");
    if (cq_pop && !cq_rvalid) $fatal(1, "otpu_hbm_ch: a pop of an empty command FIFO");
    if (wq_pop && !wq_rvalid) $fatal(1, "otpu_hbm_ch: a pop of an empty write FIFO");
    if (ORDER && start && wr_pend != 0) $fatal(1, "otpu_hbm_ch: a read started with writes pending");
    if (ORDER && wgo && (rd_res != 0 || rd_ot != 0)) $fatal(1, "otpu_hbm_ch: a write with reads outstanding");
  end
`endif
endmodule
