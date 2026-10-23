// One memory channel of the slice (otpu_native_dram's native port, core clock) onto an AXI4 master
// in the shell's clock (aclk: clk_main_a0, 250 MHz), toward the slot's DIMM (sh_ddr, through
// otpu_f2_mem): a command per 64-byte beat, each one single-beat AXI transaction (len 0, size 6)
// at BASE + beat x 64. The F2 board's counterpart of otpu_mem_ch (docs/f2.md).
//
// Crossing: the commands (we, beat) and the write data (a beat and its byte mask) cross into aclk
// through asynchronous FIFOs (otpu_afifo), the read data back through another, popped as soon as
// it is there (the native port's read data has no backpressure): a read goes out only when the
// read-data FIFO has room for it and for every read in flight (credits). The count of write
// responses (B) crosses back gray-coded: n_wdone, one step at most a cycle.
//
// Order. The native port's contract (otpu_native_dram): one master's commands on one channel
// execute in command order, so a read issued after a write to its beat returns the written data
// without waiting for n_wdone, and a write counted in n_wdone is visible to the other master (the
// host's PCIS). AXI orders neither the read and write channels against each other nor anything
// before a response; the DIMM's controller may reorder them. So the commands go out in order,
// one a cycle at most, and
//   - a read waits while a write to its beat is in flight (issued, its B not back),
//   - a write waits while a read of its beat is in flight (issued, its data not back),
// so accesses to one beat reach the controller one at a time, in command order; accesses to
// different beats overlap freely. A write counts in n_wdone once its B is back: the controller
// then holds it for every later access, the host's included. All of this master's writes use one
// AXI ID and its reads another (otpu_f2_mem), so its B's and its R's come back in order, and the
// in-flight tables retire from their heads.
//
// n_err: a response other than OKAY (sticky until rst; otpu_board's STATUS AXI_ERR).
// Resets: rst (core) and arst (aclk) are the same event (otpu_f2_sys: the shell's rst_main_n and
// the MMCM's lock), each side's synchronized; nothing is in flight while the accelerator is idle.
module otpu_f2_nch #(
  parameter logic [63:0] BASE = 64'h0,    // the channel's first byte in the DIMM
  parameter int CQ  = 16,                 // command and write-data FIFOs, beats
  // reads in flight: the DIMM's ~550 ns round trip (decode on the card: 7.0 GB/s with ROD 32)
  // needs 2 x 64 of them at 128 B a core cycle (100 MHz); 128 leaves room for the latency the
  // controller adds at the higher rate (tests/test_f2_rtl.py test_f2_read_throughput)
  parameter int RD  = 128,                // read-data FIFO (credits), beats
  parameter int ROD = 128,
  parameter int WOD = 16,                 // writes in flight
  parameter bit HAZ = 1'b1                // 0: no same-beat waits (tests only: tb_f2mem fails)
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
  // ---- AXI4 master (aclk): single beats, no IDs (otpu_f2_mem gives them)
  input  logic          aclk,
  input  logic          arst,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [63:0]   m_awaddr,
  output logic          m_wvalid,
  input  logic          m_wready,
  output logic [511:0]  m_wdata,
  output logic [63:0]   m_wstrb,
  input  logic          m_bvalid,
  output logic          m_bready,
  input  logic [1:0]    m_bresp,
  output logic          m_arvalid,
  input  logic          m_arready,
  output logic [63:0]   m_araddr,
  input  logic          m_rvalid,
  output logic          m_rready,
  input  logic [511:0]  m_rdata,
  input  logic [1:0]    m_rresp
);
  localparam int RW = $clog2(ROD), WW = $clog2(WOD);
  initial if (ROD < 2 || WOD < 2 || ROD != (1 << RW) || WOD != (1 << WW) || ROD > RD)
    $fatal(1, "otpu_f2_nch: ROD and WOD powers of two >= 2, ROD <= RD");

  // ---- into aclk: commands and write data
  logic        c_v, c_pop, c_we;
  logic [24:0] c_a;
  logic        cq_v, cq_pop, cq_we, c_held;
  logic [24:0] cq_a;
  logic        d_v, d_pop;
  logic [511:0] d_data;
  logic [63:0] d_mask;
  otpu_afifo #(.W(26), .DEPTH(CQ)) u_cq (
    .wclk(clk), .wrst(rst), .wvalid(n_cvalid), .wready(n_cready), .wdata({n_cwe, n_caddr}),
    .wslot('0), .wcommit(), .wused(),
    .rclk(aclk), .rrst(arst), .rvalid(cq_v), .rready(cq_pop), .rdata({cq_we, cq_a}));
  otpu_afifo #(.W(576), .DEPTH(CQ)) u_dq (
    .wclk(clk), .wrst(rst), .wvalid(n_wvalid), .wready(n_wready), .wdata({n_wmask, n_wdata}),
    .wslot('0), .wcommit(), .wused(),
    .rclk(aclk), .rrst(arst), .rvalid(d_v), .rready(d_pop), .rdata({d_mask, d_data}));

  // Hold the command head in aclk registers before address hazard comparisons.
  // Refill on the same edge as issue, but never bypass an empty head. Prefetch
  // consumes neither write data nor in-flight credits; only c_pop does that.
  assign c_v = c_held && !arst;
  assign cq_pop = cq_v && (!c_held || c_pop) && !arst;
  always_ff @(posedge aclk) begin
    if (arst) c_held <= 1'b0;
    else if (!c_held || c_pop) begin
      c_held <= cq_v;
      if (cq_v) begin c_we <= cq_we; c_a <= cq_a; end
    end
  end

  // ---- back to the core: read data (popped as it arrives)
  logic                  r_push;
  logic [$clog2(RD):0]   r_used;
  otpu_afifo #(.W(512), .DEPTH(RD)) u_rq (
    .wclk(aclk), .wrst(arst), .wvalid(r_push), .wready(), .wdata(m_rdata),
    .wslot('0), .wcommit(), .wused(r_used),
    .rclk(clk), .rrst(rst), .rvalid(n_rvalid), .rready(1'b1), .rdata(n_rdata));

  // ---- in flight: the beats of the issued writes (to their B) and reads (to their data)
  logic [WOD-1:0][24:0] wa;
  logic [WOD-1:0]       wv;
  logic [WW-1:0]        wh, wt;          // head (oldest), tail (next free)
  logic [WW:0]          wn;
  logic [ROD-1:0][24:0] ra;
  logic [ROD-1:0]       rv;
  logic [RW-1:0]        rh, rt;
  logic [RW:0]          rn;
  logic                 hz_w, hz_r;     // the head command's beat: a write / a read in flight
  always_comb begin
    hz_w = 1'b0;
    for (int k = 0; k < WOD; k++) hz_w |= HAZ && wv[k] && wa[k] == c_a;
    hz_r = 1'b0;
    for (int k = 0; k < ROD; k++) hz_r |= HAZ && rv[k] && ra[k] == c_a;
  end

  // ---- issue: the head command, when its output registers are free (AW and W together; AR: a
  // two-entry queue, so that a read goes out every cycle while the arbiter takes one every cycle;
  // the one AR register of before took a read every second cycle at most: 32 B an aclk cycle)
  logic aw_v, w_v;
  logic [1:0]  arn;                     // AR queue entries (registered: issue never waits on ready)
  logic [63:0] arq1;
  logic        ar_pop;
  logic iss_w, iss_r, b_tk, r_tk;
  // reads in flight plus the read-data FIFO's entries the core has not popped
  logic [$clog2(RD)+1:0] r_cred;
  assign r_cred = ($clog2(RD) + 2)'(rn) + ($clog2(RD) + 2)'(r_used);
  assign iss_w = c_v && c_we && d_v && !aw_v && !w_v && !wn[WW] && !hz_r;
  assign iss_r = c_v && !c_we && arn != 2'd2 && !rn[RW] && r_cred < ($clog2(RD) + 2)'(RD) && !hz_w;
  assign c_pop = iss_w || iss_r;
  assign d_pop = iss_w;
  assign b_tk = m_bvalid;               // bready is 1
  assign r_tk = m_rvalid;               // rready is 1: the credits hold its room
  assign r_push = r_tk;
  assign m_bready = 1'b1;
  assign m_rready = 1'b1;
  assign m_awvalid = aw_v;
  assign m_wvalid = w_v;
  assign m_arvalid = arn != 2'd0;
  assign ar_pop = m_arvalid && m_arready;

  always_ff @(posedge aclk) begin
    if (iss_w) begin
      m_awaddr <= BASE + {33'd0, c_a, 6'd0};
      m_wdata  <= d_data;
      m_wstrb  <= d_mask;
      wa[wt]   <= c_a;
    end
    if (ar_pop) m_araddr <= arq1;       // the queue's head is m_araddr
    if (iss_r) begin
      if (arn == 2'd0 || (arn == 2'd1 && ar_pop)) m_araddr <= BASE + {33'd0, c_a, 6'd0};
      else arq1 <= BASE + {33'd0, c_a, 6'd0};
      ra[rt]   <= c_a;
    end
    if (arst) begin
      aw_v <= 1'b0; w_v <= 1'b0; arn <= 2'd0;
      wv <= '0; wh <= '0; wt <= '0; wn <= '0;
      rv <= '0; rh <= '0; rt <= '0; rn <= '0;
    end else begin
      aw_v <= iss_w || (aw_v && !m_awready);
      w_v  <= iss_w || (w_v && !m_wready);
      arn  <= arn + 2'(iss_r) - 2'(ar_pop);
      if (iss_w) begin wv[wt] <= 1'b1; wt <= wt + 1'b1; end
      if (b_tk)  begin wv[wh] <= 1'b0; wh <= wh + 1'b1; end
      wn <= wn + (WW + 1)'(iss_w) - (WW + 1)'(b_tk);
      if (iss_r) begin rv[rt] <= 1'b1; rt <= rt + 1'b1; end
      if (r_tk)  begin rv[rh] <= 1'b0; rh <= rh + 1'b1; end
      rn <= rn + (RW + 1)'(iss_r) - (RW + 1)'(r_tk);
    end
  end

  // ---- write responses counted, gray-coded into the core clock; errors, sticky
  logic [15:0] bcnt, bgray;
  logic        aerr;
  always_ff @(posedge aclk) begin
    if (arst) begin
      bcnt <= '0; bgray <= '0; aerr <= 1'b0;
    end else begin
      if (b_tk) bcnt <= bcnt + 1'b1;
      bgray <= (bcnt + 16'(b_tk)) ^ ((bcnt + 16'(b_tk)) >> 1);
      if ((b_tk && m_bresp != 2'b00) || (r_tk && m_rresp != 2'b00)) aerr <= 1'b1;
    end
  end
  (* ASYNC_REG = "TRUE" *) logic [15:0] bg1, bg2;
  (* ASYNC_REG = "TRUE" *) logic        e1, e2;
  logic [15:0] bbin;
  always_comb begin
    for (int i = 0; i < 16; i++) bbin[i] = ^(bg2 >> i);
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      bg1 <= '0; bg2 <= '0; e1 <= 1'b0; e2 <= 1'b0;
      n_wdone <= '0; n_err <= 1'b0;
    end else begin
      bg1 <= bgray;
      bg2 <= bg1;
      e1  <= aerr;
      e2  <= e1;
      n_wdone <= bbin;
      n_err <= n_err || e2;
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge aclk) if (!arst) begin
    if (b_tk && wn == 0) $fatal(1, "otpu_f2_nch: B with no write in flight");
    if (r_tk && rn == 0) $fatal(1, "otpu_f2_nch: read data with no read in flight");
    if (r_push && r_used == RD) $fatal(1, "otpu_f2_nch: read-data FIFO overflow");
  end
`endif
endmodule
