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
// Each side's reset goes through a register of its own here (rst_q, hrst_q: the bridges sit
// along the HBM's ports, the leaves of the top's reset trees; otpu_f2_hbm_mem registers them
// once more on the way), so a reset pin is a flip-flop's cycle from its driver.
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

  logic rst_q, hrst_q;
  always_ff @(posedge clk) rst_q <= rst;
  always_ff @(posedge hclk) hrst_q <= hrst;

  // ================================================================== core side
  // ---- the native port into two FIFOs of 4 (readies from flip-flops: room for two more), the
  // command FIFO's head into a register (h_*: the hazard compares below and the issue start
  // from flip-flops. The a2af0904 placement ran the head read by cf_h through the 80 compares
  // into the crossing's 600 write enables, 8 levels, 6.3 of 7.1 ns of route at 200 MHz
  // (rhino's e165f4f8 on its p4rl8 build: the same path); the crossing now writes its tail
  // every cycle, otpu_afifo WT, so the issue reaches only its pointer). The register is a core
  // cycle on a command's way (a stream's commands pipeline through it one a cycle: the first
  // issues two cycles after its take, not one), 5 ns on the HBM's hundreds
  logic [25:0]  cf [4];
  logic [575:0] df [4];
  logic [2:0]   cf_n, df_n;
  logic [1:0]   cf_h, df_h;
  logic         h_v, h_we;
  logic [24:0]  h_a;
  logic         c_pop, d_pop;              // the head issues; a write's data with it
  wire c_take = n_cvalid && n_cready;
  wire d_take = n_wvalid && n_wready;
  wire h_load = cf_n != 0 && (!h_v || c_pop);   // the FIFO's head into the register
  always_ff @(posedge clk) begin
    if (h_load) {h_we, h_a} <= cf[cf_h];
    if (rst_q) begin
      cf_n <= '0; df_n <= '0; cf_h <= '0; df_h <= '0; n_cready <= 1'b0; n_wready <= 1'b0;
      h_v <= 1'b0;
    end else begin
      if (c_take) cf[2'(cf_h + cf_n)] <= {n_cwe, n_caddr};
      if (d_take) df[2'(df_h + df_n)] <= {n_wmask, n_wdata};
      cf_n <= cf_n + 3'(c_take) - 3'(h_load);
      df_n <= df_n + 3'(d_take) - 3'(d_pop);
      if (h_load) cf_h <= cf_h + 1'b1;
      if (!h_v || c_pop) h_v <= cf_n != 0;
      if (d_pop) df_h <= df_h + 1'b1;
      // room for this cycle's take and the next's: at most 2 after this cycle's changes
      n_cready <= (cf_n + 3'(c_take) - 3'(h_load)) <= 3'd2;
      n_wready <= (df_n + 3'(d_take) - 3'(d_pop)) <= 3'd2;
    end
  end

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
  wire iss_w = h_v && h_we && df_n != 0 && x_ready && !wn[WW] && !hz_r;
  wire iss_r = h_v && !h_we && x_ready && !rn[RW] && !hz_w;
  assign c_pop = iss_w || iss_r;
  assign d_pop = iss_w;
  logic        x_v, x_we;
  logic [24:0] x_a;
  logic [575:0] x_d;
  logic        xr_v;
  logic [511:0] xr_d;
  logic x_pop;
  // (RWR, WT: x_ready and the RAM's write enables are flip-flops; the tail takes the head and
  // its data every cycle there is room, the issue moves the pointer)
  otpu_afifo #(.W(602), .DEPTH(CQ), .RWR(1'b1), .WT(1'b1)) u_xq (
    .wclk(clk), .wrst(rst_q), .wvalid(c_pop), .wready(x_ready), .wdata({h_we, h_a, df[df_h]}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(hclk), .rrst(hrst_q), .rvalid(x_v), .rready(x_pop), .rdata({x_we, x_a, x_d}));

  always_ff @(posedge clk) begin
    if (iss_w) wa[wt] <= h_a;
    if (iss_r) ra[rt] <= h_a;
    if (rst_q) begin
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
  // whether the held command is the open burst's next beat, and not at a BMAX boundary, decided
  // as it loads (the 6fac2d06 synthesis: the 25-bit compare against o_nx ran into the pop's 600
  // enables at -0.19 ns, 450 MHz). While it is held, o_nx does not move (a burst opens or grows
  // only by popping it), so the flag stays true; as it loads, o_nx is this cycle's next value
  logic        k_nx, k_bnd;
  logic [575:0] k_d;
  logic        k_pop;                  // the held command joins or opens a burst
  // the head loads (the register is free, or its command goes): the beat's 576 enables take
  // this net alone, not qualified by x_v or the reset (what loads while no command is there is
  // never read: k_v is 0, and the W FIFO's count moves only on k_pop && k_we), and synthesis
  // replicates its LUT per 64 of them (max_fanout; rhino's round-3 placement, docs/wide_port.md:
  // one LUT into 604 clock enables, 3.6 ns of route at 450 MHz)
  (* max_fanout = 64 *) logic k_ld;
  assign k_ld = !k_v || k_pop;
  assign x_pop = x_v && k_ld;
  always_ff @(posedge hclk) if (k_ld) k_d <= x_d;
  always_ff @(posedge hclk) begin
    if (hrst_q) k_v <= 1'b0;
    else if (k_ld) begin
      k_v <= x_v;
      if (x_v) begin
        k_we <= x_we; k_a <= x_a;
        // (the three candidates compared in parallel, the late select last)
        k_nx <= k_open ? x_a == k_a + 1'b1 : k_join ? x_a == o_nx + 1'b1 : x_a == o_nx;
        k_bnd <= BW == 0 ? 1'b1 : (x_a % BMAX) == 0;
      end
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
  logic         wf_nz;                  // wf_n != 0, a register kept alongside (below)
  // the head continues the open burst: same direction, the next beat, room, no BMAX boundary
  wire cont = o_v && k_we == o_we && k_nx && o_len != (BW + 1)'(BMAX) && !k_bnd;
  // close the open burst: full, the head does not continue it, or it waited GATHER cycles
  wire q_room = o_we ? awq_n != 2'd2 : arq_n != 2'd2;
  wire close = o_v && q_room && (o_len == (BW + 1)'(BMAX) || (k_v && !cont) ||
                                 (!k_v && o_age >= 4'(GATHER)));
  wire w_room = !wf_n[WDW];             // wf_n <= WD - 1 (WD a power of two)
  // the head joins the open burst, or opens one where none is or the open one closes now. With
  // the head held, "the open burst closes now" is o_v && q_room && (full || !cont), and cont
  // implies not full, so (cont || !o_v || close) is (cont || !o_v || q_room): two LUT levels
  // from the registers (cont and q_room, then this), where the form with close was four
  // (rhino's round-4 placement, docs/wide_port.md "Round 5 RTL": wf_n_reg[1] -> wf_n_reg[4]
  // through six levels at 450 MHz: wf_n != 0 into the W valid, the W take, this pop, the count)
  assign k_pop = k_v && (!k_we || w_room) && (cont || !o_v || q_room);
`ifndef SYNTHESIS
  always @(posedge hclk) if (!hrst_q && k_pop != (k_v && (!k_we || w_room) && (cont || !o_v || close)))
    $fatal(1, "otpu_f2_hbm_nch: k_pop's short form differs from its definition");
`endif
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
  // W: the oldest sent AW's burst, from the W FIFO's head (two AXI beats per native beat; the
  // valid from two registers' flags: wb_n != 0 is one LUT, wf_nz a flip-flop)
  assign wb_v = wb_n != 0 && wf_nz;
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
    // the W FIFO's tail (not live unless the FIFO is full) takes the held beat every cycle; a
    // push only moves the count, so its write enables are not the pop's
    if (!wf_n[WDW]) wf[WDW'(wf_h + wf_n)] <= k_d;
    if (aw_tk) begin wb_len[wb_t] <= awq[awq_h].len; bb_len[2'(bb_h + bb_n)] <= awq[awq_h].len; end
    if (hrst_q) begin
      o_v <= 1'b0; o_age <= '0; arq_n <= '0; awq_n <= '0; arq_h <= 1'b0; awq_h <= 1'b0;
      wf_n <= '0; wf_nz <= 1'b0; wf_h <= '0; wb_n <= '0; bb_n <= '0; wb_h <= '0; bb_h <= '0; wb_t <= '0;
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
      // W: the count by +1 / -1 / 0 as one addend (one carry chain behind one LUT level: the
      // push k_pop && k_we is two levels, the pop w_tk && w_cnt[0] one), its nonzero flag kept
      // alongside from the count's bits above the lowest (one LUT) and the same push / pop
      begin
        logic w_push, w_pop_;
        w_push = k_pop && k_we;
        w_pop_ = w_tk && w_cnt[0];
        wf_n <= wf_n + {{WDW{w_pop_ && !w_push}}, w_push ^ w_pop_};
        wf_nz <= w_push || (|wf_n[WDW:1]) || (wf_nz && !w_pop_);
      end
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
    if (hrst_q) begin r_half <= 1'b0; rr_v <= 1'b0; end
    else begin
      rr_v <= r_tk;
      if (rr_v) begin
        r_half <= !r_half;
        if (!r_half) r_lo <= rr_d;
      end
    end
  end
  // (RWR, WT as u_xq: the 512-bit RAM's write enables and address are registered copies, the
  // paired beat's push moves the pointer; the writer never sees the FIFO full: RD reads in flight)
  otpu_afifo #(.W(512), .DEPTH(RD), .RWR(1'b1), .WT(1'b1)) u_rq (
    .wclk(hclk), .wrst(hrst_q), .wvalid(rr_v && r_half), .wready(), .wdata({rr_d, r_lo}),
    .wslot('0), .wcommit(), .wused(), .rused(),
    .rclk(clk), .rrst(rst_q), .rvalid(xr_v), .rready(1'b1), .rdata(xr_d));
  assign n_rvalid = xr_v;
  assign n_rdata = xr_d;

  // ---- the write count and errors into the core clock
  (* ASYNC_REG = "TRUE" *) logic [15:0] bg1, bg2;
  (* ASYNC_REG = "TRUE" *) logic        e1, e2;
  logic [15:0] bbin;
  always_comb for (int i = 0; i < 16; i++) bbin[i] = ^(bg2 >> i);
  always_ff @(posedge clk) begin
    if (rst_q) begin
      bg1 <= '0; bg2 <= '0; e1 <= 1'b0; e2 <= 1'b0; n_wdone <= '0; n_err <= 1'b0;
    end else begin
      bg1 <= bgray; bg2 <= bg1; e1 <= aerr; e2 <= e1;
      n_wdone <= bbin;
      n_err <= n_err || e2;
    end
  end
  assign wdone_s = n_wdone;

`ifndef SYNTHESIS
  always_ff @(posedge hclk) if (!hrst_q) begin
    if (b_tk && bb_n == 0) $fatal(1, "otpu_f2_hbm_nch: B with no write in flight");
    if (k_pop && k_we && !w_room) $fatal(1, "otpu_f2_hbm_nch: W FIFO overflow");
    if (wf_nz != (wf_n != 0)) $fatal(1, "otpu_f2_hbm_nch: wf_nz %0d, wf_n %0d", wf_nz, wf_n);
  end
  always_ff @(posedge clk) if (!rst_q && n_rvalid && rn == 0)
    $fatal(1, "otpu_f2_hbm_nch: read data with no read in flight");
`endif
endmodule
