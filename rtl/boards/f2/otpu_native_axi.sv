// One native memory channel (otpu_native_dram's n_* port: one command per 64-byte beat, write data
// in its own stream, reads returned in order with no back-pressure, n_wdone counting writes)
// onto an AXI4 512-bit master at the core clock: the F2 HBM build's channel c, BASE_ADDR =
// c x 512 MiB, into its SmartConnect (clock and width conversion to HBM AXI3 port c at 450 MHz;
// docs/f2_hbm.md "The memory path").
//
// From Alex's burst200 branch (ergodex-openTPU alex-burst200, commit 06c169cb,
// rtl/boards/f2/otpu_native_axi.sv, the design his F2 image closed timing and ran Gemma / Qwen
// with), with these changes:
//   - BASE_ADDR aligns to the window CH_LOG2 (Alex: 2 GiB, the default; f2-hbm: 512 MiB, a
//     pseudo-channel), fail closed at elaboration; a command whose beat lies outside the window
//     sets n_err (the address is masked to the window, never into a neighbour's);
//   - XL: the queues keep XL slots below their readies (n_cready = cn < QD - XL), for a native
//     port that sees the ready XL / 2 cycles late and whose take lands XL / 2 cycles later still
//     (the build's NXR = 3 SLR stages each way: XL = 6, as otpu_f2_hbm_nch's), and what arrives
//     is taken (n_cvalid / n_wvalid are then takes, not requests);
//   - the read burst leaves through an AR register (ar_v / ar_addr / ar_cnt), so the next burst
//     gathers while the last one waits for ARREADY: a stream takes a command every cycle
//     (Alex's single pending burst blocked the queue for its AR's cycle: 16 commands in 17).
// Coalescing (Alex's): adjacent read commands (the next beat, within a 4 KiB page) gather into
// one INCR burst of up to READ_BURST native beats; a burst closes at READ_BURST, at a 4 KiB
// boundary, at a credit limit, when the queue's head cannot join it (a write, another address)
// or after READ_IDLE cycles with nothing to take (READ_WAIT 0, added here: when the queue runs
// dry, so nothing waits for a command not yet queued). Writes stay single beats. Order: a read is
// not taken while a write is unanswered (B), a write not issued while a read is unanswered
// (R), so the native contract (a read after a write to its beat sees it) holds whatever the
// AXI side reorders between its read and write channels. All IDs 0: one channel's reads come
// back in order. R and B are always ready; the native adapter reserves room for its reads.
module otpu_native_axi #(
  parameter logic [33:0] BASE_ADDR = 34'd0,
  parameter int CH_LOG2 = 31,          // the window's bytes, log2 (BASE_ADDR aligned to it)
  parameter int QD = 16,
  parameter int XL = 0,                // queue slots kept below the readies (2 NXR)
  parameter int MAX_OUT = 128,
  parameter int READ_BURST = 16,
  parameter int READ_IDLE = 4,
  parameter bit READ_WAIT = 1'b1,      // 1 (Alex's): a burst waits for commands still to come
                                       // (up to READ_IDLE cycles); 0: it takes only what is
                                       // queued, closing when the queue runs dry
  parameter bit ORDER = 1'b1,          // the direction barriers (0: tests only, tb_f2hbm_axi)
  // with ORDER, the barriers per page (P2, docs/v4/p2-f2.md "Page buckets"): a read waits only for
  // the unanswered writes of its 4 KiB page's bucket, a write only for the reads in flight of its
  // page's bucket (NBK buckets, an XOR fold of the page), so a write to one page does not stall a
  // stream from others; the native contract (a read after a write to its beat sees it, one before
  // it does not) holds per beat. 0: every read waits for every write and the reverse (Alex's)
  parameter bit PORDER = 1'b0,
  parameter int NBK = 32,
  // ZB (P2, docs/v4/p2-f2.md "Zero-bubble bursts"): a read that cannot join the open burst (not
  // adjacent) closes it into the AR register and opens the next in the same cycle, when the AR
  // register is free; 0 (Alex's): the close takes a cycle without a take. A stream of short runs
  // (a lane's 9-beat rows on a channel) then takes a command every cycle instead of one in
  // (run + 1)
  parameter bit ZB = 1'b0
)(
  input logic clk, rst,
  input logic n_cvalid, output logic n_cready,
  input logic n_cwe, input logic [24:0] n_caddr,
  input logic n_wvalid, output logic n_wready,
  input logic [511:0] n_wdata, input logic [63:0] n_wmask,
  output logic n_rvalid, output logic [511:0] n_rdata,
  output logic [15:0] n_wdone, output logic n_err,
  output logic [33:0] m_awaddr, output logic m_awvalid,
  input logic m_awready,
  output logic [511:0] m_wdata, output logic [63:0] m_wstrb,
  output logic m_wvalid, input logic m_wready,
  input logic m_bvalid, input logic [1:0] m_bresp, output logic m_bready,
  output logic [33:0] m_araddr, output logic [7:0] m_arlen,
  output logic m_arvalid,
  input logic m_arready,
  input logic [511:0] m_rdata, input logic m_rvalid, m_rlast,
  input logic [1:0] m_rresp, output logic m_rready
);
  localparam int PW = $clog2(QD), CW = $clog2(QD+1), OW = $clog2(MAX_OUT+1);
  localparam int AB = (CH_LOG2 - 6 < 25) ? CH_LOG2 - 6 : 25;    // beat bits inside the window
  localparam logic [33:0] WMASK = (34'd1 << CH_LOG2) - 34'd1;
  initial begin
    if (QD < 2 || (1 << PW) != QD) $fatal(1,"QD must be a power of two >= 2");
    if (XL < 0 || XL > QD - 2) $fatal(1,"XL must leave at least two queue slots");
    if (READ_BURST < 1 || READ_BURST > 16) $fatal(1,"READ_BURST must be 1..16");
    if (READ_IDLE < 1) $fatal(1,"READ_IDLE must be positive");
    if (MAX_OUT < 2) $fatal(1,"MAX_OUT must be >= 2");
    if (CH_LOG2 < 12 || CH_LOG2 > 31) $fatal(1,"CH_LOG2 must be 12..31");
    if ((BASE_ADDR & WMASK) != 0) $fatal(1,"BASE_ADDR must align to the window (2^CH_LOG2)");
    if ({1'b0, BASE_ADDR} + {1'b0, WMASK} > 35'h3_FFFF_FFFF) $fatal(1,"window past 16 GiB");
  end
  function automatic logic [33:0] axaddr(input logic [24:0] a);
    return BASE_ADDR | (34'(a) << 6 & WMASK);
  endfunction
  (* ram_style="distributed" *) logic [25:0] cq [0:QD-1];
  (* ram_style="distributed" *) logic [575:0] wq [0:QD-1];
  logic [PW-1:0] cp, cr, wp, wr;
  logic [CW-1:0] cn, wn;
  logic [OW-1:0] reads, writes;
  logic aw_sent, w_sent;
  localparam int BP = $clog2(MAX_OUT), IW = $clog2(READ_IDLE+1);
  logic [24:0] pending_addr, next_addr;
  logic [4:0] pending_count;
  logic [IW-1:0] idle_count;
  logic pending_valid;                 // the open burst is complete; the AR register is busy
  logic ar_v;                          // the AR register: the burst on m_ar*
  logic [24:0] ar_addr;
  logic [4:0] ar_cnt;
  // Credit count includes collected but not yet issued native reads.
  // The length queue checks RLAST independently of native beat accounting.
  (* ram_style="distributed" *) logic [4:0] lengths [0:(1<<BP)-1];
  logic [BP-1:0] lp, lr;
  logic [OW-1:0] ln;
  logic [4:0] remaining;

  // XL > 0: n_cvalid / n_wvalid are takes already made against a ready seen earlier (the SLR
  // chain carries n_cvalid && n_cready), so whatever arrives is pushed: the XL slots hold it
  wire pushc = n_cvalid && (XL > 0 || n_cready);
  wire pushw = n_wvalid && (XL > 0 || n_wready);
  wire head_we = cq[cr][25];
  // ---- page buckets (PORDER): per bucket the read beats taken and not yet back, the writes
  // handed out and not yet answered; their nonzero flags as registers (the take's and the write's
  // checks select one), the buckets of the bursts in flight (AR order) and of the writes (AW order)
  localparam int KB = $clog2(NBK);
  function automatic logic [KB-1:0] bkt(input logic [24:0] a);
    logic [18:0] pg;
    logic [KB-1:0] k;
    pg = a[24:6];
    k = '0;
    for (int i = 0; i < 19; i += KB) k ^= KB'(pg >> i);
    return k;
  endfunction
  logic [NBK-1:0] rbz, wbz;                        // the bucket has reads / writes outstanding
  wire [KB-1:0] head_k = bkt(cq[cr][24:0]);
  wire rd_clear = !PORDER ? (writes == 0) : !wbz[head_k];
  wire wr_clear = !PORDER ? (reads == 0) : !rbz[head_k];
  wire adjacent = cq[cr][24:0] == next_addr && cq[cr][24:6] == pending_addr[24:6];
  wire can_take = !pending_valid && cn != 0 && !head_we && (rd_clear || !ORDER) && reads < MAX_OUT;
  wire join = can_take && (pending_count == 0 || adjacent);
  wire ar_free;
  // ZB: the head opens the next burst while the open one goes to the free AR register (never a
  // head that would close at once: one beat bursts, a page's last beat, the last credit, a dry
  // queue without READ_WAIT: those keep the close-then-take sequence)
  wire sw_new_full = READ_BURST == 1 || cq[cr][5:0] == 6'd63 || reads == OW'(MAX_OUT-1) ||
                     (!READ_WAIT && cn == CW'(1));
  wire sw = ZB && can_take && pending_count != 0 && !adjacent && ar_free && !sw_new_full;
  wire take_read = join || sw;
  wire [4:0] cnt_t = pending_count + 5'd1;
  wire full_t = cnt_t == 5'(READ_BURST) || cq[cr][5:0] == 6'd63 || reads == OW'(MAX_OUT-1) ||
                (!READ_WAIT && cn == CW'(1));
  // (READ_WAIT 0: a stream arriving a command a cycle goes out a beat a burst with no wait, a
  // backlog in the queue as bursts: coalescing adds no latency; with 64 reads in flight a
  // channel the decode token is latency-bound, and every cycle a burst's first beat waits for
  // its last is a cycle on its credit: docs/f2_hbm.md "The memory path")
  // Never wait for another command at a direction/page/discontinuity boundary. A short idle
  // timer also flushes isolated final reads.
  wire flush = pending_count != 0 && !pending_valid && !take_read &&
               (cn != 0 || idle_count == IW'(READ_IDLE-1) || reads == OW'(MAX_OUT));
  wire close = (join && full_t) || flush;
  wire ar_fire = m_arvalid && m_arready;
  assign ar_free = !ar_v || ar_fire;
  wire load_ar = (close || pending_valid || sw) && ar_free;
  wire write_go = cn != 0 && head_we && wn != 0 && (wr_clear || !ORDER) && writes < MAX_OUT;
  wire aw_fire = m_awvalid && m_awready;
  wire w_fire = m_wvalid && m_wready;
  wire popw = write_go && (aw_sent || aw_fire) && (w_sent || w_fire);
  wire popc = take_read || popw;
  assign n_cready = !rst && cn < CW'(QD - XL);
  assign n_wready = !rst && wn < CW'(QD - XL);
  assign m_araddr = axaddr(ar_addr);
  assign m_arlen = {3'b0,ar_cnt} - 8'd1;
  assign m_awaddr = axaddr(cq[cr][24:0]);
  assign m_arvalid = !rst && ar_v;
  // Once a channel has started, keep it asserted until its handshake completes.
  assign m_awvalid = !rst && write_go && !aw_sent;
  assign m_wvalid = !rst && write_go && !w_sent;
  assign m_wdata = wq[wr][511:0];
  assign m_wstrb = wq[wr][575:512];
  assign m_rready = !rst;
  assign m_bready = !rst;
  wire r_fire = m_rvalid && m_rready;
  wire b_fire = m_bvalid && m_bready;
  wire start_response = r_fire && remaining == 0;
  wire bypass_length = start_response && ln == 0 && ar_fire;
  wire push_length = ar_fire && !bypass_length;
  wire pop_length = start_response && ln != 0;
  wire [4:0] response_count = remaining != 0 ? remaining :
                              (ln != 0 ? lengths[lr] : ar_cnt);
  wire response_known = remaining != 0 || ln != 0 || ar_fire;
  wire outside = AB < 25 && (n_caddr >> AB) != 0;
  assign n_rvalid = r_fire;
  assign n_rdata = m_rdata;
  always_ff @(posedge clk) begin
    if (rst) begin
      cp <= 0; cr <= 0; wp <= 0; wr <= 0; cn <= 0; wn <= 0;
      reads <= 0; writes <= 0; aw_sent <= 0; w_sent <= 0;
      n_wdone <= 0; n_err <= 0;
      pending_addr <= 0; next_addr <= 0; pending_count <= 0;
      pending_valid <= 0; idle_count <= 0;
      ar_v <= 0; ar_addr <= 0; ar_cnt <= 0;
      lp <= 0; lr <= 0; ln <= 0; remaining <= 0;
    end else begin
      // the open burst: this cycle's take joins it; a complete burst moves to the AR register
      // when that is free (this cycle's AR accepted, or none held), else waits there
      if (take_read) begin
        if (pending_count == 0) pending_addr <= cq[cr][24:0];
        next_addr <= cq[cr][24:0] + 1'b1;
        pending_count <= cnt_t;
        idle_count <= 0;
      end else if (pending_count != 0 && !pending_valid && !flush)
        idle_count <= idle_count + 1'b1;
      if (close && !ar_free) pending_valid <= 1;
      if (load_ar) begin
        ar_v <= 1;
        ar_addr <= pending_count == 0 ? cq[cr][24:0] : pending_addr;
        ar_cnt <= join ? cnt_t : pending_count;
        // (sw: the open burst left for the AR register; the head opens the next)
        pending_count <= sw ? 5'd1 : 5'd0; pending_valid <= 0; idle_count <= 0;
        if (sw) pending_addr <= cq[cr][24:0];
      end else if (ar_fire) ar_v <= 0;
      if (push_length) begin lengths[lp] <= ar_cnt; lp <= lp+1'b1; end
      if (pop_length) lr <= lr+1'b1;
      case ({push_length,pop_length})
        2'b10: ln <= ln+1'b1;
        2'b01: ln <= ln-1'b1;
        default: ;
      endcase
      if (r_fire) remaining <= response_known ? response_count-1'b1 : 5'd0;
      if (pushc) begin cq[cp] <= {n_cwe,n_caddr}; cp <= cp+1'b1; end
      if (pushw) begin wq[wp] <= {n_wmask,n_wdata}; wp <= wp+1'b1; end
      if (popc) cr <= cr+1'b1;
      if (popw) wr <= wr+1'b1;
      case ({pushc,popc})
        2'b10: cn <= cn+1'b1;
        2'b01: cn <= cn-1'b1;
        default: ;
      endcase
      case ({pushw,popw})
        2'b10: wn <= wn+1'b1;
        2'b01: wn <= wn-1'b1;
        default: ;
      endcase
      if (popc) begin aw_sent <= 0; w_sent <= 0; end
      else begin
        if (aw_fire) aw_sent <= 1;
        if (w_fire) w_sent <= 1;
      end
      case ({take_read,r_fire})
        2'b10: reads <= reads+1'b1;
        2'b01: reads <= reads-1'b1;
        default: ;
      endcase
      case ({popw,b_fire})
        2'b10: writes <= writes+1'b1;
        2'b01: writes <= writes-1'b1;
        default: ;
      endcase
      if (b_fire) n_wdone <= n_wdone+1'b1;
      if (r_fire && (m_rresp != 0 || !response_known || m_rlast != (response_count == 1))) n_err <= 1;
      if (b_fire && (m_bresp != 0 || (writes == 0 && !popw))) n_err <= 1;
      if (pushc && outside) n_err <= 1;
    end
  end

  if (PORDER) begin : g_pord
    logic [OW-1:0] rb [NBK];                       // read beats taken, not back, per bucket
    logic [OW-1:0] wb [NBK];                       // writes handed out, not answered, per bucket
    (* ram_style = "distributed" *) logic [KB-1:0] rbq [0:(1<<BP)-1];   // bursts' buckets, AR order
    (* ram_style = "distributed" *) logic [KB-1:0] wbq [0:(1<<OW)-1];   // writes' buckets, in order
    logic [BP-1:0] rq_p, rq_r;
    logic [OW-1:0] wq_p, wq_r;
    wire [KB-1:0] r_k = rbq[rq_r];                 // the returning burst's bucket
    wire [KB-1:0] b_k = wbq[wq_r];                 // the answered write's bucket
    always_ff @(posedge clk) begin
      if (rst) begin
        rq_p <= '0; rq_r <= '0; wq_p <= '0; wq_r <= '0; rbz <= '0; wbz <= '0;
        for (int k = 0; k < NBK; k++) begin rb[k] <= '0; wb[k] <= '0; end
      end else begin
        if (ar_fire) begin rbq[rq_p] <= bkt(ar_addr); rq_p <= rq_p + 1'b1; end
        // (a burst's last beat by its known length, as RLAST is checked: response_count, never
        // the slave's RLAST, which may be wrong: n_err)
        if (r_fire && response_known && response_count == 5'd1) rq_r <= rq_r + 1'b1;
        if (popw) begin wbq[wq_p] <= head_k; wq_p <= wq_p + 1'b1; end
        if (b_fire) wq_r <= wq_r + 1'b1;
        for (int k = 0; k < NBK; k++) begin
          automatic logic inc_r = take_read && head_k == KB'(k);
          automatic logic dec_r = r_fire && r_k == KB'(k);
          automatic logic inc_w = popw && head_k == KB'(k);
          automatic logic dec_w = b_fire && b_k == KB'(k);
          automatic logic [OW-1:0] nr = rb[k] + OW'(inc_r) - OW'(dec_r);
          automatic logic [OW-1:0] nw = wb[k] + OW'(inc_w) - OW'(dec_w);
          rb[k] <= nr; rbz[k] <= nr != 0;
          wb[k] <= nw; wbz[k] <= nw != 0;
        end
      end
    end
`ifndef SYNTHESIS
    // the buckets' sums are the totals (every read beat and write in exactly one bucket)
    always_ff @(posedge clk)
      if (!rst) begin
        automatic longint sr = 0, sww = 0;
        for (int k = 0; k < NBK; k++) begin sr += rb[k]; sww += wb[k]; end
        if (sr != reads || sww != writes)
          $fatal(1, "otpu_native_axi: page buckets %0d reads / %0d writes, totals %0d / %0d", sr, sww, reads, writes);
      end
`endif
  end else begin : g_nopord
    assign rbz = '0; assign wbz = '0;
  end
endmodule
