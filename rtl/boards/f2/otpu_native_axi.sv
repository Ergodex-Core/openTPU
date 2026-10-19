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
// or after READ_IDLE cycles with nothing to take. Writes stay single beats. Order: a read is
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
  parameter bit ORDER = 1'b1           // the direction barriers (0: tests only, tb_f2hbm_axi)
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
  wire adjacent = cq[cr][24:0] == next_addr && cq[cr][24:6] == pending_addr[24:6];
  wire take_read = !pending_valid && cn != 0 && !head_we && (writes == 0 || !ORDER) &&
                   reads < MAX_OUT && (pending_count == 0 || adjacent);
  wire [4:0] cnt_t = pending_count + 5'd1;
  wire full_t = cnt_t == 5'(READ_BURST) || cq[cr][5:0] == 6'd63 || reads == OW'(MAX_OUT-1);
  // Never wait for another command at a direction/page/discontinuity boundary. A short idle
  // timer also flushes isolated final reads.
  wire flush = pending_count != 0 && !pending_valid && !take_read &&
               (cn != 0 || idle_count == IW'(READ_IDLE-1) || reads == OW'(MAX_OUT));
  wire close = (take_read && full_t) || flush;
  wire ar_fire = m_arvalid && m_arready;
  wire ar_free = !ar_v || ar_fire;
  wire load_ar = (close || pending_valid) && ar_free;
  wire write_go = cn != 0 && head_we && wn != 0 && reads == 0 && writes < MAX_OUT;
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
        ar_cnt <= take_read ? cnt_t : pending_count;
        pending_count <= 0; pending_valid <= 0; idle_count <= 0;
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
endmodule
