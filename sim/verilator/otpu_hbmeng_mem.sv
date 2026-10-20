// The v5 F2 HBM memory path in the decode receipt (rtlsim OTPU_HBMENG=1; otpu_top HBMENG): the
// core-clock model of otpu_f2_hbm_eng + otpu_hbm_ch per channel in front of otpu_native_mem
// (the receipts' memory model: its LAT, +axi_lat, is the HBM port's own read latency, 280 ns =
// 56 cycles at 200 MHz). What is modelled, per slice (SID) and channel, all at the core clock:
//   - the SLR chains: NXR register stages each way (the takes with their payloads down, the
//     readies, read beats and write-done pulses up: otpu_f2_hbm_eng g_xr), and the per-slice
//     write bus (one channel's write beat a cycle: a round robin over the channels whose
//     engine shows room, the requests and the grant registered as in the RTL; n_wready[c] is
//     the registered grant, two cycles after the request);
//   - the engine's command queue (32 entries, XL = 2 NXR kept below its ready) and write queue
//     (16), the `cont` bit (the beat follows the last command's and is not the first of an
//     aligned BURST group), the open burst extended while `cont` and under BURST, closed by a
//     command that cannot join, by BURST, or by the idle timer (IDLE hclk cycles = ceil(IDLE /
//     2.25) core cycles with nothing queued), at most OT bursts in flight, the direction
//     barriers (a read burst only while no write lacks its B, a write only while no read beat
//     is outstanding);
//   - the structural cycles of the RTL, LAT_DN from the chain's end to the port's AR (the
//     command FIFO's crossing and output register, the generator, the AR register, the HDK's
//     register slice: ~7 hclk) and LAT_UP from the port's R to the chain's start (the slice,
//     the R register, the read path's input register, URAM write and read a cycle apart, the
//     URAM's two output registers, the second 256-bit beat, the crossing FIFO's input
//     register, crossing and output register: ~8 hclk + 4 clk; was ~3 hclk + 4 clk = 5 with
//     the block-RAM crossing FIFO of 6a52448c); the HBM's own latency is otpu_native_mem's LAT (+axi_lat) and its
//     write B otpu_native_mem's n_wdone lag plus WLAT (+hbm_wlat, 172 ns = 35 cycles at 200)
//     less LAT_DN;
//   - the port's beats: the native port behind the model performs at most a command a cycle,
//     which is the engine's own limit too (the read FIFO drains a 512-bit beat a core cycle; the
//     AXI side's 1.125 native beats a core cycle only fills that FIFO faster), so RATE (0: the
//     AXI side's 9 beats per 8 cycles, 1: a beat a cycle) governs only how early a burst's
//     beats reach otpu_native_mem's queue and changes no steady-state rate.
// At the dump, per channel: HBMENG ch<c> rd=<beats> wr=<beats> bursts=<n> beats=<n>
// inflight=<mean read beats in flight x1000> lengths <len>:<count> ... (rtlsim
// stats["hbmeng"]). Same images, pokes and dump as otpu_native_mem (it is the model behind).
module otpu_hbmeng_mem #(
  parameter int WORDS = 1 << 18,
  parameter int LAT   = 20,
  parameter int SID   = 0,
  parameter int NCH   = 2,
  parameter int NXR   = 3,               // SLR stages each way (the build's 3)
  parameter int BURST = 8,
  parameter int IDLE  = 2,               // hclk cycles
  parameter int OT    = 64,
  parameter int RATE  = 0,               // 0: 9 beats per 8 cycles into the model, 1: one a cycle
  parameter int LAT_DN = 3,              // core cycles, the chain's end to the port's AR
  parameter int LAT_UP = 7               // core cycles, the port's R to the chain's start
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  run_rst,
  input  logic [NCH-1:0]            n_cvalid,
  output logic [NCH-1:0]            n_cready,
  input  logic [NCH-1:0]            n_cwe,
  input  logic [NCH-1:0][24:0]      n_caddr,
  input  logic [NCH-1:0]            n_wvalid,
  output logic [NCH-1:0]            n_wready,
  input  logic [NCH-1:0][511:0]     n_wdata,
  input  logic [NCH-1:0][63:0]      n_wmask,
  output logic [NCH-1:0]            n_rvalid,
  output logic [NCH-1:0][511:0]     n_rdata,
  output logic [NCH-1:0][15:0]      n_wdone,
  input  logic                  dump
);
  localparam int XL = 2 * NXR + 2;     // (the chain each way, the arbiter's two registers)
  localparam int QN = 32, WN = 16;
  localparam int IDLEC = (IDLE * 4 + 8) / 9;          // ceil(IDLE / 2.25), at least 1
  localparam int IDLC = (IDLEC < 1) ? 1 : IDLEC;
  localparam int CW = (NCH > 1) ? $clog2(NCH) : 1;
  localparam int BW = (BURST > 1) ? $clog2(BURST) : 1;
  int wlat = 35;
  initial begin
    void'($value$plusargs("hbm_wlat=%d", wlat));
  end
  // the model's native ports (m_*), behind the engines
  logic [NCH-1:0]        m_cvalid, m_cready, m_cwe, m_wvalid, m_wready, m_rvalid;
  logic [NCH-1:0][24:0]  m_caddr;
  logic [NCH-1:0][511:0] m_wdata, m_rdata;
  logic [NCH-1:0][63:0]  m_wmask;
  logic [NCH-1:0][15:0]  m_wdone;
  otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .SID(SID), .NCH(NCH)) u_mem (
    .clk, .rst, .run_rst,
    .n_cvalid(m_cvalid), .n_cready(m_cready), .n_cwe(m_cwe), .n_caddr(m_caddr),
    .n_wvalid(m_wvalid), .n_wready(m_wready), .n_wdata(m_wdata), .n_wmask(m_wmask),
    .n_rvalid(m_rvalid), .n_rdata(m_rdata), .n_wdone(m_wdone), .dump);
  longint cyc = 0;
  always @(posedge clk) if (!rst) cyc <= cyc + 1;
  int nrd [NCH], nwr [NCH], nburst [NCH], nbeat [NCH], hist [NCH][9];
  longint infl [NCH];

  // ---- the write bus: one beat a cycle, round robin over the channels whose engine (as seen
  // through the chain) shows room; the engines' readies seen late, as the chain carries them
  logic [NCH-1:0] e_wready_x;          // the engines' write readies after the up chain
  logic [NCH-1:0] req_q, gnt, gnt_q;   // (as otpu_f2_hbm_eng: the requests registered, the
  logic [CW-1:0]  ptr;                 //  round robin, its grant registered: the ready)
  always_comb begin
    logic found;
    found = 1'b0; gnt = '0;
    for (int i = 0; i < NCH; i++) begin
      int c;
      c = (int'(ptr) + i) % NCH;
      if (!found && req_q[c]) begin found = 1'b1; gnt[c] = 1'b1; end
    end
  end
  assign n_wready = gnt_q;
  always_ff @(posedge clk)
    if (rst) begin ptr <= '0; req_q <= '0; gnt_q <= '0; end
    else begin
      req_q <= n_wvalid & e_wready_x;
      gnt_q <= gnt;
      for (int c = 0; c < NCH; c++) if (gnt[c]) ptr <= CW'(c + 1);
    end

  for (genvar c = 0; c < NCH; c++) begin : g_ch
    // ---- the SLR chain: down the takes with their payloads, up the readies and responses
    localparam int DW = 1 + 1 + 25 + 1 + 512 + 64, UW = 1 + 1 + 1 + 512 + 1;
    logic [NXR:0][DW-1:0] dn;
    logic [NXR:0][UW-1:0] up;
    logic b_cvalid, b_cready, b_cwe, b_wvalid, b_wready, b_rvalid, b_wdp;
    logic [24:0] b_caddr;
    logic [511:0] b_wdata, b_rdata;
    logic [63:0] b_wmask;
    assign dn[NXR] = {n_cvalid[c] && n_cready[c], n_cwe[c], n_caddr[c], n_wvalid[c] && n_wready[c],
                      n_wdata[c], n_wmask[c]};
    assign {b_cvalid, b_cwe, b_caddr, b_wvalid, b_wdata, b_wmask} = dn[0];
    assign up[0] = {b_cready, b_wready, b_rvalid, b_rdata, b_wdp};
    assign {n_cready[c], e_wready_x[c], n_rvalid[c], n_rdata[c]} = up[NXR][UW-1:1];
    for (genvar k = 0; k < NXR; k++) begin : g_x
      always_ff @(posedge clk) begin dn[k] <= dn[k + 1]; up[k + 1] <= up[k]; end
    end
    always_ff @(posedge clk)                     // the write count from the pulses (the wrapper's)
      if (rst) n_wdone[c] <= '0;
      else if (up[NXR][0]) n_wdone[c] <= n_wdone[c] + 1'b1;
    // ---- the engine's front: the queues with XL slack, the cont bit
    logic [26:0]  q [QN];                        // {cont, we, beat}
    logic [575:0] wq [WN];
    logic [5:0]   qh, qt, qn;
    logic [4:0]   wh, wt, wn;
    logic         last_v, last_we;
    logic [24:0]  next_beat;
    wire          cont = (BURST > 1) && !b_cwe && last_v && !last_we && (b_caddr == next_beat) &&
                         (b_caddr[BW-1:0] != '0);
    assign b_cready = !rst && qn < 6'(QN - XL);
    assign b_wready = !rst && wn < 5'(WN - XL);
    // ---- the generator: the open burst, the issued bursts' native commands (a queue of
    // {we, beat, due}: the HBM port sees them at due), the barriers and counts
    logic         ob_v;
    logic [24:0]  ob_beat;
    logic [3:0]   ob_len;
    int           idle;
    int           rd_ot;                         // bursts issued, their last beat not back
    int           rd_res;                        // read beats popped, not yet delivered up
    int           wr_pend;                       // writes popped, their B not yet back
    // the native commands towards the model, in order, with the time they may go
    localparam int CN = 1024;
    logic [25:0]  cq [CN];                       // {we, beat}
    longint       cdue [CN];
    logic [9:0]   ch_, ct;
    int           cn;
    longint       tok;                           // the port's beat tokens x8 (RATE 0: +9 per 8 cycles)
    // the model's read beats, delayed LAT_UP, and its write-done count's lag into WLAT
    logic [511:0] rd_d [64];
    longint       rd_due [64];
    logic [5:0]   rh, rt;
    int           rn;
    logic [15:0]  wd_seen;
    longint       bdue [64];
    logic [5:0]   bh, bt;
    int           bn;
    wire h_v   = qn != 0;
    wire h_we  = q[qh][25];
    wire h_cont = q[qh][26];
    wire [24:0] h_beat = q[qh][24:0];
    wire ext   = h_v && !h_we && h_cont && ob_v && ob_len < 4'(BURST) && (idle < IDLC);
    wire start = h_v && !h_we && !ob_v && wr_pend == 0;
    wire close = ob_v && !ext && (ob_len == 4'(BURST) || idle >= IDLC || (h_v && !(!h_we && h_cont))) &&
                 rd_ot < OT;
    wire wgo   = h_v && h_we && wn != 0 && !ob_v && rd_res == 0 && rd_ot == 0 && cn < CN - 2;
    wire pop   = ext || start || wgo;
    wire c_go  = cn != 0 && cdue[ch_] <= cyc && m_cready[c] && tok >= 8;
    assign m_cvalid[c] = c_go;
    assign {m_cwe[c], m_caddr[c]} = cq[ch_];
    assign m_wvalid[c] = c_go && m_cwe[c];
    assign {m_wmask[c], m_wdata[c]} = wq[wh];
    assign b_rvalid = rn != 0 && rd_due[rh] <= cyc;
    assign b_rdata = rd_d[rh];
    assign b_wdp = bn != 0 && bdue[bh] <= cyc;
    always_ff @(posedge clk) begin
      if (rst) begin
        qh <= '0; qt <= '0; qn <= '0; wh <= '0; wt <= '0; wn <= '0;
        last_v <= 1'b0; last_we <= 1'b0; next_beat <= '0;
        ob_v <= 1'b0; ob_beat <= '0; ob_len <= '0; idle <= 0; rd_res <= 0; wr_pend <= 0;
        ch_ <= '0; ct <= '0; cn <= 0; tok <= 0;
        rh <= '0; rt <= '0; rn <= 0; wd_seen <= '0; bh <= '0; bt <= '0; bn <= 0;
      end else begin
        // the chain's takes land
        if (b_cvalid) begin
          q[qt[4:0]] <= {cont, b_cwe, b_caddr}; qt <= qt + 1'b1;
          last_v <= 1'b1; last_we <= b_cwe; next_beat <= b_caddr + 1'b1;
        end
        if (b_wvalid) begin wq[wt[3:0]] <= {b_wmask, b_wdata}; wt <= wt + 1'b1; end
        qn <= qn + 6'(b_cvalid) - 6'(pop);
        wn <= wn + 5'(b_wvalid) - 5'(wgo);
        if (pop) qh <= qh + 1'b1;
        // the open burst
        if (start) begin ob_v <= 1'b1; ob_beat <= h_beat; ob_len <= 4'd1; idle <= 0; end
        else if (ext) begin ob_len <= ob_len + 1'b1; idle <= 0; end
        else if (ob_v && !h_v) idle <= idle + 1;
        if (close) begin                         // the AR: its beats towards the port, LAT_DN on
          ob_v <= 1'b0; idle <= 0;
          for (int i = 0; i < 8; i++)
            if (i < int'(ob_len)) begin
              cq[10'(ct + 10'(i))] <= {1'b0, ob_beat + 25'(i)};
              cdue[10'(ct + 10'(i))] <= cyc + LAT_DN;
            end
          ct <= ct + 10'(ob_len);
          nburst[c] <= nburst[c] + 1; nbeat[c] <= nbeat[c] + int'(ob_len);
          hist[c][ob_len] <= hist[c][ob_len] + 1;
        end
        if (wgo) begin                           // the write: AW + W, LAT_DN on
          cq[ct] <= {1'b1, h_beat}; cdue[ct] <= cyc + LAT_DN; ct <= ct + 1'b1;
        end
        cn <= cn + (close ? int'(ob_len) : 0) + int'(wgo) - int'(c_go);
        // (rd_ot, the bursts in flight, is kept below against the model's returns)
        rd_res <= rd_res + (ext || start ? 1 : 0) - int'(b_rvalid);
        wr_pend <= wr_pend + int'(wgo) - int'(b_wdp);
        // the native commands out: a beat a cycle at most, under the port's tokens
        if (c_go) begin
          ch_ <= ch_ + 1'b1;
          if (m_cwe[c]) begin wh <= wh + 1'b1; nwr[c] <= nwr[c] + 1; end
          else nrd[c] <= nrd[c] + 1;
        end
        tok <= (c_go ? tok - 8 : tok) + ((RATE == 0) ? 9 : 8);
        if (tok > 64) tok <= 64;
        // the model's read beats back, LAT_UP later, in order
        if (m_rvalid[c]) begin
          rd_d[rt] <= m_rdata[c]; rd_due[rt] <= cyc + LAT_UP; rt <= rt + 1'b1;
        end
        rn <= rn + int'(m_rvalid[c]) - int'(b_rvalid);
        if (b_rvalid) rh <= rh + 1'b1;
        // the model's write-done steps into B's, WLAT - LAT_DN later (the model's own lag is
        // its n_wdone's +4..11 cycles)
        if (m_wdone[c] != wd_seen) begin
          for (int i = 0; i < 16; i++)
            if (i < int'(16'(m_wdone[c] - wd_seen))) bdue[6'(bt + 6'(i))] <= cyc + ((wlat > LAT_DN + 4) ? wlat - LAT_DN - 4 : 0);
          bt <= bt + 6'(16'(m_wdone[c] - wd_seen));
          bn <= bn + int'(16'(m_wdone[c] - wd_seen)) - int'(b_wdp);
          wd_seen <= m_wdone[c];
        end else bn <= bn - int'(b_wdp);
        if (b_wdp) bh <= bh + 1'b1;
        infl[c] <= infl[c] + rd_res;
      end
    end
    // rd_ot: bursts whose last beat the model has not yet returned (its reads come back in
    // order, each burst's beats consecutively): the model's returns counted against the
    // bursts' lengths in order
    logic [3:0] blen [128];
    logic [6:0] blh, blt;
    logic [3:0] brem;
    always_ff @(posedge clk) begin
      if (rst) begin blh <= '0; blt <= '0; brem <= '0; rd_ot <= 0; end
      else begin
        if (close) begin blen[blt] <= ob_len; blt <= blt + 1'b1; end
        if (m_rvalid[c]) begin
          if (brem == 4'd0) begin
            if (blen[blh] == 4'd1) begin blh <= blh + 1'b1; rd_ot <= rd_ot + int'(close) - 1; end
            else begin brem <= blen[blh] - 4'd1; rd_ot <= rd_ot + int'(close); end
          end else if (brem == 4'd1) begin
            brem <= 4'd0; blh <= blh + 1'b1; rd_ot <= rd_ot + int'(close) - 1;
          end else begin brem <= brem - 1'b1; rd_ot <= rd_ot + int'(close); end
        end else rd_ot <= rd_ot + int'(close);
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk) if (!rst) begin
      if (b_cvalid && qn >= 6'(QN)) $fatal(1, "otpu_hbmeng_mem ch%0d: command queue overflow", c);
      if (b_wvalid && wn >= 5'(WN)) $fatal(1, "otpu_hbmeng_mem ch%0d: write queue overflow", c);
      if (m_rvalid[c] && rn >= 64) $fatal(1, "otpu_hbmeng_mem ch%0d: read return queue overflow", c);
    end
`endif
  end
  always @(posedge clk) if (dump)
    for (int c = 0; c < NCH; c++) begin
      $write("HBMENG ch%0d rd=%0d wr=%0d bursts=%0d beats=%0d inflight=%0d lengths", c, nrd[c], nwr[c],
             nburst[c], nbeat[c], (cyc > 0) ? (infl[c] * 1000 / cyc) : 0);
      for (int i = 1; i <= 8; i++) if (hist[c][i]) $write(" %0d:%0d", i, hist[c][i]);
      $display("");
    end
endmodule
