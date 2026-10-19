// Unit test of the native DRAM adapter (rtl/mem/otpu_native_dram.sv) at NCH channels
// (tests/test_native_nch.py): random traffic on its three slice ports against a native memory
// model of NCH channels (random readies, latency and bubbles; each channel performs its commands
// in order, reads with no backpressure, n_wdone a few cycles late), checked
//   - against the address map: the channels' physical memories start as init(logical beat) under
//     the map this file states (lbeat: the inverse of otpu_native_dram's chan_of / cb_of, and of
//     opentpu/host/board.py's beat_at), so every read of a beat not yet written returns init of
//     its logical beat only if the adapter put it where the host expects it;
//   - against the ports' ordering contract: port B reads see port B's earlier writes (one queue
//     per channel, in order), port A reads a region nothing writes (runs, reuse, a_flush), SW and
//     A writes go to regions of their own; after wr_idle the physical memories hold, under the
//     map, every byte the three ports wrote;
//   - REF = 1 (NCH = 2): a copy of the two-channel adapter (otpu_native_dram_ref, the tree's
//     before the generalization: tests/test_native_nch.py makes it from git) runs on the same
//     inputs, and every output of the two must agree in every cycle.
// Plusargs: +seed=, +cycles=. Prints the counts and PASS, or the first errors and FAIL.
module tb_native_nch;
  parameter int NCH = 2;
  parameter bit REF = 1'b0;
  parameter bit CHASH = 1'b1;
  parameter bit MAPBUG = 1'b0;
  parameter bit HIGH = 1'b0;
  parameter bit WDQ = 1'b0;          // the write-done compare a register (the F2 HBM board's)         // port B's region above 4 GiB (NCH >= 4: a slice past 4 GiB)       // 1: this file's map leaves the pair's fold out (must FAIL)
  localparam int CW = $clog2(NCH), PIB = CW - 1, NP = NCH / 2;
  // regions, in words: A reads [0, RO), B [RO, 2 RO), SW writes [2 RO, 3 RO), A writes [3 RO, 4 RO)
  localparam int RO = 1 << 14;

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1;
  int unsigned seed = 1, cycles = 20000, errors = 0;
  longint cyc = 0;
  always_ff @(posedge clk) cyc <= cyc + 1;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("cycles=%d", cycles));
    void'($urandom(seed));
  end
  task automatic fail(input string s);
    errors++;
    if (errors <= 10) $display("ERROR @%0d: %s", cyc, s);
  endtask

  // ---------------------------------------------------------------- the map
  function automatic logic [24:0] fold(input logic [24:0] u);     // u's PIB-bit groups XORed
    logic [24:0] f;
    f = '0;
    if (PIB > 0) for (int k = 0; k < 27; k += PIB) f ^= (u >> k) & ((25'd1 << PIB) - 1);
    return f;
  endfunction
  // the logical beat (byte address / 64) on channel c at channel beat u
  function automatic longint lbeat(input int c, input logic [24:0] u);
    longint m;
    int p, h;
    p = c / 2; h = c % 2;
    m = longint'(u) * NP + longint'(CHASH && !MAPBUG ? (25'(p) ^ fold(u)) & 25'(NP - 1) : 25'(p));
    return 2 * m + (h ^ (CHASH ? $countones(m) % 2 : 0));
  endfunction
  function automatic logic [511:0] init(input longint b);
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = 32'(b) * 32'h9E37_79B9 ^ (i * 32'h85EB_CA6B);
    return v;
  endfunction

  // ---------------------------------------------------------------- the DUT
  logic a_rdy, a_req, a_we, a_rvalid, sw_rdy, sw_req, b_rdy, b_req, b_tag, b_we, b_par;
  logic b_rvalid, b_rtag, wr_idle, rd_idle, a_flush;
  logic [31:0] a_addr, a_wdata, sw_addr, sw_wdata, b_addr;
  logic [3:0] a_be, sw_be;
  logic [255:0] a_rdata;
  logic [31:0] b_wmask;
  logic [1023:0] b_wdata, b_rdata;
  logic [NCH-1:0] n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid;
  logic [NCH-1:0][24:0] n_caddr;
  logic [NCH-1:0][511:0] n_wdata, n_rdata;
  logic [NCH-1:0][63:0] n_wmask;
  logic [NCH-1:0][15:0] n_wdone;
  otpu_native_dram #(.NCH(NCH), .CHASH(CHASH), .WDQ(WDQ)) u_dut (
    .clk, .rst, .a_flush, .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr),
    .a_wdata_x(a_wdata), .a_be_x(a_be), .a_rvalid, .a_rdata, .sw_rdy, .sw_req, .sw_addr,
    .sw_wdata, .sw_be, .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par,
    .b_rvalid, .b_rtag, .b_rdata, .wr_idle, .rd_idle,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid,
    .n_rdata, .n_wdone);

  if (REF) begin : g_ref
    // the two-channel adapter on the same inputs: every output equal, every cycle
    logic r_a_rdy, r_a_rvalid, r_sw_rdy, r_b_rdy, r_b_rvalid, r_b_rtag, r_wr_idle, r_rd_idle;
    logic [255:0] r_a_rdata;
    logic [1023:0] r_b_rdata;
    logic [1:0] r_cvalid, r_cwe, r_wvalid;
    logic [1:0][24:0] r_caddr;
    logic [1:0][511:0] r_wdata;
    logic [1:0][63:0] r_wmask;
    otpu_native_dram_ref #(.CHASH(CHASH)) u_ref (
      .clk, .rst, .a_flush, .a_rdy_x(r_a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr),
      .a_wdata_x(a_wdata), .a_be_x(a_be), .a_rvalid(r_a_rvalid), .a_rdata(r_a_rdata),
      .sw_rdy(r_sw_rdy), .sw_req, .sw_addr, .sw_wdata, .sw_be, .b_rdy(r_b_rdy), .b_req, .b_tag,
      .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid(r_b_rvalid), .b_rtag(r_b_rtag),
      .b_rdata(r_b_rdata), .wr_idle(r_wr_idle), .rd_idle(r_rd_idle),
      .n_cvalid(r_cvalid), .n_cready(n_cready[1:0]), .n_cwe(r_cwe), .n_caddr(r_caddr),
      .n_wvalid(r_wvalid), .n_wready(n_wready[1:0]), .n_wdata(r_wdata), .n_wmask(r_wmask),
      .n_rvalid(n_rvalid[1:0]), .n_rdata(n_rdata[1:0]), .n_wdone(n_wdone[1:0]));
    always_ff @(negedge clk) if (!rst) begin
      if ({r_a_rdy, r_sw_rdy, r_b_rdy, r_wr_idle, r_rd_idle} !== {a_rdy, sw_rdy, b_rdy, wr_idle, rd_idle})
        fail("ref: a readiness or idle output differs");
      if (r_a_rvalid !== a_rvalid || (a_rvalid && r_a_rdata !== a_rdata)) fail("ref: port A data differs");
      if (r_b_rvalid !== b_rvalid || (b_rvalid && (r_b_rtag !== b_rtag || r_b_rdata !== b_rdata)))
        fail("ref: port B data differs");
      if (r_cvalid !== n_cvalid[1:0] || r_wvalid !== n_wvalid[1:0]) fail("ref: native valids differ");
      for (int c = 0; c < 2; c++) begin
        if (r_cvalid[c] && (r_cwe[c] !== n_cwe[c] || r_caddr[c] !== n_caddr[c]))
          fail($sformatf("ref: channel %0d command differs", c));
        if (r_wvalid[c] && (r_wdata[c] !== n_wdata[c] || r_wmask[c] !== n_wmask[c]))
          fail($sformatf("ref: channel %0d write data differs", c));
      end
    end
  end

  // ---------------------------------------------------------------- the channels (model)
  logic [511:0] pm [NCH][logic [24:0]];
  function automatic logic [511:0] prd(input int c, input logic [24:0] u);
    return pm[c].exists(u) ? pm[c][u] : init(lbeat(c, u));
  endfunction
  function automatic logic [511:0] merge(input logic [511:0] o, d, input logic [63:0] m);
    for (int i = 0; i < 64; i++) if (m[i]) o[8 * i +: 8] = d[8 * i +: 8];
    return o;
  endfunction
  typedef struct { logic we; logic [24:0] u; } ch_cmd_t;
  typedef struct { longint t; logic [511:0] d; } ch_rd_t;
  ch_cmd_t      ccq [NCH][$];
  logic [511:0] cwd [NCH][$];
  logic [63:0]  cwm [NCH][$];
  ch_rd_t       crq [NCH][$];
  longint       cwt [NCH][$];
  int           nrd_ch [NCH], nwr_ch [NCH];
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic crdy, wrdy, rv;
    logic [511:0] rd;
    logic [15:0] wdone;
    longint rlast;
    assign n_cready[c] = crdy;
    assign n_wready[c] = wrdy;
    assign n_rvalid[c] = rv;
    assign n_rdata[c] = rd;
    assign n_wdone[c] = wdone;
    always_ff @(posedge clk) begin
      crdy <= ccq[c].size() < 8 && $urandom % 4 != 0;
      wrdy <= cwd[c].size() < 8 && $urandom % 4 != 0;
      if (rst) begin
        ccq[c].delete(); cwd[c].delete(); cwm[c].delete(); crq[c].delete(); cwt[c].delete();
        rv <= 1'b0; wdone <= '0; rlast = 0;
      end else begin
        if (n_cvalid[c] && n_cready[c]) ccq[c].push_back('{n_cwe[c], n_caddr[c]});
        if (n_wvalid[c] && n_wready[c]) begin cwd[c].push_back(n_wdata[c]); cwm[c].push_back(n_wmask[c]); end
        // perform the oldest command (a write once its data is in), now and then
        if (ccq[c].size() && $urandom % 3 != 0 && (!ccq[c][0].we || cwd[c].size())) begin
          if (ccq[c][0].we) begin
            pm[c][ccq[c][0].u] = merge(prd(c, ccq[c][0].u), cwd[c][0], cwm[c][0]);
            void'(cwd[c].pop_front()); void'(cwm[c].pop_front());
            cwt[c].push_back(cyc + 2 + $urandom % 8);
            nwr_ch[c]++;
          end else begin
            automatic longint t = cyc + 4 + $urandom % 30;
            if (t < rlast) t = rlast;
            rlast = t;
            crq[c].push_back('{t, prd(c, ccq[c][0].u)});
            nrd_ch[c]++;
          end
          void'(ccq[c].pop_front());
        end
        if (crq[c].size() && crq[c][0].t <= cyc && $urandom % 5 != 0) begin
          rv <= 1'b1; rd <= crq[c][0].d; void'(crq[c].pop_front());
        end else rv <= 1'b0;
        if (cwt[c].size() && cwt[c][0] <= cyc) begin wdone <= wdone + 1'b1; void'(cwt[c].pop_front()); end
      end
    end
  end

  // ---------------------------------------------------------------- the slice's ports
  // the logical bytes the ports wrote (B's region, SW's, A's), and port B's view of its region
  logic [7:0]   lw [longint];
  logic [511:0] bview [longint];             // logical beat -> its contents as B left them
  function automatic logic [511:0] bv(input longint b);
    return bview.exists(b) ? bview[b] : init(b);
  endfunction
  typedef struct { logic tag; logic [1023:0] d; } bexp_t;
  bexp_t        bq_exp [$];
  logic [255:0] aq_exp [$];
  int           nb_rd = 0, nb_wr = 0, na_rd = 0, na_wr = 0, nsw = 0;
  bit           gen = 1'b0;
  logic [31:0]  a_next;                      // the next word of port A's current stream
  always_ff @(posedge clk) begin
    if (rst) begin
      a_req <= 1'b0; sw_req <= 1'b0; b_req <= 1'b0; a_flush <= 1'b0; a_next <= '0;
    end else begin
      a_flush <= gen && $urandom % 500 == 0;
      // ---- port B: whole chunks in its region; a write's word mask random
      if (b_req && b_rdy) begin
        if (b_we) begin
          for (int h = 0; h < 2; h++) begin
            automatic longint lb = longint'(b_addr[31:4]) + h;
            automatic logic [511:0] o = bv(lb);
            for (int w = 0; w < 16; w++)
              if (b_wmask[16 * h + w]) begin
                o[32 * w +: 32] = b_wdata[512 * h + 32 * w +: 32];
                for (int k = 0; k < 4; k++) lw[lb * 64 + 4 * w + k] = b_wdata[512 * h + 32 * w + 8 * k +: 8];
              end
            bview[lb] = o;
          end
          nb_wr++;
        end else begin
          bq_exp.push_back('{b_tag, {bv(longint'(b_addr[31:4]) + 1), bv(longint'(b_addr[31:4]))}});
          nb_rd++;
        end
      end
      if (!(b_req && !b_rdy)) begin
        b_req <= gen && $urandom % 3 == 0;
        b_we <= $urandom % 3 == 0;
        b_tag <= $urandom;
        begin
          automatic logic [31:0] w = (HIGH ? 32'h4000_0000 : 0) + RO + 32 * ($urandom % ($urandom % 2 ? 24 : RO / 32));
          b_addr <= w; b_par <= ^w[31:5];
        end
        for (int i = 0; i < 32; i++) b_wdata[32 * i +: 32] <= $urandom;
        b_wmask <= $urandom % 2 ? '1 : $urandom;
      end
      // ---- port A: streams of reads in its region (runs, reuse), writes in A's write region
      if (a_req && a_rdy) begin
        if (a_we) begin
          for (int k = 0; k < 4; k++) if (a_be[k]) lw[longint'(a_addr) * 4 + k] = a_wdata[8 * k +: 8];
          na_wr++;
        end else begin
          automatic logic [511:0] v = init(longint'(a_addr[29:4]));
          aq_exp.push_back(v[256 * a_addr[3] +: 256]);
          na_rd++;
        end
      end
      if (!(a_req && !a_rdy)) begin
        automatic bit we = $urandom % 16 == 0;
        a_req <= gen && $urandom % 2 == 0;
        a_we <= we;
        if ($urandom % 64 == 0) a_next <= 32'($urandom % RO);
        else a_next <= (a_next + 1 + ($urandom % 4 == 0 ? 7 : 0)) % RO;
        a_addr <= we ? 32'(3 * RO + $urandom % RO) : a_next;
        a_wdata <= $urandom; a_be <= $urandom;
      end
      // ---- port SW: word writes in its region (byte enables), bursts on few beats
      if (sw_req && sw_rdy) begin
        for (int k = 0; k < 4; k++) if (sw_be[k]) lw[longint'(sw_addr) * 4 + k] = sw_wdata[8 * k +: 8];
        nsw++;
      end
      if (!(sw_req && !sw_rdy)) begin
        sw_req <= gen && $urandom % 3 == 0;
        sw_addr <= 32'(2 * RO + ($urandom % 2 ? $urandom % 64 : 16 * ($urandom % (RO / 16)) + $urandom % 16));
        sw_wdata <= $urandom; sw_be <= $urandom;
      end
      // ---- read data
      if (b_rvalid) begin
        if (!bq_exp.size()) fail("B data with no read");
        else begin
          automatic bexp_t e = bq_exp.pop_front();
          if (b_rtag !== e.tag || b_rdata !== e.d) fail($sformatf("B read %0d: wrong data or tag", nb_rd));
        end
      end
      if (a_rvalid) begin
        if (!aq_exp.size()) fail("A data with no read");
        else begin
          automatic logic [255:0] e = aq_exp.pop_front();
          if (a_rdata !== e) fail("A read: wrong data (the map, or a run's beat)");
        end
      end
    end
  end

  initial begin
    repeat (5) @(posedge clk);
    @(negedge clk) rst = 1'b0;
    repeat (5) @(posedge clk);
    gen = 1'b1;
    repeat (cycles) @(posedge clk);
    gen = 1'b0;
    for (int t = 0; t < 100000; t++) begin
      @(posedge clk);
      if (!a_req && !b_req && !sw_req && wr_idle && rd_idle && !bq_exp.size() && !aq_exp.size()) break;
    end
    repeat (20) @(posedge clk);
    if (!wr_idle || !rd_idle || bq_exp.size() || aq_exp.size()) fail("did not drain");
    // every byte written, under the map, in the channels' memories
    begin
      int bad = 0;
      foreach (lw[a]) begin
        automatic longint lb = a / 64;
        automatic int c = -1;
        automatic logic [24:0] u = '0;
        // find the beat's channel: the adapter's chan_of / cb_of, inverted by lbeat's search
        for (int k = 0; k < NCH && c < 0; k++)
          if (lbeat(k, 25'((lb / 2) / NP)) == lb) begin c = k; u = 25'((lb / 2) / NP); end
        if (c < 0) begin bad++; continue; end
        if (prd(c, u)[8 * (a % 64) +: 8] !== lw[a]) bad++;
      end
      if (bad) fail($sformatf("%0d written bytes not where the map puts them", bad));
    end
    $display("NCH %0d: B %0d reads %0d writes, A %0d reads %0d writes, SW %0d writes", NCH, nb_rd,
             nb_wr, na_rd, na_wr, nsw);
    for (int c = 0; c < NCH; c++) $display("ch%0d: %0d reads %0d writes", c, nrd_ch[c], nwr_ch[c]);
    if (errors) $display("FAIL (%0d errors)", errors);
    else $display("PASS");
    $finish;
  end
endmodule
