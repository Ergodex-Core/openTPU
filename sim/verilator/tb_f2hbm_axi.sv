// The F2 HBM build's memory path since the alex-shell rework (tests/test_f2_rtl.py
// test_f2_hbm_axi): otpu_f2_hbm_axi (NPORT channels' native ports across the NXR-stage SLR chain
// into otpu_native_axi each, the PCIS through otpu_f2_pcis_gate) in front of a model of what the
// CL puts behind it: the SmartConnect fabric (encrypted IP: modelled by its behaviour) and the
// HBM. Per channel p:
//   - the channel's master (core clock) and the host's branch (aclk; the host's 1 x NPORT demux
//     by address bits 33:29) into port p's SmartConnect: each AR / AW taken with random readies
//     in its own clock, reaching the HBM side after a clock-crossing latency (LREQ + jitter),
//     in arrival order (the SmartConnect's master is single-threaded: one ID on the HBM port);
//     a write's service time up to 4 LREQ longer than a read's, so reads overtake writes;
//   - the HBM (hclk): reads in order, writes in order, the two directions interleaved in any
//     order (the controller's), each after a random service time; a 512-bit beat is two 256-bit
//     HBM beats (the width conversion: a beat back every 2 hclk cycles at most per port);
//     responses back to their slave after LRSP + jitter, in order per slave;
//   - the host's responses in its own order across ports (SmartConnect's same-ID order).
// Checked: the native contract (every read returns the beat as the channel's earlier commands
// left it; n_wdone reaches the writes; afterwards every write is in the HBM at p x 512 MiB +
// beat x 64), every channel transaction inside its 512 MiB, INCR, at most 16 beats, within
// 4 KiB, ID 0; streams coalesce into bursts (the histogram printed; some bursts longer than one
// beat); the host's writes land, its reads (issued right after its writes are handed over,
// before their B: the gate must hold them) return its own writes, B / R in order with its IDs;
// pcis_wbusy high while a host write is unanswered. Plusargs +seed= +cycles= +nb=.
module tb_f2hbm_axi;
  parameter int NPORT = 8;
  parameter int NXR = 3;
  parameter int CORE_HALF = 5, AXI_HALF = 4, HBM_HALF = 2;   // 100 / 125 / 250: the ratios
  parameter int LREQ = 40, LRSP = 40;     // the SmartConnect's crossing latencies (time units)
  parameter int READ_BURST = 4, READ_IDLE = 1;
  parameter bit ORDER = 1'b1;             // 0: the order rules off (must fail)

  logic clk = 1'b0, aclk = 1'b0, hclk = 1'b0;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;
  always #(HBM_HALF) hclk = !hclk;
  logic rst = 1'b1, arst = 1'b1;

  int unsigned seed = 1, cycles = 20000, nb = 6, errors = 0;
  bit gen = 1'b0;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("cycles=%d", cycles));
    void'($value$plusargs("nb=%d", nb));
    void'($urandom(seed));
  end
  task automatic fail(input string s);
    errors++;
    if (errors <= 10) $display("ERROR @%0t: %s", $time, s);
  endtask
  function automatic logic [511:0] init(input logic [27:0] b);    // a 64-byte beat's first contents
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = b * 32'h9E37_79B9 ^ (i * 32'h85EB_CA6B);
    return v;
  endfunction
  function automatic logic [511:0] rnd512();
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = $urandom;
    return v;
  endfunction
  function automatic logic [511:0] merge(input logic [511:0] o, d, input logic [63:0] m);
    for (int i = 0; i < 64; i++) if (m[i]) o[8 * i +: 8] = d[8 * i +: 8];
    return o;
  endfunction

  // ---------------------------------------------------------------- the DUT
  logic [NPORT-1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NPORT-1:0][24:0]  n_caddr;
  logic [NPORT-1:0][511:0] n_wdata, n_rdata;
  logic [NPORT-1:0][63:0]  n_wmask;
  logic [NPORT-1:0][15:0]  n_wdone;
  logic [15:0]  s_awid, s_arid, s_bid, s_rid;
  logic [63:0]  s_awaddr, s_araddr;
  logic [7:0]   s_awlen, s_arlen;
  logic         s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid, s_bready;
  logic         s_arvalid, s_arready, s_rvalid, s_rready, s_rlast, pcis_wbusy;
  logic [511:0] s_wdata, s_rdata;
  logic [63:0]  s_wstrb;
  logic [1:0]   s_bresp, s_rresp;
  logic [15:0]  h_awid, h_arid, h_bid, h_rid;
  logic [63:0]  h_awaddr, h_araddr;
  logic [7:0]   h_awlen, h_arlen;
  logic [2:0]   h_awsize, h_arsize;
  logic [1:0]   h_awburst, h_arburst, h_bresp, h_rresp;
  logic         h_awvalid, h_awready, h_wvalid, h_wready, h_wlast, h_bvalid, h_bready;
  logic         h_arvalid, h_arready, h_rvalid, h_rready, h_rlast;
  logic [511:0] h_wdata, h_rdata;
  logic [63:0]  h_wstrb;
  logic [NPORT-1:0][33:0]  c_awaddr, c_araddr;
  logic [NPORT-1:0][7:0]   c_arlen;
  logic [NPORT-1:0]        c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic [NPORT-1:0]        c_arvalid, c_arready, c_rvalid, c_rlast, c_rready;
  logic [NPORT-1:0][511:0] c_wdata, c_rdata;
  logic [NPORT-1:0][63:0]  c_wstrb;
  logic [NPORT-1:0][1:0]   c_bresp, c_rresp;
  otpu_f2_hbm_axi #(.NPORT(NPORT), .NXR(NXR), .READ_BURST(READ_BURST), .READ_IDLE(READ_IDLE),
                  .ORDER(ORDER)) u_dut (
    .clk, .rst, .aclk, .arst,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid,
    .n_rdata, .n_wdone, .n_err,
    .pcis_awid(s_awid), .pcis_awaddr(s_awaddr), .pcis_awlen(s_awlen), .pcis_awsize(3'd6),
    .pcis_awburst(2'b01), .pcis_awvalid(s_awvalid), .pcis_awready(s_awready),
    .pcis_wdata(s_wdata), .pcis_wstrb(s_wstrb), .pcis_wlast(s_wlast), .pcis_wvalid(s_wvalid),
    .pcis_wready(s_wready), .pcis_bid(s_bid), .pcis_bresp(s_bresp), .pcis_bvalid(s_bvalid),
    .pcis_bready(s_bready), .pcis_arid(s_arid), .pcis_araddr(s_araddr), .pcis_arlen(s_arlen),
    .pcis_arsize(3'd6), .pcis_arburst(2'b01), .pcis_arvalid(s_arvalid),
    .pcis_arready(s_arready), .pcis_rid(s_rid), .pcis_rdata(s_rdata), .pcis_rresp(s_rresp),
    .pcis_rlast(s_rlast), .pcis_rvalid(s_rvalid), .pcis_rready(s_rready), .pcis_wbusy,
    .h_awid, .h_awaddr, .h_awlen, .h_awsize, .h_awburst, .h_awvalid, .h_awready, .h_wdata,
    .h_wstrb, .h_wlast, .h_wvalid, .h_wready, .h_bid, .h_bresp, .h_bvalid, .h_bready, .h_arid,
    .h_araddr, .h_arlen, .h_arsize, .h_arburst, .h_arvalid, .h_arready, .h_rid, .h_rdata,
    .h_rresp, .h_rlast, .h_rvalid, .h_rready,
    .ch_awaddr(c_awaddr), .ch_awvalid(c_awvalid), .ch_awready(c_awready), .ch_wdata(c_wdata),
    .ch_wstrb(c_wstrb), .ch_wvalid(c_wvalid), .ch_wready(c_wready), .ch_bvalid(c_bvalid),
    .ch_bresp(c_bresp), .ch_bready(c_bready), .ch_araddr(c_araddr), .ch_arlen(c_arlen),
    .ch_arvalid(c_arvalid), .ch_arready(c_arready), .ch_rdata(c_rdata), .ch_rvalid(c_rvalid),
    .ch_rlast(c_rlast), .ch_rresp(c_rresp), .ch_rready(c_rready));

  // ---------------------------------------------------------------- the HBM's memory
  logic [511:0] mem [logic [27:0]];          // 64-byte beats of the 16 GiB
  function automatic logic [511:0] rd(input logic [27:0] b);
    return mem.exists(b) ? mem[b] : init(b);
  endfunction

  // ---------------------------------------------------------------- the SmartConnects + HBM
  // a transaction on port p: its slave (0 the channel, 1 the host), first beat, beats, the time it
  // reaches the HBM side; host transactions carry their sequence number (the host's order)
  typedef struct { bit src; logic [27:0] b; int n; longint due; int seq; } tx_t;
  typedef struct { logic [511:0] d; logic [63:0] m; } wb_t;
  typedef struct { logic [511:0] d; bit last; longint due; int seq; } rsp_t;
  tx_t  arq [NPORT][$], awq [NPORT][$];
  wb_t  wq [NPORT][2][$];                    // write beats per slave, in AW order
  rsp_t rq [NPORT][2][$];                    // read beats back to each slave
  rsp_t bq [NPORT][2][$];                    // B's back to each slave
  int   hist [17];
  int   nbursts = 0, nlong = 0;
  for (genvar p = 0; p < NPORT; p++) begin : g_port
    longint rnext = 0;
    // the HBM side (hclk): one read and one write may complete a cycle, in order per direction
    always @(posedge hclk) if (!rst) begin
      if (arq[p].size() && arq[p][0].due <= $time && $urandom % 3 != 0) begin
        automatic tx_t t = arq[p].pop_front();
        for (int i = 0; i < t.n; i++) begin
          automatic longint d = $time + LRSP + $urandom % 16;
          if (d < rnext + 4 * HBM_HALF) d = rnext + 4 * HBM_HALF;     // 2 HBM beats a 512-bit beat
          rnext = d;
          rq[p][t.src].push_back('{rd(t.b + 28'(i)), i == t.n - 1, d, t.seq});
        end
      end
      if (awq[p].size() && awq[p][0].due <= $time && wq[p][awq[p][0].src].size() >= awq[p][0].n
          && $urandom % 3 != 0) begin
        automatic tx_t t = awq[p].pop_front();
        for (int i = 0; i < t.n; i++) begin
          automatic wb_t w = wq[p][t.src].pop_front();
          mem[t.b + 28'(i)] = merge(rd(t.b + 28'(i)), w.d, w.m);
        end
        bq[p][t.src].push_back('{'0, 1'b1, $time + LRSP + $urandom % 16, t.seq});
      end
    end
    // the channel's slave (core clock)
    always @(posedge clk) begin
      c_awready[p] <= $urandom % 4 != 0;
      c_wready[p] <= $urandom % 4 != 0;
      c_arready[p] <= $urandom % 4 != 0;
      if (!rst) begin
        if (c_arvalid[p] && c_arready[p]) begin
          if (c_araddr[p][33:29] != 5'(p)) fail($sformatf("ch%0d: a read outside its pseudo-channel", p));
          if (c_araddr[p][5:0] != 0 || c_arlen[p] > 15 || c_araddr[p][11:6] + c_arlen[p] > 63)
            fail($sformatf("ch%0d: AR unaligned, longer than 16 or across 4 KiB", p));
          hist[c_arlen[p] + 1]++; nbursts++; if (c_arlen[p] != 0) nlong++;
          arq[p].push_back('{1'b0, c_araddr[p][33:6], c_arlen[p] + 1, $time + LREQ + $urandom % 32, 0});
        end
        if (c_awvalid[p] && c_awready[p]) begin
          if (c_awaddr[p][33:29] != 5'(p)) fail($sformatf("ch%0d: a write outside its pseudo-channel", p));
          awq[p].push_back('{1'b0, c_awaddr[p][33:6], 1, $time + LREQ + $urandom % (4 * LREQ), 0});
        end
        if (c_wvalid[p] && c_wready[p]) wq[p][0].push_back('{c_wdata[p], c_wstrb[p]});
        if ((c_rvalid[p] && !c_rready[p]) || (c_bvalid[p] && !c_bready[p]))
          fail($sformatf("ch%0d: R / B not ready", p));
      end
    end
    // R and B from registers (the slave is always ready for them: checked above)
    assign c_rresp[p] = 2'b00;
    assign c_bresp[p] = 2'b00;
    always @(posedge clk) begin
      c_rvalid[p] <= 1'b0; c_bvalid[p] <= 1'b0;
      if (!rst && rq[p][0].size() && rq[p][0][0].due <= $time) begin
        c_rvalid[p] <= 1'b1; c_rdata[p] <= rq[p][0][0].d; c_rlast[p] <= rq[p][0][0].last;
        void'(rq[p][0].pop_front());
      end
      if (!rst && bq[p][0].size() && bq[p][0][0].due <= $time) begin
        c_bvalid[p] <= 1'b1; void'(bq[p][0].pop_front());
      end
    end
  end

  // the host's branch (aclk): the demux by address, the host's own order back
  int hseq_r = 0, hseq_b = 0, hnext_r = 0, hnext_b = 0, hwport [$];
  wb_t hwpend [$];
  logic [15:0] hr_id_q [$], hb_id_q [$];
  always @(posedge aclk) begin
    h_awready <= $urandom % 4 != 0;
    h_wready <= $urandom % 4 != 0;
    h_arready <= $urandom % 4 != 0;
    if (!arst) begin
      if (h_arvalid && h_arready) begin
        automatic int p = h_araddr[33:29];
        if (h_araddr[63:34] != 0 || p >= NPORT) fail("host: read outside the channels");
        else arq[p].push_back('{1'b1, h_araddr[33:6], h_arlen + 1, $time + LREQ + $urandom % 32, hseq_r++});
      end
      if (h_awvalid && h_awready) begin
        automatic int p = h_awaddr[33:29];
        if (h_awaddr[63:34] != 0 || p >= NPORT) fail("host: write outside the channels");
        else begin
          awq[p].push_back('{1'b1, h_awaddr[33:6], h_awlen + 1, $time + LREQ + $urandom % (4 * LREQ), hseq_b++});
          for (int i = 0; i <= h_awlen; i++) hwport.push_back(p);
        end
      end
      if (h_wvalid && h_wready) hwpend.push_back('{h_wdata, h_wstrb});   // W may lead its AW
      while (hwport.size() && hwpend.size()) wq[hwport.pop_front()][1].push_back(hwpend.pop_front());
      // R and B from registers, held until taken; the next of the host's order loaded
      if (h_arvalid && h_arready) hr_id_q.push_back(h_arid);
      if (h_awvalid && h_awready) hb_id_q.push_back(h_awid);
      if (!h_rvalid || h_rready) begin
        h_rvalid <= 1'b0;
        for (int p = 0; p < NPORT; p++)
          if (rq[p][1].size() && rq[p][1][0].seq == hnext_r && rq[p][1][0].due <= $time) begin
            h_rvalid <= 1'b1; h_rdata <= rq[p][1][0].d; h_rlast <= rq[p][1][0].last;
            h_rid <= hr_id_q[0];
            if (rq[p][1][0].last) begin hnext_r++; void'(hr_id_q.pop_front()); end
            void'(rq[p][1].pop_front());
            break;
          end
      end
      if (!h_bvalid || h_bready) begin
        h_bvalid <= 1'b0;
        for (int p = 0; p < NPORT; p++)
          if (bq[p][1].size() && bq[p][1][0].seq == hnext_b && bq[p][1][0].due <= $time) begin
            h_bvalid <= 1'b1; h_bid <= hb_id_q[0];
            hnext_b++; void'(hb_id_q.pop_front());
            void'(bq[p][1].pop_front());
            break;
          end
      end
    end else begin
      h_rvalid <= 1'b0; h_bvalid <= 1'b0;
    end
  end
  assign h_rresp = 2'b00;
  assign h_bresp = 2'b00;

  // ---------------------------------------------------------------- the native masters (clk)
  logic [511:0] sh [NPORT][logic [24:0]];
  typedef struct { logic we; logic [24:0] a; } cmd_t;
  cmd_t         cq [NPORT][$];
  logic [511:0] dq [NPORT][$], eq [NPORT][$];
  logic [63:0]  mq [NPORT][$];
  int           nwr [NPORT], nrd [NPORT];
  function automatic logic [511:0] nview(input int c, input logic [24:0] a);
    return sh[c].exists(a) ? sh[c][a] : rd({5'(c), 23'(a)});
  endfunction
  for (genvar c = 0; c < NPORT; c++) begin : g_m
    logic [24:0] run;                     // a stream: consecutive beats (bursts)
    int          slen;                    // the stream's beats still to come
    logic        swe;
    always_ff @(posedge clk) begin
      if (rst) begin
        n_cvalid[c] <= 1'b0; n_wvalid[c] <= 1'b0; run <= '0; slen <= 0;
      end else begin
        if (gen && cq[c].size() < 24 && $urandom % 2 == 0) begin
          automatic cmd_t x;
          x.we = $urandom % 3 == 0;
          if (slen != 0) begin               // a stream: consecutive beats, one direction
            x.we = swe; x.a = 25'(8192 + run); run <= run + 1; slen <= slen - 1;
          end else case ($urandom % 8)
            0, 1: x.a = 25'(1000 + $urandom % 4096);
            2, 3: begin
              x.a = 25'(8192 + run); run <= run + 1; slen <= 3 + $urandom % 30; swe <= $urandom % 4 == 0;
              x.we = swe;
            end
            default: x.a = 25'($urandom % nb);
          endcase
          if (x.we) begin
            automatic logic [511:0] d = rnd512();
            automatic logic [63:0]  m = ($urandom % 3 == 0) ? {$urandom, $urandom} : '1;
            sh[c][x.a] = merge(nview(c, x.a), d, m);
            dq[c].push_back(d); mq[c].push_back(m);
            nwr[c]++;
          end else begin
            eq[c].push_back(nview(c, x.a));
            nrd[c]++;
          end
          cq[c].push_back(x);
        end
        // the adapter's port: a command or a beat shown, taken on its ready (held until then,
        // or, with the SLR chain, taken the cycle it is shown with the ready it sees)
        if (n_cvalid[c] && n_cready[c]) begin void'(cq[c].pop_front()); n_cvalid[c] <= 1'b0; end
        else if (!n_cvalid[c] && cq[c].size() && $urandom % 8 != 0) begin
          n_cvalid[c] <= 1'b1; n_cwe[c] <= cq[c][0].we; n_caddr[c] <= cq[c][0].a;
        end
        if (n_wvalid[c] && n_wready[c]) begin
          void'(dq[c].pop_front()); void'(mq[c].pop_front()); n_wvalid[c] <= 1'b0;
        end else if (!n_wvalid[c] && dq[c].size() && $urandom % 8 != 0) begin
          n_wvalid[c] <= 1'b1; n_wdata[c] <= dq[c][0]; n_wmask[c] <= mq[c][0];
        end
        if (n_rvalid[c]) begin
          if (eq[c].size() == 0) fail($sformatf("ch%0d: read data with no read", c));
          else begin
            automatic logic [511:0] e = eq[c].pop_front();
            if (n_rdata[c] !== e) fail($sformatf("ch%0d: read returned stale or wrong data", c));
          end
        end
        if (n_err[c]) fail($sformatf("ch%0d: n_err", c));
      end
    end
  end

  // ---------------------------------------------------------------- the host (aclk)
  // its beats: the upper half of each port's 512 MiB (the channels use the lower)
  logic [511:0] hsh [logic [27:0]];
  typedef struct { logic [15:0] id; logic [63:0] a; logic [7:0] len; } hx_t;
  hx_t          haw[$], har[$], hb_exp[$], hr_exp[$];
  logic [511:0] hw[$], hre[$];
  logic [63:0]  hm[$];
  logic         hl[$];
  int           hwr = 0, hrd = 0, hr_beat = 0, hb_out = 0, wbusy_low = 0;
  logic [2:0]   hb_age;
  logic [63:0]  hlast = 64'h30000000;   // the last write's address (half the reads go there)
  function automatic logic [511:0] hview(input logic [27:0] b);
    return hsh.exists(b) ? hsh[b] : rd(b);
  endfunction
  function automatic logic [63:0] haddr();
    automatic int p = $urandom % NPORT;
    return {30'd0, 5'(p), 1'b1, 22'(64 * ($urandom % 6)), 6'd0};
  endfunction
  always_ff @(posedge aclk) begin
    if (arst) begin
      s_awvalid <= 1'b0; s_wvalid <= 1'b0; s_arvalid <= 1'b0; s_bready <= 1'b0; s_rready <= 1'b0;
      hb_age <= '0;
    end else begin
      s_bready <= $urandom % 3 != 0;
      s_rready <= $urandom % 3 != 0;
      if (gen && hr_exp.size() == 0 && haw.size() < 4 && $urandom % 12 == 0) begin
        automatic hx_t x;
        x.id = $urandom; x.len = $urandom % 4; x.a = haddr();
        for (int b = 0; b <= x.len; b++) begin
          automatic logic [511:0] d = rnd512();
          automatic logic [63:0]  m = ($urandom % 2) ? {$urandom, $urandom} : '1;
          hsh[x.a[33:6] + b] = merge(hview(x.a[33:6] + b), d, m);
          hw.push_back(d); hm.push_back(m); hl.push_back(b == x.len);
        end
        haw.push_back(x); hb_exp.push_back(x); hwr++; hlast = x.a;
      end
      // reads right after the writes are handed over (before their B's): the gate holds them
      if (gen && hr_exp.size() == 0 && haw.size() == 0 && hw.size() == 0 && !s_awvalid && !s_wvalid
          && $urandom % 10 == 0) begin
        automatic int batch = 1 + $urandom % 8;
        for (int i = 0; i < batch; i++) begin
          automatic hx_t x;
          x.id = $urandom; x.len = $urandom % 4; x.a = ($urandom % 2) ? hlast : haddr();
          for (int b = 0; b <= x.len; b++) hre.push_back(hview(x.a[33:6] + b));
          har.push_back(x); hr_exp.push_back(x); hrd++;
        end
      end
      if (s_awvalid && s_awready) begin void'(haw.pop_front()); s_awvalid <= 1'b0; end
      else if (!s_awvalid && haw.size()) begin
        s_awvalid <= 1'b1; s_awid <= haw[0].id; s_awaddr <= haw[0].a; s_awlen <= haw[0].len;
      end
      if (s_wvalid && s_wready) begin
        void'(hw.pop_front()); void'(hm.pop_front()); void'(hl.pop_front()); s_wvalid <= 1'b0;
      end else if (!s_wvalid && hw.size() && $urandom % 3 != 0) begin
        s_wvalid <= 1'b1; s_wdata <= hw[0]; s_wstrb <= hm[0]; s_wlast <= hl[0];
      end
      if (s_arvalid && s_arready) begin void'(har.pop_front()); s_arvalid <= 1'b0; end
      else if (!s_arvalid && har.size()) begin
        s_arvalid <= 1'b1; s_arid <= har[0].id; s_araddr <= har[0].a; s_arlen <= har[0].len;
      end
      if (s_bvalid && s_bready) begin
        if (hb_exp.size() == 0) fail("host: B with no write");
        else begin
          if (s_bid !== hb_exp[0].id) fail("host: B's ID out of order");
          void'(hb_exp.pop_front());
        end
      end
      // pcis_wbusy: high from two cycles after a write's AW is taken until its B is given back
      hb_out <= hb_out + (s_awvalid && s_awready) - (s_bvalid && s_bready);
      hb_age <= {hb_age[1:0], hb_out != 0};
      if (&hb_age && hb_out != 0 && !pcis_wbusy) wbusy_low++;
      if (s_rvalid && s_rready) begin
        if (hre.size() == 0) fail("host: read data with no read");
        else begin
          automatic logic [511:0] e = hre.pop_front();
          if (s_rdata !== e) fail($sformatf("host: read %0d is not its own writes' (%h, expected %h)",
                                            hrd, s_rdata[31:0], e[31:0]));
          if (s_rid !== hr_exp[0].id) fail("host: R's ID out of order");
          if (s_rlast !== (hr_beat == hr_exp[0].len)) fail("host: RLAST at wrong beat");
          if (s_rlast) begin void'(hr_exp.pop_front()); hr_beat = 0; end
          else hr_beat++;
        end
      end
    end
  end

  // ---------------------------------------------------------------- run
  initial begin
    repeat (20) @(posedge clk);
    @(negedge aclk) arst = 1'b0;
    @(negedge clk) rst = 1'b0;
    repeat (10) @(posedge clk);
    gen = 1'b1;
    repeat (cycles) @(posedge clk);
    @(negedge clk);
    gen = 1'b0;
    for (int t = 0; t < 400000; t++) begin
      automatic bit idle = hb_exp.size() == 0 && hr_exp.size() == 0;
      @(posedge clk);
      for (int c = 0; c < NPORT; c++)
        idle &= cq[c].size() == 0 && eq[c].size() == 0 && n_wdone[c] == 16'(nwr[c]);
      if (idle) break;
    end
    repeat (100) @(posedge clk);
    for (int c = 0; c < NPORT; c++) begin
      if (n_wdone[c] != 16'(nwr[c])) fail($sformatf("ch%0d: n_wdone %0d, %0d writes", c, n_wdone[c], nwr[c]));
      if (eq[c].size()) fail($sformatf("ch%0d: %0d reads unanswered", c, eq[c].size()));
      foreach (sh[c][a]) if (rd({5'(c), 23'(a)}) !== sh[c][a])
        fail($sformatf("ch%0d: beat %0d not in the HBM after n_wdone", c, a));
    end
    if (hb_exp.size() || hr_exp.size()) fail("host: responses missing");
    foreach (hsh[a]) if (rd(a) !== hsh[a]) fail("host: a write is not in the HBM");
    if (wbusy_low) fail($sformatf("pcis_wbusy low for %0d cycles with a host write unanswered", wbusy_low));
    if (READ_BURST > 1 && nlong * 20 < nbursts) fail($sformatf("few bursts: %0d of %0d reads", nlong, nbursts));
    $write("NPORT %0d: %0d channel read bursts, %0d of several beats; lengths:", NPORT, nbursts, nlong);
    for (int i = 1; i <= 16; i++) if (hist[i]) $write(" %0d:%0d", i, hist[i]);
    $display("; host %0d writes %0d reads", hwr, hrd);
    for (int c = 0; c < NPORT; c++) $display("ch%0d %0d writes %0d reads", c, nwr[c], nrd[c]);
    if (errors) $display("FAIL (%0d errors)", errors);
    else $display("PASS");
    $finish;
  end
endmodule
