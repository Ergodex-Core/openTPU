// Unit test of the F2 slot's memory path (rtl/boards/f2: otpu_f2_nch x 2 in front of otpu_f2_mem;
// tests/test_f2_rtl.py): two native masters (the slice's channels, core clock) and the host's PCIS
// (bursts, random IDs; aclk) on an AXI4 model of the DIMM that reorders as an AXI slave may: each
// transaction waits a random latency, and any ready head of any ID's read or write queue may go
// next (one ID's reads in order, its writes in order; nothing else ordered).
// Checked against the native port's contract (otpu_native_dram): every read returns the beat as
// the master's own earlier commands left it (a read after a write to its beat sees the write, a
// read before one does not), whatever the DIMM's order; n_wdone reaches the writes' count, and a
// write counted there is in the DIMM; the host reads its own writes back (its reads issued once
// its writes are delivered), and its B's and R's carry its IDs in its order. The native masters
// use a few beats each, so most commands meet one to the same beat in flight. HAZ = 0 (no
// same-beat waits) fails.
// Plusargs: +seed=, +cycles= (core cycles of traffic), +nb= (beats per master). Prints the counts
// and PASS, or the first errors and FAIL.
module tb_f2mem;
  parameter bit HAZ = 1'b1;
  parameter int CORE_HALF = 5, AXI_HALF = 2;
  localparam logic [63:0] CHB = 64'h8000_0000, HB = 64'h4_0000_0000;  // channel 1, the host's base

  logic clk = 1'b0, aclk = 1'b0;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;
  logic rst = 1'b1, arst = 1'b1;

  int unsigned seed = 1, cycles = 20000, nb = 6, errors = 0;
  bit error_phase = 0, inject_error = 0;
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
  function automatic logic [511:0] rnd512();
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = $urandom;
    return v;
  endfunction
  function automatic logic [511:0] init(input logic [57:0] beat);   // the DIMM's first contents
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = beat[31:0] * 32'h9E37_79B9 ^ (i * 32'h85EB_CA6B) ^ beat[57:32];
    return v;
  endfunction
  function automatic logic [511:0] merge(input logic [511:0] o, d, input logic [63:0] m);
    for (int i = 0; i < 64; i++) if (m[i]) o[8 * i +: 8] = d[8 * i +: 8];
    return o;
  endfunction

  // ---------------------------------------------------------------- the DUT
  logic [1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [1:0][24:0]  n_caddr;
  logic [1:0][511:0] n_wdata, n_rdata;
  logic [1:0][63:0]  n_wmask;
  logic [1:0][15:0]  n_wdone;
  logic [1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
  logic [1:0]        m_arvalid, m_arready, m_rvalid, m_rready;
  logic [1:0][63:0]  m_awaddr, m_araddr, m_wstrb;
  logic [1:0][511:0] m_wdata;
  logic [511:0]      m_rdata;
  logic [1:0]        m_bresp, m_rresp;
  for (genvar c = 0; c < 2; c++) begin : g_ch
    otpu_f2_nch #(.BASE(CHB * c), .HAZ(HAZ), .CQ(4), .RD(8), .ROD(8), .WOD(4)) u_nch (
      .clk, .rst,
      .n_cvalid(n_cvalid[c]), .n_cready(n_cready[c]), .n_cwe(n_cwe[c]), .n_caddr(n_caddr[c]),
      .n_wvalid(n_wvalid[c]), .n_wready(n_wready[c]), .n_wdata(n_wdata[c]),
      .n_wmask(n_wmask[c]), .n_rvalid(n_rvalid[c]), .n_rdata(n_rdata[c]),
      .n_wdone(n_wdone[c]), .n_err(n_err[c]),
      .aclk, .arst,
      .m_awvalid(m_awvalid[c]), .m_awready(m_awready[c]), .m_awaddr(m_awaddr[c]),
      .m_wvalid(m_wvalid[c]), .m_wready(m_wready[c]), .m_wdata(m_wdata[c]),
      .m_wstrb(m_wstrb[c]), .m_bvalid(m_bvalid[c]), .m_bready(m_bready[c]), .m_bresp(m_bresp),
      .m_arvalid(m_arvalid[c]), .m_arready(m_arready[c]), .m_araddr(m_araddr[c]),
      .m_rvalid(m_rvalid[c]), .m_rready(m_rready[c]), .m_rdata(m_rdata), .m_rresp(m_rresp));
  end
  logic [15:0]  h_awid, h_arid, h_bid, h_rid;
  logic [63:0]  h_awaddr, h_araddr;
  logic [7:0]   h_awlen, h_arlen;
  logic         h_awvalid, h_awready, h_wvalid, h_wready, h_wlast, h_bvalid, h_bready;
  logic         h_arvalid, h_arready, h_rvalid, h_rready, h_rlast, pcis_wbusy;
  logic [511:0] h_wdata, h_rdata;
  logic [63:0]  h_wstrb;
  logic [1:0]   h_bresp, h_rresp;
  logic [15:0]  d_awid, d_arid, d_bid, d_rid;
  logic [63:0]  d_awaddr, d_araddr, d_wstrb;
  logic [7:0]   d_awlen, d_arlen;
  logic [2:0]   d_awsize, d_arsize;
  logic [1:0]   d_awburst, d_arburst, d_bresp, d_rresp;
  logic         d_awvalid, d_awready, d_wvalid, d_wready, d_wlast, d_bvalid, d_bready;
  logic         d_arvalid, d_arready, d_rvalid, d_rready, d_rlast;
  logic [511:0] d_wdata, d_rdata;
  otpu_f2_mem #(.HQ(4)) u_mem (
    .aclk, .arst,
    .c_awvalid(m_awvalid), .c_awready(m_awready), .c_awaddr(m_awaddr),
    .c_wvalid(m_wvalid), .c_wready(m_wready), .c_wdata(m_wdata), .c_wstrb(m_wstrb),
    .c_bvalid(m_bvalid), .c_bready(m_bready), .c_bresp(m_bresp),
    .c_arvalid(m_arvalid), .c_arready(m_arready), .c_araddr(m_araddr),
    .c_rvalid(m_rvalid), .c_rready(m_rready), .c_rdata(m_rdata), .c_rresp(m_rresp),
    .h_awid, .h_awaddr, .h_awlen, .h_awsize(3'd6), .h_awburst(2'b01), .h_awvalid, .h_awready,
    .h_wdata, .h_wstrb, .h_wlast, .h_wvalid, .h_wready, .h_bid, .h_bresp, .h_bvalid, .h_bready,
    .h_arid, .h_araddr, .h_arlen, .h_arsize(3'd6), .h_arburst(2'b01), .h_arvalid, .h_arready,
    .h_rid, .h_rdata, .h_rresp, .h_rlast, .h_rvalid, .h_rready, .pcis_wbusy,
    .d_awid, .d_awaddr, .d_awlen, .d_awsize, .d_awburst, .d_awvalid, .d_awready,
    .d_wdata, .d_wstrb, .d_wlast, .d_wvalid, .d_wready, .d_bid, .d_bresp, .d_bvalid, .d_bready,
    .d_arid, .d_araddr, .d_arlen, .d_arsize, .d_arburst, .d_arvalid, .d_arready,
    .d_rid, .d_rdata, .d_rresp, .d_rlast, .d_rvalid, .d_rready);

  // ---------------------------------------------------------------- the DIMM model (aclk)
  logic [511:0] mem [logic [57:0]];
  function automatic logic [511:0] rd(input logic [57:0] b);
    return mem.exists(b) ? mem[b] : init(b);
  endfunction
  typedef struct { logic [15:0] id; logic [63:0] addr; logic [7:0] len; longint due; } ax_t;
  ax_t          awq[$], arq[$];          // accepted, in arrival order (the scheduler keeps ID order)
  logic [511:0] wd[$];
  logic [63:0]  ws[$];
  logic [15:0]  bq[$];
  typedef struct { logic [15:0] id; logic [511:0] d; logic last; } rbeat_t;
  rbeat_t       rq[$];
  longint       now = 0;
  always_ff @(posedge aclk) begin
    now <= now + 1;
    if (!arst) begin
      if (d_awvalid && d_awready) begin
        awq.push_back('{d_awid, d_awaddr, d_awlen, now + 4 + $urandom % 60 + (gen ? 0 : 100)});
      end
      if (d_wvalid && d_wready) begin
        wd.push_back(d_wdata); ws.push_back(d_wstrb);
      end
      if (d_arvalid && d_arready) arq.push_back('{d_arid, d_araddr, d_arlen, now + 4 + $urandom % 60});
      // one transaction executes now and then: any whose ID has nothing older in its direction
      if ($urandom % 3 == 0) begin
        automatic int cand[$];                      // 0..: writes; 1000..: reads
        automatic int wofs = 0;
        for (int i = 0; i < awq.size(); i++) begin
          automatic bit older = 0;
          for (int j = 0; j < i; j++) older |= awq[j].id == awq[i].id;
          if (!older && awq[i].due <= now && wofs + awq[i].len + 1 <= wd.size()) cand.push_back(i);
          wofs += awq[i].len + 1;
        end
        for (int i = 0; i < arq.size(); i++) begin
          automatic bit older = 0;
          for (int j = 0; j < i; j++) older |= arq[j].id == arq[i].id;
          if (!older && arq[i].due <= now) cand.push_back(1000 + i);
        end
        if (cand.size() && rq.size() < 64 && bq.size() < 64) begin
          automatic int k = cand[$urandom % cand.size()];
          if (k < 1000) begin             // a write: its beats are at its offset in wd
            automatic int o = 0;
            for (int j = 0; j < k; j++) o += awq[j].len + 1;
            for (int b = 0; b <= awq[k].len; b++) begin
              automatic logic [57:0] a = awq[k].addr[63:6] + b;
              mem[a] = merge(rd(a), wd[o + b], ws[o + b]);
            end
            for (int b = 0; b <= awq[k].len; b++) begin wd.delete(o); ws.delete(o); end
            bq.push_back(awq[k].id);
            awq.delete(k);
          end else begin
            k -= 1000;
            for (int b = 0; b <= arq[k].len; b++)
              rq.push_back('{arq[k].id, rd(arq[k].addr[63:6] + b), b == arq[k].len});
            arq.delete(k);
          end
        end
      end
      if (d_bvalid && d_bready) void'(bq.pop_front());
      if (d_rvalid && d_rready) void'(rq.pop_front());
    end
    // the response channels from the queues as this edge leaves them: what a continuous
    // assignment from the queues shows until the next edge, written here (registered) because
    // combinational logic over a queue's size and head is not re-evaluated on every mutation
    // by every simulator (ArcSim: docs/arcsim.md)
    d_bvalid <= bq.size() > 0;
    d_bid    <= bq.size() ? bq[0] : '0;
    d_rvalid <= rq.size() > 0;
    d_rid    <= rq.size() ? rq[0].id : '0;
    d_rdata  <= rq.size() ? rq[0].d : '0;
    d_rlast  <= rq.size() ? rq[0].last : 1'b0;
  end
  always_ff @(posedge aclk) begin
    d_awready <= $urandom % 4 != 0;
    d_wready  <= $urandom % 4 != 0;
    d_arready <= $urandom % 4 != 0;
  end
  assign d_bresp  = inject_error ? 2'b10 : 2'b00;
  assign d_rresp  = inject_error ? 2'b10 : 2'b00;

  // ---------------------------------------------------------------- the native masters (clk)
  logic [511:0] sh [2][logic [24:0]];    // each master's view of its beats, in command order
  typedef struct { logic we; logic [24:0] a; } cmd_t;
  cmd_t         cq [2][$];
  logic [511:0] dq [2][$], eq [2][$];
  logic [63:0]  mq [2][$];
  int           nwr [2], nrd [2], ngot [2];
  bit           gen = 1'b0;
  function automatic logic [511:0] nview(input int c, input logic [24:0] a);
    return sh[c].exists(a) ? sh[c][a] : init((CHB * c) >> 6 | 58'(a));
  endfunction
  for (genvar c = 0; c < 2; c++) begin : g_m
    always_ff @(posedge clk) begin
      if (rst) begin
        n_cvalid[c] <= 1'b0; n_wvalid[c] <= 1'b0;
      end else begin
        if (gen && cq[c].size() < 24 && $urandom % 2 == 0) begin
          automatic cmd_t x;
          x.we = $urandom % 2;
          x.a = ($urandom % 8 == 0) ? 25'($urandom % 4096) : 25'($urandom % nb);
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
        // commands and write data shown apart, each held until taken
        if (n_cvalid[c] && n_cready[c]) begin void'(cq[c].pop_front()); n_cvalid[c] <= 1'b0; end
        else if (!n_cvalid[c] && cq[c].size() && $urandom % 4 != 0) begin
          n_cvalid[c] <= 1'b1; n_cwe[c] <= cq[c][0].we; n_caddr[c] <= cq[c][0].a;
        end
        if (n_wvalid[c] && n_wready[c]) begin
          void'(dq[c].pop_front()); void'(mq[c].pop_front()); n_wvalid[c] <= 1'b0;
        end else if (!n_wvalid[c] && dq[c].size() && $urandom % 4 != 0) begin
          n_wvalid[c] <= 1'b1; n_wdata[c] <= dq[c][0]; n_wmask[c] <= mq[c][0];
        end
        if (n_rvalid[c]) begin
          if (eq[c].size() == 0) fail($sformatf("ch%0d: read data with no read", c));
          else begin
            automatic logic [511:0] e = eq[c].pop_front();
            if (n_rdata[c] !== e) fail($sformatf("ch%0d: read %0d returned stale or wrong data", c, ngot[c]));
          end
          ngot[c]++;
        end
        if (n_err[c] && !error_phase) fail($sformatf("ch%0d: n_err", c));
      end
    end
  end
  // the core sees a write in the DIMM once n_wdone counts it (checked at the end: all of them)

  // ---------------------------------------------------------------- the host (aclk)
  logic [511:0] hsh [logic [57:0]];
  typedef struct { logic [15:0] id; logic [63:0] a; logic [7:0] len; } hx_t;
  hx_t          haw[$], har[$], hb_exp[$], hr_exp[$];
  logic [511:0] hw[$], hre[$];
  logic [63:0]  hm[$];
  logic         hl[$];                   // each beat's WLAST
  int           hwr = 0, hrd = 0;
  int           hr_beat = 0;
  function automatic logic [511:0] hview(input logic [57:0] b);
    return hsh.exists(b) ? hsh[b] : init(b);
  endfunction
  always_ff @(posedge aclk) begin
    if (arst) begin
      h_awvalid <= 1'b0; h_wvalid <= 1'b0; h_arvalid <= 1'b0;
      h_bready <= 1'b0; h_rready <= 1'b0;
    end else begin
      h_bready <= $urandom % 3 != 0;
      h_rready <= $urandom % 3 != 0;
      // A later write may pass an earlier read on AXI. Keep the host's snapshot fixed until
      // its read completes; only the native masters promise read-before-write ordering.
      if (gen && hr_exp.size() == 0 && haw.size() < 4 && $urandom % 12 == 0) begin
        automatic hx_t x;
        x.id = $urandom; x.len = $urandom % 4; x.a = HB + 64 * ($urandom % 8);
        for (int b = 0; b <= x.len; b++) begin
          automatic logic [511:0] d = rnd512();
          automatic logic [63:0]  m = ($urandom % 2) ? {$urandom, $urandom} : '1;
          hsh[x.a[63:6] + b] = merge(hview(x.a[63:6] + b), d, m);
          hw.push_back(d); hm.push_back(m); hl.push_back(b == x.len);
        end
        haw.push_back(x); hb_exp.push_back(x); hwr++;
      end
      // a read only once every earlier write is delivered (PCIe: reads do not pass writes)
      if (gen && hr_exp.size() == 0 && haw.size() == 0 && hw.size() == 0 && !h_awvalid && !h_wvalid
          && $urandom % 10 == 0) begin
        automatic int batch = 1 + $urandom % 8;
        for (int i = 0; i < batch; i++) begin
          automatic hx_t x;
          x.id = $urandom; x.len = $urandom % 4; x.a = HB + 64 * ($urandom % 8);
          for (int b = 0; b <= x.len; b++) hre.push_back(hview(x.a[63:6] + b));
          har.push_back(x); hr_exp.push_back(x); hrd++;
        end
      end
      if (h_awvalid && h_awready) begin void'(haw.pop_front()); h_awvalid <= 1'b0; end
      else if (!h_awvalid && haw.size()) begin
        h_awvalid <= 1'b1; h_awid <= haw[0].id; h_awaddr <= haw[0].a; h_awlen <= haw[0].len;
      end
      if (h_wvalid && h_wready) begin
        void'(hw.pop_front()); void'(hm.pop_front()); void'(hl.pop_front()); h_wvalid <= 1'b0;
      end else if (!h_wvalid && hw.size() && $urandom % 3 != 0) begin
        h_wvalid <= 1'b1; h_wdata <= hw[0]; h_wstrb <= hm[0]; h_wlast <= hl[0];
      end
      if (h_arvalid && h_arready) begin void'(har.pop_front()); h_arvalid <= 1'b0; end
      else if (!h_arvalid && har.size()) begin
        h_arvalid <= 1'b1; h_arid <= har[0].id; h_araddr <= har[0].a; h_arlen <= har[0].len;
      end
      if (h_bvalid && h_bready) begin
        if (hb_exp.size() == 0) fail("host: B with no write");
        else begin
          if (h_bid !== hb_exp[0].id) fail("host: B's ID out of order");
          void'(hb_exp.pop_front());
        end
      end
      if (h_rvalid && h_rready) begin
        if (hre.size() == 0) fail("host: read data with no read");
        else begin
          automatic logic [511:0] e = hre.pop_front();
          if (h_rdata !== e) fail($sformatf("host: read data is not its own writes' (read %0d: %h, expected %h)",
                                            hrd, h_rdata[31:0], e[31:0]));
          if (h_rid !== hr_exp[0].id) fail("host: R's ID out of order");
          if (h_rlast !== (hr_beat == hr_exp[0].len)) fail("host: RLAST at wrong beat");
          if (h_rlast) begin void'(hr_exp.pop_front()); hr_beat = 0; end
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
    // The first posted write and the following read can reach their AXI channels together.
    // Delay the model's write so a read slipping past it demonstrably returns stale data.
    @(negedge aclk);
    hsh[HB >> 6] = 512'h1234_5678;
    hw.push_back(512'h1234_5678); hm.push_back('1); hl.push_back(1'b1);
    haw.push_back('{16'h1234, HB, 8'd0}); hb_exp.push_back('{16'h1234, HB, 8'd0}); hwr++;
    hre.push_back(512'h1234_5678);
    har.push_back('{16'h5678, HB, 8'd0}); hr_exp.push_back('{16'h5678, HB, 8'd0}); hrd++;
    for (int t = 0; t < 1000; t++) begin
      @(negedge aclk);
      if (hb_exp.size() == 0 && hr_exp.size() == 0) break;
    end
    if (hb_exp.size() || hr_exp.size()) $fatal(1, "host: initial write/read did not drain");
    gen = 1'b1;
    repeat (cycles) @(posedge clk);
    @(negedge clk);
    gen = 1'b0;
    // drain: every command answered, every write counted in n_wdone and in the DIMM
    for (int t = 0; t < 200000; t++) begin
      @(posedge clk);
      if (cq[0].size() == 0 && cq[1].size() == 0 && eq[0].size() == 0 && eq[1].size() == 0 &&
          n_wdone[0] == 16'(nwr[0]) && n_wdone[1] == 16'(nwr[1]) && hb_exp.size() == 0 &&
          hr_exp.size() == 0) break;
    end
    repeat (50) @(posedge clk);
    for (int c = 0; c < 2; c++) begin
      if (n_wdone[c] != 16'(nwr[c])) fail($sformatf("ch%0d: n_wdone %0d, %0d writes", c, n_wdone[c], nwr[c]));
      if (eq[c].size()) fail($sformatf("ch%0d: %0d reads unanswered", c, eq[c].size()));
      foreach (sh[c][a]) if (rd((CHB * c) >> 6 | 58'(a)) !== sh[c][a])
        fail($sformatf("ch%0d: beat %0d not in the DIMM after n_wdone", c, a));
    end
    if (hb_exp.size() || hr_exp.size()) fail("host: responses missing");
    foreach (hsh[a]) if (rd(a) !== hsh[a]) fail("host: a write is not in the DIMM");

    // Read errors and write errors both cross back and remain sticky until the common reset.
    @(negedge clk);
    error_phase = 1; inject_error = 1;
    cq[0].push_back('{1'b0, 25'd0}); eq[0].push_back(nview(0, 25'd0)); nrd[0]++;
    cq[1].push_back('{1'b1, 25'd0}); dq[1].push_back(512'hdead_beef); mq[1].push_back('1); nwr[1]++;
    for (int t = 0; t < 2000; t++) begin
      @(negedge clk);
      if (&n_err && n_wdone[1] == 16'(nwr[1]) && eq[0].size() == 0) break;
    end
    inject_error = 0;
    repeat (20) @(negedge clk);
    if (n_err !== 2'b11) fail("native response errors did not become sticky");
    arst = 1; rst = 1;
    @(negedge clk);
    if (n_wdone !== '0 || n_err !== '0) fail("native completion/error outputs did not reset immediately");
    repeat (20) @(negedge clk);
    if (n_wdone !== '0 || n_err !== '0) fail("native completion/error CDC did not reset");
    $display("ch0 %0d writes %0d reads, ch1 %0d writes %0d reads, host %0d writes %0d reads",
             nwr[0], nrd[0], nwr[1], nrd[1], hwr, hrd);
    if (errors) $display("FAIL (%0d errors)", errors);
    else $display("PASS");
    $finish;
  end
endmodule
