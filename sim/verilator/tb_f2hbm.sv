// Unit test of the F2 HBM build's memory fabric (rtl/boards/f2/otpu_f2_hbm_mem.sv: NPORT channel
// masters otpu_native_axi, the host's PCIS path otpu_f2_hbm_host, a port arbiter otpu_f2_hbm_port
// each, all at the core clock; tests/test_f2_rtl.py) through a behavioural stand-in for the
// HDK's cl_axi_sc_1x1 SmartConnect per port (tb_f2hbm_sc.sv: the crossing into the HBM clock,
// 512 -> 256 bits, AXI4 -> AXI3, in-order responses) onto a model of the HBM's AXI3 ports: one
// 16 GiB memory, each port taking AW / W / AR with random readies, executing transactions after
// a random latency in any order that keeps the reads in order and the writes in order (nothing
// else ordered: the controller's reorder between its read and write channels). NPORT native
// masters (core clock; S slices x NCH channels) and the host (aclk) run random traffic. Checked:
//   - the native port's contract (otpu_native_dram): every read returns the beat as the
//     channel's own earlier commands left it, whatever the HBM's order; n_wdone reaches the
//     writes' count; after it, every write is in the HBM at BASE p x 512 MiB + beat x 64;
//   - every port's transaction stays in its port's 512 MiB (no switch crossing), bursts are
//     INCR of 32-byte beats within 4 KiB, and runs of beats did go out as bursts of several;
//   - the host's PCIS: its writes land in every port's memory (BAR4 offset = HBM address), its
//     reads (issued once its writes are delivered: PCIe order) return its own writes, B's and R's
//     in its order with its IDs, reads past the last port return zeros.
// HAZ = 0 (no direction barrier in the masters) fails. Plusargs: +seed=, +cycles=, +nb= (beats
// per channel). Prints the counts and PASS, or the first errors and FAIL.
module tb_f2hbm;
  parameter int NPORT = 8;
  parameter bit HAZ = 1'b1;
  parameter int CORE_HALF = 5, AXI_HALF = 4, HBM_HALF = 2;   // 100 / 125 / 250: the ratios

  logic clk = 1'b0, aclk = 1'b0, hclk = 1'b0;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;
  always #(HBM_HALF) hclk = !hclk;
  logic rst = 1'b1, arst = 1'b1, hrst = 1'b1;

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
  function automatic logic [255:0] init(input logic [28:0] u);    // a 32-byte unit's first contents
    logic [255:0] v;
    for (int i = 0; i < 8; i++) v[32 * i +: 32] = u * 32'h9E37_79B9 ^ (i * 32'h85EB_CA6B);
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
  logic [15:0]  h_awid, h_arid, h_bid, h_rid;
  logic [63:0]  h_awaddr, h_araddr;
  logic [7:0]   h_awlen, h_arlen;
  logic         h_awvalid, h_awready, h_wvalid, h_wready, h_wlast, h_bvalid, h_bready;
  logic         h_arvalid, h_arready, h_rvalid, h_rready, h_rlast, pcis_wbusy;
  logic [511:0] h_wdata, h_rdata;
  logic [63:0]  h_wstrb;
  logic [1:0]   h_bresp, h_rresp;
  // the ports' AXI4 sides (core clock) into the SmartConnect stand-ins, their AXI3 sides (hclk)
  // into the HBM model
  logic [NPORT-1:0][15:0]  a_awid, a_arid, a_bid, a_rid;
  logic [NPORT-1:0][33:0]  a_awaddr, a_araddr;
  logic [NPORT-1:0][7:0]   a_awlen, a_arlen;
  logic [NPORT-1:0]        a_awvalid, a_awready, a_wvalid, a_wready, a_wlast, a_bvalid, a_bready;
  logic [NPORT-1:0]        a_arvalid, a_arready, a_rvalid, a_rready, a_rlast;
  logic [NPORT-1:0][511:0] a_wdata, a_rdata;
  logic [NPORT-1:0][63:0]  a_wstrb;
  logic [NPORT-1:0][1:0]   a_bresp, a_rresp;
  logic [NPORT-1:0][33:0]  m_awaddr, m_araddr;
  logic [NPORT-1:0][3:0]   m_awlen, m_arlen;
  logic [NPORT-1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
  logic [NPORT-1:0]        m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [NPORT-1:0][255:0] m_wdata, m_rdata;
  logic [NPORT-1:0][31:0]  m_wstrb;
  logic [NPORT-1:0][1:0]   m_bresp, m_rresp;
  otpu_f2_hbm_mem #(.NPORT(NPORT), .HAZ(HAZ)) u_dut (
    .clk, .rst, .aclk, .arst,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid,
    .n_rdata, .n_wdone, .n_err,
    .pcis_awid(h_awid), .pcis_awaddr(h_awaddr), .pcis_awlen(h_awlen), .pcis_awsize(3'd6),
    .pcis_awburst(2'b01), .pcis_awvalid(h_awvalid), .pcis_awready(h_awready),
    .pcis_wdata(h_wdata), .pcis_wstrb(h_wstrb), .pcis_wlast(h_wlast), .pcis_wvalid(h_wvalid),
    .pcis_wready(h_wready), .pcis_bid(h_bid), .pcis_bresp(h_bresp), .pcis_bvalid(h_bvalid),
    .pcis_bready(h_bready), .pcis_arid(h_arid), .pcis_araddr(h_araddr), .pcis_arlen(h_arlen),
    .pcis_arsize(3'd6), .pcis_arburst(2'b01), .pcis_arvalid(h_arvalid),
    .pcis_arready(h_arready), .pcis_rid(h_rid), .pcis_rdata(h_rdata), .pcis_rresp(h_rresp),
    .pcis_rlast(h_rlast), .pcis_rvalid(h_rvalid), .pcis_rready(h_rready), .pcis_wbusy,
    .m_awid(a_awid), .m_awaddr(a_awaddr), .m_awlen(a_awlen), .m_awvalid(a_awvalid),
    .m_awready(a_awready), .m_wdata(a_wdata), .m_wstrb(a_wstrb), .m_wlast(a_wlast),
    .m_wvalid(a_wvalid), .m_wready(a_wready), .m_bid(a_bid), .m_bresp(a_bresp),
    .m_bvalid(a_bvalid), .m_bready(a_bready), .m_arid(a_arid), .m_araddr(a_araddr),
    .m_arlen(a_arlen), .m_arvalid(a_arvalid), .m_arready(a_arready), .m_rid(a_rid),
    .m_rdata(a_rdata), .m_rresp(a_rresp), .m_rlast(a_rlast), .m_rvalid(a_rvalid),
    .m_rready(a_rready));
  for (genvar p = 0; p < NPORT; p++) begin : g_sc
    tb_f2hbm_sc u_sc (
      .clk, .rst, .hclk,
      .s_awid(a_awid[p]), .s_awaddr(a_awaddr[p]), .s_awlen(a_awlen[p]), .s_awvalid(a_awvalid[p]),
      .s_awready(a_awready[p]), .s_wdata(a_wdata[p]), .s_wstrb(a_wstrb[p]), .s_wlast(a_wlast[p]),
      .s_wvalid(a_wvalid[p]), .s_wready(a_wready[p]), .s_bid(a_bid[p]), .s_bresp(a_bresp[p]),
      .s_bvalid(a_bvalid[p]), .s_bready(a_bready[p]), .s_arid(a_arid[p]), .s_araddr(a_araddr[p]),
      .s_arlen(a_arlen[p]), .s_arvalid(a_arvalid[p]), .s_arready(a_arready[p]), .s_rid(a_rid[p]),
      .s_rdata(a_rdata[p]), .s_rresp(a_rresp[p]), .s_rlast(a_rlast[p]), .s_rvalid(a_rvalid[p]),
      .s_rready(a_rready[p]),
      .m_awaddr(m_awaddr[p]), .m_awlen(m_awlen[p]), .m_awvalid(m_awvalid[p]),
      .m_awready(m_awready[p]), .m_wdata(m_wdata[p]), .m_wstrb(m_wstrb[p]), .m_wlast(m_wlast[p]),
      .m_wvalid(m_wvalid[p]), .m_wready(m_wready[p]), .m_bresp(m_bresp[p]), .m_bvalid(m_bvalid[p]),
      .m_bready(m_bready[p]), .m_araddr(m_araddr[p]), .m_arlen(m_arlen[p]),
      .m_arvalid(m_arvalid[p]), .m_arready(m_arready[p]), .m_rdata(m_rdata[p]),
      .m_rresp(m_rresp[p]), .m_rlast(m_rlast[p]), .m_rvalid(m_rvalid[p]), .m_rready(m_rready[p]));
  end

  // ---------------------------------------------------------------- the HBM (hclk)
  // (the AXI3 side has no IDs: the SmartConnect returns one port's responses in order; the
  // model keeps the reads in order and the writes in order and reorders between them)
  logic [NPORT-1:0][5:0] m_awid, m_arid, m_bid, m_rid;
  assign m_awid = '0;
  assign m_arid = '0;
  logic [255:0] mem [logic [28:0]];          // 32-byte units of the 16 GiB
  function automatic logic [255:0] rd(input logic [28:0] u);
    return mem.exists(u) ? mem[u] : init(u);
  endfunction
  function automatic logic [511:0] rd64(input logic [27:0] b);      // a 64-byte beat
    return {rd({b, 1'b1}), rd({b, 1'b0})};
  endfunction
  int nburst [NPORT], nlong [NPORT];
  for (genvar p = 0; p < NPORT; p++) begin : g_hbm
    typedef struct { logic [5:0] id; logic [33:0] addr; logic [3:0] len; longint due; } ax_t;
    typedef struct { logic [5:0] id; logic [255:0] d; logic last; } rb_t;
    ax_t          awq[$], arq[$];
    logic [255:0] wd[$];
    logic [31:0]  ws[$];
    logic [5:0]   bq[$];
    rb_t          rq [2][$];                 // read beats due, per ID bit (the two masters)
    longint       now = 0;
    logic         rsel;
    always_ff @(posedge hclk) begin
      now <= now + 1;
      m_awready[p] <= $urandom % 4 != 0;
      m_wready[p] <= $urandom % 4 != 0;
      m_arready[p] <= $urandom % 4 != 0;
      if (!hrst) begin
        if (m_awvalid[p] && m_awready[p]) begin
          nburst[p]++; if (m_awlen[p] > 1) nlong[p]++;
          if (m_awaddr[p][33:29] != 5'(p)) fail($sformatf("port %0d: a write outside its pseudo-channel", p));
          if (m_awaddr[p][4:0] != 0 || (m_awaddr[p][11:5] + m_awlen[p]) > 127)
            fail($sformatf("port %0d: AW unaligned or across 4 KiB", p));
          awq.push_back('{m_awid[p], m_awaddr[p], m_awlen[p], now + 4 + $urandom % 40});
        end
        if (m_wvalid[p] && m_wready[p]) begin wd.push_back(m_wdata[p]); ws.push_back(m_wstrb[p]); end
        if (m_arvalid[p] && m_arready[p]) begin
          nburst[p]++; if (m_arlen[p] > 1) nlong[p]++;
          if (m_araddr[p][33:29] != 5'(p)) fail($sformatf("port %0d: a read outside its pseudo-channel", p));
          if (m_araddr[p][4:0] != 0 || (m_araddr[p][11:5] + m_arlen[p]) > 127)
            fail($sformatf("port %0d: AR unaligned or across 4 KiB", p));
          arq.push_back('{m_arid[p], m_araddr[p], m_arlen[p], now + 8 + $urandom % 60});
        end
        // one transaction executes now and then: any whose ID has nothing older in its direction
        if ($urandom % 2 == 0) begin
          automatic int cand[$];
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
          if (cand.size() && rq[0].size() + rq[1].size() < 64 && bq.size() < 64) begin
            automatic int k = cand[$urandom % cand.size()];
            if (k < 1000) begin
              automatic int o = 0;
              for (int j = 0; j < k; j++) o += awq[j].len + 1;
              for (int b = 0; b <= awq[k].len; b++) begin
                automatic logic [28:0] u = awq[k].addr[33:5] + b;
                automatic logic [255:0] v = rd(u);
                for (int i = 0; i < 32; i++) if (ws[o + b][i]) v[8 * i +: 8] = wd[o + b][8 * i +: 8];
                mem[u] = v;
              end
              for (int b = 0; b <= awq[k].len; b++) begin wd.delete(o); ws.delete(o); end
              bq.push_back(awq[k].id);
              awq.delete(k);
            end else begin
              k -= 1000;
              for (int b = 0; b <= arq[k].len; b++)
                rq[arq[k].id[0]].push_back('{arq[k].id, rd(arq[k].addr[33:5] + b), b == arq[k].len});
              arq.delete(k);
            end
          end
        end
        // the responses are registers: the beat shown is popped once taken, the next loaded
        // (a continuous assignment from a queue would race the blocks that pop it)
        if (m_bvalid[p] && m_bready[p]) void'(bq.pop_front());
        if (!m_bvalid[p] || m_bready[p]) begin
          m_bvalid[p] <= bq.size() > 0;
          m_bid[p] <= bq.size() ? bq[0] : '0;
        end
        if (m_rvalid[p] && m_rready[p]) void'(rq[rsel].pop_front());
        if (!m_rvalid[p] || m_rready[p]) begin
          // the next beat from either ID's queue (beat-level interleaving)
          automatic logic nsel = (rq[0].size() && rq[1].size()) ? 1'($urandom) : rq[1].size() != 0;
          rsel <= nsel;
          m_rvalid[p] <= rq[nsel].size() > 0;
          m_rid[p] <= rq[nsel].size() ? rq[nsel][0].id : '0;
          m_rdata[p] <= rq[nsel].size() ? rq[nsel][0].d : '0;
          m_rlast[p] <= rq[nsel].size() ? rq[nsel][0].last : 1'b0;
        end
      end else begin
        m_bvalid[p] <= 1'b0; m_rvalid[p] <= 1'b0;
      end
    end
    assign m_bresp[p] = 2'b00;
    assign m_rresp[p] = 2'b00;
  end

  // ---------------------------------------------------------------- the native masters (clk)
  logic [511:0] sh [NPORT][logic [24:0]];
  typedef struct { logic we; logic [24:0] a; } cmd_t;
  cmd_t         cq [NPORT][$];
  logic [511:0] dq [NPORT][$], eq [NPORT][$];
  logic [63:0]  mq [NPORT][$];
  int           nwr [NPORT], nrd [NPORT];
  function automatic logic [511:0] nview(input int c, input logic [24:0] a);
    return sh[c].exists(a) ? sh[c][a] : rd64({5'(c), 23'(a)});
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
            2: begin
              x.a = 25'(8192 + run); run <= run + 1; slen <= 3 + $urandom % 14; swe <= x.we;
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
  // its beats: the upper half of each port's 512 MiB (the channels use the lower), and past the
  // last port (dropped writes, reads of zeros)
  logic [511:0] hsh [logic [27:0]];
  typedef struct { logic [15:0] id; logic [63:0] a; logic [7:0] len; } hx_t;
  hx_t          haw[$], har[$], hb_exp[$], hr_exp[$];
  logic [511:0] hw[$], hre[$];
  logic [63:0]  hm[$];
  logic         hl[$];
  int           hwr = 0, hrd = 0, hr_beat = 0;
  function automatic logic [511:0] hview(input logic [27:0] b);
    if (b[27:23] >= NPORT) return '0;
    return hsh.exists(b) ? hsh[b] : rd64(b);
  endfunction
  function automatic logic [63:0] haddr();
    automatic int p = $urandom % (NPORT < 32 ? NPORT + 1 : NPORT);
    return {30'd0, 5'(p), 1'b1, 22'(64 * ($urandom % 6)), 6'd0};
  endfunction
  always_ff @(posedge aclk) begin
    if (arst) begin
      h_awvalid <= 1'b0; h_wvalid <= 1'b0; h_arvalid <= 1'b0; h_bready <= 1'b0; h_rready <= 1'b0;
    end else begin
      h_bready <= $urandom % 3 != 0;
      h_rready <= $urandom % 3 != 0;
      if (gen && hr_exp.size() == 0 && haw.size() < 4 && $urandom % 12 == 0) begin
        automatic hx_t x;
        x.id = $urandom; x.len = $urandom % 4; x.a = haddr();
        for (int b = 0; b <= x.len; b++) begin
          automatic logic [511:0] d = rnd512();
          automatic logic [63:0]  m = ($urandom % 2) ? {$urandom, $urandom} : '1;
          if (x.a[33:29] < NPORT) hsh[x.a[33:6] + b] = merge(hview(x.a[33:6] + b), d, m);
          hw.push_back(d); hm.push_back(m); hl.push_back(b == x.len);
        end
        haw.push_back(x); hb_exp.push_back(x); hwr++;
      end
      if (gen && hr_exp.size() == 0 && haw.size() == 0 && hw.size() == 0 && !h_awvalid && !h_wvalid
          && $urandom % 10 == 0) begin
        automatic int batch = 1 + $urandom % 8;
        for (int i = 0; i < batch; i++) begin
          automatic hx_t x;
          x.id = $urandom; x.len = $urandom % 4; x.a = haddr();
          for (int b = 0; b <= x.len; b++) hre.push_back(hview(x.a[33:6] + b));
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
          if (h_rdata !== e) fail($sformatf("host: read %0d is not its own writes' (%h, expected %h)",
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
    @(negedge hclk) hrst = 1'b0;
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
    begin
      automatic int bursts = 0, longs = 0;
      for (int c = 0; c < NPORT; c++) begin
        if (n_wdone[c] != 16'(nwr[c])) fail($sformatf("ch%0d: n_wdone %0d, %0d writes", c, n_wdone[c], nwr[c]));
        if (eq[c].size()) fail($sformatf("ch%0d: %0d reads unanswered", c, eq[c].size()));
        foreach (sh[c][a]) if (rd64({5'(c), 23'(a)}) !== sh[c][a])
          fail($sformatf("ch%0d: beat %0d not in the HBM after n_wdone", c, a));
        bursts += nburst[c]; longs += nlong[c];
      end
      if (hb_exp.size() || hr_exp.size()) fail("host: responses missing");
      foreach (hsh[a]) if (rd64(a) !== hsh[a]) fail("host: a write is not in the HBM");
      if (HAZ && longs * 20 < bursts) fail($sformatf("few bursts: %0d of %0d transactions", longs, bursts));
      $display("NPORT %0d: %0d port transactions, %0d of them bursts of several beats; host %0d writes %0d reads",
               NPORT, bursts, longs, hwr, hrd);
    end
    for (int c = 0; c < NPORT; c++) $display("ch%0d %0d writes %0d reads", c, nwr[c], nrd[c]);
    if (errors) $display("FAIL (%0d errors)", errors);
    else $display("PASS");
    $finish;
  end
endmodule
