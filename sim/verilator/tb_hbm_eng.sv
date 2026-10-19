// The v5 F2 HBM memory path (tests/test_f2_rtl.py test_hbm_eng): otpu_f2_hbm_eng (NPORT = S x
// NCH channels' native ports across the NXR-stage SLR chains and the per-slice write bus into
// one otpu_hbm_ch each; the PCIS through otpu_f2_pcis_gate) in front of models of what the CL
// puts behind it, at the clocks' ratios (core 200, shell 250, HBM 450: CORE_HALF 9, AXI_HALF
// 7, HBM_HALF 4):
//   - the HBM port (hclk), AXI3 256 bits: AW / AR taken with random readies and at most 64 in
//     flight a direction, reads served in order per ID group (bit 5: the mux's source) after
//     LRD (~126 hclk cycles, 280 ns) plus jitter, the two groups overtaking each other, a beat
//     a cycle with random bubbles; writes applied once their beats are in, B after LWR (~77,
//     172 ns) plus jitter; the one memory for all ports (the HBM's global address switch for
//     the host's port);
//   - the host chain (cl_axi_sc_1x1 from the PCIS gate's AXI4 512-bit branch at aclk to an
//     AXI3 256-bit master at hclk, single-threaded, in order, two beats a beat), onto its own
//     port NPORT (HOST_MUX 0) or into the DUT's hs_* and port NPORT - 1 through the mux (1).
// Checked: the native contract on every channel (a read returns the beat as the channel's
// earlier commands left it, n_wdone reaches the writes, every write is in the HBM at the
// end); every engine transaction inside its 512 MiB window, 64-byte aligned, 32-byte beats,
// INCR, ARLEN <= 15 and inside a 512 B group, AWLEN 1, WLAST on the second beat; n_err low;
// the host's writes land, its reads (right after the writes: the gate holds them) return its
// own writes, B / R in its order with its IDs; pcis_wbusy while a host write is unanswered;
// streams coalesce (the burst histogram). ORDER 0 (no barriers) must show stale reads.
// Plusargs +seed= +cycles= +nb=.
module tb_hbm_eng;
  parameter int NPORT = 8;
  parameter int S = 1;
  parameter int NCH = 8;
  parameter int NXR = 3;
  parameter bit HOST_MUX = 1'b0;
  parameter int CORE_HALF = 9, AXI_HALF = 7, HBM_HALF = 4;   // 200 / 250 / 450: the ratios
  parameter int LRD = 126, LWR = 77;      // the HBM's latencies, hclk cycles
  parameter int LSC = 30;                 // the host chain's crossing, hclk cycles each way
  parameter int BURST = 8, IDLE = 2, OT = 64;
  parameter bit ORDER = 1'b1;             // 0: the barriers off (must fail)

  logic clk = 1'b0, aclk = 1'b0, hclk = 1'b0;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;
  always #(HBM_HALF) hclk = !hclk;
  logic rst = 1'b1, arst = 1'b1, hrst = 1'b1;

  int unsigned seed = 1, cycles = 20000, nb = 6, errors = 0, nstale_ch = 0, nstale_host = 0;
  bit gen = 1'b0, dump_dbg = 1'b0;
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
  // the ports: 0 .. NPORT - 1 the DUT's, NPORT the host's own (HOST_MUX 0)
  logic [NPORT:0][5:0]    hp_awid, hp_wid, hp_bid, hp_arid, hp_rid;
  logic [NPORT:0][33:0]   hp_awaddr, hp_araddr;
  logic [NPORT:0][3:0]    hp_awlen, hp_arlen;
  logic [NPORT:0][2:0]    hp_awsize, hp_arsize;
  logic [NPORT:0][1:0]    hp_awburst, hp_arburst, hp_bresp, hp_rresp;
  logic [NPORT:0]         hp_awvalid, hp_awready, hp_wvalid, hp_wready, hp_wlast, hp_bvalid, hp_bready;
  logic [NPORT:0]         hp_arvalid, hp_arready, hp_rvalid, hp_rready, hp_rlast;
  logic [NPORT:0][255:0]  hp_wdata, hp_rdata;
  logic [NPORT:0][31:0]   hp_wstrb;
  // the host chain's AXI3 master (the SmartConnect model's output)
  logic [5:0]    sc_awid, sc_wid, sc_bid, sc_arid, sc_rid;
  logic [33:0]   sc_awaddr, sc_araddr;
  logic [3:0]    sc_awlen, sc_arlen;
  logic [2:0]    sc_awsize, sc_arsize;
  logic [1:0]    sc_awburst, sc_arburst, sc_bresp, sc_rresp;
  logic          sc_awvalid, sc_awready, sc_wvalid, sc_wready, sc_wlast, sc_bvalid, sc_bready;
  logic          sc_arvalid, sc_arready, sc_rvalid, sc_rready, sc_rlast;
  logic [255:0]  sc_wdata, sc_rdata;
  logic [31:0]   sc_wstrb;
  otpu_f2_hbm_eng #(.NPORT(NPORT), .S(S), .NCH(NCH), .NXR(NXR), .BURST(BURST), .IDLE(IDLE), .OT(OT),
                    .HOST_MUX(HOST_MUX), .ORDER(ORDER)) u_dut (
    .clk, .rst, .hclk, .hrst, .aclk, .arst,
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
    .hs_awid(sc_awid), .hs_awaddr(sc_awaddr), .hs_awlen(sc_awlen), .hs_awsize(sc_awsize),
    .hs_awburst(sc_awburst), .hs_awvalid(HOST_MUX ? sc_awvalid : 1'b0), .hs_awready(),
    .hs_wid(sc_wid), .hs_wdata(sc_wdata), .hs_wstrb(sc_wstrb), .hs_wlast(sc_wlast),
    .hs_wvalid(HOST_MUX ? sc_wvalid : 1'b0), .hs_wready(),
    .hs_bid(), .hs_bresp(), .hs_bvalid(), .hs_bready(sc_bready),
    .hs_arid(sc_arid), .hs_araddr(sc_araddr), .hs_arlen(sc_arlen), .hs_arsize(sc_arsize),
    .hs_arburst(sc_arburst), .hs_arvalid(HOST_MUX ? sc_arvalid : 1'b0), .hs_arready(),
    .hs_rid(), .hs_rdata(), .hs_rresp(), .hs_rlast(), .hs_rvalid(), .hs_rready(sc_rready),
    .p_awid(hp_awid[NPORT-1:0]), .p_awaddr(hp_awaddr[NPORT-1:0]), .p_awlen(hp_awlen[NPORT-1:0]),
    .p_awsize(hp_awsize[NPORT-1:0]), .p_awburst(hp_awburst[NPORT-1:0]),
    .p_awvalid(hp_awvalid[NPORT-1:0]), .p_awready(hp_awready[NPORT-1:0]),
    .p_wid(hp_wid[NPORT-1:0]), .p_wdata(hp_wdata[NPORT-1:0]), .p_wstrb(hp_wstrb[NPORT-1:0]),
    .p_wlast(hp_wlast[NPORT-1:0]), .p_wvalid(hp_wvalid[NPORT-1:0]), .p_wready(hp_wready[NPORT-1:0]),
    .p_bid(hp_bid[NPORT-1:0]), .p_bresp(hp_bresp[NPORT-1:0]), .p_bvalid(hp_bvalid[NPORT-1:0]),
    .p_bready(hp_bready[NPORT-1:0]),
    .p_arid(hp_arid[NPORT-1:0]), .p_araddr(hp_araddr[NPORT-1:0]), .p_arlen(hp_arlen[NPORT-1:0]),
    .p_arsize(hp_arsize[NPORT-1:0]), .p_arburst(hp_arburst[NPORT-1:0]),
    .p_arvalid(hp_arvalid[NPORT-1:0]), .p_arready(hp_arready[NPORT-1:0]),
    .p_rid(hp_rid[NPORT-1:0]), .p_rdata(hp_rdata[NPORT-1:0]), .p_rresp(hp_rresp[NPORT-1:0]),
    .p_rlast(hp_rlast[NPORT-1:0]), .p_rvalid(hp_rvalid[NPORT-1:0]), .p_rready(hp_rready[NPORT-1:0]));
  // the host chain's AXI3: onto the DUT's hs_* (the mux) or the model's port NPORT
  if (HOST_MUX) begin : g_hm
    assign sc_awready = u_dut.hs_awready; assign sc_wready = u_dut.hs_wready;
    assign sc_arready = u_dut.hs_arready;
    assign sc_bid = u_dut.hs_bid; assign sc_bresp = u_dut.hs_bresp; assign sc_bvalid = u_dut.hs_bvalid;
    assign sc_rid = u_dut.hs_rid; assign sc_rdata = u_dut.hs_rdata; assign sc_rresp = u_dut.hs_rresp;
    assign sc_rlast = u_dut.hs_rlast; assign sc_rvalid = u_dut.hs_rvalid;
    assign hp_awvalid[NPORT] = 1'b0; assign hp_wvalid[NPORT] = 1'b0; assign hp_arvalid[NPORT] = 1'b0;
    assign hp_bready[NPORT] = 1'b1; assign hp_rready[NPORT] = 1'b1;
    assign hp_awid[NPORT] = '0; assign hp_awaddr[NPORT] = '0; assign hp_awlen[NPORT] = '0;
    assign hp_awsize[NPORT] = '0; assign hp_awburst[NPORT] = '0; assign hp_wid[NPORT] = '0;
    assign hp_wdata[NPORT] = '0; assign hp_wstrb[NPORT] = '0; assign hp_wlast[NPORT] = 1'b0;
    assign hp_arid[NPORT] = '0; assign hp_araddr[NPORT] = '0; assign hp_arlen[NPORT] = '0;
    assign hp_arsize[NPORT] = '0; assign hp_arburst[NPORT] = '0;
  end else begin : g_hp
    assign hp_awid[NPORT] = sc_awid; assign hp_awaddr[NPORT] = sc_awaddr; assign hp_awlen[NPORT] = sc_awlen;
    assign hp_awsize[NPORT] = sc_awsize; assign hp_awburst[NPORT] = sc_awburst;
    assign hp_awvalid[NPORT] = sc_awvalid; assign sc_awready = hp_awready[NPORT];
    assign hp_wid[NPORT] = sc_wid; assign hp_wdata[NPORT] = sc_wdata; assign hp_wstrb[NPORT] = sc_wstrb;
    assign hp_wlast[NPORT] = sc_wlast; assign hp_wvalid[NPORT] = sc_wvalid; assign sc_wready = hp_wready[NPORT];
    assign sc_bid = hp_bid[NPORT]; assign sc_bresp = hp_bresp[NPORT]; assign sc_bvalid = hp_bvalid[NPORT];
    assign hp_bready[NPORT] = sc_bready;
    assign hp_arid[NPORT] = sc_arid; assign hp_araddr[NPORT] = sc_araddr; assign hp_arlen[NPORT] = sc_arlen;
    assign hp_arsize[NPORT] = sc_arsize; assign hp_arburst[NPORT] = sc_arburst;
    assign hp_arvalid[NPORT] = sc_arvalid; assign sc_arready = hp_arready[NPORT];
    assign sc_rid = hp_rid[NPORT]; assign sc_rdata = hp_rdata[NPORT]; assign sc_rresp = hp_rresp[NPORT];
    assign sc_rlast = hp_rlast[NPORT]; assign sc_rvalid = hp_rvalid[NPORT]; assign hp_rready[NPORT] = sc_rready;
  end

  // ---------------------------------------------------------------- the HBM's memory
  logic [511:0] mem [logic [27:0]];          // 64-byte beats of the 16 GiB
  function automatic logic [511:0] rd(input logic [27:0] b);
    return mem.exists(b) ? mem[b] : init(b);
  endfunction

  // ---------------------------------------------------------------- the HBM ports (hclk)
  // a transaction: its ID, first 32-byte beat, beats, the time it may be served; the two ID
  // groups (bit 5) each in order, overtaking each other
  typedef struct { logic [5:0] id; logic [28:0] b; int n; longint due; } tx_t;
  typedef struct { logic [255:0] d; logic [31:0] m; bit last; } wb_t;
  typedef struct { logic [5:0] id; logic [255:0] d; bit last; } rb_t;
  tx_t  arq [NPORT+1][2][$], awq [NPORT+1][$];    // (writes in AW order: their beats are)
  wb_t  wq [NPORT+1][$];
  rb_t  rq [NPORT+1][$];                  // the R beats to send, in order
  tx_t  bq [NPORT+1][2][$];               // the B's, per group
  int   hist [17];
  int   nbursts = 0, nlong = 0;
  for (genvar p = 0; p <= NPORT; p++) begin : g_port
    int rd_ot = 0, wr_ot = 0;
    int wseen = 0;                        // W beats taken, not yet consumed by a write
    logic [5:0] hb_id;
    always @(posedge hclk) begin
      hp_awready[p] <= $urandom % 4 != 0 && wr_ot < 64;
      hp_wready[p] <= $urandom % 4 != 0;
      hp_arready[p] <= $urandom % 4 != 0 && rd_ot < 64;
      if (!hrst) begin
        if (hp_arvalid[p] && hp_arready[p]) begin
          automatic bit eng = !hp_arid[p][5] && p < NPORT;
          if (eng && hp_araddr[p][33:29] != 5'(p)) fail($sformatf("port%0d: an engine read outside its pseudo-channel", p));
          if (hp_araddr[p][4:0] != 0 || hp_arsize[p] != 3'd5 || hp_arburst[p] != 2'b01)
            fail($sformatf("port%0d: AR not a 32-byte INCR burst", p));
          if (eng && (hp_araddr[p][5] != 1'b0 || hp_araddr[p][8:5] + hp_arlen[p] > 15 || hp_arlen[p][0] != 1'b1))
            fail($sformatf("port%0d: an engine burst unaligned, across 512 B or an odd number of beats", p));
          if (eng) begin hist[(hp_arlen[p] + 1) / 2]++; nbursts++; if (hp_arlen[p] > 1) nlong++; end
          arq[p][hp_arid[p][5]].push_back('{hp_arid[p], hp_araddr[p][33:5], hp_arlen[p] + 1,
                                            $time + 2 * HBM_HALF * LRD + $urandom % 64});
          rd_ot++;
        end
        if (hp_awvalid[p] && hp_awready[p]) begin
          automatic bit eng = !hp_awid[p][5] && p < NPORT;
          if (eng && hp_awaddr[p][33:29] != 5'(p)) fail($sformatf("port%0d: an engine write outside its pseudo-channel", p));
          if (hp_awaddr[p][4:0] != 0 || hp_awsize[p] != 3'd5 || hp_awburst[p] != 2'b01)
            fail($sformatf("port%0d: AW not a 32-byte INCR burst", p));
          if (eng && (hp_awlen[p] != 4'd1 || hp_awaddr[p][5] != 1'b0))
            fail($sformatf("port%0d: an engine write not one 64-byte beat", p));
          awq[p].push_back('{hp_awid[p], hp_awaddr[p][33:5], hp_awlen[p] + 1,
                             $time + 2 * HBM_HALF * LWR + $urandom % 64});
          wr_ot++;
        end
        if (hp_wvalid[p] && hp_wready[p]) begin
          wq[p].push_back('{hp_wdata[p], hp_wstrb[p], hp_wlast[p]});
          wseen++;
        end
        // reads: a due head of either group starts a burst's beats into rq (one burst at a time)
        if (rq[p].size() == 0) begin
          automatic int g = $urandom % 2;
          automatic bit got = 1'b0;
          for (int k = 0; k < 2 && !got; k++) begin
            automatic int gg = (g + k) % 2;
            if (arq[p][gg].size() && arq[p][gg][0].due <= $time) begin
              automatic tx_t t = arq[p][gg].pop_front();
              for (int i = 0; i < t.n; i++) begin
                automatic logic [28:0] b = t.b + 29'(i);
                automatic logic [511:0] v = rd(b[28:1]);
                rq[p].push_back('{t.id, b[0] ? v[511:256] : v[255:0], i == t.n - 1});
              end
              got = 1'b1;
            end
          end
        end
        // writes: the head once due with its beats all in (in AW order: the port's W beats
        // are); the B's per ID group, the groups overtaking each other
        if (awq[p].size() && awq[p][0].due <= $time && wq[p].size() >= awq[p][0].n) begin
          automatic tx_t t = awq[p].pop_front();
          for (int i = 0; i < t.n; i++) begin
            automatic wb_t w = wq[p][i];
            automatic logic [28:0] b = t.b + 29'(i);
            automatic logic [511:0] v = rd(b[28:1]);
            automatic logic [511:0] d = b[0] ? {w.d, 256'd0} : {256'd0, w.d};
            automatic logic [63:0] m = b[0] ? {w.m, 32'd0} : {32'd0, w.m};
            if (w.last != (i == t.n - 1)) fail($sformatf("port%0d: WLAST at the wrong beat", p));
            mem[b[28:1]] = merge(v, d, m);
          end
          for (int i = 0; i < t.n; i++) wq[p].delete(0);
          bq[p][t.id[5]].push_back('{t.id, t.b, t.n, $time + 2 * HBM_HALF * 8 + $urandom % 32});
        end
        // R and B from registers, held until taken
        if (!hp_rvalid[p] || hp_rready[p]) begin
          hp_rvalid[p] <= 1'b0;
          if (rq[p].size() && $urandom % 10 != 0) begin
            automatic rb_t r = rq[p].pop_front();
            hp_rvalid[p] <= 1'b1; hp_rid[p] <= r.id; hp_rdata[p] <= r.d; hp_rlast[p] <= r.last;
            if (r.last) rd_ot--;
          end
        end
        if (!hp_bvalid[p] || hp_bready[p]) begin
          automatic int g = $urandom % 2;
          hp_bvalid[p] <= 1'b0;
          for (int k = 0; k < 2; k++) begin
            automatic int gg = (g + k) % 2;
            if (bq[p][gg].size() && bq[p][gg][0].due <= $time) begin
              automatic tx_t t = bq[p][gg].pop_front();
              hp_bvalid[p] <= 1'b1; hp_bid[p] <= t.id;
              wr_ot--;
              break;
            end
          end
        end
      end else begin
        hp_rvalid[p] <= 1'b0; hp_bvalid[p] <= 1'b0;
      end
    end
    assign hp_rresp[p] = 2'b00;
    assign hp_bresp[p] = 2'b00;
  end

  // ---------------------------------------------------------------- the host chain (cl_axi_sc_1x1)
  // h_* (AXI4 512 bits, aclk) -> sc_* (AXI3 256 bits, hclk): single-threaded (ID 0), in order,
  // each AXI4 beat two AXI3 beats, LSC hclk cycles of crossing each way
  typedef struct { logic [15:0] id; logic [33:0] a; int n; longint due; } hx_t;
  typedef struct { logic [255:0] d; logic [31:0] m; bit last; longint due; } hw_t;
  typedef struct { logic [511:0] d; bit last; longint due; } hr_t;
  hx_t hawq [$], harq [$];               // towards hclk
  hw_t hwq [$];
  hx_t hb_ord [$], hr_ord [$];           // the AXI4 IDs in order, for B / R back
  hr_t hrq [$];                          // R beats back (512-bit), towards aclk
  longint hbq [$];                       // B's back
  logic [255:0] r_half;
  bit r_have = 1'b0;
  int r_cnt = 0;                         // AXI3 beats of the current AXI4 read burst returned
  always @(posedge aclk) begin
    h_awready <= $urandom % 4 != 0 && hawq.size() < 8;
    h_wready <= $urandom % 4 != 0 && hwq.size() < 32;
    h_arready <= $urandom % 4 != 0 && harq.size() < 8;
    if (!arst) begin
      if (h_awvalid && h_awready) begin
        if (h_awaddr[63:34] != 0 || h_awaddr[33:29] >= NPORT) fail("host: write outside the channels");
        hawq.push_back('{h_awid, h_awaddr[33:0], h_awlen + 1, $time + 2 * HBM_HALF * LSC});
        hb_ord.push_back('{h_awid, h_awaddr[33:0], h_awlen + 1, 0});
      end
      if (h_arvalid && h_arready) begin
        if (h_araddr[63:34] != 0 || h_araddr[33:29] >= NPORT) fail("host: read outside the channels");
        harq.push_back('{h_arid, h_araddr[33:0], h_arlen + 1, $time + 2 * HBM_HALF * LSC});
        hr_ord.push_back('{h_arid, h_araddr[33:0], h_arlen + 1, 0});
      end
      if (h_wvalid && h_wready) begin
        hwq.push_back('{h_wdata[255:0], h_wstrb[31:0], 1'b0, $time + 2 * HBM_HALF * LSC});
        hwq.push_back('{h_wdata[511:256], h_wstrb[63:32], h_wlast, $time + 2 * HBM_HALF * LSC});
      end
      // R and B back to the gate, in order
      if (!h_rvalid || h_rready) begin
        h_rvalid <= 1'b0;
        if (hrq.size() && hrq[0].due <= $time) begin
          h_rvalid <= 1'b1; h_rdata <= hrq[0].d; h_rlast <= hrq[0].last; h_rid <= hr_ord[0].id;
          if (hrq[0].last) void'(hr_ord.pop_front());
          void'(hrq.pop_front());
        end
      end
      if (!h_bvalid || h_bready) begin
        h_bvalid <= 1'b0;
        if (hbq.size() && hbq[0] <= $time) begin
          h_bvalid <= 1'b1; h_bid <= hb_ord[0].id;
          void'(hb_ord.pop_front()); void'(hbq.pop_front());
        end
      end
    end else begin
      h_rvalid <= 1'b0; h_bvalid <= 1'b0;
    end
  end
  assign h_rresp = 2'b00;
  assign h_bresp = 2'b00;
  assign sc_awid = 6'd0; assign sc_wid = 6'd0; assign sc_arid = 6'd0;
  assign sc_awsize = 3'd5; assign sc_arsize = 3'd5; assign sc_awburst = 2'b01; assign sc_arburst = 2'b01;
  always @(posedge hclk) begin
    if (hrst) begin
      sc_awvalid <= 1'b0; sc_wvalid <= 1'b0; sc_arvalid <= 1'b0; sc_bready <= 1'b0; sc_rready <= 1'b0;
    end else begin
      sc_bready <= $urandom % 3 != 0;
      sc_rready <= $urandom % 3 != 0;
      if (!sc_awvalid || sc_awready) begin
        sc_awvalid <= 1'b0;
        if (hawq.size() && hawq[0].due <= $time) begin
          sc_awvalid <= 1'b1; sc_awaddr <= hawq[0].a; sc_awlen <= 4'(2 * hawq[0].n - 1);
          void'(hawq.pop_front());
        end
      end
      if (!sc_arvalid || sc_arready) begin
        sc_arvalid <= 1'b0;
        if (harq.size() && harq[0].due <= $time) begin
          sc_arvalid <= 1'b1; sc_araddr <= harq[0].a; sc_arlen <= 4'(2 * harq[0].n - 1);
          void'(harq.pop_front());
        end
      end
      if (!sc_wvalid || sc_wready) begin
        sc_wvalid <= 1'b0;
        if (hwq.size() && hwq[0].due <= $time) begin
          sc_wvalid <= 1'b1; sc_wdata <= hwq[0].d; sc_wstrb <= hwq[0].m; sc_wlast <= hwq[0].last;
          void'(hwq.pop_front());
        end
      end
      if (sc_rvalid && sc_rready) begin
        if (sc_rid != 6'd0) fail("host chain: an R with another ID");
        if (!r_have) begin r_half = sc_rdata; r_have = 1'b1; end
        else begin
          r_have = 1'b0;
          hrq.push_back('{{sc_rdata, r_half}, sc_rlast, $time + 2 * HBM_HALF * LSC});
        end
      end
      if (sc_bvalid && sc_bready) begin
        if (sc_bid != 6'd0) fail("host chain: a B with another ID");
        hbq.push_back($time + 2 * HBM_HALF * LSC);
      end
    end
  end

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
            if (n_rdata[c] !== e) begin
              nstale_ch++; fail($sformatf("ch%0d: read returned stale or wrong data", c));
            end
          end
        end
        if (n_err[c]) fail($sformatf("ch%0d: n_err", c));
      end
    end
`ifdef HBMENG_DEBUG
    always @(posedge dump_dbg) if (eq[c].size() || n_wdone[c] != 16'(nwr[c]))
      $display("ch%0d stuck: cq %0d/%0d wq %0d/%0d ob_v %0d ob_cl %0d ob_open %0d ob_full %0d ob_len %0d ar_v %0d ot_room %0d rd_ot %0d rd_res %0d beat_room %0d wr_pend %0d no_wr %0d no_rd %0d aw_v %0d w_v %0d w_busy %0d rq_wcount %0d herr %0d idle_to %0d cq_rvalid %0d h_we %0d h_cont %0d wq_rvalid %0d",
        c, u_dut.g_port[c].u_eng.cq_wcount, u_dut.g_port[c].u_eng.u_cq.rcount, u_dut.g_port[c].u_eng.wq_wcount, u_dut.g_port[c].u_eng.u_wq.rcount,
        u_dut.g_port[c].u_eng.ob_v, u_dut.g_port[c].u_eng.ob_cl, u_dut.g_port[c].u_eng.ob_open, u_dut.g_port[c].u_eng.ob_full, u_dut.g_port[c].u_eng.ob_len,
        u_dut.g_port[c].u_eng.ar_v, u_dut.g_port[c].u_eng.ot_room, u_dut.g_port[c].u_eng.rd_ot, u_dut.g_port[c].u_eng.rd_res, u_dut.g_port[c].u_eng.beat_room,
        u_dut.g_port[c].u_eng.wr_pend, u_dut.g_port[c].u_eng.no_wr, u_dut.g_port[c].u_eng.no_rd, u_dut.g_port[c].u_eng.aw_v, u_dut.g_port[c].u_eng.w_v,
        u_dut.g_port[c].u_eng.w_busy, u_dut.g_port[c].u_eng.rq_wcount, u_dut.g_port[c].u_eng.herr, u_dut.g_port[c].u_eng.idle_to,
        u_dut.g_port[c].u_eng.cq_rvalid, u_dut.g_port[c].u_eng.h_we, u_dut.g_port[c].u_eng.h_cont, u_dut.g_port[c].u_eng.wq_rvalid);
`endif
  end

  // ---------------------------------------------------------------- the host (aclk)
  // its beats: the upper half of each port's 512 MiB (the channels use the lower)
  logic [511:0] hsh [logic [27:0]];
  typedef struct { logic [15:0] id; logic [63:0] a; logic [7:0] len; } hg_t;
  hg_t          haw[$], har[$], hb_exp[$], hr_exp[$];
  logic [511:0] hw[$], hre[$];
  logic [63:0]  hm[$];
  logic         hl[$];
  int           hwr = 0, hrd = 0, hr_beat = 0, hb_out = 0, wbusy_low = 0;
  logic [2:0]   hb_age;
  logic [63:0]  hlast = 64'h30000000;
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
        automatic hg_t x;
        x.id = $urandom; x.len = $urandom % 4; x.a = haddr();
        for (int b = 0; b <= x.len; b++) begin
          automatic logic [511:0] d = rnd512();
          automatic logic [63:0]  m = ($urandom % 2) ? {$urandom, $urandom} : '1;
          hsh[x.a[33:6] + b] = merge(hview(x.a[33:6] + b), d, m);
          hw.push_back(d); hm.push_back(m); hl.push_back(b == x.len);
        end
        haw.push_back(x); hb_exp.push_back(x); hwr++; hlast = x.a;
      end
      if (gen && hr_exp.size() == 0 && haw.size() == 0 && hw.size() == 0 && !s_awvalid && !s_wvalid
          && $urandom % 10 == 0) begin
        automatic int batch = 1 + $urandom % 8;
        for (int i = 0; i < batch; i++) begin
          automatic hg_t x;
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
      hb_out <= hb_out + (s_awvalid && s_awready) - (s_bvalid && s_bready);
      hb_age <= {hb_age[1:0], hb_out != 0};
      if (&hb_age && hb_out != 0 && !pcis_wbusy) wbusy_low++;
      if (s_rvalid && s_rready) begin
        if (hre.size() == 0) fail("host: read data with no read");
        else begin
          automatic logic [511:0] e = hre.pop_front();
          if (s_rdata !== e) begin
            nstale_host++;
            fail($sformatf("host: read %0d is not its own writes' (%h, expected %h)", hrd, s_rdata[31:0], e[31:0]));
          end
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
    for (int c = 0; c < NPORT; c++) begin
      if (n_wdone[c] != 16'(nwr[c])) fail($sformatf("ch%0d: n_wdone %0d, %0d writes", c, n_wdone[c], nwr[c]));
      if (eq[c].size()) fail($sformatf("ch%0d: %0d reads unanswered", c, eq[c].size()));
      foreach (sh[c][a]) if (rd({5'(c), 23'(a)}) !== sh[c][a])
        fail($sformatf("ch%0d: beat %0d not in the HBM after n_wdone", c, a));
    end
    if (hb_exp.size() || hr_exp.size()) fail("host: responses missing");
    foreach (hsh[a]) if (rd(a) !== hsh[a]) fail("host: a write is not in the HBM");
    if (wbusy_low) fail($sformatf("pcis_wbusy low for %0d cycles with a host write unanswered", wbusy_low));
    if (BURST > 1 && nlong * 20 < nbursts) fail($sformatf("few bursts: %0d of %0d reads", nlong, nbursts));
    $write("NPORT %0d: %0d engine read bursts, %0d of several beats; lengths:", NPORT, nbursts, nlong);
    for (int i = 1; i <= 8; i++) if (hist[i]) $write(" %0d:%0d", i, hist[i]);
    $display("; host %0d writes %0d reads", hwr, hrd);
    for (int c = 0; c < NPORT; c++) $display("ch%0d %0d writes %0d reads", c, nwr[c], nrd[c]);
dump_dbg = 1'b1; #1;
    if (errors) $display("FAIL (%0d errors: %0d stale channel reads, %0d stale host reads)",
                         errors, nstale_ch, nstale_host);
    else $display("PASS");
    $finish;
  end
endmodule
