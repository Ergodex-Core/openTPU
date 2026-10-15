// The slot's DIMM (sh_ddr's AXI4 slave: 512 bits, 16-bit IDs, aclk) shared by three masters: the
// slice's two channel bridges (otpu_f2_nch: single beats, 0 and 1) and the host (the shell's PCIS,
// AppPF BAR4: bursts with its own IDs, 2). One clock (aclk, clk_main_a0).
//
// Writes: an AW is granted round-robin among the masters showing one, and the W beats of that
// master follow until WLAST, before the next AW is granted (W is never interleaved). Reads: an AR
// is granted round-robin. Each master's transactions go to the DIMM with the ID of the master
// (0, 1, 2), so its B's and R's come back in its order and route back by ID; the host's own IDs
// wait in a FIFO per direction and go back with its responses (its B's and R's are then in its
// AW / AR order, which AXI allows for any IDs).
//
// The host's order: its PCIS reads wait until its writes are answered (pcis_wbusy: an AW taken
// from the shell, its B not given back, or AW/W already presented), so a read after a write
// returns the written data (PCIe
// does not let a read pass a posted write; the DIMM's controller and this arbiter could). The
// control path (otpu_f2_sys) holds the OCL's writes on pcis_wbusy too: a RUN, or a word a WAITW
// waits for, written after data reaches the slice after the data. Both hold only for what the
// shell has delivered: the host reads BAR4 once after its writes (opentpu/host/f2.py) so that
// every write is delivered first.
module otpu_f2_mem #(
  parameter int HQ = 32                 // host transactions outstanding per direction (ID FIFOs)
) (
  input  logic                  aclk,
  input  logic                  arst,
  // ---- masters 0, 1: the channel bridges (single beats, no IDs)
  input  logic [1:0]            c_awvalid,
  output logic [1:0]            c_awready,
  input  logic [1:0][63:0]      c_awaddr,
  input  logic [1:0]            c_wvalid,
  output logic [1:0]            c_wready,
  input  logic [1:0][511:0]     c_wdata,
  input  logic [1:0][63:0]      c_wstrb,
  output logic [1:0]            c_bvalid,
  input  logic [1:0]            c_bready,
  output logic [1:0]            c_bresp,  // one per master: c_bresp[m] (the same lines)
  input  logic [1:0]            c_arvalid,
  output logic [1:0]            c_arready,
  input  logic [1:0][63:0]      c_araddr,
  output logic [1:0]            c_rvalid,
  input  logic [1:0]            c_rready,
  output logic [511:0]          c_rdata,
  output logic [1:0]            c_rresp,
  // ---- master 2: the host (the shell's PCIS)
  input  logic [15:0]           h_awid,
  input  logic [63:0]           h_awaddr,
  input  logic [7:0]            h_awlen,
  input  logic [2:0]            h_awsize,
  input  logic [1:0]            h_awburst,
  input  logic                  h_awvalid,
  output logic                  h_awready,
  input  logic [511:0]          h_wdata,
  input  logic [63:0]           h_wstrb,
  input  logic                  h_wlast,
  input  logic                  h_wvalid,
  output logic                  h_wready,
  output logic [15:0]           h_bid,
  output logic [1:0]            h_bresp,
  output logic                  h_bvalid,
  input  logic                  h_bready,
  input  logic [15:0]           h_arid,
  input  logic [63:0]           h_araddr,
  input  logic [7:0]            h_arlen,
  input  logic [2:0]            h_arsize,
  input  logic [1:0]            h_arburst,
  input  logic                  h_arvalid,
  output logic                  h_arready,
  output logic [15:0]           h_rid,
  output logic [511:0]          h_rdata,
  output logic [1:0]            h_rresp,
  output logic                  h_rlast,
  output logic                  h_rvalid,
  input  logic                  h_rready,
  output logic                  pcis_wbusy,
  // ---- the DIMM (sh_ddr)
  output logic [15:0]           d_awid,
  output logic [63:0]           d_awaddr,
  output logic [7:0]            d_awlen,
  output logic [2:0]            d_awsize,
  output logic [1:0]            d_awburst,
  output logic                  d_awvalid,
  input  logic                  d_awready,
  output logic [511:0]          d_wdata,
  output logic [63:0]           d_wstrb,
  output logic                  d_wlast,
  output logic                  d_wvalid,
  input  logic                  d_wready,
  input  logic [15:0]           d_bid,
  input  logic [1:0]            d_bresp,
  input  logic                  d_bvalid,
  output logic                  d_bready,
  output logic [15:0]           d_arid,
  output logic [63:0]           d_araddr,
  output logic [7:0]            d_arlen,
  output logic [2:0]            d_arsize,
  output logic [1:0]            d_arburst,
  output logic                  d_arvalid,
  input  logic                  d_arready,
  input  logic [15:0]           d_rid,
  input  logic [511:0]          d_rdata,
  input  logic [1:0]            d_rresp,
  input  logic                  d_rlast,
  input  logic                  d_rvalid,
  output logic                  d_rready
);
  localparam int HW = $clog2(HQ);
  initial if (HQ < 2 || HQ != (1 << HW))
    $fatal(1, "otpu_f2_mem: HQ must be a power of two >= 2");

  // ---- the host's IDs, in AW / AR order (their responses come back in that order: ID 2)
  logic [HQ-1:0][15:0] hbq, hrq;
  logic [HW:0]         hb_n, hr_n;
  logic [HW-1:0]       hb_h, hb_t, hr_h, hr_t;
  logic                h_aw_tk, h_ar_tk, h_b_tk, h_r_tk;

  // ---- write path: AW granted round-robin, then that master's W beats to WLAST
  logic [2:0] awreq, aw_gnt;
  logic [1:0] aw_last, wsel;
  logic       wbusy;                    // a granted AW's W beats are still to go
  logic       aw_o;                     // the DIMM's AW register holds one
  assign awreq = {h_awvalid && !hb_n[HW], c_awvalid};
  always_comb begin
    aw_gnt = '0;
    if (!aw_o && !wbusy)
      for (int k = 1; k <= 3; k++) begin
        automatic int m = (int'(aw_last) + k) % 3;
        if (awreq[m] && aw_gnt == '0) aw_gnt[m] = 1'b1;
      end
  end
  assign c_awready = aw_gnt[1:0];
  assign h_awready = aw_gnt[2];
  assign h_aw_tk = aw_gnt[2];
  assign d_awvalid = aw_o;

  // W: the granted master's beats through to the DIMM (registered: d_w*)
  logic w_o;
  logic w_take;
  logic s_wvalid, s_wlast;
  always_comb begin
    case (wsel)
      2'd0: s_wvalid = c_wvalid[0];
      2'd1: s_wvalid = c_wvalid[1];
      default: s_wvalid = h_wvalid;
    endcase
    s_wlast = (wsel == 2'd2) ? h_wlast : 1'b1;
  end
  assign w_take = wbusy && s_wvalid && (!w_o || d_wready);
  assign c_wready = {w_take && wsel == 2'd1, w_take && wsel == 2'd0};
  assign h_wready = w_take && wsel == 2'd2;
  assign d_wvalid = w_o;

  always_ff @(posedge aclk) begin
    if (aw_gnt != '0) begin
      d_awid    <= {14'd0, aw_gnt[2], aw_gnt[1]};
      d_awaddr  <= aw_gnt[2] ? h_awaddr : aw_gnt[1] ? c_awaddr[1] : c_awaddr[0];
      d_awlen   <= aw_gnt[2] ? h_awlen : 8'd0;
      d_awsize  <= aw_gnt[2] ? h_awsize : 3'd6;
      d_awburst <= aw_gnt[2] ? h_awburst : 2'b01;
    end
    if (w_take) begin
      d_wdata <= (wsel == 2'd2) ? h_wdata : c_wdata[wsel[0]];
      d_wstrb <= (wsel == 2'd2) ? h_wstrb : c_wstrb[wsel[0]];
      d_wlast <= s_wlast;
    end
    if (arst) begin
      aw_o <= 1'b0; w_o <= 1'b0; wbusy <= 1'b0; aw_last <= 2'd2; wsel <= 2'd0;
    end else begin
      aw_o <= (aw_gnt != '0) || (aw_o && !d_awready);
      w_o  <= w_take || (w_o && !d_wready);
      if (aw_gnt != '0) begin
        aw_last <= aw_gnt[2] ? 2'd2 : aw_gnt[1] ? 2'd1 : 2'd0;
        wsel    <= aw_gnt[2] ? 2'd2 : aw_gnt[1] ? 2'd1 : 2'd0;
        wbusy   <= 1'b1;
      end else if (w_take && s_wlast) begin
        wbusy   <= 1'b0;
      end
    end
  end

  // B: by ID, straight through (bready from the master it belongs to)
  logic [1:0] bm;
  assign bm = d_bid[1:0];
  assign c_bvalid = {d_bvalid && bm == 2'd1, d_bvalid && bm == 2'd0};
  assign c_bresp  = d_bresp;
  assign h_bvalid = d_bvalid && bm == 2'd2;
  assign h_bresp  = d_bresp;
  assign h_bid    = hbq[hb_h];
  assign d_bready = (bm == 2'd2) ? h_bready : c_bready[bm[0]];
  assign h_b_tk   = h_bvalid && h_bready;
  // Include a write waiting for arbitration, and W arriving before AW. Otherwise an AR or
  // OCL control write can pass the very first PCIS write before hb_n increments.
  assign pcis_wbusy = hb_n != 0 || h_awvalid || h_wvalid;

  // ---- read path: AR granted round-robin into a two-entry queue toward the DIMM (granted while
  // it has room, a registered count: one AR every cycle while the DIMM takes one every cycle;
  // the one AR register of before granted every second cycle at most); the host's reads wait for
  // its writes
  localparam int ARW = 16 + 64 + 8 + 3 + 2;
  logic [2:0] arreq, ar_gnt;
  logic [1:0] ar_last;
  logic [1:0] arn;
  logic [ARW-1:0] arq0, arq1, ar_new;
  logic       ar_pop;
  assign arreq = {h_arvalid && !hr_n[HW] && !pcis_wbusy, c_arvalid};
  always_comb begin
    ar_gnt = '0;
    if (arn != 2'd2)
      for (int k = 1; k <= 3; k++) begin
        automatic int m = (int'(ar_last) + k) % 3;
        if (arreq[m] && ar_gnt == '0) ar_gnt[m] = 1'b1;
      end
  end
  assign c_arready = ar_gnt[1:0];
  assign h_arready = ar_gnt[2];
  assign h_ar_tk = ar_gnt[2];
  assign d_arvalid = arn != 2'd0;
  assign ar_pop = d_arvalid && d_arready;
  assign {d_arid, d_araddr, d_arlen, d_arsize, d_arburst} = arq0;
  assign ar_new = {14'd0, ar_gnt[2], ar_gnt[1],
                   ar_gnt[2] ? h_araddr : ar_gnt[1] ? c_araddr[1] : c_araddr[0],
                   ar_gnt[2] ? h_arlen : 8'd0, ar_gnt[2] ? h_arsize : 3'd6,
                   ar_gnt[2] ? h_arburst : 2'b01};
  always_ff @(posedge aclk) begin
    if (ar_pop) arq0 <= arq1;
    if (ar_gnt != '0) begin
      if (arn == 2'd0 || (arn == 2'd1 && ar_pop)) arq0 <= ar_new;
      else arq1 <= ar_new;
    end
    if (arst) begin
      arn <= 2'd0; ar_last <= 2'd2;
    end else begin
      arn <= arn + 2'(ar_gnt != '0) - 2'(ar_pop);
      if (ar_gnt != '0) ar_last <= ar_gnt[2] ? 2'd2 : ar_gnt[1] ? 2'd1 : 2'd0;
    end
  end

  // R: by ID, straight through
  logic [1:0] rm;
  assign rm = d_rid[1:0];
  assign c_rvalid = {d_rvalid && rm == 2'd1, d_rvalid && rm == 2'd0};
  assign c_rdata  = d_rdata;
  assign c_rresp  = d_rresp;
  assign h_rvalid = d_rvalid && rm == 2'd2;
  assign h_rdata  = d_rdata;
  assign h_rresp  = d_rresp;
  assign h_rlast  = d_rlast;
  assign h_rid    = hrq[hr_h];
  assign d_rready = (rm == 2'd2) ? h_rready : c_rready[rm[0]];
  assign h_r_tk   = h_rvalid && h_rready && d_rlast;

  // ---- the host's ID FIFOs
  always_ff @(posedge aclk) begin
    if (h_aw_tk) hbq[hb_t] <= h_awid;
    if (h_ar_tk) hrq[hr_t] <= h_arid;
    if (arst) begin
      hb_n <= '0; hb_h <= '0; hb_t <= '0;
      hr_n <= '0; hr_h <= '0; hr_t <= '0;
    end else begin
      if (h_aw_tk) hb_t <= hb_t + 1'b1;
      if (h_b_tk)  hb_h <= hb_h + 1'b1;
      hb_n <= hb_n + (HW + 1)'(h_aw_tk) - (HW + 1)'(h_b_tk);
      if (h_ar_tk) hr_t <= hr_t + 1'b1;
      if (h_r_tk)  hr_h <= hr_h + 1'b1;
      hr_n <= hr_n + (HW + 1)'(h_ar_tk) - (HW + 1)'(h_r_tk);
    end
  end
endmodule
