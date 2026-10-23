// One HBM AXI port (AXI3, 256 bits, hclk) shared by its channel's bridge (otpu_f2_hbm_nch: master
// 0) and the host's PCIS path (otpu_f2_hbm_host: master 1), the F2 HBM build's otpu_f2_mem per
// port: the host reaches each pseudo-channel through the port that owns it, so neither master's
// traffic crosses the HBM's switch and no port needs a crossbar (docs/f2_hbm.md).
//
// AW: granted round-robin between the masters showing one, then that master's W beats follow to
// WLAST before the next AW is granted (W is never interleaved). AR: round-robin. A master's
// transactions carry ID = its index, so its R and B come back in its own order (the HBM orders
// one ID's responses; it may interleave the two IDs' read bursts, beat by beat, which the ID bit
// routes). The address, W and AR outputs are registered (aw_o, w_o, ar_o: one transaction each);
// R and B pass through by ID. A port with no channel (an unused bridge) is tied off by its parent.
// The reset through a register of its own (hrst_q: the arbiters sit at the HBM's ports, a
// reset tree's leaves; otpu_f2_hbm_nch).
module otpu_f2_hbm_port (
  input  logic                hclk,
  input  logic                hrst,
  // ---- masters: [0] the channel bridge, [1] the host
  input  logic [1:0]          s_awvalid,
  output logic [1:0]          s_awready,
  input  logic [1:0][33:0]    s_awaddr,
  input  logic [1:0][3:0]     s_awlen,
  input  logic [1:0]          s_wvalid,
  output logic [1:0]          s_wready,
  input  logic [1:0][255:0]   s_wdata,
  input  logic [1:0][31:0]    s_wstrb,
  input  logic [1:0]          s_wlast,
  output logic [1:0]          s_bvalid,
  input  logic [1:0]          s_bready,
  output logic [1:0]          s_bresp,       // (the same lines for both)
  input  logic [1:0]          s_arvalid,
  output logic [1:0]          s_arready,
  input  logic [1:0][33:0]    s_araddr,
  input  logic [1:0][3:0]     s_arlen,
  output logic [1:0]          s_rvalid,
  input  logic [1:0]          s_rready,
  output logic [255:0]        s_rdata,
  output logic [1:0]          s_rresp,
  output logic                s_rlast,
  // ---- the HBM port (AXI3 slave)
  output logic [5:0]          m_awid,
  output logic [33:0]         m_awaddr,
  output logic [3:0]          m_awlen,
  output logic                m_awvalid,
  input  logic                m_awready,
  output logic [255:0]        m_wdata,
  output logic [31:0]         m_wstrb,
  output logic                m_wlast,
  output logic                m_wvalid,
  input  logic                m_wready,
  input  logic [5:0]          m_bid,
  input  logic [1:0]          m_bresp,
  input  logic                m_bvalid,
  output logic                m_bready,
  output logic [5:0]          m_arid,
  output logic [33:0]         m_araddr,
  output logic [3:0]          m_arlen,
  output logic                m_arvalid,
  input  logic                m_arready,
  input  logic [5:0]          m_rid,
  input  logic [255:0]        m_rdata,
  input  logic [1:0]          m_rresp,
  input  logic                m_rlast,
  input  logic                m_rvalid,
  output logic                m_rready
);
  logic hrst_q;
  always_ff @(posedge hclk) hrst_q <= hrst;
  // ---- AW, then the granted master's W to WLAST
  logic aw_o, w_o, wbusy, wsel, aw_last;
  logic [1:0] aw_gnt;
  always_comb begin
    aw_gnt = '0;
    if (!aw_o && !wbusy) begin
      if (s_awvalid[!aw_last]) aw_gnt[!aw_last] = 1'b1;
      else if (s_awvalid[aw_last]) aw_gnt[aw_last] = 1'b1;
    end
  end
  assign s_awready = aw_gnt;
  assign m_awvalid = aw_o;
  // the W readies from the arbiter's state alone (AXI lets ready wait for no valid): a master's
  // pop does not run through the other master's valid (the 89bbb6bc synthesis: a bridge's W
  // FIFO count into the host's state through the W select, -0.31 ns at 450 MHz)
  wire w_room = wbusy && (!w_o || m_wready);
  assign s_wready = {w_room && wsel, w_room && !wsel};
  wire w_take = w_room && s_wvalid[wsel];
  assign m_wvalid = w_o;
  always_ff @(posedge hclk) begin
    if (aw_gnt != '0) begin
      m_awid <= {5'd0, aw_gnt[1]};
      m_awaddr <= s_awaddr[aw_gnt[1]];
      m_awlen <= s_awlen[aw_gnt[1]];
    end
    if (w_take) begin
      m_wdata <= s_wdata[wsel]; m_wstrb <= s_wstrb[wsel]; m_wlast <= s_wlast[wsel];
    end
    if (hrst_q) begin
      aw_o <= 1'b0; w_o <= 1'b0; wbusy <= 1'b0; wsel <= 1'b0; aw_last <= 1'b1;
    end else begin
      aw_o <= (aw_gnt != '0) || (aw_o && !m_awready);
      w_o <= w_take || (w_o && !m_wready);
      if (aw_gnt != '0) begin
        aw_last <= aw_gnt[1]; wsel <= aw_gnt[1]; wbusy <= 1'b1;
      end else if (w_take && s_wlast[wsel]) wbusy <= 1'b0;
    end
  end
  // B by ID
  assign s_bvalid = {m_bvalid && m_bid[0], m_bvalid && !m_bid[0]};
  assign s_bresp = m_bresp;
  assign m_bready = s_bready[m_bid[0]];

  // ---- AR, round-robin
  logic ar_o, ar_last;
  logic [1:0] ar_gnt;
  always_comb begin
    ar_gnt = '0;
    if (!ar_o) begin
      if (s_arvalid[!ar_last]) ar_gnt[!ar_last] = 1'b1;
      else if (s_arvalid[ar_last]) ar_gnt[ar_last] = 1'b1;
    end
  end
  assign s_arready = ar_gnt;
  assign m_arvalid = ar_o;
  always_ff @(posedge hclk) begin
    if (ar_gnt != '0) begin
      m_arid <= {5'd0, ar_gnt[1]};
      m_araddr <= s_araddr[ar_gnt[1]];
      m_arlen <= s_arlen[ar_gnt[1]];
    end
    if (hrst_q) begin
      ar_o <= 1'b0; ar_last <= 1'b1;
    end else begin
      ar_o <= (ar_gnt != '0) || (ar_o && !m_arready);
      if (ar_gnt != '0) ar_last <= ar_gnt[1];
    end
  end
  // R by ID
  assign s_rvalid = {m_rvalid && m_rid[0], m_rvalid && !m_rid[0]};
  assign s_rdata = m_rdata;
  assign s_rresp = m_rresp;
  assign s_rlast = m_rlast;
  assign m_rready = s_rready[m_rid[0]];
endmodule
