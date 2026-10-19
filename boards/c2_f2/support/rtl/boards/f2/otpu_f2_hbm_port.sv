// One HBM port's AXI4 slave side (the HDK's cl_axi_sc_1x1 SmartConnect: 512 bits, 16-bit IDs,
// at the core clock; it does the crossing into the HBM's 450 MHz, the 512 -> 256 width
// conversion and the AXI4 -> AXI3 conversion into the port) shared by its channel's master
// (otpu_native_axi: master 0, ID 0) and the host's PCIS path (otpu_f2_hbm_host: master 1, ID 1),
// the F2 HBM build's otpu_f2_mem per port, at the core clock since round 5 (docs/v4/f2-round5.md:
// nothing of openTPU's runs on the HBM clock; the hclk arbiter it replaces was the 450 MHz
// otpu_f2_hbm_port of rounds 1-4). The host reaches each pseudo-channel through the port that
// owns it, so neither master's traffic crosses the HBM's switch and no port needs a crossbar.
//
// AW: granted round-robin between the masters showing one, then that master's W beats follow to
// WLAST before the next AW is granted (W is never interleaved: AXI4's W order is AW order). AR:
// round-robin. A master's transactions carry ID = its index in bit 0 (the SmartConnect reflects
// the ID on B and R and returns one SI's responses in issue order), which routes R and B back.
// The address, W and AR outputs are registers (aw_o, w_o, ar_o: one transaction each, the
// SmartConnect's entry registers beside them); R and B pass through by ID. The reset through a
// register of its own (rst_q: the ports sit along the HBM, the leaves of the reset tree).
module otpu_f2_hbm_port (
  input  logic                clk,
  input  logic                rst,
  // ---- masters: [0] the channel's, [1] the host's
  input  logic [1:0]          s_awvalid,
  output logic [1:0]          s_awready,
  input  logic [1:0][33:0]    s_awaddr,
  input  logic [1:0][7:0]     s_awlen,
  input  logic [1:0]          s_wvalid,
  output logic [1:0]          s_wready,
  input  logic [1:0][511:0]   s_wdata,
  input  logic [1:0][63:0]    s_wstrb,
  input  logic [1:0]          s_wlast,
  output logic [1:0]          s_bvalid,
  input  logic [1:0]          s_bready,
  output logic [1:0]          s_bresp,       // (the same lines for both)
  input  logic [1:0]          s_arvalid,
  output logic [1:0]          s_arready,
  input  logic [1:0][33:0]    s_araddr,
  input  logic [1:0][7:0]     s_arlen,
  output logic [1:0]          s_rvalid,
  input  logic [1:0]          s_rready,
  output logic [511:0]        s_rdata,
  output logic [1:0]          s_rresp,
  output logic                s_rlast,
  // ---- the port (AXI4 slave: the SmartConnect's SI)
  output logic [15:0]         m_awid,
  output logic [33:0]         m_awaddr,
  output logic [7:0]          m_awlen,
  output logic                m_awvalid,
  input  logic                m_awready,
  output logic [511:0]        m_wdata,
  output logic [63:0]         m_wstrb,
  output logic                m_wlast,
  output logic                m_wvalid,
  input  logic                m_wready,
  input  logic [15:0]         m_bid,
  input  logic [1:0]          m_bresp,
  input  logic                m_bvalid,
  output logic                m_bready,
  output logic [15:0]         m_arid,
  output logic [33:0]         m_araddr,
  output logic [7:0]          m_arlen,
  output logic                m_arvalid,
  input  logic                m_arready,
  input  logic [15:0]         m_rid,
  input  logic [511:0]        m_rdata,
  input  logic [1:0]          m_rresp,
  input  logic                m_rlast,
  input  logic                m_rvalid,
  output logic                m_rready
);
  logic rst_q;
  always_ff @(posedge clk) rst_q <= rst;
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
  // the W readies from the arbiter's state alone (AXI lets ready wait for no valid)
  wire w_room = wbusy && (!w_o || m_wready);
  assign s_wready = {w_room && wsel, w_room && !wsel};
  wire w_take = w_room && s_wvalid[wsel];
  assign m_wvalid = w_o;
  always_ff @(posedge clk) begin
    if (aw_gnt != '0) begin
      m_awid <= {15'd0, aw_gnt[1]};
      m_awaddr <= s_awaddr[aw_gnt[1]];
      m_awlen <= s_awlen[aw_gnt[1]];
    end
    if (w_take) begin
      m_wdata <= s_wdata[wsel]; m_wstrb <= s_wstrb[wsel]; m_wlast <= s_wlast[wsel];
    end
    if (rst_q) begin
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
  always_ff @(posedge clk) begin
    if (ar_gnt != '0) begin
      m_arid <= {15'd0, ar_gnt[1]};
      m_araddr <= s_araddr[ar_gnt[1]];
      m_arlen <= s_arlen[ar_gnt[1]];
    end
    if (rst_q) begin
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
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst_q) begin
    if (m_bvalid && m_bid[15:1] != '0) $fatal(1, "otpu_f2_hbm_port: B with an ID not given");
    if (m_rvalid && m_rid[15:1] != '0) $fatal(1, "otpu_f2_hbm_port: R with an ID not given");
  end
`endif
endmodule
