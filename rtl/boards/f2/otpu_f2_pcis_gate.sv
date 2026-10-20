// The host's PCIS (AppPF BAR4: AXI4, 512 bits, aclk = clk_main_a0, 250 MHz) onto the host branch
// of the F2 HBM build's SmartConnect fabric (cl_opentpu_hbm.sv u_sc S_HOST: a 1 x NPORT
// SmartConnect demuxes it by address to the channels' SmartConnects, BAR4 offset = HBM address;
// docs/f2_hbm.md "The memory path"). It replaces otpu_f2_hbm_host's 450 MHz half: the address
// decode, the clock and width conversion are SmartConnect's.
//
// Every channel crosses a two-entry skid buffer (otpu_skidbuf: its input's ready a flip-flop,
// its output from registers), AW / W / AR towards the fabric, B / R towards the shell, so every
// output on either side is a register (the block sits beside the shell, pb_otpu_pcis in SLR1;
// the fabric is in SLR0: Alex's burst200 placed his f2_pcis_pipe register slice the same way).
//
// Order (otpu_f2_hbm_host's rule): a write is pending from its AW's take at the shell until its
// B is given back (wcnt), or while write data waits in its buffer. An AR leaves its buffer only
// while no write is pending (wpend: registers only, so a write taken in the cycle before is
// seen), so the host's read after its write returns the written data whichever channels the
// two use. pcis_wbusy, wpend registered, holds the OCL's writes (otpu_axil_cdc hold: it starts
// a write only from an AW three cycles old and registers hold again), so a RUN after the data
// finds it in the HBM: the SmartConnect returns the HBM's B, sent once the write is in the
// controller.
module otpu_f2_pcis_gate #(
  parameter bit HOLD = 1'b1             // reads held behind pending writes (0: tests only)
) (
  input  logic          aclk,
  input  logic          arst,
  // ---- the shell's PCIS (slave side)
  input  logic [15:0]   s_awid,
  input  logic [63:0]   s_awaddr,
  input  logic [7:0]    s_awlen,
  input  logic [2:0]    s_awsize,
  input  logic [1:0]    s_awburst,
  input  logic          s_awvalid,
  output logic          s_awready,
  input  logic [511:0]  s_wdata,
  input  logic [63:0]   s_wstrb,
  input  logic          s_wlast,
  input  logic          s_wvalid,
  output logic          s_wready,
  output logic [15:0]   s_bid,
  output logic [1:0]    s_bresp,
  output logic          s_bvalid,
  input  logic          s_bready,
  input  logic [15:0]   s_arid,
  input  logic [63:0]   s_araddr,
  input  logic [7:0]    s_arlen,
  input  logic [2:0]    s_arsize,
  input  logic [1:0]    s_arburst,
  input  logic          s_arvalid,
  output logic          s_arready,
  output logic [15:0]   s_rid,
  output logic [511:0]  s_rdata,
  output logic [1:0]    s_rresp,
  output logic          s_rlast,
  output logic          s_rvalid,
  input  logic          s_rready,
  // ---- towards the SmartConnect fabric (master side, the same clock)
  output logic [15:0]   m_awid,
  output logic [63:0]   m_awaddr,
  output logic [7:0]    m_awlen,
  output logic [2:0]    m_awsize,
  output logic [1:0]    m_awburst,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [511:0]  m_wdata,
  output logic [63:0]   m_wstrb,
  output logic          m_wlast,
  output logic          m_wvalid,
  input  logic          m_wready,
  input  logic [15:0]   m_bid,
  input  logic [1:0]    m_bresp,
  input  logic          m_bvalid,
  output logic          m_bready,
  output logic [15:0]   m_arid,
  output logic [63:0]   m_araddr,
  output logic [7:0]    m_arlen,
  output logic [2:0]    m_arsize,
  output logic [1:0]    m_arburst,
  output logic          m_arvalid,
  input  logic          m_arready,
  input  logic [15:0]   m_rid,
  input  logic [511:0]  m_rdata,
  input  logic [1:0]    m_rresp,
  input  logic          m_rlast,
  input  logic          m_rvalid,
  output logic          m_rready,
  output logic          pcis_wbusy
);
  logic arst_q;
  always_ff @(posedge aclk) arst_q <= arst;
  logic aw_any, w_any, ar_v, ar_r;
  logic [8:0] wcnt;                    // writes taken at the shell, B not yet given back
  wire wpend = wcnt != 0 || aw_any || w_any;
  otpu_skidbuf #(.W(16 + 64 + 8 + 3 + 2)) u_pcis_aw (
    .clk(aclk), .rst(arst_q), .i_v(s_awvalid), .i_r(s_awready),
    .i_d({s_awid, s_awaddr, s_awlen, s_awsize, s_awburst}),
    .o_v(m_awvalid), .o_r(m_awready), .o_d({m_awid, m_awaddr, m_awlen, m_awsize, m_awburst}),
    .any(aw_any));
  otpu_skidbuf #(.W(512 + 64 + 1)) u_pcis_w (
    .clk(aclk), .rst(arst_q), .i_v(s_wvalid), .i_r(s_wready), .i_d({s_wdata, s_wstrb, s_wlast}),
    .o_v(m_wvalid), .o_r(m_wready), .o_d({m_wdata, m_wstrb, m_wlast}), .any(w_any));
  otpu_skidbuf #(.W(16 + 64 + 8 + 3 + 2)) u_pcis_ar (
    .clk(aclk), .rst(arst_q), .i_v(s_arvalid), .i_r(s_arready),
    .i_d({s_arid, s_araddr, s_arlen, s_arsize, s_arburst}),
    .o_v(ar_v), .o_r(ar_r), .o_d({m_arid, m_araddr, m_arlen, m_arsize, m_arburst}), .any());
  assign m_arvalid = ar_v && !(HOLD && wpend);
  assign ar_r = m_arready && !(HOLD && wpend);
  otpu_skidbuf #(.W(16 + 2)) u_pcis_b (
    .clk(aclk), .rst(arst_q), .i_v(m_bvalid), .i_r(m_bready), .i_d({m_bid, m_bresp}),
    .o_v(s_bvalid), .o_r(s_bready), .o_d({s_bid, s_bresp}), .any());
  otpu_skidbuf #(.W(16 + 512 + 2 + 1)) u_pcis_r (
    .clk(aclk), .rst(arst_q), .i_v(m_rvalid), .i_r(m_rready), .i_d({m_rid, m_rdata, m_rresp, m_rlast}),
    .o_v(s_rvalid), .o_r(s_rready), .o_d({s_rid, s_rdata, s_rresp, s_rlast}), .any());
  wire aw_tk = s_awvalid && s_awready;
  wire b_tk = s_bvalid && s_bready;
  always_ff @(posedge aclk) begin
    if (arst_q) begin
      wcnt <= '0; pcis_wbusy <= 1'b0;
    end else begin
      wcnt <= wcnt + 9'(aw_tk) - 9'(b_tk);
      pcis_wbusy <= wpend;
    end
  end
endmodule
