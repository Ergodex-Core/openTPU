// AXI4-Lite across the SLRs of the F2 HBM build (otpu_f2_hbm_sys u_cx; docs/wide_port.md
// "Placement"): from the OCL's clock crossing (otpu_axil_cdc u_ocl, its clk side, in SLR0 with
// the HBM fabric) to the board's control registers (otpu_board u_iso, in SLR2 with the slice)
// through NXR register stages each way (otpu_xr; g_xr.g_x[k], one per SLR: k = 0 beside the
// master, k = NXR - 1 beside the slave; NXR 0: wired). An AXI4-Lite handshake (a valid held until
// a ready that depends on it in the cycle) cannot be pipelined as it is; this master is
// otpu_axil_cdc's clk side, which runs one access at a time: AW and W shown together and held
// until taken, AR held until taken, and ready for the B or R from the access until it comes
// (m_bready / m_rready = busy). So every channel crosses as a pulse with its word. Down (u_m):
// a request pulse when the master shows a valid not yet sent (sent_* holds it off until the
// acknowledgement); on the slave side (u_s) the request is held (v_*) until the slave takes it,
// and the take goes back up as a pulse that is the master's ready for one cycle (sent_* clears
// with it: the master drops its valid the next cycle, as AXI requires, and a new one is a new
// request). Up: a response (B, R) is taken on the slave side the cycle it is shown (the master
// is ready for it) and goes up as a pulse that is the master's valid for one cycle. An access
// costs 2 NXR cycles more each way; the control registers are accessed microseconds apart.
// The stages take no reset (a pulse in flight at a reset flushes in NXR cycles while both
// sides hold it); u_m resets with the master's side (rst, SLR0's), u_s with the slave's (srst,
// the board's copy, NXR cycles later).
module otpu_axil_xr #(
  parameter int NXR = 0                 // register stages each way (0: wired)
) (
  input  logic         clk,
  input  logic         rst,             // the master side's reset
  input  logic         srst,            // the slave side's reset
  // ---- the master (otpu_axil_cdc m_*)
  input  logic [11:0]  s_awaddr,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [31:0]  s_wdata,
  input  logic [3:0]   s_wstrb,
  input  logic         s_wvalid,
  output logic         s_wready,
  output logic [1:0]   s_bresp,
  output logic         s_bvalid,
  input  logic         s_bready,
  input  logic [11:0]  s_araddr,
  input  logic         s_arvalid,
  output logic         s_arready,
  output logic [31:0]  s_rdata,
  output logic [1:0]   s_rresp,
  output logic         s_rvalid,
  input  logic         s_rready,
  // ---- the slave (otpu_board s_ctl_*)
  output logic [11:0]  m_awaddr,
  output logic         m_awvalid,
  input  logic         m_awready,
  output logic [31:0]  m_wdata,
  output logic [3:0]   m_wstrb,
  output logic         m_wvalid,
  input  logic         m_wready,
  input  logic [1:0]   m_bresp,
  input  logic         m_bvalid,
  output logic         m_bready,
  output logic [11:0]  m_araddr,
  output logic         m_arvalid,
  input  logic         m_arready,
  input  logic [31:0]  m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rvalid,
  output logic         m_rready
);
  if (NXR == 0) begin : g_wire
    assign m_awaddr = s_awaddr; assign m_awvalid = s_awvalid; assign s_awready = m_awready;
    assign m_wdata = s_wdata; assign m_wstrb = s_wstrb; assign m_wvalid = s_wvalid;
    assign s_wready = m_wready; assign s_bresp = m_bresp; assign s_bvalid = m_bvalid;
    assign m_bready = s_bready; assign m_araddr = s_araddr; assign m_arvalid = s_arvalid;
    assign s_arready = m_arready; assign s_rdata = m_rdata; assign s_rresp = m_rresp;
    assign s_rvalid = m_rvalid; assign m_rready = s_rready;
  end else begin : g_xr
    localparam int DW = 1 + 12 + 1 + 32 + 4 + 1 + 12;   // down: AW, W, AR pulses and words
    localparam int UW = 3 + 1 + 2 + 1 + 2 + 32;         // up: the takes, B, R
    logic [NXR:0][DW-1:0] dn;             // dn[0] the master's, dn[NXR] the slave's
    logic [NXR:0][UW-1:0] up;             // up[NXR] the slave's, up[0] the master's
    otpu_axil_xr_m u_m (
      .clk, .rst, .s_awaddr, .s_awvalid, .s_awready, .s_wdata, .s_wstrb, .s_wvalid, .s_wready,
      .s_bresp, .s_bvalid, .s_bready, .s_araddr, .s_arvalid, .s_arready, .s_rdata, .s_rresp,
      .s_rvalid, .s_rready, .dn(dn[0]), .up(up[0]));
    otpu_axil_xr_s u_s (
      .clk, .rst(srst), .m_awaddr, .m_awvalid, .m_awready, .m_wdata, .m_wstrb, .m_wvalid,
      .m_wready, .m_bresp, .m_bvalid, .m_bready, .m_araddr, .m_arvalid, .m_arready, .m_rdata,
      .m_rresp, .m_rvalid, .m_rready, .dn(dn[NXR]), .up(up[NXR]));
    for (genvar k = 0; k < NXR; k++) begin : g_x   // k: the stage's SLR (0 the master's)
      otpu_xr #(.W(DW)) u_dn (.clk, .rst(1'b0), .d(dn[k]), .q(dn[k + 1]));
      otpu_xr #(.W(UW)) u_up (.clk, .rst(1'b0), .d(up[k + 1]), .q(up[k]));
    end
  end
endmodule

// The master's side (otpu_axil_xr): requests down as pulses, the takes back as the readies, the
// responses up as the valids
module otpu_axil_xr_m (
  input  logic         clk,
  input  logic         rst,
  input  logic [11:0]  s_awaddr,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [31:0]  s_wdata,
  input  logic [3:0]   s_wstrb,
  input  logic         s_wvalid,
  output logic         s_wready,
  output logic [1:0]   s_bresp,
  output logic         s_bvalid,
  input  logic         s_bready,
  input  logic [11:0]  s_araddr,
  input  logic         s_arvalid,
  output logic         s_arready,
  output logic [31:0]  s_rdata,
  output logic [1:0]   s_rresp,
  output logic         s_rvalid,
  input  logic         s_rready,
  output logic [62:0]  dn,
  input  logic [40:0]  up
);
  logic sent_aw, sent_w, sent_ar;
  wire p_aw = s_awvalid && !sent_aw;
  wire p_w  = s_wvalid && !sent_w;
  wire p_ar = s_arvalid && !sent_ar;
  assign dn = {p_aw, s_awaddr, p_w, s_wdata, s_wstrb, p_ar, s_araddr};
  assign {s_awready, s_wready, s_arready, s_bvalid, s_bresp, s_rvalid, s_rresp, s_rdata} = up;
  always_ff @(posedge clk) begin
    if (rst) begin
      sent_aw <= 1'b0; sent_w <= 1'b0; sent_ar <= 1'b0;
    end else begin
      sent_aw <= (sent_aw || p_aw) && !s_awready;
      sent_w  <= (sent_w || p_w) && !s_wready;
      sent_ar <= (sent_ar || p_ar) && !s_arready;
    end
  end
`ifndef SYNTHESIS
  // the master is ready for its response (otpu_axil_cdc: busy from the request until it)
  always_ff @(posedge clk) if (!rst) begin
    if (s_bvalid && !s_bready) $fatal(1, "otpu_axil_xr: B shown to a master not ready for it");
    if (s_rvalid && !s_rready) $fatal(1, "otpu_axil_xr: R shown to a master not ready for it");
  end
`endif
endmodule

// The slave's side (otpu_axil_xr): the requests held until taken, the takes and the responses
// up as pulses
module otpu_axil_xr_s (
  input  logic         clk,
  input  logic         rst,
  output logic [11:0]  m_awaddr,
  output logic         m_awvalid,
  input  logic         m_awready,
  output logic [31:0]  m_wdata,
  output logic [3:0]   m_wstrb,
  output logic         m_wvalid,
  input  logic         m_wready,
  input  logic [1:0]   m_bresp,
  input  logic         m_bvalid,
  output logic         m_bready,
  output logic [11:0]  m_araddr,
  output logic         m_arvalid,
  input  logic         m_arready,
  input  logic [31:0]  m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rvalid,
  output logic         m_rready,
  input  logic [62:0]  dn,
  output logic [40:0]  up
);
  logic        p_aw, p_w, p_ar;
  logic [11:0] d_awaddr, d_araddr;
  logic [31:0] d_wdata;
  logic [3:0]  d_wstrb;
  assign {p_aw, d_awaddr, p_w, d_wdata, d_wstrb, p_ar, d_araddr} = dn;
  wire t_aw = m_awvalid && m_awready;
  wire t_w  = m_wvalid && m_wready;
  wire t_ar = m_arvalid && m_arready;
  assign m_bready = 1'b1;
  assign m_rready = 1'b1;
  logic        t_aw_q, t_w_q, t_ar_q, b_q, r_q;
  logic [1:0]  bresp_q, rresp_q;
  logic [31:0] rdata_q;
  assign up = {t_aw_q, t_w_q, t_ar_q, b_q, bresp_q, r_q, rresp_q, rdata_q};
  always_ff @(posedge clk) begin
    if (p_aw) m_awaddr <= d_awaddr;
    if (p_w) begin m_wdata <= d_wdata; m_wstrb <= d_wstrb; end
    if (p_ar) m_araddr <= d_araddr;
    bresp_q <= m_bresp; rresp_q <= m_rresp; rdata_q <= m_rdata;
    if (rst) begin
      m_awvalid <= 1'b0; m_wvalid <= 1'b0; m_arvalid <= 1'b0;
      t_aw_q <= 1'b0; t_w_q <= 1'b0; t_ar_q <= 1'b0; b_q <= 1'b0; r_q <= 1'b0;
    end else begin
      m_awvalid <= p_aw || (m_awvalid && !t_aw);
      m_wvalid  <= p_w || (m_wvalid && !t_w);
      m_arvalid <= p_ar || (m_arvalid && !t_ar);
      t_aw_q <= t_aw; t_w_q <= t_w; t_ar_q <= t_ar;
      b_q <= m_bvalid; r_q <= m_rvalid;
    end
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk) if (!rst) begin
    if ((p_aw && m_awvalid && !t_aw) || (p_w && m_wvalid && !t_w) || (p_ar && m_arvalid && !t_ar))
      $fatal(1, "otpu_axil_xr: a request over one still held");
  end
`endif
endmodule
