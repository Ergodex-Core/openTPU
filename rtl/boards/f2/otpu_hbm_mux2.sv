// A registered AXI3 2 x 1 at hclk (docs/superpowers/specs/2026-10-20-opentpu-design-v5.md
// section 3): the host's chain (cl_axi_sc_1x1's AXI3 master, s1) shares HBM port NPORT - 1
// with that channel's engine (s0) when every port is a channel's (S x NCH = 32). Every slave
// channel enters a two-entry skid buffer (otpu_skidbuf: the readies the masters see are
// flip-flops), every master output is a register. AW and AR: the channel first, the host once
// it has been denied STARVE cycles; W follows AW order through a 4-entry order FIFO (no write
// interleaving on the port); IDs out are {source, id[4:0]} (both masters are single-threaded
// at ID 0: the engine's, SmartConnect's), B and R come back by that bit into a skid buffer per
// source with the low five ID bits. Two to three LUT levels on any path.
module otpu_hbm_mux2 #(
  parameter int STARVE = 8
) (
  input  logic          hclk,
  input  logic          hrst,
  // ---- s0: the channel's engine
  input  logic [5:0]    s0_awid,
  input  logic [33:0]   s0_awaddr,
  input  logic [3:0]    s0_awlen,
  input  logic [2:0]    s0_awsize,
  input  logic [1:0]    s0_awburst,
  input  logic          s0_awvalid,
  output logic          s0_awready,
  input  logic [5:0]    s0_wid,
  input  logic [255:0]  s0_wdata,
  input  logic [31:0]   s0_wstrb,
  input  logic          s0_wlast,
  input  logic          s0_wvalid,
  output logic          s0_wready,
  output logic [5:0]    s0_bid,
  output logic [1:0]    s0_bresp,
  output logic          s0_bvalid,
  input  logic          s0_bready,
  input  logic [5:0]    s0_arid,
  input  logic [33:0]   s0_araddr,
  input  logic [3:0]    s0_arlen,
  input  logic [2:0]    s0_arsize,
  input  logic [1:0]    s0_arburst,
  input  logic          s0_arvalid,
  output logic          s0_arready,
  output logic [5:0]    s0_rid,
  output logic [255:0]  s0_rdata,
  output logic [1:0]    s0_rresp,
  output logic          s0_rlast,
  output logic          s0_rvalid,
  input  logic          s0_rready,
  // ---- s1: the host's chain
  input  logic [5:0]    s1_awid,
  input  logic [33:0]   s1_awaddr,
  input  logic [3:0]    s1_awlen,
  input  logic [2:0]    s1_awsize,
  input  logic [1:0]    s1_awburst,
  input  logic          s1_awvalid,
  output logic          s1_awready,
  input  logic [5:0]    s1_wid,
  input  logic [255:0]  s1_wdata,
  input  logic [31:0]   s1_wstrb,
  input  logic          s1_wlast,
  input  logic          s1_wvalid,
  output logic          s1_wready,
  output logic [5:0]    s1_bid,
  output logic [1:0]    s1_bresp,
  output logic          s1_bvalid,
  input  logic          s1_bready,
  input  logic [5:0]    s1_arid,
  input  logic [33:0]   s1_araddr,
  input  logic [3:0]    s1_arlen,
  input  logic [2:0]    s1_arsize,
  input  logic [1:0]    s1_arburst,
  input  logic          s1_arvalid,
  output logic          s1_arready,
  output logic [5:0]    s1_rid,
  output logic [255:0]  s1_rdata,
  output logic [1:0]    s1_rresp,
  output logic          s1_rlast,
  output logic          s1_rvalid,
  input  logic          s1_rready,
  // ---- the port
  output logic [5:0]    m_awid,
  output logic [33:0]   m_awaddr,
  output logic [3:0]    m_awlen,
  output logic [2:0]    m_awsize,
  output logic [1:0]    m_awburst,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [5:0]    m_wid,
  output logic [255:0]  m_wdata,
  output logic [31:0]   m_wstrb,
  output logic          m_wlast,
  output logic          m_wvalid,
  input  logic          m_wready,
  input  logic [5:0]    m_bid,
  input  logic [1:0]    m_bresp,
  input  logic          m_bvalid,
  output logic          m_bready,
  output logic [5:0]    m_arid,
  output logic [33:0]   m_araddr,
  output logic [3:0]    m_arlen,
  output logic [2:0]    m_arsize,
  output logic [1:0]    m_arburst,
  output logic          m_arvalid,
  input  logic          m_arready,
  input  logic [5:0]    m_rid,
  input  logic [255:0]  m_rdata,
  input  logic [1:0]    m_rresp,
  input  logic          m_rlast,
  input  logic          m_rvalid,
  output logic          m_rready
);
  localparam int AXW = 5 + 34 + 4 + 3 + 2;        // {id[4:0], addr, len, size, burst}
  localparam int WW  = 5 + 256 + 32 + 1;          // {id[4:0], data, strb, last}
  localparam int BW_ = 5 + 2;
  localparam int RW_ = 5 + 256 + 2 + 1;
  localparam int SW = $clog2(STARVE + 1);
  // ---- the slaves' channels through skid buffers (their readies are flip-flops)
  logic [1:0]           aw_v, aw_r, ar_v, ar_r, w_v, w_r;
  logic [1:0][AXW-1:0]  aw_d, ar_d;
  logic [1:0][WW-1:0]   w_d;
  otpu_skidbuf #(.W(AXW)) u_aw0 (.clk(hclk), .rst(hrst), .i_v(s0_awvalid), .i_r(s0_awready),
    .i_d({s0_awid[4:0], s0_awaddr, s0_awlen, s0_awsize, s0_awburst}), .o_v(aw_v[0]), .o_r(aw_r[0]), .o_d(aw_d[0]), .any());
  otpu_skidbuf #(.W(AXW)) u_aw1 (.clk(hclk), .rst(hrst), .i_v(s1_awvalid), .i_r(s1_awready),
    .i_d({s1_awid[4:0], s1_awaddr, s1_awlen, s1_awsize, s1_awburst}), .o_v(aw_v[1]), .o_r(aw_r[1]), .o_d(aw_d[1]), .any());
  otpu_skidbuf #(.W(AXW)) u_ar0 (.clk(hclk), .rst(hrst), .i_v(s0_arvalid), .i_r(s0_arready),
    .i_d({s0_arid[4:0], s0_araddr, s0_arlen, s0_arsize, s0_arburst}), .o_v(ar_v[0]), .o_r(ar_r[0]), .o_d(ar_d[0]), .any());
  otpu_skidbuf #(.W(AXW)) u_ar1 (.clk(hclk), .rst(hrst), .i_v(s1_arvalid), .i_r(s1_arready),
    .i_d({s1_arid[4:0], s1_araddr, s1_arlen, s1_arsize, s1_arburst}), .o_v(ar_v[1]), .o_r(ar_r[1]), .o_d(ar_d[1]), .any());
  otpu_skidbuf #(.W(WW)) u_w0 (.clk(hclk), .rst(hrst), .i_v(s0_wvalid), .i_r(s0_wready),
    .i_d({s0_wid[4:0], s0_wdata, s0_wstrb, s0_wlast}), .o_v(w_v[0]), .o_r(w_r[0]), .o_d(w_d[0]), .any());
  otpu_skidbuf #(.W(WW)) u_w1 (.clk(hclk), .rst(hrst), .i_v(s1_wvalid), .i_r(s1_wready),
    .i_d({s1_wid[4:0], s1_wdata, s1_wstrb, s1_wlast}), .o_v(w_v[1]), .o_r(w_r[1]), .o_d(w_d[1]), .any());

  // ---- AW and AR: one output register each, the channel first unless the host starved
  logic [SW-1:0] aw_st, ar_st;             // cycles the host's request was denied
  logic          aw_pri1, ar_pri1;         // the host has priority (registered)
  wire aw_free = !m_awvalid || m_awready;
  wire ar_free = !m_arvalid || m_arready;
  // the W order FIFO: the sources of the AWs accepted, their W not yet complete
  logic [3:0] ord_src;
  logic [1:0] ord_h, ord_t;
  logic [2:0] ord_n;
  wire ord_full = ord_n[2];
  wire aw_sel1 = aw_v[1] && (aw_pri1 || !aw_v[0]);
  wire aw_take = aw_free && !ord_full && (aw_v[0] || aw_v[1]);
  assign aw_r[0] = aw_take && !aw_sel1;
  assign aw_r[1] = aw_take && aw_sel1;
  wire ar_sel1 = ar_v[1] && (ar_pri1 || !ar_v[0]);
  wire ar_take = ar_free && (ar_v[0] || ar_v[1]);
  assign ar_r[0] = ar_take && !ar_sel1;
  assign ar_r[1] = ar_take && ar_sel1;
  // ---- W: the head source's beats until its last
  wire       w_src = ord_src[ord_h];
  wire       w_free = !m_wvalid || m_wready;
  wire       w_take = w_free && ord_n != 3'd0 && w_v[w_src];
  assign w_r[0] = w_take && !w_src;
  assign w_r[1] = w_take && w_src;
  wire       w_last_tk = w_take && w_d[w_src][0];
  always_ff @(posedge hclk) begin
    if (hrst) begin
      m_awvalid <= 1'b0; m_arvalid <= 1'b0; m_wvalid <= 1'b0;
      aw_st <= '0; ar_st <= '0; aw_pri1 <= 1'b0; ar_pri1 <= 1'b0;
      ord_h <= '0; ord_t <= '0; ord_n <= '0;
    end else begin
      if (aw_free) begin
        m_awvalid <= aw_take;
        if (aw_take) begin
          {m_awid[4:0], m_awaddr, m_awlen, m_awsize, m_awburst} <= aw_d[aw_sel1];
          m_awid[5] <= aw_sel1;
          ord_src[ord_t] <= aw_sel1; ord_t <= ord_t + 1'b1;
        end
      end
      if (ar_free) begin
        m_arvalid <= ar_take;
        if (ar_take) begin
          {m_arid[4:0], m_araddr, m_arlen, m_arsize, m_arburst} <= ar_d[ar_sel1];
          m_arid[5] <= ar_sel1;
        end
      end
      // the host's wait: counts while it is valid and denied; priority once it reaches STARVE
      aw_st <= (aw_v[1] && !aw_r[1]) ? ((aw_st == SW'(STARVE)) ? aw_st : aw_st + 1'b1) : '0;
      aw_pri1 <= (aw_v[1] && !aw_r[1]) && (aw_st == SW'(STARVE - 1) || aw_st == SW'(STARVE));
      ar_st <= (ar_v[1] && !ar_r[1]) ? ((ar_st == SW'(STARVE)) ? ar_st : ar_st + 1'b1) : '0;
      ar_pri1 <= (ar_v[1] && !ar_r[1]) && (ar_st == SW'(STARVE - 1) || ar_st == SW'(STARVE));
      if (w_free) begin
        m_wvalid <= w_take;
        if (w_take) begin
          {m_wid[4:0], m_wdata, m_wstrb, m_wlast} <= w_d[w_src];
          m_wid[5] <= w_src;
        end
      end
      if (w_last_tk) ord_h <= ord_h + 1'b1;
      ord_n <= ord_n + 3'(aw_take) - 3'(w_last_tk);
    end
  end
  // ---- B and R back by the ID's bit 5, into a skid buffer per source
  logic [1:0] b_ir, r_ir;
  wire b_src = m_bid[5];
  wire r_src = m_rid[5];
  assign m_bready = b_ir[b_src];
  assign m_rready = r_ir[r_src];
  logic [1:0][BW_-1:0] b_od;
  logic [1:0][RW_-1:0] r_od;
  otpu_skidbuf #(.W(BW_)) u_b0 (.clk(hclk), .rst(hrst), .i_v(m_bvalid && !b_src), .i_r(b_ir[0]),
    .i_d({m_bid[4:0], m_bresp}), .o_v(s0_bvalid), .o_r(s0_bready), .o_d(b_od[0]), .any());
  otpu_skidbuf #(.W(BW_)) u_b1 (.clk(hclk), .rst(hrst), .i_v(m_bvalid && b_src), .i_r(b_ir[1]),
    .i_d({m_bid[4:0], m_bresp}), .o_v(s1_bvalid), .o_r(s1_bready), .o_d(b_od[1]), .any());
  otpu_skidbuf #(.W(RW_)) u_r0 (.clk(hclk), .rst(hrst), .i_v(m_rvalid && !r_src), .i_r(r_ir[0]),
    .i_d({m_rid[4:0], m_rdata, m_rresp, m_rlast}), .o_v(s0_rvalid), .o_r(s0_rready), .o_d(r_od[0]), .any());
  otpu_skidbuf #(.W(RW_)) u_r1 (.clk(hclk), .rst(hrst), .i_v(m_rvalid && r_src), .i_r(r_ir[1]),
    .i_d({m_rid[4:0], m_rdata, m_rresp, m_rlast}), .o_v(s1_rvalid), .o_r(s1_rready), .o_d(r_od[1]), .any());
  assign {s0_bid[4:0], s0_bresp} = b_od[0];
  assign {s1_bid[4:0], s1_bresp} = b_od[1];
  assign s0_bid[5] = 1'b0;
  assign s1_bid[5] = 1'b0;
  assign {s0_rid[4:0], s0_rdata, s0_rresp, s0_rlast} = r_od[0];
  assign {s1_rid[4:0], s1_rdata, s1_rresp, s1_rlast} = r_od[1];
  assign s0_rid[5] = 1'b0;
  assign s1_rid[5] = 1'b0;
`ifndef SYNTHESIS
  always @(posedge hclk) if (!hrst) begin
    if (aw_take && ord_n == 3'd4) $fatal(1, "otpu_hbm_mux2: the W order FIFO overflowed");
    if (w_last_tk && ord_n == 3'd0) $fatal(1, "otpu_hbm_mux2: a W beat without an AW");
  end
`endif
endmodule
