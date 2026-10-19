// openTPU AWS F2 small-shell top, the v5 full-HBM build (targets/f2-hbm.toml [variants.v5] /
// [variants.v5s3], platform mem_path "engine"; docs/full_hbm.md, the 2026-10-20 spec): module
// cl_opentpu, as the HDK flow names the CL; boards/f2/build.py puts this file in place of
// cl_opentpu.sv (and leaves cl_opentpu_hbm.sv, the SmartConnect path's top, out). The HDK
// supplies cl_ports.vh, the unused-DDR template and the cl_hbm IP; build.py copies the HDK's
// cl_mem_hbm_wrapper (hdk/cl/examples/cl_mem_perf) next to it.
//
// Clocks and resets as cl_opentpu_hbm.sv's: clk_main_a0 (250 MHz, aclk: OCL, PCIS, the host's
// chain) drives u_mmcm for the core clock (clk) and 2x it (clk_f, the pumped TMEM); clk_hbm_ref
// drives u_hmmcm for hclk (450 MHz: the HBM's AXI ports, the engines' port sides, the host's
// mux). The core and clk_main_a0 meet only in otpu_axil_cdc (cl_timing_user_hbm.tcl); the
// engines cross core <-> hclk in otpu_hbm_afifo's gray pointers (its own bounds, the same file);
// the host's chain crosses 250 <-> 450 inside AWS's cl_axi_sc_1x1 as shipped.
//
// The memory path (docs/full_hbm.md): u_sys (otpu_f2_hbmeng_sys) gives every HBM port p < NPORT
// = S x NCH an AXI3 256-bit master at hclk (p_*: channel p's 450 MHz engine, otpu_hbm_ch, behind
// the SLR chains and the slice's write bus), which goes through otpu_hbm_geometry (32-byte
// aligned addresses) and the HDK's cl_axi3_256b_reg_slice (g_ch[p].u_rs256) onto HBM port p: no
// SmartConnect on the channel path (AWS's cl_mem_perf's shape). The host's PCIS through
// u_sys's gate (h_*, aclk) -> u_hrs512 -> cl_axi_sc_1x1 (u_hsc, the HDK's block design as
// shipped: 250 -> 450 MHz, AXI4 512 -> AXI3 256) -> u_hrs256 -> HBM port NPORT when a port is
// free (S x NCH <= 31: the global address switch reaches every pseudo-channel), or, at
// S x NCH = 32 (f2-hbm:v5), back into u_sys as hs_* (an AXI3 master at hclk) onto port NPORT - 1
// through its channel's otpu_hbm_mux2 (HOST_MUX). The HBM is ready before the core may run
// (o_hbm_ready: otpu_board's CALIB bits, vLED bit 1).
module cl_opentpu (
  `include "cl_ports.vh"
);
  `include "cl_id_defines.vh"
  `include "opentpu_geometry.vh"        // build.py: localparam S, NCH, the core MMCM (the target's)
  localparam int NPORT = S * NCH;
  localparam bit HOST_MUX = NPORT == 32;  // every port a channel's: the host through port 31's mux
  localparam int NHBM = NPORT + (HOST_MUX ? 0 : 1);   // HBM AXI ports: the channels', and the host's own
  initial if (NPORT > 32) $fatal(1, "cl_opentpu: at most 32 HBM ports (S x NCH)");

  logic clk, clk_f, clk_fb, clk_fb_buf, clk_raw, clk_f_raw, mmcm_locked;
  logic hclk, hclk_raw, hclk_fb, hclk_fb_buf, hmmcm_locked;
  logic [2:0] board_led, board_led_core;
  (* ASYNC_REG = "TRUE" *) logic [2:0] main_reset_n, core_reset_n, bridge_reset_n, hbm_reset_n;
  (* ASYNC_REG = "TRUE" *) logic [2:0] led_s1, led_s2;
  wire clock_reset = !rst_main_n;
  wire bridge_reset = clock_reset || !mmcm_locked || !hmmcm_locked;
  wire arst = !bridge_reset_n[2];
  wire rst = !core_reset_n[2];
  wire hrst = !hbm_reset_n[2];
  logic rst_main_n_sync;                // (unused_ddr_template.inc's)
  assign rst_main_n_sync = main_reset_n[2];
  logic hbm_ready;

  // 250 MHz -> the VCO -> the core clock and 2x it, phase aligned (the pumped TMEM relies on it)
  initial if (MMCM_DIV0 != 2.0 * MMCM_DIV1 || 250000.0 * MMCM_MULT / MMCM_DIVCLK / MMCM_DIV0 != CORE_KHZ_CL)
    $fatal(1, "cl_opentpu: the core MMCM does not make CORE_KHZ_CL and 2x it");
  MMCME4_ADV #(.CLKIN1_PERIOD(4.0), .DIVCLK_DIVIDE(MMCM_DIVCLK), .CLKFBOUT_MULT_F(MMCM_MULT),
               .CLKOUT0_DIVIDE_F(MMCM_DIV0), .CLKOUT1_DIVIDE(MMCM_DIV1), .STARTUP_WAIT("FALSE")) u_mmcm (
    .CLKIN1(clk_main_a0), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clk_fb_buf),
    .CLKFBOUT(clk_fb), .CLKOUT0(clk_raw), .CLKOUT1(clk_f_raw), .LOCKED(mmcm_locked),
    .RST(clock_reset), .PWRDWN(1'b0),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .CDDCREQ(1'b0));
  BUFG u_fb (.I(clk_fb), .O(clk_fb_buf));
  BUFG u_clk (.I(clk_raw), .O(clk));
  BUFG u_clk_f (.I(clk_f_raw), .O(clk_f));
  // 100 MHz HBM reference -> 900 MHz VCO -> 450 MHz, the HBM's AXI clock
  MMCME4_ADV #(.CLKIN1_PERIOD(10.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(9.0),
               .CLKOUT0_DIVIDE_F(2.0), .STARTUP_WAIT("FALSE")) u_hmmcm (
    .CLKIN1(clk_hbm_ref), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(hclk_fb_buf),
    .CLKFBOUT(hclk_fb), .CLKOUT0(hclk_raw), .LOCKED(hmmcm_locked),
    .RST(clock_reset), .PWRDWN(1'b0),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .CDDCREQ(1'b0));
  BUFG u_hfb (.I(hclk_fb), .O(hclk_fb_buf));
  BUFG u_hclk (.I(hclk_raw), .O(hclk));

  always_ff @(posedge clk_main_a0 or posedge clock_reset)
    if (clock_reset) main_reset_n <= '0;
    else main_reset_n <= {main_reset_n[1:0], 1'b1};
  always_ff @(posedge clk_main_a0 or posedge bridge_reset)
    if (bridge_reset) bridge_reset_n <= '0;
    else bridge_reset_n <= {bridge_reset_n[1:0], 1'b1};
  always_ff @(posedge clk or posedge bridge_reset)
    if (bridge_reset) core_reset_n <= '0;
    else core_reset_n <= {core_reset_n[1:0], 1'b1};
  always_ff @(posedge hclk or posedge bridge_reset)
    if (bridge_reset) hbm_reset_n <= '0;
    else hbm_reset_n <= {hbm_reset_n[1:0], 1'b1};
  always_ff @(posedge clk)
    if (rst) board_led_core <= '0;
    else board_led_core <= board_led;
  always_ff @(posedge clk_main_a0) begin
    led_s1 <= board_led_core;
    led_s2 <= led_s1;
  end

  // ---- the accelerator and its channels' masters
  logic [15:0]  h_awid, h_arid, h_bid, h_rid;
  logic [63:0]  h_awaddr, h_araddr;
  logic [7:0]   h_awlen, h_arlen;
  logic [2:0]   h_awsize, h_arsize;
  logic [1:0]   h_awburst, h_arburst, h_bresp, h_rresp;
  logic         h_awvalid, h_awready, h_wlast, h_wvalid, h_wready, h_bvalid, h_bready;
  logic         h_arvalid, h_arready, h_rlast, h_rvalid, h_rready;
  logic [511:0] h_wdata, h_rdata;
  logic [63:0]  h_wstrb;
  // the ports' AXI3 masters at hclk (the engines'), before the geometry and the 450 MHz slice
  logic [NPORT-1:0][5:0]   p_awid, p_wid, p_bid, p_arid, p_rid;
  logic [NPORT-1:0][33:0]  p_awaddr, p_araddr;
  logic [NPORT-1:0][3:0]   p_awlen, p_arlen;
  logic [NPORT-1:0][2:0]   p_awsize, p_arsize;
  logic [NPORT-1:0][1:0]   p_awburst, p_arburst, p_bresp, p_rresp;
  logic [NPORT-1:0]        p_awvalid, p_awready, p_wlast, p_wvalid, p_wready, p_bvalid, p_bready;
  logic [NPORT-1:0]        p_arvalid, p_arready, p_rlast, p_rvalid, p_rready;
  logic [NPORT-1:0][255:0] p_wdata, p_rdata;
  logic [NPORT-1:0][31:0]  p_wstrb;
  // the host's AXI3 at hclk into u_sys's mux (HOST_MUX; else unused: tied, never valid)
  logic [5:0]   hs_awid, hs_wid, hs_bid, hs_arid, hs_rid;
  logic [33:0]  hs_awaddr, hs_araddr;
  logic [3:0]   hs_awlen, hs_arlen;
  logic [2:0]   hs_awsize, hs_arsize;
  logic [1:0]   hs_awburst, hs_arburst, hs_bresp, hs_rresp;
  logic         hs_awvalid, hs_awready, hs_wlast, hs_wvalid, hs_wready, hs_bvalid, hs_bready;
  logic         hs_arvalid, hs_arready, hs_rlast, hs_rvalid, hs_rready;
  logic [255:0] hs_wdata, hs_rdata;
  logic [31:0]  hs_wstrb;

  otpu_f2_hbmeng_sys #(
    `include "opentpu_params.vh"
  ) u_sys (
    .clk, .clk_f, .rst, .aclk(clk_main_a0), .arst, .hclk, .hrst, .hbm_ready, .led(board_led),
    .ocl_awaddr(ocl_cl_awaddr), .ocl_awvalid(ocl_cl_awvalid), .ocl_awready(cl_ocl_awready),
    .ocl_wdata(ocl_cl_wdata), .ocl_wstrb(ocl_cl_wstrb), .ocl_wvalid(ocl_cl_wvalid),
    .ocl_wready(cl_ocl_wready), .ocl_bresp(cl_ocl_bresp), .ocl_bvalid(cl_ocl_bvalid),
    .ocl_bready(ocl_cl_bready), .ocl_araddr(ocl_cl_araddr), .ocl_arvalid(ocl_cl_arvalid),
    .ocl_arready(cl_ocl_arready), .ocl_rdata(cl_ocl_rdata), .ocl_rresp(cl_ocl_rresp),
    .ocl_rvalid(cl_ocl_rvalid), .ocl_rready(ocl_cl_rready),
    .pcis_awid(sh_cl_dma_pcis_awid), .pcis_awaddr(sh_cl_dma_pcis_awaddr),
    .pcis_awlen(sh_cl_dma_pcis_awlen), .pcis_awsize(sh_cl_dma_pcis_awsize),
    .pcis_awburst(sh_cl_dma_pcis_awburst), .pcis_awvalid(sh_cl_dma_pcis_awvalid),
    .pcis_awready(cl_sh_dma_pcis_awready), .pcis_wdata(sh_cl_dma_pcis_wdata),
    .pcis_wstrb(sh_cl_dma_pcis_wstrb), .pcis_wlast(sh_cl_dma_pcis_wlast),
    .pcis_wvalid(sh_cl_dma_pcis_wvalid), .pcis_wready(cl_sh_dma_pcis_wready),
    .pcis_bid(cl_sh_dma_pcis_bid), .pcis_bresp(cl_sh_dma_pcis_bresp),
    .pcis_bvalid(cl_sh_dma_pcis_bvalid), .pcis_bready(sh_cl_dma_pcis_bready),
    .pcis_arid(sh_cl_dma_pcis_arid), .pcis_araddr(sh_cl_dma_pcis_araddr),
    .pcis_arlen(sh_cl_dma_pcis_arlen), .pcis_arsize(sh_cl_dma_pcis_arsize),
    .pcis_arburst(sh_cl_dma_pcis_arburst), .pcis_arvalid(sh_cl_dma_pcis_arvalid),
    .pcis_arready(cl_sh_dma_pcis_arready), .pcis_rid(cl_sh_dma_pcis_rid),
    .pcis_rdata(cl_sh_dma_pcis_rdata), .pcis_rresp(cl_sh_dma_pcis_rresp),
    .pcis_rlast(cl_sh_dma_pcis_rlast), .pcis_rvalid(cl_sh_dma_pcis_rvalid),
    .pcis_rready(sh_cl_dma_pcis_rready),
    .h_awid, .h_awaddr, .h_awlen, .h_awsize, .h_awburst, .h_awvalid, .h_awready, .h_wdata,
    .h_wstrb, .h_wlast, .h_wvalid, .h_wready, .h_bid, .h_bresp, .h_bvalid, .h_bready, .h_arid,
    .h_araddr, .h_arlen, .h_arsize, .h_arburst, .h_arvalid, .h_arready, .h_rid, .h_rdata,
    .h_rresp, .h_rlast, .h_rvalid, .h_rready,
    .hs_awid, .hs_awaddr, .hs_awlen, .hs_awsize, .hs_awburst, .hs_awvalid, .hs_awready, .hs_wid,
    .hs_wdata, .hs_wstrb, .hs_wlast, .hs_wvalid, .hs_wready, .hs_bid, .hs_bresp, .hs_bvalid,
    .hs_bready, .hs_arid, .hs_araddr, .hs_arlen, .hs_arsize, .hs_arburst, .hs_arvalid,
    .hs_arready, .hs_rid, .hs_rdata, .hs_rresp, .hs_rlast, .hs_rvalid, .hs_rready,
    .p_awid, .p_awaddr, .p_awlen, .p_awsize, .p_awburst, .p_awvalid, .p_awready, .p_wid, .p_wdata,
    .p_wstrb, .p_wlast, .p_wvalid, .p_wready, .p_bid, .p_bresp, .p_bvalid, .p_bready, .p_arid,
    .p_araddr, .p_arlen, .p_arsize, .p_arburst, .p_arvalid, .p_arready, .p_rid, .p_rdata, .p_rresp,
    .p_rlast, .p_rvalid, .p_rready);

  // ---- the memory path behind u_sys (docs/full_hbm.md): per port the engine's AXI3 master at
  // hclk (u_sys p_*[p]) -> otpu_hbm_geometry (32-byte aligned addresses) -> the HDK's
  // cl_axi3_256b_reg_slice at 450 MHz -> HBM port p, every AXI input of the wrapper from that
  // slice's registers (cl_mem_perf's q1_* stage). No SmartConnect on the channel path.
  // Resets: each domain's synchronizer registered again (max_fanout: replicated per slice)
  (* max_fanout = 16 *) logic sc_host_aresetn, sc_hbm_aresetn;
  always_ff @(posedge clk_main_a0) sc_host_aresetn <= bridge_reset_n[2];
  always_ff @(posedge hclk) sc_hbm_aresetn <= hbm_reset_n[2];
  // HBM ports: x_*[p] (p < NPORT channel p; NPORT the host's own port without the mux), 34-bit
  // addresses, 6-bit IDs (AXI3)
  logic [5:0]  x_awid [0:NHBM-1], x_arid [0:NHBM-1], x_bid [0:NHBM-1], x_rid [0:NHBM-1];
  logic [33:0]  x_awaddr [0:NHBM-1], x_araddr [0:NHBM-1];
  logic [3:0]   x_awlen [0:NHBM-1], x_arlen [0:NHBM-1];
  logic [2:0]   x_awsize [0:NHBM-1], x_arsize [0:NHBM-1];
  logic [1:0]   x_awburst [0:NHBM-1], x_arburst [0:NHBM-1], x_bresp [0:NHBM-1], x_rresp [0:NHBM-1];
  logic         x_awvalid [0:NHBM-1], x_awready [0:NHBM-1], x_wlast [0:NHBM-1], x_wvalid [0:NHBM-1], x_wready [0:NHBM-1];
  logic         x_bvalid [0:NHBM-1], x_bready [0:NHBM-1], x_arvalid [0:NHBM-1], x_arready [0:NHBM-1];
  logic         x_rlast [0:NHBM-1], x_rvalid [0:NHBM-1], x_rready [0:NHBM-1];
  logic [255:0] x_wdata [0:NHBM-1], x_rdata [0:NHBM-1];
  logic [31:0]  x_wstrb [0:NHBM-1];
  logic [33:0] pg_awaddr [0:NPORT-1], pg_araddr [0:NPORT-1];
  for (genvar p = 0; p < NPORT; p++) begin : g_ch
    otpu_hbm_geometry u_geo (.s_awaddr(p_awaddr[p]), .s_araddr(p_araddr[p]),
                             .m_awaddr(pg_awaddr[p]), .m_araddr(pg_araddr[p]));
    cl_axi3_256b_reg_slice u_rs256 (
      .aclk(hclk), .aresetn(sc_hbm_aresetn),
      .s_axi_awid(p_awid[p]), .s_axi_awaddr(pg_awaddr[p]), .s_axi_awlen(p_awlen[p]), .s_axi_awsize(p_awsize[p]),
      .s_axi_awburst(p_awburst[p]), .s_axi_awlock(2'b0), .s_axi_awcache(4'd3), .s_axi_awprot(3'd2),
      .s_axi_awqos(4'd0), .s_axi_awvalid(p_awvalid[p]), .s_axi_awready(p_awready[p]), .s_axi_wid(p_wid[p]),
      .s_axi_wdata(p_wdata[p]), .s_axi_wstrb(p_wstrb[p]), .s_axi_wlast(p_wlast[p]), .s_axi_wvalid(p_wvalid[p]),
      .s_axi_wready(p_wready[p]), .s_axi_bid(p_bid[p]), .s_axi_bresp(p_bresp[p]), .s_axi_bvalid(p_bvalid[p]),
      .s_axi_bready(p_bready[p]), .s_axi_arid(p_arid[p]), .s_axi_araddr(pg_araddr[p]), .s_axi_arlen(p_arlen[p]),
      .s_axi_arsize(p_arsize[p]), .s_axi_arburst(p_arburst[p]), .s_axi_arlock(2'b0), .s_axi_arcache(4'd3),
      .s_axi_arprot(3'd2), .s_axi_arqos(4'd0), .s_axi_arvalid(p_arvalid[p]), .s_axi_arready(p_arready[p]),
      .s_axi_rid(p_rid[p]), .s_axi_rdata(p_rdata[p]), .s_axi_rresp(p_rresp[p]), .s_axi_rlast(p_rlast[p]),
      .s_axi_rvalid(p_rvalid[p]), .s_axi_rready(p_rready[p]), .m_axi_awid(x_awid[p]), .m_axi_awaddr(x_awaddr[p]),
      .m_axi_awlen(x_awlen[p]), .m_axi_awsize(x_awsize[p]), .m_axi_awburst(x_awburst[p]), .m_axi_awlock(),
      .m_axi_awcache(), .m_axi_awprot(), .m_axi_awqos(), .m_axi_awvalid(x_awvalid[p]),
      .m_axi_awready(x_awready[p]), .m_axi_wid(), .m_axi_wdata(x_wdata[p]), .m_axi_wstrb(x_wstrb[p]),
      .m_axi_wlast(x_wlast[p]), .m_axi_wvalid(x_wvalid[p]), .m_axi_wready(x_wready[p]), .m_axi_bid(x_bid[p]),
      .m_axi_bresp(x_bresp[p]), .m_axi_bvalid(x_bvalid[p]), .m_axi_bready(x_bready[p]), .m_axi_arid(x_arid[p]),
      .m_axi_araddr(x_araddr[p]), .m_axi_arlen(x_arlen[p]), .m_axi_arsize(x_arsize[p]), .m_axi_arburst(x_arburst[p]),
      .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
      .m_axi_arvalid(x_arvalid[p]), .m_axi_arready(x_arready[p]), .m_axi_rid(x_rid[p]), .m_axi_rdata(x_rdata[p]),
      .m_axi_rresp(x_rresp[p]), .m_axi_rlast(x_rlast[p]), .m_axi_rvalid(x_rvalid[p]), .m_axi_rready(x_rready[p])
    );
  end
  // the host: h_* (otpu_f2_pcis_gate, 250 MHz, beside the shell in its PCIS SLR) -> u_hrs512 (the
  // same SLR: the SLR crossing runs flop to flop from its registers to cl_axi_sc_1x1's entry
  // slice in SLR0) -> cl_axi_sc_1x1 (AWS's, as shipped) -> u_hrs256 at 450 -> HBM port NPORT
  logic [15:0]  hs_awid, hs_arid, hs_bid, hs_rid;
  logic [63:0]  hs_awaddr, hs_araddr;
  logic [7:0]   hs_awlen, hs_arlen;
  logic [2:0]   hs_awsize, hs_arsize;
  logic [1:0]   hs_awburst, hs_arburst, hs_bresp, hs_rresp;
  logic         hs_awvalid, hs_awready, hs_wlast, hs_wvalid, hs_wready;
  logic         hs_bvalid, hs_bready, hs_arvalid, hs_arready;
  logic         hs_rlast, hs_rvalid, hs_rready;
  logic [511:0] hs_wdata, hs_rdata;
  logic [63:0]  hs_wstrb;
  logic [5:0]  hm_awid, hm_arid, hm_bid, hm_rid;
  logic [33:0]  hm_awaddr, hm_araddr;
  logic [3:0]   hm_awlen, hm_arlen;
  logic [2:0]   hm_awsize, hm_arsize;
  logic [1:0]   hm_awburst, hm_arburst, hm_bresp, hm_rresp;
  logic         hm_awvalid, hm_awready, hm_wlast, hm_wvalid, hm_wready;
  logic         hm_bvalid, hm_bready, hm_arvalid, hm_arready;
  logic         hm_rlast, hm_rvalid, hm_rready;
  logic [255:0] hm_wdata, hm_rdata;
  logic [31:0]  hm_wstrb;
    axi_register_slice u_hrs512 (
      .aclk(clk_main_a0), .aresetn(sc_host_aresetn),
      .s_axi_awid(h_awid), .s_axi_awaddr(h_awaddr), .s_axi_awlen(h_awlen), .s_axi_awsize(h_awsize),
      .s_axi_awburst(h_awburst), .s_axi_awlock(1'b0), .s_axi_awcache(4'd3), .s_axi_awprot(3'd2),
      .s_axi_awregion(4'd0), .s_axi_awqos(4'd0), .s_axi_awvalid(h_awvalid), .s_axi_awready(h_awready),
      .s_axi_wdata(h_wdata), .s_axi_wstrb(h_wstrb), .s_axi_wlast(h_wlast), .s_axi_wvalid(h_wvalid),
      .s_axi_wready(h_wready), .s_axi_bid(h_bid), .s_axi_bresp(h_bresp), .s_axi_bvalid(h_bvalid),
      .s_axi_bready(h_bready), .s_axi_arid(h_arid), .s_axi_araddr(h_araddr), .s_axi_arlen(h_arlen),
      .s_axi_arsize(h_arsize), .s_axi_arburst(h_arburst), .s_axi_arlock(1'b0), .s_axi_arcache(4'd3),
      .s_axi_arprot(3'd2), .s_axi_arregion(4'd0), .s_axi_arqos(4'd0), .s_axi_arvalid(h_arvalid),
      .s_axi_arready(h_arready), .s_axi_rid(h_rid), .s_axi_rdata(h_rdata), .s_axi_rresp(h_rresp),
      .s_axi_rlast(h_rlast), .s_axi_rvalid(h_rvalid), .s_axi_rready(h_rready), .m_axi_awid(hs_awid),
      .m_axi_awaddr(hs_awaddr), .m_axi_awlen(hs_awlen), .m_axi_awsize(hs_awsize), .m_axi_awburst(hs_awburst),
      .m_axi_awlock(), .m_axi_awcache(), .m_axi_awprot(), .m_axi_awregion(),
      .m_axi_awqos(), .m_axi_awvalid(hs_awvalid), .m_axi_awready(hs_awready), .m_axi_wdata(hs_wdata),
      .m_axi_wstrb(hs_wstrb), .m_axi_wvalid(hs_wvalid), .m_axi_wlast(hs_wlast), .m_axi_wready(hs_wready),
      .m_axi_bresp(hs_bresp), .m_axi_bvalid(hs_bvalid), .m_axi_bid(hs_bid), .m_axi_bready(hs_bready),
      .m_axi_arid(hs_arid), .m_axi_araddr(hs_araddr), .m_axi_arlen(hs_arlen), .m_axi_arsize(hs_arsize),
      .m_axi_arburst(hs_arburst), .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
      .m_axi_arregion(), .m_axi_arqos(), .m_axi_arvalid(hs_arvalid), .m_axi_arready(hs_arready),
      .m_axi_rid(hs_rid), .m_axi_rdata(hs_rdata), .m_axi_rresp(hs_rresp), .m_axi_rlast(hs_rlast),
      .m_axi_rvalid(hs_rvalid), .m_axi_rready(hs_rready)
    );
  logic [29:0] hm_awaddr_hi, hm_araddr_hi;
  cl_axi_sc_1x1_wrapper u_hsc (
    .aclk_250(clk_main_a0), .aresetn_250(sc_host_aresetn), .aclk_450(hclk),
    .AXI4_araddr(hs_araddr), .AXI4_arburst(hs_arburst), .AXI4_arcache(4'd3), .AXI4_arid(hs_arid),
    .AXI4_arlen(hs_arlen), .AXI4_arlock(1'b0), .AXI4_arprot(3'd2), .AXI4_arqos(4'd0),
    .AXI4_arready(hs_arready), .AXI4_arsize(hs_arsize), .AXI4_arvalid(hs_arvalid),
    .AXI4_awaddr(hs_awaddr), .AXI4_awburst(hs_awburst), .AXI4_awcache(4'd3), .AXI4_awid(hs_awid),
    .AXI4_awlen(hs_awlen), .AXI4_awlock(1'b0), .AXI4_awprot(3'd2), .AXI4_awqos(4'd0),
    .AXI4_awready(hs_awready), .AXI4_awsize(hs_awsize), .AXI4_awvalid(hs_awvalid),
    .AXI4_bid(hs_bid), .AXI4_bready(hs_bready), .AXI4_bresp(hs_bresp), .AXI4_bvalid(hs_bvalid),
    .AXI4_rdata(hs_rdata), .AXI4_rid(hs_rid), .AXI4_rlast(hs_rlast), .AXI4_rready(hs_rready),
    .AXI4_rresp(hs_rresp), .AXI4_rvalid(hs_rvalid), .AXI4_wdata(hs_wdata), .AXI4_wlast(hs_wlast),
    .AXI4_wready(hs_wready), .AXI4_wstrb(hs_wstrb), .AXI4_wvalid(hs_wvalid),
    .AXI3_araddr({hm_araddr_hi, hm_araddr}), .AXI3_arburst(hm_arburst), .AXI3_arcache(),
    .AXI3_arlen(hm_arlen), .AXI3_arlock(), .AXI3_arprot(), .AXI3_arqos(), .AXI3_arready(hm_arready),
    .AXI3_arsize(hm_arsize), .AXI3_arvalid(hm_arvalid),
    .AXI3_awaddr({hm_awaddr_hi, hm_awaddr}), .AXI3_awburst(hm_awburst), .AXI3_awcache(),
    .AXI3_awlen(hm_awlen), .AXI3_awlock(), .AXI3_awprot(), .AXI3_awqos(), .AXI3_awready(hm_awready),
    .AXI3_awsize(hm_awsize), .AXI3_awvalid(hm_awvalid),
    .AXI3_bready(hm_bready), .AXI3_bresp(hm_bresp), .AXI3_bvalid(hm_bvalid),
    .AXI3_rdata(hm_rdata), .AXI3_rlast(hm_rlast), .AXI3_rready(hm_rready), .AXI3_rresp(hm_rresp),
    .AXI3_rvalid(hm_rvalid), .AXI3_wdata(hm_wdata), .AXI3_wlast(hm_wlast), .AXI3_wready(hm_wready),
    .AXI3_wstrb(hm_wstrb), .AXI3_wvalid(hm_wvalid));
  logic [33:0] hmg_awaddr, hmg_araddr;
  otpu_hbm_geometry u_hgeo (.s_awaddr(hm_awaddr), .s_araddr(hm_araddr),
                            .m_awaddr(hmg_awaddr), .m_araddr(hmg_araddr));
  assign hm_awid = '0;                  // (cl_axi_sc_1x1's AXI3 side has no IDs: as cl_hbm_axi4.sv)
  assign hm_arid = '0;
  // the host's 450 MHz slice: its master hx_* onto its own port NPORT, or (HOST_MUX) into
  // u_sys's mux on port NPORT - 1 (hs_*), which keeps the slice before the mux as before a port
  logic [5:0]   hx_awid, hx_bid, hx_arid, hx_rid;
  logic [33:0]  hx_awaddr, hx_araddr;
  logic [3:0]   hx_awlen, hx_arlen;
  logic [2:0]   hx_awsize, hx_arsize;
  logic [1:0]   hx_awburst, hx_arburst, hx_bresp, hx_rresp;
  logic         hx_awvalid, hx_awready, hx_wlast, hx_wvalid, hx_wready, hx_bvalid, hx_bready;
  logic         hx_arvalid, hx_arready, hx_rlast, hx_rvalid, hx_rready;
  logic [255:0] hx_wdata, hx_rdata;
  logic [31:0]  hx_wstrb;
    cl_axi3_256b_reg_slice u_hrs256 (
      .aclk(hclk), .aresetn(sc_hbm_aresetn),
      .s_axi_awid(hm_awid), .s_axi_awaddr(hmg_awaddr), .s_axi_awlen(hm_awlen), .s_axi_awsize(hm_awsize),
      .s_axi_awburst(hm_awburst), .s_axi_awlock(2'b0), .s_axi_awcache(4'd3), .s_axi_awprot(3'd2),
      .s_axi_awqos(4'd0), .s_axi_awvalid(hm_awvalid), .s_axi_awready(hm_awready), .s_axi_wid(6'd0),
      .s_axi_wdata(hm_wdata), .s_axi_wstrb(hm_wstrb), .s_axi_wlast(hm_wlast), .s_axi_wvalid(hm_wvalid),
      .s_axi_wready(hm_wready), .s_axi_bid(hm_bid), .s_axi_bresp(hm_bresp), .s_axi_bvalid(hm_bvalid),
      .s_axi_bready(hm_bready), .s_axi_arid(hm_arid), .s_axi_araddr(hmg_araddr), .s_axi_arlen(hm_arlen),
      .s_axi_arsize(hm_arsize), .s_axi_arburst(hm_arburst), .s_axi_arlock(2'b0), .s_axi_arcache(4'd3),
      .s_axi_arprot(3'd2), .s_axi_arqos(4'd0), .s_axi_arvalid(hm_arvalid), .s_axi_arready(hm_arready),
      .s_axi_rid(hm_rid), .s_axi_rdata(hm_rdata), .s_axi_rresp(hm_rresp), .s_axi_rlast(hm_rlast),
      .s_axi_rvalid(hm_rvalid), .s_axi_rready(hm_rready), .m_axi_awid(hx_awid), .m_axi_awaddr(hx_awaddr),
      .m_axi_awlen(hx_awlen), .m_axi_awsize(hx_awsize), .m_axi_awburst(hx_awburst), .m_axi_awlock(),
      .m_axi_awcache(), .m_axi_awprot(), .m_axi_awqos(), .m_axi_awvalid(hx_awvalid),
      .m_axi_awready(hx_awready), .m_axi_wid(), .m_axi_wdata(hx_wdata), .m_axi_wstrb(hx_wstrb),
      .m_axi_wlast(hx_wlast), .m_axi_wvalid(hx_wvalid), .m_axi_wready(hx_wready), .m_axi_bid(hx_bid),
      .m_axi_bresp(hx_bresp), .m_axi_bvalid(hx_bvalid), .m_axi_bready(hx_bready), .m_axi_arid(hx_arid),
      .m_axi_araddr(hx_araddr), .m_axi_arlen(hx_arlen), .m_axi_arsize(hx_arsize), .m_axi_arburst(hx_arburst),
      .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
      .m_axi_arvalid(hx_arvalid), .m_axi_arready(hx_arready), .m_axi_rid(hx_rid), .m_axi_rdata(hx_rdata),
      .m_axi_rresp(hx_rresp), .m_axi_rlast(hx_rlast), .m_axi_rvalid(hx_rvalid), .m_axi_rready(hx_rready)
    );
  if (HOST_MUX) begin : g_hmux          // into u_sys's otpu_hbm_mux2 on port NPORT - 1
    assign hs_awid = hx_awid; assign hs_awaddr = hx_awaddr; assign hs_awlen = hx_awlen;
    assign hs_awsize = hx_awsize; assign hs_awburst = hx_awburst; assign hs_awvalid = hx_awvalid;
    assign hx_awready = hs_awready; assign hs_wid = '0; assign hs_wdata = hx_wdata;
    assign hs_wstrb = hx_wstrb; assign hs_wlast = hx_wlast; assign hs_wvalid = hx_wvalid;
    assign hx_wready = hs_wready; assign hx_bid = hs_bid; assign hx_bresp = hs_bresp;
    assign hx_bvalid = hs_bvalid; assign hs_bready = hx_bready; assign hs_arid = hx_arid;
    assign hs_araddr = hx_araddr; assign hs_arlen = hx_arlen; assign hs_arsize = hx_arsize;
    assign hs_arburst = hx_arburst; assign hs_arvalid = hx_arvalid; assign hx_arready = hs_arready;
    assign hx_rid = hs_rid; assign hx_rdata = hs_rdata; assign hx_rresp = hs_rresp;
    assign hx_rlast = hs_rlast; assign hx_rvalid = hs_rvalid; assign hs_rready = hx_rready;
  end else begin : g_hport              // the host's own port NPORT
    assign x_awid[NPORT] = hx_awid; assign x_awaddr[NPORT] = hx_awaddr; assign x_awlen[NPORT] = hx_awlen;
    assign x_awsize[NPORT] = hx_awsize; assign x_awburst[NPORT] = hx_awburst; assign x_awvalid[NPORT] = hx_awvalid;
    assign hx_awready = x_awready[NPORT]; assign x_wdata[NPORT] = hx_wdata; assign x_wstrb[NPORT] = hx_wstrb;
    assign x_wlast[NPORT] = hx_wlast; assign x_wvalid[NPORT] = hx_wvalid; assign hx_wready = x_wready[NPORT];
    assign hx_bid = x_bid[NPORT]; assign hx_bresp = x_bresp[NPORT]; assign hx_bvalid = x_bvalid[NPORT];
    assign x_bready[NPORT] = hx_bready; assign x_arid[NPORT] = hx_arid; assign x_araddr[NPORT] = hx_araddr;
    assign x_arlen[NPORT] = hx_arlen; assign x_arsize[NPORT] = hx_arsize; assign x_arburst[NPORT] = hx_arburst;
    assign x_arvalid[NPORT] = hx_arvalid; assign hx_arready = x_arready[NPORT]; assign hx_rid = x_rid[NPORT];
    assign hx_rdata = x_rdata[NPORT]; assign hx_rresp = x_rresp[NPORT]; assign hx_rlast = x_rlast[NPORT];
    assign hx_rvalid = x_rvalid[NPORT]; assign x_rready[NPORT] = hx_rready;
    assign hs_awid = '0; assign hs_awaddr = '0; assign hs_awlen = '0; assign hs_awsize = '0;
    assign hs_awburst = '0; assign hs_awvalid = 1'b0; assign hs_wid = '0; assign hs_wdata = '0;
    assign hs_wstrb = '0; assign hs_wlast = 1'b0; assign hs_wvalid = 1'b0; assign hs_bready = 1'b1;
    assign hs_arid = '0; assign hs_araddr = '0; assign hs_arlen = '0; assign hs_arsize = '0;
    assign hs_arburst = '0; assign hs_arvalid = 1'b0; assign hs_rready = 1'b1;
  end

  // ---- the HBM: the HDK's wrapper (cl_hbm, both stacks; ports past NHBM tied off there),
  // port p behind channel p's 450 MHz register slice (p < NPORT) or the host's (p = NPORT: its
  // own port, the global address switch reaching every channel's pseudo-channel; with the mux
  // the host's transactions enter port NPORT - 1 behind its engine); every AXI input of the
  // wrapper straight from those slices' registers
  // the wrapper's stats bus (its soft reset, its ready bits): idle; ready comes out directly
  cfg_bus_t hbm_stat_bus();
  assign hbm_stat_bus.addr = '0;
  assign hbm_stat_bus.wdata = '0;
  assign hbm_stat_bus.wr = 1'b0;
  assign hbm_stat_bus.rd = 1'b0;
  assign hbm_stat_bus.user = '0;
  cl_mem_hbm_wrapper #(.NUM_OF_AXI_PORTS(NHBM), .AXI4_INTERFACE(0)) u_hbm (
    .apb_clk(clk_hbm_ref), .i_clk_250m(clk_main_a0), .i_rst_250m_n(main_reset_n[2]),
    .i_clk_450m(hclk), .i_rst_450m_n(!hrst),
    .i_axi_araddr(x_araddr), .i_axi_arburst(x_arburst), .i_axi_arid(x_arid),
    .i_axi_arlen(x_arlen), .i_axi_arsize(x_arsize), .i_axi_arvalid(x_arvalid),
    .i_axi_awaddr(x_awaddr), .i_axi_awburst(x_awburst), .i_axi_awid(x_awid),
    .i_axi_awlen(x_awlen), .i_axi_awsize(x_awsize), .i_axi_awvalid(x_awvalid),
    .i_axi_rready(x_rready), .i_axi_bready(x_bready), .i_axi_wdata(x_wdata),
    .i_axi_wlast(x_wlast), .i_axi_wstrb(x_wstrb), .i_axi_wvalid(x_wvalid),
    .o_axi_arready(x_arready), .o_axi_awready(x_awready), .o_axi_rdata(x_rdata),
    .o_axi_rid(x_rid), .o_axi_rlast(x_rlast), .o_axi_rresp(x_rresp), .o_axi_rvalid(x_rvalid),
    .o_axi_wready(x_wready), .o_axi_bid(x_bid), .o_axi_bresp(x_bresp), .o_axi_bvalid(x_bvalid),
    .i_hbm_apb_preset_n_1(hbm_apb_preset_n_1), .o_hbm_apb_paddr_1(hbm_apb_paddr_1),
    .o_hbm_apb_pprot_1(hbm_apb_pprot_1), .o_hbm_apb_psel_1(hbm_apb_psel_1),
    .o_hbm_apb_penable_1(hbm_apb_penable_1), .o_hbm_apb_pwrite_1(hbm_apb_pwrite_1),
    .o_hbm_apb_pwdata_1(hbm_apb_pwdata_1), .o_hbm_apb_pstrb_1(hbm_apb_pstrb_1),
    .o_hbm_apb_pready_1(hbm_apb_pready_1), .o_hbm_apb_prdata_1(hbm_apb_prdata_1),
    .o_hbm_apb_pslverr_1(hbm_apb_pslverr_1),
    .i_hbm_apb_preset_n_0(hbm_apb_preset_n_0), .o_hbm_apb_paddr_0(hbm_apb_paddr_0),
    .o_hbm_apb_pprot_0(hbm_apb_pprot_0), .o_hbm_apb_psel_0(hbm_apb_psel_0),
    .o_hbm_apb_penable_0(hbm_apb_penable_0), .o_hbm_apb_pwrite_0(hbm_apb_pwrite_0),
    .o_hbm_apb_pwdata_0(hbm_apb_pwdata_0), .o_hbm_apb_pstrb_0(hbm_apb_pstrb_0),
    .o_hbm_apb_pready_0(hbm_apb_pready_0), .o_hbm_apb_prdata_0(hbm_apb_prdata_0),
    .o_hbm_apb_pslverr_0(hbm_apb_pslverr_0),
    .hbm_stat_bus(hbm_stat_bus), .o_cl_sh_hbm_stat_int(), .o_hbm_ready(hbm_ready));

  // ---- the DIMM: pins only (sh_ddr without it), its stats bus answered
  `include "unused_ddr_template.inc"

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_flr_done = 1'b1;
  assign cl_sh_status0 = 32'h4f545055; // OTPU
  assign cl_sh_status1 = 32'h0048_424d; // HBM
  assign cl_sh_status2 = 32'd0;
  // vLED[0] both MMCMs locked, [1] HBM ready, [4:2] board status.
  assign cl_sh_status_vled = {11'd0, led_s2, hbm_ready, mmcm_locked && hmmcm_locked};

  // Unused shell interfaces: no outbound PCIe DMA, SDA, interrupts or JTAG.
  assign cl_sh_dma_wr_full = '0;
  assign cl_sh_dma_rd_full = '0;
  assign cl_sh_pcim_awid = '0;
  assign cl_sh_pcim_awaddr = '0;
  assign cl_sh_pcim_awlen = '0;
  assign cl_sh_pcim_awsize = '0;
  assign cl_sh_pcim_awburst = '0;
  assign cl_sh_pcim_awcache = '0;
  assign cl_sh_pcim_awlock = '0;
  assign cl_sh_pcim_awprot = '0;
  assign cl_sh_pcim_awqos = '0;
  assign cl_sh_pcim_awuser = '0;
  assign cl_sh_pcim_awvalid = '0;
  assign cl_sh_pcim_wid = '0;
  assign cl_sh_pcim_wdata = '0;
  assign cl_sh_pcim_wstrb = '0;
  assign cl_sh_pcim_wlast = '0;
  assign cl_sh_pcim_wuser = '0;
  assign cl_sh_pcim_wvalid = '0;
  assign cl_sh_pcim_bready = '0;
  assign cl_sh_pcim_arid = '0;
  assign cl_sh_pcim_araddr = '0;
  assign cl_sh_pcim_arlen = '0;
  assign cl_sh_pcim_arsize = '0;
  assign cl_sh_pcim_arburst = '0;
  assign cl_sh_pcim_arcache = '0;
  assign cl_sh_pcim_arlock = '0;
  assign cl_sh_pcim_arprot = '0;
  assign cl_sh_pcim_arqos = '0;
  assign cl_sh_pcim_aruser = '0;
  assign cl_sh_pcim_arvalid = '0;
  assign cl_sh_pcim_rready = '0;
  assign cl_sh_apppf_irq_req = '0;
  assign cl_sh_dma_pcis_ruser = '0;
  assign cl_sda_awready = '0;
  assign cl_sda_wready = '0;
  assign cl_sda_bresp = '0;
  assign cl_sda_bvalid = '0;
  assign cl_sda_arready = '0;
  assign cl_sda_rdata = '0;
  assign cl_sda_rresp = '0;
  assign cl_sda_rvalid = '0;
  assign tdo = '0;
  assign PCIE_EP_TXP = '0;
  assign PCIE_EP_TXN = '0;
  assign PCIE_RP_PERSTN = '0;
  assign PCIE_RP_TXP = '0;
  assign PCIE_RP_TXN = '0;
endmodule
