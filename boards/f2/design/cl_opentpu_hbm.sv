// openTPU AWS F2 small-shell top, the HBM build (targets/f2-hbm*.toml, docs/f2_hbm.md): module
// cl_opentpu, as the HDK flow names the CL; boards/f2/build.py puts this file in place of
// cl_opentpu.sv for a target whose platform mem is "hbm". The HDK supplies cl_ports.vh, the
// unused-DDR template (the DIMM's pins, sh_ddr without the DIMM) and the cl_hbm IP; build.py
// copies the HDK's cl_mem_hbm_wrapper (hdk/cl/examples/cl_mem_perf) next to it.
//
// Clocks: clk_main_a0 (250 MHz) is the OCL / PCIS clock (aclk) and drives u_mmcm: the core clock
// (clk, CLKOUT0) and 2x it (clk_f, CLKOUT1, the pumped TMEM's: TMEM_PMP 1, phase aligned with clk
// from the one VCO), as build.py's HBM_CORE_MMCM gives them in opentpu_geometry.vh: VCO 1200 MHz
// /6 /3 = 200 / 400 MHz, or /8 /4 = 150 / 300 (f2-hbm:...c150), or VCO 1050 (PFD 50) /6 /3 =
// 175 / 350. clk_hbm_ref (100 MHz, the shell's HBM reference) is the
// HBM's reference and APB clock and drives u_hmmcm: VCO 900 MHz, /2 = 450 MHz (hclk, the HBM's
// AXI ports). The core and clk_main_a0 meet in openTPU's logic only through otpu_axil_cdc's held
// words and synchronized levels (cl_timing_user_hbm.tcl bounds them); the HBM clock meets the
// others only inside the SmartConnect fabric (u_sc: its own CDC constraints) and the HDK's HBM
// wrapper (its XPM crossings).
//
// The memory path (docs/f2_hbm.md; Alex's burst200 design, ergodex-openTPU 06c169cb, at S x NCH
// channels): u_sys's channel p is an AXI4 512-bit master at the core clock (otpu_native_axi,
// ch_*[p]) into u_sc, a block design generated on the build host (boards/f2/build/scripts/
// create_hbm_sc.tcl: per channel a 1 x 1 SmartConnect, that master at the core clock onto
// M_H<p>, AXI3 256 bits @450 MHz; the host's PCIS through its own 1 x 1 onto M_HOST, HBM port
// NPORT), its instance generated from the BD wrapper's own ports (otpu_hbm_sc_inst.vh); M_H<p>
// / M_HOST -> otpu_hbm_geometry (32-byte aligned addresses) -> HBM AXI port p / NPORT.
// Resets: the shell's rst_main_n and the MMCMs' lock, synchronized into each clock; the HBM
// wrapper's own (its 250 MHz reset, then its APB / AXI resets) from the same. The HBM is ready
// (its two stacks' APB init complete: cl_mem_hbm_wrapper o_hbm_ready) before the core may run:
// otpu_board's CALIB bits (vLED bit 1).
module cl_opentpu (
  `include "cl_ports.vh"
);
  `include "cl_id_defines.vh"
  `include "opentpu_geometry.vh"        // build.py: localparam S, NCH, the core MMCM (the target's)
  localparam int NPORT = S * NCH;
  localparam int NHBM = NPORT + 1;      // HBM AXI ports: the channels' 0 .. NPORT - 1, the host's NPORT
  initial if (NHBM > 32) $fatal(1, "cl_opentpu: the host needs a free HBM port (S x NCH <= 31)");

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
  logic [NPORT-1:0][33:0]  ch_awaddr, ch_araddr;
  logic [NPORT-1:0][7:0]   ch_arlen;
  logic [NPORT-1:0]        ch_awvalid, ch_awready, ch_wvalid, ch_wready, ch_bvalid, ch_bready;
  logic [NPORT-1:0]        ch_arvalid, ch_arready, ch_rvalid, ch_rlast, ch_rready;
  logic [NPORT-1:0][511:0] ch_wdata, ch_rdata;
  logic [NPORT-1:0][63:0]  ch_wstrb;
  logic [NPORT-1:0][1:0]   ch_bresp, ch_rresp;

  otpu_f2_hbm_sys #(
    `include "opentpu_params.vh"
  ) u_sys (
    .clk, .clk_f, .rst, .aclk(clk_main_a0), .arst, .hbm_ready, .led(board_led),
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
    .ch_awaddr, .ch_awvalid, .ch_awready, .ch_wdata, .ch_wstrb, .ch_wvalid, .ch_wready, .ch_bvalid,
    .ch_bresp, .ch_bready, .ch_araddr, .ch_arlen, .ch_arvalid, .ch_arready, .ch_rdata, .ch_rvalid,
    .ch_rlast, .ch_rresp, .ch_rready);

  // ---- the memory path behind u_sys, the shape of AWS's cl_dram_hbm_dma (hdk/cl/examples/
  // cl_dram_hbm_dma/design/cl_hbm_axi4.sv: axi_register_slice -> a 1 x 1 SmartConnect
  // (cl_axi_sc_1x1) -> cl_axi3_256b_reg_slice at 450 MHz -> the HBM wrapper), per channel and for
  // the host (docs/f2_hbm.md "The memory path"; the HDK's prebuilt register slices and, for the
  // host, its cl_axi_sc_1x1 block design as shipped; each channel's 1 x 1 SmartConnect is
  // create_hbm_sc.tcl's, cl_axi_sc_1x1's settings with S00 at the core clock). No crossbar.
  // Resets: each domain's synchronizer registered again (max_fanout: replicated per slice)
  (* max_fanout = 16 *) logic sc_aresetn, sc_host_aresetn, sc_hbm_aresetn;
  always_ff @(posedge clk) sc_aresetn <= core_reset_n[2];
  always_ff @(posedge clk_main_a0) sc_host_aresetn <= bridge_reset_n[2];
  always_ff @(posedge hclk) sc_hbm_aresetn <= hbm_reset_n[2];
  // HBM ports: x_*[p] (p < NPORT channel p, NPORT the host), 34-bit addresses, 6-bit IDs (AXI3)
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
  // channel p: cs_* (its 512-bit register slice -> its SmartConnect), cm_* (its SmartConnect ->
  // its 256-bit register slice)
  logic [15:0]  cs_awid [0:NPORT-1], cs_arid [0:NPORT-1], cs_bid [0:NPORT-1], cs_rid [0:NPORT-1];
  logic [63:0]  cs_awaddr [0:NPORT-1], cs_araddr [0:NPORT-1];
  logic [7:0]   cs_awlen [0:NPORT-1], cs_arlen [0:NPORT-1];
  logic [2:0]   cs_awsize [0:NPORT-1], cs_arsize [0:NPORT-1];
  logic [1:0]   cs_awburst [0:NPORT-1], cs_arburst [0:NPORT-1], cs_bresp [0:NPORT-1], cs_rresp [0:NPORT-1];
  logic         cs_awvalid [0:NPORT-1], cs_awready [0:NPORT-1], cs_wlast [0:NPORT-1], cs_wvalid [0:NPORT-1], cs_wready [0:NPORT-1];
  logic         cs_bvalid [0:NPORT-1], cs_bready [0:NPORT-1], cs_arvalid [0:NPORT-1], cs_arready [0:NPORT-1];
  logic         cs_rlast [0:NPORT-1], cs_rvalid [0:NPORT-1], cs_rready [0:NPORT-1];
  logic [511:0] cs_wdata [0:NPORT-1], cs_rdata [0:NPORT-1];
  logic [63:0]  cs_wstrb [0:NPORT-1];
  logic [5:0]  cm_awid [0:NPORT-1], cm_arid [0:NPORT-1], cm_bid [0:NPORT-1], cm_rid [0:NPORT-1];
  logic [33:0]  cm_awaddr [0:NPORT-1], cm_araddr [0:NPORT-1];
  logic [3:0]   cm_awlen [0:NPORT-1], cm_arlen [0:NPORT-1];
  logic [2:0]   cm_awsize [0:NPORT-1], cm_arsize [0:NPORT-1];
  logic [1:0]   cm_awburst [0:NPORT-1], cm_arburst [0:NPORT-1], cm_bresp [0:NPORT-1], cm_rresp [0:NPORT-1];
  logic         cm_awvalid [0:NPORT-1], cm_awready [0:NPORT-1], cm_wlast [0:NPORT-1], cm_wvalid [0:NPORT-1], cm_wready [0:NPORT-1];
  logic         cm_bvalid [0:NPORT-1], cm_bready [0:NPORT-1], cm_arvalid [0:NPORT-1], cm_arready [0:NPORT-1];
  logic         cm_rlast [0:NPORT-1], cm_rvalid [0:NPORT-1], cm_rready [0:NPORT-1];
  logic [255:0] cm_wdata [0:NPORT-1], cm_rdata [0:NPORT-1];
  logic [31:0]  cm_wstrb [0:NPORT-1];
  logic [33:0] cmg_awaddr [0:NPORT-1], cmg_araddr [0:NPORT-1];
  for (genvar p = 0; p < NPORT; p++) begin : g_ch
    // (32-byte aligned addresses into the 450 MHz slice: Alex's otpu_hbm_geometry, before the
    // last register stage so the wrapper's AXI inputs come from flip-flops alone, as
    // cl_mem_perf's q1_* stage)
    otpu_hbm_geometry u_geo (.s_awaddr(cm_awaddr[p]), .s_araddr(cm_araddr[p]),
                             .m_awaddr(cmg_awaddr[p]), .m_araddr(cmg_araddr[p]));
    axi_register_slice u_rs512 (
      .aclk(clk), .aresetn(sc_aresetn),
      .s_axi_awid(16'd0), .s_axi_awaddr({30'd0, ch_awaddr[p]}), .s_axi_awlen(8'd0), .s_axi_awsize(3'd6),
      .s_axi_awburst(2'b01), .s_axi_awlock(1'b0), .s_axi_awcache(4'd3), .s_axi_awprot(3'd2),
      .s_axi_awregion(4'd0), .s_axi_awqos(4'd0), .s_axi_awvalid(ch_awvalid[p]), .s_axi_awready(ch_awready[p]),
      .s_axi_wdata(ch_wdata[p]), .s_axi_wstrb(ch_wstrb[p]), .s_axi_wlast(1'b1), .s_axi_wvalid(ch_wvalid[p]),
      .s_axi_wready(ch_wready[p]), .s_axi_bid(), .s_axi_bresp(ch_bresp[p]), .s_axi_bvalid(ch_bvalid[p]),
      .s_axi_bready(ch_bready[p]), .s_axi_arid(16'd0), .s_axi_araddr({30'd0, ch_araddr[p]}), .s_axi_arlen(ch_arlen[p]),
      .s_axi_arsize(3'd6), .s_axi_arburst(2'b01), .s_axi_arlock(1'b0), .s_axi_arcache(4'd3),
      .s_axi_arprot(3'd2), .s_axi_arregion(4'd0), .s_axi_arqos(4'd0), .s_axi_arvalid(ch_arvalid[p]),
      .s_axi_arready(ch_arready[p]), .s_axi_rid(), .s_axi_rdata(ch_rdata[p]), .s_axi_rresp(ch_rresp[p]),
      .s_axi_rlast(ch_rlast[p]), .s_axi_rvalid(ch_rvalid[p]), .s_axi_rready(ch_rready[p]), .m_axi_awid(cs_awid[p]),
      .m_axi_awaddr(cs_awaddr[p]), .m_axi_awlen(cs_awlen[p]), .m_axi_awsize(cs_awsize[p]), .m_axi_awburst(cs_awburst[p]),
      .m_axi_awlock(), .m_axi_awcache(), .m_axi_awprot(), .m_axi_awregion(),
      .m_axi_awqos(), .m_axi_awvalid(cs_awvalid[p]), .m_axi_awready(cs_awready[p]), .m_axi_wdata(cs_wdata[p]),
      .m_axi_wstrb(cs_wstrb[p]), .m_axi_wvalid(cs_wvalid[p]), .m_axi_wlast(cs_wlast[p]), .m_axi_wready(cs_wready[p]),
      .m_axi_bresp(cs_bresp[p]), .m_axi_bvalid(cs_bvalid[p]), .m_axi_bid(cs_bid[p]), .m_axi_bready(cs_bready[p]),
      .m_axi_arid(cs_arid[p]), .m_axi_araddr(cs_araddr[p]), .m_axi_arlen(cs_arlen[p]), .m_axi_arsize(cs_arsize[p]),
      .m_axi_arburst(cs_arburst[p]), .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
      .m_axi_arregion(), .m_axi_arqos(), .m_axi_arvalid(cs_arvalid[p]), .m_axi_arready(cs_arready[p]),
      .m_axi_rid(cs_rid[p]), .m_axi_rdata(cs_rdata[p]), .m_axi_rresp(cs_rresp[p]), .m_axi_rlast(cs_rlast[p]),
      .m_axi_rvalid(cs_rvalid[p]), .m_axi_rready(cs_rready[p])
    );
    cl_axi3_256b_reg_slice u_rs256 (
      .aclk(hclk), .aresetn(sc_hbm_aresetn),
      .s_axi_awid(cm_awid[p]), .s_axi_awaddr(cmg_awaddr[p]), .s_axi_awlen(cm_awlen[p]), .s_axi_awsize(cm_awsize[p]),
      .s_axi_awburst(cm_awburst[p]), .s_axi_awlock(2'b0), .s_axi_awcache(4'd3), .s_axi_awprot(3'd2),
      .s_axi_awqos(4'd0), .s_axi_awvalid(cm_awvalid[p]), .s_axi_awready(cm_awready[p]), .s_axi_wid(6'd0),
      .s_axi_wdata(cm_wdata[p]), .s_axi_wstrb(cm_wstrb[p]), .s_axi_wlast(cm_wlast[p]), .s_axi_wvalid(cm_wvalid[p]),
      .s_axi_wready(cm_wready[p]), .s_axi_bid(cm_bid[p]), .s_axi_bresp(cm_bresp[p]), .s_axi_bvalid(cm_bvalid[p]),
      .s_axi_bready(cm_bready[p]), .s_axi_arid(cm_arid[p]), .s_axi_araddr(cmg_araddr[p]), .s_axi_arlen(cm_arlen[p]),
      .s_axi_arsize(cm_arsize[p]), .s_axi_arburst(cm_arburst[p]), .s_axi_arlock(2'b0), .s_axi_arcache(4'd3),
      .s_axi_arprot(3'd2), .s_axi_arqos(4'd0), .s_axi_arvalid(cm_arvalid[p]), .s_axi_arready(cm_arready[p]),
      .s_axi_rid(cm_rid[p]), .s_axi_rdata(cm_rdata[p]), .s_axi_rresp(cm_rresp[p]), .s_axi_rlast(cm_rlast[p]),
      .s_axi_rvalid(cm_rvalid[p]), .s_axi_rready(cm_rready[p]), .m_axi_awid(x_awid[p]), .m_axi_awaddr(x_awaddr[p]),
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
  // the channels' SmartConnects (u_sc: one 1 x 1 per channel, S_N<p> = cs_*[p], M_H<p> = cm_*[p]),
  // the instance generated from the block design's wrapper (create_hbm_sc.tcl writes
  // otpu_hbm_sc_inst.vh: every wrapper port mapped, address / ID widths adapted, the unused
  // inputs tied, an ID the wrapper lacks tied to 0)
  `include "otpu_hbm_sc_inst.vh"
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
      .s_axi_rvalid(hm_rvalid), .s_axi_rready(hm_rready), .m_axi_awid(x_awid[NPORT]), .m_axi_awaddr(x_awaddr[NPORT]),
      .m_axi_awlen(x_awlen[NPORT]), .m_axi_awsize(x_awsize[NPORT]), .m_axi_awburst(x_awburst[NPORT]), .m_axi_awlock(),
      .m_axi_awcache(), .m_axi_awprot(), .m_axi_awqos(), .m_axi_awvalid(x_awvalid[NPORT]),
      .m_axi_awready(x_awready[NPORT]), .m_axi_wid(), .m_axi_wdata(x_wdata[NPORT]), .m_axi_wstrb(x_wstrb[NPORT]),
      .m_axi_wlast(x_wlast[NPORT]), .m_axi_wvalid(x_wvalid[NPORT]), .m_axi_wready(x_wready[NPORT]), .m_axi_bid(x_bid[NPORT]),
      .m_axi_bresp(x_bresp[NPORT]), .m_axi_bvalid(x_bvalid[NPORT]), .m_axi_bready(x_bready[NPORT]), .m_axi_arid(x_arid[NPORT]),
      .m_axi_araddr(x_araddr[NPORT]), .m_axi_arlen(x_arlen[NPORT]), .m_axi_arsize(x_arsize[NPORT]), .m_axi_arburst(x_arburst[NPORT]),
      .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
      .m_axi_arvalid(x_arvalid[NPORT]), .m_axi_arready(x_arready[NPORT]), .m_axi_rid(x_rid[NPORT]), .m_axi_rdata(x_rdata[NPORT]),
      .m_axi_rresp(x_rresp[NPORT]), .m_axi_rlast(x_rlast[NPORT]), .m_axi_rvalid(x_rvalid[NPORT]), .m_axi_rready(x_rready[NPORT])
    );

  // ---- the HBM: the HDK's wrapper (cl_hbm, both stacks; ports past NHBM tied off there),
  // port p behind channel p's 450 MHz register slice (p < NPORT) or the host's (p = NPORT: its
  // own port, the global address switch reaching every channel's pseudo-channel); every AXI
  // input of the wrapper straight from those slices' registers
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
