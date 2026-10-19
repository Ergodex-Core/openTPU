// openTPU AWS F2 small-shell top, the HBM build (targets/f2-hbm*.toml, docs/f2_hbm.md): module
// cl_opentpu, as the HDK flow names the CL; boards/f2/build.py puts this file in place of
// cl_opentpu.sv for a target whose platform mem is "hbm". The HDK supplies cl_ports.vh, the
// unused-DDR template (the DIMM's pins, sh_ddr without the DIMM) and the cl_hbm IP; build.py
// copies the HDK's cl_mem_hbm_wrapper (hdk/cl/examples/cl_mem_perf) next to it.
//
// Clocks: clk_main_a0 (250 MHz) is the OCL / PCIS clock (aclk) and drives u_mmcm: VCO 1200 MHz,
// /6 = 200 MHz core clock (clk), /3 = 400 MHz (clk_f, the pumped TMEM's: TMEM_PMP 1, phase
// aligned with clk from the one VCO). clk_hbm_ref (100 MHz, the shell's HBM reference) is the
// HBM's reference and APB clock and drives u_hmmcm: VCO 900 MHz, /2 = 450 MHz (hclk, the HBM's
// AXI ports). The core and clk_main_a0 meet in openTPU's logic only through otpu_axil_cdc's held
// words and synchronized levels (cl_timing_user_hbm.tcl bounds them); the HBM clock meets the
// others only inside the SmartConnect fabric (u_sc: its own CDC constraints) and the HDK's HBM
// wrapper (its XPM crossings).
//
// The memory path (docs/f2_hbm.md; Alex's burst200 design, ergodex-openTPU 06c169cb, at S x NCH
// channels): u_sys's channel p is an AXI4 512-bit master at the core clock (otpu_native_axi,
// ch_*[p]) into u_sc, a block design generated on the build host (boards/f2/build/scripts/
// create_hbm_sc.tcl: per channel a SmartConnect with that master @200 and the host's branch
// @250 onto M_H<p>, AXI3 256 bits @450 MHz; a 1 x NPORT SmartConnect demuxes the host's PCIS
// by address), its instance generated from the BD wrapper's own ports (otpu_hbm_sc_inst.vh);
// M_H<p> -> otpu_hbm_geometry (32-byte aligned addresses) -> HBM AXI port p.
// Resets: the shell's rst_main_n and the MMCMs' lock, synchronized into each clock; the HBM
// wrapper's own (its 250 MHz reset, then its APB / AXI resets) from the same. The HBM is ready
// (its two stacks' APB init complete: cl_mem_hbm_wrapper o_hbm_ready) before the core may run:
// otpu_board's CALIB bits (vLED bit 1).
module cl_opentpu (
  `include "cl_ports.vh"
);
  `include "cl_id_defines.vh"
  `include "opentpu_geometry.vh"        // build.py: localparam S, NCH (the target's)
  localparam int NPORT = S * NCH;

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

  // 250 MHz -> 1200 MHz VCO -> 200 / 400 MHz, phase aligned (the pumped TMEM relies on it)
  MMCME4_ADV #(.CLKIN1_PERIOD(4.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(4.8),
               .CLKOUT0_DIVIDE_F(6.0), .CLKOUT1_DIVIDE(3), .STARTUP_WAIT("FALSE")) u_mmcm (
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

  // ---- the SmartConnect fabric (u_sc): its masters M_H<p> (x_*, hclk), the instance generated
  // from the block design's wrapper (create_hbm_sc.tcl writes otpu_hbm_sc_inst.vh: every
  // wrapper port mapped to these signals, the unused inputs tied, an HBM-side ID the wrapper
  // lacks tied to 0). Its reset (aresetn, associated with core_clk): the core synchronizer's,
  // registered
  logic [33:0]  x_araddr [0:NPORT-1], x_awaddr [0:NPORT-1];
  logic [1:0]   x_arburst [0:NPORT-1], x_awburst [0:NPORT-1];
  logic [5:0]   x_arid [0:NPORT-1], x_awid [0:NPORT-1], x_rid [0:NPORT-1], x_bid [0:NPORT-1];
  logic [3:0]   x_arlen [0:NPORT-1], x_awlen [0:NPORT-1];
  logic [2:0]   x_arsize [0:NPORT-1], x_awsize [0:NPORT-1];
  logic         x_arvalid [0:NPORT-1], x_awvalid [0:NPORT-1], x_rready [0:NPORT-1];
  logic         x_bready [0:NPORT-1], x_wlast [0:NPORT-1], x_wvalid [0:NPORT-1];
  logic         x_arready [0:NPORT-1], x_awready [0:NPORT-1], x_rlast [0:NPORT-1];
  logic         x_rvalid [0:NPORT-1], x_wready [0:NPORT-1], x_bvalid [0:NPORT-1];
  logic [255:0] x_wdata [0:NPORT-1], x_rdata [0:NPORT-1];
  logic [31:0]  x_wstrb [0:NPORT-1];
  logic [1:0]   x_rresp [0:NPORT-1], x_bresp [0:NPORT-1];
  logic         sc_aresetn, sc_host_aresetn;
  always_ff @(posedge clk) sc_aresetn <= core_reset_n[2];
  always_ff @(posedge clk_main_a0) sc_host_aresetn <= bridge_reset_n[2];   // the host's SmartConnect
  `include "otpu_hbm_sc_inst.vh"

  // ---- the HBM: the HDK's wrapper (cl_hbm, both stacks; ports past NPORT tied off there),
  // port p behind M_H<p> with its addresses 32-byte aligned (otpu_hbm_geometry, Alex's)
  logic [33:0]  w_araddr [0:NPORT-1], w_awaddr [0:NPORT-1];
  for (genvar p = 0; p < NPORT; p++) begin : g_hbm
    otpu_hbm_geometry u_geo (.s_awaddr(x_awaddr[p]), .s_araddr(x_araddr[p]),
                             .m_awaddr(w_awaddr[p]), .m_araddr(w_araddr[p]));
  end
  // the wrapper's stats bus (its soft reset, its ready bits): idle; ready comes out directly
  cfg_bus_t hbm_stat_bus();
  assign hbm_stat_bus.addr = '0;
  assign hbm_stat_bus.wdata = '0;
  assign hbm_stat_bus.wr = 1'b0;
  assign hbm_stat_bus.rd = 1'b0;
  assign hbm_stat_bus.user = '0;
  cl_mem_hbm_wrapper #(.NUM_OF_AXI_PORTS(NPORT), .AXI4_INTERFACE(0)) u_hbm (
    .apb_clk(clk_hbm_ref), .i_clk_250m(clk_main_a0), .i_rst_250m_n(main_reset_n[2]),
    .i_clk_450m(hclk), .i_rst_450m_n(!hrst),
    .i_axi_araddr(w_araddr), .i_axi_arburst(x_arburst), .i_axi_arid(x_arid),
    .i_axi_arlen(x_arlen), .i_axi_arsize(x_arsize), .i_axi_arvalid(x_arvalid),
    .i_axi_awaddr(w_awaddr), .i_axi_awburst(x_awburst), .i_axi_awid(x_awid),
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
