// openTPU AWS F2 small-shell top, the HBM build (targets/f2-hbm*.toml, docs/f2_hbm.md,
// docs/v4/f2-round5.md): module cl_opentpu, as the HDK flow names the CL; boards/f2/build.py puts
// this file in place of cl_opentpu.sv for a target whose platform mem is "hbm". The HDK supplies
// cl_ports.vh, the unused-DDR template (the DIMM's pins, sh_ddr without the DIMM), the cl_hbm IP,
// the cl_axi_sc_1x1 SmartConnect block design (hdk/common/ip: one AXI4 slave of 512 bits with
// 16-bit IDs on its aclk_250 pin, one AXI3 master of 256 bits without IDs on aclk_450, the
// clock crossing, the width and the protocol conversion inside it, as cl_mem_perf's and
// cl_dram_hbm_dma's HBM paths use it) and cl_mem_hbm_wrapper (hdk/cl/examples/cl_mem_perf,
// copied by build.py next to this file).
//
// Structure (round 5: nothing of openTPU's on the HBM clock, as Alex's burst200 image, the only
// openTPU F2 image that closed 200 MHz): otpu_f2_hbm_sys holds the board, the channels' AXI4
// masters, the host path and the port arbiters, all at the core clock, and leaves this top
// NPORT AXI4 masters of 512 bits (u_sys m_*). Each goes into its own cl_axi_sc_1x1
// (g_sc[p].u_sc: its aclk_250 pin takes the 200 MHz core clock; the IP's crossing is
// asynchronous, 250 and 450 MHz being unrelated, so the pin's name is the HDK's only), whose
// AXI3 side goes into port p of the HDK's HBM wrapper at 450 MHz. The openTPU cells on hclk are
// the three flip-flops of hbm_reset_n, the wrapper's synchronous reset (cl_timing_user_hbm.tcl
// checks that nothing else is).
//
// Clocks: clk_main_a0 (250 MHz) is the OCL / PCIS clock (aclk) and drives u_mmcm: VCO 1200 MHz,
// /6 = 200 MHz core clock (clk), /3 = 400 MHz (clk_f, the pumped TMEM's: TMEM_PMP 1, phase
// aligned with clk from the one VCO). clk_hbm_ref (100 MHz, the shell's HBM reference) is the
// HBM's reference and APB clock and drives u_hmmcm: VCO 900 MHz, /2 = 450 MHz (hclk, the HBM's
// AXI ports and the SmartConnects' aclk_450). The openTPU logic's two groups (core, aclk) meet
// only through otpu_afifo crossings, gray counters, held words and synchronized levels
// (cl_timing_user_hbm.tcl bounds them); the SmartConnects and the wrapper carry their own.
// Resets: the shell's rst_main_n and the MMCMs' lock, synchronized into each clock; the
// SmartConnects take the core reset on aresetn_250 (synchronous to their aclk_250 = clk; the IP
// synchronizes it into its other domain); the HBM wrapper's own (its 250 MHz reset, then its APB
// / AXI resets) from the same. The HBM is ready (its two stacks' APB init complete:
// cl_mem_hbm_wrapper o_hbm_ready) before the core may run: otpu_board's CALIB bits (vLED bit 1).
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

  // ---- the accelerator and its memory fabric: NPORT AXI4 masters at the core clock
  logic [NPORT-1:0][15:0]  m_awid, m_arid, m_bid, m_rid;
  logic [NPORT-1:0][33:0]  m_awaddr, m_araddr;
  logic [NPORT-1:0][7:0]   m_awlen, m_arlen;
  logic [NPORT-1:0]        m_awvalid, m_awready, m_wlast, m_wvalid, m_wready;
  logic [NPORT-1:0]        m_bvalid, m_bready, m_arvalid, m_arready;
  logic [NPORT-1:0]        m_rlast, m_rvalid, m_rready;
  logic [NPORT-1:0][511:0] m_wdata, m_rdata;
  logic [NPORT-1:0][63:0]  m_wstrb;
  logic [NPORT-1:0][1:0]   m_bresp, m_rresp;

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
    .m_awid, .m_awaddr, .m_awlen, .m_awvalid, .m_awready, .m_wdata, .m_wstrb, .m_wlast,
    .m_wvalid, .m_wready, .m_bid, .m_bresp, .m_bvalid, .m_bready, .m_arid, .m_araddr, .m_arlen,
    .m_arvalid, .m_arready, .m_rid, .m_rdata, .m_rresp, .m_rlast, .m_rvalid, .m_rready);

  // ---- the HBM: a SmartConnect per port (the HDK's cl_axi_sc_1x1) into the HDK's wrapper
  // (cl_hbm, both stacks; ports past NPORT tied off there)
  logic [33:0]  w_araddr [0:NPORT-1], w_awaddr [0:NPORT-1];
  logic [1:0]   w_arburst [0:NPORT-1], w_awburst [0:NPORT-1];
  logic [5:0]   w_arid [0:NPORT-1], w_awid [0:NPORT-1], w_rid [0:NPORT-1], w_bid [0:NPORT-1];
  logic [3:0]   w_arlen [0:NPORT-1], w_awlen [0:NPORT-1];
  logic [2:0]   w_arsize [0:NPORT-1], w_awsize [0:NPORT-1];
  logic         w_arvalid [0:NPORT-1], w_awvalid [0:NPORT-1], w_rready [0:NPORT-1];
  logic         w_bready [0:NPORT-1], w_wlast [0:NPORT-1], w_wvalid [0:NPORT-1];
  logic         w_arready [0:NPORT-1], w_awready [0:NPORT-1], w_rlast [0:NPORT-1];
  logic         w_rvalid [0:NPORT-1], w_wready [0:NPORT-1], w_bvalid [0:NPORT-1];
  logic [255:0] w_wdata [0:NPORT-1], w_rdata [0:NPORT-1];
  logic [31:0]  w_wstrb [0:NPORT-1];
  logic [1:0]   w_rresp [0:NPORT-1], w_bresp [0:NPORT-1];
  for (genvar p = 0; p < NPORT; p++) begin : g_sc
    // the SmartConnect's AXI3 address is 64 bits (the HDK's bd: the PCIS format); the HBM's
    // 34 go on, the IDs the wrapper takes are 0 (the IP's master has none and the port's
    // responses are in order; the wrapper's bid / rid are left unread)
    logic [63:0] sc_awaddr, sc_araddr;
    assign w_awaddr[p] = sc_awaddr[33:0];
    assign w_araddr[p] = sc_araddr[33:0];
    assign w_awid[p] = 6'd0;
    assign w_arid[p] = 6'd0;
    cl_axi_sc_1x1_wrapper u_sc (
      .aclk_250(clk), .aresetn_250(!rst), .aclk_450(hclk),
      // AXI4 (core clock): the port's master
      .AXI4_awid(m_awid[p]), .AXI4_awaddr({30'd0, m_awaddr[p]}), .AXI4_awlen(m_awlen[p]),
      .AXI4_awsize(3'd6), .AXI4_awburst(2'b01), .AXI4_awlock(1'b0), .AXI4_awcache(4'b0000),
      .AXI4_awprot(3'b000), .AXI4_awqos(4'b0000), .AXI4_awvalid(m_awvalid[p]),
      .AXI4_awready(m_awready[p]),
      .AXI4_wdata(m_wdata[p]), .AXI4_wstrb(m_wstrb[p]), .AXI4_wlast(m_wlast[p]),
      .AXI4_wvalid(m_wvalid[p]), .AXI4_wready(m_wready[p]),
      .AXI4_bid(m_bid[p]), .AXI4_bresp(m_bresp[p]), .AXI4_bvalid(m_bvalid[p]),
      .AXI4_bready(m_bready[p]),
      .AXI4_arid(m_arid[p]), .AXI4_araddr({30'd0, m_araddr[p]}), .AXI4_arlen(m_arlen[p]),
      .AXI4_arsize(3'd6), .AXI4_arburst(2'b01), .AXI4_arlock(1'b0), .AXI4_arcache(4'b0000),
      .AXI4_arprot(3'b000), .AXI4_arqos(4'b0000), .AXI4_arvalid(m_arvalid[p]),
      .AXI4_arready(m_arready[p]),
      .AXI4_rid(m_rid[p]), .AXI4_rdata(m_rdata[p]), .AXI4_rresp(m_rresp[p]),
      .AXI4_rlast(m_rlast[p]), .AXI4_rvalid(m_rvalid[p]), .AXI4_rready(m_rready[p]),
      // AXI3 (hclk): the HBM port
      .AXI3_awaddr(sc_awaddr), .AXI3_awlen(w_awlen[p]), .AXI3_awsize(w_awsize[p]),
      .AXI3_awburst(w_awburst[p]), .AXI3_awlock(), .AXI3_awcache(), .AXI3_awprot(),
      .AXI3_awqos(), .AXI3_awvalid(w_awvalid[p]), .AXI3_awready(w_awready[p]),
      .AXI3_wdata(w_wdata[p]), .AXI3_wstrb(w_wstrb[p]), .AXI3_wlast(w_wlast[p]),
      .AXI3_wvalid(w_wvalid[p]), .AXI3_wready(w_wready[p]),
      .AXI3_bresp(w_bresp[p]), .AXI3_bvalid(w_bvalid[p]), .AXI3_bready(w_bready[p]),
      .AXI3_araddr(sc_araddr), .AXI3_arlen(w_arlen[p]), .AXI3_arsize(w_arsize[p]),
      .AXI3_arburst(w_arburst[p]), .AXI3_arlock(), .AXI3_arcache(), .AXI3_arprot(),
      .AXI3_arqos(), .AXI3_arvalid(w_arvalid[p]), .AXI3_arready(w_arready[p]),
      .AXI3_rdata(w_rdata[p]), .AXI3_rresp(w_rresp[p]), .AXI3_rlast(w_rlast[p]),
      .AXI3_rvalid(w_rvalid[p]), .AXI3_rready(w_rready[p]));
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
    .i_axi_araddr(w_araddr), .i_axi_arburst(w_arburst), .i_axi_arid(w_arid),
    .i_axi_arlen(w_arlen), .i_axi_arsize(w_arsize), .i_axi_arvalid(w_arvalid),
    .i_axi_awaddr(w_awaddr), .i_axi_awburst(w_awburst), .i_axi_awid(w_awid),
    .i_axi_awlen(w_awlen), .i_axi_awsize(w_awsize), .i_axi_awvalid(w_awvalid),
    .i_axi_rready(w_rready), .i_axi_bready(w_bready), .i_axi_wdata(w_wdata),
    .i_axi_wlast(w_wlast), .i_axi_wstrb(w_wstrb), .i_axi_wvalid(w_wvalid),
    .o_axi_arready(w_arready), .o_axi_awready(w_awready), .o_axi_rdata(w_rdata),
    .o_axi_rid(w_rid), .o_axi_rlast(w_rlast), .o_axi_rresp(w_rresp), .o_axi_rvalid(w_rvalid),
    .o_axi_wready(w_wready), .o_axi_bid(w_bid), .o_axi_bresp(w_bresp), .o_axi_bvalid(w_bvalid),
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
