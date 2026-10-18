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
// AXI ports). The three groups meet only through otpu_afifo crossings, gray counters, held words
// and synchronized levels (cl_timing_user_hbm.tcl bounds them).
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

  // ---- the accelerator and its memory fabric
  logic [NPORT-1:0][5:0]   hbm_awid, hbm_arid, hbm_bid, hbm_rid;
  logic [NPORT-1:0][33:0]  hbm_awaddr, hbm_araddr;
  logic [NPORT-1:0][3:0]   hbm_awlen, hbm_arlen;
  logic [NPORT-1:0]        hbm_awvalid, hbm_awready, hbm_wlast, hbm_wvalid, hbm_wready;
  logic [NPORT-1:0]        hbm_bvalid, hbm_bready, hbm_arvalid, hbm_arready;
  logic [NPORT-1:0]        hbm_rlast, hbm_rvalid, hbm_rready;
  logic [NPORT-1:0][255:0] hbm_wdata, hbm_rdata;
  logic [NPORT-1:0][31:0]  hbm_wstrb;
  logic [NPORT-1:0][1:0]   hbm_bresp, hbm_rresp;

  otpu_f2_hbm_sys #(
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
    .hbm_awid, .hbm_awaddr, .hbm_awlen, .hbm_awvalid, .hbm_awready, .hbm_wdata, .hbm_wstrb,
    .hbm_wlast, .hbm_wvalid, .hbm_wready, .hbm_bid, .hbm_bresp, .hbm_bvalid, .hbm_bready,
    .hbm_arid, .hbm_araddr, .hbm_arlen, .hbm_arvalid, .hbm_arready, .hbm_rid, .hbm_rdata,
    .hbm_rresp, .hbm_rlast, .hbm_rvalid, .hbm_rready);

  // ---- the HBM: the HDK's wrapper (cl_hbm, both stacks; ports past NPORT tied off there)
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
  for (genvar p = 0; p < NPORT; p++) begin : g_hbm
    assign w_awid[p] = hbm_awid[p];       assign w_awaddr[p] = hbm_awaddr[p];
    assign w_awlen[p] = hbm_awlen[p];     assign w_awsize[p] = 3'd5;
    assign w_awburst[p] = 2'b01;          assign w_awvalid[p] = hbm_awvalid[p];
    assign hbm_awready[p] = w_awready[p];
    assign w_wdata[p] = hbm_wdata[p];     assign w_wstrb[p] = hbm_wstrb[p];
    assign w_wlast[p] = hbm_wlast[p];     assign w_wvalid[p] = hbm_wvalid[p];
    assign hbm_wready[p] = w_wready[p];
    assign hbm_bid[p] = w_bid[p];         assign hbm_bresp[p] = w_bresp[p];
    assign hbm_bvalid[p] = w_bvalid[p];   assign w_bready[p] = hbm_bready[p];
    assign w_arid[p] = hbm_arid[p];       assign w_araddr[p] = hbm_araddr[p];
    assign w_arlen[p] = hbm_arlen[p];     assign w_arsize[p] = 3'd5;
    assign w_arburst[p] = 2'b01;          assign w_arvalid[p] = hbm_arvalid[p];
    assign hbm_arready[p] = w_arready[p];
    assign hbm_rid[p] = w_rid[p];         assign hbm_rdata[p] = w_rdata[p];
    assign hbm_rresp[p] = w_rresp[p];     assign hbm_rlast[p] = w_rlast[p];
    assign hbm_rvalid[p] = w_rvalid[p];   assign w_rready[p] = hbm_rready[p];
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
