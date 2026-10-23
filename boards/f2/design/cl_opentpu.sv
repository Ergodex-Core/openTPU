// openTPU AWS F2 small-shell top. See targets/f2.toml and docs/f2.md.
// The HDK supplies cl_ports.vh and sh_ddr; no shell sources are copied here.
module cl_opentpu (
  `include "cl_ports.vh"
);
  `include "cl_id_defines.vh"
  logic clk, clk_f, clk_fb, clk_fb_buf, clk_raw, clk_f_raw, mmcm_locked;
  logic [2:0] board_led, board_led_core;
  (* ASYNC_REG = "TRUE" *) logic [2:0] main_reset_n, core_reset_n, bridge_reset_n;
  (* ASYNC_REG = "TRUE" *) logic [2:0] led_s1, led_s2;
  wire clock_reset = !rst_main_n;
  wire bridge_reset = clock_reset || !mmcm_locked;
  wire arst = !bridge_reset_n[2];
  wire rst = !core_reset_n[2];
  logic ddr_ready;

  // 250 MHz input -> 1000 MHz VCO -> phase-aligned 100/200 MHz clocks.
  // Both clocks MUST remain related: TMEM's pumped RAMs rely on their phase.
  MMCME4_ADV #(.CLKIN1_PERIOD(4.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(4.0),
               .CLKOUT0_DIVIDE_F(10.0), .CLKOUT1_DIVIDE(5), .STARTUP_WAIT("FALSE")) u_mmcm (
    .CLKIN1(clk_main_a0), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clk_fb_buf),
    .CLKFBOUT(clk_fb), .CLKOUT0(clk_raw), .CLKOUT1(clk_f_raw), .LOCKED(mmcm_locked),
    .RST(clock_reset), .PWRDWN(1'b0),
    .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DWE(1'b0),
    .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .CDDCREQ(1'b0));
  BUFG u_fb (.I(clk_fb), .O(clk_fb_buf));
  BUFG u_clk (.I(clk_raw), .O(clk));
  BUFG u_clk_f (.I(clk_f_raw), .O(clk_f));
  always_ff @(posedge clk_main_a0 or posedge clock_reset)
    if (clock_reset) main_reset_n <= '0;
    else main_reset_n <= {main_reset_n[1:0], 1'b1};
  always_ff @(posedge clk_main_a0 or posedge bridge_reset)
    if (bridge_reset) bridge_reset_n <= '0;
    else bridge_reset_n <= {bridge_reset_n[1:0], 1'b1};
  always_ff @(posedge clk or posedge bridge_reset)
    if (bridge_reset) core_reset_n <= '0;
    else core_reset_n <= {core_reset_n[1:0], 1'b1};
  // Resolve the status logic in its source clock before the LED synchronizers.
  always_ff @(posedge clk)
    if (rst) board_led_core <= '0;
    else board_led_core <= board_led;
  always_ff @(posedge clk_main_a0) begin
    led_s1 <= board_led_core;
    led_s2 <= led_s1;
  end

  // DDR calibration survives core reset; shell reset resets the whole design.
  logic [15:0]  ddr_awid;
  logic [63:0]  ddr_awaddr;
  logic [7:0]   ddr_awlen;
  logic [2:0]   ddr_awsize;
  logic [1:0]   ddr_awburst;
  logic         ddr_awvalid;
  logic         ddr_awready;
  logic [511:0] ddr_wdata;
  logic [63:0]  ddr_wstrb;
  logic         ddr_wlast;
  logic         ddr_wvalid;
  logic         ddr_wready;
  logic [15:0]  ddr_bid;
  logic [1:0]   ddr_bresp;
  logic         ddr_bvalid;
  logic         ddr_bready;
  logic [15:0]  ddr_arid;
  logic [63:0]  ddr_araddr;
  logic [7:0]   ddr_arlen;
  logic [2:0]   ddr_arsize;
  logic [1:0]   ddr_arburst;
  logic         ddr_arvalid;
  logic         ddr_arready;
  logic [15:0]  ddr_rid;
  logic [511:0] ddr_rdata;
  logic [1:0]   ddr_rresp;
  logic         ddr_rlast;
  logic         ddr_rvalid;
  logic         ddr_rready;

  otpu_f2_sys #(
    `include "opentpu_params.vh"
  ) u_sys (
    .clk(clk),
    .clk_f(clk_f),
    .rst(rst),
    .aclk(clk_main_a0),
    .arst(arst),
    .ddr_ready(ddr_ready),
    .led(board_led),
    .ocl_awaddr(ocl_cl_awaddr),
    .ocl_awvalid(ocl_cl_awvalid),
    .ocl_awready(cl_ocl_awready),
    .ocl_wdata(ocl_cl_wdata),
    .ocl_wstrb(ocl_cl_wstrb),
    .ocl_wvalid(ocl_cl_wvalid),
    .ocl_wready(cl_ocl_wready),
    .ocl_bresp(cl_ocl_bresp),
    .ocl_bvalid(cl_ocl_bvalid),
    .ocl_bready(ocl_cl_bready),
    .ocl_araddr(ocl_cl_araddr),
    .ocl_arvalid(ocl_cl_arvalid),
    .ocl_arready(cl_ocl_arready),
    .ocl_rdata(cl_ocl_rdata),
    .ocl_rresp(cl_ocl_rresp),
    .ocl_rvalid(cl_ocl_rvalid),
    .ocl_rready(ocl_cl_rready),
    .pcis_awid(sh_cl_dma_pcis_awid),
    .pcis_awaddr(sh_cl_dma_pcis_awaddr),
    .pcis_awlen(sh_cl_dma_pcis_awlen),
    .pcis_awsize(sh_cl_dma_pcis_awsize),
    .pcis_awburst(sh_cl_dma_pcis_awburst),
    .pcis_awvalid(sh_cl_dma_pcis_awvalid),
    .pcis_awready(cl_sh_dma_pcis_awready),
    .pcis_wdata(sh_cl_dma_pcis_wdata),
    .pcis_wstrb(sh_cl_dma_pcis_wstrb),
    .pcis_wlast(sh_cl_dma_pcis_wlast),
    .pcis_wvalid(sh_cl_dma_pcis_wvalid),
    .pcis_wready(cl_sh_dma_pcis_wready),
    .pcis_bid(cl_sh_dma_pcis_bid),
    .pcis_bresp(cl_sh_dma_pcis_bresp),
    .pcis_bvalid(cl_sh_dma_pcis_bvalid),
    .pcis_bready(sh_cl_dma_pcis_bready),
    .pcis_arid(sh_cl_dma_pcis_arid),
    .pcis_araddr(sh_cl_dma_pcis_araddr),
    .pcis_arlen(sh_cl_dma_pcis_arlen),
    .pcis_arsize(sh_cl_dma_pcis_arsize),
    .pcis_arburst(sh_cl_dma_pcis_arburst),
    .pcis_arvalid(sh_cl_dma_pcis_arvalid),
    .pcis_arready(cl_sh_dma_pcis_arready),
    .pcis_rid(cl_sh_dma_pcis_rid),
    .pcis_rdata(cl_sh_dma_pcis_rdata),
    .pcis_rresp(cl_sh_dma_pcis_rresp),
    .pcis_rlast(cl_sh_dma_pcis_rlast),
    .pcis_rvalid(cl_sh_dma_pcis_rvalid),
    .pcis_rready(sh_cl_dma_pcis_rready),
    .ddr_awid(ddr_awid),
    .ddr_awaddr(ddr_awaddr),
    .ddr_awlen(ddr_awlen),
    .ddr_awsize(ddr_awsize),
    .ddr_awburst(ddr_awburst),
    .ddr_awvalid(ddr_awvalid),
    .ddr_awready(ddr_awready),
    .ddr_wdata(ddr_wdata),
    .ddr_wstrb(ddr_wstrb),
    .ddr_wlast(ddr_wlast),
    .ddr_wvalid(ddr_wvalid),
    .ddr_wready(ddr_wready),
    .ddr_bid(ddr_bid),
    .ddr_bresp(ddr_bresp),
    .ddr_bvalid(ddr_bvalid),
    .ddr_bready(ddr_bready),
    .ddr_arid(ddr_arid),
    .ddr_araddr(ddr_araddr),
    .ddr_arlen(ddr_arlen),
    .ddr_arsize(ddr_arsize),
    .ddr_arburst(ddr_arburst),
    .ddr_arvalid(ddr_arvalid),
    .ddr_arready(ddr_arready),
    .ddr_rid(ddr_rid),
    .ddr_rdata(ddr_rdata),
    .ddr_rresp(ddr_rresp),
    .ddr_rlast(ddr_rlast),
    .ddr_rvalid(ddr_rvalid),
    .ddr_rready(ddr_rready)
  );

  // Match the HDK's eight-stage management bus pipelines for DDR status.
  logic [7:0] stat_addr, stat_irq;
  logic [31:0] stat_wdata, stat_rdata;
  logic stat_wr, stat_rd, stat_ack;
  lib_pipe #(.WIDTH(42), .STAGES(8)) u_stat_req (
    .clk(clk_main_a0), .rst_n(main_reset_n[2]),
    .in_bus({sh_cl_ddr_stat_wr, sh_cl_ddr_stat_rd, sh_cl_ddr_stat_addr, sh_cl_ddr_stat_wdata}),
    .out_bus({stat_wr, stat_rd, stat_addr, stat_wdata}));
  lib_pipe #(.WIDTH(41), .STAGES(8)) u_stat_rsp (
    .clk(clk_main_a0), .rst_n(main_reset_n[2]),
    .in_bus({stat_ack, stat_irq, stat_rdata}),
    .out_bus({cl_sh_ddr_stat_ack, cl_sh_ddr_stat_int, cl_sh_ddr_stat_rdata}));

  sh_ddr #(.DDR_PRESENT(1)) SH_DDR (
    .clk(clk_main_a0), .rst_n(main_reset_n[2]),
    .stat_clk(clk_main_a0), .stat_rst_n(main_reset_n[2]),
    .CLK_DIMM_DP(CLK_DIMM_DP),
    .CLK_DIMM_DN(CLK_DIMM_DN),
    .M_ACT_N(M_ACT_N),
    .M_MA(M_MA),
    .M_BA(M_BA),
    .M_BG(M_BG),
    .M_CKE(M_CKE),
    .M_ODT(M_ODT),
    .M_CS_N(M_CS_N),
    .M_CLK_DN(M_CLK_DN),
    .M_CLK_DP(M_CLK_DP),
    .M_PAR(M_PAR),
    .M_DQ(M_DQ),
    .M_ECC(M_ECC),
    .M_DQS_DP(M_DQS_DP),
    .M_DQS_DN(M_DQS_DN),
    .cl_RST_DIMM_N(RST_DIMM_N),
    .cl_sh_ddr_axi_awid(ddr_awid),
    .cl_sh_ddr_axi_awaddr(ddr_awaddr),
    .cl_sh_ddr_axi_awlen(ddr_awlen),
    .cl_sh_ddr_axi_awsize(ddr_awsize),
    .cl_sh_ddr_axi_awburst(ddr_awburst),
    .cl_sh_ddr_axi_awvalid(ddr_awvalid),
    .cl_sh_ddr_axi_awready(ddr_awready),
    .cl_sh_ddr_axi_wdata(ddr_wdata),
    .cl_sh_ddr_axi_wstrb(ddr_wstrb),
    .cl_sh_ddr_axi_wlast(ddr_wlast),
    .cl_sh_ddr_axi_wvalid(ddr_wvalid),
    .cl_sh_ddr_axi_wready(ddr_wready),
    .cl_sh_ddr_axi_bid(ddr_bid),
    .cl_sh_ddr_axi_bresp(ddr_bresp),
    .cl_sh_ddr_axi_bvalid(ddr_bvalid),
    .cl_sh_ddr_axi_bready(ddr_bready),
    .cl_sh_ddr_axi_arid(ddr_arid),
    .cl_sh_ddr_axi_araddr(ddr_araddr),
    .cl_sh_ddr_axi_arlen(ddr_arlen),
    .cl_sh_ddr_axi_arsize(ddr_arsize),
    .cl_sh_ddr_axi_arburst(ddr_arburst),
    .cl_sh_ddr_axi_arvalid(ddr_arvalid),
    .cl_sh_ddr_axi_arready(ddr_arready),
    .cl_sh_ddr_axi_rid(ddr_rid),
    .cl_sh_ddr_axi_rdata(ddr_rdata),
    .cl_sh_ddr_axi_rresp(ddr_rresp),
    .cl_sh_ddr_axi_rlast(ddr_rlast),
    .cl_sh_ddr_axi_rvalid(ddr_rvalid),
    .cl_sh_ddr_axi_rready(ddr_rready),
    .cl_sh_ddr_axi_awuser(1'b0),
    .cl_sh_ddr_axi_aruser(1'b0),
    .sh_ddr_stat_bus_addr(stat_addr),
    .sh_ddr_stat_bus_wdata(stat_wdata),
    .sh_ddr_stat_bus_wr(stat_wr),
    .sh_ddr_stat_bus_rd(stat_rd),
    .sh_ddr_stat_bus_ack(stat_ack),
    .sh_ddr_stat_bus_rdata(stat_rdata),
    .ddr_sh_stat_int(stat_irq),
    .sh_cl_ddr_is_ready(ddr_ready)
  );

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_flr_done = 1'b1;
  assign cl_sh_status0 = 32'h4f545055; // OTPU
  assign cl_sh_status1 = 32'd0;
  assign cl_sh_status2 = 32'd0;
  // vLED[0] MMCM locked, [1] DDR calibrated, [4:2] board status.
  assign cl_sh_status_vled = {11'd0, led_s2, ddr_ready, mmcm_locked};

  // Unused shell interfaces: no outbound PCIe DMA, SDA, HBM, interrupts or JTAG.
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
  assign hbm_apb_paddr_1 = '0;
  assign hbm_apb_pprot_1 = '0;
  assign hbm_apb_psel_1 = '0;
  assign hbm_apb_penable_1 = '0;
  assign hbm_apb_pwrite_1 = '0;
  assign hbm_apb_pwdata_1 = '0;
  assign hbm_apb_pstrb_1 = '0;
  assign hbm_apb_pready_1 = '0;
  assign hbm_apb_prdata_1 = '0;
  assign hbm_apb_pslverr_1 = '0;
  assign hbm_apb_paddr_0 = '0;
  assign hbm_apb_pprot_0 = '0;
  assign hbm_apb_psel_0 = '0;
  assign hbm_apb_penable_0 = '0;
  assign hbm_apb_pwrite_0 = '0;
  assign hbm_apb_pwdata_0 = '0;
  assign hbm_apb_pstrb_0 = '0;
  assign hbm_apb_pready_0 = '0;
  assign hbm_apb_prdata_0 = '0;
  assign hbm_apb_pslverr_0 = '0;
  assign PCIE_EP_TXP = '0;
  assign PCIE_EP_TXN = '0;
  assign PCIE_RP_PERSTN = '0;
  assign PCIE_RP_TXP = '0;
  assign PCIE_RP_TXN = '0;
endmodule
