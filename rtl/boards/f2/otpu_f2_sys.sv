// The accelerator on an AWS F2 slot (cl_opentpu, boards/f2/design: the shell's ports, the clocks,
// sh_ddr): otpu_board with one slice, its two memory channels on the slot's DIMM through one
// otpu_f2_nch each (channel c at DIMM byte c x 2 GiB), the host's PCIS on the same DIMM, the three
// shared by otpu_f2_mem, and the shell's OCL onto the control registers through otpu_axil_cdc.
// The core runs on clk (the CL's MMCM), the bridges' AXI side, the OCL, the PCIS and the DIMM on
// aclk (clk_main_a0, 250 MHz). docs/f2.md.
//   ddr_ready: sh_ddr's sh_cl_ddr_is_ready (aclk): both channels' STATUS CALIB bits.
// The parameters are the target's (targets/f2.toml through opentpu.target rtl_params; the defaults
// are its values: tests/test_f2_build.py).
module otpu_f2_sys #(
  parameter int D          = 128,
  parameter int MCOLS      = 4,
  parameter int ACT_ROWS   = 16,
  parameter int ACT_BLOCKS = 128,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int RPB        = 8 * LANES,
  parameter int WPB        = 1,
  parameter int TMEM_IMPL  = 3,
  parameter bit TMEM_RQ    = 1'b0,
  parameter int TMEM_NSB   = 2,
  parameter int TMEM_SBP   = 3,
  parameter int TMEM_SBH   = 0,
  parameter bit TMEM_RPL   = 1'b1,
  parameter bit TMEM_TDP   = 1'b1,
  parameter bit TMEM_GST   = 1'b1,
  parameter int TMEM_PMP   = 1,
  parameter int TMEM_PRK   = 0,
  parameter int TMEM_SDM   = -1,
  parameter int TMEM_WARB  = 0,       // the VPU / MXU-drain write arbitration (otpu_slice)
  parameter bit MXROT      = 1'b0,    // the MXU's TMEM ports as rotators (otpu_slice; CAPS2 bit7)
  parameter int ACT_RAM    = 0,       // the ACT RAM's RAM style (otpu_slice: 0 auto, 1 block, 2 ultra)
  parameter int TMEM_RAM   = 0,       // TMEM_IMPL 3's RAM style (otpu_slice: 0 / 1 block, 2 ultra)
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 4,
  parameter int VPU_CL     = 2,
  parameter int ULANES     = 8,
  parameter int TRACE_DEPTH = 16384,
  parameter int PQ_WIN     = 1024,
  parameter bit CHASH      = 1'b1,
  parameter bit DSTEP      = 1'b1,
  parameter bit SEQ_WC     = 1'b1,
  parameter bit T2         = 1'b0,       // t2 weights, MM QUAD, QACT DUP4 (docs/ternary.md)
  parameter bit PACK       = 1'b0,       // the MXU packs two int8 weight rows a DSP (docs/mxu_packing.md)
  parameter int CORE_KHZ   = 100000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS    = 2133,
  parameter int PLATFORM   = 3,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  parameter logic [63:0] CH_BYTES = 64'h8000_0000   // a channel's window of the DIMM
) (
  input  logic          clk,
  input  logic          clk_f,
  input  logic          rst,
  input  logic          aclk,
  input  logic          arst,
  input  logic          ddr_ready,
  output logic [2:0]    led,
  // ---- the shell's OCL (AppPF BAR0), aclk
  input  logic [31:0]   ocl_awaddr,
  input  logic          ocl_awvalid,
  output logic          ocl_awready,
  input  logic [31:0]   ocl_wdata,
  input  logic [3:0]    ocl_wstrb,
  input  logic          ocl_wvalid,
  output logic          ocl_wready,
  output logic [1:0]    ocl_bresp,
  output logic          ocl_bvalid,
  input  logic          ocl_bready,
  input  logic [31:0]   ocl_araddr,
  input  logic          ocl_arvalid,
  output logic          ocl_arready,
  output logic [31:0]   ocl_rdata,
  output logic [1:0]    ocl_rresp,
  output logic          ocl_rvalid,
  input  logic          ocl_rready,
  // ---- the shell's PCIS (AppPF BAR4), aclk
  input  logic [15:0]   pcis_awid,
  input  logic [63:0]   pcis_awaddr,
  input  logic [7:0]    pcis_awlen,
  input  logic [2:0]    pcis_awsize,
  input  logic [1:0]    pcis_awburst,
  input  logic          pcis_awvalid,
  output logic          pcis_awready,
  input  logic [511:0]  pcis_wdata,
  input  logic [63:0]   pcis_wstrb,
  input  logic          pcis_wlast,
  input  logic          pcis_wvalid,
  output logic          pcis_wready,
  output logic [15:0]   pcis_bid,
  output logic [1:0]    pcis_bresp,
  output logic          pcis_bvalid,
  input  logic          pcis_bready,
  input  logic [15:0]   pcis_arid,
  input  logic [63:0]   pcis_araddr,
  input  logic [7:0]    pcis_arlen,
  input  logic [2:0]    pcis_arsize,
  input  logic [1:0]    pcis_arburst,
  input  logic          pcis_arvalid,
  output logic          pcis_arready,
  output logic [15:0]   pcis_rid,
  output logic [511:0]  pcis_rdata,
  output logic [1:0]    pcis_rresp,
  output logic          pcis_rlast,
  output logic          pcis_rvalid,
  input  logic          pcis_rready,
  // ---- the DIMM (sh_ddr), aclk
  output logic [15:0]   ddr_awid,
  output logic [63:0]   ddr_awaddr,
  output logic [7:0]    ddr_awlen,
  output logic [2:0]    ddr_awsize,
  output logic [1:0]    ddr_awburst,
  output logic          ddr_awvalid,
  input  logic          ddr_awready,
  output logic [511:0]  ddr_wdata,
  output logic [63:0]   ddr_wstrb,
  output logic          ddr_wlast,
  output logic          ddr_wvalid,
  input  logic          ddr_wready,
  input  logic [15:0]   ddr_bid,
  input  logic [1:0]    ddr_bresp,
  input  logic          ddr_bvalid,
  output logic          ddr_bready,
  output logic [15:0]   ddr_arid,
  output logic [63:0]   ddr_araddr,
  output logic [7:0]    ddr_arlen,
  output logic [2:0]    ddr_arsize,
  output logic [1:0]    ddr_arburst,
  output logic          ddr_arvalid,
  input  logic          ddr_arready,
  input  logic [15:0]   ddr_rid,
  input  logic [511:0]  ddr_rdata,
  input  logic [1:0]    ddr_rresp,
  input  logic          ddr_rlast,
  input  logic          ddr_rvalid,
  output logic          ddr_rready
);
  // ---- control registers: the OCL across into clk
  logic [11:0] c_awaddr, c_araddr;
  logic        c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic        c_arvalid, c_arready, c_rvalid, c_rready;
  logic [31:0] c_wdata, c_rdata;
  logic [3:0]  c_wstrb;
  logic [1:0]  c_bresp, c_rresp;
  logic        pcis_wbusy;
  otpu_axil_cdc u_ocl (
    .sclk(aclk), .srst(arst), .hold(pcis_wbusy),
    .s_awaddr(ocl_awaddr), .s_awvalid(ocl_awvalid), .s_awready(ocl_awready),
    .s_wdata(ocl_wdata), .s_wstrb(ocl_wstrb), .s_wvalid(ocl_wvalid), .s_wready(ocl_wready),
    .s_bresp(ocl_bresp), .s_bvalid(ocl_bvalid), .s_bready(ocl_bready),
    .s_araddr(ocl_araddr), .s_arvalid(ocl_arvalid), .s_arready(ocl_arready),
    .s_rdata(ocl_rdata), .s_rresp(ocl_rresp), .s_rvalid(ocl_rvalid), .s_rready(ocl_rready),
    .mclk(clk), .mrst(rst),
    .m_awaddr(c_awaddr), .m_awvalid(c_awvalid), .m_awready(c_awready),
    .m_wdata(c_wdata), .m_wstrb(c_wstrb), .m_wvalid(c_wvalid), .m_wready(c_wready),
    .m_bresp(c_bresp), .m_bvalid(c_bvalid), .m_bready(c_bready),
    .m_araddr(c_araddr), .m_arvalid(c_arvalid), .m_arready(c_arready),
    .m_rdata(c_rdata), .m_rresp(c_rresp), .m_rvalid(c_rvalid), .m_rready(c_rready));

  // ---- the slice's two channels
  logic [1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [1:0][24:0]  n_caddr;
  logic [1:0][511:0] n_wdata, n_rdata;
  logic [1:0][63:0]  n_wmask;
  logic [1:0][15:0]  n_wdone;
  logic [1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
  logic [1:0]        m_arvalid, m_arready, m_rvalid, m_rready;
  logic [1:0][63:0]  m_awaddr, m_araddr, m_wstrb;
  logic [1:0][511:0] m_wdata;
  logic [511:0]      m_rdata;
  logic [1:0]        m_bresp, m_rresp;
  for (genvar c = 0; c < 2; c++) begin : g_ch
    otpu_f2_nch #(.BASE(CH_BYTES * c)) u_nch (
      .clk, .rst,
      .n_cvalid(n_cvalid[c]), .n_cready(n_cready[c]), .n_cwe(n_cwe[c]), .n_caddr(n_caddr[c]),
      .n_wvalid(n_wvalid[c]), .n_wready(n_wready[c]), .n_wdata(n_wdata[c]),
      .n_wmask(n_wmask[c]), .n_rvalid(n_rvalid[c]), .n_rdata(n_rdata[c]),
      .n_wdone(n_wdone[c]), .n_err(n_err[c]),
      .aclk, .arst,
      .m_awvalid(m_awvalid[c]), .m_awready(m_awready[c]), .m_awaddr(m_awaddr[c]),
      .m_wvalid(m_wvalid[c]), .m_wready(m_wready[c]), .m_wdata(m_wdata[c]),
      .m_wstrb(m_wstrb[c]), .m_bvalid(m_bvalid[c]), .m_bready(m_bready[c]), .m_bresp(m_bresp),
      .m_arvalid(m_arvalid[c]), .m_arready(m_arready[c]), .m_araddr(m_araddr[c]),
      .m_rvalid(m_rvalid[c]), .m_rready(m_rready[c]), .m_rdata(m_rdata), .m_rresp(m_rresp));
  end

  otpu_f2_mem u_mem (
    .aclk, .arst,
    .c_awvalid(m_awvalid), .c_awready(m_awready), .c_awaddr(m_awaddr),
    .c_wvalid(m_wvalid), .c_wready(m_wready), .c_wdata(m_wdata), .c_wstrb(m_wstrb),
    .c_bvalid(m_bvalid), .c_bready(m_bready), .c_bresp(m_bresp),
    .c_arvalid(m_arvalid), .c_arready(m_arready), .c_araddr(m_araddr),
    .c_rvalid(m_rvalid), .c_rready(m_rready), .c_rdata(m_rdata), .c_rresp(m_rresp),
    .h_awid(pcis_awid), .h_awaddr(pcis_awaddr), .h_awlen(pcis_awlen), .h_awsize(pcis_awsize),
    .h_awburst(pcis_awburst), .h_awvalid(pcis_awvalid), .h_awready(pcis_awready),
    .h_wdata(pcis_wdata), .h_wstrb(pcis_wstrb), .h_wlast(pcis_wlast), .h_wvalid(pcis_wvalid),
    .h_wready(pcis_wready), .h_bid(pcis_bid), .h_bresp(pcis_bresp), .h_bvalid(pcis_bvalid),
    .h_bready(pcis_bready),
    .h_arid(pcis_arid), .h_araddr(pcis_araddr), .h_arlen(pcis_arlen), .h_arsize(pcis_arsize),
    .h_arburst(pcis_arburst), .h_arvalid(pcis_arvalid), .h_arready(pcis_arready),
    .h_rid(pcis_rid), .h_rdata(pcis_rdata), .h_rresp(pcis_rresp), .h_rlast(pcis_rlast),
    .h_rvalid(pcis_rvalid), .h_rready(pcis_rready),
    .pcis_wbusy,
    .d_awid(ddr_awid), .d_awaddr(ddr_awaddr), .d_awlen(ddr_awlen), .d_awsize(ddr_awsize),
    .d_awburst(ddr_awburst), .d_awvalid(ddr_awvalid), .d_awready(ddr_awready),
    .d_wdata(ddr_wdata), .d_wstrb(ddr_wstrb), .d_wlast(ddr_wlast), .d_wvalid(ddr_wvalid),
    .d_wready(ddr_wready), .d_bid(ddr_bid), .d_bresp(ddr_bresp), .d_bvalid(ddr_bvalid),
    .d_bready(ddr_bready),
    .d_arid(ddr_arid), .d_araddr(ddr_araddr), .d_arlen(ddr_arlen), .d_arsize(ddr_arsize),
    .d_arburst(ddr_arburst), .d_arvalid(ddr_arvalid), .d_arready(ddr_arready),
    .d_rid(ddr_rid), .d_rdata(ddr_rdata), .d_rresp(ddr_rresp), .d_rlast(ddr_rlast),
    .d_rvalid(ddr_rvalid), .d_rready(ddr_rready));

  // ---- the accelerator (HAS_I2C 0: no I2C on the slot; temp 0: the shell owns the SYSMON)
  logic [3:0] i2c_lo;
  otpu_board #(.S(1), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL), .ULANES(ULANES),
               .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS),
               .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_I2C(1'b0), .CHASH(CHASH),
               .DSTEP(DSTEP), .HOSTCAL(1'b0), .SEQ_WC(SEQ_WC), .T2(T2), .PACK(PACK), .PLATFORM(PLATFORM),
               .ARCH_HASH(ARCH_HASH), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
               .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
               .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM), .TMEM_WARB(TMEM_WARB), .MXROT(MXROT), .ACT_RAM(ACT_RAM), .TMEM_RAM(TMEM_RAM)) u_board (
    .clk, .clk_f, .rst, .ctl_mrst(rst),
    .calib({2{ddr_ready}}), .ded(2'b00), .temp(12'd0), .led, .i2c_lo, .i2c_pin(5'h1f),
    .s_ctl_awaddr(c_awaddr), .s_ctl_awvalid(c_awvalid), .s_ctl_awready(c_awready),
    .s_ctl_wdata(c_wdata), .s_ctl_wstrb(c_wstrb), .s_ctl_wvalid(c_wvalid),
    .s_ctl_wready(c_wready), .s_ctl_bresp(c_bresp), .s_ctl_bvalid(c_bvalid),
    .s_ctl_bready(c_bready), .s_ctl_araddr(c_araddr), .s_ctl_arvalid(c_arvalid),
    .s_ctl_arready(c_arready), .s_ctl_rdata(c_rdata), .s_ctl_rresp(c_rresp),
    .s_ctl_rvalid(c_rvalid), .s_ctl_rready(c_rready),
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_err);
endmodule
