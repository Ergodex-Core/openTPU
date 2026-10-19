// The accelerator on an AWS F2 slot with the HBM (cl_opentpu_hbm, boards/f2/design): otpu_board
// with S slices of NCH channels each, channel c of slice s on HBM AXI port (and pseudo-channel)
// p = NCH s + c through one otpu_native_axi each (BASE p x 512 MiB), the host's PCIS onto every
// port through otpu_f2_hbm_host, the two masters of each port shared by an otpu_f2_hbm_port, and
// the shell's OCL onto the control registers through otpu_axil_cdc (held while the host's PCIS
// writes are in flight). Clocks: the core on clk (and clk_f, 2x, the pumped TMEM), the OCL and
// PCIS on aclk (clk_main_a0, 250 MHz). Nothing here runs on the HBM's 450 MHz clock: the ports
// leave this module as AXI4 masters of 512 bits at the core clock (m_*), and the CL top puts
// the HDK's cl_axi_sc_1x1 SmartConnect (clock, width and protocol conversion) between each and
// its HBM port (docs/f2_hbm.md, docs/v4/f2-round5.md).
//   hbm_ready: cl_mem_hbm_wrapper's o_hbm_ready (aclk): every channel's STATUS CALIB bits.
// The parameters are the target's (targets/f2-hbm.toml through opentpu.target rtl_params).
module otpu_f2_hbm_sys #(
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
  parameter int CORE_KHZ   = 200000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS    = 0,
  parameter int PLATFORM   = 3,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  parameter int S          = 1,          // slices (S x NCH <= 32 HBM ports)
  parameter int NCH        = 8,          // channels (HBM ports, pseudo-channels) per slice
  parameter int CH_LOG2    = 29,         // a pseudo-channel's 512 MiB
  parameter int DRAM_LOG2  = 32          // what a slice addresses (otpu_board; CFG_DRAM)
) (
  input  logic          clk,
  input  logic          clk_f,
  input  logic          rst,
  input  logic          aclk,
  input  logic          arst,
  input  logic          hbm_ready,
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
  // ---- the ports' AXI4 masters 0 .. S NCH - 1 (core clock; 512 bits, 34-bit addresses,
  //      INCR bursts of 64-byte beats, IDs in bit 0): into the SmartConnects (the CL top)
  output logic [S*NCH-1:0][15:0]  m_awid,
  output logic [S*NCH-1:0][33:0]  m_awaddr,
  output logic [S*NCH-1:0][7:0]   m_awlen,
  output logic [S*NCH-1:0]        m_awvalid,
  input  logic [S*NCH-1:0]        m_awready,
  output logic [S*NCH-1:0][511:0] m_wdata,
  output logic [S*NCH-1:0][63:0]  m_wstrb,
  output logic [S*NCH-1:0]        m_wlast,
  output logic [S*NCH-1:0]        m_wvalid,
  input  logic [S*NCH-1:0]        m_wready,
  input  logic [S*NCH-1:0][15:0]  m_bid,
  input  logic [S*NCH-1:0][1:0]   m_bresp,
  input  logic [S*NCH-1:0]        m_bvalid,
  output logic [S*NCH-1:0]        m_bready,
  output logic [S*NCH-1:0][15:0]  m_arid,
  output logic [S*NCH-1:0][33:0]  m_araddr,
  output logic [S*NCH-1:0][7:0]   m_arlen,
  output logic [S*NCH-1:0]        m_arvalid,
  input  logic [S*NCH-1:0]        m_arready,
  input  logic [S*NCH-1:0][15:0]  m_rid,
  input  logic [S*NCH-1:0][511:0] m_rdata,
  input  logic [S*NCH-1:0][1:0]   m_rresp,
  input  logic [S*NCH-1:0]        m_rlast,
  input  logic [S*NCH-1:0]        m_rvalid,
  output logic [S*NCH-1:0]        m_rready
);
  localparam int NPORT = S * NCH;
  initial if (NPORT > 32 || CH_LOG2 != 29)
    $fatal(1, "otpu_f2_hbm_sys: at most 32 HBM ports of 512 MiB (S x NCH <= 32, CH_LOG2 29)");
  // HBM ready, into the core clock for the calibration bits (otpu_board synchronizes it again)
  logic hbm_rdy;
  assign hbm_rdy = hbm_ready;

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

  // ---- the slices' channels: a bridge each
  logic [NPORT-1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NPORT-1:0][24:0]  n_caddr;
  logic [NPORT-1:0][511:0] n_wdata, n_rdata;
  logic [NPORT-1:0][63:0]  n_wmask;
  logic [NPORT-1:0][15:0]  n_wdone;
  // the slices' channels, the host, the HBM ports
  otpu_f2_hbm_mem #(.NPORT(NPORT), .CH_LOG2(CH_LOG2)) u_mem (
    .clk, .rst, .aclk, .arst,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid,
    .n_rdata, .n_wdone, .n_err,
    .pcis_awid, .pcis_awaddr, .pcis_awlen, .pcis_awsize, .pcis_awburst, .pcis_awvalid,
    .pcis_awready, .pcis_wdata, .pcis_wstrb, .pcis_wlast, .pcis_wvalid, .pcis_wready, .pcis_bid,
    .pcis_bresp, .pcis_bvalid, .pcis_bready, .pcis_arid, .pcis_araddr, .pcis_arlen, .pcis_arsize,
    .pcis_arburst, .pcis_arvalid, .pcis_arready, .pcis_rid, .pcis_rdata, .pcis_rresp,
    .pcis_rlast, .pcis_rvalid, .pcis_rready, .pcis_wbusy,
    .m_awid, .m_awaddr, .m_awlen, .m_awvalid, .m_awready, .m_wdata, .m_wstrb, .m_wlast,
    .m_wvalid, .m_wready, .m_bid, .m_bresp, .m_bvalid, .m_bready, .m_arid, .m_araddr, .m_arlen,
    .m_arvalid, .m_arready, .m_rid, .m_rdata, .m_rresp, .m_rlast, .m_rvalid, .m_rready);

  // ---- the accelerator (HAS_I2C 0: no I2C on the slot; temp 0: the shell owns the SYSMON)
  logic [3:0] i2c_lo;
  otpu_board #(.S(S), .NCH(NCH), .WDQ(1'b1), .CH_LOG2(CH_LOG2), .DRAM_LOG2(DRAM_LOG2), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL), .ULANES(ULANES),
               .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS),
               .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_I2C(1'b0), .CHASH(CHASH),
               .DSTEP(DSTEP), .HOSTCAL(1'b0), .SEQ_WC(SEQ_WC), .PLATFORM(PLATFORM),
               .ARCH_HASH(ARCH_HASH), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
               .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
               .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM)) u_board (
    .clk, .clk_f, .rst, .ctl_mrst(rst),
    .calib({NPORT{hbm_rdy}}), .ded('0), .temp(12'd0), .led, .i2c_lo, .i2c_pin(5'h1f),
    .s_ctl_awaddr(c_awaddr), .s_ctl_awvalid(c_awvalid), .s_ctl_awready(c_awready),
    .s_ctl_wdata(c_wdata), .s_ctl_wstrb(c_wstrb), .s_ctl_wvalid(c_wvalid),
    .s_ctl_wready(c_wready), .s_ctl_bresp(c_bresp), .s_ctl_bvalid(c_bvalid),
    .s_ctl_bready(c_bready), .s_ctl_araddr(c_araddr), .s_ctl_arvalid(c_arvalid),
    .s_ctl_arready(c_arready), .s_ctl_rdata(c_rdata), .s_ctl_rresp(c_rresp),
    .s_ctl_rvalid(c_rvalid), .s_ctl_rready(c_rready),
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_err);
endmodule
