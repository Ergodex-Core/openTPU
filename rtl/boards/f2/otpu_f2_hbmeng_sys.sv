// The accelerator on an AWS F2 slot with the HBM, v5's memory path (cl_opentpu_hbm5, boards/f2/
// design; docs/full_hbm.md, docs/superpowers/specs/2026-10-20-opentpu-design-v5.md): otpu_board
// with S slices of NCH channels each, channel c of slice s on HBM AXI port (and pseudo-channel)
// p = NCH s + c through its own 450 MHz engine (otpu_f2_hbm_eng u_mem: the SLR chains, the
// per-slice write bus, otpu_hbm_ch per port, whose AXI3 256-bit master p_* at hclk goes through
// the HDK's cl_axi3_256b_reg_slice onto cl_mem_hbm_wrapper port p); the host's PCIS through
// otpu_f2_pcis_gate to the fabric's host branch (h_*, aclk: AWS's cl_axi_sc_1x1 as shipped) and
// from there onto its own port NPORT (HOST_MUX 0: S x NCH <= 31) or, HOST_MUX 1 (S x NCH = 32),
// back in as an AXI3 master at hclk (hs_*) muxed onto port NPORT - 1 with that channel's engine
// (otpu_hbm_mux2); the shell's OCL onto the control registers through otpu_axil_cdc (held while
// the host's PCIS writes are in flight). A sibling of otpu_f2_hbm_sys (the SmartConnect path,
// unchanged for the other targets): the same control, board and host-gate structure, the
// memory side and the hclk ports differ. Clocks: the core on clk (and clk_f, 2x, the pumped
// TMEM), the OCL and PCIS on aclk (clk_main_a0, 250 MHz), the engines' port sides and the mux
// on hclk (450 MHz). The parameters are the target's (targets/f2-hbm.toml through
// opentpu.target rtl_params); NRD the adapter's reads in flight a channel (128 on v5).
module otpu_f2_hbmeng_sys #(
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
  parameter bit T2         = 1'b0,
  parameter int PORTB      = 1,
  parameter int RL         = 0,
  parameter int ALINE      = 32,
  parameter int CORE_KHZ   = 200000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS    = 0,
  parameter int PLATFORM   = 3,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  parameter int S          = 1,          // slices (S x NCH <= 32 HBM ports)
  parameter int NCH        = 8,          // channels (HBM ports, pseudo-channels) per slice
  parameter int CH_LOG2    = 29,         // a pseudo-channel's 512 MiB
  parameter int DRAM_LOG2  = 32,         // what a slice addresses (otpu_board; CFG_DRAM)
  parameter int NXR        = 3,          // SLR crossing stages each way (0: wired)
  parameter int NRD        = 128,        // the adapter's B read beats in flight a channel
  parameter int BURST      = 8,          // otpu_hbm_ch: beats a read burst at most (512 B)
  parameter int IDLE       = 2,          //   hclk cycles a burst waits with an empty queue
  parameter int OT         = 64,         //   read bursts in flight a port
  parameter int RDEPTH     = 256,        //   the read FIFO's beats
  parameter bit HOST_MUX   = 1'b0        // the host on port NPORT - 1 through otpu_hbm_mux2
) (
  input  logic          clk,
  input  logic          clk_f,
  input  logic          rst,
  input  logic          aclk,
  input  logic          arst,
  input  logic          hclk,
  input  logic          hrst,
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
  // ---- the host branch towards cl_axi_sc_1x1 (aclk; AXI4 512 bits)
  output logic [15:0]   h_awid,
  output logic [63:0]   h_awaddr,
  output logic [7:0]    h_awlen,
  output logic [2:0]    h_awsize,
  output logic [1:0]    h_awburst,
  output logic          h_awvalid,
  input  logic          h_awready,
  output logic [511:0]  h_wdata,
  output logic [63:0]   h_wstrb,
  output logic          h_wlast,
  output logic          h_wvalid,
  input  logic          h_wready,
  input  logic [15:0]   h_bid,
  input  logic [1:0]    h_bresp,
  input  logic          h_bvalid,
  output logic          h_bready,
  output logic [15:0]   h_arid,
  output logic [63:0]   h_araddr,
  output logic [7:0]    h_arlen,
  output logic [2:0]    h_arsize,
  output logic [1:0]    h_arburst,
  output logic          h_arvalid,
  input  logic          h_arready,
  input  logic [15:0]   h_rid,
  input  logic [511:0]  h_rdata,
  input  logic [1:0]    h_rresp,
  input  logic          h_rlast,
  input  logic          h_rvalid,
  output logic          h_rready,
  // ---- HOST_MUX: cl_axi_sc_1x1's AXI3 master (hclk, after its register slice) back in
  input  logic [5:0]    hs_awid,
  input  logic [33:0]   hs_awaddr,
  input  logic [3:0]    hs_awlen,
  input  logic [2:0]    hs_awsize,
  input  logic [1:0]    hs_awburst,
  input  logic          hs_awvalid,
  output logic          hs_awready,
  input  logic [5:0]    hs_wid,
  input  logic [255:0]  hs_wdata,
  input  logic [31:0]   hs_wstrb,
  input  logic          hs_wlast,
  input  logic          hs_wvalid,
  output logic          hs_wready,
  output logic [5:0]    hs_bid,
  output logic [1:0]    hs_bresp,
  output logic          hs_bvalid,
  input  logic          hs_bready,
  input  logic [5:0]    hs_arid,
  input  logic [33:0]   hs_araddr,
  input  logic [3:0]    hs_arlen,
  input  logic [2:0]    hs_arsize,
  input  logic [1:0]    hs_arburst,
  input  logic          hs_arvalid,
  output logic          hs_arready,
  output logic [5:0]    hs_rid,
  output logic [255:0]  hs_rdata,
  output logic [1:0]    hs_rresp,
  output logic          hs_rlast,
  output logic          hs_rvalid,
  input  logic          hs_rready,
  // ---- the HBM ports (hclk): AXI3 256-bit masters, port p < S x NCH (to the HDK's
  //      cl_axi3_256b_reg_slice, then cl_mem_hbm_wrapper)
  output logic [S*NCH-1:0][5:0]    p_awid,
  output logic [S*NCH-1:0][33:0]   p_awaddr,
  output logic [S*NCH-1:0][3:0]    p_awlen,
  output logic [S*NCH-1:0][2:0]    p_awsize,
  output logic [S*NCH-1:0][1:0]    p_awburst,
  output logic [S*NCH-1:0]         p_awvalid,
  input  logic [S*NCH-1:0]         p_awready,
  output logic [S*NCH-1:0][5:0]    p_wid,
  output logic [S*NCH-1:0][255:0]  p_wdata,
  output logic [S*NCH-1:0][31:0]   p_wstrb,
  output logic [S*NCH-1:0]         p_wlast,
  output logic [S*NCH-1:0]         p_wvalid,
  input  logic [S*NCH-1:0]         p_wready,
  input  logic [S*NCH-1:0][5:0]    p_bid,
  input  logic [S*NCH-1:0][1:0]    p_bresp,
  input  logic [S*NCH-1:0]         p_bvalid,
  output logic [S*NCH-1:0]         p_bready,
  output logic [S*NCH-1:0][5:0]    p_arid,
  output logic [S*NCH-1:0][33:0]   p_araddr,
  output logic [S*NCH-1:0][3:0]    p_arlen,
  output logic [S*NCH-1:0][2:0]    p_arsize,
  output logic [S*NCH-1:0][1:0]    p_arburst,
  output logic [S*NCH-1:0]         p_arvalid,
  input  logic [S*NCH-1:0]         p_arready,
  input  logic [S*NCH-1:0][5:0]    p_rid,
  input  logic [S*NCH-1:0][255:0]  p_rdata,
  input  logic [S*NCH-1:0][1:0]    p_rresp,
  input  logic [S*NCH-1:0]         p_rlast,
  input  logic [S*NCH-1:0]         p_rvalid,
  output logic [S*NCH-1:0]         p_rready
);
  localparam int NPORT = S * NCH;
  initial if (NPORT > 32 || CH_LOG2 != 29 || (NPORT > 31 && !HOST_MUX))
    $fatal(1, "otpu_f2_hbmeng_sys: at most 32 HBM ports of 512 MiB; 32 channels need HOST_MUX");
  // ---- board-bound across the SLRs: the core reset and the HBM's ready, NXR stages
  logic rst_b, hbm_rdy;
  logic [NXR:0][1:0] rs;
  assign rs[0] = {rst, hbm_ready};
  assign {rst_b, hbm_rdy} = rs[NXR];
  for (genvar k = 0; k < NXR; k++) begin : g_rs
    otpu_xr #(.W(2)) u_x (.clk, .rst(1'b0), .d(rs[k]), .q(rs[k + 1]));
  end

  // ---- control registers: the OCL across into clk (as otpu_f2_hbm_sys)
  logic [11:0] c_awaddr, c_araddr;
  logic        c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic        c_arvalid, c_arready, c_rvalid, c_rready;
  logic [31:0] c_wdata, c_rdata;
  logic [3:0]  c_wstrb;
  logic [1:0]  c_bresp, c_rresp;
  logic        pcis_wbusy;
  logic [11:0] x_awaddr, x_araddr;
  logic        x_awvalid, x_awready, x_wvalid, x_wready, x_bvalid, x_bready;
  logic        x_arvalid, x_arready, x_rvalid, x_rready;
  logic [31:0] x_wdata, x_rdata;
  logic [3:0]  x_wstrb;
  logic [1:0]  x_bresp, x_rresp;
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
  otpu_axil_xr #(.NXR(NXR)) u_cx (
    .clk, .rst, .srst(rst_b),
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready),
    .s_wdata(c_wdata), .s_wstrb(c_wstrb), .s_wvalid(c_wvalid), .s_wready(c_wready),
    .s_bresp(c_bresp), .s_bvalid(c_bvalid), .s_bready(c_bready),
    .s_araddr(c_araddr), .s_arvalid(c_arvalid), .s_arready(c_arready),
    .s_rdata(c_rdata), .s_rresp(c_rresp), .s_rvalid(c_rvalid), .s_rready(c_rready),
    .m_awaddr(x_awaddr), .m_awvalid(x_awvalid), .m_awready(x_awready),
    .m_wdata(x_wdata), .m_wstrb(x_wstrb), .m_wvalid(x_wvalid), .m_wready(x_wready),
    .m_bresp(x_bresp), .m_bvalid(x_bvalid), .m_bready(x_bready),
    .m_araddr(x_araddr), .m_arvalid(x_arvalid), .m_arready(x_arready),
    .m_rdata(x_rdata), .m_rresp(x_rresp), .m_rvalid(x_rvalid), .m_rready(x_rready));

  // ---- the slices' channels into the engines
  logic [NPORT-1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NPORT-1:0][24:0]  n_caddr;
  logic [NPORT-1:0][511:0] n_wdata, n_rdata;
  logic [NPORT-1:0][63:0]  n_wmask;
  logic [NPORT-1:0][15:0]  n_wdone;
  otpu_f2_hbm_eng #(.NPORT(NPORT), .S(S), .NCH(NCH), .NXR(NXR), .CH_LOG2(CH_LOG2), .BURST(BURST),
                    .IDLE(IDLE), .OT(OT), .RDEPTH(RDEPTH), .HOST_MUX(HOST_MUX)) u_mem (
    .clk, .rst, .hclk, .hrst, .aclk, .arst,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid,
    .n_rdata, .n_wdone, .n_err,
    .pcis_awid, .pcis_awaddr, .pcis_awlen, .pcis_awsize, .pcis_awburst, .pcis_awvalid,
    .pcis_awready, .pcis_wdata, .pcis_wstrb, .pcis_wlast, .pcis_wvalid, .pcis_wready, .pcis_bid,
    .pcis_bresp, .pcis_bvalid, .pcis_bready, .pcis_arid, .pcis_araddr, .pcis_arlen, .pcis_arsize,
    .pcis_arburst, .pcis_arvalid, .pcis_arready, .pcis_rid, .pcis_rdata, .pcis_rresp,
    .pcis_rlast, .pcis_rvalid, .pcis_rready, .pcis_wbusy,
    .h_awid, .h_awaddr, .h_awlen, .h_awsize, .h_awburst, .h_awvalid, .h_awready, .h_wdata,
    .h_wstrb, .h_wlast, .h_wvalid, .h_wready, .h_bid, .h_bresp, .h_bvalid, .h_bready, .h_arid,
    .h_araddr, .h_arlen, .h_arsize, .h_arburst, .h_arvalid, .h_arready, .h_rid, .h_rdata,
    .h_rresp, .h_rlast, .h_rvalid, .h_rready,
    .hs_awid, .hs_awaddr, .hs_awlen, .hs_awsize, .hs_awburst, .hs_awvalid, .hs_awready,
    .hs_wid, .hs_wdata, .hs_wstrb, .hs_wlast, .hs_wvalid, .hs_wready,
    .hs_bid, .hs_bresp, .hs_bvalid, .hs_bready,
    .hs_arid, .hs_araddr, .hs_arlen, .hs_arsize, .hs_arburst, .hs_arvalid, .hs_arready,
    .hs_rid, .hs_rdata, .hs_rresp, .hs_rlast, .hs_rvalid, .hs_rready,
    .p_awid, .p_awaddr, .p_awlen, .p_awsize, .p_awburst, .p_awvalid, .p_awready,
    .p_wid, .p_wdata, .p_wstrb, .p_wlast, .p_wvalid, .p_wready,
    .p_bid, .p_bresp, .p_bvalid, .p_bready,
    .p_arid, .p_araddr, .p_arlen, .p_arsize, .p_arburst, .p_arvalid, .p_arready,
    .p_rid, .p_rdata, .p_rresp, .p_rlast, .p_rvalid, .p_rready);

  // ---- the accelerator (as otpu_f2_hbm_sys, plus NRD)
  logic [3:0] i2c_lo;
  otpu_board #(.S(S), .NCH(NCH), .CH_LOG2(CH_LOG2), .DRAM_LOG2(DRAM_LOG2), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL), .ULANES(ULANES),
               .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS),
               .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_I2C(1'b0), .CHASH(CHASH),
               .DSTEP(DSTEP), .HOSTCAL(1'b0), .SEQ_WC(SEQ_WC), .T2(T2), .PORTB(PORTB), .RL(RL), .ALINE(ALINE), .PLATFORM(PLATFORM),
               .ARCH_HASH(ARCH_HASH), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
               .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
               .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM), .NRD(NRD)) u_board (
    .clk, .clk_f, .rst(rst_b), .ctl_mrst(rst_b),
    .calib({NPORT{hbm_rdy}}), .ded('0), .temp(12'd0), .led, .i2c_lo, .i2c_pin(5'h1f),
    .s_ctl_awaddr(x_awaddr), .s_ctl_awvalid(x_awvalid), .s_ctl_awready(x_awready),
    .s_ctl_wdata(x_wdata), .s_ctl_wstrb(x_wstrb), .s_ctl_wvalid(x_wvalid),
    .s_ctl_wready(x_wready), .s_ctl_bresp(x_bresp), .s_ctl_bvalid(x_bvalid),
    .s_ctl_bready(x_bready), .s_ctl_araddr(x_araddr), .s_ctl_arvalid(x_arvalid),
    .s_ctl_arready(x_arready), .s_ctl_rdata(x_rdata), .s_ctl_rresp(x_rresp),
    .s_ctl_rvalid(x_rvalid), .s_ctl_rready(x_rready),
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_err);
endmodule
