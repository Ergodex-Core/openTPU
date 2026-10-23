// openTPU: S slices, each with its own DRAM, plus the collective unit (simulation top).
// AXI = 0: the behavioural fixed-latency DRAM. AXI = 1: the board's memory path -- the native
// adapter (otpu_native_dram) in front of a two-channel native memory model with random stalls
// and latency (sim/verilator/otpu_native_mem.sv; requires D = 128); the dump prints the
// adapter's counters. AXI = 2: the same adapter in front of the card's channels as they are
// (sim/verilator/otpu_ldc_mem.sv: per channel the bridge otpu_mem_ch and LiteDRAM's own
// controller, generated with the production core's settings, in the controller clock).
// AXI = 1 with MXU_C > 1 and LMEM (the default): the lane memory of the F2 build (otpu_lane_mem,
// docs/v4/p2-f2.md "M4"): MXU_C image copies, copy l behind its own adapter on its own model
// (otpu_native_mem COPY l, or with NAXI the build's per-channel path: otpu_naxi_mem's SLR chain
// and otpu_native_axi), lane l reading copy l; at the dump every copy is compared with copy 0.
// LMEM 0: lanes 1 .. MXU_C-1 on the model's behavioural lane ports (P2's M2 stand-in).
module otpu_top
  import otpu_pkg::*;
#(
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int DRAM_WORDS = 1 << 18,
  parameter int DRAM_LAT   = 8,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,
  parameter int WPB        = 2,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int MXU_C      = 1,       // MXU chunk lanes (otpu_mxu_lanes; the memory models' lane ports)
  parameter int QUANT_XD   = 1024,    // quantizer row buffer chunks (0: two passes over TMEM; otpu_quant XD)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int AXI        = 0,       // memory path: see the top
  parameter int TMEM_IMPL  = 0,       // TMEM: 0 a copy per read port, 1 banked (otpu_slice)
  parameter bit TMEM_RQ    = 1'b0,    // TMEM_IMPL 1: request-priority bank addresses (otpu_slice)
  parameter int TMEM_NCP   = 3,       // TMEM_IMPL 1: copies (otpu_slice)
  parameter int TMEM_NSB   = 8,       // TMEM_IMPL 3: sub-banks per bank (otpu_slice)
  parameter int TMEM_SBP   = 3,       // TMEM_IMPL 3: the sub-bank field's word-address bit
  parameter int TMEM_SBH   = 0,       // TMEM_IMPL 3: the sub-bank hashed (otpu_slice)
  parameter bit TMEM_RPL   = 1'b0,    // TMEM_IMPL 3: lost reads replayed (otpu_slice)
  parameter bit TMEM_TDP   = 1'b0,    // TMEM_IMPL 3: true dual-port RAMs (otpu_slice)
  parameter bit TMEM_GST   = 1'b0,    // TMEM_IMPL 3: QUANT 3 and COLL the DMA's guests (otpu_slice)
  parameter int TMEM_PMP   = 0,       // TMEM_TDP: 2 the pumped RAMs' model (otpu_slice)
  parameter int TMEM_PRK   = 0,       // TMEM_IMPL 3: the read ports' ranks (otpu_slice)
  parameter int TMEM_SDM   = -1,      // TMEM_TDP: the read ports' RAM sides (otpu_slice)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait counts (otpu_slice)
  parameter bit COLL_GATHER = 1'b1,   // 0 (S = 1 only): the collective unit runs BAR alone
                                      // (otpu_coll_bar), as the card builds it (otpu_board)
  parameter bit LMEM       = 1'b1,    // AXI = 1, MXU_C > 1: the lane memory (see the top)
  parameter bit SWFILL     = 1'b1,    // AXI = 1: SW partial beats with a fill read (otpu_native_dram;
                                      //   0: with their byte masks, the F2 build's; OTPU_SWFILL)
  parameter bit NAXI       = 1'b0,    // AXI = 1: the F2 HBM build's channel path in front of the
  parameter int NAXI_XR    = 3,       //   model (sim/verilator/otpu_naxi_mem.sv: NAXI_XR SLR
  parameter int NAXI_BURST = 8,       //   stages each way, otpu_native_axi with its read bursts,
  parameter int NAXI_IDLE  = 1,       //   gather timeout and wait, the SmartConnect's
  parameter bit NAXI_WAIT  = 1'b1,    //   transactions; the build's settings: otpu_f2_hbm_sys;
  parameter int NAXI_OT    = 32,      //   rtlsim OTPU_NAXI)
  parameter bit NAXI_PORDER = 1'b0,   //   otpu_native_axi's barriers per page bucket (OTPU_NAXI_PORDER)
  parameter bit NAXI_ZB    = 1'b0     //   and its zero-bubble burst switch (OTPU_NAXI_ZB)
) (
  input  logic          clk,
  input  logic          clk_f,         // TMEM_PMP 1: 2 x clk, rising with it (otpu_slice)
  input  logic          sys_rst,
  input  logic          rst,
  input  logic [31:0]   rinit [8],     // the run's arguments (R8..R15 at the start)
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  output logic          all_halted,
  output logic          any_error,
  output logic [31:0]   icount [S],
  input  logic          dump
);
  logic [S-1:0] halted, error, coll_req, coll_gl, ldb;
  assign ld_busy = |ldb;
  cmd_t         coll_cmd [S];
  logic         coll_ack;
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  wire coll_gnt = &coll_gl;

  for (genvar s = 0; s < S; s++) begin : g_slice
    logic          a_req, a_we, a_rvalid, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata;
    logic [31:0]   a_addr, a_wdata, b_addr;
    logic [255:0]  a_rdata;
    logic [3:0]    a_be;
    logic [D*8-1:0] b_rdata;
    logic          a_rdy, b_rdy, wr_idle, rd_idle, sw_req, sw_rdy, a_inval;
    logic [31:0]   sw_addr, sw_wdata;
    logic [3:0]    sw_be;
    // the MXU's chunk lanes' reads (MXU_C > 1; otpu_slice lb_* / la_*)
    logic [MXU_C-1:0]          lb_req, lb_rdy, lb_rvalid, la_req, la_rdy, la_rvalid;
    logic [MXU_C-1:0][31:0]    lb_addr, la_addr;
    logic [MXU_C-1:0][D*8-1:0] lb_rdata;
    logic [MXU_C-1:0][255:0]   la_rdata;

    if (AXI == 0) begin : g_dram
      assign a_rdy = 1'b1;
      assign sw_rdy = 1'b1;
      assign b_rdy = 1'b1;
      assign wr_idle = 1'b1;
      otpu_dram #(.WORDS(DRAM_WORDS), .D(D), .LAT(DRAM_LAT), .SID(s), .NL(MXU_C)) u_dram (
        .clk, .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
        .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata, .rd_idle,
        .lb_req, .lb_addr, .lb_rdy, .lb_rvalid, .lb_rdata,
        .la_req, .la_addr, .la_rdy, .la_rvalid, .la_rdata,
        .dump);
    end else if (AXI == 1 && MXU_C > 1 && LMEM) begin : g_lmem
      // the lane memory: MXU_C copies, each an adapter on its own channel pair and model
      logic [2*MXU_C-1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [2*MXU_C-1:0][24:0] caddr;
      logic [2*MXU_C-1:0][511:0] wdata, rdata;
      logic [2*MXU_C-1:0][63:0] wmask;
      logic [2*MXU_C-1:0][15:0] wdone;
      otpu_lane_mem #(.D(D), .C(MXU_C), .SWFILL(SWFILL)) u_lmem (
        .clk, .rst(sys_rst), .a_flush(rst || ld_start || a_inval),   // as otpu_board
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
        .b_rdata, .wr_idle, .rd_idle,
        .lb_req, .lb_addr, .lb_rdy, .lb_rvalid, .lb_rdata,
        .la_req, .la_addr, .la_rdy, .la_rvalid, .la_rdata,
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      for (genvar l = 0; l < MXU_C; l++) begin : g_cp
        if (NAXI) begin : g_naxi
          otpu_naxi_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .NCH(2), .NXR(NAXI_XR),
                          .READ_BURST(NAXI_BURST), .READ_IDLE(NAXI_IDLE), .READ_WAIT(NAXI_WAIT),
                          .OT(NAXI_OT), .COPY(l), .HASH(1'b1), .PORDER(NAXI_PORDER), .ZB(NAXI_ZB)) u_mem (
            .clk, .rst(sys_rst), .run_rst(rst),
            .n_cvalid(cvalid[2*l +: 2]), .n_cready(cready[2*l +: 2]), .n_cwe(cwe[2*l +: 2]),
            .n_caddr(caddr[2*l +: 2]), .n_wvalid(wvalid[2*l +: 2]), .n_wready(wready[2*l +: 2]),
            .n_wdata(wdata[2*l +: 2]), .n_wmask(wmask[2*l +: 2]), .n_rvalid(rvalid[2*l +: 2]),
            .n_rdata(rdata[2*l +: 2]), .n_wdone(wdone[2*l +: 2]), .dump);
        end else begin : g_mod
          otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .COPY(l), .HASH(1'b1)) u_mem (
            .clk, .rst(sys_rst), .run_rst(rst),
            .n_cvalid(cvalid[2*l +: 2]), .n_cready(cready[2*l +: 2]), .n_cwe(cwe[2*l +: 2]),
            .n_caddr(caddr[2*l +: 2]), .n_wvalid(wvalid[2*l +: 2]), .n_wready(wready[2*l +: 2]),
            .n_wdata(wdata[2*l +: 2]), .n_wmask(wmask[2*l +: 2]), .n_rvalid(rvalid[2*l +: 2]),
            .n_rdata(rdata[2*l +: 2]), .n_wdone(wdone[2*l +: 2]),
            .lb_req(1'b0), .lb_addr('0), .lb_rdy(), .lb_rvalid(), .lb_rdata(),
            .la_req(1'b0), .la_addr('0), .la_rdy(), .la_rvalid(), .la_rdata(), .l_idle(), .dump);
        end
      end
      // the adapters' counters, copy 0 as "NATIVE ch<c>" (the one-adapter path's lines), the
      // others "NATIVE cp<l> ch<c>"; each copy's model prints its image's hash at the dump
      // ("MEM copy<l> hash=": otpu_native_mem HASH), which rtlsim requires equal over the copies
      // (a write that missed a copy, or a copy's fill read of other data, shows there)
      always @(posedge clk) if (dump) begin
        for (int c = 0; c < 2; c++) begin
          $write("NATIVE ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d ", c,
                 u_lmem.g_cp[0].u_adapt.st_arun[c], u_lmem.g_cp[0].u_adapt.st_ard[c],
                 u_lmem.g_cp[0].u_adapt.st_brd[c], u_lmem.g_cp[0].u_adapt.st_srd[c]);
          $display("b_wr=%0d a_wr=%0d sw_wr=%0d part_b=%0d part_a=%0d part_sw=%0d",
                   u_lmem.g_cp[0].u_adapt.st_bwr[c], u_lmem.g_cp[0].u_adapt.st_awr[c],
                   u_lmem.g_cp[0].u_adapt.st_swr[c], u_lmem.g_cp[0].u_adapt.st_pb[c],
                   u_lmem.g_cp[0].u_adapt.st_pa[c], u_lmem.g_cp[0].u_adapt.st_ps[c]);
        end
      end
      for (genvar l = 1; l < MXU_C; l++) begin : g_cnt
        always @(posedge clk) if (dump)
          for (int c = 0; c < 2; c++)
            $display("NATIVE cp%0d ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d b_wr=%0d sw_wr=%0d part_sw=%0d",
                     l, c, u_lmem.g_cp[l].u_adapt.st_arun[c], u_lmem.g_cp[l].u_adapt.st_ard[c],
                     u_lmem.g_cp[l].u_adapt.st_brd[c], u_lmem.g_cp[l].u_adapt.st_srd[c],
                     u_lmem.g_cp[l].u_adapt.st_bwr[c], u_lmem.g_cp[l].u_adapt.st_swr[c],
                     u_lmem.g_cp[l].u_adapt.st_ps[c]);
      end
    end else begin : g_native
      logic ad_idle, l_idle;              // the adapter's reads, the lane ports' reads
      assign rd_idle = ad_idle && l_idle;
      logic [1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [1:0][24:0] caddr;
      logic [1:0][511:0] wdata, rdata;
      logic [1:0][63:0] wmask;
      logic [1:0][15:0] wdone;
      otpu_native_dram #(.D(D), .SWFILL(SWFILL)) u_adapt (
        .clk, .rst(sys_rst), .a_flush(rst || ld_start || a_inval),   // as otpu_board
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata,
        .sw_rdy, .sw_rdy2(), .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
        .b_rdata, .wr_idle, .rd_idle(ad_idle),
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      if (AXI == 2) begin : g_ldc
        // (the co-simulated controller has no lane ports: MXU_C 1 only)
        initial if (MXU_C > 1) $fatal(1, "otpu_top: MXU_C %0d needs the native memory model (AXI 1)", MXU_C);
        assign lb_rdy = '0; assign lb_rvalid = '0; assign lb_rdata = '0;
        assign la_rdy = '0; assign la_rvalid = '0; assign la_rdata = '0;
        assign l_idle = 1'b1;
        otpu_ldc_mem #(.WORDS(DRAM_WORDS), .SID(s)) u_mem (
          .clk, .rst(sys_rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end else begin : g_model
        otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .NL(MXU_C)) u_mem (
          .clk, .rst(sys_rst), .run_rst(rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone),
          .lb_req, .lb_addr, .lb_rdy, .lb_rvalid, .lb_rdata,
          .la_req, .la_addr, .la_rdy, .la_rvalid, .la_rdata, .l_idle, .dump);
      end
      // the adapter's counters (rtlsim: A runs and fill reads, the partial writes by source)
      always @(posedge clk) if (dump)
        for (int c = 0; c < 2; c++) begin
          $write("NATIVE ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d ", c, u_adapt.st_arun[c],
                 u_adapt.st_ard[c], u_adapt.st_brd[c], u_adapt.st_srd[c]);
          $display("b_wr=%0d a_wr=%0d sw_wr=%0d part_b=%0d part_a=%0d part_sw=%0d",
                   u_adapt.st_bwr[c], u_adapt.st_awr[c], u_adapt.st_swr[c], u_adapt.st_pb[c],
                   u_adapt.st_pa[c], u_adapt.st_ps[c]);
        end
    end

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB),
                 .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .MXU_C(MXU_C), .QUANT_XD(QUANT_XD), .VPU_CL(VPU_CL), .ULANES(ULANES),
                 .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ), .TMEM_NCP(TMEM_NCP),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
                 .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
                 .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM), .SEQ_WC(SEQ_WC))
    u_slice (
      .clk, .clk_f, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy(ldb[s]),
      .a_rdy, .b_rdy, .sw_rdy, .wr_idle, .rd_idle,
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
      .lb_req, .lb_addr, .lb_rdy, .lb_rvalid, .lb_rdata,
      .la_req, .la_addr, .la_rdy, .la_rvalid, .la_rdata,
      .coll_req(coll_req[s]), .coll_cmd(coll_cmd[s]), .coll_ack,
      .coll_ren(coll_ren[s]), .coll_raddr(coll_raddr[s]), .coll_rdata(coll_rdata[s]),
      .coll_wen, .coll_waddr, .coll_wdata, .coll_gnt_local(coll_gl[s]), .coll_gnt,
      .halted(halted[s]), .error(error[s]), .wait_to(), .a_inval, .icount(icount[s]), .pf(), .dump);
  end

  logic coll_gerr;                    // a GATHER on the BAR-only collective unit
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_top: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER) begin : g_coll
    otpu_coll #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
  end

  assign all_halted = &halted;
  assign any_error  = |error || coll_gerr;
endmodule
