// openTPU: S slices, each with its own DRAM, plus the collective unit (simulation top).
// AXI = 0: the behavioural fixed-latency DRAM. AXI = 1: the board's memory path -- the native
// adapter (otpu_native_dram) in front of a two-channel native memory model with random stalls
// and latency (sim/verilator/otpu_native_mem.sv; requires D = 128); the dump prints the
// adapter's counters. AXI = 2: the same adapter in front of the card's channels as they are
// (sim/verilator/otpu_ldc_mem.sv: per channel the bridge otpu_mem_ch and LiteDRAM's own
// controller, generated with the production core's settings, in the controller clock).
// COLL_HUB (S > 1): the collective as otpu_board builds it at S > 1, not wired to the slices:
// otpu_coll_hub next to the unit, two register stages each way per slice (g_xs's otpu_xr pairs,
// the collective's part of them) and an otpu_coll_port per slice, with the board's COLL_PIPE and
// COLL_F; the slices' control stays wired (the board's crosses too, a constant delay).
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
  parameter bit COLL_HUB   = 1'b0,    // S > 1: the collective through the board's hub (see the top)
  parameter int COLL_PIPE  = 0,       // COLL_HUB: the hub's reads in flight (otpu_board COLL_PIPE)
  parameter int COLL_F     = 8,       // COLL_HUB: the writes a port holds (otpu_board COLL_F)
  parameter bit WDQ        = 1'b0,    // AXI: the adapter's write-done compare a register (otpu_board WDQ)
  parameter int NCH        = 2        // AXI = 1: the adapter's and the model's channels (F2 HBM: 8)
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
  localparam bit HUB = COLL_HUB && S > 1;
  localparam int PIPE = HUB ? COLL_PIPE : 0;   // (COLL_PIPE without the hub: ignored)
  logic [S-1:0] halted, error, coll_req, ldb;
  assign ld_busy = |ldb;
  // the collective unit's ports (coll_*) and each slice's (sc_*): wired together, or the hub and
  // a port per slice between them (HUB)
  cmd_t         coll_cmd [S];
  logic         coll_ack, coll_gnt;
  logic         coll_rgnt, coll_rvalid, coll_rtake;   // COLL_PIPE > 0 (otpu_coll_pu)
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  logic [S-1:0]                  sc_req, sc_ack, sc_gl, sc_gnt;
  cmd_t                          sc_cmd [S];
  logic [S-1:0][LANES-1:0]       sc_ren, sc_wen;
  logic [S-1:0][LANES-1:0][31:0] sc_raddr, sc_rdata, sc_waddr, sc_wdata;

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

    if (AXI == 0) begin : g_dram
      assign a_rdy = 1'b1;
      assign sw_rdy = 1'b1;
      assign b_rdy = 1'b1;
      assign wr_idle = 1'b1;
      otpu_dram #(.WORDS(DRAM_WORDS), .D(D), .LAT(DRAM_LAT), .SID(s)) u_dram (
        .clk, .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
        .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata, .rd_idle,
        .dump);
    end else begin : g_native
      logic [NCH-1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [NCH-1:0][24:0] caddr;
      logic [NCH-1:0][511:0] wdata, rdata;
      logic [NCH-1:0][63:0] wmask;
      logic [NCH-1:0][15:0] wdone;
      otpu_native_dram #(.D(D), .NCH(NCH), .RD(NCH > 2 ? 64 : 128), .WDQ(WDQ)) u_adapt (
        .clk, .rst(sys_rst), .a_flush(rst || ld_start || a_inval),   // as otpu_board
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
        .b_rdata, .wr_idle, .rd_idle,
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      initial if (AXI == 2 && NCH != 2) $fatal(1, "otpu_top: the LiteDRAM channels (AXI = 2) are two");
      if (AXI == 2) begin : g_ldc
        otpu_ldc_mem #(.WORDS(DRAM_WORDS), .SID(s)) u_mem (
          .clk, .rst(sys_rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end else begin : g_model
        otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .NCH(NCH)) u_mem (
          .clk, .rst(sys_rst), .run_rst(rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end
      // the adapter's counters (rtlsim: A runs and fill reads, the partial writes by source)
      always @(posedge clk) if (dump)
        for (int c = 0; c < NCH; c++) begin
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
                 .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .QUANT_XD(QUANT_XD), .VPU_CL(VPU_CL), .ULANES(ULANES),
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
      .coll_req(sc_req[s]), .coll_cmd(sc_cmd[s]), .coll_ack(sc_ack[s]),
      .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
      .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
      .coll_gnt_local(sc_gl[s]), .coll_gnt(sc_gnt[s]),
      .halted(halted[s]), .error(error[s]), .wait_to(), .a_inval, .icount(icount[s]), .pf(), .dump);
  end

  logic coll_gerr;                    // a GATHER on the BAR-only collective unit
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_top: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER && PIPE == 0) begin : g_coll
    // RQ: its reads from its own request register on the slices' ports; through the hub each
    // slice's otpu_coll_port presents them a cycle later (otpu_board at S > 1)
    otpu_coll #(.S(S), .LANES(LANES), .RQ(!HUB)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_rtake = 1'b0;
    assign coll_gerr = 1'b0;
  end else if (COLL_GATHER) begin : g_coll
    otpu_coll_pu #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_gnt(coll_rgnt), .r_valid(coll_rvalid),
      .r_data(coll_rdata[0]), .r_take(coll_rtake),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .w_gnt(coll_gnt));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
    assign coll_rtake = 1'b0;
  end

  if (!HUB) begin : g_cw
    // the unit on the slices' ports: its grant the AND of every slice's TMEM grant
    assign coll_req = sc_req;
    assign coll_cmd = sc_cmd;
    assign sc_ack = {S{coll_ack}};
    assign sc_ren = coll_ren;
    assign sc_raddr = coll_raddr;
    assign coll_rdata = sc_rdata;
    for (genvar s = 0; s < S; s++) begin : g_s
      assign sc_wen[s] = coll_wen;
      assign sc_waddr[s] = coll_waddr;
      assign sc_wdata[s] = coll_wdata;
    end
    assign coll_gnt = &sc_gl;
    assign sc_gnt = {S{coll_gnt}};
    assign coll_rgnt = 1'b0;
    assign coll_rvalid = 1'b0;
  end else begin : g_ch
    // otpu_board g_xs's collective: the hub, two register stages each way per slice (the low
    // bits, the pulses and the request, cleared by reset), a port per slice
    logic                       h_ack, h_wv;
    logic [S-1:0]               h_rv, h_req, h_dv, h_pop;
    logic [LANES-1:0]           h_rm, h_wm;
    logic [LANES-1:0][31:0]     h_ra, h_wa, h_wd;
    logic [S-1:0][LANES-1:0][31:0] h_dd;
    cmd_t                       h_cmd [S];
    otpu_coll_hub #(.S(S), .LANES(LANES), .F(COLL_F), .K(PIPE)) u_chub (
      .clk, .rst,
      .u_req(coll_req), .u_cmds(coll_cmd), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_rgnt(coll_rgnt), .u_rvalid(coll_rvalid), .u_rtake(coll_rtake),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop));
    localparam int DW = 3 + LANES + 32 * LANES + LANES + 64 * LANES;
    localparam int UW = 3 + 32 * LANES + $bits(cmd_t);
    for (genvar s = 0; s < S; s++) begin : g_x
      logic [DW-1:0]          dn_d, dn_m, dn_q;
      logic [UW-1:0]          up_d, up_m, up_q;
      logic                   p_rv, p_wv, p_dv, p_pop;
      logic [LANES-1:0]       p_rm, p_wm;
      logic [LANES-1:0][31:0] p_ra, p_wa, p_wd, p_dd;
      assign dn_d = {h_wd, h_wa, h_wm, h_ra, h_rm, h_wv, h_rv[s], h_ack};
      assign {p_wd, p_wa, p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q;
      assign up_d = {sc_cmd[s], p_dd, p_pop, p_dv, sc_req[s]};
      assign {h_cmd[s], h_dd[s], h_pop[s], h_dv[s], h_req[s]} = up_q;
      always_ff @(posedge clk) begin
        dn_m <= dn_d; dn_q <= dn_m;
        up_m <= up_d; up_q <= up_m;
        if (rst) begin
          dn_m[2:0] <= '0; dn_q[2:0] <= '0;
          up_m[2:0] <= '0; up_q[2:0] <= '0;
        end
      end
      otpu_coll_port #(.LANES(LANES), .F(COLL_F), .K(PIPE)) u_cport (
        .clk, .rst, .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa), .wd(p_wd),
        .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
        .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
        .gnt_local(sc_gl[s]), .gnt(sc_gnt[s]));
    end
  end

  assign all_halted = &halted;
  assign any_error  = |error || coll_gerr;
endmodule
