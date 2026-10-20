// The F2 HBM build's memory side at the core clock (otpu_f2_hbm_sys u_mem; docs/f2_hbm.md "The
// memory path"): NPORT channels' native ports, each across the NXR-stage SLR chain (otpu_xr,
// g_xr.g_x[k]: k = 0 in SLR0 beside the fabric, NXR - 1 in SLR2 beside the slice's adapter) into
// its otpu_native_axi (g_port[p].u_naxi, BASE p x 512 MiB, XL = 2 NXR) and out as an AXI4
// 512-bit master at the core clock (ch_*: the channel's SmartConnect S_N<p>, which converts to
// HBM AXI3 port p, 256 bits at 450 MHz); and the host's PCIS through otpu_f2_pcis_gate (u_pcis:
// registers and the read-after-write / OCL order) to the fabric's host branch (h_*, aclk). Alex's
// burst200 design (ergodex-openTPU 06c169cb) scaled from his 2 channels to NPORT: none of this
// runs at 450 MHz; the SmartConnects and the HDK's HBM wrapper are the only logic there.
// Replaces otpu_f2_hbm_mem (the bridges otpu_f2_hbm_nch, the port arbiters otpu_f2_hbm_port, the
// host path otpu_f2_hbm_host) in the build.
//
// Down the chain, per channel: the adapter's takes (n_cvalid && n_cready, n_wvalid && n_wready:
// it sees u_naxi's readies NXR cycles late) with the command and the beat; up: the readies, read
// data, write count and error. No reset reaches a stage: whatever is in one flushes in NXR cycles
// while both sides hold reset.
module otpu_f2_hbm_axi #(
  parameter int NPORT      = 8,         // channels: slice s's channel c on HBM port NCH s + c
  parameter int NXR        = 0,         // SLR crossing stages each way (0: wired)
  parameter int CH_LOG2    = 29,        // a channel's window (a pseudo-channel's 512 MiB)
  parameter int QD         = 16,        // u_naxi's queues
  parameter int READ_BURST = 8,         // u_naxi's longest read burst, native beats (8: 512 B,
                                        //   the 16-beat AXI3 burst after the 512 -> 256 SmartConnect)
  parameter int READ_IDLE  = 1,         // u_naxi's burst gather timeout, core cycles
  parameter bit READ_WAIT  = 1'b1,      // u_naxi's bursts wait READ_IDLE for the next command
                                        //   (0: only what is queued; under the SmartConnect's
                                        //   32 transactions 8-beat waiting bursts measured best)
  parameter int MAX_OUT    = 128,       // u_naxi's reads in flight (the adapter keeps <= 64)
  parameter bit ORDER      = 1'b1,      // the read / write order rules (0: tests only)
  parameter bit PORDER     = 1'b0,      // ... per page bucket (otpu_native_axi PORDER)
  parameter bit ZB         = 1'b0       // otpu_native_axi's zero-bubble burst switch
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    aclk,
  input  logic                    arst,
  // ---- the channels' native ports (core clock)
  input  logic [NPORT-1:0]        n_cvalid,
  output logic [NPORT-1:0]        n_cready,
  input  logic [NPORT-1:0]        n_cwe,
  input  logic [NPORT-1:0][24:0]  n_caddr,
  input  logic [NPORT-1:0]        n_wvalid,
  output logic [NPORT-1:0]        n_wready,
  input  logic [NPORT-1:0][511:0] n_wdata,
  input  logic [NPORT-1:0][63:0]  n_wmask,
  output logic [NPORT-1:0]        n_rvalid,
  output logic [NPORT-1:0][511:0] n_rdata,
  output logic [NPORT-1:0][15:0]  n_wdone,
  output logic [NPORT-1:0]        n_err,
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
  output logic          pcis_wbusy,
  // ---- the host branch towards the fabric (S_HOST, aclk; AXI4 512 bits)
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
  // ---- the channels' masters (S_N<p>, core clock; AXI4 512 bits, ID 0, INCR, size 64 bytes;
  //      writes single beats: awlen 0, wlast 1)
  output logic [NPORT-1:0][33:0]  ch_awaddr,
  output logic [NPORT-1:0]        ch_awvalid,
  input  logic [NPORT-1:0]        ch_awready,
  output logic [NPORT-1:0][511:0] ch_wdata,
  output logic [NPORT-1:0][63:0]  ch_wstrb,
  output logic [NPORT-1:0]        ch_wvalid,
  input  logic [NPORT-1:0]        ch_wready,
  input  logic [NPORT-1:0]        ch_bvalid,
  input  logic [NPORT-1:0][1:0]   ch_bresp,
  output logic [NPORT-1:0]        ch_bready,
  output logic [NPORT-1:0][33:0]  ch_araddr,
  output logic [NPORT-1:0][7:0]   ch_arlen,
  output logic [NPORT-1:0]        ch_arvalid,
  input  logic [NPORT-1:0]        ch_arready,
  input  logic [NPORT-1:0][511:0] ch_rdata,
  input  logic [NPORT-1:0]        ch_rvalid,
  input  logic [NPORT-1:0]        ch_rlast,
  input  logic [NPORT-1:0][1:0]   ch_rresp,
  output logic [NPORT-1:0]        ch_rready
);
  initial if (NPORT < 1 || NPORT > 32 || NPORT * (64'd1 << CH_LOG2) > (64'd1 << 34))
    $fatal(1, "otpu_f2_hbm_axi: at most 32 channels within the HBM's 16 GiB");
  // a read burst of READ_BURST 64-byte beats is 2 READ_BURST 32-byte beats on the HBM's AXI3
  // port: at most 16 (AXI3's longest; the 1 x 1 SmartConnect would otherwise split it)
  initial if (READ_BURST > 8) $fatal(1, "otpu_f2_hbm_axi: READ_BURST over 8 beats (16 AXI3 beats)");
  // the reset into a register here, replicated toward the channels (each u_naxi takes it as is)
  (* max_fanout = 4 *) logic rst_q;
  always_ff @(posedge clk) rst_q <= rst;
  // ---- the native ports' crossing (u_naxi's side b_*)
  logic [NPORT-1:0]        b_cvalid, b_cready, b_cwe, b_wvalid, b_wready, b_rvalid, b_err;
  logic [NPORT-1:0][24:0]  b_caddr;
  logic [NPORT-1:0][511:0] b_wdata, b_rdata;
  logic [NPORT-1:0][63:0]  b_wmask;
  logic [NPORT-1:0][15:0]  b_wdone;
  if (NXR == 0) begin : g_wire
    assign b_cvalid = n_cvalid; assign b_cwe = n_cwe; assign b_caddr = n_caddr;
    assign b_wvalid = n_wvalid; assign b_wdata = n_wdata; assign b_wmask = n_wmask;
    assign n_cready = b_cready; assign n_wready = b_wready; assign n_rvalid = b_rvalid;
    assign n_rdata = b_rdata; assign n_wdone = b_wdone; assign n_err = b_err;
  end else begin : g_xr
    localparam int DW = 1 + 1 + 25 + 1 + 512 + 64;   // down: take, we, beat, take, data, mask
    localparam int UW = 1 + 1 + 1 + 512 + 16 + 1;    // up: readies, rvalid, data, wdone, err
    logic [NXR:0][NPORT-1:0][DW-1:0] dn;   // dn[NXR] the adapter's, dn[0] u_naxi's
    logic [NXR:0][NPORT-1:0][UW-1:0] up;   // up[0] u_naxi's, up[NXR] the adapter's
    for (genvar p = 0; p < NPORT; p++) begin : g_ch
      assign dn[NXR][p] = {n_cvalid[p] && n_cready[p], n_cwe[p], n_caddr[p],
                           n_wvalid[p] && n_wready[p], n_wdata[p], n_wmask[p]};
      assign {b_cvalid[p], b_cwe[p], b_caddr[p], b_wvalid[p], b_wdata[p], b_wmask[p]} = dn[0][p];
      assign up[0][p] = {b_cready[p], b_wready[p], b_rvalid[p], b_rdata[p], b_wdone[p], b_err[p]};
      assign {n_cready[p], n_wready[p], n_rvalid[p], n_rdata[p], n_wdone[p], n_err[p]} = up[NXR][p];
    end
    for (genvar k = 0; k < NXR; k++) begin : g_x   // k: the stage's SLR (0 u_naxi's)
      otpu_xr #(.W(NPORT * DW)) u_dn (.clk, .rst(1'b0), .d(dn[k + 1]), .q(dn[k]));
      otpu_xr #(.W(NPORT * UW)) u_up (.clk, .rst(1'b0), .d(up[k]), .q(up[k + 1]));
    end
  end

  for (genvar p = 0; p < NPORT; p++) begin : g_port
    otpu_native_axi #(.BASE_ADDR(34'(p) << CH_LOG2), .CH_LOG2(CH_LOG2), .QD(QD), .XL(2 * NXR),
                      .MAX_OUT(MAX_OUT), .READ_BURST(READ_BURST), .READ_IDLE(READ_IDLE),
                      .READ_WAIT(READ_WAIT), .ORDER(ORDER), .PORDER(PORDER), .ZB(ZB)) u_naxi (
      .clk, .rst(rst_q),
      .n_cvalid(b_cvalid[p]), .n_cready(b_cready[p]), .n_cwe(b_cwe[p]), .n_caddr(b_caddr[p]),
      .n_wvalid(b_wvalid[p]), .n_wready(b_wready[p]), .n_wdata(b_wdata[p]), .n_wmask(b_wmask[p]),
      .n_rvalid(b_rvalid[p]), .n_rdata(b_rdata[p]), .n_wdone(b_wdone[p]), .n_err(b_err[p]),
      .m_awaddr(ch_awaddr[p]), .m_awvalid(ch_awvalid[p]), .m_awready(ch_awready[p]),
      .m_wdata(ch_wdata[p]), .m_wstrb(ch_wstrb[p]), .m_wvalid(ch_wvalid[p]), .m_wready(ch_wready[p]),
      .m_bvalid(ch_bvalid[p]), .m_bresp(ch_bresp[p]), .m_bready(ch_bready[p]),
      .m_araddr(ch_araddr[p]), .m_arlen(ch_arlen[p]), .m_arvalid(ch_arvalid[p]),
      .m_arready(ch_arready[p]), .m_rdata(ch_rdata[p]), .m_rvalid(ch_rvalid[p]),
      .m_rlast(ch_rlast[p]), .m_rresp(ch_rresp[p]), .m_rready(ch_rready[p]));
  end

  otpu_f2_pcis_gate #(.HOLD(ORDER)) u_pcis (
    .aclk, .arst,
    .s_awid(pcis_awid), .s_awaddr(pcis_awaddr), .s_awlen(pcis_awlen), .s_awsize(pcis_awsize),
    .s_awburst(pcis_awburst), .s_awvalid(pcis_awvalid), .s_awready(pcis_awready),
    .s_wdata(pcis_wdata), .s_wstrb(pcis_wstrb), .s_wlast(pcis_wlast), .s_wvalid(pcis_wvalid),
    .s_wready(pcis_wready), .s_bid(pcis_bid), .s_bresp(pcis_bresp), .s_bvalid(pcis_bvalid),
    .s_bready(pcis_bready), .s_arid(pcis_arid), .s_araddr(pcis_araddr), .s_arlen(pcis_arlen),
    .s_arsize(pcis_arsize), .s_arburst(pcis_arburst), .s_arvalid(pcis_arvalid),
    .s_arready(pcis_arready), .s_rid(pcis_rid), .s_rdata(pcis_rdata), .s_rresp(pcis_rresp),
    .s_rlast(pcis_rlast), .s_rvalid(pcis_rvalid), .s_rready(pcis_rready),
    .m_awid(h_awid), .m_awaddr(h_awaddr), .m_awlen(h_awlen), .m_awsize(h_awsize),
    .m_awburst(h_awburst), .m_awvalid(h_awvalid), .m_awready(h_awready),
    .m_wdata(h_wdata), .m_wstrb(h_wstrb), .m_wlast(h_wlast), .m_wvalid(h_wvalid),
    .m_wready(h_wready), .m_bid(h_bid), .m_bresp(h_bresp), .m_bvalid(h_bvalid),
    .m_bready(h_bready), .m_arid(h_arid), .m_araddr(h_araddr), .m_arlen(h_arlen),
    .m_arsize(h_arsize), .m_arburst(h_arburst), .m_arvalid(h_arvalid), .m_arready(h_arready),
    .m_rid(h_rid), .m_rdata(h_rdata), .m_rresp(h_rresp), .m_rlast(h_rlast),
    .m_rvalid(h_rvalid), .m_rready(h_rready), .pcis_wbusy);
endmodule
