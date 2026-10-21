`timescale 1ns/1ps
// Retain a completed collective behind an unfinished DMA in the simulated ROB.
// Drive only the sequencer's instruction-loader and unit-handshake interfaces.
module tb_seq_dispatch;
  import otpu_pkg::*;
  parameter bit WAITC = 1'b0;
  logic clk = 0;
  always #5 clk = !clk;
  logic rst = 1;
  cmd_t ucmd [NUNITS];
  logic [NUNITS-1:0] ustart, urdy = '1, udone = '0;
  logic urel, halted, error, ch_req;
  logic [31:0] icount, ch_addr, ch_n;
  seq_ev_t ev;
  logic [31:0] rinit [8] = '{default: '0};
  logic rl_v = 0;
  logic [31:0] rl_val = 0;
  logic im_we = 0, sc_we = 0;
  logic [31:0] im_row = 0, sc_row = 0;
  logic [255:0] im_data = '0, sc_data = '0;
  otpu_seq #(.IMEM_WORDS(256), .TMEM_WORDS(256), .WIN(4), .WAITC(WAITC)) dut (.*);

  logic [255:0] prog [3];
  cmd_t dma_cmd, coll_cmd;
  int dma_slot = -1, coll_slot = -1;
  initial begin
    // LD DRAM[0:32] -> TMEM[0:8], then independent GATHER TMEM[32:40] -> [64:72].
    prog[0] = {32'd0, 32'd0, 32'd0, 32'd0, 32'd8, 32'd0, 32'd0, 24'd0, OP_LD};
    prog[1] = {32'd0, 32'd8, 32'd8, 32'd8, 32'h0008_0001, 32'd64, 32'd32,
               24'd0, OP_GATHER};
    prog[2] = {224'd0, 24'd0, OP_HALT};
    repeat (2) @(negedge clk);
    // All four sidecar entries are independent (no waits), including GATHER.
    sc_we = 1;
    for (int i = 0; i < 3; i++) begin
      im_we = 1; im_row = 32'(i); im_data = prog[i];
      @(negedge clk);
    end
    im_we = 0; sc_we = 0;
    @(negedge clk);
    rst = 0;
    for (int c = 0; c < 100; c++) begin
      @(negedge clk);
      if (error) $fatal(1, "sequencer rejected lifetime program");
      if (ustart[U_DMA]) begin
        if (dma_slot != -1 || ucmd[U_DMA].op != OP_LD)
          $fatal(1, "unexpected DMA start");
        dma_slot = int'(dut.ucs[U_DMA]); dma_cmd = ucmd[U_DMA];
        urdy[U_DMA] = 0;
      end
      if (ustart[U_COLL]) begin
        if (dma_slot != 0 || ucmd[U_COLL].op != OP_GATHER)
          $fatal(1, "collective did not start behind slot-zero DMA");
        coll_slot = int'(dut.ucs[U_COLL]); coll_cmd = ucmd[U_COLL];
        urdy[U_COLL] = 0;
        break;
      end
    end
    if (coll_slot < 0) $fatal(1, "collective never started");
    // The active command must remain unchanged through its completion edge.
    repeat (3) begin
      @(negedge clk);
      if (ucmd[U_COLL] !== coll_cmd) $fatal(1, "active collective command changed");
    end
    udone[U_COLL] = 1;
    @(negedge clk);
    if (ucmd[U_COLL] !== coll_cmd) $fatal(1, "command changed on completion edge");
    udone[U_COLL] = 0; urdy[U_COLL] = 1;
    // No collective candidate remains: its idle read uses default slot zero.
    // ucs still names the completed GATHER, while q legally becomes the LD.
    repeat (3) @(negedge clk);
    if (!dut.sv[dma_slot] || dut.sdone[dma_slot] || halted)
      $fatal(1, "older DMA did not remain outstanding");
    if (int'(dut.ucs[U_COLL]) != coll_slot || ustart[U_COLL] ||
        ucmd[U_COLL] !== dma_cmd || ucmd[U_COLL] === coll_cmd)
      $fatal(1, "missing idle command reload with a stale completed-slot id");
    if (dut.hz_rob) begin
      if (!dut.sv[coll_slot] || !dut.sdone[coll_slot] ||
          !dut.sstarted[coll_slot] || !dut.older[coll_slot][dma_slot])
        $fatal(1, "ROB did not retain the completed collective behind the DMA");
    end else if (dut.sv[coll_slot])
      $fatal(1, "non-ROB completion did not free the collective");
    udone[U_DMA] = 1;
    @(negedge clk);
    udone[U_DMA] = 0; urdy[U_DMA] = 1;
    for (int c = 0; c < 20 && !halted; c++) @(negedge clk);
    if (!halted || error || dut.sv != '0 || icount != 3)
      $fatal(1, "completed window did not retire and halt");
    $display("DISPATCH_LIFETIME PASS waitc=%0d inorder=%0d rob=%0d cycles=%0d instructions=%0d",
             WAITC, dut.hz_ino, dut.hz_rob, dut.cyc, icount);
    $finish;
  end
  initial begin
    #10000;
    $fatal(1, "dispatch lifetime timeout");
  end
endmodule
