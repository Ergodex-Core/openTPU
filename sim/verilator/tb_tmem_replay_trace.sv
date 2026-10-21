// Directed single- and eight-lane traffic at the slice's otpu_tmem boundary. This is a resource trace for a
// separate bank/replay model, not a model of unit execution or the slice's whole arbiter.
// One TMEMTRACE JSON object describes inputs and combinational outputs just before each core edge.
module tb_tmem_replay_trace #(
  parameter int PMP = 0,
  parameter int SIDE_OVERRIDE = -1  // -1: the slice's PMP-dependent side map
);
  localparam int WORDS = 256, LANES = 8, NRP = 8, NWP = 4, CYCLES = 256;
  // Packed vectors are written high port to low port; the low-to-high PRANK3 values are
  // [4,0,5,2,3,7,6,1] and the RGRP values are [3,0,4,2,2,4,4,1].
  localparam logic [NRP-1:0][2:0] PRANK3 =
      {3'd1, 3'd6, 3'd7, 3'd3, 3'd2, 3'd5, 3'd0, 3'd4};
  localparam logic [NRP-1:0][2:0] RGRP =
      {3'd1, 3'd4, 3'd4, 3'd2, 3'd2, 3'd4, 3'd0, 3'd3};
  localparam logic [NRP-1:0] SIDE_MAP = SIDE_OVERRIDE < 0 ?
      (PMP == 0 ? 8'hf0 : 8'h0c) : NRP'(SIDE_OVERRIDE);

  logic clk = 0, clk_f = 0;
  logic [NRP-1:0][LANES-1:0] r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0] w_en, w_req;
  logic [NWP-1:0] w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  logic [NRP-1:0] blk_x, blk_s, lost, frz;

  otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(1),
              .GEN_R(8'h01), .GEN_W(4'h2), .IMPL(3), .PRANK(PRANK3), .GRP(RGRP),
              .NSB(2), .SBP(3), .SBH(0), .RPL(1'b1), .RPLM(8'h7f), .TOPP(1),
              .TDP(1'b1), .SIDE(SIDE_MAP), .GST(1'b1), .SH_HOST(1), .SH_MASK(8'ha0),
              .PMP(PMP), .IQ(1'b1)) dut (
    .clk, .clk_f, .r_en, .r_req, .r_addr, .r_data, .w_en, .w_req, .w_gnt,
    .w_addr, .w_data, .blk_x, .blk_s, .lost, .frz, .dump(1'b0));

  task automatic ask(input int port, input int word_address);
    r_req[port][0] = 1'b1;
    r_addr[port][0] = 32'(word_address);
  endtask

  task automatic emit_trace(input int cyc);
    bit first;
    $write("TMEMTRACE: {\"cycle\":%0d,\"pmp\":%0d,\"side_mask\":%0d,\"reads\":[",
           cyc, PMP, $unsigned(SIDE_MAP));
    for (int p = 0; p < NRP; p++) begin
      if (p != 0) $write(",");
      $write("[");
      first = 1'b1;
      for (int l = 0; l < LANES; l++)
        if (r_req[p][l]) begin
          if (!first) $write(",");
          $write("%0d", $unsigned(r_addr[p][l]));
          first = 1'b0;
        end
      $write("]");
    end
    $write("],\"taken\":[");
    for (int p = 0; p < NRP; p++) begin
      if (p != 0) $write(",");
      if (|r_en[p]) $write("true");
      else $write("false");
    end
    $write("],\"issued_writes\":[");
    first = 1'b1;
    for (int p = 0; p < NWP; p++)
      for (int l = 0; l < LANES; l++)
        if (w_en[p][l]) begin
          if (!first) $write(",");
          $write("%0d", $unsigned(w_addr[p][l]));
          first = 1'b0;
        end
    $write("],\"writes\":[");
    first = 1'b1;
    for (int b = 0; b < LANES; b++)
      if (dut.pw_v[b][0]) begin
        if (!first) $write(",");
        $write("%0d", $unsigned(dut.pw_a[b][0]) * LANES + b);
        first = 1'b0;
      end
    $display("],\"lost\":%0d,\"frozen\":%0d,\"blk_x\":%0d,\"blk_s\":%0d}",
             $unsigned(lost), $unsigned(frz), $unsigned(blk_x), $unsigned(blk_s));
  endtask

  initial begin
    if (PMP != 0 && PMP != 1) $fatal(1, "PMP must be 0 or 1");
    r_en = '0; r_req = '0; r_addr = '0;
    w_en = '0; w_req = '0; w_gnt = '0; w_addr = '0; w_data = '0;
    for (int cyc = 0; cyc < CYCLES; cyc++) begin
      int bank;
      r_en = '0; r_req = '0; r_addr = '0;
      w_en = '0; w_req = '0; w_gnt = '0; w_addr = '0; w_data = '0;
      bank = cyc % LANES;

      if (cyc < 16) begin
        // Same-word sharing: each side has several askers for a single bank/RAM/word.
        for (int p = 0; p < NRP; p++)
          if (p != 5 && p != 7) ask(p, 16 + bank);
      end else if (cyc < 40) begin
        // Two sub-banks of one lane bank, with same-word sharing within each sub-bank.
        ask(0, 16 + bank); ask(2, 24 + bank); ask(3, 16 + bank);
        ask(4, 24 + bank); ask(6, 16 + bank);
      end else if (cyc < 132) begin
        // Staggered, distinct words in the same RAM: three replayable askers on port B
        // for either side map. TOPP/DMA takes that RAM on four cycles out of five.
        if (cyc % 5 != 4) ask(1, 48);
        ask(0, 0);
        if (cyc >= 42) ask(!SIDE_MAP[2] ? 2 : 4, 16);
        if (cyc >= 44) ask(!SIDE_MAP[3] ? 3 : 6, 32);
      end else if (cyc < 176) begin
        // Host, QUANT3 guest, and COLL guest overlap in raw requests. Only the first
        // eligible guest takes the host's data selector when the host is idle.
        if (cyc % 6 < 2) ask(1, 96);
        if (cyc % 6 >= 1 && cyc % 6 <= 3) ask(5, 112);
        if (cyc % 6 >= 3) ask(7, 128);
      end else if (cyc < 232) begin
        // Same-word bypass first, followed by an idle gap before the conflicting
        // write sequence. Never overwrite a word with a pending replay.
        if (cyc < 182) begin
          w_req[0][0] = 1'b1; w_en[0][0] = 1'b1; w_gnt[0] = 1'b1;
          w_addr[0][0] = 32'd80; w_data[0][0] = 32'(cyc);
          if (cyc >= 178) ask(SIDE_MAP[4] ? 4 : 2, 80);
        end
        // IQ delays the granted input write to pw. Address 80 shares bank/sub-bank 0
        // with the side-A read of word 0, but is never the read's own word.
        if (cyc >= 188 && cyc < 224) begin
          w_req[0][0] = 1'b1; w_en[0][0] = 1'b1; w_gnt[0] = 1'b1;
          w_addr[0][0] = 32'd80; w_data[0][0] = 32'(cyc);
        end
        if (cyc >= 192) ask(SIDE_MAP[4] ? 4 : 2, 0);
      end else if (cyc < 248) begin
        // All eight banks: the MXU's scattered read differs from VPU B in
        // only bank0. One conflicting bank replays the entire port request.
        for (int l = 0; l < LANES; l++) begin
          r_req[0][l] = 1'b1; r_addr[0][l] = 32'(l == 0 ? 16 : l);
          r_req[4][l] = 1'b1; r_addr[4][l] = 32'(l);
        end
      end
      // Final cycles drain all replay registers without new requests.

      // A frozen unit issues no fresh requests from any port of its arbitration group.
      // The replay itself remains in the TMEM's registered rpv/rpe/rpa state.
      for (int p = 0; p < NRP; p++)
        for (int q = 0; q < NRP; q++)
          if (frz[q] && RGRP[q] == RGRP[p]) r_req[p] = '0;

      r_en = r_req;
      // The slice's guest grants depend on the raw requests, not the guest's own RAM asker.
      if (|r_req[1]) begin r_en[5] = '0; r_en[7] = '0; end
      else if (|r_req[5]) r_en[7] = '0;
      #1;
      // COLL cannot replay; if its raw request loses, the slice suppresses its grant.
      if (blk_x[7]) r_en[7] = '0;
      #1;
      emit_trace(cyc);

      // As in tb_tmem: fast edge aligned with the core edge, then one between edges.
      clk = 1; clk_f = 1; #1; clk_f = 0; #1;
      clk = 0; clk_f = 1; #1; clk_f = 0; #1;
    end
    $finish;
  end
endmodule
