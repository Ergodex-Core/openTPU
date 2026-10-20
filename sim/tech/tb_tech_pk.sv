// The packed PE (otpu_pe_pk, rtl/tech/<family>/otpu_tech_dsp.sv; otpu_mxu PACK) under random
// stimulus: every cycle prints both variants' (FIRST 0 / 1) P and PCOUT. Built once with
// rtl/tech/generic and once with rtl/tech/xilinxus on Vivado's DSP48E2 model (xsim:
// sim/tech/run_xsim_pk.sh, tests/test_tech.py); the traces must be equal. The operands cover the
// whole byte range (-128 included: the 27-bit pre-adder and the 48-bit sum wrap alike) and the
// enables are random. The stimulus starts after the unisim global reset (glbl's 100 ns GSR).
`timescale 1ns / 1ps
module tb_tech_pk;
  logic clk = 1'b0;
  logic en = 1'b0;
  logic [7:0] x0 = '0, x1 = '0, wlo = '0, whi = '0;
  logic sel = 1'b0;
  logic [47:0] pcin = '0;
  integer seed, cyc;
  logic [1:0][47:0] po, pco;
  for (genvar v = 0; v < 2; v++) begin : g_pe
    otpu_pe_pk #(.FIRST(v[0])) u (.clk, .en, .x0, .x1, .wlo, .whi, .sel, .pcin, .pcout(pco[v]),
                                  .p(po[v]));
  end
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    #200;
    for (cyc = 0; cyc < 4000; cyc++) begin
      en = (cyc < 16) ? 1'b1 : ($random(seed) % 8 != 0);
      x0 = $random(seed); x1 = $random(seed); wlo = $random(seed); whi = $random(seed);
      sel = $random(seed);
      pcin = {$random(seed), $random(seed)};
      #5 clk = 1'b1;
      #5 clk = 1'b0;
      if (cyc >= 16) begin                     // the pipelines filled with defined values
        $write("%0d", cyc);
        for (int v = 0; v < 2; v++) $write(" %012x %012x", po[v], pco[v]);
        $write("\n");
      end
    end
    $finish;
  end
endmodule
