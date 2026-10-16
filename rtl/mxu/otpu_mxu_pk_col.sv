// One physical column of the systolic MXU with packed positions (docs/mxu_packing.md): NCH
// chains of CLS packed PEs (otpu_pe_pk: two weight rows a, b of the same position, one
// activation, two products a DSP), the lanes split at each chain end, and two column ends
// (otpu_mxce) that apply the groups' B = m << sh and accumulate the row's blocks: acc_a is
// weight row a's integer, acc_b row b's. PACK 0 builds the same column unpacked (otpu_pe, one
// weight row, one column end): the out-of-context comparison (tools/mxu_pk_ooc.tcl). Standalone,
// in no file list: otpu_mxu's array is unchanged (the integration is phase 3, docs/mxu_packing.md
// 6); tests/test_mxu_pk.py drives it through sim/verilator/tb_mxu_pk.sv.
//
// Timing from the cycle t a block's operands are at the ports (otpu_mxu's S0: the ACT RAM's
// registered read and the chunk register). The activation is registered once (S1, as the ACT
// RAM's a0 is a cycle ahead of the decoded weights), then skewed by its chain stage k = i % CLS
// into the DSP's B register; the weights twice (S1, then S2's decode register in front of the
// skew, or, WREG, the DSP's A / D registers after it: otpu_mxu's column 0 and its later
// columns), then skewed by k. Stage k's product is in M at t + k + 3, its P at t + k + 4; the
// chain's end (its last P, unregistered) is in cycle t + CLS + 2, the column ends' input
// registers load at E = t + CLS + 3 and the accumulators at E + 4 = t + CLS + 7 (otpu_mxu: CLS +
// 6 + j edges after S0, j = 0). bm (per group, per lane; 0 on a cycle without a block, as
// otpu_mxu's bm2) and first / last travel with the block: bm to the column ends' input
// registers, first to the accumulator's OPMODE register (E + 3), av with the accumulators.
//
// The unpack. A chain's packed sum is P = L + H * 2^19 with L = sum wa * act and H = sum wb *
// act over its CLS positions; |L|, |H| <= CLS * 127 * 127 = 258,064 < 2^18 for operands in
// [-127, 127] (quant.py's int8; docs/mxu_packing.md 3 for -128). So L = sext(P[18:0]) is wiring
// (the low column end's A), and H = P[47:19] + P[18] (the borrow a negative L takes from the
// high lane) is formed in the high column end's pre-adder: its A takes P[42:19], its D the bit
// P[18] (otpu_mxce CPG 2: (a + d) * bm), so the fix-up costs no fabric logic and no cycle, and
// the chain's last P feeds the column ends' input registers directly, as in otpu_mxu. The split
// has to be here, at the chain end: the column end multiplies each lane by its own B (the two
// weight rows have their own block offsets dw) and sums eight chains and up to 128 blocks, far
// past the lane's 2^18.
module otpu_mxu_pk_col #(
  parameter int D    = 128,
  parameter int CLS  = 16,          // products per chain (the lane bound: CLS * 127^2 < 2^18)
  parameter int NCH  = D / CLS,     // chains built (a test's single chain: 1; the others' groups are 0)
  parameter bit PACK = 1'b1,        // 0: the unpacked column (otpu_pe, acc_a only)
  parameter bit WREG = 1'b0         // the weights in the DSPs' A / D registers (otpu_mxu's columns past the first)
) (
  input  logic                   clk,
  input  logic                   v,          // a block (an advance) at the ports
  input  logic                   first,      // the row's first block
  input  logic                   last,       // the row's last block: av with its accumulators
  input  logic [D-1:0][7:0]      act,
  input  logic [D-1:0][7:0]      wa,         // weight row a's block
  input  logic [D-1:0][7:0]      wb,         // weight row b's (PACK)
  input  logic [7:0][16:0]       bma,        // group g's B = m << sh for row a ...
  input  logic [7:0][16:0]       bmb,        // ... and for row b
  output logic [47:0]            acc_a,
  output logic [47:0]            acc_b,
  output logic                   av
);
  initial if (D != 8 * CLS) $fatal(1, "otpu_mxu_pk_col: D %0d is not 8 chains of %0d", D, CLS);
  initial if (NCH < 1 || NCH > 8) $fatal(1, "otpu_mxu_pk_col: NCH %0d", NCH);
  initial if (CLS * 127 * 127 >= (1 << 18)) $fatal(1, "otpu_mxu_pk_col: CLS %0d breaks the lane", CLS);

  // S1: the block's operands registered; S2: the weights again (the decode register's place)
  logic [D-1:0][7:0] a1, wa1, wb1, wa2, wb2;
  logic [7:0][16:0]  bma1, bmb1;
  logic              f1, l1;
  always_ff @(posedge clk) begin
    a1 <= act; wa1 <= wa; wb1 <= wb;
    wa2 <= wa1; wb2 <= wb1;
    bma1 <= v ? bma : '0; bmb1 <= v ? bmb : '0;
    f1 <= v && first; l1 <= v && last;
  end
  wire [D-1:0][7:0] was = WREG ? wa1 : wa2, wbs = WREG ? wb1 : wb2;

  // the chains: position i = c * CLS + k, its operands skewed by k
  logic [NCH-1:0][47:0] pend;                        // chain c's end (its last P)
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic [47:0] pc [CLS];
    for (genvar k = 0; k < CLS; k++) begin : g_k
      localparam int I = c * CLS + k;
      logic [7:0] ak, wak, wbk;
      otpu_skew #(.W(8), .N(k)) u_a (.clk, .en(1'b1), .d(a1[I]), .q(ak));
      otpu_skew #(.W(8), .N(k)) u_wa (.clk, .en(1'b1), .d(was[I]), .q(wak));
      if (PACK) begin : g_pk
        logic signed [47:0] pk;
        otpu_skew #(.W(8), .N(k)) u_wb (.clk, .en(1'b1), .d(wbs[I]), .q(wbk));
        otpu_pe_pk #(.FIRST(k == 0), .LAST(k == CLS - 1), .WREG(WREG)) u_pe (
          .clk, .en(1'b1), .act(ak), .wa(wak), .wb(wbk),
          .pcin((k == 0) ? 48'd0 : pc[(k == 0) ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
        if (k == CLS - 1) begin : g_end
          assign pend[c] = pk;
        end
      end else begin : g_un
        logic signed [23:0] pk;
        otpu_pe #(.FIRST(k == 0), .LAST(k == CLS - 1), .WREG(WREG)) u_pe (
          .clk, .en(1'b1), .act(ak), .wlo(wak), .whi(8'd0), .sel(1'b0),
          .pcin((k == 0) ? 48'd0 : pc[(k == 0) ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
        if (k == CLS - 1) begin : g_end
          assign pend[c] = 48'(pk);
        end
      end
    end
  end

  // the column ends: group g is chain g; the lanes unpacked by wiring (the high lane's borrow
  // on D). bm reaches the input registers with the chain ends, first the accumulator's OPMODE
  // register three cycles later, av the accumulators
  logic [7:0][23:0] ca, cb, db;
  always_comb
    for (int g = 0; g < 8; g++) begin
      ca[g] = '0; cb[g] = '0; db[g] = '0;
      if (g < NCH) begin
        ca[g] = PACK ? {{5{pend[g][18]}}, pend[g][18:0]} : pend[g][23:0];
        cb[g] = pend[g][42:19];
        db[g] = 24'(pend[g][18]);
      end
    end
  logic [7:0][16:0] bmad, bmbd;
  logic             fst;
  otpu_skew #(.W(8 * 17), .N(CLS + 1)) u_bma (.clk, .en(1'b1), .d(bma1), .q(bmad));
  otpu_skew #(.W(1), .N(CLS + 4)) u_f (.clk, .en(1'b1), .d(f1), .q(fst));
  otpu_skew #(.W(1), .N(CLS + 6)) u_l (.clk, .en(1'b1), .d(l1), .q(av));
  otpu_mxce #(.CPG(1)) u_ce_a (.clk, .a(ca), .d('0), .bm(bmad), .first(fst), .acc(acc_a));
  if (PACK) begin : g_ceb
    otpu_skew #(.W(8 * 17), .N(CLS + 1)) u_bmb (.clk, .en(1'b1), .d(bmb1), .q(bmbd));
    otpu_mxce #(.CPG(2)) u_ce_b (.clk, .a(cb), .d(db), .bm(bmbd), .first(fst), .acc(acc_b));
  end else begin : g_nceb
    assign acc_b = '0;
  end
endmodule
