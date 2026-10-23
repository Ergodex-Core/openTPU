// The MXU's array side (docs/vla_frequency_plan.md 6i): what otpu_mxu's compute pipeline was
// between its S0 registers and the result FIFO, in a hierarchy of its own so that the F2 build
// can place it in another SLR than the issuer, the drain and the TMEM (boards/f2/build/constraints/
// cl_pnr_user_hbm.tcl pb_otpu_mxu, with the chunk and scale FIFO RAMs and the ACT RAM): the
// chunk's decode (S1) and the shifts (the groups' B multipliers), the weight skews and hops, the
// PEs (otpu_pe_pk / otpu_pe), the column ends (otpu_mxce) with their deskews, and the result
// row's assembly (rowi / rsa / rsw, PACK's second row) and composition (rowp, otpu_mxu's result
// FIFO row). ARR 0 is otpu_mxu's pipeline exactly, with nothing between the S0 registers and
// the array's logic; ARR 1 puts every signal that crosses the hierarchy on a flip-flop at both
// ends: the S0 bundle (the popped chunk's registers, the scale entries, the per-command
// constants) goes through two input register stages (NX), the ACT RAM's read, which the S0
// bundle must meet at the act skews, is requested through otpu_mxu's registered act_* outputs
// and otpu_actram's input stage (RREG: the same two cycles), the chunk and scale FIFO RAMs'
// read registers sit in the array's region (their write pins and read addresses come from
// registers in otpu_mxu's), and the row (rowp), its push (rv, rvb) and push_n (the push a
// cycle ahead, otpu_mxu's result FIFO write-enable replicas) leave through output registers:
// three cycles more from the pop to the result FIFO (LROW + 3), nothing else changes. The PEs'
// select registers take the per-command constants one stage before the data bundle's last
// (selp / hip / hqp: what c_sel / c_hi / c_hq were to them), so each DSP's INMODE register still
// holds the data's command (the simulation checks it against sel0 / hi0 as before).
module otpu_mxu_arr
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int MCOLS = 8,
  parameter int CL    = 16,
  parameter bit T2    = 1'b0,
  parameter bit PACK  = 1'b0,
  parameter bit ARR   = 1'b0          // the registered boundary (above)
) (
  input  logic                      clk,
  input  logic                      rst,
  // the S0 bundle (otpu_mxu's registers)
  input  logic [8:0]                m0_i,     // cm_t (9 bits)
  input  logic [D*8-1:0]            w0_i,
  input  logic [D*8-1:0]            w0b_i,
  input  logic                      cu0_i, rs0_i, pr0_i, t20_i, qd0_i, pk0_i,
  input  logic [1:0]                wf0_i,
  input  logic [MCOLS-1:0]          hi0_i, hq0_i,
  input  logic [7:0]                M0_i,
  input  logic [MCOLS-1:0][1:0]     sel0_i,
  input  logic [95:0]               sr0_i,
  input  logic [63:0]               sr1_i,
  input  logic [39:0]               sr0b_i,
  // the pop head's constants as the PEs take them without ARR (c_sel / c_hi / c_hq)
  input  logic [MCOLS-1:0][1:0]     c_sel,
  input  logic [MCOLS-1:0]          c_hi, c_hq,
  // the ACT RAM's registered read (aligned with the bundle: otpu_actram RREG = ARR)
  input  logic [MCOLS*D*8-1:0]      act_data,
  input  logic [MCOLS*64-1:0]       act_scale,
  // the result FIFO's row and push
  output logic                      rv, rvb,  // a row (PACK: a pair's second row) pushes now
  output logic                      push_n,   // rv || rvb a cycle ahead
  output logic [MCOLS*80+32+(PACK?1:0)-1:0] rowp
);
  localparam logic [1:0] WF_W8 = 2'd0, WF_NV = 2'd3;
  localparam int CLS = (D / 8 < CL) ? D / 8 : CL;
  localparam int CPG = D / 8 / CLS;
  localparam int LROW = CLS + MCOLS + 5;
  localparam int PWD = PACK ? 48 : 24;
  localparam int RWX = PACK ? 1 : 0;
  localparam int RW_ = MCOLS * 48 + MCOLS * 32 + 32 + RWX;
  localparam int NX = ARR ? 2 : 0;
  typedef struct packed {
    logic       v;
    logic       last;
    logic       first;
    logic       h;
    logic       o;
    logic       o2, o3;
    logic [1:0] q;
  } cm_t;
  // ---- the input stages (NX of them; none without ARR): the bundle, as one vector
  localparam int BW_ = 9 + 2 * D * 8 + 6 + 2 + 2 * MCOLS + 8 + 2 * MCOLS + 96 + 64 + 40;
  logic [BW_-1:0] b_in;
  logic [NX:0][BW_-1:0] b_st;
  assign b_in = {m0_i, w0_i, w0b_i, cu0_i, rs0_i, pr0_i, t20_i, qd0_i, pk0_i, wf0_i, hi0_i, hq0_i,
                 M0_i, sel0_i, sr0_i, sr1_i, sr0b_i};
  assign b_st[0] = b_in;
  for (genvar k = 0; k < NX; k++) begin : g_st
    always_ff @(posedge clk) b_st[k + 1] <= b_st[k];
  end
  cm_t                 m0;
  logic [D*8-1:0]      w0, w0b;
  logic                cu0, rs0, pr0, t20, qd0, pk0;
  logic [1:0]          wf0;
  logic [MCOLS-1:0]    hi0, hq0;
  logic [7:0]          M0;
  logic [MCOLS-1:0][1:0] sel0;
  logic [95:0]         sr0;
  logic [63:0]         sr1;
  logic [39:0]         sr0b;
  assign {m0, w0, w0b, cu0, rs0, pr0, t20, qd0, pk0, wf0, hi0, hq0, M0, sel0, sr0, sr1, sr0b} = b_st[NX];
  // the PEs' selects: one stage before the data's last (the constants of the data's command),
  // or the pop head's constants themselves without ARR
  logic [MCOLS-1:0][1:0] selp;
  logic [MCOLS-1:0]      hip, hqp;
  if (ARR) begin : g_selp
    logic [MCOLS-1:0][1:0] s1;
    logic [MCOLS-1:0]      h1, q1;
    logic [7:0]            mm;
    logic [95:0] x0; logic [63:0] x1; logic [39:0] xb;
    logic [D*8-1:0] xw, xwb;
    cm_t xm;
    logic xa, xb_, xc, xd, xe, xf;
    logic [1:0] xg;
    assign {xm, xw, xwb, xa, xb_, xc, xd, xe, xf, xg, h1, q1, mm, s1, x0, x1, xb} = b_st[NX - 1];
    assign selp = s1, hip = h1, hqp = q1;
  end else begin : g_selc
    assign selp = c_sel, hip = c_hi, hqp = c_hq;
  end
  logic [MCOLS*D*8-1:0] a0;
  assign a0 = act_data;
  wire  w40 = (wf0 != WF_W8);
  // 4-bit elements: nibble i of the chunk half; int4 two's complement, E2M1 as twice its value
  // ({0, 1, 2, 3, 4, 6, 8, 12}, sign in bit 3); NVFP4 elements are E2M1's
  function automatic logic [7:0] dec4(input logic [3:0] c, input logic fp);
    if (!fp) return {{4{c[3]}}, c};
    case (c)
      4'd5: return 8'd6;
      4'd6: return 8'd8;
      4'd7: return 8'd12;
      4'd8: return 8'd0;
      4'd9: return -8'sd1;
      4'd10: return -8'sd2;
      4'd11: return -8'sd3;
      4'd12: return -8'sd4;
      4'd13: return -8'sd6;
      4'd14: return -8'sd8;
      4'd15: return -8'sd12;
      default: return 8'(c);
    endcase
  endfunction
  // the weight of position i for a column that takes the high block (PAIR) or the high half
  function automatic logic [7:0] wsel(input logic [D*8-1:0] w, input int i, input logic hb,
                                      input logic [1:0] wf);
    if (wf == WF_W8) return w[i*8 +: 8];
    return dec4(w[(hb ? D*4 : 0) + 4*i +: 4], wf != 2'd1);
  endfunction
  // t2: element i of block q of the chunk (bits 2D q + 2i), the integer c - 1 in {-1, 0, 1, 2}
  // ({6{c == 0}, c - 1}: the 2-bit difference sign-extended)
  function automatic logic [7:0] wt2(input logic [D*8-1:0] w, input int i, input logic [1:0] q);
    logic [1:0] c;
    c = w[2*D*q + 2*i +: 2];
    return {{6{c == 2'd0}}, c - 2'd1};
  endfunction
  // ---- the shifts: per block its eight groups' W - dw and m (S1), per column and group
  // B = sh >= 0 ? m << sh : 0 with sh = W - dw - da (S2), skewed to the column end
  typedef struct packed {
    logic [7:0][10:0] s;       // W - dw, signed
    logic [7:0][5:0]  m;
  } bdec_t;
  function automatic bdec_t bdec(input logic [63:0] b, input logic [1:0] wf, input logic u);
    bdec_t r;
    logic [10:0] w;
    w = (wf == WF_W8) ? 11'd16 : (wf == WF_NV) ? 11'd13 : 11'd11;
    for (int g = 0; g < 8; g++) begin
      logic [3:0] ex;
      ex = b[8*g + 3 +: 4];
      if (u) begin
        r.s[g] = w; r.m[g] = 6'd1;
      end else if (wf == WF_W8) begin
        r.s[g] = w - 11'(b[7:0]); r.m[g] = 6'd1;
      end else if (wf == WF_NV) begin
        // E4M3 s = {0, E, M}: m = E ? 8 + M : M, dw = 15 - max(E, 1): W - dw = max(E, 1) - 2
        r.m[g] = (ex != 4'd0) ? {3'b001, b[8*g +: 3]} : {3'b000, b[8*g +: 3]};
        r.s[g] = 11'((ex == 4'd0) ? 4'd1 : ex) - 11'd2;
      end else begin
        r.s[g] = w - 11'(b[7:0]); r.m[g] = b[8 + 6*(g/2) +: 6];
      end
    end
    return r;
  endfunction
  // the low block: the advance's half (4-bit without PAIR) or the entry's low block. t2: the
  // entry's words w0..w3 ({Sw, w1, w0}, {w3, w2}); the low stream's is word q, the high's q | 1,
  // QUAD's third and fourth streams w2 and w3
  wire [3:0][31:0] tw0 = {sr1[63:32], sr1[31:0], sr0[63:32], sr0[31:0]};
  wire [63:0] blo0 = (T2 && t20) ? {32'd0, tw0[m0.q]} : (w40 && !pr0 && m0.h) ? sr1 : sr0[63:0];
  // (PACK: the high stream is the pair's second row, its dw byte in the slot's bank 1 entry)
  wire [63:0] bhi0 = (PACK && pk0) ? {56'd0, sr0b[7:0]} :
                     (T2 && t20) ? {32'd0, tw0[{m0.q[1], 1'b1}]} : sr1;
  bdec_t dlo1, dhi1, dq21, dq31;
  logic  vlo1, vhi1, vq21, vq31;
  logic  pk1;
  logic [MCOLS-1:0][3:0][7:0] da1;
  logic [MCOLS-1:0] hi1, hq1;
  always_ff @(posedge clk) begin
    dlo1 <= bdec(blo0, wf0, cu0 || rs0);
    dhi1 <= bdec(bhi0, wf0, cu0 || rs0);
    vlo1 <= m0.v;
    vhi1 <= m0.v && m0.o;
    pk1 <= pk0;
    for (int j = 0; j < MCOLS; j++) begin
      da1[j] <= act_scale[j*64 + 32 +: 32];
      hi1[j] <= hi0[j];
    end
  end
  if (T2) begin : g_q1
    always_ff @(posedge clk) begin
      dq21 <= bdec({32'd0, tw0[2]}, wf0, cu0 || rs0);
      dq31 <= bdec({32'd0, tw0[3]}, wf0, cu0 || rs0);
      vq21 <= m0.v && m0.o2;
      vq31 <= m0.v && m0.o3;
      hq1 <= hq0;
    end
  end else begin : g_nq1
    assign dq21 = '0, dq31 = '0;
    assign vq21 = 1'b0, vq31 = 1'b0;
    assign hq1 = '0;
  end
  logic [MCOLS-1:0][7:0][16:0] bm2;
  always_ff @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      for (int g = 0; g < 8; g++) begin
        logic [10:0] sh;
        logic [10:0] s;
        logic [5:0]  m;
        logic        v;
        // (QUAD: one of four, by the column's hq and hi)
        if (T2 && hq1[j]) begin
          s = hi1[j] ? dq31.s[g] : dq21.s[g];
          m = hi1[j] ? dq31.m[g] : dq21.m[g];
          v = hi1[j] ? vq31 : vq21;
        end else begin
          s = hi1[j] ? dhi1.s[g] : dlo1.s[g];
          m = hi1[j] ? dhi1.m[g] : dlo1.m[g];
          v = hi1[j] ? vhi1 : vlo1;
        end
        sh = s - 11'(da1[j][g / 2]);
        bm2[j][g] <= (v && !sh[10]) ? 17'(17'(m) << sh[4:0]) : 17'd0;
      end
  // PACK: lane b's B: a packed pass's second row's (the high stream's s / m, 0 when the row
  // does not exist); unpacked, the column's own (bm2: a product in lane b is PAIR's / QUAD's
  // high block, whose s / m bm2 selects by hi1; in lane a lane b's chain end is exactly 0)
  logic [MCOLS-1:0][7:0][16:0] bm2b;
  if (PACK) begin : g_bmb
    always_ff @(posedge clk)
      for (int j = 0; j < MCOLS; j++)
        for (int g = 0; g < 8; g++) begin
          logic [10:0] sh;
          sh = dhi1.s[g] - 11'(da1[j][g / 2]);
          if (pk1) bm2b[j][g] <= (vhi1 && !sh[10]) ? 17'(17'(dhi1.m[g]) << sh[4:0]) : 17'd0;
          else begin
            logic [10:0] sh2;
            logic [10:0] s;
            logic [5:0]  m;
            logic        v;
            if (T2 && hq1[j]) begin
              s = hi1[j] ? dq31.s[g] : dq21.s[g];
              m = hi1[j] ? dq31.m[g] : dq21.m[g];
              v = hi1[j] ? vq31 : vq21;
            end else begin
              s = hi1[j] ? dhi1.s[g] : dlo1.s[g];
              m = hi1[j] ? dhi1.m[g] : dlo1.m[g];
              v = hi1[j] ? vhi1 : vlo1;
            end
            sh2 = s - 11'(da1[j][g / 2]);
            bm2b[j][g] <= (v && !sh2[10]) ? 17'(17'(m) << sh2[4:0]) : 17'd0;
          end
        end
  end else begin : g_nbmb
    assign bm2b = '0;
  end
  // the row's Sw (UNIT: 1) and each column's Sa * 2^-W, with every advance (the row's last
  // carries them to the result FIFO): fp_mul's flush and rounding at the bottom of the range
  wire [31:0] sw0 = cu0 ? F_ONE : sr0[95:64];
  wire [31:0] sw0b = cu0 ? F_ONE : sr0b[39:8];                 // PACK: the pair's second row's
  wire [31:0] wpow = (wf0 == WF_W8) ? 32'h3780_0000 : (wf0 == WF_NV) ? 32'h3900_0000
                                                                     : 32'h3A00_0000;
  // x * 2^-k as fp_mul computes it (flushed operands and result, the canonical NaN), without a
  // multiplier: the exponent less k; a result under the normal range flushes to a signed zero,
  // except the one that rounds up to the smallest normal (exponent k, fraction all ones)
  function automatic f32_t fp_pow2m(input f32_t x, input logic [4:0] k);
    logic [8:0] e;
    if (x[30:23] == 8'd0) return {x[31], 31'd0};
    if (x[30:23] == 8'hFF) return (x[22:0] != 0) ? F_NAN : x;
    e = 9'(x[30:23]) - 9'(k);
    if (!e[8] && e != 0) return {x[31], e[7:0], x[22:0]};
    if (e == 0 && x[22:0] == 23'h7F_FFFF) return {x[31], 8'd1, 23'd0};
    return {x[31], 31'd0};
  endfunction
  wire [4:0] wk = (wf0 == WF_W8) ? 5'd16 : (wf0 == WF_NV) ? 5'd13 : 5'd11;
  logic [31:0] sw1, sw1b;
  logic [MCOLS-1:0][31:0] sap1;
  always_ff @(posedge clk) begin
    sw1 <= sw0;
    sw1b <= sw0b;
    for (int j = 0; j < MCOLS; j++) sap1[j] <= fp_pow2m(act_scale[j*64 +: 32], wk);
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      if (fp_pow2m(act_scale[j*64 +: 32], wk) != fp_mul(act_scale[j*64 +: 32], wpow))
        $fatal(1, "otpu_mxu: Sa %h * 2^-%0d is not fp_mul's", act_scale[j*64 +: 32], wk);
`endif

  // ---- the systolic array (docs/mxu_systolic.md). The chunk is decoded once (S1), position i
  // is delayed by its chain stage k = i % CLS, and the weights then move one register hop per
  // column: column j sees them j cycles after column 0. Both streams flow: the low block (or the
  // 4-bit half of a non-PAIR advance) and PAIR's high block; each column takes one in its DSPs'
  // pre-adder (A = low, D = high, INMODE by its per-command hi0[j]). Column j's activation byte i
  // is delayed k + j (shift registers) into the DSP's B register. Along D: chains of CLS products,
  // one per DSP48 (otpu_pe: M register, P = PCIN + M); a chain's end (S0 + CLS + 2 + j, in the
  // column end's input registers) is one group's sum. Exact integers.
  // T2 adds two streams, QUAD's blocks 4c+2 and 4c+3 (t2 the low stream block q of the chunk,
  // the high q | 1): every hop carries four, and in front of each PE's A / D a 2:1 select by the
  // column's hq (a per-command constant) picks the pair {4c+3, 4c+2} or {high, low}, so a
  // column's DSP still sees two streams and its INMODE select (hi) as before.
  localparam int WS = T2 ? 32 : 16;                   // the streams' bits
  logic [D-1:0][WS-1:0] wd;                           // S1: {[4c+3, 4c+2,] high block, low block / half}
  always_ff @(posedge clk)
    for (int i = 0; i < D; i++) begin
      logic [7:0] lo, hi;
      lo = (T2 && t20) ? wt2(w0, i, m0.q) : wsel(w0, i, m0.h, wf0);
      // (PACK: a packed pass's high stream is the pair's second row's chunk, bank 1)
      hi = (PACK && pk0) ? w0b[i*8 +: 8] :
           (T2 && t20) ? wt2(w0, i, {m0.q[1], 1'b1}) : wsel(w0, i, 1'b1, wf0);
      wd[i] <= WS'({wt2(w0, i, 2'd3), wt2(w0, i, 2'd2), hi, lo});
    end
  logic [D-1:0][WS-1:0] wc [MCOLS];                  // the streams at column j
  for (genvar i = 0; i < D; i++) begin : g_wsk
    otpu_skew #(.W(WS), .N(i % CLS)) u_w (.clk, .en(1'b1), .d(wd[i]), .q(wc[0][i]));
  end
  always_ff @(posedge clk)
    for (int j = 1; j < MCOLS; j++) wc[j] <= wc[j-1];
  logic [7:0] as_k [MCOLS][D];
  logic signed [PWD-1:0] preg [MCOLS][D];             // running sums (the DSPs' P registers;
                                                      // a chain's last: its P, unregistered;
                                                      // PACK: 48 bits, the two lanes)
  logic [47:0] pc [MCOLS][D];                         // their cascade outputs
`ifndef SYNTHESIS
  bit hi_v;                                           // hi0 and the PEs' selects are loaded
  initial hi_v = 1'b0;
  always @(posedge clk) hi_v <= 1'b1;
`endif
  for (genvar j = 0; j < MCOLS; j++) begin : g_ask
    for (genvar i = 0; i < D; i++) begin : g_p
      otpu_skew #(.W(8), .N(i % CLS + j)) u_a (.clk, .en(1'b1), .d(a0[(j*D + i)*8 +: 8]),
                                             .q(as_k[j][i]));
      // columns past the first register their weights in the DSP (WREG), from the previous
      // column's hop: a copy of wc[j]
      localparam int JW = (j == 0) ? 0 : j - 1;
      // (T2: column 0 is never QUAD's hq: c_hq[0] is 0, so its unregistered weights see no select)
      wire [15:0] wp = (T2 && j != 0 && hqp[j]) ? 16'(wc[JW][i] >> 16) : wc[JW][i][15:0];
      if (PACK) begin : g_pk
        // the packed position: the low stream on A, the high stream << 19 on D, the column's
        // lane select in the INMODE register (both lanes for a packed pass, one otherwise)
        otpu_pe_pk #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0)) u_pe (
          .clk, .en(1'b1), .act(as_k[j][i]), .wa(wp[7:0]), .wb(wp[15:8]),
          .sel(selp[j]),
          .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
          .p(preg[j][i]));
`ifndef SYNTHESIS
        // the DSP's select register is sel0[j] (both are c_sel[j], loaded every cycle)
        always @(posedge clk) begin
          if (hi_v && u_pe.selr != sel0[j])
            $fatal(1, "otpu_mxu: PE %0d.%0d's select %0d is not sel0 %0d", j, i, u_pe.selr, sel0[j]);
          if (hi_v && j != 0 && {u_pe.wbr, u_pe.war} !=
              ((T2 && hq0[j]) ? 16'(wc[j][i] >> 16) : wc[j][i][15:0]))
            $fatal(1, "otpu_mxu: PE %0d.%0d's weights are not wc", j, i);
        end
`endif
      end else begin : g_un
        otpu_pe #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0)) u_pe (
          .clk, .en(1'b1), .act(as_k[j][i]), .wlo(wp[7:0]), .whi(wp[15:8]),
          .sel(hip[j]),
          .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
          .p(preg[j][i]));
`ifndef SYNTHESIS
        // the DSP's select register is hi0[j] (both are c_hi[j], loaded every cycle)
        always @(posedge clk) begin
          if (hi_v && u_pe.selr != hi0[j])
            $fatal(1, "otpu_mxu: PE %0d.%0d's select %0d is not hi0 %0d", j, i, u_pe.selr, hi0[j]);
          if (hi_v && j != 0 && {u_pe.whr, u_pe.wlr} !=
              ((T2 && hq0[j]) ? 16'(wc[j][i] >> 16) : wc[j][i][15:0]))
            $fatal(1, "otpu_mxu: PE %0d.%0d's weights are not wc", j, i);
        end
`endif
      end
    end
  end

  // ---- column ends: group g's chain ends (CPG of them: the first on A, the second on D) with
  // their B (skewed from S2 to the column end's input registers, S0 + CLS + 2 + j), the row's
  // first block at the accumulator's OPMODE register (two cycles later), the row sum in the
  // accumulator CLS + 6 + j edges after S0, deskewed to S0 + LROW.
  // PACK: the chain end P = L + H 2^19 is split in the wires (docs/mxu_packing.md 3.2): lane
  // a's column end takes L = sext19(P[18:0]) on A; lane b's (otpu_mxce CPG 2) takes P[42:19] on
  // A and the borrow P[18] on D, so H = P[47:19] + P[18] forms in its pre-adder. Each lane has
  // its own B (bm2, bm2b) and accumulator (acc_al, acc_bl), the first / last skews shared.
  logic [MCOLS-1:0][47:0] acc_al, acc_bl;
  for (genvar j = 0; j < MCOLS; j++) begin : g_col_end
    logic [7:0][23:0] ca, cd;
    logic [7:0][16:0] bmd;
    logic             fst;
    logic [47:0]      acc;
    always_comb
      for (int g = 0; g < 8; g++) begin
        ca[g] = PACK ? {{5{preg[j][g * CPG * CLS + CLS - 1][18]}},
                        preg[j][g * CPG * CLS + CLS - 1][18:0]}
                     : 24'(preg[j][g * CPG * CLS + CLS - 1]);
        cd[g] = (CPG == 2) ? 24'(preg[j][(g * CPG + 1) * CLS + CLS - 1]) : 24'd0;
      end
    otpu_skew #(.W(8 * 17), .N(CLS - 1 + j)) u_bm (.clk, .en(1'b1), .d(bm2[j]), .q(bmd));
    otpu_skew #(.W(1), .N(CLS + 4 + j)) u_f (.clk, .en(1'b1), .d(m0.v && m0.first), .q(fst));
    otpu_mxce #(.CPG(CPG)) u_ce (.clk, .a(ca), .d(cd), .bm(bmd), .first(fst), .acc);
    otpu_skew #(.W(48), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(1'b1), .d(acc), .q(acc_al[j]));
    if (PACK) begin : g_b
      logic [7:0][23:0] cb, db;
      logic [7:0][16:0] bmbd;
      logic [47:0]      accb;
      always_comb
        for (int g = 0; g < 8; g++) begin
          cb[g] = preg[j][g * CLS + CLS - 1][42:19];
          db[g] = 24'(preg[j][g * CLS + CLS - 1][18]);
        end
      otpu_skew #(.W(8 * 17), .N(CLS - 1 + j)) u_bmb (.clk, .en(1'b1), .d(bm2b[j]), .q(bmbd));
      otpu_mxce #(.CPG(2)) u_ceb (.clk, .a(cb), .d(db), .bm(bmbd), .first(fst), .acc(accb));
      otpu_skew #(.W(48), .N(MCOLS - 1 - j)) u_dskb (.clk, .en(1'b1), .d(accb), .q(acc_bl[j]));
    end else begin : g_nb
      assign acc_bl[j] = '0;
    end
  end
  // the row's last block, PAIR and M at S0 + LROW; Sw and the Sa' from S1. PACK: packed, and
  // the pair's second row exists (tl_o); its Sw
  logic             tl_v, tl_pr, tl_qd, tl_pk, tl_o;
  logic [7:0]       tl_M;
  logic [31:0]      tl_sw, tl_swb;
  logic [MCOLS-1:0][31:0] tl_sa;
  otpu_delay #(.W(1 + 1 + 8), .N(LROW)) u_tl (.clk, .en(1'b1), .d({m0.v && m0.last, pr0, M0}),
                                             .q({tl_v, tl_pr, tl_M}));
  if (T2) begin : g_tlq
    otpu_delay #(.W(1), .N(LROW)) u_tq (.clk, .en(1'b1), .d(qd0), .q(tl_qd));
  end else begin : g_ntlq
    assign tl_qd = 1'b0;
  end
  otpu_skew #(.W(32 + MCOLS * 32), .N(LROW - 1)) u_ts (.clk, .en(1'b1), .d({sw1, sap1}),
                                                      .q({tl_sw, tl_sa}));
  if (PACK) begin : g_tlp
    otpu_delay #(.W(2), .N(LROW)) u_tp (.clk, .en(1'b1), .d({pk0, m0.o}), .q({tl_pk, tl_o}));
    otpu_skew #(.W(32), .N(LROW - 1)) u_tsb (.clk, .en(1'b1), .d(sw1b), .q(tl_swb));
  end else begin : g_ntlp
    assign tl_pk = 1'b0, tl_o = 1'b0, tl_swb = '0;
  end
  // PAIR: row j (j < M <= MCOLS/2) adds its partner column j + M's integer; QUAD (T2): row j
  // (j < M <= MCOLS/4) adds columns j + M, j + 2M and j + 3M's, two adders deep. Registered for
  // the result FIFO (rv: push). PACK: a column's integer is its two lanes' sum (one of them 0
  // for an unpacked command); a packed pass's row is the a lanes', then its pair's second row
  // the b lanes' a cycle later (rvb: pushed with a skip bit when the row does not exist)
  logic             rv_c, rvb0, rvb_c, skb1, skb2;
  logic [MCOLS-1:0][47:0] rowi, rowb1, rowb2;
  logic [MCOLS-1:0][31:0] rsa, rsab2;
  logic [31:0]      rsw, rswb1, rswb2;
  always_ff @(posedge clk) begin
    rv_c <= !rst && tl_v;
    for (int j = 0; j < MCOLS; j++) begin
      logic [MCOLS-1:0][47:0] x;
      logic [47:0] pb, p2, p3;
      for (int k = 0; k < MCOLS; k++) x[k] = (PACK && !tl_pk) ? acc_al[k] + acc_bl[k] : acc_al[k];
      pb = '0; p2 = '0; p3 = '0;
      for (int m = 1; m <= MCOLS / 2; m++)
        if (j < MCOLS / 2 && j + m < MCOLS && tl_M == 8'(m)) pb = x[j + m];
      for (int m = 1; m <= MCOLS / 4; m++)
        if (j < MCOLS / 4 && tl_M == 8'(m)) begin
          p2 = x[j + 2 * m];
          p3 = x[j + 3 * m];
        end
      if (T2 && tl_qd && j < MCOLS / 4) rowi[j] <= (x[j] + pb) + (p2 + p3);
      else rowi[j] <= (tl_pr && j < MCOLS / 2) ? x[j] + pb : x[j];
    end
    rsa <= tl_sa;
    rsw <= tl_sw;
  end
  if (PACK) begin : g_rowb
    always_ff @(posedge clk) begin
      rvb0 <= !rst && tl_v && tl_pk;
      rvb_c <= !rst && rvb0;
      rowb1 <= acc_bl; rowb2 <= rowb1;
      rswb1 <= tl_swb; rswb2 <= rswb1;
      skb1 <= !tl_o; skb2 <= skb1;
      rsab2 <= rsa;
    end
`ifndef SYNTHESIS
    always @(posedge clk) if (!rst && rv_c && rvb_c)
      $fatal(1, "otpu_mxu: a packed pass's second row and another pass's first push together");
`endif
  end else begin : g_nrowb
    assign rvb0 = 1'b0, rvb_c = 1'b0, skb1 = 1'b0, skb2 = 1'b0;
    assign rowb1 = '0, rowb2 = '0, rsab2 = '0, rswb1 = '0, rswb2 = '0;
  end

  // ---- the row for the result FIFO (otpu_mxu's rowp), and the push: through a register with ARR
  logic [RW_-1:0] rowp_c;
  always_comb begin
    if (PACK) rowp_c = rvb_c ? RW_'({skb2, rswb2, rsab2, rowb2}) : RW_'({1'b0, rsw, rsa, rowi});
    else rowp_c = RW_'({rsw, rsa, rowi});
  end
  if (ARR) begin : g_oreg
    always_ff @(posedge clk) begin
      rv <= rv_c; rvb <= rvb_c;
      if (rv_c || rvb_c) rowp <= rowp_c;
    end
    assign push_n = rv_c || rvb_c;           // what rv || rvb will be next cycle
  end else begin : g_owire
    assign rv = rv_c, rvb = rvb_c, rowp = rowp_c;
    assign push_n = tl_v || rvb0;
  end
endmodule
