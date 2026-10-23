// ACT RAM: the MXU's stationary operand. ROWS rows x BLOCKS blocks x D int8, plus one 64-bit
// scale record per (row, block): {da3, da2, da1, da0, Sa}, the row's fp32 Sa and the block's four
// sub-block offsets (QACT, docs/isa.md "Quantization"). The quantizer writes up to LANES consecutive bytes (and one scale) per
// cycle; its byte index is LANES aligned, so the bytes sit in one block. With `w_dup` (QACT DUP)
// the bytes and the scale also go to row w_row + w_off (w_off: the QACT's row count; DUP rows
// stay below MCOLS). The MXU reads one block of MCOLS rows per cycle: rows g*MCOLS ..
// g*MCOLS+MCOLS-1 of group `r_grp` (ROWS > MCOLS: the MXU replays a weight chunk for each group
// of an MM), block r_blk, or r_blk2 for the rows in r_hi (MM PAIR: the odd blocks).
// T2 (docs/ternary.md): `w_dup4` (QACT DUP4) writes rows w_row + c w_off, c = 0..3, and MM QUAD
// reads r_blk3 / r_blk4 for the rows in r_hq (r_hi picking the latter): four blocks a cycle.
// Without T2 neither is built.
//
// Block RAM: one memory per column j (rows j, j+MCOLS, ...) and piece p (bytes p*PB ..
// p*PB+PB-1: four 64-bit block RAMs at PB = 32), GROUPS*BLOCKS words of PB bytes with byte
// write enables, group-major. The read is registered and advances with the column's `ren` bit
// (otpu_mxu: the columns of the MM's rows), then registered once more (rdq / rsq, the RAM's
// output register: the MXU's S0, which its entry reaches through a register of its own since
// round 4, docs/wide_port.md); a read of a block written at the same edge returns the old bytes. Writes
// land three cycles after they are presented: the quantizer's write is registered as it arrives
// and again (LANES bytes, their index and row, the scale), so it crosses from the quantizer in
// that narrow form in two hops and the lane placement (LANES -> D bytes) starts at a flip-flop
// by the block RAMs, then registered again in place (bytes; the block address, column select
// and group once per column and piece). The quantizer reports done two cycles after its last
// write and the sequencer releases a dependent MM two cycles after that (otpu_quant), so a
// consumed read (r_use, the MXU's pop) never names a block whose write is still in flight: the
// simulation checks it.
module otpu_actram #(
  parameter int D      = 32,
  parameter int MCOLS  = 8,
  parameter int ROWS   = MCOLS,     // a multiple of MCOLS, MCOLS a power of two if more
  parameter int BLOCKS = 64,
  parameter int LANES  = 8,
  parameter bit T2     = 1'b0,      // QACT DUP4, MM QUAD's four blocks (docs/ternary.md)
  parameter bit DUP8   = 1'b0       // QACT DUP8: eight copies (docs/row_lanes.md)
) (
  input  logic                   clk,
  input  logic [LANES-1:0]       we,          // lane l writes byte w_idx + l
  input  logic [7:0]             w_row,
  input  logic [31:0]            w_idx,       // byte index within the row: block*D + i
  input  logic [LANES-1:0][7:0]  w_data,
  input  logic                   w_dup,
  input  logic                   w_dup4,      // T2: also rows w_row + 2 w_off, + 3 w_off
  input  logic                   w_dup8,      // DUP8: also rows w_row + 4 .. 7 w_off
  input  logic [7:0]             w_off,
  input  logic                   swe,
  input  logic [7:0]             s_row,
  input  logic [15:0]            s_blk,
  input  logic [63:0]            s_data,
  input  logic [MCOLS-1:0]       ren,         // per column
  input  logic                   r_use,       // this edge's read is consumed (simulation checks)
  input  logic [15:0]            r_blk,
  input  logic [15:0]            r_blk2,
  input  logic [MCOLS-1:0]       r_hi,
  input  logic [15:0]            r_blk3,      // T2 (QUAD): for the rows in r_hq, r_blk4 for
  input  logic [15:0]            r_blk4,      // those also in r_hi
  input  logic [MCOLS-1:0]       r_hq,
  input  logic [7:0]             r_grp,
  output wire  [MCOLS*D*8-1:0]   r_data,      // row j at [j*D*8 +: D*8], byte i at [+8i]
  output wire  [MCOLS*64-1:0]    r_scale
);
  localparam int BW = $clog2(BLOCKS);
  localparam int DW = $clog2(D);
  localparam int GROUPS = ROWS / MCOLS;
  localparam int GW = (GROUPS > 1) ? $clog2(GROUPS) : 1;
  localparam int PB = (D > 32) ? 32 : D;      // bytes per piece
  localparam int NP = D / PB;
  localparam int RAW = $clog2(GROUPS * BLOCKS);  // the RAMs' address bits
  initial if (D % LANES != 0) $fatal(1, "otpu_actram: LANES must divide D");
  initial if (D % PB != 0) $fatal(1, "otpu_actram: D must be a multiple of 32 above 32");
  initial if (ROWS % MCOLS != 0 || (GROUPS > 1 && (MCOLS & (MCOLS - 1)) != 0))
    $fatal(1, "otpu_actram: ROWS must be a multiple of MCOLS, a power of two if larger");
  // a row's column and group
  function automatic int col(input logic [7:0] r);
    return (MCOLS > 1) ? int'(r) % MCOLS : 0;
  endfunction
  function automatic logic [GW-1:0] grp(input logic [7:0] r);
    return (GROUPS > 1) ? GW'(int'(r) / MCOLS) : '0;
  endfunction
  // a write to row r (with dup, also to row r + off; dup4, T2: and r + 2 off, r + 3 off, all
  // below MCOLS, so in group 0 as row r + off) selects column j, in the group returned
  function automatic logic sel(input int j, input logic [7:0] r, input logic dup,
                               input logic dup4, input logic dup8, input logic [7:0] off);
    logic d4, d8;
    d8 = DUP8 && dup8;
    d4 = (T2 && dup4) || d8;
    return (col(r) == j) || ((dup || d4) && col(8'(r + off)) == j) ||
           (d4 && (col(8'(r + 2 * off)) == j || col(8'(r + 3 * off)) == j)) ||
           (d8 && (col(8'(r + 4 * off)) == j || col(8'(r + 5 * off)) == j ||
                   col(8'(r + 6 * off)) == j || col(8'(r + 7 * off)) == j));
  endfunction
  function automatic logic [GW-1:0] sgrp(input int j, input logic [7:0] r, input logic [7:0] off);
    return (col(r) == j) ? grp(r) : grp(8'(r + off));
  endfunction
  // the quantizer's write as it arrives (_i) and again (_h): the long hop from the quantizer is
  // these two narrow registers (on 959b425 at 133.33 MHz the quantizer sat 600 RPM units from
  // these block RAMs and the wide write register below, fed straight from it, missed timing; on
  // 812bb01, with one, its replicas by the placement logic were the worst path, 96% route)
  (* shreg_extract = "no" *) logic [LANES-1:0]      we_i, we_h;
  (* shreg_extract = "no" *) logic [7:0]            w_row_i, w_off_i, s_row_i;
  (* shreg_extract = "no" *) logic [7:0]            w_row_h, w_off_h, s_row_h;
  (* shreg_extract = "no" *) logic [DW+BW-1:0]      w_idx_i, w_idx_h;
  (* shreg_extract = "no" *) logic [LANES-1:0][7:0] w_data_i, w_data_h;
  (* shreg_extract = "no" *) logic                  w_dup_i, swe_i, w_dup_h, swe_h;
  (* shreg_extract = "no" *) logic                  w_dup4_i, w_dup4_h;
  (* shreg_extract = "no" *) logic                  w_dup8_i, w_dup8_h;
  (* shreg_extract = "no" *) logic [BW-1:0]         s_blk_i, s_blk_h;
  (* shreg_extract = "no" *) logic [63:0]           s_data_i, s_data_h;
  always_ff @(posedge clk) begin
    we_i <= we;
    w_row_i <= w_row;
    w_idx_i <= w_idx[DW+BW-1:0];
    w_data_i <= w_data;
    w_dup_i <= w_dup;
    w_dup4_i <= T2 && w_dup4;
    w_dup8_i <= DUP8 && w_dup8;
    w_off_i <= w_off;
    swe_i <= swe;
    s_row_i <= s_row;
    s_blk_i <= s_blk[BW-1:0];
    s_data_i <= s_data;
    we_h <= we_i;
    w_row_h <= w_row_i;
    w_idx_h <= w_idx_i;
    w_data_h <= w_data_i;
    w_dup_h <= w_dup_i;
    w_dup4_h <= w_dup4_i;
    w_dup8_h <= w_dup8_i;
    w_off_h <= w_off_i;
    swe_h <= swe_i;
    s_row_h <= s_row_i;
    s_blk_h <= s_blk_i;
    s_data_h <= s_data_i;
  end

  // the write, as byte enables and bytes in place (registered; the block address, column select
  // and group per column and piece in g_row), and the scale write (registered); per column:
  // selected, and the group of the row it writes
  logic [D-1:0]     wbe;
  logic [D*8-1:0]   wd;
  logic [MCOLS-1:0] ssel;
  logic [GW-1:0]    sg [MCOLS];
  logic [BW-1:0]    sb;
  logic [63:0]      sd;
  // the placement register's inputs: per column, selected and the group of the row it writes
  logic [MCOLS-1:0] wsel_n, ssel_n;
  logic [GW-1:0]    wg_n [MCOLS], sg_n [MCOLS];
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      wsel_n[j] = sel(j, w_row_h, w_dup_h, w_dup4_h, w_dup8_h, w_off_h) && (|we_h);
      ssel_n[j] = swe_h && sel(j, s_row_h, w_dup_h, w_dup4_h, w_dup8_h, w_off_h);
      wg_n[j] = sgrp(j, w_row_h, w_off_h);
      sg_n[j] = sgrp(j, s_row_h, w_off_h);
    end
  always_ff @(posedge clk) begin
    wbe <= '0;
    wd <= '0;
    for (int l = 0; l < LANES; l++) begin
      wbe[w_idx_h[DW-1:0] + DW'(l)] <= we_h[l];
      wd[8 * (w_idx_h[DW-1:0] + DW'(l)) +: 8] <= w_data_h[l];
    end
    ssel <= ssel_n;
    for (int j = 0; j < MCOLS; j++) sg[j] <= sg_n[j];
    sb <= s_blk_h;
    sd <= s_data_h;
  end
`ifndef SYNTHESIS
  initial begin we_i = '0; swe_i = 1'b0; we_h = '0; swe_h = 1'b0; ssel = '0; end
`endif

  for (genvar j = 0; j < MCOLS; j++) begin : g_row
    logic [63:0]    rs;
    wire [BW-1:0] rb = (T2 && r_hq[j]) ? (r_hi[j] ? r_blk4[BW-1:0] : r_blk3[BW-1:0])
                                       : (r_hi[j] ? r_blk2[BW-1:0] : r_blk[BW-1:0]);
    wire [GW+BW-1:0] ra = (GROUPS > 1) ? {r_grp[GW-1:0], rb} : (GW+BW)'(rb);
    wire [GW+BW-1:0] sa = (GROUPS > 1) ? {sg[j], sb} : (GW+BW)'(sb);
    // a piece's block RAMs take their write address from a copy of the block address (and
    // select, group) of their own: on 812bb01 one wb drove all MCOLS * D / 8 of them across
    // the systolic array's span, 6.7 ns of wire with no logic
    for (genvar p = 0; p < NP; p++) begin : g_pc
      (* keep *) logic [BW-1:0] wb;
      (* keep *) logic          wsel;
      (* keep *) logic [GW-1:0] wg;
      logic [PB*8-1:0] rd, rdq;
      always_ff @(posedge clk) begin
        wb <= w_idx_h[DW +: BW];
        wsel <= wsel_n[j];
        wg <= wg_n[j];
        rdq <= rd;
      end
      wire [GW+BW-1:0] wa = (GROUPS > 1) ? {wg, wb} : (GW+BW)'(wb);
      // the piece's bytes: a technology RAM with byte enables (rtl/tech)
      otpu_ram_1r1w_be #(.W(PB*8), .N(GROUPS*BLOCKS), .STYLE("auto")) u_act (
        .clk, .we(wsel), .wbe(wbe[p*PB +: PB]), .wa(wa[RAW-1:0]), .wd(wd[p*PB*8 +: PB*8]),
        .re(ren[j]), .ra(ra[RAW-1:0]), .rd);
      assign r_data[j*D*8 + p*PB*8 +: PB*8] = rdq;
`ifndef SYNTHESIS
      initial wsel = 1'b0;
`endif
    end
    // the column's block scale records
    logic [63:0] rsq;
    otpu_ram_1r1w #(.W(64), .N(GROUPS*BLOCKS), .STYLE("auto")) u_asc (
      .clk, .we(ssel[j]), .wa(sa[RAW-1:0]), .wd(sd), .re(ren[j]), .ra(ra[RAW-1:0]), .rd(rs));
    always_ff @(posedge clk) rsq <= rs;
    assign r_scale[j*64 +: 64] = rsq;
`ifndef SYNTHESIS
    // a consumed read of a block (or its scale) with a write in any register: it would return
    // the old bytes
    wire [GW+BW-1:0] wa = g_pc[0].wa;
    wire             wsel_1 = sel(j, w_row_i, w_dup_i, w_dup4_i, w_dup8_i, w_off_i) && (|we_i);
    wire             ssel_1 = swe_i && sel(j, s_row_i, w_dup_i, w_dup4_i, w_dup8_i, w_off_i);
    wire [GW+BW-1:0] wa_1 = (GROUPS > 1) ? {sgrp(j, w_row_i, w_off_i), w_idx_i[DW +: BW]}
                                         : (GW+BW)'(w_idx_i[DW +: BW]);
    wire [GW+BW-1:0] sa_1 = (GROUPS > 1) ? {sgrp(j, s_row_i, w_off_i), s_blk_i} : (GW+BW)'(s_blk_i);
    wire [GW+BW-1:0] wa_n = (GROUPS > 1) ? {wg_n[j], w_idx_h[DW +: BW]} : (GW+BW)'(w_idx_h[DW +: BW]);
    wire [GW+BW-1:0] sa_n = (GROUPS > 1) ? {sg_n[j], s_blk_h} : (GW+BW)'(s_blk_h);
    always @(posedge clk)
      if (ren[j] && r_use && ((g_pc[0].wsel && wa == ra) || (wsel_n[j] && wa_n == ra) ||
                           (wsel_1 && wa_1 == ra) || (ssel[j] && sa == ra) ||
                           (ssel_n[j] && sa_n == ra) || (ssel_1 && sa_1 == ra)))
        $fatal(1, "otpu_actram: column %0d block %0d read while its write is in flight at %0t",
               j, ra, $time);
`endif
  end
endmodule
