// MXU, the MX column (docs/isa.md MM, docs/mx_mxu.md): streams N rows x KB blocks of weights from
// DRAM (port B, one D-byte chunk per cycle, prefetched through a FIFO) and their scale rows (port
// A, ALINE-byte lines), and reuses each chunk across up to MCOLS stationary rows held in ACT RAM. Each
// output is one exact integer, accumulated in its column end over the whole row:
//   c   = sum_{i in g} act[j][i] * w[i]              group g: D/8 elements, a chain of the array
//   sh  = W - dw(n, k, g) - da[j][k][g/2]            W 16 (int8x), 11 (int4x, fp4x6), 13 (nvfp4)
//   acc = sum_k sum_g (sh >= 0 ? m(n, k, g) << sh : 0) * c                 48 bits, exact
// and one fp32 epilogue per output, in E lanes the columns share (the result FIFO between them):
//   y   = (i2f(acc) * Sw[n]) * (Sa[j] * 2^-W)          then ACC / ASCALE / RMAX as before
// 4-bit weights (flags[5:4] = WF: 1 int4, 2 E2M1, 3 NVFP4): a D-byte chunk holds two blocks (the
// low half first) and is consumed in two advances. PAIR (flags[6], 4-bit, 2M <= MCOLS; "Column
// reuse"): a chunk is consumed in one advance, columns j < M take its low block 2c (ACT block
// ab+2c), columns j+M its high block 2c+1 (ACT block ab+2c+1); the two columns accumulate
// separately and their integers are added before the result FIFO (+0 for the missing block of an
// odd KB: its multipliers are 0). RSC (flags[7]): each scale row is Sw alone. UNIT: Sw 1, dw 0, m 1.
// t2 (T2 builds; docs/ternary.md): WF 1 with rs bit 0, 2-bit elements c - 1, a chunk holds four
// blocks of D/4 bytes (block q at bits 2D q), consumed in four advances (PAIR: two, blocks 2c and
// 2c+1 on the low and high streams as int4's PAIR; QUAD, rs bit 1 with PAIR, 4M <= MCOLS: one,
// columns qM + j take block 4c + q from ACT block ab + 4c + q, the four columns' integers added
// before the result FIFO). A t2 scale row has a 16-byte header; an entry holds the chunk's four
// block words ({Sw, w1, w0}, {w3, w2}). The stride is rs & ~3 for every format (bits 1:0 are
// flags). Without T2 none of this is built (the t2 terms are constant 0).
// A wide port B (PORTB 2, 4; docs/wide_port.md): a FIFO entry is PORTB chunks, and an MM whose
// column reuse takes more than a chunk a cycle (int8 PAIR: 2, QUAD: 4; 4-bit QUAD: 2) asks for
// an advance's chunks in one request (b_n: chunks - 1, fewer at a row's end), which fills one
// entry (b_rcm: the chunks a response carries, b_rlast: the entry's last; the memory may split
// it). PAIR and QUAD then hold for every format: the four streams are the entry's blocks (int8:
// chunk c; 4-bit: chunk c / 2's half c mod 2), column cM + j takes block rc' + c, as t2's QUAD,
// and the scale entry's third RAM holds blocks 2 and 3's data (int8: a byte each; 4-bit: a
// word; NVFP4: eight bytes). A 4-bit / NVFP4 QUAD advance whose 16 / 32 scale bytes cross a
// port-A line first asks for the first line alone (staged, sp), then the second with the chunks.
// PORTB 1 builds none of it.
// Row lanes (RL builds; docs/row_lanes.md): MM RL (rs bits 3:2 = log2 L, L = 2, 4, 8; M = 1) streams
// L rows at once from groups of L rows (chunk q of row g L + l at sa + g rs + (q L + l) D: an
// advance's L chunks one aligned request, an entry) with the scale rows in groups (ssa + g srs:
// the L Sw words, then from H = max(4 L, A) block k's data of the L rows, A = L bpc bs bytes an
// advance in one line). The streams are numbered block-major, s = p L + l (block p of lane l's
// chunk), and column J takes stream J: the lane structure is PAIR / QUAD's with M' = L (columns
// J >= L take block 1, J >= 2 L blocks 2 and 3, the row sum adds columns j + L, j + 2 L, j + 3 L
// into j < L), the per-column stream select (s_hi / s_hq) apart from the ACT RAM's block select
// (hi / hq: block p = J / L). A pipeline row's L results are output rows n0 .. n0 + L - 1 (the
// drain at out + n0 + l, stepping L; the last row's N - (G - 1) L), each with its lane's Sw
// (the entry's L Sw words, per column into the result FIFO), RMAX the max over the lanes'
// columns, ASCALE's one factor. RL is the streams the build has: 4 (QUAD's) or 8 (MCOLS 8: the
// hops 64 bits a position, the PE select 4:1, four more scale decoders and entry slots).
// Port A's line (ALINE, 32 or 64 bytes; docs/row_lanes.md "Port A at 64 bytes"): what one scale
// request returns, the aligned ALINE bytes holding the word asked for. Every window into the
// line (the tags' byte offset, the Sw word, the same-line compares, a QUAD advance's data across
// two lines) is indexed by the line's size; an RL advance's A <= 32 bytes and a group's Sw words
// never cross a line of either size, and at 64 the Sw of a group whose header and first advance
// take 33 .. 64 bytes (int4 L 4: H = A = 32; int8 L 8) shares the first advance's line instead
// of a request of its own: one line an advance.
// The group walk (RL builds; srs bit 0 with rs bits 3:2; docs/row_lanes.md "Prefill over the
// group layout"): a plain MM of any M over a matrix stored in row-lane groups of L, the rows one
// at a time (prefill and the verify runs over a decode image). Only the issuer's addresses
// differ from a plain MM's: row g L + l's chunks are L D apart at sa + g rs + l D (the row step
// D within a group, rs - (L - 1) D at its end), its Sw at the group's 4 l and block k's data at
// H + k L bs + l bs (the block step L bs, the first block's offset stepping bs - 4 a row). The
// pipeline, the scale entries and the drain are a plain MM's (cmd_rl is 0: no lanes).
//
// The array is the systolic one (docs/mxu_systolic.md): chains of CLS = min(CL, D/8) products
// (DSP48 cascades), so that every chain end is one group. A column end (otpu_mxce, rtl/tech) takes
// the column's eight group sums, multiplies each by its B = m << sh (the shift in fabric, two
// cycles ahead), sums them down a cascade and accumulates the row in a ninth DSP. The columns are
// skewed by a cycle each; their row sums are deskewed and registered, PAIR's halves added, and the
// row (MCOLS integers, the rows' Sa * 2^-W and Sw) written to the result FIFO. The drain picks up
// to E results of the head row a cycle (fewer at a TMEM bank conflict) into the lanes: i2f48 (round
// to nearest even), the two multiplies, then the TMEM write or the read-modify-write (ACC, ASCALE)
// as before; everything after the result FIFO advances only while the TMEM grant is given.
//
// The scale reader: for every advance the issuer asks port A for the ALINE-byte line that holds the
// advance's block data (int8x: one byte; int4x / fp4x6: a word, PAIR two; NVFP4: eight bytes,
// PAIR sixteen; RSC: Sw), and for a row's Sw: in the same line when Sw is in the line of the row's
// first advance or of the previous row's last one, else in a request of its own (a cycle without
// a chunk). A tag per request says what the line holds; the responses fill one scale entry per
// chunk ({Sw, the low block's data}, {the high block's data}) in the scale FIFO.
//
// Replay (ROWS > MCOLS, M > MCOLS): each streamed row is consumed G = ceil(M / MCOLS) times, once
// per group of MCOLS stationary rows (ACT RAM rows g*MCOLS ..). Group 0 pops the row's chunks from
// the FIFO without freeing them, the later groups read the same FIFO entries again (head + k) and
// the last group frees them, so a weight is streamed from DRAM once for all M rows (a row must fit
// the FIFO: KB <= DEPTH). Each (row, group) is a row of the pipeline and the drain; the drain steps
// its TMEM addresses by MCOLS * ors per group.
//
// The MXU holds two commands: the issuer streams the newer one's chunks while the consumer
// finishes (and drains) the older one. The issuer yields DRAM ports to the DMA and QST
// (a_gnt/b_gnt). A row's first advance pops only once all of the row's chunks and scales are in
// the FIFOs, so a row never stalls in its middle and the array needs no clock enable.
module otpu_mxu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int MCOLS = 8,
  parameter int ROWS  = MCOLS,   // ACT RAM rows: the most stationary rows of a command
  parameter int DEPTH = 16,
  parameter int LANES = 8,       // TMEM banks (the drain's lanes write distinct banks)
  parameter int IMPL  = 2,       // the dot product: 2 systolic (the only one; CFG_UARCH reports it)
  parameter int CL    = 16,      // products per cascade chain (at most D/8)
  parameter int E     = 0,       // epilogue lanes (0: MCOLS / 2, at least 1; more than MCOLS: MCOLS)
  parameter bit T2    = 1'b0,    // t2 weights and MM QUAD (docs/ternary.md; CAPS bit28)
  parameter int PORTB = 1,       // chunks a cycle on port B (1, 2, 4; docs/wide_port.md)
  parameter int RL    = 0,       // row lanes: the weight streams, 0 (none), 4 or 8 (docs/row_lanes.md)
  parameter int ALINE = 32,      // port A's line, bytes (32, 64): a scale request's response
  parameter int SID   = 0
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  start,       // accept: the issuer may stream it
  input  logic                  go,          // release: the consumer may use it (in order)
  input  cmd_t                  cmd,
  output logic                  rdy,
  output logic                  done,
  output logic                  computing,   // a chunk is consumed this cycle (profiling)
  // profiling: this cycle's stream state (the FIFO level; work but no chunk / chunks but no
  // consumption) and, a cycle after a command ends, its counters {deny, frz, bp, starve}
  output logic [$clog2(DEPTH):0] pf_level,
  output logic                  pf_starve,
  output logic                  pf_block,
  output logic                  pf_u,
  output logic [3:0][31:0]      pf_uv,
  // ACT RAM read
  output logic [15:0]           act_blk,
  output logic [15:0]           act_blk2,    // PAIR: the odd block, read by the rows in act_hi
  output logic [MCOLS-1:0]      act_hi,
  output logic [15:0]           act_blk3,    // QUAD (T2): blocks 4c+2 and 4c+3, read by the rows
  output logic [15:0]           act_blk4,    // in act_hq (act_hi: the latter)
  output logic [MCOLS-1:0]      act_hq,
  output logic [7:0]            act_grp,
  output logic [MCOLS-1:0]      act_ren,     // per column: its ACT RAM read register advances
  input  logic [MCOLS*D*8-1:0]  act_data,
  input  logic [MCOLS*64-1:0]   act_scale,   // per column the record {da3, da2, da1, da0, Sa}
  // DRAM port A (scale lines) and B (chunks)
  output logic                  a_req,
  output logic [31:0]           a_addr,
  input  logic                  a_gnt,
  input  logic                  a_rvalid,
  input  logic [8*ALINE-1:0]    a_rdata,     // the aligned ALINE bytes holding word a_addr
  output logic                  b_req,
  output logic [31:0]           b_addr,
  output logic [1:0]            b_n,         // PORTB > 1: the request's chunks - 1 (0 else)
  input  logic                  b_gnt,
  input  logic                  b_rvalid,
  input  logic [PORTB*D*8-1:0]  b_rdata,     // chunk c of the entry at [c*D*8 +: D*8]
  input  logic [PORTB-1:0]      b_rcm,       // PORTB > 1: the chunks this response carries
  input  logic                  b_rlast,     // PORTB > 1: ... and the entry is complete
  // TMEM (read port for ACC, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata,
  input  logic                    t_gnt
);
  localparam int BW = $clog2(LANES);
  localparam int PW = $clog2(DEPTH);
  localparam int LM = 2, LA = 4;
  // epilogue lanes; NL: the drain's lanes per step
  // (a target's E holds at a smaller MCOLS too: the card's 4 is 2 at MCOLS 2)
  localparam int EL = (E > MCOLS) ? MCOLS : (E > 0) ? E : ((MCOLS >= 2) ? MCOLS / 2 : 1);
  localparam int NL = EL;
  // chains of CLS products, CPG of them per group of D/8 elements (1, or 2 above D = 128)
  localparam int CLS = (D / 8 < CL) ? D / 8 : CL;
  localparam int CPG = D / 8 / CLS;
  // S0 to the columns' row sums, deskewed (the column end's accumulator is valid CLS + 6 + j
  // edges after S0, column j deskewed by MCOLS - 1 - j)
  localparam int LROW = CLS + MCOLS + 5;
  // result FIFO rows: a row pops only while fewer than RF are between the pop and the drain, so
  // RF covers the pipeline's rows in flight (one-advance rows: one per cycle of its latency)
  localparam int RF = (LROW + 7 <= 32) ? 32 : 64;
  localparam int RFW = $clog2(RF);
  localparam int MW = $clog2(MCOLS) + 1;
  localparam int TD = 128;                  // port A requests in flight (their tags)
  localparam int TDW = $clog2(TD);
  // a lane: the pick, i2f48 (4), x Sw, x Sa'
  localparam int LLAT = 1 + 4 + 2 * LM;
  initial if (IMPL != 2) $fatal(1, "otpu_mxu: IMPL 2 (systolic) is the only dot product");
  initial if (D % 16 != 0 || (D / 8) % CLS != 0 || CPG > 2)
    $fatal(1, "otpu_mxu: D a multiple of 16, D/8 a multiple of CLS, at most two chains a group");
  initial if (EL > LANES) $fatal(1, "otpu_mxu: E %0d > LANES", EL);
  initial if (ROWS < MCOLS || ROWS > 255) $fatal(1, "otpu_mxu: MCOLS <= ROWS < 256");
  initial if (PORTB != 1 && PORTB != 2 && PORTB != 4) $fatal(1, "otpu_mxu: PORTB 1, 2 or 4");
  initial if (RL != 0 && RL != 4 && RL != 8) $fatal(1, "otpu_mxu: RL 0, 4 or 8");
  initial if (RL != 0 && PORTB == 1) $fatal(1, "otpu_mxu: row lanes need a wide port B");
  // QUAD's datapath (four streams, four scale decoders, the four-column row sum): t2, or any
  // format on a wide port
  localparam bit QD = T2 || PORTB > 1;
  localparam bit WP = PORTB > 1;
  // row lanes: RLB builds the mode, NS the weight streams (8: the eight-stream build), NSW the
  // stream select's bits, NSX the chunks the S1 stream decode sees (zero past PORTB)
  localparam bit RLB = RL != 0;
  localparam int NS = (RL >= 8) ? 8 : QD ? 4 : 2;
  localparam int NSW = (NS >= 8) ? 2 : 1;
  localparam int NSX = (NS >= 8) ? 8 : 4;
  // port A's line: LB address bits, LW bits of data, LWW bits of word index
  localparam int LB  = $clog2(ALINE);
  localparam int LW  = 8 * ALINE;
  localparam int LWW = LB - 2;
  initial if (ALINE != 32 && ALINE != 64) $fatal(1, "otpu_mxu: ALINE 32 or 64");

  // conflict-free drain run for a row stride: LANES / gcd(ors, LANES), at most NL
  function automatic logic [MW-1:0] drain_run(input logic [15:0] ors);
    int d;
    d = 1;                                   // ors a multiple of LANES: one lane per cycle
    for (int t = BW - 1; t >= 0; t--)        // the lowest set bit t: gcd = 2^t
      if (ors[t]) d = LANES >> t;
    return MW'((d < NL) ? d : NL);
  endfunction
  localparam logic [1:0] WF_W8 = 2'd0, WF_NV = 2'd3;
  // t2: WF 1 (int4) with rs bit 0; QUAD: and rs bit 1 with PAIR (constant 0 without T2)
  wire cmd_t2 = T2 && cmd.flags[5:4] == 2'd1 && cmd.w5[0];
  // the matrix in row-lane groups: rs bits 3:2 = log2 L (constant 0 without RL): the lanes
  // mode, or with srs bit 0 the group walk (a plain MM over the groups, the rows one at a time)
  wire       cmd_lay = RLB && cmd.w5[3:2] != 2'd0;
  wire       cmd_gw  = cmd_lay && cmd.w7[0];
  wire       cmd_rl  = cmd_lay && !cmd.w7[0];
  wire [1:0] cmd_lg = cmd_lay ? cmd.w5[3:2] : 2'd0;
  wire [3:0] cmd_L  = cmd_rl ? 4'd1 << cmd_lg : 4'd1;
  wire [2:0] cmd_bpc = cmd_t2 ? 3'd4 : (cmd.flags[5:4] != WF_W8) ? 3'd2 : 3'd1;
  // PAIR: 4-bit (a wide port: every format); RL: a lane's chunk whole (bpc blocks: PAIR's
  // structure for 4-bit, QUAD's for t2)
  wire cmd_pair = cmd_rl ? (cmd_bpc != 3'd1) : cmd.flags[6] && (cmd.flags[5:4] != WF_W8 || WP);
  // (a wide port: PAIR and QUAD for every format)
  wire cmd_qd = cmd_rl ? cmd_t2 : (cmd_t2 || WP) && cmd.flags[6] && cmd.w5[1];
  // the pipeline's M: the command's, or (RL) its lanes
  wire [7:0] cmd_M = cmd_rl ? 8'(cmd_L) : cmd.w6[23:16];
  // PORTB > 1: the chunks an advance streams (int8: PAIR 2, QUAD 4; 4-bit QUAD 2; RL: L; else
  // 1), and a row's last advance's (fewer when KB is not a multiple of the reuse; RL: L too);
  // chunks - 1
  wire [1:0] cmd_nb = cmd_rl ? 2'(cmd_L - 4'd1) : !WP || cmd_t2 ? 2'd0 :
                      (cmd.flags[5:4] == WF_W8) ? (cmd_qd ? 2'd3 : cmd.flags[6] ? 2'd1 : 2'd0) :
                      cmd_qd ? 2'd1 : 2'd0;
  wire [1:0] cmd_km = 2'(cmd.w4[23:16] - 8'd1);           // (KB - 1) mod 4
  wire [1:0] cmd_nl = cmd_rl ? cmd_nb :
                      (cmd.flags[5:4] == WF_W8) ? cmd_nb & cmd_km : cmd_nb & {1'b0, cmd_km[1]};
`ifndef SYNTHESIS
  // a replayed row (M > MCOLS) must fit the FIFO: its chunks (4-bit: two blocks each, t2 four)
  always @(posedge clk) if (!rst && start) begin
    if (int'(cmd.w6[23:16]) > ROWS) $fatal(1, "otpu_mxu: M %0d > ROWS %0d", cmd.w6[23:16], ROWS);
    if (int'(cmd.w6[23:16]) > MCOLS && (cmd_t2 ? (int'(cmd.w4[23:16]) + 3) / 4 :
        (cmd.flags[5:4] != WF_W8) ? (int'(cmd.w4[23:16]) + 1) / 2 : int'(cmd.w4[23:16])) > DEPTH)
      $fatal(1, "otpu_mxu: a replayed row (M > MCOLS) must fit the FIFO: KB %0d, DEPTH %0d",
             cmd.w4[23:16], DEPTH);
    if (cmd_pair && 2 * int'(cmd_M) > MCOLS)
      $fatal(1, "otpu_mxu: PAIR needs 2*M <= MCOLS");
    if (cmd.w5[3:2] != 2'd0 && !cmd_lay)
      $fatal(1, "otpu_mxu: row lanes (rs bits 3:2) on a build without RL");
    if (cmd_gw && (cmd.flags[6] || cmd.w5[1]))
      $fatal(1, "otpu_mxu: the group walk (srs bit 0) takes no PAIR / QUAD");
    if (cmd_rl && (int'(cmd_L) > PORTB || int'(cmd_L) * int'(cmd_bpc) > MCOLS ||
                   int'(cmd_L) * int'(cmd_bpc) > NS || cmd.w6[23:16] != 8'd1 || cmd.flags[6] ||
                   cmd.w5[1] || (cmd.w4[31:24] % 8'(cmd_bpc)) != 8'd0))
      $fatal(1, "otpu_mxu: RL %0d lanes of %0d blocks a chunk: PORTB %0d, MCOLS %0d, streams %0d, M %0d, flags %h, k0 %0d",
             cmd_L, cmd_bpc, PORTB, MCOLS, NS, cmd.w6[23:16], cmd.flags, cmd.w4[31:24]);
    if (!T2 && cmd.flags[5:4] == 2'd1 && cmd.w5[0])
      $fatal(1, "otpu_mxu: t2 weights (rs bits %0d) on a build without T2", cmd.w5[1:0]);
    if (cmd.w5[1] && !cmd_qd)
      $fatal(1, "otpu_mxu: QUAD (rs bit 1) needs PAIR and t2 or a wide port (PORTB > 1)");
    if (int'(cmd_nb) + 1 > PORTB)
      $fatal(1, "otpu_mxu: an MM of %0d chunks a cycle on a port of %0d", int'(cmd_nb) + 1, PORTB);
    if (cmd_qd && (4 * int'(cmd_M) > MCOLS || cmd.w4[25:24] != 2'd0))
      $fatal(1, "otpu_mxu: QUAD needs 4*M <= MCOLS and k0 a multiple of 4 (M %0d, k0 %0d)",
             cmd_M, cmd.w4[31:24]);
    if (cmd_t2 && cmd.flags[7] && !cmd.flags[0]) $fatal(1, "otpu_mxu: RSC does not take t2");
  end
`endif

  // ================================================================== issuer
  logic        i_act, i_unit, i_w4, i_pair, i_rsc;
  logic        i_t2r, i_qdr;                // t2 weights; QUAD (registers; i_t2, i_qd: constant 0
  wire         i_t2 = T2 && i_t2r;         // without T2)
  wire         i_qd = QD && i_qdr;
  logic        i_rlr;                       // row lanes (constant 0 without RL), log2 L
  logic [1:0]  i_lg;
  wire         i_rl = RLB && i_rlr;
  logic        i_gwr;                       // the group walk (constant 0 without RL): the lane
  wire         i_gw = RLB && i_gwr;         // of the row in hand, lane 0's scale offset, the
  logic [2:0]  i_wl;                        // per-row offset step (bs - 4), the row and Sw
  logic [31:0] w_so0, w_ds, w_rd, w_sd;     // steps at a group's end (rs - (L - 1) D, srs -
  wire  [2:0]  w_last = 3'((4'd1 << i_lg) - 4'd1);   // 4 (L - 1)); the group's last lane
  logic [31:0] i_cstep;                     // the chunk address step within a row
  logic        i_ini;                       // the command's first cycle: its addresses load
  logic [1:0]  i_wf;
  logic [7:0]  i_k0;
  logic [31:0] i_left, i_rs, i_srs;
  logic [15:0] i_KB, i_k;
  logic        i_klast, i_l1;               // i_k + 1 == i_KB; i_left == 1
  logic [31:0] row_addr, chunk_addr;
  logic [31:0] srow_addr, srow_next;        // the row's scale row; the next row's
  logic [31:0] scale_addr;                  // the advance's block data in the scale row
  logic [31:0] i_soff;                      // block k0's data offset in a scale row
  logic [5:0]  i_sstep;                     // the data address step per advance
  logic        i_swd;                       // the row's Sw came with an earlier line
  logic        i_swh;                       // the row's Sw is in the line of its first advance
  logic        i_swn;                       // the next row's Sw is in the advance's line
  logic        i_sws;                       // a request for the row's Sw alone comes first
  logic [PW:0] occ;                         // entries issued and not yet popped (<= DEPTH)
  // PORTB > 1: an advance's chunks - 1, a row's last advance's; a 4-bit / NVFP4 QUAD advance
  // whose scale data crosses a line: its first line asked for (i_sp2), the second line (i_sa2)
  logic [1:0]  i_nb, i_nl;
  logic        i_sp2;
  logic [31:0] i_sa2;
  logic [TDW:0] tg_n;                       // port A requests in flight

  // ================================================================== command queue (2)
  // a row is ceil(KB/2) advances under PAIR (4-bit: of one chunk each), ceil(KB/4) under
  // QUAD. KB is 8 bits, k0 above it
  wire [7:0]  cmd_KB8 = cmd.w4[23:16];
  wire [15:0] cmd_KBa = cmd_qd ? (16'(cmd_KB8) + 16'd3) >> 2 :
                        cmd_pair ? (16'(cmd_KB8) + 16'd1) >> 1 : 16'(cmd_KB8);
  // RL: the pipeline rows are the groups, ceil(N / L); the last one's results
  wire [15:0] cmd_nrow = cmd_rl ? 16'((32'(cmd.w4[15:0]) + 32'(cmd_L) - 1) >> cmd_lg)
                                : cmd.w4[15:0];
  wire [3:0]  cmd_Mz = cmd_rl ? 4'(32'(cmd.w4[15:0]) - 32'(cmd_nrow - 16'd1) * 32'(cmd_L)) : 4'd1;
  wire [31:0] cmd_total = 32'(cmd_nrow) * 32'(cmd_KBa);   // advances (chunk requests)
  // cmd_total == 0 without the multiply (a product of two factors is zero iff one is)
  wire        cmd_tz = (cmd.w4[15:0] == 16'd0) || (cmd_KB8 == 8'd0);
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && start && cmd_tz != (cmd_total == 0))
      $fatal(1, "otpu_mxu: cmd_tz %0d but total %0d", cmd_tz, cmd_total);
`endif
  // The two entries are the head (h) and the next command (n), not two slots and a head pointer:
  // every head field is its own flop, with no 2:1 mux on a pointer in front of the consumer and
  // the drain. Completing the head swaps them (n keeps the old head, as the idle slot did).
  typedef struct packed {
    logic [31:0]      out, total;
    logic             tz;              // total == 0 (registered: off the drain path)
    logic [15:0]      KB, KBa;         // KBa: advances per row
    logic             pair;
    logic [MCOLS-1:0] hi;              // PAIR: the columns that take the odd blocks (j >= M)
    logic             t2, qd;          // t2 weights; QUAD (T2)
    logic [MCOLS-1:0] hq;              // QUAD: the columns of blocks 4c+2, 4c+3 (j >= 2M); hi
                                       // then holds those of the odd blocks (M <= j < 2M, j >= 3M)
    logic [31:0]      mxo;             // RMAX output base: out + M * ors (no multiply later)
    logic [7:0]       M, ab;           // (RL: M is the lanes L, the pipeline's M)
    logic             rl;              // row lanes (docs/row_lanes.md)
    logic [15:0]      nrow;            // RL: the pipeline rows (the groups, ceil(N / L))
    logic [MW-1:0]    Mz;              // RL: the last row's results (N - (nrow - 1) L); else Ml
    logic [MW-1:0]    dnlz;            // ... its first drain step's lanes and whether it drains it
    logic             dflz;
    logic [3:0]       dinc;            // the drain's step to the next row's addresses (RL: L)
    logic [MW-1:0]    run;             // drain lanes per cycle without a bank conflict
    logic             unit, acc, rmax, asc, go, rsc;
    logic [1:0]       wf;
    logic [31:0]      asa;
    logic [MCOLS-1:0][31:0] jo;        // j * ors
    logic [7:0]       G;               // groups: ceil(M / MCOLS)
    logic [MW-1:0]    Ml;              // rows of the last group
    logic [31:0]      gs;              // drain address step to the next group (MCOLS * ors)
    logic             g1;              // G == 1: a row's first group is its last
    logic [MW-1:0]    dnl;             // the last group's first drain step: its lanes (min(run,
    logic             dfl;             // Ml)) and whether it drains the group (Ml <= run)
    logic [MCOLS-1:0] cm;              // the columns its advances read (act_ren)
  } qent_t;
  qent_t       h, n;                    // the head, the next command
  qent_t       cmd_e;                   // cmd as an entry (not yet released)
  logic [1:0]  q_n;

  always_comb begin
    int g;
    cmd_e = '0;
    cmd_e.out   = cmd.w3;
    cmd_e.total = cmd_total;
    cmd_e.tz    = cmd_tz;
    cmd_e.KB    = 16'(cmd_KB8);
    cmd_e.KBa   = cmd_KBa;
    cmd_e.pair  = cmd_pair;
    cmd_e.t2    = cmd_t2;
    cmd_e.qd    = cmd_qd;
    for (int j = 0; j < MCOLS; j++) begin
      cmd_e.hi[j] = cmd_pair && 32'(j) >= 32'(cmd_M) &&
                    !(cmd_qd && 32'(j) >= 32'd2 * 32'(cmd_M) && 32'(j) < 32'd3 * 32'(cmd_M));
      cmd_e.hq[j] = cmd_qd && 32'(j) >= 32'd2 * 32'(cmd_M);
    end
    cmd_e.mxo   = cmd.w3 + 32'(cmd.w6[23:16]) * 32'(cmd.w6[15:0]);
    cmd_e.M     = cmd_M;
    cmd_e.rl    = cmd_rl;
    cmd_e.nrow  = cmd_nrow;
    cmd_e.dinc  = cmd_rl ? cmd_L : 4'd1;
    // RL: a row's L results at out + n0 + l, consecutive words (a stride of one)
    cmd_e.run   = drain_run(cmd_rl ? 16'd1 : cmd.w6[15:0]);
    cmd_e.ab    = cmd.w6[31:24];
    cmd_e.unit  = cmd.flags[0];
    cmd_e.rsc   = cmd.flags[7] && !cmd.flags[0];
    cmd_e.wf    = cmd.flags[5:4];
    cmd_e.acc   = cmd.flags[1];
    cmd_e.rmax  = cmd.flags[2];
    cmd_e.asc   = cmd.flags[3];
    cmd_e.go    = 1'b0;
    cmd_e.asa   = cmd.w2;
    for (int j = 0; j < MCOLS; j++)
      cmd_e.jo[j] = cmd_rl ? (32'(j) < 32'(cmd_L) ? 32'(j) : 32'd0) : 32'(j) * 32'(cmd.w6[15:0]);
    g = (int'(cmd_M) + MCOLS - 1) / MCOLS;
    if (g < 1) g = 1;
    cmd_e.G     = 8'(g);
    cmd_e.Ml    = MW'(int'(cmd_M) - (g - 1) * MCOLS);
    cmd_e.gs    = 32'(MCOLS) * 32'(cmd.w6[15:0]);
    cmd_e.g1    = (g == 1);
    cmd_e.dnl   = (cmd_e.run < cmd_e.Ml) ? cmd_e.run : cmd_e.Ml;
    cmd_e.dfl   = (cmd_e.Ml <= cmd_e.run);
    cmd_e.Mz    = cmd_rl ? MW'(cmd_Mz) : cmd_e.Ml;
    cmd_e.dnlz  = (cmd_e.run < cmd_e.Mz) ? cmd_e.run : cmd_e.Mz;
    cmd_e.dflz  = (cmd_e.Mz <= cmd_e.run);
    // the ACT RAM columns its advances read: every column with replay (G > 1), else rows j < M
    // (PAIR: and the high rows j < 2M; QUAD j < 4M; RL: L bpc); the others are no output's
    for (int j = 0; j < MCOLS; j++)
      cmd_e.cm[j] = (g > 1) ||
                    32'(j) < (cmd_qd ? 32'd4 : cmd_pair ? 32'd2 : 32'd1) * 32'(cmd_M);
  end

  // Two heads (docs/mxu_systolic.md): the drain's head is h, the oldest command: its results are
  // drained, its completion signalled. The pop head is h, or n once the head's chunks have all
  // popped (pn), if the pipeline treats n alike (the same PAIR and M, the per-command state read
  // after S0): the next command's rows then enter the pipeline behind the head's instead of after
  // its drain. pop_q is the pop head's entry, a register loaded with h and n (pn ? n : h), so
  // neither the pop nor the drain reads its command through a mux.
  logic        pn;
  qent_t       pop_q, pop_e;
  assign pop_e = pop_q;

  wire [31:0] c_out = h.out;
  wire        c_tz = h.tz;
  // the pop head's (c_KB .. c_wf, p_*) and the drain head's (the rest)
  wire [15:0] c_KB = pop_e.KB, c_KBa = pop_e.KBa;
  wire        c_pair = pop_e.pair;
  wire [MCOLS-1:0] c_hi = pop_e.hi;
  wire        c_t2 = T2 && pop_e.t2, c_qd = QD && pop_e.qd;
  wire [MCOLS-1:0] c_hq = QD ? pop_e.hq : '0;
  wire [7:0]  c_M = h.M, c_ab = pop_e.ab;
  wire        c_rl = RLB && h.rl;           // (the drain head's; constant 0 without RL)
  wire [7:0]  c_Mr = c_rl ? 8'd1 : h.M;     // the RMAX words and ASCALE factors: the MM's rows
  wire [7:0]  p_M = pop_e.M, p_G = pop_e.G;
  wire        p_act = (q_n != 0) && pop_e.go;
  wire [MW-1:0] c_run = h.run;
  wire        c_unit = pop_e.unit, c_rsc = pop_e.rsc, c_acc = h.acc, c_rmax = h.rmax;
  wire        c_asc = h.asc;
  wire [1:0]  c_wf = pop_e.wf;
  wire        c_w4 = (c_wf != WF_W8);
  wire [31:0] c_asa = h.asa;
  wire        c_act = (q_n != 0) && h.go;
  wire [7:0]  c_G = h.G;
  logic [31:0] alpha [MCOLS];
  logic [1:0]  al_st;                       // ASCALE factors: 0 to load, 1 requested (in TMEM's
                                            // request register), 3 read (the words arrive), 2 loaded
  logic [7:0]  al_i, mx_i;                  // next ASCALE factor to load / RMAX value to write

  // FIFOs of chunks and of their scale entries (one per chunk; the two DRAM ports return
  // independently, in order; an issued chunk's slot is reserved, so neither FIFO can overflow)
  // block RAM: written in their own reset-free process below. The chunk FIFO is its own module
  // (kept as a hierarchy): inline, Vivado absorbed its read register into the DSP input
  // registers of the products, which left an asynchronous read.
  logic [PW-1:0]  f_head, f_tail;
  logic [PW-1:0]  s_head, s_tail;
  logic [PW:0]    f_count, s_count;
  // FIFO read addresses, kept equal to (last group ? head : head + entry k of the row) by the
  // pops: a register, so no adder sits in front of the block RAM address
  logic [PW-1:0]  f_rd, s_rd;
  // The chunk FIFO is FG block RAM column groups (whole 36-bit BRAM columns each), each with its
  // own copies of the write and read addresses (f_tl, f_rdl: f_tail's and f_rd's updates): one
  // register no longer drives every block RAM of the chunk across its span. f_tail and f_rd remain
  // for the checks. A wide port's entry is PORTB chunks, each its own groups (a response may
  // carry some of an entry's chunks: b_rcm, their write enables).
  localparam int FW1 = D * 8;               // a chunk's bits
  localparam int FW = PORTB * FW1;          // an entry's
  localparam int FC = (FW1 + 35) / 36;      // 36-bit block RAM columns a chunk
  localparam int FGC = (FC < 4) ? FC : 4;   // column groups a chunk
  localparam int FG = PORTB * FGC;
  function automatic int fg_lo(input int g);                    // (of chunk g / FGC)
    int c;
    c = 0;
    for (int i = 0; i < g % FGC; i++) c = c + FC / FGC + ((i < FC % FGC) ? 1 : 0);
    return (g / FGC) * FW1 + ((36 * c < FW1) ? 36 * c : FW1);
  endfunction
  // an entry is complete (f_tail moves): a response (PORTB > 1: its last part)
  wire b_push = WP ? b_rvalid && b_rlast : b_rvalid;
  (* keep *) logic [FG-1:0][PW-1:0] f_tl, f_rdl;

  // ================================================================== consumer control
  logic [15:0] ck;
  logic [7:0]  cg;                          // the group of the row being consumed (replay)
  logic [31:0] c_left;                      // chunks of the pop head not yet freed
  // a command accepted last cycle became the head: c_left takes its total now, from h (a flop),
  // not from cmd_total in the start cycle. It cannot pop yet (released a cycle after its start at
  // the earliest), and nothing may treat the stale c_left == 0 as drained meanwhile
  logic        cl_ld;
  logic [RFW:0] rows_live;                  // the head's rows popped (first block), not drained
  logic [RFW:0] rows_p;                     // the pop head's, while it is not the head
  wire last_k   = (ck + 1 == c_KBa);
  wire last_g   = (cg + 1 == p_G);          // the row's last group: its pops free FIFO entries
  wire more     = p_act && (c_left != 0);
  // the chunk of advance ck (4-bit without PAIR: two advances a chunk; t2 four, with PAIR two,
  // with QUAD one) and whether the advance finishes it
  wire [15:0] ckc = c_t2 ? (c_qd ? ck : c_pair ? {1'b0, ck[15:1]} : {2'b0, ck[15:2]}) :
                    (c_w4 && !c_pair) ? {1'b0, ck[15:1]} : ck;
  wire cdone    = c_t2 ? (c_qd || (c_pair ? ck[0] : ck[1:0] == 2'd3) || last_k) :
                  (!c_w4 || c_pair || ck[0] || last_k);
  // advance ck's chunk and scale entry are in the FIFOs: the next entries (one group), or entry
  // ckc of the row (group 0 of several; the later groups find the whole row)
  wire f_av     = (p_G == 8'd1) ? (f_count != 0) : (cg != 0 || 32'(f_count) > 32'(ckc));
  wire s_av     = (p_G == 8'd1) ? (s_count != 0) : (cg != 0 || 32'(s_count) > 32'(ckc));
  // pop: one block (PAIR: one chunk) advances into the pipeline; fpop: its chunk leaves the FIFO
  // (4-bit without PAIR: after the high half, or after the row's last block; replay: in the row's
  // last group). A row's first advance pops only once all of the row's chunks and scale entries
  // are in the FIFOs (a later group finds them there), so a row never stalls in its middle: a
  // cycle without a pop is a bubble between rows
  wire [15:0] c_rch = c_t2 ? (c_qd ? c_KBa : c_pair ? 16'((32'(c_KBa) + 1) / 2)
                                                     : 16'((32'(c_KBa) + 3) / 4)) :
                      (c_w4 && !c_pair) ? 16'((32'(c_KBa) + 1) / 2) : c_KBa;   // chunks a row
  wire row_in   = cg != 0 ||
                  (32'(f_count) >= 32'(c_rch) && (c_unit || 32'(s_count) >= 32'(c_rch)));
  wire pop      = more && f_av && (c_unit || s_av) &&
                  (ck != 0 || (row_in && (RFW+2)'(rows_live) + (RFW+2)'(rows_p) < (RFW+2)'(RF)));
  wire fpop     = pop && last_g && cdone;
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (more && ck != 0 && !pop) $fatal(1, "otpu_mxu: row stalled at advance %0d", ck);
    if (more && 32'(c_rch) > DEPTH) $fatal(1, "otpu_mxu: a row (%0d chunks) must fit the FIFO", c_rch);
  end
  always @(posedge clk) if (!rst && more) begin
    if (f_rd != (last_g ? f_head : f_head + PW'(ckc)))
      $fatal(1, "otpu_mxu: FIFO read address %0d, head %0d, k %0d", f_rd, f_head, ck);
    if (!c_unit && s_rd != (last_g ? s_head : s_head + PW'(ckc)))
      $fatal(1, "otpu_mxu: scale read address %0d, head %0d, k %0d", s_rd, s_head, ck);
  end
`endif
  // The issuer walks advances: a chunk request with every 8-bit block, every even 4-bit block (an
  // odd one's chunk is already on its way, so it needs no FIFO slot) and every PAIR chunk; a scale
  // line request with each advance (RSC: with each chunk), and the row's Sw line first when it is
  // in neither neighbour's line (i_sws: a cycle of its own, port A only). t2: a chunk with every
  // fourth advance (PAIR: every second, QUAD: each).
  wire adv_b    = i_t2 ? (i_qd || (i_pair ? !i_k[0] : i_k[1:0] == 2'd0)) :
                  (!i_w4 || i_pair || !i_k[0]);
  // PORTB > 1: a 4-bit / NVFP4 QUAD advance's 16 / 32 scale bytes cross the line (its data's
  // offset 24 / 16; the ISA aligns them to 8 / 16 bytes): the first line alone first (i_spx, a
  // cycle without a chunk, as i_sws), then the advance with the second line (i_sp2)
  wire i_qw     = WP && i_qd && !i_t2 && i_w4 && !i_unit && !i_rsc && !i_rl;
  wire i_spx    = i_qw && !i_sws && !i_sp2 && (i_wf == WF_NV ? &scale_addr[LB-1:4] : &scale_addr[LB-1:3]);
  wire need_b   = !i_sws && !i_spx && adv_b;
  wire need_a   = !i_unit && (i_sws || !i_rsc || adv_b);
  wire want_iss = i_act && (!need_b || occ < (PW+1)'(DEPTH)) && (!need_a || !tg_n[TDW]);
  wire go_iss   = want_iss && (!need_b || b_gnt) && (!need_a || a_gnt);
  // the tag of this cycle's port A request
  typedef struct packed {
    logic [1:0] wf;
    logic       pair, rsc;
    logic [LB-1:0] doff;       // the block data's byte in the line
    logic [1:0] hv;            // the entry's halves it writes (01 low, 10 high, 11 both)
    logic       ent;           // it writes the entry (data, or RSC's Sw)
    logic       cpl;           // ... and completes it
    logic       swe;           // the entry's Sw is this line's word swo
    logic [LWW-1:0] swo;
    logic       swr;           // the next rows' Sw is this line's word sro (Sw-only: the row's)
    logic [LWW-1:0] sro;
    // t2 (T2): the entry holds the chunk's four words, {Sw, w1, w0} and {w3, w2}. Without PAIR
    // block 4c (4c+2) is staged (stg: r_pw) and written with 4c+1 (4c+3) as {w, r_pw} (pk); a
    // row's last block alone is written as {0, w}. PAIR: the two words of an advance (x8), as
    // int4's two halves; QUAD: the four (x16), both halves.
    logic       t2, qd, stg, pk;
    // PORTB > 1, a QUAD advance's scale data across two lines: the first line is staged (spl,
    // r_stl; with swe its Sw loads r_sw), the second completes the entry from both (sp2)
    logic       spl, sp2;
    // row lanes: the advance's A = L bpc bs bytes at doff are the entry's slots (lane l's
    // block p at slot p L + l), the group's L Sw words at swo (swe) / sro (swr)
    logic       rl;
    logic [1:0] lg;
  } atag_t;
  atag_t a_tag;
  wire   i_s2 = WP && i_sp2;
  always_comb begin
    a_tag = '0;
    a_tag.wf = i_wf; a_tag.pair = i_pair; a_tag.rsc = i_rsc;
    a_tag.t2 = i_t2; a_tag.qd = i_qd;
    a_tag.rl = i_rl; a_tag.lg = i_lg;
    a_tag.doff = scale_addr[LB-1:0];
    a_tag.swo = srow_addr[LB-1:2];
    a_tag.sro = i_sws ? srow_addr[LB-1:2] : srow_next[LB-1:2];
    if (i_sws) begin
      a_tag.swr = 1'b1;
    end else if (i_rsc) begin
      a_tag.ent = 1'b1; a_tag.cpl = 1'b1; a_tag.swe = 1'b1;
    end else if (i_t2 && !i_rl) begin
      a_tag.stg = !i_qd && !i_pair && !i_k[0] && !i_klast;
      a_tag.ent = !a_tag.stg;
      a_tag.pk = !i_qd && !i_pair && i_k[0];
      a_tag.hv = i_qd ? 2'b11 : (i_pair ? i_k[0] : i_k[1]) ? 2'b10 : 2'b01;
      a_tag.cpl = i_qd || (i_pair ? i_k[0] : i_k[1:0] == 2'd3) || i_klast;
      a_tag.swe = (i_k == 0) && !i_swd && i_swh;
      a_tag.swr = i_klast && !i_l1 && i_swn;
    end else if (i_spx) begin
      a_tag.spl = 1'b1;
      a_tag.swe = (i_k == 0) && !i_swd && i_swh;
    end else begin
      a_tag.ent = 1'b1;
      // (RL: every slot of the advance's L bpc, in one line: both halves, complete)
      a_tag.hv = i_rl ? 2'b11 : (i_pair || !i_w4) ? (i_pair ? 2'b11 : 2'b01)
                                                  : (i_k[0] ? 2'b10 : 2'b01);
      a_tag.cpl = i_rl || !i_w4 || i_pair || i_k[0] || i_klast;
      // (a second line: the row's Sw came with the first; the next row's is left to its own
      // request, or its first advance's line)
      a_tag.sp2 = i_s2;
      a_tag.swe = (i_k == 0) && !i_swd && i_swh && !i_s2;
      a_tag.swr = i_klast && !i_l1 && i_swn && !i_s2;
    end
  end

  assign rdy = !i_act && !i_ini && (q_n < 2);
  assign computing = pop;
  assign pf_level  = f_count;
  assign pf_starve = more && f_count == 0;
  assign pf_block  = more && f_count != 0 && !pop;
  assign b_req  = go_iss && need_b;
  // not gated with b_req: the address (and the slice's parity of it) does not wait for b_gnt
  assign b_addr = chunk_addr >> 2;
  assign b_n    = WP ? (i_klast ? i_nl : i_nb) : 2'd0;
  assign a_req  = go_iss && need_a;
  assign a_addr = !a_req ? '0 : (i_sws || i_rsc) ? (srow_addr >> 2) : i_s2 ? (i_sa2 >> 2)
                                                                           : (scale_addr >> 2);
  // QUAD's blocks 4c+1, 4c+2, 4c+3 are in the row (not past a KB that is not a multiple of 4)
  wire q_o1 = !(last_k && c_KB[1:0] == 2'd1);
  wire q_o2 = !(last_k && (c_KB[1:0] == 2'd1 || c_KB[1:0] == 2'd2));
  wire q_o3 = !(last_k && c_KB[1:0] != 2'd0);
  assign act_blk = 16'(c_ab) + (c_qd ? {ck[13:0], 2'b00} : c_pair ? {ck[14:0], 1'b0} : ck);
  // past an odd KB the high block is not the MM's (block ab + KB, outside its ACT range: a later
  // QACT may be writing it, and the scoreboard does not hold that one back): the high rows read
  // the even block again, which their zero multipliers (m0.o low) ignore. QUAD likewise: a
  // column past the row's last block reads block 4c
  assign act_blk2 = c_qd ? 16'(c_ab) + {ck[13:0], 1'b0, q_o1}
                         : 16'(c_ab) + {ck[14:0], !(last_k && c_KB[0])};
  assign act_blk3 = QD ? 16'(c_ab) + {ck[13:0], q_o2, 1'b0} : '0;
  assign act_blk4 = QD ? 16'(c_ab) + {ck[13:0], q_o3, q_o3} : '0;
  assign act_hi = c_hi;
  assign act_hq = c_hq;
  assign act_grp = cg;
`ifndef SYNTHESIS
  // co-sim statistic of the scale reads: port A line requests, and those that fetch a row's Sw
  // alone (i_sws: its Sw in neither neighbour's line). Totals in xs_a / xs_sw (otpu_slice prints
  // them at the dump: rtlsim's stats["mxu_scale"]); under +trace one line a command as its
  // issue ends, "MXS<sid> a=<requests> sw=<Sw alone>", in issue order (= the MMs' order;
  // tools/perf_qwen.py's MM table). Not a trace event: the board does not record it.
  longint unsigned xs_a, xs_sw;
  int unsigned     xc_a, xc_sw;
  bit              xs_trace;
  initial begin
    xs_a = 0; xs_sw = 0; xc_a = 0; xc_sw = 0;
    xs_trace = $test$plusargs("trace");
  end
  always @(posedge clk)
    if (rst) begin
      xc_a = 0; xc_sw = 0;
    end else if (go_iss) begin
      if (a_req) begin
        xs_a++; xc_a++;
        if (i_sws) begin xs_sw++; xc_sw++; end
      end
      if (!i_sws && !i_spx && i_left == 1) begin
        if (xs_trace) $display("MXS%0d a=%0d sw=%0d", SID, xc_a, xc_sw);
        xc_a = 0; xc_sw = 0;
      end
    end
`endif
  // The ACT RAM's read enables (one per column): an advance reads its block in the cycle it pops
  // (S0 takes the read register's output), and only the columns of an output row read it. A
  // column's enable is the pop head's cm, gated with the head being released (p_act: every pop
  // has it), so a command of M < MCOLS rows (decode: M = 1) reads M of the MCOLS columns, and no
  // column reads while no command is released. Both are registers (pop_q, q_n) one LUT from the
  // block RAMs' enables: pop itself is not on that path. A column that does not read holds its
  // last block, which only its own products (no output's) take.
  assign act_ren = {MCOLS{p_act}} & pop_e.cm;
`ifndef SYNTHESIS
  // every column of an output row reads its block when the advance pops: rows j < Ml of a
  // command's last group (PAIR: and its high rows j < 2M), every row of the earlier groups
  always @(posedge clk)
    if (!rst && pop)
      for (int j = 0; j < MCOLS; j++)
        if (!act_ren[j] && (!last_g || 32'(j) < (c_qd ? 32'd4 : c_pair ? 32'd2 : 32'd1) * 32'(pop_e.Ml)))
          $fatal(1, "otpu_mxu: column %0d of an output row does not read its ACT block (M %0d)",
                 j, p_M);
  // co-sim statistic of the ACT RAM reads (otpu_slice prints them at the dump: rtlsim's
  // stats["act_reads"]): cycles, column reads (enabled), pops, and the columns of output rows at
  // the pops, and the columns that ever read. Each column read is D bytes and an 8-byte scale
  // record.
  longint unsigned xa_cyc, xa_rd, xa_pop, xa_use;
  logic [MCOLS-1:0] xa_any;
  initial begin xa_cyc = 0; xa_rd = 0; xa_pop = 0; xa_use = 0; xa_any = '0; end
  always @(posedge clk)
    if (!rst) begin
      xa_cyc++;
      xa_rd += longint'($countones(act_ren));
      xa_any |= act_ren;
      if (pop) begin
        xa_pop++;
        for (int j = 0; j < MCOLS; j++)
          if (!last_g || 32'(j) < (c_qd ? 32'd4 : c_pair ? 32'd2 : 32'd1) * 32'(pop_e.Ml)) xa_use++;
      end
    end
`endif

  // ---- the tags of the requests in flight (LUT RAM) and the responses: a response is
  // registered with its tag (R1), then written into the scale entry at s_tail (R2)
  atag_t       tgm [TD];
  logic [TDW-1:0] tg_t, tg_h;
  always_ff @(posedge clk) if (a_req) tgm[tg_t] <= a_tag;
  logic        rv1;
  logic [LW-1:0] ln1;
  atag_t       tg1;
  always_ff @(posedge clk) begin
    rv1 <= !rst && a_rvalid;
    if (a_rvalid) begin
      ln1 <= a_rdata;
      tg1 <= tgm[tg_h];
    end
  end
  // the entry's data from the line: {Sw, the low block's}, {the high block's}
  logic [31:0] r_sw, r_swe, r_swr;          // the current rows' Sw; the line's words
  logic [31:0] r_pw;                        // t2 without PAIR: the staged word (block 4c, 4c+2)
  logic [95:0] r_wd0;
  logic [63:0] r_wd1;
  logic [127:0] r_wd2;                      // PORTB > 1, QUAD: blocks 2 and 3's data
  logic [127:0] r_wd3;                      // RL 8: slots 4 .. 7 (32 bits each)
  logic [7:1][31:0] r_swh;                  // RL: the group's other lanes' Sw words (0: r_sw)
  logic        r_we0, r_we1, r_we2;
  logic [LW-1:0] r_stl;                     // PORTB > 1: the staged first line (spl)
  always_comb begin
    logic [7:0]   xb, xb2;
    logic [31:0]  xw;
    logic [63:0]  x8;
    logic [127:0] x16;
    logic [2*LW-1:0] x2;
    logic [255:0] x32, x32l, xa;
    logic [63:0]  h0, h1;
    logic [127:0] h2, h3;
    logic [15:0]  x2b;
    xb  = ln1[8 * tg1.doff +: 8];
    xb2 = ln1[8 * {tg1.doff[LB-1:1], 1'b1} +: 8];       // int8 PAIR: the odd block's byte
    x2b = ln1[16 * tg1.doff[LB-1:1] +: 16];
    xw  = ln1[32 * tg1.doff[LB-1:2] +: 32];
    x8  = ln1[64 * tg1.doff[LB-1:3] +: 64];
    x16 = ln1[128 * tg1.doff[LB-1:4] +: 128];
    x32l = 256'(ln1 >> (256 * (tg1.doff >> 5)));          // the aligned 32 bytes (ALINE 64)
    // a 4-bit / NVFP4 QUAD advance's 16 / 32 bytes (8-byte aligned): from the line, or across
    // the staged line and this one (sp2)
    x2  = (WP && tg1.sp2) ? {ln1, r_stl} : {ln1, ln1};
    x32 = x2[64 * tg1.doff[LB-1:3] +: 256];
    // RL: the advance's A-byte window (A-aligned: 2, 4, 8, 16 or 32 bytes by the format and
    // L), slot s = its bytes s bs .. (lane s mod L, block s / L)
    case ((tg1.wf == WF_W8) ? 32'(1) << tg1.lg : (tg1.wf == WF_NV) ? 32'(16) << tg1.lg :
          (T2 && tg1.t2) ? 32'(16) << tg1.lg : 32'(8) << tg1.lg)
      2: xa = 256'(x2b);
      4: xa = 256'(xw);
      8: xa = 256'(x8);
      16: xa = 256'(x16);
      default: xa = x32l;
    endcase
    h2  = '0; h3 = '0;
    if (tg1.rsc) begin
      h0 = '0; h1 = '0;
    end else if (RLB && tg1.rl) begin
      if (tg1.wf == WF_W8) begin                      // a byte a slot
        h0 = {56'd0, xa[7:0]}; h1 = {56'd0, xa[15:8]};
        h2 = {56'd0, xa[31:24], 56'd0, xa[23:16]};
        h3 = {24'd0, xa[63:56], 24'd0, xa[55:48], 24'd0, xa[47:40], 24'd0, xa[39:32]};
      end else if (tg1.wf == WF_NV) begin             // eight bytes a slot (L 2: four slots)
        h0 = xa[63:0]; h1 = xa[127:64]; h2 = xa[255:128];
      end else begin                                  // a word a slot
        h0 = {32'd0, xa[31:0]}; h1 = {32'd0, xa[63:32]};
        h2 = {32'd0, xa[127:96], 32'd0, xa[95:64]};
        h3 = xa[255:128];
      end
    end else if (T2 && tg1.t2) begin
      h0 = tg1.qd ? x16[63:0] : tg1.pair ? x8 : tg1.pk ? {xw, r_pw} : {32'd0, xw};
      h1 = tg1.qd ? x16[127:64] : tg1.pair ? x8 : tg1.pk ? {xw, r_pw} : {32'd0, xw};
    end else if (tg1.wf == WF_W8) begin
      // (a wide port: PAIR's two bytes, QUAD's four, a block's data in each slot)
      h0 = (WP && tg1.qd) ? {56'd0, xw[7:0]} : {56'd0, xb};
      h1 = (WP && tg1.qd) ? {56'd0, xw[15:8]} : (WP && tg1.pair) ? {56'd0, xb2} : '0;
      h2 = (WP && tg1.qd) ? {56'd0, xw[31:24], 56'd0, xw[23:16]} : '0;
    end else if (WP && tg1.qd) begin                  // 4-bit / NVFP4 QUAD: four blocks' data
      h0 = (tg1.wf == WF_NV) ? x32[63:0] : {32'd0, x32[31:0]};
      h1 = (tg1.wf == WF_NV) ? x32[127:64] : {32'd0, x32[63:32]};
      h2 = (tg1.wf == WF_NV) ? x32[255:128] : {32'd0, x32[127:96], 32'd0, x32[95:64]};
    end else if (tg1.wf == WF_NV) begin
      h0 = tg1.pair ? x16[63:0] : x8; h1 = tg1.pair ? x16[127:64] : x8;
    end else begin
      h0 = {32'd0, tg1.pair ? x8[31:0] : xw}; h1 = {32'd0, tg1.pair ? x8[63:32] : xw};
    end
    r_swe = ln1[32 * tg1.swo +: 32];
    r_swr = ln1[32 * tg1.sro +: 32];
    r_wd0 = {tg1.swe ? r_swe : r_sw, h0};
    r_wd1 = h1;
    r_wd2 = h2;
    r_wd3 = h3;
    r_we0 = rv1 && tg1.ent && tg1.hv != 2'b10;
    r_we1 = rv1 && tg1.ent && tg1.hv[1];
    r_we2 = WP && rv1 && tg1.ent && ((tg1.qd && !tg1.t2) || (RLB && tg1.rl));
  end
  wire r_cpl = rv1 && tg1.ent && tg1.cpl;
  // RL: the group's L Sw words from the line's words swo .. (swe: this entry's; swr: the next
  // group's), r_swl[0] being r_sw itself
  wire [7:0][31:0] r_swle, r_swlr;
  for (genvar c = 0; c < 8; c++) begin : g_swl
    assign r_swle[c] = ln1[32 * LWW'(tg1.swo + LWW'(c)) +: 32];
    assign r_swlr[c] = ln1[32 * LWW'(tg1.sro + LWW'(c)) +: 32];
  end
  always_ff @(posedge clk)
    if (rv1) begin
      if (tg1.swr) r_sw <= r_swr;
      else if (tg1.swe) r_sw <= r_swe;
      if (RLB) begin
        if (tg1.swr) r_swh <= r_swlr[7:1];
        else if (tg1.swe) r_swh <= r_swle[7:1];
      end
      if (T2 && tg1.stg) r_pw <= ln1[32 * tg1.doff[LB-1:2] +: 32];
      if (WP && tg1.spl) r_stl <= ln1;
    end
  logic [95:0] sr0;                          // the S0 scale entry: {Sw, low block}, high block
  logic [63:0] sr1;
  otpu_ram_1r1w #(.W(96), .N(DEPTH), .STYLE("block")) u_fs0 (
    .clk, .we(r_we0), .wa(s_tail), .wd(r_wd0), .re(1'b1), .ra(s_rd), .rd(sr0));
  otpu_ram_1r1w #(.W(64), .N(DEPTH), .STYLE("block")) u_fs1 (
    .clk, .we(r_we1), .wa(s_tail), .wd(r_wd1), .re(1'b1), .ra(s_rd), .rd(sr1));
  logic [127:0] sr2;                         // PORTB > 1: QUAD's blocks 2 and 3
  if (WP) begin : g_fs2
    otpu_ram_1r1w #(.W(128), .N(DEPTH), .STYLE("block")) u_fs2 (
      .clk, .we(r_we2), .wa(s_tail), .wd(r_wd2), .re(1'b1), .ra(s_rd), .rd(sr2));
  end else begin : g_nfs2
    assign sr2 = '0;
  end
  // row lanes: slots 4 .. 7 (the eight-stream build) and the other lanes' Sw (L - 1 words)
  logic [127:0] sr3;
  logic [7:1][31:0] sr4;
  if (RLB && NS >= 8) begin : g_fs3
    otpu_ram_1r1w #(.W(128), .N(DEPTH), .STYLE("block")) u_fs3 (
      .clk, .we(r_we2), .wa(s_tail), .wd(r_wd3), .re(1'b1), .ra(s_rd), .rd(sr3));
  end else begin : g_nfs3
    assign sr3 = '0;
  end
  if (RLB) begin : g_fs4
    localparam int SWN = (NS >= 8) ? 7 : 3;
    logic [SWN*32-1:0] q;
    otpu_ram_1r1w #(.W(SWN * 32), .N(DEPTH), .STYLE("block")) u_fs4 (
      .clk, .we(r_we0), .wa(s_tail), .wd(tg1.swe ? r_swle[SWN:1] : r_swh[SWN:1]), .re(1'b1),
      .ra(s_rd), .rd(q));
    always_comb begin
      sr4 = '0;
      sr4[SWN:1] = q;
    end
  end else begin : g_nfs4
    assign sr4 = '0;
  end

  // ================================================================== compute pipeline
  typedef struct packed {
    logic       v;
    logic       last;     // last block of its row (the row's group)
    logic       first;    // first block of its row (the row's group)
    logic       h;        // 4-bit: the chunk's high half
    logic       o;        // PAIR: the chunk's high block is in the row (not past an odd KB)
    logic       o2, o3;   // QUAD: blocks 4c+2, 4c+3 are in the row (o: 4c+1)
    logic [1:0] q;        // t2: the low stream's block in the chunk (the high stream's: q | 1)
  } cm_t;

  cm_t                 m0;
  logic [FW-1:0]       w0;                  // the entry (FIFO read register): its chunks
  logic [MCOLS*D*8-1:0] a0;
  // S0's ACT RAM block and scale records are the ACT RAM's registered read
  assign a0 = act_data;
  logic cu0, pr0, rs0;
  logic [1:0] wf0;
  logic [MCOLS-1:0] hi0;                    // PAIR: the columns of the high block
  logic t20, qd0;                           // t2; QUAD
  logic [MCOLS-1:0] hq0;                    // QUAD: the columns of blocks 4c+2, 4c+3
  logic [7:0] M0;                           // the command's M (constant while its blocks flow)
  logic rl0;                                // row lanes (constant 0 without RL), log2 L
  logic [1:0] lg0;
  wire  w40 = (wf0 != WF_W8);
  // the per-column stream select (the PE's pair, s_hq, and its INMODE, s_hi): RL column J
  // takes stream J, else the block's hq / hi
  logic [MCOLS-1:0] s_hi, s_hi0;
  logic [MCOLS-1:0][NSW-1:0] s_hq, s_hq0;
  wire c_rlp = RLB && pop_e.rl;             // the pop head's
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      s_hi[j] = c_rlp ? 1'(j) : c_hi[j];
      s_hq[j] = c_rlp ? NSW'(j >> 1) : NSW'(c_hq[j]);
    end
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

  // FIFO writes (the slot was reserved when the chunk was issued; see occ); the RAMs are
  // technology RAMs (rtl/tech)
  for (genvar g = 0; g < FG; g++) begin : g_fd
    localparam int LO = fg_lo(g);
    localparam int W = ((g % FGC == FGC - 1) ? FW1 * (g / FGC + 1) : fg_lo(g + 1)) - LO;
    (* keep_hierarchy = "yes" *)
    otpu_ram_1r1w #(.W(W), .N(DEPTH), .STYLE("block")) u_fd (
      .clk, .we(b_rvalid && (!WP || b_rcm[g / FGC])), .wa(f_tl[g]), .wd(b_rdata[LO +: W]),
      .re(1'b1), .ra(f_rdl[g]), .rd(w0[LO +: W]));
  end
`ifndef SYNTHESIS
  // the groups' address copies are f_tail and f_rd (all reset together)
  bit fd_rst;                                         // (registers start arbitrary)
  initial fd_rst = 1'b0;
  always @(posedge clk) begin
    if (rst) fd_rst <= 1'b1;
    for (int g = 0; g < FG; g++)
      if (!rst && fd_rst && (f_tl[g] != f_tail || f_rdl[g] != f_rd))
        $fatal(1, "otpu_mxu: FIFO group %0d addresses %0d / %0d, not %0d / %0d", g, f_tl[g],
               f_rdl[g], f_tail, f_rd);
  end
`endif

  // S0: the popped chunk, its ACT RAM block and records, its scale entry
  always_ff @(posedge clk) begin
    m0 <= '0;
    if (pop) begin
      m0.v <= 1'b1;
      m0.last <= last_k;
      m0.first <= (ck == 16'd0);
      m0.h <= c_w4 && !c_pair && ck[0];
      m0.o <= c_pair && (c_qd ? q_o1 : !(last_k && c_KB[0]));
      m0.o2 <= c_qd && q_o2;
      m0.o3 <= c_qd && q_o3;
      // t2: block ck mod 4 (PAIR: 2 (ck mod 2) and the next; QUAD: all four, 0 on the low)
      m0.q <= (!c_t2 || c_qd) ? 2'd0 : c_pair ? {ck[0], 1'b0} : ck[1:0];
    end
    cu0 <= c_unit;
    rs0 <= c_rsc;
    wf0 <= c_wf;
    pr0 <= c_pair;
    hi0 <= c_hi;
    t20 <= c_t2;
    qd0 <= c_qd;
    hq0 <= c_hq;
    M0 <= p_M;
    rl0 <= c_rlp;
    lg0 <= !RLB ? 2'd0 : pop_e.M[3] ? 2'd3 : pop_e.M[2] ? 2'd2 : 2'd1;   // (log2 L: M' = L)
    s_hi0 <= s_hi;
    s_hq0 <= s_hq;
  end

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
  // (t2's packed words: not RL's, whose t2 entry holds a word a slot as the others')
  wire t2p0 = T2 && t20 && !rl0;
  wire [63:0] blo0 = t2p0 ? {32'd0, tw0[m0.q]} : (w40 && !pr0 && m0.h) ? sr1 : sr0[63:0];
  wire [63:0] bhi0 = t2p0 ? {32'd0, tw0[{m0.q[1], 1'b1}]} : sr1;
  bdec_t dlo1, dhi1, dq21, dq31;
  logic  vlo1, vhi1, vq21, vq31;
  logic [MCOLS-1:0][3:0][7:0] da1;
  logic [MCOLS-1:0] hi1, hq1;
  logic [MCOLS-1:0] s_hi1;
  logic [MCOLS-1:0][NSW-1:0] s_hq1;
  always_ff @(posedge clk) begin
    dlo1 <= bdec(blo0, wf0, cu0 || rs0);
    dhi1 <= bdec(bhi0, wf0, cu0 || rs0);
    vlo1 <= m0.v;
    vhi1 <= m0.v && m0.o;
    for (int j = 0; j < MCOLS; j++) begin
      da1[j] <= act_scale[j*64 + 32 +: 32];
      hi1[j] <= hi0[j];
      s_hi1[j] <= s_hi0[j];
      s_hq1[j] <= s_hq0[j];
    end
  end
  // (QUAD's third and fourth streams: t2's words 2 and 3, or a wide port's entry RAM)
  if (QD) begin : g_q1
    always_ff @(posedge clk) begin
      dq21 <= bdec(t2p0 || !WP ? {32'd0, tw0[2]} : sr2[63:0], wf0, cu0 || rs0);
      dq31 <= bdec(t2p0 || !WP ? {32'd0, tw0[3]} : sr2[127:64], wf0, cu0 || rs0);
      vq21 <= m0.v && m0.o2;
      vq31 <= m0.v && m0.o3;
      hq1 <= hq0;
    end
  end else begin : g_nq1
    assign dq21 = '0, dq31 = '0;
    assign vq21 = 1'b0, vq31 = 1'b0;
    assign hq1 = '0;
  end
  // RL 8: slots 4 .. 7's decoders (streams 4 .. 7)
  bdec_t dq41, dq51, dq61, dq71;
  if (RLB && NS >= 8) begin : g_q4
    always_ff @(posedge clk) begin
      dq41 <= bdec({32'd0, sr3[31:0]}, wf0, cu0 || rs0);
      dq51 <= bdec({32'd0, sr3[63:32]}, wf0, cu0 || rs0);
      dq61 <= bdec({32'd0, sr3[95:64]}, wf0, cu0 || rs0);
      dq71 <= bdec({32'd0, sr3[127:96]}, wf0, cu0 || rs0);
    end
  end else begin : g_nq4
    assign dq41 = '0, dq51 = '0, dq61 = '0, dq71 = '0;
  end
  // the streams' decoded scales (block-major: stream s = slot s) and the blocks' validity
  wire bdec_t [7:0] dec1 = {dq71, dq61, dq51, dq41, dq31, dq21, dhi1, dlo1};
  wire [3:0] vblk1 = {vq31, vq21, vhi1, vlo1};
  logic [MCOLS-1:0][7:0][16:0] bm2;
  always_ff @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      for (int g = 0; g < 8; g++) begin
        logic [10:0] sh;
        logic [10:0] s;
        logic [5:0]  m;
        logic        v;
        logic [2:0]  si;
        logic [1:0]  vi;
        // the column's stream (QUAD: one of four, by the column's hq and hi; RL: stream J)
        // and its block's validity (by the block, hq and hi)
        si = 3'({s_hq1[j], s_hi1[j]});
        vi = {QD && hq1[j], hi1[j]};
        s = dec1[si].s[g];
        m = dec1[si].m[g];
        v = vblk1[vi];
        sh = s - 11'(da1[j][g / 2]);
        bm2[j][g] <= (v && !sh[10]) ? 17'(17'(m) << sh[4:0]) : 17'd0;
      end
  // the row's Sw (UNIT: 1) and each column's Sa * 2^-W, with every advance (the row's last
  // carries them to the result FIFO): fp_mul's flush and rounding at the bottom of the range
  wire [31:0] sw0 = cu0 ? F_ONE : sr0[95:64];
  // RL: column J's lane is J mod L, its Sw the entry's word J mod L (0: sw0)
  wire [7:0][31:0] swl0 = {sr4[7], sr4[6], sr4[5], sr4[4], sr4[3], sr4[2], sr4[1], sr0[95:64]};
  logic [MCOLS-1:0][31:0] swc0;
  always_comb
    for (int j = 0; j < MCOLS; j++)
      swc0[j] = (RLB && rl0 && !cu0) ? swl0[3'(j & ((1 << lg0) - 1))] : sw0;
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
  logic [MCOLS-1:0][31:0] sw1;              // (per column: RL's lanes; else one value)
  logic [MCOLS-1:0][31:0] sap1;
  always_ff @(posedge clk) begin
    for (int j = 0; j < MCOLS; j++) begin
      sw1[j] <= swc0[j];
      sap1[j] <= fp_pow2m(act_scale[j*64 +: 32], wk);
    end
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
  // A wide port (PORTB > 1) makes the four streams the entry's blocks: int8 chunks 0 .. 3, 4-bit
  // chunk 0's halves and chunk 1's (PAIR: the first two), so QUAD's columns take them as t2's.
  localparam int WS = 8 * NS;                         // the streams' bits
  logic [D-1:0][WS-1:0] wd;                           // S1: {[4c+3, 4c+2,] high block, low block / half}
  wire  [NSX*FW1-1:0] w0x = (NSX * FW1)'(w0);         // (the entry, chunks past PORTB zero)
  // row lanes: stream s = block p = s / L of lane l = s mod L's chunk (int8 the chunk, 4-bit
  // its half p, t2 its block p), block-major
  function automatic logic [7:0] wrl(input logic [NSX*FW1-1:0] e, input int i, input int s,
                                     input logic [1:0] lg, input logic [1:0] wf, input logic t2);
    logic [FW1-1:0] c;
    logic [2:0] l, p;
    l = 3'(s & ((1 << lg) - 1));
    p = 3'(s >> lg);
    c = e[FW1 * l +: FW1];
    if (T2 && t2) return wt2(c, i, p[1:0]);
    if (wf == WF_W8) return c[8 * i +: 8];
    return dec4(c[(p[0] ? D * 4 : 0) + 4 * i +: 4], wf != 2'd1);
  endfunction
  always_ff @(posedge clk)
    for (int i = 0; i < D; i++) begin
      logic [7:0] lo, hi, q2, q3;
      logic [NS-1:0][7:0] st;
      lo = (T2 && t20) ? wt2(w0, i, m0.q) : wsel(w0, i, m0.h, wf0);
      hi = (T2 && t20) ? wt2(w0, i, {m0.q[1], 1'b1}) : wsel(w0, i, 1'b1, wf0);
      q2 = wt2(w0, i, 2'd2);
      q3 = wt2(w0, i, 2'd3);
      if (WP && !(T2 && t20)) begin
        if (wf0 == WF_W8) begin
          hi = w0x[FW1 + 8 * i +: 8];
          q2 = w0x[2 * FW1 + 8 * i +: 8];
          q3 = w0x[3 * FW1 + 8 * i +: 8];
        end else begin
          q2 = wsel(w0x[FW1 +: FW1], i, 1'b0, wf0);
          q3 = wsel(w0x[FW1 +: FW1], i, 1'b1, wf0);
        end
      end
      st = WS'({q3, q2, hi, lo});
      if (RLB && rl0)
        for (int k = 0; k < NS; k++) st[k] = wrl(w0x, i, k, lg0, wf0, t20);
      wd[i] <= st;
    end
  logic [D-1:0][WS-1:0] wc [MCOLS];                  // the streams at column j
  for (genvar i = 0; i < D; i++) begin : g_wsk
    otpu_skew #(.W(WS), .N(i % CLS)) u_w (.clk, .en(1'b1), .d(wd[i]), .q(wc[0][i]));
  end
  always_ff @(posedge clk)
    for (int j = 1; j < MCOLS; j++) wc[j] <= wc[j-1];
  logic [7:0] as_k [MCOLS][D];
  logic signed [23:0] preg [MCOLS][D];                // running sums (the DSPs' P registers;
                                                      // a chain's last: its P, unregistered)
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
      // (T2: column 0 is never QUAD's hq: c_hq[0] is 0, so its unregistered weights see no
      // select; RL: column J takes stream J, the pair J / 2 by s_hq, within it by s_hi)
      wire [15:0] wp = (QD && j != 0) ? wc[JW][i][16 * s_hq[j] +: 16] : wc[JW][i][15:0];
      otpu_pe #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0)) u_pe (
        .clk, .en(1'b1), .act(as_k[j][i]), .wlo(wp[7:0]), .whi(wp[15:8]),
        .sel(s_hi[j]),
        .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
        .p(preg[j][i]));
`ifndef SYNTHESIS
      // the DSP's select register is hi0[j] (both are c_hi[j], loaded every cycle)
      always @(posedge clk) begin
        if (hi_v && u_pe.selr != s_hi0[j])
          $fatal(1, "otpu_mxu: PE %0d.%0d's select %0d is not s_hi0 %0d", j, i, u_pe.selr, s_hi0[j]);
        if (hi_v && j != 0 && {u_pe.whr, u_pe.wlr} !=
            (QD ? wc[j][i][16 * s_hq0[j] +: 16] : wc[j][i][15:0]))
          $fatal(1, "otpu_mxu: PE %0d.%0d's weights are not wc", j, i);
      end
`endif
    end
  end

  // ---- column ends: group g's chain ends (CPG of them: the first on A, the second on D) with
  // their B (skewed from S2 to the column end's input registers, S0 + CLS + 2 + j), the row's
  // first block at the accumulator's OPMODE register (two cycles later), the row sum in the
  // accumulator CLS + 6 + j edges after S0, deskewed to S0 + LROW
  logic [MCOLS-1:0][47:0] acc_al;
  for (genvar j = 0; j < MCOLS; j++) begin : g_col_end
    logic [7:0][23:0] ca, cd;
    logic [7:0][16:0] bmd;
    logic             fst;
    logic [47:0]      acc;
    always_comb
      for (int g = 0; g < 8; g++) begin
        ca[g] = preg[j][g * CPG * CLS + CLS - 1];
        cd[g] = (CPG == 2) ? preg[j][(g * CPG + 1) * CLS + CLS - 1] : 24'd0;
      end
    otpu_skew #(.W(8 * 17), .N(CLS - 1 + j)) u_bm (.clk, .en(1'b1), .d(bm2[j]), .q(bmd));
    otpu_skew #(.W(1), .N(CLS + 4 + j)) u_f (.clk, .en(1'b1), .d(m0.v && m0.first), .q(fst));
    otpu_mxce #(.CPG(CPG)) u_ce (.clk, .a(ca), .d(cd), .bm(bmd), .first(fst), .acc);
    otpu_skew #(.W(48), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(1'b1), .d(acc), .q(acc_al[j]));
  end
  // the row's last block, PAIR and M at S0 + LROW; Sw and the Sa' from S1
  logic             tl_v, tl_pr, tl_qd;
  logic [7:0]       tl_M;
  logic [MCOLS-1:0][31:0] tl_sw;            // (per column: RL's lanes' Sw)
  logic [MCOLS-1:0][31:0] tl_sa;
  otpu_delay #(.W(1 + 1 + 8), .N(LROW)) u_tl (.clk, .en(1'b1), .d({m0.v && m0.last, pr0, M0}),
                                             .q({tl_v, tl_pr, tl_M}));
  if (QD) begin : g_tlq
    otpu_delay #(.W(1), .N(LROW)) u_tq (.clk, .en(1'b1), .d(qd0), .q(tl_qd));
  end else begin : g_ntlq
    assign tl_qd = 1'b0;
  end
  otpu_skew #(.W(2 * MCOLS * 32), .N(LROW - 1)) u_ts (.clk, .en(1'b1), .d({sw1, sap1}),
                                                     .q({tl_sw, tl_sa}));
  // PAIR: row j (j < M <= MCOLS/2) adds its partner column j + M's integer; QUAD (T2): row j
  // (j < M <= MCOLS/4) adds columns j + M, j + 2M and j + 3M's, two adders deep. Registered for
  // the result FIFO (rv: push)
  logic             rv;
  logic [MCOLS-1:0][47:0] rowi;
  logic [MCOLS-1:0][31:0] rsa;
  logic [MCOLS-1:0][31:0] rsw;
  always_ff @(posedge clk) begin
    rv <= !rst && tl_v;
    for (int j = 0; j < MCOLS; j++) begin
      logic [47:0] pb, p2, p3;
      pb = '0; p2 = '0; p3 = '0;
      for (int m = 1; m <= MCOLS / 2; m++)
        if (j < MCOLS / 2 && j + m < MCOLS && tl_M == 8'(m)) pb = acc_al[j + m];
      for (int m = 1; m <= MCOLS / 4; m++)
        if (j < MCOLS / 4 && tl_M == 8'(m)) begin
          p2 = acc_al[j + 2 * m];
          p3 = acc_al[j + 3 * m];
        end
      if (QD && tl_qd && j < MCOLS / 4) rowi[j] <= (acc_al[j] + pb) + (p2 + p3);
      else rowi[j] <= (tl_pr && j < MCOLS / 2) ? acc_al[j] + pb : acc_al[j];
    end
    rsa <= tl_sa;
    rsw <= tl_sw;
  end

  // ================================================================== result FIFO
  // Rows in LUT RAM; the drain reads the head row from flip-flops. rf_q trails the head by the
  // last cycle's pop (rf_hs: the head is rf_q + rf_hs), so the rows are read at rf_q, rf_q + 1
  // and rf_q + 2 (a row written this cycle bypassed) with no grant on the address; each cycle the
  // head row and the row after it are registered (rh0, rh1) and the last pop picks the head
  // (rfh). One RAM copy per read address: a single read port each infers as LUT RAM.
  // A row: MCOLS integers, MCOLS Sa', MCOLS Sw (one value a row unless RL: its lanes')
  localparam int RW_ = MCOLS * 48 + MCOLS * 32 + MCOLS * 32;
  logic [RFW-1:0] rf_q, rf_t;
  logic [RFW:0]   rf_n;
  logic           rf_hs;                     // the last cycle popped a row
  wire            rf_push = rv;
  logic [RW_-1:0] rowp, rh0, rh1, rfh;
  logic [2:0][RW_-1:0] rq;                   // rows rf_q + c, as they are after this cycle
  logic [2:0][RFW-1:0] rf_qc;
  always_comb begin
    rowp = {rsw, rsa, rowi};
    for (int c = 0; c < 3; c++) rf_qc[c] = rf_q + RFW'(c);
  end
  // each copy has its own write enable and address (replicas of rv and rf_t): one register
  // driving the three copies' write ports was the worst slow-clock path of the MX experiment's
  // 4-column OOC (docs/mx_mxu.md section 5, +0.329 ns)
  for (genvar c = 0; c < 3; c++) begin : g_rfm
    (* ram_style = "distributed" *) logic [RW_-1:0] m [RF];
    (* keep *) logic           wv;
    (* keep *) logic [RFW-1:0] wt;
    always_ff @(posedge clk) begin
      wv <= !rst && tl_v;
      if (rst) wt <= '0;
      else if (wv) wt <= wt + 1'b1;
    end
    always_ff @(posedge clk) if (!rst && wv) m[wt] <= rowp;
    assign rq[c] = (wv && wt == rf_qc[c]) ? rowp : m[rf_qc[c]];
`ifndef SYNTHESIS
    always @(posedge clk) if (!rst && (wv != rf_push || wt != rf_t))
      $fatal(1, "otpu_mxu: result FIFO copy %0d's write is not rf_push / rf_t", c);
`endif
  end
  always_ff @(posedge clk) begin
    rh0 <= rf_hs ? rq[1] : rq[0];           // the head (rf_q + rf_hs) as it is after this cycle
    rh1 <= rf_hs ? rq[2] : rq[1];           // the row after it
  end
  assign rfh = rf_hs ? rh1 : rh0;
  wire [MCOLS-1:0][47:0] rfh_i  = rfh[0 +: MCOLS * 48];
  wire [MCOLS-1:0][31:0] rfh_sa = rfh[MCOLS * 48 +: MCOLS * 32];
  wire [MCOLS-1:0][31:0] rfh_sw = rfh[MCOLS * 80 +: MCOLS * 32];

  // ================================================================== drain
  // lanes this cycle: results dj .. dj+ncnt-1 of the head row, stopping at a bank conflict
  logic [31:0] drow [MCOLS];                 // replay: dad of the head row's group 0
  logic [31:0] dad [MCOLS];                  // head row's TMEM addresses: out + n + j * ors,
                                             // kept incrementally
  logic [MW-1:0] dj;
  logic [MW-1:0] ncnt;
  logic [7:0]    dg;                         // the group of the row being drained
  logic [7:0]    dg1;                        // dg + 1, kept alongside
  // The drain's step is registered, with its next value computed a cycle ahead (below, at the
  // head's entries): no compare or subtract between the head's entry and the pick. d_last: dg1
  // == c_G; ncnt = min(run, c_Mn - dj) and dln its lanes (k < ncnt); d_fin: the step drains the
  // group's last results (dj + ncnt == c_Mn). drain_go's counts are tested by registered flags
  // (rf_nz: rf_n != 0, rl_nz: rows_live != 0) for the same reason.
  logic          d_last, d_fin, rf_nz, rl_nz;
  logic [NL-1:0] dln;
  // the head's rows left to drain (RL: the last one has Mz results, not Ml): d_lz the row being
  // drained is the last, d_lz2 the next one is
  logic [15:0]   d_rem;
  logic          d_lz, d_lz2;
  wire  [MW-1:0] c_Ml = d_lz ? h.Mz : h.Ml;
  wire  [MW-1:0] c_Mn = d_last ? c_Ml : MW'(MCOLS);   // its results
  // The drain lanes' addresses, dad[dj + k], are registered (dal): set wherever dj or dad move
  logic [NL-1:0][31:0] dal;
  function automatic logic [NL-1:0][31:0] dlane_a(input logic [31:0] a [MCOLS],
                                                  input logic [MW-1:0] j);
    for (int k = 0; k < NL; k++) dlane_a[k] = (32'(j) + 32'(k) < MCOLS) ? a[MW'(j) + MW'(k)] : '0;
  endfunction
  logic [NL-1:0][47:0] dx_l;
  logic [NL-1:0][31:0] dsa_l, dsw_l;
  logic [NL-1:0][7:0]  dcol_l;
  // (the result FIFO's head row is the head command's while it has rows not yet drained)
  wire  drain_go = rf_nz && rl_nz && (!c_asc || al_st == 2'd2);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && drain_go && dg1 != dg + 8'd1) $fatal(1, "otpu_mxu: dg1 %0d, dg %0d", dg1, dg);
`endif
  // Results j and j' of a row sit ors * (j' - j) words apart: the same bank iff that is a multiple
  // of LANES. So consecutive results are conflict-free in runs of LANES / gcd(ors, LANES)
  // (capped at NL), a per-command constant: the lane count is min(run, M - dj).
  // A step of a group with `left` results still to drain: {drains them all, lanes}
  function automatic logic [MW:0] dstep(input logic [MW-1:0] left, input logic [MW-1:0] run);
    return {left <= run, (run < left) ? run : left};
  endfunction
  function automatic logic [NL-1:0] dlanes(input logic [MW-1:0] cnt);
    for (int k = 0; k < NL; k++) dlanes[k] = 32'(k) < 32'(cnt);
  endfunction
`ifndef SYNTHESIS
  // the registered step is its definition from the head's entry, dg1, dj and the counts
  always @(posedge clk) if (!rst && q_n != 0) begin
    logic [MW-1:0] mn, lf;
    mn = (dg1 == c_G) ? c_Ml : MW'(MCOLS);
    lf = mn - dj;
    if (d_last != (dg1 == c_G) || ncnt != ((c_run < lf) ? c_run : lf) || dln != dlanes(ncnt) ||
        d_fin != (MW'(dj + ncnt) == mn))
      $fatal(1, "otpu_mxu: drain step d_last %0d ncnt %0d d_fin %0d (dg1 %0d G %0d dj %0d M %0d run %0d)",
             d_last, ncnt, d_fin, dg1, c_G, dj, mn, c_run);
  end
  always @(posedge clk) if (!rst && q_n != 0 && dal != dlane_a(dad, dj))
    $fatal(1, "otpu_mxu: the drain lanes' addresses are not dad[dj + k] (dj %0d)", dj);
  always @(posedge clk) if (!rst && (rf_nz != (rf_n != 0) || rl_nz != (rows_live != 0)))
    $fatal(1, "otpu_mxu: rf_nz %0d (rf_n %0d), rl_nz %0d (rows_live %0d)", rf_nz, rf_n, rl_nz, rows_live);
`endif
  always_comb begin
    dx_l = '0; dsa_l = '0; dsw_l = '0; dcol_l = '0;
    for (int k = 0; k < NL; k++)
      if (32'(dj) + 32'(k) < MCOLS) begin
        dx_l[k] = rfh_i[MW'(dj) + MW'(k)];
        dsa_l[k] = rfh_sa[MW'(dj) + MW'(k)];
        dsw_l[k] = rfh_sw[MW'(dj) + MW'(k)];
        dcol_l[k] = 8'(dj) + 8'(k);
      end
  end
`ifndef SYNTHESIS
  // PAIR reads a block's data and its partner's in one line (int8: a byte pair)
  always @(posedge clk)
    if (a_req && i_pair && !i_rl && !i_sws && !i_rsc && (i_w4 ? scale_addr[2] : scale_addr[0]))
      $fatal(1, "otpu_mxu: PAIR scales must be 8-byte (int8: 2-byte) aligned");
  // RL: an advance's A bytes A-aligned in a line, the group's Sw words 4 L aligned
  always @(posedge clk)
    if (a_req && i_rl && !i_sws && !i_rsc && (scale_addr & (32'(i_sstep) - 1)) != 0)
      $fatal(1, "otpu_mxu: RL scales misaligned (%h, %0d bytes an advance)", scale_addr, i_sstep);
  always @(posedge clk)
    if (a_req && i_rl && (i_sws || i_rsc) && (srow_addr[LB-1:0] & LB'((32'd4 << i_lg) - 1)) != 0)
      $fatal(1, "otpu_mxu: RL Sw words misaligned (%h)", srow_addr);
  // t2's QUAD reads a chunk's four words in one 16-byte group; int8's QUAD four bytes in a
  // word (4-bit and NVFP4: 8 / 16-byte aligned, across a line in two requests)
  always @(posedge clk)
    if (a_req && i_qd && !i_rl && !i_sws && !i_rsc && (i_t2 ? scale_addr[3:2] != 2'd0 :
                                              !i_w4 ? scale_addr[1:0] != 2'd0 :
                                              i_wf == WF_NV ? scale_addr[3:0] != 4'd0 :
                                              scale_addr[2:0] != 3'd0))
      $fatal(1, "otpu_mxu: QUAD scales misaligned (%h)", scale_addr);
  // the lane count is what a greedy pick that stops at the first bank conflict would take
  always @(posedge clk) if (drain_go) begin
    logic [LANES-1:0] used;
    int n;
    used = '0; n = 0;
    for (int k = 0; k < NL && 32'(dj) + 32'(k) < 32'(c_Mn); k++) begin
      if (used[dad[MW'(dj) + MW'(k)][BW-1:0]]) break;
      used[dad[MW'(dj) + MW'(k)][BW-1:0]] = 1'b1;
      n++;
    end
    if (n != 32'(ncnt)) $fatal(1, "otpu_mxu: drain lanes %0d, greedy pick %0d", ncnt, n);
  end
`endif
  wire drain_row_done = drain_go && d_fin;   // dj + ncnt == c_Mn
  // the next drain step's candidates, from registers (the grant only picks one): the rest of the
  // group, or (d_fin) the next group, the row's last (d_nl) or not; a new head's is its entry's
  wire [MW-1:0]    d_rest = c_Mn - dj - ncnt;
  wire             d_nl = d_last ? h.g1 : (dg1 + 8'd1 == c_G);
  wire [MW:0]      st_mid = dstep(d_rest, c_run);
  wire [MW:0]      st_grp = d_nl ? (d_lz2 ? {h.dflz, h.dnlz} : {h.dfl, h.dnl})
                                 : dstep(MW'(MCOLS), c_run);
  wire [NL-1:0]    ln_mid = dlanes(st_mid[MW-1:0]), ln_grp = dlanes(st_grp[MW-1:0]);
  wire             d_pop = t_gnt && drain_row_done;   // a row leaves the result FIFO
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rf_n != 0 && rfh != g_rfm[0].m[RFW'(rf_q + RFW'(rf_hs))])
    $fatal(1, "otpu_mxu: registered head row is not row %0d", rf_q + RFW'(rf_hs));
`endif

  // ---- the epilogue lanes: the pick registered (P1), i2f48, x Sw, x Sa' (otpu_mx_lane), the
  // pick's lanes, addresses and columns alongside; everything advances with the TMEM grant. The
  // lane output (lo_*) is the drain's TMEM request: a write, or ACC's read of the old values
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad;
    logic [NL-1:0][7:0]     col;
  } lmeta_t;
  lmeta_t p1, lo;
  logic [NL-1:0][31:0] lo_ad;
  logic [NL-1:0][7:0]  lo_col;
  logic [NL-1:0][47:0] p1_x;
  logic [NL-1:0][31:0] p1_sa, p1_sw;
  f32_t                lo_y [NL];
  always_ff @(posedge clk)
    if (rst) p1 <= '0;
    else if (t_gnt) begin
      p1.v <= drain_go;
      p1.m <= drain_go ? dln : '0;
      p1.ad <= dal;
      p1.col <= dcol_l;
    end
  always_ff @(posedge clk)
    if (t_gnt) begin
      p1_x <= dx_l; p1_sa <= dsa_l; p1_sw <= dsw_l;
    end
  for (genvar k = 0; k < NL; k++) begin : g_lane
    otpu_mx_lane #(.LM(LM)) u_l (.clk, .en(t_gnt), .x(p1_x[k]), .sw(p1_sw[k]), .sa(p1_sa[k]),
                                 .y(lo_y[k]));
  end
  // the picks' addresses and columns in a delay line; their valid and lane bits in reset
  // registers (a run stopped in the middle of a drain leaves no write to come out after rst)
  lmeta_t lo_p;
  otpu_delay #(.W($bits(lmeta_t)), .N(LLAT - 2)) u_lm (.clk, .en(t_gnt), .d(p1), .q(lo_p));
  logic [LLAT-2:0][NL:0] lvm;
  always_ff @(posedge clk)
    if (rst) lvm <= '0;
    else if (t_gnt) begin
      lvm[0] <= {p1.v, p1.m};
      for (int s = 1; s <= LLAT - 2; s++) lvm[s] <= lvm[s-1];
    end
  always_ff @(posedge clk)
    if (t_gnt) begin
      lo_ad <= lo_p.ad;
      lo_col <= lo_p.col;
    end
  always_comb begin
    {lo.v, lo.m} = lvm[LLAT-2];
    lo.ad = lo_ad;
    lo.col = lo_col;
  end
  logic [3:0] ln_n;                          // picks in the lanes (any nonzero = busy)

  // read-modify-write pipeline for ACC: request now, read from the request register next cycle
  // (rA, TMEM's input stage), data the cycle after (r0), (old*alpha)+new, write
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad, nv;
    logic [NL-1:0][7:0]     col;
  } rmw_t;
  rmw_t r0, rw;                               // r0: data arriving now; rw: at the write stage
  rmw_t rA;                                   // the read in the request register (t_ren)
  rmw_t rx;                                   // RMAX (no ACC): written lanes, compared next cycle
  f32_t ry [NL];
  // r1: r0 one granted cycle later, with the old values registered (xo): no path from the TMEM
  // block RAMs into the fmadd's DSP inputs in one cycle. The ASCALE factor (alr) is selected from
  // r0 as it moves into r1, so the fmadd's b operand is a flop too (no mux in front of the DSP);
  // c_asc and alpha cannot change while a valid entry sits in r0/r1 (the head changes only once
  // drained, alpha loads only before an ASCALE command drains)
  rmw_t r1;
  f32_t xo [NL], alr [NL];
  always_ff @(posedge clk)
    if (rst) r1 <= '0;
    else if (t_gnt) begin
      r1 <= r0;
      for (int k = 0; k < NL; k++) begin
        xo[k]  <= t_rdata[k];
        alr[k] <= c_asc ? alpha[c_rl ? '0 : r0.col[k][MW-2:0]] : F_ONE;
      end
    end
  for (genvar k = 0; k < NL; k++) begin : g_rmw
    otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(t_gnt), .a(xo[k]), .b(alr[k]), .c(r1.nv[k]),
                                        .y(ry[k]));
  end
  otpu_delay #(.W($bits(rmw_t)), .N(LM + LA)) u_rw (.clk, .en(t_gnt), .d(r1), .q(rw));
  logic [3:0] rmw_n;                          // rows' lanes in flight (any nonzero = busy)

  // RMAX: the running max is kept as its sort key (mk = fkey(max); the max is always stored
  // ftz'd, so fkey(max) is the key it is compared by); written out through unkey.
  logic [31:0] mk [MCOLS];
  logic [MCOLS-1:0] mx_have;
  logic mx_done;
  function automatic f32_t unkey(input logic [31:0] k);
    return k[31] ? {1'b0, k[30:0]} : ~k;
  endfunction
  // rw's column hits, one-hot, computed from r1 and carried alongside u_rw (same length and
  // enable); the last stage is a reset flop, not an SRL tap. c_rmax is fixed while entries are
  // in flight (the head changes only once drained)
  logic [NL-1:0][MCOLS-1:0] rh, rh_p, rwh, rxh;
  always_comb
    for (int k = 0; k < NL; k++)
      for (int j = 0; j < MCOLS; j++) begin
        rh[k][j]  = r1.v && c_rmax && r1.m[k] && (r1.col[k][MW-2:0] == (MW-1)'(j));
        rxh[k][j] = rx.v && rx.m[k] && (rx.col[k][MW-2:0] == (MW-1)'(j));
      end
  otpu_delay #(.W(NL * MCOLS), .N(LM + LA - 1)) u_rwh (.clk, .en(t_gnt), .d(rh), .q(rh_p));
  always_ff @(posedge clk) if (rst) rwh <= '0; else if (t_gnt) rwh <= rh_p;
  // per column: the candidate key is selected by the hits alone, the compare only makes the
  // enable. At most one lane hits a column per cycle (lanes drain distinct columns) and rx / rw
  // are exclusive (rx only for !c_acc commands, rw only for c_acc ones)
  logic [MCOLS-1:0] mx_hit, mx_upd;
  logic [31:0] mx_cand [MCOLS];
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      mx_hit[j] = 1'b0; mx_upd[j] = 1'b0; mx_cand[j] = '0;
      for (int k = 0; k < NL; k++) begin
        if (rxh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(rx.nv[k]);
          mx_upd[j] = mx_upd[j] | !mx_have[j] | (fkey(rx.nv[k]) > mk[j]);
        end
        if (rwh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(ry[k]);
          mx_upd[j] = mx_upd[j] | !mx_have[j] | (fkey(ry[k]) > mk[j]);
        end
      end
    end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && t_gnt)
      for (int j = 0; j < MCOLS; j++) begin
        int n;
        n = 0;
        for (int k = 0; k < NL; k++) n = n + int'(rxh[k][j]) + int'(rwh[k][j]);
        if (n > 1) $fatal(1, "otpu_mxu: several RMAX updates of column %0d in one cycle", j);
      end
`endif

  wire c_drained = c_act && !cl_ld && (pn || c_left == 0) && (rows_live == 0) && (rmw_n == 0) &&
                   (ln_n == 0) && !rA.v && !r0.v && !r1.v && !rx.v;
  // RMAX writes start a cycle after the head has drained: mx_q registers the test, so the
  // drained compares (c_left == 0, rows_live, rmw_n, ...) are off the TMEM write request and the
  // grant it feeds. The test holds while drained (nothing new enters the head until it
  // completes, which needs mx_done), and it is false in the completing cycle (mx_done), so a new
  // head never sees a stale mx_q.
  logic mx_q;
  // mx_go (mx_q && !mx_done) is a register of its own (mx_gq, r2.1): it starts the MXU's TMEM
  // write request (the RMAX lanes), which the slice's write-bank arbiter turns into the grant;
  // from mx_done through the request, the arbiter, the grant's read enables and the TMEM's
  // replay logic was r2.1's m8 163ab46d's one failing path (-0.003 ns, 14 levels)
  logic mx_gq;
  wire mx_go     = mx_gq;
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && mx_gq != (mx_q && !mx_done))
    $fatal(1, "otpu_mxu: mx_gq %0d, mx_q %0d, mx_done %0d", mx_gq, mx_q, mx_done);
`endif
  wire c_fin     = c_drained && (!c_rmax || mx_done || c_tz);
  // the head completes (while the pop head may pop the next command's chunks)
  wire c_done    = c_fin;
  wire al_go     = c_act && c_asc && al_st == 2'd0;

  // RMAX lanes (mx_i + k < c_M), registered as the drain's; next: mx_i + LANES. Their TMEM
  // addresses h.mxo + mx_i + k are registered too (mxl, loaded from h.mxo while no RMAX write
  // runs: mx_go is never set in a head's first cycle), so no adder precedes the write request
  logic [LANES-1:0] mxm, mxm_nx;
  logic [LANES-1:0][31:0] mxl;
  always_comb for (int k = 0; k < LANES; k++) mxm_nx[k] = 32'(mx_i) + LANES + 32'(k) < 32'(c_Mr);
  // RL: the one RMAX word is the max over the lanes' columns' running maxima (the key max of
  // those that saw a value), registered (the maxima settle before the head drains)
  logic [31:0] mkr;
  always_ff @(posedge clk) begin
    logic [31:0] m;
    logic        have;
    m = '0; have = 1'b0;
    for (int j = 0; j < MCOLS; j++)
      if (mx_have[j] && (!have || mk[j] > m)) begin
        m = mk[j]; have = 1'b1;
      end
    mkr <= m;
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && q_n != 0)
    for (int k = 0; k < LANES; k++)
      if (mxm[k] != (32'(mx_i) + 32'(k) < 32'(c_Mr)))
        $fatal(1, "otpu_mxu: RMAX lane %0d: mxm %0d, mx_i %0d, M %0d", k, mxm[k], mx_i, c_M);
  always @(posedge clk) if (!rst && mx_go)
    for (int k = 0; k < LANES; k++)
      if (mxl[k] != h.mxo + 32'(mx_i) + 32'(k))
        $fatal(1, "otpu_mxu: RMAX lane %0d address %0h, mxo %0h, mx_i %0d", k, mxl[k], h.mxo, mx_i);
`endif

  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the read request is a register, a
  // stage of the grant's pipeline (it advances with t_gnt), so no unit logic (the ASCALE
  // addresses' adder and compare, the drain's lane mux) precedes TMEM's lane routing, ranks and
  // block RAM pins; the word arrives a cycle later (rA, al_st 3)
  logic [LANES-1:0]       t_ren_c;
  logic [LANES-1:0][31:0] t_raddr_c;
  always_ff @(posedge clk)
    if (rst) t_ren <= '0;
    else if (t_gnt) begin t_ren <= t_ren_c; t_raddr <= t_raddr_c; end
  always_comb begin
    t_ren_c = '0; t_raddr_c = '0; t_wen = '0; t_waddr = '0; t_wdata = '0;
    if (lo.v) begin
      for (int k = 0; k < NL; k++) begin
        if (c_acc) begin
          t_ren_c[k] = lo.m[k];
          t_raddr_c[k] = lo.ad[k];
        end else begin
          t_wen[k] = lo.m[k];
          t_waddr[k] = lo.ad[k];
          t_wdata[k] = lo_y[k];
        end
      end
    end
    if (rw.v) begin
      for (int k = 0; k < NL; k++) begin
        if (rw.m[k]) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = rw.ad[k];
          t_wdata[k] = ry[k];
        end
      end
    end
    // ASCALE factors and RMAX values move LANES words per cycle (consecutive words: distinct
    // banks)
    if (al_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(al_i) + 32'(k) < 32'(c_Mr)) begin
          t_ren_c[k] = 1'b1;
          t_raddr_c[k] = c_asa + 32'(al_i) + 32'(k);
        end
      end
    end
    if (mx_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (mxm[k]) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = mxl[k];
          t_wdata[k] = unkey(c_rl ? mkr : mk[MW'(32'(mx_i) + 32'(k))]);
        end
      end
    end
  end

  // ---- statistics for the profiler (per completed command): cycles the stream was starved
  // (work, no chunk), backpressured (chunks, no consumption), the drain frozen by the TMEM grant,
  // the issuer denied a DRAM port. The conditions are registered (st_c) and summed a cycle
  // late, off the grant paths: a command's count is st + g, g the last cycle's pending bit (none
  // after a command's end: that cycle's conditions belong to no command).
  logic [31:0] st_starve, st_bp, st_frz, st_deny;
  logic [3:0]  st_c, st_g;                  // {deny, frz, bp, starve}
  logic        st_f;                        // the last cycle ended a command
  assign st_g = st_f ? 4'd0 : st_c;

  // the issuer's next line compares (on the next addresses, registered)
  function automatic logic same_line(input logic [31:0] a, input logic [31:0] b);
    return a[31:LB] == b[31:LB];
  endfunction

  always_ff @(posedge clk) begin
    done <= 1'b0;
    pf_u <= 1'b0;
    if (rst) begin
      i_act <= 1'b0; i_ini <= 1'b0; i_sp2 <= 1'b0;
      q_n <= '0; pn <= 1'b0; rows_p <= '0;
      occ <= '0; tg_n <= '0; tg_t <= '0; tg_h <= '0;
      f_head <= '0; f_tail <= '0; f_count <= '0; f_rd <= '0; f_tl <= '0; f_rdl <= '0;
      s_head <= '0; s_tail <= '0; s_count <= '0; s_rd <= '0;
      ck <= '0; cg <= '0; dg <= '0; dg1 <= 8'd1; c_left <= '0; cl_ld <= 1'b0; rows_live <= '0;
      rf_q <= '0; rf_hs <= 1'b0; rf_t <= '0; rf_n <= '0; rf_nz <= 1'b0; rl_nz <= 1'b0;
      dj <= '0; mx_done <= 1'b0; mx_have <= '0;
      d_last <= 1'b0; d_fin <= 1'b0; ncnt <= '0; dln <= '0; mxm <= '0;
      d_rem <= '0; d_lz <= 1'b0; d_lz2 <= 1'b0;
      al_st <= 2'd0; al_i <= '0; mx_i <= '0;
      r0 <= '0; rA <= '0; rx <= '0; rmw_n <= '0; ln_n <= '0;
      st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      st_c <= '0; st_f <= 1'b0; mx_q <= 1'b0; mx_gq <= 1'b0;
    end else begin
      logic [1:0] qn;
      logic [RFW:0] rn;
      logic [RFW:0] rl, rp;
      qent_t hn, nn;                            // the entries' next values (before a swap)
      qent_t hp, np;                            // the same without this cycle's accept (pop_q's:
                                                // a command accepted this cycle is not popped
                                                // before cl_ld, which reloads pop_q)
      logic pnx;
      qn = q_n;
      rn = rf_n;
      rl = rows_live;
      rp = rows_p;
      hn = h;
      nn = n;
      hp = h;
      np = n;
      pnx = pn;
      cl_ld <= 1'b0;
      if (cl_ld) c_left <= h.total;
      // ---- accept a command (the head if the queue is empty, else the next entry); the issuer
      // loads its addresses in the next cycle (i_ini)
      if (start) begin
        if (q_n == 0) begin
          hn = cmd_e;                             // becomes the head now
          cl_ld <= 1'b1;
        end else nn = cmd_e;
        qn = qn + 1;
        if (!cmd_tz) begin
          i_ini  <= 1'b1;
          i_left <= cmd_total;
          i_KB   <= cmd_KBa;
          i_rs   <= cmd.w5 & ~32'd15;             // (rs bits 3:0 are t2's, QUAD's and RL's flags)
          i_srs  <= RLB ? {cmd.w7[31:1], 1'b0} : cmd.w7;   // (bit 0: the group walk)
          i_unit <= cmd.flags[0];
          i_rsc  <= cmd.flags[7] && !cmd.flags[0];
          i_wf   <= cmd.flags[5:4];
          i_w4   <= (cmd.flags[5:4] != WF_W8);
          i_pair <= cmd_pair;
          i_t2r  <= cmd_t2;
          i_qdr  <= cmd_qd;
          i_rlr  <= cmd_rl;
          i_gwr  <= cmd_gw;
          i_lg   <= cmd_lg;
          i_nb   <= cmd_nb;
          i_nl   <= cmd_nl;
          i_k0   <= cmd.w4[31:24];
          row_addr <= cmd.w1; chunk_addr <= cmd.w1;
          srow_addr <= cmd.w2;
        end
      end
      if (i_ini) begin
        logic [31:0] so, sa, sn;
        logic [5:0]  ss;
        // (QUAD: four blocks' data an advance, a wide port's int8 PAIR two bytes)
        case (i_wf)
          WF_W8: begin so = 32'd4 + 32'(i_k0); ss = i_qd ? 6'd4 : i_pair ? 6'd2 : 6'd1; end
          WF_NV: begin
            so = 32'd16 + {21'd0, i_k0, 3'd0}; ss = i_qd ? 6'd32 : i_pair ? 6'd16 : 6'd8;
          end
          default: begin
            so = 32'd8 + {22'd0, i_k0, 2'd0}; ss = i_qd ? 6'd16 : i_pair ? 6'd8 : 6'd4;
          end
        endcase
        if (i_t2) begin                           // t2: a 16-byte header, a word a block
          so = 32'd16 + {22'd0, i_k0, 2'd0};
          ss = i_qd ? 6'd16 : i_pair ? 6'd8 : 6'd4;
        end
        if (i_rl || i_gw) begin                   // row lanes: a block's data of the L rows,
          logic [5:0] bs, A, H;                   // the advance's A bytes, the header H
          bs = (i_wf == WF_W8) ? 6'd1 : (i_wf == WF_NV) ? 6'd8 : 6'd4;
          A = 6'((32'(bs) * (i_t2 ? 32'd4 : i_w4 ? 32'd2 : 32'd1)) << i_lg);
          H = (6'(6'd4 << i_lg) > A) ? 6'(6'd4 << i_lg) : A;
          so = 32'(H) + ((32'(i_k0) * 32'(bs)) << i_lg);
          ss = A;
          if (i_gw) begin                         // the group walk: lane 0's offset, a block
            ss = 6'(bs << i_lg);                  // step of L bs; the row's lane and steps
            w_so0 <= so;
            w_ds  <= 32'(bs) - 32'd4;
            w_rd  <= i_rs - (32'(D) << i_lg) + 32'(D);
            w_sd  <= i_srs - (32'd4 << i_lg) + 32'd4;
            i_wl  <= '0;
          end
        end
        i_cstep <= i_gw ? (32'(D) << i_lg) : WP ? 32'(D) * (32'(i_nb) + 32'd1) : 32'(D);
        sa = srow_addr + so;
        sn = srow_addr + (i_gw ? 32'd4 : i_srs);
        i_ini <= 1'b0;
        i_act <= 1'b1;
        i_k <= '0;
        i_klast <= (i_KB == 16'd1);
        i_l1 <= (i_left == 32'd1);
        i_soff <= so;
        i_sstep <= ss;
        scale_addr <= sa;
        srow_next <= sn;
        i_swd <= 1'b0;
        i_swh <= same_line(srow_addr, sa);
        i_swn <= same_line(sa, sn);
        i_sws <= !i_unit && !i_rsc && !same_line(srow_addr, sa);
      end
      // ---- release (in order): the head if not yet released, else the second entry
      if (go) begin
        if (q_n != 0 && !h.go) begin
          hn.go = 1'b1; hp.go = 1'b1;
        end else begin
          nn.go = 1'b1; np.go = 1'b1;
        end
      end
      // ---- issue: the row's Sw line alone, the first of two lines (PORTB > 1), or one advance
      // (a chunk, a wide port's chunks, and its line)
      if (go_iss) begin
        if (i_sws) begin
          i_swd <= 1'b1;
          i_sws <= 1'b0;
        end else if (i_spx) begin
          i_sp2 <= 1'b1;
          i_sa2 <= scale_addr + 32'(ALINE);
          if (i_k == 0 && !i_swd && i_swh) i_swd <= 1'b1;   // (the row's Sw came with it)
        end else begin
          i_sp2 <= 1'b0;
          i_left <= i_left - 1;
          i_l1 <= (i_left == 32'd2);
          if (i_left == 1) i_act <= 1'b0;
          if (i_klast) begin
            logic [31:0] sa, sn, rd, so;
            // (the group walk: the next row is the group's next lane, D and 4 on, its block
            // data bs - 4 further in; past the group's last lane the next group's lane 0)
            rd = i_gw ? ((i_wl == w_last) ? w_rd : 32'(D)) : i_rs;
            so = i_gw ? ((i_wl == w_last) ? w_so0 : i_soff + w_ds) : i_soff;
            sa = srow_next + so;
            sn = srow_next + (i_gw ? ((i_wl + 3'd1 == w_last) ? w_sd : 32'd4) : i_srs);
            i_k <= '0;
            i_klast <= (i_KB == 16'd1);
            row_addr <= row_addr + rd;
            chunk_addr <= row_addr + rd;
            srow_addr <= srow_next;
            srow_next <= sn;
            scale_addr <= sa;
            i_soff <= so;
            if (i_gw) i_wl <= (i_wl == w_last) ? 3'd0 : i_wl + 3'd1;
            // the next row's Sw came with this line, or is in its first advance's, or needs a
            // request of its own
            i_swd <= a_tag.swr;
            i_swh <= same_line(srow_next, sa);
            i_swn <= same_line(sa, sn);
            i_sws <= !i_unit && !i_rsc && !a_tag.swr && !same_line(srow_next, sa);
          end else begin
            logic [31:0] sa;
            sa = scale_addr + 32'(i_sstep);
            i_k <= i_k + 1;
            i_klast <= (i_k + 16'd2 == i_KB);
            // 4-bit: two blocks per chunk (PAIR: both in one advance); t2 four (PAIR two
            // advances, QUAD one)
            if (i_t2 ? (i_qd || (i_pair ? i_k[0] : i_k[1:0] == 2'd3)) :
                (!i_w4 || i_pair || i_k[0]))
              chunk_addr <= chunk_addr + i_cstep;
            scale_addr <= sa;
            i_swn <= same_line(sa, srow_next);
          end
        end
      end
      // ---- FIFO pushes
      if (b_push) begin
        f_tail <= f_tail + 1;
        for (int g = 0; g < FG; g++) f_tl[g] <= f_tl[g] + 1;
      end
      if (a_req) tg_t <= tg_t + 1'b1;
      if (a_rvalid) tg_h <= tg_h + 1'b1;
      tg_n <= tg_n + (a_req ? 1'b1 : 1'b0) - (rv1 ? 1'b1 : 1'b0);
      if (r_cpl) s_tail <= s_tail + 1'b1;
      f_count <= f_count + (b_push ? 1 : 0) - (fpop ? 1 : 0);
      occ <= occ + (b_req ? 1'b1 : 1'b0) - (fpop ? 1'b1 : 1'b0);
      s_count <= s_count + (r_cpl ? 1 : 0) - ((fpop && !c_unit) ? 1 : 0);
      // ---- pop one advance (the row's last group frees its chunk and scale entry)
      if (fpop) begin
        f_head <= f_head + 1;
        if (!c_unit) s_head <= s_head + 1;
      end
      if (pop) begin
        if (last_g) c_left <= c_left - 1;
        // the next read: the next entry, or back to the head for the row's next group
        if (last_k && !last_g) begin
          f_rd <= f_head;
          f_rdl <= {FG{f_head}};
          if (!c_unit) s_rd <= s_head;
        end else if (cdone) begin
          f_rd <= f_rd + 1;
          for (int g = 0; g < FG; g++) f_rdl[g] <= f_rdl[g] + 1;
          if (!c_unit) s_rd <= s_rd + 1;
        end
        if (ck == 0) begin
          if (!pn) rl = rl + 1;
          else rp = rp + 1;
        end
        ck <= last_k ? '0 : ck + 1;
        if (last_k) cg <= last_g ? '0 : cg + 1;
      end
      // ---- a finished row enters the result FIFO
      if (rf_push) begin
        rf_t <= rf_t + 1;
        rn = rn + 1;
      end
      // ---- drain (holds while the TMEM grant is withheld)
      if (t_gnt) begin
        r0 <= rA;
        rA <= '0;
        rx <= '0;
        if (lo.v) begin
          if (c_acc) begin
            rA.v <= 1'b1;
            rA.m <= lo.m;
            rA.ad <= lo.ad;
            for (int k = 0; k < NL; k++) rA.nv[k] <= lo_y[k];
            rA.col <= lo.col;
          end
          if (c_rmax && !c_acc) begin
            rx.v <= 1'b1;
            rx.m <= lo.m;
            for (int k = 0; k < NL; k++) rx.nv[k] <= lo_y[k];
            rx.col <= lo.col;
          end
        end
        if (drain_go) begin
          if (drain_row_done) begin
            dj <= '0;
            if (d_last) begin                       // the next weight row, group 0 (RL: the
              dg <= '0; dg1 <= 8'd1;                //   next group of L rows)
              for (int j = 0; j < MCOLS; j++) begin
                logic [31:0] a;
                a = drow[j] + 32'(h.dinc);
                dad[j] <= a;
                drow[j] <= a;
                if (j < NL) dal[j] <= a;
              end
              d_rem <= d_rem - 16'd1;
              d_lz <= d_lz2;
              d_lz2 <= (d_rem == 16'd3);
            end else begin                          // the next group of this row
              dg <= dg + 8'd1; dg1 <= dg1 + 8'd1;
              for (int j = 0; j < MCOLS; j++) begin
                logic [31:0] a;
                a = dad[j] + h.gs;
                dad[j] <= a;
                if (j < NL) dal[j] <= a;
              end
            end
            rn = rn - 1;
            rl = rl - 1;
          end else begin
            dj <= dj + ncnt;
            dal <= dlane_a(dad, dj + ncnt);
          end
        end
        // RMAX (rx: registered so the lane selection and the max compare are in different cycles)
        for (int j = 0; j < MCOLS; j++) begin
          if (mx_upd[j]) mk[j] <= mx_cand[j];
          if (mx_hit[j]) mx_have[j] <= 1'b1;
        end
        rmw_n <= rmw_n + ((lo.v && c_acc) ? 4'd1 : 4'd0) - (rw.v ? 4'd1 : 4'd0);
        ln_n <= ln_n + (drain_go ? 4'd1 : 4'd0) - (lo.v ? 4'd1 : 4'd0);
        if (mx_go) begin
          if (32'(mx_i) + LANES >= 32'(c_Mr)) mx_done <= 1'b1;
          else mx_i <= mx_i + 8'(LANES);
        end
        if (al_go) al_st <= 2'd1;
        if (al_st == 2'd1) al_st <= 2'd3;     // the read leaves the request register
        if (al_st == 2'd3) begin
          // one write per entry (a constant index): a variable-index write of several lanes makes
          // Vivado try, and crash while dissolving, a RAM for these MCOLS registers
          for (int j = 0; j < MCOLS; j++)
            if (32'(j) >= 32'(al_i) && 32'(j) < 32'(al_i) + LANES)
              alpha[j] <= t_rdata[$clog2(LANES)'(32'(j) - 32'(al_i))];
          if (32'(al_i) + LANES >= 32'(c_Mr)) al_st <= 2'd2;
          else begin
            al_i <= al_i + 8'(LANES);
            al_st <= 2'd0;
          end
        end
      end
      rf_n <= rn;
      rf_q <= rf_q + RFW'(rf_hs);
      rf_hs <= d_pop;
      // RMAX lane addresses: they advance with mx_i while the writes run, else h.mxo + k
      for (int k = 0; k < LANES; k++)
        if (!mx_go) mxl[k] <= h.mxo + 32'(k);
        else if (t_gnt && 32'(mx_i) + LANES < 32'(c_Mr)) mxl[k] <= mxl[k] + 32'(LANES);
      rows_live <= rl;
      rows_p <= rp;
      // rn != 0, rl != 0 from the counts' own zero / one tests: the grant enters last
      rf_nz <= rf_push || (rf_nz && !(rf_n == 1 && d_pop));
      rl_nz <= (pop && ck == 0 && !pn) || (rl_nz && !(rows_live == 1 && d_pop));
      // the pop head moves to the next command once the head's chunks have all popped, if the
      // pipeline treats both alike (PAIR, and M under PAIR: pr0, hi0 and M0 are read late)
      if (!pn && !cl_ld && c_left == 0 && q_n == 2'd2 && n.go && !c_done &&
          n.pair == h.pair && (!h.pair || n.M == h.M) && (!QD || n.qd == h.qd) &&
          (!RLB || (n.rl == h.rl && (!h.rl || n.M == h.M)))) begin
        pnx = 1'b1;
        c_left <= n.total;
      end
      // ---- statistics
      st_c <= {want_iss && !go_iss, !t_gnt && (drain_go || lo.v || rw.v || mx_go || al_go),
               more && f_count != 0 && !pop, more && f_count == 0};
      st_f <= c_done;
      mx_q <= c_drained && c_rmax && !mx_done && !c_tz;
      // mx_q's next value and mx_done's (set by the last RMAX lanes' grant; a completing head
      // clears it, and then mx_q's next is low: it completes only with mx_done or c_tz)
      mx_gq <= c_drained && c_rmax && !mx_done && !c_tz &&
               !(t_gnt && mx_go && 32'(mx_i) + LANES >= 32'(c_Mr));
      st_starve <= st_starve + 32'(st_g[0]);
      st_bp <= st_bp + 32'(st_g[1]);
      st_frz <= st_frz + 32'(st_g[2]);
      st_deny <= st_deny + 32'(st_g[3]);
      // ---- the consumer's command is complete
      if (c_done) begin
        done <= 1'b1;
        qn = qn - 1;
        dg <= '0; dg1 <= 8'd1;
        if (!pn) begin                            // the pop head moves along
          ck <= '0; cg <= '0;
          // the next head's chunk count: the queued entry's, or (cl_ld) the total of a command
          // accepted this cycle, a cycle later
          c_left <= (q_n == 2'd2) ? n.total : '0;
          if (start && q_n == 2'd1) cl_ld <= 1'b1;
        end else begin                            // it already runs the next command: its rows
          rows_live <= rp;
          rl_nz <= (rp != 0);
          rows_p <= '0;
        end
        pnx = 1'b0;                               // the next command is both heads now
        dj <= '0;
        for (int j = 0; j < MCOLS; j++) begin
          logic [31:0] a;
          a = (start && q_n == 2'd1) ? cmd.w3 + cmd_e.jo[j] : n.out + n.jo[j];
          dad[j] <= a;
          drow[j] <= a;
          if (j < NL) dal[j] <= a;
        end
        mx_done <= 1'b0; mx_have <= '0;
        al_st <= 2'd0; al_i <= '0; mx_i <= '0;
        pf_u <= 1'b1;
        pf_uv <= {st_deny + 32'(st_g[3]), st_frz + 32'(st_g[2]), st_bp + 32'(st_g[1]),
                  st_starve + 32'(st_g[0])};
        st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      end
      // the head's output base (set when a command becomes head)
      if (start && q_n == 0) begin
        for (int j = 0; j < MCOLS; j++) begin
          logic [31:0] a;
          a = cmd.w3 + cmd_e.jo[j];
          dad[j] <= a;
          drow[j] <= a;
          if (j < NL) dal[j] <= a;
        end
      end
      // the entries: completing the head swaps them (the next entry, or the command accepted
      // this cycle, becomes the head; the old head stays in n)
      begin
        qent_t hx, nx, hpx, npx;
        if (c_done) begin
          hx = nn; nx = hn; hpx = np; npx = hp;
        end else begin
          hx = hn; nx = nn; hpx = hp; npx = np;
        end
        h <= hx; n <= nx;
        // the drain's next step: a new head starts at its first row's group 0 (all of it if the
        // row is one group); a drain step moves on in the group, or to the next group / row
        if (c_done || (start && q_n == 0)) begin
          logic [MW:0] st;
          st = hx.g1 ? (hx.nrow == 16'd1 ? {hx.dflz, hx.dnlz} : {hx.dfl, hx.dnl})
                     : dstep(MW'(MCOLS), hx.run);
          d_last <= hx.g1;
          {d_fin, ncnt} <= st;
          dln <= dlanes(st[MW-1:0]);
          d_rem <= hx.nrow;
          d_lz <= (hx.nrow == 16'd1);
          d_lz2 <= (hx.nrow == 16'd2);
          for (int k = 0; k < LANES; k++) mxm[k] <= 32'(k) < 32'((RLB && hx.rl) ? 8'd1 : hx.M);
        end else begin
          if (t_gnt && drain_go) begin
            if (!d_fin) begin
              {d_fin, ncnt} <= st_mid;
              dln <= ln_mid;
            end else begin
              d_last <= d_nl;
              {d_fin, ncnt} <= st_grp;
              dln <= ln_grp;
            end
          end
          if (t_gnt && mx_go && 32'(mx_i) + LANES < 32'(c_Mr)) mxm <= mxm_nx;
        end
        // the pop head's entry, from the entries, not from the command being accepted: a
        // command that becomes the head as it is accepted gets its entry a cycle later (cl_ld;
        // the pipeline is empty then, and nothing pops before c_left loads)
        pop_q <= pnx ? npx : hpx;
      end
      pn <= pnx;
      q_n <= qn;
    end
  end
`ifndef SYNTHESIS
  // the head / next mapping relies on the rdy contract (start only with a free entry); the tags
  // in flight never exceed TD (the issuer waits), nor the scale entries the chunk slots
  always @(posedge clk) if (!rst) begin
    if (start && q_n >= 2'd2) $fatal(1, "otpu_mxu: start with %0d commands queued", q_n);
    if (start && (i_act || i_ini)) $fatal(1, "otpu_mxu: start while the issuer streams");
    if (q_n != 0 && !cl_ld && pop_q != (pn ? n : h))
      $fatal(1, "otpu_mxu: pop_q is not the pop head's entry");
    if (32'(tg_n) > TD) $fatal(1, "otpu_mxu: %0d port A requests in flight", tg_n);
    if (32'(s_count) > DEPTH) $fatal(1, "otpu_mxu: scale FIFO overflow (%0d)", s_count);
    if (rv1 && tg_n == 0) $fatal(1, "otpu_mxu: a port A response without a request");
  end
`endif

endmodule

// A shared epilogue lane (docs/mx_mxu.md 4): one result a cycle, once per output:
//   v  = i2f48(x)            round to nearest even (|x| < 2^47: normal or zero, no overflow)
//   y  = (v * sw) * sa       two otpu_fmul (sa: Sa * 2^-W, already scaled)
// Latency 4 + 2 LM: |x| | leading zeros | normalize | round | x sw (LM) | x sa (LM). Every
// register advances with en (the MXU's TMEM grant).
module otpu_mx_lane
  import otpu_fp::*;
#(parameter int LM = 2) (
  input  logic         clk,
  input  logic         en,
  input  logic [47:0]  x,
  input  f32_t         sw,
  input  f32_t         sa,
  output f32_t         y
);
  function automatic logic [5:0] lzc48(input logic [47:0] m);
    if (m[47:16] != 0) return lzc32(m[47:16]);
    return 6'd32 + 6'(lzc32({m[15:0], 16'd0}));
  endfunction
  logic        s1, z1, s2, z2, s3, z3;
  logic [47:0] g1, g2, g3;
  logic [5:0]  lz2, lz3;
  f32_t        fv;
  always_ff @(posedge clk) if (en) begin
    // L1: sign, magnitude
    s1 <= x[47];
    z1 <= (x == 0);
    g1 <= x[47] ? (~x + 48'd1) : x;
    // L2: leading zeros
    s2 <= s1; z2 <= z1; g2 <= g1;
    lz2 <= lzc48(g1);
    // L3: normalize (the leading one at bit 47)
    s3 <= s2; z3 <= z2; lz3 <= lz2;
    g3 <= g2 << lz2;
    // L4: round to 24 bits, nearest even
    begin
      logic [23:0] mm;
      logic [24:0] mr;
      logic g, st;
      logic [8:0] e;
      e = 9'd174 - 9'(lz3);                     // 127 + 47 - lz
      mm = g3[47:24];
      g = g3[23];
      st = |g3[22:0];
      mr = {1'b0, mm} + ((g && (st || mm[0])) ? 25'd1 : 25'd0);
      if (mr[24]) e = e + 9'd1;                 // mr == 2^24: its fraction bits are 0
      fv <= z3 ? F_ZERO : {s3, e[7:0], mr[22:0]};
    end
  end
  f32_t swd, sad, y1;
  otpu_delay #(.W(32), .N(4)) u_sw (.clk, .en, .d(sw), .q(swd));
  otpu_delay #(.W(32), .N(4 + LM)) u_sa (.clk, .en, .d(sa), .q(sad));
  otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en, .a(fv), .b(swd), .y(y1));
  otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en, .a(y1), .b(sad), .y);
endmodule

// Simple dual-port block RAM, registered read with enable (read-first: a read of the word
// written at the same edge returns the old word).
module otpu_ram_sdp #(parameter int W = 32, parameter int N = 1024) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  (* ram_style = "block" *) logic [W-1:0] mem [N];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  always_ff @(posedge clk) if (re) rd <= mem[ra];
endmodule

// N-cycle delay line with an enable: no reset and no forced last flip-flop (unlike otpu_delay), so
// a long one maps to shift-register LUTs and its last stage may move into a DSP48's input register
// (the systolic MXU's operand skews)
module otpu_skew #(parameter int W = 8, parameter int N = 1) (
  input  logic         clk,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_regs
    logic [W-1:0] r [N];
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N; k++) r[k] <= r[k-1];
    end
    assign q = r[N-1];
  end
endmodule
