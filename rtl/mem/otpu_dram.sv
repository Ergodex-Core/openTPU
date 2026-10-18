// Behavioural per-slice DRAM for simulation (the board version sits behind a DDR3/HBM
// controller). Port A: one read of the aligned ALINE bytes holding word a_addr (the MXU's scale
// lines) or one byte-enabled word write per cycle; port SW: one
// byte-enabled word write per cycle (the quantizer's QST stores). Port B: one
// D-byte burst per cycle, read (the MXU's streamed operand, DMA loads) or word-masked write
// (DMA stores). Reads return LAT cycles later, in order; data is sampled at the request.
// PORTB > 1 (docs/wide_port.md): a read takes b_n + 1 consecutive chunks (at most PORTB) and
// returns them in one response (b_rcm its chunks, b_rlast always: never split).
module otpu_dram #(
  parameter int WORDS = 1 << 18,
  parameter int D     = 32,
  parameter int LAT   = 8,
  parameter int SID   = 0,
  parameter int PORTB = 1,
  parameter int ALINE = 32        // port A's line, bytes (32, 64)
) (
  input  logic              clk,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [31:0]       a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [8*ALINE-1:0] a_rdata,   // the ALINE/4 words of the aligned line, the lowest first
  input  logic              sw_req,
  input  logic [31:0]       sw_addr,
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  input  logic              b_req,
  input  logic              b_tag,      // requester tag, returned with the read data
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,    // word enables for a burst write
  input  logic [D*8-1:0]    b_wdata,
  input  logic [31:0]       b_addr,     // word address of the chunk
  input  logic [1:0]        b_n,        // PORTB > 1: a read's chunks - 1
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [PORTB*D*8-1:0] b_rdata, // chunk c at [c*D*8 +: D*8] (past b_n: zero)
  output logic [PORTB-1:0]  b_rcm,
  output logic              b_rlast,
  output logic              rd_idle,    // no read in flight
  input  logic              dump
);
  localparam int AW = $clog2(WORDS);
  localparam int CW = D / 4;
  localparam int LWW = $clog2(ALINE / 4);     // the word index in a port-A line

`ifdef OTPU_ARCSIM
  // ArcSim (docs/arcsim.md): the image is the harness's (sim/arcsim/otpu_arcsim.cpp), behind
  // DPI-C, as otpu_native_mem's -- circt-verilog's $fread takes memories of up to 1024 elements,
  // and the array in the model's state would be a value of the image's size to its compiler
  import "DPI-C" function int otpu_mem_rd(input int sid, input longint i);
  import "DPI-C" function void otpu_mem_wr(input int sid, input longint i, input int v);
  import "DPI-C" function void otpu_mem_wrm(input int sid, input longint i, input int mask,
                                            input int v);
  import "DPI-C" function void otpu_mem_init(input int sid, input longint words);
  import "DPI-C" function void otpu_mem_dump(input int sid);
  function automatic logic [31:0] rdw(input logic [AW-1:0] i);
    return otpu_mem_rd(SID, longint'(i));
  endfunction
`else
  logic [31:0] mem [WORDS];
`endif

  logic [LAT-1:0]    av, bv, bt;
  logic [8*ALINE-1:0] ad [LAT];
  logic [PORTB*D*8-1:0] bd [LAT];
  logic [PORTB-1:0]  bm [LAT];

`ifndef SYNTHESIS
  // the host's writes during the run (WAITW's tests): <dir>/poke_<SID>.txt, lines "cycle word
  // value" (hex, in cycle order), each word written as the cycle count reaches its cycle
  localparam int NPOKE = 64;
  longint      pk_c [NPOKE];
  logic [31:0] pk_a [NPOKE], pk_v [NPOKE];
  int          pk_n = 0, pk_i = 0;
  longint      pk_cyc = 0;
`endif
  always_ff @(posedge clk) begin
`ifdef OTPU_ARCSIM
    // the reads first: they sample the words as they were before this cycle's writes, as the
    // array's non-blocking writes let them below; the harness's writes land at once, in the
    // same order (the last to a word wins, as the last non-blocking write does)
    av[0] <= a_req && !a_we;
    for (int w = 0; w < ALINE / 4; w++) ad[0][32*w +: 32] <= rdw({a_addr[AW-1:LWW], LWW'(w)});
    bv[0] <= b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < PORTB * CW; w++)
      bd[0][32*w +: 32] <= (PORTB == 1 || w / CW <= int'(b_n)) ? rdw(AW'(b_addr + w)) : '0;
    for (int c = 0; c < PORTB; c++) bm[0][c] <= PORTB == 1 || c <= int'(b_n);
    pk_cyc <= pk_cyc + 1;
    if (pk_i < pk_n && pk_cyc >= pk_c[pk_i]) begin
      otpu_mem_wr(SID, longint'(pk_a[pk_i][AW-1:0]), int'(pk_v[pk_i]));
      pk_i <= pk_i + 1;
    end
    if (a_req && a_we) otpu_mem_wrm(SID, longint'(a_addr[AW-1:0]), int'(a_be), int'(a_wdata));
    if (sw_req) otpu_mem_wrm(SID, longint'(sw_addr[AW-1:0]), int'(sw_be), int'(sw_wdata));
    if (b_req && b_we)
      for (int w = 0; w < CW; w++)
        if (b_wmask[w]) otpu_mem_wr(SID, longint'(AW'(b_addr + w)), int'(b_wdata[32*w +: 32]));
`else
`ifndef SYNTHESIS
    pk_cyc <= pk_cyc + 1;
    if (pk_i < pk_n && pk_cyc >= pk_c[pk_i]) begin
      mem[pk_a[pk_i][AW-1:0]] <= pk_v[pk_i];
      pk_i <= pk_i + 1;
    end
`endif
    if (a_req && a_we) begin
      for (int b = 0; b < 4; b++)
        if (a_be[b]) mem[a_addr[AW-1:0]][8*b +: 8] <= a_wdata[8*b +: 8];
    end
    if (sw_req) begin
      for (int b = 0; b < 4; b++)
        if (sw_be[b]) mem[sw_addr[AW-1:0]][8*b +: 8] <= sw_wdata[8*b +: 8];
    end
    av[0] <= a_req && !a_we;
    for (int w = 0; w < ALINE / 4; w++) ad[0][32*w +: 32] <= mem[{a_addr[AW-1:LWW], LWW'(w)}];
    bv[0] <= b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < PORTB * CW; w++)
      bd[0][32*w +: 32] <= (PORTB == 1 || w / CW <= int'(b_n)) ? mem[AW'(b_addr + w)] : '0;
    for (int c = 0; c < PORTB; c++) bm[0][c] <= PORTB == 1 || c <= int'(b_n);
    if (b_req && b_we)
      for (int w = 0; w < CW; w++) if (b_wmask[w]) mem[AW'(b_addr + w)] <= b_wdata[32*w +: 32];
`endif
    for (int k = 1; k < LAT; k++) begin
      av[k] <= av[k-1];
      ad[k] <= ad[k-1];
      bv[k] <= bv[k-1];
      bt[k] <= bt[k-1];
      bd[k] <= bd[k-1];
      bm[k] <= bm[k-1];
    end
  end
  assign a_rvalid = av[LAT-1];
  assign a_rdata  = ad[LAT-1];
  assign b_rvalid = bv[LAT-1];
  assign b_rtag   = bt[LAT-1];
  assign b_rdata  = bd[LAT-1];
  assign b_rcm    = bm[LAT-1];
  assign b_rlast  = 1'b1;
  assign rd_idle  = !(|av) && !(|bv);

`ifndef SYNTHESIS
  // Images are raw binary: dram_<SID>.bin holds big-endian words ($fread order) and the dump
  // dram_out_<SID>.bin is little-endian (the byte order of the DRAM itself).
  string dir;
  integer fd, nread;
  initial begin
    av = '0;
    bv = '0;
    bt = '0;
`ifdef OTPU_ARCSIM
    otpu_mem_init(SID, longint'(WORDS));        // (zeros, then the harness's dram_<SID>.bin)
`else
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
`endif
    if ($value$plusargs("dir=%s", dir)) begin
`ifndef OTPU_ARCSIM
      fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
      if (fd != 0) begin
        nread = $fread(mem, fd);
        $fclose(fd);
      end
`endif
      fd = $fopen($sformatf("%s/poke_%0d.txt", dir, SID), "r");
      if (fd != 0) begin
        while (pk_n < NPOKE && $fscanf(fd, "%h %h %h", pk_c[pk_n], pk_a[pk_n], pk_v[pk_n]) == 3)
          pk_n++;
        $fclose(fd);
      end
    end
  end
  always @(posedge clk) if (dump) begin
`ifdef OTPU_ARCSIM
    otpu_mem_dump(SID);
`else
    fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
    $fclose(fd);
`endif
  end
`endif
endmodule
