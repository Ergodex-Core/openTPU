// Behavioural per-slice DRAM for simulation (the board version sits behind a DDR3/HBM
// controller). Port A: one read of the aligned 32 bytes holding word a_addr (the MXU's scale
// lines) or one byte-enabled word write per cycle; port SW: one
// byte-enabled word write per cycle (the quantizer's QST stores). Port B: one
// D-byte burst per cycle, read (the MXU's streamed operand, DMA loads) or word-masked write
// (DMA stores). Reads return LAT cycles later, in order; data is sampled at the request.
// Lane ports (NL > 1: the MXU's chunk lanes 1 .. NL-1, otpu_mxu_lanes; [0] unused): per lane
// one chunk read (lb_*) and one scale-line read (la_*) a cycle, as ports B and A, LAT cycles.
module otpu_dram #(
  parameter int WORDS = 1 << 18,
  parameter int D     = 32,
  parameter int LAT   = 8,
  parameter int SID   = 0,
  parameter int NL    = 1
) (
  input  logic              clk,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [31:0]       a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [255:0]      a_rdata,    // words (a_addr & ~7) .. (a_addr | 7), the lowest first
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
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              rd_idle,    // no read in flight
  // the chunk lanes' reads ([0] unused)
  input  logic [NL-1:0]          lb_req,
  input  logic [NL-1:0][31:0]    lb_addr,    // word address of the chunk
  output logic [NL-1:0]          lb_rdy,
  output logic [NL-1:0]          lb_rvalid,
  output logic [NL-1:0][D*8-1:0] lb_rdata,
  input  logic [NL-1:0]          la_req,
  input  logic [NL-1:0][31:0]    la_addr,    // word address
  output logic [NL-1:0]          la_rdy,
  output logic [NL-1:0]          la_rvalid,
  output logic [NL-1:0][255:0]   la_rdata,
  input  logic              dump
);
  localparam int AW = $clog2(WORDS);
  localparam int CW = D / 4;

  logic [31:0] mem [WORDS];

  logic [LAT-1:0]    av, bv, bt;
  logic [255:0]      ad [LAT];
  logic [D*8-1:0]    bd [LAT];
  logic [NL-1:0][LAT-1:0] lbv, lav;
  logic [D*8-1:0]    lbd [NL][LAT];
  logic [255:0]      lad [NL][LAT];
  assign lb_rdy = '1;
  assign la_rdy = '1;

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
    for (int w = 0; w < 8; w++) ad[0][32*w +: 32] <= mem[{a_addr[AW-1:3], 3'(w)}];
    bv[0] <= b_req && !b_we;
    bt[0] <= b_tag;
    for (int w = 0; w < CW; w++) bd[0][32*w +: 32] <= mem[AW'(b_addr + w)];
    if (b_req && b_we)
      for (int w = 0; w < CW; w++) if (b_wmask[w]) mem[AW'(b_addr + w)] <= b_wdata[32*w +: 32];
    for (int k = 1; k < LAT; k++) begin
      av[k] <= av[k-1];
      ad[k] <= ad[k-1];
      bv[k] <= bv[k-1];
      bt[k] <= bt[k-1];
      bd[k] <= bd[k-1];
    end
    // the lanes' reads: sampled at the request, back LAT cycles later
    for (int l = 1; l < NL; l++) begin
      lbv[l][0] <= lb_req[l];
      for (int w = 0; w < CW; w++) lbd[l][0][32*w +: 32] <= mem[AW'(lb_addr[l] + w)];
      lav[l][0] <= la_req[l];
      for (int w = 0; w < 8; w++) lad[l][0][32*w +: 32] <= mem[{la_addr[l][AW-1:3], 3'(w)}];
      for (int k = 1; k < LAT; k++) begin
        lbv[l][k] <= lbv[l][k-1]; lbd[l][k] <= lbd[l][k-1];
        lav[l][k] <= lav[l][k-1]; lad[l][k] <= lad[l][k-1];
      end
    end
  end
  assign a_rvalid = av[LAT-1];
  assign a_rdata  = ad[LAT-1];
  assign b_rvalid = bv[LAT-1];
  assign b_rtag   = bt[LAT-1];
  assign b_rdata  = bd[LAT-1];
  always_comb begin
    lb_rvalid = '0; la_rvalid = '0; lb_rdata = '0; la_rdata = '0;
    for (int l = 1; l < NL; l++) begin
      lb_rvalid[l] = lbv[l][LAT-1]; lb_rdata[l] = lbd[l][LAT-1];
      la_rvalid[l] = lav[l][LAT-1]; la_rdata[l] = lad[l][LAT-1];
    end
  end
  assign rd_idle  = !(|av) && !(|bv) && !(|lbv) && !(|lav);

`ifndef SYNTHESIS
  // Images are raw binary: dram_<SID>.bin holds big-endian words ($fread order) and the dump
  // dram_out_<SID>.bin is little-endian (the byte order of the DRAM itself).
  string dir;
  integer fd, nread;
  initial begin
    av = '0;
    bv = '0;
    bt = '0;
    lbv = '0;
    lav = '0;
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
      if (fd != 0) begin
        nread = $fread(mem, fd);
        $fclose(fd);
      end
      fd = $fopen($sformatf("%s/poke_%0d.txt", dir, SID), "r");
      if (fd != 0) begin
        while (pk_n < NPOKE && $fscanf(fd, "%h %h %h", pk_c[pk_n], pk_a[pk_n], pk_v[pk_n]) == 3)
          pk_n++;
        $fclose(fd);
      end
    end
  end
  always @(posedge clk) if (dump) begin
    fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
    $fclose(fd);
  end
`endif
endmodule
