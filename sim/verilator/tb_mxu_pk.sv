// Drives one packed MXU column (rtl/mxu/otpu_mxu_pk_col.sv, docs/mxu_packing.md) from a
// stimulus file, one line a cycle, eight hex fields:
//   v first last act wa wb bma bmb      (1, 1, 1, D*8, D*8, D*8, 8*17, 8*17 bits)
// (the lines are the cycles: a line with v 0 is a cycle without a block), and prints
// "ROW <n> <acc_a> <acc_b>" (hex) at every cycle av is set, n counting them, then
// "PKTEST cycles=<c> rows=<n>". tests/test_mxu_pk.py builds it (opentpu.rtlsim.build, the
// column's NCH / PACK / WREG through Verilator's -G) and checks the rows against numpy.
module tb_mxu_pk #(parameter int NCH = 8, parameter bit PACK = 1'b1, parameter bit WREG = 1'b0);
  localparam int D = 128, CLS = 16;
  logic clk;
  logic v, first, last;
  logic [D-1:0][7:0]  act, wa, wb;
  logic [7:0][16:0]   bma, bmb;
  logic [47:0]        acc_a, acc_b;
  logic               av;
  otpu_mxu_pk_col #(.D(D), .CLS(CLS), .NCH(NCH), .PACK(PACK), .WREG(WREG)) u_col (
    .clk, .v, .first, .last, .act, .wa, .wb, .bma, .bmb, .acc_a, .acc_b, .av);
  int fd, n, cycles, rows, drain;
  string fname;
  initial clk = 1'b0;
  always #5 clk = ~clk;
  // the rows, as the column's registers hold them during the cycle (sampled at its end)
  always @(posedge clk)
    if (av) begin
      $display("ROW %0d %012h %012h", rows, acc_a, acc_b);
      rows++;
    end
  initial begin
    if (!$value$plusargs("vec=%s", fname)) $fatal(1, "need +vec=");
    fd = $fopen(fname, "r");
    if (fd == 0) $fatal(1, "cannot open %s", fname);
    cycles = 0; rows = 0; drain = 0;
    v = 1'b0; first = 1'b0; last = 1'b0; act = '0; wa = '0; wb = '0; bma = '0; bmb = '0;
    // each line is driven from the falling edge and taken at the rising one; after the file,
    // CLS + 8 cycles without a block let the last row's accumulators land
    while (drain < CLS + 8) begin
      @(negedge clk);
      n = (drain == 0 && !$feof(fd)) ?
          $fscanf(fd, "%h %h %h %h %h %h %h %h\n", v, first, last, act, wa, wb, bma, bmb) : 0;
      if (n != 8) begin
        v = 1'b0; first = 1'b0; last = 1'b0;
        drain++;
      end
      @(posedge clk);
      cycles++;
    end
    @(negedge clk);
    $display("PKTEST cycles=%0d rows=%0d", cycles, rows);
    $finish;
  end
endmodule
