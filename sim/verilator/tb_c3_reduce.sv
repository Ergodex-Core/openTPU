`timescale 1ns/1ps
module tb_c3_reduce;
  import otpu_fp::*;
  logic clk = 0, rst = 1, start = 0, out_valid, out_ready = 0, active;
  logic [3:0] expected, in_valid = 0, in_ready;
  logic [3:0][31:0] in_bits, in_id;
  logic [31:0] out_bits, out_id;
  otpu_c3_reduce dut (.*);
  int fd, scan, cases, delay_cycles [4], cycle, count = 0;
  logic [31:0] want_bits, want_id;
  logic [3:0] submitted;
  string path;
  task automatic tick;
    #5; submitted |= in_valid & in_ready;
    clk = 1; #5; clk = 0;
  endtask
  initial begin
    if (!$value$plusargs("input=%s", path)) $fatal(1, "need input");
    fd = $fopen(path, "r"); scan = $fscanf(fd, "%d\n", cases);
    repeat (2) tick(); rst = 0;
    repeat (cases) begin
      scan = $fscanf(fd, "%h %h %h\n", expected, want_bits, want_id);
      if (scan != 3) $fatal(1, "short expected");
      for (int e = 0; e < 4; e++) begin
        scan = $fscanf(fd, "%h %h %d\n", in_bits[e], in_id[e], delay_cycles[e]);
        if (scan != 3) $fatal(1, "short shard");
      end
      submitted = 0; in_valid = 0; start = 1; out_ready = 0; tick(); start = 0; cycle = 0;
      while (!out_valid) begin
        for (int e = 0; e < 4; e++)
          in_valid[e] = expected[e] && !submitted[e] && cycle >= delay_cycles[e];
        tick(); cycle++;
        if (cycle > 1000) $fatal(1, "reduce timeout");
      end
      if (out_bits !== want_bits || out_id != want_id || submitted != expected)
        $fatal(1, "reducer case %0d got %08h/%0d expected %08h/%0d", count, out_bits, out_id, want_bits, want_id);
      in_valid = 0;
      repeat (7) begin
        tick();
        if (!out_valid || out_bits !== want_bits || out_id != want_id) $fatal(1, "result did not hold");
      end
      out_ready = 1; tick();
      if (out_valid || active) $fatal(1, "result did not complete");
      count++;
    end
    $fclose(fd); $display("C3_REDUCE_PASS cases=%0d", count); $finish;
  end
endmodule
