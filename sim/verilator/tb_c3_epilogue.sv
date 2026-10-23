`timescale 1ns/1ps
module tb_c3_epilogue;
  import otpu_fp::*;
  logic clk = 0, rst = 1, in_valid = 0, in_ready, out_valid, out_ready = 0;
  logic [47:0] in_acc;
  logic [31:0] in_sw, in_sa, in_id, out_bits, out_id;
  otpu_c3_epilogue dut (.*);
  logic [47:0] av [8192];
  logic [31:0] sw [8192], sa [8192], expected [8192];
  int fd, scan, count, sent = 0, received = 0, cycle = 0;
  logic held = 0;
  logic [31:0] held_bits, held_id;
  int unsigned random_state = 17;
  string path;
  function automatic int unsigned random_word();
    random_state ^= random_state << 13; random_state ^= random_state >> 17;
    random_state ^= random_state << 5; return random_state;
  endfunction
  task automatic tick;
    bit take_in, take_out;
    #5;
    take_in = in_valid && in_ready; take_out = out_valid && out_ready;
    if (!rst) begin
      if (held && (!out_valid || out_bits != held_bits || out_id != held_id))
        $fatal(1, "epilogue output changed during stall");
      held = out_valid && !out_ready; held_bits = out_bits; held_id = out_id;
      if (take_out) begin
        if (out_id != received || out_bits !== expected[received])
          $fatal(1, "epilogue case %0d got id %0d bits %08h expected %08h", received, out_id, out_bits, expected[received]);
      end
    end else held = 0;
    clk = 1; #5; clk = 0;
    if (take_in) begin sent++; in_valid = 0; end
    if (take_out) received++;
    cycle++;
  endtask
  initial begin
    if (!$value$plusargs("input=%s", path)) $fatal(1, "need input");
    fd = $fopen(path, "r");
    scan = $fscanf(fd, "%d\n", count);
    if (count < 1 || count > 8192) $fatal(1, "invalid count");
    for (int i = 0; i < count; i++) begin
      scan = $fscanf(fd, "%h %h %h %h\n", av[i], sw[i], sa[i], expected[i]);
      if (scan != 4) $fatal(1, "short vectors");
    end
    $fclose(fd);
    repeat (2) tick(); rst = 0;
    while (received < count) begin
      if (cycle > 100000) $fatal(1, "epilogue timeout");
      out_ready = (random_word()%100 >= 45);
      if (!in_valid && sent < count && random_word()%100 >= 20) begin
        in_valid = 1; in_acc = av[sent]; in_sw = sw[sent]; in_sa = sa[sent]; in_id = 32'(sent);
      end
      tick();
    end
    $display("C3_EPILOGUE_PASS cases=%0d", count); $finish;
  end
endmodule
