// Synthetic, independently stalled row service around REAL C3 RTL. This is not
// an HBM/AXI controller. A header costs one service cycle (32 bytes), then each
// 128-byte weight beat is held until accepted. Registered arithmetic executes.
`timescale 1ns/1ps
module tb_c3;
  import otpu_fp::*;
  parameter int ENGINES = 4;
  localparam int MAX_ROWS = 8192, MAX_KB = 28;
  logic clk = 0, rst = 1, act_we = 0, cmd_valid = 0, cmd_ready;
  logic [5:0] act_index, cmd_kb;
  logic [1023:0] act_data;
  logic [19:0] act_offsets;
  logic [31:0] cmd_sa;
  logic [ENGINES-1:0][31:0] cmd_rows, cmd_base;
  logic [ENGINES-1:0] in_valid = 0, in_ready, logit_valid, logit_ready = 0;
  logic [ENGINES-1:0][1023:0] in_weights;
  logic [ENGINES-1:0][4:0] in_dw;
  logic [ENGINES-1:0][31:0] in_sw, logit_bits, logit_id;
  logic result_valid, result_ready = 0;
  logic [31:0] result_bits, result_id;
  logic [ENGINES-1:0] event_dot, event_epilogue, event_done;
  logic [ENGINES-1:0][31:0] event_dot_id, event_epilogue_id;
  otpu_c3_cluster #(.ENGINES(ENGINES), .MAX_KB(MAX_KB)) dut (.*);

  logic [1023:0] a_mem [MAX_KB], w_mem [MAX_ROWS][MAX_KB];
  logic [19:0] da_mem [MAX_KB];
  logic [4:0] dw_mem [MAX_ROWS][MAX_KB];
  logic [31:0] sw_mem [MAX_ROWS];
  int vocab, kb, fd, log_fd, scan, cycle = 0, epoch = 0;
  int stalls = 0, output_stalls = 0, latency = 0, reset_after = -1, repeats = 1, slow_engine = -1, result_hold = 9, output_hold = 0;
  int pos [ENGINES], available [ENGINES];
  bit header [ENGINES];
  int unsigned random_state = 1;
  logic [ENGINES-1:0] stalled_logit = 0;
  logic [ENGINES-1:0][31:0] held_bits, held_id;
  bit stalled_result = 0;
  logic [31:0] held_result_bits, held_result_id;
  bit final_taken;
  string input_path, log_path;

  function automatic int unsigned random_word();
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
  endfunction
  task automatic record(input string kind, input int e, input int id, input int block_no,
                        input logic [31:0] bits);
    $fdisplay(log_fd, "%0d %0d %s %0d %0d %0d %08h", cycle, epoch, kind, e, id, block_no, bits);
  endtask
  task automatic tick;
    logic [ENGINES-1:0] accepted;
    #5;
    accepted = in_valid & in_ready;
    final_taken = !rst && result_valid && result_ready;
    if (!rst) begin
      if (act_we) record("A", -1, 0, int'(act_index), 0);
      if (cmd_valid && cmd_ready) record("C", -1, 0, 0, 0);
      for (int e = 0; e < ENGINES; e++) begin
        if (stalled_logit[e] && (!logit_valid[e] || logit_bits[e] != held_bits[e] || logit_id[e] != held_id[e]))
          $fatal(1, "unstable blocked diagnostic logit");
        stalled_logit[e] = logit_valid[e] && !logit_ready[e];
        held_bits[e] = logit_bits[e]; held_id[e] = logit_id[e];
        if (accepted[e]) record("W", e, int'(cmd_base[e]) + pos[e]/kb, pos[e]%kb, 0);
        if (in_valid[e] && !in_ready[e]) record("B", e, int'(cmd_base[e]) + pos[e]/kb, pos[e]%kb, 0);
        if (event_dot[e]) record("D", e, int'(event_dot_id[e]), 0, 0);
        if (event_epilogue[e]) record("E", e, int'(event_epilogue_id[e]), 0, 0);
        if (logit_valid[e] && !logit_ready[e]) record("O", e, int'(logit_id[e]), 0, logit_bits[e]);
        if (logit_valid[e] && logit_ready[e]) record("L", e, int'(logit_id[e]), 0, logit_bits[e]);
        if (event_done[e]) record("S", e, int'(dut.done_id[e]), 0, dut.done_bits[e]);
      end
      if (stalled_result && (!result_valid || result_bits != held_result_bits || result_id != held_result_id))
        $fatal(1, "unstable blocked global result");
      stalled_result = result_valid && !result_ready;
      held_result_bits = result_bits; held_result_id = result_id;
      if (result_valid) record(final_taken ? "R" : "F", -1, int'(result_id), 0, result_bits);
    end else begin stalled_logit = 0; stalled_result = 0; end
    clk = 1;
    #5;
    clk = 0;
    if (!rst) for (int e = 0; e < ENGINES; e++) if (accepted[e]) begin
      pos[e]++;
      in_valid[e] = 0;
      if (pos[e]%kb == 0) begin
        header[e] = 0;
        available[e] = cycle + 1 + ((e == slow_engine) ? 17 : 0);
      end
    end
    cycle++;
  endtask
  task automatic reset_and_flush;
    rst = 1; act_we = 0; cmd_valid = 0; in_valid = 0; logit_ready = '1; result_ready = 1;
    repeat (2) tick();
    rst = 0;
    repeat (24) begin
      if ((|logit_valid) || result_valid || (|event_done)) $fatal(1, "stale result after reset");
      tick();
    end
  endtask
  task automatic launch;
    int shard;
    shard = (vocab + ENGINES - 1)/ENGINES;
    for (int e = 0; e < ENGINES; e++) begin
      cmd_base[e] = 32'(e*shard);
      cmd_rows[e] = 32'((vocab > e*shard) ? ((vocab-e*shard < shard) ? vocab-e*shard : shard) : 0);
      pos[e] = 0; header[e] = 0;
    end
    in_valid = 0; result_ready = 0;
    for (int k = 0; k < kb; k++) begin
      act_we = 1; act_index = 6'(k); act_data = a_mem[k]; act_offsets = da_mem[k]; tick();
    end
    act_we = 0; cmd_valid = 1;
    if (!cmd_ready) $fatal(1, "command unexpectedly blocked");
    tick(); cmd_valid = 0;
    for (int e = 0; e < ENGINES; e++) available[e] = cycle + latency + 3*e;
  endtask
  initial begin
    if (!$value$plusargs("input=%s", input_path) || !$value$plusargs("events=%s", log_path))
      $fatal(1, "need input and events paths");
    scan = $value$plusargs("seed=%d", random_state);
    scan = $value$plusargs("stalls=%d", stalls);
    scan = $value$plusargs("output_stalls=%d", output_stalls);
    scan = $value$plusargs("latency=%d", latency);
    scan = $value$plusargs("reset_after=%d", reset_after);
    scan = $value$plusargs("repeats=%d", repeats);
    scan = $value$plusargs("slow_engine=%d", slow_engine);
    scan = $value$plusargs("result_hold=%d", result_hold);
    scan = $value$plusargs("output_hold=%d", output_hold);
    if (random_state == 0) random_state = 1;
    fd = $fopen(input_path, "r"); log_fd = $fopen(log_path, "w");
    if (!fd || !log_fd) $fatal(1, "cannot open files");
    scan = $fscanf(fd, "%d %d %h\n", vocab, kb, cmd_sa);
    if (scan != 3 || vocab < 1 || vocab > MAX_ROWS || kb < 1 || kb > MAX_KB)
      $fatal(1, "invalid vectors");
    cmd_kb = 6'(kb);
    for (int k = 0; k < kb; k++) begin
      scan = $fscanf(fd, "%h %h\n", a_mem[k], da_mem[k]);
      if (scan != 2) $fatal(1, "short activation");
    end
    for (int n = 0; n < vocab; n++) begin
      scan = $fscanf(fd, "%h\n", sw_mem[n]);
      if (scan != 1) $fatal(1, "short scale");
      for (int k = 0; k < kb; k++) begin
        scan = $fscanf(fd, "%h %h\n", w_mem[n][k], dw_mem[n][k]);
        if (scan != 2) $fatal(1, "short weight row");
      end
    end
    $fclose(fd);
    reset_and_flush();
    for (int rep = 0; rep < repeats; rep++) begin
      int started, result_seen;
      bit reset_used;
      epoch++; launch(); started = cycle; result_seen = -1; reset_used = 0;
      while (!final_taken) begin
        if (cycle > 2000000) $fatal(1, "C3 timeout");
        if (!reset_used && reset_after >= 0 && cycle-started >= reset_after) begin
          record("X", -1, 0, 0, 0);
          reset_and_flush(); epoch++; launch(); started = cycle; result_seen = -1; reset_used = 1;
        end
        for (int e = 0; e < ENGINES; e++) begin
          logit_ready[e] = (cycle-started >= output_hold) && (random_word()%100 >= output_stalls);
          if (!in_valid[e] && pos[e] < int'(cmd_rows[e])*kb && cycle >= available[e]) begin
            if (random_word()%100 >= stalls) begin
              int n, k;
              n = int'(cmd_base[e]) + pos[e]/kb; k = pos[e]%kb;
              if (!header[e]) begin header[e] = 1; record("H", e, n, 0, sw_mem[n]); end
              else begin
                in_valid[e] = 1; in_weights[e] = w_mem[n][k]; in_dw[e] = dw_mem[n][k]; in_sw[e] = sw_mem[n];
              end
            end
          end
        end
        if (result_valid && result_seen < 0) result_seen = cycle;
        result_ready = result_seen >= 0 && cycle-result_seen >= result_hold;
        tick();
      end
      // Completion must be unique; start again without reset to expose stale tags.
      in_valid = 0; logit_ready = '1; result_ready = 1;
      repeat (12) begin tick(); if (result_valid || (|logit_valid)) $fatal(1, "extra completion"); end
    end
    $fclose(log_fd);
    $display("C3_RTL_PASS engines=%0d rows=%0d KB=%0d epochs=%0d cycles=%0d", ENGINES, vocab, kb, epoch, cycle);
    $finish;
  end
endmodule
