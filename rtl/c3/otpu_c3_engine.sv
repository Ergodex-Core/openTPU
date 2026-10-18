// C3 streaming vocabulary engine, intentionally outside the board file list.
// Input contract: cmd_rows contiguous rows starting at cmd_base; cmd_kb blocks
// per row, in order. One block is 128 signed int8 weights and its original dw.
// Sw is sampled on the first block. Activation snapshots must be loaded while
// idle before command acceptance. A future sidecar/HBM adapter owns requests,
// headers, burst boundaries, response tags and reassembly; this is its consumer.
// Product and balanced sum stages accept one block/cycle. Epilogues/reduction
// overlap later rows. Whole-pipeline backpressure is lossless, including KB=1.
module otpu_c3_engine
  import otpu_fp::*;
#(parameter int MAX_KB = 28) (
  input logic clk, rst,
  input logic act_we,
  input logic [5:0] act_index,
  input logic [1023:0] act_data,
  input logic [19:0] act_offsets,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [31:0] cmd_rows, cmd_base,
  input logic [5:0] cmd_kb,
  input f32_t cmd_sa,
  input logic in_valid,
  output logic in_ready,
  input logic [1023:0] in_weights,
  input logic [4:0] in_dw,
  input f32_t in_sw,
  output logic logit_valid,
  input logic logit_ready,
  output f32_t logit_bits,
  output logic [31:0] logit_id,
  output logic done_valid,
  input logic done_ready,
  output f32_t done_bits,
  output logic [31:0] done_id,
  output logic event_dot, event_epilogue,
  output logic [31:0] event_dot_id, event_epilogue_id
);
  logic [1023:0] acts [MAX_KB];
  logic [19:0] offsets [MAX_KB];
  logic running;
  logic [31:0] rows, base, sent, retired;
  logic [5:0] kb, block_index;
  f32_t sa, row_sw, best_bits;
  logic [31:0] best_id;
  logic have_best;
  wire step;
  assign cmd_ready = !running && !done_valid && !rst;
  assign in_ready = running && (sent < rows) && step && !rst;
  wire take = in_valid && in_ready;
  wire first = (block_index == 0);
  wire last = (block_index + 6'd1 == kb);
  always_ff @(posedge clk) begin
    if (act_we && cmd_ready && act_index < MAX_KB) begin
      acts[act_index] <= act_data;
      offsets[act_index] <= act_offsets;
    end
    if (rst) begin
      running <= 0; done_valid <= 0; have_best <= 0;
      rows <= 0; base <= 0; sent <= 0; retired <= 0; block_index <= 0;
      best_bits <= 0; best_id <= 0; done_bits <= 0; done_id <= 0;
    end else begin
      if (done_valid && done_ready) done_valid <= 0;
      if (cmd_valid && cmd_ready) begin
        rows <= cmd_rows; base <= cmd_base; kb <= cmd_kb; sa <= cmd_sa;
        running <= 1; sent <= 0; retired <= 0; block_index <= 0; have_best <= 0;
      end
      if (take) begin
        if (first) row_sw <= in_sw;
        block_index <= last ? 6'd0 : block_index + 6'd1;
        if (last) sent <= sent + 32'd1;
      end
      if (logit_valid && logit_ready) begin
        logic choose;
        choose = !have_best || (fkey(logit_bits) > fkey(best_bits)) ||
                 ((fkey(logit_bits) == fkey(best_bits)) && logit_id < best_id);
        if (choose) begin best_bits <= ftz(logit_bits); best_id <= logit_id; end
        have_best <= 1; retired <= retired + 32'd1;
        if (retired + 32'd1 == rows) begin
          done_bits <= choose ? ftz(logit_bits) : best_bits;
          done_id <= choose ? logit_id : best_id;
          done_valid <= 1; running <= 0;
        end
      end
    end
  end

  typedef struct packed {
    logic v, first, last;
    logic [31:0] id, sw;
    logic [4:0] dw;
    logic [19:0] da;
  } meta_t;
  meta_t meta [9];
  logic signed [15:0] prod [128];
  logic signed [16:0] sum1 [64];
  logic signed [17:0] sum2 [32];
  logic signed [18:0] sum3 [16];
  logic signed [19:0] sum4 [8];
  logic signed [20:0] sum5 [4];
  logic signed [47:0] shifted [4], pair_sum [2], chunk_sum, acc, ep_acc;
  logic ep_valid;
  logic [31:0] ep_id, ep_sw;
  assign event_dot = step && meta[8].v && !rst;
  assign event_dot_id = meta[8].id;
  assign event_epilogue = ep_valid && step && !rst;
  assign event_epilogue_id = ep_id;
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < 9; i++) meta[i] <= '0;
      ep_valid <= 0; acc <= 0;
    end else if (step) begin
      meta[0] <= '{v:take, first:first, last:last, id:base + sent,
                   sw:first ? in_sw : row_sw, dw:in_dw, da:offsets[block_index]};
      for (int i = 1; i < 9; i++) meta[i] <= meta[i-1];
      ep_valid <= meta[8].v && meta[8].last;
      if (meta[8].v) begin
        acc <= (meta[8].first ? 48'sd0 : acc) + chunk_sum;
        ep_acc <= (meta[8].first ? 48'sd0 : acc) + chunk_sum;
        ep_id <= meta[8].id; ep_sw <= meta[8].sw;
      end
    end
    if (step) begin
      for (int i = 0; i < 128; i++)
        prod[i] <= $signed(in_weights[8*i +: 8]) * $signed(acts[block_index][8*i +: 8]);
      for (int i = 0; i < 64; i++) sum1[i] <= 17'(prod[2*i]) + 17'(prod[2*i+1]);
      for (int i = 0; i < 32; i++) sum2[i] <= 18'(sum1[2*i]) + 18'(sum1[2*i+1]);
      for (int i = 0; i < 16; i++) sum3[i] <= 19'(sum2[2*i]) + 19'(sum2[2*i+1]);
      for (int i = 0; i < 8; i++) sum4[i] <= 20'(sum3[2*i]) + 20'(sum3[2*i+1]);
      for (int i = 0; i < 4; i++) sum5[i] <= 21'(sum4[2*i]) + 21'(sum4[2*i+1]);
      for (int i = 0; i < 4; i++) begin
        logic [5:0] ds;
        ds = 6'(meta[5].dw) + 6'(meta[5].da[5*i +: 5]);
        shifted[i] <= ds > 16 ? 48'sd0 : (48'(sum5[i]) <<< (6'd16 - ds));
      end
      for (int i = 0; i < 2; i++) pair_sum[i] <= shifted[2*i] + shifted[2*i+1];
      chunk_sum <= pair_sum[0] + pair_sum[1];
    end
  end
  otpu_c3_epilogue u_ep (.clk, .rst, .in_valid(ep_valid), .in_ready(step),
    .in_acc(ep_acc), .in_sw(ep_sw), .in_sa(sa), .in_id(ep_id),
    .out_valid(logit_valid), .out_ready(logit_ready), .out_bits(logit_bits), .out_id(logit_id));
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (cmd_valid && cmd_ready && (cmd_rows == 0 || cmd_kb == 0 || cmd_kb > MAX_KB))
      $fatal(1, "C3 invalid command shape");
    if (act_we && !cmd_ready) $fatal(1, "C3 activation write during command");
    if (act_we && act_index >= MAX_KB) $fatal(1, "C3 activation index out of bounds");
    if (logit_valid && logit_ready && (!running || retired >= rows))
      $fatal(1, "C3 unexpected result");
  end
`endif
endmodule
