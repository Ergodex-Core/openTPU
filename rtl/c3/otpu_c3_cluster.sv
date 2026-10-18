// Standalone C3 engine cluster. No board ISA, AXI, shell, QACT or sampling path
// is implied. Zero-row shards are masked; only greedy selection is implemented.
module otpu_c3_cluster
  import otpu_fp::*;
#(parameter int ENGINES = 4, MAX_KB = 28) (
  input logic clk, rst,
  input logic act_we,
  input logic [5:0] act_index,
  input logic [1023:0] act_data,
  input logic [19:0] act_offsets,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [ENGINES-1:0][31:0] cmd_rows, cmd_base,
  input logic [5:0] cmd_kb,
  input f32_t cmd_sa,
  input logic [ENGINES-1:0] in_valid,
  output logic [ENGINES-1:0] in_ready,
  input logic [ENGINES-1:0][1023:0] in_weights,
  input logic [ENGINES-1:0][4:0] in_dw,
  input logic [ENGINES-1:0][31:0] in_sw,
  output logic [ENGINES-1:0] logit_valid,
  input logic [ENGINES-1:0] logit_ready,
  output logic [ENGINES-1:0][31:0] logit_bits, logit_id,
  output logic result_valid,
  input logic result_ready,
  output f32_t result_bits,
  output logic [31:0] result_id,
  output logic [ENGINES-1:0] event_dot, event_epilogue, event_done,
  output logic [ENGINES-1:0][31:0] event_dot_id, event_epilogue_id
);
  logic [ENGINES-1:0] ready, done_valid, done_ready, expected;
  logic [ENGINES-1:0][31:0] done_bits, done_id;
  logic active;
  assign cmd_ready = (&ready) && !active && !result_valid && !rst;
  wire start = cmd_valid && cmd_ready;
  assign event_done = done_valid & done_ready;
  for (genvar e = 0; e < ENGINES; e++) begin : g_engine
    assign expected[e] = cmd_rows[e] != 0;
    otpu_c3_engine #(.MAX_KB(MAX_KB)) u_engine (.clk, .rst,
      .act_we, .act_index, .act_data, .act_offsets,
      .cmd_valid(start && expected[e]), .cmd_ready(ready[e]),
      .cmd_rows(cmd_rows[e]), .cmd_base(cmd_base[e]), .cmd_kb, .cmd_sa,
      .in_valid(in_valid[e]), .in_ready(in_ready[e]), .in_weights(in_weights[e]),
      .in_dw(in_dw[e]), .in_sw(in_sw[e]), .logit_valid(logit_valid[e]),
      .logit_ready(logit_ready[e]), .logit_bits(logit_bits[e]), .logit_id(logit_id[e]),
      .done_valid(done_valid[e]), .done_ready(done_ready[e]),
      .done_bits(done_bits[e]), .done_id(done_id[e]),
      .event_dot(event_dot[e]), .event_epilogue(event_epilogue[e]),
      .event_dot_id(event_dot_id[e]), .event_epilogue_id(event_epilogue_id[e]));
  end
  otpu_c3_reduce #(.ENGINES(ENGINES)) u_reduce (.clk, .rst, .start, .expected,
    .in_valid(done_valid), .in_ready(done_ready), .in_bits(done_bits), .in_id(done_id),
    .out_valid(result_valid), .out_ready(result_ready), .out_bits(result_bits),
    .out_id(result_id), .active);
endmodule
