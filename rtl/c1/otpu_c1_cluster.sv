// Independent PC engines and bounded global in-order retirement, one command at a time.
// Canonical row assignment is absolute row % ENGINES, including n0 and partial tails.
// A reset aborts the command; the next command MUST carry a new generation tag.
module otpu_c1_cluster #(
  parameter int ENGINES = 4,
  parameter int SLOTS = 4,
  parameter int SW = (SLOTS > 1) ? $clog2(SLOTS) : 1
) (
  input logic clk, rst,
  input logic act_we,
  input logic [8:0] act_addr,
  input logic [255:0] act_data,
  input logic [7:0] act_da,
  input logic [31:0] act_sa,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [31:0] cmd_n0, cmd_n, cmd_generation,
  input logic [7:0] cmd_kb, cmd_k0, cmd_ab,
  output logic [ENGINES-1:0] req_valid,
  input logic [ENGINES-1:0] req_ready,
  output logic [ENGINES-1:0][SW-1:0] req_slot,
  output logic [ENGINES-1:0][31:0] req_row, req_generation,
  output logic [ENGINES-1:0][7:0] req_kb, req_k0,
  input logic [ENGINES-1:0] rsp_valid,
  output logic [ENGINES-1:0] rsp_ready,
  input logic [ENGINES-1:0][SW-1:0] rsp_slot,
  input logic [ENGINES-1:0][31:0] rsp_row, rsp_generation,
  input logic [ENGINES-1:0][9:0] rsp_group,
  input logic [ENGINES-1:0][255:0] rsp_weights,
  input logic [ENGINES-1:0][7:0] rsp_dw,
  input logic [ENGINES-1:0][31:0] rsp_sw,
  output logic [ENGINES-1:0] rsp_drop,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] out_row, out_generation, out_sw, out_sa,
  output logic signed [47:0] out_acc,
  output logic [ENGINES-1:0][SW:0] occupancy,
  output logic busy
);
  logic [31:0] next_row[ENGINES], end_row, retire_row, generation;
  logic [7:0] kb, k0, ab;
  logic [ENGINES-1:0] job_ready, engine_valid, engine_ready;
  logic [ENGINES-1:0][31:0] engine_row, engine_gen, engine_sw, engine_sa;
  logic [ENGINES-1:0][47:0] engine_acc;
  assign cmd_ready = !rst && !busy;
  wire [31:0] selected = retire_row % ENGINES;
  assign out_valid = !rst && busy && engine_valid[selected] && engine_row[selected] == retire_row;
  assign out_row = engine_row[selected];
  assign out_generation = engine_gen[selected];
  assign out_acc = engine_acc[selected];
  assign out_sw = engine_sw[selected];
  assign out_sa = engine_sa[selected];
  always_comb begin
    engine_ready = '0;
    if (out_valid) engine_ready[selected] = out_ready;
  end
  for (genvar e=0; e<ENGINES; e++) begin : engines
    otpu_c1_row_engine #(.SLOTS(SLOTS)) engine (
      .clk, .rst, .act_we, .act_addr, .act_data, .act_da, .act_sa,
      .job_valid(busy && next_row[e] < end_row), .job_ready(job_ready[e]),
      .job_row(next_row[e]), .job_generation(generation), .job_kb(kb), .job_k0(k0), .job_ab(ab),
      .req_valid(req_valid[e]), .req_ready(req_ready[e]), .req_slot(req_slot[e]),
      .req_row(req_row[e]), .req_generation(req_generation[e]), .req_kb(req_kb[e]), .req_k0(req_k0[e]),
      .rsp_valid(rsp_valid[e]), .rsp_ready(rsp_ready[e]), .rsp_slot(rsp_slot[e]),
      .rsp_row(rsp_row[e]), .rsp_generation(rsp_generation[e]), .rsp_group(rsp_group[e]),
      .rsp_weights(rsp_weights[e]), .rsp_dw(rsp_dw[e]), .rsp_sw(rsp_sw[e]), .rsp_drop(rsp_drop[e]),
      .out_valid(engine_valid[e]), .out_ready(engine_ready[e]), .out_row(engine_row[e]),
      .out_generation(engine_gen[e]), .out_acc(engine_acc[e]), .out_sw(engine_sw[e]),
      .out_sa(engine_sa[e]), .occupancy(occupancy[e])
    );
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      busy <= 0; retire_row <= 0; end_row <= 0;
      for (int e=0;e<ENGINES;e++) next_row[e] <= 0;
    end else begin
      if (cmd_valid && cmd_ready) begin
        busy <= cmd_n != 0;
        end_row <= cmd_n0 + cmd_n; retire_row <= cmd_n0; generation <= cmd_generation;
        kb <= cmd_kb; k0 <= cmd_k0; ab <= cmd_ab;
        for (int e=0;e<ENGINES;e++)
          next_row[e] <= cmd_n0 + (32'(e) + ENGINES - (cmd_n0 % ENGINES)) % ENGINES;
      end
      for (int e=0;e<ENGINES;e++)
        if (busy && next_row[e] < end_row && job_ready[e]) next_row[e] <= next_row[e] + ENGINES;
      if (out_valid && out_ready) begin
        retire_row <= retire_row + 1;
        if (retire_row + 1 == end_row) busy <= 0;
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (act_we && busy) $fatal(1,"C1 cluster ACT write during command");
    if (cmd_valid && cmd_ready && (cmd_n == 0 || cmd_n0 + cmd_n < cmd_n0))
      $fatal(1,"C1 invalid row range");
    if (out_valid && out_generation != generation) $fatal(1,"C1 stale result escaped");
  end
`endif
endmodule
