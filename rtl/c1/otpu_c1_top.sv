// Standalone plain-output GEMV kernel; no TMEM/ACC or AXI integration.
// New commands wait for the final fp32 output; reset flushes both pipeline levels.
module otpu_c1_top #(
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
  output logic [31:0] out_row, out_generation, out_bits,
  output logic int_valid, int_ready,
  output logic [31:0] int_row, int_generation, int_sw, int_sa,
  output logic signed [47:0] int_acc,
  output logic [ENGINES-1:0][SW:0] occupancy,
  output logic busy
);
  logic cluster_ready, cluster_busy;
  logic [31:0] generation;
  logic [3:0] in_flight;
  assign cmd_ready = cluster_ready && in_flight == 0;
  assign busy = cluster_busy || in_flight != 0;
  assign out_generation = generation;
  otpu_c1_cluster #(.ENGINES(ENGINES), .SLOTS(SLOTS)) cluster (
    .clk, .rst, .act_we, .act_addr, .act_data, .act_da, .act_sa,
    .cmd_valid(cmd_valid && in_flight == 0), .cmd_ready(cluster_ready),
    .cmd_n0, .cmd_n, .cmd_generation, .cmd_kb, .cmd_k0, .cmd_ab,
    .req_valid, .req_ready, .req_slot, .req_row, .req_generation, .req_kb, .req_k0,
    .rsp_valid, .rsp_ready, .rsp_slot, .rsp_row, .rsp_generation, .rsp_group,
    .rsp_weights, .rsp_dw, .rsp_sw, .rsp_drop,
    .out_valid(int_valid), .out_ready(int_ready), .out_row(int_row),
    .out_generation(int_generation), .out_sw(int_sw), .out_sa(int_sa), .out_acc(int_acc),
    .occupancy, .busy(cluster_busy)
  );
  otpu_c1_epilogue epilogue (
    .clk, .rst, .in_valid(int_valid), .in_ready(int_ready),
    .in_acc(int_acc), .in_sw(int_sw), .in_sa(int_sa), .in_id(int_row),
    .out_valid, .out_ready, .out_bits, .out_id(out_row)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      in_flight <= 0; generation <= 0;
    end else begin
      if (cmd_valid && cmd_ready) generation <= cmd_generation;
      case ({int_valid && int_ready, out_valid && out_ready})
        2'b10: in_flight <= in_flight + 1'b1;
        2'b01: in_flight <= in_flight - 1'b1;
        default: ;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (in_flight > 8) $fatal(1,"C1 epilogue count overflow");
    if (act_we && busy) $fatal(1,"C1 ACT overwrite before fp32 retirement");
  end
`endif
endmodule
