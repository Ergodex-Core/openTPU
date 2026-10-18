// C1 experimental PC-local int8 engine. This is a synthesizable abstract row/beat
// interface, NOT an AXI/HBM controller. Each issued row reserves one of SLOTS entries
// until its ordered result handshake. Responses can interleave rows but must advance
// group indices in order within a row. Tags include command generation and full row ID.
// After reset the producer must choose a NEW command generation; old responses drain
// with rsp_drop and can never mutate a new generation. Reset discards outstanding work.
// ACT writes are permitted only while idle; ab is a block offset, groups are D/4=32 bytes.
module otpu_c1_row_engine #(
  parameter int SLOTS = 4,
  parameter int SW = (SLOTS > 1) ? $clog2(SLOTS) : 1,
  parameter int ACT_GROUPS = 512
) (
  input logic clk, rst,
  input logic act_we,
  input logic [$clog2(ACT_GROUPS)-1:0] act_addr,
  input logic [255:0] act_data,
  input logic [7:0] act_da,
  input logic [31:0] act_sa,
  input logic job_valid,
  output logic job_ready,
  input logic [31:0] job_row, job_generation,
  input logic [7:0] job_kb, job_k0, job_ab,
  output logic req_valid,
  input logic req_ready,
  output logic [SW-1:0] req_slot,
  output logic [31:0] req_row, req_generation,
  output logic [7:0] req_kb, req_k0,
  input logic rsp_valid,
  output logic rsp_ready,
  input logic [SW-1:0] rsp_slot,
  input logic [31:0] rsp_row, rsp_generation,
  input logic [9:0] rsp_group,
  input logic [255:0] rsp_weights,
  input logic [7:0] rsp_dw,
  input logic [31:0] rsp_sw,
  output logic rsp_drop,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] out_row, out_generation, out_sw, out_sa,
  output logic signed [47:0] out_acc,
  output logic [SW:0] occupancy
);
  initial begin
    if (SLOTS < 1 || (SLOTS & (SLOTS-1)) != 0) $fatal(1, "SLOTS must be a power of two");
  end
  logic [255:0] activations [ACT_GROUPS];
  logic [7:0] da [ACT_GROUPS];
  logic [31:0] sa [ACT_GROUPS];
  always_ff @(posedge clk) if (act_we && !rst) begin
    activations[act_addr] <= act_data;
    da[act_addr] <= act_da;
    sa[act_addr] <= act_sa;
  end
  logic [SLOTS-1:0] active, issued, complete;
  logic [31:0] rows[SLOTS], generations[SLOTS], scales_w[SLOTS], scales_a[SLOTS];
  logic [7:0] blocks[SLOTS], offsets_k[SLOTS], offsets_a[SLOTS];
  logic [9:0] received[SLOTS];
  logic signed [47:0] accumulators[SLOTS];
  logic [SW-1:0] head, tail;
  function automatic logic [SW-1:0] advance(input logic [SW-1:0] x);
    return SW'((int'(x) + 1) & (SLOTS-1));
  endfunction
  assign job_ready = !rst && occupancy < (SW+1)'(SLOTS);
  assign out_valid = !rst && occupancy != 0 && complete[head];
  assign out_row = rows[head];
  assign out_generation = generations[head];
  assign out_acc = accumulators[head];
  assign out_sw = scales_w[head];
  assign out_sa = scales_a[head];
  always_comb begin
    req_valid = 0;
    req_slot = head;
    for (int j = 0; j < SLOTS; j++) begin
      if (!req_valid && active[(int'(head)+j)&(SLOTS-1)] &&
          !issued[(int'(head)+j)&(SLOTS-1)]) begin
        req_valid = !rst;
        req_slot = SW'((int'(head)+j)&(SLOTS-1));
      end
    end
  end
  assign req_row = rows[req_slot];
  assign req_generation = generations[req_slot];
  assign req_kb = blocks[req_slot];
  assign req_k0 = offsets_k[req_slot];

  // Invalid/stale responses are consumed and reported, never allowed to block a PC.
  wire match_rsp = int'(rsp_slot) < SLOTS && active[rsp_slot] && issued[rsp_slot] &&
                   !complete[rsp_slot] && rows[rsp_slot] == rsp_row &&
                   generations[rsp_slot] == rsp_generation &&
                   received[rsp_slot] == rsp_group &&
                   received[rsp_slot] < {blocks[rsp_slot], 2'b00};
  wire [$clog2(ACT_GROUPS):0] aidx = ($clog2(ACT_GROUPS)+1)'(
                                      int'(offsets_a[rsp_slot])*4 + int'(rsp_group));
  assign rsp_ready = !rst;
  wire accept = rsp_valid && rsp_ready && match_rsp && int'(aidx) < ACT_GROUPS;
  assign rsp_drop = rsp_valid && rsp_ready && !accept;

  // Registered multiplier plus balanced reduction tree, one 32-element group/cycle.
  // The seven valid/tag stages stop no global clock: idle or interleaved rows insert bubbles.
  logic [6:0] valid_pipe, last_pipe;
  logic [SW-1:0] slot_pipe[7];
  logic signed [10:0] shift_pipe[6];
  logic signed [20:0] sums[6][32];
  logic signed [47:0] scaled;
  always_ff @(posedge clk) begin
    valid_pipe[0] <= accept;
    slot_pipe[0] <= rsp_slot;
    last_pipe[0] <= received[rsp_slot] + 10'd1 == {blocks[rsp_slot], 2'b00};
    shift_pipe[0] <= 11'sd16 - $signed({3'b000,rsp_dw}) - $signed({3'b000,da[aidx[$clog2(ACT_GROUPS)-1:0]]});
    for (int i = 0; i < 32; i++)
      sums[0][i] <= 21'($signed(rsp_weights[i*8+:8]) * $signed(activations[aidx[$clog2(ACT_GROUPS)-1:0]][i*8+:8]));
    for (int stage = 1; stage < 6; stage++) begin
      valid_pipe[stage] <= valid_pipe[stage-1];
      slot_pipe[stage] <= slot_pipe[stage-1];
      last_pipe[stage] <= last_pipe[stage-1];
      shift_pipe[stage] <= shift_pipe[stage-1];
      for (int i = 0; i < (32 >> stage); i++)
        sums[stage][i] <= sums[stage-1][2*i] + sums[stage-1][2*i+1];
    end
    valid_pipe[6] <= valid_pipe[5];
    slot_pipe[6] <= slot_pipe[5];
    last_pipe[6] <= last_pipe[5];
    scaled <= shift_pipe[5] < 0 ? 48'sd0 :
              $signed({{27{sums[5][0][20]}}, sums[5][0]}) <<< shift_pipe[5];
    if (rst) valid_pipe <= '0;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      active <= '0; issued <= '0; complete <= '0;
      head <= '0; tail <= '0; occupancy <= '0;
    end else begin
      case ({job_valid && job_ready, out_valid && out_ready})
        2'b10: occupancy <= occupancy + 1'b1;
        2'b01: occupancy <= occupancy - 1'b1;
        default: ;
      endcase
      if (job_valid && job_ready) begin
        active[tail] <= 1; issued[tail] <= 0; complete[tail] <= 0;
        rows[tail] <= job_row; generations[tail] <= job_generation;
        blocks[tail] <= job_kb; offsets_k[tail] <= job_k0; offsets_a[tail] <= job_ab;
        received[tail] <= 0; accumulators[tail] <= 0;
        tail <= advance(tail);
      end
      if (req_valid && req_ready) issued[req_slot] <= 1;
      if (accept) begin
        received[rsp_slot] <= received[rsp_slot] + 1'b1;
        if (rsp_group == 0) begin
          scales_w[rsp_slot] <= rsp_sw;
          scales_a[rsp_slot] <= sa[aidx[$clog2(ACT_GROUPS)-1:0]];
        end
      end
      if (valid_pipe[6]) begin
        accumulators[slot_pipe[6]] <= accumulators[slot_pipe[6]] + scaled;
        if (last_pipe[6]) complete[slot_pipe[6]] <= 1;
      end
      if (out_valid && out_ready) begin
        active[head] <= 0; issued[head] <= 0; complete[head] <= 0;
        head <= advance(head);
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (occupancy > (SW+1)'(SLOTS)) $fatal(1,"C1 row credits overflow");
    if (act_we && occupancy != 0) $fatal(1,"C1 ACT overwritten while rows are live");
    if (job_valid && job_ready && (job_kb == 0 ||
        (int'(job_ab)+int'(job_kb))*4 > ACT_GROUPS)) $fatal(1,"C1 invalid ACT row range");
  end
`endif
endmodule
