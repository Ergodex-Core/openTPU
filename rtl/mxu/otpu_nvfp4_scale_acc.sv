// Folded, single-context numerical backend. Every accepted K16 block executes:
// subtotal conversion -> FP32 block multiply -> ordered FP32 add; last then
// performs a separate FP32 global multiply. No fused rounding boundary.
// mode 0/1: externally supplied STRICT tree subtotal, gradual / DAZ-FTZ.
// mode 2: F24 integer subtotal, gradual; explicitly approximate vs strict tree.
// mode 2 rejects discarded alignment until an error policy is separately approved.
// Fault bits: [0] input/protocol, [1] convert, [2] scale, [3] add, [4] global.
// A fault poisons the row until last; output is canonical +0 and invalid for use.
module otpu_nvfp4_scale_acc (
  input logic clk, rst,
  input logic in_valid,
  output logic in_ready,
  input logic [1:0] in_mode,
  input logic signed [40:0] in_dot,
  input logic signed [8:0] in_e_star,
  input logic [31:0] in_strict_dot,
  input logic [7:0] in_scale,
  input logic [31:0] in_global,
  input logic in_first, in_last, in_invalid, in_discarded,
  input logic [31:0] in_k16_index, in_tag,
  input logic [31:0] in_group, in_epoch,
  input logic in_scale_valid,
  input logic [31:0] in_scale_group, in_scale_k16_index, in_scale_epoch,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] out_bits, out_tag,
  output logic [1:0] out_mode,
  output logic [4:0] out_fault,
  output logic [31:0] out_group, out_epoch,
  output logic buffer_free, output_committed,
  output logic [31:0] free_group, free_k16_index, free_epoch
);
  import otpu_nvfp4_fp::*;
  typedef enum logic [2:0] {WAIT_INPUT,CONVERT,SCALE,ACCUMULATE,GLOBAL,OUTPUT} state_t;
  state_t state;
  logic active, last;
  logic [1:0] mode;
  logic [31:0] tag, group_id, epoch, global_factor, next_index;
  logic signed [40:0] dot;
  logic signed [8:0] e_star;
  logic [31:0] strict_dot, subtotal, scaled, accumulator;
  logic [7:0] scale;
  logic [4:0] faults;
  // The complete tagged packet is snapshotted only when its matching scale is
  // available. buffer_free releases that input packet, not a whole DMA allocation.
  assign in_ready = !rst && state == WAIT_INPUT && in_scale_valid;
  assign out_valid = !rst && state == OUTPUT;
  assign buffer_free = in_valid && in_ready;
  assign free_group = in_group;
  assign free_k16_index = in_k16_index;
  assign free_epoch = in_epoch;
  // Local result acceptance after ALL rounding stages. No DMA visibility claim.
  assign output_committed = out_valid && out_ready && out_fault == 0;

  always_ff @(posedge clk) begin : backend
    result_t r,s;
    logic protocol_bad;
    if (rst) begin
      state <= WAIT_INPUT;
      active <= 0;
      faults <= 0;
      accumulator <= 0;
      out_fault <= 0;
      out_bits <= 0;
      out_tag <= 0;
      out_mode <= 0;
      out_group <= 0;
      out_epoch <= 0;
    end else case (state)
      WAIT_INPUT: if (in_valid && in_scale_valid) begin
        protocol_bad = in_invalid || in_mode == 3 || in_scale > 8'h7e ||
                       in_global[31] || in_global[30:23] == 255 ||
                       (in_mode == 2 && in_discarded) ||
                       in_scale_group != in_group || in_scale_epoch != in_epoch ||
                       in_scale_k16_index != in_k16_index ||
                       (in_k16_index == 32'hffffffff && !in_last);
        if (!active) begin
          mode <= in_mode; tag <= in_tag; global_factor <= in_global;
          group_id <= in_group; epoch <= in_epoch;
          accumulator <= 0;
          protocol_bad = protocol_bad || !in_first || in_k16_index != 0;
          faults <= {4'd0,protocol_bad};
        end else begin
          protocol_bad = protocol_bad || in_first || in_mode != mode || in_tag != tag ||
                         in_global != global_factor || in_k16_index != next_index;
          protocol_bad = protocol_bad || in_group != group_id || in_epoch != epoch;
          faults <= faults | {4'd0,protocol_bad};
        end
        active <= 1;
        next_index <= in_k16_index+32'd1;
        dot <= in_dot; e_star <= in_e_star; strict_dot <= in_strict_dot;
        scale <= in_scale; last <= in_last;
        state <= CONVERT;
      end
      CONVERT: begin
        r = '0;
        if (mode == 2) r = from_dot(dot,e_star);
        else if (strict_dot[30:23] == 255) r.fault = 1;
        else r.bits = strict_dot;
        subtotal <= r.bits;
        faults <= faults | {3'd0,r.fault,1'b0};
        state <= SCALE;
      end
      SCALE: begin
        s = scale32(scale);
        r = mul32(subtotal,s.bits,mode == 1);
        scaled <= r.bits;
        faults <= faults | {2'd0,(r.fault || s.fault),2'd0};
        state <= ACCUMULATE;
      end
      ACCUMULATE: begin
        r = add32(accumulator,scaled,mode == 1);
        accumulator <= r.bits;
        faults <= faults | {1'b0,r.fault,3'd0};
        if (last) state <= GLOBAL;
        else state <= WAIT_INPUT;
      end
      GLOBAL: begin
        r = mul32(accumulator,global_factor,mode == 1);
        out_fault <= faults | {r.fault,4'd0};
        out_bits <= (|faults || r.fault) ? 32'd0 : r.bits;
        out_tag <= tag; out_mode <= mode;
        out_group <= group_id; out_epoch <= epoch;
        state <= OUTPUT;
      end
      OUTPUT: if (out_ready) begin
        active <= 0;
        state <= WAIT_INPUT;
      end
      default: begin state <= WAIT_INPUT; active <= 0; end
    endcase
  end
endmodule
