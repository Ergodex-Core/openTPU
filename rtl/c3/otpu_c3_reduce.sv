// Independent shard completions may arrive in any order or simultaneously.
// Each expected source is accepted once; result remains stable until consumed.
// fp32 total order is the ISA's fkey(ftz), ties use lowest GLOBAL token ID.
module otpu_c3_reduce
  import otpu_fp::*;
#(parameter int ENGINES = 4) (
  input logic clk, rst,
  input logic start,
  input logic [ENGINES-1:0] expected,
  input logic [ENGINES-1:0] in_valid,
  output logic [ENGINES-1:0] in_ready,
  input logic [ENGINES-1:0][31:0] in_bits, in_id,
  output logic out_valid,
  input logic out_ready,
  output f32_t out_bits,
  output logic [31:0] out_id,
  output logic active
);
  logic [ENGINES-1:0] mask, seen, next_seen;
  logic have, next_have;
  logic [31:0] next_bits, next_id;
  assign in_ready = {ENGINES{active && !rst}} & mask & ~seen;
  always_comb begin
    next_seen = seen; next_have = have; next_bits = out_bits; next_id = out_id;
    for (int e = 0; e < ENGINES; e++) if (in_valid[e] && in_ready[e]) begin
      next_seen[e] = 1;
      if (!next_have || fkey(in_bits[e]) > fkey(next_bits) ||
          (fkey(in_bits[e]) == fkey(next_bits) && in_id[e] < next_id)) begin
        next_bits = ftz(in_bits[e]); next_id = in_id[e];
      end
      next_have = 1;
    end
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      active <= 0; out_valid <= 0; have <= 0; seen <= 0; mask <= 0;
      out_bits <= 0; out_id <= 0;
    end else begin
      if (out_valid && out_ready) out_valid <= 0;
      if (start) begin
        active <= 1; seen <= 0; mask <= expected; have <= 0;
      end else if (active) begin
        seen <= next_seen; have <= next_have; out_bits <= next_bits; out_id <= next_id;
        if (next_seen == mask) begin active <= 0; out_valid <= 1; end
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && start && (active || out_valid || expected == 0))
    $fatal(1, "C3 reducer start while busy or empty");
`endif
endmodule
