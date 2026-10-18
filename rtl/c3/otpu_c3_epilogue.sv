// C3 isolated head epilogue. The integer conversion is the existing otpu_mx_lane
// datapath; both multiplies use otpu_fmul. Original Sa is scaled by 2^-16 with
// exactly the ISA's fp32/FTZ semantics, including zero, infinity and NaN.
// Eight granted stages, one row/cycle throughput. Backpressure freezes all data
// and tags. Reset discards in-flight valids without resetting datapath flops.
module otpu_c3_epilogue
  import otpu_fp::*;
(
  input logic clk, rst,
  input logic in_valid,
  output logic in_ready,
  input logic [47:0] in_acc,
  input f32_t in_sw, in_sa,
  input logic [31:0] in_id,
  output logic out_valid,
  input logic out_ready,
  output f32_t out_bits,
  output logic [31:0] out_id
);
  localparam int LAT = 8;
  wire en = !out_valid || out_ready;
  assign in_ready = en && !rst;
  logic [LAT-1:0] valid;
  logic [31:0] ids [LAT];
  assign out_valid = valid[LAT-1] && !rst;
  assign out_id = ids[LAT-1];
  always_ff @(posedge clk) begin
    if (rst) valid <= '0;
    else if (en) begin
      valid[0] <= in_valid;
      for (int i = 1; i < LAT; i++) valid[i] <= valid[i-1];
    end
    if (en) begin
      ids[0] <= in_id;
      for (int i = 1; i < LAT; i++) ids[i] <= ids[i-1];
    end
  end
  function automatic logic [5:0] lzc48(input logic [47:0] m);
    if (m[47:16] != 0) return lzc32(m[47:16]);
    return 6'd32 + 6'(lzc32({m[15:0], 16'd0}));
  endfunction
  logic s1, z1, s2, z2, s3, z3;
  logic [47:0] g1, g2, g3;
  logic [5:0] lz2, lz3;
  f32_t fv;
  always_ff @(posedge clk) if (en) begin
    s1 <= in_acc[47]; z1 <= (in_acc == 0);
    g1 <= in_acc[47] ? (~in_acc + 48'd1) : in_acc;
    s2 <= s1; z2 <= z1; g2 <= g1; lz2 <= lzc48(g1);
    s3 <= s2; z3 <= z2; lz3 <= lz2; g3 <= g2 << lz2;
    begin
      logic [23:0] mm;
      logic [24:0] mr;
      logic [8:0] e;
      mm = g3[47:24];
      mr = {1'b0, mm} + ((g3[23] && ((|g3[22:0]) || mm[0])) ? 25'd1 : 25'd0);
      e = 9'd174 - 9'(lz3);
      if (mr[24]) e = e + 9'd1;
      fv <= z3 ? F_ZERO : {s3, e[7:0], mr[22:0]};
    end
  end
  f32_t swd, sap, sad, y1;
  otpu_delay #(.W(32), .N(4)) u_sw (.clk, .en, .d(in_sw), .q(swd));
  otpu_fmul u_scale (.clk, .en, .a(in_sa), .b(32'h3780_0000), .y(sap));
  otpu_delay #(.W(32), .N(4)) u_sa (.clk, .en, .d(sap), .q(sad));
  otpu_fmul u_m1 (.clk, .en, .a(fv), .b(swd), .y(y1));
  otpu_fmul u_m2 (.clk, .en, .a(y1), .b(sad), .y(out_bits));
endmodule
