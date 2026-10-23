// Proposal v1 F24 subtotal ONLY: dot = out_dot * 2^(out_e_star - 25).
// BF16 gradual decode, E2M1 weights, RNE alignment; 41-bit exact integer sum.
// Scale is coassociated metadata, not applied here. No FP32 or MM integration.
// Seven registered stages, II=1 when unstalled. A full output freezes ALL stages.
// Nonfinite active BF16 or signed/NaN scale: invalid=1, dot/e/discarded=0.
// Reset discards in-flight transactions. Upstream owns tag generations/lifetimes.
module otpu_nvfp4_dot16 #(parameter integer TAG_W = 32) (
  input  logic clk,
  input  logic rst,
  input  logic in_valid,
  output logic in_ready,
  input  logic [255:0] in_a,       // lane i: [16*i +: 16] BF16 bits
  input  logic [63:0] in_w,        // lane i: [4*i +: 4], even K low nibble
  input  logic [15:0] in_mask,     // missing tail terms: exact +0, no validation
  input  logic [7:0] in_scale,     // unsigned finite E4M3FN: 00..7e
  input  logic [TAG_W-1:0] in_tag,
  output logic out_valid,
  input  logic out_ready,
  output logic signed [40:0] out_dot,
  output logic signed [8:0] out_e_star,
  output logic out_discarded,     // any nonzero alignment bits discarded
  output logic out_invalid,
  output logic [7:0] out_scale,
  output logic [TAG_W-1:0] out_tag
);
  logic advance;
  logic [6:0] valid;
  logic signed [8:0] exps [0:6];
  logic [6:0] invalid, discarded;
  logic [7:0] scales [0:6];
  logic [TAG_W-1:0] tags [0:6];
  logic [7:0] mant_in [0:15], mant [0:15];
  logic signed [8:0] exp_in [0:15], exp_lane [0:15];
  logic sign_a [0:15], sign_in [0:15];
  logic [3:0] codes0 [0:15], codes1 [0:15];
  logic signed [8:0] top_in;
  logic bad_in;
  logic signed [34:0] aligned [0:15], triple [0:15];
  logic signed [36:0] products [0:15];
  logic signed [37:0] sum2 [0:7];
  logic signed [38:0] sum4 [0:3];
  logic signed [39:0] sum8 [0:1];
  logic signed [40:0] sum16;

  assign advance = !valid[6] || out_ready;
  assign in_ready = !rst && advance;
  assign out_valid = !rst && valid[6];
  assign out_invalid = invalid[6];
  assign out_dot = invalid[6] ? 41'sd0 : sum16;
  assign out_e_star = invalid[6] ? 9'sd0 : exps[6];
  assign out_discarded = !invalid[6] && discarded[6];
  assign out_scale = scales[6];
  assign out_tag = tags[6];

  always_comb begin
    top_in = -9'sd133;
    bad_in = in_scale > 8'h7e;
    for (integer i = 0; i < 16; i = i + 1) begin
      sign_in[i] = in_a[16*i+15];
      mant_in[i] = '0;
      exp_in[i] = -9'sd133;
      if (in_mask[i]) begin
        if (in_a[16*i+7 +: 8] == 8'hff) bad_in = 1'b1;
        else if (in_a[16*i+7 +: 8] == 0)
          mant_in[i] = {1'b0, in_a[16*i +: 7]};
        else begin
          mant_in[i] = {1'b1, in_a[16*i +: 7]};
          exp_in[i] = $signed({1'b0, in_a[16*i+7 +: 8]}) - 9'sd134;
        end
      end
      if (mant_in[i] != 0 && exp_in[i] > top_in) top_in = exp_in[i];
    end
  end

  // |mant * 2^(24-gap)| rounded ties to even. gap > 32 always rounds to zero.
  // The mask detects discarded bits even when rounding leaves the integer unchanged.
  function automatic logic [32:0] align_mant(input logic [7:0] m,
                                             input logic signed [8:0] e,
                                             input logic signed [8:0] top);
    integer gap, shift;
    logic [31:0] q, rem_bits, half_value, mask_bits;
    logic lost, increment;
    gap = int'(top) - int'(e);
    q = 0; rem_bits = 0; half_value = 0; mask_bits = 0;
    lost = 0; increment = 0; shift = gap - 24;
    if (gap <= 24) q = {24'd0, m} << (24-gap);
    else if (shift > 8) lost = |m;
    else begin
      mask_bits = (32'd1 << shift) - 1;
      rem_bits = {24'd0, m} & mask_bits;
      half_value = 32'd1 << (shift-1);
      q = {24'd0, m} >> shift;
      lost = |rem_bits;
      increment = (rem_bits > half_value) || ((rem_bits == half_value) && q[0]);
      q = q + {31'd0, increment};
    end
    return {lost, q};
  endfunction

  function automatic logic signed [36:0] product(input logic signed [34:0] t,
                                                input logic signed [34:0] t3,
                                                input logic [3:0] c);
    logic signed [36:0] u, v;
    u = {{2{t[34]}}, t};
    v = {{2{t3[34]}}, t3};
    case (c[2:0])
      0: u = 0;
      1: u = u;
      2: u = u <<< 1;
      3: u = v;
      4: u = u <<< 2;
      5: u = v <<< 1;
      6: u = u <<< 3;
      7: u = v <<< 2;
      default: u = 0;
    endcase
    return c[3] ? -u : u;
  endfunction

  always_ff @(posedge clk) begin : pipeline
    logic [32:0] a;
    logic signed [34:0] t;
    logic lost;
    if (rst) begin
      valid <= '0;
      // Data registers intentionally have no reset; only valid transactions are observable.
    end else if (advance) begin
      valid <= {valid[5:0], in_valid};
      exps[0] <= top_in;
      invalid[0] <= bad_in;
      discarded[0] <= 1'b0;
      scales[0] <= in_scale;
      tags[0] <= in_tag;
      for (integer s = 1; s < 7; s = s + 1) begin
        exps[s] <= exps[s-1];
        invalid[s] <= invalid[s-1];
        discarded[s] <= discarded[s-1];
        scales[s] <= scales[s-1];
        tags[s] <= tags[s-1];
      end
      lost = 0;
      for (integer i = 0; i < 16; i = i + 1) begin
        mant[i] <= mant_in[i];
        exp_lane[i] <= exp_in[i];
        sign_a[i] <= sign_in[i];
        codes0[i] <= in_w[4*i +: 4];
        codes1[i] <= codes0[i];
        a = align_mant(mant[i], exp_lane[i], exps[0]);
        t = $signed({3'd0, a[31:0]});
        if (sign_a[i]) t = -t;
        aligned[i] <= t;
        triple[i] <= t + (t <<< 1);
        lost = lost | a[32];
        products[i] <= product(aligned[i], triple[i], codes1[i]);
      end
      discarded[1] <= lost;
      for (integer i = 0; i < 8; i = i + 1)
        sum2[i] <= {products[2*i][36], products[2*i]} +
                   {products[2*i+1][36], products[2*i+1]};
      for (integer i = 0; i < 4; i = i + 1)
        sum4[i] <= {sum2[2*i][37], sum2[2*i]} + {sum2[2*i+1][37], sum2[2*i+1]};
      for (integer i = 0; i < 2; i = i + 1)
        sum8[i] <= {sum4[2*i][38], sum4[2*i]} + {sum4[2*i+1][38], sum4[2*i+1]};
      sum16 <= {sum8[0][39], sum8[0]} + {sum8[1][39], sum8[1]};
    end
  end
endmodule
