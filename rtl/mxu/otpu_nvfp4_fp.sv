// Portable conformance-first FP32 arithmetic for the standalone W4A16 slice.
// RNE at every call, explicit gradual or DAZ/FTZ; invalid/overflow => fault, bits=0.
// Exact wide integer addition prioritizes cancellation correctness over area/timing.
// This is not qualified F2 floating IP and has no frequency/resource claim.
package otpu_nvfp4_fp;
  typedef struct packed {
    logic fault;
    logic [31:0] bits;
  } result_t;

  function automatic logic [31:0] operand(input logic [31:0] a, input logic ftz);
    return (ftz && a[30:23] == 0) ? {a[31],31'd0} : a;
  endfunction

  // The exact input is (-1)^sign * magnitude * 2^power. 280 bits cover the
  // largest exact finite-FP32 sum in units of 2^-149 (including its carry).
  function automatic result_t round_binary(input logic [279:0] magnitude,
                                            input integer power,
                                            input logic sign,
                                            input logic ftz);
    result_t r;
    logic [279:0] q, remainder_bits, half_value, mask_bits;
    integer top, e, shift;
    logic increment;
    r = '0;
    r.bits[31] = sign;
    top = -1;
    for (integer i=0; i<280; i=i+1)
      if (magnitude[i]) top = i;
    if (top < 0) return r;
    e = top + power;
    shift = (e < -126) ? -149-power : top-23;
    q = '0; remainder_bits = '0; half_value = '0; mask_bits = '0;
    increment = 0;
    if (shift <= 0) q = magnitude << (-shift);
    else if (shift <= 280) begin
      q = magnitude >> shift;
      mask_bits = ~('1 << shift);
      remainder_bits = magnitude & mask_bits;
      half_value = 280'd1 << (shift-1);
      increment = remainder_bits > half_value || (remainder_bits == half_value && q[0]);
      q = q + {{279{1'b0}},increment};
    end
    if (e < -126) begin
      // q==2^23 is the subnormal rounding carry into the smallest normal.
      r.bits[30:0] = {7'd0,q[23:0]};
      if (ftz && !q[23]) r.bits[30:0] = '0;
    end else begin
      if (q[24]) begin q = q >> 1; e = e+1; end
      if (e > 127) begin r = '0; r.fault = 1; end
      else r.bits[30:0] = {8'(e+127),q[22:0]};
    end
    return r;
  endfunction

  function automatic result_t from_dot(input logic signed [40:0] dot,
                                       input logic signed [8:0] e_star);
    logic [40:0] magnitude;
    result_t r;
    r = '0;
    if (e_star < -9'sd133 || e_star > 9'sd120) begin r.fault=1; return r; end
    // Unsigned two's-complement magnitude also handles -2^40.
    magnitude = dot[40] ? (~$unsigned(dot) + 41'd1) : $unsigned(dot);
    return round_binary({239'd0,magnitude}, int'(e_star)-25, dot[40], 1'b0);
  endfunction

  function automatic result_t mul32(input logic [31:0] a_in,
                                    input logic [31:0] b_in,
                                    input logic ftz);
    result_t r;
    logic [31:0] a,b;
    logic [23:0] ma,mb;
    logic [47:0] p;
    integer ea,eb;
    r = '0;
    // Invalid is checked before DAZ or a zero multiplier can hide it.
    if (a_in[30:23] == 255 || b_in[30:23] == 255) begin r.fault=1; return r; end
    a = operand(a_in,ftz); b = operand(b_in,ftz);
    ma = {a[30:23] != 0,a[22:0]}; mb = {b[30:23] != 0,b[22:0]};
    ea = a[30:23] == 0 ? -149 : int'({1'b0,a[30:23]})-150;
    eb = b[30:23] == 0 ? -149 : int'({1'b0,b[30:23]})-150;
    p = ma*mb;
    return round_binary({232'd0,p},ea+eb,a[31]^b[31],ftz);
  endfunction

  function automatic result_t add32(input logic [31:0] a_in,
                                    input logic [31:0] b_in,
                                    input logic ftz);
    result_t r;
    logic [31:0] a,b;
    logic [279:0] ma,mb,magnitude;
    logic signed [280:0] sa,sb,total;
    r = '0;
    if (a_in[30:23] == 255 || b_in[30:23] == 255) begin r.fault=1; return r; end
    a = operand(a_in,ftz); b = operand(b_in,ftz);
    ma = {256'd0,a[30:23] != 0,a[22:0]};
    mb = {256'd0,b[30:23] != 0,b[22:0]};
    if (a[30:23] != 0) ma = ma << (a[30:23]-8'd1);
    if (b[30:23] != 0) mb = mb << (b[30:23]-8'd1);
    sa = $signed({1'b0,ma}); if (a[31]) sa = -sa;
    sb = $signed({1'b0,mb}); if (b[31]) sb = -sb;
    total = sa+sb;
    magnitude = total[280] ? 280'(-total) : total[279:0];
    return round_binary(magnitude,-149,
                        total == 0 ? (a[31] && b[31]) : total[280],ftz);
  endfunction

  function automatic result_t scale32(input logic [7:0] scale);
    result_t r;
    logic [3:0] m;
    integer e;
    r = '0;
    if (scale > 8'h7e) begin r.fault=1; return r; end
    m = {scale[6:3] != 0,scale[2:0]};
    e = scale[6:3] == 0 ? -9 : int'({1'b0,scale[6:3]})-10;
    return round_binary({276'd0,m},e,1'b0,1'b0);
  endfunction
endpackage
