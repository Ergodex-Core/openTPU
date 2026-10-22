// Folded strict BF16 x E2M1 K16 tree: 16 RNE32 products, then 8/4/2/1
// adjacent-pair RNE32 additions. No F24 alignment, reassociation or scale fusion.
// One accepted atom at a time; scale/tag/profile retained until output drains.
module otpu_nvfp4_strict_dot16 #(parameter integer TAG_W=32) (
  input logic clk,rst,
  input logic in_valid,
  output logic in_ready,
  input logic [255:0] in_a,
  input logic [63:0] in_w,
  input logic [15:0] in_mask,
  input logic in_ftz,
  input logic [7:0] in_scale,
  input logic [TAG_W-1:0] in_tag,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] out_bits,
  output logic out_fault,out_ftz,
  output logic [7:0] out_scale,
  output logic [TAG_W-1:0] out_tag
);
  import otpu_nvfp4_fp::*;
  typedef enum logic [1:0] {IDLE,PRODUCTS,REDUCE,OUTPUT} state_t;
  state_t state;
  logic [255:0] a;
  logic [63:0] w;
  logic [15:0] mask;
  logic ftz,fault;
  logic [31:0] partial[0:15];
  logic [3:0] index;
  logic [3:0] pairs;
  assign in_ready=!rst && state==IDLE;
  assign out_valid=!rst && state==OUTPUT;

  function automatic logic [31:0] e2m1(input logic [3:0] code);
    logic [31:0] q;
    case(code[2:0])
      0:q=32'h00000000; 1:q=32'h3f000000;
      2:q=32'h3f800000; 3:q=32'h3fc00000;
      4:q=32'h40000000; 5:q=32'h40400000;
      6:q=32'h40800000; 7:q=32'h40c00000;
      default:q=0;
    endcase
    q[31]=code[3];
    return q;
  endfunction

  always_ff @(posedge clk) begin : tree
    result_t r;
    if(rst) begin
      state<=IDLE;fault<=0;out_fault<=0;out_bits<=0;
      out_tag<='0;out_scale<=0;out_ftz<=0;
    end else case(state)
      IDLE: if(in_valid) begin
        a<=in_a;w<=in_w;mask<=in_mask;ftz<=in_ftz;
        out_tag<=in_tag;out_scale<=in_scale;out_ftz<=in_ftz;
        fault<=0;index<=0;state<=PRODUCTS;
      end
      PRODUCTS: begin
        // A masked term is a literal +0, even if its unused bits encode NaN/-0.
        r='0;
        if(mask[index]) r=mul32({a[16*index +:16],16'd0},e2m1(w[4*index +:4]),ftz);
        partial[index]<=r.bits;fault<=fault || r.fault;
        if(index==15) begin index<=0;pairs<=8;state<=REDUCE;end
        else index<=index+1'b1;
      end
      REDUCE: begin
        r=add32(partial[2*index],partial[2*index+1],ftz);
        partial[index]<=r.bits;fault<=fault || r.fault;
        if(index+1'b1==pairs) begin
          index<=0;
          if(pairs==1) begin
            out_fault<=fault || r.fault;
            out_bits<=(fault || r.fault) ? 32'd0 : r.bits;
            state<=OUTPUT;
          end else pairs<=pairs>>1;
        end else index<=index+1'b1;
      end
      OUTPUT:if(out_ready) state<=IDLE;
      default:state<=IDLE;
    endcase
  end
endmodule
