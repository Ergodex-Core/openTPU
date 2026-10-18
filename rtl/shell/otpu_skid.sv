// A two-entry skid buffer: the input's ready is a flip-flop (high while the second slot is
// free), the beat shown downstream comes from the first slot's registers, a beat a cycle through
// it while the consumer keeps up, and no combinational path from an input to an output. The
// shell-facing side of the F2 blocks (otpu_f2_hbm_host's PCIS AW / W / AR, otpu_axil_cdc's OCL
// AW / W / AR): every output the shell sees is a register, and a shell register -> CL LUT ->
// shell register path (two SLR hops in the 4 ns of clk_main_a0, docs/wide_port.md "Round 4
// RTL" and "Round 5 RTL") cannot form.
(* keep_hierarchy = "yes" *)
module otpu_skid #(
  parameter int W = 1
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         i_v,
  output logic         i_r,
  input  logic [W-1:0] i_d,
  output logic         o_v,
  input  logic         o_r,
  output logic [W-1:0] o_d,
  output logic         any                 // a beat in either slot
);
  logic         r1_v;                      // the second slot (holds the beat taken as the
  logic [W-1:0] r1_d;                      //   consumer stalled)
  assign i_r = !r1_v;
  assign any = o_v || r1_v;
  wire i_tk = i_v && !r1_v;
  wire o_tk = o_v && o_r;
  always_ff @(posedge clk) begin
    if (o_tk || !o_v) o_d <= r1_v ? r1_d : i_d;
    else if (i_tk) r1_d <= i_d;
    if (rst) begin
      o_v <= 1'b0; r1_v <= 1'b0;
    end else if (o_tk || !o_v) begin
      o_v <= r1_v || i_tk; r1_v <= 1'b0;
    end else if (i_tk) begin
      r1_v <= 1'b1;
    end
  end
endmodule
