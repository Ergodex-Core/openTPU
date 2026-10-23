// One register stage of an SLR crossing: otpu_board at S > 1 (a pair per direction and slice, one
// stage on each side, u_*_tx / u_*_rx), and the F2 HBM build's fabric (otpu_f2_hbm_mem,
// otpu_f2_hbm_sys, otpu_axil_xr: the native ports, the control path and the resets between the
// HBM fabric in SLR0 and the slice in SLR2, a stage in each SLR: docs/wide_port.md
// "Placement"). The low RW bits (valid pulses, request levels) clear with rst; the others are
// data. max_fanout: a stage's reset bit drives a slice's whole core reset (as core_rst). The
// stage's r_reg cells are what a floorplan marks USER_SLL_REG (bcu1525_floorplan.tcl) and what
// a pblock per SLR places (boards/f2/build/constraints/small_shell_cl_pnr_user_hbm.xdc).
module otpu_xr #(
  parameter int W  = 1,
  parameter int RW = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  (* max_fanout = 256 *) logic [W-1:0] r;     // r_reg: the floorplan's USER_SLL_REG cells
  assign q = r;
  always_ff @(posedge clk) begin
    r <= d;
    if (rst) for (int i = 0; i < RW; i++) r[i] <= 1'b0;
  end
endmodule
