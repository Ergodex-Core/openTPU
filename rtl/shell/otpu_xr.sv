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
  // (shreg_extract no: a chain of stages must stay flip-flops, one per SLR, never an SRL: the AWS
  // Shell Interface Specification's rule for crossing registers)
  (* max_fanout = 256, shreg_extract = "no" *) logic [W-1:0] r;   // r_reg: the floorplan's USER_SLL_REG cells
  assign q = r;
  always_ff @(posedge clk) begin
    r <= d;
    if (rst) for (int i = 0; i < RW; i++) r[i] <= 1'b0;
  end
endmodule

// A pumped SLR chain (P3's F2 build, docs/v4/p3-status.md "SLR crossings"): W bits a core cycle
// cross as two halves of HW = ceil(W / 2) bits on clk_f (2 x clk, phase aligned with it: the CL's
// MMCM, the pumped TMEM's clock), through NXF stages of otpu_xr on clk_f, so a boundary carries
// half the signals of a core-clock chain (super long lines: 23,040 a boundary on the VU47P) at the
// same bandwidth. NXF even; the latency NXF / 2 + 2 core cycles from d to q, every core cycle's
// word in order and whole:
//   ser   (clk_f, the d end): the low half at the edge in mid-cycle, the high half at the core edge
//   g_x   NXF otpu_xr stages on clk_f (the floorplan places them by SLR: two per SLR at NXF 6)
//   des   the low half in a core register at the core edge where it is at the chain's end, the
//         high half in a clk_f register at the next mid edge, the word in a core register at the
//         core edge after (o)
// Every crossing between the clocks is local and takes half a core period (clk -> clk_f at the
// mid edge, clk_f -> clk at the core edge): no multicycle path. Each end has its own phase: a core
// toggle tg and its clk_f copy tgf, and sel <= (tg == tgf) on clk_f, which is high exactly in the
// first clk_f period of each core cycle once the clocks run (self-aligning, no reset); the two ends
// are kept apart (keep: they sit SLRs apart). No reset on the data: whatever is in the chain
// flushes in its latency while both sides hold reset, as otpu_xr's chains.
module otpu_xr2 #(
  parameter int W   = 1,
  parameter int NXF = 6
) (
  input  logic         clk,
  input  logic         clk_f,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  localparam int HW = (W + 1) / 2;
  initial if (NXF < 2 || NXF % 2 != 0) $fatal(1, "otpu_xr2: NXF %0d (even, >= 2)", NXF);
  // the serializer's phase and halves
  (* keep = "true" *) logic tg_s = 1'b0, tgf_s = 1'b0;
  (* keep = "true", max_fanout = 32 *) logic sel_s = 1'b0;
  always_ff @(posedge clk) tg_s <= !tg_s;
  always_ff @(posedge clk_f) begin
    tgf_s <= tg_s;
    sel_s <= (tg_s == tgf_s);
  end
  wire [2*HW-1:0] dx = (2*HW)'(d);
  (* shreg_extract = "no" *) logic [HW-1:0] ser;
  always_ff @(posedge clk_f) ser <= sel_s ? dx[HW-1:0] : dx[2*HW-1:HW];
  // the stages
  logic [NXF:0][HW-1:0] c;
  assign c[0] = ser;
  for (genvar k = 0; k < NXF; k++) begin : g_x
    otpu_xr #(.W(HW)) u_s (.clk(clk_f), .rst(1'b0), .d(c[k]), .q(c[k + 1]));
  end
  // the deserializer's phase and halves
  (* keep = "true" *) logic tg_d = 1'b0, tgf_d = 1'b0;
  (* keep = "true", max_fanout = 32 *) logic sel_d = 1'b0;
  always_ff @(posedge clk) tg_d <= !tg_d;
  always_ff @(posedge clk_f) begin
    tgf_d <= tg_d;
    sel_d <= (tg_d == tgf_d);
  end
  (* shreg_extract = "no" *) logic [HW-1:0] lo_c, hi_f;
  (* shreg_extract = "no" *) logic [W-1:0] o;
  always_ff @(posedge clk) lo_c <= c[NXF];
  always_ff @(posedge clk_f) if (sel_d) hi_f <= c[NXF];
  always_ff @(posedge clk) o <= W'({hi_f, lo_c});
  assign q = o;
endmodule
