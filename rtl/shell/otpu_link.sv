// A valid / ready link between two placement regions where every net that crosses is a
// flip-flop to a flip-flop, in both directions (otpu_xr's rule, with a handshake): the sender
// (otpu_link_tx) takes a beat while it holds a credit and registers it with its take bit, the
// receiver (otpu_link_rx) registers them again (the stage the placer puts at the boundary),
// queues the beat (DEPTH, the credits) and returns a registered pop pulse per beat it gives
// on, which the sender registers once more before counting it back. Nothing combinational
// runs between the two: tx's o_v / o_d flip-flops into rx's input registers, rx's o_pop
// flip-flop into tx's. The F2 HBM build's PCIS registers (SLR1, by the shell) and the host
// path's queues (SLR0): rtl/boards/f2/otpu_f2_hbm_host.sv (round 6, docs/wide_port.md
// "Round 6 RTL"). A beat crosses in two cycles (tx, rx's stage) and a credit comes back in
// six, so DEPTH beats per six cycles is the link's rate: 4 / 6 of a beat a cycle at DEPTH 4.
(* keep_hierarchy = "yes" *)
module otpu_link_tx #(
  parameter int W     = 1,
  parameter int DEPTH = 4                  // the receiver's queue, the credits
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         i_v,
  output logic         i_r,                // a flip-flop: a credit is held
  input  logic [W-1:0] i_d,
  output logic         o_v,                // the stage: the take, registered
  output logic [W-1:0] o_d,
  input  logic         i_pop,              // the receiver's pop pulse (its flip-flop)
  output logic         busy                // a beat in the link or queued, as the credits see it
);
  localparam int CW = $clog2(DEPTH + 1);
  logic [CW-1:0] cr;
  logic          pop_q;
  wire take = i_v && i_r;
  assign busy = o_v || cr != CW'(DEPTH);
  always_ff @(posedge clk) begin
    if (take) o_d <= i_d;
    if (rst) begin
      o_v <= 1'b0; pop_q <= 1'b0; cr <= CW'(DEPTH); i_r <= 1'b1;
    end else begin
      o_v <= take;
      pop_q <= i_pop;
      cr <= cr - CW'(take) + CW'(pop_q);
      i_r <= (cr - CW'(take) + CW'(pop_q)) != 0;
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && take && cr == 0) $fatal(1, "otpu_link_tx: a take without a credit");
`endif
endmodule

(* keep_hierarchy = "yes" *)
module otpu_link_rx #(
  parameter int W     = 1,
  parameter int DEPTH = 4
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         i_v,                // the sender's stage flip-flops
  input  logic [W-1:0] i_d,
  output logic         o_pop,              // a flip-flop: a beat left the queue last cycle
  output logic         o_v,
  input  logic         o_r,
  output logic [W-1:0] o_d
);
  localparam int AW = $clog2(DEPTH);
  logic          v_q;                      // the stage
  logic [W-1:0]  d_q;
  logic [W-1:0]  mem [DEPTH];
  logic [AW-1:0] wp, rp;
  logic [AW:0]   n;
  assign o_v = n != 0;
  assign o_d = mem[rp];
  wire pop = o_v && o_r;
  always_ff @(posedge clk) begin
    d_q <= i_d;
    if (v_q) mem[wp] <= d_q;
    if (rst) begin
      v_q <= 1'b0; wp <= '0; rp <= '0; n <= '0; o_pop <= 1'b0;
    end else begin
      v_q <= i_v;
      if (v_q) wp <= wp + 1'b1;
      if (pop) rp <= rp + 1'b1;
      n <= n + (AW + 1)'(v_q) - (AW + 1)'(pop);
      o_pop <= pop;
    end
  end
`ifndef SYNTHESIS
  initial for (int i = 0; i < DEPTH; i++) mem[i] = '0;
  always @(posedge clk) if (!rst && v_q && n == (AW + 1)'(DEPTH)) $fatal(1, "otpu_link_rx: overflow");
`endif
endmodule
