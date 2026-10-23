// Collective unit: BAR and GATHER across S slices. When every slice is waiting on the same
// collective, GATHER streams each slice's segment, up to LANES consecutive words per cycle, and
// broadcasts it into the TMEM of all slices at dst + s*seg + r*drs + c; then all slices are
// released together. `gnt` is the AND of every slice's TMEM grant; without it the unit holds.
// RQ (TMEM's input stage, release r2, docs/tmem_nocopy.md 7): the reads go to the slices from a
// request register (stage a), a cycle after they form, as every unit's do, and a read's data is
// taken one granted cycle later than without it. For the slices' TMEMs on the unit's own ports
// (otpu_top; otpu_board at S = 1); otpu_board's hub at S > 1 takes RQ 0 (the ports present).
module otpu_coll
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8,
  parameter bit RQ    = 1'b1
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  input  logic                            gnt,
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic [S-1:0][LANES-1:0][31:0]   r_data,
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data
);
  typedef enum logic [1:0] {C_IDLE, C_RUN, C_ACK, C_COOL} st_t;
  st_t st;
  logic [31:0] dst, drs, seg;
  logic [15:0] rows, cols;
  logic [31:0] s, r, c, ps, pr, pcl;
  logic        pv, issuing;
  // row base addresses kept by adding the strides (no multipliers):
  // rrow = w1[s] + r*w4[s], wrow = dst + s*seg + r*drs, wseg = dst + s*seg
  logic [31:0] rrow, wrow, wseg;
  // the read stage's address (rrow + c) and lane mask (c + l < cols), registered with c and
  // rrow: the TMEM read address pins take a flip-flop, not an adder (c -> TMEM block RAM
  // address was a 0.24 ns path at 125.49 MHz)
  logic [31:0]      ra;
  logic [LANES-1:0] rm;
  // the read requests from flip-flops: rd_on is st == C_RUN && issuing, ssel the one-hot of s
  // (a 32-bit compare of s and the enable gating were in front of the TMEM read address: the
  // build's u_coll s -> TMEM block RAM address path)
  logic             rd_on;
  logic [S-1:0]     ssel;
  // the write stage's address base and lane mask, registered with pcl (they feed the TMEM
  // arbiter, so no adder or compare sits in front of it)
  logic [31:0]      pwa;
  logic [LANES-1:0] pm;
  // write stage q: stage p's address, mask and TMEM read data, registered (no path from the
  // TMEM block RAMs to their write port in one cycle); qlast: it holds the command's last write.
  // The lanes' addresses are registered each (qwl[l] = pwa + l): the arbiter's bank masks start
  // at flip-flops, not at a carry chain (133.33 MHz, e698dcd: qwa -> qwa + l -> the collective's
  // bank mask -> every lower unit's grant -> the MXU's / VPU's enables and TMEM's pw_*, 8-10
  // levels, +0.03 ns)
  logic                   qv, qlast;
  logic [LANES-1:0][31:0] qwl;
  logic [LANES-1:0]       qm;
  logic [LANES-1:0][31:0] qd;
  // RQ: the request register (stage a): the read's slice, lanes and per-lane addresses, and its
  // position (as, ar, acl) and write base (awa) for stage p
  logic                   av;
  logic [S-1:0]           asel;
  logic [LANES-1:0]       am;
  logic [LANES-1:0][31:0] aal;
  logic [31:0]            as_, ar, acl, awa;

  always_comb begin
    ack = (st == C_ACK);
    r_en = '0; r_addr = '0;
    w_en = '0; w_addr = '0; w_data = '0;
    for (int k = 0; k < S; k++)
      for (int l = 0; l < LANES; l++)
        if (RQ ? av && asel[k] && am[l] : rd_on && ssel[k] && rm[l]) begin
          r_en[k][l] = 1'b1;
          r_addr[k][l] = RQ ? aal[l] : ra + 32'(l);
        end
    if (st == C_RUN && qv) begin
      for (int l = 0; l < LANES; l++)
        if (qm[l]) begin
          w_en[l] = 1'b1;
          w_addr[l] = qwl[l];
          w_data[l] = qd[l];
        end
    end
  end

`ifndef SYNTHESIS
  // every slice names the same collective, and a GATHER the same dst (R[rb] + w2, resolved),
  // rows, cols, drs and seg (docs/isa.md): the unit takes slice 0's
  always @(posedge clk)
    if (!rst && st == C_IDLE && &req)
      for (int k = 1; k < S; k++)
        if (cmds[k].op != cmds[0].op ||
            (cmds[0].op == OP_GATHER && (cmds[k].w2 != cmds[0].w2 || cmds[k].w3 != cmds[0].w3 ||
                                         cmds[k].w5 != cmds[0].w5 || cmds[k].w6 != cmds[0].w6)))
          $fatal(1, "otpu_coll: slices 0 and %0d disagree on the collective", k);
`endif

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      pv <= 1'b0;
      qv <= 1'b0;
      rd_on <= 1'b0;
      av <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          if (cmds[0].op == OP_BAR || cmds[0].w3[15:0] == 0 || cmds[0].w3[31:16] == 0) begin
            st <= C_ACK;
          end else begin
            dst  <= cmds[0].w2;
            rows <= cmds[0].w3[15:0];
            cols <= cmds[0].w3[31:16];
            drs  <= cmds[0].w5;
            seg  <= cmds[0].w6;
            s <= '0; r <= '0; c <= '0;
            ssel <= S'(1);
            rrow <= cmds[0].w1;
            ra <= cmds[0].w1;
            for (int l = 0; l < LANES; l++) rm[l] <= (l < int'(cmds[0].w3[31:16]));
            wrow <= cmds[0].w2; wseg <= cmds[0].w2;
            pv <= 1'b0;
            qv <= 1'b0;
            av <= 1'b0;
            issuing <= 1'b1;
            rd_on <= 1'b1;
            st <= C_RUN;
          end
        end
        C_RUN: if (gnt) begin
          qv <= pv;
          for (int l = 0; l < LANES; l++) qwl[l] <= pwa + 32'(l);
          qm <= pm;
          for (int l = 0; l < LANES; l++)
            for (int k = 0; k < S; k++) if (k == int'(ps)) qd[l] <= r_data[k][l];
          qlast <= pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols);
          if (RQ) begin                                 // the read granted is stage a's
            av <= issuing;
            asel <= ssel; am <= rm;
            for (int l = 0; l < LANES; l++) aal[l] <= ra + 32'(l);
            as_ <= s; ar <= r; acl <= c;
            awa <= wrow + c;
            pv <= av;
            ps <= as_; pr <= ar; pcl <= acl;
            pwa <= awa;
            pm <= am;                                   // (rm[l] is c + l < cols)
          end else begin
            pv <= issuing;
            ps <= s; pr <= r; pcl <= c;
            pwa <= wrow + c;
            for (int l = 0; l < LANES; l++) pm[l] <= (c + 32'(l) < 32'(cols));
          end
          if (issuing) begin
            logic [31:0] nc, nrrow;
            nc = c; nrrow = rrow;
            if (c + LANES >= 32'(cols)) begin
              nc = '0;
              if (r + 1 == 32'(rows)) begin
                r <= '0;
                if (s + 1 == S) begin
                  issuing <= 1'b0;
                  rd_on <= 1'b0;
                end else begin
                  s <= s + 1;
                  ssel <= ssel << 1;
                  for (int k = 0; k < S; k++) if (k == int'(s) + 1) nrrow = cmds[k].w1;
                  wseg <= wseg + seg;
                  wrow <= wseg + seg;
                end
              end else begin
                r <= r + 1;
                for (int k = 0; k < S; k++) if (k == int'(s)) nrrow = rrow + cmds[k].w4;
                wrow <= wrow + drs;
              end
            end else begin
              nc = c + LANES;
            end
            c <= nc; rrow <= nrrow;
            ra <= nrrow + nc;
            for (int l = 0; l < LANES; l++) rm[l] <= (nc + 32'(l) < 32'(cols));
          end
          if (pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols))
            pv <= 1'b0;                                 // the last read's data moves to q
          if (qv && qlast) begin                         // the last write is taken this cycle
            qv <= 1'b0;
            st <= C_ACK;
          end
        end
        C_ACK:  st <= C_COOL;
        C_COOL: st <= C_IDLE;
        default: st <= C_IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && (rd_on != (st == C_RUN && issuing) || (rd_on && ssel != S'(1) << s)))
      $fatal(1, "otpu_coll: rd_on %0d / ssel %b do not match st %0d issuing %0d s %0d",
             rd_on, ssel, st, issuing, s);
  // qwl[l] is the registered pwa + l (qwa: the one address it replaced)
  logic [31:0] qwa;
  bit rst_seen;                                 // (registers start arbitrary)
  initial rst_seen = 1'b0;
  always @(posedge clk) begin
    if (rst) rst_seen <= 1'b1;
    else if (st == C_RUN && gnt) qwa <= pwa;
    if (!rst && rst_seen && st == C_RUN && qv)
      for (int l = 0; l < LANES; l++)
        if (qwl[l] != qwa + 32'(l))
          $fatal(1, "otpu_coll: lane %0d's write address %0d is not %0d", l, qwl[l], qwa + 32'(l));
  end
`endif
endmodule

// The collective unit with reads in flight (COLL_PIPE > 0 at S > 1: otpu_board g_xs, otpu_top
// COLL_HUB; behind otpu_coll_hub K = COLL_PIPE). otpu_coll against the hub of K = 0 takes a read's
// data at its next granted cycle, so one read is in flight and a beat of LANES words costs the
// read's round trip through the crossings and the slice's port (~9 cycles; on the F2 card at
// S = 4, 11% of a Gemma token's cycles, docs/v4/p4-status.md). Here the read and write sides are
// decoupled. The reads (otpu_coll's iterator: slice s's segment row by row, up to LANES words a
// beat) go out whenever the hub takes them (r_gnt), ahead of their data; the data comes back in
// the reads' order (r_valid / r_data, taken with r_take), and a second iterator over the same
// beats gives each its write, to every slice at dst + s*seg + r*drs + c, through the write stage
// q (lane addresses registered each, as otpu_coll's), sent when the hub takes it (w_gnt). The
// command, the BAR, the ack and the cool-down are otpu_coll's. RQ 0 only (the ports present).
module otpu_coll_pu
  import otpu_pkg::*;
#(
  parameter int S     = 2,
  parameter int LANES = 8
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic                            r_gnt,     // the read presented is taken
  input  logic                            r_valid,   // the oldest read's data is in r_data
  input  logic [LANES-1:0][31:0]          r_data,
  output logic                            r_take,    // ... and is taken this cycle
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data,
  input  logic                            w_gnt      // the write presented is taken
);
  typedef enum logic [1:0] {C_IDLE, C_RUN, C_ACK, C_COOL} st_t;
  st_t st;
  logic [31:0] dst, drs, seg;
  logic [15:0] rows, cols;
  // the read side (otpu_coll's, from flip-flops): slice s (ssel its one-hot), row r, column c;
  // rrow = w1[s] + r*w4[s], ra = rrow + c, rm[l] = c + l < cols; rd_on: a read is presented
  logic [31:0]      s, r, c, rrow, ra;
  logic [LANES-1:0] rm;
  logic [S-1:0]     ssel;
  logic             rd_on;
  // the write side, the beat the oldest data is for: slice ws, row wr, column wc; wrow = dst +
  // ws*seg + wr*drs, wseg = dst + ws*seg, wa = wrow + wc, wm[l] = wc + l < cols
  logic [31:0]      ws, wr, wc, wrow, wseg, wa;
  logic [LANES-1:0] wm;
  // the write stage q; qlast: it holds the command's last write
  logic                   qv, qlast;
  logic [LANES-1:0]       qm;
  logic [LANES-1:0][31:0] qwl, qd;
  wire wtake = qv && w_gnt;

  assign ack = (st == C_ACK);
  assign r_take = st == C_RUN && r_valid && (!qv || w_gnt);
  always_comb begin
    r_en = '0; r_addr = '0;
    for (int k = 0; k < S; k++)
      for (int l = 0; l < LANES; l++)
        if (rd_on && ssel[k] && rm[l]) begin
          r_en[k][l] = 1'b1;
          r_addr[k][l] = ra + 32'(l);
        end
    w_en = qv ? qm : '0;
    w_addr = qwl;
    w_data = qd;
  end

`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && st == C_IDLE && &req)
      for (int k = 1; k < S; k++)
        if (cmds[k].op != cmds[0].op ||
            (cmds[0].op == OP_GATHER && (cmds[k].w2 != cmds[0].w2 || cmds[k].w3 != cmds[0].w3 ||
                                         cmds[k].w5 != cmds[0].w5 || cmds[k].w6 != cmds[0].w6)))
          $fatal(1, "otpu_coll_pu: slices 0 and %0d disagree on the collective", k);
`endif

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      rd_on <= 1'b0;
      qv <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          if (cmds[0].op == OP_BAR || cmds[0].w3[15:0] == 0 || cmds[0].w3[31:16] == 0) begin
            st <= C_ACK;
          end else begin
            dst  <= cmds[0].w2;
            rows <= cmds[0].w3[15:0];
            cols <= cmds[0].w3[31:16];
            drs  <= cmds[0].w5;
            seg  <= cmds[0].w6;
            s <= '0; r <= '0; c <= '0;
            ssel <= S'(1);
            rrow <= cmds[0].w1;
            ra <= cmds[0].w1;
            for (int l = 0; l < LANES; l++) rm[l] <= (l < int'(cmds[0].w3[31:16]));
            ws <= '0; wr <= '0; wc <= '0;
            wrow <= cmds[0].w2; wseg <= cmds[0].w2; wa <= cmds[0].w2;
            for (int l = 0; l < LANES; l++) wm[l] <= (l < int'(cmds[0].w3[31:16]));
            qv <= 1'b0;
            rd_on <= 1'b1;
            st <= C_RUN;
          end
        end
        C_RUN: begin
          if (rd_on && r_gnt) begin                    // the next read (otpu_coll's iterator)
            logic [31:0] nc, nrrow;
            nc = c; nrrow = rrow;
            if (c + LANES >= 32'(cols)) begin
              nc = '0;
              if (r + 1 == 32'(rows)) begin
                r <= '0;
                if (s + 1 == S) begin
                  rd_on <= 1'b0;
                end else begin
                  s <= s + 1;
                  ssel <= ssel << 1;
                  for (int k = 0; k < S; k++) if (k == int'(s) + 1) nrrow = cmds[k].w1;
                end
              end else begin
                r <= r + 1;
                for (int k = 0; k < S; k++) if (k == int'(s)) nrrow = rrow + cmds[k].w4;
              end
            end else begin
              nc = c + LANES;
            end
            c <= nc; rrow <= nrrow;
            ra <= nrrow + nc;
            for (int l = 0; l < LANES; l++) rm[l] <= (nc + 32'(l) < 32'(cols));
          end
          if (r_take) begin                            // the oldest data into the write stage
            logic [31:0] nc, nwrow;
            qv <= 1'b1;
            qd <= r_data;
            qm <= wm;
            for (int l = 0; l < LANES; l++) qwl[l] <= wa + 32'(l);
            qlast <= ws + 1 == S && wr + 1 == 32'(rows) && wc + LANES >= 32'(cols);
            nc = wc; nwrow = wrow;
            if (wc + LANES >= 32'(cols)) begin
              nc = '0;
              if (wr + 1 == 32'(rows)) begin
                wr <= '0;
                if (ws + 1 != S) begin
                  ws <= ws + 1;
                  wseg <= wseg + seg;
                  nwrow = wseg + seg;
                end
              end else begin
                wr <= wr + 1;
                nwrow = wrow + drs;
              end
            end else begin
              nc = wc + LANES;
            end
            wc <= nc; wrow <= nwrow;
            wa <= nwrow + nc;
            for (int l = 0; l < LANES; l++) wm[l] <= (nc + 32'(l) < 32'(cols));
          end else if (wtake) begin
            qv <= 1'b0;
          end
          if (wtake && qlast) begin                    // the last write is taken this cycle
            qv <= 1'b0;
            st <= C_ACK;
          end
        end
        C_ACK:  st <= C_COOL;
        C_COOL: st <= C_IDLE;
        default: st <= C_IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && ((rd_on && st != C_RUN) || (rd_on && ssel != S'(1) << s) ||
                 (r_take && wtake && qlast) || (st == C_RUN && wtake && qlast && rd_on)))
      $fatal(1, "otpu_coll_pu: rd_on %0d / ssel %b / st %0d / s %0d, or data past the last write",
             rd_on, ssel, st, s);
`endif
endmodule

// The collective unit without GATHER (otpu_board COLL_GATHER = 0, one slice only;
// docs/infra_hunt.md 5): BAR as otpu_coll runs it (acknowledged two cycles after the request,
// then a cycle cool-down), no TMEM access (its read and write ports tied off, so synthesis drops
// the TMEM's collective write port). A GATHER is not run: it is acknowledged like a BAR, moves no
// data, and raises gerr until reset (STATUS ERROR through otpu_board). The compiler never emits
// GATHER at S = 1 (opentpu.language.all_gather), and the host refuses to load a program with one
// on such a bitstream (CAPS2 bit0 = 0: opentpu/host/board.py). Ports as otpu_coll's, plus gerr.
module otpu_coll_bar
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  input  logic                            gnt,
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic [S-1:0][LANES-1:0][31:0]   r_data,
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data,
  output logic                            gerr      // a GATHER came (sticky until rst)
);
  initial if (S != 1) $fatal(1, "otpu_coll_bar: one slice only (GATHER across slices needs otpu_coll)");
  typedef enum logic [1:0] {C_IDLE, C_ACK, C_COOL} st_t;
  st_t st;
  assign ack    = (st == C_ACK);
  assign r_en   = '0;
  assign r_addr = '0;
  assign w_en   = '0;
  assign w_addr = '0;
  assign w_data = '0;
  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      gerr <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          st <= C_ACK;
          if (cmds[0].op == OP_GATHER) gerr <= 1'b1;
        end
        C_ACK:  st <= C_COOL;
        default: st <= C_IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && st == C_IDLE && &req && cmds[0].op == OP_GATHER && !$test$plusargs("coll_gerr_ok"))
      $fatal(1, "otpu_coll_bar: GATHER on a collective unit without it (COLL_GATHER = 0)");
`endif
endmodule

// The collective unit's side of the SLR crossings (otpu_board at S > 1, every slice a few
// register stages away). otpu_coll runs unchanged against it, but its `gnt` is this hub's, from
// the hub's own registers: no slice's TMEM grant in the cycle (at S = 1 the grant is the AND of
// the slices' grants in the same cycle, a path through every SLR). Each slice's otpu_coll_port
// takes the unit's reads and writes and has them granted in its own TMEM, later. A read holds the
// unit until its data is back (the unit takes a read's data at its next granted cycle: one read
// in flight); writes go to every slice's port, at most F not yet written per port (each port's
// pops come back as credits). The unit's ack goes to the slices once every port has written
// everything, and a slice's request counts again only after it dropped (the unit is idle two
// cycles after its ack; the slices see the ack later). The ISA's rule that a GATHER's source and
// destination do not overlap makes the order of a slice's reads and writes free.
// K > 0 (COLL_PIPE, docs/v4/p4-status.md "The pipelined collective"): the hub of otpu_coll_pu,
// up to K reads not yet taken by the unit (in flight, or back in a FIFO of K): u_rgnt takes the
// unit's read while fewer are held, the FIFO's oldest is u_rvalid / u_rdata[*] (taken with
// u_rtake), u_gnt is the writes' room alone. A slice's port returns its reads in order; the reads
// of the next slice go out once every read of the last one is back (the FIFO keeps the order,
// one return a cycle). The writes, the credits, the ack and the requests as at K = 0.
module otpu_coll_hub
  import otpu_pkg::*;
#(
  parameter int S     = 2,
  parameter int LANES = 8,
  parameter int F     = 8,              // writes a port holds
  parameter int K     = 0               // 0: one read in flight (otpu_coll); K: otpu_coll_pu's
) (
  input  logic                            clk,
  input  logic                            rst,
  // otpu_coll's ports (K = 0) / otpu_coll_pu's (K > 0: u_gnt its w_gnt, and u_rgnt .. u_rtake)
  output logic [S-1:0]                    u_req,
  output cmd_t                            u_cmds [S],
  output logic                            u_gnt,
  input  logic                            u_ack,
  input  logic [S-1:0][LANES-1:0]         u_ren,
  input  logic [S-1:0][LANES-1:0][31:0]   u_raddr,
  output logic [S-1:0][LANES-1:0][31:0]   u_rdata,
  output logic                            u_rgnt,
  output logic                            u_rvalid,
  input  logic                            u_rtake,
  input  logic [LANES-1:0]                u_wen,
  input  logic [LANES-1:0][31:0]          u_waddr,
  input  logic [LANES-1:0][31:0]          u_wdata,
  // the slices' ports, through the crossing registers
  input  logic [S-1:0]                    x_req,
  input  cmd_t                            x_cmds [S],
  output logic                            x_ack,     // to every slice (pulse)
  output logic [S-1:0]                    x_rv,      // a read for slice s (pulse)
  output logic [LANES-1:0]                x_rm,
  output logic [LANES-1:0][31:0]          x_ra,
  input  logic [S-1:0]                    x_dv,      // its data (pulse)
  input  logic [S-1:0][LANES-1:0][31:0]   x_dd,
  output logic                            x_wv,      // a write for every slice (pulse)
  output logic [LANES-1:0]                x_wm,
  output logic [LANES-1:0][31:0]          x_wa,
  output logic [LANES-1:0][31:0]          x_wd,
  input  logic [S-1:0]                    x_pop      // a port wrote one (pulse)
);
  localparam int CW = $clog2(F + 2);
  logic                   pend;
  logic [S-1:0]           blk;            // the slice's request was acknowledged, not yet dropped
  logic [S-1:0][CW-1:0]   o;              // writes sent to a port and not popped (as seen here)
  logic                   room, drained;

  always_comb begin
    room = 1'b1;
    drained = 1'b1;
    for (int k = 0; k < S; k++) begin
      if (o[k] >= CW'(F)) room = 1'b0;
      if (o[k] != '0) drained = 1'b0;
    end
  end
  assign u_req = rst ? '0 : x_req & ~blk;   // (a request still crossing when the run stops)
  for (genvar k = 0; k < S; k++) begin : g_k
    assign u_cmds[k] = x_cmds[k];
  end
  always_comb begin
    x_wm = u_wen;
    x_wa = u_waddr;
    x_wd = u_wdata;
    x_ack = pend && drained;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      pend <= 1'b0;
      blk <= '0;
      o <= '0;
    end else begin
      if (u_ack) pend <= 1'b1;
      else if (x_ack) pend <= 1'b0;
      for (int k = 0; k < S; k++) begin
        if (u_ack) blk[k] <= 1'b1;
        else if (!x_req[k]) blk[k] <= 1'b0;
        o[k] <= o[k] + CW'(x_wv) - CW'(x_pop[k]);
      end
    end
  end

  if (K == 0) begin : g_one
    // one read in flight: the unit waits for its data (rd_wait), which it takes from hd at its
    // next granted cycle
    logic                   rd_wait;
    logic [LANES-1:0][31:0] hd;
    assign u_gnt = !rd_wait && room;
    assign u_rgnt = 1'b0;
    assign u_rvalid = 1'b0;
    for (genvar k = 0; k < S; k++) begin : g_kd
      assign u_rdata[k] = hd;
    end
    always_comb begin
      x_rv = '0;
      x_rm = '0;
      x_ra = '0;
      for (int k = 0; k < S; k++) begin   // the unit reads one slice at a time (the others' '0)
        x_rv[k] = u_gnt && |u_ren[k];
        x_rm |= u_ren[k];
        x_ra |= u_raddr[k];
      end
      x_wv = u_gnt && |u_wen;
    end
    always_ff @(posedge clk) begin
      for (int k = 0; k < S; k++) if (x_dv[k]) hd <= x_dd[k];
      if (rst) rd_wait <= 1'b0;
      else if (|x_rv) rd_wait <= 1'b1;
      else if (|x_dv) rd_wait <= 1'b0;
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && ((|x_rv && rd_wait) || (|x_dv && !rd_wait) || $countones(x_rv) > 1))
        $fatal(1, "otpu_coll_hub: read %b while waiting %0d, data %b", x_rv, rd_wait, x_dv);
`endif
  end else begin : g_pipe
    // K reads held: in flight (infl: sent, data not back) or in the FIFO (fn), held = infl + fn;
    // rsel: the slice of the reads in flight. K a power of two (the FIFO's pointers wrap)
    localparam int KW = $clog2(K + 1), PW = $clog2(K);
    logic [KW-1:0]          infl, held, fn;
    logic [S-1:0]           rsel, rs;
    logic [LANES-1:0][31:0] fd [K];
    logic [PW-1:0]          fwp, frp;
    logic [LANES-1:0][31:0] dsel;
    initial if (K != (1 << PW)) $fatal(1, "otpu_coll_hub: K %0d is not a power of two", K);
    assign u_gnt = room;
    assign u_rvalid = fn != '0;
    for (genvar k = 0; k < S; k++) begin : g_kd
      assign u_rdata[k] = fd[frp];
    end
    always_comb begin
      for (int k = 0; k < S; k++) rs[k] = |u_ren[k];
      u_rgnt = held < KW'(K) && (infl == '0 || (rs & rsel) != '0);
      x_rv = '0;
      x_rm = '0;
      x_ra = '0;
      for (int k = 0; k < S; k++) begin   // the unit reads one slice at a time (the others' '0)
        x_rv[k] = u_rgnt && rs[k];
        x_rm |= u_ren[k];
        x_ra |= u_raddr[k];
      end
      x_wv = room && |u_wen;
      dsel = '0;
      for (int k = 0; k < S; k++) if (x_dv[k]) dsel = x_dd[k];   // (one slice's reads in flight)
    end
    always_ff @(posedge clk) begin
      if (|x_dv) fd[fwp] <= dsel;
      if (rst) begin
        infl <= '0;
        held <= '0;
        fn <= '0;
        fwp <= '0;
        frp <= '0;
        rsel <= '0;
      end else begin
        infl <= infl + KW'(|x_rv) - KW'(|x_dv);
        held <= held + KW'(|x_rv) - KW'(u_rtake);
        fn <= fn + KW'(|x_dv) - KW'(u_rtake);
        if (|x_dv) fwp <= fwp + 1'b1;
        if (u_rtake) frp <= frp + 1'b1;
        if (|x_rv) rsel <= rs;
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && ((|x_rv && held == KW'(K)) || (|x_dv && infl == '0) || (u_rtake && fn == '0) ||
                   $countones(x_rv) > 1 || $countones(x_dv) > 1 || (|x_dv && (x_dv & ~rsel) != '0)))
        $fatal(1, "otpu_coll_hub: read %b with %0d held, data %b with %0d in flight (slice %b), take %0d of %0d",
               x_rv, held, x_dv, infl, rsel, u_rtake, fn);
`endif
  end
endmodule

// A slice's collective port at S > 1 (otpu_board): the hub's reads (one at a time) and writes (a
// FIFO of F) presented on the slice's collective TMEM port and granted by the slice's own TMEM
// arbiter (gnt = gnt_local: the collective group's grant, reads and writes together), the read
// data returned the cycle after it is fresh, each written entry returned as a pop.
// K > 0 (otpu_coll_hub K, COLL_PIPE): up to K reads queue (the hub holds no more), each a mask
// and lane 0's address (lane l's is + l, as the unit forms them); the oldest is presented from a
// head register (flip-flops: TMEM's input stage) and the next takes its place in the cycle it is
// granted, so a read can go every granted cycle, and the data return in the reads' order. The
// writes the same way: a FIFO of F behind a head register, so the collective's write request
// (its lanes, addresses and data, into the slice's write-bank masks) leaves from flip-flops, not
// from the FIFO's LUT RAM and its count (470cf908's worst core path at 175 MHz: u_cport rp ->
// the FIFO's RAMD32 -> the arbiter -> the MXU's enables, 9 levels, +0.000 ns).
module otpu_coll_port #(
  parameter int LANES = 8,
  parameter int F     = 8,
  parameter int K     = 0
) (
  input  logic                    clk,
  input  logic                    rst,
  // from the hub, through the crossing registers
  input  logic                    rv,
  input  logic [LANES-1:0]        rm,
  input  logic [LANES-1:0][31:0]  ra,
  input  logic                    wv,
  input  logic [LANES-1:0]        wm,
  input  logic [LANES-1:0][31:0]  wa,
  input  logic [LANES-1:0][31:0]  wd,
  // to the hub
  output logic                    dv,
  output logic [LANES-1:0][31:0]  dd,
  output logic                    pop,
  // the slice's collective port (otpu_slice coll_*)
  output logic [LANES-1:0]        coll_ren,
  output logic [LANES-1:0][31:0]  coll_raddr,
  input  logic [LANES-1:0][31:0]  coll_rdata,
  output logic [LANES-1:0]        coll_wen,
  output logic [LANES-1:0][31:0]  coll_waddr,
  output logic [LANES-1:0][31:0]  coll_wdata,
  input  logic                    gnt_local,
  output logic                    gnt
);
  localparam int PW = $clog2(F);         // F a power of two: the pointers wrap
  logic got;

  assign gnt = gnt_local;
  always_ff @(posedge clk) begin
    if (got) dd <= coll_rdata;          // fresh the cycle after the granted read
    if (rst) dv <= 1'b0;
    else dv <= got;
  end

  if (K == 0) begin : g_one
    // hq: hv's second cycle. The held read is asked from it (TMEM's input stage, release r2,
    // docs/tmem_nocopy.md 7: every unit's read goes to TMEM a cycle after it forms, the writes a
    // cycle after their grant), and held there until granted, with no replay
    logic                   hv, hq;
    logic [LANES-1:0]       hm;
    logic [LANES-1:0][31:0] ha;
    logic [LANES-1:0]       fm [F];
    logic [LANES-1:0][31:0] fa [F];
    logic [LANES-1:0][31:0] fd [F];
    logic [PW-1:0]          wp, rp;
    logic [PW:0]            n;
    wire                    wtake = n != '0 && gnt_local;

    assign coll_ren = (hv && hq) ? hm : '0;
    assign coll_raddr = ha;
    assign coll_wen = n != '0 ? fm[rp] : '0;
    assign coll_waddr = fa[rp];
    assign coll_wdata = fd[rp];

    always_ff @(posedge clk) begin
      if (rv) begin
        hm <= rm;
        ha <= ra;
      end
      if (wv) begin
        fm[wp] <= wm;
        fa[wp] <= wa;
        fd[wp] <= wd;
      end
      if (rst) begin
        hv <= 1'b0;
        hq <= 1'b0;
        got <= 1'b0;
        pop <= 1'b0;
        wp <= '0;
        rp <= '0;
        n <= '0;
      end else begin
        got <= hv && hq && gnt_local;
        pop <= wtake;
        if (rv) hv <= 1'b1;
        else if (hv && hq && gnt_local) hv <= 1'b0;
        hq <= hv && !(hq && gnt_local);
        if (wv) wp <= wp + 1'b1;
        if (wtake) rp <= rp + 1'b1;
        n <= n + (PW+1)'(wv) - (PW+1)'(wtake);
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && ((rv && hv) || (wv && n == (PW+1)'(F) && !wtake)))
        $fatal(1, "otpu_coll_port: a read while one is held (%0d) or a write into a full FIFO", hv);
`endif
  end else begin : g_pipe
    // the head read (hv, presented) and the queue behind it (qn reads); K a power of two
    localparam int KP = $clog2(K);
    logic                   hv;
    logic [LANES-1:0]       hm;
    logic [LANES-1:0][31:0] ha;
    logic [LANES-1:0]       qm [K];
    logic [31:0]            qa [K];
    logic [KP-1:0]          qwp, qrp;
    logic [KP:0]            qn;
    logic [LANES-1:0]       nm;
    logic [31:0]            na;
    wire                    rtake = hv && gnt_local;       // the head read is granted
    wire                    hfree = !hv || rtake;          // the head takes the next read
    wire                    qpop = hfree && qn != '0;
    wire                    qpush = rv && !(hfree && qn == '0);
    // the head write (gv, presented) and the FIFO behind it (n writes)
    logic                   gv;
    logic [LANES-1:0]       gm;
    logic [LANES-1:0][31:0] ga, gd;
    logic [LANES-1:0]       fm [F];
    logic [LANES-1:0][31:0] fa [F];
    logic [LANES-1:0][31:0] fd [F];
    logic [PW-1:0]          wp, rp;
    logic [PW:0]            n;
    wire                    wtake = gv && gnt_local;       // the head write is granted
    wire                    gfree = !gv || wtake;
    wire                    fpop = gfree && n != '0;
    wire                    fpush = wv && !(gfree && n == '0);
    initial if (K != (1 << KP)) $fatal(1, "otpu_coll_port: K %0d is not a power of two", K);

    assign coll_ren = hv ? hm : '0;
    assign coll_raddr = ha;
    assign coll_wen = gv ? gm : '0;
    assign coll_waddr = ga;
    assign coll_wdata = gd;
    always_comb begin                   // the head's next read: the oldest queued, else the new
      nm = qn != '0 ? qm[qrp] : rm;
      na = qn != '0 ? qa[qrp] : ra[0];
    end
    always_ff @(posedge clk) begin
      if (qpush) begin
        qm[qwp] <= rm;
        qa[qwp] <= ra[0];
      end
      if (hfree) begin
        hm <= nm;
        for (int l = 0; l < LANES; l++) ha[l] <= na + 32'(l);
      end
      if (fpush) begin
        fm[wp] <= wm;
        fa[wp] <= wa;
        fd[wp] <= wd;
      end
      if (gfree) begin                  // the head's next write: the oldest held, else the new
        gm <= n != '0 ? fm[rp] : wm;
        ga <= n != '0 ? fa[rp] : wa;
        gd <= n != '0 ? fd[rp] : wd;
      end
      if (rst) begin
        hv <= 1'b0;
        got <= 1'b0;
        qwp <= '0;
        qrp <= '0;
        qn <= '0;
        gv <= 1'b0;
        pop <= 1'b0;
        wp <= '0;
        rp <= '0;
        n <= '0;
      end else begin
        got <= rtake;
        if (hfree) hv <= qn != '0 || rv;
        if (qpush) qwp <= qwp + 1'b1;
        if (qpop) qrp <= qrp + 1'b1;
        qn <= qn + (KP+1)'(qpush) - (KP+1)'(qpop);
        pop <= wtake;
        if (gfree) gv <= n != '0 || wv;
        if (fpush) wp <= wp + 1'b1;
        if (fpop) rp <= rp + 1'b1;
        n <= n + (PW+1)'(fpush) - (PW+1)'(fpop);
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst) begin
        if (qpush && qn == (KP+1)'(K))
          $fatal(1, "otpu_coll_port: a read into a full queue (%0d)", qn);
        if (fpush && n == (PW+1)'(F))
          $fatal(1, "otpu_coll_port: a write into a full FIFO (%0d)", n);
        if (rv)                         // (the unit's lanes are lane 0's address + l)
          for (int l = 0; l < LANES; l++)
            if (rm[l] && ra[l] != ra[0] + 32'(l))
              $fatal(1, "otpu_coll_port: lane %0d's read address %0d is not %0d", l, ra[l], ra[0] + 32'(l));
      end
`endif
  end
endmodule
