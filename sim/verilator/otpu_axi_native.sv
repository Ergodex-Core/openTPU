// Simulation model of the F2 memory path's AXI side for one native channel: an AXI4 slave as the
// burst200 fabric's SmartConnect presents the HBM to otpu_native_axi (rtl/boards/f2/burst200:
// read bursts of up to 16 beats inside a 4 KiB page, single-beat writes, 512-bit data, the
// channel's 2 GiB window at HBM address WIN x 2 GiB, one ID), in front of otpu_native_mem's native
// port (one command per 64-byte beat, write data in write-command order, read data in order with
// no backpressure, n_wdone the writes performed). otpu_top AXI = 3 and tb_board MEM_NATIVE = 3 put
// one per channel between otpu_native_axi and the memory model, so the adapter's coalescing, its
// direction barrier, its credits and its write acknowledgements are in the loop of every run.
//
// The model takes AR, AW and W independently, each randomly withheld (+f2_stall=percent, seed
// +axi_seed=N as the memory model), up to MAXR read bursts and MAXW writes accepted and not yet
// answered (+f2_maxr, +f2_maxw: the SmartConnect's outstanding limits); it issues the accepted
// bursts' beats in order, one native command a cycle (reads before writes: the adapter never has
// both in flight), returns the read data in order with RLAST on each burst's last beat and random
// bubbles (+f2_rgap=percent), and answers a write once the memory has performed it (n_wdone), so a
// B never precedes the write's visibility (the contract otpu_native_axi's barrier relies on: a
// read issued after every earlier write's B sees them). RRESP and BRESP are OKAY. It stops the
// simulation on a broken AXI master obligation (AR / AW / W changed while waiting, a burst over
// 16 beats or across a 4 KiB page, an address outside the window, a response it did not owe).
module otpu_axi_native #(
  parameter int WIN  = 0,                // the channel's window: HBM address bits [33:31]
  parameter int MAXR = 32,               // read bursts accepted and not complete
  parameter int MAXW = 32                // writes accepted and not answered
) (
  input  logic         clk,
  input  logic         rst,
  // AXI4 slave (otpu_native_axi's master port)
  input  logic [33:0]  s_awaddr,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [511:0] s_wdata,
  input  logic [63:0]  s_wstrb,
  input  logic         s_wvalid,
  output logic         s_wready,
  output logic         s_bvalid,
  output logic [1:0]   s_bresp,
  input  logic         s_bready,
  input  logic [33:0]  s_araddr,
  input  logic [7:0]   s_arlen,
  input  logic         s_arvalid,
  output logic         s_arready,
  output logic [511:0] s_rdata,
  output logic         s_rvalid,
  output logic         s_rlast,
  output logic [1:0]   s_rresp,
  input  logic         s_rready,
  // native master (otpu_native_mem's channel)
  output logic         n_cvalid,
  input  logic         n_cready,
  output logic         n_cwe,
  output logic [24:0]  n_caddr,
  output logic         n_wvalid,
  input  logic         n_wready,
  output logic [511:0] n_wdata,
  output logic [63:0]  n_wmask,
  input  logic         n_rvalid,
  input  logic [511:0] n_rdata,
  input  logic [15:0]  n_wdone
);
  int stall = 0, rgap = 0, maxr = MAXR, maxw = MAXW;
  initial begin
    void'($value$plusargs("f2_stall=%d", stall));
    void'($value$plusargs("f2_rgap=%d", rgap));
    void'($value$plusargs("f2_maxr=%d", maxr));
    void'($value$plusargs("f2_maxw=%d", maxw));
  end
  function automatic bit chance(input int pct);
    return ($urandom % 100) < pct;
  endfunction

  typedef struct { logic [24:0] beat; int len; int done; } ar_t;
  typedef struct { logic [511:0] d; logic last; } r_t;
  ar_t          arq [$];                 // bursts accepted, beats not all issued
  logic [24:0]  awq [$];                 // write addresses accepted, not issued
  logic [511:0] wq_d [$];                // write data accepted, not given to the memory
  logic [63:0]  wq_m [$];
  logic         tagq [$];                // read beats issued: is it its burst's last
  r_t           rq [$];                  // read data back, not yet on R
  int           bursts = 0;              // read bursts accepted, last beat not yet on R
  int           wrs = 0;                 // writes accepted, B not yet given
  int           bpend = 0;               // writes performed, B not yet given
  logic [15:0]  wd_q;
  logic         ar_ok, aw_ok, w_ok;
  logic         cv, cwe, wv, rv, rl, bv;
  logic [24:0]  ca;
  logic [511:0] wd, rd;
  logic [63:0]  wm;
  // the master's held values (its obligations while waiting)
  logic         ar_w, aw_w, w_w;
  logic [33:0]  ar_a, aw_a;
  logic [7:0]   ar_l;
  logic [511:0] w_d;
  logic [63:0]  w_m;

  assign s_arready = !rst && ar_ok && arq.size() < maxr && bursts < maxr;
  assign s_awready = !rst && aw_ok && awq.size() < maxw && wrs < maxw;
  assign s_wready = !rst && w_ok && wq_d.size() < maxw;
  assign s_bvalid = bv;
  assign s_bresp = 2'b00;
  assign s_rvalid = rv;
  assign s_rdata = rd;
  assign s_rlast = rl;
  assign s_rresp = 2'b00;
  assign n_cvalid = cv;
  assign n_cwe = cwe;
  assign n_caddr = ca;
  assign n_wvalid = wv;
  assign n_wdata = wd;
  assign n_wmask = wm;

  always_ff @(posedge clk) begin
    if (rst) begin
      arq.delete(); awq.delete(); wq_d.delete(); wq_m.delete(); tagq.delete(); rq.delete();
      bursts = 0; wrs = 0; bpend = 0;
      wd_q <= n_wdone;
      ar_ok <= 1'b0; aw_ok <= 1'b0; w_ok <= 1'b0;
      cv <= 1'b0; wv <= 1'b0; rv <= 1'b0; bv <= 1'b0; rl <= 1'b0;
      ar_w <= 1'b0; aw_w <= 1'b0; w_w <= 1'b0;
    end else begin
      logic c_busy, w_busy, r_busy, b_busy;
      // ---- the master's obligations: a request waiting stays as it is
      if (ar_w && (!s_arvalid || s_araddr != ar_a || s_arlen != ar_l))
        $fatal(1, "otpu_axi_native %0d: AR changed while waiting", WIN);
      if (aw_w && (!s_awvalid || s_awaddr != aw_a))
        $fatal(1, "otpu_axi_native %0d: AW changed while waiting", WIN);
      if (w_w && (!s_wvalid || s_wdata != w_d || s_wstrb != w_m))
        $fatal(1, "otpu_axi_native %0d: W changed while waiting", WIN);
      ar_w <= s_arvalid && !s_arready; ar_a <= s_araddr; ar_l <= s_arlen;
      aw_w <= s_awvalid && !s_awready; aw_a <= s_awaddr;
      w_w <= s_wvalid && !s_wready; w_d <= s_wdata; w_m <= s_wstrb;
      ar_ok <= !chance(stall);
      aw_ok <= !chance(stall);
      w_ok <= !chance(stall);
      // ---- accept
      if (s_arvalid && s_arready) begin
        if (int'(s_araddr[33:31]) != WIN || s_araddr[5:0] != 6'd0)
          $fatal(1, "otpu_axi_native %0d: read address %h outside the window or not a beat", WIN,
                 s_araddr);
        if (s_arlen > 8'd15 || int'(s_araddr[11:6]) + int'(s_arlen) > 63)
          $fatal(1, "otpu_axi_native %0d: read burst at %h of %0d beats over 16 or a 4 KiB page",
                 WIN, s_araddr, int'(s_arlen) + 1);
        arq.push_back('{s_araddr[30:6], int'(s_arlen) + 1, 0});
        bursts++;
      end
      if (s_awvalid && s_awready) begin
        if (int'(s_awaddr[33:31]) != WIN || s_awaddr[5:0] != 6'd0)
          $fatal(1, "otpu_axi_native %0d: write address %h outside the window or not a beat", WIN,
                 s_awaddr);
        awq.push_back(s_awaddr[30:6]);
        wrs++;
      end
      if (s_wvalid && s_wready) begin
        wq_d.push_back(s_wdata);
        wq_m.push_back(s_wstrb);
      end
      // ---- the memory: one command a cycle, held until taken; write data likewise
      c_busy = cv && !n_cready;
      if (!c_busy) begin
        cv <= 1'b0;
        if (arq.size() != 0) begin
          ar_t h;
          h = arq[0];
          cv <= 1'b1; cwe <= 1'b0;
          ca <= h.beat + 25'(h.done);
          h.done = h.done + 1;
          tagq.push_back(h.done == h.len);
          if (h.done == h.len) void'(arq.pop_front());
          else arq[0] = h;
        end else if (awq.size() != 0) begin
          cv <= 1'b1; cwe <= 1'b1;
          ca <= awq.pop_front();
        end
      end
      w_busy = wv && !n_wready;
      if (!w_busy) begin
        wv <= 1'b0;
        if (wq_d.size() != 0) begin
          wv <= 1'b1;
          wd <= wq_d.pop_front();
          wm <= wq_m.pop_front();
        end
      end
      // ---- read data back (in order, no backpressure), then R
      if (n_rvalid) begin
        if (tagq.size() == 0) $fatal(1, "otpu_axi_native %0d: read data with no read", WIN);
        rq.push_back('{n_rdata, tagq.pop_front()});
      end
      r_busy = rv && !s_rready;
      if (rv && s_rready && rl) bursts--;
      if (!r_busy) begin
        rv <= 1'b0;
        if (rq.size() != 0 && !chance(rgap)) begin
          rv <= 1'b1;
          rd <= rq[0].d;
          rl <= rq[0].last;
          void'(rq.pop_front());
        end
      end
      // ---- writes performed, then B
      bpend += int'(16'(n_wdone - wd_q));
      wd_q <= n_wdone;
      b_busy = bv && !s_bready;
      if (bv && s_bready) wrs--;
      if (!b_busy) begin
        bv <= 1'b0;
        if (bpend != 0 && !chance(rgap)) begin
          bv <= 1'b1;
          bpend--;
        end
      end
      if (bpend > wrs) $fatal(1, "otpu_axi_native %0d: more writes performed than accepted", WIN);
    end
  end
endmodule
