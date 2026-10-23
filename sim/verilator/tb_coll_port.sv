`timescale 1ns/1ps
// A queue scoreboard independent of the DUT's pointers. Exercise empty bypass,
// last-entry replacement, full pop/push, stalls, wraparound and reset while full.
module tb_coll_port;
  parameter int F = 8;
  parameter int LANES = 8;
  logic clk = 0;
  always #5 clk = !clk;
  logic rst = 1, rv = 0, wv = 0, gnt_local = 0;
  logic [LANES-1:0] rm = '0, wm, coll_ren, coll_wen;
  logic [LANES-1:0] coll_wbanks;
  logic [LANES-1:0][31:0] ra = '0, wa, wd, coll_rdata = '0;
  logic [LANES-1:0][31:0] dd, coll_raddr, coll_waddr, coll_wdata;
  logic dv, pop, gnt, coll_rany;
  otpu_coll_port #(.LANES(LANES), .F(F)) dut (.*);
  typedef struct packed {
    logic [LANES-1:0] m;
    logic [LANES-1:0][31:0] a, d;
  } entry_t;
  entry_t queue[$], item;
  int checks = 0, empty_push = 0, replace_last = 0, replace_full = 0, stalls = 0;
  bit take, expect_pop = 0;
  bit read_held = 0, read_asked = 0, read_got = 0, expect_dv = 0;
  bit next_asked, next_got;
  logic [LANES-1:0] read_mask = '0, expect_ren;
  logic [LANES-1:0][31:0] read_addr = '0, expect_dd = '0;
  logic [LANES-1:0] expect_banks;
  int seed = 32'h326f19;
  initial begin
    seed = $urandom(seed);
    repeat (2) @(negedge clk);
    for (int c = 0; c < 12000; c++) begin
      rst = c % 401 == 0;
      // Directed fill/drain/replace phases, followed by random backpressure.
      if (c % 401 < F + 2) begin
        gnt_local = 0; wv = queue.size() < F;
      end else if (c % 401 < F + 12) begin
        gnt_local = 1; wv = 1;
      end else if (c % 401 < 2 * F + 12) begin
        gnt_local = 1; wv = 0;
      end else if (c % 401 < 2 * F + 24) begin
        gnt_local = 1; wv = 1;
      end else begin
        gnt_local = ($urandom_range(0, 3) != 0);
        wv = ($urandom_range(0, 3) != 0) && (queue.size() < F || gnt_local);
      end
      wm = LANES'($urandom);
      rv = !read_held && ($urandom_range(0, 2) == 0);
      rm = LANES'($urandom);
      for (int l = 0; l < LANES; l++) begin
        wa[l] = $urandom; wd[l] = $urandom;
        ra[l] = $urandom; coll_rdata[l] = $urandom;
      end
      #1;
      if (!rst) begin
        expect_ren = (read_held && read_asked) ? read_mask : '0;
        if (coll_ren !== expect_ren || coll_rany !== (|expect_ren))
          $fatal(1, "read presentation at cycle %0d", c);
        if (read_held && coll_raddr !== read_addr) $fatal(1, "read address at %0d", c);
        if (dv !== expect_dv || (dv && dd !== expect_dd)) $fatal(1, "read response at %0d", c);
        expect_banks = '0;
        if (queue.size() != 0)
          for (int l = 0; l < LANES; l++)
            if (queue[0].m[l]) expect_banks[queue[0].a[l] % LANES] = 1'b1;
        if (coll_wbanks !== expect_banks) $fatal(1, "write banks at cycle %0d", c);
        if (coll_wen !== ((queue.size() != 0) ? queue[0].m : '0))
          $fatal(1, "head mask at cycle %0d", c);
        if (queue.size() != 0 &&
            (coll_waddr !== queue[0].a || coll_wdata !== queue[0].d))
          $fatal(1, "head payload at cycle %0d", c);
        if (pop !== expect_pop || gnt !== gnt_local) $fatal(1, "handshake at %0d", c);
        checks++;
      end
      take = queue.size() != 0 && gnt_local;
      if (!rst) begin
        if (wv && queue.size() == 0) empty_push++;
        if (wv && take && queue.size() == 1) replace_last++;
        if (wv && take && queue.size() == F) replace_full++;
        if (!gnt_local && queue.size() != 0) stalls++;
      end
      @(posedge clk);
      expect_pop = !rst && take;
      if (rst) begin
        queue.delete();
        read_held = 0; read_asked = 0; read_got = 0; expect_dv = 0;
      end
      else begin
        next_asked = read_held && !(read_asked && gnt_local);
        next_got = read_held && read_asked && gnt_local;
        expect_dv = read_got;
        if (read_got) expect_dd = coll_rdata;
        read_got = next_got;
        read_held = rv || (read_held && !(read_asked && gnt_local));
        read_asked = next_asked;
        if (rv) begin read_mask = rm; read_addr = ra; end
        if (take) item = queue.pop_front();
        if (wv) queue.push_back('{m: wm, a: wa, d: wd});
      end
      @(negedge clk);
    end
    if (!(empty_push && replace_last && replace_full && stalls)) $fatal(1, "missing coverage");
    $display("HEAD PASS checks=%0d empty=%0d last=%0d full=%0d stall=%0d",
             checks, empty_push, replace_last, replace_full, stalls);
    $finish;
  end
endmodule
