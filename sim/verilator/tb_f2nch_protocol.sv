// Deterministic native-channel interface regression. This deliberately does not
// inspect the command FIFO or issue logic, so a registered FIFO head can change
// latency without changing the test's contract. No proprietary shell is needed.
module tb_f2nch_protocol;
  parameter int CORE_HALF = 5, AXI_HALF = 2;
  localparam logic [63:0] BASE = 64'h8000_0000;
  logic clk = 0, aclk = 0, rst = 1, arst = 1;
  always #(CORE_HALF) clk = !clk;
  always #(AXI_HALF) aclk = !aclk;

  logic n_cvalid = 0, n_cready, n_cwe = 0, n_wvalid = 0, n_wready;
  logic [24:0] n_caddr = 0;
  logic [511:0] n_wdata = 0, n_rdata;
  logic [63:0] n_wmask = 0;
  logic n_rvalid, n_err;
  logic [15:0] n_wdone;
  logic m_awvalid, m_awready = 0, m_wvalid, m_wready = 0;
  logic m_bvalid = 0, m_bready, m_arvalid, m_arready = 0;
  logic m_rvalid = 0, m_rready;
  logic [63:0] m_awaddr, m_araddr, m_wstrb;
  logic [511:0] m_wdata, m_rdata = 0;
  otpu_f2_nch #(.BASE(BASE), .CQ(4), .RD(8), .ROD(4), .WOD(4)) dut (
    .clk, .rst, .n_cvalid, .n_cready, .n_cwe, .n_caddr,
    .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid, .n_rdata, .n_wdone, .n_err,
    .aclk, .arst, .m_awvalid, .m_awready, .m_awaddr,
    .m_wvalid, .m_wready, .m_wdata, .m_wstrb, .m_bvalid, .m_bready, .m_bresp(2'b00),
    .m_arvalid, .m_arready, .m_araddr, .m_rvalid, .m_rready, .m_rdata, .m_rresp(2'b00));

  logic [63:0] expected_aw[$], expected_ar[$], expected_mask[$];
  logic [511:0] expected_w[$], expected_r[$];
  int aw_count = 0, w_count = 0, ar_count = 0, b_count = 0, r_count = 0;
  bit aw_held = 0, w_held = 0, ar_held = 0;
  logic [63:0] held_aw, held_ar, held_mask;
  logic [511:0] held_w;
  logic [15:0] previous_wdone = 0;

  function automatic logic [511:0] pattern(input int tag);
    for (int k = 0; k < 16; k++) pattern[32*k +: 32] = 32'(tag) ^ (32'h9e37_79b9 * 32'(k + 1));
  endfunction
  function automatic logic [63:0] address(input int beat);
    return BASE + 64'(beat) * 64;
  endfunction

  // Check independent ready/valid channels, including their payloads throughout
  // stalls. Expected queues describe transactions, never implementation cycles.
  always @(posedge aclk) begin
    if (arst) begin
      aw_held = 0; w_held = 0; ar_held = 0;
      aw_count = 0; w_count = 0; ar_count = 0; b_count = 0;
    end else begin
      if (aw_held && (!m_awvalid || m_awaddr !== held_aw)) $fatal(1, "AW changed while stalled");
      if (w_held && (!m_wvalid || m_wdata !== held_w || m_wstrb !== held_mask))
        $fatal(1, "W changed while stalled");
      if (ar_held && (!m_arvalid || m_araddr !== held_ar)) $fatal(1, "AR changed while stalled");
      aw_held = m_awvalid && !m_awready; held_aw = m_awaddr;
      w_held = m_wvalid && !m_wready; held_w = m_wdata; held_mask = m_wstrb;
      ar_held = m_arvalid && !m_arready; held_ar = m_araddr;
      if (m_awvalid && m_awready) begin
        if (!expected_aw.size()) $fatal(1, "unexpected AW, possibly a reset command survived");
        if (m_awaddr !== expected_aw.pop_front()) $fatal(1, "AW address/order mismatch");
        aw_count++;
      end
      if (m_wvalid && m_wready) begin
        if (!expected_w.size() || !expected_mask.size()) $fatal(1, "unexpected W");
        if (m_wdata !== expected_w.pop_front() || m_wstrb !== expected_mask.pop_front())
          $fatal(1, "W data/mask/order mismatch");
        w_count++;
      end
      if (m_arvalid && m_arready) begin
        if (!expected_ar.size()) $fatal(1, "unexpected AR, possibly a reset command survived");
        if (m_araddr !== expected_ar.pop_front()) $fatal(1, "AR address/order mismatch");
        ar_count++;
      end
      if (m_bvalid && m_bready) b_count++;
    end
  end
  always @(posedge clk) begin
    if (rst) begin r_count = 0; previous_wdone = 0; end
    else begin
      if (n_err) $fatal(1, "unexpected native error");
      if (n_wdone < previous_wdone || int'(n_wdone) > b_count)
        $fatal(1, "write completion count is not a monotonic count of B responses");
      previous_wdone = n_wdone;
      if (n_rvalid) begin
        if (!expected_r.size()) $fatal(1, "unexpected native read completion");
        if (n_rdata !== expected_r.pop_front()) $fatal(1, "native read data/order mismatch");
        r_count++;
      end
    end
  end

  task automatic command(input bit we, input int beat);
    @(negedge clk); n_cvalid = 1; n_cwe = we; n_caddr = 25'(beat);
    do @(posedge clk); while (!n_cready);
    @(negedge clk); n_cvalid = 0;
  endtask
  task automatic data(input logic [511:0] value, input logic [63:0] mask);
    @(negedge clk); n_wvalid = 1; n_wdata = value; n_wmask = mask;
    do @(posedge clk); while (!n_wready);
    @(negedge clk); n_wvalid = 0;
  endtask
  task automatic expect_write(input int beat, input logic [511:0] value, input logic [63:0] mask);
    expected_aw.push_back(address(beat)); expected_w.push_back(value); expected_mask.push_back(mask);
  endtask
  task automatic expect_read(input int beat);
    expected_ar.push_back(address(beat));
  endtask
  task automatic wait_axi(input int aws, input int ws, input int ars);
    do @(negedge aclk); while (aw_count < aws || w_count < ws || ar_count < ars);
    if (aw_count != aws || w_count != ws || ar_count != ars) $fatal(1, "extra AXI transaction");
  endtask
  task automatic no_read(input int cycles);
    repeat (cycles) begin
      @(negedge aclk);
      if (m_arvalid) $fatal(1, "read passed a pending same-address write");
    end
  endtask
  task automatic no_write(input int cycles);
    repeat (cycles) begin
      @(negedge aclk);
      if (m_awvalid || m_wvalid) $fatal(1, "write passed a pending same-address read");
    end
  endtask
  task automatic quiet(input int cycles);
    repeat (cycles) begin
      @(negedge aclk);
      if (m_awvalid || m_wvalid || m_arvalid || n_rvalid)
        $fatal(1, "command escaped a missing-data stall or an empty/reset queue");
    end
  endtask
  task automatic respond_b();
    @(negedge aclk); m_bvalid = 1;
    do @(posedge aclk); while (!m_bready);
    @(negedge aclk); m_bvalid = 0;
  endtask
  task automatic respond_r(input logic [511:0] value);
    expected_r.push_back(value);
    @(negedge aclk); m_rvalid = 1; m_rdata = value;
    do @(posedge aclk); while (!m_rready);
    @(negedge aclk); m_rvalid = 0;
  endtask
  task automatic reset_channel();
    @(negedge aclk); rst = 1; arst = 1;
    repeat (6) @(negedge clk);
    repeat (6) @(negedge aclk);
    arst = 0;
    @(negedge clk); rst = 0;
  endtask

  initial begin
    reset_channel();
    // The write has no data, so it and the read behind it must remain pending.
    // Reset discards both, including any future prefetch/head register contents.
    command(1, 9); command(0, 10);
    quiet(16);
    reset_channel();
    quiet(16);
    if (n_wdone != 0) $fatal(1, "reset retained write completions");

    // Fresh data arrives before its command: it must not revive the discarded
    // write. Distinct addresses/patterns expose stale command/data pairings.
    expect_write(3, pattern(101), '1);
    data(pattern(101), '1);
    quiet(16);
    command(1, 3); expect_read(3); command(0, 3);
    @(negedge aclk); m_awready = 1; m_wready = 0;
    wait_axi(1, 0, 0);
    @(negedge aclk); m_awready = 0;
    no_read(12); // AW accepted while W is held; no read can bypass it.
    if (!m_wvalid) $fatal(1, "write data was lost after AW acceptance");
    m_wready = 1;
    wait_axi(1, 1, 0);
    @(negedge aclk); m_wready = 0;
    no_read(12); // Both AXI write channels accepted, but B is still missing.
    respond_b();
    repeat (12) @(negedge aclk); // Hold the newly issued AR to check stability.
    m_arready = 1;
    wait_axi(1, 1, 1);
    @(negedge aclk); m_arready = 0;

    // Now a read is outstanding. The next same-address write, then read, are
    // queued together; neither may overtake the preceding operation.
    expect_write(3, pattern(202), 64'hffff_0000_ffff_0000);
    command(1, 3); data(pattern(202), 64'hffff_0000_ffff_0000);
    expect_read(3); command(0, 3);
    no_write(16);
    respond_r(pattern(101));
    @(negedge aclk); m_wready = 1; m_awready = 0;
    wait_axi(1, 2, 1);
    @(negedge aclk); m_wready = 0;
    no_read(12); // W accepted first this time; AW must remain stable.
    if (!m_awvalid) $fatal(1, "write address was lost after W acceptance");
    m_awready = 1;
    wait_axi(2, 2, 1);
    @(negedge aclk); m_awready = 0;
    no_read(12);
    respond_b();
    @(negedge aclk); m_arready = 1;
    wait_axi(2, 2, 2);
    @(negedge aclk); m_arready = 0;
    // Model the masked second write independently of the DUT payload checker.
    begin
      automatic logic [511:0] merged = pattern(101);
      automatic logic [511:0] newer = pattern(202);
      for (int k = 0; k < 64; k++) if ((64'hffff_0000_ffff_0000 >> k) & 1)
        merged[8*k +: 8] = newer[8*k +: 8];
      respond_r(merged);
    end
    do @(negedge clk); while (r_count != 2 || n_wdone != 2);
    quiet(16);

    // Fill behind a stalled AR, consume/refill the command head repeatedly,
    // then fully drain and send one more command through the empty transition.
    for (int k = 0; k < 4; k++) begin expect_read(20 + k); command(0, 20 + k); end
    @(negedge aclk); m_arready = 1;
    wait_axi(2, 2, 6);
    @(negedge aclk); m_arready = 0;
    for (int k = 0; k < 4; k++) respond_r(pattern(300 + k));
    do @(negedge clk); while (r_count != 6);
    quiet(16);
    expect_read(31); command(0, 31);
    @(negedge aclk); m_arready = 1;
    wait_axi(2, 2, 7);
    @(negedge aclk); m_arready = 0;
    respond_r(pattern(999));
    do @(negedge clk); while (r_count != 7);
    quiet(16);
    if (expected_aw.size() || expected_w.size() || expected_mask.size() || expected_ar.size() || expected_r.size())
      $fatal(1, "scoreboards did not drain");
    if (n_wdone != 2 || b_count != 2) $fatal(1, "write completions did not drain");
    $display("PASS F2 native protocol: reset-pending, AW-first, W-first, same-address ordering, drain/refill");
    $finish;
  end
  initial begin
    repeat (10000) @(posedge aclk);
    $fatal(1, "protocol regression timed out: AW=%0d W=%0d AR=%0d B=%0d native-R=%0d", aw_count, w_count, ar_count, b_count, r_count);
  end
endmodule
