// F2 OCL crossing: unrelated clocks, independent AW/W, backpressure in both domains,
// PCIS hold, byte masks, error responses, and ignored accesses outside the 4 KiB window.
module tb_f2axil;
  parameter int CORE_HALF = 5, AXI_HALF = 2;
  logic sclk = 0, mclk = 0, srst = 1, mrst = 1;
  always #(AXI_HALF) sclk = !sclk;
  always #(CORE_HALF) mclk = !mclk;
  logic hold = 0;
  logic [31:0] s_awaddr, s_wdata, s_araddr, s_rdata;
  logic [3:0] s_wstrb;
  logic s_awvalid = 0, s_wvalid = 0, s_arvalid = 0, s_bready = 0, s_rready = 0;
  logic s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
  logic [1:0] s_bresp, s_rresp;
  logic [11:0] m_awaddr, m_araddr;
  logic [31:0] m_wdata, m_rdata;
  logic [3:0] m_wstrb;
  logic m_awvalid, m_wvalid, m_bready, m_arvalid, m_rready;
  logic m_awready = 0, m_wready = 0, m_arready = 0, m_bvalid = 0, m_rvalid = 0;
  logic [1:0] m_bresp, m_rresp;
  otpu_axil_cdc dut (.*);

  logic [31:0] mem [1024], shadow [1024];
  logic saw_aw = 0, saw_w = 0;
  logic [11:0] wa;
  logic [31:0] wd;
  logic [3:0] wm;
  int bdelay = 0, rdelay = 0, writes = 0, reads = 0;
  int unsigned seed = 1, count = 300;
  function automatic logic [1:0] resp(input logic [31:0] a);
    return a[31:12] != 0 ? 2'b00 : a[9] ? 2'b10 : 2'b00;
  endfunction
  function automatic logic [31:0] merge(input logic [31:0] old, d, input logic [3:0] mask);
    for (int k = 0; k < 4; k++) merge[k*8 +: 8] = mask[k] ? d[k*8 +: 8] : old[k*8 +: 8];
  endfunction

  // The register slave deliberately accepts AW and W separately, then delays its response.
  always @(posedge mclk) begin
    if (mrst) begin
      saw_aw = 0; saw_w = 0; bdelay = 0; rdelay = 0;
      m_awready <= 0; m_wready <= 0; m_arready <= 0;
      m_bvalid <= 0; m_rvalid <= 0;
    end else begin
      if (m_awvalid && m_awready) begin
        if (saw_aw) $fatal(1, "duplicate AW");
        saw_aw = 1; wa = m_awaddr;
      end
      if (m_wvalid && m_wready) begin
        if (saw_w) $fatal(1, "duplicate W");
        saw_w = 1; wd = m_wdata; wm = m_wstrb;
      end
      if (saw_aw && saw_w && bdelay == 0 && !m_bvalid) begin
        mem[wa[11:2]] = merge(mem[wa[11:2]], wd, wm);
        bdelay = 1 + $urandom % 12; writes++;
        m_bresp <= resp({20'd0, wa});
      end
      if (bdelay > 0) begin
        bdelay--;
        if (bdelay == 0) m_bvalid <= 1;
      end
      if (m_bvalid && m_bready) begin m_bvalid <= 0; saw_aw = 0; saw_w = 0; end
      if (m_arvalid && m_arready) begin
        if (rdelay || m_rvalid) $fatal(1, "duplicate AR");
        m_rdata <= mem[m_araddr[11:2]];
        m_rresp <= resp({20'd0, m_araddr});
        rdelay = 1 + $urandom % 12; reads++;
      end
      if (rdelay > 0) begin
        rdelay--;
        if (rdelay == 0) m_rvalid <= 1;
      end
      if (m_rvalid && m_rready) m_rvalid <= 0;
      m_awready <= !saw_aw && !m_bvalid && $urandom % 3 != 0;
      m_wready <= !saw_w && !m_bvalid && $urandom % 3 != 0;
      m_arready <= !rdelay && !m_rvalid && $urandom % 3 != 0;
    end
  end

  task automatic put(input logic [31:0] a, d, input logic [3:0] mask, input int order);
    automatic bit awdone = 0, wdone = 0, done = 0;
    automatic int before_writes = writes;
    automatic int awdelay = order == 1 ? 8 : 0;
    automatic int wdelay = order == 2 ? 8 : 0;
    @(negedge sclk);
    s_awaddr = a; s_wdata = d; s_wstrb = mask;
    for (int t = 0; t < 1000; t++) begin
      automatic logic req_was = dut.x_req;
      hold = t < 4 || $urandom % 5 == 0;
      s_awvalid = !awdone && t >= awdelay;
      s_wvalid = !wdone && t >= wdelay;
      s_bready = $urandom % 3 != 0;
      @(posedge sclk);
      if (s_awvalid && s_awready) awdone = 1;
      if (s_wvalid && s_wready) wdone = 1;
      if (s_bvalid && s_bready) begin
        if (!awdone || !wdone || s_bresp !== resp(a)) $fatal(1, "incorrect write response");
        done = 1;
      end
      @(negedge sclk);
      // (the AW and W may enter their skid buffers during the hold; the write must not start:
      // its request toggle, with x_wr, waits for a cycle without hold)
      if (hold && dut.x_req != req_was && dut.x_wr) $fatal(1, "write started during PCIS hold");
      if (done) break;
    end
    if (!done) $fatal(1, "write timed out");
    s_awvalid = 0; s_wvalid = 0; s_bready = 0; hold = 0;
    if (a[31:12] == 0) begin
      shadow[a[11:2]] = merge(shadow[a[11:2]], d, mask);
      if (writes != before_writes + 1) $fatal(1, "write did not cross exactly once");
    end else if (writes != before_writes) $fatal(1, "outside-window write reached core");
  endtask

  task automatic get(input logic [31:0] a);
    automatic bit ardone = 0, done = 0;
    automatic int before_reads = reads;
    automatic logic [31:0] expected = a[31:12] == 0 ? shadow[a[11:2]] : 32'd0;
    @(negedge sclk);
    s_araddr = a;
    for (int t = 0; t < 1000; t++) begin
      hold = 1;                         // PCIS writes must not block control reads
      s_arvalid = !ardone;
      s_rready = $urandom % 3 != 0;
      @(posedge sclk);
      if (s_arvalid && s_arready) ardone = 1;
      if (s_rvalid && s_rready) begin
        if (!ardone || s_rdata !== expected || s_rresp !== resp(a))
          $fatal(1, "incorrect read response: address %h got %h expected %h", a, s_rdata, expected);
        done = 1;
      end
      @(negedge sclk);
      if (done) break;
    end
    if (!done) $fatal(1, "read timed out");
    s_arvalid = 0; s_rready = 0; hold = 0;
    if (reads != before_reads + int'(a[31:12] == 0)) $fatal(1, "incorrect core read count");
  endtask

  // AXI response payloads must remain stable while the shell is stalled.
  logic b_stall = 0, r_stall = 0;
  logic [1:0] prev_bresp, prev_rresp;
  logic [31:0] prev_rdata;
  always @(posedge sclk) begin
    if (!srst) begin
      if (b_stall && (!s_bvalid || s_bresp !== prev_bresp)) $fatal(1, "B changed under backpressure");
      if (r_stall && (!s_rvalid || s_rresp !== prev_rresp || s_rdata !== prev_rdata))
        $fatal(1, "R changed under backpressure");
    end
    b_stall <= !srst && s_bvalid && !s_bready;
    r_stall <= !srst && s_rvalid && !s_rready;
    prev_bresp <= s_bresp; prev_rresp <= s_rresp; prev_rdata <= s_rdata;
  end

  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("count=%d", count));
    void'($urandom(seed));
    for (int k = 0; k < 1024; k++) begin mem[k] = 32'hbeef0000 ^ k; shadow[k] = mem[k]; end
    repeat (20) @(posedge sclk);
    @(negedge mclk) mrst = 0;
    @(negedge sclk) srst = 0;
    for (int i = 0; i < count; i++) begin
      automatic logic [31:0] a = 4 * ($urandom % 1024);
      if (i % 5 == 0) a |= 32'h1000;    // CSR windows belonging to the other boards
      put(a, $urandom, 4'($urandom), i % 3);
      get(a);
    end
    $display("PASS: %0d write/read pairs, %0d core writes, %0d core reads", count, writes, reads);
    $finish;
  end
endmodule
