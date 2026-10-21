// Directed credit-boundary test for the NCH8 wide-port native adapter.
// All eight channels receive each four-chunk B read. Native responses are held until every
// channel reserves RD B beats, then held longer to force port-B backpressure. A runs and SW
// fill reads share the outstanding read tags. This is the adapter only, not the F2 AXI path.
module tb_native_rd_stress;
  parameter int RD = 128;
  localparam int NCH = 8;
  localparam int PORTB = 4;
  localparam int PIB = 2;
  localparam int NP = NCH / 2;
  localparam int B_TARGET = 2 * RD + 96; // crosses the B FIFO, order and read-tag ring depths

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1, release_reads = 1'b0;
  longint cycle = 0;

  function automatic logic [24:0] fold(input logic [24:0] u);
    logic [24:0] f;
    f = '0;
    for (int k = 0; k < 27; k += PIB) f ^= (u >> k) & 25'(NP - 1);
    return f;
  endfunction
  function automatic longint lbeat(input int c, input logic [24:0] u);
    longint m;
    m = longint'(u) * NP + longint'((25'(c / 2) ^ fold(u)) & 25'(NP - 1));
    return 2 * m + ((c % 2) ^ ($countones(m) % 2));
  endfunction
  function automatic logic [511:0] init(input longint b);
    logic [511:0] v;
    for (int i = 0; i < 16; i++) v[32 * i +: 32] = 32'(b) * 32'h9E37_79B9 ^ (i * 32'h85EB_CA6B);
    return v;
  endfunction

  logic a_rdy, a_req, a_we, a_rvalid, a_flush;
  logic [31:0] a_addr, a_wdata;
  logic [3:0] a_be;
  logic [511:0] a_rdata;
  logic sw_rdy, sw_req;
  logic [31:0] sw_addr, sw_wdata;
  logic [3:0] sw_be;
  logic b_rdy, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag, b_rlast;
  logic [31:0] b_addr;
  logic [1:0] b_n;
  logic [31:0] b_wmask;
  logic [1023:0] b_wdata;
  logic [4095:0] b_rdata;
  logic [PORTB-1:0] b_rcm;
  logic wr_idle, rd_idle;
  logic [NCH-1:0] n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid;
  logic [NCH-1:0][24:0] n_caddr;
  logic [NCH-1:0][511:0] n_wdata, n_rdata;
  logic [NCH-1:0][63:0] n_wmask;
  logic [NCH-1:0][15:0] n_wdone;

  otpu_native_dram #(.NCH(NCH), .PORTB(PORTB), .ALINE(64), .RD(RD)) u_dut (
    .clk, .rst, .a_flush, .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr),
    .a_wdata_x(a_wdata), .a_be_x(a_be), .a_rvalid, .a_rdata, .sw_rdy, .sw_req, .sw_addr,
    .sw_wdata, .sw_be, .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par,
    .b_n, .b_rvalid, .b_rtag, .b_rdata, .b_rcm, .b_rlast, .wr_idle, .rd_idle,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone);

  // In-order native channel model. The SW fill reads see prior writes to their beats.
  typedef struct packed { logic we; logic [24:0] addr; } cmd_t;
  cmd_t cmds [NCH][$];
  logic [511:0] writes [NCH][$];
  logic [63:0] masks [NCH][$];
  logic [511:0] responses [NCH][$];
  logic [511:0] mem [NCH][logic [24:0]];
  int native_reads [NCH], native_writes [NCH];
  assign n_cready = {NCH{!rst}};
  assign n_wready = {NCH{!rst}};
  for (genvar c = 0; c < NCH; c++) begin : g_native
    always @(posedge clk) begin
      if (rst) begin
        cmds[c].delete(); writes[c].delete(); masks[c].delete(); responses[c].delete();
        n_rvalid[c] <= 1'b0; n_wdone[c] <= '0;
        native_reads[c] <= 0; native_writes[c] <= 0;
      end else begin
        n_rvalid[c] <= 1'b0;
        if (release_reads && responses[c].size() != 0) begin
          n_rvalid[c] <= 1'b1;
          n_rdata[c] <= responses[c][0];
          void'(responses[c].pop_front());
        end
        if (n_cvalid[c] && n_cready[c]) cmds[c].push_back('{we: n_cwe[c], addr: n_caddr[c]});
        if (n_wvalid[c] && n_wready[c]) begin
          writes[c].push_back(n_wdata[c]); masks[c].push_back(n_wmask[c]);
        end
        if (cmds[c].size() != 0 && (!cmds[c][0].we || writes[c].size() != 0)) begin
          cmd_t cmd;
          logic [511:0] data;
          cmd = cmds[c].pop_front();
          data = mem[c].exists(cmd.addr) ? mem[c][cmd.addr] : init(lbeat(c, cmd.addr));
          if (cmd.we) begin
            logic [511:0] wd;
            logic [63:0] mask;
            wd = writes[c].pop_front(); mask = masks[c].pop_front();
            for (int i = 0; i < 64; i++) if (mask[i]) data[8 * i +: 8] = wd[8 * i +: 8];
            mem[c][cmd.addr] = data;
            n_wdone[c] <= n_wdone[c] + 1'b1;
            native_writes[c] <= native_writes[c] + 1;
          end else begin
            responses[c].push_back(data);
            native_reads[c] <= native_reads[c] + 1;
          end
        end
      end
    end
  end

  typedef struct packed { logic tag; logic [4095:0] data; } b_expected_t;
  b_expected_t b_expected [$];
  logic [511:0] a_expected [$];
  int b_sent = 0, b_got = 0, a_sent = 0, a_got = 0, sw_sent = 0;
  int b_stalls = 0, peak_rb [NCH], peak_tags [NCH];
  int sat_wait = 0;
  bit saturated = 0, b_done = 0, mixed_at_full = 0;

  // Source ports change only on falling edges; a valid request stays stable through a stall.
  always @(negedge clk) begin
    if (rst) begin
      a_req = 0; a_we = 0; a_addr = 0; a_wdata = 0; a_be = 0; a_flush = 0;
      sw_req = 0; sw_addr = 0; sw_wdata = 0; sw_be = 0;
      b_req = 0; b_we = 0; b_tag = 0; b_addr = 0; b_par = 0; b_n = 0;
      b_wmask = 0; b_wdata = 0;
    end else begin
      b_req = !b_done;
      b_we = 0; b_tag = 1'(b_sent); b_addr = 32'(32'h0002_0000 + 128 * (b_sent % 8));
      b_par = ^b_addr[31:5]; b_n = 2'd3;
      a_req = a_sent < 16; a_we = 0; a_addr = 32'(16 * a_sent);
      sw_req = sw_sent < 8; sw_addr = 32'(32'h0001_0000 + 16 * sw_sent);
      sw_wdata = 32'hA500_0000 | 32'(sw_sent); sw_be = 4'hF;
      a_flush = 0;
    end
  end

  always @(posedge clk) begin
    if (rst) begin
      cycle <= 0;
      for (int c = 0; c < NCH; c++) begin peak_rb[c] <= 0; peak_tags[c] <= 0; end
    end else begin
      bit all_full;
      cycle <= cycle + 1;
      all_full = 1;
      for (int c = 0; c < NCH; c++) begin
        if (int'(u_dut.rb_res[c]) > peak_rb[c]) peak_rb[c] <= int'(u_dut.rb_res[c]);
        if (int'(u_dut.tg_n[c]) > peak_tags[c]) peak_tags[c] <= int'(u_dut.tg_n[c]);
        if (u_dut.rb_res[c] != RD) all_full = 0;
        if (u_dut.rb_res[c] > RD) $fatal(1, "B reservation over RD on channel %0d", c);
      end
      if (all_full) begin
        int a_reads, sw_reads;
        a_reads = 0; sw_reads = 0;
        for (int c = 0; c < NCH; c++) begin
          a_reads += int'(u_dut.st_ard[c]);
          sw_reads += int'(u_dut.st_srd[c]);
          if (u_dut.tg_n[c] > RD) mixed_at_full <= 1;
        end
        if (a_reads != 0 && sw_reads != 0) saturated <= 1;
      end
      if (saturated && !release_reads) begin
        sat_wait <= sat_wait + 1;
        if (sat_wait == 40) release_reads <= 1;
      end
      if (b_req && !b_rdy) b_stalls <= b_stalls + 1;
      // Complete a stalled valid request before ending the B stream.
      if (release_reads && b_req && b_rdy && b_sent + 1 == B_TARGET) b_done <= 1;
      if (b_req && b_rdy) begin
        b_expected_t e;
        e.tag = b_tag;
        for (int i = 0; i < 8; i++) e.data[512 * i +: 512] = init(longint'(b_addr[31:4]) + i);
        b_expected.push_back(e);
        b_sent <= b_sent + 1;
      end
      if (a_req && a_rdy) begin
        a_expected.push_back(init(longint'(a_addr[31:4])));
        a_sent <= a_sent + 1;
      end
      if (sw_req && sw_rdy) sw_sent <= sw_sent + 1;
      if (b_rvalid) begin
        b_expected_t e;
        if (b_expected.size() == 0) $fatal(1, "unsolicited B response");
        e = b_expected.pop_front();
        if (b_rtag !== e.tag || b_rdata !== e.data || b_rcm !== 4'hF || !b_rlast)
          $fatal(1, "B response %0d out of order or corrupt", b_got);
        b_got <= b_got + 1;
      end
      if (a_rvalid) begin
        logic [511:0] e;
        if (a_expected.size() == 0) $fatal(1, "unsolicited A response");
        e = a_expected.pop_front();
        if (a_rdata !== e) $fatal(1, "A response %0d out of order or corrupt", a_got);
        a_got <= a_got + 1;
      end
    end
  end

  initial begin
    if (RD != 64 && RD != 128) $fatal(1, "test supports RD64/RD128");
    repeat (5) @(negedge clk);
    #1 rst = 0;
    for (int t = 0; t < 20000; t++) begin
      @(negedge clk);
      if (release_reads && !b_req && !a_req && !sw_req && wr_idle && rd_idle &&
          b_expected.size() == 0 && a_expected.size() == 0) begin
        int a_reads, sw_reads, sw_writes;
        a_reads = 0; sw_reads = 0; sw_writes = 0;
        if (!saturated || !mixed_at_full || b_stalls == 0 || b_sent != B_TARGET || b_got != b_sent ||
            a_sent != 16 || a_got != a_sent || sw_sent != 8)
          $fatal(1, "incomplete saturation/backpressure/drain: B %0d/%0d stall %0d A %0d/%0d SW %0d",
                 b_got, b_sent, b_stalls, a_got, a_sent, sw_sent);
        for (int c = 0; c < NCH; c++) begin
          if (peak_rb[c] != RD || peak_tags[c] < RD || native_reads[c] < B_TARGET)
            $fatal(1, "channel %0d: peak B reservation %0d, tags %0d, native reads %0d (RD %0d)",
                   c, peak_rb[c], peak_tags[c], native_reads[c], RD);
          a_reads += int'(u_dut.st_ard[c]);
          sw_reads += int'(u_dut.st_srd[c]);
          sw_writes += int'(u_dut.st_swr[c]);
          if (cmds[c].size() != 0 || responses[c].size() != 0 || writes[c].size() != 0)
            $fatal(1, "native channel %0d did not drain", c);
        end
        if (a_reads == 0 || sw_reads == 0 || sw_writes == 0)
          $fatal(1, "missing concurrent A/SW traffic: A %0d SW fill %0d SW writes %0d",
                 a_reads, sw_reads, sw_writes);
        $display("PASS native RD%0d NCH8 PORTB4 ALINE64: B %0d, stalls %0d, A %0d, SW fills %0d, SW writes %0d; all channels reserved %0d",
                 RD, b_got, b_stalls, a_reads, sw_reads, sw_writes, RD);
        $finish;
      end
    end
    $fatal(1, "timeout waiting for RD%0d saturation or drain: B %0d/%0d A %0d/%0d SW %0d release %0d",
           RD, b_got, b_sent, a_got, a_sent, sw_sent, release_reads);
  end
endmodule
