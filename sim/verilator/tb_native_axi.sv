`timescale 1ns/1ps
// otpu_native_axi alone (tests/test_f2_rtl.py::test_native_axi), ported to Verilator from Alex's
// burst200 branch (ergodex-openTPU alex-burst200, commit 06c169cb, tests/f2/tb_native_axi.sv: an
// AXI target with independently random readies and in-order read responses; data before / after
// / with commands; credits and the queue saturated while responses are held; 4 KiB and window
// boundaries; isolated reads flushed by the idle timer; injected RRESP and RLAST errors; reset).
// Added here: XL > 0 puts the native side behind XL / 2 register stages each way (the F2 HBM
// build's SLR chain: the master sees n_cready late, its take lands late, nothing may be lost:
// otpu_native_axi XL), a stream phase on an always-ready target (a command taken every cycle:
// the AR register), the burst lengths' histogram, and CH_LOG2 29 (a 512 MiB window at
// BASE 3 x 512 MiB: a command outside it raises n_err).
module tb_native_axi #(parameter BURST=16, OUT=32, XL=0, CH_LOG2=31, IDLE=4, WAIT=1);
  localparam int D = XL / 2;
  localparam logic [33:0] BASE = CH_LOG2 < 31 ? 34'(3) << CH_LOG2 : 34'd0;
  logic clk=0; always #5 clk=~clk;
  logic rst=1;
  logic t_cvalid=0,t_cready,t_cwe=0; logic[24:0] t_caddr=0;
  logic t_wvalid=0,t_wready; logic[511:0] t_wdata=0; logic[63:0] t_wmask=0;
  logic n_cvalid,n_cready,n_cwe; logic[24:0] n_caddr;
  logic n_wvalid,n_wready; logic[511:0] n_wdata; logic[63:0] n_wmask;
  logic n_rvalid; logic[511:0] n_rdata; logic[15:0] n_wdone; logic n_err;
  logic [33:0] m_awaddr,m_araddr; logic m_awvalid,m_awready=0,m_arvalid,m_arready=0;
  logic [7:0] m_arlen;
  logic [511:0] m_wdata,m_rdata=0; logic[63:0] m_wstrb;
  logic m_wvalid,m_wready=0,m_bvalid=0,m_bready,m_rvalid=0,m_rready,m_rlast=1;
  logic [1:0] m_bresp=0,m_rresp=0;
  otpu_native_axi #(.QD(XL ? 16 : 4),.XL(XL),.MAX_OUT(OUT),.READ_BURST(BURST),.READ_IDLE(IDLE),.READ_WAIT(WAIT),
                    .CH_LOG2(CH_LOG2),.BASE_ADDR(BASE)) dut(.*);
  // the native side: wired, or D stages each way (ready up, the take and its payload down)
  if (D == 0) begin : g_wire
    assign n_cvalid=t_cvalid; assign n_cwe=t_cwe; assign n_caddr=t_caddr; assign t_cready=n_cready;
    assign n_wvalid=t_wvalid; assign n_wdata=t_wdata; assign n_wmask=t_wmask; assign t_wready=n_wready;
  end else begin : g_xr
    logic [D-1:0] cr_d, wr_d;
    logic [D-1:0][26:0] c_d;
    logic [D-1:0][576:0] w_d;
    assign t_cready = cr_d[D-1]; assign t_wready = wr_d[D-1];
    assign {n_cvalid,n_cwe,n_caddr} = c_d[D-1];
    assign {n_wvalid,n_wmask,n_wdata} = w_d[D-1];
    always @(posedge clk) begin
      cr_d <= {cr_d[D-2:0], n_cready}; wr_d <= {wr_d[D-2:0], n_wready};
      c_d <= {c_d[D-2:0], {t_cvalid&&t_cready, t_cwe, t_caddr}};
      w_d <= {w_d[D-2:0], {t_wvalid&&t_wready, t_wmask, t_wdata}};
      // (n_cready is the threshold QD - XL; a take in flight lands in the slots kept below it)
      if (!rst && n_cvalid && dut.cn == 16 && !dut.popc) $fatal(1,"a command landed on a full queue");
      if (!rst && n_wvalid && dut.wn == 16 && !dut.popw) $fatal(1,"write data landed on a full queue");
      if (!rst && (dut.cn > 16 || dut.wn > 16)) $fatal(1,"queue overflow");
    end
  end
  integer hist[1:16];
  initial for (integer i=1;i<=16;i++) hist[i]=0;
  logic [511:0] mem[0:255], expected[0:4095], aw_data, w_data;
  logic [63:0] w_mask;
  integer aw_addr, read_addr[0:4095], ar_head=0,ar_tail=0;
  bit read_last[0:4095];
  integer max_burst=0, bursts=0;
  logic ar_stalled=0; logic [33:0] prev_araddr; logic [7:0] prev_arlen;
  integer expected_head=0,expected_tail=0, count_read=0, count_write=0;
  integer b_pending=0; bit aw_held=0,w_held=0;
  integer seed=42, dummy, cycles=0;
  bit allow_error=0, inject_error=0, inject_badlast=0, hold_reads=0, fast=0;
  // Randomly and independently stalled AXI target, with ordered read responses.
  always @(negedge clk) begin
    if (rst) begin m_awready=0;m_wready=0;m_arready=0;m_bvalid=0;m_rvalid=0; end
    else begin
      m_awready=!aw_held && (fast || $urandom_range(0,3)!=0);
      m_wready=!w_held && (fast || $urandom_range(0,3)!=0);
      m_arready=fast || $urandom_range(0,3)!=0;
      m_bvalid=b_pending>0 && (fast || $urandom_range(0,3)==0);
      m_rvalid=!hold_reads && ar_head<ar_tail && (fast || $urandom_range(0,3)!=0);
      m_rresp=inject_error ? 2'b10 : 2'b00;
      if (m_rvalid) begin m_rdata=mem[read_addr[ar_head]];m_rlast=read_last[ar_head]^inject_badlast;end
    end
  end
  always @(posedge clk) if (!rst) begin
    if(ar_stalled && (!m_arvalid || m_araddr!==prev_araddr || m_arlen!==prev_arlen)) $fatal(1,"AR changed while stalled");
    ar_stalled=m_arvalid&&!m_arready;prev_araddr=m_araddr;prev_arlen=m_arlen;
    if(dut.reads>OUT || dut.ln>OUT) $fatal(1,"credit overflow");
    cycles=cycles+1;
    if (cycles>60000) $fatal(1,"timeout");
    if (m_awvalid&&m_awready) begin
      if (ar_head!=ar_tail) $fatal(1,"write overtook an earlier read");
      if ((m_awaddr & ~((34'd1<<CH_LOG2)-1)) != BASE) $fatal(1,"write outside the window");
      aw_addr=(m_awaddr>>6)&255; aw_held=1;
    end
    if (m_wvalid&&m_wready) begin w_data=m_wdata;w_mask=m_wstrb;w_held=1;end
    if (m_bvalid&&m_bready) begin b_pending=b_pending-1;count_write=count_write+1;end
    if (aw_held&&w_held) begin
      for(integer b=0;b<64;b=b+1) if(w_mask[b]) mem[aw_addr][8*b+:8]=w_data[8*b+:8];
      b_pending=b_pending+1;aw_held=0;w_held=0;
    end
    if (m_arvalid&&m_arready) begin
      if(b_pending!=0 || aw_held || w_held) $fatal(1,"read overtook uncompleted write");
      if(m_arlen>=BURST || m_araddr[11:0]+(m_arlen+1)*64>4096) $fatal(1,"invalid burst/page crossing");
      if ((m_araddr & ~((34'd1<<CH_LOG2)-1)) != BASE) $fatal(1,"read outside the window");
      bursts=bursts+1;hist[m_arlen+1]++;if(m_arlen+1>max_burst) max_burst=m_arlen+1;
      for(integer j=0;j<=m_arlen;j=j+1) begin
        read_addr[ar_tail]=((m_araddr>>6)+j)&255;
        read_last[ar_tail]=(j==m_arlen);ar_tail=ar_tail+1;
      end
    end
    if(m_rvalid&&m_rready) ar_head=ar_head+1;
    if(n_rvalid) begin
      if(expected_head>=expected_tail) $fatal(1,"unexpected response");
      if(n_rdata!==expected[expected_head]) $fatal(1,"read mismatch %0d",expected_head);
      expected_head=expected_head+1;count_read=count_read+1;
    end
    if(n_err && !allow_error) $fatal(1,"adapter protocol error");
  end
  task cmd(input bit we,input integer addr);
    @(negedge clk); t_cvalid=1;t_cwe=we;t_caddr=addr;
    do @(posedge clk); while(!t_cready);
    @(negedge clk);t_cvalid=0;
  endtask
  task data(input logic[511:0] value,input logic[63:0] mask);
    @(negedge clk);t_wvalid=1;t_wdata=value;t_wmask=mask;
    do @(posedge clk); while(!t_wready);
    @(negedge clk);t_wvalid=0;
  endtask
  task read_cmd(input integer addr,input logic[511:0] value);
    expected[expected_tail]=value; expected_tail=expected_tail+1;cmd(0,addr);
  endtask
  logic[511:0] gold[0:255],val;
  logic[63:0] mask;
  integer addr, nw=0;
  initial begin
    dummy=$value$plusargs("seed=%d",seed);
    dummy=$urandom(seed);
    for(integer a=0;a<256;a=a+1) begin mem[a]=0;gold[a]=0;end
    repeat(6) @(negedge clk);rst=0;
    for(integer i=0;i<400;i=i+1) begin
      addr=$urandom_range(0,63);
      if(i%3!=2) begin
        for(integer w=0;w<16;w=w+1) val[32*w+:32]=$urandom;
        mask={$urandom,$urandom};
        // Exercise data before command, command before data, simultaneous streams.
        case(i%3)
          0: begin data(val,mask);cmd(1,addr);end
          1: begin cmd(1,addr);data(val,mask);end
        endcase
        for(integer b=0;b<64;b=b+1) if(mask[b]) gold[addr][8*b+:8]=val[8*b+:8];
        nw=nw+1;
      end else read_cmd(addr,gold[addr]);
    end
    for(integer j=0;j<64;j=j+1) read_cmd(j,gold[j]);
    wait(expected_head==expected_tail && n_wdone==nw);
    // Saturate read credits and command queue while the target holds responses.
    hold_reads=1;
    fork
      begin repeat(80) @(negedge clk);hold_reads=0;end
      begin
        @(negedge clk);t_cvalid=1;t_cwe=0;
        for(integer j=0;j<64;j=j+1) begin
          t_caddr=j;expected[expected_tail]=gold[j];expected_tail=expected_tail+1;
          do @(posedge clk); while(!t_cready);
          @(negedge clk);
        end
        t_cvalid=0;
      end
    join
    wait(expected_head==expected_tail);
    // Cross 4 KiB and native-window boundaries; no burst may wrap either.
    for(integer j=60;j<85;j=j+1) read_cmd(j,gold[j]);
    if (CH_LOG2 == 31) for(integer j=33554428;j<33554432;j=j+1) read_cmd(j,gold[j&255]);
    else for(integer j=(1<<(CH_LOG2-6))-4;j<(1<<(CH_LOG2-6));j=j+1) read_cmd(j,gold[j&255]);
    read_cmd(0,gold[0]);read_cmd(19,gold[19]);read_cmd(100,gold[100]);
    wait(expected_head==expected_tail);
    repeat(3) @(negedge clk);
    read_cmd(130,gold[130]); // isolated command must flush by idle timeout
    wait(expected_head==expected_tail);
    if(max_burst != (OUT<BURST ? OUT:BURST)) $fatal(1,"full burst not exercised: %0d",max_burst);
    // a stream on an always-ready target: the queue takes a command every cycle (bursts leave
    // through the AR register while the next one gathers); count the takes' cycles
    fast=1;
    begin
      integer t0, t1, nst;
      nst = OUT < 64 ? OUT : 64;
      repeat(4) @(negedge clk);
      t_cvalid=1;t_cwe=0;t0=-1;
      for(integer j=0;j<512;j=j+1) begin
        t_caddr=j&255;expected[expected_tail]=gold[j&255];expected_tail=expected_tail+1;
        do @(posedge clk); while(!t_cready);
        if (t0 < 0) t0=cycles;
        @(negedge clk);
      end
      t1=cycles;t_cvalid=0;
      wait(expected_head==expected_tail);
      $display("stream: 512 reads taken in %0d cycles", t1-t0+1);
      if (OUT >= 128 && BURST > 1 && t1-t0+1 > 512+8) $fatal(1,"stream slower than a command a cycle: %0d", t1-t0+1);
    end
    fast=0;
    if (CH_LOG2 < 31) begin
      allow_error=1;
      read_cmd(1<<(CH_LOG2-6), gold[0]);     // masked into the window (beat 0), flagged
      wait(expected_head==expected_tail);
      repeat(2) @(negedge clk);
      if (!n_err) $fatal(1,"a command outside the window did not raise n_err");
      rst=1;repeat(4) @(negedge clk);rst=0;repeat(3) @(negedge clk);
      allow_error=0;
    end
    repeat(3) @(negedge clk);
    allow_error=1;inject_error=1;
    read_cmd(0,gold[0]);
    wait(n_err);
    inject_error=0;
    repeat(3) @(negedge clk);
    rst=1;
    repeat(4) @(negedge clk);
    rst=0;
    repeat(3) @(negedge clk);
    if(n_err || n_wdone!=0) $fatal(1,"reset did not clear status");
    // Inject wrong RLAST and check the independent burst-length scoreboard.
    inject_badlast=1;
    read_cmd(0,gold[0]);read_cmd(1,gold[1]);
    wait(expected_head==expected_tail);repeat(2) @(negedge clk);
    if(!n_err) $fatal(1,"bad RLAST not detected");
    inject_badlast=0;rst=1;repeat(4) @(negedge clk);rst=0;
    repeat(3) @(negedge clk);
    if(n_err) $fatal(1,"reset did not clear RLAST error");
    $display("burst limit %0d, credits %0d, largest burst %0d, transactions %0d",BURST,OUT,max_burst,bursts);
    $write("burst lengths:");
    for (integer i=1;i<=16;i++) if (hist[i]) $write(" %0d:%0d", i, hist[i]);
    $display("");
    $display("PASS native AXI: %0d reads, %0d masked writes; randomized stalls and direction barriers",count_read,count_write);
    $finish;
  end
endmodule
