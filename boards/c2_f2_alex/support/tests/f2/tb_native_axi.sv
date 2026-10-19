`timescale 1ns/1ps
module tb_native_axi #(parameter BURST=16, OUT=32);
  logic clk=0; always #5 clk=~clk;
  logic rst=1;
  logic n_cvalid=0,n_cready,n_cwe=0; logic[24:0] n_caddr=0;
  logic n_wvalid=0,n_wready; logic[511:0] n_wdata=0; logic[63:0] n_wmask=0;
  logic n_rvalid; logic[511:0] n_rdata; logic[15:0] n_wdone; logic n_err;
  logic [33:0] m_awaddr,m_araddr; logic m_awvalid,m_awready=0,m_arvalid,m_arready=0;
  logic [7:0] m_arlen;
  logic [511:0] m_wdata,m_rdata=0; logic[63:0] m_wstrb;
  logic m_wvalid,m_wready=0,m_bvalid=0,m_bready,m_rvalid=0,m_rready,m_rlast=1;
  logic [1:0] m_bresp=0,m_rresp=0;
  otpu_native_axi #(.QD(4),.MAX_OUT(OUT),.READ_BURST(BURST)) dut(.*);
  logic [511:0] mem[0:255], expected[0:2047], aw_data, w_data;
  logic [63:0] w_mask;
  integer aw_addr, read_addr[0:2047], ar_head=0,ar_tail=0;
  bit read_last[0:2047];
  integer max_burst=0, bursts=0;
  logic ar_stalled=0; logic [33:0] prev_araddr; logic [7:0] prev_arlen;
  integer expected_head=0,expected_tail=0, count_read=0, count_write=0;
  integer b_pending=0; bit aw_held=0,w_held=0;
  integer seed=42, dummy, cycles=0;
  bit allow_error=0, inject_error=0, inject_badlast=0, hold_reads=0;
  // Randomly and independently stalled AXI target, with ordered read responses.
  always @(negedge clk) begin
    if (rst) begin m_awready=0;m_wready=0;m_arready=0;m_bvalid=0;m_rvalid=0; end
    else begin
      m_awready=!aw_held && ($urandom_range(0,3)!=0);
      m_wready=!w_held && ($urandom_range(0,3)!=0);
      m_arready=$urandom_range(0,3)!=0;
      m_bvalid=b_pending>0 && $urandom_range(0,3)==0;
      m_rvalid=!hold_reads && ar_head<ar_tail && $urandom_range(0,3)!=0;
      m_rresp=inject_error ? 2'b10 : 2'b00;
      if (m_rvalid) begin m_rdata=mem[read_addr[ar_head]];m_rlast=read_last[ar_head]^inject_badlast;end
    end
  end
  always @(posedge clk) if (!rst) begin
    if(ar_stalled && (!m_arvalid || m_araddr!==prev_araddr || m_arlen!==prev_arlen)) $fatal(1,"AR changed while stalled");
    ar_stalled=m_arvalid&&!m_arready;prev_araddr=m_araddr;prev_arlen=m_arlen;
    if(dut.reads>OUT || dut.ln>OUT) $fatal(1,"credit overflow");
    cycles=cycles+1;
    if (cycles>30000) $fatal(1,"timeout");
    if (m_awvalid&&m_awready) begin
      if (ar_head!=ar_tail) $fatal(1,"write overtook an earlier read");
      aw_addr=m_awaddr>>6; aw_held=1;
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
      bursts=bursts+1;if(m_arlen+1>max_burst) max_burst=m_arlen+1;
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
    @(negedge clk); n_cvalid=1;n_cwe=we;n_caddr=addr;
    do @(posedge clk); while(!n_cready);
    @(negedge clk);n_cvalid=0;
  endtask
  task data(input logic[511:0] value,input logic[63:0] mask);
    @(negedge clk);n_wvalid=1;n_wdata=value;n_wmask=mask;
    do @(posedge clk); while(!n_wready);
    @(negedge clk);n_wvalid=0;
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
        @(negedge clk);n_cvalid=1;n_cwe=0;
        for(integer j=0;j<64;j=j+1) begin
          n_caddr=j;expected[expected_tail]=gold[j];expected_tail=expected_tail+1;
          do @(posedge clk); while(!n_cready);
          @(negedge clk);
        end
        n_cvalid=0;
      end
    join
    wait(expected_head==expected_tail);
    // Cross 4 KiB and native-window boundaries; no burst may wrap either.
    for(integer j=60;j<85;j=j+1) read_cmd(j,gold[j]);
    for(integer j=33554428;j<33554432;j=j+1) read_cmd(j,gold[j&255]);
    read_cmd(0,gold[0]);read_cmd(19,gold[19]);read_cmd(100,gold[100]);
    wait(expected_head==expected_tail);
    repeat(3) @(negedge clk);
    read_cmd(130,gold[130]); // isolated command must flush by idle timeout
    wait(expected_head==expected_tail);
    if(max_burst != (OUT<BURST ? OUT:BURST)) $fatal(1,"full burst not exercised: %0d",max_burst);
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
    $display("PASS native AXI: %0d reads, %0d masked writes; randomized stalls and direction barriers",count_read,count_write);
    $finish;
  end
endmodule
