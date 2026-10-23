module tb_c2_mxu;
  localparam int D=128,C=4,DEPTH=256,MAX_CMD=128;
  logic clk=0,rst=1;
  always #5 clk=~clk;
  logic cmd_valid,cmd_ready,cmd_decode,cmd_acc;
  logic [8:0] cmd_kb;
  logic [3:0] cmd_mask;
  logic [31:0] cmd_tag;
  logic [3:0][31:0] cmd_addr;
  logic [3:0] w_valid,w_ready,s_valid,s_ready,a_valid,a_ready;
  logic [3:0][1023:0] w_data,a_data;
  logic [3:0][7:0] s_dw;
  logic [3:0][31:0] s_sw,a_da,a_sa;
  logic old_req,old_ready,old_valid;
  logic [31:0] old_tag;
  logic [3:0] old_mask;
  logic [3:0][31:0] old_addr,old_data;
  logic out_valid,out_ready,out_decode,out_acc;
  logic [31:0] out_tag;
  logic [3:0] out_mask;
  logic [3:0][31:0] out_addr,out_data;
  logic [3:0][47:0] out_integer;
  logic pf_advance,pf_mode_wait,pf_credit_wait,pf_old_wait,pf_out_wait;
  logic [3:0] pf_w_wait,pf_s_wait,pf_a_wait;
  logic [15:0] pf_live,pf_results;
  logic [3:0][8:0] pf_w_level,pf_s_level,pf_a_level;
  otpu_c2_mxu dut(.*);
  int dec[MAX_CMD],ac[MAX_CMD],kb[MAX_CMD],mask[MAX_CMD],tag[MAX_CMD];
  logic [31:0] addr[MAX_CMD][C],sw[MAX_CMD][C],sa[MAX_CMD][C];
  logic [31:0] dw[MAX_CMD][C][DEPTH],da[MAX_CMD][C][DEPTH];
  logic [1023:0] weights[MAX_CMD][C][DEPTH],acts[MAX_CMD][C][DEPTH];
  logic [31:0] memory[65536];
  int nc,next_cmd=0,feed=-1,received=0,cy=0,feed_cycle=0;
  int wi[C],si[C],ai[C];
  int wstall=0,sstall=0,astall=0,ostall=0,rstall=0,hold_cycles=0,old_hold=0,delay_pc0=0;
  int seed=1,limit=1000000;
  int reset_at=0,reset_skip=0,reset_left=0;
  bit reset_happened=0;
  int hold_response=0;
  bit response_pending=0;
  logic [3:0][31:0] response_data;
  int input_fd,evfd,resfd;
  string input_path,events_path,result_path;
  function automatic logic [31:0] read32();
    logic [31:0] x;
    if($fread(x,input_fd)!=4) $fatal(1,"C2 truncated input");
    return x;
  endfunction
  initial begin
    if(!$value$plusargs("input=%s",input_path)) $fatal(1,"input required");
    if(!$value$plusargs("events=%s",events_path)) $fatal(1,"events required");
    if(!$value$plusargs("results=%s",result_path)) $fatal(1,"results required");
    void'($value$plusargs("wstall=%d",wstall));void'($value$plusargs("sstall=%d",sstall));
    void'($value$plusargs("astall=%d",astall));void'($value$plusargs("ostall=%d",ostall));
    void'($value$plusargs("rstall=%d",rstall));void'($value$plusargs("hold=%d",hold_cycles));
    void'($value$plusargs("old_hold=%d",old_hold));
    void'($value$plusargs("reset_at=%d",reset_at));void'($value$plusargs("reset_skip=%d",reset_skip));
    void'($value$plusargs("hold_response=%d",hold_response));
    void'($value$plusargs("delay_pc0=%d",delay_pc0));void'($value$plusargs("seed=%d",seed));
    void'($urandom(seed));
    input_fd=$fopen(input_path,"rb");evfd=$fopen(events_path,"w");resfd=$fopen(result_path,"w");
    if(input_fd==0 || evfd==0 || resfd==0) $fatal(1,"C2 input/output open failed");
    nc=int'(read32());
    if(nc<1 || nc>MAX_CMD) $fatal(1,"C2 command count");
    for(int n=0;n<nc;n++) begin
      dec[n]=int'(read32());ac[n]=int'(read32());kb[n]=int'(read32());
      mask[n]=int'(read32());tag[n]=int'(read32());
      for(int c=0;c<C;c++) begin
        addr[n][c]=read32();
        if(addr[n][c]>=65536) $fatal(1,"C2 TMEM address");
        memory[addr[n][c]]=read32();
      end
      for(int c=0;c<C;c++) begin
        sw[n][c]=read32();sa[n][c]=read32();
        for(int k=0;k<kb[n];k++) begin
          logic [1023:0] chunk;
          dw[n][c][k]=read32();da[n][c][k]=read32();
          if($fread(chunk,input_fd)!=128) $fatal(1,"C2 weight input");
          weights[n][c][k]=chunk;
          if($fread(chunk,input_fd)!=128) $fatal(1,"C2 activation input");
          acts[n][c][k]=chunk;
        end
      end
    end
    $fclose(input_fd);
    repeat(4) @(negedge clk);
    rst=0;
  end
  // Every source has an independently randomized delivery opportunity. Sources
  // only assert valid when the independently generated RTL ready is high; no
  // source changes a stalled transaction. There is no aggregate memory gate.
  always @(negedge clk) begin
    cmd_valid=0;w_valid=0;s_valid=0;a_valid=0;old_valid=0;out_ready=0;
    if(reset_at>0 && !reset_happened && cy>=reset_at) begin
      if(received!=0) $fatal(1,"C2 reset prelude unexpectedly committed");
      rst=1;reset_happened=1;reset_left=3;next_cmd=reset_skip;feed=-1;
      for(int c=0;c<C;c++) begin wi[c]=0;si[c]=0;ai[c]=0;end
    end else if(reset_left>0) begin
      reset_left=reset_left-1;
      if(reset_left==0) rst=0;
    end
    if(!rst) begin
      if(next_cmd<((reset_at>0 && !reset_happened) ? reset_skip:nc)) begin
        cmd_valid=1;cmd_decode=1'(dec[next_cmd]);cmd_acc=1'(ac[next_cmd]);
        cmd_kb=9'(kb[next_cmd]);cmd_mask=4'(mask[next_cmd]);cmd_tag=32'(tag[next_cmd]);
        for(int c=0;c<C;c++) cmd_addr[c]=addr[next_cmd][c];
      end
      if(feed>=0) for(int c=0;c<C;c++) begin
        if(w_ready[c] && $urandom_range(0,99)>=wstall && (c!=0 || cy-feed_cycle>=delay_pc0)) begin
          w_valid[c]=1;w_data[c]=weights[feed][c][wi[c]];
        end
        if(s_ready[c] && $urandom_range(0,99)>=sstall) begin
          s_valid[c]=1;s_dw[c]=8'(dw[feed][c][si[c]]);s_sw[c]=sw[feed][c];
        end
        if(a_ready[c] && $urandom_range(0,99)>=astall) begin
          a_valid[c]=1;a_data[c]=acts[feed][c][ai[c]];a_da[c]=da[feed][c][ai[c]];a_sa[c]=sa[feed][c];
        end
      end
      out_ready=cy>=hold_cycles && $urandom_range(0,99)>=ostall;
      // out_ready changed above in this procedural block; old_ready's continuous
      // assignment need not have reevaluated yet. Use its handshake predicate
      // explicitly, otherwise this source can pulse valid on a newly stalled edge.
      old_valid=old_req && (!out_valid || out_ready) && cy>=old_hold && $urandom_range(0,99)>=rstall;
      for(int c=0;c<C;c++) old_data[c]=memory[old_addr[c][15:0]];
      if(hold_response!=0) begin
        if(!response_pending && old_req && cy>=old_hold && $urandom_range(0,99)>=rstall) begin
          response_pending=1;
          for(int c=0;c<C;c++) response_data[c]=memory[old_addr[c][15:0]];
        end
        old_valid=response_pending;old_data=response_data;
      end
    end
  end
  logic stalled=0;
  logic old_stalled=0;
  logic response_stalled=0;
  logic [127:0] response_held;
  logic [C*32+32+4-1:0] old_held;
  logic [C*(32+32+48)+32+4+2-1:0] held;
  always @(posedge clk) if(rst) begin
    stalled=0;old_stalled=0;response_stalled=0;response_pending=0;
    if(reset_happened) cy=cy+1;
  end
  always @(posedge clk) if(!rst) begin
    cy=cy+1;
    if(cy>limit) $fatal(1,"C2 simulation timeout");
    if(stalled && (!out_valid || held!={out_decode,out_acc,out_tag,out_mask,out_addr,out_data,out_integer}))
      $fatal(1,"C2 output changed under backpressure");
    stalled=out_valid && !out_ready;
    held={out_decode,out_acc,out_tag,out_mask,out_addr,out_data,out_integer};
    if(old_stalled && (!old_req || old_held!={old_tag,old_mask,old_addr}))
      $fatal(1,"C2 ACC read request changed before response");
    old_stalled=old_req && !(old_valid && old_ready);
    old_held={old_tag,old_mask,old_addr};
    if(response_stalled && (!old_valid || old_data!=response_held))
      $fatal(1,"C2 ACC response source violated backpressure");
    response_stalled=old_valid && !old_ready;response_held=old_data;
    if(old_valid && old_ready) response_pending=0;
    $fdisplay(evfd,"%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
      cy,cmd_valid&&cmd_ready,cmd_tag,pf_advance,dut.command.tag,w_valid&w_ready,s_valid&s_ready,
      a_valid&a_ready,out_valid&&out_ready,out_tag,pf_old_wait,pf_out_wait,pf_mode_wait,pf_credit_wait,
      pf_w_wait,pf_s_wait,pf_a_wait,pf_live,pf_results,
      (pf_w_level[0]>pf_w_level[1] ? pf_w_level[0]:pf_w_level[1]),
      (pf_w_level[2]>pf_w_level[3] ? pf_w_level[2]:pf_w_level[3]),
      dut.fill_bank,dut.compute_bank,cmd_valid && !dut.filling && dut.occupied[dut.fill_bank],
      old_valid && !old_ready,dut.read_advance,dut.command_bank[dut.compute_bank].tag,dut.fetch_bank);
    if(cmd_valid && cmd_ready) begin
      feed=next_cmd;next_cmd=next_cmd+1;feed_cycle=cy;
      for(int c=0;c<C;c++) begin wi[c]=0;si[c]=0;ai[c]=0;end
    end
    for(int c=0;c<C;c++) begin
      if(w_valid[c]&&w_ready[c]) wi[c]=wi[c]+1;
      if(s_valid[c]&&s_ready[c]) si[c]=si[c]+1;
      if(a_valid[c]&&a_ready[c]) ai[c]=ai[c]+1;
    end
    if(out_valid && out_ready) begin
      $fwrite(resfd,"%0d %0d %0d %0d %0d",cy,out_tag,out_decode,out_acc,out_mask);
      for(int c=0;c<C;c++) begin
        $fwrite(resfd," %0d %08x %012x",out_addr[c],out_data[c],out_integer[c]);
        if(out_mask[c]) memory[out_addr[c][15:0]]=out_data[c];
      end
      $fwrite(resfd,"\n");received=received+1;
      if(received==(nc-(reset_happened ? reset_skip:0))) begin
        $fclose(evfd);$fclose(resfd);
        $display("C2 RTL completed %0d groups in %0d cycles",received,cy);
        $finish;
      end
    end
  end
endmodule
