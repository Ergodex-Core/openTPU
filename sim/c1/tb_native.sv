module tb_native;
  logic clk=0;always #5 clk=~clk;
  logic rst=1,enable=0;
  logic [22:0] base=23'd64;
  logic [15:0] req_valid,req_ready,rsp_valid,rsp_ready;
  logic [15:0][1:0] req_slot,rsp_slot;
  logic [15:0][31:0] req_row,req_generation,rsp_row,rsp_generation;
  logic [15:0][7:0] req_kb;
  logic [15:0][9:0] rsp_group;
  logic [15:0][255:0] rsp_weights;
  logic [7:0] n_cvalid,n_cready,n_rvalid,n_err=0;
  logic [7:0][24:0] n_caddr;
  logic [7:0][511:0] n_rdata;
  logic drained,error;
  logic [7:0][31:0] pc_issued,pc_returned,pc_peak,pc_blocked,pc_credit;
  otpu_c1_native #(.RD(8)) dut(.*);
  int head[8],tail[8],seen[64],cmds[16];
  int addresses[8][4096],due[8][4096];
  int cycle=0,outputs=0,phase=0;
  logic [31:0] rng=32'h12345678;
  logic [7:0] stalled;
  logic [7:0][24:0] held_addr;
  function automatic logic [31:0] value(input int addr,wordno);
    return 32'(addr*16+wordno)^32'h83ac9217;
  endfunction
  always @(negedge clk) begin
    rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
    for(int e=0;e<16;e++)begin
      req_valid[e]=enable&&cmds[e]<4;
      req_row[e]=32'(e+cmds[e]*16);
      req_slot[e]=2'(cmds[e]);req_generation[e]=32'(phase+1);req_kb[e]=9;
      rsp_ready[e]=rng[e]||cycle%7==0;
    end
    for(int p=0;p<8;p++)begin
      n_cready[p]=rng[p+8]||cycle%3==0;
      n_rvalid[p]=head[p]<tail[p]&&cycle>=due[p][head[p]]&&(rng[p]||cycle%5==0);
      for(int w=0;w<16;w++)n_rdata[p][32*w+:32]=value(addresses[p][head[p]],w);
    end
  end
  always @(posedge clk)begin
    cycle++;
    if(!rst)begin
      if(error)$fatal(1,"unexpected error");
      for(int e=0;e<16;e++)begin
        if(req_valid[e]&&req_ready[e])cmds[e]++;
        if(rsp_valid[e]&&rsp_ready[e])begin
          int rowid,groupid,addr;
          rowid=int'(rsp_row[e]);groupid=int'(rsp_group[e]);
          if(rowid%16!=e||rsp_generation[e]!=32'(phase+1)||groupid!=seen[rowid]||rsp_slot[e]!=2'(rowid/16))$fatal(1,"bad response tag e%0d r%0d g%0d want%0d",e,rowid,groupid,seen[rowid]);
          addr=64+((rowid/16)*2+(rowid%2))*18+groupid/2;
          for(int w=0;w<8;w++)if(rsp_weights[e][32*w+:32]!=value(addr,w+(groupid%2)*8))$fatal(1,"bad response data");
          seen[rowid]++;outputs++;
        end
      end
      for(int p=0;p<8;p++)begin
        if(stalled[p]&&(!n_cvalid[p]||n_caddr[p]!=held_addr[p]))$fatal(1,"unstable native command");
        stalled[p]=n_cvalid[p]&&!n_cready[p];held_addr[p]=n_caddr[p];
        if(n_cvalid[p]&&n_cready[p])begin
          addresses[p][tail[p]]=int'(n_caddr[p]);due[p][tail[p]]=cycle+33+(p==0?97:0);tail[p]++;
        end
        if(n_rvalid[p])head[p]++;
      end
      if(outputs==64*36)begin
        if(!drained)$fatal(1,"unreturned reads after completion");
        for(int p=0;p<8;p++)if(pc_issued[p]!=144||pc_returned[p]!=144)$fatal(1,"incorrect counters");
        $display("C1_NATIVE_PASS phase=%0d cycles=%0d outputs=%0d",phase,cycle,outputs);
        if(phase==1)$finish;
        phase=1;rst=1;enable=0;
        for(int e=0;e<16;e++)cmds[e]=0;
        for(int r=0;r<64;r++)seen[r]=0;
        outputs=0;
      end
    end
    if(cycle>50000)$fatal(1,"native test timeout outputs=%0d",outputs);
  end
  initial begin
    repeat(3)@(negedge clk);rst=0;enable=1;
    wait(phase==1);repeat(3)@(negedge clk);rst=0;enable=1;
  end
endmodule
