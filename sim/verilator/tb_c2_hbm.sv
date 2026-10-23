// Native-interface simulation only: randomized in-order per-PC response service.
// This does not model AXI HBM timing, the F2 shell, or FPGA clock closure.
module tb_c2_hbm;
  logic clk=0,rst=1,hbm_ready=1;
  always #5 clk=~clk;
  logic [2:0] led;
  logic [11:0] s_ctl_awaddr=0,s_ctl_araddr=0;
  logic s_ctl_awvalid=0,s_ctl_awready,s_ctl_wvalid=0,s_ctl_wready;
  logic [31:0] s_ctl_wdata=0,s_ctl_rdata;
  logic [3:0] s_ctl_wstrb=15;
  logic [1:0] s_ctl_bresp,s_ctl_rresp;
  logic s_ctl_bvalid,s_ctl_bready=0,s_ctl_arvalid=0,s_ctl_arready,s_ctl_rvalid,s_ctl_rready=0;
  logic [7:0] n_cvalid,n_cready=0,n_cwe,n_wvalid,n_wready='1,n_rvalid=0,n_err=0;
  logic [7:0][24:0] n_caddr;
  logic [7:0][511:0] n_wdata,n_rdata=0;
  logic [7:0][63:0] n_wmask;
  logic [7:0][15:0] n_wdone=0;
  otpu_c2_hbm_kernel #(.BUILD_ID(32'hc2abcdef)) dut(.*);
  logic [511:0] memory[8][32768];
  int qaddr[8][64],qdue[8][64],head[8],tail[8],pending[8],accepted[8],replied[8];
  int cy=0,seed=1,latency=30,stall=20,delay_pc0=0,fault=0,mode,rows,kb,base,beats,nops;
  int fd,evfd,resfd,run_no=0,start_cycle=0,total_pending=0;
  bit injected=0;
  string input_path,events_path,result_path;
  function automatic logic[31:0] read32();
    logic[31:0] v;
    if($fread(v,fd)!=4)$fatal(1,"truncated input");
    return v;
  endfunction
  task automatic write_reg(input logic[11:0] addr,input logic[31:0] value,
                            input logic[1:0] expected=0,input logic[3:0] strobe=15);
    bit first_aw;
    logic[1:0] response;
    first_aw=$urandom_range(0,1)!=0;
    @(negedge clk);s_ctl_awaddr=addr;s_ctl_wdata=value;s_ctl_wstrb=strobe;
    if(first_aw)begin
      s_ctl_awvalid=1;do @(posedge clk);while(!s_ctl_awready);
      @(negedge clk);s_ctl_awvalid=0;
      repeat($urandom_range(0,2))@(negedge clk);
      s_ctl_wvalid=1;do @(posedge clk);while(!s_ctl_wready);
      @(negedge clk);s_ctl_wvalid=0;
    end else begin
      s_ctl_wvalid=1;do @(posedge clk);while(!s_ctl_wready);
      @(negedge clk);s_ctl_wvalid=0;
      repeat($urandom_range(0,2))@(negedge clk);
      s_ctl_awvalid=1;do @(posedge clk);while(!s_ctl_awready);
      @(negedge clk);s_ctl_awvalid=0;
    end
    do @(posedge clk);while(!s_ctl_bvalid);
    response=s_ctl_bresp;
    repeat($urandom_range(1,4))begin
      @(posedge clk);if(!s_ctl_bvalid||s_ctl_bresp!=response)$fatal(1,"unstable B response");
    end
    if(response!=expected)$fatal(1,"write %h response %d expected %d",addr,response,expected);
    @(negedge clk);s_ctl_bready=1;@(posedge clk);@(negedge clk);s_ctl_bready=0;
  endtask
  task automatic read_reg(input logic[11:0] addr,output logic[31:0] value,input logic[1:0] expected=0);
    logic[1:0] response;
    @(negedge clk);s_ctl_araddr=addr;s_ctl_arvalid=1;
    do @(posedge clk);while(!s_ctl_arready);
    // ARADDR is not owned by the slave after acceptance. Poison it immediately
    // so every CSR/result read checks the accepted-address register.
    @(negedge clk);s_ctl_arvalid=0;s_ctl_araddr=12'hfff;
    do @(posedge clk);while(!s_ctl_rvalid);
    value=s_ctl_rdata;response=s_ctl_rresp;
    repeat($urandom_range(1,4))begin
      @(posedge clk);if(!s_ctl_rvalid||s_ctl_rdata!=value||s_ctl_rresp!=response)$fatal(1,"unstable R response");
    end
    if(response!=expected)$fatal(1,"read %h response %d expected %d",addr,response,expected);
    @(negedge clk);s_ctl_rready=1;@(posedge clk);@(negedge clk);s_ctl_rready=0;
  endtask
  task automatic read_blocks_write(input logic[11:0] addr,input logic[31:0] expected_value);
    logic[31:0] value;
    // A following MEM_ADDR write must remain unaccepted during address capture,
    // decode, result fetch, and a stalled R response.
    @(negedge clk);s_ctl_araddr=addr;s_ctl_arvalid=1;
    do @(posedge clk);while(!s_ctl_arready);
    @(negedge clk);s_ctl_arvalid=0;s_ctl_araddr=12'hfff;
    s_ctl_awaddr=12'h024;s_ctl_awvalid=1;s_ctl_wdata=32'h12345678;
    s_ctl_wstrb=15;s_ctl_wvalid=1;
    do begin
      @(posedge clk);
      if(s_ctl_awready||s_ctl_wready||s_ctl_arready)$fatal(1,"write/read accepted before R response");
    end while(!s_ctl_rvalid);
    value=s_ctl_rdata;
    if(value!=expected_value||s_ctl_rresp!=0)$fatal(1,"accepted AR address/data changed");
    repeat(5)begin
      @(posedge clk);
      if(!s_ctl_rvalid||s_ctl_rdata!=value||s_ctl_rresp!=0)$fatal(1,"held R response changed");
      if(s_ctl_awready||s_ctl_wready||s_ctl_arready)$fatal(1,"write/read accepted while R stalled");
    end
    @(negedge clk);s_ctl_rready=1;
    @(posedge clk);
    if(s_ctl_awready||s_ctl_wready)$fatal(1,"write overlapped R acknowledgement");
    @(negedge clk);s_ctl_rready=0;
    do @(posedge clk);while(!(s_ctl_awready&&s_ctl_wready));
    @(negedge clk);s_ctl_awvalid=0;s_ctl_wvalid=0;
    do @(posedge clk);while(!s_ctl_bvalid);
    if(s_ctl_bresp!=0)$fatal(1,"queued write failed");
    @(negedge clk);s_ctl_bready=1;@(posedge clk);@(negedge clk);s_ctl_bready=0;
    read_reg(12'h024,value);
    if(value!=32'h12345678)$fatal(1,"queued write lost or reordered");
  endtask
  task automatic reset_read(input logic[11:0] addr,input int elapsed);
    logic[31:0] value;
    write_reg(12'h024,32'h701ff);
    @(negedge clk);s_ctl_araddr=addr;s_ctl_arvalid=1;
    do @(posedge clk);while(!s_ctl_arready);
    @(negedge clk);s_ctl_arvalid=0;s_ctl_araddr=12'hfff;
    // elapsed 0: accepted address; 1: decode/RAM fetch; 2: held response.
    repeat(elapsed)@(negedge clk);
    rst=1;
    repeat(2)@(negedge clk);
    rst=0;
    repeat(5)begin
      @(posedge clk);
      if(s_ctl_rvalid||s_ctl_bvalid)$fatal(1,"reset leaked a stale response");
    end
    read_reg(12'h024,value);
    if(value!=0)$fatal(1,"reset leaked a read-side increment");
    read_reg(12'h000,value);
    if(value!=32'h43324d42)$fatal(1,"read pipeline did not recover after reset");
  endtask
  task automatic result_boundary(input logic[31:0] window,input logic[31:0] expected_value);
    logic[31:0] value;
    write_reg(12'h024,window+511);
    read_reg(12'h02c,value);
    if(value!=expected_value)$fatal(1,"last result-window word selected wrong bank/kind");
    read_reg(12'h024,value);
    if(value!=window+512)$fatal(1,"result read did not increment exactly once");
    repeat(2)begin
      read_reg(12'h02c,value,2);
      if(value!=0)$fatal(1,"invalid result read returned stale data");
      read_reg(12'h024,value);
      if(value!=window+512)$fatal(1,"invalid boundary read advanced MEM_ADDR");
    end
    // The queued write may not change the word selected by an accepted read.
    write_reg(12'h024,window+511);
    read_blocks_write(12'h02c,expected_value);
  endtask
  task automatic wait_done(output logic[31:0] status);
    do begin
      read_reg(12'h00c,status);
      if(cy-start_cycle>1000000)$fatal(1,"kernel timeout");
    end while(!(status&2));
    if(status&17)$fatal(1,"DONE while still busy/draining");
    if(total_pending!=0)$fatal(1,"DONE with owned native returns pending %0d",total_pending);
  endtask
  task automatic configure();
    write_reg(12'h010,4);write_reg(12'h014,32'(mode));write_reg(12'h018,32'(kb));
    write_reg(12'h01c,32'(rows));write_reg(12'h020,32'(base));
  endtask
  initial begin:driver
    logic[31:0] v,status,bits,lo,hi,before_cycles;
    logic[511:0] chunk;
    if(!$value$plusargs("input=%s",input_path)||!$value$plusargs("events=%s",events_path)||
       !$value$plusargs("results=%s",result_path))$fatal(1,"paths required");
    void'($value$plusargs("seed=%d",seed));void'($value$plusargs("latency=%d",latency));
    void'($value$plusargs("stall=%d",stall));void'($value$plusargs("delay_pc0=%d",delay_pc0));
    void'($value$plusargs("fault=%d",fault));void'($urandom(seed));
    fd=$fopen(input_path,"rb");evfd=$fopen(events_path,"w");resfd=$fopen(result_path,"w");
    if(!fd||!evfd||!resfd)$fatal(1,"file open");
    rows=int'(read32());kb=int'(read32());mode=int'(read32());base=int'(read32());
    beats=int'(read32());nops=int'(read32());
    for(int p=0;p<8;p++)for(int k=0;k<beats;k++)begin
      if($fread(chunk,fd)!=64)$fatal(1,"truncated native image");memory[p][k]=chunk;
    end
    repeat(4)@(negedge clk);rst=0;
    read_reg(0,v);if(v!=32'h43324d42)$fatal(1,"identity");
    read_blocks_write(12'h000,32'h43324d42);
    for(int elapsed=0;elapsed<3;elapsed++)begin
      reset_read(12'h000,elapsed);
      reset_read(12'h02c,elapsed);
    end
    // ABI rejection and autoincrement checks before loading valid data.
    write_reg(12'h024,32'hdeadbeef);write_reg(12'h028,42,2);
    read_reg(12'h024,v);if(v!=32'hdeadbeef)$fatal(1,"bad write advanced MEM_ADDR");
    read_reg(12'h02c,v,2);read_reg(12'h024,v);if(v!=32'hdeadbeef)$fatal(1,"bad read advanced MEM_ADDR");
    write_reg(12'h014,0,2,4'b0011);
    for(int i=0;i<nops;i++)begin
      logic[31:0] addr,val;
      addr=read32();val=read32();write_reg(addr[11:0],val);
    end
    $fclose(fd);
    for(int bad=0;bad<7;bad++)begin
      configure();
      case(bad)
        0:write_reg(12'h018,0);
        1:write_reg(12'h018,256);
        2:write_reg(12'h01c,0);
        3:write_reg(12'h01c,513);
        4:write_reg(12'h014,2);
        5:write_reg(12'h020,32'h800000);
        6:hbm_ready=0;
      endcase
      write_reg(12'h010,1);wait_done(status);
      read_reg(12'h030,v);if(v!=1)$fatal(1,"invalid config not rejected %0d",bad);
      hbm_ready=1;
    end
    configure();
    if(fault!=0)begin
      start_cycle=cy;write_reg(12'h010,1);
      // No configuration or setup memory writes are legal during a run.
      write_reg(12'h018,1,2);write_reg(12'h024,0);write_reg(12'h028,123,2);
      if(fault==3)write_reg(12'h010,2);
      wait_done(status);read_reg(12'h030,v);
      if(fault==3)begin if(!(status&8)||v!=0)$fatal(1,"abort status");end
      else if(!(status&4)||v!=(fault==1?4:2))$fatal(1,"fault status %h error %h",status,v);
      $fwrite(resfd,"F %0d %0d %0d\n",fault,status,v);
      fault=0;run_no=1;configure();
    end
    start_cycle=cy;write_reg(12'h010,1);wait_done(status);
    if(status!=2)$fatal(1,"success status %h",status);
    for(int addr='h030;addr<='h07c;addr+=4)begin
      read_reg(12'(addr),v);$fwrite(resfd,"C %0d %0d\n",addr,v);
    end
    for(int p=0;p<8;p++)for(int off=0;off<=20;off+=4)begin
      read_reg(12'(256+p*32+off),v);$fwrite(resfd,"C %0d %0d\n",256+p*32+off,v);
    end
    read_reg(12'h038,before_cycles);
    for(int r=0;r<rows;r++)begin
      write_reg(12'h024,32'h50000+32'(r));read_reg(12'h02c,bits);
      read_reg(12'h024,v);if(v!=32'h50001+32'(r))$fatal(1,"read autoincrement");
      write_reg(12'h024,32'h60000+32'(r));read_reg(12'h02c,lo);
      write_reg(12'h024,32'h70000+32'(r));read_reg(12'h02c,hi);
      $fwrite(resfd,"R %0d %08x %04x%08x\n",r,bits,hi[15:0],lo);
    end
    if(rows==512)begin
      result_boundary(32'h50000,bits);
      result_boundary(32'h60000,lo);
      result_boundary(32'h70000,hi);
    end
    read_reg(12'h038,v);if(v!=before_cycles)$fatal(1,"DONE cycles changed during readback");
    $fclose(resfd);$fclose(evfd);$finish;
  end
  always @(negedge clk)begin:native_service
    n_cready=0;n_rvalid=0;n_err=0;
    if(!rst)begin
      for(int p=0;p<8;p++)begin
        n_cready[p]=$urandom_range(0,99)>=stall;
        if(pending[p]>0&&qdue[p][head[p]%64]<=cy&&$urandom_range(0,99)>=stall)begin
          n_rvalid[p]=1;n_rdata[p]=memory[p][qaddr[p][head[p]%64]-base];
        end
      end
      if(!injected&&fault!=0&&fault!=3&&total_pending>0&&cy-start_cycle>10)begin
        if(fault==1)begin
          if(rows!=1||pending[7]!=0)$fatal(1,"fault1 requires one row");
          n_rvalid[7]=1;n_rdata[7]='0;
        end else n_err[0]=1;
        injected=1;
      end
    end
  end
  always @(posedge clk)begin:native_monitor
    cy=cy+1;
    if(cy>4000000)$fatal(1,"global timeout");
    if(!rst)begin
      if(n_cwe!=0||n_wvalid!=0)$fatal(1,"read-only kernel wrote native memory");
      for(int p=0;p<8;p++)begin
        if(n_rvalid[p]&&pending[p]>0)begin
          pending[p]--;head[p]++;replied[p]++;
        end
        if(n_cvalid[p]&&n_cready[p])begin
          if(n_caddr[p]<25'(base)||n_caddr[p]>=25'(base+beats))$fatal(1,"native address range");
          if(pending[p]>=32)$fatal(1,"native credit exceeded");
          qaddr[p][tail[p]%64]=int'(n_caddr[p]);
          qdue[p][tail[p]%64]=cy+latency+(p==0?delay_pc0:0)+int'($urandom_range(0,5));
          pending[p]++;tail[p]++;accepted[p]++;
        end
      end
      total_pending=0;for(int p=0;p<8;p++)total_pending+=pending[p];
      if(dut.busy)$fwrite(evfd,"%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
        run_no,cy,n_cvalid&n_cready,n_rvalid,dut.core_accept,dut.core_commit,
        dut.core_out_mask,dut.pf_advance,total_pending,dut.state,
        dut.row_cursor,dut.core_out_tag,dut.u_mxu.command.tag);
    end
  end
endmodule
