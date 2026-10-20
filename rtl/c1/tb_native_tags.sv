`timescale 1ns/1ps
// Exercise every native port, engine/slot, changing row/generation tags,
// pointer wrap, saturated reservations, response backpressure, and full512groups.
module tb_c1_native_packed;
  reg clk=0,rst=1,enable=0;
  always #2.5 clk=~clk;
  reg [22:0] base=23'd4096;
  reg [15:0] req_valid=0,rsp_ready=0;
  wire [15:0] req_ready,rsp_valid;
  reg [15:0][1:0] req_slot=0;
  reg [15:0][31:0] req_row=0,req_generation=0;
  reg [15:0][7:0] req_kb=0;
  wire [15:0][1:0] rsp_slot;
  wire [15:0][31:0] rsp_row,rsp_generation;
  wire [15:0][9:0] rsp_group;
  wire [15:0][255:0] rsp_weights;
  wire [7:0] n_cvalid;
  reg [7:0] n_cready=0,n_rvalid=0,n_err=0;
  wire [7:0][24:0] n_caddr;
  reg [7:0][511:0] n_rdata=0;
  wire drained,error;
  wire [7:0][31:0] pc_issued,pc_returned,pc_peak,pc_blocked,pc_credit;
  otpu_c1_native dut(.*);
  integer cycle=0,total_groups=0;
  integer sent_jobs[16],out_job[16],out_group[16];
  integer head[8],tail[8],due[8][4096];
  reg [24:0] addresses[8][4096];
  reg [511:0] expected_data;
  reg [24:0] expected_addr;
  reg [31:0] expected_generation;
  integer finished;
  function automatic [511:0] payload(input [24:0] address);
    for(integer w=0;w<16;w=w+1)payload[w*32+:32]=32'h13579bdf ^ {7'd0,address} ^ (32'h1020304*w);
  endfunction
  initial begin
    for(integer e=0;e<16;e=e+1)begin sent_jobs[e]=0;out_job[e]=0;out_group[e]=0;end
    for(integer p=0;p<8;p=p+1)begin head[p]=0;tail[p]=0;end
    repeat(24) @(negedge clk);
    rst=0;enable=1;
  end
  always @(negedge clk)if(!rst)begin
    cycle=cycle+1;
    for(integer e=0;e<16;e=e+1)begin
      req_valid[e]=sent_jobs[e]<4;
      req_slot[e]=sent_jobs[e]%4;
      req_row[e]=e+16*sent_jobs[e];
      req_generation[e]=32'hcafe1234 ^ (32'h01010203*sent_jobs[e]);
      req_kb[e]=8'd128;
      rsp_ready[e]=((cycle+e*3)%113)>=43 && (cycle+e)%7!=0;
    end
    for(integer p=0;p<8;p=p+1)begin
      n_cready[p]=(cycle+p)%7!=0;
      n_rvalid[p]=head[p]<tail[p] && due[p][head[p]%4096]<=cycle && (cycle+p)%5!=0;
      n_rdata[p]=n_rvalid[p]?payload(addresses[p][head[p]%4096]):0;
    end
    if(cycle>100000)$fatal(1,"Native tag test timed out groups=%0d",total_groups);
  end
  always @(posedge clk)if(!rst)begin
    if(error)$fatal(1,"Unexpected native adapter error");
    for(integer p=0;p<8;p=p+1)begin
      if(n_cvalid[p]&&n_cready[p])begin
        if(tail[p]-head[p]>=4095)$fatal(1,"Test memory queue overflow");
        addresses[p][tail[p]%4096]=n_caddr[p];
        due[p][tail[p]%4096]=cycle+31+p*3;
        tail[p]=tail[p]+1;
      end
      if(n_rvalid[p])head[p]=head[p]+1;
    end
    for(integer e=0;e<16;e=e+1)begin
      if(req_valid[e]&&req_ready[e])sent_jobs[e]=sent_jobs[e]+1;
      if(rsp_valid[e]&&rsp_ready[e])begin
        if(out_job[e]>=4)$fatal(1,"Extra response engine%0d",e);
        expected_generation=32'hcafe1234 ^ (32'h01010203*out_job[e]);
        expected_addr=4096+(out_job[e]*2+(e%2))*128*2+out_group[e]/2;
        expected_data=payload(expected_addr);
        if(rsp_slot[e]!==2'(out_job[e]%4)||rsp_row[e]!==32'(e+16*out_job[e])||rsp_generation[e]!==expected_generation||rsp_group[e]!==10'(out_group[e]))
          $fatal(1,"Tag mismatch engine%0d job%0d group%0d got slot%0d row%0d gen%08x group%0d",e,out_job[e],out_group[e],rsp_slot[e],rsp_row[e],rsp_generation[e],rsp_group[e]);
        if(rsp_weights[e] !== (out_group[e]%2?expected_data[511:256]:expected_data[255:0]))$fatal(1,"Payload mismatch engine%0d group%0d",e,out_group[e]);
        total_groups=total_groups+1;
        if(out_group[e]==511)begin out_group[e]=0;out_job[e]=out_job[e]+1;end
        else out_group[e]=out_group[e]+1;
      end
    end
    finished=1;
    for(integer e=0;e<16;e=e+1)if(out_job[e]!=4)finished=0;
    if(finished)begin
      if(total_groups!=32768||!drained)$fatal(1,"Final group/drain mismatch");
      for(integer p=0;p<8;p=p+1)begin
        if(pc_issued[p]!==32'd2048||pc_returned[p]!==32'd2048||head[p]!=tail[p])$fatal(1,"Port count mismatch%0d",p);
        if(pc_peak[p]<16||pc_peak[p]>64)$fatal(1,"Insufficient or invalid outstanding coverage%0d",p);
      end
      $display("C1_NATIVE_PACKED_EXACT groups=%0d native_reads=16384 rows=64 cycles=%0d",total_groups,cycle);
      $finish;
    end
  end
endmodule
