"""Native C/W skew, delayed B-derived visibility and modulo count wrap."""
import struct
import pytest
from opentpu import nvfp4 as N
from opentpu.mxu_job import local_job_fields
from test_nvfp4_local_job import desc_pack
from test_nvfp4_ocl import fixture
from test_nvfp4_scale_acc_rtl import simulate,ROOT

@pytest.mark.parametrize('backend',['iverilog','verilator'])
def test_native_job_independent_command_data(tmp_path,backend):
    ds=[];data=[];expected=[]
    for i,(k,mode) in enumerate([(17,0),(64,1),(1,0),(33,1)]):
        j,_=fixture(k,mode,i);a=[0x3f80+(n%31) for n in range(k)];c=[n%16 for n in range(k)];s=[0x38]*((k+15)//16)
        ds.append(f'{desc_pack(local_job_fields(j,j.generation)):0411x}')
        b=struct.pack('<'+'H'*k,*a).ljust(128,b'\0')+N.pack_codes(c).ljust(64,b'\xff')+bytes(s).ljust(64,b'\xff')
        data.extend(f'{int.from_bytes(b[n:n+64],"little"):0128x}' for n in range(0,256,64))
        expected.append(f'{N.strict_row(a,N.pack_codes(c),s,j.global_bits,profile=j.profile):08x}')
    for name,lines in [('desc.hex',ds),('data.hex',data),('expected.hex',expected)]:
        (tmp_path/name).write_text('\n'.join(lines)+'\n')
    sources=[ROOT/'rtl/mxu'/n for n in ['otpu_nvfp4_fp.sv','otpu_nvfp4_job.sv','otpu_nvfp4_strict_dot16.sv',
              'otpu_nvfp4_scale_acc.sv','otpu_nvfp4_local_job.sv','otpu_nvfp4_hbm_job.sv']]
    simulate(tmp_path,backend,sources,TB)

TB=r'''
module tb;
 import otpu_nvfp4_job::*;
 logic clk=0,rst=1,start=0,abort_request=0,complete_ready=0;
 always #5 clk=~clk;
 descriptor_t descriptor;
 logic busy,too_late,complete,output_visible,memory_outstanding;
 logic [7:0] status,faults;
 logic [31:0] bits;
 logic n_cvalid,n_cwe,n_wvalid,n_cready,n_wready,n_rvalid=0,n_err=0;
 logic [24:0] n_caddr;
 logic [511:0] n_wdata,n_rdata=0;
 logic [63:0] n_wmask;
 logic [15:0] n_wdone=16'hfffe;
 otpu_nvfp4_hbm_job dut(.*);
 logic [1640:0] ds[4];logic [511:0] data[16],mem[512];logic [31:0] expected[4];
 integer j=0,cyc=0,write_age=0,rd_delay=0,b_delay=0,cf=-1,wf=-1,skews=0;
 logic read_pending=0,captured_c=0,captured_w=0;
 logic [511:0] rd_saved,write_saved;
 logic [24:0] write_addr;
 assign n_cready=!rst && (dut.state==7 ? (j[0] || write_age>=12) : cyc%3!=0);
 assign n_wready=!rst && (j[0] ? write_age>=12 : 1'b1);
 always @(posedge clk)begin
   cyc<=cyc+1;n_rvalid<=0;
   if(dut.state==7)write_age<=write_age+1;else write_age<=0;
   if(n_cvalid && n_cready)begin
     if(n_cwe)begin captured_c<=1;write_addr<=n_caddr;cf<=cyc;end
     else begin
       if(read_pending)$fatal(1,"multiple reads outstanding");
       read_pending<=1;rd_saved<=mem[int'(n_caddr)];rd_delay<=7;
     end
   end
   if(read_pending)begin
     if(rd_delay==0)begin n_rvalid<=1;n_rdata<=rd_saved;read_pending<=0;end
     else rd_delay<=rd_delay-1;
   end
   if(n_wvalid && n_wready)begin
     if(n_wmask!=15)$fatal(1,"output strobe");captured_w<=1;write_saved<=n_wdata;wf<=cyc;
   end
   if(captured_c && captured_w)begin
     if(b_delay==9)begin
       if(write_addr!=25'('h4000/64))$fatal(1,"write address");
       mem[int'(write_addr)][31:0]<=write_saved[31:0];n_wdone<=n_wdone+16'd1;
       captured_c<=0;captured_w<=0;b_delay<=0;
     end else begin
       if(complete || output_visible)$fatal(1,"visibility before write acknowledgement");
       b_delay<=b_delay+1;
     end
   end
 end
 initial begin
   $readmemh("desc.hex",ds);$readmemh("data.hex",data);$readmemh("expected.hex",expected);
   for(integer i=0;i<512;i++)mem[i]=0;
   repeat(3)@(negedge clk);rst=0;
   for(j=0;j<4;j++)begin
     descriptor=descriptor_t'(ds[j]);mem['h1000/64]=data[4*j];mem['h1040/64]=data[4*j+1];
     mem['h2000/64]=data[4*j+2];mem['h3000/64]=data[4*j+3];mem['h4000/64]=512'hdeadbeef;
     cf=-1;wf=-1;start=1;@(negedge clk);start=0;
     while(!complete)@(negedge clk);
     if(status!=1 || faults!=0 || bits!==expected[j] || !output_visible || memory_outstanding)$fatal(1,"native strict result j=%0d",j);
     if(mem['h4000/64][31:0]!==expected[j] || cf<0 || wf<0 || (cf-wf<10 && wf-cf<10))$fatal(1,"native independent C/W or Y");
     skews++;
     repeat(11)begin @(negedge clk);if(!complete || bits!==expected[j] || !busy)$fatal(1,"held completion");end
     complete_ready=1;@(negedge clk);complete_ready=0;repeat(4)@(negedge clk);
   end
   if(n_wdone!=2)$fatal(1,"B count wrap");
   $display("TEST_PASS native_job command_first=2 data_first=2 delayed_B=4 count_wrap=1 strict_rows=4");$finish;
 end
 initial begin #200000;$fatal(1,"timeout");end
endmodule
'''
