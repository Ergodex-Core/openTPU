"""The production channel owner must hide sidecar B counts, including wrap."""
import pytest
from test_nvfp4_scale_acc_rtl import simulate,ROOT

@pytest.mark.parametrize('backend',['iverilog','verilator'])
def test_native_owner(tmp_path,backend):
    simulate(tmp_path,backend,[ROOT/'rtl/mxu/otpu_nvfp4_native_owner.sv'],TB)

TB=r'''
module tb;
 logic clk=0,rst=1,owns=0;
 always #5 clk=~clk;
 logic b_cvalid=0,b_cwe=0,b_wvalid=0,b_cready,b_wready,b_rvalid;
 logic [24:0] b_caddr=3;
 logic [511:0] b_wdata=123,b_rdata;
 logic [63:0] b_wmask=15;
 logic [15:0] b_wdone;
 logic v_cvalid=0,v_cwe=0,v_wvalid=0;
 logic [24:0] v_caddr=9;
 logic [511:0] v_wdata=456;
 logic [63:0] v_wmask=255;
 logic n_cvalid,n_cwe,n_wvalid,n_cready=0,n_wready=0,n_rvalid=0;
 logic [24:0] n_caddr;
 logic [511:0] n_wdata,n_rdata=789;
 logic [63:0] n_wmask;
 logic [15:0] n_wdone=0;
 otpu_nvfp4_native_owner dut(.*);
 integer legacy=0,sidecar=0;
 task automatic tick;
   @(posedge clk);@(negedge clk);
 endtask
 task automatic check;
   #1;
   if(b_wdone!==16'(legacy))$fatal(1,"legacy write count got=%d expected=%d raw=%d",b_wdone,legacy,n_wdone);
   if(b_cready!==(!owns && n_cready) || b_wready!==(!owns && n_wready) || b_rvalid!==(!owns && n_rvalid))$fatal(1,"legacy routing");
   if(n_cvalid!==(owns ? v_cvalid : b_cvalid) || n_wvalid!==(owns ? v_wvalid : b_wvalid))$fatal(1,"owner valid");
   if(n_caddr!==(owns ? v_caddr : b_caddr) || n_wdata!==(owns ? v_wdata : b_wdata) || n_wmask!==(owns ? v_wmask : b_wmask))$fatal(1,"owner payload");
 endtask
 initial begin
   repeat(3)tick();rst=0;tick();
   // Legacy completions before, between and after many ownership handovers.
   // Actual accepted writes are represented by B-derived raw count increments.
   for(integer i=0;i<131080;i++)begin
     owns=0;b_cvalid=1;b_wvalid=1;n_cready=i[0];n_wready=!i[0];n_rvalid=i[1];
     check();tick();n_wdone=n_wdone+16'd1;legacy++;check();tick();check();
     if(i%17==0)begin
       owns=1;v_cvalid=1;v_wvalid=1;check();tick();
       // Independent ready combinations and a delayed response while credit held.
       for(integer k=0;k<4;k++)begin n_cready=k[0];n_wready=k[1];check();tick();end
       n_wdone=n_wdone+16'd1;sidecar++;check();tick();check();
       v_cvalid=0;v_wvalid=0;repeat(3)begin tick();check();end
       owns=0;check();tick();check();
     end
   end
   // A coordinated reset resets both counters; no stale delta survives.
   rst=1;n_wdone=0;legacy=0;owns=0;tick();rst=0;tick();check();
   $display("TEST_PASS native_owner legacy_writes=131080 sidecar_writes=%0d raw_wraps=2",sidecar);
   $finish;
 end
endmodule
'''
