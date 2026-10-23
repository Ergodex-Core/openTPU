// Exclusive-owner native channel mux. Acquisition requires legacy quiescence;
// release requires all sidecar responses drained and terminal completion ACK.
module otpu_nvfp4_native_owner (
 input logic clk,rst,owns,
 input logic b_cvalid,b_cwe,b_wvalid,
 input logic [24:0] b_caddr,
 input logic [511:0] b_wdata,
 input logic [63:0] b_wmask,
 output logic b_cready,b_wready,b_rvalid,
 output logic [511:0] b_rdata,
 output logic [15:0] b_wdone,
 input logic v_cvalid,v_cwe,v_wvalid,
 input logic [24:0] v_caddr,
 input logic [511:0] v_wdata,
 input logic [63:0] v_wmask,
 output logic n_cvalid,n_cwe,n_wvalid,
 output logic [24:0] n_caddr,
 output logic [511:0] n_wdata,
 output logic [63:0] n_wmask,
 input logic n_cready,n_wready,n_rvalid,
 input logic [511:0] n_rdata,
 input logic [15:0] n_wdone
);
 logic [15:0] offset,previous_done;
 always_ff @(posedge clk)begin
   if(rst)begin offset<=0;previous_done<=0;end
   else begin
     previous_done<=n_wdone;
     if(owns)offset<=offset+(n_wdone-previous_done);
   end
 end
 assign n_cvalid=owns ? v_cvalid : b_cvalid;
 assign n_cwe=owns ? v_cwe : b_cwe;
 assign n_caddr=owns ? v_caddr : b_caddr;
 assign n_wvalid=owns ? v_wvalid : b_wvalid;
 assign n_wdata=owns ? v_wdata : b_wdata;
 assign n_wmask=owns ? v_wmask : b_wmask;
 assign b_cready=!owns && n_cready;
 assign b_wready=!owns && n_wready;
 assign b_rvalid=!owns && n_rvalid;
 assign b_rdata=n_rdata;
 // Subtract the current B delta too; a legacy idle check must never see it.
 assign b_wdone=n_wdone-offset-(owns ? (n_wdone-previous_done) : 16'd0);
endmodule
