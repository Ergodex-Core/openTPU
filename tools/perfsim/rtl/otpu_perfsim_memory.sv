// Alternative test-only boundary for unmodified otpu_slice. No FPGA implementation claim.
module otpu_perfsim_memory #(parameter int WORDS=262144,D=32,SID=0) (
 input logic clk,rst,
 input logic a_req,a_we,input logic [31:0] a_addr,a_wdata,input logic [3:0] a_be,
 output logic a_rdy,a_rvalid,output logic [255:0] a_rdata,
 input logic sw_req,input logic [31:0] sw_addr,sw_wdata,input logic [3:0] sw_be,output logic sw_rdy,
 input logic b_req,b_tag,b_we,input logic [D/4-1:0] b_wmask,input logic [D*8-1:0] b_wdata,input logic [31:0] b_addr,
 output logic b_rdy,b_rvalid,b_rtag,output logic [D*8-1:0] b_rdata,output logic rd_idle,wr_idle,input logic dump
);
 import "DPI-C" function void pm_init(input string profile,directory,root,site,executable,input int width,words);
 import "DPI-C" function void pm_step(input longint cycle,
  input int ar,aw,input int unsigned aa,av,am,input int br,bw,input int unsigned ba,bm,input int bt,input bit [1023:0] bd,
  input int sr,input int unsigned sa,sv,sm,output int ready,arv,output bit [255:0] ard,
  output int brv,brt,output bit [1023:0] brd,output int ri,wi);
 import "DPI-C" function void pm_dump();
 import "DPI-C" function void pm_finish();
 string profile,dir,root,site,exe;
 bit running=0;
 longint cycle=-1; // otpu_seq.cyc is zero at the first edge after rst
 int ready,arv,brv,brt,ri,wi;
 bit [255:0] ard;
 bit [1023:0] brd;
 initial begin
  if(SID!=0 || D>128) $fatal(1,"perfsim native harness supports S=1, D<=128 only");
  if(!$value$plusargs("memory_profile=%s",profile) || !$value$plusargs("dir=%s",dir) ||
     !$value$plusargs("python_root=%s",root) || !$value$plusargs("python_site=%s",site) ||
     !$value$plusargs("python_executable=%s",exe)) $fatal(1,"missing perfsim memory arguments");
  a_rdy=0;b_rdy=0;sw_rdy=0;a_rvalid=0;b_rvalid=0;b_rtag=0;rd_idle=1;wr_idle=1;
  a_rdata='0;b_rdata='0;
  pm_init(profile,dir,root,site,exe,D,WORDS);
 end
 // After the falling-edge reset changes, settle native ready/responses for the next core edge.
 // Logical periods are supplied to the shared controller in ps, independently of #5 clock units.
 always @(negedge clk) begin
  #1;
  if(!rst || running) begin
   running=1;
   cycle=cycle+1;
   pm_step(cycle,int'(a_req && !rst),int'(a_we),a_addr,a_wdata,32'(a_be),int'(b_req && !rst),int'(b_we),b_addr,
      32'(b_wmask),int'(b_tag),1024'(b_wdata),int'(sw_req && !rst),sw_addr,sw_wdata,32'(sw_be),
      ready,arv,ard,brv,brt,brd,ri,wi);
   a_rdy=ready[0];b_rdy=ready[1];sw_rdy=ready[2];a_rvalid=arv[0];a_rdata=ard;
   b_rvalid=brv[0];b_rtag=brt[0];b_rdata=brd[D*8-1:0];rd_idle=ri[0];wr_idle=wi[0];
  end
 end
 always @(posedge clk) if(dump) pm_dump();
 final pm_finish();
endmodule
