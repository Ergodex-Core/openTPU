// Functional AXI3 256-bit memory model for the real SmartConnect/CDC test.
// Not a timing or bandwidth model of HBM silicon.
module axi_hbm_model (
 input logic clk,rst_n,
 input logic [33:0] awaddr,araddr,
 input logic [3:0] awlen,arlen,
 input logic [2:0] awsize,arsize,
 input logic awvalid,arvalid,
 output logic awready,arready,
 input logic [255:0] wdata,input logic [31:0] wstrb,
 input logic wlast,wvalid,output logic wready,
 output logic bvalid,output logic [1:0] bresp,input logic bready,
 output logic [255:0] rdata,output logic rlast,rvalid,
 output logic [1:0] rresp,input logic rready
);
 logic [255:0] mem [0:4095];
 logic wb,rb;
 logic [11:0] wa,ra;
 logic [4:0] wl,rl;
 logic [15:0] rng=16'h5a3d;
 assign awready=rst_n&&!wb&&!bvalid&&rng[0];
 assign arready=rst_n&&!rb&&rng[1];
 assign wready=rst_n&&wb&&rng[2];
 assign rvalid=rst_n&&rb;
 assign rlast=rl==0;
 assign rdata=mem[ra];
 assign rresp=0;assign bresp=0;
 always_ff @(posedge clk) begin
  rng<={rng[14:0],rng[15]^rng[13]^rng[12]^rng[10]};
  if(!rst_n) begin wb<=0;rb<=0;bvalid<=0;wa<=0;ra<=0;wl<=0;rl<=0;end
  else begin
   if(awvalid&&awready) begin
    if(awsize!=5 || awaddr[4:0]!=0) $fatal(1,"bad HBM write geometry");
    wb<=1;wa<=awaddr[16:5];wl<={1'b0,awlen};
   end
   if(wvalid&&wready) begin
    if(wlast!=(wl==0)) $fatal(1,"bad WLAST after width conversion");
    for(integer b=0;b<32;b=b+1) if(wstrb[b]) mem[wa][8*b+:8]<=wdata[8*b+:8];
    wa<=wa+1'b1;wl<=wl-1'b1;
    if(wlast) begin wb<=0;bvalid<=1;end
   end
   if(bvalid&&bready)bvalid<=0;
   if(arvalid&&arready) begin
    if(arsize!=5 || araddr[4:0]!=0) $fatal(1,"bad HBM read geometry");
    rb<=1;ra<=araddr[16:5];rl<={1'b0,arlen};
   end
   if(rvalid&&rready) begin
    ra<=ra+1'b1;rl<=rl-1'b1;
    if(rlast)rb<=0;
   end
  end
 end
endmodule
