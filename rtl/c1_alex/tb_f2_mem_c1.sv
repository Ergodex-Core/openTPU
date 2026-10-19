// Actual vendor-IP simulation. The frozen functional HBM model is not silicon timing.
`timescale 1ns/1ps
module tb_f2_mem_c1;
integer seed=1, phase=0;
logic core_clk=0,host_clk=0,hbm_clk=0,aresetn=0;
logic hold_model_ar=0;
initial begin
 if(!$value$plusargs("SEED=%d",seed)) seed=1;
 if(!$value$plusargs("PHASE=%d",phase)) phase=0;
end
initial begin #0.001; #(phase*0.317); forever #2.5 core_clk=~core_clk; end
initial begin #0.001; #(phase*0.593); forever #2 host_clk=~host_clk; end
// 1-ps simulator resolution: 2.222 ns, nominal 450 MHz (0.01% quantization).
initial begin #0.001; #(phase*0.829); forever #1.111 hbm_clk=~hbm_clk; end

wire [33:0] N_araddr [0:7];
wire [1:0] N_arburst [0:7];
wire [3:0] N_arcache [0:7];
wire N_arid [0:7];
wire [7:0] N_arlen [0:7];
wire N_arlock [0:7];
wire [2:0] N_arprot [0:7];
wire [3:0] N_arqos [0:7];
wire N_arready [0:7];
wire [2:0] N_arsize [0:7];
wire N_arvalid [0:7];
wire [33:0] N_awaddr [0:7];
wire [1:0] N_awburst [0:7];
wire [3:0] N_awcache [0:7];
wire N_awid [0:7];
wire [7:0] N_awlen [0:7];
wire N_awlock [0:7];
wire [2:0] N_awprot [0:7];
wire [3:0] N_awqos [0:7];
wire N_awready [0:7];
wire [2:0] N_awsize [0:7];
wire N_awvalid [0:7];
wire N_bid [0:7];
wire N_bready [0:7];
wire [1:0] N_bresp [0:7];
wire N_bvalid [0:7];
wire [511:0] N_rdata [0:7];
wire N_rid [0:7];
wire N_rlast [0:7];
wire N_rready [0:7];
wire [1:0] N_rresp [0:7];
wire N_rvalid [0:7];
wire [511:0] N_wdata [0:7];
wire N_wlast [0:7];
wire N_wready [0:7];
wire [63:0] N_wstrb [0:7];
wire N_wvalid [0:7];
wire [33:0] H_araddr [0:7];
wire [1:0] H_arburst [0:7];
wire [3:0] H_arcache [0:7];
wire [3:0] H_arlen [0:7];
wire [1:0] H_arlock [0:7];
wire [2:0] H_arprot [0:7];
wire [3:0] H_arqos [0:7];
wire H_arready [0:7];
wire [2:0] H_arsize [0:7];
wire H_arvalid [0:7];
wire [33:0] H_awaddr [0:7];
wire [1:0] H_awburst [0:7];
wire [3:0] H_awcache [0:7];
wire [3:0] H_awlen [0:7];
wire [1:0] H_awlock [0:7];
wire [2:0] H_awprot [0:7];
wire [3:0] H_awqos [0:7];
wire H_awready [0:7];
wire [2:0] H_awsize [0:7];
wire H_awvalid [0:7];
wire H_bready [0:7];
wire [1:0] H_bresp [0:7];
wire H_bvalid [0:7];
wire [255:0] H_rdata [0:7];
wire H_rlast [0:7];
wire H_rready [0:7];
wire [1:0] H_rresp [0:7];
wire H_rvalid [0:7];
wire [255:0] H_wdata [0:7];
wire H_wlast [0:7];
wire H_wready [0:7];
wire [31:0] H_wstrb [0:7];
wire H_wvalid [0:7];
logic [63:0] S_araddr=0;
logic [1:0] S_arburst=0;
logic [3:0] S_arcache=0;
logic [15:0] S_arid=0;
logic [7:0] S_arlen=0;
logic S_arlock=0;
logic [2:0] S_arprot=0;
logic [3:0] S_arqos=0;
wire S_arready;
logic [2:0] S_arsize=0;
logic S_arvalid=0;
logic [63:0] S_awaddr=0;
logic [1:0] S_awburst=0;
logic [3:0] S_awcache=0;
logic [15:0] S_awid=0;
logic [7:0] S_awlen=0;
logic S_awlock=0;
logic [2:0] S_awprot=0;
logic [3:0] S_awqos=0;
wire S_awready;
logic [2:0] S_awsize=0;
logic S_awvalid=0;
wire [15:0] S_bid;
logic S_bready=0;
wire [1:0] S_bresp;
wire S_bvalid;
wire [511:0] S_rdata;
wire [15:0] S_rid;
wire S_rlast;
logic S_rready=0;
wire [1:0] S_rresp;
wire S_rvalid;
logic [511:0] S_wdata=0;
logic S_wlast=0;
wire S_wready;
logic [63:0] S_wstrb=0;
logic S_wvalid=0;
wire [63:0] P_araddr;
wire [1:0] P_arburst;
wire [3:0] P_arcache;
wire [15:0] P_arid;
wire [7:0] P_arlen;
wire P_arlock;
wire [2:0] P_arprot;
wire [3:0] P_arqos;
wire P_arready;
wire [2:0] P_arsize;
wire P_arvalid;
wire [63:0] P_awaddr;
wire [1:0] P_awburst;
wire [3:0] P_awcache;
wire [15:0] P_awid;
wire [7:0] P_awlen;
wire P_awlock;
wire [2:0] P_awprot;
wire [3:0] P_awqos;
wire P_awready;
wire [2:0] P_awsize;
wire P_awvalid;
wire [15:0] P_bid;
wire P_bready;
wire [1:0] P_bresp;
wire P_bvalid;
wire [511:0] P_rdata;
wire [15:0] P_rid;
wire P_rlast;
wire P_rready;
wire [1:0] P_rresp;
wire P_rvalid;
wire [511:0] P_wdata;
wire P_wlast;
wire P_wready;
wire [63:0] P_wstrb;
wire P_wvalid;
f2_mem_wrapper dut (
 .aresetn(aresetn),
 .core_clk(core_clk),
 .host_clk(host_clk),
 .hbm_clk(hbm_clk),
 .S_N0_araddr(N_araddr[0]),
 .S_N0_arburst(N_arburst[0]),
 .S_N0_arcache(N_arcache[0]),
 .S_N0_arid(N_arid[0]),
 .S_N0_arlen(N_arlen[0]),
 .S_N0_arlock(N_arlock[0]),
 .S_N0_arprot(N_arprot[0]),
 .S_N0_arqos(N_arqos[0]),
 .S_N0_arready(N_arready[0]),
 .S_N0_arsize(N_arsize[0]),
 .S_N0_arvalid(N_arvalid[0]),
 .S_N0_awaddr(N_awaddr[0]),
 .S_N0_awburst(N_awburst[0]),
 .S_N0_awcache(N_awcache[0]),
 .S_N0_awid(N_awid[0]),
 .S_N0_awlen(N_awlen[0]),
 .S_N0_awlock(N_awlock[0]),
 .S_N0_awprot(N_awprot[0]),
 .S_N0_awqos(N_awqos[0]),
 .S_N0_awready(N_awready[0]),
 .S_N0_awsize(N_awsize[0]),
 .S_N0_awvalid(N_awvalid[0]),
 .S_N0_bid(N_bid[0]),
 .S_N0_bready(N_bready[0]),
 .S_N0_bresp(N_bresp[0]),
 .S_N0_bvalid(N_bvalid[0]),
 .S_N0_rdata(N_rdata[0]),
 .S_N0_rid(N_rid[0]),
 .S_N0_rlast(N_rlast[0]),
 .S_N0_rready(N_rready[0]),
 .S_N0_rresp(N_rresp[0]),
 .S_N0_rvalid(N_rvalid[0]),
 .S_N0_wdata(N_wdata[0]),
 .S_N0_wlast(N_wlast[0]),
 .S_N0_wready(N_wready[0]),
 .S_N0_wstrb(N_wstrb[0]),
 .S_N0_wvalid(N_wvalid[0]),
 .M_H0_araddr(H_araddr[0]),
 .M_H0_arburst(H_arburst[0]),
 .M_H0_arcache(H_arcache[0]),
 .M_H0_arlen(H_arlen[0]),
 .M_H0_arlock(H_arlock[0]),
 .M_H0_arprot(H_arprot[0]),
 .M_H0_arqos(H_arqos[0]),
 .M_H0_arready(H_arready[0]),
 .M_H0_arsize(H_arsize[0]),
 .M_H0_arvalid(H_arvalid[0]),
 .M_H0_awaddr(H_awaddr[0]),
 .M_H0_awburst(H_awburst[0]),
 .M_H0_awcache(H_awcache[0]),
 .M_H0_awlen(H_awlen[0]),
 .M_H0_awlock(H_awlock[0]),
 .M_H0_awprot(H_awprot[0]),
 .M_H0_awqos(H_awqos[0]),
 .M_H0_awready(H_awready[0]),
 .M_H0_awsize(H_awsize[0]),
 .M_H0_awvalid(H_awvalid[0]),
 .M_H0_bready(H_bready[0]),
 .M_H0_bresp(H_bresp[0]),
 .M_H0_bvalid(H_bvalid[0]),
 .M_H0_rdata(H_rdata[0]),
 .M_H0_rlast(H_rlast[0]),
 .M_H0_rready(H_rready[0]),
 .M_H0_rresp(H_rresp[0]),
 .M_H0_rvalid(H_rvalid[0]),
 .M_H0_wdata(H_wdata[0]),
 .M_H0_wlast(H_wlast[0]),
 .M_H0_wready(H_wready[0]),
 .M_H0_wstrb(H_wstrb[0]),
 .M_H0_wvalid(H_wvalid[0]),
 .S_N1_araddr(N_araddr[1]),
 .S_N1_arburst(N_arburst[1]),
 .S_N1_arcache(N_arcache[1]),
 .S_N1_arid(N_arid[1]),
 .S_N1_arlen(N_arlen[1]),
 .S_N1_arlock(N_arlock[1]),
 .S_N1_arprot(N_arprot[1]),
 .S_N1_arqos(N_arqos[1]),
 .S_N1_arready(N_arready[1]),
 .S_N1_arsize(N_arsize[1]),
 .S_N1_arvalid(N_arvalid[1]),
 .S_N1_awaddr(N_awaddr[1]),
 .S_N1_awburst(N_awburst[1]),
 .S_N1_awcache(N_awcache[1]),
 .S_N1_awid(N_awid[1]),
 .S_N1_awlen(N_awlen[1]),
 .S_N1_awlock(N_awlock[1]),
 .S_N1_awprot(N_awprot[1]),
 .S_N1_awqos(N_awqos[1]),
 .S_N1_awready(N_awready[1]),
 .S_N1_awsize(N_awsize[1]),
 .S_N1_awvalid(N_awvalid[1]),
 .S_N1_bid(N_bid[1]),
 .S_N1_bready(N_bready[1]),
 .S_N1_bresp(N_bresp[1]),
 .S_N1_bvalid(N_bvalid[1]),
 .S_N1_rdata(N_rdata[1]),
 .S_N1_rid(N_rid[1]),
 .S_N1_rlast(N_rlast[1]),
 .S_N1_rready(N_rready[1]),
 .S_N1_rresp(N_rresp[1]),
 .S_N1_rvalid(N_rvalid[1]),
 .S_N1_wdata(N_wdata[1]),
 .S_N1_wlast(N_wlast[1]),
 .S_N1_wready(N_wready[1]),
 .S_N1_wstrb(N_wstrb[1]),
 .S_N1_wvalid(N_wvalid[1]),
 .M_H1_araddr(H_araddr[1]),
 .M_H1_arburst(H_arburst[1]),
 .M_H1_arcache(H_arcache[1]),
 .M_H1_arlen(H_arlen[1]),
 .M_H1_arlock(H_arlock[1]),
 .M_H1_arprot(H_arprot[1]),
 .M_H1_arqos(H_arqos[1]),
 .M_H1_arready(H_arready[1]),
 .M_H1_arsize(H_arsize[1]),
 .M_H1_arvalid(H_arvalid[1]),
 .M_H1_awaddr(H_awaddr[1]),
 .M_H1_awburst(H_awburst[1]),
 .M_H1_awcache(H_awcache[1]),
 .M_H1_awlen(H_awlen[1]),
 .M_H1_awlock(H_awlock[1]),
 .M_H1_awprot(H_awprot[1]),
 .M_H1_awqos(H_awqos[1]),
 .M_H1_awready(H_awready[1]),
 .M_H1_awsize(H_awsize[1]),
 .M_H1_awvalid(H_awvalid[1]),
 .M_H1_bready(H_bready[1]),
 .M_H1_bresp(H_bresp[1]),
 .M_H1_bvalid(H_bvalid[1]),
 .M_H1_rdata(H_rdata[1]),
 .M_H1_rlast(H_rlast[1]),
 .M_H1_rready(H_rready[1]),
 .M_H1_rresp(H_rresp[1]),
 .M_H1_rvalid(H_rvalid[1]),
 .M_H1_wdata(H_wdata[1]),
 .M_H1_wlast(H_wlast[1]),
 .M_H1_wready(H_wready[1]),
 .M_H1_wstrb(H_wstrb[1]),
 .M_H1_wvalid(H_wvalid[1]),
 .S_N2_araddr(N_araddr[2]),
 .S_N2_arburst(N_arburst[2]),
 .S_N2_arcache(N_arcache[2]),
 .S_N2_arid(N_arid[2]),
 .S_N2_arlen(N_arlen[2]),
 .S_N2_arlock(N_arlock[2]),
 .S_N2_arprot(N_arprot[2]),
 .S_N2_arqos(N_arqos[2]),
 .S_N2_arready(N_arready[2]),
 .S_N2_arsize(N_arsize[2]),
 .S_N2_arvalid(N_arvalid[2]),
 .S_N2_awaddr(N_awaddr[2]),
 .S_N2_awburst(N_awburst[2]),
 .S_N2_awcache(N_awcache[2]),
 .S_N2_awid(N_awid[2]),
 .S_N2_awlen(N_awlen[2]),
 .S_N2_awlock(N_awlock[2]),
 .S_N2_awprot(N_awprot[2]),
 .S_N2_awqos(N_awqos[2]),
 .S_N2_awready(N_awready[2]),
 .S_N2_awsize(N_awsize[2]),
 .S_N2_awvalid(N_awvalid[2]),
 .S_N2_bid(N_bid[2]),
 .S_N2_bready(N_bready[2]),
 .S_N2_bresp(N_bresp[2]),
 .S_N2_bvalid(N_bvalid[2]),
 .S_N2_rdata(N_rdata[2]),
 .S_N2_rid(N_rid[2]),
 .S_N2_rlast(N_rlast[2]),
 .S_N2_rready(N_rready[2]),
 .S_N2_rresp(N_rresp[2]),
 .S_N2_rvalid(N_rvalid[2]),
 .S_N2_wdata(N_wdata[2]),
 .S_N2_wlast(N_wlast[2]),
 .S_N2_wready(N_wready[2]),
 .S_N2_wstrb(N_wstrb[2]),
 .S_N2_wvalid(N_wvalid[2]),
 .M_H2_araddr(H_araddr[2]),
 .M_H2_arburst(H_arburst[2]),
 .M_H2_arcache(H_arcache[2]),
 .M_H2_arlen(H_arlen[2]),
 .M_H2_arlock(H_arlock[2]),
 .M_H2_arprot(H_arprot[2]),
 .M_H2_arqos(H_arqos[2]),
 .M_H2_arready(H_arready[2]),
 .M_H2_arsize(H_arsize[2]),
 .M_H2_arvalid(H_arvalid[2]),
 .M_H2_awaddr(H_awaddr[2]),
 .M_H2_awburst(H_awburst[2]),
 .M_H2_awcache(H_awcache[2]),
 .M_H2_awlen(H_awlen[2]),
 .M_H2_awlock(H_awlock[2]),
 .M_H2_awprot(H_awprot[2]),
 .M_H2_awqos(H_awqos[2]),
 .M_H2_awready(H_awready[2]),
 .M_H2_awsize(H_awsize[2]),
 .M_H2_awvalid(H_awvalid[2]),
 .M_H2_bready(H_bready[2]),
 .M_H2_bresp(H_bresp[2]),
 .M_H2_bvalid(H_bvalid[2]),
 .M_H2_rdata(H_rdata[2]),
 .M_H2_rlast(H_rlast[2]),
 .M_H2_rready(H_rready[2]),
 .M_H2_rresp(H_rresp[2]),
 .M_H2_rvalid(H_rvalid[2]),
 .M_H2_wdata(H_wdata[2]),
 .M_H2_wlast(H_wlast[2]),
 .M_H2_wready(H_wready[2]),
 .M_H2_wstrb(H_wstrb[2]),
 .M_H2_wvalid(H_wvalid[2]),
 .S_N3_araddr(N_araddr[3]),
 .S_N3_arburst(N_arburst[3]),
 .S_N3_arcache(N_arcache[3]),
 .S_N3_arid(N_arid[3]),
 .S_N3_arlen(N_arlen[3]),
 .S_N3_arlock(N_arlock[3]),
 .S_N3_arprot(N_arprot[3]),
 .S_N3_arqos(N_arqos[3]),
 .S_N3_arready(N_arready[3]),
 .S_N3_arsize(N_arsize[3]),
 .S_N3_arvalid(N_arvalid[3]),
 .S_N3_awaddr(N_awaddr[3]),
 .S_N3_awburst(N_awburst[3]),
 .S_N3_awcache(N_awcache[3]),
 .S_N3_awid(N_awid[3]),
 .S_N3_awlen(N_awlen[3]),
 .S_N3_awlock(N_awlock[3]),
 .S_N3_awprot(N_awprot[3]),
 .S_N3_awqos(N_awqos[3]),
 .S_N3_awready(N_awready[3]),
 .S_N3_awsize(N_awsize[3]),
 .S_N3_awvalid(N_awvalid[3]),
 .S_N3_bid(N_bid[3]),
 .S_N3_bready(N_bready[3]),
 .S_N3_bresp(N_bresp[3]),
 .S_N3_bvalid(N_bvalid[3]),
 .S_N3_rdata(N_rdata[3]),
 .S_N3_rid(N_rid[3]),
 .S_N3_rlast(N_rlast[3]),
 .S_N3_rready(N_rready[3]),
 .S_N3_rresp(N_rresp[3]),
 .S_N3_rvalid(N_rvalid[3]),
 .S_N3_wdata(N_wdata[3]),
 .S_N3_wlast(N_wlast[3]),
 .S_N3_wready(N_wready[3]),
 .S_N3_wstrb(N_wstrb[3]),
 .S_N3_wvalid(N_wvalid[3]),
 .M_H3_araddr(H_araddr[3]),
 .M_H3_arburst(H_arburst[3]),
 .M_H3_arcache(H_arcache[3]),
 .M_H3_arlen(H_arlen[3]),
 .M_H3_arlock(H_arlock[3]),
 .M_H3_arprot(H_arprot[3]),
 .M_H3_arqos(H_arqos[3]),
 .M_H3_arready(H_arready[3]),
 .M_H3_arsize(H_arsize[3]),
 .M_H3_arvalid(H_arvalid[3]),
 .M_H3_awaddr(H_awaddr[3]),
 .M_H3_awburst(H_awburst[3]),
 .M_H3_awcache(H_awcache[3]),
 .M_H3_awlen(H_awlen[3]),
 .M_H3_awlock(H_awlock[3]),
 .M_H3_awprot(H_awprot[3]),
 .M_H3_awqos(H_awqos[3]),
 .M_H3_awready(H_awready[3]),
 .M_H3_awsize(H_awsize[3]),
 .M_H3_awvalid(H_awvalid[3]),
 .M_H3_bready(H_bready[3]),
 .M_H3_bresp(H_bresp[3]),
 .M_H3_bvalid(H_bvalid[3]),
 .M_H3_rdata(H_rdata[3]),
 .M_H3_rlast(H_rlast[3]),
 .M_H3_rready(H_rready[3]),
 .M_H3_rresp(H_rresp[3]),
 .M_H3_rvalid(H_rvalid[3]),
 .M_H3_wdata(H_wdata[3]),
 .M_H3_wlast(H_wlast[3]),
 .M_H3_wready(H_wready[3]),
 .M_H3_wstrb(H_wstrb[3]),
 .M_H3_wvalid(H_wvalid[3]),
 .S_N4_araddr(N_araddr[4]),
 .S_N4_arburst(N_arburst[4]),
 .S_N4_arcache(N_arcache[4]),
 .S_N4_arid(N_arid[4]),
 .S_N4_arlen(N_arlen[4]),
 .S_N4_arlock(N_arlock[4]),
 .S_N4_arprot(N_arprot[4]),
 .S_N4_arqos(N_arqos[4]),
 .S_N4_arready(N_arready[4]),
 .S_N4_arsize(N_arsize[4]),
 .S_N4_arvalid(N_arvalid[4]),
 .S_N4_awaddr(N_awaddr[4]),
 .S_N4_awburst(N_awburst[4]),
 .S_N4_awcache(N_awcache[4]),
 .S_N4_awid(N_awid[4]),
 .S_N4_awlen(N_awlen[4]),
 .S_N4_awlock(N_awlock[4]),
 .S_N4_awprot(N_awprot[4]),
 .S_N4_awqos(N_awqos[4]),
 .S_N4_awready(N_awready[4]),
 .S_N4_awsize(N_awsize[4]),
 .S_N4_awvalid(N_awvalid[4]),
 .S_N4_bid(N_bid[4]),
 .S_N4_bready(N_bready[4]),
 .S_N4_bresp(N_bresp[4]),
 .S_N4_bvalid(N_bvalid[4]),
 .S_N4_rdata(N_rdata[4]),
 .S_N4_rid(N_rid[4]),
 .S_N4_rlast(N_rlast[4]),
 .S_N4_rready(N_rready[4]),
 .S_N4_rresp(N_rresp[4]),
 .S_N4_rvalid(N_rvalid[4]),
 .S_N4_wdata(N_wdata[4]),
 .S_N4_wlast(N_wlast[4]),
 .S_N4_wready(N_wready[4]),
 .S_N4_wstrb(N_wstrb[4]),
 .S_N4_wvalid(N_wvalid[4]),
 .M_H4_araddr(H_araddr[4]),
 .M_H4_arburst(H_arburst[4]),
 .M_H4_arcache(H_arcache[4]),
 .M_H4_arlen(H_arlen[4]),
 .M_H4_arlock(H_arlock[4]),
 .M_H4_arprot(H_arprot[4]),
 .M_H4_arqos(H_arqos[4]),
 .M_H4_arready(H_arready[4]),
 .M_H4_arsize(H_arsize[4]),
 .M_H4_arvalid(H_arvalid[4]),
 .M_H4_awaddr(H_awaddr[4]),
 .M_H4_awburst(H_awburst[4]),
 .M_H4_awcache(H_awcache[4]),
 .M_H4_awlen(H_awlen[4]),
 .M_H4_awlock(H_awlock[4]),
 .M_H4_awprot(H_awprot[4]),
 .M_H4_awqos(H_awqos[4]),
 .M_H4_awready(H_awready[4]),
 .M_H4_awsize(H_awsize[4]),
 .M_H4_awvalid(H_awvalid[4]),
 .M_H4_bready(H_bready[4]),
 .M_H4_bresp(H_bresp[4]),
 .M_H4_bvalid(H_bvalid[4]),
 .M_H4_rdata(H_rdata[4]),
 .M_H4_rlast(H_rlast[4]),
 .M_H4_rready(H_rready[4]),
 .M_H4_rresp(H_rresp[4]),
 .M_H4_rvalid(H_rvalid[4]),
 .M_H4_wdata(H_wdata[4]),
 .M_H4_wlast(H_wlast[4]),
 .M_H4_wready(H_wready[4]),
 .M_H4_wstrb(H_wstrb[4]),
 .M_H4_wvalid(H_wvalid[4]),
 .S_N5_araddr(N_araddr[5]),
 .S_N5_arburst(N_arburst[5]),
 .S_N5_arcache(N_arcache[5]),
 .S_N5_arid(N_arid[5]),
 .S_N5_arlen(N_arlen[5]),
 .S_N5_arlock(N_arlock[5]),
 .S_N5_arprot(N_arprot[5]),
 .S_N5_arqos(N_arqos[5]),
 .S_N5_arready(N_arready[5]),
 .S_N5_arsize(N_arsize[5]),
 .S_N5_arvalid(N_arvalid[5]),
 .S_N5_awaddr(N_awaddr[5]),
 .S_N5_awburst(N_awburst[5]),
 .S_N5_awcache(N_awcache[5]),
 .S_N5_awid(N_awid[5]),
 .S_N5_awlen(N_awlen[5]),
 .S_N5_awlock(N_awlock[5]),
 .S_N5_awprot(N_awprot[5]),
 .S_N5_awqos(N_awqos[5]),
 .S_N5_awready(N_awready[5]),
 .S_N5_awsize(N_awsize[5]),
 .S_N5_awvalid(N_awvalid[5]),
 .S_N5_bid(N_bid[5]),
 .S_N5_bready(N_bready[5]),
 .S_N5_bresp(N_bresp[5]),
 .S_N5_bvalid(N_bvalid[5]),
 .S_N5_rdata(N_rdata[5]),
 .S_N5_rid(N_rid[5]),
 .S_N5_rlast(N_rlast[5]),
 .S_N5_rready(N_rready[5]),
 .S_N5_rresp(N_rresp[5]),
 .S_N5_rvalid(N_rvalid[5]),
 .S_N5_wdata(N_wdata[5]),
 .S_N5_wlast(N_wlast[5]),
 .S_N5_wready(N_wready[5]),
 .S_N5_wstrb(N_wstrb[5]),
 .S_N5_wvalid(N_wvalid[5]),
 .M_H5_araddr(H_araddr[5]),
 .M_H5_arburst(H_arburst[5]),
 .M_H5_arcache(H_arcache[5]),
 .M_H5_arlen(H_arlen[5]),
 .M_H5_arlock(H_arlock[5]),
 .M_H5_arprot(H_arprot[5]),
 .M_H5_arqos(H_arqos[5]),
 .M_H5_arready(H_arready[5]),
 .M_H5_arsize(H_arsize[5]),
 .M_H5_arvalid(H_arvalid[5]),
 .M_H5_awaddr(H_awaddr[5]),
 .M_H5_awburst(H_awburst[5]),
 .M_H5_awcache(H_awcache[5]),
 .M_H5_awlen(H_awlen[5]),
 .M_H5_awlock(H_awlock[5]),
 .M_H5_awprot(H_awprot[5]),
 .M_H5_awqos(H_awqos[5]),
 .M_H5_awready(H_awready[5]),
 .M_H5_awsize(H_awsize[5]),
 .M_H5_awvalid(H_awvalid[5]),
 .M_H5_bready(H_bready[5]),
 .M_H5_bresp(H_bresp[5]),
 .M_H5_bvalid(H_bvalid[5]),
 .M_H5_rdata(H_rdata[5]),
 .M_H5_rlast(H_rlast[5]),
 .M_H5_rready(H_rready[5]),
 .M_H5_rresp(H_rresp[5]),
 .M_H5_rvalid(H_rvalid[5]),
 .M_H5_wdata(H_wdata[5]),
 .M_H5_wlast(H_wlast[5]),
 .M_H5_wready(H_wready[5]),
 .M_H5_wstrb(H_wstrb[5]),
 .M_H5_wvalid(H_wvalid[5]),
 .S_N6_araddr(N_araddr[6]),
 .S_N6_arburst(N_arburst[6]),
 .S_N6_arcache(N_arcache[6]),
 .S_N6_arid(N_arid[6]),
 .S_N6_arlen(N_arlen[6]),
 .S_N6_arlock(N_arlock[6]),
 .S_N6_arprot(N_arprot[6]),
 .S_N6_arqos(N_arqos[6]),
 .S_N6_arready(N_arready[6]),
 .S_N6_arsize(N_arsize[6]),
 .S_N6_arvalid(N_arvalid[6]),
 .S_N6_awaddr(N_awaddr[6]),
 .S_N6_awburst(N_awburst[6]),
 .S_N6_awcache(N_awcache[6]),
 .S_N6_awid(N_awid[6]),
 .S_N6_awlen(N_awlen[6]),
 .S_N6_awlock(N_awlock[6]),
 .S_N6_awprot(N_awprot[6]),
 .S_N6_awqos(N_awqos[6]),
 .S_N6_awready(N_awready[6]),
 .S_N6_awsize(N_awsize[6]),
 .S_N6_awvalid(N_awvalid[6]),
 .S_N6_bid(N_bid[6]),
 .S_N6_bready(N_bready[6]),
 .S_N6_bresp(N_bresp[6]),
 .S_N6_bvalid(N_bvalid[6]),
 .S_N6_rdata(N_rdata[6]),
 .S_N6_rid(N_rid[6]),
 .S_N6_rlast(N_rlast[6]),
 .S_N6_rready(N_rready[6]),
 .S_N6_rresp(N_rresp[6]),
 .S_N6_rvalid(N_rvalid[6]),
 .S_N6_wdata(N_wdata[6]),
 .S_N6_wlast(N_wlast[6]),
 .S_N6_wready(N_wready[6]),
 .S_N6_wstrb(N_wstrb[6]),
 .S_N6_wvalid(N_wvalid[6]),
 .M_H6_araddr(H_araddr[6]),
 .M_H6_arburst(H_arburst[6]),
 .M_H6_arcache(H_arcache[6]),
 .M_H6_arlen(H_arlen[6]),
 .M_H6_arlock(H_arlock[6]),
 .M_H6_arprot(H_arprot[6]),
 .M_H6_arqos(H_arqos[6]),
 .M_H6_arready(H_arready[6]),
 .M_H6_arsize(H_arsize[6]),
 .M_H6_arvalid(H_arvalid[6]),
 .M_H6_awaddr(H_awaddr[6]),
 .M_H6_awburst(H_awburst[6]),
 .M_H6_awcache(H_awcache[6]),
 .M_H6_awlen(H_awlen[6]),
 .M_H6_awlock(H_awlock[6]),
 .M_H6_awprot(H_awprot[6]),
 .M_H6_awqos(H_awqos[6]),
 .M_H6_awready(H_awready[6]),
 .M_H6_awsize(H_awsize[6]),
 .M_H6_awvalid(H_awvalid[6]),
 .M_H6_bready(H_bready[6]),
 .M_H6_bresp(H_bresp[6]),
 .M_H6_bvalid(H_bvalid[6]),
 .M_H6_rdata(H_rdata[6]),
 .M_H6_rlast(H_rlast[6]),
 .M_H6_rready(H_rready[6]),
 .M_H6_rresp(H_rresp[6]),
 .M_H6_rvalid(H_rvalid[6]),
 .M_H6_wdata(H_wdata[6]),
 .M_H6_wlast(H_wlast[6]),
 .M_H6_wready(H_wready[6]),
 .M_H6_wstrb(H_wstrb[6]),
 .M_H6_wvalid(H_wvalid[6]),
 .S_N7_araddr(N_araddr[7]),
 .S_N7_arburst(N_arburst[7]),
 .S_N7_arcache(N_arcache[7]),
 .S_N7_arid(N_arid[7]),
 .S_N7_arlen(N_arlen[7]),
 .S_N7_arlock(N_arlock[7]),
 .S_N7_arprot(N_arprot[7]),
 .S_N7_arqos(N_arqos[7]),
 .S_N7_arready(N_arready[7]),
 .S_N7_arsize(N_arsize[7]),
 .S_N7_arvalid(N_arvalid[7]),
 .S_N7_awaddr(N_awaddr[7]),
 .S_N7_awburst(N_awburst[7]),
 .S_N7_awcache(N_awcache[7]),
 .S_N7_awid(N_awid[7]),
 .S_N7_awlen(N_awlen[7]),
 .S_N7_awlock(N_awlock[7]),
 .S_N7_awprot(N_awprot[7]),
 .S_N7_awqos(N_awqos[7]),
 .S_N7_awready(N_awready[7]),
 .S_N7_awsize(N_awsize[7]),
 .S_N7_awvalid(N_awvalid[7]),
 .S_N7_bid(N_bid[7]),
 .S_N7_bready(N_bready[7]),
 .S_N7_bresp(N_bresp[7]),
 .S_N7_bvalid(N_bvalid[7]),
 .S_N7_rdata(N_rdata[7]),
 .S_N7_rid(N_rid[7]),
 .S_N7_rlast(N_rlast[7]),
 .S_N7_rready(N_rready[7]),
 .S_N7_rresp(N_rresp[7]),
 .S_N7_rvalid(N_rvalid[7]),
 .S_N7_wdata(N_wdata[7]),
 .S_N7_wlast(N_wlast[7]),
 .S_N7_wready(N_wready[7]),
 .S_N7_wstrb(N_wstrb[7]),
 .S_N7_wvalid(N_wvalid[7]),
 .M_H7_araddr(H_araddr[7]),
 .M_H7_arburst(H_arburst[7]),
 .M_H7_arcache(H_arcache[7]),
 .M_H7_arlen(H_arlen[7]),
 .M_H7_arlock(H_arlock[7]),
 .M_H7_arprot(H_arprot[7]),
 .M_H7_arqos(H_arqos[7]),
 .M_H7_arready(H_arready[7]),
 .M_H7_arsize(H_arsize[7]),
 .M_H7_arvalid(H_arvalid[7]),
 .M_H7_awaddr(H_awaddr[7]),
 .M_H7_awburst(H_awburst[7]),
 .M_H7_awcache(H_awcache[7]),
 .M_H7_awlen(H_awlen[7]),
 .M_H7_awlock(H_awlock[7]),
 .M_H7_awprot(H_awprot[7]),
 .M_H7_awqos(H_awqos[7]),
 .M_H7_awready(H_awready[7]),
 .M_H7_awsize(H_awsize[7]),
 .M_H7_awvalid(H_awvalid[7]),
 .M_H7_bready(H_bready[7]),
 .M_H7_bresp(H_bresp[7]),
 .M_H7_bvalid(H_bvalid[7]),
 .M_H7_rdata(H_rdata[7]),
 .M_H7_rlast(H_rlast[7]),
 .M_H7_rready(H_rready[7]),
 .M_H7_rresp(H_rresp[7]),
 .M_H7_rvalid(H_rvalid[7]),
 .M_H7_wdata(H_wdata[7]),
 .M_H7_wlast(H_wlast[7]),
 .M_H7_wready(H_wready[7]),
 .M_H7_wstrb(H_wstrb[7]),
 .M_H7_wvalid(H_wvalid[7]),
 .S_HOST_araddr(P_araddr),
 .S_HOST_arburst(P_arburst),
 .S_HOST_arcache(P_arcache),
 .S_HOST_arid(P_arid),
 .S_HOST_arlen(P_arlen),
 .S_HOST_arlock(P_arlock),
 .S_HOST_arprot(P_arprot),
 .S_HOST_arqos(P_arqos),
 .S_HOST_arready(P_arready),
 .S_HOST_arsize(P_arsize),
 .S_HOST_arvalid(P_arvalid),
 .S_HOST_awaddr(P_awaddr),
 .S_HOST_awburst(P_awburst),
 .S_HOST_awcache(P_awcache),
 .S_HOST_awid(P_awid),
 .S_HOST_awlen(P_awlen),
 .S_HOST_awlock(P_awlock),
 .S_HOST_awprot(P_awprot),
 .S_HOST_awqos(P_awqos),
 .S_HOST_awready(P_awready),
 .S_HOST_awsize(P_awsize),
 .S_HOST_awvalid(P_awvalid),
 .S_HOST_bid(P_bid),
 .S_HOST_bready(P_bready),
 .S_HOST_bresp(P_bresp),
 .S_HOST_bvalid(P_bvalid),
 .S_HOST_rdata(P_rdata),
 .S_HOST_rid(P_rid),
 .S_HOST_rlast(P_rlast),
 .S_HOST_rready(P_rready),
 .S_HOST_rresp(P_rresp),
 .S_HOST_rvalid(P_rvalid),
 .S_HOST_wdata(P_wdata),
 .S_HOST_wlast(P_wlast),
 .S_HOST_wready(P_wready),
 .S_HOST_wstrb(P_wstrb),
 .S_HOST_wvalid(P_wvalid)
);
f2_pcis_pipe host_pipe (
 .aclk(host_clk),
 .aresetn(aresetn),
 .s_axi_awregion(4'b0),
 .s_axi_arregion(4'b0),
 .m_axi_awregion(),
 .m_axi_arregion(),
 .s_axi_araddr(S_araddr),
 .m_axi_araddr(P_araddr),
 .s_axi_arburst(S_arburst),
 .m_axi_arburst(P_arburst),
 .s_axi_arcache(S_arcache),
 .m_axi_arcache(P_arcache),
 .s_axi_arid(S_arid),
 .m_axi_arid(P_arid),
 .s_axi_arlen(S_arlen),
 .m_axi_arlen(P_arlen),
 .s_axi_arlock(S_arlock),
 .m_axi_arlock(P_arlock),
 .s_axi_arprot(S_arprot),
 .m_axi_arprot(P_arprot),
 .s_axi_arqos(S_arqos),
 .m_axi_arqos(P_arqos),
 .s_axi_arready(S_arready),
 .m_axi_arready(P_arready),
 .s_axi_arsize(S_arsize),
 .m_axi_arsize(P_arsize),
 .s_axi_arvalid(S_arvalid),
 .m_axi_arvalid(P_arvalid),
 .s_axi_awaddr(S_awaddr),
 .m_axi_awaddr(P_awaddr),
 .s_axi_awburst(S_awburst),
 .m_axi_awburst(P_awburst),
 .s_axi_awcache(S_awcache),
 .m_axi_awcache(P_awcache),
 .s_axi_awid(S_awid),
 .m_axi_awid(P_awid),
 .s_axi_awlen(S_awlen),
 .m_axi_awlen(P_awlen),
 .s_axi_awlock(S_awlock),
 .m_axi_awlock(P_awlock),
 .s_axi_awprot(S_awprot),
 .m_axi_awprot(P_awprot),
 .s_axi_awqos(S_awqos),
 .m_axi_awqos(P_awqos),
 .s_axi_awready(S_awready),
 .m_axi_awready(P_awready),
 .s_axi_awsize(S_awsize),
 .m_axi_awsize(P_awsize),
 .s_axi_awvalid(S_awvalid),
 .m_axi_awvalid(P_awvalid),
 .s_axi_bid(S_bid),
 .m_axi_bid(P_bid),
 .s_axi_bready(S_bready),
 .m_axi_bready(P_bready),
 .s_axi_bresp(S_bresp),
 .m_axi_bresp(P_bresp),
 .s_axi_bvalid(S_bvalid),
 .m_axi_bvalid(P_bvalid),
 .s_axi_rdata(S_rdata),
 .m_axi_rdata(P_rdata),
 .s_axi_rid(S_rid),
 .m_axi_rid(P_rid),
 .s_axi_rlast(S_rlast),
 .m_axi_rlast(P_rlast),
 .s_axi_rready(S_rready),
 .m_axi_rready(P_rready),
 .s_axi_rresp(S_rresp),
 .m_axi_rresp(P_rresp),
 .s_axi_rvalid(S_rvalid),
 .m_axi_rvalid(P_rvalid),
 .s_axi_wdata(S_wdata),
 .m_axi_wdata(P_wdata),
 .s_axi_wlast(S_wlast),
 .m_axi_wlast(P_wlast),
 .s_axi_wready(S_wready),
 .m_axi_wready(P_wready),
 .s_axi_wstrb(S_wstrb),
 .m_axi_wstrb(P_wstrb),
 .s_axi_wvalid(S_wvalid),
 .m_axi_wvalid(P_wvalid)
);
logic [7:0] n_cvalid=0,n_cready,n_cwe=0,n_wvalid=0,n_wready,n_rvalid,n_err;
logic [7:0][24:0] n_caddr=0;
logic [7:0][511:0] n_wdata=0,n_rdata;
logic [7:0][63:0] n_wmask='1;
logic [7:0][15:0] n_wdone;
logic [511:0] expected [0:7][0:2047];
integer rq_addr[0:7][0:4095],issued[0:7],returned[0:7],consumed[0:7],peak[0:7];
integer bursts[0:7][0:4095],bt[0:7],bh[0:7],axi_out[0:7],axi_total[0:7],axi_ret[0:7];
integer epoch=0,core_cycle=0,host_b_count=0,host_r_count=0,reset_pending=0;
integer hbm_reads[0:7],hbm_writes[0:7],native_total[0:7];
function automatic [511:0] pattern(input integer pc,beat,salt);
 for(integer w=0;w<16;w=w+1)
  pattern[w*32+:32]=32'h913a0000 ^ (pc*32'h11771531) ^ (beat*32'h1021) ^ (w*32'h01010101) ^ salt;
endfunction
for(genvar p=0;p<8;p=p+1) begin:g_pc
 wire [33:0] local_aw,local_ar,hb_aw,hb_ar;
 wire model_arready;
 // Legal endpoint backpressure before AR acceptance. This is asserted only
 // for the reset fixture, with no previous read outstanding in the models.
 assign H_arready[p]=model_arready && !hold_model_ar;
 assign N_awaddr[p]=local_aw | (34'(p)<<29);
 assign N_araddr[p]=local_ar | (34'(p)<<29);
 otpu_native_axi #(.BASE_ADDR(34'd0)) native (
 .clk(core_clk),.rst(!aresetn),.n_cvalid(n_cvalid[p]),.n_cready(n_cready[p]),
 .n_cwe(n_cwe[p]),.n_caddr(n_caddr[p]),.n_wvalid(n_wvalid[p]),.n_wready(n_wready[p]),
 .n_wdata(n_wdata[p]),.n_wmask(n_wmask[p]),.n_rvalid(n_rvalid[p]),.n_rdata(n_rdata[p]),
 .n_wdone(n_wdone[p]),.n_err(n_err[p]),.m_awaddr(local_aw),.m_araddr(local_ar),

.m_awvalid(N_awvalid[p]),
 .m_awready(N_awready[p]),
 .m_wdata(N_wdata[p]),
 .m_wstrb(N_wstrb[p]),
 .m_wvalid(N_wvalid[p]),
 .m_wready(N_wready[p]),
 .m_bvalid(N_bvalid[p]),
 .m_bresp(N_bresp[p]),
 .m_bready(N_bready[p]),
 .m_arlen(N_arlen[p]),
 .m_arvalid(N_arvalid[p]),
 .m_arready(N_arready[p]),
 .m_rdata(N_rdata[p]),
 .m_rvalid(N_rvalid[p]),
 .m_rlast(N_rlast[p]),
 .m_rresp(N_rresp[p]),
 .m_rready(N_rready[p]));
assign N_arburst[p]=1;
assign N_arcache[p]=0;
assign N_arid[p]=0;
assign N_arlock[p]=0;
assign N_arprot[p]=0;
assign N_arqos[p]=0;
assign N_arsize[p]=6;
assign N_awburst[p]=1;
assign N_awcache[p]=0;
assign N_awid[p]=0;
assign N_awlen[p]=0;
assign N_awlock[p]=0;
assign N_awprot[p]=0;
assign N_awqos[p]=0;
assign N_awsize[p]=6;
assign N_wlast[p]=1;
otpu_hbm_geometry geometry(.s_awaddr(H_awaddr[p]),.s_araddr(H_araddr[p]),.m_awaddr(hb_aw),.m_araddr(hb_ar));
axi_hbm_model memory(.clk(hbm_clk),.rst_n(aresetn),.awaddr(hb_aw),.araddr(hb_ar),.awsize(3'd5),.arsize(3'd5),
.awlen(H_awlen[p]),
 .arlen(H_arlen[p]),
 .awvalid(H_awvalid[p]),
 .arvalid(H_arvalid[p] && !hold_model_ar),
 .awready(H_awready[p]),
 .arready(model_arready),
 .wdata(H_wdata[p]),
 .wstrb(H_wstrb[p]),
 .wlast(H_wlast[p]),
 .wvalid(H_wvalid[p]),
 .wready(H_wready[p]),
 .bvalid(H_bvalid[p]),
 .bresp(H_bresp[p]),
 .bready(H_bready[p]),
 .rdata(H_rdata[p]),
 .rlast(H_rlast[p]),
 .rvalid(H_rvalid[p]),
 .rresp(H_rresp[p]),
 .rready(H_rready[p]));
// Check FULL addresses before the frozen model indexes only [16:5]. No
// transaction may silently alias a wrong physical window or a model-size wrap.
always @(posedge hbm_clk) if(aresetn) begin
 if(H_awvalid[p] && H_awready[p]) begin
  if(((^{H_awaddr[p],H_awlen[p],H_awsize[p],H_awburst[p]})===1'bx) || (H_awaddr[p]>>29)!=p || H_awaddr[p][28:17]!=0)
   $fatal(1,"HBM write wrong full address pc=%0d addr=%h",p,H_awaddr[p]);
  if((H_awlen[p]!=0 && H_awburst[p]!=1) || (H_awlen[p]!=0 && H_awsize[p]!=5) ||
     ({1'b0,hb_aw[11:0]}+((int'(H_awlen[p])+1)*32))>4096)
   $fatal(1,"HBM illegal write geometry pc=%0d",p);
  hbm_writes[p]=hbm_writes[p]+int'(H_awlen[p])+1;
 end
 if(H_arvalid[p] && H_arready[p]) begin
  if(((^{H_araddr[p],H_arlen[p],H_arsize[p],H_arburst[p]})===1'bx) || (H_araddr[p]>>29)!=p || H_araddr[p][28:17]!=0)
   $fatal(1,"HBM read wrong full address pc=%0d addr=%h",p,H_araddr[p]);
  if((H_arlen[p]!=0 && H_arburst[p]!=1) || (H_arlen[p]!=0 && H_arsize[p]!=5) ||
     ({1'b0,hb_ar[11:0]}+((int'(H_arlen[p])+1)*32))>4096)
   $fatal(1,"HBM illegal read geometry pc=%0d",p);
  hbm_reads[p]=hbm_reads[p]+int'(H_arlen[p])+1;
 end
end
// Independent AXI burst scoreboard, not the adapter's outstanding register.
always @(posedge core_clk) begin
 if(!aresetn) begin bt[p]=0;bh[p]=0;axi_out[p]=0;axi_total[p]=0;axi_ret[p]=0;end
 else begin
  if(N_rready[p]!==1'b1) $fatal(1,"native RREADY must remain high pc=%0d",p);
  if(N_arvalid[p] && N_arready[p]) begin
   if(((^{N_araddr[p],N_arlen[p]})===1'bx) || N_araddr[p]>>29!=p || N_araddr[p][5:0]!=0 || N_arlen[p]>15 ||
      (int'(N_araddr[p][11:0])+64*(int'(N_arlen[p])+1))>4096)
    $fatal(1,"native burst address/4KiB violation pc=%0d",p);
   bursts[p][bt[p]]=int'(N_arlen[p])+1;bt[p]=bt[p]+1;
   axi_out[p]=axi_out[p]+int'(N_arlen[p])+1;
   axi_total[p]=axi_total[p]+int'(N_arlen[p])+1;
   if(axi_out[p]>32) $fatal(1,"native AXI exceeds reserved receive capacity pc=%0d",p);
  end
  if(N_rvalid[p] && N_rready[p]) begin
   if(bh[p]>=bt[p] || axi_out[p]<=0 || N_rresp[p]!==2'b00 || N_rid[p]!==1'b0)
    $fatal(1,"unsolicited/error native AXI beat pc=%0d",p);
   if(N_rlast[p] !== (bursts[p][bh[p]]==1)) $fatal(1,"native RLAST conservation pc=%0d",p);
   bursts[p][bh[p]]=bursts[p][bh[p]]-1;
   if(bursts[p][bh[p]]==0)bh[p]=bh[p]+1;
   axi_out[p]=axi_out[p]-1;axi_ret[p]=axi_ret[p]+1;
  end
  if(n_rvalid[p] !== (N_rvalid[p]&&N_rready[p]) || (n_rvalid[p] && n_rdata[p]!==N_rdata[p]))
   $fatal(1,"native response seam mismatch pc=%0d",p);
 end
end
end
// Reservations are released by a deliberately delayed downstream consumer,
// not upon R arrival. This models the kernel's finite RD32 return RAM.
always @(posedge core_clk) begin
 if(!aresetn) begin
  core_cycle=0;
  for(integer p=0;p<8;p=p+1)begin issued[p]=0;returned[p]=0;consumed[p]=0;peak[p]=0;end
 end else begin
  core_cycle=core_cycle+1;
  for(integer p=0;p<8;p=p+1) begin
   if(n_err[p]!==1'b0)$fatal(1,"native error pc=%0d",p);
   if(n_cvalid[p] && n_cready[p] && !n_cwe[p]) begin
    if(n_caddr[p]>=2048 || issued[p]>=4096)$fatal(1,"native test queue range");
    rq_addr[p][issued[p]]=n_caddr[p];issued[p]=issued[p]+1;
    if(issued[p]-consumed[p]>32)$fatal(1,"RD32 reservation overflow pc=%0d",p);
    if(issued[p]-consumed[p]>peak[p])peak[p]=issued[p]-consumed[p];
    native_total[p]=native_total[p]+1;
   end
   if(n_rvalid[p])begin
    if(returned[p]>=issued[p])$fatal(1,"unsolicited native return pc=%0d",p);
    if(n_rdata[p]!==expected[p][rq_addr[p][returned[p]]])
     $fatal(1,"native byte mismatch pc=%0d index=%0d addr=%0d",p,returned[p],rq_addr[p][returned[p]]);
    returned[p]=returned[p]+1;
   end
   // Long pauses fill all32 slots; otherwise drain once every three cycles.
   if(returned[p]>consumed[p] && core_cycle%97>55 && (core_cycle+p)%3==0)consumed[p]=consumed[p]+1;
  end
 end
end

// Ready/valid stability under host response stalls, including all16 ID bits.
logic held_r=0,held_b=0;
logic [530:0] held_r_bits;
logic [17:0] held_b_bits;
always @(posedge host_clk) begin
 if(!aresetn)begin held_r=0;held_b=0;end
 else begin
  if(held_r && (!S_rvalid || {S_rid,S_rlast,S_rresp,S_rdata}!==held_r_bits))$fatal(1,"host R changed while stalled");
  if(held_b && (!S_bvalid || {S_bid,S_bresp}!==held_b_bits))$fatal(1,"host B changed while stalled");
  held_r=S_rvalid&&!S_rready;held_b=S_bvalid&&!S_bready;
  held_r_bits={S_rid,S_rlast,S_rresp,S_rdata};held_b_bits={S_bid,S_bresp};
  if(S_bvalid&&S_bready)host_b_count=host_b_count+1;
  if(S_rvalid&&S_rready)host_r_count=host_r_count+1;
  if(S_arvalid&&S_arready && (int'(S_araddr[11:0])+(int'(S_arlen)+1)*(1<<S_arsize))>4096)$fatal(1,"test issued illegal 4KiB AR");
  if(S_awvalid&&S_awready && (int'(S_awaddr[11:0])+(int'(S_awlen)+1)*(1<<S_awsize))>4096)$fatal(1,"test issued illegal 4KiB AW");
 end
end

task automatic native_write(input integer pc,addr,input logic[511:0] data);
 expected[pc][addr]=data;
 @(negedge core_clk);n_cvalid[pc]=1;n_cwe[pc]=1;n_caddr[pc]=addr;
 n_wvalid[pc]=1;n_wdata[pc]=data;n_wmask[pc]='1;
 fork
  begin do @(posedge core_clk);while(!n_cready[pc]);@(negedge core_clk);n_cvalid[pc]=0;end
  begin do @(posedge core_clk);while(!n_wready[pc]);@(negedge core_clk);n_wvalid[pc]=0;end
 join
endtask

task automatic native_read(input integer pc,addr);
 @(negedge core_clk);
 while(issued[pc]-consumed[pc]>=32)@(negedge core_clk);
 n_cvalid[pc]=1;n_cwe[pc]=0;n_caddr[pc]=addr;
 do @(posedge core_clk);while(!n_cready[pc]);
 @(negedge core_clk);n_cvalid[pc]=0;
endtask

task automatic drain_native;
 bit done;
 do begin
  @(negedge core_clk);done=1;
  for(integer p=0;p<8;p=p+1)if(issued[p]!=consumed[p])done=0;
 end while(!done);
 for(integer p=0;p<8;p=p+1)
  if(axi_out[p]!=0 || bt[p]!=bh[p] || axi_total[p]!=issued[p] || axi_ret[p]!=returned[p])
   $fatal(1,"native burst conservation final pc=%0d",p);
endtask

task automatic host_send_write(input integer pc,addr,sz,count,input logic[15:0] id,input integer salt,mask_kind);
 if((addr%4096)+count*(1<<sz)>4096)$fatal(1,"illegal host test write");
 @(negedge host_clk);S_awaddr=(64'(pc)<<29)+addr;S_awsize=sz;S_awlen=count-1;S_awburst=1;S_awid=id;S_awvalid=1;
 fork
  begin do @(posedge host_clk);while(!S_awready);@(negedge host_clk);S_awvalid=0;end
  begin
   for(integer i=0;i<count;i=i+1)begin
    @(negedge host_clk);S_wvalid=1;S_wlast=(i==count-1);S_wdata=pattern(pc,(addr+i*(1<<sz))/64,salt);S_wstrb=0;
    for(integer b=0;b<(1<<sz);b=b+1)begin
     if(mask_kind==0 || (mask_kind==1 && b%3==1))begin
      S_wstrb[(addr+i*(1<<sz))%64+b]=1;
      expected[pc][(addr+i*(1<<sz))/64][8*((addr+i*(1<<sz))%64+b)+:8]=S_wdata[8*((addr+i*(1<<sz))%64+b)+:8];
     end
    end
    do @(posedge host_clk);while(!S_wready);
    @(negedge host_clk);S_wvalid=0;
   end
  end
 join
endtask

task automatic host_b(input logic[15:0] id);
 @(negedge host_clk);S_bready=0;repeat(7)@(negedge host_clk);S_bready=1;
 do @(posedge host_clk);while(!S_bvalid);
 if(S_bresp!==2'b00 || S_bid!==id)$fatal(1,"host B response/id expected=%h got=%h",id,S_bid);
 @(negedge host_clk);S_bready=0;
endtask

task automatic host_ar(input integer pc,addr,sz,count,input logic[15:0] id);
 @(negedge host_clk);S_araddr=(64'(pc)<<29)+addr;S_arsize=sz;S_arlen=count-1;S_arburst=1;S_arid=id;S_arvalid=1;
 do @(posedge host_clk);while(!S_arready);
 @(negedge host_clk);S_arvalid=0;
endtask

task automatic host_r(input integer pc,addr,sz,count,input logic[15:0] id);
 for(integer i=0;i<count;i=i+1)begin
  @(negedge host_clk);S_rready=0;repeat(1+((i+seed)%7))@(negedge host_clk);S_rready=1;
  do @(posedge host_clk);while(!S_rvalid);
  if(S_rresp!==2'b00 || S_rid!==id || S_rlast!==(i==count-1))$fatal(1,"host R response/id/last");
  for(integer b=0;b<(1<<sz);b=b+1)
   if(S_rdata[8*((addr+i*(1<<sz))%64+b)+:8]!==expected[pc][(addr+i*(1<<sz))/64][8*((addr+i*(1<<sz))%64+b)+:8])
    $fatal(1,"host byte mismatch pc=%0d addr=%0d beat=%0d byte=%0d",pc,addr,i,b);
  @(negedge host_clk);S_rready=0;
 end
endtask

// AXI may reorder DIFFERENT IDs. These collectors accept either order but
// reject duplicate IDs and compare each response against its actual PC data.
task automatic host_b_pair(input logic[15:0] id0,id1);
 bit seen0,seen1;seen0=0;seen1=0;
 repeat(2)begin
  @(negedge host_clk);S_bready=0;repeat(9)@(negedge host_clk);S_bready=1;
  do @(posedge host_clk);while(!S_bvalid);
  if(S_bresp!==2'b00)$fatal(1,"paired host B error/unknown");
  if(S_bid===id0 && !seen0)seen0=1;
  else if(S_bid===id1 && !seen1)seen1=1;
  else $fatal(1,"paired host B duplicate/wrong ID");
  @(negedge host_clk);S_bready=0;
 end
endtask

task automatic host_r_pair(input integer pc0,pc1,addr,input logic[15:0] id0,id1);
 bit seen0,seen1;integer pc;seen0=0;seen1=0;
 repeat(2)begin
  @(negedge host_clk);S_rready=0;repeat(5)@(negedge host_clk);S_rready=1;
  do @(posedge host_clk);while(!S_rvalid);
  if(S_rresp!==2'b00 || S_rlast!==1'b1)$fatal(1,"paired host R error/unknown/last");
  if(S_rid===id0 && !seen0)begin seen0=1;pc=pc0;end
  else if(S_rid===id1 && !seen1)begin seen1=1;pc=pc1;end
  else $fatal(1,"paired host R duplicate/wrong ID");
  if(S_rdata!==expected[pc][addr/64])$fatal(1,"paired host R PC/data mismatch");
  @(negedge host_clk);S_rready=0;
 end
endtask

task automatic host_checks;
 for(integer p=0;p<8;p=p+1)begin
  // All byte lanes with legal aligned 1/2/4/8-byte requests.
  for(integer sz=0;sz<4;sz=sz+1)begin
   for(integer off=0;off<64;off=off+(1<<sz))begin
    host_send_write(p,16384+off,sz,1,16'h8023+p,seed+off+sz,0);host_b(16'h8023+p);
    host_ar(p,16384+off,sz,1,16'hf025+p);host_r(p,16384+off,sz,1,16'hf025+p);
   end
  end
  // Sparse and zero-strobe writes preserve untouched bytes.
  host_send_write(p,16384,6,1,16'hb071+p,seed+91,1);host_b(16'hb071+p);
  host_send_write(p,16384,6,1,16'hb171+p,seed+92,2);host_b(16'hb171+p);
  host_ar(p,16384,6,1,16'hc027+p);host_r(p,16384,6,1,16'hc027+p);
  // Legal16-beat burst ends at4KiB; separate16-beat transaction starts next page.
  host_send_write(p,19456,6,16,16'hbb71+p,seed+3,0);host_b(16'hbb71+p);
  host_send_write(p,20480,6,16,16'hbb81+p,seed+4,0);host_b(16'hbb81+p);
  host_ar(p,19456,6,16,16'hcc27+p);host_r(p,19456,6,16,16'hcc27+p);
  host_ar(p,20480,6,16,16'hcc37+p);host_r(p,20480,6,16,16'hcc37+p);
 end
 // Two outstanding reads on the SAME full ID, different destinations. The
 // data comparison checks same-ID response ordering across physical PCs.
 host_ar(0,16384,6,1,16'hdabc);host_ar(7,16384,6,1,16'hdabc);
 host_r(0,16384,6,1,16'hdabc);host_r(7,16384,6,1,16'hdabc);
 // Two different full IDs simultaneously outstanding in both directions.
 host_send_write(2,24576,6,1,16'h10ab,seed+11,0);
 host_send_write(6,24576,6,1,16'hec01,seed+12,0);
 host_b_pair(16'h10ab,16'hec01);
 host_ar(2,24576,6,1,16'he012);host_ar(6,24576,6,1,16'hf012);
 host_r_pair(2,6,24576,16'he012,16'hf012);
endtask

task automatic native_stream(input integer pc);
 for(integer repeat_i=0;repeat_i<6;repeat_i=repeat_i+1)
  for(integer a=48;a<144;a=a+1)native_read(pc,a);
endtask

task automatic reset_reads(input integer pc);
 for(integer a=48;a<64;a=a+1)native_read(pc,a);
 wait(axi_total[pc]==16);
endtask

task automatic reset_fabric;
 @(negedge core_clk);aresetn=0;hold_model_ar=0;n_cvalid=0;n_wvalid=0;S_arvalid=0;S_awvalid=0;S_wvalid=0;S_bready=0;S_rready=0;
 repeat(100)@(negedge core_clk);aresetn=1;
 repeat(100)@(negedge core_clk);
endtask

initial begin
 for(integer p=0;p<8;p=p+1)begin hbm_reads[p]=0;hbm_writes[p]=0;native_total[p]=0;end
 reset_fabric();
 for(integer p=0;p<8;p=p+1)begin
  for(integer a=48;a<144;a=a+1)native_write(p,a,pattern(p,a,seed));
  native_write(p,256,pattern(p,256,seed));
 end
 for(integer p=0;p<8;p=p+1)wait(n_wdone[p]==97);
 // Block model AR acceptance so each native burst has entered the actual
 // vendor fabric and remains pending when the shared fabric reset asserts.
 @(negedge hbm_clk);hold_model_ar=1;
 host_ar(0,48*64,6,16,16'h8123);
 fork
  reset_reads(0);reset_reads(1);reset_reads(2);reset_reads(3);
  reset_reads(4);reset_reads(5);reset_reads(6);reset_reads(7);
 join
 @(negedge core_clk);
 for(integer p=0;p<8;p=p+1)if(issued[p]==16 && returned[p]==0 && axi_out[p]==16)reset_pending=reset_pending+1;
 if(reset_pending!=8)$fatal(1,"reset failed to exercise pending native fabric bursts");
 reset_fabric();epoch=1;
 // Reinitialize after reset; memory persistence is not required by the test.
 for(integer p=0;p<8;p=p+1)begin
  for(integer a=48;a<144;a=a+1)native_write(p,a,pattern(p,a,seed+101));
  native_write(p,256,pattern(p,256,seed+101));
 end
 for(integer p=0;p<8;p=p+1)wait(n_wdone[p]==97);
 fork
  begin: native_traffic
   fork
    native_stream(0);native_stream(1);native_stream(2);native_stream(3);
    native_stream(4);native_stream(5);native_stream(6);native_stream(7);
   join
   drain_native();
  end
  begin: host_traffic host_checks();end
 join
 // Host writes become visible on all native interfaces, all512 bits checked.
 for(integer p=0;p<8;p=p+1)begin
  native_read(p,256);
  for(integer a=304;a<336;a=a+1)native_read(p,a);
 end
 native_read(2,384);native_read(6,384);drain_native();
 // Unmapped host address must DECERR, not wrap into a model's low bits.
 host_ar(8,0,2,1,16'h4567);
 @(negedge host_clk);S_rready=1;
 do @(posedge host_clk);while(!S_rvalid);
 if(S_rresp!==2'b11 || S_rid!==16'h4567 || S_rlast!==1'b1)$fatal(1,"unmapped address did not DECERR");
 @(negedge host_clk);S_rready=0;
 for(integer p=0;p<8;p=p+1)begin
  if(hbm_reads[p]==0 || hbm_writes[p]==0 || native_total[p]<600 || peak[p]!=32)$fatal(1,"missing per-PC coverage");
  $display("C1_PC pc=%0d accepted=%0d returned=%0d consumed=%0d axi_beats=%0d peak=%0d hbm_reads=%0d hbm_writes=%0d",p,issued[p],returned[p],consumed[p],axi_total[p],peak[p],hbm_reads[p],hbm_writes[p]);
 end
 if(host_b_count!=994 || host_r_count!=1229)$fatal(1,"host count conservation B=%0d R=%0d",host_b_count,host_r_count);
 $display("C1_ALEX_FABRIC_PASS seed=%0d phase=%0d pcs=8 reset_pending=%0d host_b=%0d host_r=%0d",seed,phase,reset_pending,host_b_count,host_r_count);
 $finish;
end
initial begin #5000000;$fatal(1,"C1 vendor fabric timeout");end
endmodule
