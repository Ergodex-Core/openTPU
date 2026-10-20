`timescale 1ns/1ps
module tb_f2_mem;
logic core_clk=0,host_clk=0,hbm_clk=0,aresetn=0;
always #2.5 core_clk=~core_clk;
always #2 host_clk=~host_clk;
always #1.111 hbm_clk=~hbm_clk;
logic [33:0] M_H0_araddr;
logic [1:0] M_H0_arburst;
logic [3:0] M_H0_arcache;
logic [3:0] M_H0_arlen;
logic [1:0] M_H0_arlock;
logic [2:0] M_H0_arprot;
logic [3:0] M_H0_arqos;
logic  M_H0_arready;
logic [2:0] M_H0_arsize;
logic  M_H0_arvalid;
logic [33:0] M_H0_awaddr;
logic [1:0] M_H0_awburst;
logic [3:0] M_H0_awcache;
logic [3:0] M_H0_awlen;
logic [1:0] M_H0_awlock;
logic [2:0] M_H0_awprot;
logic [3:0] M_H0_awqos;
logic  M_H0_awready;
logic [2:0] M_H0_awsize;
logic  M_H0_awvalid;
logic  M_H0_bready;
logic [1:0] M_H0_bresp;
logic  M_H0_bvalid;
logic [255:0] M_H0_rdata;
logic  M_H0_rlast;
logic  M_H0_rready;
logic [1:0] M_H0_rresp;
logic  M_H0_rvalid;
logic [255:0] M_H0_wdata;
logic  M_H0_wlast;
logic  M_H0_wready;
logic [31:0] M_H0_wstrb;
logic  M_H0_wvalid;
logic [33:0] M_H4_araddr;
logic [1:0] M_H4_arburst;
logic [3:0] M_H4_arcache;
logic [3:0] M_H4_arlen;
logic [1:0] M_H4_arlock;
logic [2:0] M_H4_arprot;
logic [3:0] M_H4_arqos;
logic  M_H4_arready;
logic [2:0] M_H4_arsize;
logic  M_H4_arvalid;
logic [33:0] M_H4_awaddr;
logic [1:0] M_H4_awburst;
logic [3:0] M_H4_awcache;
logic [3:0] M_H4_awlen;
logic [1:0] M_H4_awlock;
logic [2:0] M_H4_awprot;
logic [3:0] M_H4_awqos;
logic  M_H4_awready;
logic [2:0] M_H4_awsize;
logic  M_H4_awvalid;
logic  M_H4_bready;
logic [1:0] M_H4_bresp;
logic  M_H4_bvalid;
logic [255:0] M_H4_rdata;
logic  M_H4_rlast;
logic  M_H4_rready;
logic [1:0] M_H4_rresp;
logic  M_H4_rvalid;
logic [255:0] M_H4_wdata;
logic  M_H4_wlast;
logic  M_H4_wready;
logic [31:0] M_H4_wstrb;
logic  M_H4_wvalid;
logic [63:0] S_HOST_araddr;
logic [1:0] S_HOST_arburst;
logic [3:0] S_HOST_arcache;
logic [15:0] S_HOST_arid;
logic [7:0] S_HOST_arlen;
logic [0:0] S_HOST_arlock;
logic [2:0] S_HOST_arprot;
logic [3:0] S_HOST_arqos;
logic  S_HOST_arready;
logic [2:0] S_HOST_arsize;
logic  S_HOST_arvalid;
logic [63:0] S_HOST_awaddr;
logic [1:0] S_HOST_awburst;
logic [3:0] S_HOST_awcache;
logic [15:0] S_HOST_awid;
logic [7:0] S_HOST_awlen;
logic [0:0] S_HOST_awlock;
logic [2:0] S_HOST_awprot;
logic [3:0] S_HOST_awqos;
logic  S_HOST_awready;
logic [2:0] S_HOST_awsize;
logic  S_HOST_awvalid;
logic [15:0] S_HOST_bid;
logic  S_HOST_bready;
logic [1:0] S_HOST_bresp;
logic  S_HOST_bvalid;
logic [511:0] S_HOST_rdata;
logic [15:0] S_HOST_rid;
logic  S_HOST_rlast;
logic  S_HOST_rready;
logic [1:0] S_HOST_rresp;
logic  S_HOST_rvalid;
logic [511:0] S_HOST_wdata;
logic  S_HOST_wlast;
logic  S_HOST_wready;
logic [63:0] S_HOST_wstrb;
logic  S_HOST_wvalid;
logic [33:0] S_N0_araddr;
logic [1:0] S_N0_arburst;
logic [3:0] S_N0_arcache;
logic [0:0] S_N0_arid;
logic [7:0] S_N0_arlen;
logic [0:0] S_N0_arlock;
logic [2:0] S_N0_arprot;
logic [3:0] S_N0_arqos;
logic  S_N0_arready;
logic [2:0] S_N0_arsize;
logic  S_N0_arvalid;
logic [33:0] S_N0_awaddr;
logic [1:0] S_N0_awburst;
logic [3:0] S_N0_awcache;
logic [0:0] S_N0_awid;
logic [7:0] S_N0_awlen;
logic [0:0] S_N0_awlock;
logic [2:0] S_N0_awprot;
logic [3:0] S_N0_awqos;
logic  S_N0_awready;
logic [2:0] S_N0_awsize;
logic  S_N0_awvalid;
logic [0:0] S_N0_bid;
logic  S_N0_bready;
logic [1:0] S_N0_bresp;
logic  S_N0_bvalid;
logic [511:0] S_N0_rdata;
logic [0:0] S_N0_rid;
logic  S_N0_rlast;
logic  S_N0_rready;
logic [1:0] S_N0_rresp;
logic  S_N0_rvalid;
logic [511:0] S_N0_wdata;
logic  S_N0_wlast;
logic  S_N0_wready;
logic [63:0] S_N0_wstrb;
logic  S_N0_wvalid;
logic [33:0] S_N1_araddr;
logic [1:0] S_N1_arburst;
logic [3:0] S_N1_arcache;
logic [0:0] S_N1_arid;
logic [7:0] S_N1_arlen;
logic [0:0] S_N1_arlock;
logic [2:0] S_N1_arprot;
logic [3:0] S_N1_arqos;
logic  S_N1_arready;
logic [2:0] S_N1_arsize;
logic  S_N1_arvalid;
logic [33:0] S_N1_awaddr;
logic [1:0] S_N1_awburst;
logic [3:0] S_N1_awcache;
logic [0:0] S_N1_awid;
logic [7:0] S_N1_awlen;
logic [0:0] S_N1_awlock;
logic [2:0] S_N1_awprot;
logic [3:0] S_N1_awqos;
logic  S_N1_awready;
logic [2:0] S_N1_awsize;
logic  S_N1_awvalid;
logic [0:0] S_N1_bid;
logic  S_N1_bready;
logic [1:0] S_N1_bresp;
logic  S_N1_bvalid;
logic [511:0] S_N1_rdata;
logic [0:0] S_N1_rid;
logic  S_N1_rlast;
logic  S_N1_rready;
logic [1:0] S_N1_rresp;
logic  S_N1_rvalid;
logic [511:0] S_N1_wdata;
logic  S_N1_wlast;
logic  S_N1_wready;
logic [63:0] S_N1_wstrb;
logic  S_N1_wvalid;
wire [63:0] P_HOST_araddr;
wire [1:0] P_HOST_arburst;
wire [3:0] P_HOST_arcache;
wire [15:0] P_HOST_arid;
wire [7:0] P_HOST_arlen;
wire [0:0] P_HOST_arlock;
wire [2:0] P_HOST_arprot;
wire [3:0] P_HOST_arqos;
wire  P_HOST_arready;
wire [2:0] P_HOST_arsize;
wire  P_HOST_arvalid;
wire [63:0] P_HOST_awaddr;
wire [1:0] P_HOST_awburst;
wire [3:0] P_HOST_awcache;
wire [15:0] P_HOST_awid;
wire [7:0] P_HOST_awlen;
wire [0:0] P_HOST_awlock;
wire [2:0] P_HOST_awprot;
wire [3:0] P_HOST_awqos;
wire  P_HOST_awready;
wire [2:0] P_HOST_awsize;
wire  P_HOST_awvalid;
wire [15:0] P_HOST_bid;
wire  P_HOST_bready;
wire [1:0] P_HOST_bresp;
wire  P_HOST_bvalid;
wire [511:0] P_HOST_rdata;
wire [15:0] P_HOST_rid;
wire  P_HOST_rlast;
wire  P_HOST_rready;
wire [1:0] P_HOST_rresp;
wire  P_HOST_rvalid;
wire [511:0] P_HOST_wdata;
wire  P_HOST_wlast;
wire  P_HOST_wready;
wire [63:0] P_HOST_wstrb;
wire  P_HOST_wvalid;
f2_mem_wrapper dut (.M_H0_araddr(M_H0_araddr),.M_H0_arburst(M_H0_arburst),.M_H0_arcache(M_H0_arcache),.M_H0_arlen(M_H0_arlen),.M_H0_arlock(M_H0_arlock),.M_H0_arprot(M_H0_arprot),.M_H0_arqos(M_H0_arqos),.M_H0_arready(M_H0_arready),.M_H0_arsize(M_H0_arsize),.M_H0_arvalid(M_H0_arvalid),.M_H0_awaddr(M_H0_awaddr),.M_H0_awburst(M_H0_awburst),.M_H0_awcache(M_H0_awcache),.M_H0_awlen(M_H0_awlen),.M_H0_awlock(M_H0_awlock),.M_H0_awprot(M_H0_awprot),.M_H0_awqos(M_H0_awqos),.M_H0_awready(M_H0_awready),.M_H0_awsize(M_H0_awsize),.M_H0_awvalid(M_H0_awvalid),.M_H0_bready(M_H0_bready),.M_H0_bresp(M_H0_bresp),.M_H0_bvalid(M_H0_bvalid),.M_H0_rdata(M_H0_rdata),.M_H0_rlast(M_H0_rlast),.M_H0_rready(M_H0_rready),.M_H0_rresp(M_H0_rresp),.M_H0_rvalid(M_H0_rvalid),.M_H0_wdata(M_H0_wdata),.M_H0_wlast(M_H0_wlast),.M_H0_wready(M_H0_wready),.M_H0_wstrb(M_H0_wstrb),.M_H0_wvalid(M_H0_wvalid),.M_H4_araddr(M_H4_araddr),.M_H4_arburst(M_H4_arburst),.M_H4_arcache(M_H4_arcache),.M_H4_arlen(M_H4_arlen),.M_H4_arlock(M_H4_arlock),.M_H4_arprot(M_H4_arprot),.M_H4_arqos(M_H4_arqos),.M_H4_arready(M_H4_arready),.M_H4_arsize(M_H4_arsize),.M_H4_arvalid(M_H4_arvalid),.M_H4_awaddr(M_H4_awaddr),.M_H4_awburst(M_H4_awburst),.M_H4_awcache(M_H4_awcache),.M_H4_awlen(M_H4_awlen),.M_H4_awlock(M_H4_awlock),.M_H4_awprot(M_H4_awprot),.M_H4_awqos(M_H4_awqos),.M_H4_awready(M_H4_awready),.M_H4_awsize(M_H4_awsize),.M_H4_awvalid(M_H4_awvalid),.M_H4_bready(M_H4_bready),.M_H4_bresp(M_H4_bresp),.M_H4_bvalid(M_H4_bvalid),.M_H4_rdata(M_H4_rdata),.M_H4_rlast(M_H4_rlast),.M_H4_rready(M_H4_rready),.M_H4_rresp(M_H4_rresp),.M_H4_rvalid(M_H4_rvalid),.M_H4_wdata(M_H4_wdata),.M_H4_wlast(M_H4_wlast),.M_H4_wready(M_H4_wready),.M_H4_wstrb(M_H4_wstrb),.M_H4_wvalid(M_H4_wvalid),.S_HOST_araddr(P_HOST_araddr),.S_HOST_arburst(P_HOST_arburst),.S_HOST_arcache(P_HOST_arcache),.S_HOST_arid(P_HOST_arid),.S_HOST_arlen(P_HOST_arlen),.S_HOST_arlock(P_HOST_arlock),.S_HOST_arprot(P_HOST_arprot),.S_HOST_arqos(P_HOST_arqos),.S_HOST_arready(P_HOST_arready),.S_HOST_arsize(P_HOST_arsize),.S_HOST_arvalid(P_HOST_arvalid),.S_HOST_awaddr(P_HOST_awaddr),.S_HOST_awburst(P_HOST_awburst),.S_HOST_awcache(P_HOST_awcache),.S_HOST_awid(P_HOST_awid),.S_HOST_awlen(P_HOST_awlen),.S_HOST_awlock(P_HOST_awlock),.S_HOST_awprot(P_HOST_awprot),.S_HOST_awqos(P_HOST_awqos),.S_HOST_awready(P_HOST_awready),.S_HOST_awsize(P_HOST_awsize),.S_HOST_awvalid(P_HOST_awvalid),.S_HOST_bid(P_HOST_bid),.S_HOST_bready(P_HOST_bready),.S_HOST_bresp(P_HOST_bresp),.S_HOST_bvalid(P_HOST_bvalid),.S_HOST_rdata(P_HOST_rdata),.S_HOST_rid(P_HOST_rid),.S_HOST_rlast(P_HOST_rlast),.S_HOST_rready(P_HOST_rready),.S_HOST_rresp(P_HOST_rresp),.S_HOST_rvalid(P_HOST_rvalid),.S_HOST_wdata(P_HOST_wdata),.S_HOST_wlast(P_HOST_wlast),.S_HOST_wready(P_HOST_wready),.S_HOST_wstrb(P_HOST_wstrb),.S_HOST_wvalid(P_HOST_wvalid),.S_N0_araddr(S_N0_araddr),.S_N0_arburst(S_N0_arburst),.S_N0_arcache(S_N0_arcache),.S_N0_arid(S_N0_arid),.S_N0_arlen(S_N0_arlen),.S_N0_arlock(S_N0_arlock),.S_N0_arprot(S_N0_arprot),.S_N0_arqos(S_N0_arqos),.S_N0_arready(S_N0_arready),.S_N0_arsize(S_N0_arsize),.S_N0_arvalid(S_N0_arvalid),.S_N0_awaddr(S_N0_awaddr),.S_N0_awburst(S_N0_awburst),.S_N0_awcache(S_N0_awcache),.S_N0_awid(S_N0_awid),.S_N0_awlen(S_N0_awlen),.S_N0_awlock(S_N0_awlock),.S_N0_awprot(S_N0_awprot),.S_N0_awqos(S_N0_awqos),.S_N0_awready(S_N0_awready),.S_N0_awsize(S_N0_awsize),.S_N0_awvalid(S_N0_awvalid),.S_N0_bid(S_N0_bid),.S_N0_bready(S_N0_bready),.S_N0_bresp(S_N0_bresp),.S_N0_bvalid(S_N0_bvalid),.S_N0_rdata(S_N0_rdata),.S_N0_rid(S_N0_rid),.S_N0_rlast(S_N0_rlast),.S_N0_rready(S_N0_rready),.S_N0_rresp(S_N0_rresp),.S_N0_rvalid(S_N0_rvalid),.S_N0_wdata(S_N0_wdata),.S_N0_wlast(S_N0_wlast),.S_N0_wready(S_N0_wready),.S_N0_wstrb(S_N0_wstrb),.S_N0_wvalid(S_N0_wvalid),.S_N1_araddr(S_N1_araddr),.S_N1_arburst(S_N1_arburst),.S_N1_arcache(S_N1_arcache),.S_N1_arid(S_N1_arid),.S_N1_arlen(S_N1_arlen),.S_N1_arlock(S_N1_arlock),.S_N1_arprot(S_N1_arprot),.S_N1_arqos(S_N1_arqos),.S_N1_arready(S_N1_arready),.S_N1_arsize(S_N1_arsize),.S_N1_arvalid(S_N1_arvalid),.S_N1_awaddr(S_N1_awaddr),.S_N1_awburst(S_N1_awburst),.S_N1_awcache(S_N1_awcache),.S_N1_awid(S_N1_awid),.S_N1_awlen(S_N1_awlen),.S_N1_awlock(S_N1_awlock),.S_N1_awprot(S_N1_awprot),.S_N1_awqos(S_N1_awqos),.S_N1_awready(S_N1_awready),.S_N1_awsize(S_N1_awsize),.S_N1_awvalid(S_N1_awvalid),.S_N1_bid(S_N1_bid),.S_N1_bready(S_N1_bready),.S_N1_bresp(S_N1_bresp),.S_N1_bvalid(S_N1_bvalid),.S_N1_rdata(S_N1_rdata),.S_N1_rid(S_N1_rid),.S_N1_rlast(S_N1_rlast),.S_N1_rready(S_N1_rready),.S_N1_rresp(S_N1_rresp),.S_N1_rvalid(S_N1_rvalid),.S_N1_wdata(S_N1_wdata),.S_N1_wlast(S_N1_wlast),.S_N1_wready(S_N1_wready),.S_N1_wstrb(S_N1_wstrb),.S_N1_wvalid(S_N1_wvalid),.aresetn(aresetn),.core_clk(core_clk),.hbm_clk(hbm_clk),.host_clk(host_clk));
f2_pcis_pipe host_pipe (.aclk(host_clk),.aresetn(aresetn),.s_axi_awregion(4'b0),.s_axi_arregion(4'b0),.m_axi_awregion(),.m_axi_arregion(),.s_axi_araddr(S_HOST_araddr),.m_axi_araddr(P_HOST_araddr),.s_axi_arburst(S_HOST_arburst),.m_axi_arburst(P_HOST_arburst),.s_axi_arcache(S_HOST_arcache),.m_axi_arcache(P_HOST_arcache),.s_axi_arid(S_HOST_arid),.m_axi_arid(P_HOST_arid),.s_axi_arlen(S_HOST_arlen),.m_axi_arlen(P_HOST_arlen),.s_axi_arlock(S_HOST_arlock),.m_axi_arlock(P_HOST_arlock),.s_axi_arprot(S_HOST_arprot),.m_axi_arprot(P_HOST_arprot),.s_axi_arqos(S_HOST_arqos),.m_axi_arqos(P_HOST_arqos),.s_axi_arready(S_HOST_arready),.m_axi_arready(P_HOST_arready),.s_axi_arsize(S_HOST_arsize),.m_axi_arsize(P_HOST_arsize),.s_axi_arvalid(S_HOST_arvalid),.m_axi_arvalid(P_HOST_arvalid),.s_axi_awaddr(S_HOST_awaddr),.m_axi_awaddr(P_HOST_awaddr),.s_axi_awburst(S_HOST_awburst),.m_axi_awburst(P_HOST_awburst),.s_axi_awcache(S_HOST_awcache),.m_axi_awcache(P_HOST_awcache),.s_axi_awid(S_HOST_awid),.m_axi_awid(P_HOST_awid),.s_axi_awlen(S_HOST_awlen),.m_axi_awlen(P_HOST_awlen),.s_axi_awlock(S_HOST_awlock),.m_axi_awlock(P_HOST_awlock),.s_axi_awprot(S_HOST_awprot),.m_axi_awprot(P_HOST_awprot),.s_axi_awqos(S_HOST_awqos),.m_axi_awqos(P_HOST_awqos),.s_axi_awready(S_HOST_awready),.m_axi_awready(P_HOST_awready),.s_axi_awsize(S_HOST_awsize),.m_axi_awsize(P_HOST_awsize),.s_axi_awvalid(S_HOST_awvalid),.m_axi_awvalid(P_HOST_awvalid),.s_axi_bid(S_HOST_bid),.m_axi_bid(P_HOST_bid),.s_axi_bready(S_HOST_bready),.m_axi_bready(P_HOST_bready),.s_axi_bresp(S_HOST_bresp),.m_axi_bresp(P_HOST_bresp),.s_axi_bvalid(S_HOST_bvalid),.m_axi_bvalid(P_HOST_bvalid),.s_axi_rdata(S_HOST_rdata),.m_axi_rdata(P_HOST_rdata),.s_axi_rid(S_HOST_rid),.m_axi_rid(P_HOST_rid),.s_axi_rlast(S_HOST_rlast),.m_axi_rlast(P_HOST_rlast),.s_axi_rready(S_HOST_rready),.m_axi_rready(P_HOST_rready),.s_axi_rresp(S_HOST_rresp),.m_axi_rresp(P_HOST_rresp),.s_axi_rvalid(S_HOST_rvalid),.m_axi_rvalid(P_HOST_rvalid),.s_axi_wdata(S_HOST_wdata),.m_axi_wdata(P_HOST_wdata),.s_axi_wlast(S_HOST_wlast),.m_axi_wlast(P_HOST_wlast),.s_axi_wready(S_HOST_wready),.m_axi_wready(P_HOST_wready),.s_axi_wstrb(S_HOST_wstrb),.m_axi_wstrb(P_HOST_wstrb),.s_axi_wvalid(S_HOST_wvalid),.m_axi_wvalid(P_HOST_wvalid));
logic [1:0] n_cvalid;
logic [1:0] n_cready;
logic [1:0] n_cwe;
logic [1:0][24:0] n_caddr;
logic [1:0] n_wvalid;
logic [1:0] n_wready;
logic [1:0][511:0] n_wdata;
logic [1:0][63:0] n_wmask;
logic [1:0] n_rvalid;
logic [1:0][511:0] n_rdata;
logic [1:0][15:0] n_wdone;
logic [1:0] n_err;
otpu_native_axi #(.BASE_ADDR(34'h0)) native0 (.clk(core_clk),
.rst(!aresetn),
.n_cvalid(n_cvalid[0]),
.n_cready(n_cready[0]),
.n_cwe(n_cwe[0]),
.n_caddr(n_caddr[0]),
.n_wvalid(n_wvalid[0]),
.n_wready(n_wready[0]),
.n_wdata(n_wdata[0]),
.n_wmask(n_wmask[0]),
.n_rvalid(n_rvalid[0]),
.n_rdata(n_rdata[0]),
.n_wdone(n_wdone[0]),
.n_err(n_err[0]),
.m_bvalid(S_N0_bvalid),
.m_awready(S_N0_awready),
.m_arlen(S_N0_arlen),
.m_awaddr(S_N0_awaddr),
.m_wstrb(S_N0_wstrb),
.m_rlast(S_N0_rlast),
.m_wready(S_N0_wready),
.m_araddr(S_N0_araddr),
.m_arvalid(S_N0_arvalid),
.m_bresp(S_N0_bresp),
.m_bready(S_N0_bready),
.m_wdata(S_N0_wdata),
.m_awvalid(S_N0_awvalid),
.m_wvalid(S_N0_wvalid),
.m_arready(S_N0_arready),
.m_rresp(S_N0_rresp),
.m_rdata(S_N0_rdata),
.m_rvalid(S_N0_rvalid),
.m_rready(S_N0_rready));
assign S_N0_arburst=2'b01;
assign S_N0_arcache='0;
assign S_N0_arid='0;
assign S_N0_arlock='0;
assign S_N0_arprot='0;
assign S_N0_arqos='0;
assign S_N0_arsize=3'd6;
assign S_N0_awburst=2'b01;
assign S_N0_awcache='0;
assign S_N0_awid='0;
assign S_N0_awlen='0;
assign S_N0_awlock='0;
assign S_N0_awprot='0;
assign S_N0_awqos='0;
assign S_N0_awsize=3'd6;
assign S_N0_wlast=1'b1;
otpu_native_axi #(.BASE_ADDR(34'h80000000)) native1 (.clk(core_clk),
.rst(!aresetn),
.n_cvalid(n_cvalid[1]),
.n_cready(n_cready[1]),
.n_cwe(n_cwe[1]),
.n_caddr(n_caddr[1]),
.n_wvalid(n_wvalid[1]),
.n_wready(n_wready[1]),
.n_wdata(n_wdata[1]),
.n_wmask(n_wmask[1]),
.n_rvalid(n_rvalid[1]),
.n_rdata(n_rdata[1]),
.n_wdone(n_wdone[1]),
.n_err(n_err[1]),
.m_bvalid(S_N1_bvalid),
.m_awready(S_N1_awready),
.m_arlen(S_N1_arlen),
.m_awaddr(S_N1_awaddr),
.m_wstrb(S_N1_wstrb),
.m_rlast(S_N1_rlast),
.m_wready(S_N1_wready),
.m_araddr(S_N1_araddr),
.m_arvalid(S_N1_arvalid),
.m_bresp(S_N1_bresp),
.m_bready(S_N1_bready),
.m_wdata(S_N1_wdata),
.m_awvalid(S_N1_awvalid),
.m_wvalid(S_N1_wvalid),
.m_arready(S_N1_arready),
.m_rresp(S_N1_rresp),
.m_rdata(S_N1_rdata),
.m_rvalid(S_N1_rvalid),
.m_rready(S_N1_rready));
assign S_N1_arburst=2'b01;
assign S_N1_arcache='0;
assign S_N1_arid='0;
assign S_N1_arlock='0;
assign S_N1_arprot='0;
assign S_N1_arqos='0;
assign S_N1_arsize=3'd6;
assign S_N1_awburst=2'b01;
assign S_N1_awcache='0;
assign S_N1_awid='0;
assign S_N1_awlen='0;
assign S_N1_awlock='0;
assign S_N1_awprot='0;
assign S_N1_awqos='0;
assign S_N1_awsize=3'd6;
assign S_N1_wlast=1'b1;
initial S_HOST_araddr='0;
initial S_HOST_arburst='0;
initial S_HOST_arcache='0;
initial S_HOST_arid='0;
initial S_HOST_arlen='0;
initial S_HOST_arlock='0;
initial S_HOST_arprot='0;
initial S_HOST_arqos='0;
initial S_HOST_arsize='0;
initial S_HOST_arvalid='0;
initial S_HOST_awaddr='0;
initial S_HOST_awburst='0;
initial S_HOST_awcache='0;
initial S_HOST_awid='0;
initial S_HOST_awlen='0;
initial S_HOST_awlock='0;
initial S_HOST_awprot='0;
initial S_HOST_awqos='0;
initial S_HOST_awsize='0;
initial S_HOST_awvalid='0;
initial S_HOST_bready=1'b1;
initial S_HOST_rready=1'b1;
initial S_HOST_wdata='0;
initial S_HOST_wlast='0;
initial S_HOST_wstrb='0;
initial S_HOST_wvalid='0;
wire [33:0] hb0_awaddr,hb0_araddr;
otpu_hbm_geometry geometry0 (.s_awaddr(M_H0_awaddr),.s_araddr(M_H0_araddr),.m_awaddr(hb0_awaddr),.m_araddr(hb0_araddr));
axi_hbm_model mem0 (.clk(hbm_clk),.rst_n(aresetn),.awaddr(hb0_awaddr),.araddr(hb0_araddr),.awlen(M_H0_awlen),.arlen(M_H0_arlen),.awsize(3'd5),.arsize(3'd5),.awvalid(M_H0_awvalid),.arvalid(M_H0_arvalid),.awready(M_H0_awready),.arready(M_H0_arready),.wdata(M_H0_wdata),.wstrb(M_H0_wstrb),.wlast(M_H0_wlast),.wvalid(M_H0_wvalid),.wready(M_H0_wready),.bvalid(M_H0_bvalid),.bresp(M_H0_bresp),.bready(M_H0_bready),.rdata(M_H0_rdata),.rlast(M_H0_rlast),.rvalid(M_H0_rvalid),.rresp(M_H0_rresp),.rready(M_H0_rready));
always @(posedge hbm_clk) if(aresetn) begin if(M_H0_awvalid && M_H0_awlen!=0 && M_H0_awsize!=5) $fatal(1,"unexpected unpacked burst"); if(M_H0_arvalid && M_H0_arlen!=0 && M_H0_arsize!=5) $fatal(1,"unexpected unpacked read burst"); end
wire [33:0] hb4_awaddr,hb4_araddr;
otpu_hbm_geometry geometry4 (.s_awaddr(M_H4_awaddr),.s_araddr(M_H4_araddr),.m_awaddr(hb4_awaddr),.m_araddr(hb4_araddr));
axi_hbm_model mem4 (.clk(hbm_clk),.rst_n(aresetn),.awaddr(hb4_awaddr),.araddr(hb4_araddr),.awlen(M_H4_awlen),.arlen(M_H4_arlen),.awsize(3'd5),.arsize(3'd5),.awvalid(M_H4_awvalid),.arvalid(M_H4_arvalid),.awready(M_H4_awready),.arready(M_H4_arready),.wdata(M_H4_wdata),.wstrb(M_H4_wstrb),.wlast(M_H4_wlast),.wvalid(M_H4_wvalid),.wready(M_H4_wready),.bvalid(M_H4_bvalid),.bresp(M_H4_bresp),.bready(M_H4_bready),.rdata(M_H4_rdata),.rlast(M_H4_rlast),.rvalid(M_H4_rvalid),.rresp(M_H4_rresp),.rready(M_H4_rready));
always @(posedge hbm_clk) if(aresetn) begin if(M_H4_awvalid && M_H4_awlen!=0 && M_H4_awsize!=5) $fatal(1,"unexpected unpacked burst"); if(M_H4_arvalid && M_H4_arlen!=0 && M_H4_arsize!=5) $fatal(1,"unexpected unpacked read burst"); end

logic [511:0] expected[0:1][0:63];integer seen[0:1];
always @(posedge core_clk) if(aresetn) begin
 for(integer c=0;c<2;c=c+1) begin
  if(n_err[c]) $fatal(1,"native AXI error channel %0d",c);
  if(n_rvalid[c]) begin
   if(n_rdata[c] !== expected[c][seen[c]]) $fatal(1,"width/CDC mismatch channel %0d beat %0d",c,seen[c]);
   seen[c]=seen[c]+1;
  end
 end
end
initial begin
 n_cvalid=0;n_cwe=0;n_caddr=0;n_wvalid=0;n_wdata=0;n_wmask='1;seen[0]=0;seen[1]=0;
 repeat(100) @(negedge core_clk);aresetn=1;
 repeat(100) @(negedge core_clk);
 for(integer i=0;i<64;i=i+1) begin
  for(integer c=0;c<2;c=c+1) begin
   for(integer word=0;word<16;word=word+1) expected[c][i][word*32+:32]=32'h12340000|(c<<12)|(i<<4)|word;
   @(negedge core_clk);n_cvalid[c]=1;n_cwe[c]=1;n_caddr[c]=i;n_wvalid[c]=1;n_wdata[c]=expected[c][i];
   fork
    begin do @(posedge core_clk);while(!n_cready[c]);@(negedge core_clk);n_cvalid[c]=0;end
    begin do @(posedge core_clk);while(!n_wready[c]);@(negedge core_clk);n_wvalid[c]=0;end
   join
  end
 end
 wait(n_wdone[0]==64 && n_wdone[1]==64);
 for(integer i=0;i<64;i=i+1) begin
  for(integer c=0;c<2;c=c+1) begin
   @(negedge core_clk);n_cvalid[c]=1;n_cwe[c]=0;n_caddr[c]=i;
   do @(posedge core_clk);while(!n_cready[c]);
   @(negedge core_clk);n_cvalid[c]=0;
  end
 end
 wait(seen[0]==64 && seen[1]==64);
 // Exercise all 64 byte lanes, both 32-byte halves and channel windows.
 for(integer c=0;c<2;c=c+1) begin
  for(integer sz=0;sz<=3;sz=sz+1) begin
   for(integer off=0;off<64;off=off+(1<<sz)) begin
    @(negedge host_clk);
    S_HOST_awaddr=(c*64'h80000000)+off;S_HOST_awsize=sz;S_HOST_awlen=0;S_HOST_awburst=1;S_HOST_awid=16'h8023;S_HOST_awvalid=1;
    S_HOST_wdata=0;S_HOST_wstrb=0;S_HOST_wlast=1;S_HOST_wvalid=1;
    for(integer b=0;b<(1<<sz);b=b+1) begin
     S_HOST_wdata[8*(off+b)+:8]=(c*127+sz*41+off+b);
     S_HOST_wstrb[off+b]=1;
     expected[c][0][8*(off+b)+:8]=(c*127+sz*41+off+b);
    end
    fork
     begin do @(posedge host_clk);while(!S_HOST_awready);@(negedge host_clk);S_HOST_awvalid=0;end
     begin do @(posedge host_clk);while(!S_HOST_wready);@(negedge host_clk);S_HOST_wvalid=0;end
    join
    do @(posedge host_clk);while(!S_HOST_bvalid);
    if(S_HOST_bresp!=0 || S_HOST_bid!=16'h8023) $fatal(1,"host write response / full ID");
    @(negedge host_clk);S_HOST_araddr=(c*64'h80000000)+off;S_HOST_arsize=sz;S_HOST_arlen=0;S_HOST_arburst=1;S_HOST_arid=16'hf025;S_HOST_arvalid=1;
    do @(posedge host_clk);while(!S_HOST_arready);
    @(negedge host_clk);S_HOST_arvalid=0;
    do @(posedge host_clk);while(!S_HOST_rvalid);
    if(S_HOST_rresp!=0 || S_HOST_rid!=16'hf025 || !S_HOST_rlast) $fatal(1,"host narrow read / full ID");
    for(integer b=0;b<(1<<sz);b=b+1)
     if(S_HOST_rdata[8*(off+b)+:8]!==expected[c][0][8*(off+b)+:8]) $fatal(1,"host data c=%0d size=%0d offset=%0d byte=%0d",c,sz,off,b);
   end
   // Check untouched bytes and visibility from the accelerator's native port.
   seen[c]=0;
   @(negedge core_clk);n_cvalid[c]=1;n_cwe[c]=0;n_caddr[c]=0;
   do @(posedge core_clk);while(!n_cready[c]);
   @(negedge core_clk);n_cvalid[c]=0;
   wait(seen[c]==1);
  end
 end
 // PCIS write-combined transfers can be multi-beat 512-bit bursts.
 // Force a 16-beat host burst: converted to 32 HBM beats and split at AXI3's limit.
 for(integer c=0;c<2;c=c+1) begin
  @(negedge host_clk);S_HOST_awaddr=c*64'h80000000+16384;S_HOST_awsize=6;S_HOST_awlen=15;S_HOST_awid=16'hbb71;S_HOST_awvalid=1;
  fork
   begin do @(posedge host_clk);while(!S_HOST_awready);@(negedge host_clk);S_HOST_awvalid=0;end
   begin
    for(integer i=0;i<16;i=i+1) begin
     @(negedge host_clk);S_HOST_wvalid=1;S_HOST_wstrb='1;S_HOST_wlast=(i==15);
     for(integer w=0;w<16;w=w+1)S_HOST_wdata[w*32+:32]=32'hdead0000|(c<<12)|(i<<4)|w;
     do @(posedge host_clk);while(!S_HOST_wready);
     @(negedge host_clk);S_HOST_wvalid=0;
    end
   end
  join
  do @(posedge host_clk);while(!S_HOST_bvalid);
  if(S_HOST_bresp!=0 || S_HOST_bid!=16'hbb71)$fatal(1,"host burst B");
  @(negedge host_clk);S_HOST_araddr=c*64'h80000000+16384;S_HOST_arsize=6;S_HOST_arlen=15;S_HOST_arid=16'hcc27;S_HOST_arvalid=1;
  do @(posedge host_clk);while(!S_HOST_arready);
  @(negedge host_clk);S_HOST_arvalid=0;
  for(integer i=0;i<16;i=i+1) begin
   S_HOST_rready=0;repeat(3)@(negedge host_clk);S_HOST_rready=1;
   do @(posedge host_clk);while(!S_HOST_rvalid);
   if(S_HOST_rresp!=0 || S_HOST_rid!=16'hcc27 || S_HOST_rlast!=(i==15))$fatal(1,"host burst R");
   for(integer w=0;w<16;w=w+1)if(S_HOST_rdata[w*32+:32] !== (32'hdead0000|(c<<12)|(i<<4)|w))$fatal(1,"host burst data");
   @(negedge host_clk);
  end
 end
 // Unmapped host addresses must return DECERR, never alias into the 4 GiB map.
 @(negedge host_clk);S_HOST_araddr=64'h100000000;S_HOST_arsize=2;S_HOST_arlen=0;S_HOST_arid=16'h4567;S_HOST_arvalid=1;S_HOST_rready=1;
 do @(posedge host_clk);while(!S_HOST_arready);
 @(negedge host_clk);S_HOST_arvalid=0;
 do @(posedge host_clk);while(!S_HOST_rvalid);
 if(S_HOST_rresp!=3 || S_HOST_rid!=16'h4567 || !S_HOST_rlast)$fatal(1,"unmapped host read must DECERR");
 $display("PASS PCIS register slice and vendor fabric: burst splitting, R stalls, unmapped DECERR; 128 native writes/reads; 240 narrow host writes/reads, byte lanes, 16-bit IDs, both windows, native readback");
 $finish;
end
initial begin #2000000;$fatal(1,"vendor fabric timeout");end
endmodule
