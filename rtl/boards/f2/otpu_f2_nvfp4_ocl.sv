// Optional NV41 sidecar router after existing OCL CDC. One transaction at a
// time; AW/W paired before routing. Legacy addresses and payloads unchanged.
module otpu_f2_nvfp4_ocl #(
  parameter logic [63:0] BANK_BASE=0,
  parameter int BANK_LOG2=29
)(
  input logic clk,rst,legacy_idle,
  input logic [11:0] s_awaddr,s_araddr,
  input logic s_awvalid,s_wvalid,s_bready,s_arvalid,s_rready,
  output logic s_awready,s_wready,s_bvalid,s_arready,s_rvalid,
  input logic [31:0] s_wdata,
  input logic [3:0] s_wstrb,
  output logic [1:0] s_bresp,s_rresp,
  output logic [31:0] s_rdata,
  output logic [11:0] l_awaddr,l_araddr,
  output logic l_awvalid,l_wvalid,l_bready,l_arvalid,l_rready,
  input logic l_awready,l_wready,l_bvalid,l_arready,l_rvalid,
  output logic [31:0] l_wdata,
  output logic [3:0] l_wstrb,
  input logic [1:0] l_bresp,l_rresp,
  input logic [31:0] l_rdata,
  output logic owns_memory,memory_outstanding,
  output logic n_cvalid,n_cwe,
  input logic n_cready,
  output logic [24:0] n_caddr,
  output logic n_wvalid,
  input logic n_wready,
  output logic [511:0] n_wdata,
  output logic [63:0] n_wmask,
  input logic n_rvalid,
  input logic [511:0] n_rdata,
  input logic [15:0] n_wdone,
  input logic n_err
);
  typedef enum logic [2:0] {IDLE,SEND_W,WAIT_B,RESP_B,SEND_R,WAIT_R,RESP_R} state_t;
  state_t state;
  logic have_aw,have_w,aw_done,w_done,selected;
  logic [11:0] addr,awaddr;
  logic [31:0] data;
  logic [3:0] strobe;
  logic m_awready,m_wready,m_bvalid,m_arready,m_rvalid;
  logic [1:0] m_bresp,m_rresp;
  logic [31:0] m_rdata;
  wire awvalid=!rst && state==SEND_W && !aw_done;
  wire wvalid=!rst && state==SEND_W && !w_done;
  wire arvalid=!rst && state==SEND_R;
  assign s_awready=!rst && state==IDLE && !have_aw;
  assign s_wready=!rst && state==IDLE && !have_w;
  assign s_arready=!rst && state==IDLE && !have_aw && !have_w && !s_awvalid && !s_wvalid;
  assign s_bvalid=!rst && state==RESP_B;
  assign s_rvalid=!rst && state==RESP_R;
  assign l_awaddr=addr;assign l_araddr=addr;assign l_wdata=data;assign l_wstrb=strobe;
  assign l_awvalid=awvalid && !selected;assign l_wvalid=wvalid && !selected;
  assign l_bready=!rst && state==WAIT_B && !selected;
  assign l_arvalid=arvalid && !selected;assign l_rready=!rst && state==WAIT_R && !selected;
  otpu_nvfp4_mailbox #(.BANK_BASE(BANK_BASE),.BANK_LOG2(BANK_LOG2)) u_mailbox (
    .clk,.rst,.legacy_idle,.s_awaddr(addr[9:0]),.s_araddr(addr[9:0]),
    .s_awvalid(awvalid && selected),.s_awready(m_awready),.s_wvalid(wvalid && selected),.s_wready(m_wready),
    .s_wdata(data),.s_wstrb(strobe),.s_bvalid(m_bvalid),.s_bresp(m_bresp),.s_bready(state==WAIT_B && selected),
    .s_arvalid(arvalid && selected),.s_arready(m_arready),.s_rvalid(m_rvalid),.s_rdata(m_rdata),
    .s_rresp(m_rresp),.s_rready(state==WAIT_R && selected),
    .owns_memory,.memory_outstanding,.n_cvalid,.n_cwe,.n_cready,.n_caddr,.n_wvalid,.n_wready,.n_wdata,
    .n_wmask,.n_rvalid,.n_rdata,.n_wdone,.n_err);
  always_ff @(posedge clk)begin
    if(rst)begin
      state<=IDLE;have_aw<=0;have_w<=0;aw_done<=0;w_done<=0;selected<=0;
      addr<=0;awaddr<=0;data<=0;strobe<=0;s_bresp<=0;s_rresp<=0;s_rdata<=0;
    end else case(state)
      IDLE:begin
        if(s_awvalid && s_awready)begin have_aw<=1;awaddr<=s_awaddr;end
        if(s_wvalid && s_wready)begin have_w<=1;data<=s_wdata;strobe<=s_wstrb;end
        if(have_aw && have_w)begin
          addr<=awaddr;selected<=awaddr[11:10]==2'b01;have_aw<=0;have_w<=0;aw_done<=0;w_done<=0;
          if(owns_memory && awaddr[11:10]!=2'b01)begin s_bresp<=2;state<=RESP_B;end
          else state<=SEND_W;
        end else if(s_arvalid && s_arready)begin
          addr<=s_araddr;selected<=s_araddr[11:10]==2'b01;state<=SEND_R;
        end
      end
      SEND_W:begin
        if(awvalid && (selected ? m_awready : l_awready))aw_done<=1;
        if(wvalid && (selected ? m_wready : l_wready))w_done<=1;
        if((aw_done || (selected ? m_awready : l_awready)) &&
           (w_done || (selected ? m_wready : l_wready)))state<=WAIT_B;
      end
      WAIT_B:if(selected ? m_bvalid : l_bvalid)begin s_bresp<=selected ? m_bresp : l_bresp;state<=RESP_B;end
      RESP_B:if(s_bready)state<=IDLE;
      SEND_R:if(selected ? m_arready : l_arready)state<=WAIT_R;
      WAIT_R:if(selected ? m_rvalid : l_rvalid)begin
        s_rresp<=selected ? m_rresp : l_rresp;s_rdata<=selected ? m_rdata : l_rdata;
        if(!selected && addr==12'h05c)s_rdata<=l_rdata|32'h40;
        if(!selected && addr==12'h00c && memory_outstanding)s_rdata<=l_rdata&~32'h8;
        state<=RESP_R;
      end
      RESP_R:if(s_rready)state<=IDLE;
      default:state<=IDLE;
    endcase
  end
endmodule
