// C1 independent E16/S4 matrix benchmark, ABI1. Scales/ACT are OCL preloaded.
// HBM stores row-major weights per physical port; this is not a full-token top.
module otpu_c1_hbm_kernel #(
  parameter logic [31:0] BUILD_ID=0,
  parameter int CORE_KHZ=200000,
  parameter int DEPTH=128, MAX_ROWS=512, RD=32
)(
  input logic clk,rst,hbm_ready,
  output logic [2:0] led,
  input logic [11:0] s_ctl_awaddr,
  input logic s_ctl_awvalid,
  output logic s_ctl_awready,
  input logic [31:0] s_ctl_wdata,
  input logic [3:0] s_ctl_wstrb,
  input logic s_ctl_wvalid,
  output logic s_ctl_wready,
  output logic [1:0] s_ctl_bresp,
  output logic s_ctl_bvalid,
  input logic s_ctl_bready,
  input logic [11:0] s_ctl_araddr,
  input logic s_ctl_arvalid,
  output logic s_ctl_arready,
  output logic [31:0] s_ctl_rdata,
  output logic [1:0] s_ctl_rresp,
  output logic s_ctl_rvalid,
  input logic s_ctl_rready,
  output logic [7:0] n_cvalid,
  input logic [7:0] n_cready,
  output logic [7:0] n_cwe,
  output logic [7:0][24:0] n_caddr,
  output logic [7:0] n_wvalid,
  input logic [7:0] n_wready,
  output logic [7:0][511:0] n_wdata,
  output logic [7:0][63:0] n_wmask,
  input logic [7:0] n_rvalid,
  input logic [7:0][511:0] n_rdata,
  input logic [7:0][15:0] n_wdone,
  input logic [7:0] n_err
);
  typedef enum logic[2:0] {IDLE,PREAD,PWRITE,COMMAND,RUN,DRAIN,FLUSH} state_t;
  state_t state;
  wire busy=state!=IDLE;
  logic done,aborted;
  logic [31:0] error_code,cfg_mode,cfg_kb,cfg_rows,cfg_base,mem_addr,generation;
  logic [8:0] preload;
  logic [63:0] cycles;
  logic [31:0] result_count,groups_accepted,groups_committed,read_issued,read_returned;
  logic [31:0] advances,native_wait,weight_wait,scale_wait,act_wait,output_wait,max_outstanding;
  logic [7:0][31:0] pc_issued,pc_returned,pc_peak,pc_blocked,pc_credit;
  wire [31:0] pc_latency[8]='{default:0};
  wire [32:0] max_span=33'((cfg_rows+15)>>4)*33'(cfg_kb)*4;
  wire config_ok=cfg_mode==0&&cfg_kb>=1&&cfg_kb<=128&&cfg_rows>=1&&cfg_rows<=512&&
                 cfg_base<32'h800000&&({1'b0,cfg_base}+max_span)<=33'h800000&&hbm_ready;
  assign n_cwe='0;assign n_wvalid='0;assign n_wdata='0;assign n_wmask='0;
  assign led={error_code!=0,hbm_ready,!rst};
  // Serialized AXI-Lite slave: AW and W may arrive independently. Response payloads
  // remain stable until acknowledged. Reads and writes cannot race MEM_ADDR.
  // Capture the accepted read address before the wide CSR mux. The OCL bridge
  // sits beside the shell in SLR1; this register keeps its crossing into the
  // SLR0 kernel separate from register selection. Read side effects and the
  // synchronous result-RAM access happen together in the following phase.
  logic aw_hold,w_hold,rd_decode,rd_pending,rd_mem;
  logic [11:0] aw_addr,rd_addr;
  logic [31:0] w_value,rd_value;
  logic [3:0] w_strobe;
  logic [1:0] rd_response,rd_bank,rd_kind;
  wire wr_fire=aw_hold&&w_hold&&!s_ctl_bvalid;
  wire ar_fire=s_ctl_arvalid&&s_ctl_arready;
  assign s_ctl_awready=!rst&&!aw_hold&&!s_ctl_bvalid&&!rd_decode&&!rd_pending&&!s_ctl_rvalid;
  assign s_ctl_wready=!rst&&!w_hold&&!s_ctl_bvalid&&!rd_decode&&!rd_pending&&!s_ctl_rvalid;
  assign s_ctl_arready=!rst&&!rd_decode&&!rd_pending&&!s_ctl_rvalid&&!aw_hold&&!w_hold&&
                       !s_ctl_awvalid&&!s_ctl_wvalid&&!s_ctl_bvalid;
  wire is_act=mem_addr<DEPTH*32;
  wire is_da=mem_addr>=32'h10000&&mem_addr<32'h10000+DEPTH;
  wire is_sa=mem_addr==32'h11000;
  wire is_dw=mem_addr>=32'h20000&&mem_addr<32'h20000+MAX_ROWS*DEPTH;
  wire is_sw=mem_addr>=32'h40000&&mem_addr<32'h40000+MAX_ROWS;
  wire is_result=(mem_addr>=32'h50000&&mem_addr<32'h50000+MAX_ROWS)||
                 (mem_addr>=32'h60000&&mem_addr<32'h60000+MAX_ROWS)||
                 (mem_addr>=32'h70000&&mem_addr<32'h70000+MAX_ROWS);
  wire [1:0] result_kind=mem_addr[17:16]-2'd1;
  wire memory_write=wr_fire&&aw_addr==12'h028&&w_strobe==4'hf&&!busy&&
                    (is_act||is_da||is_sa||is_dw||is_sw);
  logic wr_ok,read_ok;
  logic [31:0] csr_value;
  wire ctrl_write=wr_fire&&aw_addr==12'h010&&w_strobe==4'hf;
  wire start=ctrl_write&&w_value==1&&!busy;
  wire abort_request=ctrl_write&&w_value==2&&busy;
  wire clear_request=ctrl_write&&w_value==4&&!busy;
  always_comb begin
    wr_ok=0;
    if(w_strobe==4'hf) case(aw_addr)
      12'h010: wr_ok=(w_value==1&&!busy)||(w_value==2&&busy)||(w_value==4&&!busy);
      12'h014,12'h018,12'h01c,12'h020: wr_ok=!busy;
      12'h024: wr_ok=1;
      12'h028: wr_ok=memory_write;
      default: ;
    endcase
    read_ok=1;csr_value=0;
    case(rd_addr)
      12'h000:csr_value=32'h43314d42;
      12'h004:csr_value=1;
      12'h008:csr_value=BUILD_ID;
      12'h00c:csr_value={27'd0,state==DRAIN,aborted,error_code!=0,done,busy};
      12'h014:csr_value=cfg_mode;
      12'h018:csr_value=cfg_kb;
      12'h01c:csr_value=cfg_rows;
      12'h020:csr_value=cfg_base;
      12'h024:csr_value=mem_addr;
      12'h02c:read_ok=is_result;
      12'h030:csr_value=error_code;
      12'h034:csr_value=result_count;
      12'h038:csr_value=cycles[31:0];
      12'h03c:csr_value=cycles[63:32];
      12'h040:csr_value=groups_accepted;
      12'h044:csr_value=groups_committed;
      12'h048:csr_value=read_issued;
      12'h04c:csr_value=read_returned;
      12'h050:csr_value=advances;
      12'h054:csr_value=native_wait;
      12'h058:csr_value=weight_wait;
      12'h05c:csr_value=scale_wait;
      12'h060:csr_value=act_wait;
      12'h064:csr_value=output_wait;
      12'h068:csr_value=max_outstanding;
      12'h06c:csr_value=CORE_KHZ;
      12'h070:csr_value=MAX_ROWS;
      12'h074:csr_value=DEPTH;
      12'h078:csr_value=RD;
      12'h07c:csr_value={31'd0,hbm_ready};
      default:begin
        read_ok=0;
        for(int p=0;p<8;p++) begin
          if(rd_addr==12'(256+p*32+0)) begin csr_value=pc_issued[p];read_ok=1;end
          if(rd_addr==12'(256+p*32+4)) begin csr_value=pc_returned[p];read_ok=1;end
          if(rd_addr==12'(256+p*32+8)) begin csr_value=pc_peak[p];read_ok=1;end
          if(rd_addr==12'(256+p*32+12)) begin csr_value=pc_blocked[p];read_ok=1;end
          if(rd_addr==12'(256+p*32+16)) begin csr_value=pc_latency[p];read_ok=1;end
          if(rd_addr==12'(256+p*32+20)) begin csr_value=pc_credit[p];read_ok=1;end
        end
      end
    endcase
  end
  logic [79:0] result_q;
  always_ff @(posedge clk) begin
    if(rst) begin
      aw_hold<=0;w_hold<=0;s_ctl_bvalid<=0;s_ctl_bresp<=0;rd_decode<=0;rd_pending<=0;
      s_ctl_rvalid<=0;s_ctl_rresp<=0;s_ctl_rdata<=0;
      cfg_mode<=0;cfg_kb<=9;cfg_rows<=257;cfg_base<=0;mem_addr<=0;
    end else begin
      if(s_ctl_awvalid&&s_ctl_awready) begin aw_hold<=1;aw_addr<=s_ctl_awaddr;end
      if(s_ctl_wvalid&&s_ctl_wready) begin w_hold<=1;w_value<=s_ctl_wdata;w_strobe<=s_ctl_wstrb;end
      if(s_ctl_bvalid&&s_ctl_bready)s_ctl_bvalid<=0;
      if(wr_fire)begin
        aw_hold<=0;w_hold<=0;s_ctl_bvalid<=1;s_ctl_bresp<=wr_ok?0:2;
        if(wr_ok)case(aw_addr)
          12'h014:cfg_mode<=w_value;
          12'h018:cfg_kb<=w_value;
          12'h01c:cfg_rows<=w_value;
          12'h020:cfg_base<=w_value;
          12'h024:mem_addr<=w_value;
          12'h028:mem_addr<=mem_addr+1;
          default:;
        endcase
      end
      if(s_ctl_rvalid&&s_ctl_rready)s_ctl_rvalid<=0;
      if(ar_fire)begin
        rd_decode<=1;rd_addr<=s_ctl_araddr;
      end
      if(rd_decode)begin
        rd_decode<=0;rd_pending<=1;rd_value<=csr_value;rd_response<=read_ok?0:2;
        rd_mem<=rd_addr==12'h02c&&read_ok;rd_bank<=mem_addr[1:0];rd_kind<=result_kind;
        if(rd_addr==12'h02c&&read_ok)mem_addr<=mem_addr+1;
      end
      if(rd_pending)begin
        rd_pending<=0;s_ctl_rvalid<=1;s_ctl_rresp<=rd_response;
        if(!rd_mem)s_ctl_rdata<=rd_value;
        else case(rd_kind)
          0:s_ctl_rdata<=result_q[31:0];
          1:s_ctl_rdata<=result_q[63:32];
          default:s_ctl_rdata<={16'd0,result_q[79:64]};
        endcase
      end
    end
  end

  logic [255:0] act_q;
  logic [31:0] da_q,sa_reg;
  logic [15:0] req_valid,req_ready,rsp_valid,rsp_ready,rsp_drop;
  logic [15:0][1:0] req_slot,rsp_slot;
  logic [15:0][31:0] req_row,req_generation,rsp_row,rsp_generation,rsp_sw;
  logic [15:0][7:0] req_kb,req_k0,rsp_dw;
  logic [15:0][9:0] rsp_group;
  logic [15:0][255:0] rsp_weights;
  logic [15:0][2:0] occupancy;
  logic cmd_ready,core_busy,core_out_valid,int_valid,int_ready;
  logic [31:0] core_out_row,core_out_generation,core_out_bits,int_row,int_generation,int_sw,int_sa;
  logic signed [47:0] int_acc;
  logic adapter_drained,adapter_error;
  wire core_reset=rst||state==IDLE||state==FLUSH;
  wire running=state==RUN&&!abort_request&&error_code==0&&!adapter_error;
  wire commit=state==RUN&&core_out_valid;
  logic [31:0] accepted_now,issued_total,returned_total,pending_total;
  always_comb begin
    accepted_now=0;issued_total=0;returned_total=0;pending_total=0;
    for(int e=0;e<16;e++)if(rsp_valid[e]&&rsp_ready[e])accepted_now++;
    for(int p=0;p<8;p++)begin
      issued_total+=pc_issued[p];returned_total+=pc_returned[p];
      pending_total+=pc_issued[p]-pc_returned[p];
    end
  end
  assign read_issued=issued_total;assign read_returned=returned_total;
  assign advances=groups_accepted;assign scale_wait=0;assign act_wait=0;assign output_wait=0;
  for(genvar bank=0;bank<8;bank++)begin: act_banks
    (* ram_style="block" *) logic [31:0] data[512];
    always_ff @(posedge clk)begin
      if(memory_write&&is_act&&mem_addr[2:0]==3'(bank))data[mem_addr[11:3]]<=w_value;
      act_q[bank*32+:32]<=data[preload];
    end
  end
  (* ram_style="block" *) logic [31:0] da_mem[128];
  always_ff @(posedge clk)begin
    if(memory_write&&is_da)da_mem[mem_addr[6:0]]<=w_value;
    da_q<=da_mem[preload[8:2]];
    if(memory_write&&is_sa)sa_reg<=w_value;
  end
  for(genvar e=0;e<16;e++)begin: scale_banks
    (* ram_style="distributed" *) logic [7:0] dw[4096];
    (* ram_style="distributed" *) logic [31:0] sw[32];
    always_ff @(posedge clk)begin
      if(memory_write&&is_dw&&mem_addr[10:7]==4'(e))dw[{mem_addr[15:11],mem_addr[6:0]}]<=w_value[7:0];
      if(memory_write&&is_sw&&mem_addr[3:0]==4'(e))sw[mem_addr[8:4]]<=w_value;
    end
    assign rsp_dw[e]=dw[{rsp_row[e][8:4],rsp_group[e][8:2]}];
    assign rsp_sw[e]=sw[rsp_row[e][8:4]];
  end
  (* ram_style="block" *) logic [31:0] fp_results[512];
  (* ram_style="block" *) logic [47:0] int_results[512];
  always_ff @(posedge clk)begin
    if(commit)fp_results[core_out_row[8:0]]<=core_out_bits;
    if(int_valid&&int_ready&&state==RUN)int_results[int_row[8:0]]<=int_acc;
    if(rd_decode&&rd_addr==12'h02c)result_q<={int_results[mem_addr[8:0]],fp_results[mem_addr[8:0]]};
  end
  otpu_c1_top #(.ENGINES(16),.SLOTS(4)) core(
    .clk,.rst(core_reset),.act_we(state==PWRITE),.act_addr(preload),.act_data(act_q),
    .act_da(da_q[preload[1:0]*8+:8]),.act_sa(sa_reg),
    .cmd_valid(state==COMMAND),.cmd_ready,.cmd_n0(32'd0),.cmd_n(cfg_rows),
    .cmd_generation(generation),.cmd_kb(cfg_kb[7:0]),.cmd_k0(8'd0),.cmd_ab(8'd0),
    .req_valid,.req_ready,.req_slot,.req_row,.req_generation,.req_kb,.req_k0,
    .rsp_valid,.rsp_ready,.rsp_slot,.rsp_row,.rsp_generation,.rsp_group,.rsp_weights,
    .rsp_dw,.rsp_sw,.rsp_drop,.out_valid(core_out_valid),.out_ready(1'b1),
    .out_row(core_out_row),.out_generation(core_out_generation),.out_bits(core_out_bits),
    .int_valid,.int_ready,.int_row,.int_generation,.int_sw,.int_sa,.int_acc,.occupancy,.busy(core_busy)
  );
  otpu_c1_native #(.RD(RD)) adapter(
    .clk,.rst(rst||start||state==FLUSH),.enable(running),.base(cfg_base[22:0]),
    .req_valid,.req_ready,.req_slot,.req_row,.req_generation,.req_kb,
    .rsp_valid,.rsp_ready,.rsp_slot,.rsp_row,.rsp_generation,.rsp_group,.rsp_weights,
    .n_cvalid,.n_cready,.n_caddr,.n_rvalid,.n_rdata,.n_err,
    .drained(adapter_drained),.error(adapter_error),.pc_issued,.pc_returned,.pc_peak,.pc_blocked,.pc_credit
  );
  always_ff @(posedge clk)begin
    if(rst)begin
      state<=IDLE;done<=0;aborted<=0;error_code<=0;generation<=0;preload<=0;
      cycles<=0;result_count<=0;groups_accepted<=0;groups_committed<=0;
      native_wait<=0;weight_wait<=0;max_outstanding<=0;
    end else begin
      if(state==COMMAND||state==RUN||state==DRAIN)cycles<=cycles+1;
      if(state==RUN)begin
        groups_accepted<=groups_accepted+accepted_now;
        if(int_valid&&int_ready)groups_committed<=groups_committed+1;
        if(|(n_cvalid&~n_cready))native_wait<=native_wait+1;
        if(accepted_now==0&&!core_out_valid)weight_wait<=weight_wait+1;
        if(pending_total>max_outstanding)max_outstanding<=pending_total;
      end
      if(clear_request)begin done<=0;aborted<=0;error_code<=0;end
      if(start)begin
        done<=0;aborted<=0;cycles<=0;result_count<=0;groups_accepted<=0;groups_committed<=0;
        native_wait<=0;weight_wait<=0;max_outstanding<=0;preload<=0;
        if(config_ok&&!(|n_err))begin error_code<=0;generation<=generation+1;state<=PREAD;end
        else begin error_code<=1;done<=1;end
      end
      case(state)
        PREAD:state<=PWRITE;
        PWRITE:begin
          if({1'b0,preload}+10'd1=={cfg_kb[7:0],2'b00})state<=COMMAND;
          else begin preload<=preload+1'b1;state<=PREAD;end
        end
        COMMAND:if(cmd_ready)state<=RUN;
        RUN:if(commit)begin
          result_count<=result_count+1;
          if(result_count+1==cfg_rows)begin done<=1;state<=IDLE;end
        end
        DRAIN:if(adapter_drained)state<=FLUSH;
        FLUSH:begin state<=IDLE;done<=1;end
        default:;
      endcase
      if(busy&&state!=DRAIN&&state!=FLUSH&&(abort_request||adapter_error||(|rsp_drop)))begin
        state<=DRAIN;aborted<=abort_request;
        if(adapter_error)error_code<=2;
        else if(|rsp_drop)error_code<=3;
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)if(!rst)begin
    if(commit&&core_out_row!=result_count)$fatal(1,"C1 result retirement order mismatch");
    if(commit&&core_out_generation!=generation)$fatal(1,"C1 stale result");
    if(commit&&result_count+1==cfg_rows&&!adapter_drained)$fatal(1,"C1 completed with outstanding reads");
  end
`endif
endmodule
