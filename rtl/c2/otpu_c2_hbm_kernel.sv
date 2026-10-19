// C2 isolated real-HBM M=1 microbenchmark, ABI 1. This is not a production ISA top.
// Raw PCIS addresses: row r belongs to PC pair 2*(r%4), +1; each PC holds one
// 64-byte half at BASE + floor(r/4)*KB + k. Both modes read identical bytes.
// Legacy executes one row at a time, routing its actual pair to MXU input 0;
// decode executes up to four rows. ACT0 and scale records are preloaded via OCL.
// Native R has no ready: every accepted read reserves ring storage until the
// paired 128-byte chunk is consumed. ABORT drains all accepted reads, then flushes
// the MXU. Whole-shell rst is only legal with the native bridges reset/quiescent.
module otpu_c2_hbm_kernel #(
  parameter logic [31:0] BUILD_ID=0,
  parameter int CORE_KHZ=200000,
  parameter int DEPTH=256, MAX_ROWS=512, RD=32
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
  localparam int RW=$clog2(RD), AW=$clog2(DEPTH), GROUPS=MAX_ROWS/4, GW=$clog2(GROUPS);
  initial begin
    if(DEPTH!=256 || MAX_ROWS!=512 || RD<2 || RD>128 || (RD&(RD-1))!=0)
      $fatal(1,"C2 kernel ABI1 geometry requires DEPTH256 MAX_ROWS512 and power-of-two RD2..128");
  end
  typedef enum logic[2:0] {IDLE,SWREAD,COMMAND,FEED,WAITALL,DRAIN,FLUSH} state_t;
  state_t state;
  wire busy=state!=IDLE;
  logic done,aborted;
  logic [31:0] error_code,cfg_mode,cfg_kb,cfg_rows,cfg_base,mem_addr;
  logic [9:0] row_cursor;
  logic [63:0] cycles;
  logic [31:0] result_count,groups_accepted,groups_committed,read_issued,read_returned;
  logic [31:0] advances,native_wait,weight_wait,scale_wait,act_wait,output_wait,max_outstanding;
  logic [31:0] pc_issued[8],pc_returned[8],pc_peak[8],pc_blocked[8],pc_latency[8],pc_credit[8];
  logic [8:0] issued[8],returned[8],loaded[4],delivered[4],s_loaded[4],s_delivered[4];
  logic [8:0] a_loaded,a_delivered;
  logic [3:0] pair_active,core_mask;
  logic [3:0][31:0] cmd_addr;
  wire [1:0] legacy_pair=row_cursor[1:0];
  wire decode=cfg_mode[0];
  wire [8:0] kb=cfg_kb[8:0];
  logic [24:0] group_base;
  // Full-width bounds below reject high bits before these bounded products matter.
  wire [8:0] cfg_groups=9'(({1'b0,cfg_rows[9:0]}+11'd3)>>2);
  wire [16:0] cfg_span=17'(cfg_groups)*17'(cfg_kb[7:0]);
  wire [15:0] group_offset=16'(row_cursor[8:2])*16'(cfg_kb[7:0]);
  wire [32:0] end_beat={1'b0,cfg_base}+33'(cfg_span);
  wire config_ok=cfg_mode<=1 && cfg_kb>=1 && cfg_kb<=255 && cfg_rows>=1 &&
                 cfg_rows<=MAX_ROWS && cfg_base<32'h800000 && end_beat<=33'h800000 && hbm_ready;
  assign n_cwe='0;assign n_wvalid='0;assign n_wdata='0;assign n_wmask='0;
  assign led={error_code!=0,hbm_ready,!rst};

  // Serialized AXI-Lite slave: AW and W may arrive independently. Response payloads
  // remain stable until acknowledged. Reads and writes cannot race MEM_ADDR.
  logic aw_hold,w_hold,rd_pending,rd_mem;
  logic [11:0] aw_addr;
  logic [31:0] w_value,rd_value;
  logic [3:0] w_strobe;
  logic [1:0] rd_response,rd_bank,rd_kind;
  wire wr_fire=aw_hold&&w_hold&&!s_ctl_bvalid;
  wire ar_fire=s_ctl_arvalid&&s_ctl_arready;
  assign s_ctl_awready=!rst&&!aw_hold&&!s_ctl_bvalid&&!rd_pending&&!s_ctl_rvalid;
  assign s_ctl_wready=!rst&&!w_hold&&!s_ctl_bvalid&&!rd_pending&&!s_ctl_rvalid;
  assign s_ctl_arready=!rst&&!rd_pending&&!s_ctl_rvalid&&!aw_hold&&!w_hold&&
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
    case(s_ctl_araddr)
      12'h000:csr_value=32'h43324d42;
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
          if(s_ctl_araddr==12'(256+p*32+0)) begin csr_value=pc_issued[p];read_ok=1;end
          if(s_ctl_araddr==12'(256+p*32+4)) begin csr_value=pc_returned[p];read_ok=1;end
          if(s_ctl_araddr==12'(256+p*32+8)) begin csr_value=pc_peak[p];read_ok=1;end
          if(s_ctl_araddr==12'(256+p*32+12)) begin csr_value=pc_blocked[p];read_ok=1;end
          if(s_ctl_araddr==12'(256+p*32+16)) begin csr_value=pc_latency[p];read_ok=1;end
          if(s_ctl_araddr==12'(256+p*32+20)) begin csr_value=pc_credit[p];read_ok=1;end
        end
      end
    endcase
  end
  logic [3:0][79:0] result_q;
  always_ff @(posedge clk) begin
    if(rst) begin
      aw_hold<=0;w_hold<=0;s_ctl_bvalid<=0;s_ctl_bresp<=0;rd_pending<=0;
      s_ctl_rvalid<=0;s_ctl_rresp<=0;s_ctl_rdata<=0;
      cfg_mode<=1;cfg_kb<=9;cfg_rows<=257;cfg_base<=0;mem_addr<=0;
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
        rd_pending<=1;rd_value<=csr_value;rd_response<=read_ok?0:2;
        rd_mem<=s_ctl_araddr==12'h02c&&read_ok;rd_bank<=mem_addr[1:0];rd_kind<=result_kind;
        if(s_ctl_araddr==12'h02c&&read_ok)mem_addr<=mem_addr+1;
      end
      if(rd_pending)begin
        rd_pending<=0;s_ctl_rvalid<=1;s_ctl_rresp<=rd_response;
        if(!rd_mem)s_ctl_rdata<=rd_value;
        else case(rd_kind)
          0:s_ctl_rdata<=result_q[rd_bank][31:0];
          1:s_ctl_rdata<=result_q[rd_bank][63:32];
          default:s_ctl_rdata<={16'd0,result_q[rd_bank][79:64]};
        endcase
      end
    end
  end

  // Fixed-bank, reset-free synchronous memories. Preload ports are write-only;
  // the measured run only reads ACT and scales, and only writes the result bank.
  logic [1023:0] act_q;
  logic [31:0] da_q,sa_reg;
  logic [3:0][31:0] sw_q;
  logic [3:0][7:0] dw_q;
  logic [3:0] s_valid,s_ready,w_valid,w_ready,a_valid,a_ready;
  logic [3:0][1023:0] w_data,a_data;
  logic [3:0][31:0] s_sw,a_da,a_sa;
  logic [3:0][7:0] s_dw;
  logic [3:0] s_fetch;
  logic a_fetch;
  logic [3:0] pair_valid,pair_ready,pair_fetch;
  logic [7:0][511:0] half_q;
  logic cmd_ready,core_out_valid,core_out_decode,core_out_acc;
  logic [31:0] core_out_tag;
  logic [3:0] core_out_mask;
  logic [3:0][31:0] core_out_addr,core_out_data;
  logic [3:0][47:0] core_out_integer;
  logic pf_advance,pf_mode_wait,pf_credit_wait,pf_old_wait,pf_out_wait;
  logic [3:0] pf_w_wait,pf_s_wait,pf_a_wait;
  logic [15:0] pf_live,pf_results;
  logic [3:0][8:0] pf_w_level,pf_s_level,pf_a_level;
  wire core_reset=rst||state==IDLE||state==FLUSH;
  wire core_commit=core_out_valid&&busy&&state!=DRAIN&&state!=FLUSH;
  wire core_accept=state==COMMAND&&cmd_ready;
  logic all_returns;
  logic feed_done;
  logic [31:0] issued_now,returned_now,outstanding_now;
  logic unexpected,overflow;
  always_comb begin
    core_mask=0;pair_active=0;
    for(int c=0;c<4;c++)begin
      core_mask[c]=decode ? (32'(row_cursor)+32'(c)<cfg_rows):(c==0);
      cmd_addr[c]=32'(row_cursor)+(decode?32'(c):0);
      pair_active[c]=decode?core_mask[c]:(legacy_pair==2'(c));
    end
    n_cvalid=0;
    for(int p=0;p<8;p++)begin
      n_caddr[p]=group_base+25'(issued[p]);
      n_cvalid[p]=state==FEED&&pair_active[p/2]&&issued[p]<kb&&
                  (issued[p]-delivered[p/2]<9'(RD))&&error_code==0&&!abort_request;
    end
    w_valid=0;w_data=0;pair_ready=0;s_fetch=0;a_fetch=0;
    a_valid=0;a_data=0;a_da=0;a_sa=0;s_dw=0;s_sw=0;
    for(int c=0;c<4;c++)begin
      if(decode)begin w_valid[c]=pair_valid[c];w_data[c]={half_q[2*c+1],half_q[2*c]};pair_ready[c]=w_ready[c];end
      else if(c==0)begin w_valid[0]=pair_valid[legacy_pair];
        w_data[0]={half_q[2*legacy_pair+1],half_q[2*legacy_pair]};pair_ready[legacy_pair]=w_ready[0];end
      pair_fetch[c]=state==FEED&&pair_active[c]&&(!pair_valid[c]||pair_ready[c])&&
                     loaded[c]<returned[2*c]&&loaded[c]<returned[2*c+1];
      s_fetch[c]=state==FEED&&core_mask[c]&&(!s_valid[c]||s_ready[c])&&s_loaded[c]<kb;
      s_dw[c]=decode?dw_q[c]:dw_q[legacy_pair];
      s_sw[c]=decode?sw_q[c]:sw_q[legacy_pair];
    end
    a_fetch=state==FEED&&(!a_valid_reg||a_ready[0])&&a_loaded<kb;
    a_valid[0]=a_valid_reg;a_data[0]=act_q;a_da[0]=da_q;a_sa[0]=sa_reg;
    feed_done=state==FEED&&a_delivered==kb;
    for(int c=0;c<4;c++)begin
      if(pair_active[c]&&delivered[c]!=kb)feed_done=0;
      if(core_mask[c]&&s_delivered[c]!=kb)feed_done=0;
    end
    issued_now=0;returned_now=0;outstanding_now=0;unexpected=0;overflow=0;all_returns=1;
    for(int p=0;p<8;p++)begin
      issued_now+=32'(n_cvalid[p]&&n_cready[p]);
      returned_now+=32'(n_rvalid[p]);
      outstanding_now+=32'(issued[p]-returned[p]);
      if(issued[p]!=returned[p])all_returns=0;
      if(n_rvalid[p]&&(!busy||returned[p]>=issued[p]))unexpected=1;
      if(issued[p]-delivered[p/2]>9'(RD))overflow=1;
    end
  end
  logic a_valid_reg;
  for(genvar j=0;j<32;j++)begin:g_act
    (* ram_style="block" *) logic [31:0] memory[DEPTH];
    always_ff @(posedge clk)begin
      if(memory_write&&is_act&&mem_addr[4:0]==5'(j))memory[mem_addr[AW+4:5]]<=w_value;
      if(a_fetch)act_q[j*32+:32]<=memory[a_loaded[AW-1:0]];
    end
  end
  (* ram_style="block" *) logic [31:0] da_mem[DEPTH];
  always_ff @(posedge clk)begin
    if(memory_write&&is_da)da_mem[mem_addr[AW-1:0]]<=w_value;
    if(memory_write&&is_sa)sa_reg<=w_value;
    if(a_fetch)da_q<=da_mem[a_loaded[AW-1:0]];
  end
  for(genvar c=0;c<4;c++)begin:g_scale_result
    (* ram_style="block" *) logic [7:0] dw_mem[GROUPS*DEPTH];
    (* ram_style="block" *) logic [31:0] sw_mem[GROUPS];
    (* ram_style="block" *) logic [79:0] res_mem[GROUPS];
    wire fetch_scale=decode?s_fetch[c]:(legacy_pair==2'(c)&&s_fetch[0]);
    wire [8:0] scale_index=decode?s_loaded[c]:s_loaded[0];
    wire result_write=core_commit&&(core_out_decode?core_out_mask[c]:(core_out_tag[1:0]==2'(c)));
    always_ff @(posedge clk)begin
      if(memory_write&&is_dw&&mem_addr[9:8]==2'(c))dw_mem[{mem_addr[16:10],mem_addr[7:0]}]<=w_value[7:0];
      if(memory_write&&is_sw&&mem_addr[1:0]==2'(c))sw_mem[mem_addr[GW+1:2]]<=w_value;
      if(state==SWREAD)sw_q[c]<=sw_mem[row_cursor[GW+1:2]];
      if(fetch_scale)dw_q[c]<=dw_mem[{row_cursor[GW+1:2],scale_index[AW-1:0]}];
      if(result_write)res_mem[core_out_tag[GW+1:2]]<=core_out_decode?
        {core_out_integer[c],core_out_data[c]}:{core_out_integer[0],core_out_data[0]};
      if(ar_fire&&s_ctl_araddr==12'h02c&&is_result)result_q[c]<=res_mem[mem_addr[GW+1:2]];
    end
  end
  for(genvar p=0;p<8;p++)begin:g_return
    (* ram_style="block" *) logic [511:0] memory[RD];
    always_ff @(posedge clk)begin
      if(n_rvalid[p]&&busy&&returned[p]<issued[p])memory[returned[p][RW-1:0]]<=n_rdata[p];
      if(pair_fetch[p/2])half_q[p]<=memory[loaded[p/2][RW-1:0]];
    end
  end
  otpu_c2_mxu #(.DEPTH(DEPTH)) u_mxu(
    .clk,.rst(core_reset),.cmd_valid(state==COMMAND),.cmd_decode(decode),.cmd_acc(1'b0),
    .cmd_ready,.cmd_kb(kb),.cmd_mask(core_mask),.cmd_tag(32'(row_cursor)),.cmd_addr,
    .w_valid,.s_valid,.a_valid,.w_ready,.s_ready,.a_ready,.w_data,.a_data,
    .s_dw,.s_sw,.a_da,.a_sa,.old_req(),.old_ready(),.old_tag(),.old_mask(),.old_addr(),
    .old_valid(1'b0),.old_data('0),.out_valid(core_out_valid),.out_ready(busy),
    .out_decode(core_out_decode),.out_acc(core_out_acc),.out_tag(core_out_tag),
    .out_mask(core_out_mask),.out_addr(core_out_addr),.out_data(core_out_data),.out_integer(core_out_integer),
    .pf_advance,.pf_mode_wait,.pf_credit_wait,.pf_old_wait,.pf_out_wait,
    .pf_w_wait,.pf_s_wait,.pf_a_wait,.pf_live,.pf_results,.pf_w_level,.pf_s_level,.pf_a_level
  );
  always_ff @(posedge clk)begin
    if(rst)begin
      state<=IDLE;done<=0;aborted<=0;error_code<=0;row_cursor<=0;group_base<=0;
      pair_valid<=0;s_valid<=0;a_valid_reg<=0;a_loaded<=0;a_delivered<=0;
      for(int p=0;p<8;p++)begin issued[p]<=0;returned[p]<=0;end
      for(int c=0;c<4;c++)begin loaded[c]<=0;delivered[c]<=0;s_loaded[c]<=0;s_delivered[c]<=0;end
    end else begin
      if(clear_request)begin done<=0;aborted<=0;error_code<=0;end
      if(start)begin
        done<=0;aborted<=0;error_code<=0;row_cursor<=0;
        if(!config_ok||n_err!=0)begin done<=1;error_code<=!config_ok?1:2;end
        else state<=SWREAD;
      end
      if(state==SWREAD)begin
        group_base<=cfg_base[24:0]+25'(group_offset);
        state<=COMMAND;pair_valid<=0;s_valid<=0;a_valid_reg<=0;a_loaded<=0;a_delivered<=0;
        for(int p=0;p<8;p++)begin issued[p]<=0;returned[p]<=0;end
        for(int c=0;c<4;c++)begin loaded[c]<=0;delivered[c]<=0;s_loaded[c]<=0;s_delivered[c]<=0;end
      end
      if(core_accept)state<=FEED;
      if(state==FEED)begin
        for(int c=0;c<4;c++)begin
          if(!pair_valid[c]||pair_ready[c])pair_valid[c]<=pair_fetch[c];
          if(pair_fetch[c])loaded[c]<=loaded[c]+1'b1;
          if(pair_valid[c]&&pair_ready[c])delivered[c]<=delivered[c]+1'b1;
          if(!s_valid[c]||s_ready[c])s_valid[c]<=s_fetch[c];
          if(s_fetch[c])s_loaded[c]<=s_loaded[c]+1'b1;
          if(s_valid[c]&&s_ready[c])s_delivered[c]<=s_delivered[c]+1'b1;
        end
        if(!a_valid_reg||a_ready[0])a_valid_reg<=a_fetch;
        if(a_fetch)a_loaded<=a_loaded+1'b1;
        if(a_valid_reg&&a_ready[0])a_delivered<=a_delivered+1'b1;
        if(feed_done)begin
          if(32'(row_cursor)+(decode?4:1)>=cfg_rows)state<=WAITALL;
          else begin row_cursor<=row_cursor+(decode?10'd4:10'd1);state<=SWREAD;end
        end
      end
      for(int p=0;p<8;p++)begin
        if(n_cvalid[p]&&n_cready[p])issued[p]<=issued[p]+1'b1;
        if(n_rvalid[p]&&busy&&returned[p]<issued[p])returned[p]<=returned[p]+1'b1;
      end
      if(core_commit&&result_count+$countones(core_out_mask)==cfg_rows)begin state<=IDLE;done<=1;end
      if(abort_request)begin state<=DRAIN;aborted<=1;pair_valid<=0;s_valid<=0;a_valid_reg<=0;end
      if(busy&&(n_err!=0||unexpected||overflow))begin
        state<=DRAIN;pair_valid<=0;s_valid<=0;a_valid_reg<=0;error_code<=error_code|(n_err!=0?32'd2:0)|(unexpected?32'd4:0)|(overflow?32'd8:0);
      end
      if(state==DRAIN&&all_returns)state<=FLUSH;
      if(state==FLUSH)begin state<=IDLE;done<=1;pair_valid<=0;s_valid<=0;a_valid_reg<=0;end
    end
    if(rst||start||clear_request)begin
      cycles<=0;result_count<=0;groups_accepted<=0;groups_committed<=0;
      read_issued<=0;read_returned<=0;advances<=0;native_wait<=0;weight_wait<=0;
      scale_wait<=0;act_wait<=0;output_wait<=0;max_outstanding<=0;
      for(int p=0;p<8;p++)begin pc_issued[p]<=0;pc_returned[p]<=0;pc_peak[p]<=0;
        pc_blocked[p]<=0;pc_latency[p]<=0;pc_credit[p]<=0;end
    end else if(busy)begin
      cycles<=cycles+1'b1;
      groups_accepted<=groups_accepted+32'(core_accept);
      groups_committed<=groups_committed+32'(core_commit);
      if(core_commit)result_count<=result_count+$countones(core_out_mask);
      read_issued<=read_issued+issued_now;read_returned<=read_returned+returned_now;
      advances<=advances+32'(pf_advance);
      native_wait<=native_wait+32'(|(n_cvalid&~n_cready));
      weight_wait<=weight_wait+32'(|pf_w_wait);scale_wait<=scale_wait+32'(|pf_s_wait);
      act_wait<=act_wait+32'(|pf_a_wait);output_wait<=output_wait+32'(pf_out_wait);
      if(outstanding_now>max_outstanding)max_outstanding<=outstanding_now;
      for(int p=0;p<8;p++)begin
        pc_issued[p]<=pc_issued[p]+32'(n_cvalid[p]&&n_cready[p]);
        pc_returned[p]<=pc_returned[p]+32'(n_rvalid[p]);
        if(32'(issued[p]-returned[p])>pc_peak[p])pc_peak[p]<=32'(issued[p]-returned[p]);
        pc_blocked[p]<=pc_blocked[p]+32'(n_cvalid[p]&&!n_cready[p]);
        pc_latency[p]<=pc_latency[p]+32'(issued[p]!=returned[p]&&!n_rvalid[p]);
        pc_credit[p]<=pc_credit[p]+32'(state==FEED&&pair_active[p/2]&&issued[p]<kb&&issued[p]-delivered[p/2]>=9'(RD));
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)if(!rst)begin
    if(busy&&overflow)$fatal(1,"C2 HBM receive reservation overflow");
    if(state==SWREAD&&!all_returns)$fatal(1,"C2 HBM changed group with live responses");
    if(core_commit&&core_out_acc)$fatal(1,"C2 HBM plain benchmark unexpectedly ACC");
  end
`endif
endmodule
