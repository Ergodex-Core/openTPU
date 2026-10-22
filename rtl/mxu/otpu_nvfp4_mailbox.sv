// NV41 ABI1.0 OCL peripheral, local offsets0..3ff (absolute400..7ff).
// Exclusive cooperative owner. All table/descriptor checks precede acceptance.
module otpu_nvfp4_mailbox #(
  parameter logic [63:0] BANK_BASE=0,
  parameter int BANK_LOG2=29
)(
  input logic clk,rst,legacy_idle,
  input logic [9:0] s_awaddr,s_araddr,
  input logic s_awvalid,s_wvalid,s_bready,s_arvalid,s_rready,
  output logic s_awready,s_wready,s_bvalid,s_arready,s_rvalid,
  input logic [31:0] s_wdata,
  input logic [3:0] s_wstrb,
  output logic [1:0] s_bresp,s_rresp,
  output logic [31:0] s_rdata,
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
  import otpu_nvfp4_job::*;
  localparam logic [31:0] MAGIC=32'h314a344e,ABI=32'h00010000;
  logic [31:0] stage [64],windows [32];
  logic [255:0] stage_written;
  logic [127:0] window_written;
  logic [63:0] cfg_gen,cfg_owner,generation,owner,accept_seq,last_epoch;
  logic epoch_valid,session_valid,armed,running,completion_valid,reset_pending;
  logic [31:0] last_error;
  descriptor_t parsed,job;
  logic [31:0] validation_error;
  logic [63:0] wb [4],ws [4];
  buffer_t operand [4];
  logic [64:0] absolute_address,allocation_end,operand_end,line_end;
  logic [63:0] offset,handle,required_bytes,padded_bytes;
  integer found,match_count;
  logic [7:0] e_status,e_faults,c_status,c_faults;
  logic [31:0] e_bits,c_bits;
  logic e_visible,c_visible,e_busy,e_late,e_done,start,abort_request;
  logic [63:0] c_sequence;
  wire command_cycle=aw_pending && w_pending && !s_bvalid && aw==10'h014 && strobe==4'hf;
  assign abort_request=command_cycle && running && !e_late &&
    ((wd==3 && !e_done) || (wd==4 && generation!=64'hffffffffffffffff));
  logic aw_pending,w_pending;
  logic [9:0] aw;
  logic [31:0] wd;
  logic [3:0] strobe;

  function automatic logic [63:0] d64(input integer n);
    return {stage[n+1],stage[n]};
  endfunction
  function automatic logic [63:0] w64(input integer n);
    return {windows[n+1],windows[n]};
  endfunction
  function automatic logic [31:0] merge_bytes(input logic [31:0] old_value,new_value,input logic [3:0] enables);
    logic [31:0] result;
    result=old_value;
    for(integer b=0;b<4;b++)if(enables[b])result[8*b+:8]=new_value[8*b+:8];
    return result;
  endfunction

  assign owns_memory=!rst && (running || completion_valid);
  assign s_awready=!rst && !aw_pending && !s_bvalid;
  assign s_wready=!rst && !w_pending && !s_bvalid;
  assign s_arready=!rst && !s_rvalid;
  always_comb begin
    parsed='0;validation_error=0;
    absolute_address=0;allocation_end=0;operand_end=0;line_end=0;
    offset=0;handle=0;required_bytes=0;padded_bytes=0;found=0;match_count=0;
    parsed.abi_major=1;parsed.contract=127;parsed.mode=stage[5][1:0];
    parsed.tokens=1;parsed.outputs=1;parsed.k=stage[4];parsed.global_bits=stage[6];
    for(integer n=0;n<8;n++)parsed.digest[n*32+:32]=stage[8+n];
    parsed.context_id=d64(16);parsed.generation=d64(18);parsed.request_id=d64(20);
    parsed.position=d64(22);parsed.epoch=d64(24);
    for(integer r=0;r<4;r++)begin
      found=0;match_count=0;handle=d64(26+8*r);offset=d64(28+8*r);
      for(integer w=0;w<4;w++)if(handle!=0 && handle==w64(8*w))begin found=w;match_count=match_count+1;end
      wb[r]=w64(8*found+2);ws[r]=w64(8*found+4);
      absolute_address={1'b0,wb[r]}+{1'b0,offset};
      allocation_end={1'b0,wb[r]}+{1'b0,ws[r]};
      operand_end={1'b0,offset}+{1'b0,d64(30+8*r)};
      required_bytes=(r==0) ? ({32'd0,stage[4]}<<1) :
                     (r==1) ? (({32'd0,stage[4]}+1)>>1) :
                     (r==2) ? (({32'd0,stage[4]}+15)>>4) : 64'd4;
      padded_bytes=(required_bytes+63)&~64'd63;
      line_end={1'b0,offset}+{1'b0,padded_bytes};
      operand[r].address=absolute_address[63:0];operand[r].span=d64(30+8*r);operand[r].stride=d64(32+8*r);
      if(match_count!=1 || absolute_address[64] || allocation_end[64] || (wb[r]-BANK_BASE)>=(64'd1<<BANK_LOG2) ||
         allocation_end>{1'b0,BANK_BASE}+(65'd1<<BANK_LOG2) || ws[r]==0 ||
         wb[r][5:0]!=0 || ws[r][5:0]!=0 || offset[5:0]!=0 ||
         operand_end>{1'b0,ws[r]} || line_end>{1'b0,ws[r]})validation_error=7;
      if(windows[8*found+7]!=0 || windows[8*found+6]==0 || (windows[8*found+6]&~32'd3)!=0 ||
         (r==3 ? !windows[8*found+6][1] : !windows[8*found+6][0]))validation_error=8;
    end
    parsed.a=operand[0];parsed.w=operand[1];parsed.scales=operand[2];parsed.y=operand[3];
    for(integer aidx=0;aidx<4;aidx++)for(integer bidx=0;bidx<aidx;bidx++)
      if(w64(8*aidx)==w64(8*bidx))validation_error=7;
    // All registered entries must be valid, even when roles share an input window.
    for(integer i=0;i<4;i++)begin
      if(w64(8*i)==0 || w64(8*i+4)==0 || windows[8*i+2][5:0]!=0 ||
         windows[8*i+4][5:0]!=0 || (w64(8*i+2)-BANK_BASE)>=(64'd1<<BANK_LOG2) ||
         {1'b0,w64(8*i+2)}+{1'b0,w64(8*i+4)}>{1'b0,BANK_BASE}+(65'd1<<BANK_LOG2))validation_error=7;
      if(windows[8*i+7]!=0 || windows[8*i+6]==0 || (windows[8*i+6]&~32'd3)!=0)validation_error=8;
    end
    for(integer r=0;r<3;r++)
      if({1'b0,wb[3]}<{1'b0,wb[r]}+{1'b0,ws[r]} &&
         {1'b0,wb[r]}<{1'b0,wb[3]}+{1'b0,ws[3]})validation_error=8;
    if(validate(parsed,generation)!=0)validation_error=2;
    if(stage[5]>1 || (stage[5]==0 && parsed.digest!=GRADUAL_DIGEST) ||
       (stage[5]==1 && parsed.digest!=FTZ_DIGEST))validation_error=3;
    if(stage[4]==0 || stage[4]>64)validation_error=4;
    if(parsed.context_id!=owner || parsed.generation!=generation || !armed)validation_error=5;
    if(epoch_valid && parsed.epoch<=last_epoch)validation_error=6;
    if(stage[0]!=MAGIC || stage[1]!=ABI || stage[2]!=256 || stage[3]!=0 || stage[7]!=0)validation_error=2;
    for(integer n=58;n<64;n++)if(stage[n]!=0)validation_error=2;
    if(!(&stage_written) || !(&window_written))validation_error=10;
    if(!legacy_idle || n_err)validation_error=9;
    if(accept_seq==64'hffffffffffffffff)validation_error=13;
  end

  otpu_nvfp4_hbm_job #(.BANK_BASE(BANK_BASE)) u_engine (
    .clk,.rst,.start,.abort_request,.descriptor(job),.busy(e_busy),.too_late(e_late),
    .complete(e_done),.complete_ready(running),.status(e_status),.faults(e_faults),.bits(e_bits),
    .output_visible(e_visible),.memory_outstanding,.n_cvalid,.n_cwe,.n_cready,.n_caddr,
    .n_wvalid,.n_wready,.n_wdata,.n_wmask,.n_rvalid,.n_rdata,.n_wdone,.n_err);

  always_ff @(posedge clk) begin
    if(rst)begin
      aw_pending<=0;w_pending<=0;s_bvalid<=0;s_rvalid<=0;s_bresp<=0;s_rresp<=0;s_rdata<=0;
      aw<=0;wd<=0;strobe<=0;stage_written<=0;window_written<=0;
      cfg_gen<=0;cfg_owner<=0;generation<=0;owner<=0;accept_seq<=0;last_epoch<=0;
      epoch_valid<=0;session_valid<=0;armed<=0;running<=0;completion_valid<=0;reset_pending<=0;
      last_error<=0;job<='0;c_status<=0;c_faults<=0;c_bits<=0;c_visible<=0;c_sequence<=0;
      start<=0;
      for(integer i=0;i<64;i++)stage[i]<=0;
      for(integer i=0;i<32;i++)windows[i]<=0;
    end else begin
      start<=0;
      if(s_bvalid && s_bready)s_bvalid<=0;
      if(s_rvalid && s_rready)s_rvalid<=0;
      if(s_awvalid && s_awready)begin aw_pending<=1;aw<=s_awaddr;end
      if(s_wvalid && s_wready)begin w_pending<=1;wd<=s_wdata;strobe<=s_wstrb;end
      if(e_done && running)begin
        running<=0;completion_valid<=1;c_status<=(reset_pending && e_status==3) ? 8'd4 : e_status;
        c_faults<=e_faults;c_bits<=e_bits;c_visible<=e_visible;c_sequence<=accept_seq;
      end
      if(aw_pending && w_pending && !s_bvalid)begin
        aw_pending<=0;w_pending<=0;s_bvalid<=1;s_bresp<=0;last_error<=0;
        if(aw[1:0]!=0)begin s_bresp<=2;last_error<=2;end
        else if(aw==10'h014)begin
          if(strobe!=4'hf)begin s_bresp<=2;last_error<=2;end
          else case(wd)
            1:if(owns_memory || e_busy)begin s_bresp<=2;last_error<=1;end
              else if(validation_error!=0)begin s_bresp<=2;last_error<=validation_error;end
              else begin
                job<=parsed;accept_seq<=accept_seq+1'b1;last_epoch<=parsed.epoch;epoch_valid<=1;
                running<=1;start<=1;stage_written<=0;
              end
            2:if(!completion_valid)begin s_bresp<=2;last_error<=12;end
              else begin
                completion_valid<=0;
                if(reset_pending)begin
                  generation<=generation+1'b1;cfg_gen<=generation+1'b1;armed<=0;
                  stage_written<=0;window_written<=0;epoch_valid<=0;reset_pending<=0;
                end
              end
            3:if(!running || e_late || e_done)begin s_bresp<=2;last_error<=11;end
              else begin end
            4:if(generation==64'hffffffffffffffff)begin s_bresp<=2;last_error<=13;end
              else if(owns_memory)begin reset_pending<=1;end
              else begin
                generation<=generation+1'b1;cfg_gen<=generation+1'b1;armed<=0;
                stage_written<=0;window_written<=0;epoch_valid<=0;
              end
            5:if(owns_memory || e_busy)begin s_bresp<=2;last_error<=1;end
              else if(cfg_gen==0 || cfg_gen<generation ||
                      (session_valid && cfg_gen==generation && cfg_owner!=owner) || !(&window_written))begin
                s_bresp<=2;last_error<=5;
              end else begin
                if(!session_valid || cfg_gen!=generation)epoch_valid<=0;
                generation<=cfg_gen;owner<=cfg_owner;session_valid<=1;armed<=1;
              end
            default:begin s_bresp<=2;last_error<=2;end
          endcase
        end else if(owns_memory || e_busy)begin s_bresp<=2;last_error<=1;end
        else if(aw>=10'h200 && aw<10'h300)begin
          stage[aw[7:2]]<=merge_bytes(stage[aw[7:2]],wd,strobe);
          stage_written[4*aw[7:2]+:4]<=stage_written[4*aw[7:2]+:4]|strobe;
        end else if(aw>=10'h100 && aw<10'h180)begin
          windows[aw[6:2]]<=merge_bytes(windows[aw[6:2]],wd,strobe);
          window_written[4*aw[6:2]+:4]<=window_written[4*aw[6:2]+:4]|strobe;armed<=0;
        end else if(aw==10'h028 || aw==10'h02c)begin
          if(aw[2])cfg_gen[63:32]<=merge_bytes(cfg_gen[63:32],wd,strobe);
          else cfg_gen[31:0]<=merge_bytes(cfg_gen[31:0],wd,strobe);armed<=0;
        end else if(aw==10'h030 || aw==10'h034)begin
          if(aw[2])cfg_owner[63:32]<=merge_bytes(cfg_owner[63:32],wd,strobe);
          else cfg_owner[31:0]<=merge_bytes(cfg_owner[31:0],wd,strobe);armed<=0;
        end else begin s_bresp<=2;last_error<=2;end
      end
      if(s_arvalid && s_arready)begin
        s_rvalid<=1;s_rresp<=0;s_rdata<=0;
        case(s_araddr)
          10'h000:s_rdata<=32'h3134564e;
          10'h004:s_rdata<=ABI;
          10'h008:s_rdata<=256;
          10'h00c:s_rdata<=32'h00030140; // K64, in-flight1, strict profile bits[17:16]
          10'h010:s_rdata<={27'd0,reset_pending,e_late,memory_outstanding,completion_valid,owns_memory};
          10'h018:s_rdata<=last_error;
          10'h01c:s_rdata<=accept_seq[31:0];10'h020:s_rdata<=accept_seq[63:32];
          10'h028:s_rdata<=cfg_gen[31:0];10'h02c:s_rdata<=cfg_gen[63:32];
          10'h030:s_rdata<=cfg_owner[31:0];10'h034:s_rdata<=cfg_owner[63:32];
          10'h038:s_rdata<=last_epoch[31:0];10'h03c:s_rdata<=last_epoch[63:32];
          10'h040:s_rdata<=completion_valid ? {15'd0,c_visible,c_faults,c_status} : 0;
          10'h044:s_rdata<=completion_valid ? c_bits : 0;
          10'h048:s_rdata<=completion_valid ? c_sequence[31:0] : 0;
          10'h04c:s_rdata<=completion_valid ? c_sequence[63:32] : 0;
          10'h0c0:s_rdata<=BANK_BASE[31:0];10'h0c4:s_rdata<=BANK_BASE[63:32];
          10'h0c8:s_rdata<=32'((64'd1<<BANK_LOG2));10'h0cc:s_rdata<=32'((64'd1<<BANK_LOG2)>>32);
          10'h0d0:s_rdata<=generation[31:0];10'h0d4:s_rdata<=generation[63:32];
          default:begin
            if(s_araddr>=10'h080 && s_araddr<10'h0a0 && s_araddr[1:0]==0)
              s_rdata<=GRADUAL_DIGEST[32*int'(s_araddr[4:2])+:32];
            else if(s_araddr>=10'h0a0 && s_araddr<10'h0c0 && s_araddr[1:0]==0)
              s_rdata<=FTZ_DIGEST[32*int'(s_araddr[4:2])+:32];
            else if(s_araddr>=10'h050 && s_araddr<10'h078 && s_araddr[1:0]==0)begin
              case(s_araddr[5:3])
                2:s_rdata<=completion_valid ? (s_araddr[2] ? job.context_id[63:32] : job.context_id[31:0]) : 0;
                3:s_rdata<=completion_valid ? (s_araddr[2] ? job.generation[63:32] : job.generation[31:0]) : 0;
                4:s_rdata<=completion_valid ? (s_araddr[2] ? job.request_id[63:32] : job.request_id[31:0]) : 0;
                5:s_rdata<=completion_valid ? (s_araddr[2] ? job.position[63:32] : job.position[31:0]) : 0;
                6:s_rdata<=completion_valid ? (s_araddr[2] ? job.epoch[63:32] : job.epoch[31:0]) : 0;
                default:s_rdata<=0;
              endcase
            end else s_rresp<=2;
          end
        endcase
      end
    end
  end
endmodule
