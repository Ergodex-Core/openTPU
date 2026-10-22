// One checked logical job -> one private FP32 result. Local streamed payloads;
// no DMA/address dereference, permissions, instruction opcode or model state.
// Supported subset: one token/output, K=1..64, strict gradual/DAZ-FTZ, no deps.
module otpu_nvfp4_local_job (
  input logic clk,rst,
  input logic [63:0] session_generation,
  input logic desc_valid,
  output logic desc_ready,
  input otpu_nvfp4_job::descriptor_t descriptor,
  output logic reply_valid,
  input logic reply_ready,
  output logic reply_accepted,
  output logic [5:0] reply_fault,
  output logic [63:0] reply_request,
  input logic packet_valid,
  output logic packet_ready,
  input logic [255:0] packet_a,
  input logic [63:0] packet_w,
  input logic [15:0] packet_mask,
  input logic [7:0] packet_scale,
  input logic packet_scale_valid,packet_first,packet_last,
  input logic [31:0] packet_group,packet_k,scale_group,scale_k,
  input logic [63:0] packet_context,packet_generation,packet_request,packet_position,
                     packet_epoch,scale_epoch,
  output logic buffer_free,
  output logic [31:0] free_group,free_k,
  output logic [63:0] free_epoch,
  output logic result_valid,
  input logic result_ready,
  output logic result_success,output_committed,session_invalidated,
  output logic [31:0] result_bits,
  output logic [7:0] result_fault,
  output logic [63:0] result_context,result_generation,result_request,result_position,result_epoch
);
  import otpu_nvfp4_job::*;
  typedef enum logic [2:0] {IDLE,WAIT_PACKET,WAIT_TREE,WAIT_EPILOGUE,RESULT} state_t;
  state_t state;
  descriptor_t job;
  logic [31:0] index,blocks;
  logic [15:0] expected_mask;
  logic protocol_fault,tree_fault_seen;
  logic [31:0] private_bits;
  logic [7:0] private_fault;
  logic tree_ready,tree_valid,tree_out_ready,tree_fault,tree_ftz;
  logic [31:0] tree_bits;
  logic [7:0] tree_scale;
  logic [31:0] tree_tag;
  logic ep_valid,ep_ready;
  logic [31:0] ep_bits,ep_tag,ep_group,ep_epoch;
  logic [1:0] ep_mode;
  logic [4:0] ep_fault;
  logic ep_free,ep_commit;
  logic [31:0] ep_free_group,ep_free_k,ep_free_epoch;
  logic core_reset,generation_fault,reply_pending;

  // A session transition is an explicit reset-like cancellation boundary. Gate
  // handshakes immediately, flush both numerical units, then retain one failed
  // private completion (fault bit 7) with the ORIGINAL full identity until drain.
  assign session_invalidated=!rst && state!=IDLE && !generation_fault &&
                              job.generation!=session_generation;
  assign core_reset=rst || session_invalidated;
  assign reply_valid=!rst && reply_pending;
  assign desc_ready=!rst && state==IDLE && (!reply_valid || reply_ready);
  assign expected_mask=(job.k-index*32'd16>=16) ? 16'hffff :
                       (16'hffff >> (32'd16-(job.k-index*32'd16)));
  assign packet_ready=!rst && !session_invalidated && state==WAIT_PACKET && !reply_valid &&
                       tree_ready && packet_scale_valid;
  assign buffer_free=packet_valid && packet_ready;
  assign free_group=packet_group;
  assign free_k=packet_k;
  assign free_epoch=packet_epoch;
  assign tree_out_ready=state==WAIT_TREE && ep_ready;
  assign result_valid=!rst && !session_invalidated && state==RESULT;
  assign result_success=result_valid && private_fault==0;
  assign result_bits=result_success ? private_bits : 32'd0;
  assign result_fault=result_valid ? private_fault : 8'd0;
  assign output_committed=result_valid && result_ready && result_success;
  assign result_context=result_valid ? job.context_id : 64'd0;
  assign result_generation=result_valid ? job.generation : 64'd0;
  assign result_request=result_valid ? job.request_id : 64'd0;
  assign result_position=result_valid ? job.position : 64'd0;
  assign result_epoch=result_valid ? job.epoch : 64'd0;

  otpu_nvfp4_strict_dot16 strict_tree(
    .clk(clk),.rst(core_reset),.in_valid(packet_valid && packet_ready),.in_ready(tree_ready),
    .in_a(packet_a),.in_w(packet_w),.in_mask(expected_mask),.in_ftz(job.mode==1),
    .in_scale(packet_scale),.in_tag(index),.out_valid(tree_valid),.out_ready(tree_out_ready),
    .out_bits(tree_bits),.out_fault(tree_fault),.out_ftz(tree_ftz),
    .out_scale(tree_scale),.out_tag(tree_tag));
  otpu_nvfp4_scale_acc epilogue(
    .clk(clk),.rst(core_reset),.in_valid(tree_valid && state==WAIT_TREE),.in_ready(ep_ready),
    .in_mode(job.mode),.in_dot(41'sd0),.in_e_star(9'sd0),.in_strict_dot(tree_bits),
    .in_scale(tree_scale),.in_global(job.global_bits),.in_first(index==0),
    .in_last(index==blocks-1),.in_invalid(protocol_fault || tree_fault || tree_tag!=index ||
                                            tree_ftz!=(job.mode==1)),.in_discarded(1'b0),
    .in_k16_index(index),.in_tag(job.request_id[31:0]),.in_group(32'd0),
    .in_epoch(job.epoch[31:0]),.in_scale_valid(1'b1),.in_scale_group(32'd0),
    .in_scale_k16_index(index),.in_scale_epoch(job.epoch[31:0]),
    .out_valid(ep_valid),.out_ready(state==WAIT_EPILOGUE),.out_bits(ep_bits),.out_tag(ep_tag),
    .out_mode(ep_mode),.out_fault(ep_fault),.out_group(ep_group),.out_epoch(ep_epoch),
    .buffer_free(ep_free),.output_committed(ep_commit),.free_group(ep_free_group),
    .free_k16_index(ep_free_k),.free_epoch(ep_free_epoch));

  always_ff @(posedge clk) begin : control
    logic [5:0] validation;
    logic bad;
    if(rst) begin
      state<=IDLE;reply_pending<=0;reply_accepted<=0;reply_fault<=0;reply_request<=0;
      protocol_fault<=0;tree_fault_seen<=0;private_bits<=0;private_fault<=0;
      generation_fault<=0;
      index<=0;blocks<=0;job<='0;
    end else begin
      if(reply_valid && reply_ready) reply_pending<=0;
      if(session_invalidated) begin
        generation_fault<=1;private_fault<=8'h80;private_bits<=0;state<=RESULT;
      end else case(state)
        IDLE: if(desc_valid && desc_ready) begin
          validation=validate(descriptor,session_generation);
          reply_pending<=1;reply_fault<=validation;reply_accepted<=validation==0;
          reply_request<=descriptor.request_id;
          if(validation==0) begin
            job<=descriptor;blocks<=(descriptor.k+15)>>4;index<=0;
            protocol_fault<=0;tree_fault_seen<=0;private_bits<=0;private_fault<=0;
            generation_fault<=0;
            state<=WAIT_PACKET;
          end
        end
        WAIT_PACKET: if(buffer_free) begin
          bad=packet_context!=job.context_id || packet_generation!=job.generation ||
              packet_request!=job.request_id || packet_position!=job.position ||
              packet_epoch!=job.epoch || job.generation!=session_generation ||
              packet_group!=0 || packet_k!=index || scale_group!=packet_group ||
              scale_k!=packet_k || scale_epoch!=packet_epoch || packet_mask!=expected_mask ||
              packet_first!=(index==0) || packet_last!=(index==blocks-1);
          protocol_fault<=protocol_fault || bad;
          state<=WAIT_TREE;
        end
        WAIT_TREE: if(tree_valid && tree_out_ready) begin
          tree_fault_seen<=tree_fault_seen || tree_fault;
          if(index==blocks-1) state<=WAIT_EPILOGUE;
          else begin index<=index+1'b1;state<=WAIT_PACKET;end
        end
        WAIT_EPILOGUE: if(ep_valid) begin
          bad=ep_tag!=job.request_id[31:0] || ep_mode!=job.mode || ep_group!=0 ||
              ep_epoch!=job.epoch[31:0] || job.generation!=session_generation;
          private_fault<={1'b0,ep_fault,tree_fault_seen,(protocol_fault || bad)};
          private_bits<=ep_bits;
          state<=RESULT;
        end
        RESULT: if(result_ready) state<=IDLE;
        default:state<=IDLE;
      endcase
    end
  end
endmodule
