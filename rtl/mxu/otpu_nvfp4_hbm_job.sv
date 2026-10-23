// One strict job over one reserved native HBM channel. Full-line reads, one
// outstanding request; successful completion only after the B-derived n_wdone.
module otpu_nvfp4_hbm_job #(
  parameter logic [63:0] BANK_BASE=64'd0
)(
  input logic clk,rst,start,abort_request,
  input otpu_nvfp4_job::descriptor_t descriptor,
  output logic busy,too_late,complete,
  input logic complete_ready,
  output logic [7:0] status,faults,
  output logic [31:0] bits,
  output logic output_visible,memory_outstanding,
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
  typedef enum logic [3:0] {IDLE,DESC,REPLY,READ_CMD,READ_WAIT,PACKET,CORE_RESULT,WRITE,WRITE_WAIT,DONE} state_t;
  state_t state;
  descriptor_t job;
  logic [31:0] index;
  logic [1:0] role;
  logic aborted,mem_error,flush,cmd_sent,data_sent;
  logic [15:0] write_base;
  logic [255:0] a;
  logic [63:0] w;
  logic [7:0] scale;
  logic [63:0] address;
  logic [31:0] private_bits;
  logic core_dready,reply_valid,reply_accepted,packet_ready,result_valid,result_success;
  logic [5:0] reply_fault;
  logic [7:0] result_fault;
  logic [31:0] result_bits;
  logic [15:0] mask;
  wire abort_now=aborted || (abort_request && !too_late);
  wire core_rst=rst || flush;
  assign busy=!rst && state!=IDLE;
  assign too_late=state==WRITE || state==WRITE_WAIT || (state==DONE && output_visible);
  assign complete=!rst && state==DONE;
  assign memory_outstanding=!rst && (state==READ_CMD || state==READ_WAIT || state==WRITE || state==WRITE_WAIT);
  assign mask=job.k-index*16>=16 ? 16'hffff : (16'hffff>>(16-(job.k-index*16)));
  always_comb begin
    case(role)
      0:address=job.a.address+64'(index)*32;
      1:address=job.w.address+64'(index)*8;
      default:address=job.scales.address+64'(index);
    endcase
    if(state==WRITE || state==WRITE_WAIT)address=job.y.address;
  end
  assign n_cvalid=!rst && (state==READ_CMD || (state==WRITE && !cmd_sent));
  assign n_cwe=state==WRITE || state==WRITE_WAIT;
  assign n_caddr=25'((address-BANK_BASE)>>6);
  assign n_wvalid=!rst && state==WRITE && !data_sent;
  assign n_wdata={480'd0,private_bits};
  assign n_wmask=64'hf;
  otpu_nvfp4_local_job u_core (
    .clk,.rst(core_rst),.session_generation(job.generation),
    .desc_valid(state==DESC && !abort_now),.desc_ready(core_dready),.descriptor(job),
    .reply_valid,.reply_ready(state==REPLY),.reply_accepted,.reply_fault,.reply_request(),
    .packet_valid(state==PACKET && !abort_now),.packet_ready,.packet_a(a),.packet_w(w),.packet_mask(mask),
    .packet_scale(scale),.packet_scale_valid(1'b1),.packet_first(index==0),.packet_last(index==(job.k-1)/16),
    .packet_group(32'd0),.packet_k(index),.scale_group(32'd0),.scale_k(index),
    .packet_context(job.context_id),.packet_generation(job.generation),.packet_request(job.request_id),
    .packet_position(job.position),.packet_epoch(job.epoch),.scale_epoch(job.epoch),
    .buffer_free(),.free_group(),.free_k(),.free_epoch(),
    .result_valid,.result_ready(state==CORE_RESULT && !abort_now),.result_success,.result_bits,.result_fault,
    .output_committed(),.session_invalidated(),.result_context(),.result_generation(),.result_request(),
    .result_position(),.result_epoch());
  always_ff @(posedge clk) begin
    if(rst) begin
      state<=IDLE;job<='0;index<=0;role<=0;aborted<=0;mem_error<=0;flush<=0;
      cmd_sent<=0;data_sent<=0;write_base<=0;private_bits<=0;
      status<=0;faults<=0;bits<=0;output_visible<=0;a<=0;w<=0;scale<=0;
    end else begin
      flush<=0;
      if(busy && n_err)mem_error<=1;
      if(abort_request && !too_late)aborted<=1;
      case(state)
        IDLE:if(start) begin
          job<=descriptor;index<=0;role<=0;aborted<=0;mem_error<=0;
          status<=0;faults<=0;bits<=0;output_visible<=0;state<=DESC;
        end
        DESC:if(abort_now)begin status<=3;flush<=1;state<=DONE;end
             else if(core_dready)state<=REPLY;
        REPLY:if(reply_valid)begin
          if(abort_now)begin status<=3;flush<=1;state<=DONE;end
          else if(!reply_accepted)begin status<=2;faults<=8'h80;flush<=1;state<=DONE;end
          else state<=READ_CMD;
        end
        READ_CMD:if(n_cready)state<=READ_WAIT; // valid stays asserted through cancellation
        READ_WAIT:if(n_rvalid)begin
          if(abort_now || mem_error || n_err)begin
            status<=abort_now ? 3 : 2;faults<=abort_now ? 0 : 8'h80;flush<=1;state<=DONE;
          end else begin
            case(role)
              0:a<=256'(n_rdata>>(8*address[5:0]));
              1:w<=64'(n_rdata>>(8*address[5:0]));
              default:scale<=8'(n_rdata>>(8*address[5:0]));
            endcase
            if(role==2)state<=PACKET;
            else begin role<=role+1'b1;state<=READ_CMD;end
          end
        end
        PACKET:if(abort_now)begin status<=3;flush<=1;state<=DONE;end
          else if(packet_ready)begin
            if(index==(job.k-1)/16)state<=CORE_RESULT;
            else begin index<=index+1'b1;role<=0;state<=READ_CMD;end
          end
        CORE_RESULT:if(abort_now)begin status<=3;flush<=1;state<=DONE;end
          else if(result_valid)begin
            if(!result_success || mem_error || n_err)begin
              status<=2;faults<=result_fault | ((mem_error || n_err) ? 8'h80 : 8'h0);state<=DONE;
            end else begin
              private_bits<=result_bits;write_base<=n_wdone;cmd_sent<=0;data_sent<=0;state<=WRITE;
            end
          end
        WRITE:begin
          if(n_cvalid && n_cready)cmd_sent<=1;
          if(n_wvalid && n_wready)data_sent<=1;
          if((cmd_sent || n_cready) && (data_sent || n_wready))state<=WRITE_WAIT;
        end
        WRITE_WAIT:if(n_wdone==write_base+16'd1)begin
          output_visible<=1;
          if(mem_error || n_err)begin status<=2;faults<=8'h80;bits<=0;end
          else begin status<=1;bits<=private_bits;end
          state<=DONE;
        end
        DONE:if(complete_ready)begin state<=IDLE;flush<=1;end
        default:state<=IDLE;
      endcase
    end
  end
endmodule
