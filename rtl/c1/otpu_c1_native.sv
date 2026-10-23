// C1 E16 adapter over eight in-order 64-byte native ports. Two engines share
// each port; up to 16 adjacent commands preserve Alex burst coalescing. Every
// accepted command reserves one 64-byte engine buffer until both halves retire.
// Native responses cannot stall. ABORT must drain pending reads before reset.
module otpu_c1_native #(
  parameter int RD=32, SW=2, RW=$clog2(RD), TQ=2*RD, TW=$clog2(TQ)
)(
  input logic clk,rst,enable,
  input logic [22:0] base,
  input logic [15:0] req_valid,
  output logic [15:0] req_ready,
  input logic [15:0][SW-1:0] req_slot,
  input logic [15:0][31:0] req_row,req_generation,
  input logic [15:0][7:0] req_kb,
  output logic [15:0] rsp_valid,
  input logic [15:0] rsp_ready,
  output logic [15:0][SW-1:0] rsp_slot,
  output logic [15:0][31:0] rsp_row,rsp_generation,
  output logic [15:0][9:0] rsp_group,
  output logic [15:0][255:0] rsp_weights,
  output logic [7:0] n_cvalid,
  input logic [7:0] n_cready,
  output logic [7:0][24:0] n_caddr,
  input logic [7:0] n_rvalid,n_err,
  input logic [7:0][511:0] n_rdata,
  output logic drained,error,
  output logic [7:0][31:0] pc_issued,pc_returned,pc_peak,pc_blocked,pc_credit
);
  typedef struct packed {
    logic engine;
    logic [SW-1:0] slot;
    logic [31:0] row,generation;
    logic [9:0] group_id;
  } tag_t;
  logic [15:0] active;
  logic [15:0][SW-1:0] slot;
  logic [15:0][31:0] row,generation;
  logic [15:0][7:0] kb;
  logic [15:0][9:0] group_id;
  logic [15:0][22:0] row_base;
  logic [15:0][RW:0] reserved;
  logic [7:0] selected;
  logic [7:0][3:0] quantum;
  logic [7:0][TW:0] outstanding;
  logic [7:0][TW-1:0] tp,tr;
  logic [7:0] issue,receive;
  localparam int TAG_W=$bits(tag_t);
  tag_t front[8];
  logic [15:0] sent,arrived,pop;
  logic [15:0] buffer_valid,upper;
  logic [15:0][511:0] buffer_data;
  tag_t buffer_tag[16];
  logic [15:0][RW-1:0] wp,rp;
  logic [15:0][RW:0] queued;

  initial begin
    if(RD<2 || (RD&(RD-1))!=0 || SW!=2) $fatal(1,"C1 native ABI requires power-of-two RD and four slots");
  end
  always_comb begin
    drained=1;
    for(int p=0;p<8;p++) begin
      drained &= outstanding[p]==0;
      n_cvalid[p]=enable && active[2*p+int'(selected[p])] &&
                  reserved[2*p+int'(selected[p])]<RD && outstanding[p]<TQ;
      n_caddr[p]={2'b0,row_base[2*p+int'(selected[p])]}+
                 25'(group_id[2*p+int'(selected[p])]>>1);
      issue[p]=n_cvalid[p]&&n_cready[p];
      receive[p]=n_rvalid[p]&&outstanding[p]!=0;
    end
    for(int e=0;e<16;e++) begin
      req_ready[e]=enable&&!active[e];
      sent[e]=issue[e/2]&&selected[e/2]==1'(e%2);
      arrived[e]=receive[e/2]&&front[e/2].engine==1'(e%2);
      rsp_valid[e]=buffer_valid[e]&&enable;
      rsp_slot[e]=buffer_tag[e].slot;
      rsp_row[e]=buffer_tag[e].row;
      rsp_generation[e]=buffer_tag[e].generation;
      rsp_group[e]=buffer_tag[e].group_id+10'(upper[e]);
      rsp_weights[e]=upper[e]?buffer_data[e][511:256]:buffer_data[e][255:0];
      pop[e]=rsp_valid[e]&&rsp_ready[e]&&upper[e];
    end
  end
  for(genvar e=0;e<16;e++) begin: engines
    (* ram_style="block" *) logic [511:0] data_q[RD];
    // Packed vectors infer distributed RAM in Vivado; tags retain their full bit layout.
    (* ram_style="distributed" *) logic [TAG_W-1:0] tag_q[RD];
    wire fetch=queued[e]!=0&&(!buffer_valid[e]||pop[e]);
    always_ff @(posedge clk) begin
      if(arrived[e]) begin data_q[wp[e]]<=n_rdata[e/2];tag_q[wp[e]]<=front[e/2];end
      if(fetch) begin buffer_data[e]<=data_q[rp[e]];buffer_tag[e]<=tag_q[rp[e]];end
      if(rst) begin
        active[e]<=0;group_id[e]<=0;reserved[e]<=0;wp[e]<=0;rp[e]<=0;queued[e]<=0;
        buffer_valid[e]<=0;upper[e]<=0;
      end else begin
        if(req_valid[e]&&req_ready[e]) begin
          active[e]<=1;slot[e]<=req_slot[e];row[e]<=req_row[e];generation[e]<=req_generation[e];
          kb[e]<=req_kb[e];group_id[e]<=0;
          row_base[e]<=base+23'(({req_row[e][8:4],req_row[e][0]})*9'(req_kb[e])*2);
        end
        if(sent[e]) begin
          group_id[e]<=group_id[e]+2;
          if(group_id[e]+2=={kb[e],2'b00})active[e]<=0;
        end
        case({sent[e],pop[e]})
          2'b10:reserved[e]<=reserved[e]+1'b1;
          2'b01:reserved[e]<=reserved[e]-1'b1;
          default:;
        endcase
        if(arrived[e])wp[e]<=wp[e]+1'b1;
        if(fetch)rp[e]<=rp[e]+1'b1;
        case({arrived[e],fetch})
          2'b10:queued[e]<=queued[e]+1'b1;
          2'b01:queued[e]<=queued[e]-1'b1;
          default:;
        endcase
        if(rsp_valid[e]&&rsp_ready[e]) begin
          if(upper[e])buffer_valid[e]<=0;
          else upper[e]<=1;
        end
        if(fetch)begin buffer_valid[e]<=1;upper[e]<=0;end
      end
    end
  end
  for(genvar p=0;p<8;p++) begin: ports
    (* ram_style="distributed" *) logic [TAG_W-1:0] tags[TQ];
    assign front[p]=tags[tr[p]];
    wire [3:0] e={3'(p),selected[p]};
    wire other_eligible=active[{3'(p),!selected[p]}]&&reserved[{3'(p),!selected[p]}]<RD;
    always_ff @(posedge clk) begin
      if(issue[p])tags[tp[p]]<={selected[p],slot[e],row[e],generation[e],group_id[e]};
      if(rst)begin
        selected[p]<=0;quantum[p]<=0;tp[p]<=0;tr[p]<=0;outstanding[p]<=0;
        pc_issued[p]<=0;pc_returned[p]<=0;pc_peak[p]<=0;pc_blocked[p]<=0;pc_credit[p]<=0;
      end else begin
        // A valid command never changes selection while native READY is low.
        if((!active[e]||reserved[e]==RD)&&other_eligible)begin selected[p]<=!selected[p];quantum[p]<=0;end
        if(issue[p])begin
          tp[p]<=tp[p]+1'b1;pc_issued[p]<=pc_issued[p]+1;
          quantum[p]<=quantum[p]+1'b1;
          if((quantum[p]==15||group_id[e]+2=={kb[e],2'b00})&&other_eligible) begin
            selected[p]<=!selected[p];quantum[p]<=0;
          end
        end
        if(receive[p])begin tr[p]<=tr[p]+1'b1;pc_returned[p]<=pc_returned[p]+1;end
        case({issue[p],receive[p]})
          2'b10:outstanding[p]<=outstanding[p]+1'b1;
          2'b01:outstanding[p]<=outstanding[p]-1'b1;
          default:;
        endcase
        if(32'(outstanding[p])+(issue[p]?1:0)-(receive[p]?1:0)>pc_peak[p])pc_peak[p]<=32'(outstanding[p])+(issue[p]?1:0)-(receive[p]?1:0);
        if(n_cvalid[p]&&!n_cready[p])pc_blocked[p]<=pc_blocked[p]+1;
        if(enable&&active[e]&&reserved[e]==RD)pc_credit[p]<=pc_credit[p]+1;
      end
    end
  end
  always_ff @(posedge clk) begin
    if(rst)error<=0;
    else begin
      if(|n_err)error<=1;
      for(int p=0;p<8;p++)if(n_rvalid[p]&&!receive[p])error<=1;
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)if(!rst) begin
    for(int e=0;e<16;e++)begin
      if(reserved[e]>RD||queued[e]>RD)$fatal(1,"C1 native storage reservation overflow");
      if(req_valid[e]&&req_ready[e]&&(req_row[e]>=512||req_row[e]%16!=e||req_kb[e]==0||req_kb[e]>128))$fatal(1,"C1 invalid native row request");
    end
  end
`endif
endmodule
