// F2 platform adapter; does not change the accelerator's native interface.
// Coalesce adjacent read commands into bounded AXI4 bursts; writes remain single beat.
// Commands and write data are independent, in-order streams. Read/write direction
// changes drain older responses, preserving the native read-after-write contract.
// All reads use ID 0. Native responses cannot stall; upstream reserves their space.
module otpu_native_axi #(
  parameter logic [33:0] BASE_ADDR = 34'd0,
  parameter int QD = 16,
  parameter int MAX_OUT = 128,
  parameter int READ_BURST = 16,
  parameter int READ_IDLE = 4
)(
  input logic clk, rst,
  input logic n_cvalid, output logic n_cready,
  input logic n_cwe, input logic [24:0] n_caddr,
  input logic n_wvalid, output logic n_wready,
  input logic [511:0] n_wdata, input logic [63:0] n_wmask,
  output logic n_rvalid, output logic [511:0] n_rdata,
  output logic [15:0] n_wdone, output logic n_err,
  output logic [33:0] m_awaddr, output logic m_awvalid,
  input logic m_awready,
  output logic [511:0] m_wdata, output logic [63:0] m_wstrb,
  output logic m_wvalid, input logic m_wready,
  input logic m_bvalid, input logic [1:0] m_bresp, output logic m_bready,
  output logic [33:0] m_araddr, output logic [7:0] m_arlen,
  output logic m_arvalid,
  input logic m_arready,
  input logic [511:0] m_rdata, input logic m_rvalid, m_rlast,
  input logic [1:0] m_rresp, output logic m_rready
);
  localparam int PW = $clog2(QD), CW = $clog2(QD+1), OW = $clog2(MAX_OUT+1);
  initial begin
    if (QD < 2 || (1 << PW) != QD) $fatal(1,"QD must be a power of two >= 2");
    if (READ_BURST < 1 || READ_BURST > 16) $fatal(1,"READ_BURST must be 1..16");
    if (READ_IDLE < 1) $fatal(1,"READ_IDLE must be positive");
    if (MAX_OUT < 2) $fatal(1,"MAX_OUT must be >= 2");
    if (BASE_ADDR[30:0] != 0) $fatal(1,"BASE_ADDR must align to 2 GiB");
  end
  (* ram_style="distributed" *) logic [25:0] cq [0:QD-1];
  (* ram_style="distributed" *) logic [575:0] wq [0:QD-1];
  logic [PW-1:0] cp, cr, wp, wr;
  logic [CW-1:0] cn, wn;
  logic [OW-1:0] reads, writes;
  logic aw_sent, w_sent;
  localparam int BP = $clog2(MAX_OUT), IW = $clog2(READ_IDLE+1);
  logic [24:0] pending_addr, next_addr;
  logic [4:0] pending_count;
  logic [IW-1:0] idle_count;
  logic pending_valid;
  // Credit count includes collected but not yet issued native reads.
  // The length queue checks RLAST independently of native beat accounting.
  (* ram_style="distributed" *) logic [4:0] lengths [0:(1<<BP)-1];
  logic [BP-1:0] lp, lr;
  logic [OW-1:0] ln;
  logic [4:0] remaining;

  wire pushc = n_cvalid && n_cready;
  wire pushw = n_wvalid && n_wready;
  wire head_we = cq[cr][25];
  wire adjacent = cq[cr][24:0] == next_addr && cq[cr][24:6] == pending_addr[24:6];
  wire take_read = !pending_valid && cn != 0 && !head_we && writes == 0 &&
                   reads < MAX_OUT && (pending_count == 0 || adjacent);
  wire write_go = cn != 0 && head_we && wn != 0 && reads == 0 && writes < MAX_OUT;
  wire ar_fire = m_arvalid && m_arready;
  wire aw_fire = m_awvalid && m_awready;
  wire w_fire = m_wvalid && m_wready;
  wire popw = write_go && (aw_sent || aw_fire) && (w_sent || w_fire);
  wire popc = take_read || popw;
  assign n_cready = !rst && cn < QD;
  assign n_wready = !rst && wn < QD;
  assign m_araddr = BASE_ADDR | {3'b0,pending_addr,6'b0};
  assign m_arlen = {3'b0,pending_count} - 8'd1;
  assign m_awaddr = BASE_ADDR | {3'b0,cq[cr][24:0],6'b0};
  assign m_arvalid = !rst && pending_valid;
  // Once a channel has started, keep it asserted until its handshake completes.
  assign m_awvalid = !rst && write_go && !aw_sent;
  assign m_wvalid = !rst && write_go && !w_sent;
  assign m_wdata = wq[wr][511:0];
  assign m_wstrb = wq[wr][575:512];
  assign m_rready = !rst;
  assign m_bready = !rst;
  wire r_fire = m_rvalid && m_rready;
  wire b_fire = m_bvalid && m_bready;
  wire start_response = r_fire && remaining == 0;
  wire bypass_length = start_response && ln == 0 && ar_fire;
  wire push_length = ar_fire && !bypass_length;
  wire pop_length = start_response && ln != 0;
  wire [4:0] response_count = remaining != 0 ? remaining :
                              (ln != 0 ? lengths[lr] : pending_count);
  wire response_known = remaining != 0 || ln != 0 || ar_fire;
  assign n_rvalid = r_fire;
  assign n_rdata = m_rdata;
  always_ff @(posedge clk) begin
    if (rst) begin
      cp <= 0; cr <= 0; wp <= 0; wr <= 0; cn <= 0; wn <= 0;
      reads <= 0; writes <= 0; aw_sent <= 0; w_sent <= 0;
      n_wdone <= 0; n_err <= 0;
      pending_addr <= 0; next_addr <= 0; pending_count <= 0;
      pending_valid <= 0; idle_count <= 0;
      lp <= 0; lr <= 0; ln <= 0; remaining <= 0;
    end else begin
      if (ar_fire) begin
        pending_valid <= 0; pending_count <= 0; idle_count <= 0;
      end else if (take_read) begin
        if (pending_count == 0) pending_addr <= cq[cr][24:0];
        next_addr <= cq[cr][24:0] + 1'b1;
        pending_count <= pending_count + 1'b1;
        idle_count <= 0;
        if (pending_count == READ_BURST-1 || cq[cr][5:0] == 6'd63 || reads == MAX_OUT-1)
          pending_valid <= 1;
      end else if (pending_count != 0 && !pending_valid) begin
        // Never wait for another command at a direction/page/discontinuity
        // boundary. A short idle timer also flushes isolated final reads.
        if (cn != 0 || idle_count == READ_IDLE-1 || reads == MAX_OUT)
          pending_valid <= 1;
        else idle_count <= idle_count + 1'b1;
      end
      if (push_length) begin lengths[lp] <= pending_count; lp <= lp+1'b1; end
      if (pop_length) lr <= lr+1'b1;
      case ({push_length,pop_length})
        2'b10: ln <= ln+1'b1;
        2'b01: ln <= ln-1'b1;
        default: ;
      endcase
      if (r_fire) remaining <= response_known ? response_count-1'b1 : 5'd0;
      if (pushc) begin cq[cp] <= {n_cwe,n_caddr}; cp <= cp+1'b1; end
      if (pushw) begin wq[wp] <= {n_wmask,n_wdata}; wp <= wp+1'b1; end
      if (popc) cr <= cr+1'b1;
      if (popw) wr <= wr+1'b1;
      case ({pushc,popc})
        2'b10: cn <= cn+1'b1;
        2'b01: cn <= cn-1'b1;
        default: ;
      endcase
      case ({pushw,popw})
        2'b10: wn <= wn+1'b1;
        2'b01: wn <= wn-1'b1;
        default: ;
      endcase
      if (popc) begin aw_sent <= 0; w_sent <= 0; end
      else begin
        if (aw_fire) aw_sent <= 1;
        if (w_fire) w_sent <= 1;
      end
      case ({take_read,r_fire})
        2'b10: reads <= reads+1'b1;
        2'b01: reads <= reads-1'b1;
        default: ;
      endcase
      case ({popw,b_fire})
        2'b10: writes <= writes+1'b1;
        2'b01: writes <= writes-1'b1;
        default: ;
      endcase
      if (b_fire) n_wdone <= n_wdone+1'b1;
      if (r_fire && (m_rresp != 0 || !response_known || m_rlast != (response_count == 1))) n_err <= 1;
      if (b_fire && (m_bresp != 0 || (writes == 0 && !popw))) n_err <= 1;
    end
  end
endmodule
