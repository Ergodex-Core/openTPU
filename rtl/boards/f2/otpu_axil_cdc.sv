// AXI4-Lite from the shell's OCL (sclk: clk_main_a0) to the control registers (mclk: the core
// clock), one access at a time: the slave side takes a write (AW and W, in any order) or a read,
// holds it in registers and toggles a request; the master side, seeing the toggle through two
// flip-flops, runs the access on its port and toggles an acknowledgement back with the response,
// which the slave side then gives. Every crossing word is stable from before its toggle until the
// toggle back (the constraints: max delay, datapath only; boards/f2/build/constraints).
//
// Slave side: AW, W and AR enter two-entry skid buffers (otpu_skidbuf, rtl/shell: the ready a
// register, the access the state machine sees from the buffer's registers), B and R leave from
// registers (s_bvalid / s_rvalid and their data below), so no OCL output is a combinational
// function of an OCL input or of hold (round 4's placement, docs/wide_port.md "Round 5 RTL":
// the shell's PCIS wvalid through pcis_wbusy into s_awready and back to the shell, 8.0 ns of
// route over two SLRs on the 4 ns clock). The AW and W buffers' exits wait for !hold.
//
// hold: a write is not started while it is high (otpu_f2_mem's pcis_wbusy: the host's writes to
// the DIMM go first; the write waits in its skid buffer). Accesses outside the control registers' 4 KiB (BAR0 offset bit 12 and up: the
// card's other windows, the LiteDRAM / DDR4 CSRs of the other boards) never reach the port: reads
// return 0, writes are dropped, both OKAY (the host's drivers probe them).
module otpu_axil_cdc (
  input  logic         sclk,
  input  logic         srst,
  input  logic         hold,
  input  logic [31:0]  s_awaddr,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [31:0]  s_wdata,
  input  logic [3:0]   s_wstrb,
  input  logic         s_wvalid,
  output logic         s_wready,
  output logic [1:0]   s_bresp,
  output logic         s_bvalid,
  input  logic         s_bready,
  input  logic [31:0]  s_araddr,
  input  logic         s_arvalid,
  output logic         s_arready,
  output logic [31:0]  s_rdata,
  output logic [1:0]   s_rresp,
  output logic         s_rvalid,
  input  logic         s_rready,
  input  logic         mclk,
  input  logic         mrst,
  output logic [11:0]  m_awaddr,
  output logic         m_awvalid,
  input  logic         m_awready,
  output logic [31:0]  m_wdata,
  output logic [3:0]   m_wstrb,
  output logic         m_wvalid,
  input  logic         m_wready,
  input  logic [1:0]   m_bresp,
  input  logic         m_bvalid,
  output logic         m_bready,
  output logic [11:0]  m_araddr,
  output logic         m_arvalid,
  input  logic         m_arready,
  input  logic [31:0]  m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rvalid,
  output logic         m_rready
);
  // ---- slave side (sclk)
  typedef enum logic [1:0] {S_IDLE, S_WAIT, S_RESP} sst_t;
  sst_t        ss;
  logic        have_aw, have_w;
  logic [31:0] x_addr, x_wdata;         // crossing: held while the request is out
  logic [3:0]  x_wstrb;
  logic        x_wr;
  logic        x_req;                   // toggles per request
  (* ASYNC_REG = "TRUE" *) logic a1, a2;
  logic        a3;
  logic [31:0] y_rdata;                 // crossing back: held from the ack's toggle on
  logic [1:0]  y_resp;

  // the skid buffers' outputs (k_*: what the state machine sees) and their takes
  logic        k_awv, k_awr, k_wv, k_wr, k_arv, k_arr;
  logic [31:0] k_awaddr, k_wdata, k_araddr;
  logic [3:0]  k_wstrb;
  otpu_skidbuf #(.W(32)) u_skid_aw (.clk(sclk), .rst(srst), .i_v(s_awvalid), .i_r(s_awready),
                                 .i_d(s_awaddr), .o_v(k_awv), .o_r(k_awr), .o_d(k_awaddr), .any());
  otpu_skidbuf #(.W(36)) u_skid_w (.clk(sclk), .rst(srst), .i_v(s_wvalid), .i_r(s_wready),
                                .i_d({s_wstrb, s_wdata}), .o_v(k_wv), .o_r(k_wr),
                                .o_d({k_wstrb, k_wdata}), .any());
  otpu_skidbuf #(.W(32)) u_skid_ar (.clk(sclk), .rst(srst), .i_v(s_arvalid), .i_r(s_arready),
                                 .i_d(s_araddr), .o_v(k_arv), .o_r(k_arr), .o_d(k_araddr), .any());
  // a write starts only from an AW / W three cycles old in its buffer (k_aw2 / k_w2: the beat's
  // age; rounds 7 and 8, docs/wide_port.md): hold, the host's write-pending bit, is registered
  // twice on its way here and once more here, so a PCIS write shown with the OCL's is counted
  // before the OCL's starts
  // (round 8: hold is registered here too, hold_q, so this block has no combinational input
  // from the host path and can sit beside the shell's AXI-Lite master, SLR1; the beat's age
  // before a write starts is three cycles for it)
  logic hold_q;
  logic [1:0] aw_age, w_age;
  wire k_aw2 = aw_age == 2'd3;
  wire k_w2 = w_age == 2'd3;
  always_ff @(posedge sclk) begin
    hold_q <= hold;
    if (srst || !k_awv || (k_awr && k_awv)) aw_age <= 2'd0;
    else if (aw_age != 2'd3) aw_age <= aw_age + 2'd1;
    if (srst || !k_wv || (k_wr && k_wv)) w_age <= 2'd0;
    else if (w_age != 2'd3) w_age <= w_age + 2'd1;
  end
  assign k_awr = ss == S_IDLE && !have_aw && !hold_q && k_aw2 && !k_arvalid_pri();
  assign k_wr  = ss == S_IDLE && !have_w && !hold_q && k_w2 && !k_arvalid_pri();
  assign k_arr = ss == S_IDLE && !have_aw && !have_w;
  // a read waiting takes the turn when no write has started
  function automatic logic k_arvalid_pri();
    return k_arv && !have_aw && !have_w;
  endfunction

  always_ff @(posedge sclk) begin
    if (srst) begin
      ss <= S_IDLE; have_aw <= 1'b0; have_w <= 1'b0; x_req <= 1'b0;
      s_bvalid <= 1'b0; s_rvalid <= 1'b0; a1 <= 1'b0; a2 <= 1'b0; a3 <= 1'b0;
    end else begin
      a1 <= y_ack; a2 <= a1; a3 <= a2;
      case (ss)
        S_IDLE: begin
          if (k_arr && k_arv) begin
            x_addr <= k_araddr; x_wr <= 1'b0; x_req <= !x_req; ss <= S_WAIT;
          end else begin
            if (k_awr && k_awv) begin x_addr <= k_awaddr; have_aw <= 1'b1; end
            if (k_wr && k_wv) begin x_wdata <= k_wdata; x_wstrb <= k_wstrb; have_w <= 1'b1; end
            if ((have_aw || (k_awr && k_awv)) && (have_w || (k_wr && k_wv))) begin
              x_wr <= 1'b1; x_req <= !x_req; ss <= S_WAIT;
            end
          end
        end
        S_WAIT: if (a2 != a3) begin
          ss <= S_RESP;
          have_aw <= 1'b0; have_w <= 1'b0;
          if (x_wr) begin s_bvalid <= 1'b1; s_bresp <= y_resp; end
          else      begin s_rvalid <= 1'b1; s_rresp <= y_resp; s_rdata <= y_rdata; end
        end
        default: begin
          if (s_bvalid && s_bready) begin s_bvalid <= 1'b0; ss <= S_IDLE; end
          if (s_rvalid && s_rready) begin s_rvalid <= 1'b0; ss <= S_IDLE; end
        end
      endcase
    end
  end

  // ---- master side (mclk)
  (* ASYNC_REG = "TRUE" *) logic q1, q2;
  logic q3, y_ack, busy, in_win;
  assign in_win = x_addr[31:12] == '0;
  always_ff @(posedge mclk) begin
    if (mrst) begin
      q1 <= 1'b0; q2 <= 1'b0; q3 <= 1'b0; y_ack <= 1'b0; busy <= 1'b0;
      m_awvalid <= 1'b0; m_wvalid <= 1'b0; m_arvalid <= 1'b0;
    end else begin
      q1 <= x_req; q2 <= q1; q3 <= q2;
      if (!busy && q2 != q3) begin
        if (!in_win) begin               // outside the registers: answered here
          y_rdata <= '0; y_resp <= 2'b00; y_ack <= !y_ack;
        end else if (x_wr) begin
          m_awaddr <= x_addr[11:0]; m_wdata <= x_wdata; m_wstrb <= x_wstrb;
          m_awvalid <= 1'b1; m_wvalid <= 1'b1; busy <= 1'b1;
        end else begin
          m_araddr <= x_addr[11:0]; m_arvalid <= 1'b1; busy <= 1'b1;
        end
      end
      if (m_awvalid && m_awready) m_awvalid <= 1'b0;
      if (m_wvalid && m_wready) m_wvalid <= 1'b0;
      if (m_arvalid && m_arready) m_arvalid <= 1'b0;
      if (busy && m_bvalid) begin y_resp <= m_bresp; y_ack <= !y_ack; busy <= 1'b0; end
      if (busy && m_rvalid) begin y_resp <= m_rresp; y_rdata <= m_rdata; y_ack <= !y_ack; busy <= 1'b0; end
    end
  end
  assign m_bready = busy;
  assign m_rready = busy;
endmodule
