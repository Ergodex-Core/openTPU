// AXI4-Lite from the shell's OCL (sclk: clk_main_a0) to the control registers (mclk: the core
// clock), one access at a time: the slave side takes a write (AW and W, in any order) or a read,
// holds it in registers and toggles a request; the master side, seeing the toggle through two
// flip-flops, runs the access on its port and toggles an acknowledgement back with the response,
// which the slave side then gives. Every crossing word is stable from before its toggle until the
// toggle back (the constraints: max delay, datapath only; boards/f2/build/constraints).
//
// hold: a write is not taken while it is high (otpu_f2_mem's pcis_wbusy: the host's writes to the
// DIMM go first). Accesses outside the control registers' 4 KiB (BAR0 offset bit 12 and up: the
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

  assign s_awready = ss == S_IDLE && !have_aw && !hold && !s_arvalid_pri();
  assign s_wready  = ss == S_IDLE && !have_w && !hold && !s_arvalid_pri();
  assign s_arready = ss == S_IDLE && !have_aw && !have_w;
  // a read waiting takes the turn when no write has started
  function automatic logic s_arvalid_pri();
    return s_arvalid && !have_aw && !have_w;
  endfunction

  always_ff @(posedge sclk) begin
    if (srst) begin
      ss <= S_IDLE; have_aw <= 1'b0; have_w <= 1'b0; x_req <= 1'b0;
      s_bvalid <= 1'b0; s_rvalid <= 1'b0; a1 <= 1'b0; a2 <= 1'b0; a3 <= 1'b0;
    end else begin
      a1 <= y_ack; a2 <= a1; a3 <= a2;
      case (ss)
        S_IDLE: begin
          if (s_arready && s_arvalid) begin
            x_addr <= s_araddr; x_wr <= 1'b0; x_req <= !x_req; ss <= S_WAIT;
          end else begin
            if (s_awready && s_awvalid) begin x_addr <= s_awaddr; have_aw <= 1'b1; end
            if (s_wready && s_wvalid) begin x_wdata <= s_wdata; x_wstrb <= s_wstrb; have_w <= 1'b1; end
            if ((have_aw || (s_awready && s_awvalid)) && (have_w || (s_wready && s_wvalid))) begin
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
