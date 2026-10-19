// A behavioural stand-in for the HDK's cl_axi_sc_1x1 SmartConnect block design (tb_f2hbm only:
// the vendor IP has no Verilator model), as the F2 HBM CL uses it per port (cl_opentpu_hbm.sv,
// docs/v4/f2-round5.md): an AXI4 slave of 512 bits with 16-bit IDs at the core clock (its
// aclk_250 pin) onto an AXI3 master of 256 bits without IDs at the HBM clock (aclk_450). What it
// reproduces of the IP: the clock crossing (queues written at one clock and read at the other),
// the width conversion (a 512-bit beat at a 64-byte aligned address is two 256-bit beats, the
// low half first; strobes split the same way), the AXI3 conversion (a burst of more than 16
// 32-byte beats is split into bursts of at most 16, each with its own WLAST; the R beats of the
// pieces come back as one AXI4 burst with RLAST on the last, the B's of the pieces as one B),
// single ordering per direction (an AXI4 slave with an ID-less master returns B's in AW order
// and R bursts in AR order, each with the ID of its request), random readies on the slave side
// and independent read and write channels (nothing orders a read against a write: the masters
// in front of it do). Every output is a register loaded from a queue's head and popped once
// taken (a continuous assignment from a queue is a race against the blocks that pop it: the
// simulator may run them first). Not a model of the IP's latency or buffering.
module tb_f2hbm_sc (
  input  logic          clk,
  input  logic          rst,
  input  logic          hclk,
  // ---- AXI4 slave (clk)
  input  logic [15:0]   s_awid,
  input  logic [33:0]   s_awaddr,
  input  logic [7:0]    s_awlen,
  input  logic          s_awvalid,
  output logic          s_awready,
  input  logic [511:0]  s_wdata,
  input  logic [63:0]   s_wstrb,
  input  logic          s_wlast,
  input  logic          s_wvalid,
  output logic          s_wready,
  output logic [15:0]   s_bid,
  output logic [1:0]    s_bresp,
  output logic          s_bvalid,
  input  logic          s_bready,
  input  logic [15:0]   s_arid,
  input  logic [33:0]   s_araddr,
  input  logic [7:0]    s_arlen,
  input  logic          s_arvalid,
  output logic          s_arready,
  output logic [15:0]   s_rid,
  output logic [511:0]  s_rdata,
  output logic [1:0]    s_rresp,
  output logic          s_rlast,
  output logic          s_rvalid,
  input  logic          s_rready,
  // ---- AXI3 master (hclk), no IDs
  output logic [33:0]   m_awaddr,
  output logic [3:0]    m_awlen,
  output logic          m_awvalid,
  input  logic          m_awready,
  output logic [255:0]  m_wdata,
  output logic [31:0]   m_wstrb,
  output logic          m_wlast,
  output logic          m_wvalid,
  input  logic          m_wready,
  input  logic [1:0]    m_bresp,
  input  logic          m_bvalid,
  output logic          m_bready,
  output logic [33:0]   m_araddr,
  output logic [3:0]    m_arlen,
  output logic          m_arvalid,
  input  logic          m_arready,
  input  logic [255:0]  m_rdata,
  input  logic [1:0]    m_rresp,
  input  logic          m_rlast,
  input  logic          m_rvalid,
  output logic          m_rready
);
  typedef struct { logic [15:0] id; logic [33:0] a; logic [7:0] len; } ax_t;
  typedef struct { logic [33:0] a; logic [3:0] len; } ax3_t;
  typedef struct { logic [255:0] d; logic [31:0] s; logic last; } w3_t;
  typedef struct { logic [15:0] id; int pieces; } wr_t;
  typedef struct { logic [15:0] id; int halves; } rr_t;
  typedef struct { logic [15:0] id; logic [511:0] d; logic last; } r4_t;
  ax_t   awq[$], arq[$];                 // AXI4 requests taken, not yet converted
  logic [576:0] wq[$];                   // AXI4 W beats taken ({last, strb, data})
  ax3_t  m_awq[$], m_arq[$];             // AXI3 requests to send
  w3_t   m_wq[$];                        // AXI3 W beats to send
  wr_t   wrq[$];                         // writes converted: their B's to collect, in order
  rr_t   rrq[$];                         // reads converted: their R halves to collect, in order
  int    b_got;                          // B's of the head write collected
  logic [255:0] r_halves[$];             // R halves received, in order
  r4_t   s_rq[$];                        // AXI4 R beats to give
  logic [15:0] s_bq[$];                  // AXI4 B's to give

  function automatic int beats_of(input logic [7:0] len);   // AXI4 beats of a burst
    return int'(len) + 1;
  endfunction

  // ---- the slave side (clk): take requests and beats, convert, give responses
  always_ff @(posedge clk) begin
    s_awready <= $urandom % 4 != 0;
    s_wready <= $urandom % 4 != 0;
    s_arready <= $urandom % 4 != 0;
    if (!rst) begin
      if (s_awvalid && s_awready) begin
        if (s_awaddr[5:0] != 0) $fatal(1, "tb_f2hbm_sc: AW not 64-byte aligned");
        awq.push_back('{s_awid, s_awaddr, s_awlen});
      end
      if (s_wvalid && s_wready) wq.push_back({s_wlast, s_wstrb, s_wdata});
      if (s_arvalid && s_arready) begin
        if (s_araddr[5:0] != 0) $fatal(1, "tb_f2hbm_sc: AR not 64-byte aligned");
        arq.push_back('{s_arid, s_araddr, s_arlen});
      end
      // a write with all its beats: into AXI3 pieces of at most 16 half beats
      if (awq.size() && wq.size() >= beats_of(awq[0].len)) begin
        automatic ax_t x = awq.pop_front();
        automatic int halves = 2 * beats_of(x.len);
        automatic int pieces = 0;
        for (int b = 0; b < beats_of(x.len); b++) begin
          automatic logic [576:0] w = wq.pop_front();
          if (w[576] != (b == beats_of(x.len) - 1)) $fatal(1, "tb_f2hbm_sc: WLAST at the wrong beat");
          for (int h = 0; h < 2; h++) begin
            automatic int i = 2 * b + h;
            m_wq.push_back('{w[256 * h +: 256], w[512 + 32 * h +: 32], (i % 16 == 15) || (i == halves - 1)});
          end
        end
        for (int i = 0; i < halves; i += 16) begin
          automatic int n = (halves - i < 16) ? halves - i : 16;
          m_awq.push_back('{x.a + 34'(32 * i), 4'(n - 1)});
          pieces++;
        end
        wrq.push_back('{x.id, pieces});
      end
      if (arq.size()) begin
        automatic ax_t x = arq.pop_front();
        automatic int halves = 2 * beats_of(x.len);
        for (int i = 0; i < halves; i += 16) begin
          automatic int n = (halves - i < 16) ? halves - i : 16;
          m_arq.push_back('{x.a + 34'(32 * i), 4'(n - 1)});
        end
        rrq.push_back('{x.id, halves});
      end
      // the head read's halves, paired into AXI4 beats, RLAST on its last
      if (rrq.size() && r_halves.size() >= 2) begin
        automatic logic [255:0] lo = r_halves.pop_front();
        automatic logic [255:0] hi = r_halves.pop_front();
        s_rq.push_back('{rrq[0].id, {hi, lo}, rrq[0].halves == 2});
        rrq[0].halves -= 2;
        if (rrq[0].halves == 0) void'(rrq.pop_front());
      end
      // the responses: the beat shown is popped once taken, the next loaded
      if (s_bvalid && s_bready) void'(s_bq.pop_front());
      if (!s_bvalid || s_bready) begin
        s_bvalid <= s_bq.size() > 0;
        if (s_bq.size()) s_bid <= s_bq[0];
      end
      if (s_rvalid && s_rready) void'(s_rq.pop_front());
      if (!s_rvalid || s_rready) begin
        s_rvalid <= s_rq.size() > 0;
        if (s_rq.size()) begin s_rid <= s_rq[0].id; s_rdata <= s_rq[0].d; s_rlast <= s_rq[0].last; end
      end
    end else begin
      s_bvalid <= 1'b0; s_rvalid <= 1'b0;
    end
  end
  assign s_bresp = 2'b00;
  assign s_rresp = 2'b00;

  // ---- the master side (hclk): send the pieces (registered), collect the responses
  assign m_bready = !rst;
  assign m_rready = !rst;
  always_ff @(posedge hclk) begin
    if (!rst) begin
      if (m_awvalid && m_awready) void'(m_awq.pop_front());
      if (!m_awvalid || m_awready) begin
        m_awvalid <= m_awq.size() > 0;
        if (m_awq.size()) begin m_awaddr <= m_awq[0].a; m_awlen <= m_awq[0].len; end
      end
      if (m_wvalid && m_wready) void'(m_wq.pop_front());
      if (!m_wvalid || m_wready) begin
        m_wvalid <= m_wq.size() > 0;
        if (m_wq.size()) begin m_wdata <= m_wq[0].d; m_wstrb <= m_wq[0].s; m_wlast <= m_wq[0].last; end
      end
      if (m_arvalid && m_arready) void'(m_arq.pop_front());
      if (!m_arvalid || m_arready) begin
        m_arvalid <= m_arq.size() > 0;
        if (m_arq.size()) begin m_araddr <= m_arq[0].a; m_arlen <= m_arq[0].len; end
      end
      if (m_bvalid && m_bready) begin
        if (m_bresp != 0) $fatal(1, "tb_f2hbm_sc: B error");
        if (wrq.size() == 0) $fatal(1, "tb_f2hbm_sc: B with no write");
        b_got++;
        if (b_got == wrq[0].pieces) begin
          s_bq.push_back(wrq[0].id);
          void'(wrq.pop_front());
          b_got = 0;
        end
      end
      if (m_rvalid && m_rready) begin
        if (m_rresp != 0) $fatal(1, "tb_f2hbm_sc: R error");
        r_halves.push_back(m_rdata);
      end
    end else begin
      m_awvalid <= 1'b0; m_wvalid <= 1'b0; m_arvalid <= 1'b0;
    end
  end
  initial b_got = 0;
endmodule
