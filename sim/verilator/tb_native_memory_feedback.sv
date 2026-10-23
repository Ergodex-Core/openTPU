// Bounded deterministic native-memory boundary exercise. The Python checker
// consumes the DUT's pre-edge request/response trace; this bench checks only
// that all independent command and write-data streams complete.
module tb_native_memory_feedback;
  localparam int NCH = 8;
  localparam int NCOMMANDS = 40;
  localparam int NWRITES = 8;
  localparam int NREADS = NCOMMANDS - NWRITES;

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1;
  logic dump;
  logic [NCH-1:0] n_cvalid, n_cready, n_cwe;
  logic [NCH-1:0][24:0] n_caddr;
  logic [NCH-1:0] n_wvalid, n_wready, n_rvalid;
  logic [NCH-1:0][511:0] n_wdata, n_rdata;
  logic [NCH-1:0][63:0] n_wmask;
  logic [NCH-1:0][15:0] n_wdone;
  int unsigned cycle = 0;
  int unsigned commands [NCH], data_beats [NCH], reads [NCH];

  otpu_native_mem #(.WORDS(16384), .NCH(NCH), .CHASH(1'b1)) u_mem (
    .clk, .rst, .run_rst(1'b0), .n_cvalid, .n_cready, .n_cwe, .n_caddr,
    .n_wvalid, .n_wready, .n_wdata, .n_wmask, .n_rvalid, .n_rdata,
    .n_wdone, .dump
  );

  always_comb begin
    dump = !rst;
    for (int c = 0; c < NCH; c++) begin
      // A command is held unchanged until its own channel accepts it.
      n_cvalid[c] = !rst && commands[c] < NCOMMANDS;
      n_cwe[c] = (commands[c] % 5) == 0;
      n_caddr[c] = 25'(2 * commands[c] + c); // 0..85, inside 1024 logical beats
      // Write beats are an independent stream, including bubbles before and
      // after corresponding write commands. All writes are whole beats.
      n_wvalid[c] = !rst && data_beats[c] < NWRITES && (cycle + c) % 4 != 0;
      n_wdata[c] = {16{32'(c * 256 + data_beats[c])}};
      n_wmask[c] = '1;
      dump &= commands[c] == NCOMMANDS && data_beats[c] == NWRITES &&
              reads[c] == NREADS && n_wdone[c] == 16'(NWRITES);
    end
  end

  initial begin
    for (int c = 0; c < NCH; c++) begin
      commands[c] = 0;
      data_beats[c] = 0;
      reads[c] = 0;
    end
    repeat (4) @(negedge clk);
    rst = 1'b0;
  end

  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (!rst) begin
      for (int c = 0; c < NCH; c++) begin
        if (n_cvalid[c] && n_cready[c]) commands[c] <= commands[c] + 1;
        if (n_wvalid[c] && n_wready[c]) data_beats[c] <= data_beats[c] + 1;
        if (n_rvalid[c]) reads[c] <= reads[c] + 1;
        if (reads[c] > NREADS || n_wdone[c] > 16'(NWRITES))
          $fatal(1, "native feedback channel %0d over-completed", c);
      end
    end
    if (cycle >= 10000) $fatal(1, "native feedback watchdog");
    if (dump) begin
      $display("TEST_PASS native memory feedback NCH8 commands=%0d writes=%0d reads=%0d cycles=%0d",
               NCH * NCOMMANDS, NCH * NWRITES, NCH * NREADS, cycle);
      // Let the DUT's dump/trace always blocks emit the final line and close.
      #1 $finish;
    end
  end
endmodule
