  // Test-only host launcher. Each stage uses the original first-launch core counter.
  // Three dump/settle edges + five core-reset edges separate launches, plus host_gap.
  // Program/host input writes are outside the measured core interval. sys_rst never repeats.
  import "DPI-C" function void pm_stage(input int index);
  int stages=1,host_gap=0;
  string stage_root,stage_dir,nm;
  logic [31:0] program_words [IMEM_WORDS];
  initial begin
    void'($value$plusargs("max_cycles=%d",max_cycles));
    void'($value$plusargs("stages=%d",stages));
    void'($value$plusargs("host_gap=%d",host_gap));
    if (!$value$plusargs("dir=%s",stage_root)) $fatal(1,"missing stage root");
    repeat (3) @(negedge clk);
    sys_rst=0;
    @(negedge clk);
    for (int stage=0;stage<stages;stage++) begin
      stage_dir=$sformatf("%s/stage%0d",stage_root,stage);
      pm_stage(stage);
      dut.g_slice[0].u_slice.u_tmem.dir=stage_dir;
      for (int i=0;i<IMEM_WORDS;i++) program_words[i]='0;
      $readmemh($sformatf("%s/prog_0.hex",stage_dir),program_words);
      for (int row=0;row<IMEM_WORDS/(D/4);row++)
        for (int word_=0;word_<D/4;word_++)
          dut.g_slice[0].u_slice.u_seq.u_imem.g_auto.mem[row][32*word_+:32]=program_words[row*(D/4)+word_];
      for (int k=0;k<8;k++) begin
        rinit[k]=0;nm=$sformatf("stage%0d_arg%0d=%%d",stage,k);
        void'($value$plusargs(nm,rinit[k]));
      end
      if (stage>0) repeat (5) @(negedge clk);
      cycles=0;
      $display("STAGE_START index=%0d global=%0d",stage,dut.g_slice[0].u_memory.cycle+1);
      rst=0;
      while (!all_halted && cycles<max_cycles) begin
        @(negedge clk);cycles++;
      end
      $display("STAGE_RESULT index=%0d cycles=%0d halted=%0d error=%0d",stage,cycles,all_halted,any_error);
      if (!all_halted || any_error) $fatal(1,"stage failed");
      dump=1;
      @(negedge clk);dump=0;
      repeat (2) @(negedge clk);
      if (stage+1<stages) begin
        repeat (host_gap) @(negedge clk);
        rst=1;
      end
    end
    $finish;
  end
