#!/usr/bin/env python3
"""Build a derived test top around unchanged RTL and the shared Python/Ramulator controller."""
import argparse,hashlib,json,subprocess,sys,sysconfig,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from opentpu.rtlsim import RTL_SOURCES,top_params
from opentpu.perfsim.io import load_bundle,digest

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--persistent',action='store_true');a=ap.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 cfg,_,_,_=load_bundle(a.bundle)
 if cfg.S!=1:raise ValueError('isolated harness currently supports S=1')
 top=(ROOT/'rtl/top/otpu_top.sv').read_text();start=top.index('    if (AXI == 0) begin : g_dram');end=top.index('\n    otpu_slice #(',start)
 block='''    otpu_perfsim_memory #(.WORDS(DRAM_WORDS),.D(D),.SID(s)) u_memory (
      .clk,.rst,.a_req,.a_we,.a_addr,.a_wdata,.a_be,.a_rdy,.a_rvalid,.a_rdata,
      .sw_req,.sw_addr,.sw_wdata,.sw_be,.sw_rdy,
      .b_req,.b_tag,.b_we,.b_wmask,.b_wdata,.b_addr,.b_rdy,.b_rvalid,.b_rtag,.b_rdata,
      .rd_idle,.wr_idle,.dump);
'''
 block+='''
    always @(posedge clk) if (!rst && $test$plusargs("uarch_trace") && u_slice.u_mxu.c_act)
      $display("UX c=%0d out=%0d pop=%0d ck=%0d f=%0d sc=%0d rf=%0d dr=%0d g=%0d w=%0d ad=%0d",
       u_slice.u_seq.cyc,u_slice.u_mxu.h.out,u_slice.u_mxu.pop,u_slice.u_mxu.ck,
       u_slice.u_mxu.f_count,u_slice.u_mxu.s_count,u_slice.u_mxu.rf_n,
       u_slice.u_mxu.drain_go,u_slice.u_mxu.t_gnt,u_slice.u_mxu.t_wen,u_slice.u_mxu.t_waddr[0]);
    always @(posedge clk) if (!rst && $test$plusargs("uarch_trace") && (u_slice.u_vpu.busy || u_slice.u_vpu.done || (|u_slice.v_wen)))
      $display("VX c=%0d dst=%0d issue=%0d en=%0d cw=%0d ca=%0d w=%0d wa=%0d di=%0d done=%0d ew=%0d begin=%0d",
       u_slice.u_seq.cyc,u_slice.u_vpu.dst,u_slice.u_vpu.issuing,u_slice.u_vpu.en,u_slice.u_vpu.cw_en,u_slice.u_vpu.cw_addr[0],u_slice.v_wen,u_slice.v_waddr[0],u_slice.u_vpu.done_i,u_slice.u_vpu.done,u_slice.u_vpu.ew_n,u_slice.u_vpu.can_begin);
'''
 block+='''
    always @(posedge clk) if (!rst && $test$plusargs("uarch_trace") && u_slice.u_dma.is_ds)
      $display("SX c=%0d su=%0d wait=%0d fill=%0d run=%0d out=%0d req=%0d gnt=%0d pe=%0d take=%0d y=%0d o=%0d og=%0d left=%0d wr=%0d rr=%0d cnt=%0d occ=%0d done=%0d",
       u_slice.u_seq.cyc,u_slice.u_dma.ds_su,u_slice.u_dma.ds_wait,u_slice.u_dma.ds_fill,u_slice.u_dma.ds_run,u_slice.u_dma.ds_out,
       u_slice.u_dma.ss_req,u_slice.u_dma.ss_gnt,u_slice.u_dma.ss_pe,u_slice.u_dma.ds_take,u_slice.u_dma.y_v,u_slice.u_dma.o_v,
       u_slice.u_dma.og,u_slice.u_dma.ds_left,u_slice.u_dma.ds_wr,u_slice.u_dma.ds_rr,u_slice.u_dma.cnt,u_slice.u_dma.occ,u_slice.u_dma.done);
'''
 derived=top[:start]+block+top[end:];derived=derived.replace('module otpu_top','module otpu_perfsim_top',1)
 (out/'otpu_perfsim_top.sv').write_text(derived)
 tb=(ROOT/'sim/verilator/tb_top.sv').read_text().replace('module tb_top;','module tb_perfsim;').replace('otpu_top #(','otpu_perfsim_top #(')
 if a.persistent:
  start=tb.index("  initial begin\n    void'($value$plusargs(\"max_cycles=")
  body=(ROOT/'tools/perfsim/rtl/persistent_body.sv').read_text()
  if cfg.SEQ_WC:
   body=body.replace('  int stages=1', '  logic [63:0] wait_words [IMEM_WORDS/8];\n  int stages=1')
   body=body.replace('      pm_stage(stage);', '''      pm_stage(stage);
      for (int i=0;i<IMEM_WORDS/8;i++) wait_words[i]='0;
      $readmemh($sformatf("%s/waits.hex",stage_dir),wait_words);
      for (int row=0;row<IMEM_WORDS/32;row++)
        for (int e=0;e<4;e++)
          dut.g_slice[0].u_slice.u_seq.g_sync.u_sync.g_auto.mem[row][59*e+:59]={wait_words[4*row+e][62:32],wait_words[4*row+e][27:0]};''')
  tb=tb[:start]+body+'\nendmodule\n'
 (out/'tb_perfsim.sv').write_text(tb)
 sources=[ROOT/'rtl'/p for p in RTL_SOURCES if p!='top/otpu_top.sv']+[ROOT/'tools/perfsim/rtl/otpu_perfsim_memory.sv',out/'otpu_perfsim_top.sv',out/'tb_perfsim.sv']
 cpp=ROOT/'tools/perfsim/rtl/driver.cpp'
 includes=f'-I{sysconfig.get_config_var("INCLUDEPY")}'
 if sys.platform=='darwin':
  includes+=f' -arch {platform.machine()}'
  ld=f'-arch {platform.machine()} -F{sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")} -framework Python'
 else:ld=f'-L{sysconfig.get_config_var("LIBDIR")} -lpython{sysconfig.get_config_var("LDVERSION")} -ldl -lm'
 params=top_params(cfg,uarch={'WIN':16,'RPB':32,'WPB':1,'MXU_IMPL':2,'FIFO_DEPTH':1024,'SEQ_WC':int(cfg.SEQ_WC)})
 cmd=['verilator','--binary','-j',str(a.jobs),'-Wno-fatal','-Wno-WIDTHEXPAND','-Wno-WIDTHTRUNC','-O3','--x-assign','0','--x-initial','0','--top-module','tb_perfsim','-Mdir',str(out/'obj'),'-CFLAGS',includes,'-LDFLAGS',ld,*[f'-G{k}={int(v)}' for k,v in params.items()],*[str(p) for p in sources],str(cpp)]
 (out/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'build.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'Verilator build failed; see {out}/build.log')
 receipt={'persistent':a.persistent,'command':cmd,'parameters':params,'executable':str(out/'obj/Vtb_perfsim'),'binary_sha256':digest(out/'obj/Vtb_perfsim'),'sources':{str(p):digest(p) for p in sources+[cpp]},'original_top_sha256':digest(ROOT/'rtl/top/otpu_top.sv'),'original_testbench_sha256':digest(ROOT/'sim/verilator/tb_top.sv'),'derivation':'Only top module names and external memory boundary replaced; original RTL units and tb counter/dump sequence retained. No original source writes.'}
 (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['executable'])
if __name__=='__main__':main()
