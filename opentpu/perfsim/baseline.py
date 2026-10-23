"""Explicit compute-RTL contract for the program-resident native-memory baseline.

Values mirror sim/verilator/tb_top.sv; no ambient rtlsim.UARCH/board environment is used.
The external native controller replaces AXI=0 memory only, not any compute RTL unit.
"""
from .config import Architecture, Unsupported


def rtl_defaults(cfg):
    return dict(S=1,D=32,MCOLS=8,ACT_BLOCKS=64,ACT_ROWS=cfg.MCOLS,
                TMEM_WORDS=65536,IMEM_WORDS=65536,DRAM_WORDS=262144,DRAM_LAT=8,
                LANES=8,WIN=32,RPB=4,WPB=2,MXU_IMPL=2,MXU_CL=16,MXU_E=0,
                QUANT_XD=1024,VPU_CL=cfg.LANES//4 if cfg.LANES>=8 else 1,
                ULANES=cfg.LANES,AXI=0,FIFO_DEPTH=128,TMEM_IMPL=0,SEQ_WC=0,
                TMEM_RQ=0,TMEM_NCP=3,TMEM_NSB=8,TMEM_SBP=3,TMEM_SBH=0,
                TMEM_RPL=0,TMEM_TDP=0,TMEM_GST=0,TMEM_PMP=0,TMEM_PRK=0,
                TMEM_SDM=-1,COLL_GATHER=1)


def rtl_parameters(cfg, architecture=None):
    arch=architecture or Architecture();arch.validate(cfg)
    fixed={'mxu_queue':2,'mxu_result_rows':32,'dma_buffer_chunks':128,'dma_store_run':32,
           'vpu_slots':10,'result_queue':128,'issue_period':1,'dispatch_latency':1,'tmem_read_latency':2}
    for field,value in fixed.items():
        if getattr(arch,field)!=value:raise Unsupported(f'RTL comparison requires {field}={value}')
    if arch.vpu_composite_lanes not in (0,cfg.LANES):raise Unsupported('RTL comparison requires full-width composite lanes')
    p=rtl_defaults(cfg)
    p.update(S=cfg.S,D=cfg.D,MCOLS=cfg.MCOLS,LANES=cfg.LANES,ACT_BLOCKS=cfg.ACT_BLOCKS,
             ACT_ROWS=cfg.act_rows,TMEM_WORDS=cfg.TMEM_WORDS,IMEM_WORDS=cfg.IMEM_WORDS,
             DRAM_WORDS=cfg.DRAM_BYTES//4,WIN=arch.window,RPB=arch.tmem_read_ports_per_bank,
             WPB=arch.tmem_write_ports_per_bank,MXU_CL=arch.mxu_chain,MXU_E=arch.mxu_epilogue_lanes,
             FIFO_DEPTH=arch.mxu_fifo_chunks,QUANT_XD=arch.quant_row_buffer_chunks,SEQ_WC=int(cfg.SEQ_WC))
    return p


def verify_rtl_parameters(receipt,cfg,architecture=None):
    expected=rtl_parameters(cfg,architecture);actual=rtl_defaults(cfg)
    unknown=set(receipt['parameters'])-set(actual)
    if unknown:raise ValueError(f'unrecognized RTL parameters: {sorted(unknown)}')
    actual.update(receipt['parameters'])
    mismatches={key:{'expected':value,'actual':actual[key]} for key,value in expected.items() if actual[key]!=value}
    if mismatches:raise ValueError(f'RTL build/config mismatch: {mismatches}')
    return expected
