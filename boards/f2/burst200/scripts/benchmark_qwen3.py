#!/usr/bin/env python3
"""README-equivalent Qwen3 benchmark with F2 transport and explicit TTFT accounting.
Run only after hardware smoke tests and token/logit validation pass.
"""
from pathlib import Path
import argparse,os,sys,json,time,hashlib,dataclasses
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OTPU_RUN_DIR',str(ROOT/'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET',str(ROOT/'targets/f2-burst200.toml'))
os.environ.setdefault('HF_HOME',str(ROOT/'build/f2/hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'build/f2/runtime/cache'))
(ROOT/'build/f2/runtime/tmp').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TMPDIR',str(ROOT/'build/f2/runtime/tmp'))
import numpy as np
from transformers import AutoTokenizer
from opentpu.host.board import Board,BoardBackend,device_config
from opentpu.llm import load_spec
from opentpu.llm import qwen3 as Q
from opentpu import imgcache as IC,progcache as PC
from opentpu.host.f2_transport import F2Transport

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--wformat',choices=['int8','fp4'],default='int8')
    ap.add_argument('--prompt',type=int,default=512)
    ap.add_argument('--tokens',type=int,default=64)
    ap.add_argument('--stream-logits',action='store_true',
                    help='Separate streamed-logit protocol; the README table comparison leaves this off')
    ap.add_argument('--prompt-runs',action='store_true',help='Use upstream cached prompt programs; report separately')
    ap.add_argument('--tiled',action='store_true',help='Use upstream tiled prompt programs (implies --prompt-runs)')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.prompt<1 or a.tokens<1:ap.error('Token counts must be positive')
    total_start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    spec=load_spec(a.model)
    tr=F2Transport();tr.streams=a.stream_logits;probe=None;eng=None
    try:
        probe=Board(transport=tr)
        info=probe.info();cfg=device_config(info)
        khz=info['core_khz']
        text=(ROOT/'tools/data/austen_pp_ch1.txt').read_text()
        ids=tok(text)['input_ids'][:a.prompt]
        if len(ids)!=a.prompt:raise RuntimeError('Reference text is too short')
        t0=time.perf_counter()
        weights=Q.load_weights(a.model)
        weight_read_s=time.perf_counter()-t0
        backend=lambda c,images:BoardBackend(c,images,transport=tr,model=a.model.name)
        transfer0=dict(tr.transfer_bytes);seconds0=dict(tr.transfer_seconds)
        t0=time.perf_counter()
        eng=Q.Engine(spec,weights,cap=max(2048,a.prompt+a.tokens+1),cfg=cfg,backend=backend,
                     wformat=a.wformat,head_format='int8',resident=True,
                     prompt_runs=a.prompt_runs or a.tiled,tile_runs=a.tiled)
        setup_s=time.perf_counter()-t0
        setup_transfers={d:dict(bytes=tr.transfer_bytes[d]-transfer0[d],
                               seconds=tr.transfer_seconds[d]-seconds0[d]) for d in ('h2c','c2h')}
        if bool(eng.backend.streams)!=a.stream_logits:
            raise RuntimeError('Backend logit-streaming mode differs from benchmark protocol')
        board=eng.backend.board
        n0=len(eng.stats);t0=time.perf_counter()
        logits=eng.prefill(ids);first=int(np.argmax(logits))
        ttft=time.perf_counter()-t0
        load_to_first=time.perf_counter()-total_start
        pcycles=sum(s['cycles'] for s in eng.stats[n0:])
        s0=board.snapshot();n0=len(eng.stats);token=first;generated=[first]
        t0=time.perf_counter()
        for _ in range(a.tokens):
            token=int(np.argmax(eng.step(token)));generated.append(token)
        eng._drain();wall=time.perf_counter()-t0
        s1=board.snapshot();cycles=sum(s['cycles'] for s in eng.stats[n0:])
        delta={k:int(s1[k]-s0[k]) for k in ('DRAM_RD','DRAM_WR','RUNNING','MXU_STARVE','DRAM_WAIT') if k in s0}
        run_s=delta['RUNNING']/(khz*1000)
        result=dict(model=str(a.model.resolve()),weight_format=a.wformat,head_format='int8',
            protocol='Greedy host argmax; '+('streamed logits' if a.stream_logits else 'non-streamed logits'),
            streamed_logits=a.stream_logits,
            prompt_programs='tiled' if a.tiled else ('cached' if a.prompt_runs else 'position_specific'),
            prompt_rows=eng.rows,image_cache_outcome=IC.last[0],
            program_cache_stats=dict(PC.stats),
            transport_read_mode=tr.read_mode,
            transport_read_threads=tr.read_threads,
            prompt_tokens=len(ids),decode_steps=a.tokens,core_khz=khz,
            weight_file_read_seconds=weight_read_s,engine_setup_seconds=setup_s,
            engine_setup_transfers=setup_transfers,
            model_upload_seconds=setup_transfers['h2c']['seconds'],
            model_upload_bytes=setup_transfers['h2c']['bytes'],
            model_load_to_first_token_seconds=load_to_first,
            warm_ttft_seconds=ttft,prefill_device_tps=len(ids)*khz*1000/pcycles,
            prefill_wall_tps=len(ids)/ttft,decode_device_tps=a.tokens*khz*1000/cycles,
            decode_wall_tps=a.tokens/wall,decode_wall_seconds=wall,decode_device_cycles=cycles,
            counters=delta,dram_gbytes_per_running_second=(delta['DRAM_RD']+delta['DRAM_WR'])*64/run_s/1e9,
            port_peak_gbytes_per_second=min(2*64*khz*1000,2*32*450e6)/1e9,
            generated_token_ids=generated,device_info=info,
            target_config_path=str(Path(os.environ['OTPU_TARGET']).resolve()),
            target_config_sha256=hashlib.sha256(Path(os.environ['OTPU_TARGET']).read_bytes()).hexdigest())
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
        print(json.dumps({k:v for k,v in result.items() if k.endswith('_tps') or k.endswith('_seconds')},indent=2))
    finally:
        if eng is not None:eng.backend.close()
        if probe is not None:probe.close()
        tr.close()
if __name__=='__main__':main()
