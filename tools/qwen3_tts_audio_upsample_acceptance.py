#!/usr/bin/env python3
"""Pinned checkpoint 2x2 upsampling/ConvNeXt feature acceptance; no waveform."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu import isa as I
from opentpu.isasim import Machine
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec,ChunkedAudioFrameSession
from opentpu.llm.qwen3_tts_audio_upsample import UpsampleChunkImage,UpsampleChunkISASession,UpsampleChunkFrameImage
from opentpu.llm.qwen3_tts_talker import TalkerSpec,_tensor
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.kernels.tts_audio_upsample import gelu_probe
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,digest
from tools.qwen3_tts_audio_front_acceptance import CachedReader,metrics
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from qwen3_tts_audio_upsample_reference import AudioUpsampleReference


def read_tensor(session,t):
    return session.machine.slices[0].dram[t.base.const:t.base.const+4*math.prod(t.shape)].view(np.float32).copy().reshape(t.shape)


def references(image,reader,codes,dtype):
    result=[{key:[] for key in ['up','depthwise','normalized','hidden','activation','projected','output']} for _ in range(2)]
    c=image.chunk.spec
    for start in range(0,len(codes),c.chunk_size):
        context=min(start,c.left_context);front=AudioFrontReference(image.audio.spec,reader.values,dtype=dtype)
        input=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+c.chunk_size,len(codes))]]
        trans=AudioTransformerReference(image.transformer.spec,reader.values,dtype=dtype).sequence(input)['output']
        traces=AudioUpsampleReference(image.upsample.channels,reader.values,dtype=dtype).sequence(trans)
        for stage in range(2):
            for key,value in traces[stage].items():result[stage][key].extend(value[(2 if stage==0 else 4)*context:])
    return [{key:np.asarray(value) for key,value in block.items()} for block in result]


def gelu_accuracy(cfg):
    x=_tensor(0,(1,1024));y=_tensor(4096,(1,1024));b=gelu_probe.trace(cfg,0,{'input':x,'output':y});program=b.finish()
    m=Machine(replace(cfg,DRAM_BYTES=8192),[[I.halt()]],[np.zeros(8192,np.uint8)])
    values=np.linspace(-16,16,4096,dtype=np.float32);values[:4]=[-1e4,1e4,0.,-0.];worst=0.
    for part in values.reshape(4,1024):
        m.slices[0].dram[:4096]=part.view(np.uint8);m.load([program]).run();actual=m.slices[0].dram[4096:8192].view(np.float32)
        wanted=np.array([.5*float(v)*(1+math.erf(float(v)/math.sqrt(2))) for v in part],np.float32)
        worst=max(worst,float(np.max(np.abs(actual-wanted))))
    return dict(grid_points=4096,grid_range=[-16,16],extra_tails=[-10000,10000],max_absolute_error=worst,bound=2e-6,accepted=worst<2e-6)


def run(a):
    started=time.monotonic();cfg=shared_tts_config()
    paths=sorted(set(list((ROOT/'opentpu/llm').glob('qwen3_tts*.py'))+list((ROOT/'opentpu/kernels').glob('tts_*.py'))+
        list((ROOT/'tests').glob('*qwen3_tts_audio*.py'))+list((ROOT/'tests/fixtures').glob('qwen3_tts*.json'))+[Path(__file__).resolve(),ROOT/'tools/qwen3_tts_audio_checkpoint.py',ROOT/'tools/qwen3_tts_audio_front_acceptance.py']))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    config=json.loads((a.data/'audio-config.json').read_text());audio=AudioFrontSpec.from_config(config);spec=AudioTransformerSpec.from_config(config)
    reader=CachedReader(AudioCheckpoint(a.data/'audio-model.safetensors'));fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    cases={}
    for f in fixture['frames']:cases.setdefault(f['text'],[]).append(f['ids'])
    cases['recorded_code_short_chunks_diagnostic']=[fixture['frames'][(t*5+t//4)%len(fixture['frames'])]['ids'] for t in range(14)]
    rows=[];replays=[];preact=[float('inf'),float('-inf')];model_budgets=[];reset_same=True
    for case,codes in cases.items():
        chunk=DecoderChunkSpec(5,3) if case=='recorded_code_short_chunks_diagnostic' else DecoderChunkSpec()
        image=UpsampleChunkImage(cfg,audio,spec,chunk_spec=chunk)
        weights=image.audio.weights(reader);weights.update(image.transformer.weights(reader));weights.update(image.upsample.weights(reader))
        session=UpsampleChunkISASession(image,image.build(weights));refs={name:references(image,reader,codes,dt) for name,dt in [('fp32',np.float32),('fp64',np.float64)]}
        outputs=[]
        for frame,ids in enumerate(codes):
            if case=='recorded_code_short_chunks_diagnostic' and frame in [5,10]:
                for name,(address,size) in image.state_regions().items():
                    if not name.startswith('audio.chunk.'):session.machine.slices[0].dram[address:address+size]=0x7f
            def observe(name,s):
                nonlocal preact
                if 'audio.upsample.' not in name:return
                clean=name[name.index('audio.upsample.'):];bits=clean.split('.');stage=int(bits[2]);m=image.upsample.blocks[stage]
                if bits[3]=='row':
                    hidden=read_tensor(s,m.hidden);preact=[min(preact[0],float(hidden.min())),max(preact[1],float(hidden.max()))]
                if name.startswith('audio.chunk.replay.'):return
                fields={}
                if bits[3]=='pair':
                    offset=0 if stage==0 else 2*int(bits[4]);fields={'up':(read_tensor(s,m.up_output),slice((2 if stage==0 else 4)*frame+offset,(2 if stage==0 else 4)*frame+offset+2))}
                else:
                    index=(2 if stage==0 else 4)*frame+int(bits[4])
                    fields={key:(read_tensor(s,getattr(m,{'activation':'activation','hidden':'hidden','projected':'projected','normalized':'normalized','depthwise':'depthwise','output':'output'}[key])).reshape(-1),index) for key in ['depthwise','normalized','hidden','activation','projected','output']}
                for key,(actual,index) in fields.items():rows.append(dict(case=case,frame=frame,stage=stage,event=clean,field=key,errors={n:metrics(actual,ref[stage][key][index]) for n,ref in refs.items()}))
            outputs.append(session.step(ids,observe))
            a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'component','case':case,'frame':frame+1,'max_feature_error':max(row['errors']['fp32']['relative_l2'] for row in rows)}))
            print(json.dumps({'case':case,'frame':frame+1,'max_feature_error':max(row['errors']['fp32']['relative_l2'] for row in rows)}),flush=True)
        session.reset();count=min(7,len(codes));replayed=np.stack([session.step(ids) for ids in codes[:count]])
        same=bool(np.array_equal(replayed.view(np.uint32),np.stack(outputs[:count]).view(np.uint32)));reset_same=reset_same and same
        replays.append(dict(case=case,frames=count,bitwise=same));model_budgets.append(dict(case=case,dram_bytes=image.nbytes,chunk_size=chunk.chunk_size,left_context=chunk.left_context))
    summary={oracle:{f'{stage}.{key}':max(row['errors'][oracle]['relative_l2'] for row in rows if row['stage']==stage and row['field']==key) for stage in range(2) for key in ['up','depthwise','normalized','hidden','activation','projected','output']} for oracle in ['fp32','fp64']}
    gelu=gelu_accuracy(cfg);component_ok=reset_same and gelu['accepted'] and all(v<=.035 for s in summary.values() for v in s.values())
    image=UpsampleChunkImage(cfg,audio,spec);parts=[p for position in [0,25,324] for p in image.upsample.parts(position)]
    combined_native=UpsampleChunkFrameImage(TalkerSpec(),cfg);integrated=None
    if a.integrated and component_ok:
        main_config=json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text());combined=UpsampleChunkFrameImage(TalkerSpec.from_config(main_config),cfg,chunk_spec=DecoderChunkSpec(2,1))
        own=combined.audio.weights(reader);own.update(combined.transformer.weights(reader));own.update(combined.upsample.weights(reader))
        main_reader=BF16Checkpoint(a.data/'model.safetensors',json.loads((a.data/'checkpoint-manifest.json').read_text()))
        weights={k:main_reader.tensor(k,shape) for k,(_,_,shape) in combined.parameters.items() if k not in own};weights.update(own)
        main=ChunkedAudioFrameSession(combined,combined.build(weights,max_bytes=3<<30));prompt=json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
        golden=[f['ids'] for f in fixture['frames'] if f['text']==prompt['text']];expected=references(combined,reader,golden,np.float32)[1]['output'].reshape(-1,4,1024)
        records=[];runs=[]
        for repetition in range(2):
            main.reset()
            for row in custom_voice_prefill(prompt['input_ids'],main_config):main.step(row.text_id,row.codec_id,input_kind=row.kind)
            outputs=[]
            for frame,want in enumerate(golden):
                ids=main.frame();output=main.read('audio_upsampled').reshape(4,1024);outputs.append(output)
                records.append(dict(repetition=repetition,frame=frame,ids=ids.tolist(),ids_match_preserved_fp32=ids.tolist()==want,device_ids_restored=bool(np.array_equal(main.read('code_ids'),ids)),error=metrics(output,expected[frame])))
                a.output.with_suffix('.integrated.partial.json').write_text(json.dumps(records));a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'integrated','repetition':repetition,'frame':frame+1}))
            runs.append(np.stack(outputs))
        same=bool(np.array_equal(runs[0].view(np.uint32),runs[1].view(np.uint32)));integrated=dict(rows=records,chunk_size=2,left_context=1,reset_replay_bitwise=same,accepted=same and all(r['ids_match_preserved_fp32'] and r['device_ids_restored'] and r['error']['relative_l2']<=.035 for r in records))
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',stable_component_ancestor='e0d68fea470e68f424df0888d38083a7d34f504d',model_revision='85e237c12c027371202489a0ec509ded67b5e4b5',source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e',arithmetic_contract='source weights evaluated with FP32 algebra and explicitly tested erf-GELU approximation; not bit-exact BF16',
        summary=summary,component_accepted=component_ok,threshold_relative_l2=.035,recorded_prefix_frames=9,short_chunk_diagnostic_frames=14,reset_replays=replays,
        gelu_approximation=gelu,observed_model_gelu_input_range=preact,model_budgets=model_budgets,standalone_public_dram_bytes=image.nbytes,combined_public_dram_bytes=combined_native.nbytes,
        upsample_parameter_views=len(image.upsample.sources),upsample_unique_source_tensors=len({v['source'] for v in image.upsample.sources.values()}),upsample_programs_per_decoded_frame=9,
        program_budgets=[{'name':n,**b} for n,_,b in parts],integrated=integrated,output_shape_per_codec_frame=[4,1024],output_is_waveform=False,
        public_300_boundary_with_upsampling_validated=False,upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],upstream_bf16_comparison=fixture['upstream_summary']['bf16'],upstream_thresholds=fixture['upstream_thresholds'],
        waveform_produced=False,raw_scaling_validated=False,complete_utterances_validated=False,quality_validated=False,physical_hardware_access=False,upstream_custom_code_executed=False,tts_enabled=False,host='opentpu-m8a-build',code_sha256=hashes,seconds=time.monotonic()-started)
    detail=a.output.with_suffix('.rows.json');detail.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(detail)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['code_sha256','integrated','program_budgets']},indent=2),flush=True)
    return 0 if component_ok and (integrated is None or integrated['accepted']) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--integrated',action='store_true')
    sys.exit(run(p.parse_args()))
