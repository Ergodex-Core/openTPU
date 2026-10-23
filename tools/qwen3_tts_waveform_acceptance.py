#!/usr/bin/env python3
"""Pinned waveform checkpoint boundaries and guarded ISA samples; never serving."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.llm.qwen3_tts_waveform import WaveformSpec,WaveformChunkImage,WaveformChunkISASession,WaveformChunkFrameImage,WaveformChunkFrameSession
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,digest
from tools.qwen3_tts_audio_front_acceptance import CachedReader,metrics
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from qwen3_tts_audio_upsample_reference import AudioUpsampleReference
from qwen3_tts_waveform_reference import WaveformReference


def read_tensor(s,t):return s.machine.slices[0].dram[t.base.const:t.base.const+4*math.prod(t.shape)].view(np.float32).copy().reshape(t.shape)


def references(image,reader,codes,dtype):
    fields={};raw=[];wave=[];arguments=[];c=image.chunk.spec
    for start in range(0,len(codes),c.chunk_size):
        context=min(start,c.left_context);front=AudioFrontReference(image.audio.spec,reader.values,dtype=dtype)
        input=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+c.chunk_size,len(codes))]]
        trans=AudioTransformerReference(image.transformer.spec,reader.values,dtype=dtype).sequence(input)['output']
        up=AudioUpsampleReference(image.upsample.channels,reader.values,dtype=dtype).sequence(trans)[1]['output']
        result=WaveformReference(image.wave.spec,reader.values,dtype=dtype).sequence(up);arguments.append(result['max_arguments'])
        for _,name,m in image.wave.ops:
            rows=m.output.shape[0]
            fields.setdefault(name,[]).extend(result['traces'][name][context*rows:])
            if getattr(m,'residual',None) is not None:fields.setdefault(name+'.branch',[]).extend(result['traces'][name+'.branch'][context*rows:])
        raw.extend(result['raw'][context*image.wave.desc.samples:]);wave.extend(result['waveform'][context*image.wave.desc.samples:])
    return dict(fields={k:np.asarray(v) for k,v in fields.items()},raw=np.asarray(raw).reshape(len(codes),-1),waveform=np.asarray(wave).reshape(len(codes),-1),argument_maxima=arguments)


def run(a):
    began=time.monotonic();cfg=shared_tts_config()
    paths=list((ROOT/'opentpu/llm').glob('qwen3_tts*.py'))+list((ROOT/'opentpu/kernels').glob('tts_*.py'))+list((ROOT/'tests').glob('*qwen3_tts*.py'))+list((ROOT/'tests/fixtures').glob('qwen3_tts*.json'))+[Path(__file__).resolve(),ROOT/'tools/qwen3_tts_audio_checkpoint.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths)) if not p.name.startswith('.')}
    config=json.loads((a.data/'audio-config.json').read_text());audio=AudioFrontSpec.from_config(config);transformer=AudioTransformerSpec.from_config(config);wave_spec=WaveformSpec.from_config(config)
    reader=CachedReader(AudioCheckpoint(a.data/'audio-model.safetensors'));fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    golden=[f['ids'] for f in fixture['frames'] if f['text']=='Hello.'][:a.frames]
    if len(golden)!=a.frames:raise ValueError('requested more retained prefix frames than available')
    chunk=DecoderChunkSpec(1,1) if a.integrated else DecoderChunkSpec()
    if a.integrated:
        main_config=json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text())
        image=WaveformChunkFrameImage(TalkerSpec.from_config(main_config),cfg,chunk_spec=chunk,waveform_spec=wave_spec)
    else:image=WaveformChunkImage(cfg,audio,transformer,chunk_spec=chunk,waveform_spec=wave_spec)
    own=image.audio.weights(reader);own.update(image.transformer.weights(reader));own.update(image.upsample.weights(reader));own.update(image.wave.weights(reader))
    refs={n:references(image,reader,golden,dt) for n,dt in [('fp32',np.float32),('fp64',np.float64)]}
    if a.integrated:
        main_reader=BF16Checkpoint(a.data/'model.safetensors',json.loads((a.data/'checkpoint-manifest.json').read_text()))
        weights={k:main_reader.tensor(k,shape) for k,(_,_,shape) in image.parameters.items() if k not in own};weights.update(own)
        session=WaveformChunkFrameSession(image,image.build(weights,max_bytes=4<<30))
    else:session=WaveformChunkISASession(image,image.build(own))
    ops={name:m for _,name,m in image.wave.ops};rows=[];outputs=[];frames=[];guards=[]
    def execute(frame,reset_replay=False):
        def observe(name,s):
            if name.startswith('audio.chunk.replay.'):return
            prefix=next((p for p in ['wave.conv.','wave.transposed.','wave.snake.'] if name.startswith(p)),None)
            if prefix is None:return
            key=name[len(prefix):];m=ops[key];actual=read_tensor(s,m.output);count=m.output.shape[0]
            fields={key:actual}
            if getattr(m,'residual',None) is not None:fields[key+'.branch']=read_tensor(s,m.branch)
            for field,value in fields.items():
                if not reset_replay:rows.append(dict(frame=frame,operation=field,errors={n:metrics(value,ref['fields'][field][frame*count:(frame+1)*count]) for n,ref in refs.items()}))
            progress=dict(phase='reset_replay' if reset_replay else 'waveform',frame=frame+1,operation=key,
                          max_fp32_error=max((r['errors']['fp32']['relative_l2'] for r in rows),default=0.))
            a.output.with_suffix('.progress.json').write_text(json.dumps(progress))
        if a.integrated:
            ids=session.frame(observe);output=session.read('audio_waveform')
            match=ids is not None and ids.tolist()==golden[frame]
            restored=ids is not None and bool(np.array_equal(session.read('code_ids'),ids))
        else:output=session.step(golden[frame],observe);match=restored=True
        raw=read_tensor(session,image.wave.desc.raw).reshape(-1);guard=float(read_tensor(session,image.wave.desc.guard).reshape(-1)[0])
        np.testing.assert_array_equal(output,np.clip(raw,-1,1))
        return output,dict(frame=frame,ids_match=match,device_ids_restored=restored,raw={n:metrics(raw,ref['raw'][frame]) for n,ref in refs.items()},waveform={n:metrics(output,ref['waveform'][frame]) for n,ref in refs.items()},guard_max_argument=guard,
            sample_count=len(output),clipped_samples=int(np.count_nonzero(np.abs(raw)>1)),actual_rms=float(np.sqrt(np.mean(output.astype(np.float64)**2))))
    def prefill():
        if not a.integrated:return
        prompt=json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
        for row in custom_voice_prefill(prompt['input_ids'],main_config):session.step(row.text_id,row.codec_id,input_kind=row.kind)
    prefill()
    for frame in range(a.frames):
        output,record=execute(frame);outputs.append(output);frames.append(record);guards.append(record['guard_max_argument'])
        a.output.with_suffix('.frames.partial.json').write_text(json.dumps(frames));print(json.dumps(record),flush=True)
    reset_same=None;reset_record=None
    if a.reset:
        session.reset();prefill();replayed,reset_record=execute(0,True);reset_same=bool(np.array_equal(replayed.view(np.uint32),outputs[0].view(np.uint32)))
    maximum={n:max((r['errors'][n]['relative_l2'] for r in rows),default=0.) for n in refs}
    checked=frames+([reset_record] if reset_record is not None else [])
    accepted=all(v<=.035 for v in maximum.values()) and all(r['ids_match'] and r['device_ids_restored'] and all(r[k][n]['relative_l2']<=.035 for k in ['raw','waveform'] for n in refs) for r in checked) and (reset_same is not False)
    parts=image.wave.parts(25)
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',component_base='7652f55231cfc8cc356fcc80fa55c0693ac9ae0e',model_revision='85e237c12c027371202489a0ec509ded67b5e4b5',source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e',
        arithmetic_contract='pinned source weights evaluated with FP32 algebra; folded static SnakeBeta exp/inverse constants; bounded sin2 device approximation; not bit-exact BF16',
        actual_neural_code_producer=a.integrated,recorded_codec_fixture_producer=not a.integrated,chunk_size=chunk.chunk_size,left_context=chunk.left_context,
        frames=frames,max_operation_relative_l2=maximum,threshold_relative_l2=.035,component_accepted=accepted,reset_first_frame_bitwise=reset_same,reset_record=reset_record,
        source_reference_max_arguments={n:ref['argument_maxima'] for n,ref in refs.items()},max_device_snake_argument=max(guards),snake_argument_limit=262144.,
        output_sample_rate=24000,samples_per_codec_frame=1920,scaling='raw source clamp [-1,1]; no normalization or output tanh',
        waveform_produced_in_isa_simulation=True,waveform_produced_on_fpga=False,complete_utterance_validated=False,quality_validated=False,
        physical_hardware_access=False,tts_enabled=False,upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],upstream_bf16_comparison=fixture['upstream_summary']['bf16'],
        public_300_waveform_boundary_validated=False,main_native_300_frame_generation_validated=False,resident_dram_bytes=image.nbytes,waveform_programs_per_codec_frame=len(parts),
        program_budgets=[{'name':name,**budget} for name,_,budget in parts],source_parameter_views=len(image.wave.sources),source_unique_tensors=len({v['source'] for v in image.wave.sources.values()}),code_sha256=hashes,seconds=time.monotonic()-began)
    detail=a.output.with_suffix('.rows.json');detail.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(detail)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['program_budgets','code_sha256','source_reference_max_arguments']},indent=2),flush=True)
    return 0 if accepted else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,choices=[1,2,3],default=1);p.add_argument('--integrated',action='store_true');p.add_argument('--reset',action='store_true');sys.exit(run(p.parse_args()))
