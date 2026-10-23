#!/usr/bin/env python3
"""Real-weight RVQ/preconv and resident codec-to-decoder diagnostics; no waveform."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec,AudioFrontImage,AudioFrontISASession,AudioFrontFrameImage
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_frame import CodecFrameISASession
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,FILES,digest
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint
from qwen3_tts_audio_reference import AudioFrontReference


def metrics(actual,want):
    x,y=np.asarray(actual,np.float64),np.asarray(want,np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite decoder feature')
    return dict(relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),max_absolute=float(np.max(np.abs(x-y))),
                actual_rms=float(np.sqrt(np.mean(x*x))),reference_rms=float(np.sqrt(np.mean(y*y))),
                actual_min=float(np.min(x)),actual_max=float(np.max(x)))


class CachedReader:
    def __init__(self,reader):self.reader,self.values=reader,{}
    def tensor(self,name,shape):
        if name not in self.values:self.values[name]=self.reader.tensor(name,shape)
        if self.values[name].shape!=tuple(shape):raise ValueError('cached decoder shape mismatch')
        return self.values[name].copy()


def run(a):
    started=time.monotonic()
    paths=[Path(__file__).resolve(),ROOT/'opentpu/llm/qwen3_tts_audio_front.py',ROOT/'opentpu/kernels/tts_audio_front.py',
           ROOT/'tools/qwen3_tts_audio_checkpoint.py',ROOT/'tests/qwen3_tts_audio_reference.py',ROOT/'tests/test_qwen3_tts_audio_front.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    config=json.loads((a.data/'audio-config.json').read_text())
    spec=AudioFrontSpec.from_config(config);cfg=shared_tts_config()
    source=AudioCheckpoint(a.data/'audio-model.safetensors');reader=CachedReader(source)
    image=AudioFrontImage(cfg,spec);weights=image.audio.weights(reader)
    packed=image.build(weights);session=AudioFrontISASession(image,packed)
    refs={'fp32':AudioFrontReference(spec,reader.values),'fp64':AudioFrontReference(spec,reader.values,dtype=np.float64)}
    fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    groups={}
    for f in fixture['frames']:groups.setdefault(f['text'],[]).append(f['ids'])
    groups['edge_and_seeded_diagnostic_codes']=[[0]*16,[2047]*16]+np.random.default_rng(2306).integers(0,2048,(64,16)).tolist()
    rows=[];replay=True
    for text,codes in groups.items():
        session.reset()
        for ref in refs.values():ref.reset()
        for index,ids in enumerate(codes):
            latent,preconv=session.step(ids);actual={'latent':latent,'preconv':preconv,'semantic':session.read('semantic'),'acoustic':session.read('acoustic')}
            expected={n:r.step(ids) for n,r in refs.items()}
            rows.append(dict(case=text,frame=index,ids=ids,errors={n:{s:metrics(actual[s],want[s]) for s in actual} for n,want in expected.items()}))
        if text!='edge_and_seeded_diagnostic_codes':
            old=session.read('preconv');session.reset()
            for ids in codes:session.step(ids)
            replay=replay and bool(np.array_equal(old.view(np.uint32),session.read('preconv').view(np.uint32)))
        a.output.with_suffix('.progress.json').write_text(json.dumps({'case':text,'rows':len(rows),'max_preconv_fp32_error':max(r['errors']['fp32']['preconv']['relative_l2'] for r in rows)}))
        print(json.dumps({'case':text,'rows':len(rows)}),flush=True)
    summary={n:{s:max(r['errors'][n][s]['relative_l2'] for r in rows) for s in ['semantic','acoustic','latent','preconv']} for n in refs}
    component_pass=all(v<=.035 for n in summary.values() for v in n.values()) and replay
    integrated=None
    if a.integrated and component_pass:
        main_config=json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text())
        combined=AudioFrontFrameImage(TalkerSpec.from_config(main_config),cfg,audio_spec=spec)
        audio_weights=combined.audio.weights(reader)
        main_reader=BF16Checkpoint(a.data/'model.safetensors',json.loads((a.data/'checkpoint-manifest.json').read_text()))
        all_weights={k:main_reader.tensor(k,shape) for k,(_,_,shape) in combined.parameters.items() if k not in combined.audio.source_views}
        all_weights.update(audio_weights)
        main=CodecFrameISASession(combined,combined.build(all_weights,max_bytes=3<<30))
        prompt=json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
        golden=[f['ids'] for f in fixture['frames'] if f['text']==prompt['text']]
        records=[];runs=[]
        for repetition in range(2):
            main.reset();reference=AudioFrontReference(spec,reader.values)
            for row in custom_voice_prefill(prompt['input_ids'],main_config):main.step(row.text_id,row.codec_id,input_kind=row.kind)
            outputs=[]
            for index,wanted in enumerate(golden):
                ids=main.frame();expected=reference.step(ids)
                actual=main.read('audio_preconv');outputs.append(actual)
                records.append(dict(repetition=repetition,frame=index,ids=ids.tolist(),codec_ids_match_preserved_fp32_path=ids.tolist()==wanted,
                                    latent=metrics(main.read('audio_rvq'),expected['latent']),preconv=metrics(actual,expected['preconv'])))
                print(json.dumps({'integrated_repetition':repetition,'frame':index+1,'code_ids_equal':ids.tolist()==wanted,'preconv_error':records[-1]['preconv']['relative_l2']}),flush=True)
                a.output.with_suffix('.integrated.partial.json').write_text(json.dumps(records))
            runs.append(np.stack(outputs))
        same=bool(np.array_equal(runs[0].view(np.uint32),runs[1].view(np.uint32)))
        integrated=dict(rows=records,reset_replay_bitwise=same,dram_bytes=combined.nbytes,
                        accepted=same and all(r['codec_ids_match_preserved_fp32_path'] and r['latent']['relative_l2']<=.035 and r['preconv']['relative_l2']<=.035 for r in records))
    parts=image.audio.parts()
    result=dict(profile=profile_metadata(),arithmetic_contract='Pinned source weights evaluated with FP32 algebra; BF16 execution is not bit-exact and is not relabeled passed',
                decoder_checkpoint=dict(bytes=FILES['speech_tokenizer/model.safetensors'][0],sha256=FILES['speech_tokenizer/model.safetensors'][1],header_sha256=source.header_sha256),
                source_views=len(reader.values),physical_parameters=len(image.parameters),source_mapping=image.audio.source_views,
                component_summary=summary,component_accepted=component_pass,component_rows=len(rows),reset_replay_bitwise=replay,
                integrated=integrated,standalone_dram_bytes=image.nbytes,programs=[{'name':n,**b} for n,_,b in parts],
                threshold_relative_l2=.035,upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],
                upstream_bf16_comparison=fixture['upstream_summary']['bf16'],upstream_thresholds=fixture['upstream_thresholds'],
                waveform_produced=False,raw_waveform_scaling_validated=False,complete_utterances_validated=False,
                remaining_stages=['decoder input projection and 8-layer sliding transformer','2x2 causal upsampling and ConvNeXt','4 waveform decoder blocks/SnakeBeta/convolution','raw waveform clamp/scaling and full utterances','quality and coordinated physical/serving acceptance'],
                code_sha256=hashes,seconds=time.monotonic()-started,host='opentpu-m8a-build',physical_hardware_access=False,upstream_custom_code_executed=False,tts_enabled=False)
    details=a.output.with_suffix('.rows.json');details.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(details)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('integrated','source_mapping')},indent=2),flush=True)
    return 0 if component_pass and (integrated is None or integrated['accepted']) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--integrated',action='store_true');a=p.parse_args();sys.exit(run(a))
