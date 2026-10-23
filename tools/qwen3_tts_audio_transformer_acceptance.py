#!/usr/bin/env python3
"""Checkpoint decoder features: source FP32/FP64, cache, chunks and resident IDs."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec,AudioTransformerImage,AudioTransformerISASession,AudioTransformerFrameImage,AudioTransformerFrameSession
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,FILES,digest
from tools.qwen3_tts_audio_front_acceptance import CachedReader,metrics
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference


def run(a):
    started=time.monotonic();cfg=shared_tts_config()
    paths=sorted(set(list((ROOT/'opentpu/llm').glob('qwen3_tts*.py'))+list((ROOT/'opentpu/kernels').glob('tts_*.py'))+
                     [Path(__file__).resolve(),ROOT/'tests/qwen3_tts_audio_transformer_reference.py',ROOT/'tests/test_qwen3_tts_audio_transformer.py',
                      ROOT/'tests/qwen3_tts_audio_reference.py',ROOT/'tools/qwen3_tts_audio_checkpoint.py',ROOT/'tools/qwen3_tts_audio_front_acceptance.py']+
                     list((ROOT/'tests/fixtures').glob('qwen3_tts*.json'))))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    c=json.loads((a.data/'audio-config.json').read_text());spec=AudioTransformerSpec.from_config(c)
    reader=CachedReader(AudioCheckpoint(a.data/'audio-model.safetensors'))
    image=AudioTransformerImage(cfg,AudioFrontSpec.from_config(c),spec)
    weights=image.audio.weights(reader);weights.update(image.transformer.weights(reader))
    session=AudioTransformerISASession(image,image.build(weights))
    fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    groups={}
    for f in fixture['frames']:groups.setdefault(f['text'],[]).append(f['ids'])
    # Repeating previously generated prefix IDs exercises cache length, not speech
    # generation/quality. Keep it visibly separate from the nine real prefixes.
    groups['repeated_generated_prefix_codes_for_cache_only']=[fixture['frames'][t%len(fixture['frames'])]['ids'] for t in range(80)]
    rows=[];long_outputs=[];batch_checks={};saved_prefix=None
    for case,codes in groups.items():
        session.reset()
        fronts={n:AudioFrontReference(image.spec,reader.values,dtype=dt) for n,dt in [('fp32',np.float32),('fp64',np.float64)]}
        refs={n:AudioTransformerReference(spec,reader.values,dtype=dt) for n,dt in [('fp32',np.float32),('fp64',np.float64)]}
        conditional=AudioTransformerReference(spec,reader.values);source_inputs=[];source_traces=[];outputs=[]
        for frame,ids in enumerate(codes):
            expected={n:refs[n].step(fronts[n].step(ids)['preconv']) for n in refs}
            actual={}
            def observe(name,s):
                if name=='audio.transformer.input':actual['input']=s.read('transformer_hidden')
                if name.endswith('.finish'):actual['layer.'+name.split('.')[2]]=s.read('transformer_hidden')
                if name=='audio.transformer.output':
                    t=image.transformer.desc.normalized;dram=s.machine.slices[0].dram
                    actual['normalized']=dram[t.base.const:t.base.const+4*spec.hidden].view(np.float32).copy()
            actual['output']=session.step(ids,observe);outputs.append(actual['output'])
            conditional_trace=conditional.step(session.read('preconv'))
            expected['conditional_fp32']=conditional_trace
            row=dict(case=case,frame=frame,visible_keys=min(frame+1,spec.window),ring_slot=frame%spec.window,
                     errors={n:{key:metrics(value,expected[n][key]) for key,value in actual.items()} for n in expected})
            rows.append(row)
            if case=='repeated_generated_prefix_codes_for_cache_only':
                source_traces.append(expected['fp32'])
            a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'component','case':case,'frame':frame+1,'rows':len(rows),
                'max_output_relative_l2':max(r['errors']['fp32']['output']['relative_l2'] for r in rows)}))
        if saved_prefix is None:saved_prefix=(codes,np.stack(outputs))
        if case=='repeated_generated_prefix_codes_for_cache_only':
            long_outputs=outputs
            front=AudioFrontReference(image.spec,reader.values)
            source_inputs=[front.step(ids)['preconv'] for ids in codes]
            full=refs['fp32'].sequence(source_inputs)
            batch_checks={key:max(metrics(trace[key],full[key][t])['relative_l2'] for t,trace in enumerate(source_traces)) for key in full}
        print(json.dumps({'case':case,'frames':len(codes),'max_output_error':max(r['errors']['fp32']['output']['relative_l2'] for r in rows if r['case']==case)}),flush=True)
    # Poison both front history and all decoder ring slots, then clear via the
    # public reset and replay the original source-format generated prefix.
    for address,size in image.state_regions().values():session.machine.slices[0].dram[address:address+size]=0x7f
    session.reset();replayed=np.stack([session.step(ids) for ids in saved_prefix[0]])
    replay=bool(np.array_equal(replayed.view(np.uint32),saved_prefix[1].view(np.uint32)))
    codes=groups['repeated_generated_prefix_codes_for_cache_only']
    # The first 64-frame chunk equals the already measured continuous prefix.
    # Recompute only the second chunk with the official 25-frame left context;
    # both its rotary positions and preconv/transformer states restart at zero.
    start=64;context=25;session.reset()
    chunk_codes=codes[start-context:];front=AudioFrontReference(image.spec,reader.values)
    chunk_ref=AudioTransformerReference(spec,reader.values)
    chunk_inputs=[front.step(ids)['preconv'] for ids in chunk_codes]
    chunk_expected=chunk_ref.sequence(chunk_inputs)['output'][context:]
    chunk_actual=[]
    for j,ids in enumerate(chunk_codes):
        output=session.step(ids)
        if j>=context:chunk_actual.append(output)
        a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'source_chunk_recompute','context':context,'frame':j+1,'frames':len(chunk_codes)}))
    chunk_errors=[metrics(x,y) for x,y in zip(chunk_actual,chunk_expected)]
    chunks=dict(chunk_size=64,left_context_size=25,source_default_chunk_size=300,
                recomputed_frames=len(chunk_codes),emitted_second_chunk_frames=len(chunk_actual),
                max_error=max(x['relative_l2'] for x in chunk_errors),rows=chunk_errors,
                continuous_vs_recomputed_relative_l2=metrics(np.stack(long_outputs[start:]),np.stack(chunk_actual))['relative_l2'],
                continuous_and_chunked_equivalence_claimed=False)
    summary={n:{key:max(r['errors'][n][key]['relative_l2'] for r in rows) for key in rows[0]['errors'][n]} for n in rows[0]['errors']}
    component_ok=replay and max(batch_checks.values())<=1e-5 and chunks['max_error']<=.035 and all(v<=.035 for x in summary.values() for v in x.values())
    budgets=[]
    for position in [0,71,72,143,7999]:
        parts=image.transformer.parts(position)
        budgets.append(dict(position=position,programs=len(parts),max_imem_words=max(b['imem_words'] for _,_,b in parts),max_tmem_words=max(b['tmem_peak_words'] for _,_,b in parts),
                            first_layer=[{'name':n,**b} for n,_,b in parts if n in ['audio.transformer.input','audio.transformer.0.qkv','audio.transformer.0.head.0','audio.transformer.0.finish','audio.transformer.output']]))
    integrated=None
    if a.integrated and component_ok:
        main_config=json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text())
        combined=AudioTransformerFrameImage(TalkerSpec.from_config(main_config),cfg,transformer_spec=spec)
        audio_weights=combined.audio.weights(reader);audio_weights.update(combined.transformer.weights(reader))
        main_reader=BF16Checkpoint(a.data/'model.safetensors',json.loads((a.data/'checkpoint-manifest.json').read_text()))
        all_weights={k:main_reader.tensor(k,shape) for k,(_,_,shape) in combined.parameters.items() if k not in audio_weights}
        all_weights.update(audio_weights);main=AudioTransformerFrameSession(combined,combined.build(all_weights,max_bytes=3<<30))
        prompt=json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
        golden=[f['ids'] for f in fixture['frames'] if f['text']==prompt['text']]
        records=[];runs=[]
        for repetition in range(2):
            main.reset();front=AudioFrontReference(image.spec,reader.values);reference=AudioTransformerReference(spec,reader.values)
            for row in custom_voice_prefill(prompt['input_ids'],main_config):main.step(row.text_id,row.codec_id,input_kind=row.kind)
            outputs=[]
            for frame,wanted in enumerate(golden):
                ids=main.frame();expected=reference.step(front.step(ids)['preconv']);actual=main.read('audio_transformed');outputs.append(actual)
                records.append(dict(repetition=repetition,frame=frame,decoder_position=frame,ids=ids.tolist(),codec_ids_match_preserved_fp32_path=ids.tolist()==wanted,output=metrics(actual,expected['output'])))
                a.output.with_suffix('.integrated.partial.json').write_text(json.dumps(records))
                a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'integrated','repetition':repetition,'frame':frame+1}))
                print(json.dumps({'integrated_repetition':repetition,'frame':frame+1,'ids_equal':records[-1]['codec_ids_match_preserved_fp32_path'],'output_error':records[-1]['output']['relative_l2']}),flush=True)
            runs.append(np.stack(outputs))
        same=bool(np.array_equal(runs[0].view(np.uint32),runs[1].view(np.uint32)))
        integrated=dict(rows=records,reset_replay_bitwise=same,dram_bytes=combined.nbytes,
                        accepted=same and all(r['codec_ids_match_preserved_fp32_path'] and r['output']['relative_l2']<=.035 for r in records))
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',component_base_revision='85a42df959fd071f40fd488e4cc7844c96ece83b',
                model_revision='85e237c12c027371202489a0ec509ded67b5e4b5',source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e',
                arithmetic_contract='Pinned source weights evaluated with FP32 algebra; not bit-exact BF16 execution',
                decoder_checkpoint=dict(bytes=FILES['speech_tokenizer/model.safetensors'][0],sha256=FILES['speech_tokenizer/model.safetensors'][1]),
                transformer_parameters=len(image.transformer.sources),summary=summary,component_rows=len(rows),component_accepted=component_ok,
                source_full_sequence_vs_incremental_errors=batch_checks,source_chunk_recompute=chunks,reset_poison_replay_bitwise=replay,integrated=integrated,
                standalone_dram_bytes=image.nbytes,program_budgets=budgets,threshold_relative_l2=.035,source_mask_reference_bound=1e-5,
                upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],upstream_bf16_comparison=fixture['upstream_summary']['bf16'],upstream_thresholds=fixture['upstream_thresholds'],
                waveform_produced=False,raw_waveform_scaling_validated=False,complete_utterances_validated=False,quality_validated=False,
                physical_hardware_access=False,upstream_custom_code_executed=False,tts_enabled=False,host='opentpu-m8a-build',code_sha256=hashes,seconds=time.monotonic()-started)
    details=a.output.with_suffix('.rows.json');details.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(details)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['integrated','source_chunk_recompute','program_budgets','code_sha256']},indent=2),flush=True)
    return 0 if component_ok and (integrated is None or integrated['accepted']) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--integrated',action='store_true')
    sys.exit(run(p.parse_args()))
