#!/usr/bin/env python3
"""Real-weight 300/25 boundaries, device fixture producer, and neural integration."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec,ChunkedAudioImage,ChunkedAudioISASession,ChunkedAudioFrameImage,ChunkedAudioFrameSession
from opentpu.llm.qwen3_tts_talker import TalkerSpec,_tensor
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.kernels.tts_audio_chunks import replay_code
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,digest
from tools.qwen3_tts_audio_front_acceptance import CachedReader,metrics
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference


def reference_outputs(image,reader,codes,dtype,chunk=300,left=25):
    outputs=[];hidden=[]
    for start in range(0,len(codes),chunk):
        context=min(start,left);front=AudioFrontReference(image.spec,reader.values,dtype=dtype)
        inputs=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+chunk,len(codes))]]
        ref=AudioTransformerReference(image.transformer.spec,reader.values,dtype=dtype).sequence(inputs)
        outputs.extend(ref['output'][context:]);hidden.extend(ref['layer.7'][context:])
    return np.asarray(outputs),np.asarray(hidden)


def run(a):
    began=time.monotonic();cfg=shared_tts_config()
    paths=sorted(set(list((ROOT/'opentpu/llm').glob('qwen3_tts*.py'))+list((ROOT/'opentpu/kernels').glob('tts_*.py'))+
        list((ROOT/'tests').glob('*qwen3_tts_audio*.py'))+list((ROOT/'tests/fixtures').glob('qwen3_tts*.json'))+
        [Path(__file__).resolve(),ROOT/'tools/qwen3_tts_audio_checkpoint.py',ROOT/'tools/qwen3_tts_audio_front_acceptance.py']))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    config=json.loads((a.data/'audio-config.json').read_text());reader=CachedReader(AudioCheckpoint(a.data/'audio-model.safetensors'))
    image=ChunkedAudioImage(cfg,AudioFrontSpec.from_config(config),AudioTransformerSpec.from_config(config))
    weights=image.audio.weights(reader);weights.update(image.transformer.weights(reader))
    fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    # Not a generated utterance: recorded neural prefix IDs are permuted into a
    # long diagnostic fixture. All request-time current-ID production below is
    # a device copy from this immutable table, and replay never uploads IDs.
    codes=[fixture['frames'][(t*5+t//17)%len(fixture['frames'])]['ids'] for t in range(602)]
    encoded=np.asarray(codes,np.float32)
    assert encoded.shape==(602,16) and np.all(encoded>=0) and np.all(encoded<2048)
    model_bytes=image.nbytes;address=image.nbytes;size=encoded.nbytes
    image.regions['diagnostic.recorded_codes']=(address,size);image.nbytes=((address+size+127)//128)*128
    producer=SimpleNamespace(codes=_tensor(address,(602,16)),ids=image.audio.desc.ids)
    packed=image.build(weights);packed[address:address+size]=encoded.view(np.uint8).reshape(-1)
    session=ChunkedAudioISASession(image,packed)
    references={n:reference_outputs(image,reader,codes,dt) for n,dt in [('fp32',np.float32),('fp64',np.float64)]}
    continuous,_=reference_outputs(image,reader,codes,np.float32,chunk=602,left=0)
    mode_difference=[metrics(x,y)['relative_l2'] for x,y in zip(continuous,references['fp32'][0])]
    rows=[];saved=[];replayed_ids=0;restores=0
    for frame in range(len(codes)):
        _,context,position=image.chunk.coordinates(frame)
        if frame in [300,600]:
            for name,(base,count) in image.state_regions().items():
                if name.startswith(('audio.state.','audio.transformer.state.')):session.machine.slices[0].dram[base:base+count]=0x7f
        _,programs,_=image.part('diagnostic.device_fixture_producer',replay_code,m=producer,slot=frame)
        session.machine.load(programs).run();last_hidden=None
        def observe(name,s):
            nonlocal replayed_ids,restores,last_hidden
            if name.startswith('audio.chunk.replay.') and name.endswith('.ids'):
                index=int(name.split('.')[3]);np.testing.assert_array_equal(s.read('ids'),codes[frame-context+index]);replayed_ids+=1
            if name=='audio.chunk.restore':np.testing.assert_array_equal(s.read('ids'),codes[frame]);restores+=1
            if name=='audio.transformer.7.finish':last_hidden=s.read('transformer_hidden')
        actual=session.step_resident(observe)
        np.testing.assert_array_equal(session.read('ids'),codes[frame])
        rows.append(dict(frame=frame,chunk_start=frame//300*300,decoder_position=position,
            output={n:metrics(actual,value[0][frame]) for n,value in references.items()},
            layer7={n:metrics(last_hidden,value[1][frame]) for n,value in references.items()}))
        if frame<5:saved.append(actual)
        if frame%10==0 or frame in [299,300,599,600,601]:
            progress={'phase':'public_300_25','frame':frame+1,'frames':602,'replayed_context_ids':replayed_ids,'max_output_relative_l2':max(r['output']['fp32']['relative_l2'] for r in rows)}
            a.output.with_suffix('.progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    table_unchanged=bool(np.array_equal(session.machine.slices[0].dram[address:address+size],encoded.view(np.uint8).reshape(-1)))
    session.reset();reset_outputs=[]
    for frame in range(5):
        _,programs,_=image.part('diagnostic.device_fixture_producer',replay_code,m=producer,slot=frame)
        session.machine.load(programs).run();reset_outputs.append(session.step_resident())
    reset_same=bool(np.array_equal(np.stack(saved).view(np.uint32),np.stack(reset_outputs).view(np.uint32)))
    max_output={n:max(row['output'][n]['relative_l2'] for row in rows) for n in references}
    max_layer7={n:max(row['layer7'][n]['relative_l2'] for row in rows) for n in references}
    component_ok=table_unchanged and reset_same and replayed_ids==50 and restores==2 and all(v<=.035 for x in [max_output,max_layer7] for v in x.values())
    parts=image.chunk.parts(300)
    ordinary_parts=image.chunk.parts(347)
    native_combined=ChunkedAudioFrameImage(TalkerSpec(),cfg)
    integrated=None
    if a.integrated and component_ok:
        main_config=json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text())
        combined=ChunkedAudioFrameImage(TalkerSpec.from_config(main_config),cfg,chunk_spec=DecoderChunkSpec(2,1))
        audio_weights=combined.audio.weights(reader);audio_weights.update(combined.transformer.weights(reader))
        main_reader=BF16Checkpoint(a.data/'model.safetensors',json.loads((a.data/'checkpoint-manifest.json').read_text()))
        all_weights={k:main_reader.tensor(k,shape) for k,(_,_,shape) in combined.parameters.items() if k not in audio_weights};all_weights.update(audio_weights)
        main=ChunkedAudioFrameSession(combined,combined.build(all_weights,max_bytes=3<<30))
        prompt=json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
        golden=[f['ids'] for f in fixture['frames'] if f['text']==prompt['text']]
        expected,_=reference_outputs(image,reader,golden,np.float32,chunk=2,left=1)
        records=[];runs=[]
        for repetition in range(2):
            main.reset()
            for row in custom_voice_prefill(prompt['input_ids'],main_config):main.step(row.text_id,row.codec_id,input_kind=row.kind)
            outputs=[]
            for frame,want in enumerate(golden):
                ids=main.frame();output=main.read('audio_transformed');outputs.append(output)
                records.append(dict(repetition=repetition,frame=frame,ids=ids.tolist(),codec_ids_match_preserved_fp32_path=ids.tolist()==want,
                    current_device_ids_restored=bool(np.array_equal(main.read('code_ids'),ids)),output=metrics(output,expected[frame])))
                a.output.with_suffix('.integrated.partial.json').write_text(json.dumps(records));a.output.with_suffix('.progress.json').write_text(json.dumps({'phase':'integrated_neural_producer_short_chunks','repetition':repetition,'frame':frame+1}))
            runs.append(np.stack(outputs))
        same=bool(np.array_equal(runs[0].view(np.uint32),runs[1].view(np.uint32)))
        integrated=dict(chunk_size=2,left_context=1,public_default_boundary_test=False,rows=records,reset_replay_bitwise=same,
            accepted=same and all(r['codec_ids_match_preserved_fp32_path'] and r['current_device_ids_restored'] and r['output']['relative_l2']<=.035 for r in records))
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',component_base_revision='e0d68fea470e68f424df0888d38083a7d34f504d',
        model_revision='85e237c12c027371202489a0ec509ded67b5e4b5',source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e',
        arithmetic_contract='source weights evaluated with FP32 algebra; not bit-exact BF16',
        corpus='602-frame recorded-prefix-code permutation diagnostic; not a generated utterance',frames=602,public_chunk_size=300,left_context=25,
        boundary_frames_zero_based=[300,600],replayed_context_ids=replayed_ids,current_id_restores=restores,
        decoder_state_poisoned_at_boundaries=True,host_current_id_uploads_after_initial_fixture=0,host_replay_id_uploads=0,
        fixture_table_unchanged=table_unchanged,reset_replay_bitwise=reset_same,max_output_relative_l2=max_output,max_layer7_relative_l2=max_layer7,
        max_source_continuous_vs_public_chunk_relative_l2=max(mode_difference),source_mode_differences_at_boundaries={str(t):mode_difference[t] for t in [299,300,301,599,600,601]},
        continuous_mode_substituted=False,component_accepted=component_ok,integrated=integrated,threshold_relative_l2=.035,
        standalone_model_dram_bytes=model_bytes,diagnostic_fixture_dram_bytes=image.nbytes-model_bytes,public_combined_dram_bytes=native_combined.nbytes,
        resident_chunk_state_dram_increment_bytes=1792,ordinary_decoder_programs=150,boundary_decoder_programs=len(parts),
        max_imem_words=max(b['imem_words'] for _,_,b in parts+ordinary_parts),max_tmem_words=max(b['tmem_peak_words'] for _,_,b in parts+ordinary_parts),
        upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],upstream_bf16_comparison=fixture['upstream_summary']['bf16'],upstream_thresholds=fixture['upstream_thresholds'],
        complete_native_main_generated_300_frame_sequence_validated=False,waveform_produced=False,quality_validated=False,physical_hardware_access=False,
        upstream_custom_code_executed=False,tts_enabled=False,host='opentpu-m8a-build',seconds=time.monotonic()-began,code_sha256=hashes)
    details=a.output.with_suffix('.rows.json');details.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(details)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['integrated','code_sha256']},indent=2),flush=True)
    return 0 if component_ok and (integrated is None or integrated['accepted']) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--integrated',action='store_true')
    sys.exit(run(p.parse_args()))
