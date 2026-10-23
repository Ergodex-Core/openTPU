#!/usr/bin/env python3
"""301 recorded-code frames through public 300/25 replay including upsampling."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_upsample import UpsampleChunkImage,UpsampleChunkISASession
from opentpu.llm.qwen3_tts_talker import _tensor
from opentpu.kernels.tts_audio_chunks import replay_code
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,digest
from tools.qwen3_tts_audio_front_acceptance import CachedReader,metrics
from tools.qwen3_tts_audio_upsample_acceptance import references


def run(a):
    began=time.monotonic();cfg=shared_tts_config()
    paths=list((ROOT/'opentpu/llm').glob('qwen3_tts*.py'))+list((ROOT/'opentpu/kernels').glob('tts_*.py'))+list((ROOT/'tests').glob('*qwen3_tts_audio*.py'))+list((ROOT/'tests/fixtures').glob('qwen3_tts*.json'))+[Path(__file__).resolve(),ROOT/'tools/qwen3_tts_audio_upsample_acceptance.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths if not p.name.startswith('.')}
    config=json.loads((a.data/'audio-config.json').read_text());reader=CachedReader(AudioCheckpoint(a.data/'audio-model.safetensors'))
    image=UpsampleChunkImage(cfg,AudioFrontSpec.from_config(config),AudioTransformerSpec.from_config(config))
    weights=image.audio.weights(reader);weights.update(image.transformer.weights(reader));weights.update(image.upsample.weights(reader))
    fixture=json.loads((ROOT/'tests/fixtures/qwen3_tts_audio_codec_cases.json').read_text())
    codes=[fixture['frames'][(t*5+t//17)%len(fixture['frames'])]['ids'] for t in range(301)];encoded=np.asarray(codes,np.float32)
    if encoded.shape!=(301,16) or any(type(v) is not int or not 0<=v<2048 for ids in codes for v in ids):raise ValueError('invalid diagnostic device producer IDs')
    model_bytes=image.nbytes;address=image.nbytes;size=encoded.nbytes;image.nbytes=((address+size+127)//128)*128
    producer=SimpleNamespace(codes=_tensor(address,(301,16)),ids=image.audio.desc.ids)
    packed=image.build(weights);packed[address:address+size]=encoded.view(np.uint8).reshape(-1);session=UpsampleChunkISASession(image,packed)
    refs={n:references(image,reader,codes,dt)[1]['output'].reshape(301,4,1024) for n,dt in [('fp32',np.float32),('fp64',np.float64)]}
    rows=[];replays=0;saved=[]
    for frame in range(301):
        if frame==300:
            for name,(base,count) in image.state_regions().items():
                if not name.startswith('audio.chunk.'):session.machine.slices[0].dram[base:base+count]=0x7f
        _,programs,_=image.part('diagnostic.device_fixture_producer',replay_code,m=producer,slot=frame);session.machine.load(programs).run()
        def observe(name,s):
            nonlocal replays
            if name.startswith('audio.chunk.replay.') and name.endswith('.ids'):
                j=int(name.split('.')[3]);np.testing.assert_array_equal(s.read('ids'),codes[275+j]);replays+=1
        actual=session.step_resident(observe);np.testing.assert_array_equal(session.read('ids'),codes[frame])
        rows.append(dict(frame=frame,errors={n:metrics(actual,ref[frame]) for n,ref in refs.items()}))
        if frame<3:saved.append(actual)
        if frame%10==0 or frame in [299,300]:
            progress=dict(frame=frame+1,frames=301,replayed_context_frames=replays,max_fp32_error=max(r['errors']['fp32']['relative_l2'] for r in rows))
            a.output.with_suffix('.progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    unchanged=bool(np.array_equal(session.machine.slices[0].dram[address:address+size],encoded.view(np.uint8).reshape(-1)))
    session.reset();again=[]
    for frame in range(3):
        _,programs,_=image.part('diagnostic.device_fixture_producer',replay_code,m=producer,slot=frame);session.machine.load(programs).run();again.append(session.step_resident())
    reset_same=bool(np.array_equal(np.stack(saved).view(np.uint32),np.stack(again).view(np.uint32)))
    maximum={n:max(r['errors'][n]['relative_l2'] for r in rows) for n in refs}
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',component_base='7652f55231cfc8cc356fcc80fa55c0693ac9ae0e',
        model_revision='85e237c12c027371202489a0ec509ded67b5e4b5',source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e',
        arithmetic_contract='source weights evaluated with FP32 algebra; not bit-exact BF16',frames=301,corpus='recorded-prefix-code diagnostic, not newly generated speech',
        public_chunk_size=300,left_context=25,replayed_context_frames=replays,decoder_and_upsample_state_poisoned_at_boundary=True,host_current_or_replay_id_uploads_after_fixture=0,
        fixture_unchanged=unchanged,reset_replay_bitwise=reset_same,max_output_relative_l2=maximum,threshold_relative_l2=.035,standalone_model_bytes=model_bytes,fixture_bytes=image.nbytes-model_bytes,
        public_300_boundary_with_upsampling_validated=unchanged and reset_same and replays==25 and all(v<=.035 for v in maximum.values()),
        native_main_300_frame_generation_validated=False,upstream_overall_gate_passed=fixture['upstream_overall_gate_passed'],upstream_bf16_comparison=fixture['upstream_summary']['bf16'],
        waveform_produced=False,quality_validated=False,physical_hardware_access=False,tts_enabled=False,seconds=time.monotonic()-began,code_sha256=hashes)
    details=a.output.with_suffix('.rows.json');details.write_text(json.dumps(rows,allow_nan=False)+'\n');result['rows_sha256']=digest(details)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='code_sha256'},indent=2),flush=True)
    return 0 if result['public_300_boundary_with_upsampling_validated'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);sys.exit(run(p.parse_args()))
