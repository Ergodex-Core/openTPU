"""PCM -> complete model -> UTF-8 on CPU ISA, never an Inferlab backend.

Only PCM bytes enter the device. Every signal/model stage and every live
handoff executes as device instructions; all Torch work is a separate audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from opentpu import target
from opentpu.asr.phonon2 import MODEL_ID,MODEL_REVISION
from opentpu.asr.phonon2_waveform import compile_waveform
from opentpu.asr.phonon2_encoder import compile_encoder_layer,shared_profile_config,SHARED_PROFILE,SHARED_AGFI,SHARED_ARCH_HASH
from opentpu.asr.phonon2_precise_subsampler import compile_precise_subsampler
from tools.phonon2_encoder_reference import Checkpoint,layer_torch
from tools.phonon2_program_device import ProgramSlotDevice
from tools.phonon2_qualify_encoder import error
from tools.phonon2_qualify_tdt import run_decoder
from tools.phonon2_subsample_reference import subsampler_torch,read_tutorial_audio
from tools.phonon2_torch_audit import frontend_torch
from tools.phonon2_tdt_reference import load_text_asset,read_weights,reference
from tools.phonon2_waveform_audit import coefficients,transport_pcm,snapshots,device_move,canonical_frontend


def run_case(checkpoint,device,pcm,root,out,name):
    started=time.monotonic();cfg=device.cfg
    native=frontend_torch(pcm);canonical=canonical_frontend(native)
    front=compile_waveform(len(pcm)//2,cfg,coefficients(root))
    weights,_=checkpoint.subsampler()
    sub=compile_precise_subsampler(weights,front.frames,front.valid,cfg)
    n,v=sub.stored_lengths[3],sub.valid_lengths[3]
    first_w,_=checkpoint.layer(0);first=compile_encoder_layer(first_w,n,v,cfg,passes=2);del first_w
    initial=device.dispatches,device.constant_bytes,device.program_bytes_uploaded
    device.reset(max(front.image.size,sub.image.size,first.image.size))
    device.constants(front.image,((0,front.input_at),));transport_pcm(device,front,pcm)
    for p in front.programs:device.phase(p,check=(name=='short'))
    actual_front=snapshots(device,front)
    expected_pcm=np.frombuffer(pcm,'<i2').astype(np.float32)/np.float32(32768)
    pcm_exact=np.array_equal(actual_front['wave'][1:front.samples+1],expected_pcm)
    pre_exact=np.array_equal(actual_front['padded_preemphasis'][256:256+front.samples],native['preemphasis'])
    feature_error=error(actual_front['features'],canonical['features'])
    feature_abs=float(np.max(np.abs(actual_front['features']-canonical['features'])))
    mask=bool(np.array_equal(actual_front['features'][front.valid:],np.zeros_like(actual_front['features'][front.valid:])))
    handoff=device_move(device,front.output_at,sub.input_at,front.frames,128,check=(name=='short'))
    device.constants(sub.image,sub.constant_ranges)
    for p in sub.programs:device.phase(p)
    canonical_state=subsampler_torch(weights,canonical['features'],front.valid)['output']
    native_state=subsampler_torch(weights,native['features'],front.valid)['output']
    del weights
    previous=sub.output_at;layers=[]
    for index in range(24):
        weights,_=checkpoint.layer(index)
        plan=first if index==0 else compile_encoder_layer(weights,n,v,cfg,passes=2)
        device.copy(previous,plan.input_at,n)
        device.constants(plan.image,((0,plan.input_at),))
        for p in plan.programs:device.phase(p)
        actual=device.read(plan.output_at,n)
        canonical_state=layer_torch(weights,canonical_state,v)
        native_state=layer_torch(weights,native_state,v)
        glu=device.read(plan.checkpoints['glu'],n)
        probability=device.read(plan.attention_scores_at,n,n)
        masks=bool(np.array_equal(glu[v:],np.zeros_like(glu[v:])) and np.array_equal(probability[:v,v:],np.zeros_like(probability[:v,v:])))
        if v<n:masks&=bool(np.allclose(probability[v:],1/n,rtol=2e-6,atol=1e-8))
        layers.append({'layer':index,'encoder_vs_canonical_dense':error(actual[:v],canonical_state[:v]),'masks_passed':masks,'finite':bool(np.isfinite(actual).all())})
        (out/'progress.json').write_text(json.dumps({'case':name,'stage':'encoder','completed_layers':index+1,'frames':n,'elapsed_seconds':time.monotonic()-started}))
        previous=plan.output_at;del weights
    table,_,_=load_text_asset(root)
    binding={'output_at':previous,'frames':n,'valid':v}
    decoder=run_decoder(checkpoint,device,binding,actual,table,out/'decoder_progress.json',check_waits=(name=='short'))
    weights,hashes=read_weights(checkpoint)
    canonical_decode=reference(weights,canonical_state,v,table=table)
    native_decode=reference(weights,native_state,v,table=table)
    def trace(a):return [[r['token'],r['duration'],r['frame'],int(r['emit'])] for r in a['records']]
    canonical_equal=decoder['trace']==trace(canonical_decode) and decoder['text_utf8'].encode()==canonical_decode['text'] and decoder['status']==canonical_decode['status']
    native_equal=decoder['trace']==trace(native_decode) and decoder['text_utf8'].encode()==native_decode['text'] and decoder['status']==native_decode['status']
    front_accepted=bool(pcm_exact and pre_exact and mask and feature_abs<.002 and np.isfinite(actual_front['features']).all())
    accepted=bool(front_accepted and all(r['masks_passed'] and r['finite'] for r in layers) and decoder['accepted'] and canonical_equal and native_equal)
    return {'name':name,'accepted':accepted,'hardware_validated':False,'model_ready':False,'pcm_sha256':hashlib.sha256(pcm).hexdigest(),
            'samples':front.samples,'frame_counts':{'mel_stored':front.frames,'mel_valid':front.valid,'encoder_stored':n,'encoder_valid':v},
            'frontend':{'accepted':front_accepted,'pcm_exact':bool(pcm_exact),'preemphasis_exact':bool(pre_exact),'mask_exact':mask,
                        'log_mel_vs_torch':error(actual_front['log_mel'],native['log_mel']),
                        'features_vs_pinned_arm_stats':feature_error,'features_max_abs_vs_pinned_arm_stats':feature_abs,
                        'features_vs_native_cpu_torch':error(actual_front['features'],native['features']),
                        'native_cpu_vs_pinned_arm_stats_max_abs':float(np.max(np.abs(native['features']-canonical['features']))),
                        'output_sha256':hashlib.sha256(actual_front['features'].tobytes()).hexdigest(),'budget':front.budget()},
            'waveform_to_subsampler_handoff':handoff,'encoder_layers':layers,'encoder_output_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'decoder':decoder,'complete_canonical_dense_trace_and_text_equal':bool(canonical_equal),
            'complete_native_torch_dense_trace_and_text_equal':bool(native_equal),
            'canonical_reference_text':canonical_decode['text'].decode(),'native_reference_text':native_decode['text'].decode(),
            'canonical_reference_trace':trace(canonical_decode),'native_reference_trace':trace(native_decode),
            'total_dispatches':device.dispatches-initial[0],'immutable_bytes_staged':device.constant_bytes-initial[1],
            'program_wait_bytes_uploaded':device.program_bytes_uploaded-initial[2],'graph_high_water_bytes':device.high_water,
            'host_audio_preprocessing_or_model_compute_in_serving_path':False,'elapsed_seconds':time.monotonic()-started}


def qualify(container,pcm,wave,root,output_dir):
    import torch
    torch.set_num_threads(1);root,out=Path(root),Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=shared_profile_config()
    if f'{target.arch_hash(cfg):08x}'!=SHARED_ARCH_HASH:raise RuntimeError('shared profile mismatch')
    prior=json.loads((root/'docs/phonon2/waveform_prior_binding.json').read_text())
    for name,sha in prior['preserved_source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('preserved source changed '+name)
    import opentpu
    runtime=Path(opentpu.__file__).resolve().parents[1];manifest=json.loads((runtime/'source_manifest.json').read_text())
    if manifest!=prior['runtime_manifest']:raise RuntimeError('runtime manifest differs')
    for name,sha in manifest['files'].items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('runtime source changed '+name)
    checkpoint,device=Checkpoint(container),ProgramSlotDevice(cfg)
    report={'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'profile':SHARED_PROFILE,'agfi':SHARED_AGFI,'declared_cfg_hash':SHARED_ARCH_HASH,
            'runtime_manifest':manifest,'prior_binding':prior,'hardware_used':False,'model_ready':False,'physical_audio_to_transcript_validated':False,
            'frontend_policy':'Nondithered pinned source equations; deterministic Torch 5c488690 ARM64 Vec4/ILP4 statistics matching retained vectors',
            'host_role':'PCM byte transport and UTF-8 byte output I/O only in proposed serving path; all Torch code is separate audit',
            'cases':[]}
    short=Path(pcm).read_bytes();raw=Path(wave).read_bytes()
    if hashlib.sha256(short).hexdigest()!='7f84e087aa28b5a33212af26a4273b4902761b2543afadc32c3236bf51ca3f74':raise ValueError('short fixture differs')
    if hashlib.sha256(raw).hexdigest()!='67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8':raise ValueError('full fixture differs')
    # Float-WAV conversion is offline fixture preparation, outside the PCM
    # inference path. Record the resulting PCM hash for the actual model input.
    full=read_tutorial_audio(wave).tobytes()
    for name,data in [('short',short),('full_speech',full),('silence_min',bytes(640)),('silence_reduction_boundary',bytes(5280*2)),('short_after_reset',short)]:
        result=run_case(checkpoint,device,data,root,out,name);report['cases'].append(result)
        (out/'waveform_report.json').write_text(json.dumps(report,indent=2))
    first,last=report['cases'][0],report['cases'][-1]
    report['reset_bitwise_passed']=bool(first['frontend']['output_sha256']==last['frontend']['output_sha256'] and first['encoder_output_sha256']==last['encoder_output_sha256'] and all(first['decoder'][k]==last['decoder'][k] for k in ['state_sha256','hidden_cell_cache_sha256','text_sha256','trace']))
    report['accepted']=bool(report['reset_bitwise_passed'] and all(c['accepted'] for c in report['cases']))
    names=['opentpu/asr/phonon2_waveform.py','tools/phonon2_waveform_audit.py','tools/phonon2_qualify_waveform.py','docs/phonon2/waveform_coefficients.npz','docs/phonon2/waveform_coefficients.json','docs/phonon2/waveform_prior_binding.json']
    report['source_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    report['environment']={'python':__import__('sys').version,'numpy':np.__version__,'torch':torch.__version__,'torch_git':torch.version.git_version}
    (out/'waveform_report.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('container','pcm','wave','root','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=qualify(a.container,a.pcm,a.wave,a.root,a.output_dir)
    print(json.dumps({'accepted':r['accepted'],'reset_bitwise_passed':r['reset_bitwise_passed'],'hardware_used':False,'cases':[{'name':c['name'],'accepted':c['accepted'],'text':c['decoder']['text_utf8'],'native_trace_equal':c['complete_native_torch_dense_trace_and_text_equal']} for c in r['cases']]}))
