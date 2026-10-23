"""Real-checkpoint encoder-to-TDT ISA audit on existing EC2; no serving/card.

The retained encoder source/output binding is checked before evidence is reused.
Every learned handoff is a device copy; host reads are audit observations only.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from opentpu import target
from opentpu.asr.phonon2 import MODEL_ID, MODEL_REVISION
from opentpu.asr.phonon2_encoder import compile_encoder_layer, shared_profile_config, SHARED_PROFILE, SHARED_AGFI, SHARED_ARCH_HASH
from opentpu.asr.phonon2_precise_subsampler import compile_precise_subsampler
from opentpu.asr.phonon2_tdt import compile_decoder, POLICY, STATUS, STEPS
from tools.phonon2_encoder_reference import Checkpoint
from tools.phonon2_program_device import ProgramSlotDevice
from tools.phonon2_qualify_encoder import error
from tools.phonon2_subsample_reference import read_tutorial_audio
from tools.phonon2_torch_audit import frontend_torch
from tools.phonon2_tdt_reference import read_weights, load_text_asset, reference, read_result


def run_encoder(checkpoint, device, frontend, progress, expected_sha):
    start = time.monotonic()
    sub_weights, _ = checkpoint.subsampler()
    sub = compile_precise_subsampler(sub_weights, len(frontend['features']), frontend['valid_frames'], device.cfg)
    del sub_weights
    n, valid = sub.stored_lengths[3], sub.valid_lengths[3]
    w, _ = checkpoint.layer(0)
    first = compile_encoder_layer(w, n, valid, device.cfg, passes=2)
    del w
    device.reset(max(sub.image.size, first.image.size))
    device.constants(sub.image, sub.constant_ranges)
    x = np.ascontiguousarray(frontend['features'], np.float32)
    device.memory[sub.input_at:sub.input_at+x.nbytes] = x.view(np.uint8).ravel()  # offline waveform oracle only
    for p in sub.programs:
        device.phase(p)
    previous = sub.output_at
    for layer in range(24):
        w, _ = checkpoint.layer(layer)
        plan = first if layer == 0 else compile_encoder_layer(w, n, valid, device.cfg, passes=2)
        del w
        device.copy(previous, plan.input_at, n)
        device.constants(plan.image, ((0, plan.input_at),))
        for p in plan.programs:
            device.phase(p)
        previous = plan.output_at
        Path(progress).write_text(json.dumps({'stage':'encoder','layer':layer+1,'frames':n,'elapsed_seconds':time.monotonic()-start}))
    actual = device.read(previous, n)
    sha = hashlib.sha256(actual.tobytes()).hexdigest()
    if sha != expected_sha:
        raise AssertionError('encoder output differs from preserved cf6bb28e receipt: '+sha)
    return {'output_at':previous,'frames':n,'valid':valid,'output_sha256':sha,'matched_preserved_receipt':True,
            'elapsed_seconds':time.monotonic()-start}, actual


def run_decoder(checkpoint, device, binding, encoder_audit, text_table, progress, *, check_waits):
    start = time.monotonic()
    weights, hashes = read_weights(checkpoint)
    plan = compile_decoder(weights, text_table, binding['frames'], binding['valid'], device.cfg, audit=True)
    a, b, nbytes = binding['output_at'], plan.input_at, encoder_audit.nbytes
    if max(a,b) < min(a+nbytes,b+nbytes):
        raise AssertionError('encoder/decoder handoff ranges overlap')
    counters = device.dispatches, device.constant_bytes, device.program_bytes_uploaded, device.wait_checked
    device.copy(a, b, plan.frames, check=check_waits)
    device.constants(plan.image, ((0, plan.input_at),))
    for index, p in enumerate(plan.programs):
        device.phase(p, check=check_waits)
        Path(progress).write_text(json.dumps({'stage':'decoder','phase':index+1,'frames':plan.frames,'elapsed_seconds':time.monotonic()-start}))
    actual = read_result(device, plan)
    expected = reference(weights, encoder_audit, plan.valid, max_steps=plan.max_steps, table=text_table)
    expected_trace = np.asarray([[r['token'],r['duration'],r['frame'],float(r['emit'])] for r in expected['records']],np.float32)
    trace_equal = np.array_equal(actual['trace'],expected_trace)
    comparisons = []
    if trace_equal:
        for index, record in enumerate(expected['records']):
            state = np.concatenate((record['hidden'][:,0],record['cell'][:,0])).ravel()
            comparisons.append({'step':index,'state_vs_dense':error(actual['audit_hidden'][index],state),
                                'cache_vs_dense':error(actual['audit_cache'][index],record['cache'][0])})
    finite = all(np.isfinite(actual[k]).all() for k in ('state','trace','hidden','cell','cache','audit_hidden','audit_cache'))
    status = int(actual['state'][STATUS])
    text_equal = actual['text'] == expected['text']
    max_state = max((r['state_vs_dense']['relative_l2'] for r in comparisons),default=float('inf'))
    max_cache = max((r['cache_vs_dense']['relative_l2'] for r in comparisons),default=float('inf'))
    accepted = bool(trace_equal and text_equal and status in (1,2) and status==expected['status'] and finite and max_state < .01 and max_cache < .01)
    result = {'accepted':accepted,'model_ready':False,'hardware_validated':False,'status':status,'expected_status':expected['status'],
              'steps':int(actual['state'][STEPS]),'token_duration_frame_trace_equal':bool(trace_equal),'text_equal':bool(text_equal),
              'text_utf8':actual['text'].decode('utf-8'),'expected_text_utf8':expected['text'].decode('utf-8'),
              'trace':actual['trace'].astype(int).tolist(),'expected_trace':expected_trace.astype(int).tolist(),
              'all_states_finite':bool(finite),'state_comparisons':comparisons,
              'maximum_state_relative_l2':max_state,'maximum_cache_relative_l2':max_cache,
              'encoder_projector_vs_dense':error(actual['encoder_projected'][:plan.valid],expected['encoder_projected'][:plan.valid]),
              'state':actual['state'].tolist(),'state_sha256':hashlib.sha256(actual['state'].tobytes()).hexdigest(),
              'hidden_cell_cache_sha256':hashlib.sha256(actual['hidden'].tobytes()+actual['cell'].tobytes()+actual['cache'].tobytes()).hexdigest(),
              'text_sha256':hashlib.sha256(actual['text']).hexdigest(),'checkpoint_record_sha256':hashes,
              'budget':plan.budget(),'actual_dispatches':device.dispatches-counters[0],
              'immutable_bytes_staged':device.constant_bytes-counters[1],'program_wait_bytes_uploaded':device.program_bytes_uploaded-counters[2],
              'wait_checker_instructions':device.wait_checked-counters[3],
              'host_encoder_reupload_or_model_compute_in_handoff':False,'elapsed_seconds':time.monotonic()-start}
    return result


def qualify(container, pcm, wave, root, output_dir):
    import torch
    torch.set_num_threads(1)
    root, out = Path(root), Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg = shared_profile_config()
    if f'{target.arch_hash(cfg):08x}' != SHARED_ARCH_HASH:
        raise RuntimeError('shared profile differs')
    binding = json.loads((root/'docs/phonon2/encoder_handoff_binding.json').read_text())
    for name, sha in binding['encoder_source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('preserved encoder source changed '+name)
    import opentpu
    runtime = Path(opentpu.__file__).resolve().parents[1]
    runtime_manifest = json.loads((runtime/'source_manifest.json').read_text())
    if runtime_manifest != binding['runtime_manifest']:
        raise RuntimeError('common runtime manifest differs')
    for name, sha in runtime_manifest['files'].items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('common runtime source changed '+name)
    table, _, policy = load_text_asset(root)
    checkpoint, device = Checkpoint(container), ProgramSlotDevice(cfg)
    report={'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'checkpoint_sha256':binding['checkpoint_sha256'],
            'profile':SHARED_PROFILE,'agfi':SHARED_AGFI,'declared_cfg_hash':SHARED_ARCH_HASH,'config':dataclasses.asdict(cfg),
            'policy':POLICY,'runtime_manifest':runtime_manifest,'encoder_binding':binding,'tokenizer_policy':policy,
            'hardware_used':False,'frontend_device_implemented':False,'model_ready':False,'audio_to_transcript_fpga_validated':False,
            'execution':'CPU ISA audit only; own native Torch reference; waveform frontend offline',
            'text_output':'Device-selected UTF-8 scatter/gather bytes and lengths; host concatenation is output I/O only',
            'cases':[]}
    def save():
        (out/'tdt_report.json').write_text(json.dumps(report,indent=2))
    cases=[('short',pcm,'short_residual'),('full_speech',wave,'full_speech_residual'),('short_after_reset',pcm,'short_residual')]
    for name,path,key in cases:
        raw=Path(path).read_bytes()
        expected_hash = '7f84e087aa28b5a33212af26a4273b4902761b2543afadc32c3236bf51ca3f74' if key=='short_residual' else '67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8'
        if hashlib.sha256(raw).hexdigest()!=expected_hash:raise ValueError('speech differs')
        frontend=frontend_torch(raw if key=='short_residual' else read_tutorial_audio(path).tobytes())
        enc,actual=run_encoder(checkpoint,device,frontend,out/'progress.json',binding['outputs'][key])
        result=run_decoder(checkpoint,device,enc,actual,table,out/'progress.json',check_waits=(name=='short'))
        report['cases'].append({'name':name,'encoder':enc,'decoder':result});save()
    first,last=report['cases'][0]['decoder'],report['cases'][-1]['decoder']
    report['reset_bitwise_passed']=all(first[k]==last[k] for k in ('state_sha256','hidden_cell_cache_sha256','text_sha256','trace'))
    report['accepted']=bool(report['reset_bitwise_passed'] and all(x['decoder']['accepted'] for x in report['cases']))
    names=['opentpu/asr/phonon2_tdt.py','tools/phonon2_tdt_reference.py','tools/phonon2_qualify_tdt.py','tools/phonon2_program_device.py',
           'docs/phonon2/token_text_vocabulary.json','docs/phonon2/tdt_semantic_policy.json','docs/phonon2/encoder_handoff_binding.json']
    report['source_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names};save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('container','pcm','wave','root','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=qualify(a.container,a.pcm,a.wave,a.root,a.output_dir)
    print(json.dumps({'accepted':r['accepted'],'reset_bitwise_passed':r['reset_bitwise_passed'],'hardware_used':False}))
