from pathlib import Path
import numpy as np
import pytest

from opentpu.asr.phonon2_waveform import compile_waveform
from opentpu.asr.phonon2_encoder import shared_profile_config
from tools.phonon2_program_device import ProgramSlotDevice
from tools.phonon2_waveform_audit import coefficients,run,transport_pcm,snapshots,device_move
from tools.phonon2_torch_audit import frontend_torch

ROOT=Path(__file__).resolve().parents[1]


def test_every_pcm16_value_is_exactly_converted_by_device_mm_and_argmax():
    pcm=np.arange(-32768,32768,dtype='<i2').tobytes()
    plan=compile_waveform(65536,shared_profile_config(),coefficients(ROOT))
    device=ProgramSlotDevice(plan.cfg);device.reset(plan.image.size)
    device.constants(plan.image,((0,plan.input_at),));transport_pcm(device,plan,pcm)
    for p in plan.programs[:2]:device.phase(p)
    actual=snapshots(device,plan)['wave'][1:65537]
    expected=np.frombuffer(pcm,'<i2').astype(np.float32)/np.float32(32768)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('name',['silence_min','silence_100ms','boundary_impulses','chirp_partial_hop','noise_exact_hop','noise_partial_hop'])
def test_retained_source_frontend_vectors(name):
    data=np.load(ROOT/'docs/phonon2/frontend_vectors.npz',allow_pickle=False)
    pcm=data[name+'.pcm_s16le'].tobytes()
    plan=compile_waveform(len(pcm)//2,shared_profile_config(),coefficients(ROOT))
    actual=run(ProgramSlotDevice(plan.cfg),plan,pcm,check=True)
    expected=frontend_torch(pcm)
    np.testing.assert_array_equal(actual['wave'][1:plan.samples+1],np.frombuffer(pcm,'<i2').astype(np.float32)/32768)
    np.testing.assert_array_equal(actual['padded_preemphasis'][256:256+plan.samples],expected['preemphasis'])
    np.testing.assert_allclose(actual['log_mel'],expected['log_mel'],atol=5e-4,rtol=2e-5)
    delta=np.max(np.abs(actual['features']-expected['features']))
    print(name,'feature_max_abs',delta,'first',actual['features'][0,0],expected['features'][0,0])
    assert np.isfinite(actual['features']).all()
    np.testing.assert_array_equal(actual['features'][plan.valid:],0)
    assert delta<1e-4


@pytest.mark.parametrize('samples,kwargs',[(319,{}),(480001,{}),(320,{'sample_rate_hz':8000}),(320,{'channels':2}),(320,{'dither':1e-5})])
def test_frontend_duration_and_policy_guards(samples,kwargs):
    with pytest.raises(ValueError):compile_waveform(samples,shared_profile_config(),coefficients(ROOT),**kwargs)


def test_frontend_reset_after_nonzero_pcm_is_bitwise():
    coeff=coefficients(ROOT);cfg=shared_profile_config();device=ProgramSlotDevice(cfg)
    first=compile_waveform(1617,cfg,coeff);second=compile_waveform(320,cfg,coeff)
    run(device,first,np.random.default_rng(31).integers(-32768,32768,1617,dtype=np.int16).tobytes())
    reused=run(device,second,bytes(640));fresh=run(ProgramSlotDevice(cfg),second,bytes(640))
    for name in reused:np.testing.assert_array_equal(reused[name],fresh[name])


def test_maximum_context_frontend_programs_fit_shared_imem_and_tmem():
    plan=compile_waveform(480000,shared_profile_config(),coefficients(ROOT))
    assert plan.budget()['max_instruction_words']<=32768
    assert plan.budget()['max_tmem_live_words']<=65536
    assert (plan.frames,plan.valid)==(3001,3000)


@pytest.mark.parametrize('samples,first',[(4960,.15975946),(5280,-.15978564),(112960,-.1601643),(480000,-.27612054)])
def test_normalization_silence_reduction_boundaries_match_retained_torch_arm_order(samples,first):
    plan=compile_waveform(samples,shared_profile_config(),coefficients(ROOT));device=ProgramSlotDevice(plan.cfg);device.reset(plan.image.size)
    x=np.full((128,plan.frames),np.float32(-24*np.log(2)),np.float32);x[:,-1]=0
    s=plan.checkpoints['transposed'];device.memory[s['offset']:s['offset']+x.nbytes]=x.view(np.uint8).ravel()
    device.phase(plan.programs[-1])
    actual=snapshots(device,plan)['features'];expected=np.full((plan.frames,128),np.float32(first));expected[-1]=0
    np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=5e-6)


@pytest.mark.parametrize('source,destination',[(0,128),(128,0),(0,2048)])
def test_device_handoff_overlap_both_directions(source,destination):
    device=ProgramSlotDevice(shared_profile_config());device.reset(4096)
    data=np.arange(1024,dtype=np.float32);device.memory[:data.nbytes]=data.view(np.uint8)
    expected=data.copy();expected[destination//4:destination//4+64]=data[source//4:source//4+64]
    device_move(device,source,destination,8,8,check=True)
    np.testing.assert_array_equal(device.memory[:data.nbytes].view(np.float32),expected)
