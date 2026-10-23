import numpy as np
import pytest
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec,AudioTransformerImage,AudioTransformerISASession,AudioTransformerFrameImage,AudioTransformerFrameSession
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from test_qwen3_tts_audio_front import Reader,raw_weights,error


def transformer_weights(layout,seed=514):
    rng=np.random.default_rng(seed);weights={}
    for name,shape in layout.sources.items():
        if name.endswith('layernorm.weight') or name.endswith('norm.weight'):a=1+.05*rng.standard_normal(shape)
        elif name.endswith('.scale'):a=rng.uniform(-.3,.3,shape)
        elif name.endswith('.bias'):a=.03*rng.standard_normal(shape)
        else:a=.4/np.sqrt(shape[1])*rng.standard_normal(shape)
        weights[name]=a.astype(np.float32)
    return weights


def tiny():
    return AudioTransformerImage(Config(D=32,DRAM_BYTES=8<<20),AudioFrontSpec(vocab=11,code_dim=32,rvq_hidden=64,preconv_hidden=128),
                                 AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32))


def make(image):
    raw=raw_weights(image.audio);raw.update(transformer_weights(image.transformer))
    weights=image.audio.weights(Reader(raw));weights.update(image.transformer.weights(Reader(raw)))
    return raw,image.build(weights)


def test_layerwise_algebra_ring_wrap_and_independent_full_sequence_mask():
    image=tiny();raw,packed=make(image);session=AudioTransformerISASession(image,packed)
    front=AudioFrontReference(image.spec,raw,dtype=np.float64)
    ref=AudioTransformerReference(image.transformer.spec,raw,dtype=np.float64)
    inputs=[];traces=[]
    for t in range(21):
        ids=[(t*3+j*7)%11 for j in range(16)];features=front.step(ids)['preconv'];inputs.append(features)
        expected=ref.step(features);got={}
        def observe(name,s):
            if name.endswith('.finish'):got['layer.'+name.split('.')[2]]=s.read('transformer_hidden')
        output=session.step(ids,observe)
        for name,x in got.items():assert error(x,expected[name])<.002
        assert error(output,expected['output'])<.002
        traces.append(expected['output'])
    batch=ref.sequence(inputs)
    np.testing.assert_allclose(traces,batch['output'],rtol=1e-10,atol=1e-10)
    changed=np.array(inputs);changed[10:]+=100
    np.testing.assert_allclose(ref.sequence(changed)['output'][:10],batch['output'][:10],rtol=0,atol=0)


def test_poisoned_future_slots_reset_bounds_and_execution_failure(monkeypatch):
    image=tiny();raw,packed=make(image);s=AudioTransformerISASession(image,packed);fresh=AudioTransformerISASession(image,packed)
    for name,(address,size) in image.state_regions().items():
        if 'transformer' in name:s.machine.slices[0].dram[address:address+size]=0x7f
    np.testing.assert_array_equal(s.step([0]*16),fresh.step([0]*16))
    for address,size in image.state_regions().values():s.machine.slices[0].dram[address:address+size]=0x7f
    s.reset();fresh.reset();np.testing.assert_array_equal(s.step([1]*16),fresh.step([1]*16))
    before=s.machine.slices[0].dram.copy();s.frames=image.transformer.spec.max_positions
    with pytest.raises(ValueError,match='maximum'):s.step([0]*16)
    np.testing.assert_array_equal(s.machine.slices[0].dram,before)
    s.reset();load=s.machine.load;calls=0
    def fail(programs):
        nonlocal calls
        calls+=1
        if calls==4:raise RuntimeError('injected transformer execution failure')
        return load(programs)
    monkeypatch.setattr(s.machine,'load',fail)
    with pytest.raises(RuntimeError):s.step([0]*16)
    with pytest.raises(ValueError,match='reset'):s.step([0]*16)
    monkeypatch.setattr(s.machine,'load',load);s.reset();s.step([0]*16)


def test_source_chunk_recompute_resets_front_cache_and_rotary_position():
    image=tiny();raw,packed=make(image);s=AudioTransformerISASession(image,packed)
    codes=[[(t*3+j*7)%11 for j in range(16)] for t in range(20)]
    got=s.chunked_features(codes,chunk_size=7,left_context_size=3);expected=[]
    for start in range(0,len(codes),7):
        context=min(3,start);front=AudioFrontReference(image.spec,raw,dtype=np.float64)
        inputs=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+7,len(codes))]]
        ref=AudioTransformerReference(image.transformer.spec,raw,dtype=np.float64)
        expected.extend(ref.sequence(inputs)['output'][context:])
    assert max(error(a,b) for a,b in zip(got,expected))<.002
    np.testing.assert_array_equal(got,s.chunked_features(codes,7,3))
    for chunk,left in [(0,3),(7,-1)]:
        with pytest.raises(ValueError):s.chunked_features(codes,chunk,left)


def test_resident_generated_codec_integration_uses_frame_position_not_prefill():
    from test_qwen3_tts_talker import make_weights
    spec=TalkerSpec(text_vocab=17,codec_vocab=13,text_hidden=64,hidden=64,n_q=4,n_kv=2,head_dim=32,ffn=128)
    audio=AudioFrontSpec(vocab=10,code_dim=32,rvq_hidden=64,preconv_hidden=128)
    trans=AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32)
    image=AudioTransformerFrameImage(spec,Config(D=32,DRAM_BYTES=32<<20),64,32,layers=3,predictor_layers=5,
                                   codebooks=16,audio_vocab=10,eos=12,pad_id=16,audio_spec=audio,transformer_spec=trans)
    raw=raw_weights(image.audio);raw.update(transformer_weights(image.transformer));weights=make_weights(image)
    weights.update(image.audio.weights(Reader(raw)));weights.update(image.transformer.weights(Reader(raw)))
    packed=image.build(weights);s=AudioTransformerFrameSession(image,packed)
    for t in [1,2,3]:s.step(t,2,input_kind='role')
    front=AudioFrontReference(audio,raw,dtype=np.float64);ref=AudioTransformerReference(trans,raw,dtype=np.float64)
    outputs=[]
    for _ in range(2):
        ids=s.frame();expected=ref.step(front.step(ids)['preconv']);actual=s.read('audio_transformed')
        assert error(actual,expected['output'])<.002;outputs.append(actual)
    s.reset()
    for t in [1,2,3]:s.step(t,2,input_kind='role')
    for output in outputs:s.frame();np.testing.assert_array_equal(s.read('audio_transformed'),output)


def test_native_shared_image_programs_residency_and_last_position():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();image=AudioTransformerImage(cfg)
    assert image.nbytes<128<<20
    for position in [0,71,72,143,7999]:
        parts=image.transformer.parts(position)
        assert len(parts)==146
        assert max(p[2]['imem_words'] for p in parts)<=cfg.IMEM_WORDS
        assert max(p[2]['tmem_peak_words'] for p in parts)<=cfg.TMEM_WORDS
    for position in [-1,8000,True]:
        with pytest.raises(ValueError):image.transformer.parts(position)
    combined=AudioTransformerFrameImage(TalkerSpec(),cfg)
    assert combined.nbytes<3<<30
    assert combined.transformer.desc.input is combined.audio.desc.preconv
