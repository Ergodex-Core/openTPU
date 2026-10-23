import numpy as np
import pytest
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec,AudioFrontImage,AudioFrontISASession,AudioFrontFrameImage
from opentpu.llm.qwen3_tts_frame import CodecFrameISASession
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from qwen3_tts_audio_reference import AudioFrontReference


class Reader:
    def __init__(self,values):self.values=values
    def tensor(self,name,shape):
        a=self.values[name]
        assert a.shape==tuple(shape)
        return a.copy()


def raw_weights(layout,seed=241):
    rng=np.random.default_rng(seed);values={}
    for view in layout.source_views.values():
        name,shape=view['source'],view['shape']
        if name in values:continue
        if name.endswith('cluster_usage'):a=rng.uniform(.1,2,shape)
        elif 'embedding_sum' in name:a=.1*rng.standard_normal(shape)
        elif name.endswith('bias'):a=.03*rng.standard_normal(shape)
        else:a=.3/np.sqrt(shape[1])*rng.standard_normal(shape)
        values[name]=a.astype(np.float32)
    return values


def tiny():return AudioFrontImage(Config(D=32,DRAM_BYTES=8<<20),AudioFrontSpec(vocab=11,code_dim=32,rvq_hidden=64,preconv_hidden=128))


def error(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)


def test_real_source_views_causal_state_chunks_reset_and_clamped_usage():
    image=tiny();raw=raw_weights(image.audio)
    raw['decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage'][0]=0
    raw['decoder.quantizer.rvq_rest.vq.layers.3._codebook.cluster_usage'][10]=-1
    weights=image.audio.weights(Reader(raw));packed=image.build(weights)
    a=AudioFrontISASession(image,packed);ref=AudioFrontReference(image.spec,raw,dtype=np.float64)
    codes=[[(frame*3+book*7)%11 for book in range(16)] for frame in range(35)]
    outputs=[]
    for ids in codes:
        latent,preconv=a.step(ids);expected=ref.step(ids)
        assert error(latent,expected['latent'])<.001
        assert error(preconv,expected['preconv'])<.001
        assert error(a.read('semantic'),expected['semantic'])<1e-6
        assert error(a.read('acoustic'),expected['acoustic'])<1e-6
        outputs.append(preconv)
    chunked=AudioFrontISASession(image,packed);got=[]
    for start,stop in [(0,1),(1,7),(7,20),(20,35)]:
        got.extend(chunked.step(ids)[1] for ids in codes[start:stop])
    np.testing.assert_array_equal(outputs,got)
    for address,size in image.state_regions().values():a.machine.slices[0].dram[address:address+size]=0x3f
    a.reset();fresh=AudioFrontISASession(image,packed)
    for ids in codes[:3]:np.testing.assert_array_equal(a.step(ids)[1],fresh.step(ids)[1])


def test_invalid_tokens_preserve_state_and_execution_failure_requires_reset(monkeypatch):
    image=tiny();session=AudioFrontISASession(image,image.build(image.audio.weights(Reader(raw_weights(image.audio)))))
    before=session.machine.slices[0].dram.copy()
    for ids in [[0]*15,[-1]+[0]*15,[11]+[0]*15,[True]+[0]*15,[1.0]*16]:
        with pytest.raises(ValueError):session.step(ids)
        np.testing.assert_array_equal(before,session.machine.slices[0].dram)
    original=session.machine.load
    def fail(programs):raise RuntimeError('injected decoder execution failure')
    monkeypatch.setattr(session.machine,'load',fail)
    with pytest.raises(RuntimeError):session.step([0]*16)
    with pytest.raises(ValueError,match='reset'):session.step([0]*16)
    monkeypatch.setattr(session.machine,'load',original);session.reset();session.step([0]*16)


def test_resident_codec_frame_to_decoder_without_host_hidden_or_code_upload():
    from test_qwen3_tts_talker import make_weights
    spec=TalkerSpec(text_vocab=17,codec_vocab=13,text_hidden=64,hidden=64,n_q=4,n_kv=2,head_dim=32,ffn=128)
    audio=AudioFrontSpec(vocab=10,code_dim=32,rvq_hidden=64,preconv_hidden=128)
    image=AudioFrontFrameImage(spec,Config(D=32,DRAM_BYTES=32<<20),64,32,layers=3,predictor_layers=5,
                               codebooks=16,audio_vocab=10,eos=12,pad_id=16,audio_spec=audio)
    raw=raw_weights(image.audio);weights=make_weights(image);weights.update(image.audio.weights(Reader(raw)))
    packed=image.build(weights);session=CodecFrameISASession(image,packed);ref=AudioFrontReference(audio,raw,dtype=np.float64)
    session.step(1,2,input_kind='role')
    for _ in range(2):
        ids=session.frame();expected=ref.step(ids)
        assert error(session.read('audio_rvq'),expected['latent'])<.001
        assert error(session.read('audio_preconv'),expected['preconv'])<.001
    session.reset();fresh=CodecFrameISASession(image,packed)
    for s in [session,fresh]:s.step(1,2,input_kind='role')
    np.testing.assert_array_equal(session.frame(),fresh.frame())
    np.testing.assert_array_equal(session.read('audio_preconv'),fresh.read('audio_preconv'))


def test_native_memory_and_programs_fit_shared_profile():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();image=AudioFrontImage(cfg)
    assert image.nbytes<64<<20 and len(image.audio.source_views)==38
    parts=image.audio.parts()
    assert len(parts)==3
    assert max(p[2]['imem_words'] for p in parts)<=cfg.IMEM_WORDS
    assert max(p[2]['tmem_peak_words'] for p in parts)<=cfg.TMEM_WORDS
    combined=AudioFrontFrameImage(TalkerSpec(),cfg)
    assert combined.nbytes<3<<30 and combined.audio.desc.ids is combined.desc.code_ids
