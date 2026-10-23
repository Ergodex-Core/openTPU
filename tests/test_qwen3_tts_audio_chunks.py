import numpy as np
import pytest
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec,ChunkedAudioImage,ChunkedAudioISASession,ChunkedAudioFrameImage,ChunkedAudioFrameSession
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec,AudioTransformerFrameImage,AudioTransformerFrameSession
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from test_qwen3_tts_audio_transformer import make
from test_qwen3_tts_audio_front import raw_weights,Reader,error
from test_qwen3_tts_audio_transformer import transformer_weights


def tiny(chunk=7,left=3):
    return ChunkedAudioImage(Config(D=32,DRAM_BYTES=8<<20),AudioFrontSpec(vocab=11,code_dim=32,rvq_hidden=64,preconv_hidden=128),
                            AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32),chunk_spec=DecoderChunkSpec(chunk,left))


def source_chunks(image,raw,codes):
    result=[];c=image.chunk.spec
    for start in range(0,len(codes),c.chunk_size):
        context=min(start,c.left_context);front=AudioFrontReference(image.spec,raw,dtype=np.float64)
        inputs=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+c.chunk_size,len(codes))]]
        ref=AudioTransformerReference(image.transformer.spec,raw,dtype=np.float64)
        result.extend(ref.sequence(inputs)['output'][context:])
    return result


@pytest.mark.parametrize('chunk,left',[(7,3),(3,5),(7,0)])
def test_resident_ring_replay_exact_source_chunks_and_current_id_restore(chunk,left):
    image=tiny(chunk,left);raw,packed=make(image);s=ChunkedAudioISASession(image,packed)
    codes=[[(t*3+j*7)%11 for j in range(16)] for t in range(24)];got=[];seen=[]
    for t,ids in enumerate(codes):
        def observe(name,session):
            if name.startswith('audio.chunk.replay.') and name.endswith('.ids'):
                j=int(name.split('.')[3]);_,context,_=image.chunk.coordinates(t)
                np.testing.assert_array_equal(session.read('ids'),codes[t-context+j]);seen.append((t,j))
            if name=='audio.chunk.restore':np.testing.assert_array_equal(session.read('ids'),ids)
        got.append(s.step(ids,observe))
        np.testing.assert_array_equal(s.read('ids'),ids)
    assert max(error(a,b) for a,b in zip(got,source_chunks(image,raw,codes)))<.002
    assert len(seen)==sum(min(t,left) for t in range(chunk,len(codes),chunk))
    for address,size in image.state_regions().values():s.machine.slices[0].dram[address:address+size]=0x7f
    s.reset();replay=[s.step(ids) for ids in codes];np.testing.assert_array_equal(got,replay)


def test_replay_failure_latches_until_reset(monkeypatch):
    image=tiny(3,2);raw,packed=make(image);s=ChunkedAudioISASession(image,packed)
    for _ in range(3):s.step([0]*16)
    original=s.machine.load;count=0
    def fail(programs):
        nonlocal count
        count+=1
        if count==4:raise RuntimeError('injected replay failure')
        return original(programs)
    monkeypatch.setattr(s.machine,'load',fail)
    with pytest.raises(RuntimeError):s.step([1]*16)
    with pytest.raises(ValueError,match='reset'):s.step([1]*16)
    monkeypatch.setattr(s.machine,'load',original);s.reset();fresh=ChunkedAudioISASession(image,packed)
    np.testing.assert_array_equal(s.step([1]*16),fresh.step([1]*16))


def test_generated_frames_replay_without_altering_main_ids_or_position():
    from test_qwen3_tts_talker import make_weights
    spec=TalkerSpec(text_vocab=17,codec_vocab=13,text_hidden=64,hidden=64,n_q=4,n_kv=2,head_dim=32,ffn=128)
    audio=AudioFrontSpec(vocab=10,code_dim=32,rvq_hidden=64,preconv_hidden=128)
    trans=AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32)
    args=(spec,Config(D=32,DRAM_BYTES=32<<20),64,32)
    kwargs=dict(layers=3,predictor_layers=5,codebooks=16,audio_vocab=10,eos=12,pad_id=16,audio_spec=audio,transformer_spec=trans)
    image=ChunkedAudioFrameImage(*args,chunk_spec=DecoderChunkSpec(3,2),**kwargs)
    baseline=AudioTransformerFrameImage(*args,**kwargs)
    raw=raw_weights(image.audio);raw.update(transformer_weights(image.transformer));weights=make_weights(image)
    weights['talker.codec_head.weight'][12]=0
    weights.update(image.audio.weights(Reader(raw)));weights.update(image.transformer.weights(Reader(raw)))
    s=ChunkedAudioFrameSession(image,image.build(weights));old=AudioTransformerFrameSession(baseline,baseline.build(weights))
    for t in [1,2,3]:
        for session in [s,old]:session.step(t,2,input_kind='role')
    codes=[];outputs=[]
    for _ in range(9):
        ids=s.frame();other=old.frame();assert ids is not None
        np.testing.assert_array_equal(ids,other);np.testing.assert_array_equal(s.read('code_ids'),ids)
        assert s.position==old.position and s.frames==old.frames
        codes.append(ids.tolist());outputs.append(s.read('audio_transformed'))
    # Reference helper only needs front and transformer specs plus chunk controls.
    view=type('View',(),{'spec':audio,'transformer':image.transformer,'chunk':image.chunk})()
    assert max(error(x,y) for x,y in zip(outputs,source_chunks(view,raw,codes)))<.002


def test_native_public_boundaries_and_resource_limits():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();image=ChunkedAudioImage(cfg)
    assert image.chunk.coordinates(299)==(0,0,299)
    assert image.chunk.coordinates(300)==(300,25,25)
    assert image.chunk.coordinates(599)==(300,25,324)
    assert image.chunk.coordinates(600)==(600,25,25)
    parts=image.chunk.parts(300);assert len(parts)==3903
    assert max(p[2]['imem_words'] for p in parts)<=cfg.IMEM_WORDS
    assert max(p[2]['tmem_peak_words'] for p in parts)<=cfg.TMEM_WORDS
    assert image.nbytes-107370624==1792
    for chunk,left in [(301,25),(300,26),(0,0),(True,1)]:
        with pytest.raises(ValueError):DecoderChunkSpec(chunk,left).check(8000)
