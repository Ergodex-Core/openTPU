import numpy as np
import pytest
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_waveform import WaveformSpec,WaveformChunkImage,WaveformChunkISASession,WaveformChunkFrameImage,WaveformChunkFrameSession
from opentpu.kernels.tts_waveform import SNAKE_ARGUMENT_LIMIT
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from qwen3_tts_audio_upsample_reference import AudioUpsampleReference
from qwen3_tts_waveform_reference import WaveformReference
from test_qwen3_tts_audio_upsample import all_weights
from test_qwen3_tts_audio_front import Reader,error


def wave_weights(layout,seed=2421):
    rng=np.random.default_rng(seed);raw={}
    for v in layout.sources.values():
        name,shape=v['source'],v['shape']
        if name in raw:continue
        if name.endswith(('.alpha','.beta')):a=.1*rng.standard_normal(shape)
        elif name.endswith('.bias'):a=.001*rng.standard_normal(shape)
        else:a=.2/np.sqrt(shape[1]*shape[-1])*rng.standard_normal(shape)
        raw[name]=a.astype(np.float32)
    return raw


def tiny():
    return WaveformChunkImage(Config(D=32,DRAM_BYTES=32<<20),AudioFrontSpec(vocab=11,code_dim=32,rvq_hidden=64,preconv_hidden=128),
        AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32),
        chunk_spec=DecoderChunkSpec(3,2),waveform_spec=WaveformSpec(input_channels=128,decoder_channels=64,rates=(2,2,2,2)))


def make(image):
    raw,weights=all_weights(image);raw.update(wave_weights(image.wave));weights.update(image.wave.weights(Reader(raw)))
    return raw,weights


def source(image,raw,codes):
    result=[];raw_samples=[];chunk=image.chunk.spec
    for start in range(0,len(codes),chunk.chunk_size):
        context=min(start,chunk.left_context);front=AudioFrontReference(image.audio.spec,raw,dtype=np.float64)
        inputs=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+chunk.chunk_size,len(codes))]]
        trans=AudioTransformerReference(image.transformer.spec,raw,dtype=np.float64).sequence(inputs)['output']
        up=AudioUpsampleReference(image.upsample.channels,raw,dtype=np.float64).sequence(trans)[1]['output']
        wave=WaveformReference(image.wave.spec,raw,dtype=np.float64).sequence(up);samples=image.wave.desc.samples
        result.extend(wave['waveform'][context*samples:].reshape(-1,samples));raw_samples.extend(wave['raw'][context*samples:].reshape(-1,samples))
    return np.asarray(result),np.asarray(raw_samples)


def test_full_waveform_causal_chunks_padding_transpose_and_source_scaling():
    image=tiny();raw,weights=make(image);s=WaveformChunkISASession(image,image.build(weights))
    codes=[[(t*3+j*7)%11 for j in range(16)] for t in range(5)]
    expected,unclamped=source(image,raw,codes);outputs=[]
    for frame,ids in enumerate(codes):
        actual=s.step(ids);assert actual.shape==(64,);assert error(actual,expected[frame])<.005
        t=image.wave.desc.raw;got=s.machine.slices[0].dram[t.base.const:t.base.const+4*64].view(np.float32)
        assert error(got,unclamped[frame])<.005
        np.testing.assert_array_equal(actual,np.clip(got,-1,1));outputs.append(actual)
    s.reset();np.testing.assert_array_equal(outputs,[s.step(ids) for ids in codes])
    assert len(image.wave.sources)==214  # Tiny rate-2 transposes have fewer tap views.
    assert len({v['source'] for v in image.wave.sources.values()})==118


def test_snake_guard_fails_closed_and_requires_reset():
    image=tiny();raw,weights=make(image);s=WaveformChunkISASession(image,image.build(weights))
    def inject(name,session):
        if name=='wave.begin':
            address=image.wave.desc.guard.base.const;session.machine.slices[0].dram[address:address+4]=np.array([SNAKE_ARGUMENT_LIMIT+1],np.float32).view(np.uint8)
    with pytest.raises(ValueError,match='qualified sine domain'):s.step([0]*16,inject)
    with pytest.raises(ValueError,match='reset'):s.step([0]*16)
    s.reset();s.step([0]*16)


def test_generated_codec_ids_feed_waveform_and_restore_after_short_chunk_replay():
    from test_qwen3_tts_talker import make_weights
    spec=TalkerSpec(text_vocab=17,codec_vocab=13,text_hidden=64,hidden=64,n_q=4,n_kv=2,head_dim=32,ffn=128)
    audio=AudioFrontSpec(vocab=10,code_dim=32,rvq_hidden=64,preconv_hidden=128)
    trans=AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32)
    image=WaveformChunkFrameImage(spec,Config(D=32,DRAM_BYTES=64<<20),64,32,layers=3,predictor_layers=5,codebooks=16,audio_vocab=10,eos=12,pad_id=16,audio_spec=audio,transformer_spec=trans,chunk_spec=DecoderChunkSpec(1,1),waveform_spec=WaveformSpec(input_channels=128,decoder_channels=64,rates=(2,2,2,2)))
    raw,own=make(image);weights=make_weights(image);weights.update(own);weights['talker.codec_head.weight'][12]=0
    s=WaveformChunkFrameSession(image,image.build(weights));codes=[];outputs=[]
    for t in [1,2,3]:s.step(t,2,input_kind='role')
    for _ in range(3):
        ids=s.frame();assert ids is not None;np.testing.assert_array_equal(s.read('code_ids'),ids)
        codes.append(ids.tolist());outputs.append(s.read('audio_waveform'))
    wanted,_=source(image,raw,codes);assert max(error(a,b) for a,b in zip(outputs,wanted))<.005


def test_native_waveform_resource_bounds_and_padded_contractions():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();image=WaveformChunkImage(cfg)
    assert len(image.wave.sources)==238
    assert image.nbytes<512<<20
    assert image.wave.desc.samples==1920
    assert any(m.nin==96 and m.padded==128 for kind,_,m in image.wave.ops if kind!='snake')
    assert any(m.nin==192 and m.padded==256 for kind,_,m in image.wave.ops if kind!='snake')
    for position in [0,25,324]:
        parts=image.wave.parts(position);assert len(parts)==79
        assert max(b['imem_words'] for _,_,b in parts)<=cfg.IMEM_WORDS
        assert max(b['tmem_peak_words'] for _,_,b in parts)<=cfg.TMEM_WORDS
    combined=WaveformChunkFrameImage(TalkerSpec(),cfg)
    assert combined.nbytes<cfg.DRAM_BYTES
