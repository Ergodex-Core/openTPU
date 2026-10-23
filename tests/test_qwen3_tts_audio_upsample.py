import math
from dataclasses import replace
import numpy as np
from opentpu import isa as I
from opentpu.isasim import Config,Machine
from opentpu.kernels.tts_audio_upsample import gelu_probe
from opentpu.llm.qwen3_tts_talker import TalkerSpec,_tensor
from opentpu.llm.qwen3_tts_audio_front import AudioFrontSpec
from opentpu.llm.qwen3_tts_audio_transformer import AudioTransformerSpec
from opentpu.llm.qwen3_tts_audio_chunks import DecoderChunkSpec,ChunkedAudioFrameSession
from opentpu.llm.qwen3_tts_audio_upsample import UpsampleChunkImage,UpsampleChunkISASession,UpsampleChunkFrameImage
from qwen3_tts_audio_reference import AudioFrontReference
from qwen3_tts_audio_transformer_reference import AudioTransformerReference
from qwen3_tts_audio_upsample_reference import AudioUpsampleReference
from test_qwen3_tts_audio_front import raw_weights,Reader,error
from test_qwen3_tts_audio_transformer import transformer_weights


def upsample_weights(layout,seed=2348):
    rng=np.random.default_rng(seed);raw={}
    for view in layout.sources.values():
        name,shape=view['source'],view['shape']
        if name in raw:continue
        if name.endswith('norm.weight'):value=1+.05*rng.standard_normal(shape)
        elif name.endswith('gamma'):value=rng.uniform(.1,.4,shape)
        elif name.endswith('bias'):value=.01*rng.standard_normal(shape)
        else:value=.3/np.sqrt(shape[1])*rng.standard_normal(shape)
        raw[name]=value.astype(np.float32)
    return raw


def tiny():
    return UpsampleChunkImage(Config(D=32,DRAM_BYTES=16<<20),AudioFrontSpec(vocab=11,code_dim=32,rvq_hidden=64,preconv_hidden=128),
        AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32),chunk_spec=DecoderChunkSpec(7,3))


def all_weights(image):
    raw=raw_weights(image.audio);raw.update(transformer_weights(image.transformer));raw.update(upsample_weights(image.upsample))
    weights=image.audio.weights(Reader(raw));weights.update(image.transformer.weights(Reader(raw)));weights.update(image.upsample.weights(Reader(raw)))
    return raw,weights


def source_chunks(image,raw,codes):
    out=[];chunk=image.chunk.spec
    for start in range(0,len(codes),chunk.chunk_size):
        context=min(start,chunk.left_context);front=AudioFrontReference(image.audio.spec,raw,dtype=np.float64)
        inputs=[front.step(ids)['preconv'] for ids in codes[start-context:min(start+chunk.chunk_size,len(codes))]]
        trans=AudioTransformerReference(image.transformer.spec,raw,dtype=np.float64).sequence(inputs)['output']
        out.extend(AudioUpsampleReference(image.upsample.channels,raw,dtype=np.float64).sequence(trans)[1]['output'][4*context:].reshape(-1,4,image.upsample.channels))
    return np.asarray(out)


def test_exact_erf_gelu_approximation_grid_and_tails_on_shared_isa():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();x=_tensor(0,(1,1024));y=_tensor(4096,(1,1024))
    b=gelu_probe.trace(cfg,0,{'input':x,'output':y});program=b.finish()
    machine=Machine(replace(cfg,DRAM_BYTES=8192),[[I.halt()]],[np.zeros(8192,np.uint8)])
    worst=0
    values=np.linspace(-16,16,4096,dtype=np.float32);values[:4]=[-1e4,1e4,0.,-0.]
    for part in values.reshape(4,1024):
        machine.slices[0].dram[:4096]=part.view(np.uint8);machine.load([program]).run()
        actual=machine.slices[0].dram[4096:8192].view(np.float32)
        expected=np.array([.5*float(v)*(1+math.erf(float(v)/math.sqrt(2))) for v in part],np.float32)
        worst=max(worst,float(np.max(np.abs(actual-expected))))
    assert worst<2e-6


def test_source_transposed_taps_causal_depthwise_chunk_reset_and_full_branch():
    image=tiny();raw,weights=all_weights(image);session=UpsampleChunkISASession(image,image.build(weights))
    codes=[[(frame*3+j*7)%11 for j in range(16)] for frame in range(12)]
    expected=source_chunks(image,raw,codes);outputs=[]
    for ids,want in zip(codes,expected):
        actual=session.step(ids);assert actual.shape==(4,128);assert error(actual,want)<.003;outputs.append(actual)
    for address,size in image.state_regions().values():session.machine.slices[0].dram[address:address+size]=0x7f
    session.reset();np.testing.assert_array_equal(outputs,[session.step(ids) for ids in codes])


def test_neural_generated_ids_feed_chunked_upsampling_without_host_features():
    from test_qwen3_tts_talker import make_weights
    spec=TalkerSpec(text_vocab=17,codec_vocab=13,text_hidden=64,hidden=64,n_q=4,n_kv=2,head_dim=32,ffn=128)
    audio=AudioFrontSpec(vocab=10,code_dim=32,rvq_hidden=64,preconv_hidden=128)
    trans=AudioTransformerSpec(latent=128,hidden=64,heads=4,head_dim=32,ffn=128,layers=2,window=4,max_positions=32)
    image=UpsampleChunkFrameImage(spec,Config(D=32,DRAM_BYTES=32<<20),64,32,layers=3,predictor_layers=5,codebooks=16,audio_vocab=10,eos=12,pad_id=16,audio_spec=audio,transformer_spec=trans,chunk_spec=DecoderChunkSpec(1,1))
    raw,own=all_weights(image);weights=make_weights(image);weights.update(own);weights['talker.codec_head.weight'][12]=0
    session=ChunkedAudioFrameSession(image,image.build(weights));codes=[];outputs=[]
    for t in [1,2,3]:session.step(t,2,input_kind='role')
    for _ in range(3):
        ids=session.frame();assert ids is not None;codes.append(ids.tolist())
        np.testing.assert_array_equal(session.read('code_ids'),ids);outputs.append(session.read('audio_upsampled').reshape(4,128))
    expected=source_chunks(image,raw,codes)
    assert max(error(a,b) for a,b in zip(outputs,expected))<.003


def test_native_upsampler_programs_and_residency_on_shared_profile():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    cfg=shared_tts_config();image=UpsampleChunkImage(cfg)
    assert image.nbytes<256<<20 and len(image.upsample.sources)==24
    for position in [0,25,324]:
        parts=image.upsample.parts(position);assert len(parts)==9
        assert max(p[2]['imem_words'] for p in parts)<=cfg.IMEM_WORDS
        assert max(p[2]['tmem_peak_words'] for p in parts)<=cfg.TMEM_WORDS
    combined=UpsampleChunkFrameImage(TalkerSpec(),cfg)
    assert combined.nbytes<3<<30
