"""Resident tokenizer transformer; diagnostics only, no waveform or fallback."""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import numpy as np
from .. import quant as Q
from ..compiler import Affine, QTensor
from ..kernels.mlp import down_chunk
from ..kernels.tts_audio_transformer import decoder_input, decoder_qkv, decoder_attention_head, decoder_finish, decoder_output
from .qwen3_tts_audio_front import AudioFrontImage, AudioFrontISASession, AudioFrontFrameImage
from .qwen3_tts_talker import _tensor
from .qwen3_tts_frame import CodecFrameISASession


@dataclass(frozen=True)
class AudioTransformerSpec:
    latent: int = 1024
    hidden: int = 512
    heads: int = 16
    head_dim: int = 64
    ffn: int = 1024
    layers: int = 8
    window: int = 72
    max_positions: int = 8000
    eps: float = 1e-5
    theta: float = 10000.

    @classmethod
    def from_config(cls, config):
        c = config['decoder_config']
        expected = dict(latent_dim=1024, hidden_size=512, num_attention_heads=16,
                        num_key_value_heads=16, head_dim=64, intermediate_size=1024,
                        num_hidden_layers=8, sliding_window=72, max_position_embeddings=8000,
                        rms_norm_eps=1e-5, rope_theta=10000, attention_bias=False,
                        hidden_act='silu', attention_dropout=0.)
        if any(c.get(k) != v for k,v in expected.items()) or c.get('rope_scaling') is not None:
            raise ValueError('unsupported pinned tokenizer transformer config')
        return cls()

    def check(self, cfg):
        if any(type(v) is not int or v <= 0 for v in [self.latent,self.hidden,self.heads,self.head_dim,self.ffn,self.layers,self.window,self.max_positions]):
            raise ValueError('positive integer decoder dimensions required')
        if cfg.S != 1 or self.head_dim % 2 or any(v % cfg.D for v in [self.latent,self.hidden,self.ffn,self.heads*self.head_dim]):
            raise ValueError('decoder affine dimensions must fit the single-slice MXU')
        if self.layers > 8 or self.window > 72 or self.max_positions > 8000 or self.window > self.max_positions:
            raise ValueError('decoder exceeds qualified layer/window/position bounds')
        if max(self.latent,self.hidden,self.ffn,self.heads*self.head_dim) > cfg.D*cfg.ACT_BLOCKS:
            raise ValueError('decoder activation exceeds ACT RAM')
        if not all(math.isfinite(v) and v > 0 for v in [self.eps,self.theta]):
            raise ValueError('invalid decoder normalization/rotary constants')


class AudioTransformerLayout:
    def __init__(self, owner, spec, input_tensor):
        spec.check(owner.cfg)
        if input_tensor.shape != (1,spec.latent):raise ValueError('decoder input shape mismatch')
        self.owner,self.spec,self.sources=owner,spec,{}
        cfg=owner.cfg
        def alloc(name,size):
            address=owner.nbytes;owner.regions[name]=(address,size)
            owner.nbytes=((address+size+127)//128)*128
            return address
        def tensor(name,shape,source=None):
            t=_tensor(alloc(name,4*math.prod(shape)),shape)
            if source:
                owner.parameters[source]=('fp32',t,shape);self.sources[source]=shape
            return t
        def weight(source,shape):
            n,k=shape
            hi=QTensor.weight(alloc(source+'.data',n*Q.row_bytes(k,'int8',cfg.D)),alloc(source+'.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            lo=QTensor.weight(alloc(source+'.low.data',n*Q.row_bytes(k,'int8',cfg.D)),alloc(source+'.low.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            owner.parameters[source]=('int8',hi,shape);owner.residual_parameters[source]=(hi,lo,shape);self.sources[source]=shape
            return hi,lo
        widths={spec.latent,spec.hidden,spec.heads*spec.head_dim,down_chunk(spec.ffn,cfg.D,'int8')}
        for width in sorted(widths):
            if width not in owner.identities:owner.identities[width]=QTensor(Affine(alloc(f'audio.transformer.identity.{width}',width*width)),None,(width,width),width,0,cfg.D)
        p='decoder.pre_transformer.'
        m=SimpleNamespace(spec=spec,identities=owner.identities,input=input_tensor)
        m.output=tensor('audio.transformer.hidden',(1,spec.hidden))
        m.attention_residual=tensor('audio.transformer.attention_residual',(1,spec.hidden))
        m.normalized=tensor('audio.transformer.normalized',(1,spec.hidden))
        m.transformed=tensor('audio.transformer.transformed',(1,spec.latent))
        m.query=tensor('audio.transformer.query',(spec.heads,spec.head_dim))
        m.context=tensor('audio.transformer.context',(spec.heads,spec.head_dim))
        m.cosine=tensor('audio.transformer.cosine',(spec.max_positions,spec.head_dim//2))
        m.sine=tensor('audio.transformer.sine',(spec.max_positions,spec.head_dim//2))
        m.input_proj,m.input_proj_residual=weight(p+'input_proj.weight',(spec.hidden,spec.latent))
        m.input_bias=tensor(p+'input_proj.bias',(spec.hidden,),p+'input_proj.bias')
        m.output_proj,m.output_proj_residual=weight(p+'output_proj.weight',(spec.latent,spec.hidden))
        m.output_bias=tensor(p+'output_proj.bias',(spec.latent,),p+'output_proj.bias')
        m.final_norm=tensor(p+'norm.weight',(spec.hidden,),p+'norm.weight')
        self.layers=[]
        for i in range(spec.layers):
            layer=SimpleNamespace(**vars(m));prefix=p+f'layers.{i}.'
            for attr,suffix in [('input_norm','input_layernorm.weight'),('post_norm','post_attention_layernorm.weight'),('attention_scale','self_attn_layer_scale.scale'),('mlp_scale','mlp_layer_scale.scale')]:
                setattr(layer,attr,tensor(prefix+suffix,(spec.hidden,),prefix+suffix))
            for attr,suffix,shape in [('wq','self_attn.q_proj.weight',(spec.heads*spec.head_dim,spec.hidden)),('wk','self_attn.k_proj.weight',(spec.heads*spec.head_dim,spec.hidden)),('wv','self_attn.v_proj.weight',(spec.heads*spec.head_dim,spec.hidden)),('wo','self_attn.o_proj.weight',(spec.hidden,spec.heads*spec.head_dim)),('gate','mlp.gate_proj.weight',(spec.ffn,spec.hidden)),('up','mlp.up_proj.weight',(spec.ffn,spec.hidden)),('down','mlp.down_proj.weight',(spec.hidden,spec.ffn))]:
                hi,lo=weight(prefix+suffix,shape);setattr(layer,attr,hi);setattr(layer,attr+'_residual',lo)
            layer.keys=[];layer.values=[]
            for head in range(spec.heads):
                for attr,tag in [('keys','k'),('values','v')]:
                    getattr(layer,attr).append(tensor(f'audio.transformer.state.layer.{i}.{head}.{tag}',(spec.window,spec.head_dim)))
            self.layers.append(layer)
        self.desc=m
        if owner.nbytes > min(cfg.DRAM_BYTES,2**32):raise MemoryError('resident decoder exceeds shared DRAM; no offload')

    def weights(self,reader):return {name:reader.tensor(name,shape) for name,shape in self.sources.items()}

    def constants(self,packed):
        s,m=self.spec,self.desc
        inv=np.float32(1)/np.power(np.float32(s.theta),np.arange(0,s.head_dim,2,dtype=np.float32)/s.head_dim)
        angles=np.arange(s.max_positions,dtype=np.float32)[:,None]*inv[None,:]
        for t,a in [(m.cosine,np.cos(angles)),(m.sine,np.sin(angles))]:
            raw=np.ascontiguousarray(a,dtype=np.float32).view(np.uint8).reshape(-1);packed[t.base.const:t.base.const+len(raw)]=raw
        return packed

    def parts(self,position):
        if type(position) is not int or not 0<=position<self.spec.max_positions:raise ValueError('decoder position exceeds qualified rotary table; reset or reject, no wrap')
        o,m=self.owner,self.desc
        parts=[o.part('audio.transformer.input',decoder_input,m=m)]
        for i,layer in enumerate(self.layers):
            parts.append(o.part(f'audio.transformer.{i}.qkv',decoder_qkv,m=layer,position=position))
            parts.extend(o.part(f'audio.transformer.{i}.head.{head}',decoder_attention_head,m=layer,position=position,head=head) for head in range(self.spec.heads))
            parts.append(o.part(f'audio.transformer.{i}.finish',decoder_finish,m=layer))
        return parts+[o.part('audio.transformer.output',decoder_output,m=m)]


class AudioTransformerImage(AudioFrontImage):
    def __init__(self,cfg,audio_spec=None,transformer_spec=AudioTransformerSpec()):
        super().__init__(cfg,**({} if audio_spec is None else {'spec':audio_spec}))
        self.transformer=AudioTransformerLayout(self,transformer_spec,self.desc.preconv)
        self.desc.transformed=self.transformer.desc.transformed
        self.desc.transformer_hidden=self.transformer.desc.output
    def build(self,weights,*,max_bytes=128<<20):return self.transformer.constants(super().build(weights,max_bytes=max_bytes))
    def state_regions(self):return {n:v for n,v in self.regions.items() if n.startswith(('audio.state.','audio.transformer.state.'))}


class AudioTransformerISASession(AudioFrontISASession):
    def step(self,ids,observe=None):
        position=self.frames
        if position>=self.image.transformer.spec.max_positions:raise ValueError('decoder maximum position reached')
        parts=self.image.transformer.parts(position)
        super().step(ids)
        try:
            for name,programs,_ in parts:
                self.machine.load(programs).run()
                if observe is not None:observe(name,self)
        except Exception:
            self.failed=True
            raise
        return self.read('transformed')

    def chunked_features(self,codes,chunk_size=300,left_context_size=25):
        if type(chunk_size) is not int or chunk_size<=0 or type(left_context_size) is not int or left_context_size<0:
            raise ValueError('invalid source chunk controls')
        result=[]
        for start in range(0,len(codes),chunk_size):
            end=min(start+chunk_size,len(codes));context=min(start,left_context_size)
            if end-start+context>self.image.transformer.spec.max_positions:raise ValueError('source chunk exceeds qualified positions')
            self.reset()
            for index in range(start-context,end):
                output=self.step(codes[index])
                if index>=start:result.append(output)
        return np.asarray(result)


class AudioTransformerFrameImage(AudioFrontFrameImage):
    def __init__(self,*args,transformer_spec=AudioTransformerSpec(),**kwargs):
        super().__init__(*args,**kwargs)
        self.transformer=AudioTransformerLayout(self,transformer_spec,self.audio.desc.preconv)
        self.desc.audio_transformed=self.transformer.desc.transformed
        self.desc.audio_transformer_hidden=self.transformer.desc.output
    def build(self,weights,**kwargs):return self.transformer.constants(super().build(weights,**kwargs))
    def feedback_parts(self,position,*,audio_position):
        # Decoder time counts codec frames, independently of text prefill length.
        return self.audio.parts()+self.transformer.parts(audio_position)+super(AudioFrontFrameImage,self).feedback_parts(position)
    def cache_regions(self):
        result=super().cache_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.transformer.state.')});return result


class AudioTransformerFrameSession(CodecFrameISASession):
    def feedback_parts(self):return self.image.feedback_parts(self.position,audio_position=self.frames)
