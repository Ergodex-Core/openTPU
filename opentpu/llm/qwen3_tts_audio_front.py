"""Decoder RVQ + causal pre-convolution, standalone or resident after codec IDs."""
from dataclasses import dataclass, replace
import math
from numbers import Integral
from types import SimpleNamespace
import numpy as np
from .. import fp32 as F, quant as Q, isa as I
from ..compiler import Affine, QTensor
from ..isasim import Machine
from ..kernels.tts_audio_front import rvq_lookup, rvq_project, decoder_preconv
from .qwen3_tts_talker import _tensor
from .qwen3_tts_talker_precision import TalkerPrecisionFrameImage


@dataclass(frozen=True)
class AudioFrontSpec:
    vocab: int = 2048
    codebooks: int = 16
    code_dim: int = 256
    rvq_hidden: int = 512
    preconv_hidden: int = 1024
    epsilon: float = 1e-5

    @classmethod
    def from_config(cls, config):
        c = config['decoder_config']
        expected = dict(codebook_size=2048, num_quantizers=16, num_semantic_quantizers=1,
                        codebook_dim=512, latent_dim=1024)
        if any(c.get(k) != v for k,v in expected.items()) or config.get('output_sample_rate') != 24000 or config.get('decode_upsample_rate') != 1920:
            raise ValueError('unsupported decoder RVQ/pre-convolution configuration')
        return cls()

    def check(self, cfg):
        if self.codebooks != 16 or any(type(v) is not int or v <= 0 for v in [self.vocab,self.code_dim,self.rvq_hidden,self.preconv_hidden]):
            raise ValueError('invalid decoder dimensions/codebook count')
        if self.vocab >= 2**24 or any(v % cfg.D for v in [self.code_dim,self.rvq_hidden]) or cfg.S != 1:
            raise ValueError('decoder dimensions do not fit the single-slice MXU')
        if self.epsilon != 1e-5:
            raise ValueError('pinned decoder codebook epsilon is 1e-5')


class AudioFrontLayout:
    def __init__(self, owner, spec, ids=None):
        spec.check(owner.cfg)
        self.owner, self.spec, self.source_views = owner, spec, {}
        cfg = owner.cfg

        def alloc(name, size):
            address = owner.nbytes
            owner.regions[name] = (address,size)
            owner.nbytes = ((address + size + 127)//128)*128
            return address

        def tensor(name, shape, source=None, source_shape=None):
            t = _tensor(alloc(name,4*math.prod(shape)),shape)
            if source:
                owner.parameters[name] = ('fp32',t,shape)
                self.source_views[name] = dict(source=source,shape=tuple(source_shape or shape),tap=None)
            return t

        def weight(name, shape, source, source_shape, tap):
            n,k = shape
            high = QTensor.weight(alloc(name+'.data',n*Q.row_bytes(k,'int8',cfg.D)),
                                  alloc(name+'.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            low = QTensor.weight(alloc(name+'.residual.data',n*Q.row_bytes(k,'int8',cfg.D)),
                                 alloc(name+'.residual.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            owner.parameters[name] = ('int8',high,shape)
            owner.residual_parameters[name] = (high,low,shape)
            self.source_views[name] = dict(source=source,shape=tuple(source_shape),tap=tap)
            return high,low

        m = SimpleNamespace(spec=spec,identities=owner.identities)
        m.ids = tensor('audio.ids',(spec.codebooks,)) if ids is None else ids
        m.embeddings, m.usages = [], []
        for book in range(spec.codebooks):
            prefix = 'decoder.quantizer.' + ('rvq_first.vq.layers.0' if book == 0 else f'rvq_rest.vq.layers.{book-1}') + '._codebook.'
            m.embeddings.append(tensor(f'audio.embedding.{book}',(spec.vocab,spec.code_dim),prefix+'embedding_sum'))
            m.usages.append(tensor(f'audio.usage.{book}',(spec.vocab,1),prefix+'cluster_usage',(spec.vocab,)))
        m.semantic = tensor('audio.semantic',(1,spec.code_dim))
        m.acoustic = tensor('audio.acoustic',(1,spec.code_dim))
        m.latent = tensor('audio.latent',(1,spec.rvq_hidden))
        m.preconv = tensor('audio.preconv',(1,spec.preconv_hidden))
        m.history0 = tensor('audio.state.history0',(1,spec.rvq_hidden))
        m.history1 = tensor('audio.state.history1',(1,spec.rvq_hidden))
        for name,branch in [('first','rvq_first'),('rest','rvq_rest')]:
            high,low=weight(f'audio.{name}.projection',(spec.rvq_hidden,spec.code_dim),
                            f'decoder.quantizer.{branch}.output_proj.weight',(spec.rvq_hidden,spec.code_dim,1),0)
            setattr(m,name,high);setattr(m,name+'_residual',low)
        m.conv,m.conv_residual=[],[]
        for tap in range(3):
            high,low=weight(f'audio.preconv.tap.{tap}',(spec.preconv_hidden,spec.rvq_hidden),
                            'decoder.pre_conv.conv.weight',(spec.preconv_hidden,spec.rvq_hidden,3),tap)
            m.conv.append(high);m.conv_residual.append(low)
        m.conv_bias=tensor('audio.preconv.bias',(spec.preconv_hidden,),'decoder.pre_conv.conv.bias')
        for width in [spec.code_dim,spec.rvq_hidden]:
            if width not in owner.identities:
                owner.identities[width]=QTensor(Affine(alloc(f'audio.identity.{width}',width*width)),None,(width,width),width,0,cfg.D)
        self.desc=m
        if owner.nbytes > min(cfg.DRAM_BYTES,2**32):
            raise MemoryError(f'decoder front needs {owner.nbytes} resident bytes; no offload')

    def weights(self, reader):
        cache,result={},{}
        for name,view in self.source_views.items():
            key=view['source']
            if key not in cache:cache[key]=reader.tensor(key,view['shape'])
            value=cache[key]
            if view['tap'] is not None:value=value[:,:,view['tap']]
            result[name]=np.ascontiguousarray(value).reshape(self.owner.parameters[name][2])
        return result

    def parts(self):
        owner,m=self.owner,self.desc
        return [owner.part('audio.rvq.lookup',rvq_lookup,m=m),owner.part('audio.rvq.project',rvq_project,m=m),
                owner.part('audio.preconv',decoder_preconv,m=m)]


class AudioFrontImage:
    def __init__(self, cfg, spec=AudioFrontSpec()):
        self.cfg,self.spec=cfg,spec
        self.nbytes,self.parameters,self.regions,self.residual_parameters,self.identities=0,{},{},{},{}
        self.audio=AudioFrontLayout(self,spec)
        self.desc=self.audio.desc

    def part(self,name,kernel,**args):
        b=kernel.trace(self.cfg,0,args);program=b.finish()
        budget=dict(instructions=len(program),imem_words=8*len(program),tmem_peak_words=b.tmem_peak,dram_bytes=self.nbytes)
        if budget['imem_words']>self.cfg.IMEM_WORDS or b.tmem_peak>self.cfg.TMEM_WORDS:
            raise MemoryError('decoder front exceeds on-chip program/scratch memory')
        return name,[program],budget

    def build(self,weights,*,max_bytes=64<<20):
        if self.nbytes>max_bytes:raise MemoryError('decoder front exceeds explicit packing budget')
        packed=np.zeros(self.nbytes,np.uint8)
        def put(address,value):
            raw=np.ascontiguousarray(value).view(np.uint8).reshape(-1);packed[address:address+len(raw)]=raw
        for name,(fmt,desc,shape) in self.parameters.items():
            a=np.asarray(weights[name],np.float32)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{name}: invalid decoder parameter')
            a=F.ftz(a)
            if fmt=='fp32':put(desc.base.const,a)
            else:
                data,scale=Q.quantize_mxu(a,'int8',self.cfg.D);put(desc.data.const,data);put(desc.scale.const,scale)
        for width,desc in self.identities.items():put(desc.data.const,np.eye(width,dtype=np.int8))
        for name,(high,low,(n,k)) in self.residual_parameters.items():
            data=packed[high.data.const:high.data.const+n*high.rs].reshape(n,high.rs)
            scale=packed[high.scale.const:high.scale.const+n*high.srs].reshape(n,high.srs)
            represented=Q.dequantize_mxu(data,scale,'int8',k,self.cfg.D).astype(np.float32)
            data,scale=Q.quantize_mxu(F.sub(np.asarray(weights[name],np.float32),represented),'int8',self.cfg.D)
            put(low.data.const,data);put(low.scale.const,scale)
        return packed

    def state_regions(self):return {n:v for n,v in self.regions.items() if n.startswith('audio.state.')}


class AudioFrontISASession:
    """Explicit CPU ISA diagnostic; not a runtime backend or fallback."""
    def __init__(self,image,packed):
        self.image,self.frames,self.failed=image,0,False
        self.machine=Machine(replace(image.cfg,DRAM_BYTES=image.nbytes),[[I.halt()]],[packed])
    def read(self,name):
        t=getattr(self.image.desc,name);return self.machine.slices[0].dram[t.base.const:t.base.const+4*math.prod(t.shape)].view(np.float32).copy().reshape(-1)
    def step(self,ids):
        if self.failed:raise ValueError('reset required after decoder execution failure')
        if len(ids)!=16 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or not 0<=v<self.image.spec.vocab for v in ids):
            raise ValueError('decoder needs 16 in-range integer code IDs')
        raw=np.asarray(ids,np.float32).view(np.uint8);address=self.image.desc.ids.base.const
        self.machine.slices[0].dram[address:address+len(raw)]=raw
        try:
            for _,programs,_ in self.image.audio.parts():self.machine.load(programs).run()
        except Exception:
            self.failed=True
            raise
        self.frames+=1
        return self.read('latent'),self.read('preconv')
    def reset(self):
        for address,size in self.image.state_regions().values():self.machine.slices[0].dram[address:address+size]=0
        self.frames,self.failed=0,False


class AudioFrontFrameImage(TalkerPrecisionFrameImage):
    def __init__(self,*args,audio_spec=AudioFrontSpec(),**kwargs):
        super().__init__(*args,**kwargs)
        if audio_spec.vocab!=self.audio_vocab:raise ValueError('talker/decoder vocabulary mismatch')
        self.audio=AudioFrontLayout(self,audio_spec,self.desc.code_ids)
        self.desc.audio_rvq=self.audio.desc.latent
        self.desc.audio_preconv=self.audio.desc.preconv
    def feedback_parts(self,position):return self.audio.parts()+super().feedback_parts(position)
    def cache_regions(self):
        result=super().cache_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.state.')});return result
