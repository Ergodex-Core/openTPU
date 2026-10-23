"""Resident 2x2 upsampling/ConvNeXt after the public-chunk decoder transformer."""
import math
from types import SimpleNamespace
import numpy as np
from .. import quant as Q
from ..compiler import Affine,QTensor
from ..kernels.mlp import down_chunk
from ..kernels.tts_audio_upsample import upsample_pair,convnext_row
from .qwen3_tts_talker import _tensor
from .qwen3_tts_audio_chunks import ChunkedAudioImage,ChunkedAudioISASession,ChunkedAudioFrameImage


class AudioUpsampleLayout:
    def __init__(self,owner):
        self.owner,self.sources=owner,{}
        cfg=owner.cfg;channels=owner.transformer.spec.latent
        if channels%cfg.D or cfg.S!=1:raise ValueError('unsupported upsampler dimensions')
        self.channels=channels
        def alloc(name,size):
            address=owner.nbytes;owner.regions[name]=(address,size);owner.nbytes=((address+size+127)//128)*128;return address
        def tensor(name,shape,source=None,source_shape=None,view='identity'):
            t=_tensor(alloc(name,4*math.prod(shape)),shape)
            if source:
                owner.parameters[name]=('fp32',t,shape);self.sources[name]=dict(source=source,shape=tuple(source_shape or shape),view=view)
            return t
        def weight(name,shape,source,source_shape=None,tap=None):
            n,k=shape
            hi=QTensor.weight(alloc(name+'.data',n*Q.row_bytes(k,'int8',cfg.D)),alloc(name+'.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            lo=QTensor.weight(alloc(name+'.low.data',n*Q.row_bytes(k,'int8',cfg.D)),alloc(name+'.low.scale',n*Q.scale_bytes(k,'int8',cfg.D)),n,k,'int8',cfg.D)
            owner.parameters[name]=('int8',hi,shape);owner.residual_parameters[name]=(hi,lo,shape)
            self.sources[name]=dict(source=source,shape=tuple(source_shape or shape),view='transposed_conv_tap' if tap is not None else 'identity',tap=tap)
            return hi,lo
        chunk=down_chunk(4*channels,cfg.D,'int8')
        for width in [channels,chunk]:
            if width not in owner.identities:owner.identities[width]=QTensor(Affine(alloc(f'audio.upsample.identity.{width}',width*width)),None,(width,width),width,0,cfg.D)
        self.final=tensor('audio.upsample.final',(4,channels));self.blocks=[]
        input=owner.transformer.desc.transformed
        for index in range(2):
            p=f'decoder.upsample.{index}.';name=f'audio.upsample.{index}.'
            m=SimpleNamespace(channels=channels,identities=owner.identities,input=input,final=self.final)
            m.up=[];m.up_low=[]
            for phase in range(2):
                hi,lo=weight(name+f'up.tap.{phase}',(channels,channels),p+'0.conv.weight',(channels,channels,2),phase)
                m.up.append(hi);m.up_low.append(lo)
            m.up_bias=tensor(name+'up.bias',(channels,),p+'0.conv.bias')
            m.up_output=tensor(name+'up.output',(2,channels))
            m.history=tensor(f'audio.upsample.state.{index}.history',(6,channels))
            m.dw=tensor(name+'dw.weight',(7,channels),p+'1.dwconv.conv.weight',(channels,1,7),'depthwise_taps')
            m.dw_bias=tensor(name+'dw.bias',(channels,),p+'1.dwconv.conv.bias')
            m.norm_weight=tensor(name+'norm.weight',(channels,),p+'1.norm.weight')
            m.norm_bias=tensor(name+'norm.bias',(channels,),p+'1.norm.bias')
            m.gamma=tensor(name+'gamma',(channels,),p+'1.gamma')
            m.fc1,m.fc1_low=weight(name+'fc1.weight',(4*channels,channels),p+'1.pwconv1.weight')
            m.fc1_bias=tensor(name+'fc1.bias',(4*channels,),p+'1.pwconv1.bias')
            m.fc2,m.fc2_low=weight(name+'fc2.weight',(channels,4*channels),p+'1.pwconv2.weight')
            m.fc2_bias=tensor(name+'fc2.bias',(channels,),p+'1.pwconv2.bias')
            m.depthwise=tensor(name+'depthwise',(1,channels));m.normalized=tensor(name+'normalized',(1,channels))
            m.hidden=tensor(name+'hidden',(1,4*channels));m.activation=tensor(name+'activation',(1,4*channels))
            m.projected=tensor(name+'projected',(1,channels));m.output=tensor(name+'output',(1,channels))
            self.blocks.append(m);input=m.output
        if owner.nbytes>min(cfg.DRAM_BYTES,2**32):raise MemoryError('upsampler exceeds resident DRAM; no offload')

    def weights(self,reader):
        cache,result={},{}
        for name,v in self.sources.items():
            if v['source'] not in cache:cache[v['source']]=reader.tensor(v['source'],v['shape'])
            raw=cache[v['source']]
            if v['view']=='transposed_conv_tap':raw=raw[:,:,v['tap']].T
            if v['view']=='depthwise_taps':raw=raw[:,0,:].T
            result[name]=np.ascontiguousarray(raw)
        return result

    def parts(self,position):
        owner=self.owner;a,b=self.blocks
        parts=[owner.part('audio.upsample.0.pair',upsample_pair,m=a)]
        for phase in range(2):
            parts.append(owner.part(f'audio.upsample.0.row.{phase}',convnext_row,m=a,phase=phase,row_position=2*position+phase,output_row=-1))
            parts.append(owner.part(f'audio.upsample.1.pair.{phase}',upsample_pair,m=b))
            for last in range(2):
                row=2*phase+last
                parts.append(owner.part(f'audio.upsample.1.row.{row}',convnext_row,m=b,phase=last,row_position=4*position+row,output_row=row))
        return parts


class UpsampleChunkImage(ChunkedAudioImage):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.upsample=AudioUpsampleLayout(self);self.desc.upsampled=self.upsample.final
    def build(self,weights,*,max_bytes=256<<20):return super().build(weights,max_bytes=max_bytes)
    def decoder_parts(self,position):return super().decoder_parts(position)+self.upsample.parts(position)
    def state_regions(self):
        result=super().state_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.upsample.state.')});return result


class UpsampleChunkISASession(ChunkedAudioISASession):
    def step_resident(self,observe=None):
        super().step_resident(observe)
        return self.read('upsampled').reshape(4,self.image.upsample.channels)


class UpsampleChunkFrameImage(ChunkedAudioFrameImage):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.upsample=AudioUpsampleLayout(self);self.desc.audio_upsampled=self.upsample.final
    def decoder_parts(self,position):return super().decoder_parts(position)+self.upsample.parts(position)
    def cache_regions(self):
        result=super().cache_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.upsample.state.')});return result
