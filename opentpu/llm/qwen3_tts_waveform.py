"""Resident waveform decoder with bounded device SnakeBeta; diagnostics only."""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import numpy as np
from .. import quant as Q
from ..compiler import Affine,QTensor
from ..kernels.tts_wave_ops import begin_wave,snake_beta,prepare_window,causal_convolution,transposed_convolution,finish_wave
from ..kernels.tts_waveform import SNAKE_ARGUMENT_LIMIT
from .qwen3_tts_talker import _tensor
from .qwen3_tts_audio_chunks import ChunkedAudioFrameSession
from .qwen3_tts_audio_upsample import UpsampleChunkImage,UpsampleChunkISASession,UpsampleChunkFrameImage


@dataclass(frozen=True)
class WaveformSpec:
    input_channels: int = 1024
    decoder_channels: int = 1536
    rates: tuple = (8,5,4,3)
    dilations: tuple = (1,3,9)
    batch: int = 4

    @classmethod
    def from_config(cls,c):
        d=c['decoder_config']
        if d.get('latent_dim')!=1024 or d.get('decoder_dim')!=1536 or d.get('upsample_rates')!=[8,5,4,3] or c.get('decode_upsample_rate')!=1920 or c.get('output_sample_rate')!=24000:
            raise ValueError('unsupported pinned waveform decoder config')
        return cls()

    def check(self,cfg):
        if type(self.input_channels) is not int or type(self.decoder_channels) is not int or self.input_channels<=0 or self.decoder_channels<=0 or self.decoder_channels%16:
            raise ValueError('invalid waveform widths')
        if len(self.rates)!=4 or any(type(r) is not int or r<=0 for r in self.rates) or self.dilations!=(1,3,9):raise ValueError('unsupported waveform structure')
        if self.batch!=4 or cfg.S!=1 or cfg.MCOLS<4:raise ValueError('waveform lowering needs four-row single-slice programs')


class WaveformLayout:
    def __init__(self,owner,spec=WaveformSpec()):
        spec.check(owner.cfg);self.owner,self.spec,self.sources,self.ops=owner,spec,{},[]
        cfg=owner.cfg
        if owner.upsample.final.shape!=(4,spec.input_channels):raise ValueError('resident waveform input shape mismatch')
        def alloc(name,size):
            address=owner.nbytes;owner.regions[name]=(address,size);owner.nbytes=((address+size+127)//128)*128;return address
        def tensor(name,shape,source=None,view='identity'):
            t=_tensor(alloc(name,4*math.prod(shape)),shape)
            if source:
                owner.parameters[name]=('fp32',t,shape);self.sources[name]=dict(source=source,shape=tuple(shape),view=view)
            return t
        def weight(name,n,k,source,raw_shape,tap,transpose):
            padded=((k+cfg.D-1)//cfg.D)*cfg.D
            high=QTensor.weight(alloc(name+'.data',n*Q.row_bytes(padded,'int8',cfg.D)),alloc(name+'.scale',n*Q.scale_bytes(padded,'int8',cfg.D)),n,padded,'int8',cfg.D)
            low=QTensor.weight(alloc(name+'.low.data',n*Q.row_bytes(padded,'int8',cfg.D)),alloc(name+'.low.scale',n*Q.scale_bytes(padded,'int8',cfg.D)),n,padded,'int8',cfg.D)
            owner.parameters[name]=('int8',high,(n,padded));owner.residual_parameters[name]=(high,low,(n,padded))
            self.sources[name]=dict(source=source,shape=tuple(raw_shape),view='transposed_tap' if transpose else 'conv_tap',tap=tap,padded=padded)
            return high,low
        guard=tensor('audio.wave.guard',(1,))
        def snake(prefix,input):
            rows,channels=input.shape;m=SimpleNamespace(input=input,nrows=rows,nin=channels,batch=4,guard=guard)
            m.alpha=tensor(prefix+'.exp_alpha',(channels,),prefix+'.alpha','exp_f32')
            m.inv_beta=tensor(prefix+'.inv_beta',(channels,),prefix+'.beta','inverse_exp_beta_eps_f32')
            m.output=tensor(prefix+'.output',(rows,channels));self.ops.append(('snake',prefix,m));return m.output
        def conv(prefix,input,nout,kernel,dilation=1,residual=None,rate=1,transpose=False):
            rows,nin=input.shape;padded=((nin+cfg.D-1)//cfg.D)*cfg.D
            if padded>cfg.D*cfg.ACT_BLOCKS:raise ValueError('waveform contraction exceeds ACT RAM')
            if padded not in owner.identities:owner.identities[padded]=QTensor(Affine(alloc(f'audio.wave.identity.{padded}',padded*padded)),None,(padded,padded),padded,0,cfg.D)
            history=1 if transpose else (kernel-1)*dilation
            m=SimpleNamespace(input=input,nrows=rows,nin=nin,nout=nout,padded=padded,batch=4,kernel=kernel,dilation=dilation,rate=rate,residual=residual,history_rows=history,identity=owner.identities[padded])
            if history:
                m.history=tensor('audio.wave.state.'+prefix,(history,nin));m.window=tensor(prefix+'.window',(history+rows,nin))
            else:m.history=None;m.window=input
            m.weights=[];m.low=[]
            raw_shape=(nin,nout,kernel) if transpose else (nout,nin,kernel)
            for tap in range(kernel):
                hi,lo=weight(prefix+f'.tap.{tap}',nout,nin,prefix+'.weight',raw_shape,tap,transpose);m.weights.append(hi);m.low.append(lo)
            m.bias=tensor(prefix+'.bias',(nout,),prefix+'.bias')
            m.branch=tensor(prefix+'.branch',(rows,nout)) if residual is not None else None
            m.output=tensor(prefix+'.output',(rows*rate,nout));self.ops.append(('transposed' if transpose else 'conv',prefix,m));return m.output
        x=conv('decoder.decoder.0.conv',owner.upsample.final,spec.decoder_channels,7)
        for stage,rate in enumerate(spec.rates):
            p=f'decoder.decoder.{stage+1}.block.';out=spec.decoder_channels//2**(stage+1)
            x=snake(p+'0',x);x=conv(p+'1.conv',x,out,2*rate,rate=rate,transpose=True)
            for unit,dilation in enumerate(spec.dilations):
                u=p+str(unit+2);residual=x
                x=snake(u+'.act1',x);x=conv(u+'.conv1.conv',x,out,7,dilation)
                x=snake(u+'.act2',x);x=conv(u+'.conv2.conv',x,out,1,residual=residual)
        x=snake('decoder.decoder.5',x);raw=conv('decoder.decoder.6.conv',x,1,7)
        samples=4*math.prod(spec.rates)
        self.desc=SimpleNamespace(raw=raw,waveform=tensor('audio.wave.samples',(samples,)),guard=guard,samples=samples,batch=4)
        if owner.nbytes>min(cfg.DRAM_BYTES,2**32):raise MemoryError('waveform model exceeds resident DRAM; no offload')

    def weights(self,reader):
        cache,result={},{}
        for name,v in self.sources.items():
            if v['source'] not in cache:cache[v['source']]=reader.tensor(v['source'],v['shape'])
            raw=cache[v['source']]
            if v['view'] in ['conv_tap','transposed_tap']:
                raw=raw[:,:,v['tap']]
                if v['view']=='transposed_tap':raw=raw.T
                raw=np.pad(raw,((0,0),(0,v['padded']-raw.shape[1])))
            if v['view']=='exp_f32':raw=np.exp(raw).astype(np.float32)
            if v['view']=='inverse_exp_beta_eps_f32':raw=(np.float32(1)/(np.exp(raw).astype(np.float32)+np.float32(1e-9))).astype(np.float32)
            if not np.isfinite(raw).all():raise ValueError('nonfinite folded waveform parameter')
            result[name]=np.ascontiguousarray(raw)
        return result

    def parts(self,position):
        o=self.owner;parts=[o.part('wave.begin',begin_wave,m=self.desc)]
        for kind,name,m in self.ops:
            if kind=='snake':parts.append(o.part('wave.snake.'+name,snake_beta,m=m))
            else:
                if m.history_rows:parts.append(o.part('wave.prepare.'+name,prepare_window,m=m,fresh=position==0))
                parts.append(o.part('wave.'+kind+'.'+name,transposed_convolution if kind=='transposed' else causal_convolution,m=m))
        last=o.part('wave.finish',finish_wave,m=self.desc)
        last[2].update(snake_guard_address=self.desc.guard.base.const,snake_argument_limit=SNAKE_ARGUMENT_LIMIT)
        return parts+[last]


def enforce_wave_guard(session,budget):
    if 'snake_guard_address' not in budget:return
    address=budget['snake_guard_address'];value=float(session.machine.slices[0].dram[address:address+4].view(np.float32)[0])
    if not math.isfinite(value) or value>budget['snake_argument_limit']:
        session.failed=True
        raise ValueError(f'device SnakeBeta argument {value} exceeds qualified sine domain; reset required, no fallback')


class WaveformChunkImage(UpsampleChunkImage):
    def __init__(self,*args,waveform_spec=WaveformSpec(),**kwargs):
        super().__init__(*args,**kwargs);self.wave=WaveformLayout(self,waveform_spec);self.desc.waveform=self.wave.desc.waveform
    def build(self,weights,*,max_bytes=512<<20):return super().build(weights,max_bytes=max_bytes)
    def decoder_parts(self,position):return super().decoder_parts(position)+self.wave.parts(position)
    def state_regions(self):
        result=super().state_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.wave.state.')});return result


class WaveformChunkISASession(UpsampleChunkISASession):
    def step_resident(self,observe=None):
        if self.failed:raise ValueError('reset required after decoder execution failure')
        try:
            for name,programs,budget in self.image.chunk.parts(self.frames):
                self.machine.load(programs).run();enforce_wave_guard(self,budget)
                if observe is not None:observe(name,self)
        except Exception:self.failed=True;raise
        self.frames+=1
        return self.read('waveform')


class WaveformChunkFrameImage(UpsampleChunkFrameImage):
    def __init__(self,*args,waveform_spec=WaveformSpec(),**kwargs):
        super().__init__(*args,**kwargs);self.wave=WaveformLayout(self,waveform_spec);self.desc.audio_waveform=self.wave.desc.waveform
    def decoder_parts(self,position):return super().decoder_parts(position)+self.wave.parts(position)
    def cache_regions(self):
        result=super().cache_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.wave.state.')});return result


class WaveformChunkFrameSession(ChunkedAudioFrameSession):
    def execute(self,parts,observe=None):
        if self.failed:raise ValueError('reset required after failed frame')
        for name,programs,budget in parts:
            try:self.machine.load(programs).run();enforce_wave_guard(self,budget)
            except Exception:self.failed=True;raise
            if observe is not None:observe(name,self)
