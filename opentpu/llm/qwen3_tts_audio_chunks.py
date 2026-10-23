"""Public 300/25 decoder chunks with resident codec-ID replay; no waveform."""
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
import numpy as np
from ..kernels.tts_audio_chunks import retain_current,restore_current,record_code,replay_code,restart_front
from .qwen3_tts_talker import _tensor
from .qwen3_tts_talker_precision import TalkerPrecisionFrameImage
from .qwen3_tts_audio_front import AudioFrontISASession
from .qwen3_tts_audio_transformer import AudioTransformerImage,AudioTransformerFrameImage
from .qwen3_tts_frame import CodecFrameISASession


@dataclass(frozen=True)
class DecoderChunkSpec:
    chunk_size: int = 300
    left_context: int = 25

    def check(self,positions):
        if type(self.chunk_size) is not int or not 1<=self.chunk_size<=300 or type(self.left_context) is not int or not 0<=self.left_context<=25:
            raise ValueError('unsupported decoder chunk controls')
        if self.chunk_size+self.left_context>positions:raise ValueError('chunk plus context exceeds qualified rotary positions')


class ResidentChunkLayout:
    def __init__(self,owner,spec):
        spec.check(owner.transformer.spec.max_positions)
        self.owner,self.spec=owner,spec
        def tensor(name,shape):
            size=4*int(np.prod(shape));address=owner.nbytes;owner.regions[name]=(address,size)
            owner.nbytes=((address+size+127)//128)*128
            return _tensor(address,shape)
        self.desc=SimpleNamespace(ids=owner.audio.desc.ids,
            current=tensor('audio.chunk.state.current',(16,)),
            codes=tensor('audio.chunk.state.codes',(max(1,spec.left_context),16)),
            history0=owner.audio.desc.history0,history1=owner.audio.desc.history1)
        if owner.nbytes>min(owner.cfg.DRAM_BYTES,2**32):raise MemoryError('resident replay exceeds device DRAM; no offload')

    def coordinates(self,frame):
        if type(frame) is not int or frame<0:raise ValueError('invalid generated frame index')
        start=frame//self.spec.chunk_size*self.spec.chunk_size
        context=min(start,self.spec.left_context)
        return start,context,context+frame-start

    def parts(self,frame):
        owner,m,s=self.owner,self.desc,self.spec
        start,context,position=self.coordinates(frame);parts=[]
        if frame==start and frame:
            parts=[owner.part('audio.chunk.save',retain_current,m=m),owner.part('audio.chunk.restart',restart_front,m=m)]
            # Decoder KV logical length restarts at zero. Every visible entry is
            # overwritten before use; stale future entries are never read.
            for j in range(context):
                parts.append(owner.part(f'audio.chunk.replay.{j}.ids',replay_code,m=m,slot=(start-context+j)%max(1,s.left_context)))
                parts.extend((f'audio.chunk.replay.{j}.'+n,p,b) for n,p,b in owner.decoder_parts(j))
            parts.append(owner.part('audio.chunk.restore',restore_current,m=m))
        parts.extend(owner.decoder_parts(position))
        if s.left_context:parts.append(owner.part('audio.chunk.record',record_code,m=m,slot=frame%s.left_context))
        return parts


class ChunkedAudioImage(AudioTransformerImage):
    def __init__(self,*args,chunk_spec=DecoderChunkSpec(),**kwargs):
        super().__init__(*args,**kwargs);self.chunk=ResidentChunkLayout(self,chunk_spec)
    def decoder_parts(self,position):return self.audio.parts()+self.transformer.parts(position)
    def state_regions(self):
        result=super().state_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.chunk.state.')});return result


class ChunkedAudioISASession(AudioFrontISASession):
    """Recorded-code CPU ISA diagnostic; never the generated-code producer."""
    def step(self,ids,observe=None):
        if self.failed:raise ValueError('reset required after decoder execution failure')
        if len(ids)!=16 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or not 0<=v<self.image.spec.vocab for v in ids):
            raise ValueError('decoder needs 16 in-range integer code IDs')
        raw=np.asarray(ids,np.float32).view(np.uint8);address=self.image.audio.desc.ids.base.const
        self.machine.slices[0].dram[address:address+len(raw)]=raw
        return self.step_resident(observe)

    def step_resident(self,observe=None):
        """Consume IDs from a validated device producer; no host ID read/upload."""
        if self.failed:raise ValueError('reset required after decoder execution failure')
        parts=self.image.chunk.parts(self.frames)
        try:
            for name,programs,_ in parts:
                self.machine.load(programs).run()
                if observe is not None:observe(name,self)
        except Exception:
            self.failed=True
            raise
        self.frames+=1
        return self.read('transformed')


class ChunkedAudioFrameImage(AudioTransformerFrameImage):
    def __init__(self,*args,chunk_spec=DecoderChunkSpec(),**kwargs):
        super().__init__(*args,**kwargs);self.chunk=ResidentChunkLayout(self,chunk_spec)
    def decoder_parts(self,position):return self.audio.parts()+self.transformer.parts(position)
    def feedback_parts(self,position,*,audio_position):
        return self.chunk.parts(audio_position)+TalkerPrecisionFrameImage.feedback_parts(self,position)
    def cache_regions(self):
        result=super().cache_regions();result.update({n:v for n,v in self.regions.items() if n.startswith('audio.chunk.state.')});return result


class ChunkedAudioFrameSession(CodecFrameISASession):
    def feedback_parts(self):return self.image.feedback_parts(self.position,audio_position=self.frames)
