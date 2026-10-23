"""Independent reviewed decoder algebra, NumPy only; never a serving backend."""
import numpy as np


class AudioFrontReference:
    def __init__(self,spec,weights,*,dtype=np.float32):
        self.spec,self.dtype=spec,dtype
        self.w={k:np.asarray(v,dtype=dtype) for k,v in weights.items()}
        self.reset()
    def reset(self):self.history=[np.zeros(self.spec.rvq_hidden,self.dtype) for _ in range(2)]
    def step(self,ids):
        vectors=[]
        for i,code in enumerate(ids):
            stem='decoder.quantizer.'+('rvq_first.vq.layers.0' if i==0 else f'rvq_rest.vq.layers.{i-1}')+'._codebook.'
            denominator=np.maximum(self.w[stem+'cluster_usage'][code],self.dtype(1e-5))
            vectors.append(self.w[stem+'embedding_sum'][code]/denominator)
        semantic=vectors[0]
        acoustic=np.zeros(self.spec.code_dim,self.dtype)
        for v in vectors[1:]:acoustic=acoustic+v
        first=self.w['decoder.quantizer.rvq_first.output_proj.weight'][:,:,0]@semantic
        rest=self.w['decoder.quantizer.rvq_rest.output_proj.weight'][:,:,0]@acoustic
        latent=first+rest
        frames=np.stack(self.history+[latent])
        preconv=np.einsum('ti,oit->o',frames,self.w['decoder.pre_conv.conv.weight'],optimize=False)+self.w['decoder.pre_conv.conv.bias']
        self.history=[self.history[1].copy(),latent.copy()]
        return {'semantic':semantic.copy(),'acoustic':acoustic.copy(),'latent':latent.copy(),'preconv':preconv.copy()}
