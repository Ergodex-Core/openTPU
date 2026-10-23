"""Independent source-algebra decoder references; never a serving backend."""
import numpy as np


class AudioTransformerReference:
    def __init__(self,spec,weights,*,dtype=np.float32):
        self.spec,self.dtype=spec,dtype
        self.w={k:np.asarray(v,dtype=dtype) for k,v in weights.items()}
        # Upstream explicitly builds rotary frequencies/trig in FP32 even when
        # other arithmetic has a different dtype. This oracle uses that contract.
        inv=np.float32(1)/np.power(np.float32(spec.theta),np.arange(0,spec.head_dim,2,dtype=np.float32)/spec.head_dim)
        angles=np.arange(spec.max_positions,dtype=np.float32)[:,None]*inv[None,:]
        self.cos=np.cos(angles).astype(dtype);self.sin=np.sin(angles).astype(dtype)
        self.reset()

    def reset(self):
        self.position=0
        self.k=[[] for _ in range(self.spec.layers)];self.v=[[] for _ in range(self.spec.layers)]

    def norm(self,x,weight):
        return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+self.dtype(self.spec.eps))*weight

    def rotate(self,x,positions):
        half=self.spec.head_dim//2
        c=self.cos[positions];s=self.sin[positions]
        if x.ndim==3:c=c[:,None,:];s=s[:,None,:]
        return np.concatenate([x[...,:half]*c-x[...,half:]*s,x[...,half:]*c+x[...,:half]*s],axis=-1)

    def affine(self,x,name):return x@self.w[name+'.weight'].T+self.w[name+'.bias']

    def mlp(self,x,p):
        gate=x@self.w[p+'mlp.gate_proj.weight'].T
        up=x@self.w[p+'mlp.up_proj.weight'].T
        return (gate/(1+np.exp(-gate))*up)@self.w[p+'mlp.down_proj.weight'].T

    def step(self,input):
        s=self.spec;p='decoder.pre_transformer.';position=self.position
        if position>=s.max_positions:raise ValueError('reference position out of range')
        x=self.affine(np.asarray(input,dtype=self.dtype),p+'input_proj');trace={'input':x.copy()}
        for i in range(s.layers):
            t=p+f'layers.{i}.';xn=self.norm(x,self.w[t+'input_layernorm.weight'])
            q=self.rotate((xn@self.w[t+'self_attn.q_proj.weight'].T).reshape(s.heads,s.head_dim),position)
            k=self.rotate((xn@self.w[t+'self_attn.k_proj.weight'].T).reshape(s.heads,s.head_dim),position)
            v=(xn@self.w[t+'self_attn.v_proj.weight'].T).reshape(s.heads,s.head_dim)
            self.k[i].append(k);self.v[i].append(v)
            self.k[i]=self.k[i][-s.window:];self.v[i]=self.v[i][-s.window:]
            keys=np.stack(self.k[i]);values=np.stack(self.v[i])
            scores=np.einsum('hd,thd->ht',q,keys,optimize=False)/self.dtype(np.sqrt(s.head_dim))
            prob=np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob=prob/np.sum(prob,axis=-1,keepdims=True)
            context=np.einsum('ht,thd->hd',prob,values,optimize=False).reshape(-1)
            residual=x+(context@self.w[t+'self_attn.o_proj.weight'].T)*self.w[t+'self_attn_layer_scale.scale']
            x=residual+self.mlp(self.norm(residual,self.w[t+'post_attention_layernorm.weight']),t)*self.w[t+'mlp_layer_scale.scale']
            trace[f'layer.{i}']=x.copy()
        normalized=self.norm(x,self.w[p+'norm.weight']);out=self.affine(normalized,p+'output_proj')
        trace['normalized']=normalized.copy();trace['output']=out.copy();self.position+=1
        return trace

    def sequence(self,inputs):
        """Full-sequence source mask; does not call or mutate incremental cache."""
        s=self.spec;p='decoder.pre_transformer.';inputs=np.asarray(inputs,dtype=self.dtype)
        n=len(inputs)
        if n>s.max_positions:raise ValueError('reference sequence out of range')
        x=self.affine(inputs,p+'input_proj');trace={'input':x.copy()};positions=np.arange(n)
        allowed=(positions[None,:]<=positions[:,None])&(positions[None,:]>positions[:,None]-s.window)
        for i in range(s.layers):
            t=p+f'layers.{i}.';xn=self.norm(x,self.w[t+'input_layernorm.weight'])
            q=self.rotate((xn@self.w[t+'self_attn.q_proj.weight'].T).reshape(n,s.heads,s.head_dim),positions)
            k=self.rotate((xn@self.w[t+'self_attn.k_proj.weight'].T).reshape(n,s.heads,s.head_dim),positions)
            v=(xn@self.w[t+'self_attn.v_proj.weight'].T).reshape(n,s.heads,s.head_dim)
            scores=np.einsum('thd,shd->hts',q,k,optimize=False)/self.dtype(np.sqrt(s.head_dim))
            scores=np.where(allowed[None,:,:],scores,-np.inf)
            prob=np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob=prob/np.sum(prob,axis=-1,keepdims=True)
            context=np.einsum('hts,shd->thd',prob,v,optimize=False).reshape(n,-1)
            residual=x+(context@self.w[t+'self_attn.o_proj.weight'].T)*self.w[t+'self_attn_layer_scale.scale']
            x=residual+self.mlp(self.norm(residual,self.w[t+'post_attention_layernorm.weight']),t)*self.w[t+'mlp_layer_scale.scale']
            trace[f'layer.{i}']=x.copy()
        normalized=self.norm(x,self.w[p+'norm.weight']);trace['normalized']=normalized
        trace['output']=self.affine(normalized,p+'output_proj')
        return trace
