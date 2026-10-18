#!/usr/bin/env python3
"""Compile actual Gemma/Bonsai operator dimensions, with explicitly synthetic data.

Reads configuration metadata only. Optional Gemma Q projection reads just that tensor from
an already local safetensors file; no network, model loader, credential or full-weight load.
"""
import argparse,json,struct,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['OTPU_IMAGE_CACHE']='0';os.environ['OTPU_IMG_CACHE']='0';os.environ['OTPU_PROG_CACHE']='0'
import numpy as np
from opentpu import Config,language as ol
from opentpu.runtime import compile_kernel,Input,Weight,Output
from opentpu.kernels.lib import rmsnorm,gelu_tanh,silu
from opentpu.perfsim.io import save_bundle,write_json,digest


@ol.jit
def projection(x,w,y):ol.store(y,ol.dot(ol.load(x),w))


@ol.jit
def vector_kernel(x,y,kind):
    value=ol.load(x)
    if kind=='rmsnorm':value=rmsnorm(value,None,1e-6)
    elif kind=='gelu':value=gelu_tanh(value)
    elif kind=='silu':value=silu(value)
    ol.store(y,value)


@ol.jit
def delta_head(state,query,key,value,decay,out):
    s=ol.load(state);q=ol.load(query);k=ol.load(key);v=ol.load(value);d=ol.load(decay)
    delta=(v-(s@k)*d)*0.25
    ol.outer(delta,k,acc=s,decay=d)
    ol.store(out[:128,:],s)
    ol.store(out[128,:],s@q)


def local_tensor(path,name):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));meta=header[name]
        lo,hi=meta['data_offsets'];f.seek(8+size+lo);raw=f.read(hi-lo)
    if meta['dtype']=='BF16':value=(np.frombuffer(raw,'<u2').astype('<u4')<<16).view('<f4')
    elif meta['dtype']=='F32':value=np.frombuffer(raw,'<f4').copy()
    elif meta['dtype']=='F16':value=np.frombuffer(raw,'<f2').astype(np.float32)
    else:raise ValueError('unsupported local tensor dtype')
    import hashlib
    return value.reshape(meta['shape']),{'path':str(path.resolve()),'tensor':name,'dtype':meta['dtype'],'shape':meta['shape'],'tensor_sha256':hashlib.sha256(raw).hexdigest(),'bytes_read':len(raw),'whole_checkpoint_loaded':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gemma-config',type=Path,required=True);ap.add_argument('--bonsai-config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--full',action='store_true')
    ap.add_argument('--gemma-weights',type=Path)
    ap.add_argument('--depth',type=int,choices=(32,128),default=32)
    a=ap.parse_args()
    if a.out.exists():ap.error('output must not exist')
    a.out.mkdir(parents=True)
    gc=json.loads(a.gemma_config.read_text());bc=json.loads(a.bonsai_config.read_text())['text_config']
    cfg=Config(D=a.depth,MCOLS=4,LANES=8,ACT_BLOCKS=256 if a.depth==32 else 64,TMEM_WORDS=65536,DRAM_BYTES=1<<25)
    gh,gf=gc['hidden_size'],gc['intermediate_size'];bh,bf=bc['hidden_size'],bc['intermediate_size']
    shapes=[('gemma_q',gh,gc['num_attention_heads']*gc['head_dim']),('gemma_mlp_up',gh,gf),('gemma_mlp_down',gf,gh),('bonsai_mlp_up',bh,bf),('bonsai_mlp_down',bf,bh)]
    rows=[];rng=np.random.default_rng(20261020)
    def emit(name,kind,m,k,n,weights=None,fmt='int8',pair=False,provenance=None,full_n=None):
        from dataclasses import replace
        c=replace(cfg,PAIR=pair);x=rng.uniform(-1,1,(m,k)).astype(np.float32)
        begin=time.perf_counter()
        if kind=='projection':
            w=weights if weights is not None else (rng.standard_normal((n,k))*0.02).astype(np.float32)
            comp,images=compile_kernel(projection,c,x=Input(x),w=Weight(w,fmt=fmt),y=Output((m,n)))
            reference=x.astype(np.float64)@w.astype(np.float64).T
        else:
            comp,images=compile_kernel(vector_kernel,c,x=Input(x),y=Output((m,k)),kind=kind)
            xx=x.astype(np.float64)
            reference=xx/np.sqrt(np.mean(xx*xx,axis=1,keepdims=True)+1e-6) if kind=='rmsnorm' else 0.5*xx*(1+np.tanh(np.sqrt(2/np.pi)*(xx+0.044715*xx**3))) if kind=='gelu' else xx/(1+np.exp(-xx))
        elapsed=time.perf_counter()-begin;address=comp.layout.params['y'][1]
        metadata={'case':name,'kind':kind,'shape_M_K_N':[m,k,n],'full_operator_N':full_n or n,'complete_operator_shape':not full_n or full_n==n,'data':'synthetic activations; local pretrained projection weights only' if provenance and 'path' in provenance else 'synthetic activations and weights','weight_format':fmt,'logits_address':address,'logits_count':reference.size,'independent_reference':'NumPy float64 operator; not end-to-end pretrained inference','weight_source':provenance,'compile_wall_seconds':elapsed}
        # Omit irrelevant trailing zero DRAM while keeping configured capacity explicit.
        used=max(address+reference.size*4,int(np.flatnonzero(images[0])[-1])+1)
        save_bundle(a.out/name,c,comp.programs[0],images[0][:used],metadata=metadata)
        np.save(a.out/name/'reference_logits.npy',reference.reshape(-1));rows.append(metadata)
        print(name,len(comp.programs[0]),elapsed,flush=True)
    for name,k,n in shapes:emit(name,'projection',1,k,n if a.full else 64,full_n=n)
    if not a.full:
        emit('gemma_prefill4_q_shard','projection',4,gh,64,full_n=gc['num_attention_heads']*gc['head_dim'])
        emit('bonsai_int4_pair_shard','projection',1,bh,64,fmt='int4',pair=True,full_n=bf)
        for name,k in [('gemma',gh),('bonsai',bh)]:emit(name+'_rmsnorm','rmsnorm',1,k,k)
        emit('gemma_gelu','gelu',1,gf,gf);emit('bonsai_silu','silu',1,bf,bf)
        if a.gemma_weights:
            w,provenance=local_tensor(a.gemma_weights,'model.layers.0.self_attn.q_proj.weight')
            emit('gemma_local_q_projection','projection',1,w.shape[1],w.shape[0],weights=w,provenance=provenance)
    if not a.full:
        ternary=rng.integers(-1,2,(64,bh)).astype(np.float32)*0.02
        emit('bonsai_ternary_int4_pair','projection',1,bh,64,weights=ternary,fmt='int4',pair=True,full_n=bf,provenance={'kind':'synthetic ternary weights; no Hadamard model basis claim'})
        if bc['linear_key_head_dim']!=128 or bc['linear_value_head_dim']!=128:raise ValueError('DeltaNet case requires 128x128 local metadata')
        state=rng.normal(0,0.02,(128,128)).astype(np.float32)
        q=rng.normal(0,0.05,128).astype(np.float32);k=rng.normal(0,0.05,128).astype(np.float32)
        v=rng.normal(0,0.02,128).astype(np.float32);d=np.array([0.9],np.float32)
        begin=time.perf_counter()
        comp,images=compile_kernel(delta_head,cfg,state=Input(state),query=Input(q),key=Input(k),value=Input(v),decay=Input(d),out=Output((129,128)))
        updated=state.astype(np.float64)*float(d[0])+np.outer((v.astype(np.float64)-(state.astype(np.float64)@k)*float(d[0]))*0.25,k)
        reference=np.concatenate([updated.reshape(-1),updated@q]);name='bonsai_deltanet_head128'
        address=comp.layout.params['out'][1]
        md={'case':name,'kind':'deltanet_head','shape_M_K_N':[128,128,128],'complete_operator_shape':True,'data':'synthetic recurrent state, q/k/v/decay; VPU fallback, no DSTEP/STREAM','weight_format':None,'logits_address':address,'logits_count':reference.size,'independent_reference':'NumPy float64 DeltaNet recurrence and output; not pretrained model','compile_wall_seconds':time.perf_counter()-begin}
        save_bundle(a.out/name,cfg,comp.programs[0],images[0][:address+reference.size*4],metadata=md)
        np.save(a.out/name/'reference_logits.npy',reference);rows.append(md)
    write_json(a.out/'catalog.json',{'cases':rows,'configuration_sources':{str(p):digest(p) for p in (a.gemma_config,a.bonsai_config)},'gemma_config':gc,'bonsai_text_config':bc,'pretrained_model_validated':False,'no_sparse_speedup':True})
if __name__=='__main__':main()
