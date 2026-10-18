#!/usr/bin/env python3
"""Generate bounded real compiler inputs, deterministic random weights, no downloads."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['OTPU_IMAGE_CACHE']='0'
os.environ['OTPU_IMG_CACHE']='0'
os.environ['OTPU_PROG_CACHE']='0'
os.environ['OTPU_BUILD_JOBS']='1'
import numpy as np
from opentpu import Config,isa as I,language as ol
from opentpu.runtime import compile_kernel,Input,Weight,Output
from opentpu.perfsim.io import save_bundle,write_json


@ol.jit
def gemm(x,w,y):
    ol.store(y,ol.dot(ol.load(x),w))


def tiny_model():
    from opentpu.llm.qwen3 import Spec
    cfg=Config(D=32,MCOLS=4,LANES=8,ACT_BLOCKS=64,TMEM_WORDS=8192,DRAM_BYTES=1<<20)
    spec=Spec(hidden=64,layers=2,n_q=2,n_kv=1,head_dim=32,ffn=96,vocab=64)
    rng=np.random.default_rng(20261019)
    def w(shape): return (rng.standard_normal(shape)*0.04).astype(np.float32)
    weights={'model.embed_tokens.weight':w((64,64)),'model.norm.weight':np.ones(64,np.float32)}
    for layer in range(spec.layers):
        p=f'model.layers.{layer}.'
        for name,n in [('input_layernorm',64),('post_attention_layernorm',64),
                       ('self_attn.q_norm',32),('self_attn.k_norm',32)]:
            weights[p+name+'.weight']=np.ones(n,np.float32)
        for name,shape in [('self_attn.q_proj',(64,64)),('self_attn.k_proj',(32,64)),
                           ('self_attn.v_proj',(32,64)),('self_attn.o_proj',(64,64)),
                           ('mlp.gate_proj',(96,64)),('mlp.up_proj',(96,64)),('mlp.down_proj',(64,96))]:
            weights[p+name+'.weight']=w(shape)
    return cfg,spec,weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=Config(D=32,MCOLS=4,TMEM_WORDS=8192,DRAM_BYTES=1<<20)
    rng=np.random.default_rng(17)
    x=rng.standard_normal((3,64)).astype(np.float32)
    w=rng.standard_normal((19,64)).astype(np.float32)
    comp,images=compile_kernel(gemm,cfg,x=Input(x),w=Weight(w),y=Output((3,19)))
    save_bundle(out/'gemm',cfg,comp.programs[0],images[0],metadata={'case':'partial GEMM 3x64x19','seed':17})
    # No memory traffic: valid timing comparison to RTL with identical idle memory conditions.
    prog=[I.vop(I.V_FILL,0,0,0,1,67,67,67,0,I.B_SCALAR,2),
          I.vop(I.V_MUL,128,0,0,1,67,67,67,0,I.B_SCALAR,3),
          I.vop(I.V_RSUM,256,128,0,1,67,1,67,0),I.halt()]
    save_bundle(out/'compute',cfg,prog,np.zeros(4,np.uint8),metadata={'case':'FILL/MUL/RSUM 67 elements','external_traffic':False})
    cfg,spec,weights=tiny_model()
    from opentpu.llm.qwen3 import rope_tables,reference_logits
    from opentpu.isasim import Machine
    img=spec.image(cfg,cap=256)
    dram=img.build(weights)[0]
    tokens=[1,5,9]
    refs=reference_logits(spec,weights,tokens)
    for pos,tok in enumerate(tokens):
        cos,sin=rope_tables(spec,pos)
        for key,value in [('x',weights['model.embed_tokens.weight'][tok]),('cos',cos),('sin',sin)]:
            data=np.ascontiguousarray(value,np.float32).view(np.uint8).ravel()
            dram[img.io[key]:img.io[key]+len(data)]=data
        progs=img.compile_step(pos)
        labels={pc:'decoder' if ins.op!=I.HALT else 'halt' for pc,ins in enumerate(progs[0])}
        save_bundle(out/f'tiny_qwen_{pos}',cfg,progs[0],dram,labels=labels,
                    metadata={'case':'tiny random full Qwen3 topology','spec':asdict(spec),'seed':20261019,
                              'position':pos,'token':tok,'phase':'prefill' if pos<2 else 'decode',
                              'logits_address':img.io['logits'],'logits_count':spec.vocab,
                              'host_inputs':[[img.io['x'],4*spec.hidden],[img.io['cos'],2*spec.rope_dim],[img.io['sin'],2*spec.rope_dim]],
                              'note':'teacher-forced tokens; two serial prompt steps, then one decode step; no host/sample cost'})
        np.save(out/f'tiny_qwen_{pos}'/'reference_logits.npy',refs[pos])
        machine=Machine(cfg,progs,[dram]).run()
        dram=machine.slices[0].dram.copy()
    write_json(out/'sequence.json',{'programs':['tiny_qwen_0','tiny_qwen_1','tiny_qwen_2'],
                                   'prefill_steps':2,'decode_steps':1,'full_pretrained_model':False})
    print(out)

if __name__=='__main__': main()
