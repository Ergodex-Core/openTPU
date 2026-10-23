#!/usr/bin/env python3
"""Complete bounded hybrid decoder using the unchanged baseline Qwen3.5 compiler.

Random two-layer topology: one two-head DeltaNet layer and one gated-attention layer.
DSTEP or STREAM plus optional production wait-count issue; independent NumPy logits.
"""
import argparse,os,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ['OTPU_IMAGE_CACHE','OTPU_IMG_CACHE','OTPU_PROG_CACHE']:os.environ[key]='0'
os.environ['OTPU_BUILD_JOBS']='1';os.environ['OTPU_QUANT_JOBS']='1'
import numpy as np
from opentpu import Config
from opentpu.llm.qwen35 import Spec,reference_logits,rope_tables
from opentpu.isasim import Machine
from opentpu.perfsim.io import save_bundle,write_json


def tiny():
 spec=Spec(128,('linear','attn'),2,1,128,32,2,128,128,256,128)
 rng=np.random.default_rng(20261022)
 def random(shape):return (rng.standard_normal(shape)*.02).astype(np.float32)
 w={'model.embed_tokens.weight':random((128,128)),'model.norm.weight':np.zeros(128,np.float32)}
 for i,kind in enumerate(spec.kinds):
  p=f'model.layers.{i}.'
  for n in ['input_layernorm','post_attention_layernorm']:w[p+n+'.weight']=np.zeros(128,np.float32)
  for n,shape in [('mlp.gate_proj',(256,128)),('mlp.up_proj',(256,128)),('mlp.down_proj',(128,256))]:w[p+n+'.weight']=random(shape)
  if kind=='linear':
   a=p+'linear_attn.'
   for n,shape in [('in_proj_qkv.weight',(768,128)),('in_proj_z.weight',(256,128)),('out_proj.weight',(128,256)),('in_proj_a.weight',(2,128)),('in_proj_b.weight',(2,128)),('conv1d.weight',(768,1,4))]:w[a+n]=random(shape)
   w[a+'A_log']=np.zeros(2,np.float32);w[a+'dt_bias']=np.zeros(2,np.float32);w[a+'norm.weight']=np.ones(128,np.float32)
  else:
   for n,shape in [('q_proj',(512,128)),('k_proj',(128,128)),('v_proj',(128,128)),('o_proj',(128,256))]:w[p+'self_attn.'+n+'.weight']=random(shape)
   for n in ['q_norm','k_norm']:w[p+'self_attn.'+n+'.weight']=np.zeros(128,np.float32)
 return spec,w


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stream',action='store_true');ap.add_argument('--waitc',action='store_true');ap.add_argument('--chunk-prefill',action='store_true');a=ap.parse_args()
 if a.out.exists():ap.error('output must not exist')
 spec,weights=tiny();cfg=Config(D=128,MCOLS=4,LANES=8,ACT_BLOCKS=64,TMEM_WORDS=65536,DRAM_BYTES=1<<25,DSTEP=not a.stream,STREAM=a.stream,SEQ_WC=a.waitc)
 img=spec.image(cfg,cap=256,rows=2 if a.chunk_prefill else 1);dram=img.build(weights)[0];tokens=[1,5,9,13] if a.chunk_prefill else [1,5,9];refs=reference_logits(spec,weights,tokens)
 names=[]
 groups=[[0,1],[2],[3]] if a.chunk_prefill else [[0],[1],[2]]
 for stage,positions in enumerate(groups):
  pos=positions[-1];tok=tokens[pos];inputs=[]
  cos=np.stack([rope_tables(spec,p)[0] for p in positions]);sin=np.stack([rope_tables(spec,p)[1] for p in positions])
  for key,value in [('x',weights['model.embed_tokens.weight'][[tokens[p] for p in positions]]),('cos',cos),('sin',sin)]:
   raw=np.ascontiguousarray(value,np.float32).view(np.uint8).ravel();addr=img.io[key];dram[addr:addr+len(raw)]=raw;inputs.append([addr,len(raw)])
  program=img.compile_rows([(0,p) for p in positions],[len(positions)-1]) if len(positions)>1 else img.compile_step(pos)
  name=f'hybrid_{stage}';names.append(name)
  save_bundle(a.out/name,cfg,program[0],dram,metadata={'case':'complete random hybrid decoder','spec':asdict(spec),'seed':20261022,'token':tok,'position':pos,'positions':positions,'phase':'prefill' if pos<2 else 'decode','logits_address':img.io['logits']+(len(positions)-1)*4*spec.vocab,'logits_count':spec.vocab,'host_inputs':inputs,'independent_reference':'unchanged baseline NumPy Qwen3.5 forward; no ISA arithmetic helpers','full_pretrained_model':False})
  np.save(a.out/name/'reference_logits.npy',refs[pos]);dram=Machine(cfg,program,[dram]).run().slices[0].dram.copy();print(name,len(program[0]),flush=True)
 write_json(a.out/'sequence.json',{'programs':names,'prefill_steps':1 if a.chunk_prefill else 2,'decode_steps':2 if a.chunk_prefill else 1,'full_pretrained_model':False,'prefill_tokens':2,'prefill_mode':'batched two-token prompt' if a.chunk_prefill else 'two serial prompt-token steps'})
if __name__=='__main__':main()
