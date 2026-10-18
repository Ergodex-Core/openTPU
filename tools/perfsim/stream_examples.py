#!/usr/bin/env python3
"""Bounded baseline DSTEP/STREAM programs with independent float64 recurrence outputs."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu import Config,isa as I
from opentpu.perfsim.io import save_bundle,write_json


def modes(out,depth,waitc):
 cfg=Config(D=depth,MCOLS=4,LANES=8,ACT_BLOCKS=64,TMEM_WORDS=8192,DRAM_BYTES=1<<20,DSTEP=True,STREAM=True,SEQ_WC=waitc)
 rng=np.random.default_rng(20261023);catalog=[]
 for mode,ops in [('delta',I.D_DELTA),('delta1',I.D_DELTA1),('scale',I.D_SCALE),('dot',I.D_DOT)]:
  for gate in ('const','col','one'):
   for q_on in (False,True):
    rows,cols=3,192;zero=not q_on and mode!='dot';name=f'{mode}_{gate}_q{int(q_on)}';catalog.append(name)
    state=rng.normal(0,.02,(rows,cols)).astype(np.float32);vec=rng.normal(0,.05,(4,cols)).astype(np.float32);vec[3]=rng.uniform(.8,1,cols)
    x=rng.normal(0,.02,rows).astype(np.float32);k=np.array([.9,.25],np.float32)
    desc=np.array(I.state_desc(rows,cols,ops,2,gate,q_on).words(),'<u4');image=np.zeros(1<<18,np.uint8)
    for addr,val in [(0,vec),(4096,x),(8192,k),(12288,desc),(65536,state)]:image[addr:addr+val.nbytes]=val.reshape(-1).view(np.uint8)
    program=[I.ld(0,100,vec.size),I.ld(4096,1000,rows),I.ld(8192,1100,2),I.ld(12288,2000,len(desc)),I.stream(2000,65536,131072,100,1000,1100,4000,zero=zero)]
    if q_on:program.append(I.st(131072+state.nbytes,4000,rows))
    program.append(I.halt());S=np.zeros_like(state,dtype=np.float64) if zero else state.astype(np.float64);V=vec.astype(np.float64);A=S@V[2]
    d={'delta':(x-A*float(k[0]))*float(k[1]),'delta1':(x-A)*float(k[1]),'scale':x.astype(np.float64)*float(k[1]),'dot':A*float(k[1])}[mode]
    G={'const':float(k[0]),'col':V[3],'one':1}[gate];Y=S*G+d[:,None]*V[1];reference=np.r_[Y.reshape(-1),Y@V[0]] if q_on else Y.reshape(-1)
    save_bundle(out/name,cfg,program,image,metadata={'case':name,'phase':'synthetic stream descriptor modes','zero':zero,'separate_source_destination':True,'logits_address':131072,'logits_count':len(reference),'independent_reference':'NumPy float64 recurrence; no ISA arithmetic helpers'})
    np.save(out/name/'reference_logits.npy',reference)
 write_json(out/'catalog.json',catalog)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--depth',type=int,default=32,choices=[32,128]);ap.add_argument('--waitc',action='store_true');ap.add_argument('--modes',action='store_true');a=ap.parse_args()
 if a.out.exists():ap.error('output must not exist')
 if a.modes:return modes(a.out,a.depth,a.waitc)
 cfg=Config(D=a.depth,MCOLS=4,LANES=8,ACT_BLOCKS=64,TMEM_WORDS=8192,DRAM_BYTES=1<<20,DSTEP=True,STREAM=True,SEQ_WC=a.waitc)
 rng=np.random.default_rng(20261021);catalog=[]
 for rows,cols in [(1,64),(7,64),(3,192),(32,256),(128,128),(256,256)]:
  for stream,zero in [(False,False),(False,True),(True,False)]:
   name=f'{"stream" if stream else "dstep"}_{rows}x{cols}'+('_zero' if zero else '')
   image=np.zeros(1<<20,np.uint8);state=rng.normal(0,.02,(rows,cols)).astype(np.float32);qk=rng.normal(0,.05,2*cols).astype(np.float32);v=rng.normal(0,.02,rows).astype(np.float32);g=np.array([.9,0,.25],np.float32)
   for addr,value in [(0,qk),(4096,v),(8192,g),(65536,state)]:image[addr:addr+value.nbytes]=value.reshape(-1).view(np.uint8)
   program=[I.ld(0,100,2*cols),I.ld(4096,700,rows),I.ld(8192,1000,3)]
   if stream:
    desc=np.array(I.gdn_desc(rows,cols).words(),'<u4');image[12288:12288+desc.nbytes]=desc.view(np.uint8)
    program.append(I.ld(12288,2000,len(desc)));program.append(I.stream(2000,65536,65536,100,700,1000,4000,ks=2))
   else:program.append(I.dstep(65536,100,700,rows,cols,1000,2,4000,zero=zero))
   program.extend([I.st(65536+state.nbytes,4000,rows),I.halt()])
   S=np.zeros_like(state,dtype=np.float64) if zero else state.astype(np.float64);q,k=qk[:cols].astype(np.float64),qk[cols:].astype(np.float64)
   delta=(v.astype(np.float64)-(S@k)*float(g[0]))*float(g[2]);updated=S*float(g[0])+delta[:,None]*k[None,:];reference=np.r_[updated.reshape(-1),updated@q]
   md={'case':name,'phase':'synthetic baseline recurrence','logits_address':65536,'logits_count':len(reference),'independent_reference':'NumPy float64 gated DeltaNet state and output; not a pretrained model'}
   save_bundle(a.out/name,cfg,program,image[:65536+state.nbytes+4*rows],metadata=md);np.save(a.out/name/'reference_logits.npy',reference);catalog.append(name)
 write_json(a.out/'catalog.json',catalog)
if __name__=='__main__':main()
