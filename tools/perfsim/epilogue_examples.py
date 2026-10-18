#!/usr/bin/env python3
"""Compiled partial/full four-row GEMMs that exercise E2 versus E4 epilogue draining."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu import Config
from opentpu.runtime import compile_kernel,Input,Weight,Output
from tools.perfsim.examples import gemm
from opentpu.perfsim.io import save_bundle,write_json


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():ap.error('output must not exist')
 cfg=Config(D=128,MCOLS=4,LANES=8,ACT_BLOCKS=64,TMEM_WORDS=65536,DRAM_BYTES=1<<25,SEQ_WC=True)
 rng=np.random.default_rng(20261024);names=[]
 for m,k,n in [(3,128,19),(4,128,67),(4,256,33)]:
  name=f'm{m}k{k}n{n}';names.append(name);x=rng.normal(0,.1,(m,k)).astype(np.float32);w=rng.normal(0,.1,(n,k)).astype(np.float32)
  compiled,images=compile_kernel(gemm,cfg,x=Input(x),w=Weight(w),y=Output((m,n)))
  addr=compiled.layout.params['y'][1]
  used=max(addr+m*n*4,int(np.flatnonzero(images[0])[-1])+1);image=images[0][:used]
  save_bundle(a.out/name,cfg,compiled.programs[0],image,metadata={'case':name,'phase':'independent epilogue kernel','position':0,'logits_address':addr,'logits_count':m*n,'host_inputs':[[0,len(image)]],'independent_reference':'NumPy float64 dense GEMM; separate quantization tolerance'})
  np.save(a.out/name/'reference_logits.npy',(x.astype(np.float64)@w.astype(np.float64).T).reshape(-1))
 write_json(a.out/'sequence.json',{'programs':names,'purpose':'independent compiled kernels, each DRAM image replaced; controller/TMEM/ACT retained','full_pretrained_model':False})
if __name__=='__main__':main()
