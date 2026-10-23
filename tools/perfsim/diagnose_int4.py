#!/usr/bin/env python3
"""Separate float-model quantization loss from packing, QACT and MXU execution errors.

Independent decoder/arithmetic below uses NumPy and ISA field definitions; it never calls
opentpu.quant or opentpu.fp32. The ISA engine is used only as a separately labelled subject.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu import isa as I
from opentpu.isasim import Machine
from opentpu.perfsim.io import load_bundle,manifest,write_json


def source_arrays():
    rng=np.random.default_rng(20261020)
    for k in (1152,1152,6912,2048,6144):
        rng.uniform(-1,1,(1,k));rng.standard_normal((64,k))
    rng.uniform(-1,1,(4,1152));rng.standard_normal((64,1152))
    return rng.uniform(-1,1,(1,2048)).astype(np.float32),(rng.standard_normal((64,2048))*0.02).astype(np.float32)


def decode_weights(image,ins,D):
    a,s,_,nkb,rs,_,srs=ins.w;n,kb=nkb&65535,(nkb>>16)&255;k0=nkb>>24
    packed=np.stack([image[a+r*rs:a+r*rs+kb*D//2] for r in range(n)])
    codes=np.empty((n,kb*D),np.int64);codes[:,0::2]=packed&15;codes[:,1::2]=packed>>4
    codes=np.where(codes>=8,codes-16,codes).reshape(n,kb,4,D//4)
    sw=np.array([np.frombuffer(image[s+r*srs:s+r*srs+4].tobytes(),'<f4')[0] for r in range(n)])
    records=np.array([[int.from_bytes(image[s+r*srs+8+4*(k+k0):s+r*srs+12+4*(k+k0)].tobytes(),'little') for k in range(kb)] for r in range(n)],np.uint32)
    dw=(records&255).astype(np.int64);m=np.stack([(records>>(8+6*g))&63 for g in range(4)],-1).astype(np.int64)
    deq=codes*m[:,:,:,None]*np.exp2(-dw[:,:,None,None])*sw[:,None,None,None]
    return codes,sw,dw,m,deq.reshape(n,-1)


def activation(x,D):
    maximum=np.abs(x).max(axis=1);group=np.abs(x).reshape(len(x),-1,D//4).max(axis=2)
    off=np.clip(np.floor(np.log2(maximum[:,None]/group)),0,31).astype(np.int64)
    scale=(maximum*np.float32(1/127)).astype(np.float32)
    inv=(np.float32(127)/maximum).astype(np.float32)
    expanded=(inv[:,None]*np.exp2(off).astype(np.float32)).astype(np.float32)
    code=np.clip(np.rint((x.reshape(len(x),-1,D//4)*expanded[:,:,None]).astype(np.float32)),-127,127).astype(np.int8)
    return code.reshape(len(x),-1),scale,off


def rel(a,b):return float(np.linalg.norm(np.asarray(a,np.float64)-b)/np.linalg.norm(b))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--rtl-run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():ap.error('output must not exist')
    a.out.mkdir(parents=True)
    cfg,program,image,obj=load_bundle(a.bundle);x,w=source_arrays()
    ld=next(i for i in program if i.op==I.LD);mm=next(i for i in program if i.op==I.MM)
    loaded=np.frombuffer(image[ld.w[0]:ld.w[0]+4*x.size].tobytes(),'<f4').reshape(x.shape)
    assert np.array_equal(loaded,x),'generator input changed; do not diagnose a different source'
    original=x.astype(np.float64)@w.astype(np.float64).T
    assert np.array_equal(original.reshape(-1),np.load(a.bundle/'reference_logits.npy'))
    code,sw,dw,m,deq=decode_weights(image,mm,cfg.D)
    q,sa,da=activation(x,cfg.D);kb=x.shape[1]//cfg.D
    acc=np.zeros((1,len(w)),np.int64);dropped=0
    for k in range(kb):
        for g in range(4):
            dot=(code[:,k,g]*q.reshape(1,kb,4,cfg.D//4)[0,k,g]).sum(axis=1)
            shift=11-dw[:,k]-da[0,k*4+g];dropped+=int(np.count_nonzero(shift<0))
            acc[0]+=dot*m[:,k,g]*np.where(shift>=0,np.left_shift(1,np.maximum(shift,0)),0)
    # IEEE float32 round at each specified epilogue stage; no shared arithmetic helpers.
    exact=((acc.astype(np.float32)*sw[None,:]).astype(np.float32)*(sa[:,None]*np.float32(2**-11)).astype(np.float32)).astype(np.float32)
    isa=Machine(cfg,[program],[image]);isa.run();state=isa.slices[0]
    md=obj['metadata'];addr,n=md['logits_address'],md['logits_count']
    actual=state.dram[addr:addr+4*n].view(np.float32)
    rtl=np.fromfile(a.rtl_run/'dram_out_0.bin',np.uint8)[addr:addr+4*n].view(np.float32)
    qeq=np.array_equal(q[0],state.act[0,:x.shape[1]]) and np.array_equal(da[0],state.adoff[0,:kb].reshape(-1)) and np.array_equal(sa.view(np.uint32),state.ascale[0,:1].view(np.uint32))
    stages={'weight_only_dequantized_dot':x.astype(np.float64)@deq.T,'activation_and_weight_dequantized_dot':(q.reshape(1,kb,4,-1).astype(np.float64)*np.exp2(-da.reshape(1,kb,4,1))*sa[:,None,None,None]).reshape(x.shape)@deq.T,'independent_integer_window_and_f32_epilogue':exact}
    differences=np.argwhere(w.astype(np.float64)!=deq);first=tuple(map(int,differences[0]))
    result={'diagnosis':'lossy int4 quantization exceeds the original float-model suitability gate; no simulator/packing/RTL arithmetic divergence in these programs','original_float_gate_passed':rel(rtl,original.reshape(-1))<.05,'original_float_relative_l2':rel(rtl,original.reshape(-1)),'first_representation_divergence':{'stage':'compiled weight quantization before execution','weight_index':first,'original_float':float(w[first]),'packed_dequantized':float(deq[first]),'expected':'lossy int4 representation; original float reference intentionally retained'},'weight_relative_l2':rel(deq,w.astype(np.float64)),'independent_activation_matches_ISA':qeq,'independent_packed_arithmetic_bit_exact_ISA':np.array_equal(exact.reshape(-1).view(np.uint32),actual.view(np.uint32)),'independent_packed_arithmetic_bit_exact_RTL':np.array_equal(exact.reshape(-1).view(np.uint32),rtl.view(np.uint32)),'dropped_groups':dropped,'stages':{name:{'relative_l2_vs_original_float':rel(value,original),'relative_l2_vs_rtl':rel(value.reshape(-1),rtl.astype(np.float64))} for name,value in stages.items()},'scope':'two seeded operator cases only; does not establish full-model int4 quality','D':cfg.D}
    write_json(a.out/'diagnosis.json',result);write_json(a.out/'manifest.json',manifest([a.bundle/'input.json',a.bundle/'prog_0.hex',a.bundle/'dram_0.bin',a.bundle/'reference_logits.npy',a.rtl_run/'dram_out_0.bin']))
    print(json.dumps(result,indent=2))
    if not all(result[k] for k in ('independent_activation_matches_ISA','independent_packed_arithmetic_bit_exact_ISA','independent_packed_arithmetic_bit_exact_RTL')):raise SystemExit(1)
if __name__=='__main__':main()
