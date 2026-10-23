"""Offline waveform audit helpers; host transports bytes, never preprocesses them."""
import hashlib
import json
from pathlib import Path
import numpy as np
from opentpu.compiler import Tensor
from opentpu.asr.phonon2_waveform import move_rows


def coefficients(root):
    root=Path(root);path=root/'docs/phonon2/waveform_coefficients.npz'
    meta=json.loads((root/'docs/phonon2/waveform_coefficients.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('coefficient asset differs')
    data=dict(np.load(path,allow_pickle=False))
    phase=np.arange(257)[:,None]*np.arange(512)[None,:]%512
    return {'hann':data['hann'],'mel':data['mel'],
            'cosine':data['root_cosine'][phase],'sine':data['root_sine'][phase]}


def snapshots(device,plan):
    return {name:device.memory[s['offset']:s['offset']+4*int(np.prod(s['shape']))].view('<f4').reshape(s['shape']).copy()
            for name,s in plan.checkpoints.items()}


def transport_pcm(device,plan,pcm):
    if len(pcm)!=2*plan.samples:raise ValueError('PCM byte length differs')
    # Alignment padding and byte copies only; no host signed conversion,
    # scaling, preemphasis, transform, masking, statistics or model computation.
    device.memory[plan.input_at:plan.input_at+plan.raw_padded_bytes]=0
    device.memory[plan.input_at:plan.input_at+len(pcm)]=np.frombuffer(pcm,np.uint8)


def run(device,plan,pcm,*,check=False):
    device.reset(plan.image.size)
    device.constants(plan.image,((0,plan.input_at),))
    transport_pcm(device,plan,pcm)
    for program in plan.programs:device.phase(program,check=check)
    return snapshots(device,plan)


def device_move(device,source,destination,rows,width,*,check=False):
    size=rows*width*4
    reverse=source<destination<source+size
    x=Tensor(source,(rows,width),(width,1));out=Tensor(destination,(rows,width),(width,1))
    device.phase(move_rows.trace(device.cfg,0,dict(x=x,out=out,rows=rows,reverse=reverse)).finish(),check=check)
    return {'source':source,'destination':destination,'bytes':size,'reverse':reverse,'host_activation_transform_or_reupload':False}


def arm_sum(x):
    """Independent NumPy expression of pinned Torch Vec4/ILP4 cascade order."""
    x=np.asarray(x,np.float32);channels,count=x.shape
    partial=np.zeros((4,channels,4),np.float32);cascade=np.zeros_like(partial)
    groups=count//16
    for group in range(groups):
        for k in range(4):partial[k]+=x[:,group*16+k*4:group*16+k*4+4]
        if (group+1)%16==0:cascade+=partial;partial[:]=0
    partial+=cascade
    for i in range(groups*4,count//4):partial[0]+=x[:,i*4:i*4+4]
    for k in range(1,4):partial[0]+=partial[k]
    total=np.zeros(channels,np.float32)
    for i in range(count//4*4,count):total+=x[:,i]
    for k in range(4):total+=partial[0,:,k]
    return total


def canonical_frontend(native):
    """Same source equations with retained ARM64 Torch reduction pinned on any CPU.

    Used only for audit. Native x86 Torch sums can differ in ill-conditioned
    silence; report that separately rather than silently altering the device.
    """
    logs=native['log_mel'];valid=native['valid_frames']
    masked=logs.T.copy();masked[:,-1]=0
    mean=arm_sum(masked)/np.float32(valid)
    centered=masked-mean[:,None];squared=centered*centered;squared[:,-1]=0
    variance=arm_sum(squared)/np.float32(valid-1)
    features=(logs-mean[None,:])/(np.sqrt(variance)[None,:]+np.float32(1e-5));features[-1]=0
    return {**native,'features':features,'mean':mean,'variance':variance}
