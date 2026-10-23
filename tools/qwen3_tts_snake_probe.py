#!/usr/bin/env python3
"""Qualify bounded sin² and inspect pinned static SnakeBeta parameters."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from opentpu import isa as I
from opentpu.isasim import Machine
from opentpu.llm.qwen3_tts_shared import shared_tts_config,profile_metadata
from opentpu.llm.qwen3_tts_talker import _tensor
from opentpu.kernels.tts_waveform import sin2_probe,SNAKE_ARGUMENT_LIMIT
from tools.qwen3_tts_audio_checkpoint import AudioCheckpoint,digest


def run(a):
    cfg=shared_tts_config();input=_tensor(0,(1,1024));output=_tensor(4096,(1,1024))
    b=sin2_probe.trace(cfg,0,{'input':input,'output':output});program=b.finish()
    m=Machine(replace(cfg,DRAM_BYTES=8192),[[I.halt()]],[np.zeros(8192,np.uint8)])
    limit=SNAKE_ARGUMENT_LIMIT;rng=np.random.default_rng(2332)
    groups={'dense':np.linspace(-limit,limit,16384,dtype=np.float32),'random':rng.uniform(-limit,limit,16384).astype(np.float32),'near_zero':np.linspace(-2,2,4096,dtype=np.float32)}
    critical=[]
    for k in [0,1,2,127,1023,8191,32767,65535,83442]:
        for phase in [0.,.5]:
            x=np.float32((k+phase)*math.pi)
            for value in [np.nextafter(x,np.float32(-np.inf)),x,np.nextafter(x,np.float32(np.inf))]:
                if abs(float(value))<=limit:critical.extend([value,-value])
    groups['critical']=np.asarray(critical,np.float32);errors={}
    for name,values in groups.items():
        worst=0.
        for start in range(0,len(values),1024):
            part=values[start:start+1024];padded=np.pad(part,(0,1024-len(part)));m.slices[0].dram[:4096]=padded.view(np.uint8);m.load([program]).run()
            actual=m.slices[0].dram[4096:8192].view(np.float32)[:len(part)];want=(np.sin(part.astype(np.float64))**2).astype(np.float32)
            worst=max(worst,float(np.max(np.abs(actual-want))))
        errors[name]=dict(points=len(values),max_absolute_error=worst)
    checkpoint=AudioCheckpoint(a.data/'audio-model.safetensors');parameters=[]
    for key,entry in checkpoint.header.items():
        if not key.startswith('decoder.decoder.') or not key.endswith('.alpha'):continue
        alpha=checkpoint.tensor(key,entry['shape']);beta=checkpoint.tensor(key[:-5]+'beta',entry['shape'])
        ae=np.exp(alpha).astype(np.float32);bi=(np.float32(1)/(np.exp(beta).astype(np.float32)+np.float32(1e-9))).astype(np.float32)
        if not np.isfinite(ae).all() or not np.isfinite(bi).all():raise ValueError('nonfinite static SnakeBeta constants')
        parameters.append(dict(source=key,channels=len(alpha),exp_alpha_range=[float(ae.min()),float(ae.max())],inv_beta_range=[float(bi.min()),float(bi.max())]))
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',argument_limit=limit,intrinsic_absolute_error_bound=2e-5,errors=errors,
        accepted=all(e['max_absolute_error']<=2e-5 for e in errors.values()),snake_parameter_groups=parameters,physical_hardware_access=False,
        waveform_validated=False,quality_validated=False,tts_enabled=False,code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),ROOT/'opentpu/kernels/tts_waveform.py']})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['accepted'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);sys.exit(run(p.parse_args()))
