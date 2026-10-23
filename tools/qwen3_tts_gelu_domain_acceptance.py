#!/usr/bin/env python3
"""Supplemental ISA probe covering the checkpoint's observed GELU input domain."""
import argparse
from dataclasses import replace
import hashlib
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
from opentpu.kernels.tts_audio_upsample import gelu_probe


def run(output):
    cfg=shared_tts_config();x=_tensor(0,(1,1024));y=_tensor(4096,(1,1024))
    b=gelu_probe.trace(cfg,0,{'input':x,'output':y});program=b.finish()
    machine=Machine(replace(cfg,DRAM_BYTES=8192),[[I.halt()]],[np.zeros(8192,np.uint8)])
    values=np.concatenate([np.linspace(-40,32,8192,dtype=np.float32),np.array([-1e4,1e4,0.,-0.],np.float32)])
    worst=0.
    for start in range(0,len(values),1024):
        part=values[start:start+1024];padded=np.pad(part,(0,1024-len(part)))
        machine.slices[0].dram[:4096]=padded.view(np.uint8);machine.load([program]).run()
        actual=machine.slices[0].dram[4096:8192].view(np.float32)[:len(part)]
        expected=np.array([.5*float(v)*(1+math.erf(float(v)/math.sqrt(2))) for v in part],np.float32)
        worst=max(worst,float(np.max(np.abs(actual-expected))))
    paths=[Path(__file__).resolve(),ROOT/'opentpu/kernels/tts_audio_upsample.py']
    result=dict(profile=profile_metadata(),runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',
        dense_grid_range=[-40,32],dense_grid_points=8192,additional_values=[-10000,10000,0.,-0.],
        observed_checkpoint_input_range=[-37.96326446533203,24.534238815307617],
        max_absolute_error=worst,bound=2e-6,accepted=worst<2e-6,physical_hardware_access=False,
        exhaustive_float32_proof=False,code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['accepted'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);sys.exit(run(p.parse_args().output))
