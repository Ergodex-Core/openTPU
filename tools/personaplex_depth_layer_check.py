#!/usr/bin/env python3
"""One real-checkpoint depth layer over16 codebooks plus reset, CPU/ISA only.

Uses cached conditioning outputs and actual installed native streaming layer.
No temporal resimulation, complete six-layer run, audio generation or FPGA use.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1")
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opentpu import target
from opentpu.llm.personaplex_depth_layer import DepthLayerSpec,DepthLayerImage,DepthLayerEngine
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_isa_check import precision_metrics,PINNED_REVISION,PROFILE
from tools.personaplex_depth_input_check import exact
os.environ["OTPU_BUILD_JOBS"]="1"

EXPECTED_NATIVE_SOURCE = {'modules/transformer.py': '0eccf9b539ba738910d201a66809c68f5dad14f6b020fee862a592fd00e1de55', 'modules/gating.py': '008663b832e66088a087b1df4448e573f0dc94b01a9bbbb6a2206687abeb2f72', 'modules/streaming.py': '9fbdd77237d53ef6d95d1a08bce3b2438d661223ad8b795edfccc1b87eb4b81b', 'models/lm.py': '73d6da43ca59bb5162cfb97d8d05e71acaf214b89dd6705effa0c65da04c015b', 'models/loaders.py': '24d484efc2198c118f6d2b22877c765cd82e89887c52774c42aac5410b299148'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_layer(layer,spec):
    """Build independent original-source FP32 reference, never a production fallback."""
    import torch
    from moshi.modules.transformer import StreamingTransformerLayer
    if spec.layers != 1 or spec.cache_policy != "native" or spec.eps != 1e-8:
        raise ValueError("Native checker requires one layer, native cache policy and eps1e-8")
    model=StreamingTransformerLayer(d_model=spec.hidden,num_heads=spec.n_heads,
        dim_feedforward=3*spec.ffn//2,causal=True,context=None,rope=None,
        norm="rms_norm_f32",layer_scale=None,gating="silu",weights_per_step=spec.steps,
        device="cpu",dtype=torch.float32).eval()
    state={"norm1.alpha":torch.from_numpy(layer["norm1"]).reshape(1,1,spec.hidden),
           "norm2.alpha":torch.from_numpy(layer["norm2"]).reshape(1,1,spec.hidden)}
    banks=layer["steps"]
    state["self_attn.in_proj_weight"]=torch.from_numpy(np.concatenate([
        np.concatenate([bank[k] for k in ("wq","wk","wv")],axis=0) for bank in banks],axis=0))
    state["self_attn.out_proj.weight"]=torch.from_numpy(np.concatenate([b["wo"] for b in banks],axis=0))
    for step,bank in enumerate(banks):
        state[f"gating.{step}.linear_in.weight"]=torch.from_numpy(np.concatenate([bank["wg"],bank["wu"]],axis=0))
        state[f"gating.{step}.linear_out.weight"]=torch.from_numpy(bank["wd"])
    model.load_state_dict(state,strict=True)
    return model


def native_pass(model,inputs):
    import torch
    outputs,positions=[],[]
    with torch.no_grad(),model.streaming(1):
        cache=model.self_attn._streaming_state.kv_cache
        complete=cache.complete
        def record(k,v):
            result=complete(k,v)
            positions.append(result.positions.detach().cpu().tolist())
            return result
        cache.complete=record
        for row in inputs:
            outputs.append(model(torch.from_numpy(row).reshape(1,1,-1)).detach().numpy().reshape(-1).copy())
    return np.stack(outputs),positions


def check_native_positions(positions,steps):
    if len(positions)!=steps:
        raise AssertionError("Native cache did not record every depth step")
    for step,actual in enumerate(positions):
        expected=[i if i<=step else -1 for i in range(steps)]
        if step==steps-1:
            expected[0]=steps  # observed pinned RingKVCache full-capacity boundary
        if actual!=expected:
            raise AssertionError(f"Native cache position contract changed at step{step}")


def run(model_dir,conditioning_dir,output_dir,*,layer_index=0):
    import torch
    from moshi.modules.transformer import StreamingTransformerLayer
    torch.set_num_threads(1)
    model_dir,conditioning_dir,output_dir=(Path(p).resolve() for p in (model_dir,conditioning_dir,output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    for protected in (ROOT,model_dir,conditioning_dir):
        if output_dir==protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError("Output must be isolated from source, checkpoint and baseline")
    basepath=conditioning_dir/"depth_input_check.json"
    if basepath.stat().st_size>1<<20:
        raise ValueError("Conditioning receipt exceeds1MiB")
    baseline=json.loads(basepath.read_text())
    if (baseline.get("status")!="completed" or baseline.get("reset_bit_exact") is not True
            or baseline.get("checkpoint_revision")!=PINNED_REVISION):
        raise ValueError("A completed pinned conditioning baseline is required")
    inputs_path,requests_path=conditioning_dir/"conditioned.npy",conditioning_dir/"requests.npy"
    if (sha256(inputs_path)!=baseline["conditioned_output"]["sha256"]
            or sha256(requests_path)!=baseline["requests_output"]["sha256"]):
        raise ValueError("Saved conditioning arrays differ from receipt")
    values=np.load(inputs_path,allow_pickle=False);requests=np.load(requests_path,allow_pickle=False)
    if values.shape!=(20,1024) or values.dtype!=np.float32 or requests.shape!=(20,3) or requests.dtype.kind not in 'iu':
        raise ValueError("Expected saved20-case production conditioning outputs")
    if not np.isfinite(values).all() or set(map(int,requests[:16,0]))!=set(range(16)):
        raise ValueError("Invalid bounded conditioning input coverage")
    order=np.argsort(requests[:16,0]);inputs=values[:16][order].copy()
    weights=MLXWeights(model_dir);metadata=weights.depth_layer_metadata(layer_index)
    files={}
    for name in ("config.json","depformer.safetensors"):
        path=model_dir/name;stat=path.stat();files[name]={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
        if files[name]!=baseline["checkpoint_files"][name]:
            raise ValueError(f"Checkpoint metadata differs from baseline: {name}")
        with (model_dir/".cache/huggingface/download"/(name+".metadata")).open() as handle:
            if handle.readline().strip()!=PINNED_REVISION:
                raise ValueError(f"Checkpoint revision mismatch: {name}")
    native_root=Path(inspect.getfile(StreamingTransformerLayer)).parents[1]
    for name,digest in EXPECTED_NATIVE_SOURCE.items():
        if sha256(native_root/name)!=digest:
            raise ValueError(f"Native contract source changed: {name}")
    cfg=target.config(target.load(str(PROFILE)));spec=DepthLayerSpec()
    layout=DepthLayerImage(spec,cfg);program=layout.compile()
    six=DepthLayerImage(replace(spec,layers=6),cfg);six_program=six.compile()
    # CPU simulator may allocate only the required mapped range. Production
    # layout/compiler capacity above was checked against the recorded4GiB profile.
    isa_cfg=replace(cfg,DRAM_BYTES=layout.required_bytes)
    receipt={"status":"running","stage":"materializing_one_layer","started_utc":datetime.now(timezone.utc).isoformat(),
        "backend":"isa","hardware_used":False,"hardware_performance_measured":False,
        "full_fpga_model_inference":False,"speech_generated":False,"quality_accepted":False,
        "scope":"One depth transformer layer,16 device-selected banks and persistent KV, plus reset replay",
        "input_scope":"Reordered cached conditioning outputs from four saved temporal frames; bounded diagnostic sequence",
        "checkpoint_revision":PINNED_REVISION,"checkpoint_files":files,"layer_index":layer_index,
        "component_metadata":metadata,"checkpoint_payload_hashes_verified":False,
        "baseline_receipt_sha256":sha256(basepath),"baseline_input_sha256":sha256(inputs_path),
        "baseline_requests_sha256":sha256(requests_path),"input_source_case_order":order.tolist(),
        "native_source_sha256":EXPECTED_NATIVE_SOURCE,"spec":asdict(spec),
        "required_bytes":layout.required_bytes,"weight_bytes":layout.weight_bytes,"kv_bytes":layout.kv_bytes,
        "program_words":len(program[0])*8,"physical_profile_dram_bytes":cfg.DRAM_BYTES,
        "isa_allocated_dram_bytes":isa_cfg.DRAM_BYTES,
        "six_layer_capacity_only":{"required_bytes":six.required_bytes,"weight_bytes":six.weight_bytes,
            "kv_bytes":six.kv_bytes,"program_words":len(six_program[0])*8,"checkpoint_execution_tested":False},
        "precision":{"weights":"Source MLX affine INT4/group64 -> FP32 offline -> MX INT8",
            "activations":"INT8 QACT","kv":"INT8 with scales","residual_norm_softmax":"FP32/FTZ ISA",
            "native_reference":"Original dequantized-source FP32 PyTorch layer, no requantization"},
        "cache_policy":"native: no context8 eviction; laststep slot0 excluded by pinned source boundary bug",
        "frames":[],"reset_frames":[],"source_files":{str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__),ROOT/"opentpu/llm/personaplex_depth_layer.py",ROOT/"opentpu/llm/personaplex_weights.py",
             ROOT/"opentpu/llm/personaplex_audio_heads.py",ROOT/"tools/personaplex_depth_input_check.py"]}}
    output_dir.mkdir(parents=True,exist_ok=False);receipt_path=output_dir/"depth_layer_check.json"
    def save():
        temp=receipt_path.with_suffix('.json.tmp');temp.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');temp.replace(receipt_path)
    def interrupted(signum,frame):
        raise InterruptedError("Depth layer check interrupted; inspect receipt before retry")
    previous=signal.signal(signal.SIGTERM,interrupted);engine=None;started=time.perf_counter();save()
    try:
        materialize=time.perf_counter();lazy=weights.depth_layer_weights(layer_index)
        layer={"norm1":lazy["norm1"],"norm2":lazy["norm2"],"steps":[{k:bank[k] for k in bank} for bank in lazy["steps"]]}
        receipt["source_materialize_cpu_seconds"]=time.perf_counter()-materialize
        receipt['stage']='native_reference';save();native=native_layer(layer,spec)
        original,positions=native_pass(native,inputs);check_native_positions(positions,spec.steps)
        receipt['native_cache_positions']=positions
        original_repeat,_=native_pass(native,inputs);exact(original_repeat,original,'native reset')
        receipt['native_reset_bit_exact']=True;del native
        receipt['stage']='building_isa_image';save();build=time.perf_counter()
        engine=DepthLayerEngine([layer],spec=spec,cfg=isa_cfg)
        receipt['image_build_cpu_seconds']=time.perf_counter()-build
        actual=[]
        for step,row in enumerate(inputs):
            receipt.update(stage='simulating_step',step=step);save();engine.step(row)
            value=engine.read_output()
            if engine.read_counter()!=step+1:
                raise AssertionError('Device counter did not advance exactly once')
            actual.append(value);receipt['frames'].append({'step':step,'precision':precision_metrics(value,original[step]),'stats':engine.last_stats});save()
        actual=np.stack(actual);cached=engine.read_cache();receipt['precision_delta']=precision_metrics(actual,original)
        engine.reset()
        if engine.read_counter()!=0:
            raise AssertionError('Device counter reset failed')
        for step,row in enumerate(inputs):
            receipt.update(stage='reset_replay',step=step);save();engine.step(row)
            exact(engine.read_output(),actual[step],f'ISA reset step{step}')
            if engine.read_counter()!=step+1:
                raise AssertionError('Device reset counter diverged')
            receipt['reset_frames'].append({'step':step,'stats':engine.last_stats})
        if not np.array_equal(engine.read_cache(),cached):
            raise AssertionError('Full16-step reset KV bytes differ')
        receipt.update(reset_bit_exact=True,kv_reset_bit_exact=True)
        for name,array in [('inputs',inputs),('isa_hidden',actual),('native_hidden',original),('isa_kv',cached)]:
            path=output_dir/(name+'.npy')
            with path.open('xb') as handle:np.save(handle,array,allow_pickle=False)
            receipt[name+'_output']={'sha256':sha256(path),'shape':list(array.shape),'dtype':str(array.dtype)}
        if (sha256(basepath)!=receipt['baseline_receipt_sha256'] or sha256(inputs_path)!=receipt['baseline_input_sha256']
                or sha256(requests_path)!=receipt['baseline_requests_sha256']):raise RuntimeError('Baseline changed during check')
        for name,before in files.items():
            stat=(model_dir/name).stat()
            if {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=before:raise RuntimeError('Checkpoint changed during check')
        for name,digest in EXPECTED_NATIVE_SOURCE.items():
            if sha256(native_root/name)!=digest:raise RuntimeError('Native source changed during check')
        receipt.update(status='completed',stage='completed')
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());raise
    finally:
        try:
            if engine is not None:engine.close()
        finally:
            signal.signal(signal.SIGTERM,previous);receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),cpu_wall_seconds=time.perf_counter()-started);save()
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True);p.add_argument('--conditioning-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--layer-index',type=int,default=0);a=p.parse_args()
    result=run(a.model_dir,a.conditioning_dir,a.output_dir,layer_index=a.layer_index)
    print(json.dumps({k:result[k] for k in ['status','precision_delta','reset_bit_exact','kv_reset_bit_exact','native_reset_bit_exact','cpu_wall_seconds']},indent=2))


if __name__=='__main__':main()
