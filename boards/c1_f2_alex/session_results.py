"""Offline C1 result-field replay only: no session acceptance, profiles, or IO."""
from datetime import datetime
import hashlib,json
from opentpu import c1_f2 as C

def need(ok,message):
    if not ok:raise ValueError(message)

def checked_result_facts(result,fixture,*,expected_identity,expected_device,repeat):
    """Caller must separately prove qualification, source, capture and restoration.

    expected_identity must ultimately come from verified C1 qualification. This
    helper cannot turn caller-supplied identity or result flags into provenance.
    """
    fixture.validate()
    keys={'id','abi','build_id','core_khz','max_rows','depth','credits'}
    need(set(expected_identity)==keys and all(type(v)is int for v in expected_identity.values()),'identity shape')
    need(expected_identity['id']==C.ID and expected_identity['abi']==C.ABI and
         0<=expected_identity['build_id']<2**32 and expected_identity['core_khz']==200000 and
         expected_identity['max_rows']==512 and expected_identity['depth']==128 and
         1<=expected_identity['credits']<=32,'C1 identity/geometry')
    actual_identity=result.get('identity',{})
    need(isinstance(actual_identity,dict) and set(actual_identity)==keys and all(type(v)is int for v in actual_identity.values()),'raw identity types')
    need(isinstance(expected_device,str) and bool(expected_device.strip()) and isinstance(result.get('device'),str) and bool(result['device'].strip()),'device string')
    need(actual_identity==expected_identity and result['device']==expected_device,'image/device mismatch')
    need(all(type(result.get(k))is int for k in ('mode','repeat','status','error','rows','kb','base_beat','weight_bytes')),'raw integer field types')
    need(result.get('evidence_kind')=='f2-hardware','not physical evidence')
    need(result.get('bit_exact') is True and result.get('status')==C.DONE and result.get('error')==0,'unsuccessful result')
    need(type(repeat)is int and repeat>=0 and result.get('repeat')==repeat and result.get('mode')==0,'repeat/mode')
    need((result.get('rows'),result.get('kb'),result.get('base_beat'))==(fixture.rows,fixture.kb,fixture.base_beat),'fixture geometry')
    need(result.get('fixture_sha256')==fixture.digest(),'fixture identity')
    need(isinstance(result.get('bits'),list) and all(type(v)is int and 0<=v<2**32 for v in result['bits']),'result bit types/range')
    need(isinstance(result.get('integers'),list) and all(type(v)is int and -(2**47)<=v<2**47 for v in result['integers']),'result integer types/range')
    bits,integers=fixture.reference()
    need(result.get('bits')==bits.tolist() and result.get('integers')==integers,'canonical arithmetic')
    cycles=result.get('cycles');need(type(cycles)is int and 0<cycles<2**32,'invalid/wrapped cycles')
    need(datetime.fromisoformat(result['measured_at_utc']).utcoffset() is not None,'measurement timezone')
    need(result.get('measured_window')=='COMMAND acceptance through final fp32 result; excludes OCL preload, ACT snapshot copy and host readback','measurement window')
    counters=result.get('counters',{});pcs=result.get('pcs',[])
    need(set(counters)==set(C.COUNTERS) and len(pcs)==8,'counter inventory')
    need(all(type(v)is int and 0<=v<2**32 for v in counters.values()),'invalid counter')
    groups=fixture.rows*fixture.kb*4;beats=fixture.expected_pc_beats();credits=expected_identity['credits']
    need(counters['accepted_32B_groups']==counters['advance']==groups and counters['committed_rows']==fixture.rows,'retirement counts')
    need(counters['read_issued']==counters['read_returned']==sum(beats) and result.get('weight_bytes')==sum(beats)*64,'native traffic')
    need(counters['max_outstanding']<=16*credits,'aggregate reservation bound')
    need(all(counters[k]==0 for k in ('scale_wait','act_wait','output_wait')),'ABI1 reserved counters')
    need(all(counters[k]<=cycles for k in ('weight_wait','native_blocked_cycles')),'cycle counter bound')
    for pc,row in enumerate(pcs):
        need(set(row)==set(C.PC_COUNTERS) and all(type(v)is int and 0<=v<2**32 for v in row.values()),'per-PC inventory/range')
        need(row['issued']==row['returned']==beats[pc] and row['max_outstanding']<=2*credits,'per-PC traffic/credits')
        need(row['reserved_latency']==0 and row['blocked_cycles']<=cycles and row['credit_wait_cycles']<=cycles,'per-PC reserved/cycle counters')
    return dict(kind='c1-checked-result-fields-only',raw_result_sha256=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                cycles=cycles,clock_mhz=200.0,rows=fixture.rows,kb=fixture.kb,accepted_32B_groups=groups,
                macs=fixture.rows*fixture.kb*128,native_read_bytes=sum(beats)*64,native_interface_floor_cycles=max(beats),
                evidence_kind='f2-hardware',session_verified=False,route_qualified=False,lens_append_authorized=False)
