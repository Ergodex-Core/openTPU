"""Exact frozen C1 source specialization, conditional on source/hardware fidelity.

Proves only disabled writes, the address high-bit bound and occupied command
FIFO/pending-address high-bit invariants after reset. Unwritten FIFO entries are
never assumed initialized. This is not whole-native RTL/netlist equivalence.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from . import native_review as N

HARDWARE_REVISION = '0d654453e2c77ea9bc9d0e854c33d9166bb27375'
SYNTH_INPUT_SHA256 = '3dc8cdb3f1106c25ce5dcd72d17e9ea5c967b327cd98fc7150c3d45df2d85b87'
TOP_SHA256 = 'adce0a570cea1a249f9b00653d274a58ed287b148420d022492e04176a8ac22a'
SOURCE_SHA256 = {
    'rtl/c1/otpu_c1_hbm_kernel.sv': '249f61da5bf6178ae3329e179483eeab8671aac1bacec4724d5f11eb9b41bd24',
    'rtl/c1/otpu_c1_native.sv': '835490b3ae1e7f90d0fe4490568857731ee4e9e0c4affa9f12dfeb4fd655a2e3',
    'boards/c1_f2_alex/support/rtl/boards/f2/otpu_native_axi.sv': '9219ade00e16bf1b3a200167bd90a44283a606842a6477ce75f9b3eae7e14f01',
    'rtl/c1_alex/generate.py': '6fda6d36a4afd945b2d70685a168dfcab849221421208606d312bed444409ee6',
}
CONFIGURATION = {'QD': 16, 'MAX_OUT': 128, 'READ_BURST': 16, 'READ_IDLE': 4}


def unique(pairs):
    result = {}
    for key, value in pairs:
        N.require(key not in result, 'duplicate source input key'); result[key] = value
    return result


def compact(text):
    return re.sub(r'\s+', '', re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S))


def address_bound(row_width, group_width, shift, output_width):
    N.require((row_width, group_width, shift, output_width) == (23, 10, 1, 25), 'unreviewed source address expression widths')
    maximum = ((1 << row_width)-1) + (((1 << group_width)-1) >> shift)
    N.require(maximum < 1 << (output_width-1), 'source address high bit is not zero')
    return {'row_width': row_width, 'group_width': group_width, 'logical_right_shift': shift,
            'output_width': output_width, 'maximum': maximum, 'bit_24': 0,
            'bit_23_is_not_assumed_zero': True}


def fifo_step(cr, count, push, pop, qd):
    """One exact bounded source update, including simultaneous read/write."""
    N.require(qd == 16 and type(cr) is int and 0 <= cr < qd and type(count) is int and 0 <= count <= qd, 'invalid occupied FIFO state')
    N.require(type(push) is bool and type(pop) is bool and (not push or count < qd) and (not pop or count > 0), 'illegal source FIFO handshake')
    cp = (cr + count) % qd
    occupied = {(cr+i) % qd for i in range(count)}
    N.require(not pop or cr in occupied, 'popped head not known written')
    known_zero = occupied | ({cp} if push else set())
    after_cr, after_cp = (cr+int(pop)) % qd, (cp+int(push)) % qd
    after_count = count+int(push)-int(pop)
    after_occupied = {(after_cr+i) % qd for i in range(after_count)}
    N.require(0 <= after_count <= qd and after_cp == (after_cr+after_count) % qd and after_occupied <= known_zero, 'occupied FIFO induction failed')
    return {'cr': after_cr, 'cp': after_cp, 'count': after_count, 'occupied': sorted(after_occupied)}


def prove_fifo():
    transitions = 0
    # cp is determined by cr/count. Each abstract occupied slot carries known
    # zero high/write bits. All other slots can have arbitrary prior contents.
    for cr in range(16):
        for count in range(17):
            for push in (False, True):
                for pop in (False, True):
                    if push and count == 16 or pop and count == 0: continue
                    fifo_step(cr, count, push, pop, 16); transitions += 1
    N.require(fifo_step(0, 0, False, False, 16)['occupied'] == [], 'reset occupancy is not empty')
    # pending_addr[24] resets to zero, holds, or loads an occupied FIFO head.
    # The load is inside take_read, whose source guard requires cn != 0.
    for cr in range(16):
        for count in range(1,17):
            occupied={(cr+i)%16 for i in range(count)}
            N.require(cr in occupied, 'pending address load lacks valid head')
    return {'qd': 16, 'abstract_states': 272, 'legal_transition_cases': transitions,
            'reset': {'cp': 0, 'cr': 0, 'cn': 0},
            'invariant': 'cp=(cr+cn) modulo16; 0<=cn<=16; every occupied entry has been written with cq[25:24]=0',
            'unwritten_entries_assumed_zero': False, 'pending_addr_bit_24_after_reset': 0,
            'scope': 'source two-state synchronous transitions after reset; no RTL/netlist state equivalence claim'}


def source_facts(texts, top):
    """Derive the reviewed specialization only for the hash-bound source dialect."""
    kernel, native, adapter = (compact(texts[name]) for name in list(SOURCE_SHA256)[:3])
    N.require("assignn_cwe='0;assignn_wvalid='0;" in kernel, 'disabled-write source connection differs')
    N.require('logic[15:0][9:0]group_id;' in native and 'logic[15:0][22:0]row_base;' in native and
              "n_caddr[p]={2'b0,row_base[2*p+int'(selected[p])]}+25'(group_id[2*p+int'(selected[p])]>>1);" in native,
              'native address width/expression differs')
    for name, value in CONFIGURATION.items():
        N.require(re.search(r'parameter(?:int)?'+name+'='+str(value)+r'[,)]', adapter) is not None, 'native default parameter differs: '+name)
    anchors = ["wirepushc=n_cvalid&&n_cready;", "assignn_cready=!rst&&cn<QD;", "wirepopc=take_read||popw;",
               "wiretake_read=!pending_valid&&cn!=0&&!head_we&&writes==0&&reads<MAX_OUT&&(pending_count==0||adjacent);",
               "wirewrite_go=cn!=0&&head_we&&wn!=0&&reads==0&&writes<MAX_OUT;",
               "wirepopw=write_go&&(aw_sent||aw_fire)&&(w_sent||w_fire);",
               "if(pushc)begincq[cp]<={n_cwe,n_caddr};cp<=cp+1'b1;end", "if(popc)cr<=cr+1'b1;",
               "case({pushc,popc})2'b10:cn<=cn+1'b1;2'b01:cn<=cn-1'b1;default:;endcase",
               "cp<=0;cr<=0;wp<=0;wr<=0;cn<=0;wn<=0;", "pending_addr<=0;next_addr<=0;pending_count<=0;",
               "endelseif(take_read)beginif(pending_count==0)pending_addr<=cq[cr][24:0];"]
    N.require(all(adapter.count(anchor)==1 for anchor in anchors), 'source FIFO/guard/update structure differs')
    N.require(len(re.findall(r'pending_addr<=',adapter))==2 and len(re.findall(r'cq\[[^]]+\]<=',adapter))==1, 'unreviewed source high-bit update')
    # The entire generated top hash is pinned. These assertions make the exact
    # parameter and direct signal wiring premises visible in the proof record.
    top = compact(top)
    for pc in range(8):
        prefix=f"otpu_native_axi#(.BASE_ADDR(34'd0))u_native{pc}("
        N.require(top.count(prefix)==1, 'generated native parameter/instance differs')
        body=top.split(prefix,1)[1].split(');',1)[0]
        N.require(all(body.count(f'.{role}({role}[{pc}])')==1 for role in ('n_cwe','n_wvalid','n_caddr','n_cvalid','n_cready')),
                  'generated native direct wiring differs')
        N.require('.clk(core_clk)' in body and '.rst(core_rst)' in body, 'generated native reset/clock differs')
    N.require(top.count(')u_board(')==1 and '.n_cwe(n_cwe)' in top and '.n_wvalid(n_wvalid)' in top and '.n_caddr(n_caddr)' in top,
              'generated kernel native output wiring differs')
    N.require(all((value<128)==not_bit7(value) for value in range(256)), 'unsigned read threshold specialization failed')
    return {'configuration': dict(CONFIGURATION), 'disabled_write_signals': {'n_cwe': 0, 'n_wvalid': 0},
            'address': address_bound(23,10,1,25), 'fifo': prove_fifo(),
            'equation_substitutions': {'head_we_for_nonempty_fifo': 0, 'cq_head_bit_24_for_nonempty_fifo': 0,
                                       'pending_addr_bit_24_after_reset': 0, 'reads_lt_128': '!reads[7]'},
            'unwritten_command_memory_contents': 'unconstrained', 'whole_native_state_equivalence_proven': False}


def not_bit7(value): return not bool(value & 128)


def review_sources(*, source_paths, synth_inputs_path, synth_inputs_sha256, generated_top_path):
    result={'schema':1,'kind':'c1-admission-source-specialization-review','source_elided_bits_proven':False,
            'source_hardware_equivalence_proven':False,'route_qualified':False,'hardware_measured':False,
            'issues':[],'canonical_facts':{},'pending_reviews':['native_state_transition_correspondence','linked_final_accounting_preservation','admission_ram_state_address_control_preservation']}
    try:
        N.require(set(source_paths)==set(SOURCE_SHA256), 'source specialization inventory differs')
        N.require(synth_inputs_sha256==SYNTH_INPUT_SHA256, 'unreviewed frozen synthesis input identity')
        before={name:N.identity(path,SOURCE_SHA256[name]) for name,path in source_paths.items()}
        input_record=N.identity(synth_inputs_path,SYNTH_INPUT_SHA256)
        top_record=N.identity(generated_top_path,TOP_SHA256)
        inputs=json.loads(Path(synth_inputs_path).read_bytes(),object_pairs_hook=unique)
        N.require(inputs['schema']==1 and inputs['kind']=='c1-alex-burst200-build-inputs' and inputs['source_revision']==HARDWARE_REVISION and inputs['build_id']==HARDWARE_REVISION[:8] and inputs['source_status']=='', 'frozen hardware input source binding differs')
        for name,row in before.items():
            if name.startswith('boards/'):
                for prefix in ('source/rtl/boards/f2/','synth/rtl/','link/rtl/'):
                    N.require(inputs['prepared_files'][prefix+'otpu_native_axi.sv']==row, 'frozen adapter prepared source differs')
            else:
                N.require(inputs['source_inputs'][name]==row, 'frozen hardware/generator input differs')
                if name.endswith('.sv'):N.require(inputs['prepared_files']['source/'+name]==row, 'frozen core prepared source differs')
        for prefix in ('source/rtl/boards/f2/','synth/rtl/','link/rtl/'):
            N.require(inputs['prepared_files'][prefix+'cl_opentpu.sv']==top_record, 'generated top differs from frozen synthesis input')
        texts={name:Path(path).read_text() for name,path in source_paths.items()}
        facts=source_facts(texts,Path(generated_top_path).read_text())
        N.require(before=={name:N.identity(path,SOURCE_SHA256[name]) for name,path in source_paths.items()} and N.identity(synth_inputs_path,SYNTH_INPUT_SHA256)==input_record and N.identity(generated_top_path,TOP_SHA256)==top_record, 'source specialization files changed')
        facts.update(hardware_revision=HARDWARE_REVISION,source_files=before,synthesis_inputs=input_record,generated_top=top_record)
        result.update(source_elided_bits_proven=True,canonical_facts=facts)
        result['facts_sha256']=hashlib.sha256(json.dumps(facts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    except (ValueError,TypeError,KeyError,IndexError,OSError) as exc:result['issues'].append({'reason':str(exc)})
    return result
