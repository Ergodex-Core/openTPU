"""C1 graph replay boundary tests, not hardware or timing evidence."""
import copy
import hashlib
from pathlib import Path
import pytest
from boards.c1_f2_alex import native_review as N, ready_review as R


def minimal():
    """17 explicitly observed constant-root graphs: valid raw facts, no acceptance."""
    e=R.Evidence();owner=N.ROOT+'GND';driver=owner+'/G'
    e.objects['cell',owner]={'NAME':owner,'REF_NAME':'GND','IS_PRIMITIVE':'1','IS_BLACKBOX':'0'}
    e.objects['pin',driver]={'NAME':driver,'REF_PIN_NAME':'G','DIRECTION':'OUT'}
    e.queries['select','cell',owner]=[owner]
    e.queries['select','pin',driver]=[driver]
    e.queries['pin_owner',driver]=[owner]
    host=N.ROOT+'u_pcis_pipe/m_axi_wready'
    roots={'host_wready':host}
    for pc in range(8):
        si=f's{pc+1:02d}'
        for channel in ('b','r'):
            roots[f'native {pc} {channel}']=f'{N.SC}{si}_entry_pipeline/{si}_mmu/inst/{channel}_sreg/s_axi_{channel}ready'
    for index,(label,pin) in enumerate(roots.items()):
        net=N.ROOT+f'local{index}'
        e.ready['root'][label]={'pin':pin}
        e.queries['select','pin',pin]=[pin]
        e.queries['segments',pin]=[net]
        e.queries['select','net',net]=[net]
        e.queries['leaves',net]=[driver]
        e.queries['ports',net]=[]
        e.networks['pin',pin]={'nets':[net],'leaves':[driver],'drivers':[driver],'ports':[]}
        e.ready['walk'][label]={'outputs':[driver],'registers':[],'expand_controls':'1'}
        e.ready['terminal'][label,owner,driver]={'cell':owner,'output':driver,'reference':'GND'}
    e.hosts=[[host]]
    for scope in (N.SC+'s00_entry_pipeline/s00_mmu/inst/w_sreg',N.ROOT+'u_pcis_pipe'):
        e.queries['ready_host_register_inventory',scope]=[]
        e.ready['candidates'][scope]={'scope':scope,'all':[],'selected':[],'selection':'name_contains_ready'}
    return e


def test_complete_raw_graph_is_observational_not_equivalence():
    facts=R.facts(minimal())
    assert len(facts['roots'])==17 and 'pending' in facts['limitation']
    assert not facts['registers']


@pytest.mark.parametrize('mutation',[
    lambda e:e.ready['root'].pop('native 7 r'),
    lambda e:e.ready['walk'].pop('host_wready'),
    lambda e:e.ready['terminal'].clear(),
    lambda e:e.ready['candidates'].clear(),
    lambda e:e.hosts.append(e.hosts[0]),
    lambda e:e.queries.__setitem__(('leaves',N.ROOT+'local0'),[]),
])
def test_missing_actual_evidence_fails(mutation):
    e=minimal();mutation(e)
    with pytest.raises(ValueError):R.facts(e)


def test_control_record_parser_preserves_raw_scope_and_duplicate_conflicts():
    e=R.Evidence()
    e.add(N.tcl_list('header schema 1 kind c1_ready_topology checkpoint_sha256 '+'a'*64+
                     ' checkpoint /source.dcp route_qualified 0 hardware_measured 0'))
    line='ready_control host_wready cell cl/state role D network {scope selected_input_and_drivers input cl/state/D properties {NAME cl/state/D DIRECTION IN} nets n drivers g/G ports {}}'
    e.add(N.tcl_list(line))
    assert e.ready['control']['host_wready','cl/state','D']['network']['drivers']==['g/G']
    e.add(N.tcl_list(line.replace('drivers g/G','drivers v/P')))
    assert e.issues[0]['reason']=='conflicting_duplicate'


@pytest.mark.parametrize('marker',[
    'incomplete capture reason node_cap',
    'selection pin missing status missing_or_ambiguous observed {}',
])
def test_capture_gaps_never_produce_facts(tmp_path,marker):
    raw=('header schema 1 kind c1_ready_topology checkpoint_sha256 '+ 'a'*64+
         ' checkpoint /source.dcp route_qualified 0 hardware_measured 0\n'
         'ready_sources helper_sha256 '+ 'b'*64+' native_helper_sha256 '+'c'*64+'\n'+marker+'\n'
         'summary queries 0 incomplete_or_error 0 output_byte_cap 0 route_qualified 0 hardware_measured 0\n')
    path=tmp_path/'raw';path.write_text(raw)
    result=R.review_artifact(topology_path=path,topology_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    assert result['issues'] and not result['canonical_facts'] and result['route_qualified'] is False


def test_wrong_hash_rejected_before_parse(tmp_path):
    path=tmp_path/'raw';path.write_text('not evidence')
    result=R.review_artifact(topology_path=path,topology_sha256='0'*64,
        checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    assert 'hash mismatch' in result['issues'][0]['reason']
