"""Independent adversarial inventory tests; no fixture grants semantic approval."""
import importlib.util
from pathlib import Path
import pytest
from boards.c1_f2_alex import admission_review as R, native_review as N

_spec=importlib.util.spec_from_file_location('admission_peer_fixture',Path(__file__).with_name('test_c1_admission_review.py'))
T=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(T)


def assert_rejected(tmp_path,e):
    result=T.replay(tmp_path,T.serialize(e))
    assert result['issues'] and result['capture_complete'] is False
    assert result['functional_semantics_proven'] is False
    assert result['paired_preservation_proven'] is False
    assert result['route_qualified'] is False


def change_pin_reference(e,pin,reference):
    """Alter all repeated property observations coherently, not only one copy."""
    e.objects['pin',pin]['REF_PIN_NAME']=reference
    for tag in ('root','edge'):
        for row in e.admission[tag].values():
            network=N.td(row['network'])
            if network.get('input')==pin:
                properties=N.td(network['properties']);properties['REF_PIN_NAME']=reference
                network['properties']=T.td(properties);row['network']=T.td(network)


def test_coherent_missing_fdre_d_inventory_rejects(tmp_path):
    e=T.fixture();pc='0';cell=N.ROOT+'u_native0/state';pin=cell+'/D';output=cell+'/Q'
    # Every repeated query/witness agrees on the same incomplete primitive.
    e.cells[cell].remove(pin)
    e.queries['admission_inputs',pc,cell].remove(pin)
    e.admission['cell'][pc,cell]['inputs']=T.seq(e.queries['admission_inputs',pc,cell])
    del e.admission['edge'][pc,pin]
    e.networks['pin',output]['leaves'].remove(pin)
    e.admission['output'][pc,output]['network']=T.td({k:T.seq(v) for k,v in e.networks['pin',output].items()})
    assert_rejected(tmp_path,e)


@pytest.mark.parametrize('pin',[N.ROOT+'u_native0/state/Q',N.ROOT+'u_native0/state/C',N.ROOT+'u_native0/state/CE',N.ROOT+'u_native0/state/D',N.ROOT+'u_native0/state/R',N.ROOT+'external/shared/clk/Q',N.ROOT+'u_mem/reads[7]_i_1/I1'])
def test_coherent_false_reference_pin_rejects(tmp_path,pin):
    e=T.fixture();change_pin_reference(e,pin,'NOT_THE_SELECTED_PIN');assert_rejected(tmp_path,e)


@pytest.mark.parametrize('mutation',['unknown_state','outside_state','missing_selected_output','ambiguous_owner','nonprimitive_state','opaque_state','output_wrong_direction','extra_interface_port','interface_multidriver','extra_state_pin'])
def test_independent_typed_object_mutations_reject(tmp_path,mutation):
    e=T.fixture();pc='0';cell=N.ROOT+'u_native0/state';out=cell+'/Q'
    if mutation=='unknown_state':
        e.objects['cell',cell]['REF_NAME']='RAMB36E2';e.admission['cell'][pc,cell]['reference']='RAMB36E2'
    elif mutation=='outside_state':
        text=T.serialize(e).replace(cell,N.ROOT+'outside0/state')
        result=T.replay(tmp_path,text)
        assert result['issues'] and not result['capture_complete'] and not result['functional_semantics_proven']
        return
    elif mutation=='missing_selected_output':e.queries['select','pin',out]=[]
    elif mutation=='ambiguous_owner':e.queries['pin_owner',out]=[cell,N.ROOT+'another']
    elif mutation=='nonprimitive_state':e.objects['cell',cell]['IS_PRIMITIVE']='0'
    elif mutation=='opaque_state':e.objects['cell',cell]['IS_BLACKBOX']='1'
    elif mutation=='output_wrong_direction':e.objects['pin',out]['DIRECTION']='IN'
    elif mutation=='extra_interface_port':
        row=e.admission['interface'][pc,'clk'];net=row['name'];e.queries['admission_net_ports',net]=['outside_port']
        network=N.td(row['network']);network['ports']=T.seq(['outside_port']);row['network']=T.td(network)
    elif mutation=='interface_multidriver':
        row=e.admission['interface'][pc,'clk'];net=row['name'];e.queries['admission_net_drivers',net].append(N.ROOT+'GND/G')
        network=N.td(row['network']);network['drivers']=T.seq(e.queries['admission_net_drivers',net]);row['network']=T.td(network)
    elif mutation=='extra_state_pin':
        pin=cell+'/UNEXPECTED';e.cells[cell].append(pin);e.queries['admission_inputs',pc,cell].append(pin)
        e.admission['cell'][pc,cell]['inputs']=T.seq(e.queries['admission_inputs',pc,cell])
        e.queries['select','pin',pin]=[pin];e.queries['pin_owner',pin]=[cell]
        e.objects['pin',pin]=dict(NAME=pin,REF_PIN_NAME='UNEXPECTED',DIRECTION='IN',IS_INVERTED='0',IS_CLOCK='0')
    assert_rejected(tmp_path,e)


def test_successful_inventory_still_leaves_every_semantic_obligation_pending(tmp_path):
    result=T.replay(tmp_path,T.serialize(T.fixture()))
    assert result['capture_complete'] and result['issues']==[]
    assert result['functional_semantics_proven'] is result['paired_preservation_proven'] is result['route_qualified'] is result['hardware_measured'] is False
    assert set(result['pending_reviews'])=={'take_read_admission_semantics','linked_final_accounting_preservation','admission_ram_state_address_control_preservation'}
