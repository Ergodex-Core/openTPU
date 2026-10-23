"""Native layer construction/cache contract and receipt protection; EC2 only."""
from dataclasses import replace
import numpy as np
import pytest
from opentpu.llm.personaplex_depth_layer import DepthLayerSpec
from tools.personaplex_depth_layer_check import native_layer, native_pass, check_native_positions, run


def zero_layer(spec):
    shapes={name:(spec.hidden,spec.hidden) for name in ('wq','wk','wv','wo')}
    shapes.update(wg=(spec.ffn,spec.hidden),wu=(spec.ffn,spec.hidden),wd=(spec.hidden,spec.ffn))
    return {'norm1':np.ones(spec.hidden,np.float32),'norm2':np.ones(spec.hidden,np.float32),
            'steps':[{key:np.zeros(shape,np.float32) for key,shape in shapes.items()} for _ in range(spec.steps)]}


def test_native_reference_matches_residual_and_observes_real_ring_boundary_reset():
    spec=DepthLayerSpec(hidden=32,n_heads=2,ffn=64,steps=4)
    native=native_layer(zero_layer(spec),spec)
    inputs=np.arange(4*32,dtype=np.float32).reshape(4,32)/32
    first,positions=native_pass(native,inputs)
    np.testing.assert_array_equal(first,inputs)  # no output/final normalization
    check_native_positions(positions,4)
    assert positions==[[0,-1,-1,-1],[0,1,-1,-1],[0,1,2,-1],[4,1,2,3]]
    again,repeated=native_pass(native,inputs)
    np.testing.assert_array_equal(again.view(np.uint32),first.view(np.uint32))
    assert repeated==positions


@pytest.mark.parametrize('change',[{'layers':2},{'cache_policy':'causal'},{'eps':1e-5}])
def test_native_checker_rejects_unverified_reference_configuration(change):
    spec=DepthLayerSpec(hidden=32,n_heads=2,ffn=64,steps=4)
    with pytest.raises(ValueError,match='Native checker'):
        native_layer(zero_layer(spec),replace(spec,**change))


@pytest.mark.parametrize('bad',[
    [[0,-1,-1,-1]],
    [[0,-1,-1,-1],[0,1,-1,-1],[0,1,2,-1],[0,1,2,3]],
    [[0,-1,-1,-1],[0,2,-1,-1],[0,1,2,-1],[4,1,2,3]],
])
def test_native_contract_guard_detects_changed_capacity_or_boundary(bad):
    with pytest.raises(AssertionError):check_native_positions(bad,4)


def test_existing_receipt_is_never_overwritten(tmp_path):
    saved=tmp_path/'prior';saved.mkdir()
    receipt=saved/'depth_layer_check.json';receipt.write_text('preserved')
    with pytest.raises(FileExistsError):run(tmp_path/'model',tmp_path/'baseline',saved)
    assert receipt.read_text()=='preserved'
