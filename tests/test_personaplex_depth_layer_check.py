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


def test_native_stack_uses_original_container_order_independent_caches_and_reset():
    from contextlib import ExitStack
    import torch
    from moshi.modules.transformer import StreamingTransformer
    from tools.personaplex_depth_layer_check import native_stack, native_stack_pass
    spec=DepthLayerSpec(hidden=32,n_heads=2,ffn=64,steps=4,layers=2)
    rng=np.random.default_rng(867)
    payload=[]
    for index in range(spec.layers):
        layer=zero_layer(spec)
        layer['norm1']*=1+index/4
        for bank in layer['steps']:
            for name,value in bank.items():bank[name]=rng.normal(0,.05,value.shape).astype(np.float32)
        payload.append(layer)
    inputs=rng.normal(size=(4,32)).astype(np.float32)
    model=native_stack(payload,spec)
    assert isinstance(model,StreamingTransformer)
    actual,positions,intermediate=native_stack_pass(model,inputs)
    for history in positions:check_native_positions(history,4)
    one=replace(spec,layers=1)
    independent=[native_layer(layer,one) for layer in payload]
    expected=[];first=[]
    with torch.no_grad(),ExitStack() as context:
        for layer in independent:context.enter_context(layer.streaming(1))
        for row in inputs:
            value=torch.from_numpy(row).reshape(1,1,-1)
            for index,layer in enumerate(independent):
                value=layer(value)
                if index==0:first.append(value.detach().numpy().reshape(-1).copy())
            expected.append(value.detach().numpy().reshape(-1).copy())
    np.testing.assert_array_equal(actual.view(np.uint32),np.stack(expected).view(np.uint32))
    np.testing.assert_array_equal(intermediate[:,0,:].view(np.uint32),np.stack(first).view(np.uint32))
    replay,repeated,replayed_intermediate=native_stack_pass(model,inputs)
    np.testing.assert_array_equal(replay.view(np.uint32),actual.view(np.uint32))
    np.testing.assert_array_equal(replayed_intermediate.view(np.uint32),intermediate.view(np.uint32))
    assert repeated==positions
    assert all(not layer._forward_hooks for layer in model.layers)


@pytest.mark.parametrize('count',[0,2])
def test_native_stack_rejects_payload_count(count):
    from tools.personaplex_depth_layer_check import native_stack
    spec=DepthLayerSpec(hidden=32,n_heads=2,ffn=64,steps=4)
    with pytest.raises(ValueError,match='layer count'):
        native_stack([zero_layer(spec)]*count,spec)


@pytest.mark.parametrize('kwargs',[
    {'layers':2}, {'layers':0}, {'layers':6,'layer_index':1}, {'layers':6},
])
def test_unbounded_or_unbased_stack_check_is_rejected(tmp_path,kwargs):
    with pytest.raises(ValueError,match='Checker supports|requires its completed'):
        run(tmp_path/'model',tmp_path/'conditioning',tmp_path/'out',**kwargs)
    assert not (tmp_path/'out').exists()
