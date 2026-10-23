"""Native-method interception and diagnostic receipt safety, EC2 execution only."""
from types import SimpleNamespace
import numpy as np
import pytest
from tools.personaplex_depth_input_check import exact, run, source_input


def test_source_reference_stops_before_transformer_and_head_and_matches_native_preparation():
    import torch
    from moshi.models.lm import ScaledEmbedding
    projections = [torch.nn.Linear(32, 16, bias=False) for _ in range(3)]
    tables = [ScaledEmbedding(9 if i == 0 else 5, 16, norm=False, zero_idx=-1) for i in range(3)]
    native = SimpleNamespace(depformer_multi_linear=True, depformer_in=projections,
        depformer_text_emb=tables[0], depformer_emb=tables[1:])
    hidden = np.linspace(-3, 2, 32, dtype=np.float32)
    for cb, token in ((0, 8), (1, -1), (2, 4)):
        result = source_input(native, hidden, cb, token)
        with torch.no_grad():
            expected = (projections[cb](torch.from_numpy(hidden)) + tables[cb](torch.tensor([token]))).numpy().reshape(-1)
        exact(result, expected, "native")
    native.depformer_multi_linear = False
    result = source_input(native, hidden, 2, 4)
    with torch.no_grad():
        expected = (projections[0](torch.from_numpy(hidden)) + tables[2](torch.tensor([4]))).numpy().reshape(-1)
    exact(result, expected, "shared projection")
    assert not hasattr(native, 'linears')  # source never reached an audio head


@pytest.mark.parametrize("fault", ["dtype", "signed_zero", "nonfinite", "shape"])
def test_diagnostic_exact_requires_finite_fp32_bits(fault):
    expected = np.array([0, 1], np.float32)
    actual = expected.copy()
    if fault == 'dtype':
        actual = actual.astype(np.float64)
    elif fault == 'signed_zero':
        actual[0] = -0.0
    elif fault == 'nonfinite':
        actual[0] = np.nan
    else:
        actual = actual[None]
    with pytest.raises(AssertionError):
        exact(actual, expected, fault)


def test_existing_receipt_is_never_overwritten(tmp_path):
    saved = tmp_path / 'prior';saved.mkdir()
    receipt = saved / 'depth_input_check.json';receipt.write_text('saved evidence')
    with pytest.raises(FileExistsError):
        run(tmp_path/'missing_checkpoint', tmp_path/'missing_baseline', saved)
    assert receipt.read_text() == 'saved evidence'
