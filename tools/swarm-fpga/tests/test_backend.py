"""Backend contracts without a socket service or physical device."""
import numpy as np
import pytest
from swarmfpga.backend import ReferenceBackend
from swarmfpga.graph import Graph


@pytest.mark.parametrize("value", [np.zeros(4), np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
def test_reference_rejects_invalid_inputs(value):
    g = Graph()
    x = g.input("x", (2, 2))
    with pytest.raises(ValueError):
        ReferenceBackend().run(g.export({"x": x}), {"x": value})


def test_unknown_precision_is_rejected_before_execution():
    g = Graph()
    x = g.input("x", (1,))
    y = g.node("exp", (x,), (1,), compute_dtype="unknown")
    with pytest.raises(ValueError, match="compute dtype"):
        ReferenceBackend().run(g.export({"y": y}), {"x": np.zeros(1)})
