"""The card's driver over the ISA simulator (opentpu/host/isaboard.py, `--backend board-isa`):
BoardBackend's program loads (with their wait entries on a WAITC target), run arguments, the
decode loop's token reads and the staged KV copies run against IsaTransport, whose DRAM is
the simulator's: a tiny Qwen3's prefill logits and the card loop's tokens are the ISA
backend's, bit for bit, and the transport counts the bytes the host moved."""
import numpy as np
import pytest

from opentpu.host import chat as CH
from opentpu.llm.qwen3 import Engine

from test_serve_kvio import _model

torch = pytest.importorskip("torch")


def test_the_cards_driver_on_the_isa_simulator_gives_the_isas_bits():
    W, spec, _ = _model("qwen3")
    backend, cfg = CH.make_backend("board-isa", spec, 512, None)
    eng = Engine(spec, W, cap=512, cfg=cfg, backend=backend, resident=True, prompt_runs=True)
    ref = Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True)
    prompt = [int(x) for x in np.random.default_rng(1).integers(0, 1000, 37)]
    a, b = eng.prefill(prompt[:-1]), ref.prefill(prompt[:-1])
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert eng.can_generate and ref.can_generate
    ta = eng.generate_card(prompt[-1], 12, stop_ids=[])
    tb = ref.generate_card(prompt[-1], 12, stop_ids=[])
    assert ta == tb and len(ta) == 12
    tr = eng.backend.board.t
    c = tr.counters()
    assert c["sim_runs"] >= 2 and c["bytes_written"] >= eng.image.nbytes
    assert c["bytes_read"] > 0 and c["sim_s"] > 0

