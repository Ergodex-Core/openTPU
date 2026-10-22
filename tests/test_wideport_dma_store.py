import pytest
from opentpu import isa as I
from tools.wideport_dma_store import DmaStore, StoreCommand


@pytest.mark.parametrize("words", [13, 16])
def test_command_store_pipeline_and_ready_hold(words):
    model = DmaStore()
    command = StoreCommand(3, I.st(0x60000, 0xE000, words), tuple(range(words)))
    model.step(start=command)
    for _ in range(6):
        assert model.snapshot()["request"] is None
        model.step()
    request = model.snapshot()["request"]
    assert request == command.request()
    for _ in range(5):
        model.step(b_ready=False, wr_idle=True)
        assert model.snapshot()["request"] == request
        assert not model.done
    model.step(b_ready=True)
    assert model.snapshot()["request"] is None
    model.step()
    for _ in range(4):
        model.step(wr_idle=False)
        assert model.busy and not model.done
    model.step(wr_idle=True)
    assert model.done and not model.busy
    assert model.finish() == dict(starts=1,writes=1,completions=1)


def test_reject_unsupported_store_and_early_restart():
    model = DmaStore()
    with pytest.raises(ValueError, match="aligned"):
        model.step(start=StoreCommand(0,I.st(4,0,16),tuple(range(16))))
    command = StoreCommand(0,I.st(0,0,16),tuple(range(16)))
    model.step(start=command)
    with pytest.raises(ValueError, match="busy"):
        model.step(start=command)
