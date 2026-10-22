"""Directed, command-generated DMA LD timing and byte-lane checks."""

import pytest

from opentpu import isa as I
from tools.wideport_dma_load import DmaLoadModel, LoadCommand


def _chunk(base):
    return b"".join((base + i).to_bytes(4, "little") for i in range(32))


def _run(model, *, limit=300):
    writes = []
    done = []
    for _ in range(limit):
        state = model.snapshot()
        writes.extend(state.writes)
        if state.done:
            done.append(state.cycle)
            break
        model.step()
    assert done, "LD never completed"
    return writes, done[0]


def test_single_chunk_request_hold_registered_write_and_done():
    dma = DmaLoadModel()
    assert dma.snapshot().ready and not dma.snapshot().request
    dma.step(start=I.ld(0x20000, 0xE000, 8))
    assert dma.snapshot().request.word_addr == 0x20000 // 4
    assert dma.snapshot().request.tag
    dma.step()
    assert dma.snapshot().request.word_addr == 0x20000 // 4
    dma.step(b_ready=True)
    assert dma.snapshot().requested == 1 and dma.snapshot().request is None
    dma.step()
    assert not dma.snapshot().writes and not dma.snapshot().done
    response_cycle = dma.snapshot().cycle
    dma.step(response=_chunk(100))
    assert not dma.snapshot().writes
    dma.step()  # cnt > 0: segment is delivered; synchronous RAM q captures data.
    assert not dma.snapshot().writes
    dma.step()  # dv register drives the TMEM write register.
    write = dma.snapshot()
    assert write.cycle == response_cycle + 3
    assert write.writes == tuple((0xE000 + i, 100 + i) for i in range(8))
    assert write.ld_fin and not write.done and write.ready
    dma.step()  # the write lands at this edge; done becomes visible afterward.
    assert dma.snapshot().done and not dma.snapshot().writes
    dma.step()
    assert not dma.snapshot().done


def test_partial_first_and_last_segment_with_delayed_second_chunk():
    dma = DmaLoadModel()
    dma.step(start=LoadCommand("unaligned", 30 * 4, 100, 8))
    assert dma.snapshot().request.word_addr == 0
    dma.step(b_ready=True)
    assert dma.snapshot().request.word_addr == 32
    dma.step(b_ready=True)
    assert dma.snapshot().request is None
    dma.step(response=_chunk(0))
    writes = []
    for _ in range(5):
        writes.extend(dma.snapshot().writes)
        dma.step()
    assert writes == [(100, 30), (101, 31)]
    assert not dma.snapshot().done and dma.snapshot().received_chunks == 0
    dma.step(response=_chunk(32))
    tail, _ = _run(dma)
    assert tail == [(102 + i, 32 + i) for i in range(6)]
    assert dma.snapshot().requested == dma.snapshot().received == 2
    assert dma.snapshot().delivered_segments == 2


def test_36_chunk_activation_and_four_small_initial_loads_close():
    # Real fixed fixture: X = 36 * 128 bytes, then two 8-word VPU operands,
    # then two 16-word scratch canaries. Every start is explicit.
    dma = DmaLoadModel()
    loads = [(0, 0, 1152), (0x10000, 4096, 8),
             (0x18000, 12288, 8), (0x20000, 0xE000, 16),
             (0x21000, 0xE100, 16)]
    for index, (dram, tmem, words) in enumerate(loads):
        dma.step(start=I.ld(dram, tmem, words))
        chunks = (words + 31) // 32
        requests = []
        for j in range(chunks):
            req = dma.snapshot().request
            assert req and req.chunk == j
            requests.append(req.word_addr)
            dma.step(b_ready=True)
        assert requests == [dram // 4 + 32 * j for j in range(chunks)]
        # All grants can precede any response. Read responses may then arrive
        # one per cycle, while the segment delivery drains behind them.
        out = []
        for j in range(chunks):
            out.extend(dma.snapshot().writes)
            dma.step(response=_chunk(index * 1000 + 32 * j))
        tail, _ = _run(dma)
        out.extend(tail)
        assert out == [(tmem + j, index * 1000 + j) for j in range(words)]
        assert dma.snapshot().requested == dma.snapshot().received == chunks
        assert dma.snapshot().reserved_chunks == dma.snapshot().received_chunks == 0
        dma.step()  # consume the one-cycle done pulse before the next start
    assert dma.finish(expected_loads=5, expected_chunks=40, expected_words=1200) == {
        "loads": 5,
        "requested_chunks": 40,
        "received_chunks": 40,
        "delivered_segments": 150,
        "written_words": 1200,
        "done_pulses": 5,
    }


def test_finish_rejects_active_transfer_and_wrong_fixture_totals():
    dma = DmaLoadModel()
    with pytest.raises(ValueError, match="no LD starts"):
        dma.finish()
    dma.step(start=I.ld(0, 8, 8))
    with pytest.raises(ValueError, match="active request"):
        dma.finish()
    dma.step(b_ready=True)
    dma.step(response=_chunk(10))
    _, _ = _run(dma)
    assert dma.finish(expected_loads=1, expected_chunks=1, expected_words=8)["written_words"] == 8
    with pytest.raises(ValueError, match="expected 5"):
        dma.finish(expected_loads=5)
    with pytest.raises(ValueError, match="expected 40"):
        dma.finish(expected_chunks=40)
    with pytest.raises(ValueError, match="expected 1200"):
        dma.finish(expected_words=1200)


def test_rejects_unsupported_instruction_bad_range_orphan_and_grant():
    dma = DmaLoadModel()
    with pytest.raises(ValueError, match="only an LD"):
        dma.step(start=I.st(0, 0, 8))
    with pytest.raises(ValueError, match="register-relative"):
        dma.step(start=I.ld(0, 0, 8, ra=1))
    with pytest.raises(ValueError, match="word-aligned"):
        dma.step(start=LoadCommand("bad", 3, 0, 8))
    with pytest.raises(ValueError, match="exceeds 36"):
        dma.step(start=I.ld(0, 0, 36 * 32 + 1))
    with pytest.raises(ValueError, match="grant without"):
        dma.step(b_ready=True)
    dma.step(start=I.ld(0, 0, 8))
    with pytest.raises(ValueError, match="while DMA busy"):
        dma.step(start=I.ld(128, 32, 8))
    with pytest.raises(ValueError, match="previously issued"):
        dma.step(response=_chunk(0))
    dma.step(b_ready=True)
    with pytest.raises(ValueError, match="exactly 128"):
        dma.step(response=b"short")
    dma.step(response=_chunk(0))
    with pytest.raises(ValueError, match="previously issued"):
        dma.step(response=_chunk(32))


def test_reset_discards_inflight_load_and_accepts_later_start():
    dma = DmaLoadModel()
    dma.step(start=I.ld(0, 0, 64))
    dma.step(b_ready=True)
    dma.step(response=_chunk(0))
    old_cycle = dma.snapshot().cycle
    dma.step(reset=True, response=_chunk(999))
    assert dma.snapshot().cycle == old_cycle + 1
    assert dma.snapshot().ready and dma.snapshot().request is None
    assert dma.snapshot().requested == dma.snapshot().received == 0
    dma.step(start=I.ld(128, 50, 8))
    assert dma.snapshot().request.word_addr == 32
