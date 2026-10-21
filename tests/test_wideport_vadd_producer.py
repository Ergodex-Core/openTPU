"""Focused command-driven V_ADD producer checks; no simulator build is needed."""

import pytest

from opentpu import isa as I
from tools.wideport_vadd_producer import VAddCommand, VAddProducer


def command(rows=1, cols=8):
    return I.vop(I.V_ADD, 50000, 4096, 12288, rows, cols, 256, 256, 256, I.B_FULL)


def test_decode_fails_closed_for_other_modes_and_out_of_tmem():
    c = VAddCommand.from_words(command(32, 256).encode())
    assert (c.rows, c.cols, c.dst, c.a, c.b) == (32, 256, 50000, 4096, 12288)
    for bad in (I.vop(I.V_SUB, 50000, 4096, 12288, 1, 8, 8, 8, 8).encode(),
                I.vop(I.V_ADD, 50000, 4096, 12288, 1, 8, 8, 8, 8, I.B_ROW).encode(),
                I.vop(I.V_ADD, 50000, 4096, 12288, 1, 8, 8, 8, 8, ra=1).encode()):
        with pytest.raises(ValueError):
            VAddCommand.from_words(bad)
    with pytest.raises(ValueError, match="beyond TMEM"):
        VAddCommand.from_words(command(32, 256).encode(), tmem_words=52000)


def test_request_and_output_metadata_follow_enabled_edges():
    p = VAddProducer()
    p.step(reset=True)
    p.step(start=command().encode(), en=True)  # accepted into the two-command queue
    assert p.snapshot()["queued"] == 1 and not p.snapshot()["issuing"]
    p.step(en=True)  # begins, without issuing yet
    assert p.snapshot()["issuing"] and not p.snapshot()["ta_words"]
    p.step(en=True)  # first and only issue
    assert p.snapshot()["ta_words"] == list(range(4096, 4104))
    assert p.snapshot()["tb_words"] == list(range(12288, 12296))
    assert p.snapshot()["ta_mask"] == p.snapshot()["tb_mask"] == 255
    p.step(en=False)  # registered requests hold on the lost-read cycle
    assert p.snapshot()["ta_words"] == list(range(4096, 4104))
    for _ in range(8):
        p.step(en=True)
        assert p.snapshot()["cw_words"] == []
    p.step(en=True)
    assert p.snapshot()["cw_words"] == list(range(50000, 50008))
    assert p.snapshot()["done_i"] == 0
    p.step(en=False)
    assert p.snapshot()["cw_words"] == list(range(50000, 50008))
    assert p.snapshot()["done_i"] == 0
    p.step(en=True)
    assert p.snapshot()["done_i"] == 1
    assert p.snapshot()["cw_words"] == []


def test_partial_last_chunk_and_two_queued_commands():
    p = VAddProducer()
    a = VAddCommand.from_words(command(1, 10).encode())
    b = VAddCommand.from_words(command(1, 8).encode())
    p.step(start=a, en=True)
    p.step(start=b, en=True)  # first begins; second queues on this edge
    p.step(en=True)
    assert p.snapshot()["ta_words"] == list(range(4096, 4104))
    p.step(en=True)
    assert p.snapshot()["ta_words"] == [4104, 4105]
    assert p.snapshot()["ta_mask"] == 3
    p.step(en=True)  # second begins after first has finished issuing
    assert p.snapshot()["issuing"] and p.snapshot()["queued"] == 0
    p.step(en=True)
    assert p.snapshot()["ta_words"] == list(range(4096, 4104))


def test_warm_reset_with_inflight_metadata_is_rejected():
    p = VAddProducer()
    p.step(start=VAddCommand.from_words(command().encode()), en=True)
    p.step(en=True)
    p.step(en=True)
    with pytest.raises(ValueError, match="warm reset"):
        p.step(reset=True)
