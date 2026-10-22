"""Adversarial ownership transitions, independent of saved RTL observations."""
import pytest
from tools.wideport_pipeline_ownership import SlotRing, Ticket, RowCredits


def test_ring_reuse_has_new_generation_after_consumption():
    ring = SlotRing(1)
    old = ring.reserve(0)
    ring.complete(old)
    ring.consume(0)
    new = ring.reserve(1)
    assert (new.slot, new.epoch, new.owner) == (0, 1, 1)
    with pytest.raises(ValueError, match="stale epoch"):
        ring.complete(old)
    ring.complete(new)
    assert ring.consume(1) == new
    assert ring.reuses == 1


def test_early_reuse_and_unavailable_operand_rejected():
    ring = SlotRing(1)
    ring.reserve(0)
    with pytest.raises(ValueError, match="premature"):
        ring.reserve(1)
    with pytest.raises(ValueError, match="unavailable"):
        ring.consume(0)
    ring.complete()
    with pytest.raises(ValueError, match="wrong FIFO owner"):
        ring.consume(1)


def test_active_accumulator_cannot_begin_another_row():
    rows = RowCredits(2)
    token = rows.first(0)
    with pytest.raises(ValueError, match="accumulator unavailable"):
        rows.first(1)
    assert rows.last(0) == token
    assert rows.available


def test_backpressure_retains_rows_and_blocks_capacity_reuse():
    rows = RowCredits(1)
    token = rows.first(0)
    rows.last(0)
    rows.result(token)
    with pytest.raises(ValueError, match="credits exhausted"):
        rows.first(1)
    with pytest.raises(ValueError, match="backpressure"):
        rows.drain(grant=False)
    assert rows.live[0] == token and not rows.available
    assert rows.drain(grant=True) == token
    assert rows.first(1) == Ticket(0, 1, 1)


def test_stale_or_duplicate_accumulator_result_rejected():
    rows = RowCredits(1)
    token = rows.first(0)
    rows.last(0)
    with pytest.raises(ValueError, match="stale"):
        rows.result(Ticket(0, 99, 0))
    rows.result(token)
    with pytest.raises(ValueError, match="duplicate"):
        rows.result(token)
