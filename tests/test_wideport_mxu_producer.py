"""Directed, clock-indexed checks of the independent MXU row-valid path."""

import pytest

from opentpu import isa as I
from tools.wideport_mxu_producer import MxuRowProducer, Rl4Int8Mm


def _mm(n, kb):
    return I.mm(sa=0x60000, ssa=0x90000, out=0x8000, n=n, kb=kb,
                rs=4 * kb * 128, ors=n + 1, m=1, ab=0, srs=64,
                wf=I.W8, rl=4)


def test_kb9_row_ends_follow_actual_pop_bubbles():
    model = MxuRowProducer([_mm(13, 9)], cl=16)
    # Four RL4 groups, each with nine advances. The gaps are bubbles, not
    # inferred delays from a subsequently observed rf_push signal.
    bursts = (range(3, 12), range(20, 29), range(36, 45), range(52, 61))
    pops = {cycle for burst in bursts for cycle in burst}
    observed = [cycle for cycle in range(120) if model.step(
        pop=cycle in pops, start=cycle == 0, done=cycle == 110,
        pn=False, rst=False)]
    assert model.lrow == 29 and model.delay == 32
    assert model.row_ends == 4
    assert observed == [11 + 32, 28 + 32, 44 + 32, 60 + 32]
    model.finish()


def test_kb1_next_pop_head_may_run_before_old_head_drains():
    model = MxuRowProducer([_mm(5, 1), _mm(1, 1)], cl=8)
    observed = [cycle for cycle in range(70) if model.step(
        pop=cycle in (3, 4, 6), start=cycle in (0, 1),
        done=cycle in (41, 50), pn=6 <= cycle <= 41,
        rst=False)]
    assert model.lrow == 21 and model.delay == 24
    assert model.row_ends == 3
    assert observed == [3 + 24, 4 + 24, 6 + 24]
    model.finish()


@pytest.mark.parametrize("reset_cycle,want", [(10, [34]), (33, [])])
def test_rv_reset_gate_does_not_retime_or_clear_unreset_delay(reset_cycle, want):
    model = MxuRowProducer([_mm(1, 1)], cl=16)
    # The pop at 2 reaches tl_v at pre-edge 33. An earlier reset does not
    # flush that unreset delay; only rst on edge 33 gates rv and suppresses 34.
    pushed = [cycle for cycle in range(50) if model.step(
        pop=cycle == 2, start=cycle == 0, done=False,
        pn=False, rst=cycle == reset_cycle)]
    assert pushed == want
    assert not any(model.stages)


def test_supported_command_shape_is_explicit():
    assert Rl4Int8Mm.from_instr(_mm(257, 1)).row_groups == 65
    assert Rl4Int8Mm.from_instr(_mm(13, 9)).advances == 36
    with pytest.raises(ValueError, match="RL4"):
        Rl4Int8Mm.from_instr(I.mm(0, 0, 0, 8, 1, 128, 8, 1, 0, 64))
    with pytest.raises(ValueError, match="KB1/KB9"):
        Rl4Int8Mm.from_instr(_mm(13, 2))
    with pytest.raises(ValueError, match="D128/MCOLS8"):
        MxuRowProducer([_mm(13, 9)], d=64)


def test_row_eligibility_uses_command_and_fifo_state():
    model = MxuRowProducer([_mm(13, 9)])
    model.step(pop=False, start=True, done=False, pn=False, rst=False)
    assert model.eligibility(f_count=8, s_count=9, pn=False) == {}
    model.step(pop=False, start=False, done=False, pn=False, rst=False)
    p = model.eligibility(f_count=8, s_count=9, pn=False)
    assert p == {"first_row": 1, "c_unit": 0, "f_av": 1, "s_av": 1, "row_in": 0, "rf_capacity": 64}
    assert model.eligibility(f_count=9, s_count=9, pn=False)["row_in"] == 1
    assert model.eligibility(f_count=9, s_count=8, pn=False)["row_in"] == 0
    model.step(pop=True, start=False, done=False, pn=False, rst=False)
    assert model.eligibility(f_count=8, s_count=8, pn=False)["first_row"] == 0


def test_pop_and_handoff_fail_closed_without_accepted_work():
    model = MxuRowProducer([_mm(1, 1)])
    with pytest.raises(ValueError, match="pop without accepted"):
        model.step(pop=True, start=False, done=False, pn=False, rst=False)
    model.step(pop=False, start=True, done=False, pn=False, rst=False)
    with pytest.raises(ValueError, match="pn has no queued"):
        model.step(pop=False, start=False, done=False, pn=True, rst=False)
    with pytest.raises(ValueError, match="before all MM advances"):
        model.step(pop=False, start=False, done=True, pn=False, rst=False)
