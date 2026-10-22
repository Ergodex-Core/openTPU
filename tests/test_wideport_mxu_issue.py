"""Directed checks of the bounded, independent MXU command issuer."""

import pytest

from opentpu import isa as I
from tools.wideport_mxu_issue import IssueCommand, MxuIssueModel


def _mm(n=13, kb=9, *, unit=False, rsc=False, k0=0, srs=64):
    return I.mm(sa=0x60000, ssa=0x90000, out=0x8000, n=n, kb=kb,
                rs=4 * kb * 128, ors=n + 1, m=1, ab=0, srs=srs,
                wf=I.W8, rl=4, k0=k0, unit=unit, rsc=rsc)


def _prime(model):
    assert not model.step(start=True).a_req
    assert model.snapshot().initializing
    assert not model.step().b_req
    assert model.snapshot().active


def test_kb9_row_and_scale_address_walk_with_joint_grants():
    model = MxuIssueModel([_mm()])
    _prime(model)
    first = model.snapshot()
    assert (first.a_addr, first.b_addr, first.b_n) == (0x90010 // 4, 0x60000 // 4, 3)
    assert (first.need_a, first.need_b, first.want) == (True, True, True)
    assert first.tag.ent and first.tag.cpl and first.tag.swe and first.tag.hv == 3
    # A request cannot be issued without B's grant, nor vice versa.
    assert not model.step(a_gnt=True).a_req
    assert model.snapshot() == first.__class__(**{**first.__dict__, "cycle": first.cycle + 1})
    assert not model.step(b_gnt=True).b_req
    assert model.snapshot().b_addr == first.b_addr

    issued = [model.step(a_gnt=True, b_gnt=True) for _ in range(10)]
    assert all(s.a_req and s.b_req for s in issued)
    assert [s.b_addr for s in issued[:9]] == [(0x60000 + k * 512) // 4 for k in range(9)]
    assert [s.a_addr for s in issued[:9]] == [(0x90000 + 16 + 4 * k) // 4 for k in range(9)]
    assert issued[9].b_addr == (0x60000 + 9 * 512) // 4
    assert issued[9].a_addr == (0x90000 + 64 + 16) // 4
    assert issued[8].tag.swr is False
    assert issued[9].tag.swe is True
    assert model.snapshot().occ == 10 and model.snapshot().tg_n == 10


def test_a_tag_response_is_registered_then_completes_scale_entry():
    model = MxuIssueModel([_mm(n=1, kb=1)])
    _prime(model)
    event = model.step(a_gnt=True, b_gnt=True)
    assert event.a_req and event.b_req and event.tag.cpl
    assert not model.snapshot().rv1
    model.step(a_rvalid=True, b_response=True)
    assert model.snapshot().rv1 and model.snapshot().r_cpl
    assert model.snapshot().tg_n == 1 and model.snapshot().s_count == 0
    retired = model.step()
    assert retired.rv1 and retired.r_cpl
    assert model.snapshot().tg_n == 0 and model.snapshot().s_count == 1
    model.step(fpop=True, scale_retire=True)
    assert (model.snapshot().occ, model.snapshot().f_count, model.snapshot().s_count) == (0, 0, 0)
    model.finish()


def test_sw_only_request_precedes_data_and_prevents_duplicate_sw():
    # RL4 W8 first scale data is H + k0*L = 16 + 12*4 = 64 bytes
    # beyond this group's Sw, so the issuer emits an A-only line request.
    model = MxuIssueModel([_mm(n=5, kb=1, k0=12)])
    _prime(model)
    sw = model.snapshot()
    assert sw.sw_only and sw.need_a and not sw.need_b
    assert sw.a_addr == 0x90000 // 4
    assert not sw.tag.ent and sw.tag.swr
    only = model.step(a_gnt=True)
    assert only.a_req and not only.b_req
    data = model.snapshot()
    assert data.need_a and data.need_b and data.a_addr == 0x90040 // 4
    assert not data.tag.swe and data.tag.swr
    model.step(a_gnt=True, b_gnt=True)
    assert not model.snapshot().sw_only  # next Sw came with the previous data line


def test_rsc_and_unit_choose_distinct_a_paths():
    rsc = MxuIssueModel([_mm(n=5, kb=1, rsc=True)])
    _prime(rsc)
    first = rsc.snapshot()
    assert first.a_addr == 0x90000 // 4
    assert first.tag.rsc and first.tag.ent and first.tag.cpl and first.tag.swe
    rsc.step(a_gnt=True, b_gnt=True)
    assert rsc.snapshot().a_addr == (0x90000 + 64) // 4

    unit = MxuIssueModel([_mm(n=1, kb=1, unit=True)])
    _prime(unit)
    assert unit.snapshot().need_b and not unit.snapshot().need_a
    assert unit.step(b_gnt=True).b_req
    assert unit.snapshot().tg_n == 0
    unit.step(b_response=True)
    unit.step(fpop=True, scale_retire=False)
    unit.finish()


def test_tag_depth_stalls_both_ports_until_registered_response_retires():
    model = MxuIssueModel([_mm(n=517, kb=1)])  # 130 RL groups
    _prime(model)
    for _ in range(128):
        assert model.step(a_gnt=True, b_gnt=True).a_req
    assert model.snapshot().tg_n == 128 and model.snapshot().occ == 128
    assert not model.snapshot().want
    assert not model.step(a_gnt=True, b_gnt=True, a_rvalid=True).a_req
    assert model.snapshot().rv1 and model.snapshot().tg_n == 128
    assert not model.step(a_gnt=True, b_gnt=True).a_req
    assert model.snapshot().tg_n == 127 and model.snapshot().want
    assert model.step(a_gnt=True, b_gnt=True).a_req


def test_reserved_b_depth_stalls_then_reopens_after_pop():
    model = MxuIssueModel([_mm(n=4100, kb=1, unit=True)])  # 1025 RL groups
    _prime(model)
    for _ in range(1024):
        assert model.step(b_gnt=True, b_response=False).b_req
    assert model.snapshot().occ == 1024 and not model.snapshot().want
    assert not model.step(b_gnt=True, b_response=True).b_req
    assert model.snapshot().f_count == 1 and model.snapshot().occ == 1024
    assert not model.step(b_gnt=True, fpop=True, scale_retire=False).b_req
    assert model.snapshot().occ == 1023 and model.snapshot().want
    assert model.step(b_gnt=True).b_req


def test_unsupported_layout_and_unmodeled_events_fail_closed():
    with pytest.raises(ValueError, match="RL4"):
        IssueCommand.from_instr(I.mm(0, 0, 0, 8, 1, 128, 8, 1, 0, 64))
    with pytest.raises(ValueError, match="KB1/KB9"):
        IssueCommand.from_instr(_mm(kb=2))
    with pytest.raises(ValueError, match="PORTB4"):
        MxuIssueModel([_mm()], portb=2)
    model = MxuIssueModel([_mm(n=1, kb=1)])
    with pytest.raises(ValueError, match="response without"):
        model.step(a_rvalid=True)
    with pytest.raises(ValueError, match="completion without"):
        model.step(b_response=True)
    with pytest.raises(ValueError, match="scale_retire"):
        model.step(fpop=True)
