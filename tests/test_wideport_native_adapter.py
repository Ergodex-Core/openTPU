"""Focused read-only native adapter timing cases; not an RTL oracle by itself."""

import pytest

from tools.wideport_native_adapter import ARead, BRead, NativeAdapter, NativePins


def pins(*, ready=True, responses=None):
    return NativePins((ready,) * 8, tuple((responses or {}).get(c) for c in range(8)))


def test_breg_tag_join_and_native_response_edge():
    m = NativeAdapter()
    request = BRead("b0", 0, tag=True)
    assert m.step(pins(), b=request).b_accepted
    assert m.step(pins()).native_commands == ()  # BREG first pushes queue on this edge
    issued = m.step(pins()).native_commands
    assert {(cmd.channel, cmd.beat_addr, cmd.accepted) for cmd in issued} == {
        (0, 0, True), (1, 0, True)
    }
    assert m.step(pins(responses={cmd.channel: cmd.id for cmd in issued})).b_response is None
    part = m.step(pins()).b_response
    assert part is not None and (part.id, part.tag, part.chunk_mask, part.last) == ("b0", True, 1, True)


def test_duplicate_pair_is_two_ordered_parts_and_holds_next_accept():
    m = NativeAdapter()
    address = next(
        a for a in range(0, 32 * 256, 32)
        if m._plan_b(BRead("search", a, chunks=4)).second_mask
    )
    plan = m._plan_b(BRead("split", address, chunks=4))
    assert plan.first_mask & plan.second_mask == 0
    m.step(pins(), b=BRead("split", address, chunks=4))
    assert not m.step(pins(), b=BRead("later", 0)).b_ready  # first part is being pushed
    assert m.step(pins()).b_ready  # second part is being pushed
    assert [entry.part.chunk_mask for entry in m.b_tags] == [plan.first_mask, plan.second_mask]
    assert [entry.part.last for entry in m.b_tags] == [False, True]


def test_a_miss_prefetch_response_register_and_reuse():
    m = NativeAdapter()
    assert m.step(pins(), a=ARead("a0", 0)).a_accepted
    first = m.step(pins()).native_commands
    assert len(first) == 1 and first[0].source == "A" and first[0].beat_addr == 0
    m.step(pins(responses={first[0].channel: first[0].id}))
    assert m.step(pins()).a_response_id is None  # AO pop, a_rvalid register loaded at edge
    assert m.step(pins(), a=ARead("reuse", 1)).a_response_id == "a0"
    assert len(m.a_order) == 1 and m.a_order[0].reuse
    assert m.step(pins()).a_response_id is None
    assert m.step(pins()).a_response_id == "reuse"


def test_response_id_must_match_independently_issued_native_command():
    m = NativeAdapter()
    m.step(pins(), b=BRead("b", 0))
    m.step(pins())
    issued = m.step(pins()).native_commands
    with pytest.raises(ValueError, match="expected"):
        m.step(pins(responses={issued[0].channel: ("observed RTL result",)}))


def test_unsupported_writes_fail_closed():
    m = NativeAdapter()
    with pytest.raises(TypeError, match="outside scope"):
        m.step(pins(), b={"we": True})
