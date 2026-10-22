"""Focused native adapter read/write cases; not an RTL oracle by itself."""

import pytest

from tools.wideport_native_adapter import ARead, BRead, BWrite, NativeAdapter, NativePins


def pins(*, ready=True, responses=None):
    return NativePins((ready,) * 8, tuple((responses or {}).get(c) for c in range(8)))


def write_pins(*, command=True, data=True, done=None, responses=None):
    return NativePins((command,) * 8, tuple((responses or {}).get(c) for c in range(8)),
                      (data,) * 8, tuple((done or {}).get(c, 0) for c in range(8)))


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


def test_b_response_sources_follow_logical_halves_across_chash_swap():
    m = NativeAdapter()
    # Chunk 1 has odd CHASH parity: its logical low half is physical channel
    # 3, and its high half is physical channel 2.
    assert m.step(pins(), b=BRead("swap", 32)).b_accepted
    assert not m.step(pins()).native_commands
    issued = m.step(pins()).native_commands
    assert {cmd.channel: cmd.id[-1] for cmd in issued} == {2: 1, 3: 0}
    assert not m.step(pins(responses={cmd.channel: cmd.id for cmd in issued})).b_response_sources
    result = m.step(pins())
    assert result.b_response is not None
    assert [(s.chunk, s.low_id, s.high_id) for s in result.b_response_sources] == [
        (0, ("B", "swap", 0, 0), ("B", "swap", 0, 1))]
    assert m.step(pins()).b_response_sources == ()


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


def test_split_b_response_sources_keep_chunk_indices_and_order():
    m = NativeAdapter()
    address = next(a for a in range(0, 32 * 256, 32)
                   if m._plan_b(BRead("search", a, chunks=4)).second_mask)
    expected = m._plan_b(BRead("split", address, chunks=4))
    m.step(pins(), b=BRead("split", address, chunks=4))
    responses = {}
    parts = []
    for _ in range(12):
        result = m.step(pins(responses=responses))
        if result.b_response is not None:
            parts.append((result.b_response, result.b_response_sources))
        responses = {cmd.channel: cmd.id for cmd in result.native_commands if cmd.accepted}
        if len(parts) == 2:
            break
    assert len(parts) == 2
    assert [part.chunk_mask for part, _ in parts] == [expected.first_mask, expected.second_mask]
    assert [part.last for part, _ in parts] == [False, True]
    for part, sources in parts:
        assert [source.chunk for source in sources] == [i for i in range(4) if part.chunk_mask & (1 << i)]
        for source in sources:
            assert source.low_id == ("B", "split", source.chunk, 0)
            assert source.high_id == ("B", "split", source.chunk, 1)


def test_a_miss_prefetch_response_register_and_reuse():
    m = NativeAdapter()
    assert m.step(pins(), a=ARead("a0", 0)).a_accepted
    first = m.step(pins()).native_commands
    assert len(first) == 1 and first[0].source == "A" and first[0].beat_addr == 0
    m.step(pins(responses={first[0].channel: first[0].id}))
    assert m.step(pins()).a_response_id is None  # AO pop, a_rvalid register loaded at edge
    served = m.step(pins(), a=ARead("reuse", 1))
    assert served.a_response_id == "a0"
    assert served.a_response_source_id == first[0].id
    assert len(m.a_order) == 1 and m.a_order[0].reuse
    assert m.step(pins()).a_response_id is None
    served = m.step(pins())
    assert served.a_response_id == "reuse"
    assert served.a_response_source_id == first[0].id


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


@pytest.mark.parametrize("mask,channels", [((1 << 16) - 1, (0,)),
                                          (((1 << 13) - 1), (0,)),
                                          (((1 << 32) - 1), (0, 1))])
def test_b_masked_write_halves_and_acknowledged_idle(mask, channels):
    m = NativeAdapter()
    data = int.from_bytes(bytes(range(128)), "little")
    w = BWrite("patch", 0, data, mask)
    assert m.step(write_pins(), b=w).b_accepted
    assert m.step(write_pins()).native_commands == ()  # BREG loads the selected halves
    issued = m.step(write_pins())
    assert not issued.wr_idle and issued.b_response is None
    assert tuple(cmd.channel for cmd in issued.native_commands) == channels
    for cmd in issued.native_commands:
        half = cmd.channel
        words = (mask >> (16 * half)) & 0xFFFF
        expected_mask = sum(0xF << (4 * i) for i in range(16) if words & (1 << i))
        assert cmd.write and cmd.cvalid and cmd.wvalid and cmd.accepted
        assert (cmd.payload, cmd.mask) == ((data >> (512 * half)) & ((1 << 512) - 1), expected_mask)
    assert m.step(write_pins()).wr_idle is False
    done = {c: 1 for c in channels}
    assert m.step(write_pins(done=done)).wr_idle is False
    assert m.step(write_pins(done=done)).wr_idle is True
    assert m.step(write_pins(done=done)).rd_idle is True


def test_b_write_command_and_data_handshakes_are_independent():
    m = NativeAdapter()
    m.step(write_pins(), b=BWrite("w", 0, 0x12345678, 1))
    m.step(write_pins())
    first = m.step(write_pins(command=False, data=True)).native_commands[0]
    assert (first.cvalid, first.wvalid, first.accepted) == (True, True, False)
    assert m.data_done[first.channel] and not m.command_done[first.channel]
    second = m.step(write_pins(command=True, data=False)).native_commands[0]
    assert (second.cvalid, second.wvalid, second.accepted) == (True, False, True)
    assert second.id == first.id and second.mask == first.mask
    assert not m.bq[first.channel]

    n = NativeAdapter()
    n.step(write_pins(), b=BWrite("other", 0, 0, 1 << 20))
    n.step(write_pins())
    first = n.step(write_pins(command=True, data=False)).native_commands[0]
    assert (first.cvalid, first.wvalid, first.accepted) == (True, True, False)
    second = n.step(write_pins(command=False, data=True)).native_commands[0]
    assert (second.cvalid, second.wvalid, second.accepted) == (False, True, True)
    assert second.id == first.id


def test_b_write_hashed_half_and_reset_write_done_baseline():
    m = NativeAdapter()
    m.step(write_pins(done={3: 17}), reset=True)
    assert m.write_issued[3] == 17
    assert m.step(write_pins(done={3: 17})).wr_idle is False
    assert m.step(write_pins(done={3: 17})).wr_idle is True
    # Chunk 1's CHASH parity swaps its logical high half into channel 2.
    data = (0xAABBCCDD << (512 + 32 * 4))
    m.step(write_pins(done={3: 17}), b=BWrite("high", 32, data, 1 << 20))
    m.step(write_pins(done={3: 17}))
    issued = m.step(write_pins(done={3: 17})).native_commands
    assert len(issued) == 1
    assert (issued[0].channel, issued[0].beat_addr, issued[0].payload,
            issued[0].mask) == (2, 0, 0xAABBCCDD << (32 * 4), 0xF << (4 * 4))


def test_b_write_invalidates_cached_a_beat_one_edge_after_issue():
    m = NativeAdapter()
    m.step(write_pins(), a=ARead("old", 0))
    assert m.cache[0] is not None
    m.step(write_pins(), b=BWrite("patch", 0, 0, 1))
    issued = None
    for _ in range(40):
        edge = m.step(write_pins())
        if any(cmd.write and cmd.accepted for cmd in edge.native_commands):
            issued = edge
            break
    assert issued is not None
    assert m.cache[0] is not None  # wk_v registered on issue edge
    m.step(write_pins())
    assert m.cache[0] is None
    assert m.step(write_pins(), a=ARead("fresh", 0)).a_accepted
    assert not m.a_order[-1].reuse


def test_write_inputs_fail_closed_and_read_constructor_stays_compatible():
    m = NativeAdapter()
    with pytest.raises(ValueError, match="write-ready and write-done"):
        m.step(pins(), b=BWrite("w", 0, 0, 1))
    for bad in (BWrite("unaligned", 1, 0, 1), BWrite("zero-mask", 0, 0, 0),
                BWrite("oversized", 0, 1 << 1024, 1)):
        with pytest.raises(ValueError):
            m.step(write_pins(), b=bad)
    assert m.step(pins(), b=BRead("old-api", 0)).b_accepted
