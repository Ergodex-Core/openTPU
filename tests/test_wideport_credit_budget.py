"""Small, offline invariants for the two-MM oracle credit ledger."""

from collections import deque

from tools.wideport_credit_budget import (
    ADVANCES, SB, WB, WEIGHT_BYTES, SCALE_WINDOW_BYTES,
    _accepted_command, _first_last, _gap, _new_owner, _pop_credit, _read_response,
    classify_byte,
)
from tools.wideport_mxu_issue import IssueCommand


def test_owner_windows_include_prefetch_but_do_not_alias():
    assert classify_byte(WB[0]) == ("weight", 0)
    assert classify_byte(WB[0] + WEIGHT_BYTES - 1) == ("weight", 0)
    assert classify_byte(WB[0] + WEIGHT_BYTES) is None
    assert classify_byte(WB[1]) == ("weight", 1)
    assert classify_byte(SB[0] + SCALE_WINDOW_BYTES - 1) == ("scale", 0)
    assert classify_byte(SB[0] + SCALE_WINDOW_BYTES) == ("scale", 1)
    assert classify_byte(SB[1]) == ("scale", 1)


def test_phase_summary_preserves_unknown_and_signed_overlap():
    assert _first_last([]) == {"first": None, "last": None, "count": 0}
    assert _first_last([3, 5]) == {"first": 3, "last": 5, "count": 2}
    assert _gap(None, 10) is None
    assert _gap(5, 9) == 3
    assert _gap(9, 5) == -5


def test_pop_and_pipelined_row_credits_cross_drain_head_boundary():
    assert [_pop_credit(n) for n in (0, 8, 9, 17, 26, 35, 36, 44, 71)] == [
        (0, False), (0, True), (0, False), (0, True), (0, True),
        (0, True), (1, False), (1, True), (1, True)]


def test_native_response_ids_come_from_channel_request_order():
    cmd = IssueCommand(sa=WB[0], ssa=SB[0], n=13, kb=9, rs=0, srs=0,
                       k0=0, unit=False, rsc=False)
    assert cmd.advances == ADVANCES
    owners = [_new_owner(0, cmd), _new_owner(1, cmd)]
    queues = [deque() for _ in range(8)]
    errors, ledger, ids = [], [], [0] * 8
    # With chash disabled, channel 0 beat 0 maps to logical byte 0, so use
    # a native beat whose logical address is WB0. The address owner is still
    # derived by logical_beat, never by a response's observed data.
    from tools.wideport_native_memory import logical_beat
    beat = next(b for b in range(WB[0] // 64 + 1) if logical_beat(b, 0, chash=False) * 64 == WB[0])
    row = {"cv": 1, "cr": 1, "we": 0, "addr": beat, "rv": 0}
    _accepted_command(queues, row, 0, 4, False, owners, errors, ledger, ids)
    _accepted_command(queues, row, 0, 5, False, owners, errors, ledger, ids)
    assert [(e["channel"], e["id"], e["owner"]) for e in ledger] == [
        (0, 0, ("weight", 0)), (0, 1, ("weight", 0))]
    _read_response(queues, {"rv": 1}, 0, 9, owners, errors)
    _read_response(queues, {"rv": 1}, 0, 10, owners, errors)
    assert [e["response_cycle"] for e in ledger] == [9, 10]
    assert owners[0]["cycles"]["native_weight_response"] == [9, 10]
    assert owners[0]["native_weight_payload_bytes"] == 128
    assert errors == []


def test_unmatched_native_response_is_assertable_violation():
    queues = [deque() for _ in range(8)]
    errors = []
    _read_response(queues, {"rv": 1}, 3, 7, [], errors)
    assert errors == ["cycle 7: channel 3 returned a read without an accepted request"]
