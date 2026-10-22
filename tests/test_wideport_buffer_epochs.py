"""Pure protocol checks; no RTL, model, network, or physical board execution."""

import pytest

from tools.wideport_buffer_epochs import EpochPageLedger


READS = {"weight": 2, "scale": 1}
PATCH = {"weight": 2, "scale": 1}


def _read_tickets(page, epoch):
    return [("weight", ("r", page, epoch, "weight", 0)),
            ("weight", ("r", page, epoch, "weight", 1)),
            ("scale", ("r", page, epoch, "scale", 0))]


def _issue_all(ledger, page, epoch):
    for kind, ticket in _read_tickets(page, epoch):
        ledger.issue(page, epoch, kind, ticket)


def _respond_all(ledger, page, epoch):
    for kind, ticket in _read_tickets(page, epoch):
        ledger.respond(page, epoch, kind, ticket, response_epoch=epoch)


def _patch_tickets(page, epoch):
    return [("weight", ("w", page, epoch, "weight", 0)),
            ("weight", ("w", page, epoch, "weight", 1)),
            ("scale", ("w", page, epoch, "scale", 0))]


def test_two_distinct_epoch_zero_pages_then_page_zero_epoch_one():
    ledger = EpochPageLedger(READS)
    ledger.read_start(0, 0)  # first MM
    ledger.read_start(1, 0)  # second MM, independent physical page
    for page in (0, 1):
        _issue_all(ledger, page, 0)
        ledger.close_reads(page, 0)
        _respond_all(ledger, page, 0)
        assert ledger.page_free(page, 0)
    ledger.begin_overwrite(0, 0, 1, PATCH)
    assert not ledger.page_free(0, 0)
    for kind, ticket in _patch_tickets(0, 1):
        ledger.start_write(0, 1, kind, ticket)
    for kind, ticket in _patch_tickets(0, 1):
        ledger.complete_write(0, 1, kind, ticket)
    assert not ledger.page_free(0, 0)
    assert ledger.snapshot(0)["ready_epoch"] == 1
    ledger.read_start(0, 1)  # third MM reuses page 0 only after both patches ack
    _issue_all(ledger, 0, 1)
    ledger.close_reads(0, 1)
    _respond_all(ledger, 0, 1)
    assert ledger.page_free(0, 1)
    assert ledger.page_free(1, 0)
    assert ledger.snapshot(0)["arrivals_ready"] is None
    assert ledger.snapshot(0)["output_committed"] is None
    assert ledger.calibrated is False and ledger.tok_s is None


def test_premature_overwrite_rejects_open_issuer_and_outstanding_response():
    ledger = EpochPageLedger(READS)
    ledger.read_start(0, 0)
    ledger.issue(0, 0, "weight", _read_tickets(0, 0)[0][1])
    with pytest.raises(ValueError, match="weight issuer open"):
        ledger.close_reads(0, 0)
    with pytest.raises(ValueError, match="premature overwrite"):
        ledger.begin_overwrite(0, 0, 1, PATCH)
    _issue_all_remaining = _read_tickets(0, 0)[1:]
    for kind, ticket in _issue_all_remaining:
        ledger.issue(0, 0, kind, ticket)
    ledger.close_reads(0, 0)
    with pytest.raises(ValueError, match="premature overwrite"):
        ledger.begin_overwrite(0, 0, 1, PATCH)
    _respond_all(ledger, 0, 0)
    ledger.begin_overwrite(0, 0, 1, PATCH)


def test_new_read_waits_for_every_weight_and_scale_patch_ack():
    ledger = EpochPageLedger(READS)
    ledger.read_start(0, 0)
    _issue_all(ledger, 0, 0)
    ledger.close_reads(0, 0)
    _respond_all(ledger, 0, 0)
    ledger.begin_overwrite(0, 0, 1, PATCH)
    writes = _patch_tickets(0, 1)
    for kind, ticket in writes:
        ledger.start_write(0, 1, kind, ticket)
    for kind, ticket in writes[:-1]:
        ledger.complete_write(0, 1, kind, ticket)
    with pytest.raises(ValueError, match="patch acknowledgements incomplete"):
        ledger.read_start(0, 1)
    with pytest.raises(ValueError, match="no matching acquired read epoch"):
        ledger.issue(0, 1, "scale", ("early", 0))
    ledger.complete_write(0, 1, *writes[-1])
    ledger.read_start(0, 1)


def test_stale_response_epoch_and_duplicate_ticket_fail_closed():
    ledger = EpochPageLedger(READS)
    ledger.read_start(0, 0)
    _issue_all(ledger, 0, 0)
    ledger.close_reads(0, 0)
    _respond_all(ledger, 0, 0)
    ledger.begin_overwrite(0, 0, 1, PATCH)
    for kind, ticket in _patch_tickets(0, 1):
        ledger.start_write(0, 1, kind, ticket)
        ledger.complete_write(0, 1, kind, ticket)
    ledger.read_start(0, 1)
    kind, ticket = _read_tickets(0, 1)[0]
    ledger.issue(0, 1, kind, ticket)
    with pytest.raises(ValueError, match="duplicate ticket"):
        ledger.issue(0, 1, kind, ticket)
    with pytest.raises(ValueError, match="stale weight response"):
        ledger.respond(0, 1, kind, ticket, response_epoch=0)
    assert not ledger.page_free(0, 1)
    ledger.respond(0, 1, kind, ticket, response_epoch=1)
    with pytest.raises(ValueError, match="duplicate read response"):
        ledger.respond(0, 1, kind, ticket, response_epoch=1)
