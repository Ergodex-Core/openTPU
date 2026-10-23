"""Observed-event protocol ledger for two reusable wide-port DRAM page pairs.

Page 0 and page 1 each contain a weight and a scale region. A read epoch is
safe to overwrite only after its expected requests were all issued *and* their
read tickets all responded. The next epoch cannot start reading until both
weight and scale patches were completely issued and acknowledged. Callers
provide observed event identities; this ledger does not predict RTL timing,
consumer arrivals, accumulator state, output commit, or throughput.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field


KINDS = ("weight", "scale")
PAGES = (0, 1)


def _counts(values: Mapping[str, int], *, label: str) -> dict[str, int]:
    if not isinstance(values, Mapping) or set(values) != set(KINDS):
        raise ValueError(f"{label}: require exactly weight and scale counts")
    counts = dict(values)
    if any(type(counts[k]) is not int or counts[k] <= 0 for k in KINDS):
        raise ValueError(f"{label}: both counts must be positive integers")
    return counts


def _kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown page kind {kind!r}; expected weight or scale")


def _ticket(ticket: Hashable) -> None:
    try:
        hash(ticket)
    except TypeError as exc:
        raise ValueError("ticket must be hashable") from exc


@dataclass
class _Read:
    epoch: int
    expected: dict[str, int]
    issued: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    tickets: dict[Hashable, tuple[str, bool]] = field(default_factory=dict)
    closed: bool = False

    @property
    def all_responded(self) -> bool:
        return all(done for _kind_name, done in self.tickets.values())

    @property
    def safe(self) -> bool:
        return self.closed and all(self.issued[k] == self.expected[k] for k in KINDS) and self.all_responded


@dataclass
class _Patch:
    epoch: int
    expected: dict[str, int]
    issued: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    tickets: dict[Hashable, tuple[str, bool]] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return (all(self.issued[k] == self.expected[k] for k in KINDS)
                and all(acked for _kind_name, acked in self.tickets.values()))


@dataclass
class _Page:
    epoch: int = 0
    used_epochs: set[int] = field(default_factory=set)
    read: _Read | None = None
    patch: _Patch | None = None
    seen_tickets: set[Hashable] = field(default_factory=set)


class EpochPageLedger:
    """Fail-closed read/overwrite epochs; page numbers are exactly 0 and 1.

    ``read_start`` acquires the current epoch. ``issue`` and ``respond`` use
    opaque, page-unique read tickets (for native channels, use ``(ch, id)``).
    Expected read counts must include *all* native reads of each owner,
    including scale prefetch; 36 logical MM entries alone are insufficient
    proof that the old page is safe to overwrite.
    ``respond`` also requires the response's data epoch, established from an
    independent memory-content/epoch oracle, so a returned old cache/prefetch
    line cannot masquerade as the new page epoch. Copying the requested epoch
    into ``response_epoch`` without that check would make this gate vacuous.
    ``close_reads`` proves no more reads will be issued for that epoch.

    ``begin_overwrite`` reserves the next epoch and declares its per-kind patch
    command counts. Each observed write uses ``start_write`` and later
    ``complete_write`` with a page-unique ticket. The next epoch becomes ready
    only after *both* kinds have all expected writes acknowledged. Native write
    acknowledgements, rather than mere command acceptance, must drive the
    latter call. Event ordering is caller-supplied; no observed result is used
    as a predictor input.
    """

    calibrated = False
    tok_s = None

    def __init__(self, expected_reads: Mapping[str, int]):
        self.expected_reads = _counts(expected_reads, label="expected_reads")
        self._pages = {p: _Page() for p in PAGES}

    def _page(self, page: int) -> _Page:
        if type(page) is not int or page not in PAGES:
            raise ValueError(f"unknown page {page!r}; expected 0 or 1")
        return self._pages[page]

    @staticmethod
    def _epoch(epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def read_start(self, page: int, epoch: int) -> None:
        """Acquire a page pair for one MM, once per epoch."""
        state = self._page(page)
        self._epoch(epoch)
        if state.patch is not None:
            raise ValueError(f"page {page} epoch {epoch}: patch acknowledgements incomplete")
        if epoch != state.epoch:
            raise ValueError(f"page {page} epoch {epoch}: current ready epoch is {state.epoch}")
        if epoch in state.used_epochs:
            raise ValueError(f"page {page} epoch {epoch}: read epoch already acquired")
        if state.read is not None and not state.read.safe:
            raise ValueError(f"page {page} epoch {epoch}: prior read epoch remains active")
        state.read = _Read(epoch, dict(self.expected_reads))
        state.used_epochs.add(epoch)

    def issue(self, page: int, epoch: int, kind: str, ticket: Hashable) -> None:
        """Issue one weight chunk or one scale-entry read ticket."""
        state = self._page(page)
        self._epoch(epoch)
        _kind(kind)
        _ticket(ticket)
        read = state.read
        if read is None or read.epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: no matching acquired read epoch")
        if read.closed:
            raise ValueError(f"page {page} epoch {epoch}: read issue after close")
        if ticket in state.seen_tickets:
            raise ValueError(f"page {page} epoch {epoch}: duplicate ticket {ticket!r}")
        if read.issued[kind] >= read.expected[kind]:
            raise ValueError(f"page {page} epoch {epoch}: excess {kind} read issue")
        read.tickets[ticket] = (kind, False)
        read.issued[kind] += 1
        state.seen_tickets.add(ticket)

    def close_reads(self, page: int, epoch: int) -> None:
        """Declare the issuer finished; every expected request must exist."""
        state = self._page(page)
        self._epoch(epoch)
        read = state.read
        if read is None or read.epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: no matching acquired read epoch")
        if read.closed:
            raise ValueError(f"page {page} epoch {epoch}: reads already closed")
        for kind in KINDS:
            if read.issued[kind] != read.expected[kind]:
                raise ValueError(f"page {page} epoch {epoch}: {kind} issuer open "
                                 f"({read.issued[kind]}/{read.expected[kind]} issued)")
        read.closed = True

    def respond(self, page: int, epoch: int, kind: str, ticket: Hashable, *, response_epoch: int) -> None:
        """Retire the exact read ticket and assert its returned data epoch."""
        state = self._page(page)
        self._epoch(epoch)
        self._epoch(response_epoch)
        _kind(kind)
        _ticket(ticket)
        read = state.read
        if read is None or read.epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: no matching acquired read epoch")
        if ticket not in read.tickets:
            raise ValueError(f"page {page} epoch {epoch}: unknown read ticket {ticket!r}")
        issued_kind, done = read.tickets[ticket]
        if issued_kind != kind:
            raise ValueError(f"page {page} epoch {epoch}: ticket {ticket!r} is {issued_kind}, not {kind}")
        if done:
            raise ValueError(f"page {page} epoch {epoch}: duplicate read response {ticket!r}")
        if response_epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: stale {kind} response {ticket!r} "
                             f"from epoch {response_epoch}")
        read.tickets[ticket] = (kind, True)

    def page_free(self, page: int, epoch: int) -> bool:
        """True only when this read epoch can be overwritten now."""
        state = self._page(page)
        self._epoch(epoch)
        return bool(state.epoch == epoch and state.read is not None and state.read.epoch == epoch
                    and state.read.safe and state.patch is None)

    def begin_overwrite(self, page: int, epoch: int, next_epoch: int,
                        expected_writes: Mapping[str, int]) -> None:
        """Reserve the next epoch after *all* old requests and responses."""
        state = self._page(page)
        self._epoch(epoch)
        self._epoch(next_epoch)
        if next_epoch != epoch + 1:
            raise ValueError(f"page {page}: overwrite must advance epoch {epoch} to {epoch + 1}")
        if not self.page_free(page, epoch):
            raise ValueError(f"page {page} epoch {epoch}: premature overwrite; issuer open or reads outstanding")
        state.patch = _Patch(next_epoch, _counts(expected_writes, label="expected_writes"))

    def start_write(self, page: int, epoch: int, kind: str, ticket: Hashable) -> None:
        """Accept one patch write command; it is not complete until acknowledged."""
        state = self._page(page)
        self._epoch(epoch)
        _kind(kind)
        _ticket(ticket)
        patch = state.patch
        if patch is None or patch.epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: no pending overwrite")
        if ticket in state.seen_tickets:
            raise ValueError(f"page {page} epoch {epoch}: duplicate ticket {ticket!r}")
        if patch.issued[kind] >= patch.expected[kind]:
            raise ValueError(f"page {page} epoch {epoch}: excess {kind} patch write")
        patch.tickets[ticket] = (kind, False)
        patch.issued[kind] += 1
        state.seen_tickets.add(ticket)

    def complete_write(self, page: int, epoch: int, kind: str, ticket: Hashable) -> None:
        """Retire one acknowledged write; both kinds must finish before reuse."""
        state = self._page(page)
        self._epoch(epoch)
        _kind(kind)
        _ticket(ticket)
        patch = state.patch
        if patch is None or patch.epoch != epoch:
            raise ValueError(f"page {page} epoch {epoch}: no pending overwrite")
        if ticket not in patch.tickets:
            raise ValueError(f"page {page} epoch {epoch}: unknown patch ticket {ticket!r}")
        issued_kind, done = patch.tickets[ticket]
        if issued_kind != kind:
            raise ValueError(f"page {page} epoch {epoch}: ticket {ticket!r} is {issued_kind}, not {kind}")
        if done:
            raise ValueError(f"page {page} epoch {epoch}: duplicate patch acknowledgement {ticket!r}")
        patch.tickets[ticket] = (kind, True)
        if patch.done:
            state.epoch = epoch
            state.patch = None

    def snapshot(self, page: int) -> dict:
        """Protocol state only; downstream arrivals and commit are unknown."""
        state = self._page(page)
        read, patch = state.read, state.patch
        return {
            "page": page, "ready_epoch": state.epoch,
            "read_epoch": read.epoch if read else None,
            "read_closed": read.closed if read else None,
            "read_issued": dict(read.issued) if read else None,
            "read_responded": (sum(done for _kind_name, done in read.tickets.values()) if read else None),
            "page_free": bool(read and read.epoch == state.epoch and read.safe and patch is None),
            "patch_epoch": patch.epoch if patch else None,
            "patch_issued": dict(patch.issued) if patch else None,
            "patch_acknowledged": (sum(done for _kind_name, done in patch.tickets.values()) if patch else None),
            "arrivals_ready": None, "output_committed": None,
            "calibrated": False, "tok_s": None,
        }
