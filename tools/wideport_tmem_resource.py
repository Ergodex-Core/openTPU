"""Cycle-stepped bank/replay resource for a future structural wide-port LPN.

This models the arbiter boundary in otpu_tmem.g_one, not instruction execution.
Callers supply post-slice-gating fresh requests, whole-port read takes, and the
registered landing writes. Unit enables, WBUF/result FIFO and DRAM feedback are
deliberately outside this component; the full LPN's TPS gate must stay closed.
No observed elapsed time, replay count, or fitted coefficient is an input.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TmemReplayConfig:
    pmp: int = 0
    lanes: int = 8
    words: int = 65536
    nsb: int = 2
    sbp: int = 3
    # Port order: MXU, DMA, QUANT, VPU A, VPU B, Q3 guest, Q2, COLL.
    ranks: tuple = (4, 0, 5, 2, 3, 7, 6, 1)
    host: int = 1
    guests: tuple = (5, 7)
    replay_ports: tuple = (0, 2, 3, 4, 5, 6)
    side_mask: int | None = None

    def __post_init__(self):
        if self.pmp not in (0, 1, 2):
            raise ValueError("PMP must be 0, 1, or the RTL's core-clock PMP2 model")
        if (self.lanes, self.nsb, self.sbp) != (8, 2, 3):
            raise ValueError("only the wide-port LANES8/NSB2/SBP3/SBH0/TDP/SHARE configuration is modeled")
        if self.words < 16 or self.words & (self.words - 1):
            raise ValueError("WORDS must be a power of two")
        if sorted(self.ranks) != list(range(8)) or self.host in self.replay_ports:
            raise ValueError("expected eight distinct ranks and a non-replaying host")
        if self.host != 1 or self.guests != (5, 7) or self.replay_ports != (0, 2, 3, 4, 5, 6):
            raise ValueError("only the slice's DMA host, Q3/COLL guests and replay mask are modeled")
        if self.side_mask is not None and not 0 <= self.side_mask < 256:
            raise ValueError("side mask must have eight bits")

    @property
    def sides(self):
        # otpu_slice TM_SIDED; changing PMP changes this split as well as writes.
        mask = self.side_mask if self.side_mask is not None else (0xF0 if self.pmp == 0 else 0x0C)
        return tuple((mask >> p) & 1 for p in range(8))


class WriteLandingPipeline:
    """Granted write -> IQ -> pw, two core edges (otpu_tmem IQ=1).

    step() returns the occupants at this cycle's pre-edge observation point and
    accepts this cycle's newly granted writes. Do not use this together with
    already registered pw occupants: that would count the delay twice.
    """
    def __init__(self):
        self.stages = deque([(), ()])

    def step(self, granted_words):
        landed = self.stages.popleft()
        self.stages.append(tuple(granted_words))
        return landed


class TmemReplayResource:
    """Address-level read-side arbitration, pending replay and age transitions.

    reads[p] contains enabled word addresses, as seen at otpu_tmem.r_req. A
    pending replay replaces that port's fresh request. taken[p] is |r_en[p];
    partial lane takes are intentionally unsupported. writes contains pw words,
    one per bank. Returned masks describe the current cycle before the edge;
    pending state advances only after those masks have been computed.
    """
    def __init__(self, config=None):
        self.cfg = config or TmemReplayConfig()
        self.pending = [None] * 8
        self.born = [None] * 8
        self.cycle = 0

    def _banks(self, words):
        banks = {}
        for word in words:
            if not isinstance(word, int) or not 0 <= word < self.cfg.words:
                raise ValueError("word address outside TMEM")
            bank = word % 8
            if bank in banks and banks[bank] != word:
                raise ValueError("one port cannot ask two different words in one bank")
            banks[bank] = word
        return banks

    @staticmethod
    def _mask(bits):
        return sum(1 << p for p, value in enumerate(bits) if value)

    def step(self, reads, taken, writes=()):
        if len(reads) != 8 or len(taken) != 8 or any(x not in (False, True) for x in taken):
            raise ValueError("expected eight fresh request vectors and read-take booleans")
        fresh = [self._banks(words) for words in reads]
        landing = self._banks(writes)
        if len(landing) != len(writes):
            raise ValueError("at most one write per bank is supported")
        # GST collapses Q3/COLL onto the DMA host; guests have no independent asker.
        host_taken = bool(fresh[self.cfg.host])
        for guest in self.cfg.guests:
            if not host_taken and fresh[guest]:
                fresh[self.cfg.host] = dict(fresh[guest])
                host_taken = True
            fresh[guest] = {}
        pending_before = [request is not None for request in self.pending]
        asks = [self.pending[p] if pending_before[p] else fresh[p] for p in range(8)]
        sides, ranks = self.cfg.sides, self.cfg.ranks

        def earlier(other, port):
            if pending_before[other] and pending_before[port]:
                nreplay = sum(p in self.cfg.replay_ports and p not in self.cfg.guests and sides[p] == sides[port]
                              for p in range(8))
                if nreplay >= 3 and other in self.cfg.replay_ports and port in self.cfg.replay_ports:
                    return (self.born[other], ranks[other]) < (self.born[port], ranks[port])
                return ranks[other] < ranks[port]
            def rank(p):
                return 1 + ranks[p] if pending_before[p] else 0 if p == self.cfg.host else 9 + ranks[p]
            return rank(other) < rank(port)

        # A landing write owns port A on PMP0, including when it supplies the
        # same word through bypass. Only a *different* word makes wm/lost true.
        write_owns = [{bank for bank, word in asks[p].items()
                       if sides[p] and self.cfg.pmp == 0 and bank in landing
                       and ((landing[bank] >> 3) & 1) == ((word >> 3) & 1)} for p in range(8)]
        write_conflict = [any(landing[b] != asks[p][b] for b in write_owns[p]) for p in range(8)]
        blockers = [[] for _ in range(8)]
        # Match cp/hm's lower-index pair definition, including write-owned RAMs.
        for i in range(8):
            for other in range(i):
                if sides[i] != sides[other]:
                    continue
                conflict = any(b in asks[other] and b not in write_owns[i]
                               and ((asks[other][b] >> 3) & 1) == ((word >> 3) & 1)
                               and asks[other][b] != word for b, word in asks[i].items())
                if conflict:
                    loser, winner = (i, other) if earlier(other, i) else (other, i)
                    blockers[loser].append(winner)
        blocked = [write_conflict[p] or bool(blockers[p]) for p in range(8)]
        lost = [p in self.cfg.replay_ports and (pending_before[p] or bool(taken[p])) and blocked[p]
                for p in range(8)]
        # The RTL assertion forbids overwriting an outstanding replay's word.
        for p in range(8):
            if pending_before[p] and (lost[p] or self.cfg.pmp == 1):
                if any(landing.get(bank) == word for bank, word in asks[p].items()):
                    raise ValueError("write overwrote a word awaiting replay")
        result = {"cycle": self.cycle, "lost": self._mask(lost), "frozen": self._mask(pending_before),
                  "blocked": self._mask(blocked), "self_blocked": 0,
                  "write_conflict": self._mask(write_conflict), "read_blockers": blockers}
        for p in range(8):
            if lost[p]:
                if not pending_before[p]:
                    self.pending[p] = dict(fresh[p])
                    self.born[p] = self.cycle
            else:
                self.pending[p] = self.born[p] = None
        self.cycle += 1
        return result
