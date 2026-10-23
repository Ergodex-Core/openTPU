"""Command-generated, single-chunk DMA ST producer for the reuse fixture.

Only aligned 1..16-word stores on the D128/LANES8 board are supported. Source
words are supplied by an independent functional memory oracle and are stable
throughout each store. The RTL has no DMA enable input: its conflict-free,
highest-priority eight-lane reads advance each cycle. The pa0/pa/st_pend input
and gather pipeline, chunk buffering, run start, B-ready hold and wr_idle
completion are modeled directly. Request/ack timestamps are not inputs.
Accepted ST starts remain an explicit frontend input.
"""
from dataclasses import dataclass
from opentpu import isa as I
from tools.wideport_native_adapter import BWrite


@dataclass(frozen=True)
class StoreCommand:
    id: object
    instruction: object
    words: tuple[int, ...]

    def request(self):
        ins = self.instruction
        if ins.op != I.ST or any((ins.ra, ins.rb, ins.rc)):
            raise ValueError("store producer requires static ST")
        address, source, n = ins.w[:3]
        if address % 128 or source % 8 or not 1 <= n <= 16 or len(self.words) != n:
            raise ValueError("store producer requires chunk/segment aligned 1..16 words")
        value = sum((int(w) & 0xFFFFFFFF) << (32 * i) for i, w in enumerate(self.words))
        return BWrite(self.id, address // 4, value, (1 << n) - 1)


class DmaStore:
    def __init__(self):
        self.busy = self.ackw = self.done = False
        self.command = None
        self.sleft = 0
        self.pa0 = self.pla0 = self.pa = self.pla = self.pend = self.pl = False
        self.occ = 0
        self.finished = self.write_run = False
        self.accepted = self.issued = self.completed = 0
        self.read_words = []

    def snapshot(self):
        request = self.command.request() if self.busy and not self.ackw and self.write_run else None
        return dict(request=request, done=self.done, busy=self.busy, read_words=self.read_words)

    def step(self, *, start=None, b_ready=False, wr_idle=False, reset=False):
        before = self.snapshot()
        self.done = False
        self.read_words = []
        if reset:
            if self.busy:
                raise ValueError("warm DMA store reset is outside the fixture")
            return before
        if start is not None:
            if self.busy:
                raise ValueError("ST start while DMA store busy")
            start.request()  # fail closed before changing state
            self.command = start
            self.busy = True
            self.ackw = self.finished = self.write_run = False
            self.sleft = (len(start.words) + 7) // 8
            self.pa0 = self.pla0 = self.pa = self.pla = self.pend = self.pl = False
            self.occ = 0
            self.accepted += 1
        elif self.ackw:
            if wr_idle:
                self.ackw = self.busy = False
                self.done = True
                self.completed += 1
        elif self.busy:
            push = bool(self.pend and self.pl)
            issue = bool(self.write_run and b_ready)
            new_occ = self.occ + int(push) - int(issue)
            if issue:
                self.write_run = False
                self.issued += 1
            elif not self.write_run and self.finished and self.occ:
                self.write_run = True
            if self.finished and not before["request"] and self.occ == 0:
                self.ackw = True
            new_pending = self.pa
            if self.pend and not self.pa and not self.pa0 and self.sleft == 0:
                self.finished = True
                new_pending = False
            rd = self.sleft > 0
            if rd:
                offset = ((len(self.command.words) + 7) // 8 - self.sleft) * 8
                self.read_words = [self.command.instruction.w[1] + i for i in
                                   range(offset, min(offset + 8, len(self.command.words)))]
            self.pa0, self.pla0, self.pa, self.pla, self.pend, self.pl = (
                rd, bool(rd and self.sleft == 1), self.pa0, self.pla0, new_pending, self.pla)
            self.sleft -= int(rd)
            self.occ = new_occ
            if not 0 <= self.occ <= 1:
                raise ValueError("single-chunk store occupancy outside capacity")
        return before

    def finish(self):
        if self.busy or self.accepted != self.issued or self.issued != self.completed:
            raise ValueError("DMA stores have not fully issued and acknowledged")
        return dict(starts=self.accepted, writes=self.issued, completions=self.completed)
