"""Command-driven VPU producer for the narrow, non-stream V_ADD/B_FULL case.

This reproduces request masks/addresses, write masks/addresses and ``done_i``.
The caller supplies accepted command-start events and the VPU's ``en`` each
core edge. It does not compute fp32 data or predict ``en``/TMEM/DRAM grants.
Only a cold reset (or a reset after the metadata pipeline has drained) is
supported: the RTL's slot delay line has no reset and can retain old metadata
across a short warm reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from opentpu import isa as I


@dataclass(frozen=True)
class VAddCommand:
    dst: int
    a: int
    b: int
    rows: int
    cols: int
    drs: int
    ars: int
    brs: int

    def __post_init__(self):
        if not (0 < self.rows < 1 << 16 and 0 < self.cols < 1 << 16):
            raise ValueError("V_ADD needs nonempty 16-bit rows and columns")
        if any(not 0 <= stride < 1 << 16 for stride in (self.drs, self.ars, self.brs)):
            raise ValueError("V_ADD strides must fit 16 bits")
        if any(not 0 <= base < 1 << 20 for base in (self.dst, self.a, self.b)):
            raise ValueError("V_ADD TMEM bases must fit the VPU's 20-bit address")
        for base, stride in ((self.dst, self.drs), (self.a, self.ars), (self.b, self.brs)):
            if base + (self.rows - 1) * stride + self.cols > 1 << 20:
                raise ValueError("V_ADD addresses wrap the VPU's 20-bit address")

    @classmethod
    def from_words(cls, words: Sequence[int], *, tmem_words: int = 1 << 16) -> VAddCommand:
        """Decode one eight-word ISA instruction, rejecting unsupported modes."""
        if len(words) != 8 or any(not isinstance(w, int) or not 0 <= w < 1 << 32 for w in words):
            raise ValueError("expected eight unsigned 32-bit instruction words")
        w0, dst, a, b, w4, w5, w6, imm = words
        if w0 != I.VOP or (w6 >> 16) != I.V_ADD or imm != 0:
            raise ValueError("only register-free V_ADD/B_FULL with zero flags/immediate is supported")
        cmd = cls(dst, a, b, w4 & 0xffff, w4 >> 16, w5 & 0xffff, w5 >> 16, w6 & 0xffff)
        if tmem_words < 1 or tmem_words > 1 << 20:
            raise ValueError("invalid TMEM size")
        for base, stride in ((cmd.dst, cmd.drs), (cmd.a, cmd.ars), (cmd.b, cmd.brs)):
            if base + (cmd.rows - 1) * stride + cmd.cols > tmem_words:
                raise ValueError("V_ADD accesses beyond TMEM")
        return cmd


@dataclass(frozen=True)
class _Beat:
    read_a: tuple[int, ...]
    read_b: tuple[int, ...]
    write: tuple[int, ...]
    last: bool


class VAddProducer:
    """Pre-edge snapshots and edge transitions for V_ADD with LANES=8, SE=1.

    ``step`` returns the snapshot *before* applying this edge's ``start`` and
    ``en``. ``start`` is the sequencer's accepted VPU start on that edge, not
    an instruction-fetch event. Queue depth is two, matching the RTL.
    """

    def __init__(self):
        self.queue: list[VAddCommand] = []
        self.active: VAddCommand | None = None
        self.row = self.chunk = 0
        self.mi: _Beat | None = None
        self.mi2: _Beat | None = None
        self.m0: _Beat | None = None
        self.slot: list[_Beat | None] = [None] * 7  # V_ADD's SL=1+LM+LA=7 enabled edges
        self.read_a: tuple[int, ...] = ()
        self.read_b: tuple[int, ...] = ()
        self.done_i = 0
        self.ew_n = 0
        self._ever_issued = False

    def snapshot(self) -> dict:
        mo = self.slot[-1]
        return {
            "ta_words": list(self.read_a), "tb_words": list(self.read_b),
            "ta_mask": sum(1 << (word - self.read_a[0]) for word in self.read_a) if self.read_a else 0,
            "tb_mask": sum(1 << (word - self.read_b[0]) for word in self.read_b) if self.read_b else 0,
            "cw_words": list(mo.write) if mo else [],
            "cw_mask": (1 << len(mo.write)) - 1 if mo else 0,
            "done_i": self.done_i, "red_thr": 1,
            "issuing": int(self.active is not None), "queued": len(self.queue), "ew_n": self.ew_n,
        }

    def _issue(self) -> _Beat | None:
        cmd = self.active
        if cmd is None:
            return None
        col = self.chunk * 8
        lanes = range(min(8, cmd.cols - col))
        return _Beat(
            tuple(cmd.a + self.row * cmd.ars + col + lane for lane in lanes),
            tuple(cmd.b + self.row * cmd.brs + col + lane for lane in lanes),
            tuple(cmd.dst + self.row * cmd.drs + col + lane for lane in lanes),
            self.row + 1 == cmd.rows and col + 8 >= cmd.cols,
        )

    def step(self, *, start: VAddCommand | Sequence[int] | None = None,
             en: bool = False, reset: bool = False) -> dict:
        before = self.snapshot()
        if reset:
            if start is not None:
                raise ValueError("cannot start V_ADD on a reset edge")
            if self._ever_issued and (self.active or self.queue or self.mi or self.mi2 or self.m0
                                      or any(self.slot)):
                raise ValueError("warm reset with in-flight V_ADD metadata is unsupported")
            self.__init__()
            return before
        if start is not None:
            if not isinstance(start, VAddCommand):
                start = VAddCommand.from_words(start)
            if len(self.queue) >= 2:
                raise ValueError("VPU command queue full at start")

        old_active = self.active
        issue = self._issue() if en else None
        mo = self.slot[-1]
        next_done = int(bool(en and mo and mo.last))
        if en:
            self.read_a = issue.read_a if issue else ()
            self.read_b = issue.read_b if issue else ()
            self.slot = [self.m0, *self.slot[:-1]]
            self.m0, self.mi2, self.mi = self.mi2, self.mi, issue
            if issue:
                self._ever_issued = True
                assert old_active is not None
                if (self.chunk + 1) * 8 >= old_active.cols:
                    self.chunk = 0
                    if self.row + 1 == old_active.rows:
                        self.active = None
                    else:
                        self.row += 1
                else:
                    self.chunk += 1
            # can_begin uses the *old* queue and old issuing flag. A concurrent
            # start cannot begin on the same edge, as in otpu_vpu's cq_n test.
            if old_active is None and self.queue:
                self.active = self.queue.pop(0)
                self.row = self.chunk = 0
                self.ew_n += 1
        if start is not None:
            self.queue.append(start)
        self.ew_n -= next_done
        if self.ew_n < 0:
            raise ValueError("V_ADD completion without an outstanding instruction")
        # RTL done_i is unconditionally cleared, then set only by an enabled
        # final mo beat (or an empty command, excluded by VAddCommand).
        self.done_i = next_done
        return before
