"""Structural MXU result-row producer for straight-line RL4 int8 MM traces.

This models only the validity path from an accepted MM command and its granted
weight/activation advances (``pop``) to ``rf_push``. It does not predict when
the DRAM, scale FIFO or result FIFO makes a pop possible. Command starts,
consumer completion (``c_done``), and pop-head selection (``pn``) are explicit
frontend inputs; the trace validator must verify their provenance separately.
No observed ``rf_push`` or ``last_k`` signal enters this model.
"""

from collections import deque
from dataclasses import dataclass

from opentpu import isa as I


@dataclass(frozen=True)
class Rl4Int8Mm:
    """One MM's immutable RTL pop-head fields for the supported lane layout."""

    n: int
    kb: int
    unit: bool = False

    @property
    def row_groups(self) -> int:
        return (self.n + 3) // 4

    @property
    def advances(self) -> int:
        # RTL cmd_KBa: W8, no PAIR/T2/QUAD => KB advances per RL group.
        return self.row_groups * self.kb

    @classmethod
    def from_instr(cls, ins) -> "Rl4Int8Mm":
        if ins.op != I.MM or I.mm_lanes(ins) != 4:
            raise ValueError("producer supports only RL4 lane-mode MM instructions")
        wf, pair, quad = I.mm_format(ins)
        if wf != I.W8 or pair or quad or ins.w[6] & I.SRS_GW:
            raise ValueError("producer supports only int8 RL4 without PAIR/QUAD/group walk")
        if any((ins.ra, ins.rb, ins.rc)):
            raise ValueError("producer needs static N and KB (no address/count register operands)")
        n = ins.w[3] & 0xFFFF
        kb = (ins.w[3] >> 16) & 0xFF
        m = (ins.w[5] >> 16) & 0xFF
        if n < 1 or kb not in (1, 9) or m != 1:
            raise ValueError("producer supports static M1 RL4 int8 KB1/KB9 with nonzero N")
        return cls(n=n, kb=kb, unit=bool(ins.flags & I.F_UNIT))


@dataclass
class _ActiveMm:
    spec: Rl4Int8Mm
    popped: int = 0


class MxuRowProducer:
    """One pre-edge step per cycle; returns the modeled ``rf_push`` bit.

    ``program_mms`` is the straight-line sequence of MM instructions in start
    order. ``start`` is the accepted U_MXU start bit (slice ``start`` bit 1),
    ``done`` is the observed consumer ``c_done``, and ``pn`` selects the next
    pop-head when the old head is still draining. These three inputs are not
    inferred from ``pop``; an unsupported or inconsistent handoff raises.

    The pipeline has an unreset m0p register, m0 register and LROW-stage
    ``otpu_delay`` followed by ``rv``. Thus a row-ending pop at pre-edge cycle
    t appears as rf_push at t + LROW + 3; only the rv update is gated by rst.
    Every stage advances even during a bubble or reset, as in the RTL.
    """

    def __init__(self, program_mms, *, d: int = 128, mcols: int = 8, cl: int = 16):
        if d != 128 or mcols != 8 or cl not in (8, 16):
            raise ValueError("producer supports D128/MCOLS8 and CL8/CL16 only")
        self.commands = tuple(Rl4Int8Mm.from_instr(ins) for ins in program_mms)
        if not self.commands:
            raise ValueError("producer needs at least one MM instruction")
        self.lrow = min(cl, d // 8) + mcols + 5
        self.delay = self.lrow + 3
        self.stages = [0] * self.delay
        self.accepted = deque()
        self.next_start = 0
        self.ck = 0
        self.cycle = 0
        self.row_ends = 0
        self.loading = False

    def eligibility(self, *, f_count, s_count, pn):
        """G=1 W8 prerequisites from commands and modeled FIFO occupancies.

        ``more`` still needs sequencer release/head state, which is outside
        this component. An empty command queue has no active pop-head fields.
        """
        # pop_q deliberately excludes a command on its acceptance edge.
        # cl_ld reloads that command cache one edge later; its old inactive
        # prerequisite values cannot admit a pop while more is false.
        if not self.accepted or self.loading:
            return {}
        if int(pn) >= len(self.accepted):
            raise ValueError("pn has no queued pop-head MM")
        spec = self.accepted[int(pn)].spec
        return {"first_row": int(self.ck == 0), "c_unit": int(spec.unit),
                "f_av": int(f_count != 0), "s_av": int(s_count != 0),
                "row_in": int(f_count >= spec.kb and (spec.unit or s_count >= spec.kb)),
                "rf_capacity": 32 if self.lrow + 7 <= 32 else 64}

    def step(self, *, pop: bool, start: bool, done: bool, pn: bool, rst: bool) -> int:
        """Advance at one core edge, returning the predicted pre-edge rf_push."""
        pop, start, done, pn, rst = map(bool, (pop, start, done, pn, rst))
        cycle = self.cycle
        old_depth = len(self.accepted)
        predicted = self.stages[0]
        if rst and start:
            raise ValueError(f"cycle {cycle}: MM start in reset")
        if pn and len(self.accepted) != 2:
            raise ValueError(f"cycle {cycle}: pn has no queued pop-head MM")
        if pn and self.accepted[0].popped != self.accepted[0].spec.advances:
            raise ValueError(f"cycle {cycle}: pn before old pop-head finished its advances")
        if pop and not self.accepted:
            raise ValueError(f"cycle {cycle}: pop without accepted MM")
        if done and not rst and not self.accepted:
            raise ValueError(f"cycle {cycle}: c_done without accepted MM")

        row_end = 0
        if pop:
            active = self.accepted[int(pn)]
            if active.popped >= active.spec.advances:
                raise ValueError(f"cycle {cycle}: MM popped beyond its declared rows")
            # c_KBa = KB in the accepted RL4 int8 shape. ck is the registered
            # advance index, not an observed trace input.
            row_end = int(self.ck + 1 == active.spec.kb)
            active.popped += 1
            self.ck = 0 if row_end else self.ck + 1
            self.row_ends += row_end

        # The only reset gate in m0p -> m0 -> tl_v -> rv is rv's !rst.
        old = self.stages
        self.stages = [int(bool(old[1] and not rst)), *old[2:], row_end]

        # Synchronous queue updates. A simultaneous start/c_done can replace
        # the retiring head; pop and pn above refer to the pre-edge entries.
        if done and not rst:
            head = self.accepted.popleft()
            if head.popped != head.spec.advances:
                raise ValueError(f"cycle {cycle}: c_done before all MM advances were popped")
            if not pn:
                self.ck = 0   # RTL c_done overrides the pop's ck update
        if start:
            if self.next_start >= len(self.commands):
                raise ValueError(f"cycle {cycle}: MM start exceeds straight-line program")
            if len(self.accepted) >= 2:
                raise ValueError(f"cycle {cycle}: MXU command queue exceeds two entries")
            self.accepted.append(_ActiveMm(self.commands[self.next_start]))
            self.next_start += 1
        if rst:
            self.accepted.clear()
            self.ck = 0
            self.next_start = 0  # a subsequent run restarts the same program
        self.loading = bool(not rst and start and (old_depth == 0 or (done and old_depth == 1 and not pn)))
        self.cycle += 1
        return predicted

    def finish(self) -> None:
        """Reject traces ending before the expected program and pulses complete."""
        if self.next_start != len(self.commands):
            raise ValueError("trace ended before all MM commands were accepted")
        if self.accepted:
            raise ValueError("trace ended with unfinished MM commands")
        if any(self.stages):
            raise ValueError("trace ended before the MXU result-valid pipeline drained")
