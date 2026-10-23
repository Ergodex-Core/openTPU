"""Clocked MXU DRAM issuer for the bounded RL4/int8 wide-port fixtures.

The issuer's requests are predicted from accepted MM instructions and modeled
grants. A generated memory model supplies in-order A responses and completed B
entries; the consumer supplies FIFO pops. No observed RTL request, ``b_push``,
``r_cpl`` or ``rv1`` signal is an input. This is deliberately not a model of the
MXU frontend, DRAM arbitration, or the result producer.
"""

from collections import deque
from dataclasses import dataclass

from opentpu import isa as I


def _u32(value):
    return value & 0xffffffff


@dataclass(frozen=True)
class IssueCommand:
    sa: int
    ssa: int
    n: int
    kb: int
    rs: int
    srs: int
    k0: int
    unit: bool
    rsc: bool

    def __post_init__(self):
        if (not 0 < self.n < 65536 or self.kb not in (1, 9) or
                not 0 <= self.k0 < 256 or self.sa % 128 or self.ssa % 16 or
                self.rs % 16 or self.srs % 4 or self.rsc and self.unit):
            raise ValueError("issuer command violates static RL4 W8 M1 KB1/KB9 layout")

    @property
    def advances(self):
        return ((self.n + 3) // 4) * self.kb

    @classmethod
    def from_instr(cls, ins):
        if ins.op != I.MM or I.mm_lanes(ins) != 4:
            raise ValueError("issuer supports only RL4 lane-mode MM")
        wf, pair, quad = I.mm_format(ins)
        if wf != I.W8 or pair or quad or I.mm_walk(ins):
            raise ValueError("issuer supports only W8 RL4 without PAIR/QUAD/group walk")
        if any((ins.ra, ins.rb, ins.rc)):
            raise ValueError("issuer requires static MM operands")
        n = ins.w[3] & 0xffff
        kb = (ins.w[3] >> 16) & 0xff
        m = (ins.w[5] >> 16) & 0xff
        sa, ssa = ins.w[:2]
        rs = ins.w[4] & ~15
        srs = ins.w[6] & ~1
        k0 = (ins.w[3] >> 24) & 0xff
        if n < 1 or kb not in (1, 9) or m != 1:
            raise ValueError("issuer supports static M1 RL4 W8 KB1/KB9, N > 0")
        if sa % 128 or ssa % 16 or rs % 16 or srs % 4:
            raise ValueError("issuer requires RL4 chunk/scale alignment")
        unit = bool(ins.flags & I.F_UNIT)
        return cls(sa=_u32(sa), ssa=_u32(ssa), n=n, kb=kb, rs=rs,
                   srs=srs, k0=k0, unit=unit,
                   rsc=bool(ins.flags & I.F_RSC) and not unit)


@dataclass(frozen=True)
class ScaleTag:
    """Relevant ``atag_t`` bits; ``ent && cpl`` produces one scale FIFO entry."""

    ent: bool
    cpl: bool
    swe: bool
    swr: bool
    hv: int
    doff: int
    swo: int
    sro: int
    rsc: bool
    rl: bool = True
    lg: int = 2
    wf: int = I.W8


@dataclass(frozen=True)
class IssueSnapshot:
    cycle: int
    active: bool
    initializing: bool
    need_a: bool
    need_b: bool
    want: bool
    a_addr: int | None
    b_addr: int | None
    b_n: int | None
    tag: ScaleTag | None
    sw_only: bool
    advance: int | None
    left: int
    occ: int
    tg_n: int
    f_count: int
    s_count: int
    rv1: bool
    r_cpl: bool


@dataclass(frozen=True)
class IssueStep:
    """Requests and scale completion sampled at this edge, before state advances."""

    cycle: int
    a_req: bool
    b_req: bool
    a_addr: int | None
    b_addr: int | None
    b_n: int | None
    tag: ScaleTag | None
    r_cpl: bool
    rv1: bool


class MxuIssueModel:
    """One ``snapshot`` and ``step`` per pre-edge cycle.

    A start takes one edge to load the instruction and an ``i_ini`` edge to
    initialize addresses; the first request can appear on the following cycle.
    ``a_rvalid`` consumes the oldest *previously issued* A tag and sets ``rv1``
    for the next cycle. ``b_response`` means a complete B FIFO entry (RTL
    ``b_rvalid && b_rlast``), not a partial beat. ``fpop`` and ``scale_retire``
    come from an independent consumer model. The latter must be supplied for
    each pop because the pop-head's UNIT bit can differ from the issuer's.
    """

    def __init__(self, program_mms, *, d=128, mcols=8, rl=4, portb=4,
                 aline=64, depth=1024, tag_depth=128):
        if (d, mcols, rl, portb, aline, depth, tag_depth) != (128, 8, 4, 4, 64, 1024, 128):
            raise ValueError("issuer requires D128/MCOLS8/RL4/PORTB4/ALINE64/DEPTH1024/TD128")
        self.commands = tuple(IssueCommand.from_instr(x) if not isinstance(x, IssueCommand)
                              else x for x in program_mms)
        if not self.commands:
            raise ValueError("issuer requires at least one MM instruction")
        self.next_start = 0
        self.cycle = 0
        self._clear()

    def _clear(self):
        self.cmd = None
        self.i_ini = False
        self.i_act = False
        self.i_left = 0
        self.i_k = 0
        self.i_klast = False
        self.i_l1 = False
        self.row_addr = self.chunk_addr = 0
        self.srow_addr = self.srow_next = self.scale_addr = 0
        self.i_soff = 0
        self.i_swd = self.i_swh = self.i_swn = self.i_sws = False
        self.occ = self.f_count = self.s_count = self.tg_n = 0
        self.tg_t = self.tg_h = 0
        self.tag_mem = [None] * 128
        self.a_pending = deque()
        self.b_pending = deque()
        self.rv1 = False
        self.tg1 = None

    @staticmethod
    def _same_line(a, b):
        return (a >> 6) == (b >> 6)

    def _tag(self):
        cmd = self.cmd
        if cmd is None:
            raise ValueError("tag requested without an active command")
        if self.i_sws:
            ent = cpl = swe = False
            swr = True
            hv = 0
        elif cmd.rsc:
            ent = cpl = swe = True
            swr = False
            hv = 0
        else:
            ent = cpl = True  # RL4 W8 completes every advance's scale entry.
            swe = self.i_k == 0 and not self.i_swd and self.i_swh
            swr = self.i_klast and not self.i_l1 and self.i_swn
            hv = 3
        return ScaleTag(ent=ent, cpl=cpl, swe=swe, swr=swr, hv=hv,
                        doff=self.scale_addr & 63, swo=(self.srow_addr >> 2) & 15,
                        sro=((self.srow_addr if self.i_sws else self.srow_next) >> 2) & 15,
                        rsc=cmd.rsc)

    def snapshot(self):
        active = self.i_act
        cmd = self.cmd
        need_b = active and not self.i_sws
        need_a = active and not cmd.unit
        want = active and (not need_b or self.occ < 1024) and (not need_a or self.tg_n < 128)
        tag = self._tag() if need_a else None
        return IssueSnapshot(
            cycle=self.cycle, active=active, initializing=self.i_ini,
            need_a=need_a, need_b=need_b, want=want,
            a_addr=((self.srow_addr if self.i_sws or cmd.rsc else self.scale_addr) >> 2)
            if need_a else None,
            b_addr=self.chunk_addr >> 2 if need_b else None,
            b_n=3 if need_b else None, tag=tag,
            sw_only=bool(active and self.i_sws), advance=self.i_k if active else None,
            left=self.i_left, occ=self.occ, tg_n=self.tg_n,
            f_count=self.f_count, s_count=self.s_count,
            rv1=self.rv1, r_cpl=bool(self.rv1 and self.tg1 and self.tg1.ent and self.tg1.cpl))

    def step(self, *, start=None, a_gnt=False, b_gnt=False, a_rvalid=False,
             b_response=False, fpop=False, scale_retire=None, rst=False):
        """Advance one core edge; return independently predicted request pulses.

        ``start=True`` consumes the next declared command. An explicit
        ``IssueCommand`` or MM instruction may be supplied instead and must
        match that next command. A response must match a prior issued request;
        zero-cycle responses and underflow fail closed.
        """
        snap = self.snapshot()
        start_event = start is not None and start is not False
        if rst:
            if start_event or a_rvalid or b_response or fpop:
                raise ValueError("reset edge cannot carry issuer/consumer events")
            self._clear()
            self.next_start = 0
            self.cycle += 1
            return IssueStep(snap.cycle, False, False, None, None, None, None, False, snap.rv1)
        if start_event:
            if self.i_act or self.i_ini or self.next_start == len(self.commands):
                raise ValueError("start while issuer busy or beyond declared commands")
            if start is not True:
                spec = start if isinstance(start, IssueCommand) else IssueCommand.from_instr(start)
                if spec != self.commands[self.next_start]:
                    raise ValueError("start command does not match declared order")
        if fpop and scale_retire is None:
            raise ValueError("fpop requires explicit pop-head scale_retire")
        if scale_retire and not fpop:
            raise ValueError("scale_retire without fpop")
        if fpop and (self.occ == 0 or self.f_count == 0):
            raise ValueError("FIFO pop before an issued and completed B entry")
        if scale_retire and self.s_count == 0:
            raise ValueError("scale FIFO pop before a completed A entry")
        if a_rvalid and not self.a_pending:
            raise ValueError("A response without a previously issued request")
        if b_response and not self.b_pending:
            raise ValueError("B completion without a previously issued request")
        if snap.rv1 and self.tg_n == 0:
            raise ValueError("registered A response without an in-flight tag")

        go_iss = snap.want and (not snap.need_a or bool(a_gnt)) and (not snap.need_b or bool(b_gnt))
        a_req = bool(go_iss and snap.need_a)
        b_req = bool(go_iss and snap.need_b)
        issued = IssueStep(snap.cycle, a_req, b_req,
                           snap.a_addr if a_req else None, snap.b_addr if b_req else None,
                           snap.b_n if b_req else None, snap.tag if a_req else None,
                           snap.r_cpl, snap.rv1)

        # A response sees the old LUT-RAM tag at tg_h; a same-edge request
        # writes at tg_t. RTL retires the previous cycle's rv1, not a_rvalid.
        returned_tag = self.tag_mem[self.tg_h] if a_rvalid else None
        if a_rvalid:
            if returned_tag != self.a_pending.popleft():
                raise ValueError("A tag ring lost issue/response order")
            self.tg_h = (self.tg_h + 1) & 127
        if b_response:
            self.b_pending.popleft()
        if a_req:
            self.tag_mem[self.tg_t] = snap.tag
            self.a_pending.append(snap.tag)
            self.tg_t = (self.tg_t + 1) & 127
        if b_req:
            self.b_pending.append((snap.b_addr, snap.b_n))

        self.tg_n += int(a_req) - int(snap.rv1)
        self.occ += int(b_req) - int(fpop)
        self.f_count += int(b_response) - int(fpop)
        self.s_count += int(snap.r_cpl) - int(bool(scale_retire))
        if not (0 <= self.tg_n <= 128 and 0 <= self.occ <= 1024 and
                0 <= self.f_count <= 1024 and 0 <= self.s_count <= 1024):
            raise ValueError("issuer/FIFO occupancy outside RTL capacity")
        self.rv1 = bool(a_rvalid)
        if a_rvalid:
            self.tg1 = returned_tag

        if self.i_ini:
            cmd = self.cmd
            so = 16 + 4 * cmd.k0  # RL4 W8: H=max(4L,L)=16; block stride L bytes.
            sa = _u32(self.srow_addr + so)
            sn = _u32(self.srow_addr + cmd.srs)
            self.i_ini = False
            self.i_act = True
            self.i_k = 0
            self.i_klast = cmd.kb == 1
            self.i_l1 = self.i_left == 1
            self.i_soff = so
            self.scale_addr = sa
            self.srow_next = sn
            self.i_swd = False
            self.i_swh = self._same_line(self.srow_addr, sa)
            self.i_swn = self._same_line(sa, sn)
            self.i_sws = not cmd.unit and not cmd.rsc and not self.i_swh
        if go_iss:
            if self.i_sws:
                self.i_swd = True
                self.i_sws = False
            else:
                cmd = self.cmd
                self.i_left -= 1
                self.i_l1 = self.i_left == 1
                if self.i_left == 0:
                    self.i_act = False
                if self.i_klast:
                    rd = cmd.rs
                    sa = _u32(self.srow_next + self.i_soff)
                    sn = _u32(self.srow_next + cmd.srs)
                    self.i_k = 0
                    self.i_klast = cmd.kb == 1
                    self.row_addr = _u32(self.row_addr + rd)
                    self.chunk_addr = self.row_addr
                    self.srow_addr = self.srow_next
                    self.srow_next = sn
                    self.scale_addr = sa
                    self.i_swd = bool(snap.tag and snap.tag.swr)
                    self.i_swh = self._same_line(self.srow_addr, sa)
                    self.i_swn = self._same_line(sa, sn)
                    self.i_sws = not cmd.unit and not cmd.rsc and not self.i_swd and not self.i_swh
                else:
                    sa = _u32(self.scale_addr + 4)
                    self.i_k += 1
                    self.i_klast = self.i_k + 1 == cmd.kb
                    self.chunk_addr = _u32(self.chunk_addr + 4 * 128)
                    self.scale_addr = sa
                    self.i_swn = self._same_line(sa, self.srow_next)
        if start_event:
            cmd = self.commands[self.next_start]
            self.next_start += 1
            self.cmd = cmd
            self.i_ini = True
            self.i_left = cmd.advances
            self.row_addr = self.chunk_addr = cmd.sa
            self.srow_addr = cmd.ssa
        self.cycle += 1
        return issued

    def finish(self):
        if self.next_start != len(self.commands) or self.i_ini or self.i_act:
            raise ValueError("trace ended before all declared MMs finished issuing")
        if self.occ or self.tg_n or self.f_count or self.s_count or self.a_pending or self.b_pending or self.rv1:
            raise ValueError("trace ended before issuer requests and FIFO entries drained")
