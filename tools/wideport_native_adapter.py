"""Cycle model of the NCH8 wide-port native adapter's A/B read and B write path.

This models ``rtl/mem/otpu_native_dram.sv`` at its slice and native-channel
pins. A and B requests here are already in the slice's output registers. The
native memory supplies independently predicted ``ready`` and ordered response
IDs/write-done counts; no RTL result, counter, or completion cycle is an input.
A and SW writes remain unsupported. A caller must model the slice's A/B registers,
its B response register, and the MXU's scale-tag pipeline separately.

``step`` returns signals visible *before* the rising edge named by ``cycle``;
it then applies that edge. Native commands are valid even when ``ready`` is
false and must be presented to the memory model for that same cycle. Response
IDs are the IDs of commands accepted in earlier cycles, in channel order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class ARead:
    id: Hashable
    word_addr: int


@dataclass(frozen=True)
class BRead:
    id: Hashable
    word_addr: int
    chunks: int = 1
    tag: bool = False  # DMA/loader B read, as opposed to MXU B read


@dataclass(frozen=True)
class BWrite:
    id: Hashable
    word_addr: int
    data: int          # little-endian 128-byte chunk, word 0 at bits 31:0
    word_mask: int     # 32 bits, one write enable per 32-bit word


@dataclass(frozen=True)
class NativePins:
    ready: tuple[bool, ...]
    response_ids: tuple[Hashable | None, ...]
    write_ready: tuple[bool, ...] | None = None
    write_done: tuple[int, ...] | None = None


@dataclass(frozen=True)
class NativeCommand:
    id: Hashable
    channel: int
    beat_addr: int
    source: str  # "A" or "B"
    accepted: bool
    write: bool = False
    payload: int | None = None  # 512-bit channel beat for a write
    mask: int | None = None     # 64 byte enables; 1 writes the byte
    cvalid: bool = True
    wvalid: bool = False


@dataclass(frozen=True)
class BPart:
    id: Hashable
    tag: bool
    chunk_mask: int
    last: bool


@dataclass(frozen=True)
class BChunkSources:
    """The two native read IDs used for one enabled 128-byte B chunk."""
    chunk: int
    low_id: Hashable   # logical bytes 0..63, regardless of CHASH channel swap
    high_id: Hashable  # logical bytes 64..127


@dataclass(frozen=True)
class StepResult:
    cycle: int
    a_ready: bool
    b_ready: bool
    a_accepted: bool
    b_accepted: bool
    native_commands: tuple[NativeCommand, ...]
    a_response_id: Hashable | None
    b_response: BPart | None
    wr_idle: bool | None = None
    rd_idle: bool = True
    a_response_source_id: Hashable | None = None
    b_response_sources: tuple[BChunkSources, ...] = ()


@dataclass(frozen=True)
class _BPlan:
    request: BRead | BWrite
    pairs: tuple[int, ...]
    beats: tuple[int, ...]
    first_mask: int
    second_mask: int


@dataclass(frozen=True)
class _BTag:
    part: BPart
    pairs: tuple[int, ...]
    chunks: tuple[tuple[int, int, bool], ...]  # (request chunk index, pair, CHASH half swap)


@dataclass(frozen=True)
class _AOrder:
    id: Hashable
    channel: int
    reuse: bool
    cache_slot: int
    drop: int


class NativeAdapter:
    """Direct NCH8/PORTB4/APF32/APR16 read and masked B-write contract.

    All queues represent pre-edge RTL state. In particular, responses arriving
    on an edge cannot satisfy a B join or A order-head until the next cycle.
    The model intentionally fails closed on any unsupported request shape.
    """

    NCH = 8
    PORTB = 4
    QD = 16
    RD = 128
    OD = 256
    AD = 64
    APF = 32
    APR = 16

    def __init__(self) -> None:
        self.cycle = 0
        self.breg: _BPlan | None = None
        self.breg_first = False
        self.breg_second = False
        self.bq: list[deque[NativeCommand]] = [deque() for _ in range(8)]
        self.aq: list[deque[tuple[ARead, int]]] = [deque() for _ in range(8)]
        self.a_issue = [0] * 8
        self.held_source: list[str | None] = [None] * 8
        self.command_done = [False] * 8
        self.data_done = [False] * 8
        self.write_issued = [0] * 8
        self.write_waiting = 0        # wq_n: accepted beats before last cycle, not yet issued
        self.write_accepted_last = 0  # popcount(wacc_q): last cycle's accepted write halves
        self.issued_all_reg = False   # iss_all_q: one-edge registered comparison to n_wdone
        self.write_invalidate: list[int | None] = [None] * 8  # wk_v/wk_m
        self.native_tags: list[deque[NativeCommand]] = [deque() for _ in range(8)]
        self.b_fifo: list[deque[Hashable]] = [deque() for _ in range(8)]
        self.a_fifo: list[deque[Hashable]] = [deque() for _ in range(8)]
        self.b_reserved = [0] * 8
        self.a_reserved = [0] * 8
        self.b_tags: deque[_BTag] = deque()
        self.a_order: deque[_AOrder] = deque()
        self.a_valid_reg: Hashable | None = None
        self.run_valid = [False] * 8
        self.run_next = [0] * 8
        self.run_left = [0] * 8
        self.cache: list[tuple[int, Hashable] | None] = [None] * 16
        self.cache_data_id: list[Hashable | None] = [None] * 16
        self.cache_next = 0
        self.a_valid_reg_source_id: Hashable | None = None

    @staticmethod
    def pair(word_addr: int) -> int:
        """CHASH fold of word-address chunk bits 31:5, PIB=2."""
        m = (word_addr >> 5) & ((1 << 27) - 1)
        p = 0
        for shift in range(0, 27, 2):
            p ^= (m >> shift) & 3
        return p

    @classmethod
    def channel(cls, word_addr: int) -> int:
        m = (word_addr >> 5) & ((1 << 27) - 1)
        return 2 * cls.pair(word_addr) + (((word_addr >> 4) & 1) ^ (m.bit_count() & 1))

    @staticmethod
    def beat(word_addr: int) -> int:
        return (word_addr >> 7) & ((1 << 25) - 1)

    @classmethod
    def _plan_b(cls, req: BRead | BWrite) -> _BPlan:
        if isinstance(req, BWrite):
            if req.word_addr & 31 or not 0 <= req.word_addr < 1 << 32:
                raise ValueError("B write needs one 128-byte chunk-aligned address")
            if not 0 < req.word_mask < 1 << 32 or not 0 <= req.data < 1 << 1024:
                raise ValueError("B write needs 1024-bit data and a nonzero 32-bit word mask")
            return _BPlan(req, (cls.pair(req.word_addr),), (cls.beat(req.word_addr),), 1, 0)
        if not 1 <= req.chunks <= 4 or req.word_addr & 31:
            raise ValueError("B read needs 1..4 chunk-aligned chunks")
        pairs = tuple(cls.pair(req.word_addr + 32 * i) for i in range(req.chunks))
        beats = tuple(cls.beat(req.word_addr + 32 * i) for i in range(req.chunks))
        seen: set[int] = set()
        first = second = 0
        for i, pair in enumerate(pairs):
            if pair in seen:
                second |= 1 << i
            else:
                first |= 1 << i
                seen.add(pair)
        return _BPlan(req, pairs, beats, first, second)

    def step(
        self,
        pins: NativePins,
        *,
        a: ARead | None = None,
        b: BRead | BWrite | None = None,
        a_flush: bool = False,
        reset: bool = False,
    ) -> StepResult:
        if len(pins.ready) != 8 or len(pins.response_ids) != 8:
            raise ValueError("NCH8 needs eight native ready and response inputs")
        if a is not None and (not isinstance(a, ARead) or a.id is None):
            raise TypeError("only identified A reads are supported")
        if b is not None and (not isinstance(b, (BRead, BWrite)) or b.id is None):
            raise TypeError("only identified B reads/writes are supported; dict/SW outside scope")
        if a is not None and not 0 <= a.word_addr < 1 << 32:
            raise ValueError("A address must be a 32-bit word address")
        if b is not None and not 0 <= b.word_addr < 1 << 32:
            raise ValueError("B address must be a 32-bit word address")
        if pins.write_ready is not None and len(pins.write_ready) != 8:
            raise ValueError("NCH8 needs eight native write-ready inputs")
        if pins.write_done is not None and (len(pins.write_done) != 8 or
                                            any(not 0 <= x < 1 << 16 for x in pins.write_done)):
            raise ValueError("NCH8 needs eight 16-bit native write-done counts")
        if (isinstance(b, BWrite) or self.write_waiting or self.write_accepted_last or
                (self.breg is not None and isinstance(self.breg.request, BWrite)) or
                any(self.command_done) or any(self.data_done) or
                any(q and q[0].write for q in self.bq)) and (
                pins.write_ready is None or pins.write_done is None):
            raise ValueError("B writes require native write-ready and write-done pins")
        if isinstance(b, BWrite):
            self._plan_b(b)  # fail closed before changing any queue state

        # Combinational outputs from pre-edge queue counts (RTL :362-389).
        b_hold = self.breg_first and self.breg is not None and bool(self.breg.second_mask)
        b_ready = all(len(q) < self.QD - 2 for q in self.bq) and len(self.b_tags) < self.OD - 2 and not b_hold
        a_ready = all(len(q) < self.QD for q in self.aq) and len(self.a_order) < self.OD
        b_accept = b is not None and b_ready
        a_accept = a is not None and a_ready
        breg_write = bool(self.breg_first and self.breg is not None and
                          isinstance(self.breg.request, BWrite))
        breg_word_mask = self.breg.request.word_mask if breg_write else 0
        wr_idle = (self.write_waiting == 0 and self.write_accepted_last == 0 and
                   self.issued_all_reg and not breg_write) if pins.write_done is not None else None
        rd_idle = not self.b_tags and not self.a_order and not (
            self.breg is not None and (self.breg_first or self.breg_second) and
            isinstance(self.breg.request, BRead))
        if reset:
            if a is not None or b is not None:
                raise ValueError("native adapter reset does not accept A/B requests")
            result = StepResult(self.cycle, a_ready, b_ready, False, False, (),
                                self.a_valid_reg, None, wr_idle, rd_idle,
                                self.a_valid_reg_source_id)
            next_issued_all = (pins.write_done is not None and
                               all(self.write_issued[c] == pins.write_done[c] for c in range(8)))
            cycle = self.cycle + 1
            self.__init__()
            self.cycle = cycle
            self.issued_all_reg = next_issued_all
            if pins.write_done is not None:
                self.write_issued = list(pins.write_done)
            return result

        # Channel issue priority A then B, with a stalled source held (:640-684).
        shown: list[NativeCommand | None] = [None] * 8
        for c in range(8):
            a_eligible = bool(self.aq[c]) and (
                self.a_issue[c] != 0 or self.a_reserved[c] <= self.AD - self.APF
            )
            b_eligible = bool(self.bq[c]) and (
                self.bq[c][0].write or self.b_reserved[c] < self.RD)
            source = self.held_source[c] or ("A" if a_eligible else "B" if b_eligible else None)
            if source == "A":
                req, beat = self.aq[c][0]
                idx = self.a_issue[c]
                shown[c] = NativeCommand(("A", req.id, beat + idx), c, beat + idx, "A", bool(pins.ready[c]))
            elif source == "B":
                cmd = self.bq[c][0]
                if cmd.write:
                    cv = not self.command_done[c]
                    wv = not self.data_done[c]
                    took_c = cv and pins.ready[c]
                    took_w = wv and pins.write_ready[c]
                    issued = ((self.command_done[c] or took_c) and
                              (self.data_done[c] or took_w))
                    shown[c] = NativeCommand(cmd.id, c, cmd.beat_addr, "B", issued,
                                             True, cmd.payload, cmd.mask, cv, wv)
                else:
                    shown[c] = NativeCommand(cmd.id, c, cmd.beat_addr, "B",
                                             bool(pins.ready[c]))

        # The oldest B part joins only *stored* responses (:705-725).
        b_response: BPart | None = None
        b_response_sources: tuple[BChunkSources, ...] = ()
        if self.b_tags:
            head = self.b_tags[0]
            if all(self.b_fifo[2 * p] and self.b_fifo[2 * p + 1] for p in head.pairs):
                b_response = head.part
                b_response_sources = tuple(
                    BChunkSources(i, self.b_fifo[2 * p + int(sw)][0],
                                  self.b_fifo[2 * p + int(not sw)][0])
                    for i, p, sw in head.chunks)

        # A order head may reuse its APR entry or wait for all prior run beats
        # slated for discard plus its selected beat (:729-755).
        a_order_out: _AOrder | None = None
        if self.a_order:
            head_a = self.a_order[0]
            if head_a.reuse or len(self.a_fifo[head_a.channel]) > head_a.drop:
                a_order_out = head_a
        a_response = self.a_valid_reg
        a_response_source = self.a_valid_reg_source_id

        # wk_v/wk_m are the previous edge's completed channel writes. The
        # invalidation is applied on this edge, after pre-edge reuse/run tests.
        cache_invalid = set()
        run_invalid = set()
        for c, beat in enumerate(self.write_invalidate):
            if beat is None:
                continue
            if self.run_valid[c] and ((beat - self.run_next[c]) & ((1 << 25) - 1)) < self.run_left[c]:
                run_invalid.add(c)
            for i, entry in enumerate(self.cache):
                if entry is not None and self.channel(entry[0] << 4) == c and self.beat(entry[0] << 4) == beat:
                    cache_invalid.add(i)

        # All actions below are one rising edge. Capture the previous BREG part
        # before replacing it with the newly accepted request.
        if self.breg is not None and (self.breg_first or self.breg_second):
            plan = self.breg
            mask = plan.first_mask if self.breg_first else plan.second_mask
            last = not b_hold
            active_pairs = tuple(plan.pairs[i] for i in range(len(plan.pairs)) if mask & (1 << i))
            if isinstance(plan.request, BRead):
                chunks = tuple((i, plan.pairs[i], bool(((plan.request.word_addr + 32 * i) >> 5).bit_count() & 1))
                               for i in range(len(plan.pairs)) if mask & (1 << i))
                self.b_tags.append(_BTag(BPart(plan.request.id, plan.request.tag, mask, last),
                                         active_pairs, chunks))
            for i in range(len(plan.pairs)):
                if not mask & (1 << i):
                    continue
                for half in (0, 1):
                    if isinstance(plan.request, BWrite):
                        # Logical low/high 64B may be swapped between the
                        # pair's physical channels by CHASH parity.
                        words = (plan.request.word_mask >> (16 * half)) & 0xFFFF
                        if not words:
                            continue
                        payload = (plan.request.data >> (512 * half)) & ((1 << 512) - 1)
                        bytemask = sum(0xF << (4 * j) for j in range(16) if words & (1 << j))
                    else:
                        payload = bytemask = None
                    physical_half = half ^ (((plan.request.word_addr + 32 * i) >> 5).bit_count() & 1)
                    c = 2 * plan.pairs[i] + physical_half
                    cmd_id = ("B", plan.request.id, i, half)
                    self.bq[c].append(NativeCommand(cmd_id, c, plan.beats[i], "B", False,
                                                    isinstance(plan.request, BWrite), payload, bytemask))
        self.breg_first = b_accept
        self.breg_second = b_hold
        if b_accept:
            self.breg = self._plan_b(b)
        elif not b_hold:
            self.breg = None

        # A accept uses pre-edge cache/run state. The cache records a pending
        # fetch immediately; ordered A outputs guarantee its data is later ready.
        if a_accept:
            assert a is not None
            c = self.channel(a.word_addr)
            beat = self.beat(a.word_addr)
            logical_beat = a.word_addr >> 4
            reuse_slot = next((i for i, v in enumerate(self.cache) if v is not None and v[0] == logical_beat), None)
            reuse = reuse_slot is not None
            hit = not reuse and self.run_valid[c] and self.run_next[c] == beat and self.run_left[c] > 0
            slot = reuse_slot if reuse else self.cache_next
            drop = 0 if reuse or hit else self.run_left[c]
            self.a_order.append(_AOrder(a.id, c, reuse, slot, drop))
            new_cache_slot = None
            if not reuse:
                self.cache[slot] = (logical_beat, a.id)
                new_cache_slot = slot
                self.cache_next = (slot + 1) % self.APR
                if hit:
                    self.run_next[c] += 1
                    self.run_left[c] -= 1
                else:
                    self.aq[c].append((a, beat))
                    self.run_valid[c] = True
                    self.run_next[c] = beat + 1
                    self.run_left[c] = self.APF - 1
        else:
            new_cache_slot = None
        for c in run_invalid:
            self.run_valid[c] = False
        for i in cache_invalid:
            if i != new_cache_slot:
                self.cache[i] = None

        # Native read-command tags and per-channel response routing. Route
        # arrivals against the *pre-edge* tag FIFO: a command accepted on this
        # edge cannot return on the same edge. The ID is a check on the
        # independent memory model, not an observed RTL result.
        issued_writes: list[NativeCommand] = []
        old_issued = list(self.write_issued)
        for c, cmd in enumerate(shown):
            response_id = pins.response_ids[c]
            if response_id is not None:
                if not self.native_tags[c]:
                    raise ValueError(f"channel {c}: response without an earlier accepted read")
                expected = self.native_tags[c].popleft()
                if response_id != expected.id:
                    raise ValueError(f"channel {c}: response {response_id!r}, expected {expected.id!r}")
                if expected.source == "A":
                    self.a_fifo[c].append(response_id)
                else:
                    self.b_fifo[c].append(response_id)
            if cmd is not None:
                if cmd.write:
                    if cmd.accepted:
                        issued_writes.append(cmd)
                        self.write_issued[c] = (self.write_issued[c] + 1) & 0xFFFF
                        self.bq[c].popleft()
                        self.held_source[c] = None
                        self.command_done[c] = self.data_done[c] = False
                    else:
                        self.held_source[c] = cmd.source
                        self.command_done[c] |= bool(cmd.cvalid and pins.ready[c])
                        self.data_done[c] |= bool(cmd.wvalid and pins.write_ready[c])
                else:
                    if cmd.accepted:
                        self.native_tags[c].append(cmd)
                        self.held_source[c] = None
                        if cmd.source == "A":
                            if self.a_issue[c] == 0:
                                self.a_reserved[c] += self.APF
                            self.a_issue[c] += 1
                            if self.a_issue[c] == self.APF:
                                self.a_issue[c] = 0
                                self.aq[c].popleft()
                        else:
                            self.b_reserved[c] += 1
                            self.bq[c].popleft()
                    else:
                        self.held_source[c] = cmd.source

        if pins.write_done is not None:
            self.issued_all_reg = all((old_issued[c] + int(any(w.channel == c for w in issued_writes))) &
                                      0xFFFF == pins.write_done[c] for c in range(8))
        self.write_waiting += self.write_accepted_last - len(issued_writes)
        if self.write_waiting < 0:
            raise ValueError("B write accounting underflow")
        self.write_accepted_last = (int(bool(breg_word_mask & 0xFFFF)) +
                                    int(bool(breg_word_mask >> 16)))
        self.write_invalidate = [next((w.beat_addr for w in issued_writes if w.channel == c), None)
                                 for c in range(8)]

        if b_response is not None:
            pairs_to_pop = self.b_tags.popleft().pairs
            for pair in pairs_to_pop:
                for c in (2 * pair, 2 * pair + 1):
                    self.b_fifo[c].popleft()
                    self.b_reserved[c] -= 1
        self.a_valid_reg = None
        self.a_valid_reg_source_id = None
        if a_order_out is not None:
            self.a_order.popleft()
            self.a_valid_reg = a_order_out.id
            if a_order_out.reuse:
                self.a_valid_reg_source_id = self.cache_data_id[a_order_out.cache_slot]
                if self.a_valid_reg_source_id is None:
                    raise ValueError("A cache reuse before its original data arrived")
            else:
                c = a_order_out.channel
                self.a_valid_reg_source_id = self.a_fifo[c][a_order_out.drop]
                self.cache_data_id[a_order_out.cache_slot] = self.a_valid_reg_source_id
                for _ in range(a_order_out.drop + 1):
                    self.a_fifo[c].popleft()
                self.a_reserved[c] -= a_order_out.drop + 1
        if a_flush:
            self.run_valid = [False] * 8
            self.cache = [None] * self.APR

        result = StepResult(self.cycle, a_ready, b_ready, a_accept, b_accept,
                            tuple(cmd for cmd in shown if cmd is not None), a_response, b_response,
                            wr_idle, rd_idle, a_response_source, b_response_sources)
        self.cycle += 1
        return result
