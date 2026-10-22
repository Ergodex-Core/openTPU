"""Bounded command-driven cycle model of the D128/LANES8 DMA LD path.

``snapshot`` describes the pins *before* a core edge; ``step`` returns that
snapshot and advances one edge. ``response`` is an independently generated
128-byte, ordered B response at the DMA input, after the slice response
register. Accepted starts are supplied by the caller. No observed DMA request,
write, or completion is used to advance this model.

This covers the five initial LDs of the three-MM page-reuse fixture (at most
36 chunks). It excludes ST, WAITW, stream and TMEM contention. In this
initial-load phase DMA has highest TMEM write priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


CHUNK_WORDS = 32
SEGMENT_WORDS = 8
CHUNK_BYTES = 128
BUFFER_DEPTH = 128
MAX_CHUNKS = 36


@dataclass(frozen=True)
class LoadCommand:
    id: Hashable
    dram_byte: int
    tmem_word: int
    words: int

    @classmethod
    def from_instruction(cls, id: Hashable, instruction: object) -> "LoadCommand":
        # OP_LD is 0x10 in otpu_pkg and opentpu.isa. Do not accept an ST with
        # the same word fields as an LD.
        if getattr(instruction, "op", None) != 0x10:
            raise ValueError("only an LD instruction is supported")
        w = getattr(instruction, "w", None)
        if w is None or len(w) < 3:
            raise ValueError("LD requires its three static immediate fields")
        if getattr(instruction, "ra", 0) or getattr(instruction, "rb", 0):
            raise ValueError("register-relative LD address is unsupported")
        return cls(id, int(w[0]), int(w[1]), int(w[2]))


@dataclass(frozen=True)
class LoadRequest:
    id: Hashable
    chunk: int
    word_addr: int
    chunks: int = 1
    tag: bool = True


@dataclass(frozen=True)
class LoadSnapshot:
    cycle: int
    ready: bool
    busy: bool
    request: LoadRequest | None
    writes: tuple[tuple[int, int], ...]  # (TMEM word address, unsigned u32)
    done: bool
    ld_fin: bool
    requested: int
    received: int
    delivered_segments: int
    reserved_chunks: int
    received_chunks: int


class DmaLoadModel:
    def __init__(self) -> None:
        self.cycle = 0
        self.command: LoadCommand | None = None
        self.busy = False
        self.done = False
        self.ld_fin = False
        self.ld_last = False
        self.dv_valid = False
        self.dv_mask = (False,) * SEGMENT_WORDS
        self.dv_addr = 0
        self.dv_pos = 0
        self.read_q = bytes(CHUNK_BYTES)
        self.write_reg: tuple[tuple[int, int], ...] = ()
        self.slots = [bytes(CHUNK_BYTES) for _ in range(BUFFER_DEPTH)]
        self.wp = self.rp = self.occ = self.cnt = 0
        self.requested = self.received = self.delivered_segments = 0
        self.ic = self.cleft = self.sw = self.so = self.sleft = 0
        self.dw = self.de = 0
        self.total_chunks = 0
        self.starts = 0
        self.planned_chunks = self.planned_words = 0
        self.total_requests = self.total_responses = 0
        self.total_segments = self.total_writes = self.total_done = 0

    @staticmethod
    def _check(command: LoadCommand) -> tuple[int, int, int]:
        if not isinstance(command, LoadCommand):
            raise TypeError("start requires LoadCommand")
        a, t, n = command.dram_byte, command.tmem_word, command.words
        if not all(isinstance(x, int) and 0 <= x < (1 << 32) for x in (a, t, n)):
            raise ValueError("LD addresses and count must be unsigned 32-bit integers")
        if a % 4 or not n or (a // 4 + n) >= (1 << 32) or (t + n) >= (1 << 32):
            raise ValueError("LD requires a word-aligned, nonempty, nonwrapping range")
        word = a // 4
        chunks = (word % CHUNK_WORDS + n + CHUNK_WORDS - 1) // CHUNK_WORDS
        if chunks > MAX_CHUNKS:
            raise ValueError(f"LD exceeds {MAX_CHUNKS}-chunk bounded model")
        segments = (word % SEGMENT_WORDS + n + SEGMENT_WORDS - 1) // SEGMENT_WORDS
        return word, chunks, segments

    def snapshot(self) -> LoadSnapshot:
        req = None
        if self.busy and self.cleft and self.occ < BUFFER_DEPTH:
            assert self.command is not None
            req = LoadRequest(self.command.id, self.requested, self.ic)
        return LoadSnapshot(
            self.cycle, not self.busy, self.busy, req, self.write_reg,
            self.done, self.ld_fin, self.requested, self.received,
            self.delivered_segments, self.occ, self.cnt,
        )

    def finish(
        self, *, expected_loads: int | None = None,
        expected_chunks: int | None = None,
        expected_words: int | None = None,
    ) -> dict[str, int]:
        """Require complete LD closure and return cumulative phase counts.

        The caller supplies fixture-specific expectations; the model itself
        does not assume a five-instruction program. A visible ``done`` pulse
        is allowed, but its generating ``ld_fin`` and all write stages must
        already be empty.
        """
        if not self.starts:
            raise ValueError("no LD starts were modeled")
        if (self.busy or self.snapshot().request is not None or self.cleft or
                self.sleft or self.occ or self.cnt or self.dv_valid or
                self.write_reg or self.ld_last or self.ld_fin):
            raise ValueError("DMA LD still has an active request, buffer, or pipeline stage")
        if (self.requested != self.received or self.requested != self.total_chunks or
                self.total_requests != self.total_responses or
                self.total_requests != self.planned_chunks or
                self.total_writes != self.planned_words or
                self.total_done != self.starts):
            raise ValueError("DMA LD counts did not close")
        for label, actual, expected in (
            ("loads", self.starts, expected_loads),
            ("chunks", self.total_requests, expected_chunks),
            ("words", self.total_writes, expected_words),
        ):
            if expected is not None and actual != expected:
                raise ValueError(f"DMA LD {label}: expected {expected}, got {actual}")
        return {
            "loads": self.starts,
            "requested_chunks": self.total_requests,
            "received_chunks": self.total_responses,
            "delivered_segments": self.total_segments,
            "written_words": self.total_writes,
            "done_pulses": self.total_done,
        }

    def step(
        self, *, start: LoadCommand | object | None = None, b_ready: bool = False,
        response: bytes | None = None, reset: bool = False,
    ) -> LoadSnapshot:
        """Advance one edge using generated grant and ordered DMA-input bytes.

        A grant applies only to the visible ``snapshot().request``. A response
        must belong to a previously granted request, not a same-edge grant.
        """
        before = self.snapshot()
        if reset:
            self.__init__()
            self.cycle = before.cycle + 1
            return before
        if start is not None and not isinstance(start, LoadCommand):
            start = LoadCommand.from_instruction(self.starts, start)
        if start is not None:
            word, chunks, segments = self._check(start)
            if self.busy:
                raise ValueError("LD start while DMA busy")
            if b_ready or response is not None:
                raise ValueError("LD start cannot share an edge with B traffic")
        elif b_ready and before.request is None:
            raise ValueError("B grant without DMA LD request")
        if response is not None:
            if not isinstance(response, bytes) or len(response) != CHUNK_BYTES:
                raise ValueError("B response must be exactly 128 generated bytes")
            if not self.busy or self.received >= self.requested:
                raise ValueError("B response has no previously issued LD request")
            if self.cnt >= BUFFER_DEPTH:
                raise ValueError("LD response buffer overflow")

        # All right-hand sides below use the pre-edge registers. The block RAM
        # is synchronous/read-first: its read_q takes the old slot at rp, even
        # if this edge writes that slot with a new B response.
        next_read_q = self.slots[self.rp]
        next_writes = ()
        if self.dv_valid:
            next_writes = tuple(
                (self.dv_addr + lane,
                          int.from_bytes(self.read_q[4 * (self.dv_pos * SEGMENT_WORDS + lane):
                                                     4 * (self.dv_pos * SEGMENT_WORDS + lane + 1)], "little"))
                for lane in range(SEGMENT_WORDS) if self.dv_mask[lane]
            )
        next_done = self.ld_fin
        next_fin = False
        next_last = False
        next_dv = False

        if start is not None:
            ow = word % SEGMENT_WORDS
            self.command = start
            self.starts += 1
            self.planned_chunks += chunks
            self.planned_words += start.words
            self.busy = True
            self.ic = word & ~(CHUNK_WORDS - 1)
            self.cleft = chunks
            self.total_chunks = chunks
            self.dw, self.de = word, word + start.words
            self.sw = word & ~(SEGMENT_WORDS - 1)
            self.so = start.tmem_word - ow
            self.sleft = segments
            self.wp = self.rp = self.occ = self.cnt = 0
            self.requested = self.received = self.delivered_segments = 0
        elif self.busy:
            issue = bool(before.request is not None and b_ready)
            deliver = self.sleft > 0 and self.cnt > 0
            eat = deliver and ((self.sw // SEGMENT_WORDS) % (CHUNK_WORDS // SEGMENT_WORDS)
                               == CHUNK_WORDS // SEGMENT_WORDS - 1 or self.sleft == 1)
            if issue:
                self.ic += CHUNK_WORDS
                self.cleft -= 1
                self.requested += 1
                self.total_requests += 1
            if deliver:
                next_dv = True
                self.dv_addr = self.so
                self.dv_pos = (self.sw % CHUNK_WORDS) // SEGMENT_WORDS
                self.dv_mask = tuple(self.dw <= self.sw + lane < self.de
                                     for lane in range(SEGMENT_WORDS))
                if self.sleft == 1:
                    next_last = True
                self.sw += SEGMENT_WORDS
                self.so += SEGMENT_WORDS
                self.sleft -= 1
                self.delivered_segments += 1
                self.total_segments += 1
            if self.ld_last:
                self.busy = False
                next_fin = True
            self.occ += int(issue) - int(eat)
            self.cnt += int(response is not None) - int(eat)
            if eat:
                self.rp = (self.rp + 1) % BUFFER_DEPTH
            if response is not None:
                self.slots[self.wp] = response
                self.wp = (self.wp + 1) % BUFFER_DEPTH
                self.received += 1
                self.total_responses += 1
            if not (0 <= self.cnt <= self.occ <= BUFFER_DEPTH):
                raise AssertionError("LD reserved/received chunk invariant failed")

        self.read_q = next_read_q
        self.write_reg = next_writes
        self.total_writes += len(next_writes)
        self.done = next_done
        self.total_done += int(next_done)
        self.ld_fin = next_fin
        self.ld_last = next_last
        self.dv_valid = next_dv
        self.cycle += 1
        return before
