"""Opt-in deterministic timing model of ``otpu_native_mem.sv``'s native ports.

This covers the direct NAXI=0, ``axi_dram=0`` path, not the native AXI adapter,
DMA, ``otpu_mem_ch``, memory contents, or the legacy shared ``$urandom`` mode.
Inputs are pre-edge native-port requests. Responses are generated from them;
no observed ready, read-valid, or write-done signal enters the model.

The matching opt-in RTL schedule uses the same stateless 32-bit hash. Draw kinds
are 0 command ready, 1 write ready, 2 read jitter, 3 write jitter, 4 read
bubble, and 5 write-done bubble. A draw is keyed by seed, free-running native
core cycle, channel, and kind. All arithmetic wraps at 32 bits.
"""

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass


MASK32 = (1 << 32) - 1
CMD_READY, WRITE_READY, READ_JITTER, WRITE_JITTER, READ_BUBBLE, WRITE_BUBBLE = range(6)


def deterministic_draw(seed: int, cycle: int, channel: int, kind: int) -> int:
    """Portable hash for the opt-in native-memory RTL schedule."""
    x = ((seed & MASK32) ^ ((cycle * 0x9E3779B9) & MASK32)
         ^ ((channel * 0x85EBCA6B) & MASK32) ^ ((kind * 0xC2B2AE35) & MASK32))
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    return (x ^ (x >> 16)) & MASK32


def logical_beat(channel_beat: int, channel: int, channels: int = 8,
                 *, chash: bool = True) -> int:
    """The native adapter's pair/half interleave, as RTL ``lbeat`` implements.

    This is a mapping helper; it does not choose the channel for an upstream
    request. The native model receives that already-selected channel.
    """
    if not 2 <= channels <= 32 or channels & (channels - 1):
        raise ValueError("channels must be a power of two in 2..32")
    if channel_beat < 0 or not 0 <= channel < channels:
        raise ValueError("invalid native channel or beat")
    pib = channels.bit_length() - 2
    fold = 0
    if chash and pib:
        for k in range(0, 27, pib):
            fold ^= (channel_beat >> k) & ((1 << pib) - 1)
    pair = channel_beat * (channels // 2) + ((channel // 2 ^ fold) & (channels // 2 - 1))
    return pair * 2 + (channel % 2 ^ (int(chash and pair.bit_count() & 1)))


@dataclass(frozen=True)
class NativeMemoryConfig:
    channels: int = 8
    seed: int = 7
    stall: int = 0
    lat: int = 120
    bandwidth: int = 100
    command_depth: int = 16
    write_depth: int = 16
    write_done_lag: int = 4
    dram: int = 0
    deterministic: bool = True
    words: int | None = None
    chash: bool = True

    def __post_init__(self):
        if not self.deterministic or self.dram != 0:
            raise ValueError("requires opt-in deterministic axi_dram=0 native memory")
        if not 2 <= self.channels <= 32 or self.channels & (self.channels - 1):
            raise ValueError("channels must be a power of two in 2..32")
        if not 0 <= self.stall <= 100 or not 0 < self.bandwidth <= 100:
            raise ValueError("stall/bandwidth outside supported percentage range")
        if min(self.lat, self.write_done_lag) < 0 or min(self.command_depth, self.write_depth) < 1:
            raise ValueError("negative latency or empty native queue")
        if self.words is not None and (self.words <= 0 or self.words % (16 * self.channels)):
            raise ValueError("logical memory size must be a positive multiple of 16*NCH words")


@dataclass(frozen=True)
class NativeCommand:
    write: bool
    beat: int                 # n_caddr: channel-local 64-byte beat address
    tag: Hashable | None = None  # checker-only identity; native RTL has no ID


@dataclass(frozen=True)
class NativeInput:
    command: NativeCommand | None = None  # n_cvalid plus n_cwe/n_caddr
    write_data: bool = False              # n_wvalid; payload irrelevant to timing


@dataclass(frozen=True)
class NativeSnapshot:
    cycle: int
    command_ready: tuple[bool, ...]
    write_ready: tuple[bool, ...]
    read_valid: tuple[bool, ...]
    read_tag: tuple[Hashable | None, ...]
    write_done: tuple[int, ...]


@dataclass(frozen=True)
class NativeEdge:
    before: NativeSnapshot
    accepted_commands: tuple[NativeCommand | None, ...]
    accepted_write_data: tuple[bool, ...]
    performed: tuple[NativeCommand | None, ...]


class NativeMemory:
    """Pre-edge snapshot then one synchronous core-edge transition per ``step``.

    Verilator's two-state initial ready registers are zero. A native reset
    clears queues/responses but not the separate ready registers or the
    free-running cycle counter. A core ``run_rst`` is not this reset.
    """

    def __init__(self, config: NativeMemoryConfig):
        self.config = config
        n = config.channels
        self.cycle = 0
        self.crr = [False] * n
        self.wrr = [False] * n
        self.commands = [deque() for _ in range(n)]
        self.write_data = [deque() for _ in range(n)]
        self.read_due = [deque() for _ in range(n)]
        self.write_due = [deque() for _ in range(n)]
        self.read_valid = [False] * n
        self.read_tag = [None] * n
        self.write_done = [0] * n
        self.credit = [0] * n
        self.last_read = [0] * n
        self.last_write = [0] * n

    def draw(self, channel: int, kind: int) -> int:
        return deterministic_draw(self.config.seed, self.cycle, channel, kind)

    def snapshot(self) -> NativeSnapshot:
        cfg = self.config
        return NativeSnapshot(
            self.cycle,
            tuple(self.crr[c] and len(self.commands[c]) < cfg.command_depth
                  for c in range(cfg.channels)),
            tuple(self.wrr[c] and len(self.write_data[c]) < cfg.write_depth
                  for c in range(cfg.channels)),
            tuple(self.read_valid), tuple(self.read_tag), tuple(self.write_done))

    def step(self, inputs: list[NativeInput] | tuple[NativeInput, ...], *, reset: bool = False) -> NativeEdge:
        if len(inputs) != self.config.channels:
            raise ValueError("one native input per channel is required")
        before = self.snapshot()
        taken_commands, taken_data, performed = [], [], []
        for c, request in enumerate(inputs):
            if not isinstance(request, NativeInput):
                raise ValueError("native inputs must be NativeInput values")
            take_c = request.command is not None and before.command_ready[c] and not reset
            take_w = request.write_data and before.write_ready[c] and not reset
            taken_commands.append(request.command if take_c else None)
            taken_data.append(bool(take_w))

            if reset:
                self.commands[c].clear()
                self.write_data[c].clear()
                self.read_due[c].clear()
                self.write_due[c].clear()
                self.read_valid[c] = False
                self.read_tag[c] = None
                self.write_done[c] = 0
                self.credit[c] = self.last_read[c] = self.last_write[c] = 0
                performed.append(None)
            else:
                if take_c:
                    command = request.command
                    if not 0 <= command.beat < 1 << 25:
                        raise ValueError("native beat address outside 25 bits")
                    if (self.config.words is not None and
                            (logical_beat(command.beat, c, self.config.channels,
                                          chash=self.config.chash) + 1) * 16 > self.config.words):
                        raise ValueError("native command beyond logical memory")
                    self.commands[c].append((self.cycle, command))
                if take_w:
                    self.write_data[c].append(self.cycle)
                self.credit[c] = min(200, self.credit[c] + self.config.bandwidth)
                command = None
                if (self.commands[c] and self.commands[c][0][0] < self.cycle
                        and self.credit[c] >= 100
                        and (not self.commands[c][0][1].write or self.write_data[c])):
                    _, command = self.commands[c].popleft()
                    self.credit[c] -= 100
                    if command.write:
                        self.write_data[c].popleft()
                        due = self.cycle + self.config.write_done_lag + self.draw(c, WRITE_JITTER) % 8
                        due = max(due, self.last_write[c])
                        self.last_write[c] = due
                        self.write_due[c].append(due)
                    else:
                        due = self.cycle + self.config.lat + self.draw(c, READ_JITTER) % 8
                        due = max(due, self.last_read[c])
                        self.last_read[c] = due
                        self.read_due[c].append((due, command.tag))
                performed.append(command)
                if (self.read_due[c] and self.read_due[c][0][0] <= self.cycle
                        and self.draw(c, READ_BUBBLE) % 100 >= self.config.stall):
                    _, self.read_tag[c] = self.read_due[c].popleft()
                    self.read_valid[c] = True
                else:
                    self.read_tag[c] = None
                    self.read_valid[c] = False
                if self.draw(c, WRITE_BUBBLE) % 100 >= self.config.stall:
                    while self.write_due[c] and self.write_due[c][0] <= self.cycle:
                        self.write_due[c].popleft()
                        self.write_done[c] = (self.write_done[c] + 1) & 0xffff

            # This is a distinct always_ff in RTL, active even on reset.
            self.crr[c] = self.draw(c, CMD_READY) % 100 >= self.config.stall
            self.wrr[c] = self.draw(c, WRITE_READY) % 100 >= self.config.stall

        self.cycle += 1
        return NativeEdge(before, tuple(taken_commands), tuple(taken_data), tuple(performed))
