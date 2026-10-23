"""IsaTransport: the card's register and memory protocol over the bit-exact ISA simulator, so the
card's driver (BoardBackend: program loads with their wait entries, run arguments, the decode
loop's token reads, the KV store's staged copies, the compile worker processes) runs end to end
on a host with no card, with the ISA's numerics: `otpu-serve --backend board-isa`.

The DRAM is kept as each slice's logical bytes (the simulator's view); the channel accesses the
driver makes (Board.write / read: 64-byte beats alternating between the slice's two channels,
CHASH's map) are scattered into it and gathered from it. A LOAD decodes the program at PROG_ADDR
(its words; a WAITC program's entries after them are ignored), a RUN runs it to its HALT before
the register write returns (the simulator's one persistent machine: TMEM and ACT RAM kept between
runs, as on the card), so HALTED is up at the first poll. CYCLES reports `cycles` (default 0: the
simulator has no clock); `sim_s` is the wall time the simulator took (loads and runs), `bytes_read`
and `bytes_written` the memory the host moved (the PCIe traffic a card would carry): the host's
own time is a run's wall time less sim_s. The registers are FakeTransport's (the current
target's configuration registers, CAPS of a card that has every unit the programs use).
"""
from __future__ import annotations

import time

import numpy as np

from . import regs as R
from .board import BEAT
from .fake import FakeTransport


def _parity(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        m ^= m >> np.uint64(s)
    return (m & np.uint64(1)).astype(np.int64)


class IsaTransport(FakeTransport):
    threaded = False
    streams = False
    host_idle = 50e-6

    def __init__(self, core_khz: int = 200_000, cycles: int = 0, chash: bool = True, **kw):
        from ..isasim import board_config
        from ..target import current
        t = current()
        c = board_config()
        super().__init__(ch_bytes=0, regmap=5, devname=None, cycles=cycles, core_khz=core_khz,
                         args=True, stream=bool(c.STREAM), gen=True, chash=chash,
                         pair=bool(c.PAIR), dstep=bool(c.DSTEP), act_rows=c.act_rows,
                         seq_wc=bool(t.arch.get("SEQ_WC", False)), **kw)
        self.ecc = False
        cfg = R.cfg(self.cfg_words)
        self.ch_bytes = cfg["channel_bytes"]
        self.dram = [np.zeros(2 * self.ch_bytes, np.uint8) for _ in range(self.slices)]
        self.cfg = None                 # the engine's Config (set by the backend's maker)
        self.machine = None
        self.progs = None
        self.sim_s = 0.0
        self.sim_runs = 0
        self.bytes_read = self.bytes_written = 0
        self.reads_n = self.writes_n = 0

    # ---- memory: channel (ch, off) <-> slice ch // 2's logical bytes
    def _logical(self, c: int, q: np.ndarray) -> np.ndarray:
        """The logical beat index (2 m + b) of channel c's beats q (chunk m = q)."""
        b = c ^ _parity(q) if self.chash else np.full(len(q), c, np.int64)
        return 2 * q + b

    def _move(self, ch: int, off: int, data: np.ndarray | None, n: int) -> np.ndarray | None:
        s, c = divmod(ch, 2)
        mem = self.dram[s]
        out = None if data is not None else np.empty(n, np.uint8)
        pos = 0
        while pos < n:
            x = off + pos
            if x % BEAT or n - pos < BEAT:                  # a partial beat
                k = min(n - pos, BEAT - x % BEAT)
                q = x // BEAT
                la = int(self._logical(c, np.array([q]))[0]) * BEAT + x % BEAT
                if data is not None:
                    mem[la:la + k] = data[pos:pos + k]
                else:
                    out[pos:pos + k] = mem[la:la + k]
                pos += k
                continue
            nb = min((n - pos) // BEAT, 1 << 20)            # whole beats, 64 MiB at a time
            q = np.arange(x // BEAT, x // BEAT + nb, dtype=np.int64)
            beats = mem.reshape(-1, BEAT)
            if not self.chash:
                sl = slice(2 * q[0] + c, 2 * (q[-1] + 1) + c, 2)
                if data is not None:
                    beats[sl] = data[pos:pos + nb * BEAT].reshape(-1, BEAT)
                else:
                    out[pos:pos + nb * BEAT] = beats[sl].reshape(-1)
            else:
                li = self._logical(c, q)
                if data is not None:
                    beats[li] = data[pos:pos + nb * BEAT].reshape(-1, BEAT)
                else:
                    out[pos:pos + nb * BEAT] = beats[li].reshape(-1)
            pos += nb * BEAT
        return out

    def mem_write(self, ch, off, data):
        d = np.ascontiguousarray(data, np.uint8).reshape(-1)
        self.bytes_written += len(d)
        self.writes_n += 1
        self._move(ch, off, d, len(d))

    def mem_read(self, ch, off, n, out=None):
        self.bytes_read += n
        self.reads_n += 1
        v = self._move(ch, off, None, n)
        if out is None:
            return v
        out[:] = v
        return out

    def _put(self, addr: int, b: np.ndarray) -> None:
        self.dram[0][addr:addr + len(b)] = b

    # ---- the device
    def _load(self) -> None:
        from .. import isa as I
        addr, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
        progs = []
        for s in range(self.slices):
            w = self.dram[s][addr:addr + 32 * n].view("<u4")
            progs.append([I.Instr.decode(w[8 * k:8 * k + 8]) for k in range(n)])
        self.progs = progs

    def _run(self) -> None:
        from ..isasim import Machine
        if self.cfg is None:
            raise RuntimeError("IsaTransport: no Config (the backend maker sets .cfg)")
        args = [self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)]
        if self.machine is None:
            self.machine = Machine(self.cfg, self.progs, [None] * self.slices, args)
            for sl, m in zip(self.machine.slices, self.dram):
                sl.dram = m                     # (the transport's arrays: no copy)
        else:
            self.machine.load(self.progs, args)
        self.machine.run(max_steps=1 << 40)
        self.sim_runs += 1

    def reg_write(self, off, val):
        if off == R.R_CTRL:
            prev = self.regs[R.R_CTRL]
            t0 = time.perf_counter()
            if val & R.CTRL_LOAD and not prev & R.CTRL_LOAD:
                self._load()
            if val & R.CTRL_RUN and not prev & R.CTRL_RUN:
                self._run()
            self.sim_s += time.perf_counter() - t0
        super().reg_write(off, val)

    def counters(self) -> dict:
        return {"sim_s": self.sim_s, "sim_runs": self.sim_runs, "bytes_read": self.bytes_read,
                "bytes_written": self.bytes_written, "reads": self.reads_n,
                "writes": self.writes_n}
