"""Generated request schedule for the bounded, initial S1 boot loader.

This is the ``otpu_slice`` loader FSM, not an instruction executor or a
sequencer model. The one initial ``ld_start`` edge is supplied by the testbench;
``rd_idle_i``, B-register grant and returned B response IDs must come from the
independent slice/adapter/native-memory models. No observed loader request,
DRAM response time, or ``ld_busy`` value drives a transition.

Only D128 boot loading is supported. HALT CHAIN, reload, host writes to the
boot image, and loader payload decoding are deliberately outside this model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct


@dataclass(frozen=True)
class BootConfig:
    byte_addr: int
    instructions: int
    seq_wc: bool
    d: int = 128

    def __post_init__(self) -> None:
        if self.d != 128:
            raise ValueError("boot loader is scoped to D128")
        if self.instructions < 1:
            raise ValueError("boot loader needs a nonempty program")
        if self.byte_addr < 0 or self.byte_addr % self.d:
            raise ValueError("boot address must be nonnegative and chunk aligned")

    @property
    def rows(self) -> int:
        return (self.instructions + 3) // 4

    @property
    def total_reads(self) -> int:
        return self.rows + (self.rows if self.seq_wc else 0)

    def address_of(self, ordinal: int) -> int:
        """Expected loader B *word* address of a request ordinal."""
        if not 0 <= ordinal < self.total_reads:
            raise ValueError("loader request ordinal outside boot image")
        chunk = ordinal if ordinal < self.rows else self.rows + (ordinal - self.rows) // 4
        return self.byte_addr // 4 + 32 * chunk

    @classmethod
    def from_fixture(cls, directory: str | Path) -> "BootConfig":
        """Bind config to saved command and image, independently of RTL trace.

        ``rtlsim`` appends program words at ``+boot_addr`` and, when SEQ_WC is
        enabled, pads to a chunk before appending the 64-bit wait entries.
        Verify those source bytes and hashes so a trace from a different
        program cannot silently supply this loader's configuration.
        """
        root = Path(directory)
        cmd = json.loads((root / "sim-command.json").read_text())
        if not isinstance(cmd, list) or "+boot" not in cmd:
            raise ValueError("saved simulation command lacks +boot")

        def arg(name: str) -> int:
            matches = [part.split("=", 1)[1] for part in cmd
                       if isinstance(part, str) and part.startswith(name + "=")]
            if len(matches) != 1:
                raise ValueError(f"saved command needs exactly one {name}")
            return int(matches[0], 10)

        manifest = json.loads((root / "feedback-inputs.json").read_text())
        params = manifest["parameters"]
        config = cls(arg("+boot_addr"), arg("+boot_n"), bool(int(params.get("SEQ_WC", 0))),
                     int(params["D"]))
        words = [int(line, 16) for line in (root / "prog_0.hex").read_text().splitlines()]
        if not 0 < len(words) <= 8 * config.instructions or manifest["instruction_count"] != config.instructions:
            raise ValueError("boot count differs from the saved program")
        # rtlsim._write_hex omits trailing zero words, including HALT's seven
        # zero operand words. The declared instruction count fixes the padding.
        words += [0] * (8 * config.instructions - len(words))
        program = b"".join(struct.pack("<I", word) for word in words)
        if hashlib.sha256(program).hexdigest() != manifest["program_sha256"]:
            raise ValueError("boot program SHA mismatch")
        expected = program
        if config.seq_wc:
            waits = [int(line, 16) for line in (root / "waits.hex").read_text().splitlines()]
            if len(waits) != config.instructions:
                raise ValueError("saved wait sidecar count differs from boot program")
            sidecar = b"".join(struct.pack("<Q", value) for value in waits)
            if hashlib.sha256(sidecar).hexdigest() != manifest["waits_sha256"]:
                raise ValueError("boot wait sidecar SHA mismatch")
            expected += bytes(-len(expected) % config.d) + sidecar
        image = (root / "dram_0.bin").read_bytes()
        stop = config.byte_addr + len(expected)
        if stop > len(image):
            raise ValueError("boot image exceeds saved DRAM file")
        # The simulator writes logical little-endian words as big-endian file
        # words for SystemVerilog $fread; reverse each stored word back.
        raw = image[config.byte_addr:stop]
        logical = b"".join(raw[i:i + 4][::-1] for i in range(0, len(raw), 4))
        if logical != expected:
            raise ValueError("saved DRAM boot bytes differ from program/sidecar")
        return config


@dataclass(frozen=True)
class BootStep:
    cycle: int
    busy: bool
    go: bool
    request: bool
    word_addr: int | None
    request_id: tuple[str, int] | None
    instruction_write: bool
    instruction_row: int | None
    sidecar_write: bool
    sidecar_row: int | None
    sidecar_quarter: int | None
    issued: int
    completed: int


class BootLoader:
    """One pre-edge step of the initial boot loader at a time."""

    def __init__(self, config: BootConfig):
        self.config = config
        self.cycle = 0
        self.busy = False
        self.go = False
        self.started = False
        self.issued = 0
        self.completed = 0
        self.word_addr = config.byte_addr // 4
        self.pending: deque[tuple[str, int]] = deque()
        self.done_cycle: int | None = None

    def step(self, *, start: bool = False, rd_idle_i: bool = False, b_gnt: bool = False,
             b_response_id: tuple[str, int] | None = None, sys_rst: bool = False) -> BootStep:
        """Sample pre-edge outputs, then apply one edge.

        ``start`` is the testbench's initial accepted ld_start, never an
        observed loader request. A returned ID is ``('boot', request_ordinal)``
        after the slice's B response register; ``None`` means no response.
        """
        if start and (sys_rst or self.started):
            raise ValueError("boot start during reset or after an earlier boot")
        if sys_rst and (self.busy or self.pending):
            raise ValueError("system reset interrupted initial boot")
        req = self.busy and self.go and self.issued < self.config.total_reads
        request_id = ("boot", self.issued) if req else None
        got = b_response_id is not None
        if got and (not self.busy or not self.go or not self.pending or
                    b_response_id != self.pending[0]):
            raise ValueError("boot response is orphaned, stale, or out of order")
        wt = self.config.seq_wc and self.completed >= self.config.rows
        out = BootStep(
            cycle=self.cycle, busy=self.busy, go=self.go, request=req,
            word_addr=self.word_addr if req else None, request_id=request_id,
            instruction_write=got and not wt,
            instruction_row=self.completed if got and not wt else None,
            sidecar_write=got and wt,
            sidecar_row=self.completed - self.config.rows if got and wt else None,
            sidecar_quarter=(self.completed - self.config.rows) % 4 if got and wt else None,
            issued=self.issued, completed=self.completed)

        if sys_rst:
            self.busy = False
        elif start:
            self.started = True
            self.busy = True
            self.go = False
            self.issued = self.completed = 0
            self.word_addr = self.config.byte_addr // 4
            self.pending.clear()
        elif self.busy:
            if not self.go:
                self.go = bool(rd_idle_i)
            else:
                if req and b_gnt:
                    assert request_id is not None
                    self.pending.append(request_id)
                    before = self.issued
                    self.issued += 1
                    if before < self.config.rows or (before - self.config.rows) % 4 == 3:
                        self.word_addr += 32
                if got:
                    self.pending.popleft()
                    self.completed += 1
                    if self.completed == self.config.total_reads:
                        self.busy = False
                        self.done_cycle = self.cycle + 1
        elif got:
            raise ValueError("boot response after loader completion")
        self.cycle += 1
        return out

    def finish(self) -> dict:
        if (not self.started or self.busy or self.pending or
                self.issued != self.config.total_reads or
                self.completed != self.config.total_reads):
            raise ValueError("boot loader ended with missing or outstanding reads")
        return {
            "reads": self.config.total_reads, "instruction_rows": self.config.rows,
            "sidecar_rows": self.config.rows if self.config.seq_wc else 0,
            "done_cycle": self.done_cycle, "calibrated": False, "tok_s": None,
            "external_inputs": ["initial testbench ld_start", "predicted slice rd_idle_i",
                                "predicted slice B-register grant", "generated B response ID"],
            "excluded": ["HALT CHAIN and reload", "sequencer execution", "host boot-image writes"],
        }
