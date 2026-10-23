"""Qualification backend for the actual program+WAITC DRAM loading layout.

This is a CPU ISA test backend, never a serving fallback. It reserves a real
nonoverlapping program slot, writes the serialized image there, and executes
instructions decoded from those bytes. The physical loader/PCIe remains gated.
"""
import numpy as np

from opentpu import isa as I, waits as W
from opentpu.isasim import Machine
from tools.phonon2_qualify_encoder import SimulatorDevice


class ProgramSlotDevice(SimulatorDevice):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.program_area_bytes = W.area(cfg.IMEM_WORDS, bool(cfg.SEQ_WC))
        self.program_at = cfg.DRAM_BYTES - self.program_area_bytes
        if self.program_at < 0 or self.program_at % cfg.D:
            raise ValueError("program slot does not fit/aligned in device DRAM")
        self.program_bytes_uploaded = 0

    def reset(self, footprint):
        if footprint > self.program_at:
            raise ValueError("graph footprint overlaps program+wait slot")
        super().reset(footprint)

    def phase(self, program, check=False):
        if getattr(program, "run_args", None):
            raise ValueError("static encoder loader does not accept unbound runtime arguments")
        words = np.asarray([op.encode() for op in program], dtype="<u4").reshape(-1)
        entries = np.full(len(program), W.DEFAULT, np.uint64)
        payload = W.loadable(words, entries, self.cfg.D) if self.cfg.SEQ_WC else words
        if len(words) > self.cfg.IMEM_WORDS or payload.nbytes > self.program_area_bytes:
            raise ValueError("program+wait image exceeds reserved slot")
        if self.machine is None:
            self.machine = Machine(self.cfg, [[I.halt()]], [None])
        a = self.program_at
        self.memory[a:a + payload.nbytes] = payload.view(np.uint8)
        loaded_words = self.memory[a:a + words.nbytes].view("<u4")
        decoded = [I.Instr.decode(loaded_words[i:i + 8]) for i in range(0, len(loaded_words), 8)]
        self.machine.load([decoded])
        if check:
            entry_at = a + (words.nbytes + self.cfg.D - 1) // self.cfg.D * self.cfg.D
            loaded_entries = self.memory[entry_at:entry_at + entries.nbytes].view("<u8").copy() if self.cfg.SEQ_WC else entries
            self.wait_checked += W.check_isasim(self.machine, [loaded_entries])
        else:
            self.machine.run(max_steps=50_000_000)
        self.dispatches += 1
        self.program_bytes_uploaded += payload.nbytes
