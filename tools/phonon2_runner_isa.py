"""Offline protocol/serialization qualification. NEVER a serving backend.

Consumes an actual prepared artifact through Runner, with a CPU ISA transport.
No Board import/open; synthetic identity/permit are explicitly software-only.
"""
from contextlib import contextmanager
import numpy as np
from opentpu import isa as I
from opentpu.isasim import Machine


class ISATransport:
    backend = "isa"

    def __init__(self, cfg, manifest):
        self.cfg, self.manifest = cfg, manifest
        self.machine = Machine(cfg, [[I.halt()]], [None])
        self.pending = None

    def lease_valid(self, lease_id):
        return lease_id == "offline-isa-no-physical-lease"

    def verify_host_sources(self, binding):
        # Artifact preparation already verified common/compute source hashes.
        # Board-only dependencies are not imported or executed in this backend.
        if binding != self.manifest["binding"]:
            raise ValueError("binding mismatch")

    def identity(self):
        return dict(D=128, MCOLS=8, LANES=8, act_rows=16, core_khz=150000,
                    calibrated=True, running=False, status=8,
                    cfg=dict(S=1, TMEM_WORDS=65536, IMEM_WORDS=32768,
                             ACT_BLOCKS=128, DRAM_BYTES=4 << 30, SEQ_WC=True,
                             hash=int("dd6b92a6",16)),
                    caps={k:True for k in ("mx","pair","gen","dstep","stream","t2")},
                    caps2=dict(portb=4,rowlanes=True,aline=64),
                    identity_source="SYNTHETIC ISA CONFIGURATION, NOT DEVICE REGISTERS")

    @contextmanager
    def session(self, check, deadline):
        yield

    def write(self, at, data):
        self.machine.slices[0].dram[at:at+len(data)] = np.frombuffer(data,np.uint8)

    def read(self, at, size):
        return self.machine.slices[0].dram[at:at+size].tobytes()

    def load_program(self, at, code, waits):
        from opentpu import waits as W
        words, entries = np.frombuffer(code,"<u4"), np.frombuffer(waits,"<u8")
        # Exercise the exact serialized DRAM layout, not the original Instr list.
        payload = W.loadable(words,entries,self.cfg.D)
        self.write(at,payload.tobytes())
        actual = np.frombuffer(self.read(at,len(code)),"<u4")
        self.pending = [I.Instr.decode(actual[i:i+8]) for i in range(0,len(actual),8)]

    def start(self):
        self.machine.load([self.pending]);self.pending=None

    def wait(self, timeout):
        self.machine.run(max_steps=50_000_000)
        return dict(status=1,ecc_ded=False)

    def stop_and_drain(self, lease):
        return self.lease_valid(lease)
