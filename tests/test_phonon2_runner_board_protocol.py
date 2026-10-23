"""Native pinned Board methods against in-memory registers/DRAM, no card open.

Run with the verified df69 host source closure first in opentpu.__path__.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu import isa as I, waits as W
from opentpu.asr.phonon2_runner import BoardTransport, Cancelled, PROGRAM_AT
from opentpu.host import board as B, regs as R


class Registers:
    """Sparse mock memory channels; no PCIe, files, subprocess or FPGA handles."""
    batched = False
    threaded = False
    def __init__(self):
        self.status = R.ST_CALIB0 | R.ST_CALIB1 | R.ST_WR_IDLE
        self.log, self.memory, self.stall = [], {}, False

    def reg_read(self, off):
        return self.status if off == R.R_STATUS else 0

    def reg_read_many(self, offs):
        return [self.reg_read(off) for off in offs]

    def reg_write(self, off, value):
        self.log.append(('reg_write', off, value))
        if off == R.R_CTRL:
            if value == R.CTRL_RUN:
                self.status |= R.ST_RUN
                if not self.stall:self.status |= R.ST_HALTED
            elif value == 0:
                self.status &= ~R.ST_RUN
                self.status |= R.ST_WR_IDLE

    def poll(self, off, mask, value, timeout=600, expect=0):
        self.log.append(('poll',off,mask,value))
        result=self.reg_read(off)
        if result & mask != value:raise TimeoutError('mock register not ready')
        return result

    def mem_write(self, channel, off, data):
        self.log.append(('mem_write',channel,off,len(data)))
        for i,b in enumerate(data):self.memory[channel,off+i]=int(b)

    def mem_read(self, channel, off, size, out=None):
        raw=np.array([self.memory.get((channel,off+i),0) for i in range(size)],np.uint8)
        if out is not None:out[:]=raw;return out
        return raw


def board():
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/'docs/phonon2/runner_source_binding.json').read_text())
    if hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()!=binding['host_files']['opentpu/host/board.py']:
        pytest.skip('this test requires the pinned df69 native Board source closure')
    # Bypass the constructor deliberately: it opens/quiesces/calibrates devices.
    b=B.Board.__new__(B.Board)
    b.t=Registers();b.lock=SimpleNamespace(holder=None)
    b.in_run=False;b._trace=None;b._t_run=0;b._ded0=None;b._ded_said=False
    b._info=dict(D=128,MCOLS=8,LANES=8,act_rows=16,regmap=5,
                 cfg=dict(S=1,channels=8,channel_bytes=512<<20,platform=5,SEQ_WC=1),
                 caps=dict(chash=False),caps2=dict(gather=False))
    return b


def test_native_load_waitc_bytes_start_wait_write_idle_and_readback():
    b=board();transport=BoardTransport(b,lambda lease:lease=='test')
    words=np.asarray(I.assemble([I.halt()]),'<u4');entries=np.full(1,W.DEFAULT,'<u8')
    with transport.session(lambda:None,__import__('time').monotonic()+10):
        transport.load_program(PROGRAM_AT,words.tobytes(),entries.tobytes())
        transport.start();stats=transport.wait(1)
    expected=W.loadable(words,entries,128).tobytes()
    assert transport.read(PROGRAM_AT,len(expected))==expected
    assert stats['status'] & R.ST_HALTED and not b.in_run
    writes=[x for x in b.t.log if x[0]=='reg_write']
    assert ('reg_write',R.R_PROG_ADDR,PROGRAM_AT) in writes
    assert ('reg_write',R.R_PROG_N,1) in writes
    assert writes[-1]==('reg_write',R.R_CTRL,0)


def test_native_wait_cancel_stops_and_drains_then_restores_transport():
    b=board();b.t.stall=True;original=b.t;transport=BoardTransport(b,lambda _:True)
    calls=[0]
    def cancel():
        calls[0]+=1
        if calls[0]>=2:raise Cancelled('cancel native wait')
    with pytest.raises(Cancelled):
        with transport.session(cancel,__import__('time').monotonic()+10):
            transport.start();transport.wait(1)
    assert b.t is original and not b.in_run
    assert not b.t.status & R.ST_RUN
    assert b.t.log[-1]==('poll',R.R_STATUS,R.ST_WR_IDLE,R.ST_WR_IDLE)


def test_native_wait_hardware_error_does_not_return_valid_counters():
    b=board();transport=BoardTransport(b,lambda _:True)
    transport.start();b.t.status|=R.ST_AXI_ERR
    with pytest.raises(RuntimeError,match='memory path'):transport.wait(1)
    assert not b.in_run
