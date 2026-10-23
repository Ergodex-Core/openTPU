"""Offline ABI-client tests. The fake is not RTL or F2 performance evidence."""
import numpy as np
import pytest

from opentpu import c2_f2 as C, c2_rtl as R
from opentpu.host.c2_f2 import canonical_bdf


@pytest.mark.parametrize("rows,kb", [(1, 1), (3, 9), (4, 54), (5, 255), (31, 9), (512, 1)])
def test_pair_pc_layout_reconstructs_original_bytes_and_tails(rows, kb):
    f = C.make_fixture(rows, kb, base_beat=123)
    banks = f.banks()
    for r in range(rows):
        for k in range(kb):
            start = (r//4*kb+k)*64
            got = np.concatenate([banks[2*(r % 4)+half][start:start+64] for half in range(2)])
            assert np.array_equal(got, f.weights[r, k].view(np.uint8))
    assert sum(f.expected_pc_beats())*64 == rows*kb*128
    for r in range(rows, ((rows+3)//4)*4):
        start = (r//4)*kb*64
        assert not np.any(banks[2*(r % 4)][start:start+kb*64])
        assert not np.any(banks[2*(r % 4)+1][start:start+kb*64])


@pytest.mark.parametrize("kb", [1, 9, 54, 255])
def test_fixture_reference_matches_existing_canonical_group_reference(kb):
    f = C.make_fixture(4, kb)
    g = R.make_group(kb, 4)
    g.weights[:] = f.weights
    g.acts[0] = f.acts
    g.dw[:] = f.dw
    g.da[0] = f.da
    g.sw[:] = f.sw
    g.sa[0] = f.sa[0]
    want, ints = R.reference(g)
    got, got_ints = f.reference()
    assert np.array_equal(got, want)
    assert got_ints == ints


def test_fixture_roundtrip_and_address_overflow(tmp_path):
    f = C.make_fixture(7, 9)
    path = tmp_path/"f.npz"
    f.save(path)
    restored = C.Fixture.load(path)
    assert restored.digest() == f.digest()
    assert restored.reference()[1] == f.reference()[1]
    f.base_beat = C.PC_BYTES//64-1
    with pytest.raises(ValueError, match="allocation"):
        f.validate()


class Fake:
    """ABI behavioral fake, using canonical arithmetic, with no simulated timing claim."""
    evidence_kind = "unit-test"

    def __init__(self):
        self.regs = {C.R_ID: C.ID, C.R_ABI: C.ABI, C.R_BUILD: 1234, C.R_CORE_KHZ: 200000,
                     C.R_MAX_ROWS: 512, C.R_DEPTH: 256, C.R_CREDITS: 32, C.R_HBM_READY: 1}
        self.mem, self.pcis, self.writes = {}, {}, []
        self.corrupt, self.hang = None, False

    def reg_read(self, addr):
        if addr == C.R_MEM_RDATA:
            word = self.regs[C.R_MEM_ADDR]
            self.regs[C.R_MEM_ADDR] += 1
            return self.mem[word]
        return self.regs.get(addr, 0)

    def reg_write(self, addr, value):
        self.writes.append((addr, value))
        if addr == C.R_MEM_WDATA:
            word = self.regs[C.R_MEM_ADDR]
            self.mem[word] = value
            self.regs[C.R_MEM_ADDR] += 1
        elif addr == C.R_CTRL:
            if value == C.CLEAR:
                self.regs[C.R_STATUS] = self.regs[C.R_ERROR] = 0
            elif value == C.ABORT:
                self.regs[C.R_STATUS] = C.DONE | C.ABORTED
            elif value == C.START:
                if self.hang:
                    self.regs[C.R_STATUS] = C.BUSY
                else:
                    self.execute()
        else:
            self.regs[addr] = value

    def mem_write(self, pc, offset, data):
        self.pcis[pc, offset] = data.copy()

    def mem_read(self, pc, offset, n):
        return self.pcis[pc, offset][:n].copy()

    def execute(self):
        rows, kb, base = (self.regs[a] for a in (C.R_ROWS, C.R_KB, C.R_BASE))
        weights = np.empty((rows, kb, 128), np.int8)
        for r in range(rows):
            for k in range(kb):
                at = (r//4*kb+k)*64
                weights[r, k] = np.concatenate([self.pcis[2*(r % 4)+h, base*64][at:at+64]
                                                 for h in range(2)]).view(np.int8)
        acts = np.array([self.mem[C.ACT+i] for i in range(kb*32)], dtype="<u4").view(np.int8).reshape(kb, 128)
        da = np.array([self.mem[C.DA+i] for i in range(kb)], dtype="<u4").view(np.uint8).reshape(kb, 4)
        dw = np.array([[self.mem[C.DW+r*256+k] for k in range(kb)] for r in range(rows)], np.uint8)
        sw = np.array([self.mem[C.SW+r] for r in range(rows)], np.uint32)
        f = C.Fixture(weights, acts, dw, da, sw, np.array([self.mem[C.SA]], np.uint32), base)
        bits, integers = f.reference()
        for r, (value, integer) in enumerate(zip(bits, integers)):
            self.mem[C.RESULT+r] = int(value)
            self.mem[C.INTLO+r] = integer & 0xFFFFFFFF
            self.mem[C.INTHI+r] = (integer >> 32) & 0xFFFF
        # This sentinel counter is only a unit-test fixture, never exported as hardware.
        self.regs.update({C.R_STATUS: C.DONE, C.R_COUNT: rows, C.R_CYCLES_LO: 1000, C.R_CYCLES_HI: 0})
        groups = rows if self.regs[C.R_MODE] == 0 else (rows+3)//4
        for name, reg in C.COUNTERS.items():
            self.regs[reg] = (groups if name in ("accepted_groups", "committed_groups") else
                              rows*kb*2 if name in ("read_issued", "read_returned") else 0)
        for pc, beats in enumerate(f.expected_pc_beats()):
            for name, off in C.PC_COUNTERS.items():
                self.regs[0x100+pc*0x20+off] = beats if name in ("issued", "returned") else 0
        if self.corrupt == "bits":
            self.mem[C.RESULT] ^= 1
        elif self.corrupt == "integer":
            self.mem[C.INTLO] ^= 1
        elif self.corrupt == "count":
            self.regs[C.R_COUNT] -= 1
        elif self.corrupt == "pc":
            self.regs[0x104] += 1
        elif self.corrupt == "groups":
            self.regs[0x044] -= 1
        elif self.corrupt == "error":
            self.regs[C.R_ERROR] = 2
        elif self.corrupt == "wrap":
            self.regs[C.R_CYCLES_HI] = 1


def bench():
    tr = Fake()
    b = C.Benchmark(tr, expected_build_id=1234)
    b.preload(C.make_fixture(7, 9))
    return tr, b


def test_checked_abi_run_both_modes_repeats_and_export_rejects_fake():
    tr, b = bench()
    results = [b.run(mode) for mode in (0, 1, 1, 0)]
    assert len({tuple(r["bits"]) for r in results}) == 1
    assert all(r["cycles"] == 1000 for r in results)
    assert all(r["bit_exact"] and r["weight_bytes"] == 7*9*128 for r in results)
    assert results[0]["counters"]["accepted_groups"] == 7
    assert results[1]["counters"]["accepted_groups"] == 2
    with pytest.raises(ValueError, match="physical F2"):
        C.lens_data(results[0], "must not export this fake")


@pytest.mark.parametrize("corrupt", ["bits", "integer", "count", "pc", "groups", "error", "wrap"])
def test_fail_closed_on_result_or_counter_corruption(corrupt):
    tr, b = bench()
    tr.corrupt = corrupt
    with pytest.raises(RuntimeError):
        b.run(1)


def test_wrong_build_identity_writes_nothing():
    tr = Fake()
    with pytest.raises(RuntimeError, match="identity"):
        C.Benchmark(tr, expected_build_id=5678)
    assert tr.writes == []


def test_mutated_preload_is_rejected_before_start():
    tr, b = bench()
    b.fixture.weights[0, 0, 0] ^= 1
    tr.writes.clear()
    with pytest.raises(RuntimeError, match="changed after"):
        b.run(1)
    assert tr.writes == []


def test_completion_timeout_aborts_and_drains():
    tr, b = bench()
    tr.hang = True
    with pytest.raises(TimeoutError, match="completion"):
        b.run(1, timeout=0.001)
    assert (C.R_CTRL, C.ABORT) in tr.writes
    assert not tr.reg_read(C.R_STATUS) & (C.BUSY | C.DRAINING)


def test_busy_device_is_not_reconfigured():
    tr, b = bench()
    tr.regs[C.R_STATUS] = C.DRAINING
    tr.writes.clear()
    with pytest.raises(RuntimeError, match="owns native"):
        b.run(0)
    assert tr.writes == []


def test_canonical_bdf_matches_existing_f2_lock_key():
    assert canonical_bdf("34:00.0") == canonical_bdf("0000:34:00.0") == "0000:34:00.0"
    with pytest.raises(ValueError):
        canonical_bdf("f2:0")
