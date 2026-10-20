"""P2's lane memory through the host's path (docs/v4/p2-f2.md "Host"): the board model
(sim/verilator/tb_board.sv) of the f2-p2 target, its four chunk lanes on four image copies
(otpu_board MXU_C 4: rtl/mem/otpu_lane_mem.sv), driven by Board over SimTransport, which writes
every copy and reads copy 0 as the F2's host does (opentpu/host/f2.py). The bitstream reports
the copies and the lanes (CFG_DRAM / CFG_UARCH); programs give the ISA simulator's DRAM bit for
bit; after every run the copies hold the same image (SimTransport checks)."""
import pytest

from opentpu.host.board import Board, SimTransport
from opentpu.host.checks import masked_program, run_demo
from conftest import sim_board_config

CFG = sim_board_config()


@pytest.mark.parametrize("seed,swfill", [(3, 1), (4, 0)])
def test_lane_memory_through_the_host_on_the_board_model(have_verilator, seed, swfill):
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=seed, native=True,
                     target="f2-p2", params={"SWFILL": swfill})
    assert t.copies == 4 and len(t.ch) == 8
    b = Board(t)
    c = b.info()["cfg"]
    assert (c["copies"], c["MXU_C"], c["channels"]) == (4, 4, 2)
    ok, msg, st = run_demo(b, CFG)
    assert ok, msg
    assert st["b_reads"] > 0 and st["cycles"] > 0
    ok, msg, st = run_demo(b, CFG, masked_program())     # partial writes: QST bytes, B word masks
    assert ok, msg
    for k in range(1, 4):                                # (SimTransport checked after each run)
        assert all((t.ch[2 * k + j] == t.ch[j]).all() for j in range(2))
