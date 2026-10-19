"""C2 OCL/native adapter: real RTL with explicitly synthetic PC service."""
import numpy as np
import pytest
from opentpu.c2_f2 import make_fixture
from opentpu.c2_hbm_rtl import run, lens_data


@pytest.mark.parametrize('mode,rows,kb,latency,stall,delay', [
    (0, 1, 1, 1, 0, 0), (1, 1, 1, 30, 30, 0),
    (0, 3, 9, 30, 35, 50), (1, 3, 9, 30, 35, 50),
    (0, 4, 65, 90, 40, 200), (1, 4, 65, 90, 40, 200),
    (0, 9, 33, 15, 50, 100), (1, 9, 33, 15, 50, 100),
    (0, 31, 9, 30, 20, 0), (1, 31, 9, 30, 20, 0),
    (1, 257, 9, 30, 20, 150), (1, 512, 1, 30, 0, 0),
    (0, 1, 255, 60, 15, 100), (1, 4, 255, 60, 15, 100),
])
def test_native_roundtrip(mode, rows, kb, latency, stall, delay):
    result = run(make_fixture(rows, kb, seed=rows*kb+mode), mode,
                 latency=latency, stall=stall, delay_pc0=delay, seed=91+mode)
    # No output is synthesized by Python: run checks FPGA-target RTL fp32 and
    # integer diagnostics against the canonical ISA plus independent int64 sum.
    events = result['events']
    assert result['counters'][0x068] <= 256
    assert result['cycles'] > 0
    if rows == 3:
        profile = lens_data(make_fixture(rows, kb, seed=rows*kb+mode), mode, result, 'test')
        assert profile['kind'] == 'rtl'
        assert len(profile['instrs']) == (rows if mode == 0 else (rows+3)//4)
        assert profile['cycles'] == result['cycles']
    if kb > 32 and delay:
        assert result['counters'][0x114] > 0  # PC0 receive reservation saturation.


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('fault', [1, 2, 3])
def test_error_or_abort_drains_then_replays_without_reset(mode, fault):
    # PC7 unsolicited response while PC0 still owns delayed native transactions
    # catches raw-return-equality early flush/overshoot. Replay uses same SRAMs.
    result = run(make_fixture(1, 65, seed=712), mode, latency=100, stall=30,
                 delay_pc0=250, fault=fault, seed=19)
    assert len(result['faults']) == 1
    assert set(result['events'][:, 0]) == {0, 1}


@pytest.mark.parametrize('mode', [0, 1])
def test_read_pipeline_full_result_windows(mode):
    # The RTL harness poisons ARADDR after acceptance, holds a following AW/W
    # behind a stalled read, and resets each read phase before loading fixtures.
    # All 512 rows populate the final word of each fp32/int48 readback window;
    # the harness checks that word, its queued-write isolation, and repeated
    # invalid reads immediately after the exactly-once MEM_ADDR increment.
    run(make_fixture(512, 1, seed=194), mode, latency=30, stall=35, seed=29+mode)


@pytest.mark.parametrize('mode', [0, 1])
def test_scale_special_values_and_signed_integer_endpoints(mode):
    f = make_fixture(4, 65, seed=81)
    f.weights[0] = -128
    f.weights[1] = 127
    f.acts[:] = -128
    f.dw[:] = 0
    f.da[:] = 0
    f.sw[:] = np.uint32([0x3f800000, 0xbf800000, 0x7f800000, 0x7fc00123])
    run(f, mode, latency=45, stall=45, seed=8)
