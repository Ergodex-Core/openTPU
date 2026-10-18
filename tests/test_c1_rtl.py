"""Canonical ISA versus the complete standalone RTL integer/fp32 path.

The C++ harness also asserts ready/valid output stability, mapping, ordered unique
retirement, generation tags, bounded occupancy and absence of duplicate live requests.
"""
import shutil
import subprocess

import pytest

from opentpu import c1_rtl as R

pytestmark = pytest.mark.skipif(not shutil.which("verilator"), reason="requires Verilator")


@pytest.mark.parametrize("slots", [1,2,4,8])
def test_slots_interleaved_delayed_pc_and_backpressure(slots, tmp_path):
    case=R.workload(n=67,n0=3,kb=9,k0=2,ab=4,mode="random",seed=slots)
    _,summary=R.run(case,tmp_path,slots=slots,latency=41,pc0_delay=93,
                    request_percent=61,output_percent=43,interleave=True,seed=slots*11)
    assert summary["outputs_bit_exact"]==67
    assert summary["weights_bytes_served"]==67*9*128
    assert 1<=summary["max_occupancy"]<=slots


@pytest.mark.parametrize("slots", [1,2,4,8])
def test_reset_aborts_live_rows_and_drops_old_generation(slots,tmp_path):
    case=R.workload(n=41,n0=2,kb=6,k0=3,ab=7,mode="random")
    events,summary=R.run(case,tmp_path,slots=slots,latency=80,pc0_delay=39,
                        request_percent=74,output_percent=51,interleave=True,reset_after=25)
    assert summary["generation"]==2 and summary["dropped_beats"]>0
    assert any(e["event"]=="reset" for e in events)


@pytest.mark.parametrize("mode", ["positive","negative","dropped"])
def test_wide_signed_accumulator_and_negative_shift(mode,tmp_path):
    case=R.workload(n=17,n0=5,kb=126,k0=1,ab=2,mode=mode)
    _,summary=R.run(case,tmp_path,slots=4,latency=3,interleave=True,output_percent=67)
    if mode=="positive":
        assert min(case.accumulators)>2**32
    elif mode=="negative":
        assert all(x&(1<<47) for x in case.accumulators)
    else:
        assert case.accumulators==[0]*17
    assert summary["outputs_bit_exact"]==17


@pytest.mark.parametrize("sa", [0,0x80000000,1,0x00800000,0x3f800000,0x7f800000,0x7fc12345])
def test_exact_fp32_special_scales_under_output_stalls(sa,tmp_path):
    case=R.workload(n=19,n0=3,kb=1,ab=0,mode="special",sa_bits=sa)
    _,summary=R.run(case,tmp_path,slots=4,latency=2,output_percent=24,request_percent=71)
    assert summary["outputs_bit_exact"]==19


@pytest.mark.parametrize("n,n0,kb", [(1,0,1),(9,13,2),(257,0,9)])
def test_tail_and_quantized_activation_snapshot(n,n0,kb,tmp_path):
    case=R.workload(n=n,n0=n0,kb=kb,k0=2,ab=3)
    events,summary=R.run(case,tmp_path,slots=4,latency=20)
    profile=R.profile(case,events,summary)
    assert profile["kind"]=="rtl-c1-kernel"
    assert profile["bytes"]==n*kb*128
    assert profile["cycles"]==summary["end_cycle"]-summary["command_cycle"]
    assert len(profile["instrs"])==n
    assert not profile["slices"][0]["ports"]


def test_slot_depth_hides_latency_but_does_not_multiply_bandwidth(tmp_path):
    case=R.workload(n=257,kb=9)
    measured={}
    for slots in (1,2,4,8):
        for latency,interval in ((80,1),(160,1),(80,4)):
            _,s=R.run(case,tmp_path/f"s{slots}-l{latency}-b{interval}",slots=slots,
                      latency=latency,beat_interval=interval)
            measured[slots,latency,interval]=s["cycles"]
            # Every PC must physically serve 17 rows * 36 beats for N257/E16.
            assert s["cycles"]>=(17*36-1)*interval
    assert measured[4,80,1]<measured[1,80,1]
    assert measured[4,160,1]>=measured[4,80,1]
    assert measured[8,80,4]>measured[8,80,1]


@pytest.mark.parametrize("reset_after", [17,65])
def test_reset_flushes_partial_sums_and_live_fp32_stages(reset_after,tmp_path):
    case=R.workload(n=67,kb=9,n0=3,k0=1,ab=4,mode="random")
    events,summary=R.run(case,tmp_path,slots=4,latency=3,output_percent=19,
                        interleave=False,reset_after=reset_after,seed=17)
    assert summary["generation"]==2 and summary["dropped_beats"]>0
    assert any(e["event"]=="beat" and e["generation"]==1 and not e["drop"] for e in events)
    if reset_after==65:
        assert any(e["event"]=="integer" and e["generation"]==1 for e in events)


@pytest.mark.parametrize("slots", [1,4])
def test_single_engine_packed_port_boundary(slots,tmp_path):
    case=R.workload(n=9,n0=3,kb=3,k0=2,ab=4,mode="random")
    _,summary=R.run(case,tmp_path,engines=1,slots=slots,latency=17,
                    request_percent=58,output_percent=43,interleave=True)
    assert summary["outputs_bit_exact"]==9


def test_next_command_waits_for_stalled_final_fp32_retirement():
    binary=R.build(engines=1,slots=1,harness="replay")
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "five exact outputs" in result.stdout
