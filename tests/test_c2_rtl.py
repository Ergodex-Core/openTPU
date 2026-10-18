"""D=128 C=4 shared-bank RTL against canonical MM, not a timing proxy."""
import pytest

from opentpu import c2_rtl as C


@pytest.mark.parametrize("kb", [1, 9, 54, 255])
def test_exact_both_modes_tails_acc(have_verilator, kb):
    groups = [C.make_group(kb, count=n, decode=decode, acc=acc, tag=i,
                          seed=1800+i, corners=(n % 2 == 0))
              for i, (decode, acc, n) in enumerate(
                  (decode, acc, n) for decode in (False, True, False)
                  for acc in (False, True) for n in (1, 2, 3, 4))]
    r = C.run(groups, seed=kb, wstall=17, sstall=31, astall=23, ostall=43, rstall=37)
    counts = r["counts"]
    assert r["bit_exact"]
    assert counts["advance"] == kb * len(groups)
    assert counts["mode_wait"] > 0
    assert counts["weight_transactions"] == sum(g.kb * (g.count if g.decode else 1) for g in groups)
    assert counts["scale_transactions"] == counts["weight_transactions"]
    assert counts["act_transactions"] == sum(g.kb * (1 if g.decode else g.count) for g in groups)


@pytest.mark.parametrize("seed", [7, 19, 91])
def test_independent_delivery_and_full_credit_drain(have_verilator, seed):
    groups = [C.make_group(1, count=1+i%4, decode=True, acc=True, tag=i, seed=seed+i)
              for i in range(20)]
    groups.append(C.make_group(9, count=4, decode=False, acc=True, tag=20, seed=seed+20))
    r = C.run(groups, seed=seed, wstall=45, sstall=37, astall=33,
              ostall=61, rstall=51, hold=450, old_hold=300, delay_pc0=4)
    c = r["counts"]
    assert c["peak_live_groups"] == 8
    assert c["peak_result_groups"] == 8
    assert c["credit_wait"] > 0 and c["old_wait"] > 100 and c["output_wait"] > 0
    assert c["mode_wait"] > 0
    assert any(e["accept"] and e["commit"] for e in r["events"])
    # The RTL/harness asserts that stalled commits and ACC requests retain their
    # complete tags, mask, addresses, fp32 data and integer payload until accepted.


def test_slow_column_changes_real_rtl_cycles(have_verilator):
    groups = [C.make_group(1, count=4, tag=i, seed=500+i) for i in range(12)]
    normal = C.run(groups)
    slow = C.run(groups, delay_pc0=40)
    assert slow["counts"]["cycles"] > normal["counts"]["cycles"]
    assert slow["counts"]["weight_wait_column_cycles"][0] > slow["counts"]["weight_wait_column_cycles"][1]
    assert slow["outputs"][-1]["bits"] == normal["outputs"][-1]["bits"]
    assert slow["counts"]["advance"] == normal["counts"]["advance"] == 12


@pytest.mark.parametrize("where", ["operand", "acc", "output"])
def test_reset_discards_pending_work_and_changes_mode(have_verilator, where):
    prelude = [C.make_group(54 if where == "operand" else 1, tag=i,
                           acc=where == "acc", seed=600+i) for i in range(8)]
    after = [C.make_group(9, count=3, decode=False, acc=True, tag=8, seed=900),
             C.make_group(1, count=4, decode=True, tag=9, seed=901)]
    r = C.run(prelude+after, reset_at=12 if where == "operand" else 70,
              reset_skip=8, hold=200, old_hold=180, wstall=13)
    assert [o["tag"] for o in r["outputs"]] == [8,9]
    assert r["bit_exact"]


def test_reject_malformed_operand_packet():
    group=C.make_group()
    group.weights=group.weights[:3]
    with pytest.raises(ValueError,match="weights shape/type"):
        C.run([group])


@pytest.mark.parametrize("stalls", [False, True])
def test_pingpong_overlap_and_bank_recycling(have_verilator, stalls):
    groups=[C.make_group(9, count=1+i%4, tag=i, acc=bool(i%2), seed=1200+i)
            for i in range(16)]
    r=C.run(groups, seed=81, wstall=31 if stalls else 0,
            sstall=51 if stalls else 0, astall=23 if stalls else 0,
            ostall=47 if stalls else 0, rstall=53 if stalls else 0)
    assert r["counts"]["fill_compute_overlap_cycles"]>0
    accepts=[e for e in r["events"] if e["accept"]]
    assert [e["fill_bank"] for e in accepts]==[i%2 for i in range(16)]
    for e in r["events"]:
        if e["advance"] and (e["weight_accept_mask"] or e["scale_accept_mask"] or e["act_accept_mask"]):
            assert e["fill_bank"]!=e["compute_bank"]
            assert e["fill_tag"]!=e["array_tag"]


def test_recycled_banks_with_unequal_kb(have_verilator):
    groups=[C.make_group([1,255,9,54][i%4], count=1+i%4, tag=i,
                         acc=bool(i%2), seed=1400+i) for i in range(16)]
    r=C.run(groups, seed=802, wstall=31, sstall=39, astall=20, ostall=40, rstall=35)
    assert r["counts"]["fill_compute_overlap_cycles"]>0
    assert r["counts"]["operand_bank_wait"]>0
    assert [o["tag"] for o in r["outputs"]]==list(range(16))


def test_held_acc_response_through_blocked_epilogue(have_verilator):
    groups=[C.make_group(1, tag=i, acc=True, seed=1500+i) for i in range(20)]
    r=C.run(groups, seed=902, hold=300, rstall=75, ostall=30, hold_response=1)
    assert r["counts"]["old_response_backpressure"]>0
    assert r["counts"]["output_wait"]>0
    assert r["bit_exact"]
