"""C3 RTL, checked against executed canonical MM and independently ordered ARGMAX."""
import itertools

import numpy as np
import pytest

from opentpu import fp32 as F, lens, quant as Q, rtlsim
from opentpu.llm import vocab_engine as C, vocab_rtl as R
from test_vocab_engine import _case, _isa, _isa_argmax


def _check(sidecar, activation, w, s, x, **options):
    expected = _isa(sidecar.layout, w, s, x, raw=activation)
    want = _isa_argmax(expected)
    sim = R.run(sidecar, activation, **options)
    for got, winner in zip(sim.logits_bits, sim.winners):
        np.testing.assert_array_equal(got, expected)
        assert winner == want
    for epoch in sim.completed_epochs:
        events = [e for e in sim.events if e["epoch"] == epoch]
        assert sum(e["event"] == "W" for e in events) == sidecar.layout.vocab*sidecar.layout.kb
        assert sum(e["event"] == "D" for e in events) == sidecar.layout.vocab*sidecar.layout.kb
        assert sum(e["event"] == "E" for e in events) == sidecar.layout.vocab
    return sim


@pytest.mark.parametrize("engines", [1, 4])
@pytest.mark.parametrize("hidden", [128, 1152, 3584])
def test_streamed_logits_match_executed_isa_under_random_stalls(have_verilator, engines, hidden):
    layout, w, s, sidecar, x = _case(vocab=37, hidden=hidden, seed=81+hidden)
    sim = _check(sidecar, C.Activation.quantize(x), w, s, x, engines=engines,
                 seed=37+hidden, stalls=35, output_stalls=45, latency=17, result_hold=11, repeats=2)
    assert len(sim.completed_epochs) == 2
    assert R.lens_profile(sim)["c3"]["result_consumer_hold_cycles"] == 11


@pytest.mark.parametrize("vocab", [1, 2, 3, 5])
def test_empty_shards_and_tiny_vocabulary_complete_once(have_verilator, vocab):
    _, w, s, sidecar, x = _case(vocab=vocab, hidden=128, seed=23)
    sim = _check(sidecar, C.Activation.quantize(x), w, s, x, stalls=20, result_hold=19)
    completed = [e["engine"] for e in sim.events if e["event"] == "S"]
    assert sorted(completed) == [e for e in range(4) if len(sidecar.layout.row_ids(e))]


@pytest.mark.parametrize("reset_after", [3, 12, 30])
def test_reset_flushes_partial_rows_pipeline_and_blocked_outputs(have_verilator, reset_after):
    _, w, s, sidecar, x = _case(vocab=65, hidden=128, seed=19)
    sim = _check(sidecar, C.Activation.quantize(x), w, s, x,
                 output_hold=100, output_stalls=30, reset_after=reset_after,
                 repeats=2, result_hold=13)
    aborted = {e["epoch"] for e in sim.events if e["event"] == "X"}
    assert len(aborted) == 2
    assert not aborted.intersection(sim.completed_epochs)
    if reset_after == 30:
        assert any(e["event"] == "O" and e["epoch"] in aborted for e in sim.events)
        assert any(e["event"] == "B" and e["epoch"] in aborted for e in sim.events)


def test_shards_finish_in_different_orders_with_cross_engine_ties(have_verilator):
    layout = C.Layout(vocab=68, hidden=128)
    w = np.zeros((layout.vocab, layout.hidden), np.uint8)
    sw = np.ones(layout.vocab, np.float32)
    sw[0] = -1  # -0 loses to +0; lowest remaining global ID must win.
    s = Q.pack_scales("int8", sw, dw=np.zeros((layout.vocab, 1), np.uint8))
    sidecar = C.Sidecar.build(layout, w, s, embedding_weights=w, embedding_scales=s)
    x = np.ones(layout.hidden, np.float32)
    orders = []
    for slow in range(4):
        sim = _check(sidecar, C.Activation.quantize(x), w, s, x, slow_engine=slow, result_hold=7)
        order = tuple(e["engine"] for e in sim.events if e["event"] == "S")
        assert order[-1] == slow
        orders.append(order)
        assert sim.winners[0] == C.Winner(0, 1)
    assert len(set(orders)) == 4


@pytest.mark.parametrize("sa", [0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC00000])
def test_rtl_epilogue_exceptions_match_isa_in_complete_engine(have_verilator, sa):
    layout, w, s, _, x = _case(vocab=8, hidden=128, seed=5)
    s[:, :4] = np.array([0, 0x80000000, 1, 0x00800000, 0x3F800000,
                         0x7F800000, 0xFF800000, 0x7FC00000], np.uint32).view(np.uint8).reshape(8, 4)
    sidecar = C.Sidecar.build(layout, w, s, embedding_weights=w, embedding_scales=s)
    act = C.Activation(bytes([1])*128, sa, bytes(4))
    _check(sidecar, act, w, s, x, seed=101, stalls=25, output_stalls=60)


def test_full_48_bit_epilogue_rounding_and_exceptional_fp32(have_verilator, tmp_path):
    rng = np.random.default_rng(91)
    integers = [0, 1, -1, (1 << 47)-1, -(1 << 47)]
    for p in range(24, 47):
        half = 1 << (p-24)
        for delta in [-1, 0, 1]:
            for v in [(1 << p)+half+delta, (1 << p)+(1 << (p-23))+half+delta]:
                integers += [v, -v]
    integers += list(map(int, rng.integers(-(1 << 47), 1 << 47, 400)))
    cases = [(v, 0x3F800000, 0x47800000) for v in integers]  # Sa=2^16: expose i2f48 alone
    corners = [0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x3F800000,
               0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00000, 0xFFC00000]
    cases += [(v, sw, sa) for v in [0, 1, -1, (1 << 47)-1] for sw in corners for sa in corners]
    cases += [(int(v), int(sw), int(sa)) for v, sw, sa in zip(
        rng.integers(-(1 << 47), 1 << 47, 400), rng.integers(0, 1 << 32, 400), rng.integers(0, 1 << 32, 400))]
    lines = [str(len(cases))]
    for acc, sw, sa in cases:
        value = F.mul(F.mul(F.i2f(acc), F.from_bits(sw)), F.mul(F.from_bits(sa), np.float32(2**-16)))
        lines.append(f"{acc & ((1 << 48)-1):012x} {sw:08x} {sa:08x} {int(F.bits(value)):08x}")
    vectors = tmp_path / "epilogue.txt"
    vectors.write_text("\n".join(lines)+"\n")
    exe = R.build_aux("tb_c3_epilogue", ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv",
                                           "rtl/c3/otpu_c3_epilogue.sv", "sim/verilator/tb_c3_epilogue.sv"])
    result = rtlsim.run_sim([exe, f"+input={vectors}"], timeout=60)
    assert result.returncode == 0 and "C3_EPILOGUE_PASS" in result.stdout, result.stdout+result.stderr


def test_reducer_all_completion_orders_ftz_nan_payloads_and_ties(have_verilator, tmp_path):
    rng = np.random.default_rng(40)
    patterns = [[0x3F800000]*4, [0x80000000, 0x80000001, 1, 0],
                [0x7F800001, 0x7FC00000, 0xFFC00000, 0x7F800000],
                [0xFF800000]*4, [0x7FC00000]*4, [0x80000000]*4]
    patterns += [list(map(int, rng.integers(0, 1 << 32, 4))) for _ in range(30)]
    cases = []
    for bits in patterns:
        # All orderings and simultaneous completions, deliberately nonmonotonic IDs.
        ids = [262143, 0, 65536, 131072]
        for order in itertools.permutations(range(4)):
            delays = [0]*4
            for i, e in enumerate(order):
                delays[e] = 3*i
            cases.append((15, bits, ids, delays))
        cases.append((15, bits, ids, [0]*4))
    for mask in range(1, 16):
        cases.append((mask, [0, 1, 0x80000001, 0x80000000], [4, 3, 2, 1], [0, 6, 3, 9]))
    lines = [str(len(cases))]
    for mask, bits, ids, delays in cases:
        winner = None
        for e in range(4):
            if mask & (1 << e):
                winner = C.better(winner, C.Winner(bits[e], ids[e]))
        lines.append(f"{mask:x} {winner.bits:08x} {winner.token_id:08x}")
        lines += [f"{b:08x} {i:08x} {d}" for b, i, d in zip(bits, ids, delays)]
    vectors = tmp_path / "reduce.txt"
    vectors.write_text("\n".join(lines)+"\n")
    exe = R.build_aux("tb_c3_reduce", ["rtl/vpu/otpu_fp.sv", "rtl/c3/otpu_c3_reduce.sv",
                                     "sim/verilator/tb_c3_reduce.sv"])
    result = rtlsim.run_sim([exe, f"+input={vectors}"], timeout=60)
    assert result.returncode == 0 and "C3_REDUCE_PASS" in result.stdout, result.stdout+result.stderr


def test_lens_records_actual_service_compute_epilogue_overlap(have_verilator, tmp_path):
    _, w, s, sidecar, x = _case(vocab=129, hidden=1152, seed=3)
    sim = _check(sidecar, C.Activation.quantize(x), w, s, x, latency=30, stalls=20)
    profile = R.lens_profile(sim)
    assert profile["kind"] == "rtl" and "synthetic" in profile["name"]
    assert not profile["c3"]["HBM_modeled"]
    assert profile["c3"]["read_and_dot_overlap_engine_cycles"] > 0
    assert profile["c3"]["read_and_epilogue_overlap_engine_cycles"] > 0
    assert profile["c3"]["event_counts"]["W"] == 129*9
    assert profile["bytes"] == 129*sidecar.layout.pitch
    path = lens.save([profile], tmp_path / "c3.otpuprof")
    loaded = lens.load(path)["profiles"][0]
    assert loaded["cycles"] == profile["cycles"]
    assert loaded["c3"] == profile["c3"]
    assert "C3" in lens.render(lens.load(path))
