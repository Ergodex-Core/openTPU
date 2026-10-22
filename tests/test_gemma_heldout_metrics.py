"""Timing-scope regressions for the frozen comparison, without a model/card."""
from tools.gemma_specdec_heldout_measure import _aggregate


def _row(visible, interval_s, request_s, cycles, native):
    return dict(visible_decode_intervals=max(visible - 1, 0),
                visible_decode_wall_s=interval_s, request_wall_s=request_s,
                visible_token_ids=list(range(visible)), raw_steps=visible + 1,
                decode_cycles=cycles, target_rows=visible + 2,
                committed_input_rows=visible + 1, device_run_stats=native)


def test_weighted_rates_keep_short_outputs_and_do_not_average_rates():
    a = _aggregate([_row(3, .1, 1., 15000000, {"b_reads": 10}),
                    _row(5, .4, 2., 30000000, {"b_reads": 20}),
                    _row(1, 0., 1., 15000000, {"b_reads": 1}),
                    _row(0, 0., 1., 15000000, {"b_reads": 2})])
    assert a["weighted_interval_wall_tps"] == 12.
    assert a["visible_tokens_per_total_request_wall_s"] == 9 / 5
    assert a["raw_steps_per_total_request_wall_s"] == 13 / 5
    assert a["requests"] == 4 and a["interval_ineligible_requests"] == 2
    assert a["raw_steps_per_device_s"] == 26.
    assert a["target_rows"] == 17 and a["committed_input_rows"] == 13
    assert a["b_reads"] == 33 and a["a_reads"] is None


def test_missing_native_counters_and_absent_intervals_are_not_zero_rates():
    a = _aggregate([_row(1, 0., .1, 150, {"b_reads": 10}),
                    _row(0, 0., .2, 150, {})])
    assert a["weighted_interval_wall_tps"] is None
    assert a["b_reads"] is None
    assert a["requests"] == 2 and a["interval_eligible_requests"] == 0
