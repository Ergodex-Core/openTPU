"""Freshly observed E2/E4 schedules; reuse the unchanged strict comparison helper."""
import json
from pathlib import Path
import pytest
from test_stream_waitc_round5 import native,test_complete_hybrid_batched_prefill_persistent_decode as compare_sequence
CASES=json.loads((Path(__file__).parent/'fixtures/rtl_epilogue_round6.json').read_text())['cases']

@native
@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_observed_e4_and_full_row_drain(case):
    compare_sequence(case)
