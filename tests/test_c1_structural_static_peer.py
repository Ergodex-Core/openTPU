"""Peer mutations keep producer summaries coherent while changing raw membership."""
from collections import Counter
import importlib.util
from pathlib import Path
import pytest
from boards.c1_f2_alex import structural_static_review as R, native_review as N

spec = importlib.util.spec_from_file_location('static_peer_fixtures', Path(__file__).with_name('test_c1_structural_static_review.py'))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)


def append_member(facts, clock, member):
    chunk = F.row(facts, 'clocks', 'register_chunk', clock)
    members = N.names(chunk[-1]) + [member]
    F.field(chunk, 'names', F.seq(members))
    F.field(F.row(facts, 'clocks', 'register_inventory', clock), 'count', str(len(members)))
    F.field(F.row(facts, 'clocks', 'clock_registers', clock), 'count', str(len(members)))
    summary = F.row(facts, 'clocks', 'fabric_clock', clock)
    F.field(summary, 'registers', str(len(members)))
    owners = Counter(n[len(N.ROOT):].split('/')[0] for n in members if n.startswith(N.ROOT))
    F.field(summary, 'owner_counts', F.data(owners))
    return facts


@pytest.mark.parametrize('clock,member', [
    ('host', 'WRAPPER/CL/u_native0/unexpected_register'),
    ('host', 'WRAPPER/CL/u_board/unexpected_register'),
    ('core', 'WRAPPER/CL/u_pcis_pipe/unexpected_register'),
    ('hbm', 'WRAPPER/SHELL/outside_register'),
    ('hbm', 'WRAPPER/CL/u_board/unexpected_register'),
    ('core', 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s09_entry_pipeline/extra'),
    ('hbm', 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/m08_exit_pipeline/extra'),
    ('host', 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s01_entry_pipeline/extra'),
    ('core', 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/m01_exit_pipeline/extra'),
])
def test_coherent_producer_summaries_cannot_hide_wrong_clock_or_extra_ports(clock, member):
    facts = append_member(F.fixture(), clock, member)
    result = R.review(facts)
    assert 'clocks' in result['pending_static_steps']
    assert 'unused_clocks' in result['pending_static_steps']
    assert any(i['step'] == 'clocks' for i in result['issues'])
    assert result['route_qualified'] is False


def test_clock_anchor_pin_and_clock_names_are_independently_bound():
    facts = F.fixture()
    anchor = F.row(facts, 'clocks', 'clock_anchor', 'core')
    F.field(anchor, 'pin', 'WRAPPER/CL/u_hbm_clk/clk_out1')
    F.field(anchor, 'selected', 'WRAPPER/CL/u_hbm_clk/clk_out1')
    result = R.review(facts)
    assert 'clocks' in result['pending_static_steps']
    assert any('anchor mapping' in i['reason'] for i in result['issues'])


def test_observed_register_clock_properties_cannot_disagree_with_domain_query():
    facts = F.fixture()
    row = F.row(facts, 'clocks', 'register_inventory', 'core')
    F.field(row, 'properties', row[row.index('properties')+1].replace('5.000', '5.001'))
    result = R.review(facts)
    assert 'clocks' in result['pending_static_steps']
    assert any('properties disagree' in i['reason'] for i in result['issues'])
