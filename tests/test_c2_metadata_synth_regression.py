"""Input/proof contract tests only; vendor regression is an explicit CLI command."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('metadata_synth', ROOT / 'tools/c2_metadata_synth_regression.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_emitted_fifo_and_delay_are_current_production_bytes(tmp_path):
    record = m.emit(tmp_path/'new')
    design = (tmp_path/'new/repro.sv').read_text()
    source = (ROOT/m.MXU).read_text()
    for key in ('TYPES', 'FIFO_DECLARATION', 'FIFO_READ', 'FIFO_WRITE', 'META_DELAY', 'EMETA_DELAY'):
        assert record['excerpts'][key] in source
        assert record['excerpts'][key] in design
    assert (tmp_path/'new/delay.sv').read_text().strip() == record['excerpts']['DELAY_MODULE']
    assert 'logic [$bits(result_t)-1:0] results[RF];' in design
    assert "assign head=result_t'(results[rh]);" in design
    assert 'results[rt]<={tail_meta,integers,tail_sw,tail_sa};' in design


def test_negative_control_changes_only_fifo_representation(tmp_path):
    current = m.emit(tmp_path/'new')
    legacy = m.emit(tmp_path/'old', legacy=True)
    for name in ('delay.sv','tb.sv','synth.tcl'):
        assert (tmp_path/'new'/name).read_bytes() == (tmp_path/'old'/name).read_bytes()
    changed = {k for k in current['excerpts'] if current['excerpts'][k] != legacy['excerpts'][k]}
    assert changed == {'FIFO_DECLARATION','FIFO_READ','FIFO_WRITE'}
    assert "results[rt]<='{meta:tail_meta,x:integers,sw:tail_sw,sa:tail_sa};" in (tmp_path/'old/repro.sv').read_text()


@pytest.mark.parametrize('mutation', ['missing_write', 'duplicate_read', 'different_width'])
def test_source_drift_does_not_silently_use_stale_fixture(mutation):
    source=(ROOT/m.MXU).read_text()
    if mutation=='missing_write': source=source.replace('results[rt]<=','other_results[rt]<=')
    elif mutation=='duplicate_read': source += '\nassign head=results[rh];\n'
    else: source=source.replace('C=4, DEPTH=256','C=8, DEPTH=256')
    with pytest.raises(ValueError): m.extract(source,(ROOT/m.DELAY).read_text())


GOOD='METADATA_FAITHFUL_POST_SYNTH_RESULT commands=32 count=44 received=32 failures=0 done=1\n'

def test_exact_netlist_result_required():
    m.check_simulation(GOOD)
    for text in ('',GOOD+GOOD,GOOD+'Fatal: simulator error\n',GOOD.replace('count=44','count=0'),GOOD.replace('done=1','done=0')):
        with pytest.raises(ValueError):m.check_simulation(text)


def test_instance_attributes_are_not_silently_dropped():
    source=(ROOT/m.MXU).read_text()
    original="otpu_delay #(.W($bits(meta_t)),.N(EP)) u_emeta"
    assert source.count(original)==1
    source=source.replace(original,'(* keep_hierarchy="yes" *) '+original)
    extracted=m.extract(source,(ROOT/m.DELAY).read_text())
    assert extracted['EMETA_DELAY'].startswith('(* keep_hierarchy="yes" *)')
