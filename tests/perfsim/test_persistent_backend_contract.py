"""The Ramulator comparison must never relabel a deterministic test backend."""
import json
import sys

import pytest

from tools.perfsim import compare_persistent


@pytest.mark.parametrize('backend', ['simple', None])
def test_reject_non_ramulator_before_execution(tmp_path, monkeypatch, backend):
    memory = {'native_ports': True}
    if backend is not None:
        memory['backend'] = backend
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'memory': memory}))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['compare_persistent', '--profile', str(profile),
        '--bundles', str(tmp_path / 'missing-bundles'), '--executable', str(tmp_path / 'missing-rtl'),
        '--bridge', str(tmp_path / 'missing-bridge'), '--out', str(out)])
    with pytest.raises(ValueError, match='requires explicit Ramulator backend'):
        compare_persistent.main()
    assert not out.exists()
