"""Apply three auditable compatibility changes to the pinned, disposable Ramulator clone."""
import subprocess
import sys
from pathlib import Path

REV = 'be93be78055d922aa1d4d33e15bcc8f2b0c61a9d'
p = Path(sys.argv[1]).resolve()
assert subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip() == REV
f = p / 'src/dram_controller/impl/generic_dram_controller.cpp'
s = f.read_text()
old = '// TODO: Add code to update statistics'
new = '''// OpenTPU bridge: report write command issue (bridge adds nCWL+nBL).
            if (req_it->callback) req_it->callback(*req_it);'''
if old in s:
    assert s.count(old) == 1
    f.write_text(s.replace(old, new))
else:
    assert new in s
f = p / 'src/dram_controller/impl/refresh/all_bank_refresh.cpp'
s = f.read_text()
old = 'm_num_ranks = m_dram->get_level_size("rank");'
new = '''// HBM2's first level below channel is pseudochannel, DDR4's is rank.
      m_num_ranks = m_dram->m_organization.count.at(1);'''
if old in s:
    f.write_text(s.replace(old, new))
else:
    assert new in s
# Strict Apple Clang requires the dependent-template disambiguator.
f = p / 'src/base/param.h'
s = f.read_text()
old, new = '_config[_name].as<T>()', '_config[_name].template as<T>()'
if old in s:
    f.write_text(s.replace(old, new))
else:
    assert new in s
