"""The combiner retains new blockers after exact inherited-context classification."""
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'boards/f2/build/scripts'


@pytest.mark.parametrize('fault', ['none', 'new_critical', 'classifier_error',
                                 'missing_inherited', 'not_a_blocker', 'duplicate'])
def test_methodology_combiner_keeps_full_severity_inventory(tmp_path, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    shutil.copy2(SCRIPTS / 'classify_methodology.tcl', tmp_path)
    # Property-level validation is tested separately. These helpers model only
    # its outputs, so a new unrelated critical finding cannot disappear here.
    (tmp_path / 'check_shell_baseline.tcl').write_text(r'''
proc otpu_shell_names {objects} {return $objects}
proc otpu_classify_shell_methodology {shell names audit} {
  if {$names ne $::live} {error "incomplete caller inventory"}
  if {$::fault eq "classifier_error"} {error "shell evidence rejected"}
  set inherited $::inherited
  if {$::fault eq "missing_inherited"} {set inherited [lrange $inherited 1 end]}
  return [dict create inherited $inherited remaining $::context]
}
''')
    (tmp_path / 'check_scoped_clocks.tcl').write_text(r'''
proc otpu_classify_scoped_clocks {names prefix audit} {
  if {$names ne $::live} {error "incomplete caller inventory"}
  set context $::context
  if {$::fault eq "not_a_blocker"} {lset context 0 WARNING#1}
  if {$::fault eq "duplicate"} {lset context 0 [lindex $::inherited 0]}
  return [dict create context $context remaining $::inherited]
}
''')
    harness = r'''
set inherited {}; set context {}
for {set i 1} {$i <= 35} {incr i} {lappend inherited TIMING-17#$i}
for {set i 1} {$i <= 3} {incr i} {lappend context TIMING-54#$i}
set live [concat $inherited $context WARNING#1]
if {$fault eq "new_critical"} {lappend live NEW-CRITICAL#1}
proc get_methodology_violations {} {return $::live}
proc list_property {object} {return {NAME SEVERITY}}
proc get_property {property object} {
  if {$property eq "NAME"} {return $object}
  if {$object eq "WARNING#1"} {return Warning}
  return {Critical Warning}
}
set audit [open audit.txt w]
set status [catch {otpu_qualify_methodology shell evidence $audit} result]
close $audit
if {$status} {puts stderr $result; exit 1}
puts "raw=[llength [dict get $result raw_blockers]]"
puts "remaining=[dict get $result blockers]"
'''
    harness = (f'set fault {fault}\n'
               f'source {{{SCRIPTS / "check_rule_severity.tcl"}}}\n'
               f'source {{{tmp_path / "classify_methodology.tcl"}}}\n' + harness)
    result = subprocess.run([tclsh], input=harness, text=True,
                            capture_output=True, cwd=tmp_path)
    audit = (tmp_path / 'audit.txt').read_text()
    assert audit.count('severity {CRITICAL WARNING}') == 38 + (fault == 'new_critical')
    assert 'WARNING#1 severity WARNING' in audit
    if fault in ('none', 'new_critical'):
        assert result.returncode == 0, result.stderr
        assert f'raw={38 + (fault == "new_critical")}' in result.stdout
        assert f'remaining={"NEW-CRITICAL#1" if fault == "new_critical" else ""}\n' in result.stdout
    else:
        assert result.returncode != 0
