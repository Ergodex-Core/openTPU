"""The burst200 build driver's variant options (boards/f2/burst200/build.py, docs/v4/f2-burst200-s3.md):
--from-netlist implements an earlier build's linked netlist and refuses one built from anything
else, --incremental-from hands implement a hashed routed reference, and --floorplan and the
directive options reach implement.tcl through strategy.tcl. Vivado is a stand-in that records
each stage's command line and writes the stage's completion marker; the HDK is its file list."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / 'boards/f2/burst200'
S3_175 = ROOT / 'targets/f2-burst200-s3-175.toml'
S3_162 = ROOT / 'targets/f2-burst200-s3-162.toml'
NETLIST = ['ip', 'pcis', 'clocks', 'synth_link']
LATER = ['implement', 'refine', 'audit']

spec = importlib.util.spec_from_file_location('burst200_build_flow', BOARD / 'build.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)

FAKE_VIVADO = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args == ['-version']:
    print('vivado v2025.2 (64-bit)\nSW Build 0 (test stand-in)')
    sys.exit(0)
d = Path(args[args.index('-source') + 1]).parent
with open(os.environ['FAKE_VIVADO_LOG'], 'a') as f:
    f.write(json.dumps({'run': str(d.parent), 'stage': d.name,
                        'argv': args[args.index('-tclargs') + 1:]}) + '\n')
if d.name in ('ip', 'pcis', 'clocks'):
    (d / 'ip_done.txt').write_text('done\n')
elif d.name == 'synth_link':
    (d / 'post_link.dcp').write_bytes(b'linked netlist ' + str(d).encode())
    (d / 'link_done.txt').write_text('LINK_COMPLETE\n')
elif d.name in ('implement', 'refine'):
    (d / 'post_route.dcp').write_bytes(b'routed ' + str(d).encode())
    (d / 'implementation_result.txt').write_text('setup_slack_ns=0.1\nhold_slack_ns=0.1\n')
    if d.name == 'refine':
        print('ROUTED_TIMING_PASS_PENDING_REPORT_REVIEW')
elif d.name == 'audit':
    (d / 'audit.txt').write_text('max_slack_ns=0.1\n')
'''

HDK_FILES = [
    'common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp',
    'common/shell_stable/build/constraints/cl_pins.xdc',
    'common/shell_stable/build/constraints/small_shell_level_1_fp_cl.xdc',
    'common/shell_stable/design/interfaces/cl_ports.vh',
    'common/shell_stable/design/sh_ddr/synth/sh_ddr.sv',
    'common/lib/interfaces.sv',
    'cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv',
    'common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci',
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_hdk(d: Path) -> Path:
    for f in HDK_FILES:
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text('stand-in\n')
    (d / 'common/shell_stable/shell_version.txt').write_text('small_shell=0x10212415\n')
    return d


class Bench:
    """Runs build.py against the stand-in Vivado and a stand-in HDK under tmp."""
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.hdk = make_hdk(tmp / 'hdk')
        self.vivado = tmp / 'vivado'
        self.vivado.write_text(f'#!{sys.executable}\n' + FAKE_VIVADO)
        self.vivado.chmod(0o755)
        self.log = tmp / 'vivado_calls.jsonl'

    def run(self, out: Path, *extra, target=S3_175, hdk=None, wait=True):
        cmd = [sys.executable, str(BOARD / 'build.py'), '--hdk', str(hdk or self.hdk),
               '--vivado', str(self.vivado), '--target', str(target), '--output', str(out),
               *map(str, extra)]
        env = dict(os.environ, FAKE_VIVADO_LOG=str(self.log))
        if not wait:
            return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, cwd=self.tmp)
        return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=self.tmp)

    def calls(self, out: Path) -> list[dict]:
        if not self.log.exists():
            return []
        rows = [json.loads(l) for l in self.log.read_text().splitlines()]
        return [r for r in rows if r['run'] == str(out.resolve())]

    def parent(self) -> Path:
        out = self.tmp / 'parent'
        r = self.run(out)
        assert r.returncode == 0, r.stderr
        return out


@pytest.fixture
def bench(tmp_path):
    return Bench(tmp_path)


def test_full_build_runs_every_stage_with_the_flows_own_strategy(bench):
    out = bench.parent()
    calls = bench.calls(out)
    assert [c['stage'] for c in calls] == NETLIST + LATER
    impl = calls[4]['argv']
    assert impl[1] == str(out / 'synth_link/post_link.dcp')
    assert impl[3] == str(out / 'synth_link/link_done.txt')
    tcl = (out / 'implement/strategy.tcl').read_text()
    for line in ['set place_directive Explore', 'set phys_directive AggressiveExplore',
                 'set route_directive Explore', 'set incremental_directive RuntimeOptimized']:
        assert line in tcl
    assert not (out / 'implement/incremental_reference.txt').exists()
    assert not (out / 'netlist_parent.json').exists()
    assert (out / 'implement/floorplan.tcl').read_bytes() == (BOARD / 'constraints/floorplan_s3.tcl').read_bytes()
    hashes = json.loads((out / 'inputs.sha256.json').read_text())
    assert hashes['targets/f2-burst200-s3-175.toml'] == sha(S3_175)
    assert hashes['rtl/shell/otpu_board.sv'] == sha(ROOT / 'rtl/shell/otpu_board.sv')


def test_variants_of_one_netlist_skip_the_netlist_stages_side_by_side(bench):
    """Two variants of build 11's kind at once: another floorplan and other directives, each
    implementing the parent's post_link.dcp without the IP, clock, synthesis and link stages."""
    parent = bench.parent()
    variant = bench.tmp / 'floorplan_variant.tcl'
    variant.write_text((BOARD / 'constraints/floorplan_s3.tcl').read_text() + '\n# variant\n')
    a, b = bench.tmp / 'variant_a', bench.tmp / 'variant_b'
    pa = bench.run(a, '--from-netlist', parent, '--floorplan', variant,
                   '--place-directive', 'AggressiveExplore', wait=False)
    pb = bench.run(b, '--from-netlist', parent, '--route-directive', 'NoTimingRelaxation',
                   '--phys-directive', 'AlternateReplication', wait=False)
    for p in (pa, pb):
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err
    dcp = parent / 'synth_link/post_link.dcp'
    for out in (a, b):
        calls = bench.calls(out)
        assert [c['stage'] for c in calls] == LATER
        assert calls[0]['argv'][1] == str(dcp)
        assert calls[0]['argv'][3] == str(parent / 'synth_link/link_done.txt')
        assert calls[1]['argv'][1] == str(out / 'implement/post_route.dcp')
        assert not any((out / s).exists() for s in NETLIST)
        prov = json.loads((out / 'netlist_parent.json').read_text())
        assert prov['run'] == str(parent) and prov['post_link_sha256'] == sha(dcp)
        assert prov['parent_inputs_sha256'] == sha(parent / 'inputs.sha256.json')
        assert {'synth_link/config.tcl', 'clocks/core_clock.tcl', 'synth_link/build.tcl',
                'rtl/shell/otpu_board.sv', 'synth_link/rtl/cl_opentpu.sv'} <= set(prov['checked'])
    assert (a / 'implement/floorplan.tcl').read_bytes() == variant.read_bytes()
    assert json.loads((a / 'inputs.sha256.json').read_text())[str(variant)] == sha(variant)
    ta, tb = (a / 'implement/strategy.tcl').read_text(), (b / 'implement/strategy.tcl').read_text()
    assert 'set place_directive AggressiveExplore' in ta and 'set route_directive Explore' in ta
    assert 'set route_directive NoTimingRelaxation' in tb
    assert 'set phys_directive AlternateReplication' in tb and 'set place_directive Explore' in tb


def _corrupt(bench: Bench, parent: Path, fault: str) -> dict:
    """Make the parent's netlist differ from the build being started; returns run() kwargs."""
    link = parent / 'synth_link'
    if fault == 'clock target':
        return dict(target=S3_162)
    if fault == 'link config':
        (link / 'config.tcl').write_text((link / 'config.tcl').read_text() + '\n')
    elif fault == 'CL top':
        (link / 'rtl/cl_opentpu.sv').write_text('// edited after synthesis\n')
    elif fault == 'shared RTL':
        h = json.loads((parent / 'inputs.sha256.json').read_text())
        h['rtl/shell/otpu_board.sv'] = '0' * 64
        (parent / 'inputs.sha256.json').write_text(json.dumps(h))
    elif fault == 'stage script':
        (link / 'build.tcl').write_text((link / 'build.tcl').read_text() + '# edited\n')
    elif fault == 'geometry':
        (parent / 'ip/geometry.tcl').write_text('set otpu_S 1\n')
    elif fault == 'unlinked':
        (link / 'link_done.txt').unlink()
    elif fault == 'vivado':
        (parent / 'vivado_version.txt').write_text('vivado v2025.2 (64-bit)\nSW Build 1\n')
    elif fault == 'hdk':
        return dict(hdk=make_hdk(bench.tmp / 'other_hdk'))
    return {}


@pytest.mark.parametrize('fault, message', [
    ('clock target', 'the core clock differs'),
    ('link config', 'the link config differs'),
    ('CL top', 'the CL top file cl_opentpu.sv differs'),
    ('shared RTL', 'rtl/shell/otpu_board.sv differs'),
    ('stage script', 'the synth_link stage script differs'),
    ('geometry', 'the geometry differs'),
    ('unlinked', 'its link stage did not complete'),
    ('vivado', 'another Vivado build made it'),
    ('hdk', 'it was linked against the HDK at'),
])
def test_netlist_reuse_refuses_a_netlist_built_from_anything_else(bench, fault, message):
    parent = bench.parent()
    kw = _corrupt(bench, parent, fault)
    out = bench.tmp / 'child'
    r = bench.run(out, '--from-netlist', parent, **kw)
    assert r.returncode != 0 and message in r.stderr, r.stderr
    assert 'build the netlist with the full flow' in r.stderr
    assert not out.exists() and bench.calls(out) == []


def test_netlist_reuse_names_the_first_run_of_a_chain(bench):
    parent = bench.parent()
    child = bench.tmp / 'child'
    assert bench.run(child, '--from-netlist', parent).returncode == 0
    r = bench.run(bench.tmp / 'grandchild', '--from-netlist', child)
    assert r.returncode != 0 and f'it reused the netlist of {parent}; name that run' in r.stderr


def test_incremental_reference_reaches_implement_hashed(bench):
    ref = bench.tmp / 'build11_post_route.dcp'
    ref.write_bytes(b'a routed reference checkpoint')
    out = bench.tmp / 'incr'
    r = bench.run(out, '--incremental-from', ref, '--incremental-directive', 'TimingClosure')
    assert r.returncode == 0, r.stderr
    assert (out / 'implement/incremental_reference.txt').read_text() == f'{ref.resolve()}\n'
    assert (out / 'implement/incremental_reference.sha256').read_text() == sha(ref) + '\n'
    assert 'set incremental_directive TimingClosure' in (out / 'implement/strategy.tcl').read_text()
    assert [c['stage'] for c in bench.calls(out)] == NETLIST + LATER
    # with a reused netlist too (the floorplan changed, the placement kept where it can be)
    parent = bench.parent()
    out2 = bench.tmp / 'incr_reuse'
    r = bench.run(out2, '--from-netlist', parent, '--incremental-from', ref)
    assert r.returncode == 0, r.stderr
    assert [c['stage'] for c in bench.calls(out2)] == LATER
    assert 'set incremental_directive RuntimeOptimized' in (out2 / 'implement/strategy.tcl').read_text()


@pytest.mark.parametrize('args, message', [
    (['--incremental-from', 'REF', '--place-directive', 'Explore'], 'an incremental run takes'),
    (['--incremental-directive', 'Quick'], '--incremental-directive needs --incremental-from'),
    (['--incremental-from', 'NOT_A_DCP'], 'not a checkpoint file'),
    (['--floorplan', 'no_such_floorplan.tcl'], 'no such file'),
    (['--place-directive', 'SSI_SpreadLogic_high'], 'invalid choice'),
])
def test_bad_variant_options_are_refused_before_anything_runs(bench, args, message):
    (bench.tmp / 'ref.dcp').write_bytes(b'x')
    (bench.tmp / 'notes.txt').write_text('x')
    args = [{'REF': str(bench.tmp / 'ref.dcp'), 'NOT_A_DCP': str(bench.tmp / 'notes.txt')}.get(x, x)
            for x in args]
    out = bench.tmp / 'bad'
    r = bench.run(out, *args)
    assert r.returncode == 2 and message in r.stderr, r.stderr
    assert not out.exists()


def test_plan_shows_the_reused_netlist_and_the_strategy(tmp_path):
    out = subprocess.run([sys.executable, str(BOARD / 'build.py'), '--hdk', '/nonexistent',
                          '--target', str(S3_175), '--plan', '--from-netlist', '/runs/b11',
                          '--place-directive', 'Quick'], text=True, capture_output=True,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert 'Netlist: /runs/b11/synth_link/post_link.dcp' in out.stdout
    assert 'place_design Quick, phys_opt_design AggressiveExplore, route_design Explore' in out.stdout
    assert out.stdout.strip().endswith('1. implement\n2. refine\n3. audit')


def test_implement_tcl_runs_the_strategy_files_directives(tmp_path):
    """implement.tcl's directive block reads strategy.tcl (evaluated here under tclsh with and
    without build.py's file), and every directive is a variable, the incremental one included."""
    tcl = (BOARD / 'scripts/implement.tcl').read_text()
    assert 'place_design -directive $place_directive -no_bufg_opt' in tcl
    assert 'phys_opt_design -directive $phys_directive' in tcl
    assert 'route_design -directive $route_directive -tns_cleanup' in tcl
    assert 'read_checkpoint -incremental $ref -directive $incremental_directive' in tcl
    assert not re.search(r'(place_design|phys_opt_design|route_design) -directive [A-Z]', tcl)
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('no tclsh')
    block = re.search(r'^set place_directive .*?\{source \$run/strategy\.tcl\}\n', tcl,
                      re.M | re.S).group(0)
    show = 'puts "$place_directive $phys_directive $route_directive $incremental_directive"\n'

    def evaluate() -> str:
        script = tmp_path / 'check.tcl'
        script.write_text(f'set run {tmp_path}\n' + block + show)
        return subprocess.run([tclsh, str(script)], capture_output=True, text=True,
                              check=True).stdout.strip()
    assert evaluate() == 'Explore AggressiveExplore Explore RuntimeOptimized'
    (tmp_path / 'strategy.tcl').write_text(flow.strategy_tcl(
        dict(place='AggressiveExplore', phys='AddRetime', route='HigherDelayCost'), 'Quick'))
    assert evaluate() == 'AggressiveExplore AddRetime HigherDelayCost Quick'
