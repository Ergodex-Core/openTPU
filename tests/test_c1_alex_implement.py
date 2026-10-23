"""Offline continuation tests. All DCPs/receipts here are explicit temporary mocks."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c1_alex_implement', ROOT / 'boards/c1_f2_alex/implement.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = m.b
_fixtures_spec = importlib.util.spec_from_file_location('c1_alex_sim_fixtures', ROOT / 'tests/test_c1_alex_sim.py')
sf = importlib.util.module_from_spec(_fixtures_spec)
_fixtures_spec.loader.exec_module(sf)


@pytest.fixture
def context(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    names = b.source_inputs()
    for name in names:
        dest = source / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    (source / 'executor.py').write_text('# synthetic executor fixture\n')
    monkeypatch.setattr(m, 'ROOT', source)
    monkeypatch.setattr(b, 'require_clean_source', lambda: '2' * 40)
    monkeypatch.setattr(m, 'executor_inputs', lambda stage: b.inventory([source / 'executor.py'], source))
    run = tmp_path / 'synthesis'; run.mkdir()
    for stage in b.STAGES:
        (run / stage).mkdir()
    # Copy baseline recipes only. These tests never execute a Vivado process.
    link = (b.SUPPORT / 'boards/f2/scripts/build_f2.tcl').read_text()
    link = b.bounded_script(link).replace('foreach {output factor} {clk_out1 1.0 clk_out2 2.0}',
                                          'foreach {output factor} {clk_out1 1.0}')
    (run / 'link/build.tcl').write_text(link)
    for stage, name in [('implement', 'implement.tcl'), ('refine', 'refine_route.tcl')]:
        (run / stage / 'build.tcl').write_text(b.bounded_script((b.SUPPORT / 'boards/f2/scripts' / name).read_text()))
    for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'):
        (run / 'implement' / name).write_text(f'# synthetic prepared {name}\n')
    (run / 'source.sv').write_text('// synthetic source\n')
    prepared = b.inventory([run / 'source.sv', run / 'link/build.tcl', run / 'implement', run / 'refine'], run)
    hdk = tmp_path / 'aws-fpga'; hdk.mkdir()
    installed = {'hdk/shell.dcp': {'sha256': 'a' * 64, 'bytes': 9}}
    inputs = {'schema': 1, 'kind': 'c1-alex-burst200-build-inputs', 'source_revision': '1' * 40,
              'source_status': '', 'build_id': '11111111', 'part': b.PART, 'core_khz': 200000,
              'hbm_mhz': 450, 'hbm_ports': 8, 'channel_bytes': 2**29,
              'alex_revision': b.ALEX_REVISION, 'archive_sha256': b.ARCHIVE_SHA256,
              'support_manifest_sha256': b.sha(b.SUPPORT / 'manifest.json'),
              'hdk': str(hdk), 'prepared_files': prepared, 'source_inputs': names,
              'installed_inputs': installed}
    b.write_json(run / 'inputs.json', inputs)
    ip, ip_sha = sf.ip_fixture(tmp_path)
    b.write_json(run / 'ip-binding.json', {'run': str(ip), 'receipt_sha256': ip_sha})
    (run / 'synth/post_synth.dcp').write_bytes(b'SYNTHETIC TEST DCP, NOT HARDWARE')
    (run / 'synth/synth_done.txt').write_text('C1_ALEX_SYNTH_ONLY_COMPLETE\n')
    b.write_json(run / 'synth/exit.json', {'returncode': 0})
    receipt = {'schema': 1, 'kind': 'c1-alex-synthesis-only', 'qualification': 'not-qualified',
               'inputs_sha256': b.sha(run / 'inputs.json'),
               'checkpoint_sha256': b.sha(run / 'synth/post_synth.dcp'), 'ip_receipt_sha256': ip_sha}
    b.write_json(run / 'synth-complete.json', receipt)
    monkeypatch.setattr(b, 'check_hdk', lambda path: installed)
    monkeypatch.setattr(b, 'verify_ip_run', lambda path, expected: None)
    receipt_sha = b.sha(run / 'synth-complete.json')
    origin = m.verify_synthesis(run, receipt_sha, hdk)
    return SimpleNamespace(tmp=tmp_path, source=source, run=run, hdk=hdk, ip=ip,
                           origin=origin, receipt_sha=receipt_sha, installed=installed,
                           previous=None, calls=[])


def fake_tool(ctx, monkeypatch, stage, *, returncode=0, setup=0.04, hold=0.01, qualified=False,
              mutate=None, wrong_digest=False, missing_checkpoint=False):
    def run(cmd, cwd, stdout, stderr):
        ctx.calls.append(cmd)
        cwd = Path(cwd)
        if stage == 'link':
            (cwd / 'link_done.txt').write_text('C1_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED\n')
            (cwd / 'audit-result.txt').write_text('structural_pass=1\nroute_qualified=0\n')
        elif stage in ('implement', 'refine'):
            (cwd / 'implementation_result.txt').write_text(f'setup_slack_ns={setup}\nhold_slack_ns={hold}\nIndependent review required.\n')
        else:
            digest = '0' * 64 if wrong_digest else b.sha(ctx.previous['checkpoint'])
            (cwd / 'audit-result.txt').write_text(f'route_qualified={int(qualified)}\ncheckpoint_sha256={digest}\n')
        if stage in m.CHECKPOINTS and not missing_checkpoint:
            (cwd / m.CHECKPOINTS[stage]).write_bytes(f'SYNTHETIC {stage} TEST DCP'.encode())
        stdout.write('SYNTHETIC TOOL LOG\n')
        if mutate:
            mutate(cwd)
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(m.subprocess, 'run', run)


def simulation_run(ctx, monkeypatch):
    """Temporary synthetic logs exercise receipt checks, never vendor hardware."""
    sim = m.simulation_helper()
    monkeypatch.setattr(sim.build, 'command', lambda *a, **kw: 'Vivado v2025.2 (64-bit)')
    real = sim.run_scenario
    def fake_scenario(out, private, seed, phase, vivado):
        def fake(cmd, **kw):
            kw['stdout'].write(sf.pass_log(seed, phase))
            return SimpleNamespace(returncode=0)
        return real(out, private, seed, phase, vivado, runner=fake)
    monkeypatch.setattr(sim, 'run_scenario', fake_scenario)
    out = ctx.tmp / 'simulation'
    sim.main(['--ip-run', str(ctx.ip), '--ip-receipt-sha256', ctx.origin['receipt']['ip_receipt_sha256'],
              '--out', str(out)])
    proof = m.verify_simulation(out, b.sha(out / 'simulation-complete.json'), ctx.origin)
    return out, proof


def stage_run(ctx, monkeypatch, stage, *, max_threads=4, **kwargs):
    out = ctx.tmp / ('stage-' + stage)
    simulation = simulation_run(ctx, monkeypatch)[1] if stage == 'qualify' else None
    inputs = m.prepare(out, stage, ctx.origin, ctx.previous, ctx.hdk, simulation, max_threads=max_threads)
    fake_tool(ctx, monkeypatch, stage, **kwargs)
    result = m.execute(out, stage, ctx.origin, ctx.previous, inputs, ctx.hdk, 'vivado')
    if stage != 'qualify':
        ctx.previous = m.verify_previous(out, b.sha(out / 'stage-result.json'), stage, ctx.origin)
    return out, result


def test_synthesis_provenance_verified(context):
    assert context.origin['inputs']['build_id'] == '11111111'
    assert context.origin['checkpoint'].name == 'post_synth.dcp'


@pytest.mark.parametrize('change', ['receipt', 'checkpoint', 'source', 'prepared', 'ip', 'marker', 'exit', 'hdk'])
def test_synthesis_refuses_changed_evidence(context, monkeypatch, change):
    c = context
    if change == 'receipt':
        (c.run / 'synth-complete.json').write_text('{}')
    elif change == 'checkpoint':
        (c.run / 'synth/post_synth.dcp').write_text('changed')
    elif change == 'source':
        name = next(iter(c.origin['inputs']['source_inputs']))
        (c.source / name).write_text('changed')
    elif change == 'prepared':
        (c.run / 'link/build.tcl').write_text('changed')
    elif change == 'ip':
        (c.ip / 'ip-complete.json').write_text('{}')
    elif change == 'marker':
        (c.run / 'synth/synth_done.txt').write_text('pending')
    elif change == 'exit':
        b.write_json(c.run / 'synth/exit.json', {'returncode': 1})
    else:
        monkeypatch.setattr(b, 'check_hdk', lambda _: {})
    with pytest.raises(RuntimeError):
        m.verify_synthesis(c.run, c.receipt_sha, c.hdk)


def test_link_is_existing_dcp_only(context):
    script = m.render_link(context.origin)
    assert 'add_files $checkpoint' in script
    assert 'link_design -mode default -reconfig_partitions {WRAPPER/CL} -top top' in script
    assert 'synth_design' not in script and 'read_verilog' not in script and 'generate_target' not in script
    assert script.count('general.maxThreads 4') == 1
    assert '::c1_alex_route::audit_linked $run' in script
    assert 'clk_out1 1.0 clk_out2 2.0' not in script


@pytest.mark.parametrize('strategy', m.PHYSICAL_STRATEGIES)
def test_eight_threads_change_only_exact_frozen_thread_parameters(context, strategy):
    original = (context.run / 'implement/build.tcl').read_text()
    four = m.physical_script(original, strategy)
    eight = m.physical_script(original, strategy, 8)
    assert eight == four.replace('set_param general.maxThreads 4\n', 'set_param general.maxThreads 8\n')
    assert 'general.maxThreads 4' not in eight
    assert m.physical_script(original, strategy, 4) == four


@pytest.mark.parametrize('threads', [True, False, 4.0, 8.0, '8', 0, 6, 16, None])
def test_thread_budget_rejects_unreviewed_values_before_output(context, threads):
    out = context.tmp / 'invalid-threads'
    with pytest.raises(RuntimeError, match='Thread limit'):
        m.prepare(out, 'implement', context.origin, None, context.hdk, max_threads=threads)
    assert not out.exists()


def test_thread_anchor_cannot_be_missing_altered_or_unbound(context):
    original = (context.run / 'implement/build.tcl').read_text()
    for changed in (original.replace('set_param general.maxThreads 4\n', ''),
                    original.replace('general.maxThreads 4', 'general.maxThreads 6'),
                    original + 'set_param general.maxThreads $unbound\n'):
        with pytest.raises(RuntimeError, match='Thread limit anchor'):
            m.physical_script(changed, 'default', 8)


@pytest.mark.parametrize('stage', ['link', 'qualify'])
def test_eight_threads_are_restricted_to_physical_stages(context, stage):
    with pytest.raises(RuntimeError, match='Thread limit'):
        m.prepare(context.tmp / 'invalid', stage, context.origin, None, context.hdk, max_threads=8)


def test_eight_thread_implementation_and_refine_bind_exact_scripts(context, monkeypatch):
    stage_run(context, monkeypatch, 'link')
    for stage in ('implement', 'refine'):
        out, _ = stage_run(context, monkeypatch, stage, max_threads=8)
        inputs = json.loads((out / 'stage-inputs.json').read_text())
        assert inputs['max_threads'] == 8
        actual = (out / 'run.tcl').read_text()
        frozen = (context.run / stage / 'build.tcl').read_text()
        assert actual == frozen.replace('set_param general.maxThreads 4\n', 'set_param general.maxThreads 8\n')
    # Even a self-consistently re-hashed input/receipt cannot relabel an8-thread
    # script as a4-thread run; ancestry and actual script are checked again.
    inputs['max_threads'] = 4
    b.write_json(out / 'stage-inputs.json', inputs)
    result = json.loads((out / 'stage-result.json').read_text())
    result['inputs_sha256'] = b.sha(out / 'stage-inputs.json')
    b.write_json(out / 'stage-result.json', result)
    with pytest.raises(RuntimeError, match='strategy/script'):
        m.verify_previous(out, b.sha(out / 'stage-result.json'), 'refine', context.origin)


def test_explicit_skip_strategy_changes_only_pre_route_physical_pass(context, monkeypatch):
    stage_run(context, monkeypatch, 'link')
    out = context.tmp / 'skip-phys-opt'
    inputs = m.prepare(out, 'implement', context.origin, context.previous, context.hdk,
                       physical_strategy='skip-pre-route-phys-opt')
    original = (context.run / 'implement/build.tcl').read_text()
    actual = (out / 'run.tcl').read_text()
    assert actual == original.replace('  phys_opt_design -directive AggressiveExplore\n',
                                     '  puts C1_PRE_ROUTE_PHYS_OPT_SKIPPED_BY_EXPLICIT_STRATEGY\n')
    assert inputs['physical_strategy'] == 'skip-pre-route-phys-opt'
    assert inputs['source_revision'] != inputs['executor_revision']
    for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'):
        assert (out / name).read_bytes() == (context.run / 'implement' / name).read_bytes()
    fake_tool(context, monkeypatch, 'implement')
    m.execute(out, 'implement', context.origin, context.previous, inputs, context.hdk, 'vivado')
    m.verify_previous(out, b.sha(out / 'stage-result.json'), 'implement', context.origin)


@pytest.mark.parametrize('strategy', ['align-promoted-carry-pblocks', 'align-complete-promoted-carry-pblocks'])
def test_carry_pblock_repair_keeps_link_ancestry_and_constraints(context, monkeypatch, strategy):
    stage_run(context, monkeypatch, 'link')
    out = context.tmp / 'carry-pblock-repair'
    inputs = m.prepare(out, 'implement', context.origin, context.previous, context.hdk,
                       physical_strategy=strategy)
    original = (context.run / 'implement/build.tcl').read_text()
    expected = original.replace('  phys_opt_design -directive AggressiveExplore\n',
                                '  puts C1_PRE_ROUTE_PHYS_OPT_SKIPPED_BY_EXPLICIT_STRATEGY\n')
    repair = m.CARRY_PBLOCK_SCRIPT + (m.complete_carry_pblock_script() if strategy == 'align-complete-promoted-carry-pblocks' else '')
    expected = expected.replace('write_checkpoint -force $run/post_opt.dcp\n',
                                repair + 'write_checkpoint -force $run/post_opt.dcp\n')
    assert (out / 'run.tcl').read_text() == expected
    assert inputs['source_checkpoint']['sha256'] == b.sha(context.previous['checkpoint'])
    assert inputs['source_revision'] != inputs['executor_revision']
    for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'):
        assert (out / name).read_bytes() == (context.run / 'implement' / name).read_bytes()
    # Additive diagnostic output must survive the ordinary stage receipt inventory.
    (out / 'complete_carry_pblock_repair.tcldata').write_text('raw repair observations\n')
    fake_tool(context, monkeypatch, 'implement')
    m.execute(out, 'implement', context.origin, context.previous, inputs, context.hdk, 'vivado')
    m.verify_previous(out, b.sha(out / 'stage-result.json'), 'implement', context.origin)
    receipt = json.loads((out / 'stage-result.json').read_text())
    assert receipt['outputs']['complete_carry_pblock_repair.tcldata']['sha256'] == b.sha(out / 'complete_carry_pblock_repair.tcldata')


def test_carry_pblock_repair_rejects_changed_post_opt_anchor(context):
    script = (context.run / 'implement/build.tcl').read_text()
    anchor = 'write_checkpoint -force $run/post_opt.dcp\n'
    for changed in (script.replace(anchor, ''), script + anchor):
        with pytest.raises(RuntimeError, match='anchor'):
            m.physical_script(changed, 'align-promoted-carry-pblocks')


@pytest.mark.parametrize('stage', ['link', 'refine', 'qualify'])
def test_carry_pblock_repair_cannot_run_in_other_stages(context, stage):
    with pytest.raises(RuntimeError, match='only to implementation'):
        m.prepare(context.tmp / stage, stage, context.origin, context.previous, context.hdk,
                  physical_strategy='align-promoted-carry-pblocks')


@pytest.mark.parametrize('strategy', ['unknown', 'Explore', '', None])
def test_unknown_physical_strategy_rejected(context, strategy):
    with pytest.raises(RuntimeError, match='strategy'):
        m.physical_script((context.run / 'implement/build.tcl').read_text(), strategy)


def test_skip_strategy_requires_exact_anchor(context):
    script = (context.run / 'implement/build.tcl').read_text()
    for changed in (script.replace('AggressiveExplore', 'Explore'), script + '  phys_opt_design -directive AggressiveExplore\n'):
        with pytest.raises(RuntimeError, match='anchor'):
            m.physical_script(changed, 'skip-pre-route-phys-opt')


def test_skip_strategy_cannot_change_refine_or_link(context):
    for stage in ('link', 'refine'):
        with pytest.raises(RuntimeError, match='only to implementation'):
            m.prepare(context.tmp / stage, stage, context.origin, context.previous, context.hdk,
                      physical_strategy='skip-pre-route-phys-opt')


def test_strategy_mismatch_rejected_despite_rebound_receipt(context, monkeypatch):
    stage_run(context, monkeypatch, 'link')
    out, _ = stage_run(context, monkeypatch, 'implement')
    inputs = json.loads((out / 'stage-inputs.json').read_text())
    inputs['physical_strategy'] = 'skip-pre-route-phys-opt'
    b.write_json(out / 'stage-inputs.json', inputs)
    result = json.loads((out / 'stage-result.json').read_text())
    result['inputs_sha256'] = b.sha(out / 'stage-inputs.json')
    b.write_json(out / 'stage-result.json', result)
    with pytest.raises(RuntimeError, match='strategy/script'):
        m.verify_previous(out, b.sha(out / 'stage-result.json'), 'implement', context.origin)


def test_full_stage_chain_preserves_candidate_and_pending_review(context, monkeypatch):
    c = context
    link, linked = stage_run(c, monkeypatch, 'link')
    assert linked['status'] == 'complete' and linked['route_qualified'] is False
    impl, candidate = stage_run(c, monkeypatch, 'implement', setup=-4.024)
    assert candidate['status'] == 'candidate' and candidate['timing']['setup_slack_ns'] == -4.024
    refine, refined = stage_run(c, monkeypatch, 'refine', setup=0.032)
    assert refined['status'] == 'complete' and refined['route_qualified'] is False
    qualify, pending = stage_run(c, monkeypatch, 'qualify')
    assert pending['status'] == 'pending-review' and pending['route_qualified'] is False
    assert (impl / 'post_route.dcp').read_bytes() == b'SYNTHETIC implement TEST DCP'
    assert pending['checkpoint_sha256'] == refined['checkpoint_sha256']
    for run in (link, impl, refine, qualify):
        inputs = json.loads((run / 'stage-inputs.json').read_text())
        assert inputs['executor_revision'] == '2' * 40
        assert inputs['source_revision'] == '1' * 40
        assert inputs['build_id'] == '11111111'
        assert inputs['hardware_actions'] is False and inputs['resynthesis'] is False
    assert all('synth' not in command[:5] for command in c.calls)


def test_only_current_qualifier_may_accept(context, monkeypatch):
    for stage in ('link', 'implement', 'refine'):
        stage_run(context, monkeypatch, stage)
    _, result = stage_run(context, monkeypatch, 'qualify', qualified=True)
    assert result['status'] == 'complete' and result['route_qualified'] is True


def test_qualification_digest_must_match_opened_checkpoint(context, monkeypatch):
    for stage in ('link', 'implement', 'refine'):
        stage_run(context, monkeypatch, stage)
    with pytest.raises(RuntimeError, match='different DCP'):
        stage_run(context, monkeypatch, 'qualify', qualified=True, wrong_digest=True)
    result = json.loads((context.tmp / 'stage-qualify/stage-result.json').read_text())
    assert result['status'] == 'failed' and result['route_qualified'] is False


def test_tool_failure_preserves_failed_attempt(context, monkeypatch):
    with pytest.raises(RuntimeError, match='Vivado process failed'):
        stage_run(context, monkeypatch, 'link', returncode=1)
    out = context.tmp / 'stage-link'
    assert (out / 'post_link.dcp').is_file()
    result = json.loads((out / 'stage-result.json').read_text())
    assert result['returncode'] == 1 and result['status'] == 'failed'
    with pytest.raises(RuntimeError, match='existing stage'):
        m.prepare(out, 'link', context.origin, None, context.hdk)
    with pytest.raises(RuntimeError, match='required gate'):
        m.verify_previous(out, b.sha(out / 'stage-result.json'), 'link', context.origin)


@pytest.mark.parametrize('setup,hold', [(-0.01, 0.01), (0.01, -0.01), ('nan', 0.01), ('inf', 0.01)])
def test_refine_nonpassing_or_nonfinite_fails(context, monkeypatch, setup, hold):
    stage_run(context, monkeypatch, 'link')
    stage_run(context, monkeypatch, 'implement')
    with pytest.raises(RuntimeError):
        stage_run(context, monkeypatch, 'refine', setup=setup, hold=hold)
    out = context.tmp / 'stage-refine'
    assert (out / 'post_route.dcp').is_file()
    assert json.loads((out / 'stage-result.json').read_text())['status'] == 'failed'


def test_missing_checkpoint_cannot_complete(context, monkeypatch):
    with pytest.raises(RuntimeError, match='checkpoint missing'):
        stage_run(context, monkeypatch, 'link', missing_checkpoint=True)


@pytest.mark.parametrize('name', ['tuning.tcl', 'incremental_reference.txt', 'incremental_reference.sha256'])
def test_no_unreviewed_control_injection(context, monkeypatch, name):
    with pytest.raises(RuntimeError, match='Unreviewed tuning'):
        stage_run(context, monkeypatch, 'link', mutate=lambda cwd: (cwd / name).write_text('injected'))
    result = json.loads((context.tmp / 'stage-link/stage-result.json').read_text())
    assert result['status'] == 'failed'


def test_source_mutation_during_tool_is_rejected(context, monkeypatch):
    def mutate(_):
        (context.run / 'synth/post_synth.dcp').write_bytes(b'changed during tool')
    with pytest.raises(RuntimeError, match='Synthesis checkpoint changed'):
        stage_run(context, monkeypatch, 'link', mutate=mutate)


def test_old_chain_mutation_detected(context, monkeypatch):
    link, _ = stage_run(context, monkeypatch, 'link')
    impl, _ = stage_run(context, monkeypatch, 'implement')
    (link / 'post_link.dcp').write_text('changed ancestor')
    with pytest.raises(RuntimeError, match='Frozen file changed'):
        m.verify_previous(impl, b.sha(impl / 'stage-result.json'), 'implement', context.origin)


@pytest.mark.parametrize('text', ['setup_slack_ns=0.1\nhold_slack_ns=0.2\nsetup_slack_ns=0.3\n',
                                'setup_slack_ns=\nhold_slack_ns=0.2\n', 'setup_slack_ns=0.1\n'])
def test_timing_parser_rejects_missing_duplicate_empty(tmp_path, text):
    path = tmp_path / 'timing'; path.write_text(text)
    with pytest.raises(RuntimeError):
        m.timing_result(path)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'x/../escape', 'x//y', 3])
def test_manifest_paths_rejected(name):
    with pytest.raises(RuntimeError):
        m.relative(name)


def test_receipt_symlink_rejected(tmp_path):
    source = tmp_path / 'receipt'; source.write_text('{}')
    link = tmp_path / 'link'; link.symlink_to(source)
    with pytest.raises(RuntimeError, match='linked receipt'):
        m.read_bound(link, b.sha(source))


def test_plan_never_invokes_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(b, 'command', lambda *a, **k: pytest.fail('no tool calls in plan'))
    out = tmp_path / 'uncreated'
    assert m.main(['--stage', 'link', '--out', str(out), '--hdk', '/unused',
        '--synth-run', '/unused', '--synth-receipt-sha256', 'a' * 64, '--plan']) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result['resynthesis'] and not result['hardware_actions']
    assert not out.exists()


@pytest.mark.parametrize('stage,extra', [('link', ['--from-run', '/x', '--from-receipt-sha256', 'a' * 64]),
                                      ('implement', []), ('refine', []), ('qualify', [])])
def test_stage_arguments_fail_closed(tmp_path, stage, extra):
    with pytest.raises(SystemExit):
        m.main(['--stage', stage, '--out', str(tmp_path / 'out'), '--hdk', '/x', '--synth-run', '/x',
                '--synth-receipt-sha256', 'b' * 64, '--plan', *extra])


def test_manifest_mutation_during_tool_is_rejected(context, monkeypatch):
    with pytest.raises(RuntimeError, match='input manifest changed'):
        stage_run(context, monkeypatch, 'link', mutate=lambda out: (out / 'stage-inputs.json').write_text('{}'))
    result = json.loads((context.tmp / 'stage-link/stage-result.json').read_text())
    assert result['status'] == 'failed' and result['route_qualified'] is False


@pytest.mark.parametrize('text', ['route_qualified=1\nroute_qualified=0\ncheckpoint_sha256=x\n',
                                'route_qualified=1\n', 'garbage\n',
                                'route_qualified=1\ncheckpoint_sha256=x\nextra=1\n'])
def test_audit_result_is_exact(tmp_path, text):
    path = tmp_path / 'audit'; path.write_text(text)
    with pytest.raises(RuntimeError):
        m.machine_result(path, ('route_qualified', 'checkpoint_sha256'))


def test_simulation_receipt_matches_original_ip_and_all_four_scenarios(context, monkeypatch):
    run, proof = simulation_run(context, monkeypatch)
    assert proof['receipt_sha256'] == b.sha(run / 'simulation-complete.json')
    assert proof['inputs_sha256'] == b.sha(run / 'inputs.json')
    assert len(json.loads((run / 'simulation-complete.json').read_text())['results']) == 4


@pytest.mark.parametrize('change', ['state', 'kind', 'results', 'omitted_evidence', 'input_kind',
    'ip_receipt', 'ip_inputs', 'revision', 'products', 'clocks', 'period', 'model', 'runner',
    'builder', 'sources', 'scenarios', 'missing_dependencies', 'command', 'consumed_inputs',
    'exit', 'console', 'result', 'rtl', 'script', 'version', 'dependency', 'added_dependency'])
def test_simulation_gate_rejects_semantically_wrong_or_changed_evidence(context, monkeypatch, change):
    run, _ = simulation_run(context, monkeypatch)
    receipt_path = run / 'simulation-complete.json'
    receipt = json.loads(receipt_path.read_text())
    inputs = json.loads((run / 'inputs.json').read_text())
    scenario = run / 'seed-17-phase-0'
    if change == 'state': receipt['state'] = 'failed'
    elif change == 'kind': receipt['kind'] = 'behavioral-proxy'
    elif change == 'results': receipt['results'].pop()
    elif change == 'omitted_evidence': receipt['evidence'].pop('seed-17-phase-0/console.log')
    elif change == 'input_kind': inputs['kind'] = 'other'
    elif change == 'ip_receipt': inputs['ip_receipt_sha256'] = '0' * 64
    elif change == 'ip_inputs': inputs['build_inputs_sha256'] = '0' * 64
    elif change == 'revision': inputs['ip_source_revision'] = '0' * 40
    elif change == 'products': inputs['ip_products'] = {}
    elif change == 'clocks': inputs['clocks_mhz']['core'] = 100
    elif change == 'period': inputs['hbm_simulated_period_ps'] = 5000
    elif change == 'model': inputs['hbm_model'] = 'silicon-timing'
    elif change == 'runner': inputs['runner_sha256'] = '0' * 64
    elif change == 'builder': inputs['builder_sha256'] = '0' * 64
    elif change == 'sources': inputs['sources']['tb_f2_mem_c1.sv'] = '0' * 64
    elif change == 'scenarios': inputs['scenarios'].reverse()
    elif change == 'missing_dependencies': inputs.pop('simulation_ip_files')
    elif change == 'command': b.write_json(scenario / 'command.json', ['true'])
    elif change == 'consumed_inputs': b.write_json(scenario / 'inputs.json', {})
    elif change == 'exit': b.write_json(scenario / 'exit.json', {'returncode': 1})
    elif change == 'console': (scenario / 'console.log').write_text('PASS\n')
    elif change == 'result': b.write_json(scenario / 'result.json', {})
    elif change == 'rtl': (scenario / 'rtl/tb_f2_mem_c1.sv').write_text('// changed\n')
    elif change == 'script': (scenario / 'simulate.tcl').write_text('puts PASS\n')
    elif change == 'version': (run / 'vivado_version.txt').write_text('Vivado v2024.1\n')
    elif change == 'dependency': (run / 'private-ip/ip/ip_project/child.dcp').write_text('changed')
    else: (run / 'private-ip/ip/ip_project/new_header.vh').write_text('extra unreviewed dependency')
    b.write_json(run / 'inputs.json', inputs)
    receipt['inputs_sha256'] = b.sha(run / 'inputs.json')
    # Rebind changed records to test semantic gates as well as plain hashes.
    for name in receipt['evidence']:
        path = run / name
        receipt['evidence'][name] = {'sha256': b.sha(path), 'bytes': path.stat().st_size}
    b.write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError):
        m.verify_simulation(run, b.sha(receipt_path), context.origin)


def test_simulation_receipt_and_raw_log_hashes_are_checked(context, monkeypatch):
    run, proof = simulation_run(context, monkeypatch)
    (run / 'seed-17-phase-0/console.log').write_text('changed raw result')
    with pytest.raises(RuntimeError, match='Frozen file changed'):
        m.verify_simulation(run, proof['receipt_sha256'], context.origin)
    with pytest.raises(RuntimeError, match='Receipt hash mismatch'):
        m.verify_simulation(run, '0' * 64, context.origin)


def test_simulation_mutation_during_qualification_rejects_positive_audit(context, monkeypatch):
    for stage in ('link', 'implement', 'refine'):
        stage_run(context, monkeypatch, stage)
    with pytest.raises(RuntimeError, match='Frozen file changed'):
        stage_run(context, monkeypatch, 'qualify', qualified=True,
                  mutate=lambda _: (context.tmp / 'simulation/seed-17-phase-0/console.log').write_text('changed'))
    receipt = json.loads((context.tmp / 'stage-qualify/stage-result.json').read_text())
    assert receipt['status'] == 'failed' and receipt['route_qualified'] is False
    inputs = json.loads((context.tmp / 'stage-qualify/stage-inputs.json').read_text())
    assert inputs['simulation']['run'] == str(context.tmp / 'simulation')
    assert len(inputs['simulation']['receipt_sha256']) == 64


@pytest.mark.parametrize('stage,extra', [
    ('qualify', ['--from-run', '/x', '--from-receipt-sha256', 'a' * 64]),
    ('link', ['--simulation-run', '/x', '--simulation-receipt-sha256', 'a' * 64]),
])
def test_simulation_arguments_are_required_only_for_qualification(tmp_path, stage, extra):
    with pytest.raises(SystemExit):
        m.main(['--stage', stage, '--out', str(tmp_path / 'out'), '--hdk', '/x', '--synth-run', '/x',
                '--synth-receipt-sha256', 'b' * 64, '--plan', *extra])
