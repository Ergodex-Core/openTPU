"""Portable parser fixtures only: no Vivado, physical qualification, or hardware.

Positive data is fabricated in pytest temporary directories to exercise the
receipt graph. It is never emitted into build evidence or used for deployment.
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

e = load('portable_evidence', ROOT / 'boards/c1_f2_alex/candidate_evidence.py')
b = e.local_module('build')
sim = e.local_module('simulate')
implementation = e.local_module('implement')
f = load('portable_build_fixture', ROOT / 'tests/test_c1_alex_build.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode() if isinstance(value, str)
                     else (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def rec(path):
    return {'sha256': e.sha(path), 'bytes': path.stat().st_size}


def read(path): return json.loads(path.read_text())


def inventory(path):
    return {p.relative_to(path).as_posix(): rec(p) for p in path.rglob('*') if p.is_file()}


def source_copy(root, role, names):
    records = {}
    for name in names:
        dest = root / 'sources' / role / name
        put(dest, (ROOT / name).read_bytes()); records[name] = rec(dest)
    return records


def pass_log(seed, phase):
    lines = []
    for pc in range(8):
        n = 610 if pc in (2, 6) else 609
        lines.append(f'C1_PC pc={pc} accepted={n} returned={n} consumed={n} axi_beats={n} peak=32 hbm_reads=2000 hbm_writes=1000')
    return '\n'.join(lines) + f'\nC1_ALEX_FABRIC_PASS seed={seed} phase={phase} pcs=8 reset_pending=8 host_b=994 host_r=1229\n'


def refresh_manifest(root, m):
    """Repin outer collection only; frozen inner receipts remain unchanged."""
    m['files'] = {name: row for name, row in inventory(root).items() if name != e.MANIFEST}
    for role, filename in e.RECEIPTS.items():
        name = m['roots'][role]['bundle_relative_path'] + '/' + filename
        m['receipts'][role] = {'relative_path': name, 'sha256': m['files'][name]['sha256']}
    put(root / e.MANIFEST, m)
    return e.sha(root / e.MANIFEST)


def fixture_bundle(tmp_path, monkeypatch, *, original_base="/unavailable/original"):
    root = tmp_path / 'portable'; root.mkdir()
    roots = {role: {'original_absolute_path': original_base + '/' + role,
                    'bundle_relative_path': 'runs/' + role} for role in e.ROLES}
    roots['simulation_private'] = {key: value + '/private-ip' for key, value in roots['simulation'].items()}
    def run(role): return root / roots[role]['bundle_relative_path']
    def old(role, name=''): return roots[role]['original_absolute_path'] + ('/' + name if name else '')
    m = {'schema': 1, 'kind': 'c1-alex-portable-candidate-evidence', 'instance_id': e.INSTANCE,
         'hardware': {'source_revision': 'a'*40, 'build_id': 'a'*8},
         'roots': roots, 'receipts': {}, 'source_snapshots': {}, 'files': {}}
    for role in e.SOURCE_ROLES:
        m['source_snapshots'][role] = {'revision': 'a'*40 if role in ('ip', 'hardware') else 'd' * 40,
                                      'bundle_relative_path': 'sources/' + role}
    shell = 'hdk/common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp'
    put(run('installed') / shell, 'SYNTHETIC SHELL DCP FOR PARSER TEST ONLY')
    monkeypatch.setattr(b, 'SHELL_SHA256', e.sha(run('installed') / shell))
    monkeypatch.setattr(e, 'local_module', lambda name: {'build': b, 'simulate': sim, 'implement': implementation}[name])
    base_inputs = dict(schema=1, kind='c1-alex-burst200-build-inputs', source_status='',
                       source_revision='a'*40, build_id='a'*8, part=b.PART,
                       core_khz=200000, hbm_mhz=450, hbm_ports=8, channel_bytes=2**29,
                       alex_revision=b.ALEX_REVISION, archive_sha256=b.ARCHIVE_SHA256,
                       hdk_revision=b.HDK_REVISION, ip_revision=b.IP_REVISION,
                       support_manifest_sha256=e.sha(b.SUPPORT / 'manifest.json'),
                       hdk=old('installed'), installed_inputs=inventory(run('installed')))
    support_names = ['boards/c1_f2_alex/support/' + name for name in inventory(b.SUPPORT)]
    for role, source_role in (('ip', 'ip'), ('synth', 'hardware')):
        source_copy(root, source_role, support_names)
        inp = copy.deepcopy(base_inputs)
        inp['source_inputs'] = source_copy(root, source_role, b.source_inputs())
        for name in e.prepared_names(b):
            put(run(role) / name, '# SYNTHETIC prepared test input ' + name + '\n')
        inp['prepared_files'] = inventory(run(role))
        put(run(role) / 'inputs.json', inp)
    products = [sim.BD, sim.WRAPPER, sim.PCIS, 'ip/ip_project/child.dcp']
    for name in products: put(run('ip') / name, 'SYNTHETIC private IP product ' + name)
    put(run('ip') / sim.BD, f.generated_bd())
    for tree in e.DEPENDENCY_TREES:
        put(run('ip') / tree / 'include.vh', '`define SYNTHETIC_TEST 1\n')
    for stage in ('ip', 'pcis', 'clocks'):
        put(run('ip') / stage / 'ip_done.txt', 'IP_DONE\n')
        put(run('ip') / stage / 'exit.json', {'returncode': 0})
        put(run('ip') / stage / 'console.log', 'SYNTHETIC log\n')
        put(run('ip') / stage / 'command.json', ['vivado', 'synthetic-test'])
    products = {name: row for name, row in inventory(run('ip')).items()
                if any(name.startswith(s + '/ip_project/') for s in ('ip', 'pcis', 'clocks'))
                and Path(name).suffix.lower() in b.PRODUCT_SUFFIXES}
    put(run('ip') / 'ip-complete.json', {'schema': 1, 'kind': 'c1-alex-private-ip',
        'qualification': 'not-qualified', 'inputs_sha256': e.sha(run('ip') / 'inputs.json'), 'products': products})
    ip_sha = e.sha(run('ip') / 'ip-complete.json')
    put(run('synth') / 'ip-binding.json', {'run': old('ip'), 'receipt_sha256': ip_sha})
    put(run('synth') / 'synth/post_synth.dcp', 'SYNTHETIC SYNTHESIS DCP FOR PARSER TEST ONLY')
    put(run('synth') / 'synth/exit.json', {'returncode': 0})
    put(run('synth') / 'synth/synth_done.txt', 'C1_ALEX_SYNTH_ONLY_COMPLETE\n')
    put(run('synth') / 'synth/command.json', ['vivado', 'synthetic-test'])
    put(run('synth') / 'synth/console.log', 'SYNTHETIC synthesis log\n')
    reports = set(e.FINAL_REPORTS) - {'route_status.rpt', 'setup.rpt', 'hold.rpt', 'supplemental.tcldata'} | {'REQUIRED_NEXT_STEPS.txt'}
    for name in reports: put(run('synth') / 'synth/synth_audit' / name, 'SYNTHETIC report\n')
    put(run('synth') / 'synth-complete.json', {'schema': 1, 'kind': 'c1-alex-synthesis-only',
        'qualification': 'not-qualified', 'inputs_sha256': e.sha(run('synth') / 'inputs.json'),
        'checkpoint_sha256': e.sha(run('synth') / 'synth/post_synth.dcp'), 'ip_receipt_sha256': ip_sha})
    # Four actual-parser scoreboard fixtures with copied reviewed simulator files.
    shutil.copytree(run('ip'), run('simulation_private'))
    si = {'schema': 1, 'kind': 'c1-alex-vendor-fabric-simulation', 'ip_receipt_sha256': ip_sha,
          'build_inputs_sha256': e.sha(run('ip') / 'inputs.json'), 'ip_source_revision': 'a'*40,
          'ip_products': products, 'simulation_ip_files': {name: row for name, row in inventory(run('simulation_private')).items()
              if any(name.startswith(stage + '/ip_project/') for stage in ('ip','pcis','clocks'))},
          'clocks_mhz': {'core': 200, 'host': 250, 'hbm_nominal': 450}, 'hbm_simulated_period_ps': 2222,
          'hbm_model': 'functional-not-silicon-timing', 'hardware_actions': False,
          'qualification': 'protocol-simulation-only-not-physical-timing', 'scenarios': [list(p) for p in sim.SCENARIOS],
          'runner_sha256': e.sha(e.HERE / 'simulate.py'), 'builder_sha256': e.sha(e.HERE / 'build.py'),
          'sources': {name: e.sha(path) for name, path in sim.SOURCES.items()}}
    source_copy(root, 'simulation', ['boards/c1_f2_alex/simulate.py', 'boards/c1_f2_alex/build.py',
                                    *[p.relative_to(ROOT).as_posix() for p in sim.SOURCES.values()]])
    put(run('simulation') / 'inputs.json', si)
    put(run('simulation') / 'vivado_version.txt', 'Vivado v2025.2 SYNTHETIC TEST\n')
    evidence = {'vivado_version.txt': rec(run('simulation') / 'vivado_version.txt')}; results = []
    for seed, phase in sim.SCENARIOS:
        prefix = f'seed-{seed}-phase-{phase}'
        dest = run('simulation') / prefix
        for name, path in sim.SOURCES.items(): put(dest / 'rtl' / name, path.read_bytes())
        put(dest / 'simulate.tcl', sim.simulation_tcl(Path(old('simulation', prefix)), Path(old('simulation_private')), seed, phase))
        put(dest / 'inputs.json', inventory(dest))
        put(dest / 'command.json', ['vivado', 'synthetic-test'])
        put(dest / 'console.log', pass_log(seed, phase))
        put(dest / 'exit.json', {'returncode': 0})
        result = sim.parse_result(pass_log(seed, phase), seed, phase); results.append(result)
        put(dest / 'result.json', result)
        evidence.update({prefix + '/' + name: row for name, row in inventory(dest).items()})
    put(run('simulation') / 'simulation-complete.json', {'schema': 1, 'kind': 'c1-alex-vendor-fabric-simulation',
        'state': 'passed', 'inputs_sha256': e.sha(run('simulation') / 'inputs.json'), 'results': results,
        'evidence': evidence, 'qualification': 'protocol-simulation-only-not-physical-timing', 'hardware_actions': False})
    prev, checkpoint = 'synth', 'synth/post_synth.dcp'
    for stage in ('link', 'implement', 'refine', 'qualify'):
        dest = run(stage)
        names = {'boards/c1_f2_alex/implement.py', 'boards/c1_f2_alex/build.py'}
        prepared = {'run.tcl'}
        if stage in ('link', 'qualify'):
            names |= {'boards/c1_f2_alex/qualify_c1_alex.tcl', 'boards/c1_f2_alex/audit_c1_alex.tcl'}
            prepared |= {'qualify_c1_alex.tcl', 'audit_c1_alex.tcl'}
        if stage == 'qualify':
            names |= {'boards/c1_f2_alex/simulate.py','boards/c1_f2_alex/routed_host_ready.tcl'}
            prepared.add('routed_host_ready.tcl')
        if stage == 'implement': prepared |= {'floorplan.tcl','core_timing.tcl','platform_constraints.tcl'}
        sources = source_copy(root, stage, names)
        for name in prepared:
            source = e.HERE / name
            put(dest / name, source.read_bytes() if name in ('qualify_c1_alex.tcl','audit_c1_alex.tcl','routed_host_ready.tcl') else '# SYNTHETIC stage recipe\n')
        if stage == 'implement':
            for name in prepared:
                original = 'build.tcl' if name == 'run.tcl' else name
                put(dest / name, (run('synth') / 'implement' / original).read_bytes())
        inp = {'schema': 1, 'kind': 'c1-alex-checkpoint-stage-inputs', 'stage': stage,
               'executor_revision': 'd'*40, 'executor_sources': sources, **m['hardware'],
               'synth_run': old('synth'), 'synth_receipt_sha256': e.sha(run('synth')/'synth-complete.json'),
               'synth_inputs_sha256': e.sha(run('synth')/'inputs.json'), 'ip_run': old('ip'), 'ip_receipt_sha256': ip_sha,
               'hdk': old('installed'), 'shell_sha256': b.SHELL_SHA256,
               'source_checkpoint': {'path': old(prev, checkpoint), **rec(run(prev)/checkpoint)},
               'previous_run': None if stage=='link' else old(prev),
               'previous_receipt_sha256': None if stage=='link' else e.sha(run(prev)/'stage-result.json'),
               'max_threads':4, 'resynthesis':False, 'hardware_actions':False, 'prepared_files':inventory(dest)}
        if stage == 'qualify': inp['simulation'] = {'run':old('simulation'),
            'receipt_sha256':e.sha(run('simulation')/'simulation-complete.json'), 'inputs_sha256':e.sha(run('simulation')/'inputs.json')}
        put(dest / 'stage-inputs.json', inp)
        put(dest / 'command.json', ['vivado','synthetic-test']);put(dest/'console.log','SYNTHETIC stage log\n')
        receipt = {'schema':1,'kind':'c1-alex-checkpoint-stage','stage':stage,'inputs_sha256':e.sha(dest/'stage-inputs.json'),
            'source_checkpoint_sha256':e.sha(run(prev)/checkpoint),'returncode':0,'hardware_actions':False,
            'status':'pending-review' if stage=='qualify' else 'candidate' if stage=='implement' else 'complete','route_qualified':False}
        if stage == 'link':
            put(dest/'audit-result.txt','structural_pass=1\nroute_qualified=0\n')
            put(dest/'link_done.txt','C1_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED\n')
            put(dest/'linked_audit/supplemental.tcldata',
                'schema 1 kind SYNTHETIC_parser_test_only route_qualified 0 review pending\n')
        if stage in ('implement','refine'):
            receipt['timing']={'setup_slack_ns':-4.024 if stage=='implement' else .1,'hold_slack_ns':.02}
            put(dest/'implementation_result.txt','\n'.join(f'{key}={value}' for key,value in receipt['timing'].items())+'\n')
        if stage == 'qualify':
            final=e.sha(run(prev)/checkpoint)
            receipt.update(checkpoint=old(prev,checkpoint),checkpoint_sha256=final)
            put(dest/'audit-result.txt',f'route_qualified=0\ncheckpoint_sha256={final}\n')
            for name in e.FINAL_REPORTS:put(dest/'qualification'/name,'SYNTHETIC report FOR PARSER TEST ONLY\n')
            put(dest/'qualification/NOT_QUALIFIED.txt','SYNTHETIC pending review\n')
            put(dest/'qualification/summary.tcldata',f'schema 1 stage final_route route_qualified 0 hardware_measured 0 checkpoint {old(prev,checkpoint)} checkpoint_sha256 {final} blockers {{}} pending_reviews {{SYNTHETIC_pending_review}}\n')
        else:
            outname='post_link.dcp' if stage=='link' else 'post_route.dcp'
            put(dest/outname,'SYNTHETIC '+stage+' DCP FOR PARSER TEST ONLY')
            receipt.update(checkpoint=outname,checkpoint_sha256=e.sha(dest/outname))
            prev,checkpoint=stage,outname
        receipt['outputs']={name:row for name,row in inventory(dest).items() if name!='stage-inputs.json'}
        put(dest/'stage-result.json',receipt)
    return root,m,refresh_manifest(root,m)

@pytest.fixture
def bundle(tmp_path,monkeypatch): return fixture_bundle(tmp_path,monkeypatch)


def repin_final(root,m):
    p=root/'runs/qualify/stage-result.json';r=read(p)
    r['outputs']={name:row for name,row in inventory(p.parent).items() if name not in ('stage-inputs.json','stage-result.json')}
    put(p,r);return refresh_manifest(root,m)


def test_relocation_replays_c1_receipts_and_keeps_pending(bundle,tmp_path,monkeypatch):
    root,m,trusted=bundle;out=tmp_path/'relocated';shutil.move(root,out)
    def forbidden(*a,**k):raise AssertionError('Offline replay must not execute tools or bundled code')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'check_output',forbidden)
    result=e.verify_bundle(out,trusted)
    assert result['route_qualified'] is False and result['review_required'] is True
    assert result['hardware_measured'] is False
    assert result['pending_reviews']==['SYNTHETIC_pending_review']
    assert result['generated_dependency_capture']=='collection_time_not_pre_synthesis_attestation'


@pytest.mark.parametrize('change',['approve_receipt','approve_summary','remove_pending','remove_marker','c2_kind','wrong_checkpoint','missing_report','changed_qualifier','bad_ancestry'])
def test_c1_raw_candidate_rejects_false_approval_or_broken_chain(bundle,change):
    root,m,_=bundle;p=root/'runs/qualify/stage-result.json';r=read(p)
    if change=='approve_receipt':r.update(status='complete',route_qualified=True);put(p,r)
    elif change=='approve_summary':
        p=root/'runs/qualify/qualification/summary.tcldata';put(p,p.read_text().replace('route_qualified 0','route_qualified 1'))
    elif change=='remove_pending':
        p=root/'runs/qualify/qualification/summary.tcldata';put(p,p.read_text().replace('SYNTHETIC_pending_review',''))
    elif change=='remove_marker':(root/'runs/qualify/qualification/NOT_QUALIFIED.txt').unlink()
    elif change=='c2_kind':r['kind']='c2-alex-checkpoint-stage';put(p,r)
    elif change=='wrong_checkpoint':r['checkpoint_sha256']='0'*64;put(p,r)
    elif change=='missing_report':(root/'runs/qualify/qualification/cdc.rpt').unlink()
    elif change=='changed_qualifier':
        put(root/'runs/qualify/qualify_c1_alex.tcl','route_qualified 1')
    else:
        p=root/'runs/qualify/stage-inputs.json';v=read(p);v['previous_receipt_sha256']='0'*64;put(p,v)
        p=root/'runs/qualify/stage-result.json';r=read(p);r['inputs_sha256']=e.sha(root/'runs/qualify/stage-inputs.json');put(p,r)
    with pytest.raises(ValueError):e.verify_bundle(root,repin_final(root,m))


def test_physical_blocker_is_preserved_as_unqualified_data(bundle):
    root,m,_=bundle;p=root/'runs/qualify/qualification/summary.tcldata'
    put(p,p.read_text().replace('blockers {}','blockers {negative_setup}'))
    result=e.verify_bundle(root,repin_final(root,m))
    assert result['blockers']==['negative_setup'] and result['route_qualified'] is False


@pytest.mark.parametrize('name',['runs/ip/ip/ip_project/child.dcp','runs/refine/post_route.dcp','runs/qualify/qualification/cdc.rpt','sources/hardware/rtl/c1/otpu_c1_native.sv','runs/simulation/seed-17-phase-0/console.log'])
def test_receipt_bound_data_tamper_rejected_after_outer_rehash(bundle,name):
    root,m,_=bundle;put(root/name,'CORRUPTED')
    with pytest.raises((ValueError,RuntimeError)):e.verify_bundle(root,refresh_manifest(root,m))


@pytest.mark.parametrize('change',['missing','extra','symlink','wrong_manifest','wrong_instance','schema2'])
def test_closed_transport_inventory(bundle,change):
    root,m,trusted=bundle;p=root/'runs/qualify/qualification/cdc.rpt'
    if change=='missing':p.unlink()
    elif change=='extra':put(root/'extra.txt','unbound')
    elif change=='symlink':p.unlink();p.symlink_to(root/'runs/qualify/qualification/drc.rpt')
    elif change=='wrong_manifest':trusted='0'*64
    else:
        if change=='wrong_instance':m['instance_id']='i-0c78de2253f66519d'
        else:m['schema']=2
        put(root/e.MANIFEST,m);trusted=e.sha(root/e.MANIFEST)
    with pytest.raises(ValueError):e.verify_bundle(root,trusted)


def test_candidate_archive_never_grants_qualification(bundle,tmp_path):
    root,m,trusted=bundle;path=tmp_path/'candidate.tar'
    with tarfile.open(path,'w') as archive:
        for item in sorted(root.rglob('*')):
            if item.is_file():archive.add(item,arcname=item.relative_to(root).as_posix(),recursive=False)
    assert e.inspect_archive(path,trusted)['route_qualified'] is False
