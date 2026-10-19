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

e = load('portable_evidence', ROOT / 'boards/c2_f2_alex/evidence.py')
b = e.local_module('build')
sim = e.local_module('simulate')
f = load('portable_build_fixture', ROOT / 'tests/test_c2_alex_build.py')


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
        lines.append(f'C2_PC pc={pc} accepted={n} returned={n} consumed={n} axi_beats={n} peak=32 hbm_reads=2000 hbm_writes=1000')
    return '\n'.join(lines) + f'\nC2_ALEX_FABRIC_PASS seed={seed} phase={phase} pcs=8 reset_pending=8 host_b=994 host_r=1229\n'


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
    m = {'schema': 1, 'kind': 'c2-alex-portable-evidence', 'instance_id': e.INSTANCE,
         'hardware': {'source_revision': e.DEPENDENCY_SOURCE, 'build_id': e.DEPENDENCY_SOURCE[:8]},
         'roots': roots, 'receipts': {}, 'source_snapshots': {}, 'files': {}}
    for role in e.SOURCE_ROLES:
        m['source_snapshots'][role] = {'revision': e.DEPENDENCY_SOURCE if role in ('ip', 'hardware') else 'd' * 40,
                                      'bundle_relative_path': 'sources/' + role}
    shell = 'hdk/common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp'
    put(run('installed') / shell, 'SYNTHETIC SHELL DCP FOR PARSER TEST ONLY')
    monkeypatch.setattr(b, 'SHELL_SHA256', e.sha(run('installed') / shell))
    monkeypatch.setattr(e, 'local_module', lambda name: {'build': b, 'simulate': sim}[name])
    base_inputs = dict(schema=1, kind='c2-alex-burst200-build-inputs', source_status='',
                       source_revision=e.DEPENDENCY_SOURCE, build_id=e.DEPENDENCY_SOURCE[:8], part=b.PART,
                       core_khz=200000, hbm_mhz=450, hbm_ports=8, channel_bytes=2**29,
                       alex_revision=b.ALEX_REVISION, archive_sha256=b.ARCHIVE_SHA256,
                       hdk_revision=b.HDK_REVISION, ip_revision=b.IP_REVISION,
                       support_manifest_sha256=e.sha(b.SUPPORT / 'manifest.json'),
                       hdk=old('installed'), installed_inputs=inventory(run('installed')))
    support_names = ['boards/c2_f2_alex/support/' + name for name in inventory(b.SUPPORT)]
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
    put(run('ip') / 'ip-complete.json', {'schema': 1, 'kind': 'c2-alex-private-ip',
        'qualification': 'not-qualified', 'inputs_sha256': e.sha(run('ip') / 'inputs.json'), 'products': products})
    ip_sha = e.sha(run('ip') / 'ip-complete.json')
    put(run('synth') / 'ip-binding.json', {'run': old('ip'), 'receipt_sha256': ip_sha})
    put(run('synth') / 'synth/post_synth.dcp', 'SYNTHETIC SYNTHESIS DCP FOR PARSER TEST ONLY')
    put(run('synth') / 'synth/exit.json', {'returncode': 0})
    put(run('synth') / 'synth/synth_done.txt', 'C2_ALEX_SYNTH_ONLY_COMPLETE\n')
    put(run('synth') / 'synth/command.json', ['vivado', 'synthetic-test'])
    put(run('synth') / 'synth/console.log', 'SYNTHETIC synthesis log\n')
    reports = set(e.FINAL_REPORTS) - {'route_status.rpt', 'setup.rpt', 'hold.rpt', 'supplemental.tcldata'} | {'REQUIRED_NEXT_STEPS.txt'}
    for name in reports: put(run('synth') / 'synth/synth_audit' / name, 'SYNTHETIC report\n')
    put(run('synth') / 'synth-complete.json', {'schema': 1, 'kind': 'c2-alex-synthesis-only',
        'qualification': 'not-qualified', 'inputs_sha256': e.sha(run('synth') / 'inputs.json'),
        'checkpoint_sha256': e.sha(run('synth') / 'synth/post_synth.dcp'), 'ip_receipt_sha256': ip_sha})
    # Preserve the distinct supplemental observer proof, including all .vh files.
    dep = run('dependencies')
    # The reviewed real observer is deliberately an ignored, run-specific file;
    # fixtures substitute a parser-only pin, so tests need no ignored artifacts.
    observer = b'# SYNTHETIC observer bytes for parser fixture only\n'
    monkeypatch.setattr(e, 'OBSERVER_SHA256', hashlib.sha256(observer).hexdigest())
    put(dep / 'attest_ip_dependencies.py', observer)
    unit = {'Id': 'otpu-c2-alex-264d8413-ip-synth.service', 'InvocationID': 'c' * 32,
            'Result': 'success', 'ActiveState': 'active'}
    identity = {'instance_id': e.INSTANCE, 'expected_source_revision': e.DEPENDENCY_SOURCE,
                'observer_sha256': e.OBSERVER_SHA256, 'unit': unit['Id'], 'invocation_id': unit['InvocationID'],
                'initial_unit': unit, 'ip_run': old('ip'), 'synth_run': old('synth'), 'started_utc': '2026-10-19T18:00:00+00:00'}
    put(dep / 'attestation/identity.json', identity)
    files = {name: row for name, row in inventory(run('ip')).items()
             if any(name.startswith(tree + '/') for tree in e.DEPENDENCY_TREES)}
    before = {'schema': 1, 'kind': 'c2-alex-ip-dependency-before-synthesis',
              'identity_sha256': e.sha(dep / 'attestation/identity.json'), 'inputs': rec(run('ip') / 'inputs.json'),
              'unit': unit, 'files': files, 'started_utc': '2026-10-19T18:00:01+00:00', 'finished_utc': '2026-10-19T18:00:02+00:00'}
    put(dep / 'attestation/before.json', before)
    complete = {'schema': 1, 'kind': 'c2-alex-supplemental-ip-dependency-attestation', 'state': 'verified',
                'source_revision': e.DEPENDENCY_SOURCE, 'instance_id': e.INSTANCE, 'unit': unit['Id'],
                'invocation_id': unit['InvocationID'], 'final_unit': unit, 'ip_run': old('ip'), 'synth_run': old('synth'),
                'identity_sha256': e.sha(dep / 'attestation/identity.json'), 'before_sha256': e.sha(dep / 'attestation/before.json'),
                'files': files, 'finished_utc': '2026-10-19T18:00:04+00:00', 'hardware_actions': False,
                'qualification': 'source-provenance-only-not-timing-or-hardware'}
    for key, role, name in [('ip_inputs','ip','inputs.json'),('ip_receipt','ip','ip-complete.json'),
        ('synth_inputs','synth','inputs.json'),('synth_receipt','synth','synth-complete.json'),
        ('ip_binding','synth','ip-binding.json'),('synth_command','synth','synth/command.json'),
        ('synth_exit','synth','synth/exit.json'),('synth_marker','synth','synth/synth_done.txt'),
        ('synth_checkpoint','synth','synth/post_synth.dcp')]:
        complete[key] = rec(run(role) / name)
    put(dep / 'attestation/complete.json', complete)
    # Four actual-parser scoreboard fixtures with copied reviewed simulator files.
    shutil.copytree(run('ip'), run('simulation_private'))
    si = {'schema': 1, 'kind': 'c2-alex-vendor-fabric-simulation', 'ip_receipt_sha256': ip_sha,
          'build_inputs_sha256': e.sha(run('ip') / 'inputs.json'), 'ip_source_revision': e.DEPENDENCY_SOURCE,
          'ip_products': products, 'simulation_ip_files': {name: row for name, row in inventory(run('simulation_private')).items()
              if any(name.startswith(stage + '/ip_project/') for stage in ('ip','pcis','clocks'))},
          'clocks_mhz': {'core': 200, 'host': 250, 'hbm_nominal': 450}, 'hbm_simulated_period_ps': 2222,
          'hbm_model': 'functional-not-silicon-timing', 'hardware_actions': False,
          'qualification': 'protocol-simulation-only-not-physical-timing', 'scenarios': [list(p) for p in sim.SCENARIOS],
          'runner_sha256': e.sha(e.HERE / 'simulate.py'), 'builder_sha256': e.sha(e.HERE / 'build.py'),
          'sources': {name: e.sha(path) for name, path in sim.SOURCES.items()}}
    source_copy(root, 'simulation', ['boards/c2_f2_alex/simulate.py', 'boards/c2_f2_alex/build.py',
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
    put(run('simulation') / 'simulation-complete.json', {'schema': 1, 'kind': 'c2-alex-vendor-fabric-simulation',
        'state': 'passed', 'inputs_sha256': e.sha(run('simulation') / 'inputs.json'), 'results': results,
        'evidence': evidence, 'qualification': 'protocol-simulation-only-not-physical-timing', 'hardware_actions': False})
    prev, checkpoint = 'synth', 'synth/post_synth.dcp'
    for stage in ('link', 'implement', 'refine', 'qualify'):
        dest = run(stage)
        names = {'boards/c2_f2_alex/implement.py', 'boards/c2_f2_alex/build.py'}
        prepared = {'run.tcl'}
        if stage in ('link', 'qualify'):
            names |= {'boards/c2_f2_alex/qualify_c2_alex.tcl', 'boards/c2_f2_alex/audit_c2_alex.tcl'}
            prepared |= {'qualify_c2_alex.tcl', 'audit_c2_alex.tcl'}
        if stage == 'qualify': names.add('boards/c2_f2_alex/simulate.py')
        if stage == 'implement': prepared |= {'floorplan.tcl','core_timing.tcl','platform_constraints.tcl'}
        sources = source_copy(root, stage, names)
        for name in prepared:
            source = e.HERE / name
            put(dest / name, source.read_bytes() if name in ('qualify_c2_alex.tcl','audit_c2_alex.tcl') else '# SYNTHETIC stage recipe\n')
        inp = {'schema': 1, 'kind': 'c2-alex-checkpoint-stage-inputs', 'stage': stage,
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
        receipt = {'schema':1,'kind':'c2-alex-checkpoint-stage','stage':stage,'inputs_sha256':e.sha(dest/'stage-inputs.json'),
            'source_checkpoint_sha256':e.sha(run(prev)/checkpoint),'returncode':0,'hardware_actions':False,
            'status':'candidate' if stage=='implement' else 'complete','route_qualified':stage=='qualify'}
        if stage == 'link':
            put(dest/'audit-result.txt','structural_pass=1\nroute_qualified=0\n')
            put(dest/'link_done.txt','C2_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED\n')
            put(dest/'linked_audit/supplemental.tcldata',
                'schema 1 kind SYNTHETIC_parser_test_only route_qualified 0 review pending\n')
        if stage in ('implement','refine'):
            receipt['timing']={'setup_slack_ns':-4.024 if stage=='implement' else .1,'hold_slack_ns':.02}
            put(dest/'implementation_result.txt','\n'.join(f'{key}={value}' for key,value in receipt['timing'].items())+'\n')
        if stage == 'qualify':
            final=e.sha(run(prev)/checkpoint)
            receipt.update(checkpoint=old(prev,checkpoint),checkpoint_sha256=final)
            put(dest/'audit-result.txt',f'route_qualified=1\ncheckpoint_sha256={final}\n')
            for name in e.FINAL_REPORTS:put(dest/'qualification'/name,'SYNTHETIC report FOR PARSER TEST ONLY\n')
            put(dest/'qualification/summary.tcldata',f'schema 1 stage final_route route_qualified 1 hardware_measured 0 checkpoint {old(prev,checkpoint)} checkpoint_sha256 {final} blockers {{}} pending_reviews {{}}\n')
        else:
            outname='post_link.dcp' if stage=='link' else 'post_route.dcp'
            put(dest/outname,'SYNTHETIC '+stage+' DCP FOR PARSER TEST ONLY')
            receipt.update(checkpoint=outname,checkpoint_sha256=e.sha(dest/outname))
            prev,checkpoint=stage,outname
        receipt['outputs']={name:row for name,row in inventory(dest).items() if name!='stage-inputs.json'}
        put(dest/'stage-result.json',receipt)
    return root,m,refresh_manifest(root,m)


@pytest.fixture
def bundle(tmp_path, monkeypatch): return fixture_bundle(tmp_path,monkeypatch)


def test_relocated_graph_replays_without_remote_paths_or_execution(bundle, tmp_path, monkeypatch):
    root,m,digest=bundle
    target=tmp_path/'a completely different path';shutil.move(root,target)
    def forbidden(*a,**k):raise AssertionError('offline evidence must not execute commands')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'check_output',forbidden)
    result=e.verify_bundle(target,digest)
    assert result['route_qualified'] is True and result['hardware_measured'] is False
    assert result['source_revision']==e.DEPENDENCY_SOURCE
    assert m['source_snapshots']['qualify']['revision']!=result['source_revision']
    assert not Path('/unavailable/original').exists()


@pytest.mark.parametrize('name',['runs/ip/ip/ip_project/child.dcp','runs/refine/post_route.dcp',
 'runs/qualify/qualification/cdc.rpt','sources/hardware/rtl/mxu/otpu_c2_mxu.sv',
 'runs/ip/ip/ip_project/f2_ip.srcs/include.vh','runs/simulation/seed-17-phase-0/console.log'])
def test_corrupt_bytes_rejected_even_outer_rehashed(bundle,name):
    root,m,_=bundle;put(root/name,'CORRUPTED')
    with pytest.raises((ValueError,RuntimeError)):e.verify_bundle(root,refresh_manifest(root,m))


@pytest.mark.parametrize('change',['missing','extra','symlink','directory_symlink','extra_directory','wrong_manifest'])
def test_collection_inventory_closed(bundle,tmp_path,change):
    root,m,digest=bundle
    path=root/'runs/qualify/qualification/cdc.rpt'
    if change=='missing':path.unlink()
    elif change=='extra':put(root/'runs/qualify/unlisted.txt','extra')
    elif change=='extra_directory':(root/'unexpected-empty-directory').mkdir()
    elif change=='symlink':
        path.unlink();path.symlink_to(root/'runs/qualify/qualification/drc.rpt')
    elif change=='directory_symlink':
        target=tmp_path/'external';shutil.move(root/'runs/qualify/qualification',target)
        (root/'runs/qualify/qualification').symlink_to(target,target_is_directory=True)
    else:digest='0'*64
    with pytest.raises(ValueError):e.verify_bundle(root,digest)


def update_last_receipt(root,m):
    path=root/'runs/qualify/stage-result.json';receipt=read(path)
    receipt['inputs_sha256']=e.sha(root/'runs/qualify/stage-inputs.json')
    receipt['outputs']={name:row for name,row in inventory(root/'runs/qualify').items() if name not in ('stage-inputs.json','stage-result.json')}
    put(path,receipt)
    return refresh_manifest(root,m)


@pytest.mark.parametrize('change',['pending_receipt','raw_pending','raw_dcp','summary_pending','summary_dcp','explicit_pending',
 'missing_report','missing_supplemental','missing_executor','changed_executor','missing_prepared','wrong_ancestry','missing_simulation','wrong_simulation'])
def test_final_qualification_semantics_reject_self_consistent_outer(bundle,change):
    root,m,_=bundle;run=root/'runs/qualify';p=run/'stage-result.json';r=read(p)
    inp=read(run/'stage-inputs.json')
    if change=='pending_receipt':r.update(status='pending-review',route_qualified=False)
    elif change=='raw_pending':put(run/'audit-result.txt',(run/'audit-result.txt').read_text().replace('route_qualified=1','route_qualified=0'))
    elif change=='raw_dcp':put(run/'audit-result.txt','route_qualified=1\ncheckpoint_sha256='+'0'*64+'\n')
    elif change.startswith('summary_'):
        p2=run/'qualification/summary.tcldata';text=p2.read_text()
        put(p2,text.replace('pending_reviews {}','pending_reviews {vendor_CDC}') if change=='summary_pending' else text.replace(r['checkpoint_sha256'],'0'*64))
    elif change=='explicit_pending':put(run/'qualification/NOT_QUALIFIED.txt','review required')
    elif change=='missing_report':(run/'qualification/bus_skew.rpt').unlink()
    elif change=='missing_supplemental':(run/'qualification/supplemental.tcldata').unlink()
    elif change=='missing_executor':inp['executor_sources'].pop('boards/c2_f2_alex/qualify_c2_alex.tcl')
    elif change=='changed_executor':
        name='boards/c2_f2_alex/qualify_c2_alex.tcl';put(root/'sources/qualify'/name,'route_qualified 1')
        inp['executor_sources'][name]=rec(root/'sources/qualify'/name)
    elif change=='missing_prepared':inp['prepared_files'].pop('qualify_c2_alex.tcl')
    elif change=='wrong_ancestry':inp['source_checkpoint']['path']='/other/checkpoint.dcp'
    elif change=='missing_simulation':inp.pop('simulation')
    else:inp['simulation']['receipt_sha256']='0'*64
    put(p,r);put(run/'stage-inputs.json',inp)
    with pytest.raises((ValueError,KeyError)):e.verify_bundle(root,update_last_receipt(root,m))


def test_linked_supplemental_is_bound_by_stage_output_inventory(bundle):
    root,m,_=bundle
    put(root/'runs/link/linked_audit/supplemental.tcldata','SYNTHETIC changed report\n')
    # Rehash the outer collection, but preserve the actual stage receipt. The
    # new collect-only report must be covered just like every other output.
    with pytest.raises(ValueError,match='Frozen inventory mismatch: link/linked_audit/supplemental.tcldata'):
        e.verify_bundle(root,refresh_manifest(root,m))


@pytest.mark.parametrize('change',['missing','failed','wrong_observer','changed_include','late','wrong_invocation','missing_before_sha'])
def test_dependency_proof_required(bundle,change):
    root,m,_=bundle;dep=root/'runs/dependencies';p=dep/'attestation/complete.json';r=read(p)
    if change=='missing':p.unlink();m['files'].pop(p.relative_to(root).as_posix());put(root/e.MANIFEST,m);digest=e.sha(root/e.MANIFEST)
    else:
        if change=='failed':put(dep/'attestation/failure.json',{'error':'late observer'})
        elif change=='wrong_observer':put(dep/'attest_ip_dependencies.py','unreviewed observer')
        elif change=='changed_include':r['files'].pop(next(name for name in r['files'] if name.endswith('.vh')))
        elif change=='wrong_invocation':r['final_unit']['InvocationID']='0'*32
        elif change=='missing_before_sha':r.pop('before_sha256')
        else:
            before=read(dep/'attestation/before.json');before['finished_utc']='2026-10-19T19:00:00+00:00'
            put(dep/'attestation/before.json',before);r['before_sha256']=e.sha(dep/'attestation/before.json')
        put(p,r);digest=refresh_manifest(root,m)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):e.verify_bundle(root,digest)


@pytest.mark.parametrize('bad',['../escape','/absolute','a//b','a/./b','a/../b','a\\b','a\x00b','a/.git/config'])
def test_unsafe_relative_paths(bad):
    with pytest.raises(ValueError):e.relative(bad)


def test_duplicate_json_keys_and_no_tcl_execution():
    with pytest.raises(ValueError,match='Duplicate'):e.decode('{"route_qualified":false,"route_qualified":true}')
    assert e.tcl_dict('x {[exec touch /never]} pending {}')=={'x':'[exec touch /never]','pending':''}
    with pytest.raises(ValueError):e.tcl_dict('x 1 x 2')


def make_tar(root,out,extra=None):
    with tarfile.open(out,'w') as tar:
        for path in sorted(root.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
        if extra:extra(tar)


def test_archive_preflight_is_not_route_acceptance(bundle,tmp_path):
    root,_,digest=bundle;archive=tmp_path/'bundle.tar';make_tar(root,archive)
    result=e.inspect_archive(archive,digest)
    assert result['route_qualified'] is False
    assert result['files']==len(inventory(root))-1
    assert not (tmp_path/'extracted').exists()


@pytest.mark.parametrize('change',['traversal','absolute','duplicate','symlink','hardlink','fifo','extra','empty_directory','sparse'])
def test_archive_unsafe_or_extra_members(bundle,tmp_path,change):
    root,_,digest=bundle;archive=tmp_path/'bad.tar'
    def extra(tar):
        info=tarfile.TarInfo('../escape' if change=='traversal' else '/escape' if change=='absolute' else e.MANIFEST if change=='duplicate' else 'unexpected')
        if change in ('symlink','hardlink'):info.type=tarfile.SYMTYPE if change=='symlink' else tarfile.LNKTYPE;info.linkname=e.MANIFEST
        elif change=='fifo':info.type=tarfile.FIFOTYPE
        elif change=='empty_directory':info.type=tarfile.DIRTYPE
        elif change=='sparse':info.pax_headers={'GNU.sparse.map':'0,0','GNU.sparse.size':'0'}
        tar.addfile(info,io.BytesIO(b''))
    make_tar(root,archive,extra)
    with pytest.raises((ValueError,tarfile.TarError)):e.inspect_archive(archive,digest)


@pytest.mark.parametrize('bound',['MAX_FILES','MAX_TOTAL','MAX_FILE','MAX_MANIFEST'])
def test_archive_resource_bounds(bundle,tmp_path,monkeypatch,bound):
    root,_,digest=bundle;archive=tmp_path/'bounded.tar';make_tar(root,archive)
    monkeypatch.setattr(e,bound,1)
    with pytest.raises(ValueError):e.inspect_archive(archive,digest)


def test_tar_metadata_allocation_is_bounded_before_read(tmp_path, monkeypatch):
    archive=tmp_path/'huge-metadata.tar'
    header=tarfile.TarInfo('pax');header.type=tarfile.XHDTYPE;header.size=1<<30
    archive.write_bytes(header.tobuf())
    with pytest.raises(ValueError,match='extended archive header'):e.inspect_archive(archive,'0'*64)


def test_concatenated_archive_cannot_hide_extra_members(bundle,tmp_path):
    root,_,digest=bundle;archive=tmp_path/'concatenated.tar';make_tar(root,archive)
    other=tmp_path/'extra.tar'
    with tarfile.open(other,'w') as tar:tar.addfile(tarfile.TarInfo('hidden'))
    with archive.open('ab') as stream:stream.write(other.read_bytes())
    with pytest.raises(ValueError,match='inventory'):e.inspect_archive(archive,digest)


def test_manifest_rejects_duplicate_keys_in_file_content(bundle):
    root,m,_=bundle
    path=root/'runs/qualify/stage-result.json';value=path.read_text()
    put(path,value.replace('"schema": 1','"schema": 1, "schema": 1'))
    with pytest.raises(ValueError,match='Duplicate'):e.verify_bundle(root,refresh_manifest(root,m))


@pytest.mark.parametrize('change',['roles','prefix','absolute','missing_inventory'])
def test_relocation_map_is_complete_and_unambiguous(bundle,change):
    root,m,_=bundle
    if change=='roles':m['roots'].pop('qualify')
    elif change=='prefix':m['roots']['qualify']['bundle_relative_path']='runs/refine'
    elif change=='absolute':m['roots']['qualify']['original_absolute_path']='/unavailable/original/refine'
    else:m['files'].pop('runs/qualify/qualification/cdc.rpt')
    put(root/e.MANIFEST,m)
    with pytest.raises(ValueError):e.verify_bundle(root,e.sha(root/e.MANIFEST))


def test_manifest_mutation_during_semantic_replay_is_rejected(bundle,monkeypatch):
    root,_,digest=bundle
    previous=e.local_module
    def load_then_mutate(name):
        if name=='build':
            path=root/e.MANIFEST;path.write_bytes(path.read_bytes()+b'\n')
        return previous(name)
    monkeypatch.setattr(e,'local_module',load_then_mutate)
    with pytest.raises(ValueError,match='Manifest changed while verifying'):e.verify_bundle(root,digest)


def synthetic_historical_link(root,monkeypatch):
    """Model an older LINK helper without treating current bytes as history."""
    name='boards/c2_f2_alex/qualify_c2_alex.tcl'
    assert e.REVIEWED_HISTORICAL['link',name]=={
        '6306dc7dc5f8c9112ce62b4140ccc0aa2846cd05fb624ef48b1af87140ab623c',
        'c072db502318cadcd4a38b47438fcaef69a5b91fe7090374a629558e502eef9d'}
    # The entire bundle is parser-only synthetic evidence. Give its LINK
    # bytes a test-local historical role; never widen the production pins or
    # add a QUALIFY exception. This works after the current helper changes.
    historical=dict(e.REVIEWED_HISTORICAL)
    historical['link',name]={rec(root/'sources/link'/name)['sha256']}
    monkeypatch.setattr(e,'REVIEWED_HISTORICAL',historical)
    return name


def test_pinned_historical_link_survives_new_final_helper(bundle,monkeypatch):
    root,m,_=bundle
    name=synthetic_historical_link(root,monkeypatch)
    replacement=b'# SYNTHETIC later reviewed final helper; parser fixture only\n'
    new_sha=hashlib.sha256(replacement).hexdigest()
    put(root/'sources/qualify'/name,replacement)
    put(root/'runs/qualify/qualify_c2_alex.tcl',replacement)
    inp=read(root/'runs/qualify/stage-inputs.json')
    inp['executor_sources'][name]=rec(root/'sources/qualify'/name)
    inp['prepared_files']['qualify_c2_alex.tcl']=rec(root/'runs/qualify/qualify_c2_alex.tcl')
    put(root/'runs/qualify/stage-inputs.json',inp)
    trusted=update_last_receipt(root,m)
    original_sha=e.sha
    monkeypatch.setattr(e,'sha',lambda p:new_sha if Path(p)==ROOT/name else original_sha(p))
    assert e.verify_bundle(root,trusted)['route_qualified'] is True
    # The exception is exact and LINK-only: changing either historical byte
    # stream, or presenting it as the current final helper, still fails.
    monkeypatch.setattr(e,'REVIEWED_HISTORICAL',{})
    with pytest.raises(ValueError,match='reviewed local validator: link'):e.verify_bundle(root,trusted)


def test_unreviewed_historical_link_is_not_accepted(bundle):
    root,m,_=bundle;name='boards/c2_f2_alex/qualify_c2_alex.tcl'
    put(root/'sources/link'/name,'changed historical helper')
    put(root/'runs/link/qualify_c2_alex.tcl','changed historical helper')
    inp=read(root/'runs/link/stage-inputs.json')
    inp['executor_sources'][name]=rec(root/'sources/link'/name)
    inp['prepared_files']['qualify_c2_alex.tcl']=rec(root/'runs/link/qualify_c2_alex.tcl')
    put(root/'runs/link/stage-inputs.json',inp)
    receipt=read(root/'runs/link/stage-result.json')
    receipt['inputs_sha256']=e.sha(root/'runs/link/stage-inputs.json')
    receipt['outputs']={name:row for name,row in inventory(root/'runs/link').items() if name not in ('stage-inputs.json','stage-result.json')}
    put(root/'runs/link/stage-result.json',receipt)
    with pytest.raises(ValueError,match='reviewed local validator: link'):e.verify_bundle(root,refresh_manifest(root,m))


def test_historical_link_pin_never_qualifies_final_helper(bundle,monkeypatch):
    root,_,trusted=bundle;name=synthetic_historical_link(root,monkeypatch)
    assert rec(root/'sources/link'/name)==rec(root/'sources/qualify'/name)
    original=e.sha
    monkeypatch.setattr(e,'sha',lambda p:'0'*64 if Path(p)==ROOT/name else original(p))
    with pytest.raises(ValueError,match='reviewed local validator: qualify'):e.verify_bundle(root,trusted)


def synchronous_fixture(root,m,monkeypatch):
    """Convert parser-only fixture to new synth + preserved older private IP."""
    revision='b'*40
    old_synth=m['roots']['synth']['original_absolute_path']
    new_synth=m['roots']['dependencies']['original_absolute_path']+'/synthesis'
    m['roots']['synth']['original_absolute_path']=new_synth
    m['hardware']={'source_revision':revision,'build_id':revision[:8]}
    m['source_snapshots']['hardware']['revision']=revision
    syn=root/'runs/synth'
    inp=read(syn/'inputs.json');inp.update(m['hardware']);put(syn/'inputs.json',inp)
    receipt=read(syn/'synth-complete.json');receipt['inputs_sha256']=e.sha(syn/'inputs.json');put(syn/'synth-complete.json',receipt)
    dep=root/'runs/dependencies';(dep/'attest_ip_dependencies.py').unlink()
    wrapper=b'# SYNTHETIC synchronous wrapper for parser test only\n'
    monkeypatch.setattr(e,'SYNCHRONOUS_WRAPPER_SHA256',hashlib.sha256(wrapper).hexdigest())
    monkeypatch.setattr(e,'REUSED_IP_RECEIPT_SHA256',e.sha(root/'runs/ip/ip-complete.json'))
    put(dep/'run_synth_with_attestation.py',wrapper)
    unit={'Id':f'otpu-c2-alex-{revision[:8]}-synth-retry.service','InvocationID':'c'*32,'Result':'success','ActiveState':'active'}
    identity=read(dep/'attestation/identity.json')
    identity.update(schema=2,kind='c2-alex-synchronous-synthesis-identity',unit=unit['Id'],initial_unit=unit,
        expected_source_revision=revision,ip_source_revision=e.DEPENDENCY_SOURCE,synth_run=new_synth,
        wrapper_sha256=e.SYNCHRONOUS_WRAPPER_SHA256,source='/approved/retry-source')
    identity.pop('observer_sha256');put(dep/'attestation/identity.json',identity)
    before=read(dep/'attestation/before.json')
    before.update(schema=2,kind='c2-alex-ip-dependency-before-synchronous-synthesis',
        identity_sha256=e.sha(dep/'attestation/identity.json'),source_inputs=inp['source_inputs'],
        unit=unit,ip_receipt=rec(root/'runs/ip/ip-complete.json'))
    put(dep/'attestation/before.json',before)
    put(dep/'attestation/command.json',['/approved/python',identity['source']+'/boards/c2_f2_alex/build.py',
        '--hdk',m['roots']['installed']['original_absolute_path'],'--out',new_synth,'--stage','synth',
        '--ip-run',m['roots']['ip']['original_absolute_path'],'--ip-jobs','1'])
    put(dep/'attestation/exit.json',{'returncode':0})
    complete=read(dep/'attestation/complete.json')
    complete.update(schema=2,kind='c2-alex-synchronous-ip-dependency-attestation',source_revision=revision,build_id=revision[:8],
        ip_source_revision=e.DEPENDENCY_SOURCE,unit=unit['Id'],final_unit=unit,synth_run=new_synth,
        identity_sha256=e.sha(dep/'attestation/identity.json'),before_sha256=e.sha(dep/'attestation/before.json'),
        source_inputs=inp['source_inputs'],synth_inputs=rec(syn/'inputs.json'),synth_receipt=rec(syn/'synth-complete.json'),
        wrapper_command=rec(dep/'attestation/command.json'),wrapper_exit=rec(dep/'attestation/exit.json'),
        launch_started_utc='2026-10-19T18:00:03+00:00')
    put(dep/'attestation/complete.json',complete)
    prev=None
    for stage in ('link','implement','refine','qualify'):
        run=root/'runs'/stage;value=read(run/'stage-inputs.json')
        value.update(m['hardware'],synth_run=new_synth,synth_inputs_sha256=e.sha(syn/'inputs.json'),
                     synth_receipt_sha256=e.sha(syn/'synth-complete.json'))
        if stage=='link':value['source_checkpoint']['path']=new_synth+'/synth/post_synth.dcp'
        else:value['previous_receipt_sha256']=e.sha(root/'runs'/prev/'stage-result.json')
        put(run/'stage-inputs.json',value)
        result=read(run/'stage-result.json');result['inputs_sha256']=e.sha(run/'stage-inputs.json')
        put(run/'stage-result.json',result);prev=stage
    return refresh_manifest(root,m)


def test_distinct_ip_and_hardware_revisions_with_synchronous_proof(bundle,monkeypatch):
    root,m,_=bundle;trusted=synchronous_fixture(root,m,monkeypatch)
    result=e.verify_bundle(root,trusted)
    assert result['source_revision']=='b'*40
    assert m['source_snapshots']['ip']['revision']==e.DEPENDENCY_SOURCE
    assert m['source_snapshots']['simulation']['revision']=='d'*40


@pytest.mark.parametrize('change',['wrapper','ip_receipt','ip_revision','hardware_revision','source_inputs','launch_before_capture',
                                  'wrapper_exit','wrapper_command','identity','before','unit','schema'])
def test_synchronous_proof_rejects_wrong_bindings(bundle,monkeypatch,change):
    root,m,_=bundle;synchronous_fixture(root,m,monkeypatch)
    dep=root/'runs/dependencies';path=dep/'attestation/complete.json';value=read(path)
    if change=='wrapper':put(dep/'run_synth_with_attestation.py','unreviewed wrapper')
    elif change=='ip_receipt':monkeypatch.setattr(e,'REUSED_IP_RECEIPT_SHA256','0'*64)
    elif change=='ip_revision':value['ip_source_revision']='0'*40
    elif change=='hardware_revision':value['source_revision']='0'*40
    elif change=='source_inputs':value['source_inputs'].pop(next(iter(value['source_inputs'])))
    elif change=='launch_before_capture':value['launch_started_utc']='2026-10-19T17:00:00+00:00'
    elif change=='wrapper_exit':
        put(dep/'attestation/exit.json',{'returncode':1});value['wrapper_exit']=rec(dep/'attestation/exit.json')
    elif change=='wrapper_command':
        command=read(dep/'attestation/command.json');command[-1]='2';put(dep/'attestation/command.json',command)
        value['wrapper_command']=rec(dep/'attestation/command.json')
    elif change=='identity':value['identity_sha256']='0'*64
    elif change=='before':value['before_sha256']='0'*64
    elif change=='unit':value['unit']='another.service'
    else:value['schema']=3
    put(path,value)
    with pytest.raises(ValueError):e.verify_bundle(root,refresh_manifest(root,m))


def test_historical_ip_audit_pin_does_not_apply_to_hardware(bundle,monkeypatch):
    root,m,trusted=bundle;name='boards/c2_f2_alex/audit_c2_alex.tcl'
    assert e.REVIEWED_HISTORICAL['ip',name]=={'5e1999a5f1ad2f515c9fac78bfd616c5e4932bce4ee07c77cc38f63916df1c2e'}
    row=rec(root/'sources/ip'/name)
    # Synthetic old/new digest substitution exercises the role boundary without
    # requiring a historical Git checkout or mutating reviewed local code.
    monkeypatch.setattr(e,'REVIEWED_HISTORICAL',{('ip',name):{row['sha256']}})
    original=e.sha;monkeypatch.setattr(e,'sha',lambda p:'0'*64 if Path(p)==ROOT/name else original(p))
    x=e.Bundle(root,trusted)
    x.source('ip',m['source_snapshots']['ip']['revision'],{name:row},current={name})
    with pytest.raises(ValueError,match='hardware'):
        x.source('hardware',m['source_snapshots']['hardware']['revision'],{name:row},current={name})


@pytest.mark.parametrize('fault', ['none', 'oversized', 'invalid_utf8', 'truncated_utf8', 'mutated'])
def test_large_supplemental_stream_keeps_size_encoding_and_hash_bounds(bundle, monkeypatch, fault):
    root,m,_=bundle
    path=root/'runs/qualify/qualification/supplemental.tcldata'
    # Cross the stream chunk boundary inside a multibyte character. Keep this
    # fixture small by lowering the small-report limit only for this unit test.
    raw=b'x'*((1<<20)-1)+'\u20ac'.encode()+b'\nSYNTHETIC report only\n'
    if fault=='invalid_utf8':raw+=b'\xff'
    if fault=='truncated_utf8':raw+=b'\xe2\x82'
    path.write_bytes(raw)
    trusted=update_last_receipt(root,m)
    x=e.Bundle(root,trusted)
    monkeypatch.setattr(e,'MAX_MANIFEST',1<<20)
    if fault=='oversized':monkeypatch.setattr(e,'MAX_SUPPLEMENTAL_TEXT',len(raw)-1)
    if fault=='mutated':path.write_bytes(raw[:-1]+b'!')
    with pytest.raises(ValueError,match='oversized raw evidence'):
        x.text('qualify','qualification/supplemental.tcldata')
    if fault=='none':
        x.supplemental_text()
    else:
        with pytest.raises((ValueError,UnicodeDecodeError)):
            x.supplemental_text()
