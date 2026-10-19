"""Offline verification and optional Lens append for one restored Alex/C2 session.

Default is read-only validation. --apply is permitted only after real hardware
evidence exists. No hardware data is generated, inferred, relabeled, or repaired.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tools/c2_f2_alex_session.py').is_file())
sys.path.insert(0,str(ROOT))
from opentpu import c2_f2 as C, lens

spec=importlib.util.spec_from_file_location('c2_session_check',ROOT/'tools/c2_f2_session.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

A=module('alex_lens_session','tools/c2_f2_alex_session.py')
E,D=A.E,A.D
VIEWER=ROOT.parent/'lens-v4'
TARGET=VIEWER/'all-profiles.otpuprof'
HTML=VIEWER/'all-profiles.html'
APP=ROOT/'opentpu/lens_app.html'
BASE_COUNT=128
BASE_SHA='3059bbbfeae93eb65595dfa8a81c3cbd67a1c7f7b465daf1c51eafe259466a58'
MATRIX=[(31,1),(257,9),(31,54),(31,255),(3,9)]
# Repository dependencies used by the local canonical replay/export path. The
# clean measured commit pins these even where the session's explicit file list
# records only its top-level entrypoints. Never execute code from the capture.
REPLAY_FILES=('opentpu/__init__.py','opentpu/c2_f2.py','opentpu/host/c2_f2.py',
    'tools/c2_f2_alex_bench.py','opentpu/fp32.py','opentpu/isa.py','opentpu/isasim.py',
    'opentpu/quant.py','opentpu/profile.py','opentpu/lens.py','opentpu/rtlsim.py',
    'opentpu/target.py','opentpu/simtmp.py')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return E.decode(Path(path).read_bytes())

def regular(path):
    path=Path(path)
    E.no_symlink(path)
    require(path.is_file() and path.stat().st_nlink==1,'Missing/nonregular/hardlinked evidence: '+str(path))

def tree_inventory(directory):
    E.no_symlink(directory)
    require(directory.is_dir(),'Missing session directory')
    files={};total=0
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(),'Symlink in physical session')
        if path.is_dir():continue
        regular(path)
        name=E.relative(path.relative_to(directory).as_posix())
        size=path.stat().st_size;total+=size
        require(size<=1<<30 and total<=2<<30 and len(files)<10000,'Session evidence exceeds capture bounds')
        files[name]=dict(sha256=sha(path),bytes=size)
    return files

def verify_capture(directory,manifest,manifest_sha256):
    directory=Path(os.path.abspath(directory));manifest=Path(os.path.abspath(manifest))
    E.digest(manifest_sha256);regular(manifest)
    require(not manifest.is_relative_to(directory),'Transfer manifest must be outside the session tree')
    require(sha(manifest)==manifest_sha256,'Physical-session capture manifest hash mismatch')
    capture=read(manifest)
    require(set(capture)=={'schema','kind','instance_id','source_revision','session','session_sha256','plan_sha256','files'},
            'Unexpected physical-session capture schema')
    require(type(capture['schema']) is int and capture['schema']==1 and
            capture['kind']=='c2-alex-physical-session-files' and capture['instance_id']==S.INSTANCE,
            'Wrong physical-session capture kind/instance')
    require(isinstance(capture['source_revision'],str) and
            re.fullmatch('[0-9a-f]{40}',capture['source_revision']),'Missing physical executor commit')
    E.absolute(capture['session'])
    require(capture['session'].startswith('/home/ubuntu/rachit/codex-c2/'),'Unexpected original session root')
    require(isinstance(capture['files'],dict) and capture['files'],'Missing session file inventory')
    for name,record in capture['files'].items():E.relative(name);E.metadata(record)
    require(tree_inventory(directory)==capture['files'],'Physical-session captured file inventory differs')
    for name,key in [('session.json','session_sha256'),('plan.json','plan_sha256')]:
        E.digest(capture[key]);require(capture['files'].get(name,{}).get('sha256')==capture[key],'Capture root record mismatch')
    require(sha(manifest)==manifest_sha256,'Capture manifest changed during verification')
    return capture

def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_report(path,value):
    """Keep the last complete journal intact if a later write fails."""
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.alex-lens-journal-',
                                         delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def unique(profiles):
    names=[p['name'] for p in profiles]
    digests=[profile_digest(p) for p in profiles]
    require(len(set(names))==len(names),'Profile names are not unique')
    require(len(set(digests))==len(digests),'Duplicate profile contents')
    return digests

def baseline():
    require(sha(TARGET)==BASE_SHA,'Combined Lens changed from the reviewed 128-profile baseline')
    doc=lens.load(TARGET)
    require(len(doc['profiles'])==BASE_COUNT,'Expected exactly 128 existing profiles')
    unique(doc['profiles'])
    return doc

def source_hashes(provenance,qualification,capture):
    commit=provenance.get('software_revision','')
    require(re.fullmatch('[0-9a-f]{40}',commit),'Missing exact host source commit')
    require(commit==capture['source_revision'],'Host source differs from trusted session capture')
    require(provenance.get('software_status')=='','Hardware host checkout was not clean')
    require(provenance.get('kind')=='c2-alex-physical-session','Wrong physical benchmark provenance')
    sources=provenance.get('sources',{})
    expected={'opentpu/c2_f2.py','opentpu/host/c2_f2.py','tools/c2_f2_alex_bench.py'}
    require(set(sources)==expected,'Incomplete host source provenance')
    frozen=qualification.get('software_sources',{})
    require(set(frozen)==set(A.SOURCE_FILES),'Incomplete qualified software inventory')
    git_hashes={}
    for name,digest in frozen.items():
        E.digest(digest)
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        git_hashes[name]=hashlib.sha256(raw).hexdigest()
        require(git_hashes[name]==digest,'Host source hash mismatch: '+name)
    for name,digest in sources.items():
        require(digest==frozen[name] and sha(ROOT/name)==digest,'Exporter/arithmetic differs from reviewed host source: '+name)
    for name in REPLAY_FILES:
        if name not in git_hashes:
            raw=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            git_hashes[name]=hashlib.sha256(raw).hexdigest()
        require(sha(ROOT/name)==git_hashes[name],'Local replay dependency differs from measured source: '+name)
    return {name:git_hashes[name] for name in REPLAY_FILES}

def checked_result(result,fixture,expected,qualification,device,repeat,mode):
    require(result.get('evidence_kind')=='f2-hardware','Only physical F2 evidence is accepted')
    require(result.get('bit_exact') is True,'Result lacks exact arithmetic verification')
    require(result.get('error')==0 and result.get('status')==C.DONE,'Hardware run did not complete cleanly')
    require(result.get('device')==device and result.get('repeat')==repeat and result.get('mode')==mode,
            'Hardware device, repeat, or mode differs from the session')
    require((result.get('rows'),result.get('kb'),result.get('base_beat'))==
            (fixture.rows,fixture.kb,fixture.base_beat),'Result fixture geometry differs')
    require(result.get('fixture_sha256')==fixture.digest(),'Result fixture digest differs')
    require(result.get('bits')==expected[0].tolist() and result.get('integers')==expected[1],
            'Recorded results do not match canonical arithmetic')
    identity=dict(id=C.ID,abi=C.ABI,build_id=int(qualification['build_id'],16),core_khz=200000,
                  max_rows=512,depth=256,credits=32)
    require(result.get('identity')==identity,'Hardware image identity or geometry differs')
    cycles=result.get('cycles')
    require(type(cycles) is int and 0<cycles<2**32,'Invalid/wrapped device cycle count')
    require(datetime.fromisoformat(result['measured_at_utc']).tzinfo is not None,'Missing measurement timezone')
    counters,pcs=result['counters'],result['pcs']
    require(set(counters)==set(C.COUNTERS) and len(pcs)==8,'Incomplete physical counters')
    require(all(type(v) is int and 0<=v<2**32 for v in counters.values()),'Invalid physical counter')
    beats=fixture.expected_pc_beats();groups=fixture.rows if mode==0 else (fixture.rows+3)//4
    require(counters['accepted_groups']==counters['committed_groups']==groups,'Incomplete group retirement')
    require(counters['read_issued']==counters['read_returned']==sum(beats),'Native traffic mismatch')
    require(result['weight_bytes']==sum(beats)*64,'Weight byte count differs')
    require(counters['max_outstanding']<=8*identity['credits'],'Aggregate credit overflow')
    for key in ('advance','native_blocked_cycles','weight_wait','scale_wait','act_wait','output_wait'):
        require(counters[key]<=cycles,'Aggregate wait exceeds measured interval: '+key)
    for pc,(values,expected_beats) in enumerate(zip(pcs,beats)):
        require(set(values)==set(C.PC_COUNTERS),'Incomplete per-PC counters')
        require(all(type(v) is int and 0<=v<2**32 for v in values.values()),'Invalid per-PC counter')
        require(values['issued']==values['returned']==expected_beats,f'PC{pc} traffic mismatch')
        require(values['max_outstanding']<=identity['credits'],f'PC{pc} credit overflow')
        for key in ('blocked_cycles','pending_no_response_cycles','credit_wait_cycles'):
            require(values[key]<=cycles,f'PC{pc} wait exceeds measured interval')
    require(result['measured_window']==
        'START acceptance through final core commit; excludes preload and host readback','Unexpected measurement interval')

def checked_session(directory,manifest,manifest_sha256,evidence_dir,evidence_manifest_sha256,
                    image_record,image_record_sha256):
    directory=Path(os.path.abspath(directory))
    capture=verify_capture(directory,manifest,manifest_sha256)
    report=read(directory/'session.json');plan=read(directory/'plan.json')
    require(report.get('phase')=='complete' and report.get('restored') is True and
            report.get('benchmark_complete') is True and report.get('load_attempted') is True,
            'Session is incomplete or original image restoration was not verified')
    require(not any(report.get(k) for k in ('error','restore_error','logging_errors','signal')),
            'Session contains a failure, signal, or missing evidence')
    require(plan.get('execute') is True and type(plan.get('repeats')) is int and
            plan['repeats']==3,'Expected executed physical plan with three repeats')
    require(report.get('instance_id')==S.INSTANCE and report.get('slot')==S.SLOT and
            report.get('initial_bdf')==S.INITIAL_BDF,'Unexpected hardware session target')
    for key in ('candidate_bdf','restored_bdf'):
        require(S.BDF_RE.fullmatch(report.get(key,'')),'Missing checked BDF: '+key)
    before,after=report['original_identity'],report['restored_identity']
    S.require_baseline(before);S.require_baseline(after)
    require({k:v for k,v in before.items() if k!='status'}=={k:v for k,v in after.items() if k!='status'},
            'Restored baseline identity differs from the original')
    q=plan['qualification'];require(report.get('qualification')==q,'Session/plan qualification differs')
    require(q.get('schema')==1 and q.get('kind')=='c2-alex-session-qualification' and
            q.get('route_qualified') is True and q.get('hardware_measured') is False,
            'Wrong Alex qualification record')
    frozen=read(directory/'qualification.json')
    require(q==dict(frozen,qualification_file=capture['session']+'/qualification.json',
                   qualification_sha256=sha(directory/'qualification.json')),
            'Frozen session qualification differs from plan')
    require(q.get('build_manifest')==capture['session']+'/synthesis-inputs.json',
            'Session synthesis locator differs')
    verified=E.verify_bundle(Path(evidence_dir),evidence_manifest_sha256)
    regular(Path(image_record));E.digest(image_record_sha256)
    require(sha(image_record)==image_record_sha256==q.get('deployment_sha256'),'Available deployment record changed')
    image=read(image_record);deployment=D.verify_deployment(image,verified)
    require(all(q.get(k)==v for k,v in deployment.items()) and q.get('verified_evidence')==verified,
            'Session does not bind this qualified portable evidence/deployment')
    require(q.get('evidence_manifest_sha256')==evidence_manifest_sha256,'Evidence manifest binding differs')
    expected_inputs=Path(evidence_dir)/E.relative(verified['synth_inputs_relative_path'])
    require(sha(directory/'synthesis-inputs.json')==sha(expected_inputs)==q['inputs_sha256'],
            'Frozen synthesis inputs differ from qualified evidence')
    inputs=read(directory/'synthesis-inputs.json')
    require(inputs.get('kind')=='c2-alex-burst200-build-inputs' and
            inputs.get('source_revision')==q['source_revision'] and inputs.get('build_id')==q['build_id'] and
            (inputs.get('core_khz'),inputs.get('hbm_mhz'),inputs.get('hbm_ports'),inputs.get('channel_bytes'))==
            (200000,450,8,2**29),'Build inputs identify another image')
    candidate=report.get('candidate_identity',{})
    require(all(candidate.get(key)==value for key,value in
                dict(id=C.ID,abi=C.ABI,build_id=int(q['build_id'],16),core_khz=200000).items()) and
            type(candidate.get('status')) is int and not candidate['status']&(C.BUSY|C.DRAINING|C.ERROR) and
            type(candidate.get('hbm_ready')) is int and candidate['hbm_ready']&1,
            'Candidate identity/readiness differs from the qualified image')
    binding=report['aws_image_binding']
    require(binding.get('owner')==D.ACCOUNT and binding.get('state',{}).get('Code')=='available' and
            binding.get('agfi_id')==q['agfi_id'] and binding.get('afi_id')==q['afi_id'],
            'AWS image binding differs')
    require(all(binding.get('tags',{}).get(k)==v for k,v in D.image_tags(image).items()),
            'AWS five-tag provenance binding differs')
    entries=plan['fixtures']
    require([(entry['rows'],entry['kb']) for entry in entries]==MATRIX,'Expected the five reviewed hardware fixtures')
    require(report.get('completed_fixtures')==[entry['frozen_path'] for entry in entries],
            'Session did not complete every fixture')
    profiles=[];evidence=[];replay_sources=None
    for index,entry in enumerate(entries):
        fixture_path=directory/'fixtures'/f'{index:02d}.npz'
        require(entry['frozen_path']==capture['session']+f'/fixtures/{index:02d}.npz',
                'Frozen fixture locator differs')
        require(sha(fixture_path)==entry['frozen_file_sha256'],'Frozen fixture bytes changed')
        fixture=C.Fixture.load(fixture_path)
        require((fixture.rows,fixture.kb)==MATRIX[index] and fixture.digest()==entry['fixture_sha256'],
                'Frozen fixture geometry/digest changed')
        bench=directory/f'benchmark-{index:02d}'
        provenance=read(bench/'provenance.json');current_replay=source_hashes(provenance,q,capture)
        require(replay_sources is None or replay_sources==current_replay,'Replay dependencies changed between fixtures')
        replay_sources=current_replay
        expected=fixture.reference()
        require(provenance.get('qualification')==frozen and
                provenance.get('qualification_sha256')==q['qualification_sha256'] and
                provenance.get('fixture_file_sha256')==sha(fixture_path),
                'Benchmark provenance differs from frozen qualification/fixture')
        results=read(bench/'results.json');bundle=lens.load(bench/'c2-f2.otpuprof')['profiles']
        order=[(repeat,mode) for repeat in range(3) for mode in ((0,1) if repeat%2==0 else (1,0))]
        require(len(results)==len(bundle)==6,'Incomplete benchmark result/profile matrix')
        for offset,((repeat,mode),result,profile) in enumerate(zip(order,results,bundle)):
            require(read(bench/f'result-{repeat}-{mode}.json')==result,'Raw per-run evidence differs')
            require(result.get('source')==provenance,'Per-run source provenance differs')
            checked_result(result,fixture,expected,q,'f2:'+report['candidate_bdf'],repeat,mode)
            name=f"C2 Alex {'decode' if mode else 'legacy M1'} N{fixture.rows} KB{fixture.kb} repeat{repeat}"
            require(profile==C.lens_data(result,name),'Lens profile differs from checked raw hardware evidence')
            lens.summary_data(profile)
            profiles.append(profile)
            evidence.append(dict(bundle=str(bench/'c2-f2.otpuprof'),bundle_sha256=sha(bench/'c2-f2.otpuprof'),
                source_profile=offset,profile_sha256=profile_digest(profile),name=profile['name'],
                cycles=profile['cycles'],evidence_kind=result['evidence_kind']))
    require(len(profiles)==30,'Expected exactly 30 physical profiles');unique(profiles)
    require(tree_inventory(directory)==capture['files'] and sha(manifest)==manifest_sha256 and
            sha(image_record)==image_record_sha256,'Physical/deployment evidence changed during validation')
    return profiles,dict(session=str(directory),session_manifest_sha256=manifest_sha256,
        session_sha256=sha(directory/'session.json'),plan_sha256=sha(directory/'plan.json'),
        evidence_manifest_sha256=evidence_manifest_sha256,deployment_sha256=image_record_sha256,
        source_revision=capture['source_revision'],build_id=q['build_id'],replay_sources=replay_sources,profiles=evidence)

def append(directory,*,manifest,manifest_sha256,evidence_dir,evidence_manifest_sha256,
           image_record,image_record_sha256,viewer=APP,viewer_sha256,apply=False):
    """Verify first; replace the reviewed local bundle only with explicit --apply.

    A trusted capture hash binds transfer bytes, not the truth of arbitrary
    submissions. Qualification, restoration, arithmetic and source provenance
    are replayed independently before accepting any profile.
    """
    viewer=Path(os.path.abspath(viewer));regular(viewer);E.digest(viewer_sha256)
    template=viewer.read_bytes()
    require(hashlib.sha256(template).hexdigest()==viewer_sha256,'Viewer template hash mismatch')
    require(template.count(b'/*__DATA__*/null')==1,'Unexpected Lens application template')
    for marker in (b'BLOCKED_KEYS',b'blocked_credits',b'native-interface-lower-bound',
                   b'unit_utilization',b'instruction_timeline'):
        require(marker in template,'Viewer lacks required measured-counter semantics')
    regular(TARGET);regular(HTML)
    original=baseline();prior_hashes=unique(original['profiles'])
    original_html_sha=sha(HTML)
    profiles,provenance=checked_session(directory,manifest,manifest_sha256,evidence_dir,
                                       evidence_manifest_sha256,image_record,image_record_sha256)
    require(len(profiles)==30,'Expected exactly 30 verified physical profiles')
    combined=original['profiles']+profiles;unique(combined)
    require(unique(combined[:BASE_COUNT])==prior_hashes,'Existing profile contents/order changed')
    report=dict(kind='verified-physical-C2-Alex-Lens-append',applied=False,baseline_sha256=BASE_SHA,
                preserved_profiles=BASE_COUNT,total_profiles=len(combined),source=provenance,
                app_sha256=viewer_sha256,app_source=str(viewer))
    if not apply:return report
    backup=VIEWER/'all-profiles.pre-alex-hardware.otpuprof'
    html_backup=VIEWER/'all-profiles.pre-alex-hardware.html'
    journal=VIEWER/'alex-hardware-profile-sources.json'
    require(not any(p.exists() for p in (backup,html_backup,journal)),'Hardware append already exists; refusing duplication')
    previous_app=lens.APP
    try:
        with tempfile.TemporaryDirectory(prefix='.c2-hardware-',dir=VIEWER) as temp:
            pending=Path(temp)/TARGET.name;pending_html=Path(temp)/HTML.name
            captured_app=Path(temp)/'lens_app.html';captured_app.write_bytes(template);lens.APP=captured_app
            lens.save(combined,pending);checked=lens.load(pending)
            require(checked['profiles']==combined and unique(checked['profiles'][:BASE_COUNT])==prior_hashes,
                    'Lens roundtrip changed existing profiles')
            rendered=lens.render(checked,title='openTPU v4 — recorded profiles and physical C2 measurements')
            require('/*__DATA__*/null' not in rendered,'Unpopulated Lens template')
            pending_html.write_text(rendered)
            require(sha(TARGET)==BASE_SHA and sha(HTML)==original_html_sha,
                    'Combined bundle or viewer changed during validation')
            require(sha(viewer)==viewer_sha256,'Viewer template changed during validation')
            shutil.copy2(TARGET,backup);shutil.copy2(HTML,html_backup)
            report.update(bundle_sha256=sha(pending),html_sha256=sha(pending_html))
            write_report(journal,report|dict(state='prepared'))
            try:
                os.replace(pending_html,HTML);os.replace(pending,TARGET)
                report.update(applied=True,state='complete')
                write_report(journal,report)
            except BaseException:
                shutil.copy2(backup,TARGET);shutil.copy2(html_backup,HTML)
                report.update(applied=False,state='rolled-back')
                try:write_report(journal,report)
                except Exception:pass  # Retained prepared journal + backups support recovery.
                raise
    finally:lens.APP=previous_app
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=Path)
    parser.add_argument('--session-manifest',required=True,type=Path)
    parser.add_argument('--session-manifest-sha256',required=True)
    parser.add_argument('--evidence-dir',required=True,type=Path)
    parser.add_argument('--evidence-manifest-sha256',required=True)
    parser.add_argument('--image-record',required=True,type=Path)
    parser.add_argument('--image-record-sha256',required=True)
    parser.add_argument('--viewer',type=Path,default=APP)
    parser.add_argument('--viewer-sha256',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(append(args.session,manifest=args.session_manifest,
        manifest_sha256=args.session_manifest_sha256,evidence_dir=args.evidence_dir,
        evidence_manifest_sha256=args.evidence_manifest_sha256,image_record=args.image_record,
        image_record_sha256=args.image_record_sha256,viewer=args.viewer,
        viewer_sha256=args.viewer_sha256,apply=args.apply),indent=2))

if __name__=='__main__':main()
