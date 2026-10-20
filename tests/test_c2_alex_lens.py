"""Offline import/display tests. All constructed records are temporary TEST fixtures.

No cloud, hardware, real Lens bundle writes, or measurements are performed here.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
L=load('alex_lens_import_test',ROOT/'tools/append_c2_alex_lens.py')
F=load('alex_lens_session_fixture',ROOT/'tests/test_c2_f2_alex_session.py')
C=L.C


def put(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def capture(directory,manifest,source='9'*40):
    value=dict(schema=1,kind='c2-alex-physical-session-files',instance_id=L.S.INSTANCE,
        source_revision=source,session='/home/ubuntu/rachit/codex-c2/UNIT-TEST-session',
        session_sha256=L.sha(directory/'session.json'),plan_sha256=L.sha(directory/'plan.json'),
        files=L.tree_inventory(directory))
    put(manifest,value);return value


@pytest.fixture
def small_capture(tmp_path):
    directory=tmp_path/'session';directory.mkdir()
    put(directory/'session.json',{'phase':'failed'});put(directory/'plan.json',{})
    manifest=tmp_path/'physical-session-files.json';capture(directory,manifest)
    return directory,manifest


def test_capture_accepts_exact_inventory_and_independent_hash(small_capture):
    directory,manifest=small_capture
    got=L.verify_capture(directory,manifest,L.sha(manifest))
    assert set(got['files'])=={'session.json','plan.json'}
    with pytest.raises(ValueError,match='hash mismatch'):
        L.verify_capture(directory,manifest,'0'*64)


@pytest.mark.parametrize('change',['extra','missing','changed','symlink','hardlink','fifo','parent_link'])
def test_capture_rejects_unbound_and_nonregular_files(small_capture,tmp_path,change):
    directory,manifest=small_capture
    if change=='extra':(directory/'extra').write_text('extra')
    elif change=='missing':(directory/'plan.json').unlink()
    elif change=='changed':(directory/'plan.json').write_text('{} ')
    elif change=='symlink':(directory/'extra').symlink_to(directory/'plan.json')
    elif change=='hardlink':os.link(directory/'plan.json',directory/'extra')
    elif change=='fifo':os.mkfifo(directory/'extra')
    else:
        alias=tmp_path/'alias';alias.symlink_to(directory,target_is_directory=True);directory=alias
    with pytest.raises(ValueError):L.verify_capture(directory,manifest,L.sha(manifest))


@pytest.mark.parametrize('change',['extra_key','wrong_instance','source','unsafe','plan_binding','inside','duplicate_json'])
def test_capture_rejects_malformed_or_ambiguous_manifest(small_capture,change):
    directory,manifest=small_capture;value=L.read(manifest)
    if change=='extra_key':value['untrusted_qualification']=True
    elif change=='wrong_instance':value['instance_id']='other'
    elif change=='source':value['source_revision']=False
    elif change=='unsafe':value['files']['../outside']=value['files']['plan.json']
    elif change=='plan_binding':value['plan_sha256']='0'*64
    elif change=='inside':manifest=directory/'capture.json'
    put(manifest,value)
    if change=='duplicate_json':manifest.write_text(manifest.read_text().replace('"schema": 1','"schema": 1,"schema": 1'))
    with pytest.raises(ValueError):L.verify_capture(directory,manifest,L.sha(manifest))


def test_incomplete_session_rejected_before_qualification(small_capture,monkeypatch):
    directory,manifest=small_capture
    monkeypatch.setattr(L.E,'verify_bundle',lambda *args:pytest.fail('Must reject failed session first'))
    with pytest.raises(ValueError,match='incomplete'):
        L.checked_session(directory,manifest,L.sha(manifest),directory,'0'*64,manifest,'0'*64)


@pytest.fixture(scope='module')
def complete_parser_fixture(tmp_path_factory):
    """Synthetic schema/arithmetic fixtures live only in pytest's temporary tree.

    The actual portable verifier is injected by the tests; no fake qualification
    can pass the real verifier and none of these fixtures are applied to Lens.
    """
    tmp=tmp_path_factory.mktemp('TEST-alex-session')
    evidence,verified,record,image=F.verified_fixture(tmp)
    directory=tmp/'session';directory.mkdir();(directory/'fixtures').mkdir()
    origin='/home/ubuntu/rachit/codex-c2/UNIT-TEST-session'
    q=dict(L.D.verify_deployment(record,verified),schema=1,kind='c2-alex-session-qualification',
        hardware_measured=False,build_manifest=origin+'/synthesis-inputs.json',
        deployment_sha256=L.sha(image),evidence_directory='/TEST/original-evidence',
        verified_evidence=verified,software_sources={name:L.sha(ROOT/name) for name in L.A.SOURCE_FILES})
    shutil.copyfile(evidence/'synth-inputs.json',directory/'synthesis-inputs.json')
    put(directory/'qualification.json',q)
    planq=dict(q,qualification_file=origin+'/qualification.json',qualification_sha256=L.sha(directory/'qualification.json'))
    entries=[]
    for index,(rows,kb) in enumerate(L.MATRIX):
        fixture=C.make_fixture(rows,kb);path=directory/'fixtures'/f'{index:02d}.npz';fixture.save(path)
        bits,integers=fixture.reference()
        entries.append(dict(rows=rows,kb=kb,frozen_path=origin+f'/fixtures/{index:02d}.npz',
            frozen_file_sha256=L.sha(path),fixture_sha256=fixture.digest()))
        bench=directory/f'benchmark-{index:02d}';bench.mkdir()
        provenance=dict(kind='c2-alex-physical-session',qualification=q,
            qualification_sha256=planq['qualification_sha256'],fixture_file_sha256=L.sha(path),
            software_revision='9'*40,software_status='',sources={name:q['software_sources'][name] for name in
                ('opentpu/c2_f2.py','opentpu/host/c2_f2.py','tools/c2_f2_alex_bench.py')})
        put(bench/'provenance.json',provenance)
        results=[];profiles=[]
        for repeat in range(3):
            for mode in ((0,1) if repeat%2==0 else (1,0)):
                beats=fixture.expected_pc_beats();counters=dict.fromkeys(C.COUNTERS,0)
                counters.update(accepted_groups=rows if mode==0 else (rows+3)//4,
                    committed_groups=rows if mode==0 else (rows+3)//4,read_issued=sum(beats),read_returned=sum(beats))
                pcs=[dict(dict.fromkeys(C.PC_COUNTERS,0),issued=n,returned=n) for n in beats]
                result=dict(test_only='SYNTHETIC PARSER FIXTURE; NEVER HARDWARE EVIDENCE',evidence_kind='f2-hardware',
                    bit_exact=True,error=0,status=C.DONE,device='f2:'+L.S.INITIAL_BDF,repeat=repeat,mode=mode,
                    rows=rows,kb=kb,base_beat=fixture.base_beat,fixture_sha256=fixture.digest(),
                    bits=bits.tolist(),integers=integers,cycles=1000000,counters=counters,pcs=pcs,
                    identity=dict(id=C.ID,abi=C.ABI,build_id=int(q['build_id'],16),core_khz=200000,
                        max_rows=512,depth=256,credits=32),weight_bytes=sum(beats)*64,
                    measured_at_utc='2000-01-01T00:00:00+00:00',source=provenance,
                    measured_window='START acceptance through final core commit; excludes preload and host readback')
                put(bench/f'result-{repeat}-{mode}.json',result);results.append(result)
                name=f"C2 Alex {'decode' if mode else 'legacy M1'} N{rows} KB{kb} repeat{repeat}"
                profiles.append(C.lens_data(result,name))
        put(bench/'results.json',results);L.lens.save(profiles,bench/'c2-f2.otpuprof')
    put(directory/'plan.json',dict(qualification=planq,fixtures=entries,execute=True,repeats=3))
    base=dict(id=L.S.R.ID_OTPU,platform=3,core_khz=200000,
        status=L.S.R.ST_WR_IDLE|L.S.R.ST_CALIB0|L.S.R.ST_CALIB1)
    put(directory/'session.json',dict(phase='complete',restored=True,benchmark_complete=True,
        load_attempted=True,instance_id=L.S.INSTANCE,slot=L.S.SLOT,initial_bdf=L.S.INITIAL_BDF,
        candidate_bdf=L.S.INITIAL_BDF,restored_bdf=L.S.INITIAL_BDF,original_identity=base,restored_identity=base,
        candidate_identity=dict(id=C.ID,abi=C.ABI,build_id=int(q['build_id'],16),core_khz=200000,status=0,hbm_ready=1),
        qualification=planq,completed_fixtures=[entry['frozen_path'] for entry in entries],
        aws_image_binding=dict(owner=L.D.ACCOUNT,state={'Code':'available'},agfi_id=record['global_image_id'],
            afi_id=record['image_id'],tags=L.D.image_tags(record))))
    manifest=tmp/'physical-session-files.json';capture(directory,manifest)
    return tmp,verified


@pytest.fixture
def parser_case(complete_parser_fixture,tmp_path,monkeypatch):
    base,verified=complete_parser_fixture
    shutil.copytree(base,tmp_path/'case');root=tmp_path/'case';directory=root/'session'
    manifest=root/'physical-session-files.json';image=root/'image.json';evidence=root/'bundle'
    monkeypatch.setattr(L.E,'verify_bundle',lambda path,digest:copy.deepcopy(verified))
    def git_show(argv,**kwargs):
        assert argv[:2]==['git','show'] and argv[2].startswith('9'*40+':')
        return (ROOT/argv[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(L.subprocess,'check_output',git_show)
    def run():return L.checked_session(directory,manifest,L.sha(manifest),evidence,
        verified['manifest_sha256'],image,L.sha(image))
    return directory,manifest,run


def test_complete_schema_exact_results_and_all_30_profiles(parser_case):
    _,_,run=parser_case;profiles,source=run()
    assert len(profiles)==30 and len(set(p['name'] for p in profiles))==30
    assert source['source_revision']=='9'*40 and source['build_id']=='264d8413'
    assert all(p['board']['test_only'].startswith('SYNTHETIC') for p in profiles)
    assert [p['board']['mode'] for p in profiles[:6]]==[0,1,1,0,0,1]


def explicit_baseline_case(parser_case):
    """Extend the synthetic parser fixture; never actual hardware evidence."""
    directory,manifest,run=parser_case
    descriptor=dict(schema=1,kind='c2-f2-restoration-baseline',instance_id=L.S.INSTANCE,
        agfi_id='agfi-00c2d61ec51bd89aa',afi_id='afi-07e20bf5679cf37d7',owner=L.D.ACCOUNT,
        shell_version='0x10212415',identity=dict(id=0x4f545055,version=0x800408,regmap=5,
            build_id=0x7e42a626,platform=3,core_khz=200000))
    put(directory/'baseline-record.json',descriptor)
    frozen=L.read(directory/'qualification.json')
    frozen.update(baseline=descriptor,baseline_record_sha256=L.sha(directory/'baseline-record.json'))
    put(directory/'qualification.json',frozen)
    plan=L.read(directory/'plan.json')
    planq=dict(frozen,qualification_file=plan['qualification']['qualification_file'],
        qualification_sha256=L.sha(directory/'qualification.json'))
    plan.update(qualification=planq,baseline=descriptor);put(directory/'plan.json',plan)
    report=L.read(directory/'session.json')
    identity=dict(descriptor['identity'],status=L.S.R.ST_WR_IDLE|L.S.R.ST_CALIB0|L.S.R.ST_CALIB1)
    image=dict(agfi=descriptor['agfi_id'],state='loaded',status_code=0,error='ok',error_code=0)
    report.update(qualification=planq,baseline=descriptor,original_identity=identity,restored_identity=identity,
                  original_image=image,restored_image=image)
    put(directory/'session.json',report)
    for bench in sorted(directory.glob('benchmark-*')):
        provenance=L.read(bench/'provenance.json')
        provenance.update(qualification=frozen,qualification_sha256=planq['qualification_sha256'])
        put(bench/'provenance.json',provenance)
        results=L.read(bench/'results.json')
        for result in results:
            result['source']=provenance
            put(bench/f"result-{result['repeat']}-{result['mode']}.json",result)
        put(bench/'results.json',results)
        L.lens.save([C.lens_data(result,
                    f"C2 Alex {'decode' if result['mode'] else 'legacy M1'} N{result['rows']} KB{result['kb']} repeat{result['repeat']}")
                    for result in results],
                    bench/'c2-f2.otpuprof')
    capture(directory,manifest)
    return directory,manifest,run


def test_explicit_restoration_baseline_keeps_all_thirty_profiles(parser_case):
    _,_,run=explicit_baseline_case(parser_case)
    profiles,_=run()
    assert len(profiles)==30


@pytest.mark.parametrize('change',['selected','original_image','restored_image','before_identity',
    'after_identity','baseline_bytes','missing_baseline_file','plan_baseline','missing_image','unbound'])
def test_explicit_restoration_baseline_rejects_mismatch_after_transfer_rehash(parser_case,change):
    directory,manifest,run=explicit_baseline_case(parser_case)
    path=directory/'session.json';report=L.read(path)
    if change=='selected':report['baseline']['agfi_id']='agfi-'+'3'*17
    elif change in ('original_image','restored_image'):report[change]['agfi']='agfi-'+'4'*17
    elif change in ('before_identity','after_identity'):
        report['original_identity' if change=='before_identity' else 'restored_identity']['build_id']+=1
    elif change=='baseline_bytes':
        p=directory/'baseline-record.json';p.write_text(p.read_text()+' ')
    elif change=='missing_baseline_file':(directory/'baseline-record.json').unlink()
    elif change=='plan_baseline':
        p=directory/'plan.json';plan=L.read(p);del plan['baseline'];put(p,plan)
    elif change=='missing_image':del report['restored_image']
    elif change=='unbound':
        p=directory/'plan.json';plan=L.read(p);del plan['qualification']['baseline']
        report['qualification']=plan['qualification'];put(p,plan)
    put(path,report);capture(directory,manifest)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):run()


@pytest.mark.parametrize('change,reason',[
    ('not_restored','incomplete'),('logging_error','failure'),('wrong_tag','five-tag'),
    ('wrong_manifest','qualification'),('candidate','Candidate identity'),('dirty_host','not clean'),('wrong_host','trusted session'),
    ('raw_mismatch','Raw per-run'),('nonhardware','Only physical'),('arithmetic','canonical arithmetic'),
    ('credit','credit overflow'),('traffic','traffic mismatch'),('profile','checked raw'),
    ('missing_repeat','Incomplete benchmark'),('wrong_qualification','qualification')])
def test_session_semantic_tampering_rejected_even_after_transfer_rehash(parser_case,change,reason):
    directory,manifest,run=parser_case
    path=directory/'session.json';report=L.read(path)
    if change=='not_restored':report['restored']=False;put(path,report)
    elif change=='logging_error':report['logging_errors']=['failed write'];put(path,report)
    elif change=='wrong_tag':report['aws_image_binding']['tags']['c2:evidence_manifest_sha256']='0'*64;put(path,report)
    elif change=='candidate':report['candidate_identity']['hbm_ready']=0;put(path,report)
    elif change=='wrong_manifest':report['qualification']['build_manifest']='/wrong';put(path,report)
    elif change=='wrong_qualification':report['qualification']['route_qualified']=False;put(path,report)
    else:
        bench=directory/'benchmark-00'
        if change in ('dirty_host','wrong_host'):
            path=bench/'provenance.json';p=L.read(path)
            p['software_status' if change=='dirty_host' else 'software_revision']='dirty' if change=='dirty_host' else '8'*40
            put(path,p)
        elif change=='profile':
            path=bench/'c2-f2.otpuprof';p=L.lens.load(path)['profiles'];p[0]['cycles']+=1;L.lens.save(p,path)
        else:
            path=bench/'results.json';results=L.read(path)
            if change=='raw_mismatch':results[0]['cycles']+=1
            elif change=='missing_repeat':results.pop()
            elif change=='nonhardware':results[0]['evidence_kind']='unit-test'
            elif change=='arithmetic':results[0]['bits'][0]^=1
            elif change=='credit':results[0]['pcs'][0]['max_outstanding']=33
            elif change=='traffic':results[0]['pcs'][0]['returned']-=1
            put(path,results)
            if change not in ('raw_mismatch','missing_repeat'):put(bench/'result-0-0.json',results[0])
    capture(directory,manifest)
    with pytest.raises(ValueError,match=reason):run()


def test_pending_portable_evidence_cannot_be_bypassed(parser_case,monkeypatch):
    _,_,run=parser_case
    monkeypatch.setattr(L.E,'verify_bundle',lambda *args:(_ for _ in ()).throw(ValueError('Pending route review')))
    with pytest.raises(ValueError,match='Pending route'):run()


def display_profile(name):
    return dict(kind='unit-test',name=name,cycles=10,instrs=[],slices=[{}],config={'S':1})


@pytest.fixture
def storage_case(tmp_path,monkeypatch):
    """Storage-only test doubles, deliberately not physical/board profiles."""
    target=tmp_path/'all-profiles.otpuprof';html=tmp_path/'all-profiles.html'
    old=[display_profile('old '+str(i)) for i in range(128)]
    incoming=[display_profile('TEST new '+str(i)) for i in range(30)]
    L.lens.save(old,target);html.write_text('original HTML')
    for key,value in [('VIEWER',tmp_path),('TARGET',target),('HTML',html),('BASE_SHA',L.sha(target))]:
        monkeypatch.setattr(L,key,value)
    monkeypatch.setattr(L,'checked_session',lambda *args:(incoming,{'test_only':True}))
    args=dict(manifest=tmp_path/'unused',manifest_sha256='0'*64,evidence_dir=tmp_path/'unused',
        evidence_manifest_sha256='0'*64,image_record=tmp_path/'unused',image_record_sha256='0'*64,
        viewer=L.APP,viewer_sha256=L.sha(L.APP))
    return tmp_path,old,args


def test_storage_default_read_only_then_preserves_128_order_and_backs_up(storage_case):
    root,old,args=storage_case;before=L.TARGET.read_bytes()
    report=L.append(root,**args)
    assert not report['applied'] and sorted(p.name for p in root.iterdir())==['all-profiles.html','all-profiles.otpuprof']
    report=L.append(root,apply=True,**args)
    assert report['applied'] and report['total_profiles']==158
    doc=L.lens.load(L.TARGET)
    assert doc['profiles'][:128]==old and len(doc['profiles'])==158
    assert (root/'all-profiles.pre-alex-hardware.otpuprof').read_bytes()==before
    assert (root/'all-profiles.pre-alex-hardware.html').read_text()=='original HTML'
    assert 'native-interface-lower-bound' in L.HTML.read_text() and 'blocked_scales' in L.HTML.read_text()
    with pytest.raises(ValueError,match='reviewed 128'):L.append(root,apply=True,**args)


def test_storage_rolls_back_both_files_if_second_replace_fails(storage_case,monkeypatch):
    root,_,args=storage_case;before=L.TARGET.read_bytes();real_replace=os.replace
    def fail_second(src,dst):
        if dst==L.TARGET:raise OSError('TEST write failure')
        return real_replace(src,dst)
    monkeypatch.setattr(L.os,'replace',fail_second)
    with pytest.raises(OSError,match='TEST'):L.append(root,apply=True,**args)
    assert L.TARGET.read_bytes()==before and L.HTML.read_text()=='original HTML'
    assert L.read(root/'alex-hardware-profile-sources.json')['state']=='rolled-back'


@pytest.mark.parametrize('change',['viewer','duplicate','changed_baseline'])
def test_storage_rejects_unreviewed_viewer_duplicate_or_changed_baseline(storage_case,monkeypatch,change):
    root,_,args=storage_case
    if change=='viewer':args['viewer_sha256']='0'*64
    elif change=='duplicate':monkeypatch.setattr(L,'checked_session',lambda *a:([display_profile('same')]*30,{}))
    else:L.TARGET.write_bytes(b'changed')
    with pytest.raises(ValueError):L.append(root,apply=True,**args)
    assert not (root/'alex-hardware-profile-sources.json').exists()


def test_actual_reviewed_baseline_read_only():
    if not L.TARGET.exists():pytest.skip('Shared Lens baseline not available')
    assert len(L.baseline()['profiles'])==128


@pytest.mark.parametrize('split',[False,True])
def test_merged_viewer_actual_bucket_renderer_preserves_blocked_causes(split):
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    page=L.APP.read_text();code=page.split('const BLOCKED_KEYS =',1)[1].split('function allCauses',1)[0]
    values=dict(n=10,bm=2,mb=8,mbc=2,mbr=5,mbs=1,mbo=0)
    js='const BLOCKED_KEYS ='+code+'\n'
    js+='const b='+json.dumps(values)+';\n'
    js+='const m={bn:[10],pre:PRE,get:k=>b[k]||0,cov:[[0],[0],[0],[0],[0]]};\n'.replace(
        'PRE',json.dumps({k:[0,values[k]] for k in ('mbc','mbr','mbs','mbo')} if split else {}))
    js+='process.stdout.write(JSON.stringify(bucketCauses(m,0)));'
    out=json.loads(subprocess.check_output([node,'-e',js],text=True))
    assert sum(out.values())==10
    if split:
        assert 'blocked' not in out
        assert [out[k] for k in ('blocked_credits','blocked_row','blocked_scales','blocked_other')]==[2,5,1,0]
    else:assert out['blocked']==8


@pytest.mark.parametrize('physical',[False,True])
def test_native_memory_table_is_metadata_scoped_and_missing_is_unmeasured(physical):
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    # Display-only object, never exported, saved or accepted by the import gate.
    profile=dict(kind='unit-test',name='display fixture',cycles=100,clock_mhz=200,
        config=dict(S=1,D=128,MCOLS=4,LANES=8),roofline=dict(kind='native-interface-lower-bound',bound=10),
        measurements=dict(unit_utilization=False,instruction_timeline=False),slices=[{}],peak_macs=512,
        board=dict(evidence_kind='f2-hardware' if physical else 'unit-test',pcs=[dict(issued=9,returned=9,
            max_outstanding=3,blocked_cycles=12,pending_no_response_cycles=70,credit_wait_cycles=40)]))
    page=L.APP.read_text();overview=page.split('function renderOverview() {',1)[1].split(
        '// ------------------------------------------------------------------ timeline (canvas)',1)[0]
    js='const target={};const $=()=>target;const fmt=String,pct=x=>(100*x).toFixed(1)+"%",UCOL=()=>"red",esc=String;\n'
    js+='const p='+json.dumps(profile)+';const M={p,cycles:100,s:0,ins:[],units:["MXU"],nb:0};\n'
    js+='function renderOverview() {'+overview+'\nrenderOverview();process.stdout.write(target.innerHTML);'
    out=subprocess.check_output([node,'-e',js],text=True)
    if physical:
        assert 'Native memory counters' in out and 'not additive lost cycles' in out
        assert '70.0% of window' in out and '40.0% of window' in out
        assert '<td class="n mono">unmeasured / unmeasured</td>' in out
        assert 'per-request latency' in out and 'paired half or core consumption' in out
        assert '110.0%' not in out  # Overlapping counters are never added.
    else:assert 'Native memory counters' not in out


@pytest.mark.parametrize('name',['opentpu/fp32.py','opentpu/lens.py','opentpu/quant.py','opentpu/isasim.py'])
def test_all_executed_replay_dependencies_must_match_measured_git(parser_case,monkeypatch,name):
    _,_,run=parser_case;original=L.sha
    monkeypatch.setattr(L,'sha',lambda path:'0'*64 if Path(path)==ROOT/name else original(path))
    with pytest.raises(ValueError,match='Local replay dependency differs'):run()


def test_storage_final_journal_failure_restores_both_originals(storage_case,monkeypatch):
    root,_,args=storage_case;before=L.TARGET.read_bytes();write=L.write_report
    def fail_complete(path,value):
        if value.get('state')=='complete':raise OSError('TEST final journal failure')
        return write(path,value)
    monkeypatch.setattr(L,'write_report',fail_complete)
    with pytest.raises(OSError,match='final journal'):L.append(root,apply=True,**args)
    assert L.TARGET.read_bytes()==before and L.HTML.read_text()=='original HTML'
    report=L.read(root/'alex-hardware-profile-sources.json')
    assert report['state']=='rolled-back' and report['applied'] is False
