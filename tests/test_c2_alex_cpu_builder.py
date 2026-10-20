"""Synthetic CPU-builder provenance; never hardware or route acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

e = load('cpu_evidence', ROOT/'boards/c2_f2_alex/evidence.py')
session = load('cpu_session_fixture', ROOT/'tests/test_c2_f2_alex_session.py')
c = load('cpu_collect', ROOT/'boards/c2_f2_alex/collect.py')

class Evidence:
    def __init__(self):
        self.data = {}
        self.manifest = {'instance_id': e.CPU_BUILDER, 'hardware': {'source_revision': 'b'*40}}
        self.roots = {'dependencies': '/fresh/retry', 'synth': '/fresh/retry/synthesis',
                      'ip': '/original/private-ip', 'installed': '/installed'}
    def original(self, role, name=''):
        return self.roots[role] + ('/'+name if name else '')
    def read(self, role, name): return self.data[role, name]
    def receipt(self, role): return self.read(role, 'complete.json')
    def record(self, role, name):
        raw = json.dumps(self.read(role, name), sort_keys=True).encode()
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    def names(self, role): return {name for r, name in self.data if r == role}
    def inventory(self, role, rows, *, exact):
        e.require(set(rows) == exact and all(self.record(role,n) == row for n,row in rows.items()), 'Inventory differs')

@pytest.fixture
def retry(monkeypatch):
    x = Evidence(); rev = 'b'*40; unit = 'otpu-c2-bbbbbbbb-synth-retry.service'; inv = 'c'*32
    d = x.data
    old_source = {'revision': e.FRESH_SOURCE, 'hardware_actions': False, 'support_verified': True, 'frozen_inputs': {'rtl/test.sv': 'same', 'boards/c2_f2_alex/audit_c2_alex.tcl': 'old'}}
    source = {'revision': rev, 'hardware_actions': False, 'support_verified': True, 'frozen_inputs': {'rtl/test.sv': 'same', 'boards/c2_f2_alex/audit_c2_alex.tcl': 'new'}, 'source': '/fresh/src'}
    d['dependencies','driver.py'] = 'synthetic reviewed retry driver'
    d['dependencies','original-driver.py'] = 'synthetic original driver'
    d['dependencies','original-source-record.json'] = old_source
    d['dependencies','source-record.json'] = source
    monkeypatch.setattr(e, 'REVIEWED_CPU_RETRY_DRIVERS', {x.record('dependencies','driver.py')['sha256']})
    monkeypatch.setattr(e, 'FRESH_DRIVER_SHA256', x.record('dependencies','original-driver.py')['sha256'])
    monkeypatch.setattr(e, 'FRESH_SOURCE_RECORD_SHA256', x.record('dependencies','original-source-record.json')['sha256'])
    installed = {'tool': 'bytes'}
    identity = dict(kind='c2-m7a-reused-ip-synthesis-identity',instance_id=e.CPU_BUILDER,source_revision=rev,
                    source_record=x.record('dependencies','source-record.json'),driver=x.record('dependencies','driver.py'),
                    hdk='/installed',installed_inputs=installed,unit=unit,invocation_id=inv,
                    unit_state={'Id':unit,'InvocationID':inv,'MemoryMax':str(32<<30),'CPUQuotaPerSecUSec':'4s'},hardware_actions=False,started_at='2026-10-20T14:00:00+00:00')
    original = dict(kind='c2-m7a-fresh-ip-synthesis-identity',instance_id=e.CPU_BUILDER,
                    source_revision=e.FRESH_SOURCE,source_record_sha256=e.FRESH_SOURCE_RECORD_SHA256,
                    hdk='/installed',installed_inputs=installed,hardware_actions=False)
    d['dependencies','identity.json']=identity;d['dependencies','original-identity.json']=original
    monkeypatch.setattr(e,'FRESH_IDENTITY_SHA256',x.record('dependencies','original-identity.json')['sha256'])
    for role,s in [('ip',old_source),('synth',source)]:
        d[role,'inputs.json']={'source_revision':s['revision'],'source_inputs':s['frozen_inputs'],'installed_inputs':installed}
    d['ip','ip-complete.json']={'synthetic':'private-ip'}
    monkeypatch.setattr(e,'FRESH_IP_RECEIPT',x.record('ip','ip-complete.json'))
    d['synth','synth-complete.json']={'synthetic':'synth'}
    d['synth','synth/post_synth.dcp']='synthetic DCP'
    for tree in e.DEPENDENCY_TREES:d['ip',tree+'/include.vh']='synthetic include'
    files={name:x.record('ip',name) for name in x.names('ip') if name.endswith('.vh')}
    d['dependencies','dependencies-before.json']=dict(kind='c2-reused-generated-dependencies-before-synthesis',
        instance_id=e.CPU_BUILDER,hardware_actions=False,source_revision=rev,invocation_id=inv,ip_receipt=x.record('ip','ip-complete.json'),files=files)
    d['dependencies','dependencies-after.json']={'files':copy.deepcopy(files),'ip_receipt':x.record('ip','ip-complete.json'),'finished_at':'2026-10-20T14:01:30+00:00'}
    identity['ip_receipt']=x.record('ip','ip-complete.json');identity['ip_run']=x.original('ip')
    d['dependencies','dependencies-before.json']['ip_run']=x.original('ip')
    d['dependencies','synthesis-command.json']=['/python','/fresh/src/boards/c2_f2_alex/build.py','--hdk','/installed',
        '--out','/fresh/retry/synthesis','--ip-jobs','1','--stage','synth','--ip-run','/original/private-ip']
    d['dependencies','synthesis-exit.json']={'returncode':0,'finished_at':'2026-10-20T14:01:00+00:00'}
    complete=dict(kind='c2-m7a-reused-ip-synthesis-complete-pending-functional-validation',hardware_actions=False,
        route_qualified=False,instance_id=e.CPU_BUILDER,source_revision=rev,unit=unit,invocation_id=inv,
        ip_run=x.original('ip'),ip_receipt=x.record('ip','ip-complete.json'),synth_receipt=x.record('synth','synth-complete.json'),
        checkpoint=x.record('synth','synth/post_synth.dcp'),checkpoint_path=x.original('synth','synth/post_synth.dcp'),
        dependency_files=len(files),finished_at='2026-10-20T14:02:00+00:00')
    original_files={n:x.record('dependencies',n) for n in ('original-identity.json','original-driver.py','original-source-record.json')}
    complete['original_files']=original_files;identity['original_files']=original_files
    for key,name in [('identity','identity.json'),('driver','driver.py'),('source_record','source-record.json'),
                     ('dependencies_before','dependencies-before.json'),('dependencies_after','dependencies-after.json')]:
        complete[key]=x.record('dependencies',name)
    d['dependencies','complete.json']=complete
    return x


def test_cpu_retry_replays_original_ip_and_fresh_dependencies(retry):
    assert e.dependency_attestation(retry) is None
    assert retry.receipt('dependencies')['route_qualified'] is False

@pytest.mark.parametrize('fault', ['driver','original_driver','ip_receipt','checkpoint','before_mutation','extra_include',
    'missing_include','failed','exit','wrong_command_ip','wrong_source','wrong_instance','invocation','timestamps'])
def test_cpu_retry_rejects_false_provenance(retry, fault):
    d=retry.data;complete=retry.receipt('dependencies')
    if fault in ('driver','ip_receipt','checkpoint'):complete[fault]={'sha256':'0'*64,'bytes':1}
    elif fault=='original_driver':complete['original_files']['original-driver.py']={'sha256':'0'*64,'bytes':1}
    elif fault=='before_mutation':d['dependencies','dependencies-before.json']['files']={}
    elif fault=='extra_include':d['ip',e.DEPENDENCY_TREES[0]+'/extra.vh']='new'
    elif fault=='missing_include':del d['ip',e.DEPENDENCY_TREES[0]+'/include.vh']
    elif fault=='failed':d['dependencies','failure.json']={'failure':True}
    elif fault=='exit':d['dependencies','synthesis-exit.json']['returncode']=1
    elif fault=='wrong_command_ip':d['dependencies','synthesis-command.json'][-1]='/another/private-ip'
    elif fault=='wrong_source':d['synth','inputs.json']['source_inputs']={'different':'bytes'}
    elif fault=='wrong_instance':complete['instance_id']=e.INSTANCE
    elif fault=='invocation':complete['invocation_id']='d'*32
    elif fault=='timestamps':complete['finished_at']='2026-10-20T13:00:00+00:00'
    with pytest.raises((ValueError,KeyError)):e.dependency_attestation(retry)


def test_retry_cannot_accept_unreviewed_driver(retry, monkeypatch):
    monkeypatch.setattr(e,'REVIEWED_CPU_RETRY_DRIVERS',frozenset())
    with pytest.raises(ValueError,match='Unreviewed CPU'):e.dependency_attestation(retry)


def test_builder_identity_does_not_change_physical_target(tmp_path):
    _,v,record,_=session.verified_fixture(tmp_path)
    v['instance_id']=e.CPU_BUILDER
    assert session.D.verify_deployment(record,v)['route_qualified'] is True
    assert session.A.S.INSTANCE == e.INSTANCE
    v['instance_id']='i-unapproved'
    with pytest.raises(ValueError,match='not qualified'):session.D.verify_deployment(record,v)


def test_collector_preserves_builder_and_uses_actual_receipt_layout():
    spec={'schema':2,'instance_id':e.CPU_BUILDER,'roots':{},'source_checkouts':{}}
    assert c.Plan(spec).manifest['instance_id']==e.CPU_BUILDER
    assert e.schema_receipts(2,e.CPU_BUILDER)['dependencies']=='complete.json'
    assert e.schema_receipts(2,e.INSTANCE)['dependencies']=='attestation/complete.json'


def test_cpu_retry_rejects_changed_dependency_after_even_with_rebound_receipt(retry):
    after=retry.read('dependencies','dependencies-after.json')
    after['files'][next(iter(after['files']))]={'sha256':'0'*64,'bytes':1}
    retry.receipt('dependencies')['dependencies_after']=retry.record('dependencies','dependencies-after.json')
    with pytest.raises(ValueError,match='changed during CPU retry'):e.dependency_attestation(retry)

def test_cpu_retry_rejects_new_rtl_even_with_consistent_source_receipts(retry):
    source=retry.read('dependencies','source-record.json')
    source['frozen_inputs']['rtl/test.sv']='changed RTL'
    retry.read('synth','inputs.json')['source_inputs']=source['frozen_inputs']
    updated=retry.record('dependencies','source-record.json')
    retry.receipt('dependencies')['source_record']=updated
    retry.read('dependencies','identity.json')['source_record']=updated
    retry.receipt('dependencies')['identity']=retry.record('dependencies','identity.json')
    with pytest.raises(ValueError,match='beyond its audit'):e.dependency_attestation(retry)


@pytest.mark.parametrize('field,value', [('MemoryMax',str(64<<30)),('CPUQuotaPerSecUSec','8s')])
def test_retry_resource_identity_must_match_reviewed_bound(retry,field,value):
    retry.read('dependencies','identity.json')['unit_state'][field]=value
    retry.receipt('dependencies')['identity']=retry.record('dependencies','identity.json')
    with pytest.raises(ValueError,match='resource bound'):e.dependency_attestation(retry)

def test_original_host_identity_cannot_be_replaced_even_with_rebound_map(retry):
    retry.read('dependencies','original-identity.json')['invocation_id']='e'*32
    row=retry.record('dependencies','original-identity.json')
    retry.receipt('dependencies')['original_files']['original-identity.json']=row
    retry.read('dependencies','identity.json')['original_files']['original-identity.json']=row
    retry.receipt('dependencies')['identity']=retry.record('dependencies','identity.json')
    with pytest.raises(ValueError,match='original private-IP executor'):e.dependency_attestation(retry)
