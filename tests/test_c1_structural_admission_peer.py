"""Independent partial-replay failure-retention tests; synthetic captures only."""
from copy import deepcopy
import pytest
from boards.c1_f2_alex import structural_review as S
from boards.c1_f2_alex import admission_specialization_review as Z
from boards.c1_f2_alex import admission_transition_review as T
from boards.c1_f2_alex import admission_pair_review as P
from boards.c1_f2_alex import admission_equation_review as E


def setup(monkeypatch,case):
    source=dict(source_elided_bits_proven=True,issues=[],facts_sha256='a'*64,
                canonical_facts={'configuration':dict(T.CONFIGURATION)})
    transition=dict(one_step_equivalent_under_source_specialization=True,issues=[],facts_sha256='b'*64,
        canonical_facts={'channels':{str(i):dict(configuration_required=dict(T.CONFIGURATION),rtl_sha256=E.RTL_SHA256,
            one_step_equivalent_under_source_specialization=True,root=f'actual/{i}/root/O') for i in range(8)}})
    calls=[]
    def auth(**kw):
        if kw['role']!='accounting_boundary':raise ValueError('unrelated actual evidence unavailable')
        return dict(raw={'sha256':('c' if kw['stage']=='link' else 'd')*64},
            checkpoint_sha256=kw['checkpoint_sha256'],helpers={'capture_accounting_boundary.tcl':'e'*64,'capture_native.tcl':'f'*64})
    def replay(**kw):
        value=deepcopy(transition)
        if case=='transition_failure' and kw['checkpoint_sha256']=='2'*64:
            value.update(issues=[{'reason':'actual final arithmetic mismatch'}],one_step_equivalent_under_source_specialization=False)
        return value
    def source_replay(**kw):
        value=deepcopy(source)
        if case=='source_failure':value.update(issues=[{'reason':'actual source changed'}],source_elided_bits_proven=False)
        return value
    def pair(**kw):
        calls.append(kw)
        if case=='pair_failure':return dict(issues=[{'reason':'actual final RAM input differs'}],paired_preservation_proven=False)
        return dict(issues=[],paired_preservation_proven=True)
    monkeypatch.setattr(S,'authenticate_capture',auth)
    monkeypatch.setattr(S,'original_structural_observation',lambda *a:dict(failed_scope_resolved=False))
    monkeypatch.setattr(Z,'review_sources',source_replay)
    monkeypatch.setattr(T,'review_artifact',replay)
    if case!='same_checkpoint':monkeypatch.setattr(P,'review_pair',pair)
    else:
        # The real public pair guard must run before even reading raw evidence.
        monkeypatch.setattr(P.R,'review_artifact',lambda **kw:pytest.fail('same-checkpoint pair reached raw parser'))
    auxiliary={k:'/synthetic/'+k for role in S.SPECS for k in (role,role+'_receipt',role+'_linked',role+'_linked_receipt')}
    return dict(reports={},auxiliary=auxiliary,checkpoint_sha256='2'*64,
        linked_checkpoint_sha256=('2' if case=='same_checkpoint' else '1')*64,
        prerequisites={'receipt_sha256':dict(link='7'*64,refine='8'*64),'source_binding':{},'source_specialization':source}),calls


@pytest.mark.parametrize('case',['all_partial_pass','transition_failure','source_failure','pair_failure','same_checkpoint'])
def test_partial_success_never_hides_other_failure_or_grants_final_acceptance(monkeypatch,case):
    kwargs,calls=setup(monkeypatch,case)
    value=S.review_artifact(**kwargs)
    assert value['resolved_pending_reviews']==[]
    assert value['route_qualified'] is value['hardware_measured'] is False
    assert value['issues'][0]==dict(reason='structural_semantics_pending',obligations=[
        'original_structural_failure_scope','paired_accounting_and_admission_preservation','ready_replica_scope_closure'])
    reasons=[i['reason'] for i in value['issues']]
    if case=='transition_failure':
        assert 'accounting_boundary_transition_unresolved' in reasons
        assert 'admission_source_composition_unresolved' in reasons
    elif case=='source_failure':
        assert 'source_specialization_unresolved' in reasons
        assert 'admission_source_composition' not in value['canonical_facts']
    elif case in ('pair_failure','same_checkpoint'):
        assert 'admission_pair_unresolved' in reasons
        assert value['canonical_facts']['admission_source_composition']['paired_preservation_proven'] is False
    else:
        assert value['canonical_facts']['admission_pair']['paired_preservation_proven'] is True
        assert value['canonical_facts']['admission_source_composition']['paired_preservation_proven'] is False
    if calls:
        assert calls[0]['linked']['topology_path'].endswith('accounting_boundary_linked')
        assert calls[0]['routed']['topology_path'].endswith('accounting_boundary')
