"""C1 typed inherited no-clock predicates; no route acceptance or mutation.
The caller must bind inputs to independently verified same-shell checkpoints,
reviewed capture helpers, complete report inventories and unchanged rule settings.
This does not classify protected CDC/pulse evidence or unrelated findings.
"""
# Generic typed relationships only; no C2 artifact policy or acceptance state.
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PIN_KEYS = ('NAME', 'REF_PIN_NAME', 'DIRECTION', 'IS_CLOCK')
CELL_KEYS = ('NAME', 'REF_NAME', 'IS_SEQUENTIAL', 'LOC', 'BEL')
RELATIONS = ('pin_selection', 'owners', 'clocks', 'roots', 'root_selection', 'root_owners', 'root_clocks')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_no_clock(shell, current):
    """Compare typed normalized maps keyed by actual TIMING-17 clock pin.
    Values: finding(CHECK, severity, waived), relations, and pin/owner/root/
    root_owner property dictionaries. Missing CLASS/pin REF_NAME is allowed only
    because exact typed get_pins/get_cells relationships are mandatory here.
    Every collection is an actual observed list, never a default empty list.
    """
    issues = []; normalized = {}
    def issue(reason, pin): issues.append((reason, pin))
    for kind, data in (('shell', shell), ('current', current)):
        result = {}
        if not data: issue(kind + ':empty_inventory', '')
        for pin, row in sorted(data.items()):
            prefix = kind + ':'
            if not isinstance(pin, str) or pin.startswith('WRAPPER/CL/') or '<hidden>' in pin:
                issue(prefix+'not_visible_shell_pin', pin)
            finding = row.get('finding', {})
            if finding.get('CHECK') != 'TIMING-17': issue(prefix+'wrong_rule', pin)
            if finding.get('severity', '').upper() not in ('CRITICAL WARNING', 'WARNING'):
                issue(prefix+'missing_or_unexpected_severity', pin)
            if finding.get('waived') is not False: issue(prefix+'waiver_state_unproved', pin)
            if row.get('capture_errors') or row.get('incomplete_relations'):
                issue(prefix+'incomplete_capture', pin)
            relations = {}
            for name in RELATIONS:
                value = row.get(name)
                if not isinstance(value, list) or any(not isinstance(n, str) for n in value) or len(value) != len(set(value)):
                    issue(prefix+'missing_or_ambiguous_'+name, pin); continue
                relations[name] = value
            if relations.get('pin_selection') != [pin]: issue(prefix+'pin_selection', pin)
            if relations.get('clocks') != []: issue(prefix+'pin_clock_not_empty', pin)
            owners = relations.get('owners', [])
            roots = relations.get('roots', [])
            root_owners = relations.get('root_owners', [])
            if len(owners) != 1 or pin.rsplit('/',1)[0] != owners[0]: issue(prefix+'owner_identity', pin)
            if len(roots) != 1 or relations.get('root_selection') != roots: issue(prefix+'root_selection', pin)
            if len(root_owners) != 1 or len(roots) != 1 or roots[0].rsplit('/',1)[0] != root_owners[0]: issue(prefix+'root_owner_identity', pin)
            for names in (owners, roots, root_owners):
                if any(n.startswith('WRAPPER/CL/') or '<hidden>' in n for n in names): issue(prefix+'non_shell_relationship', pin)
            props = {}
            expected = {'pin': pin, 'owner': owners[0] if len(owners)==1 else '', 'root': roots[0] if len(roots)==1 else '', 'root_owner': root_owners[0] if len(root_owners)==1 else ''}
            for group, name in expected.items():
                value = row.get(group+'_properties', {})
                keys = PIN_KEYS if group in ('pin','root') else CELL_KEYS
                if not isinstance(value, dict) or any(k not in value for k in keys):
                    issue(prefix+'missing_'+group+'_properties', pin); continue
                prop = {k: value[k] for k in keys}; props[group] = prop
                if prop['NAME'] != name: issue(prefix+'wrong_'+group+'_name', pin)
                if group in ('pin','root') and name.rsplit('/',1)[-1] != prop['REF_PIN_NAME']:
                    issue(prefix+'wrong_library_pin', pin)
                if group in ('owner','root_owner') and not prop['REF_NAME']: issue(prefix+'missing_reference', pin)
                if group=='pin' and (prop['DIRECTION']!='IN' or prop['IS_CLOCK']!='1'): issue(prefix+'not_clock_input', pin)
                if group=='root' and prop['DIRECTION']!='OUT': issue(prefix+'not_root_output', pin)
            result[pin] = {'finding':finding,'relations':relations,'properties':props}
        normalized[kind] = result
    if set(shell) != set(current): issue('different_pin_sets', '')
    for pin in sorted(set(shell)&set(current)):
        if normalized['shell'][pin] != normalized['current'][pin]: issue('different_relationship_or_property', pin)
    return {'schema':1,'kind':'typed_inherited_no_clock_predicate','predicate_satisfied':not issues,
            'route_qualified':False,'requires_external_provenance':True,'pins':len(current),
            'canonical_shell_sha256':digest(normalized['shell']),'canonical_current_sha256':digest(normalized['current']),
            'unresolved_count':len(issues),'unresolved_reasons':dict(Counter(k for k,v in issues)),
            'unresolved_samples':issues[:16],'unresolved_sha256':digest(issues),
            'scope':'Visible inherited TIMING-17 evidence only; same-shell provenance/rule and waiver invariants must be verified by caller. No CDC or pulse acceptance.'}


def parse_evidence(m, paths):
    findings={};rules={};queries={};objects={};conflicts=set();errors=[]
    for path in paths:
        for row in m.records(path):
            if not row:continue
            if row[0]=='finding' and row[1]=='methodology':
                fields=m.pairs(row[2:]);props=m.td(fields['properties']);rule=m.td(fields['rule_properties'])
                if props.get('CHECK')=='TIMING-17':
                    name=props['NAME'];m.require(name not in findings,'Duplicate finding');findings[name]=(props,rule)
            elif row[0]=='rule_configuration':
                value=m.td(row[2]);key=(row[1],value['NAME']);m.require(key not in rules or rules[key]==value,'Conflicting rule setting');rules[key]=value
            elif row[0]=='query' and 'TIMING-17#' in row[1]:
                key=m.label(row[1]);value=m.pairs(row[2:]);queries.setdefault(key,[]).append(value)
                if value.get('status')!='observed':errors.append(key)
            elif row[0]=='incomplete' and 'TIMING-17#' in row[1]:errors.append(m.label(row[1]))
            elif row[0]=='object_properties':
                # Object identity is (type, NAME), never NAME alone. Nets and
                # pins can legitimately share NAME but differ in CLASS/IS_CLOCK.
                key=m.label(row[1])
                if len(key)<2 or key[-2] not in ('pin','cell'):continue
                value=m.properties(row[2]);name=value.get('NAME');kind=key[-2]
                m.require(name==key[-1],'Typed property identity differs')
                prior=objects.setdefault((kind,name),{})
                for prop,v in value.items():
                    if prop in prior and prior[prop]!=v:conflicts.add((kind,name))
                    prior[prop]=v
    return dict(findings=findings,rules=rules,queries=queries,objects=objects,conflicts=conflicts,errors=errors)


def normalize(m, evidence, no_waivers):
    result={}
    for finding,(props,rule) in evidence['findings'].items():
        base=('methodology','finding',finding)
        def observed(*tail):
            records=evidence['queries'].get(base+tail,[])
            if not records or any(r.get('status')!='observed' for r in records):return None
            values=[m.tcl_list(r['value']) for r in records]
            m.require(all(v==values[0] for v in values),'Conflicting typed relationship')
            return values[0]
        related=observed('related_pins')
        m.require(related and len(related)==1,'TIMING-17 lacks unique actual related pin')
        pin=related[0];m.require(pin not in result,'Duplicate canonical no-clock pin')
        owners=observed('owning_cells');roots=observed('fanin_roots');root_owners=observed('root','owning_cells')
        owner=owners[0] if owners and len(owners)==1 else ''
        root=roots[0] if roots and len(roots)==1 else ''
        root_owner=root_owners[0] if root_owners and len(root_owners)==1 else ''
        identities=[('pin',pin),('cell',owner),('pin',root),('cell',root_owner)]
        waived=no_waivers and props.get('IS_WAIVED','0').lower() in ('0','false','') and not props.get('WAIVER','')
        row={'finding':{'CHECK':props.get('CHECK'),'severity':props.get('SEVERITY',rule.get('SEVERITY','')).upper(),'waived':False if waived else None},
             'pin_selection':observed('pin_selection',pin),'owners':owners,'clocks':observed('associated_clocks'),'roots':roots,
             'root_selection':observed('root','pin_selection',root),'root_owners':root_owners,'root_clocks':observed('root','associated_clocks'),
             'capture_errors':[list(k) for k in evidence['errors'] if k[:3]==base],
             'incomplete_relations':[list(k) for k in identities if k in evidence['conflicts']]}
        for group,key in zip(('pin','owner','root','root_owner'),identities):row[group+'_properties']=evidence['objects'].get(key,{})
        result[pin]=row
    return result


def no_waivers(path):
    return bool(re.search(r'(^|\n)No waivers found\.\s*$',Path(path).read_text()))



def review_raw(parser, shell_objects, current_objects, supplemental, shell_waivers, current_waivers):
    shell = parse_evidence(parser, [shell_objects])
    current = parse_evidence(parser, [current_objects, supplemental])
    result = check_no_clock(normalize(parser, shell, no_waivers(shell_waivers)), normalize(parser, current, no_waivers(current_waivers)))
    result['rule_settings_equal'] = bool(shell['rules']) and shell['rules'] == current['rules']
    result['rule_count'] = len(current['rules'])
    result['rule_settings_sha256'] = digest(sorted((list(k),v) for k,v in current['rules'].items()))
    result['predicate_satisfied'] = result['predicate_satisfied'] and result['rule_settings_equal']
    return result
