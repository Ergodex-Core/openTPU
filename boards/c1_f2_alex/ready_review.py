"""Pure replay of bounded C1 ready observations, never route qualification.

This validates collection completeness and canonical circuit facts. Paired
LINK/routed transition equivalence requires actual observations and remains
pending; no C2 register counts, LUT signatures or removed-state rules apply.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import native_review as N


class Evidence(N.Evidence):
    def __init__(self):
        super().__init__()
        self.ready = {key: {} for key in ('root','edge','terminal','walk','control','clock','register','candidates')}
        self.sources = []; self.hosts = []

    def add(self, w):
        if not w or not w[0].startswith('ready_'):
            return super().add(w)
        N.require(self.headers and not self.seen_summary, 'ready record outside capture')
        tag = w[0][6:]; self.counts[w[0]] += 1
        if tag == 'sources':
            self.sources.append(N.pairs(w[1:], ('helper_sha256','native_helper_sha256')))
            return
        if tag == 'host_candidates':
            self.hosts.append(N.names(N.pairs(w[1:], ('pins',))['pins']))
            return
        fields = {
            'root': ('pin',), 'edge': ('cell','input','network'),
            'terminal': ('output','cell','reference'),
            'walk': ('outputs','registers','expand_controls'),
            'control': ('cell','role','network'), 'clock': ('cell','network'),
            'register': ('cell','outputs','clock_scope'),
            'candidates': ('scope','all','selected','selection'),
        }
        N.require(tag in fields, 'unknown ready record')
        if tag == 'candidates':
            row=N.pairs(w[1:],fields[tag]); key=row['scope']
        else:
            row=N.pairs(w[2:],fields[tag]); key=w[1]
            if tag in ('edge','terminal','control','clock','register'):
                key=(w[1],row['cell'])
                if tag == 'edge': key += (row['input'],)
                if tag == 'terminal': key += (row['output'],)
                if tag == 'control': key += (row['role'],)
        if 'network' in row: row['network']=self.input_record(row['network'])
        for name in ('outputs','registers','all','selected'):
            if name in row: row[name]=N.names(row[name])
        self.store(self.ready[tag],key,row)


def network(e,pin):
    e.selected('pin',pin)
    row=e.networks.get(('pin',pin))
    N.require(row is not None and row['nets'], 'missing output/root network: '+pin)
    N.require(row['nets']==e.query('segments',pin), 'network net selection mismatch')
    leaves=set(); ports=set()
    for net in row['nets']:
        e.selected('net',net)
        leaves.update(e.query('leaves',net));ports.update(e.query('ports',net))
    N.require(set(row['leaves'])==leaves and set(row['ports'])==ports, 'incomplete network leaves/ports')
    drivers=set()
    for leaf in leaves:
        direction=e.prop('pin',leaf,'DIRECTION'); e.pin(leaf,direction)
        N.require(direction in ('IN','OUT'), 'unknown leaf direction')
        if direction=='OUT': drivers.add(leaf)
    N.require(set(row['drivers'])==drivers and len(drivers)==1, 'ambiguous network driver')
    return row


def facts(e):
    expected={f'native {pc} {channel}' for pc in range(8) for channel in ('b','r')}|{'host_wready'}
    N.require(set(e.ready['root'])==expected, 'missing/extra ready roots')
    N.require(len(e.hosts)==1 and len(e.hosts[0])==1, 'ambiguous host anchor')
    host=e.hosts[0][0]
    N.require(host in (N.ROOT+'u_pcis_pipe/m_axi_wready',N.ROOT+'u_pcis_pipe/m_axi_wready[0]'), 'unexpected host anchor')
    roots={}
    for label,row in e.ready['root'].items():
        if label=='host_wready': expected_pin=host
        else:
            _,pc,channel=label.split();si=f's{int(pc)+1:02d}'
            expected_pin=f'{N.SC}{si}_entry_pipeline/{si}_mmu/inst/{channel}_sreg/s_axi_{channel}ready'
        N.require(row['pin']==expected_pin, 'ready root identity mismatch')
        roots[label]=network(e,row['pin'])
        N.require(label in e.ready['walk'] and e.ready['walk'][label]['expand_controls']=='1', 'missing root cone')
        N.require(set(roots[label]['drivers'])<=set(e.ready['walk'][label]['outputs']), 'root driver omitted')
    graphs={}
    for label,walk in e.ready['walk'].items():
        N.require(walk['expand_controls'] in ('0','1') and walk['outputs'], 'invalid ready walk')
        observed=set(walk['outputs']);nodes={}
        for output in walk['outputs']:
            e.pin(output,'OUT');owner=e.query('pin_owner',output)[0]
            ref=e.prop('cell',owner,'REF_NAME');e.primitive(owner,ref)
            props=e.objects[('cell',owner)]
            node={'output':output,'cell':owner,'properties':props}
            if re.fullmatch('LUT[1-6]',ref):
                init=e.prop('cell',owner,'INIT')
                N.require(re.fullmatch(r"[0-9]+'[hHbB][0-9a-fA-F]+",init), 'invalid LUT INIT')
                inputs=e.query('ready_inputs',label,owner)
                N.require(len(inputs)==int(ref[-1]), 'LUT input inventory')
                node['inputs']={}
                for pin in inputs:
                    edge=e.ready['edge'].get((label,owner,pin))
                    N.require(edge is not None, 'missing ready edge')
                    driver=e.input(edge['network'],pin)
                    N.require(driver in observed, 'cone driver omitted')
                    N.require(e.prop('pin',pin,'IS_INVERTED') in ('0','1','false','true','FALSE','TRUE'), 'unknown input inversion')
                    node['inputs'][pin]=edge['network']
                node['output_network']=network(e,output)
            else:
                terminal=e.ready['terminal'].get((label,owner,output))
                N.require(terminal is not None and terminal['reference']==ref, 'missing terminal identity')
                N.require(ref in ('GND','VCC') or e.prop('cell',owner,'IS_SEQUENTIAL').lower() in ('1','true'), 'unsupported terminal')
            nodes[output]=node
        graphs[label]={'nodes':nodes,'registers':walk['registers'],'expand_controls':walk['expand_controls']}
        if walk['expand_controls']=='1':
            for reg in walk['registers']:
                N.require((label,reg) in e.ready['register'], 'missing register controls')
    registers=[]
    for (label,reg),r in e.ready['register'].items():
        N.require(r['clock_scope']=='names_and_properties_only', 'register clock scope')
        e.primitive(reg,'FDRE')
        N.require(r['outputs']==e.query('ready_register_outputs',label,reg) and r['outputs'], 'register output inventory')
        outputs={output:network(e,output) for output in r['outputs']}
        controls={}
        for role in ('D','CE','R','C'):
            control=e.ready['clock'].get((label,reg)) if role=='C' else e.ready['control'].get((label,reg,role))
            N.require(control is not None, 'missing FDRE control: '+role)
            driver=e.input(control['network'],reg+'/'+role)
            inversion=e.prop('pin',reg+'/'+role,'IS_INVERTED')
            N.require(inversion in ('0','1','false','true','FALSE','TRUE'), 'missing FDRE polarity')
            controls[role]={'network':control['network'],'inversion':inversion,'clocks':e.query('pin_clock',reg+'/'+role)}
            if role!='C':
                matching=[key for key in graphs if N.tcl_list(key)==['control',label,reg,role]]
                N.require(len(matching)==1 and graphs[matching[0]]['expand_controls']=='0' and
                          driver in graphs[matching[0]]['nodes'], 'missing control cone')
        registers.append({'label':label,'cell':reg,'outputs':outputs,'controls':controls})
    # Candidate capture is explicitly discovery by scoped name, not a claim that
    # arbitrarily renamed routed replicas are absent.
    candidates={}
    N.require(set(e.ready['candidates'])=={N.SC+'s00_entry_pipeline/s00_mmu/inst/w_sreg',N.ROOT+'u_pcis_pipe'}, 'host discovery scope incomplete')
    for scope,row in e.ready['candidates'].items():
        N.require(row['selection']=='name_contains_ready', 'unknown candidate scope')
        N.require(row['all']==e.query('ready_host_register_inventory',scope), 'candidate inventory mismatch')
        N.require(row['selected']==[name for name in row['all'] if 'ready' in name.lower()], 'candidate selection mismatch')
        for reg in row['selected']:
            N.require(any(cell==reg and N.tcl_list(label)==['host_candidate',reg]
                          for label,cell in e.ready['register']), 'missing selected candidate')
        candidates[scope]=row
    return {'roots':roots,'graphs':graphs,'registers':registers,'candidate_discovery':candidates,
            'limitation':'Observed ready graph and controls only; sequential equivalence and renamed-replica closure are pending.'}


def review_artifact(*,topology_path,topology_sha256,checkpoint_sha256,helper_sha256,native_helper_sha256):
    result={'route_qualified':False,'hardware_measured':False,'issues':[],'canonical_facts':{},
            'pending_reviews':['paired_link_routed_ready_equivalence','renamed_replica_scope_closure']}
    try:
        bound=N.identity(topology_path,topology_sha256)
        e=Evidence()
        for record in N.records(topology_path): e.add(record)
        N.require(N.identity(topology_path,topology_sha256)==bound,'capture changed')
        N.require(len(e.headers)==1 and len(e.summaries)==1 and len(e.sources)==1,'missing capture envelope')
        h=e.headers[0];s=e.summaries[0]
        N.require(h['schema']=='1' and h['kind']=='c1_ready_topology' and h['checkpoint_sha256']==checkpoint_sha256,'checkpoint/schema mismatch')
        N.require(re.fullmatch('[0-9a-f]{64}',checkpoint_sha256 or ''),'invalid checkpoint hash')
        N.require(all(re.fullmatch('[0-9a-f]{64}',digest or '') for digest in (helper_sha256,native_helper_sha256)), 'invalid collector hash')
        N.require(h['route_qualified']==h['hardware_measured']==s['route_qualified']==s['hardware_measured']=='0','invalid evidence claim')
        N.require(e.sources[0]==dict(helper_sha256=helper_sha256,native_helper_sha256=native_helper_sha256),'collector source mismatch')
        N.require(N.number(s['queries'])==e.counts['query'] and s['incomplete_or_error']==s['output_byte_cap']=='0','incomplete capture summary')
        N.require(not e.issues and not e.selections,'capture errors or missing selections')
        result['capture']=bound;result['canonical_facts']=facts(e)
        result['facts_sha256']=hashlib.sha256(json.dumps(result['canonical_facts'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result['unavailable_properties']=[list(row) for row in sorted(e.unavailable)]
    except (ValueError,OSError,KeyError,TypeError) as exc:
        result['issues'].append({'reason':str(exc)})
    return result
