"""Bounded pure C1 native-topology replay; no Tcl execution or route acceptance.

The caller supplies independently trusted capture and checkpoint hashes. These
narrow predicates prove the source-backed linked shape only. Routed rewrites,
reset/status, timing and native-ready equivalence require separate evidence.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat

MAX_FILE = 128 << 20
MAX_RECORD = 32 << 20
ROOT = 'WRAPPER/CL/'
SC = ROOT + 'u_mem/f2_mem_i/memory_sc/inst/'
PROPERTIES = {
    'pin': {'NAME', 'REF_PIN_NAME', 'DIRECTION', 'IS_CLOCK', 'IS_INVERTED'},
    'cell': {'NAME', 'REF_NAME', 'ORIG_REF_NAME', 'IS_PRIMITIVE', 'IS_SEQUENTIAL',
             'IS_BLACKBOX', 'INIT', 'ORIG_CELL_NAME', 'PHYS_OPT_MODIFIED',
             'IS_C_INVERTED', 'IS_D_INVERTED', 'IS_R_INVERTED', 'IS_CLR_INVERTED'},
    'net': set(),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def tcl_list(text):
    """Parse list data only. No eval, variable substitution or Tcl interpreter."""
    require(len(text) <= MAX_RECORD, 'oversized Tcl record')
    result, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        word, quoted, depth = [], text[i] == '"', 0
        if text[i] == '{':
            depth = 1; i += 1
            while i < len(text) and depth:
                char = text[i]
                if char == '\\' and i + 1 < len(text):
                    word.extend(text[i:i+2]); i += 2; continue
                if char == '{': depth += 1
                if char == '}': depth -= 1
                require(depth < 100, 'Tcl nesting bound')
                if depth: word.append(char)
                i += 1
            require(depth == 0, 'unbalanced Tcl braces')
        else:
            if quoted: i += 1
            while i < len(text) and (quoted or not text[i].isspace()):
                char = text[i]
                if quoted and char == '"':
                    i += 1; quoted = False; break
                if char == '\\':
                    require(i + 1 < len(text), 'trailing Tcl escape')
                    char = {'n': '\n', 'r': '\r', 't': '\t'}.get(text[i+1], text[i+1]); i += 1
                word.append(char); i += 1
            require(not quoted, 'unclosed Tcl quote')
        require(i == len(text) or text[i].isspace(), 'invalid Tcl separator')
        result.append(''.join(word))
    return result


def pairs(words, keys=None):
    require(len(words) % 2 == 0, 'odd dictionary')
    result = {}
    for key, value in zip(words[::2], words[1::2]):
        require(key not in result, 'duplicate dictionary key: ' + key)
        result[key] = value
    if keys is not None: require(set(result) == set(keys), 'unexpected dictionary schema')
    return result


def td(text, keys=None): return pairs(tcl_list(text), keys)
def names(text):
    values = tcl_list(text)
    require(len(values) == len(set(values)) and '...' not in values and all(values), 'duplicate/abbreviated names')
    return values

def number(text):
    require(re.fullmatch(r'0|[1-9][0-9]*', text), 'invalid nonnegative integer')
    return int(text)


def records(path):
    """Linear framing; a multiline record is parsed once, not after every line."""
    chunks, size, mode, depth, escaped, count = [], 0, 'start', 0, False, 0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            chunks.append(line); size += len(line)
            require(size <= MAX_RECORD, 'logical record bound')
            for char in line:
                if escaped: escaped = False; continue
                if char == '\\':
                    escaped = True
                    if mode == 'start': mode = 'bare'
                    continue
                if mode == 'brace':
                    if char == '{': depth += 1
                    elif char == '}':
                        depth -= 1
                        if not depth: mode = 'bare'
                    require(depth < 100, 'record nesting bound')
                elif mode == 'quote':
                    if char == '"': mode = 'bare'
                elif char.isspace(): mode = 'start'
                elif mode == 'start':
                    if char == '{': mode, depth = 'brace', 1
                    elif char == '"': mode = 'quote'
                    else: mode = 'bare'
            if mode not in ('brace', 'quote') and not escaped:
                count += 1; require(count <= 1000000, 'record count bound')
                yield tcl_list(''.join(chunks)); chunks, size = [], 0
    require(not chunks, 'unterminated record')


def identity(path, expected):
    path = Path(path)
    require(re.fullmatch('[0-9a-f]{64}', expected or ''), 'invalid capture hash')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'linked capture')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= MAX_FILE, 'unsafe capture file')
    before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(digest == expected, 'capture hash mismatch')
    after = path.stat()
    require(before == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'capture changed')
    return {'sha256': digest, 'bytes': info.st_size}


class Evidence:
    def __init__(self):
        self.headers = []; self.summaries = []; self.queries = {}; self.objects = {}
        self.unavailable = set(); self.networks = {}; self.edges = {}; self.disabled = {}
        self.receives = {}; self.cells = {}; self.selections = []; self.incomplete = []
        self.issues = []; self.counts = Counter(); self.seen_summary = False

    def issue(self, reason, where=''):
        require(len(self.issues) < 10000, 'issue bound')
        self.issues.append({'reason': reason, 'where': str(where)})

    def store(self, table, key, value):
        if key in table and table[key] != value: self.issue('conflicting_duplicate', key)
        else: table[key] = value

    def add(self, w):
        require(w, 'empty record')
        tag = w[0]; self.counts[tag] += 1
        require(not self.seen_summary, 'record after summary')
        require(tag == 'header' or self.headers, 'record before header')
        if tag == 'header':
            require(not self.headers and sum(self.counts.values()) == 1, 'repeated/noninitial header')
            self.headers.append(pairs(w[1:], ('schema','kind','checkpoint_sha256','checkpoint','route_qualified','hardware_measured')))
        elif tag == 'summary':
            self.summaries.append(pairs(w[1:], ('queries','incomplete_or_error','output_byte_cap','route_qualified','hardware_measured')))
            self.seen_summary = True
        elif tag == 'query':
            label = tuple(tcl_list(w[1])); row = pairs(w[2:], ('status','total','names'))
            require(row['status'] == 'observed', 'invalid query status')
            value = names(row['names']); total = number(row['total'])
            if total != len(value): self.issue('incomplete_query_inventory', label)
            self.store(self.queries, label, value)
        elif tag == 'property':
            require(len(w) >= 6 and w[1] in PROPERTIES and w[3] in PROPERTIES[w[1]], 'unknown typed property')
            kind, name, key = w[1:4]; row = pairs(w[4:]); obj = self.objects.setdefault((kind,name), {})
            require(row in ({'status':'unavailable'},) or set(row) == {'status','value'} and row['status'] == 'observed', 'property status schema')
            token = (kind,name,key)
            if row['status'] == 'unavailable':
                if key in obj: self.issue('conflicting_property_availability', token)
                self.unavailable.add(token)
            else:
                if token in self.unavailable: self.issue('conflicting_property_availability', token)
                if key == 'NAME' and row['value'] != name: self.issue('property_identity_mismatch', token)
                self.store(obj, key, row['value'])
        elif tag == 'selection':
            require(w[1] in ('pin','net','cell'), 'unknown selection kind')
            row = pairs(w[3:], ('status','observed'))
            require(row['status'] == 'missing_or_ambiguous', 'unknown selection status')
            names(row['observed']); self.selections.append((w[1],w[2],row))
        elif tag == 'incomplete':
            row = pairs(w[2:]); require('reason' in row, 'missing incomplete reason')
            self.incomplete.append((w[1],row)); self.issue('incomplete_capture', (w[1],row))
        elif tag == 'cell_pins':
            row = pairs(w[2:], ('names',)); self.store(self.cells,w[1],names(row['names']))
        elif tag == 'network':
            require(w[1] in ('pin','net'), 'unknown network kind')
            row = pairs(w[3:], ('nets','leaves','drivers','ports'))
            self.store(self.networks,(w[1],w[2]),{k:names(v) for k,v in row.items()})
        elif tag == 'leaf':
            require(w[1] in ('pin','net'), 'unknown leaf kind')
            row = pairs(w[3:], ('pin','properties','owners'))
            td(row['properties']); names(row['owners'])
        elif tag == 'cone_edge':
            label = tuple(tcl_list(w[1])); row = pairs(w[2:], ('cell','input','network'))
            self.store(self.edges,(label,row['cell'],row['input']),self.input_record(row['network']))
        elif tag == 'disabled_write':
            row = pairs(w[2:], ('channel','network')); pc = number(w[1])
            require(pc < 8 and row['channel'] in ('aw','w'), 'disabled write scope')
            require((pc,row['channel']) not in self.disabled, 'duplicate disabled write row')
            self.disabled[(pc,row['channel'])] = self.input_record(row['network'])
        elif tag == 'receive_candidate':
            row = pairs(w[2:], ('cell','output','endpoints')); pc = number(w[1])
            require(pc < 8, 'receive scope')
            self.store(self.receives,(pc,row['cell'],row['output']),names(row['endpoints']))
        elif tag == 'cone_terminal': pairs(w[2:], ('pin','cell','reference'))
        elif tag == 'cone':
            row = pairs(w[2:], ('visited','terminals')); td(row['visited']); td(row['terminals'])
        elif tag == 'anchor':
            row = pairs(w[2:], ('kind','name')); require(row['kind'] in ('pin','net'), 'unknown anchor kind')
        else: raise ValueError('unknown record kind: ' + tag)

    @staticmethod
    def input_record(text):
        row = td(text)
        if not row: return {}
        require(set(row) == {'scope','input','properties','nets','drivers','ports'} and row['scope'] == 'selected_input_and_drivers', 'input observation scope')
        return {**row,'properties':td(row['properties']),**{k:names(row[k]) for k in ('nets','drivers','ports')}}

    def query(self, *label):
        require(tuple(label) in self.queries, 'missing query: '+str(label))
        return self.queries[tuple(label)]

    def prop(self, kind, name, key):
        value = self.objects.get((kind,name),{}).get(key)
        require(value is not None and value != '', 'missing mandatory property: '+str((kind,name,key)))
        return value

    def selected(self, kind, name):
        require(self.query('select',kind,name) == [name], 'not uniquely selected: '+name)

    def pin(self, name, direction):
        self.selected('pin',name)
        require(self.prop('pin',name,'NAME') == name and self.prop('pin',name,'DIRECTION') == direction, 'pin identity/direction: '+name)
        require(self.query('pin_owner',name) == [name.rsplit('/',1)[0]], 'pin ownership: '+name)

    def primitive(self, name, reference):
        self.selected('cell',name)
        require(self.prop('cell',name,'NAME') == name and self.prop('cell',name,'REF_NAME') == reference, 'primitive reference: '+name)
        require(self.prop('cell',name,'IS_PRIMITIVE').lower() in ('1','true') and self.prop('cell',name,'IS_BLACKBOX').lower() in ('0','false'), 'opaque/nonprimitive cell: '+name)

    def input(self, row, expected):
        require(row.get('input') == expected and row.get('scope') == 'selected_input_and_drivers', 'missing input witness: '+expected)
        self.selected('pin',expected)
        require(self.prop('pin',expected,'NAME') == expected and self.prop('pin',expected,'DIRECTION') == 'IN', 'input identity/direction')
        for key,value in row['properties'].items(): require(self.prop('pin',expected,key) == value, 'input property disagreement')
        require(row['nets'] and row['nets'] == self.query('input_nets',expected), 'input nets disagree')
        drivers, ports = set(), set()
        for net in row['nets']:
            drivers.update(self.query('input_drivers',expected,net)); ports.update(self.query('input_ports',expected,net))
        require(set(row['drivers']) == drivers and not ports and not row['ports'] and len(drivers) == 1, 'input driver/port inventory')
        driver = next(iter(drivers)); self.pin(driver,'OUT')
        return driver


def disabled_review(e):
    require(set(e.disabled) == {(pc,ch) for pc in range(8) for ch in ('aw','w')}, 'missing/extra disabled write inputs')
    facts = []
    for (pc,ch),row in sorted(e.disabled.items()):
        si = f's{pc+1:02d}'; pin = f'{SC}{si}_entry_pipeline/{si}_mmu/inst/{ch}_sreg/s_axi_{ch}valid'
        driver = e.input(row,pin); owner = driver.rsplit('/',1)[0]
        e.primitive(owner,'GND')
        require(e.prop('pin',driver,'REF_PIN_NAME') == 'G', 'constant output is not GND/G')
        facts.append({'pc':pc,'channel':ch,'input':pin,'nets':row['nets'],'driver':driver,'reference':'GND','scope':'selected_input_and_drivers'})
    return facts


def endpoint_family(pc, endpoint, reference):
    require(endpoint.startswith(ROOT), 'receive endpoint outside CL')
    name = endpoint[len(ROOT):]; family = None; expected = 'FDRE'
    if name == 'u_board/adapter/error_reg/D': family = 'error'
    elif m := re.fullmatch(r'u_native([0-7])/(lr|ln|lp)_reg\[([0-9]+)\]/(CE|D)',name):
        port,reg,bit,terminal = m.groups()
        if int(port)==pc and int(bit)<(8 if reg=='ln' else 7) and (terminal=='CE' or reg=='ln'): family='native_'+reg
    elif m := re.fullmatch(r'u_native([0-7])/lengths_reg_(0_63|64_127)_0_4/RAM[A-H]/WE',name):
        if int(m[1])==pc: family='native_lengths'; expected='RAMD64E'
    elif m := re.fullmatch(r'u_board/adapter/ports\[([0-7])\]\.(pc_peak|outstanding|tr|pc_returned)_reg\[([0-7])\]\[([0-9]+)\](_rep(__[0-9]+)?)?/(CE|D)',name):
        port,reg,index,bit,replica,_,terminal=m.groups()
        if int(port)==pc and int(index)==pc and int(bit)<{'pc_peak':32,'outstanding':7,'tr':6,'pc_returned':32}[reg] and (not replica or reg=='tr') and (terminal=='CE' or reg=='pc_peak'): family='port_'+reg
    elif m := re.fullmatch(r'u_board/adapter/engines\[([0-9]+)\]\.(.+)',name):
        engine,tail=int(m[1]),m[2]
        if engine//2==pc:
            if n := re.fullmatch(r'data_q_reg_([0-7])/WEBWE\[([0-7])\]',tail):
                ram,bit=int(n[1]),int(n[2])
                if ram!=7 or bit<4: family=f'engine_{engine}_data';expected='RAMB18E2' if ram==7 else 'RAMB36E2'
            elif n := re.fullmatch(r'(queued|wp)_reg\[([0-9]+)\]\[([0-9]+)\]/(CE|D)',tail):
                reg,index,bit,terminal=n.groups()
                if int(index)==engine and int(bit)<(6 if reg=='queued' else 5) and (terminal=='CE' or reg=='queued'): family=f'engine_{engine}_{reg}'
            elif n := re.fullmatch(r'tag_q_reg\[([0-9]+)\]\[(slot|row|group_id|generation)\]\[([0-9]+)\]/CE',tail):
                entry,field,bit=n.groups()
                if int(entry)<32 and int(bit)<{'slot':2,'row':32,'group_id':10,'generation':32}[field]: family=f'engine_{engine}_tag'
    require(family is not None and reference==expected, 'unreviewed receive sink: '+endpoint+' ('+reference+')')
    return family


def required_endpoints(pc):
    values = {ROOT+'u_board/adapter/error_reg/D'}
    for reg,width in (('tr',6),('outstanding',7),('pc_returned',32)):
        values.update(f'{ROOT}u_board/adapter/ports[{pc}].{reg}_reg[{pc}][{bit}]/CE' for bit in range(width))
    for reg,width in (('lr',7),('ln',8),('lp',7)):
        values.update(f'{ROOT}u_native{pc}/{reg}_reg[{bit}]/CE' for bit in range(width))
    for engine in (2*pc,2*pc+1):
        for ram in range(8):
            values.update(f'{ROOT}u_board/adapter/engines[{engine}].data_q_reg_{ram}/WEBWE[{bit}]' for bit in range(4 if ram==7 else 8))
    return values


def receive_review(e):
    facts=[]
    for pc in range(8):
        net=f'{ROOT}n{pc}_rvalid'; e.selected('net',net)
        network=e.networks.get(('net',net),{})
        require(network.get('nets')==[net] and not network.get('ports'), 'missing/escaping native RVALID network')
        leaves=e.query('leaves',net)
        require(set(network['leaves'])==set(leaves) and not e.query('ports',net), 'RVALID leaf/port inventory mismatch')
        drivers=[pin for pin in leaves if e.prop('pin',pin,'DIRECTION')=='OUT']
        si=f's{pc+1:02d}'; source=f'{SC}{si}_entry_pipeline/{si}_mmu/inst/r_sreg/'
        require(len(drivers)==1 and network['drivers']==drivers and drivers[0].startswith(source), 'wrong RVALID source channel')
        e.pin(drivers[0],'OUT')
        require(e.prop('cell',drivers[0].rsplit('/',1)[0],'REF_NAME') not in ('GND','VCC'), 'constant RVALID')
        suffix='' if pc==0 else f'__{pc-1}'
        relocated={f'{ROOT}u_mem/reads[7]_i_1{suffix}/I0',f'{ROOT}u_mem/n_err_i_1{suffix}/I1'}
        gate_inputs=[]
        require(any(pin.startswith(ROOT+f'u_native{pc}/') and e.prop('pin',pin,'DIRECTION')=='IN' for pin in leaves), 'missing native RVALID receiver')
        for pin in leaves:
            if pin in drivers: continue
            e.pin(pin,'IN')
            if pin.startswith(ROOT+f'u_native{pc}/') or pin.startswith(source): continue
            if pin in relocated:
                require(re.fullmatch('LUT[1-6]',e.prop('cell',pin.rsplit('/',1)[0],'REF_NAME')), 'relocated input is not LUT')
                continue
            gate_inputs.append(pin)
        require(len(gate_inputs)==1, 'unsupported/missing/multiple receive gates')
        gate_input=gate_inputs[0]; gate=gate_input.rsplit('/',1)[0]
        require(gate.startswith(ROOT+'u_mem/') and gate_input==gate+'/I0', 'wrong receive gate location/input')
        e.primitive(gate,'LUT2'); require(e.prop('cell',gate,'INIT').lower()=="4'h2", 'receive truth table differs')
        require(set(e.cells.get(gate,[]))=={gate+'/I0',gate+'/I1',gate+'/O'}, 'receive LUT pin inventory')
        for suffix,direction in (('I0','IN'),('I1','IN'),('O','OUT')):
            pin=gate+'/'+suffix;e.pin(pin,direction)
            require(e.prop('pin',pin,'REF_PIN_NAME')==suffix, 'receive pin reference mismatch')
            require(e.prop('pin',pin,'IS_INVERTED').lower() in ('0','false'), 'unsupported receive pin inversion')
        label=('receive',str(pc),gate)
        rvalid=e.input(e.edges.get((label,gate,gate+'/I0'),{}),gate+'/I0')
        reset=e.input(e.edges.get((label,gate,gate+'/I1'),{}),gate+'/I1')
        require(rvalid==drivers[0] and reset==ROOT+'core_rst_reg/Q', 'receive gate input mapping')
        e.primitive(ROOT+'core_rst_reg','FDRE')
        endpoints=e.receives.get((pc,gate,gate+'/O'))
        require(endpoints and endpoints==e.query('receive_endpoints',str(pc),gate+'/O'), 'missing/changed complete receive fanout')
        families=Counter();references={}
        for endpoint in endpoints:
            require(e.prop('pin',endpoint,'DIRECTION')=='IN','receive endpoint direction')
            owner=endpoint.rsplit('/',1)[0]
            require(e.query('pin_owner',endpoint)==[owner], 'receive endpoint ownership')
            reference=e.prop('cell',owner,'REF_NAME');references[owner]=reference
            families[endpoint_family(pc,endpoint,reference)]+=1
        require(required_endpoints(pc)<=set(endpoints), 'missing required receive state/payload sink')
        for engine in (2*pc,2*pc+1):
            require(all(f'engine_{engine}_{reg}' in families for reg in ('queued','wp','tag')), 'missing paired-engine bookkeeping sink')
        facts.append({'pc':pc,'gate':gate,'init':"4'h2",'rvalid_driver':rvalid,'reset_driver':reset,
                      'endpoints':sorted(endpoints),'cell_references':references,'sink_families':dict(sorted(families.items())),
                      'equation':'RVALID && !core_rst','scope':'source_backed_linked_receive_shape'})
    return facts


def review_artifact(*, topology_path, topology_sha256, checkpoint_sha256, checkpoint_path=None):
    require(re.fullmatch('[0-9a-f]{64}',checkpoint_sha256 or ''), 'invalid checkpoint hash')
    record=identity(topology_path,topology_sha256); e=Evidence()
    try:
        for w in records(topology_path): e.add(w)
        require(len(e.headers)==len(e.summaries)==1,'missing header/summary')
        header=e.headers[0];summary=e.summaries[0]
        require(header['schema']=='1' and header['kind']=='c1_native_topology' and header['checkpoint_sha256']==checkpoint_sha256 and header['route_qualified']==header['hardware_measured']=='0','capture header binding')
        require(header['checkpoint'].startswith('/') and (checkpoint_path is None or header['checkpoint']==checkpoint_path),'checkpoint path binding')
        require(summary['route_qualified']==summary['hardware_measured']=='0','collector claims acceptance')
        require(number(summary['queries'])==e.counts['query']+sum(row[1]['reason']=='api_error' and 'property' not in row[0] and row[0]!='capture' for row in e.incomplete),'query count differs')
        if number(summary['incomplete_or_error']) or number(summary['output_byte_cap']): e.issue('incomplete_summary',summary)
        if e.incomplete and number(summary['incomplete_or_error'])==0: e.issue('incomplete_markers_disagree_with_summary')
    except (ValueError,KeyError,IndexError,UnicodeError) as exc: e.issue('malformed_capture',str(exc))
    facts={}
    if not e.issues:
        for name,predicate in (('disabled_write_inputs',disabled_review),('receive_gates',receive_review)):
            try: facts[name]=predicate(e)
            except (ValueError,KeyError,IndexError) as exc: e.issue(name+'_unresolved',str(exc))
    require(identity(topology_path,topology_sha256)==record,'capture changed during replay')
    return {'schema':1,'kind':'c1-native-narrow-semantic-review','capture':record,'checkpoint_sha256':checkpoint_sha256,
            'issues':e.issues,'canonical_facts':facts,'unavailable_properties':[list(v) for v in sorted(e.unavailable)],
            'missing_selections':e.selections,'pending_reviews':['current_routed_native_ready_equivalence','current_host_wready_equivalence','current_reset_status','physical_timing_and_exceptions'],
            'route_qualified':False,'hardware_measured':False}
