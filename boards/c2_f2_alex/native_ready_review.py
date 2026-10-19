"""Read-only, bounded LUT equivalence replay; does not grant qualification."""
from pathlib import Path
import importlib.util
import json
import re

def _dependencies():
    spec = importlib.util.spec_from_file_location('c2_native_reset_helpers', Path(__file__).with_name('reset_status_review.py'))
    module = importlib.util.module_from_spec(spec)
    import sys
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module.file_record, module.g


file_record, g = _dependencies()

RESET = 'WRAPPER/CL/core_rst_reg/Q'
OBSERVER = 'WRAPPER/CL/u_status_cdc/src_ff_reg[0]/D'
BANK_SCOPE = re.compile(r'^WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s0([1-8])_entry_pipeline/s0\1_mmu/inst/[br]_sreg/')
SCOPE = re.compile(r'^WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s0([1-8])_entry_pipeline/s0\1_mmu/inst/([br])_sreg/(m_vector_i\[1024\]|skid2vector_q|state\[(m_valid_i|s_ready_i|s_stall_d)\])_i_1__\d+$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(values, message):
    require(len(values) == len(set(values)) and '...' not in values, message)
    return values


def read(path, checkpoint, kind, expected_sha=None):
    before = file_record(path)
    if expected_sha is not None:
        require(before['sha256'] == expected_sha, 'Raw capture SHA mismatch')
    data = {k: {} for k in ('cells', 'pins', 'cell_pins', 'fanout', 'inventories')}
    headers = []; complete = []
    for count, w in enumerate(g.records(Path(path)), 1):
        require(count < 100000, 'Record bound')
        tag = w[0]
        if tag == 'schema': headers.append(g.pairs(w))
        elif tag == 'complete': complete.append(g.pairs(w[1:]))
        elif tag in ('cell', 'pin', 'cell_pins', 'fanout'):
            key = {'cell': 'cells', 'pin': 'pins'}.get(tag, tag)
            item = g.pairs(w[2:])
            if 'properties' in item: item['properties'] = g.td(item['properties'])
            require(w[1] not in data[key] or data[key][w[1]] == item, f'Conflicting {tag}: {w[1]}')
            data[key][w[1]] = item
        elif tag in ('comparison_cells', 'input_inverters', 'focus_cells'):
            require(tag not in data['inventories'], 'Duplicate inventory')
            data['inventories'][tag] = unique(g.tcl_list(g.pairs(w[1:])['names']), 'Bad inventory')
        elif tag in ('boundary_pin', 'polarity_inventory', 'polarity_cell', 'polarity_pin'):
            require(kind == 'narrow_routed_structure', 'Unexpected supplemental record')
        else: raise ValueError(f'Unknown/error/incomplete record: {tag}')
    require(headers == [{'schema':'1','kind':kind,'checkpoint_sha256':checkpoint,'route_qualified':'0'}], 'Checkpoint/header mismatch')
    require(complete == [{'cells':str(len(data['cells'])),'route_qualified':'0'}], 'Completion inventory mismatch')
    require(len(data['cells']) <= 4096, 'Cell bound')
    require(file_record(path) == before, 'Capture changed during read')
    data['input'] = before
    return data


def properties(data, cell):
    p = data['cells'][cell]['properties']
    require(p.get('NAME') == cell and p.get('CLASS') == 'cell', 'Cell identity mismatch')
    return p


def pin(data, cell, name, direction):
    full = cell + '/' + name
    p = data['pins'][full]
    pp = p['properties']
    require(pp.get('NAME') == full and pp.get('CLASS') == 'pin' and pp.get('REF_PIN_NAME') == name and pp.get('DIRECTION') == direction, f'Pin identity: {full}')
    require(pp.get('IS_INVERTED') in ('0','1'), f'Missing pin inversion: {full}')
    require(not g.tcl_list(p['ports']), f'Unexpected top-level port: {full}')
    return p


def lut(data, cell):
    p = properties(data, cell)
    m = re.fullmatch(r'LUT([1-6])', p['REF_NAME'])
    require(m is not None, f'Unsupported primitive: {cell}')
    n = int(m[1]); width = 1 << n
    m = re.fullmatch(r"(\d+)'h([0-9A-Fa-f]+)", p['INIT'])
    require(m is not None and int(m[1]) == width and int(m[2],16) < (1 << width), f'Invalid INIT: {cell}')
    expected = {cell + '/O', *(cell + f'/I{i}' for i in range(n))}
    names = unique(g.tcl_list(data['cell_pins'][cell]['names']), 'Bad cell pins')
    require(set(names) == expected, f'Unexpected LUT pins: {cell}')
    inputs = []
    for i in range(n):
        row = pin(data,cell,f'I{i}','IN')
        drivers = unique(g.tcl_list(row['drivers']), 'Bad driver inventory')
        require(len(drivers) == 1, f'Nonunique driver: {cell}/I{i}')
        inputs.append((drivers[0],int(row['properties']['IS_INVERTED'])))
    out = pin(data,cell,'O','OUT')
    require(g.tcl_list(out['drivers']) == [cell+'/O'], 'LUT output driver mismatch')
    leaves = unique(g.tcl_list(data['fanout'][cell+'/O']['names']), 'Bad fanout inventory')
    directions = g.tcl_list(data['fanout'][cell+'/O']['directions'])
    require(len(leaves) == len(directions) and leaves.count(cell+'/O') == 1, 'Incomplete fanout')
    require(all(d == ('OUT' if name == cell+'/O' else 'IN') for name,d in zip(leaves,directions)), 'Unexpected output network direction')
    return {'init':int(m[2],16),'inputs':inputs,'output_inverted':int(out['properties']['IS_INVERTED']), 'sinks': sorted(set(leaves)-{cell+'/O'})}


def truth(lut_value, variables):
    index = sum((variables[name] ^ inverted) << i for i,(name,inverted) in enumerate(lut_value['inputs']))
    return ((lut_value['init'] >> index) & 1) ^ lut_value['output_inverted']


def compare_cell(before, after, allowed_removed=()):
    require(set(allowed_removed) <= set(before['sinks']), 'Unobserved removed sink allowance')
    require(set(before['sinks']) - set(allowed_removed) == set(after['sinks']), 'Output sink identity changed')
    variables = sorted({x[0] for x in before['inputs']} | {x[0] for x in after['inputs']})
    require(len(variables) <= 6, 'Comparison variable bound')
    require({x[0] for x in before['inputs']} == {x[0] for x in after['inputs']}, 'Input driver identity changed')
    for bits in range(1 << len(variables)):
        assignment = {name:(bits>>i)&1 for i,name in enumerate(variables)}
        require(truth(before,assignment) == truth(after,assignment), f'Truth table mismatch: {assignment}')
    return {'variables':variables,'assignments':1<<len(variables),'sinks':before['sinks'], 'before_init':before['init'],'after_init':after['init'], 'removed_sinks':sorted(allowed_removed), 'surviving_sinks':after['sinks']}



def reset_anchor(before, after):
    cell = RESET.rsplit('/', 1)[0]
    anchors = []
    for data in (before, after):
        cp = properties(data, cell)
        require(cp.get('REF_NAME') == 'FDRE', 'Unsupported reset anchor primitive')
        require(cp.get('INIT') in ("1'b0", "1'b1"), 'Missing reset anchor INIT')
        names = unique(g.tcl_list(data['cell_pins'][cell]['names']), 'Reset anchor pins')
        require(set(names) == {cell+'/'+p for p in ('C','CE','D','Q','R')}, 'Reset anchor inventory')
        facts = {'REF_NAME':cp['REF_NAME'], 'INIT':cp['INIT'], 'pins':{}}
        for ref in ('C','CE','D','Q','R'):
            row = pin(data, cell, ref, 'OUT' if ref=='Q' else 'IN')
            drivers = unique(g.tcl_list(row['drivers']), 'Reset anchor driver inventory')
            require(len(drivers)==1, 'Reset anchor nonunique driver')
            if ref=='Q':require(drivers==[RESET], 'Reset anchor output identity')
            facts['pins'][ref] = {'drivers':drivers, 'inverted':row['properties']['IS_INVERTED'], 'clocks':sorted(unique(g.tcl_list(row['clocks']), 'Reset anchor clocks'))}
        require(facts['pins']['C']['clocks'], 'Reset anchor clock missing')
        anchors.append(facts)
    require(anchors[0]==anchors[1], 'Reset source register identity/configuration changed')
    return anchors[0]



def scope_closure(before, after, selected, inverters):
    selected_inputs = {cell+f'/I{i}' for cell in selected for i in range(len(lut(after,cell)['inputs']))}
    excluded = []
    for cell in inverters:
        for sink in lut(before,cell)['sinks']:
            if sink == OBSERVER:
                excluded.append(sink)
            else:
                require(sink in selected_inputs, f'Unreviewed original ready consumer: {sink}')
                require(g.tcl_list(before['pins'][sink]['drivers']) == [cell+'/O'], 'Inverter fanout/driver disagreement')
    fanout = after['fanout'][RESET]
    names = unique(g.tcl_list(fanout['names']), 'Reset output fanout inventory')
    directions = g.tcl_list(fanout['directions'])
    require(len(names)==len(directions) and names.count(RESET)==1, 'Reset fanout incomplete')
    require(all(d==('OUT' if n==RESET else 'IN') for n,d in zip(names,directions)), 'Reset fanout direction')
    bank_sinks = {n for n in names if BANK_SCOPE.match(n)}
    expected = {n for n in selected_inputs if g.tcl_list(after['pins'][n]['drivers']) == [RESET]}
    require(bank_sinks == expected, 'Unreviewed routed bank reset consumer')
    return {'routed_bank_reset_sinks':sorted(bank_sinks), 'separately_reviewed_status_observer':sorted(set(excluded)), 'scope':'Status observer is excluded from native-ready truth-function claim.'}


def review(before, after, removals=None):
    issues = []; facts = []; anchor = reset_anchor(before, after)
    selected = before['inventories'].get('comparison_cells',[])
    require(len(selected)==80 and set(selected)=={name for name in after['cells'] if SCOPE.fullmatch(name)}, 'Selected 80-cell inventory mismatch')
    scope_counts = {}
    for name in selected:
        m = SCOPE.fullmatch(name); require(m is not None,'Unexpected comparison cell')
        scope_counts.setdefault((m[1],m[2]),set()).add(m[3])
    expected_roles = {'m_vector_i[1024]','skid2vector_q','state[m_valid_i]','state[s_ready_i]','state[s_stall_d]'}
    require(len(scope_counts)==16 and all(v==expected_roles for v in scope_counts.values()), 'Channel/role inventory mismatch')
    inverters = set(before['inventories'].get('input_inverters',[]))
    closure = scope_closure(before, after, selected, inverters)
    used = set()
    for name in selected:
        try:
            a=lut(before,name); b=lut(after,name)
            resolved=[]; substitutions=[]
            for driver, polarity in a['inputs']:
                owner, leaf = driver.rsplit('/',1)
                if owner in inverters:
                    inv=lut(before,owner)
                    require(leaf=='O' and inv['init']==1 and inv['inputs']==[(RESET,0)] and inv['output_inverted']==0, 'Unproven reset inverter')
                    require(any(target.startswith(name+'/I') for target in inv['sinks']), 'Inverter output does not reach selected cell')
                    resolved.append((RESET,polarity^1)); used.add(owner); substitutions.append(driver)
                else: resolved.append((driver,polarity))
            a['inputs']=resolved
            require(any(driver==RESET for driver,_ in a['inputs']) and any(driver==RESET for driver,_ in b['inputs']), 'Reset input missing')
            allowed=([removals['luts'][name]['removed_sink']] if removals and name in removals['luts'] else [])
            fact=compare_cell(a,b,allowed);fact.update(cell=name, substituted_inverters=substitutions);facts.append(fact)
        except (ValueError,KeyError) as error:
            issues.append({'cell':name,'reason':str(error)})
    require(used==inverters, 'Unused/unverified input inverter inventory')
    return {'issues':issues,'canonical_facts':facts,'reset_source_anchor':anchor,'scope_closure':closure,'inputs':{'linked':before['input'],'refined':after['input']},'route_qualified':False,'scope':'Only selected 80 LUT combinational truth functions and output leaf identities; no sequential equivalence claim.'}


def read_removed(path, phase, checkpoint):
    """Read the narrow two-checkpoint removed-register witness without Tcl eval."""
    before = file_record(path)
    data = {k:{} for k in ('cell_selection','pin_selection','cell','cell_pins','clock_pin',
                           'pin','driver_cell','fanout','original_net_selection','original_leaf_selection')}
    headers=[]; selected=[]; complete=[]
    for count,w in enumerate(g.records(Path(path)),1):
        require(count<10000, 'Removed-register record bound')
        tag=w[0]
        if tag=='schema':headers.append(g.pairs(w))
        elif tag=='selected_cells':selected.append(unique(g.tcl_list(g.pairs(w[1:])['names']), 'Removed selection duplicates'))
        elif tag=='complete':complete.append(g.pairs(w[1:]))
        elif tag in data:
            row=g.pairs(w[2:])
            if 'properties' in row:row['properties']=g.td(row['properties'])
            key=(w[1],row['original']) if tag.startswith('original_') else w[1]
            require(key not in data[tag] or (tag=='driver_cell' and data[tag][key]==row), 'Duplicate/conflicting removed-register record')
            data[tag][key]=row
        else:raise ValueError('Unknown/error/incomplete removed-register record: '+tag)
    require(headers==[{'schema':'1','kind':'removed_ready_registers','phase':phase,'checkpoint_sha256':checkpoint,'route_qualified':'0'}], 'Removed-register checkpoint/header mismatch')
    require(len(selected)==1, 'Removed-register selection missing')
    require(complete==[{'cells':str(len(data['cell'])),'route_qualified':'0'}], 'Removed-register completion mismatch')
    require(file_record(path)==before, 'Removed-register witness changed')
    data['selected']=selected[0];data['input']=before
    return data


def unused_removal_review(before, after, linked_path, refined_path, linked_checkpoint, checkpoint):
    """Discharge only exact FDREs whose original Q has no consumer or port.

    This is unused-output elimination, not a constant-register assumption. An
    actual Q consumer, retained final alias, or replacement mapping is outside
    this deliberately small rule and remains a review failure.
    """
    a=read_removed(linked_path,'linked',linked_checkpoint)
    b=read_removed(refined_path,'refined',checkpoint)
    mapping={}
    for name in before['inventories']['comparison_cells']:
        old=set(lut(before,name)['sinks']);new=set(lut(after,name)['sinks'])
        require(not new-old, 'Added native-ready sink is not unused-register removal')
        removed=old-new
        if removed:
            expected=name.rsplit('/',1)[0]+'/m_vector_i_reg[1024]/CE'
            require(removed=={expected} and '/m_vector_i[1024]_' in name, 'Unrecognized removed native-ready sink')
            mapping[expected.rsplit('/',1)[0]]=name
    require(len(mapping)==16, 'Expected exact sixteen unused response-ID registers')
    expected=set(mapping)
    expected_pins={name+'/'+ref for name in expected for ref in ('C','CE','D','Q','R')}
    for data in (a,b):
        require(set(data['selected'])==expected, 'Removed-register selected inventory mismatch')
        require(set(data['cell_selection'])==expected and set(data['pin_selection'])==expected_pins, 'Removed-register selection coverage mismatch')
    require(set(a['cell'])==expected and set(a['cell_pins'])==expected, 'Original removed-register cell coverage')
    require(set(a['clock_pin'])=={n+'/C' for n in expected}, 'Original removed-register clock coverage')
    require(set(a['pin'])==expected_pins-{n+'/C' for n in expected}, 'Original removed-register pin coverage')
    require(set(a['fanout'])=={n+'/Q' for n in expected}, 'Original Q fanout coverage')
    require(not a['original_net_selection'] and not a['original_leaf_selection'], 'Unexpected original selection phase')
    require(not any(b[k] for k in ('cell','cell_pins','clock_pin','pin','driver_cell','fanout')), 'Removed register or replacement still present')
    facts={};net_keys=set();leaf_keys=set()
    for name,source in sorted(mapping.items()):
        require(g.tcl_list(a['cell_selection'][name]['names'])==[name] and not g.tcl_list(b['cell_selection'][name]['names']), 'Cell removal not observed')
        cp=a['cell'][name]['properties']
        require(cp.get('NAME')==name and cp.get('CLASS')=='cell' and cp.get('REF_NAME')=='FDRE' and cp.get('IS_SEQUENTIAL')=='1', 'Removed primitive identity mismatch')
        require(set(unique(g.tcl_list(a['cell_pins'][name]['names']), 'Duplicate original pin'))=={name+'/'+ref for ref in ('C','CE','D','Q','R')}, 'Original primitive pin inventory')
        for ref in ('C','CE','D','Q','R'):
            pn=name+'/'+ref
            require(g.tcl_list(a['pin_selection'][pn]['names'])==[pn] and not g.tcl_list(b['pin_selection'][pn]['names']), 'Pin removal not observed')
            row=a['clock_pin' if ref=='C' else 'pin'][pn];pp=row['properties']
            require(pp.get('NAME')==pn and pp.get('CLASS')=='pin' and pp.get('REF_NAME')=='FDRE' and pp.get('REF_PIN_NAME')==ref and pp.get('DIRECTION')==('OUT' if ref=='Q' else 'IN'), 'Removed primitive pin identity')
        require(g.tcl_list(a['pin'][name+'/CE']['drivers'])==[source+'/O'], 'Removed CE is not driven by compared LUT')
        require(a['driver_cell'][source]['properties']==before['cells'][source]['properties'], 'Original compared LUT changed across witnesses')
        q=name+'/Q';row=a['pin'][q]
        require(g.tcl_list(row['drivers'])==[q] and not g.tcl_list(row['ports']), 'Original unused Q driver/port mismatch')
        leaves=unique(g.tcl_list(a['fanout'][q]['names']), 'Original Q leaf inventory invalid')
        require(leaves==[q] and g.tcl_list(a['fanout'][q]['directions'])==['OUT'], 'Original register Q has an actual consumer')
        nets=unique(g.tcl_list(row['nets']), 'Original Q net inventory invalid')
        require(nets, 'Original Q net inventory missing')
        for net in nets:
            key=(name,net);net_keys.add(key)
            require(key in b['original_net_selection'] and not g.tcl_list(b['original_net_selection'][key]['names']), 'Original Q net still present or not reselected')
        key=(name,q);leaf_keys.add(key)
        require(key in b['original_leaf_selection'] and not g.tcl_list(b['original_leaf_selection'][key]['names']), 'Original Q leaf still present or not reselected')
        facts[source]={'removed_sink':name+'/CE','removed_cell':name,'primitive_properties':cp,
                       'original_q_driver':q,'original_q_leaf_pins':leaves,'original_q_ports':[],
                       'original_q_nets':nets,'final_cell_pins_nets_absent':True,
                       'disposition':'unused_output_register_removed','constant_value_asserted':False}
    require(set(b['original_net_selection'])==net_keys and set(b['original_leaf_selection'])==leaf_keys, 'Final reselection inventory mismatch')
    return {'luts':facts,'inputs':{'linked':a['input'],'refined':b['input']},
            'rule':'Exact original FDRE Q has no other leaf pin and no top-level port; its cell, five pins, and complete Q net aliases are absent in final checkpoint.'}


def review_artifact(*, reports, auxiliary, checkpoint_sha256, linked_checkpoint_sha256):
    """Replay authenticated raw captures; caller binds their hashes to its manifest.

    Cell removal remains an issue until separately captured original and routed
    netlist facts establish a specific valid disposition. No success/count flag
    from a diagnostic is accepted as equivalence proof.
    """
    try:
        for digest in (checkpoint_sha256, linked_checkpoint_sha256):
            require(re.fullmatch('[0-9a-f]{64}', digest or '') is not None, 'Invalid checkpoint digest')
        linked = read(auxiliary['native_ready_linked'], linked_checkpoint_sha256, 'linked_native_ready')
        refined = read(auxiliary['native_ready_refined'], checkpoint_sha256, 'narrow_routed_structure')
        removals=None
        removal_keys=('native_ready_removed_linked','native_ready_removed_refined')
        if any(key in auxiliary for key in removal_keys):
            require(all(key in auxiliary for key in removal_keys), 'Both removed-register witnesses required')
            removals=unused_removal_review(linked,refined,auxiliary[removal_keys[0]],auxiliary[removal_keys[1]],linked_checkpoint_sha256,checkpoint_sha256)
        result = review(linked, refined, removals)
        facts = {key:result[key] for key in ('reset_source_anchor','scope_closure','inputs','scope')}
        facts['luts'] = result['canonical_facts']
        if removals is not None:facts['unused_register_removals']=removals
        facts['checkpoint_sha256'] = checkpoint_sha256
        facts['linked_checkpoint_sha256'] = linked_checkpoint_sha256
        return {'issues':result['issues'], 'canonical_facts':facts, 'route_qualified':False}
    except (ValueError, KeyError, IndexError, OSError) as error:
        return {'issues':[{'reason':'missing_or_malformed_native_ready_evidence','detail':str(error)}],
                'canonical_facts':{}, 'route_qualified':False}
