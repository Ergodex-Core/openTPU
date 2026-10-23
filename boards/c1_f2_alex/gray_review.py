#!/usr/bin/env python3
"""C1 bounded pure gray evidence predicates; never qualify an entire image.
Supply hashes obtained from the trusted external SSM receipt. All three design
inputs must be from the same externally verified checkpoint; this tool cannot
infer that provenance from arbitrary file contents. Vendor hash identifies the
actual installed script reviewed for its source-period/min-period formulas.
"""
# Reuses generic vendor/topology predicates only; no C2 artifact disposition or
# historical success is imported. Every C1 result requires fresh C1 evidence.
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import stat

MAX_FILE = 256 << 20
MAX_RECORD = 16 << 20
VENDOR_SHA = '5b15cbd342000de8a719ff79e6e9532d5123b59d2912cfa48453d6662e29cea6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tcl_list(text):
    result, i = [], 0
    require(len(text) <= MAX_RECORD, 'Tcl record exceeds bound')
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        word, quoted, depth = [], text[i] == '"', 0
        if text[i] == '{':
            depth = 1
            i += 1
            while i < len(text) and depth:
                char = text[i]
                if char == '\\' and i + 1 < len(text):
                    word.extend(text[i:i+2]); i += 2; continue
                if char == '{': depth += 1
                if char == '}': depth -= 1
                require(depth < 100, 'Tcl nesting exceeds bound')
                if depth: word.append(char)
                i += 1
            require(depth == 0, 'Unbalanced Tcl braces')
        else:
            if quoted: i += 1
            while i < len(text) and (quoted or not text[i].isspace()):
                char = text[i]
                if quoted and char == '"':
                    i += 1; quoted = False; break
                if char == '\\':
                    require(i + 1 < len(text), 'Trailing Tcl escape')
                    char = {'n': '\n', 'r': '\r', 't': '\t'}.get(text[i+1], text[i+1]); i += 1
                word.append(char); i += 1
            require(not quoted, 'Unclosed Tcl quote')
        require(i == len(text) or text[i].isspace(), 'Malformed Tcl separator')
        result.append(''.join(word))
    return result


def pairs(words):
    require(len(words) % 2 == 0, 'Odd dictionary')
    result = {}
    for k, v in zip(words[::2], words[1::2]):
        require(k not in result, 'Duplicate dictionary key: ' + k)
        result[k] = v
    return result


def td(text):
    return pairs(tcl_list(text))


def records(path):
    # Frame each character exactly once; parse each complete record exactly once.
    # A multiline pulse report therefore does not cause quadratic re-parsing.
    chunks, size, mode, depth, escaped, count = [], 0, 'start', 0, False, 0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            chunks.append(line); size += len(line)
            require(size <= MAX_RECORD, 'Oversized logical record')
            for ch in line:
                if escaped:
                    escaped = False; continue
                if ch == '\\':
                    escaped = True
                    if mode == 'start': mode = 'bare'
                    continue
                if mode == 'brace':
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if not depth: mode = 'bare'
                    require(depth < 100, 'Record nesting exceeds bound')
                elif mode == 'quote':
                    if ch == '"': mode = 'bare'
                elif ch.isspace():
                    mode = 'start'
                elif mode == 'start':
                    if ch == '{': mode, depth = 'brace', 1
                    elif ch == '"': mode = 'quote'
                    else: mode = 'bare'
            if mode not in ('brace', 'quote') and not escaped:
                count += 1
                require(count <= 1000000, 'Record count exceeds bound')
                yield tcl_list(''.join(chunks))
                chunks, size = [], 0
    require(not chunks, 'Unterminated final record')


def label(text):
    parts = tcl_list(text)
    for _ in range(12):
        if not parts or parts[0] in ('gray', 'gray_ancestor'): return tuple(parts)
        nested = tcl_list(parts[0])
        if len(nested) <= 1: return tuple(parts)
        parts = nested + parts[1:]
    raise ValueError('Label nesting exceeds bound')


def properties(text):
    result = {}
    for name, item in td(text).items():
        item = td(item)
        require(item.get('ok') == '1' and 'value' in item, 'Failed captured property')
        result[name] = item['value']
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path, wanted):
    p = Path(path); s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and 0 < s.st_size <= MAX_FILE, 'Unsafe/empty/oversized input')
    with p.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(re.fullmatch('[0-9a-f]{64}', wanted) and h == wanted, 'Input hash mismatch: ' + str(p))
    return {'path': str(p), 'bytes': s.st_size, 'sha256': h}


def number(value):
    d = Decimal(value)
    require(d.is_finite(), 'Nonfinite numeric evidence')
    return d


def rounded_match(observed, expected):
    # Export/path strings display at least three decimals. Never allow >0.0005ns.
    value = number(observed)
    places = max(3, -value.as_tuple().exponent)
    return abs(value - expected) <= Decimal(5).scaleb(-places-1)


class Evidence:
    def __init__(self):
        self.props, self.queries, self.drivers, self.stages, self.timing = {}, {}, {}, {}, {}
        self.groups, self.ancestors, self.relations = {}, {}, {}
        self.headers, self.summaries, self.coverage = {}, {}, {}
        self.counts = Counter(); self.issues = []

    def issue(self, reason, where=''):
        require(len(self.issues) < 100000, 'Unresolved count exceeds bound')
        self.issues.append((reason, str(where)))

    def put(self, target, key, value):
        if key in target:
            self.issue('duplicate_record', key)
        else: target[key] = value

    def read(self, path, kind):
        queries = 0; pending = 0
        for row in records(path):
            if not row: continue
            tag = row[0]; self.counts[kind + '/' + tag] += 1
            if tag == 'schema': self.put(self.headers, kind, pairs(row))
            elif tag == 'query':
                queries += 1; data = pairs(row[2:]); key = label(row[1])
                if data.get('status') != 'observed':
                    pending += 1; self.issue('query_error', key)
                if key == ('gray_destination_stage_inventory',) or (key[:1] == ('gray',) and key[-1] in ('associated_clocks', 'owning_cells')):
                    self.put(self.queries, key, data.get('value', ''))
            elif tag == 'incomplete':
                pending += 1; self.issue('incomplete_capture', row[1:])
            elif tag == 'object_properties':
                key = label(row[1])
                if key[:1] in (('gray',), ('gray_ancestor',)):
                    self.put(self.props, key, properties(row[2]))
            elif tag in ('data_driver', 'gray_stage_inventory', 'gray_bit_timing'):
                key = label(row[1]); data = pairs(row[2:])
                target = {'data_driver': self.drivers, 'gray_stage_inventory': self.stages, 'gray_bit_timing': self.timing}[tag]
                if tag == 'gray_bit_timing': key += (data['user_ignored'],)
                self.put(target, key, data)
            elif tag == 'vendor_gray_bus':
                self.put(self.groups, row[1], td(row[2]))
            elif tag == 'gray_ancestor_relation': self.put(self.relations, row[1], pairs(row[2:]))
            elif tag == 'gray_ancestor_coverage': self.put(self.coverage, kind, pairs(row[1:]))
            elif tag in ('supplemental_summary', 'diagnostic_summary'):
                self.put(self.summaries, kind, td(row[1]))
        summary = self.summaries.get(kind, {})
        if summary.get('queries') != str(queries) or summary.get('incomplete_or_error') != str(pending):
            self.issue('summary_count_mismatch', kind)
        if not self.groups and kind == 'ancestors': self.issue('empty_ancestor_groups')
        require(len(self.groups) <= 1024 and len(self.props) <= 100000, 'Evidence inventory exceeds bound')

    def getprop(self, key):
        if key not in self.props: self.issue('missing_properties', key)
        return self.props.get(key, {})

    def data_pin(self, base, pin, direction):
        prop = self.getprop(base + ('pin', pin))
        if (prop.get('NAME'), prop.get('REF_PIN_NAME'), prop.get('DIRECTION')) != (pin, pin[-1], direction):
            self.issue('data_pin_identity', base)
        if tcl_list(self.queries.get(base + ('owning_cells',), '')) != [pin[:-2]]:
            self.issue('data_pin_owner', base)

    def clock(self, base, pin):
        p = self.getprop(base + ('pin', pin))
        if (p.get('NAME'), p.get('REF_PIN_NAME'), p.get('DIRECTION'), p.get('IS_CLOCK')) != (pin, 'C', 'IN', '1'):
            self.issue('clock_pin_identity', base)
        if tcl_list(self.queries.get(base + ('owning_cells',), '')) != [pin[:-2]]:
            self.issue('clock_pin_owner', base)
        c = self.getprop(base + ('clock',))
        names = tcl_list(self.queries.get(base + ('associated_clocks',), ''))
        if len(names) != 1 or names != [c.get('NAME')]: self.issue('clock_collection', base)
        try:
            period = number(c['PERIOD']); require(period > 0, 'Nonpositive period')
        except (KeyError, ValueError, InvalidOperation):
            self.issue('clock_period', base); return ('', Decimal(0))
        return (c.get('NAME', ''), period)

    def review(self, exported, checkpoint):
        if self.headers.get('ancestors', {}).get('checkpoint_sha256') != checkpoint:
            self.issue('ancestor_checkpoint_mismatch')
        coverage = self.coverage.get('ancestors', {})
        if coverage.get('skipped') != '0' or coverage.get('selected') != coverage.get('attempted'):
            self.issue('ancestor_coverage')
        cells = {}
        for key, p in self.props.items():
            if len(key) >= 2 and key[-2] == 'cell':
                name = key[-1]
                if name in cells and cells[name] != p: self.issue('conflicting_cell_properties', name)
                cells[name] = p
        expected_stage_cells = set()
        canonical = []; normal_counts = Counter(); ignored_counts = Counter(); refs = Counter()
        expected_bases = set(); used_paths = set()
        for root, group in sorted(self.groups.items()):
            source, dest = td(group['source']), td(group['destination'])
            if not source or set(source) != set(dest) or sorted(map(int, source)) != list(range(len(source))): self.issue('group_bit_inventory', root)
            relation = self.relations.get(root, {})
            marker = '/gen_xpm_memory_fifo.inst_fifo/'
            expected_fifo = root.rsplit(marker, 1)[0] + marker[:-1] if marker in root else ''
            if not expected_fifo or relation.get('fifo_candidate') != expected_fifo or relation.get('smartconnect_candidate') != 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc':
                self.issue('ancestor_exact_hierarchy', root)
            for name, role in ((root, 'gray_cdc'), (relation.get('fifo_candidate', ''), 'xpm_fifo'), (relation.get('smartconnect_candidate', ''), 'smartconnect')):
                found = [(k, p) for k, p in self.props.items() if k[:2] == ('gray_ancestor', name)]
                if len(found) != 1: self.issue('ancestor_identity_missing', name); continue
                key, prop = found[0]; refs[(role, prop.get('ORIG_REF_NAME', ''))] += 1
                if prop.get('NAME') != name or prop.get('IS_BLACKBOX') != '0' or role not in tcl_list(key[-1]):
                    self.issue('ancestor_identity', name)
                expected_original = {'gray_cdc': 'xpm_cdc_gray', 'xpm_fifo': 'sc_util_v1_0_4_xpm_memory_fifo', 'smartconnect': 'f2_mem_memory_sc_0'}[role]
                if prop.get('ORIG_REF_NAME') != expected_original or not prop.get('REF_NAME'):
                    self.issue('ancestor_reference_family', name)
            if td(relation.get('source_cells', '')) != source or td(relation.get('destination_cells', '')) != dest:
                self.issue('ancestor_group_mismatch', root)
            root_clocks = set()
            for bit in sorted(source, key=int):
                base = ('gray', root, 'bit', bit); expected_bases.add(base)
                src, dst = source[bit], dest.get(bit, '')
                if src != root + '/src_gray_ff_reg[' + bit + ']' or dst != root + '/dest_graysync_ff_reg[0][' + bit + ']': self.issue('bit_cell_name', base)
                stages = td(self.stages.get(base, {}).get('stages', ''))
                expected_stage_cells.update(stages.values())
                if sorted(map(int, stages)) != list(range(len(stages))) or len(stages) < 2: self.issue('stage_inventory', base)
                self.data_pin(base + ('source',), src + '/Q', 'OUT')
                self.data_pin(base + ('first_destination',), dst + '/D', 'IN')
                sc = self.clock(base + ('source_clock',), src + '/C'); dc = None
                for stage, cell in sorted(stages.items(), key=lambda x: int(x[0])):
                    if cell != root + '/dest_graysync_ff_reg[' + stage + '][' + bit + ']': self.issue('stage_cell_name', base)
                    self.data_pin(base + ('stage', stage, 'data'), cell + '/D', 'IN')
                    self.data_pin(base + ('stage', stage, 'output'), cell + '/Q', 'OUT')
                    cp = cells.get(cell, {})
                    if cp.get('NAME') != cell or cp.get('IS_SEQUENTIAL') != '1' or cp.get('ASYNC_REG', '').lower() not in ('1', 'true'): self.issue('stage_primitive_properties', cell)
                    clock = self.clock(base + ('stage', stage, 'clock'), cell + '/C')
                    if dc is None: dc = clock
                    elif clock != dc: self.issue('stage_clock_mismatch', base)
                    previous = src if stage == '0' else stages.get(str(int(stage)-1), '')
                    for driver_key in [base + ('stage', stage)] + ([base + ('first_destination',)] if stage == '0' else []):
                        d = self.drivers.get(driver_key, {})
                        if d.get('driver_count') != '1' or tcl_list(d.get('drivers', '')) != [previous + '/Q']: self.issue('data_driver_mismatch', driver_key)
                if cells.get(src, {}).get('IS_SEQUENTIAL') != '1': self.issue('source_not_sequential', base)
                if dc is None: dc = ('', Decimal(0))
                if not sc[0] or not dc[0] or sc[0] == dc[0]: self.issue('distinct_clock_witness', base)
                root_clocks.add((sc[0], str(sc[1]), dc[0], str(dc[1])))
                ex = exported.get(root, {})
                for command, expected in (('set_max_delay', sc[1]), ('set_bus_skew', min(sc[1], dc[1]))):
                    if command not in ex or not rounded_match(ex[command]['value'], expected): self.issue('vendor_export_bound', (root, command))
                path_facts = []
                for ignored in ('0', '1'):
                    summary = self.timing.get(base + (ignored,), {})
                    paths = sorted([(k, p) for k, p in self.props.items() if k[:4] == base and len(k) == 9 and k[4:7] == ('path', 'user_ignored', ignored)], key=lambda x: int(x[0][-1]))
                    used_paths.update(k for k, _ in paths)
                    count = len(paths); (normal_counts if ignored == '0' else ignored_counts)[count] += 1
                    if summary.get('returned') != str(count) or summary.get('capped') != '0' or summary.get('max_paths') != '16' or summary.get('nworst') != '16' or count >= 16 or (ignored == '0' and not count): self.issue('path_count_or_cap', (base, ignored))
                    if [k[-1] for k, _ in paths] != [str(i) for i in range(count)]: self.issue('path_index_inventory', base)
                    if summary.get('expected_source') != src+'/Q' or summary.get('expected_source_clock_pin') != src+'/C' or summary.get('expected_destination') != dst+'/D': self.issue('timing_query_identity', base)
                    if ignored == '1' and count: self.issue('ignored_paths_need_review', base)
                    for key, p in paths:
                        if p.get('STARTPOINT_PIN') not in (src+'/C', src+'/Q') or p.get('ENDPOINT_PIN') != dst+'/D': self.issue('path_endpoint_relation', key)
                        if (p.get('STARTPOINT_CLOCK'), p.get('ENDPOINT_CLOCK')) != (sc[0], dc[0]): self.issue('path_clock_relation', key)
                        match = re.fullmatch(r'MaxDelay Path ([0-9.]+)ns -datapath_only', p.get('EXCEPTION', ''))
                        if not match or not rounded_match(match[1], sc[1]) or not rounded_match(p.get('REQUIREMENT', 'NaN'), sc[1]): self.issue('effective_max_delay', key)
                        for name in ('SLACK', 'DATAPATH_DELAY'): number(p[name])
                        path_facts.append((ignored, key[-1], p))
                canonical.append({'root': root, 'bit': bit, 'source': src, 'stages': stages, 'source_clock': [sc[0], str(sc[1])], 'destination_clock': [dc[0], str(dc[1])], 'paths': path_facts})
            if len(root_clocks) != 1: self.issue('group_clock_mismatch', root)
        observed_stage_cells = tcl_list(self.queries.get(('gray_destination_stage_inventory',), ''))
        if len(observed_stage_cells) != len(set(observed_stage_cells)) or set(observed_stage_cells) != expected_stage_cells:
            self.issue('global_stage_cell_inventory')
        if set(self.stages) != expected_bases or set(self.timing) != {b+(i,) for b in expected_bases for i in ('0','1')}:
            self.issue('global_bit_inventory_mismatch')
        all_paths = {k for k in self.props if len(k) == 9 and k[:1] == ('gray',) and k[4] == 'path'}
        if all_paths != used_paths: self.issue('orphan_path_records')
        if set(exported) != set(self.groups): self.issue('export_group_inventory')
        return {'schema': 1, 'kind': 'offline_gray_evidence_review', 'route_qualified': False, 'hardware_measured': False,
                'gray_evidence_consistent': not self.issues, 'checkpoint_sha256': checkpoint,
                'groups': len(self.groups), 'bits': len(canonical), 'normal_path_count_histogram': dict(normal_counts), 'ignored_path_count_histogram': dict(ignored_counts),
                'unique_ancestor_cells': sum(k[:1] == ('gray_ancestor',) for k in self.props), 'ancestor_reference_histogram': [{'role': k[0], 'original_ref': k[1], 'uses': v} for k,v in sorted(refs.items())],
                'canonical_gray_sha256': digest(canonical), 'canonical_export_sha256': digest(exported), 'canonical_ancestor_sha256': digest(sorted((list(k),v) for k,v in self.props.items() if k[:1] == ('gray_ancestor',))), 'unresolved_count': len(self.issues), 'unresolved_reasons': dict(Counter(k for k,v in self.issues)),
                'unresolved_samples': self.issues[:16], 'unresolved_sha256': digest(sorted(self.issues)),
                'record_counts': dict(self.counts), 'remaining': ['Final routed effective path/slack and bus-skew evidence', 'Independent shell, CDC waiver, pulse, DRC and physical qualification', 'Installed vendor source is current intent evidence, not retroactive synthesis attestation', 'Input checkpoint co-provenance remains an external receipt obligation']}


def export_gray(path):
    source = scope = ''; result = {}
    selectors = r'-from \[get_cells (?:\{src_gray_ff_reg\*\}|src_gray_ff_reg\*)\] -to \[get_cells \{dest_graysync_ff_reg\[0\]\*\}\]'
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        m = re.fullmatch(r"# Constraints from file : '([^']+)'", line)
        if m: source = m[1]; continue
        if line.startswith('current_instance'):
            words = tcl_list(line); scope = '' if words == ['current_instance', '-quiet'] else words[-1]; continue
        if source != 'xpm_cdc_gray.tcl' or not line or line.startswith('#'): continue
        if line.startswith(('set_max_delay', 'set_bus_skew')):
            m = re.fullmatch(r'(set_max_delay -datapath_only|set_bus_skew) ' + selectors + r' ([0-9.]+)', line)
            require(m is not None and scope, 'Unrecognized gray export clause: ' + line)
            command = m[1].split()[0]
            require(command not in result.setdefault(scope, {}), 'Duplicate gray export clause')
            result[scope][command] = {'value': m[2], 'source': source}
    require(bool(result), 'No gray export constraints')
    return result



def read_objects(e, path, checkpoint):
    binding=None;rows={}
    for row in records(path):
        if not row: continue
        if row[0]=='checkpoint':
            require(binding is None and len(row)==4 and row[2]=='sha256','Malformed/duplicate checkpoint identity')
            binding={'path':row[1],'sha256':row[3]}
        elif row[0]=='vendor_gray_bus': e.put(e.groups,row[1],td(row[2]))
        elif row[0]=='bus_skew':
            require(row[1] not in rows,'Duplicate bus-skew ID');rows[row[1]]=td(row[2])
    require(binding and binding['sha256']==checkpoint,'Final objects checkpoint mismatch')
    require(e.groups and rows,'Missing final groups or bus-skew inventory')
    # Normalize only already-observed identities into the linked reviewer's input
    # slots. This does not fabricate a separate ancestor diagnostic or its receipt.
    e.headers['ancestors']={'checkpoint_sha256':binding['sha256']}
    e.coverage['ancestors']=e.coverage.get('supplemental',{})
    return binding,rows


def final_gray_checks(e,rows):
    issues=[];matches=[];negative=[]
    for key,prop in e.props.items():
        if len(key)==9 and key[:1]==('gray',) and key[4:7]==('path','user_ignored','0'):
            if number(prop['SLACK'])<0:negative.append(key)
    if negative:issues.append(('negative_final_gray_paths',str(len(negative))))
    for root,group in sorted(e.groups.items()):
        source=td(group['source']);destination=td(group['destination'])
        wanted_from=sorted(source.values());wanted_to=sorted(destination.values())
        found=[(ident,row) for ident,row in rows.items() if sorted(tcl_list(row['from']))==wanted_from and sorted(tcl_list(row['to']))==wanted_to]
        if len(found)!=1:issues.append(('final_bus_skew_exact_coverage',root));continue
        ident,row=found[0];bit=sorted(source,key=int)[0];base=('gray',root,'bit',bit)
        sp=number(e.props[base+('source_clock','clock')]['PERIOD']);dp=number(e.props[base+('stage','0','clock','clock')]['PERIOD'])
        requirement=number(row['requirement']);actual=number(row['actual']);slack=number(row['slack'])
        if requirement<=0 or not rounded_match(row['requirement'],min(sp,dp)):issues.append(('final_bus_skew_vendor_bound',root))
        # Vivado reports signed relative-delay skew (including relative CRPR).
        # A negative actual skew is valid when the reported slack is MET.
        if slack<0 or actual>requirement:issues.append(('final_bus_skew_physical_failure',root))
        if abs((requirement-actual)-slack)>Decimal('.001'):
            issues.append(('final_bus_skew_slack_inconsistent',root))
        matches.append({'root':root,'id':ident,'row':row,'derived_bound_ns':str(min(sp,dp))})
    return {'issues':issues,'negative_path_count':len(negative),'negative_path_sha256':digest(negative),'matched_gray_bus_skew_rows':len(matches),
            'remaining_non_gray_bus_skew_rows':len(rows)-len(matches),'canonical_gray_bus_skew_sha256':digest(matches)}



def review_final(objects, supplemental, export, vendor, checkpoint):
    require(hashlib.sha256(Path(vendor).read_bytes()).hexdigest() == VENDOR_SHA, 'Unreviewed vendor gray source')
    evidence = Evidence()
    evidence.read(supplemental, 'supplemental')
    binding, rows = read_objects(evidence, objects, checkpoint)
    result = evidence.review(export_gray(export), checkpoint)
    physical = final_gray_checks(evidence, rows)
    evidence.issues.extend(physical.pop('issues'))
    result.update(kind='final_gray_predicate', gray_evidence_consistent=not evidence.issues,
                  unresolved_count=len(evidence.issues), unresolved_reasons=dict(Counter(k for k,v in evidence.issues)),
                  unresolved_samples=evidence.issues[:16], unresolved_sha256=digest(sorted(evidence.issues)),
                  final_gray_checks=physical, checkpoint_binding=binding)
    return result
