#!/usr/bin/env python3
"""Pure parsers/predicates for current C2 exception scope; never executes Tcl.

This module reports scoped evidence consistency only. The caller must bind all
texts to one final checkpoint, approved vendor/source provenance and the final
timing/skew/CDC gates. It cannot independently qualify an image.
"""
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import re
import importlib.util
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)


def fixed_table(text, headings):
    """Parse Vivado fixed columns, retaining wrapped field pieces separately."""
    lines=text.splitlines();indices=[];first=None
    for i,line in enumerate(lines):
        if line.startswith('Position  '):
            require(first is None,'Duplicate report table header');first=i
            cursor=0
            for name in headings:
                at=line.find(name,cursor);require(at>=cursor,'Missing/out-of-order table column '+name)
                indices.append(at);cursor=at+len(name)
    require(first is not None and first+1<len(lines),'Missing table')
    require(re.fullmatch(r'[- ]+',lines[first+1]) is not None,'Missing table divider')
    rows=[];row=None;ended=False
    for line in lines[first+2:]:
        if not line.strip():continue
        if line.startswith('Warning: the percentages reported '):ended=True;continue
        require(not ended,'Unexpected data after table note')
        parts=[line[start:indices[n+1] if n+1<len(indices) else None].strip() for n,start in enumerate(indices)]
        if parts[0]:
            require(parts[0].isdigit(),'Unexpected report row prefix')
            row={name:[part] if part else [] for name,part in zip(headings,parts)};rows.append(row)
        else:
            require(row is not None,'Orphan wrapped report line')
            for name,part in zip(headings[1:],parts[1:]):
                if part:row[name].append(part)
    for row in rows:
        row['Position']=int(row['Position'][0])
        for name in headings[1:]:
            row[name]='\n'.join(row[name])
    require(rows,'Empty exception table')
    return rows


def parse_ignored(text):
    rows=fixed_table(text,('Position','From','Through','To','Setup','Hold','Status'))
    known=re.compile(r'Non-existent path|Totally overridden path by DPO [0-9]+')
    return rows,[(r['Position'],r['Status']) for r in rows if not known.fullmatch(r['Status'])]


def parse_coverage(text):
    rows=fixed_table(text,('Position','Type','Setup','Hold','From','Through','To','Endpoints','From (%)','Through (%)','To (%)'))
    result={}
    for row in rows:
        require(row['Position'] not in result,'Duplicate coverage position')
        require(row['Endpoints'].isdigit(),'Missing/noninteger covered endpoints')
        row['Endpoints']=int(row['Endpoints'])
        for field in ('From (%)','Through (%)','To (%)'):
            if row[field]:
                value=Decimal(row[field]);require(value.is_finite() and 0<=value<=100,'Invalid coverage percentage')
        result[row['Position']]=row
    return result


def report_inventory(ignored,coverage):
    rows,unknown=parse_ignored(ignored);cov=parse_coverage(coverage)
    pairs=[];nonexistent=[];unresolved=list(unknown)
    for row in rows:
        m=re.fullmatch(r'Totally overridden path by DPO ([0-9]+)',row['Status'])
        if m:
            target=int(m[1]);old=cov.get(row['Position']);new=cov.get(target)
            if old is None or new is None:
                unresolved.append((row['Position'],'override coverage ID absent'));continue
            pair=dict(old=row['Position'],new=target,old_setup=old['Setup'],new_setup=new['Setup'],
                      old_endpoints=old['Endpoints'],new_endpoints=new['Endpoints'],
                      same_type=old['Type']==new['Type']=='Max Delay Datapath Only',
                      same_setup=old['Setup']==new['Setup'],same_endpoint_count=old['Endpoints']==new['Endpoints'])
            pairs.append(pair)
        elif row['Status']=='Non-existent path':nonexistent.append(row['Position'])
    return dict(ignored_rows=len(rows),coverage_rows=len(cov),status_histogram=dict(Counter(r['Status'].split(' by DPO ')[0] for r in rows)),
                overrides=pairs,nonexistent_positions=nonexistent,unresolved=unresolved,
                route_qualified=False,acceptance='IDs and counts alone do not establish scope or endpoint identity')


def validate_memory_override(old,new,old_coverage,new_coverage,ignored_status,expected_period):
    """Narrow static predicate after original vendor intent/scope is proven.

    `old`/`new` are parsed exported clauses with resolved shared Tcl aliases.
    Through and destination predicates are booleans from exact grammar matching,
    never a hierarchy-prefix or rule-count guess. Endpoint counts are only a
    consistency check: the actual Vivado override relationship and same scoped
    vendor clauses establish replacement coverage.
    """
    issues=[]
    def check(ok,why):
        if not ok:issues.append(why)
    check(old.get('source')==new.get('source')=='xpm_memory_xdc.tcl','vendor_source')
    check(bool(old.get('scope')) and old.get('scope')==new.get('scope'),'same_scope')
    check(old.get('kind')==new.get('kind')=='set_max_delay' and old.get('datapath_only') is True and new.get('datapath_only') is True,'datapath_only')
    check(old.get('from')==new.get('from') and old.get('vendor_clka_leaf_source') is True and new.get('vendor_clka_leaf_source') is True,'exact_clock_leaf_source')
    check(old.get('vendor_lutram_through') is True and new.get('vendor_doutb_destination') is True,'vendor_replacement_selectors')
    try:
        expected=Decimal(str(expected_period));a=Decimal(str(old.get('value')));b=Decimal(str(new.get('value')))
        check(expected.is_finite() and expected>0 and a==b==expected,'same_expected_clock_bound')
    except Exception:issues.append('invalid_bound')
    check(ignored_status==f"Totally overridden path by DPO {new.get('id')}",'exact_effective_override')
    check(old_coverage.get('Type')==new_coverage.get('Type')=='Max Delay Datapath Only','coverage_type')
    check(old_coverage.get('Setup')==new_coverage.get('Setup') and old_coverage.get('Setup','').startswith('max_dpo='),'coverage_bound')
    try:
        check(Decimal(old_coverage.get('Setup','').removeprefix('max_dpo='))==Decimal(str(expected_period)),'effective_coverage_numeric_bound')
    except Exception:issues.append('effective_coverage_numeric_bound')
    check(type(old_coverage.get('Endpoints')) is int and old_coverage['Endpoints']>0 and old_coverage.get('Endpoints')==new_coverage.get('Endpoints'),'nonempty_same_covered_endpoint_count')
    check(new_coverage.get('Through','')=='','replacement_has_no_unreviewed_through')
    return issues


def logical_xdc(text):
    """Frame commands without substitutions or execution; retain source/ID comments."""
    source='';comments=[];chunks=[];brace=bracket=0;quote=False;escaped=False
    for lineno,line in enumerate(text.splitlines(keepends=True),1):
        if not chunks and (not line.strip() or line.lstrip().startswith('#')):
            m=re.fullmatch(r"\s*# Constraints from file : '([^']+)'\s*",line)
            if m:source=m[1];comments=[]
            elif line.strip():comments=(comments+[line.strip()])[-2:]
            continue
        if not chunks:start=lineno
        chunks.append(line)
        for ch in line:
            if escaped:escaped=False;continue
            if ch=='\\':escaped=True;continue
            if brace:
                if ch=='{':brace+=1
                elif ch=='}':brace-=1
                continue
            if ch=='"':quote=not quote;continue
            if not quote and ch=='{':brace=1
            elif ch=='[':bracket+=1
            elif ch==']':bracket-=1;require(bracket>=0,'Unbalanced XDC bracket')
        require(brace<100 and bracket<100,'Deep XDC nesting')
        if brace or bracket or quote or escaped or re.search(r'(?<!\\)(?:\\\\)*\\\r?\n$',line):continue
        yield start,source,comments,''.join(chunks).strip()
        chunks=[];comments=[]
    require(not chunks,'Incomplete XDC command')
def norm_command(command):
    return re.sub(r'\\\r?\n[ \t]*',' ',command).strip()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _list(text):
    return _peer('gray_review').tcl_list(text)


_PEERS={}
def _peer(name):
    if name not in _PEERS:
        spec=importlib.util.spec_from_file_location('c2_exception_'+name,Path(__file__).with_name(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_PEERS[name]=module
    return _PEERS[name]


EXCEPTION_TYPES={'set_false_path','set_clock_groups','set_max_delay','set_min_delay','set_bus_skew','set_multicycle_path','set_disable_timing'}
MEMORY_FROM='[filter [all_fanout -from [get_ports -scoped_to_current_instance -no_traverse clka] -flat -endpoints_only] IS_LEAF]'
MEMORY_THROUGH='[get_pins -of_objects [get_cells -hier * -filter {PRIMITIVE_SUBGROUP==LUTRAM || PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==drom}] -filter DIRECTION==OUT]'
MEMORY_TO='[get_cells -hierarchical -filter {NAME =~ *doutb*reg*}]'
HANDSHAKE_SELECTORS='-from [get_cells src_hsdata_ff_reg*] -to [get_cells dest_hsdata_ff_reg*]'


def parse_export(text):
    scope='';variables={};pending=None;rows=[];ids=set()
    for line,source,comments,command in logical_xdc(text):
        for comment in comments:
            match=re.fullmatch(r'# id:([0-9]+)',comment)
            if match:pending=int(match[1])
        command=norm_command(command)
        if command.startswith('current_instance'):
            words=_list(command)[1:]
            if words[:1]==['-quiet']:words=words[1:]
            require(len(words)<=1,'Ambiguous scope');scope=words[0] if words else '';continue
        assignment=re.fullmatch(r'set (_xlnx_shared_i[0-9]+) (.+)',command,re.S)
        if assignment:
            variables[assignment[1]]=(scope,assignment[2]);continue
        kind=command.split(None,1)[0] if command else ''
        if kind not in EXCEPTION_TYPES:
            pending=None;continue
        require(pending is not None,'Exception without exported ID')
        require(pending==0 or pending not in ids,'Duplicate nonzero exception ID')
        ids.add(pending)
        def expand(match):
            key=match[1];require(key in variables and variables[key][0]==scope,'Unbound or cross-scope XDC alias')
            return variables[key][1]
        expanded=re.sub(r'\$(_xlnx_shared_i[0-9]+)\b',expand,command)
        require('$' not in expanded,'Unsupported XDC variable')
        rows.append(dict(id=pending,scope=scope,source=source,kind=kind,command=expanded))
        pending=None
    require(rows,'No exported exceptions')
    return rows


def memory_clause(row):
    result=dict(row);command=row['command'];prefix='set_max_delay -datapath_only -from '+MEMORY_FROM
    require(command.startswith(prefix+' '),'Nonstandard vendor memory source clause')
    match=re.fullmatch(re.escape(prefix)+r' -(through|to) (.+) ([0-9]+\.[0-9]+)',command)
    require(match is not None,'Malformed vendor memory bound/selector')
    option,selector,value=match.groups()
    require((option,selector) in (('through',MEMORY_THROUGH),('to',MEMORY_TO)),'Unknown vendor memory selector')
    result.update(datapath_only=True,value=value,**{'from':MEMORY_FROM},vendor_clka_leaf_source=True,
                  vendor_lutram_through=option=='through',vendor_doutb_destination=option=='to')
    return result


def other_clause_family(row):
    """Recognize grammar/purpose; exact record-set disposition remains required."""
    source,scope,kind,command=(row[k] for k in ('source','scope','kind','command'))
    if source=='xpm_cdc_async_rst.tcl' and kind=='set_false_path':
        if scope and command=='set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]':
            return 'source_bound_xpm_async_reset'
        if not scope and re.fullmatch(r'set_false_path -through \[get_pins (SL|WRAPPER)/(?!CL/)[^ ]+/src_arst\]',command):
            return 'inherited_shell_async_reset'
    if source=='xpm_cdc_single.tcl' and scope and command=='set_false_path -to [get_cells {syncstages_ff_reg[0]}]':
        return 'source_bound_xpm_single_first_stage'
    if source=='xpm_cdc_array_single.tcl' and scope and command=='set_false_path -to [get_cells {syncstages_ff_reg[0][*]}]':
        return 'source_bound_xpm_array_first_stage'
    if scope or 'WRAPPER/CL/' in command:return None
    if source=='top_timing.xdc':
        if command=='set_clock_groups -asynchronous -group [get_clocks clk_core] -group [get_clocks clk_spi]':
            return 'inherited_shell_core_spi_clockgroup'
        if kind=='set_false_path' and command.startswith('set_false_path -from [get_pins {{WRAPPER/IO_SHIM/CDC_ASYNC_CLK_EN_N/arststages_ff_reg[0]/C}'):
            return 'inherited_shell_clock_enable_reset'
    if source=='io_timing.xdc' and kind in ('set_max_delay','set_min_delay'):
        value=command.rsplit(' ',1)[-1]
        if value==('50.000' if kind=='set_max_delay' else '-50.000') and '[get_ports ' in command and '[get_clocks ' in command:
            return 'inherited_shell_io_limits'
    if source=='axi_jtag.xdc' and re.fullmatch(r'set_max_delay -datapath_only -from \[get_cells \{WRAPPER/RL/RL_DEBUG_BRIDGE/inst/axi_jtag/inst/u_jtag_proc/t(?:di|ms)_output_reg\[0\]\}\] 16\.000',command):
        return 'inherited_shell_jtag_limits'
    if source=='rl_clkwiz_main.xdc' and kind=='set_false_path' and command.startswith('set_false_path -from [get_pins [list '):
        return 'inherited_shell_mmcm_reconfiguration'
    if source=='sync_cdc.xdc' and kind=='set_false_path':
        if row['id']==0 and re.match(r'set_false_path -setup -hold -to \[get_(?:cells|pins) \[list ',command):
            return 'inherited_shell_flattened_false_paths'
        if command=='set_false_path -to [get_pins {WRAPPER/RL/RL_MGT/MGT_SPI_SLV/SYNC_INP/in_q_reg[0]/D}]':
            return 'inherited_shell_spi_input'
    return None


def review_exceptions(export_text,ignored_text,coverage_text,*,fifo_write_periods,handshake_witnesses=None):
    """Return issues and canonical facts; no caller-supplied approval flags.

    fifo_write_periods maps exact vendor FIFO instance -> measured source PERIOD
    of its addra-to-rd-clock gray synchronizer (same FIFO write clock). Those
    relationships must come from same-DCP gray evidence and verified standard
    vendor FIFO source, not the numeric bound being checked here.

    handshake_witnesses maps exact OCL handshake instance to observed width,
    forward_sync_stages, source_period and destination_period. A missing witness
    is explicit pending evidence. Final skew/path timing is checked separately.
    """
    rows=parse_export(export_text);ignored,unknown=parse_ignored(ignored_text);coverage=parse_coverage(coverage_text)
    byid={r['id']:r for r in rows if r['id']};issues=[];facts=[];memory={};handshake={};other=[]
    for row in rows:
        try:
            if row['source']=='xpm_memory_xdc.tcl':memory[row['id']]=memory_clause(row)
            elif row['source']=='xpm_cdc_handshake.tcl':
                require(row['kind'] in ('set_max_delay','set_bus_skew'),'Unknown handshake exception')
                prefix=('set_max_delay -datapath_only' if row['kind']=='set_max_delay' else 'set_bus_skew')+' '+HANDSHAKE_SELECTORS+' '
                require(row['command'].startswith(prefix),'Nonstandard handshake selector')
                value=Decimal(row['command'][len(prefix):]);require(value.is_finite() and value>0,'Invalid handshake bound')
                require(row['kind'] not in handshake.setdefault(row['scope'],{}),'Duplicate handshake clause')
                handshake[row['scope']][row['kind']]=dict(row,value=str(value))
            elif row['source']=='xpm_cdc_gray.tcl':
                require(row['kind'] in ('set_max_delay','set_bus_skew'),'Unknown gray clause')
                facts.append(('gray_delegated',row['scope'],row['kind']))
            else:
                family=other_clause_family(row)
                if family is None:issues.append({'id':row['id'],'reason':'Unknown non-gray exception grammar/family'})
                other.append(dict(row,family=family))
        except (ValueError,ArithmeticError) as error:issues.append({'id':row['id'],'reason':str(error)})
    overridden=set();replacements=set()
    for item in ignored:
        position=item['Position'];row=byid.get(position)
        if position in memory:
            match=re.fullmatch(r'Totally overridden path by DPO ([0-9]+)',item['Status'])
            if not match:issues.append({'id':position,'reason':'Unrecognized memory override status'});continue
            target=int(match[1]);old=memory[position];new=memory.get(target)
            if new is None:issues.append({'id':position,'reason':'Memory replacement not same vendor family'});continue
            scope_match=re.fullmatch(r'(.*/gen_xpm_memory_fifo\.inst_fifo)/gen_mem_rep\[[0-9]+\]\.inst_xpm_memory',old['scope'])
            fifo=scope_match[1] if scope_match else None
            expected=fifo_write_periods.get(fifo) if fifo else None
            problems=validate_memory_override(old,new,coverage.get(position,{}),coverage.get(target,{}),item['Status'],expected)
            for problem in problems:issues.append({'id':position,'reason':problem})
            overridden.add(position);replacements.add(target)
            facts.append(('memory_override',old['scope'],position,target,str(expected),coverage.get(target,{}).get('Endpoints')))
        elif item['Status']=='Non-existent path':
            # Constraints with no path are not weakened active timing. Record
            # their exact identity; all scoped CL exceptions still pass family
            # checks below, and CDC/clock/structural review remains independent.
            if row is None:issues.append({'id':position,'reason':'Ignored ID missing from export'})
            else:facts.append(('nonexistent',row['source'],row['scope'],row['command']))
        elif item['Status']=='Totally overridden path by FP 0':
            expected={
                'set_false_path -through [get_pins WRAPPER/RL/HOST_PCIE4C/inst/sys_reset_in_async_rst_inst/src_arst]',
                'set_false_path -through [get_pins WRAPPER/RL/CDC_ASYNC_VR_ALERT_N/src_arst]',
                'set_false_path -through [get_pins WRAPPER/IO_SHIM/CDC_ASYNC_RST_HBM_N/src_arst]'}
            if row is None or row['source']!='xpm_cdc_async_rst.tcl' or row['scope'] or row['command'] not in expected:
                issues.append({'id':position,'reason':'Unknown false-path override'})
            else:facts.append(('inherited_shell_reset_override',row['command']))
        elif item['Status']=='Invalid endpoint':
            if row is None or row['source']!='sync_cdc.xdc' or row['scope'] or row['command']!='set_false_path -to [get_pins {WRAPPER/RL/RL_MGT/MGT_SPI_SLV/SYNC_INP/in_q_reg[0]/D}]':
                issues.append({'id':position,'reason':'Unknown invalid endpoint'})
            else:facts.append(('inherited_unused_spi_endpoint',row['command']))
        else:issues.append({'id':position,'reason':'Unknown ignored exception status'})
    if set(memory)!=overridden|replacements or overridden&replacements:
        issues.append({'reason':'Memory clause inventory not exactly covered by independent replacement pairs'})
    if replacements & {r['Position'] for r in ignored}:issues.append({'reason':'Effective memory replacement is itself ignored'})
    for root,pair in handshake.items():
        witness=(handshake_witnesses or {}).get(root)
        if witness is None:issues.append({'scope':root,'reason':'Missing handshake clock/width/depth witness'});continue
        try:
            require(set(pair)=={'set_max_delay','set_bus_skew'},'Incomplete handshake clause pair')
            require(type(witness.get('width')) is int and 1<=witness['width']<=100,'Unsupported handshake width')
            require(type(witness.get('forward_sync_stages')) is int and 2<=witness['forward_sync_stages']<=10,'Invalid handshake synchronizer depth')
            source=Decimal(str(witness['source_period']));dest=Decimal(str(witness['destination_period']))
            require(source.is_finite() and dest.is_finite() and source>0 and dest>0 and source!=dest,'Missing/invalid handshake clocks')
            expected=dest*witness['forward_sync_stages']
            require(all(Decimal(v['value'])==expected for v in pair.values()),'Handshake bound differs from observed vendor formula')
            require(not {v['id'] for v in pair.values()} & {r['Position'] for r in ignored},'Handshake exception is ignored')
            cov=coverage.get(pair['set_max_delay']['id'],{})
            require(cov.get('Type')=='Max Delay Datapath Only' and cov.get('Endpoints')==witness['width'],'Handshake width/effective coverage mismatch')
            require(Decimal(cov.get('Setup','').removeprefix('max_dpo='))==expected,'Handshake effective coverage bound mismatch')
            facts.append(('handshake',root,witness,str(expected)))
        except (KeyError,ValueError,ArithmeticError) as error:issues.append({'scope':root,'reason':str(error)})
    # All other clauses are returned as an exact artifact-specific record set.
    # The final-review record must supply a narrow reviewed disposition for this
    # canonical digest. It cannot substitute a generic success flag or count.
    groups=defaultdict(list)
    for row in other:groups[row['family']].append(row)
    return {'issues':issues,'canonical_facts':{'facts':facts,'other_clauses':other,
            'other_clauses_sha256':digest(other),'export_count':len(rows),'coverage_count':len(coverage),
            'other_clause_groups':{key:{'records_sha256':digest(value),'count':len(value)} for key,value in groups.items()},
            'memory_pairs':len(overridden),'handshake_roots':sorted(handshake)},'route_qualified':False}


def parse_findings_report(text):
    pattern=re.compile(r'^([A-Z0-9]+-[0-9]+)#([0-9]+) (Critical Warning|Warning|Advisory|Info|Error|Fatal)\n(.*?)(?=^[A-Z0-9]+-[0-9]+#[0-9]+ (?:Critical Warning|Warning|Advisory|Info|Error|Fatal)\n|\Z)',re.M|re.S)
    rows=[];seen=set()
    for match in pattern.finditer(text):
        check,ordinal,severity,body=match.groups();name=check+'#'+ordinal
        require(name not in seen,'Duplicate finding report ID');seen.add(name)
        rows.append(dict(name=name,check=check,severity=severity,body=body.strip()))
    summary={}
    for line in text.splitlines():
        match=re.match(r'^\|\s*([A-Z0-9]+-[0-9]+)\s*\|\s*(Critical Warning|Warning|Advisory|Info|Error|Fatal)\s*\|.*\|\s*([0-9]+)\s*\|$',line)
        if match:
            key=(match[1],match[2]);require(key not in summary,'Duplicate summary rule');summary[key]=int(match[3])
    require(summary and Counter((r['check'],r['severity']) for r in rows)==summary,'Finding summary/details mismatch')
    return rows


def finding_family(kind,row):
    check,severity,body=(row[k] for k in ('check','severity','body'))
    if kind=='drc':
        if severity not in ('Warning','Advisory','Info'):return None
        if check in {'DPIP-2','DPOP-3','DPOP-4','AVAL-155','AVAL-156'} and 'WRAPPER/CL/u_board/u_mxu/' in body and ('DSP ' in body or 'DSP48E2' in body):
            return 'c2_dsp_pipeline_power_recommendations'
        if check=='BUFC-1' and re.search(r'Input buffer WRAPPER/CL/SH_DDR/[^\s]+ .*? has no loads\.',body,re.S):
            return 'disabled_ddr_stub_input_buffers'
        if check=='RTSTAT-10' and body.startswith('No routable loads'):
            return 'unused_net_topology_required'
        if check=='HDPRA-57' and "RP pblock 'pblock_CL'" in body and "TAG='57_G79_RP1_S'" in body:
            return 'inherited_shell_pblock_gap_advisory'
        return None
    if kind!='methodology':return None
    if check=='TIMING-17' and severity=='Critical Warning' and body.startswith('Non-clocked sequential cell'):
        return 'exact_shell_no_clock_witness_required'
    if severity not in ('Warning','Advisory','Info'):return None
    if check=='SYNTH-6' and 'WRAPPER/CL/u_board/' in body and 'no output register was merged into the block' in body:
        return 'c2_bram_output_register_recommendations'
    if check in ('CLKC-40','CLKC-56') and re.search(r'WRAPPER/CL/u_(?:core|hbm)_clk/inst/mmcme4_adv_inst',body):
        return 'source_bound_clock_wizard_placement_recommendations'
    if check=='LUTAR-1' and 'LUT cell WRAPPER/CL/SH_DDR/CCF_DDR_STAT_REQ/' in body:
        return 'disabled_ddr_stub_reset_support'
    if check=='LUTAR-1' and re.search(r'LUT cell WRAPPER/(?:HBM_SHIM/PIPE_WDATA_[01]/|IO_SHIM/CDC_ASYNC_CLK_EN_N_i_1)',body):
        return 'exact_inherited_shell_reset_logic'
    if check=='CKLD-2' and 'Clock net WRAPPER/IO_SHIM/IBUF_SCK/O' in body:
        return 'exact_inherited_shell_spi_clock'
    if check=='CLKC-3' and 'WRAPPER/IO_SHIM/' in body and 'has a LOC constraint' in body:
        return 'exact_inherited_shell_clock_buffer_placement'
    if check=='CLKC-40' and 'WRAPPER/IO_SHIM/MMCM_CLK_HBM_REF/inst/mmcme4_adv_inst' in body:
        return 'exact_inherited_shell_hbm_reference_mmcm'
    if check=='TIMING-18' and re.search(r'(?:input|output) delay is missing on SPI_(?:CS2_N_SLAVE_R|MISO_SLAVE_R).*clk_spi',body,re.S):
        return 'exact_inherited_shell_spi_io_delay'
    if check in ('XDCC-1','XDCC-7') and body.count('create_clock -period 10.005 [get_ports CLK_DIMM_DP]')==2:
        return 'exact_inherited_ddr_clock_duplicate'
    if check=='TIMING-9' and 'report_cdc' in body:
        return 'complete_current_cdc_review_required'
    return None


def review_methodology(drc_text,methodology_text,*,witnesses=None):
    """Classify reviewed warning purposes and pin full exact finding records.

    Every group needs an artifact-specific final-review disposition bound to its
    records_sha256. Families ending `required` additionally need the named live
    topology/CDC witness. Merely recognizing a rule or scope is never acceptance.
    """
    issues=[];groups=defaultdict(list)
    for kind,text in (('drc',drc_text),('methodology',methodology_text)):
        for row in parse_findings_report(text):
            family=finding_family(kind,row)
            if family is None:issues.append(dict(kind=kind,name=row['name'],reason='Unreviewed finding family/severity'))
            groups[(kind,family)].append(row)
    facts=[]
    for (kind,family),rows in sorted(groups.items(),key=lambda x:str(x[0])):
        facts.append(dict(kind=kind,family=family,count=len(rows),records_sha256=digest(rows),records=rows))
        if family and family.endswith('_required') and not (witnesses or {}).get(family):
            issues.append(dict(kind=kind,family=family,reason='Required raw prerequisite witness absent'))
    return {'issues':issues,'canonical_facts':{'groups':facts},'route_qualified':False}


OCL_PREFIX='WRAPPER/CL/u_ocl_cdc/inst/gen_clock_conv.gen_async_lite_conv.clock_conv_lite_'
OCL_ROOTS={OCL_PREFIX+suffix+'/handshake' for suffix in ('fwd_ar','fwd_aw','fwd_w','resp_b','resp_r')}
# Exact installed Vivado 2025.2 source bytes independently retrieved in the
# hash-bound vendor source receipt. This pins reviewed numeric intent; synthesis
# and generated-IP provenance remain required by the outer receipt chain.
VENDOR_SOURCES={
    'memory_vendor':('98b4fef69706ae1849a1640292cd982b101f1b8f76f2f41abfab88b942dec873',3485),
    'handshake_vendor':('c524b464dff2e91e30389fa624522c37bd1585068041e303918218ac66625c1c',1988),
}


def vendor_source_identity(auxiliary):
    result={}
    for key,(expected,size) in VENDOR_SOURCES.items():
        require(key in auxiliary,'Missing reviewed vendor source '+key)
        path=Path(auxiliary[key]);require(path.is_file() and not path.is_symlink() and path.stat().st_size==size,'Unexpected vendor source size/type')
        actual=hashlib.sha256(path.read_bytes()).hexdigest();require(actual==expected,'Unreviewed vendor source '+key)
        result[key]={'sha256':actual,'bytes':size}
    return result


def parse_handshake_witness(path,checkpoint_sha256):
    """Replay raw typed register/clock observations; derive width and depth."""
    g=_peer('gray_review');header=None;roots=None;properties={};inventories={};registers={};complete=None
    for row in g.records(path):
        require(row,'Empty handshake record');tag=row[0]
        if tag=='schema':
            require(header is None,'Duplicate handshake header');header=g.pairs(row)
            require(header=={'schema':'1','kind':'c2_alex_handshake_witness','checkpoint_sha256':checkpoint_sha256,'route_qualified':'0'},'Handshake checkpoint/schema mismatch')
        elif tag=='roots':
            require(roots is None,'Duplicate handshake roots');data=g.pairs(row[1:]);require(set(data)=={'names'},'Unexpected root fields')
            roots=g.tcl_list(data['names']);require(len(roots)==len(set(roots)) and set(roots)==OCL_ROOTS,'Unexpected handshake root inventory')
        elif tag=='root':
            data=g.pairs(row[1:]);require(set(data)=={'name','properties'} and data['name'] not in properties,'Duplicate/malformed handshake root')
            properties[data['name']]=g.td(data['properties'])
        elif tag=='inventory':
            data=g.pairs(row[1:]);require(set(data)=={'root','role','names'},'Unexpected inventory fields')
            key=(data['root'],data['role']);require(key not in inventories,'Duplicate handshake inventory')
            names=g.tcl_list(data['names']);require(names and len(names)==len(set(names)) and len(names)<=100,'Missing/duplicate/excess registers')
            inventories[key]=names
        elif tag=='register':
            data=g.pairs(row[1:]);require(set(data)=={'root','role','cell','pin','owners','clocks','clock'},'Unexpected register fields')
            for key in ('cell','pin','clock'):data[key]=g.td(data[key])
            for key in ('owners','clocks'):data[key]=g.tcl_list(data[key])
            key=(data['root'],data['role'],data['cell'].get('NAME'));require(key not in registers,'Duplicate handshake register')
            registers[key]=data
        elif tag=='complete':
            require(complete is None,'Duplicate handshake completion');complete=g.pairs(row[1:])
        else:raise ValueError('Unknown handshake record '+tag)
    require(header is not None and roots is not None and complete=={'roots':'5','route_qualified':'0'},'Incomplete handshake evidence')
    require(set(properties)==OCL_ROOTS,'Incomplete root property inventory')
    expected={(root,role) for root in OCL_ROOTS for role in ('source','destination','forward_sync')}
    require(set(inventories)==expected,'Incomplete handshake role inventory')
    require(set(registers)=={(root,role,name) for (root,role),names in inventories.items() for name in names},'Register evidence does not cover exact inventories')
    result={}
    for root in sorted(OCL_ROOTS):
        prop=properties[root]
        require(prop.get('NAME')==root and prop.get('IS_BLACKBOX')=='0' and (prop.get('ORIG_REF_NAME') or prop.get('REF_NAME'))=='xpm_cdc_handshake','Handshake vendor identity/opacity differs')
        periods={};bitsets={}
        for role in ('source','destination','forward_sync'):
            leaf={'source':'src_hsdata_ff_reg','destination':'dest_hsdata_ff_reg','forward_sync':'xpm_cdc_single_src2dest_inst/syncstages_ff_reg'}[role]
            bits=[];clocks=set()
            for name in inventories[(root,role)]:
                match=re.fullmatch(re.escape(root+'/'+leaf)+r'\[([0-9]+)\]',name)
                require(match is not None,'Unexpected handshake register selector');bits.append(int(match[1]))
                data=registers[(root,role,name)];cell=data['cell'];pin=data['pin'];clock=data['clock']
                require(cell.get('NAME')==name and cell.get('IS_SEQUENTIAL')=='1' and cell.get('REF_NAME') in {'FDRE','FDSE','FDCE','FDPE'},'Non-register handshake endpoint')
                require(pin.get('NAME')==name+'/C' and pin.get('REF_PIN_NAME')=='C' and pin.get('IS_CLOCK')=='1' and pin.get('DIRECTION')=='IN' and data['owners']==[name],'Handshake clock pin identity/owner')
                require(data['clocks']==[clock.get('NAME')] and bool(clock.get('NAME')),'Handshake clock collection mismatch')
                period=Decimal(clock['PERIOD']);require(period.is_finite() and period>0,'Invalid handshake period')
                clocks.add((clock['NAME'],period))
                if role=='forward_sync':require(cell.get('ASYNC_REG','').upper() in {'1','TRUE'},'Handshake stage lacks ASYNC_REG')
            require(sorted(bits)==list(range(len(bits))),'Noncontiguous handshake bit/stage inventory')
            require(len(clocks)==1,'Mixed handshake clocks');periods[role]=next(iter(clocks));bitsets[role]=bits
        require(len(bitsets['source'])==len(bitsets['destination']),'Handshake source/destination width mismatch')
        require(2<=len(bitsets['forward_sync'])<=10,'Handshake stage count outside vendor range')
        require(periods['destination']==periods['forward_sync'] and periods['source'][0]!=periods['destination'][0],'Handshake clock direction mismatch')
        result[root]={'width':len(bitsets['source']),'forward_sync_stages':len(bitsets['forward_sync']),
            'source_period':str(periods['source'][1]),'destination_period':str(periods['destination'][1]),
            'source_clock':periods['source'][0],'destination_clock':periods['destination'][0],
            'source_registers':sorted(inventories[(root,'source')]),'destination_registers':sorted(inventories[(root,'destination')])}
    return result


def fifo_write_periods_from_gray(evidence):
    """Use observed addra synchronizer clocks, independent of memory XDC bound."""
    g=_peer('gray_review');result={};before=len(evidence.issues)
    suffix='/gen_async_clocks.inst_cdc_addra_to_rd_clk'
    for root,group in sorted(evidence.groups.items()):
        if not root.endswith(suffix):continue
        fifo=root[:-len(suffix)];require(fifo.endswith('/gen_xpm_memory_fifo.inst_fifo'),'Unexpected addra FIFO scope')
        require(evidence.relations.get(root,{}).get('fifo_candidate')==fifo,'FIFO gray ancestor relation mismatch')
        source=g.td(group['source']);require(source,'Missing addra source registers')
        clocks={evidence.clock(('gray',root,'bit',bit,'source_clock'),cell+'/C') for bit,cell in source.items()}
        require(len(clocks)==1,'Mixed FIFO write clocks');clock,period=next(iter(clocks))
        require(bool(clock) and period>0,'Missing FIFO write clock')
        require(fifo not in result,'Duplicate FIFO write evidence');result[fifo]=str(period)
    require(result and len(evidence.issues)==before,'Incomplete FIFO write clock evidence')
    return result


def review_handshake_skew(rows,handshakes,gray_groups):
    """Every exported physical skew row must match one exact observed bus."""
    g=_peer('gray_review');matched={};gray_used=set()
    for ident,row in rows.items():
        source=sorted(g.tcl_list(row['from']));dest=sorted(g.tcl_list(row['to']))
        gray=[root for root,group in gray_groups.items() if source==sorted(g.td(group['source']).values()) and dest==sorted(g.td(group['destination']).values())]
        if gray:
            require(len(gray)==1 and gray[0] not in gray_used,'Duplicate gray bus-skew coverage');gray_used.add(gray[0]);continue
        candidates=[root for root,w in handshakes.items() if source==w['source_registers'] and dest==w['destination_registers']]
        require(len(candidates)==1,'Unknown/mixed non-gray bus-skew endpoints');root=candidates[0]
        require(root not in matched,'Duplicate handshake bus-skew coverage')
        witness=handshakes[root];expected=Decimal(witness['destination_period'])*witness['forward_sync_stages']
        requirement=Decimal(row['requirement']);actual=Decimal(row['actual']);slack=Decimal(row['slack'])
        require(requirement.is_finite() and requirement==expected,'Handshake physical bus-skew formula mismatch')
        # Actual skew is signed after relative clock-delay/CRPR adjustment.
        require(actual.is_finite() and slack.is_finite() and slack>=0 and actual<=expected,'Handshake bus-skew physical failure')
        require(abs((expected-actual)-slack)<=Decimal('.001'),'Handshake bus-skew slack inconsistent')
        matched[root]={'id':ident,'row':row,'derived_bound_ns':str(expected)}
    require(set(matched)==set(handshakes) and gray_used==set(gray_groups),'Incomplete exact bus-skew inventory')
    return matched


def current_cdc_inventory(text,object_rows):
    """Cross-check complete report summary/detail plus raw object serialization.

    This establishes full inventory, not acceptance of every CDC record. The
    outer source-pinned artifact dispositions must cover each exact record;
    gray and reset/status replay independently establish their active topology.
    """
    expected={};actual=Counter();rows=[];source=dest=kind=''
    for line in text.splitlines():
        summary=re.match(r'^(CDC-[0-9]+)\s+(Critical|Warning|Info)\s+([0-9]+)\s+',line)
        if summary:
            require(summary[1] not in expected,'Duplicate CDC summary');expected[summary[1]]=int(summary[3])
        elif line.startswith('Source Clock: '):source=line[len('Source Clock: '):]
        elif line.startswith('Destination Clock: '):dest=line[len('Destination Clock: '):]
        elif line.startswith('CDC Type: '):kind=line[len('CDC Type: '):]
        elif re.match(r'^\s*[0-9]+\s+CDC-',line):
            match=re.fullmatch(r'\s*([0-9]+)\s+(CDC-[0-9]+)\s+(Critical|Warning|Info)\s+(.+?)\s+([0-9]+)\s+(.+?)\s+(\S+)\s+(\S+)\s+([NY])\s*',line)
            require(match is not None and source and dest and kind,'Malformed CDC detail')
            ident,rule,severity,description,depth,exception,start,end,waived=match.groups()
            require('...' not in start+end,'Truncated CDC endpoint')
            row=dict(source_clock=source,destination_clock=dest,cdc_type=kind,row=ident,rule=rule,severity=severity,description=description,depth=depth,exception=exception,**{'from':start,'to':end},waived=waived)
            rows.append(row);actual[rule]+=1
    require(expected and dict(actual)==expected,'CDC summary/detail mismatch')
    fields=set(rows[0]);normalized=[{key:row[key] for key in fields} for row in object_rows]
    require(Counter(digest(row) for row in rows)==Counter(digest(row) for row in normalized),'CDC report/object inventory mismatch')
    return {'rows':rows,'records_sha256':digest(rows),'counts':dict(actual),'acceptance':'requires exact artifact dispositions and independent active topology predicates'}


def unused_net_inventory(supplemental,findings):
    """Read full typed related-net/port collections without truncated previews."""
    g=_peer('gray_review');selected={}
    for row in g.records(supplemental):
        if row and row[0]=='query':
            key=g.label(row[1])
            if len(key)==4 and key[:2]==('drc','finding') and key[2].startswith('RTSTAT-10#') and key[3] in {'related_nets','related_ports'}:
                require(key not in selected,'Duplicate RTSTAT relation');data=g.pairs(row[2:])
                require(data.get('status')=='observed','Failed RTSTAT relation')
                names=g.tcl_list(data['value']);require(len(names)==len(set(names)) and all(name and name!='...' for name in names),'Truncated/duplicate RTSTAT relation')
                selected[key]=names
    result={}
    for row in findings:
        if row['check']!='RTSTAT-10':continue
        match=re.search(r'\b([0-9]+) net\(s\) have no routable loads\.',row['body']);require(match is not None,'Unrecognized RTSTAT description')
        nets=selected.get(('drc','finding',row['name'],'related_nets'));ports=selected.get(('drc','finding',row['name'],'related_ports'))
        require(nets is not None and ports is not None and len(nets)==int(match[1]),'RTSTAT full relation inventory missing/incomplete')
        result[row['name']]={'nets':sorted(nets),'ports':sorted(ports),'records_sha256':digest({'nets':sorted(nets),'ports':sorted(ports)})}
    return result


def review_artifact(*,reports,auxiliary,checkpoint_sha256,prerequisites=None):
    """Replay source-pinned raw files; caller binds hashes and exact dispositions.

    prerequisites are full independently computed results from final_review,
    never loaded from a user-supplied success receipt. Raw pending reports are
    left unchanged. No generic family/count constitutes an acceptance waiver.
    """
    g=_peer('gray_review');issues=[];facts={};prerequisites=prerequisites or {}
    try:
        facts['vendor_sources']=vendor_source_identity(auxiliary)
        evidence=g.Evidence();evidence.read(reports['supplemental.tcldata'],'supplemental')
        binding,skew=g.read_objects(evidence,reports['objects.tcldata'],checkpoint_sha256)
        periods=fifo_write_periods_from_gray(evidence)
        require(prerequisites.get('gray',{}).get('gray_evidence_consistent') is True and prerequisites['gray'].get('checkpoint_sha256')==checkpoint_sha256,'Independent same-DCP gray predicate incomplete')
        require('handshake' in auxiliary,'Missing raw handshake capture')
        handshakes=parse_handshake_witness(auxiliary['handshake'],checkpoint_sha256)
        facts['handshake_skew']=review_handshake_skew(skew,handshakes,evidence.groups)
        result=review_exceptions(Path(reports['timing_all_ids.xdc']).read_text(),Path(reports['ignored_exceptions.rpt']).read_text(),Path(reports['exception_coverage.rpt']).read_text(),fifo_write_periods=periods,handshake_witnesses=handshakes)
        issues.extend(result['issues']);facts['exceptions']=result['canonical_facts'];facts['fifo_write_periods']=periods
        require(set(result['canonical_facts']['handshake_roots'])==set(handshakes),'Export/raw handshake root mismatch')
        drc=Path(reports['drc.rpt']).read_text();method=Path(reports['methodology.rpt']).read_text()
        facts['unused_nets']=unused_net_inventory(reports['supplemental.tcldata'],parse_findings_report(drc))
        objects=[g.td(row[1]) for row in g.records(reports['objects.tcldata']) if row and row[0]=='cdc']
        facts['cdc_inventory']=current_cdc_inventory(Path(reports['cdc.rpt']).read_text(),objects)
        no_clock=prerequisites.get('no_clock',{});reset=prerequisites.get('reset_status',{})
        witnesses={'unused_net_topology_required':facts['unused_nets'],
            'exact_shell_no_clock_witness_required':no_clock.get('predicate_satisfied') is True and no_clock.get('canonical_current_sha256')==no_clock.get('canonical_shell_sha256') and no_clock.get('unresolved_count')==0,
            'complete_current_cdc_review_required':facts['cdc_inventory'] and reset.get('issues')==[] and bool(reset.get('canonical_facts'))}
        result=review_methodology(drc,method,witnesses=witnesses)
        issues.extend(result['issues']);facts['methodology']=result['canonical_facts']
        facts['external_obligation']='All exact raw finding/exception/CDC records and auxiliary hashes must be covered by source-pinned artifact dispositions in final_review.'
    except (ValueError,KeyError,IndexError,ArithmeticError,OSError) as error:
        issues.append({'reason':'Incomplete/malformed artifact evidence','detail':str(error)})
    return {'issues':issues,'canonical_facts':facts,'route_qualified':False}
