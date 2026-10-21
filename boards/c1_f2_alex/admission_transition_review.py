"""One-step proof of the bounded C1 admission state slice.

The source specialization (read-only command stream and eliminated high bits)
must be proved separately. This covers the 81 observed admission FDREs and the
24 command RAM bits only; no AXI response/length-queue equivalence is claimed.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import admission_review as A, admission_equation_review as E
from . import admission_pair_review as Q
from . import native_review as N, ready_pair_review as P

CONFIGURATION = {'QD': 16, 'MAX_OUT': 128, 'READ_BURST': 16, 'READ_IDLE': 4}


def increment(b, vector, *, decrement=False):
    carry = 1; result = []
    for bit in vector:
        result.append(b.ite(carry, b.invert(bit), bit))
        carry = b.ite(b.invert(bit) if decrement else bit, carry, 0)
    return result


def expected_next(b, states, inputs, stored, take):
    """Specialized frozen RTL, with active-high synchronous reset dominant."""
    both = lambda *values: E._all(b, values)
    select = lambda condition, yes, no: [b.ite(condition, a, z) for a, z in zip(yes, no)]
    zero = lambda values: E._zero(b, values)
    equal = lambda values, constant: E._all(b, [v if (constant >> i) & 1 else b.invert(v) for i,v in enumerate(values)])
    changed = {key: list(value) for key,value in states.items()}
    rst = inputs['rst']; pending = states['pending_valid'][0]
    ar_fire = both(pending, inputs['m_arready'])  # Evaluated in the !rst branch.
    push = both(inputs['n_cvalid'], b.invert(states['cn'][4]))
    count_zero = zero(states['pending_count']); cn_zero = zero(states['cn'])
    # Read/write response events are unmasked here because outer reset dominates.
    def counted(name, up, down):
        current = states[name]
        return select(up, select(down, current, increment(b,current)),
                      select(down, increment(b,current,decrement=True), current))
    changed['cp'] = select(push, increment(b,states['cp']), states['cp'])
    changed['cr'] = select(take, increment(b,states['cr']), states['cr'])
    changed['cn'] = counted('cn', push, take)
    changed['reads'] = counted('reads', take, inputs['m_rvalid'])
    changed['writes'] = counted('writes', 0, inputs['m_bvalid'])
    collect = both(b.invert(ar_fire),take)
    changed['pending_addr'] = select(both(collect,count_zero), stored[6:24], states['pending_addr'])
    changed['next_addr'] = select(collect, increment(b,stored+[0]), states['next_addr'])
    changed['pending_count'] = select(ar_fire, [0]*5,
                                     select(take, increment(b,states['pending_count']), states['pending_count']))
    collected_limit = b.ite(equal(states['pending_count'],15), 1,
                           b.ite(equal(stored[:6],63),1,equal(states['reads'],127)))
    waiting = both(b.invert(count_zero),b.invert(pending))
    flush = b.ite(b.invert(cn_zero),1,
                 b.ite(equal(states['idle_count'],3),1,equal(states['reads'],128)))
    changed['pending_valid'] = [b.ite(ar_fire,0,
        b.ite(take,b.ite(collected_limit,1,pending),b.ite(both(waiting,flush),1,pending)))]
    changed['idle_count'] = select(ar_fire,[0]*3,select(take,[0]*3,
        select(both(waiting,b.invert(flush)),increment(b,states['idle_count']),states['idle_count'])))
    return {name: [b.ite(rst,0,value) for value in values] for name,values in changed.items()}


def model(pc, channel, equation):
    """Build current-state cuts only after the complete equation replay passes."""
    b = P.BDD(); cuts = {}; state_outputs = {}; states = {}
    scope = N.ROOT + 'u_native' + str(pc) + '/'
    for output in equation['state_output_assumptions']:
        reg = output[:-2]; local = reg[len(scope):]
        match = re.fullmatch(r'(\w+)_reg(?:\[([0-9]+)\])?',local)
        N.require(match is not None, 'unrecognized admission state mapping')
        family = match[1]; index = int(match[2]) if match[2] is not None else 0
        variable = (f'10-address-{index:02}-{family}' if family in ('next_addr','pending_addr')
                    else '00-state-' + local.removesuffix('_reg'))
        value = b.var(variable); cuts[output] = value
        states.setdefault(family,{})[index] = value
        state_outputs.setdefault(family,{})[index] = output
    stored = []
    for bit in range(24):
        output = equation['ram_bit_mapping'][str(bit)]['output']
        value=b.var(f'10-address-{bit:02}-cq');cuts[output]=value;stored.append(value)
    for info in channel['interfaces'].values():
        output=info['driver'];node=channel['nodes'].get(output)
        if node is None or node['reference'] in ('GND','VCC'):continue
        if output not in cuts:cuts[output]=b.var('20-input-'+min(node['roles']))
    c=E.Circuit(channel['nodes'],cuts,b)
    inputs={role:c.value(channel['interfaces'][role]['driver'])
            for role in ('rst','n_cvalid','m_bvalid','m_arready','m_rvalid')}
    normalized={family:[values[index] for index in sorted(values)] for family,values in states.items()}
    outputs={family:[values[index] for index in sorted(values)] for family,values in state_outputs.items()}
    return b,c,normalized,outputs,inputs,stored


def review_channel(pc, channel):
    equation=E.review_channel(pc,channel)
    b,c,states,outputs,inputs,stored=model(pc,channel,equation)
    N.require({key:len(value) for key,value in states.items()} == {
        'cn':5,'cp':4,'cr':4,'idle_count':3,'next_addr':25,'pending_addr':18,
        'pending_count':5,'pending_valid':1,'reads':8,'writes':8}, 'admission transition state widths')
    N.require(channel['interfaces']['clk']['driver'] == N.ROOT+'u_core_clk/inst/clkout1_buf/O',
              'admission transition clock identity')
    expected=expected_next(b,states,inputs,stored,c.value(channel['root']))
    checked=[]
    for family,observed_outputs in outputs.items():
        N.require(len(expected[family])==len(observed_outputs),'admission transition vector width')
        for output,wanted in zip(observed_outputs,expected[family]):
            node=channel['nodes'][output];controls=E.roles(node)
            for role,edge in controls.items():Q.polarity(node,role,edge)
            N.require(sorted(role for role,edge in controls.items()
                             if P.bit(edge['properties']['IS_CLOCK'])) == ['C'],
                      'admission register clock pin roles')
            actual=b.ite(c.edge(controls['R']),0,
                         b.ite(c.edge(controls['CE']),c.edge(controls['D']),c.value(output)))
            N.require(actual==wanted,'admission next-state equation differs: '+output)
            checked.append({'cell':node['cell'],'init':0,'next_state_equivalent':True,
                            'clock_driver':controls['C']['driver'],
                            'effective_clock_inversion':P.bit(controls['C']['properties']['IS_INVERTED'])})
    N.require(len(checked)==81,'admission transition register inventory')
    # Equation replay established exact data bits, all ten address pins, common
    # write event, zero INIT and rising clock for every RAMD32. Its primitive
    # model therefore writes bit n_caddr[b] at cp on push and reads at cr; no
    # unconstrained unknown RAM control is silently accepted.
    memories=[{'cell':row['cell'],'bit':int(bit),'depth':32,'used_addresses':16,
               'init':0,'write_enable':'!rst && n_cvalid && cn<16',
               'write_data_role':row['address_role'],'write_address':'cp[3:0]',
               'read_address':'cr[3:0]','clock_driver':row['inputs']['CLK']}
              for bit,row in sorted(equation['ram_bit_mapping'].items(),key=lambda item:int(item[0]))]
    return {'root':channel['root'],'checked_registers':checked,'checked_memories':memories,
            'one_step_equivalent_under_source_specialization':True,
            'configuration_required':dict(CONFIGURATION),'rtl_sha256':E.RTL_SHA256,
            'equation':equation['equation'],'boolean_nodes':len(b.nodes),
            'boolean_variables':sorted(b.variables),'source_specialization_verified':False,
            'whole_native_equivalence_proven':False,'paired_preservation_proven':False}


def review_artifact(**binding):
    result={'schema':1,'kind':'c1-admission-transition-review','issues':[],
            'one_step_equivalent_under_source_specialization':False,
            'source_specialization_verified':False,'whole_native_equivalence_proven':False,
            'paired_preservation_proven':False,'route_qualified':False,'hardware_measured':False,
            'canonical_facts':{},'pending_reviews':['source_specialization_and_configuration_binding',
                'linked_final_accounting_preservation','admission_ram_state_address_control_preservation']}
    try:
        graph=A.review_artifact(**binding)
        N.require(graph.get('capture_complete') is True and graph.get('issues')==[],
                  'authenticated admission graph incomplete: '+str(graph.get('issues')))
        channels=graph['canonical_facts']['channels']
        N.require(set(channels)=={str(i) for i in range(8)},'admission transition channel inventory')
        canonical={'graph_facts_sha256':graph['facts_sha256'],
                   'channels':{pc:review_channel(pc,channels[pc]) for pc in sorted(channels)}}
        result.update(one_step_equivalent_under_source_specialization=True,capture=graph['capture'],canonical_facts=canonical)
        result['facts_sha256']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    except (ValueError,OSError,KeyError,TypeError,IndexError,RecursionError) as exc:
        result['issues'].append({'reason':str(exc)})
    return result
