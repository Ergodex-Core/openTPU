from dataclasses import replace
import numpy as np
import pytest
from opentpu import Config, isa as I
from opentpu.isasim import Machine
from opentpu.perfsim import Architecture, MemoryConfig, Simulator, Unsupported
from opentpu.perfsim.engine import Banks
from opentpu.perfsim.memory import MemorySystem


def run(program, image=None, cfg=None, **kw):
    cfg=cfg or Config(TMEM_WORDS=1024,DRAM_BYTES=65536)
    image=np.zeros(65536,np.uint8) if image is None else image
    sim=Simulator(cfg,program,image,**kw)
    result=sim.run()
    ref=Machine(cfg,[program],[image]).run().slices[0]
    assert np.array_equal(sim.state.dram,ref.dram)
    assert np.array_equal(sim.state.tmem,ref.tmem)
    assert np.array_equal(sim.state.act,ref.act)
    return sim,result


def fill(dst,n,v=1):
    return I.vop(I.V_FILL,dst,0,0,1,n,n,n,0,I.B_SCALAR,v)


def test_dependencies_war_waw_raw_and_overlap():
    p=[fill(0,64,2), I.vop(I.V_MUL,64,0,0,1,64,64,64,0,I.B_SCALAR,3),
       fill(0,64,9), fill(128,64,5), I.st(0,64,64), I.halt()]
    sim,r=run(p)
    assert np.all(sim.state.dram[:256].view(np.float32)==6)
    a,b,c,d,e=r['instructions']
    assert b['start']>=a['end'] and c['start']>=b['end']
    assert d['start']<c['end']
    assert e['start']>=b['end']


def test_persistent_memory_after_compute_only_program():
    cfg=Config(TMEM_WORDS=1024,DRAM_BYTES=65536)
    memory=MemorySystem(MemoryConfig())
    first=Simulator(cfg,[fill(0,19),I.halt()],np.zeros(4,np.uint8),memory_system=memory)
    a=first.run()
    last=memory.cycle
    second=Simulator(cfg,[fill(0,19),I.halt()],np.zeros(4,np.uint8),memory_system=memory)
    b=second.run()
    assert second.memory_offset==last+1
    assert a['instructions'][0]['id']!=b['instructions'][0]['id']
    with pytest.raises(ValueError,match='monotonically'):
        memory.begin_cycle(memory.cycle,5000)
    memory.close()


def test_scalar_loop_rld_and_barrier():
    p=[fill(0,1,3), I.rld(1,0), I.loop(2,0,rcount=1),
       I.vop(I.V_ADD,0,0,0,1,1,1,1,0,I.B_SCALAR,1),I.nop(), I.bar(), I.st(0,0,1),I.halt()]
    sim,r=run(p)
    assert sim.state.dram[:4].view(np.float32)[0]==6
    assert r['frontend_stalls']['scalar_dependency']>0


@pytest.mark.parametrize('f',[I.V_EXP2,I.V_RECIP,I.V_RSQRT,I.V_LOG2,I.V_RSUM,I.V_RMAX,I.V_RSSQ,I.V_RDOT,I.V_ARGMAX])
def test_nonlinear_and_reduction_partial_vectors(f):
    run([fill(0,19,2),I.vop(f,64,0,0,1,19,32,32,0,I.B_SCALAR,3),I.halt()])


def test_bank_conflict_and_capacity():
    b=Banks(Config(),Architecture(tmem_write_ports_per_bank=1))
    assert b.grant(writes=[0,1])
    assert not b.grant(writes=[8])
    b.reset()
    assert b.grant(writes=[8])
    with pytest.raises(Unsupported):
        b.grant(writes=[0,8])


def test_closed_loop_latency_and_backpressure():
    p=[I.ld(0,0,128),I.vop(I.V_ADD,128,0,0,1,128,128,128,0,I.B_SCALAR,1),I.st(512,128,128),I.halt()]
    _,fast=run(p,memory=MemoryConfig(simple_latency_ticks=2,outstanding=4))
    _,slow=run(p,memory=MemoryConfig(simple_latency_ticks=80,outstanding=4))
    assert slow['cycles']>fast['cycles']
    assert slow['memory']['read_bytes']==fast['memory']['read_bytes']==512
    assert slow['memory']['write_bytes']==512
    assert slow['memory']['peak_outstanding']<=4


def test_memory_retry_unique_ids_and_partial_write_rmw():
    m=MemorySystem(MemoryConfig(record_trace=True,outstanding=4,simple_queue_per_channel=1,simple_channels=1,simple_latency_ticks=9))
    m.begin_cycle(0,1000)
    ticket=m.request(3,27,True,'B',0)
    assert m.request(32,16,False,'B',1) is None
    for cycle in range(1,100):
        m.begin_cycle(cycle,1000)
        if ticket.done: break
    assert ticket.done and m.idle
    accepts=[x for x in m.trace if x['event']=='accept']
    completes=[x for x in m.trace if x['event']=='complete']
    assert len(accepts)==len(completes)==4
    assert len({x['id'] for x in accepts})==4
    assert m.stats['read_bytes']==m.stats['write_bytes']==32
    assert m.stats['backend_rejected_ticks']>0


def test_configuration_and_unknown_opcode_fail_closed():
    with pytest.raises(Unsupported):
        Simulator(Config(S=2),[I.halt()],np.zeros(4,np.uint8))
    with pytest.raises(Unsupported):
        Simulator(Config(),[I.Instr(0xff)],np.zeros(4,np.uint8))
    with pytest.raises(ValueError,match='no fallback'):
        MemorySystem(MemoryConfig(backend='ramulator'))
    with pytest.raises(Unsupported):
        MemorySystem(MemoryConfig(controller_period_ps=2000))


def test_memory_clock_independent_of_core_clock():
    p=[I.ld(0,0,32),I.halt()]
    _,a=run(p,architecture=Architecture(core_period_ps=5000))
    _,b=run(p,architecture=Architecture(core_period_ps=1000))
    assert b['cycles']>a['cycles']
    assert b['seconds']>a['seconds']/5


@pytest.mark.parametrize('fmt,pair',[('int8',False),('int4',False),('int4',True),('fp4',False),('nvfp4',False)])
def test_compiled_partial_gemm_and_packed_formats(fmt,pair):
    from opentpu import language as ol
    from opentpu.runtime import compile_kernel, Input,Weight,Output,collect
    @ol.jit
    def kernel(x,w,y):
        ol.store(y,ol.dot(ol.load(x),w))
    cfg=Config(MCOLS=4,PAIR=pair,TMEM_WORDS=4096,DRAM_BYTES=65536)
    rng=np.random.default_rng(7)
    x=rng.normal(size=(3,64)).astype(np.float32)
    w=rng.normal(size=(13,64)).astype(np.float32)
    comp,images=compile_kernel(kernel,cfg,x=Input(x),w=Weight(w,fmt=fmt),y=Output((3,13)))
    sim,r=run(comp.programs[0],images[0],cfg)
    y=collect(comp,[sim.state.dram],[sim.state.tmem],{}).outputs['y']
    relative=np.linalg.norm(y-x@w.T)/np.linalg.norm(x@w.T)
    assert relative<(0.03 if fmt=='int8' else 0.3)
    assert sum(c.work['macs'] for c in sim.commands)==3*13*64
    assert sum(c.work['mxu_advances'] for c in sim.commands)>0


def test_partial_writes_serialize_read_modify_write():
    m=MemorySystem(MemoryConfig(record_trace=True,outstanding=8,simple_latency_ticks=3))
    m.begin_cycle(0,1000)
    a=m.request(1,1,True,'SW',0)
    m.begin_cycle(1,1000)
    b=m.request(2,1,True,'SW',1)
    for cycle in range(2,50):
        m.begin_cycle(cycle,1000)
        if a.done and b.done: break
    assert a.done and b.done
    first_write_end=next(x['time_ps'] for x in m.trace if x['event']=='complete' and x['owner']==0 and x['write'])
    second_read=next(x['time_ps'] for x in m.trace if x['event']=='accept' and x['owner']==1)
    assert second_read>=first_write_end


def test_identical_zero_and_dense_weights_do_not_get_sparse_speedup():
    cfg=Config(MCOLS=4,TMEM_WORDS=1024,DRAM_BYTES=65536)
    p=[fill(0,32),I.qact(0,1,0,1,32),I.mm(0,0,128,7,1,32,7,1,0,0,unit=True),I.halt()]
    zero=np.zeros(65536,np.uint8); dense=zero.copy(); dense[:224]=1
    _,a=run(p,zero,cfg); _,b=run(p,dense,cfg)
    assert a['cycles']==b['cycles']
    assert a['memory']==b['memory']


def test_window_pressure_and_fifo_depth_affect_timing():
    p=[fill(i*32,32) for i in range(12)]+[I.halt()]
    _,small=run(p,architecture=Architecture(window=2))
    _,large=run(p,architecture=Architecture(window=16))
    assert small['frontend_stalls']['window_full']>0
    assert small['cycles']>=large['cycles']


def test_unknown_flags_are_not_silently_ignored():
    with pytest.raises(Unsupported):
        Simulator(Config(),[I.Instr(I.LD,flags=128),I.halt()],np.zeros(4,np.uint8))


def test_persistent_backend_keeps_time_monotonic():
    m=MemorySystem(MemoryConfig(record_trace=True))
    p=[I.ld(0,0,8),I.halt()]
    a=Simulator(Config(),p,np.zeros(32,np.uint8),memory_system=m); a.run()
    split=len(m.trace)
    b=Simulator(Config(),p,a.state.dram,memory_system=m); b.run()
    assert m.trace[split]['time_ps']>m.trace[split-1]['time_ps']
    m.close()
