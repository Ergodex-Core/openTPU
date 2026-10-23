"""C1 host/RTL-fixture agreement and fail-closed ABI tests; no FPGA is accessed."""
import struct
import numpy as np
import pytest
from opentpu import c1_f2 as C, c1_rtl as R, c1_hbm_rtl as H
from opentpu.host.c1_f2 import canonical_bdf

@pytest.mark.parametrize('n,kb,mode',[(1,1,'quantized'),(17,9,'random'),(31,128,'negative'),(512,1,'special')])
def test_host_layout_and_ocl_words_match_actual_rtl_fixture(n,kb,mode,tmp_path):
    case=R.workload(n=n,kb=kb,ab=0,k0=0,mode=mode)
    fixture=C.Fixture.from_workload(case,base_beat=64)
    bits,integers=fixture.reference()
    assert np.array_equal(bits,case.expected)
    assert integers==[v-(1<<48) if v&(1<<47) else v for v in case.accumulators]
    path=tmp_path/'rtl.bin';H.write_fixture(case,path)
    raw=path.read_bytes();rows,blocks,plain,base,beats,nops=struct.unpack('>6I',raw[:24])
    assert (rows,blocks,plain,base)==(n,kb,0,fixture.base_beat)
    at=24
    for bank in fixture.banks():
        got=b''.join(raw[i:i+64][::-1] for i in range(at,at+beats*64,64))
        assert got==bank.tobytes();at+=beats*64
    ops=[struct.unpack('>II',raw[i:i+8]) for i in range(at,len(raw),8)]
    assert len(ops)==nops
    fake=Fake();bench=C.Benchmark(fake,expected_build_id=1234);bench.preload(fixture)
    assert [pair for pair in fake.writes if pair[0] in (C.R_MEM_ADDR,C.R_MEM_WDATA)]==ops
    assert sum(fixture.expected_pc_beats())*64==n*kb*128


def test_roundtrip_bounds_and_tail_padding(tmp_path):
    f=C.make_fixture(17,3);path=tmp_path/'f.npz';f.save(path)
    restored=C.Fixture.load(path)
    assert restored.digest()==f.digest()
    banks=f.banks()
    for row in range(17,32):
        start=((row//16)*2+row%2)*f.kb*128
        assert not np.any(banks[(row%16)//2][start:start+f.kb*128])
    f.base_beat=C.PC_BYTES//64-1
    with pytest.raises(ValueError,match='allocation'):f.validate()
    with pytest.raises(ValueError,match='128'):C.make_fixture(1,129)
    with pytest.raises(ValueError,match='origin-zero'):
        C.Fixture.from_workload(R.workload(n=1,kb=1,ab=1))


class Fake:
    evidence_kind='unit-test'
    def __init__(self):
        self.regs={C.R_ID:C.ID,C.R_ABI:C.ABI,C.R_BUILD:1234,C.R_CORE_KHZ:200000,
                   C.R_MAX_ROWS:512,C.R_DEPTH:128,C.R_CREDITS:32,C.R_HBM_READY:1}
        self.mem={};self.pcis={};self.writes=[];self.corrupt=None;self.hang=False
    def reg_read(self,addr):
        if addr==C.R_MEM_RDATA:
            word=self.regs[C.R_MEM_ADDR];self.regs[C.R_MEM_ADDR]+=1;return self.mem[word]
        return self.regs.get(addr,0)
    def reg_write(self,addr,value):
        self.writes.append((addr,value))
        if addr==C.R_MEM_WDATA:
            self.mem[self.regs[C.R_MEM_ADDR]]=value;self.regs[C.R_MEM_ADDR]+=1
        elif addr==C.R_CTRL:
            if value==C.CLEAR:self.regs[C.R_STATUS]=self.regs[C.R_ERROR]=0
            elif value==C.ABORT:self.regs[C.R_STATUS]=C.DONE|C.ABORTED
            elif value==C.START:
                if self.hang:self.regs[C.R_STATUS]=C.BUSY
                else:self.execute()
        else:self.regs[addr]=value
    def mem_write(self,pc,offset,data):self.pcis[pc,offset]=data.copy()
    def mem_read(self,pc,offset,n):
        data=self.pcis[pc,offset][:n].copy()
        if self.corrupt=='preload':data[0]^=1
        return data
    def execute(self):
        n,kb,base=(self.regs[a] for a in (C.R_ROWS,C.R_KB,C.R_BASE))
        w=np.empty((n,kb,128),np.int8)
        for row in range(n):
            at=((row//16)*2+row%2)*kb*128
            w[row]=self.pcis[(row%16)//2,base*64][at:at+kb*128].view(np.int8).reshape(kb,128)
        a=np.array([self.mem[C.ACT+i] for i in range(kb*32)],dtype='<u4').view(np.int8).reshape(kb,128)
        da=np.array([self.mem[C.DA+i] for i in range(kb)],dtype='<u4').view(np.uint8).reshape(kb,4)
        dw=np.array([[self.mem[C.DW+r*128+k] for k in range(kb)] for r in range(n)],np.uint8)
        sw=np.array([self.mem[C.SW+r] for r in range(n)],np.uint32)
        f=C.Fixture(w,a,dw,da,sw,np.array([self.mem[C.SA]],np.uint32),base)
        bits,ints=f.reference()
        for r,(value,integer) in enumerate(zip(bits,ints)):
            self.mem[C.RESULT+r]=int(value);self.mem[C.INTLO+r]=integer&0xffffffff;self.mem[C.INTHI+r]=(integer>>32)&0xffff
        self.regs.update({C.R_STATUS:C.DONE,C.R_COUNT:n,C.R_CYCLES_LO:1000,C.R_CYCLES_HI:0})
        for name,reg in C.COUNTERS.items():
            self.regs[reg]=(n*kb*4 if name in ('accepted_32B_groups','advance') else n if name=='committed_rows' else n*kb*2 if name in ('read_issued','read_returned') else 0)
        for pc,beats in enumerate(f.expected_pc_beats()):
            for name,off in C.PC_COUNTERS.items():self.regs[0x100+pc*0x20+off]=beats if name in ('issued','returned') else 0
        if self.corrupt=='bits':self.mem[C.RESULT]^=1
        elif self.corrupt=='integer':self.mem[C.INTLO]^=1
        elif self.corrupt=='count':self.regs[C.R_COUNT]-=1
        elif self.corrupt=='pc':self.regs[0x104]+=1
        elif self.corrupt=='groups':self.regs[0x40]-=1
        elif self.corrupt=='rows':self.regs[0x44]-=1
        elif self.corrupt=='advance':self.regs[0x50]-=1
        elif self.corrupt=='error':self.regs[C.R_ERROR]=2
        elif self.corrupt=='wrap':self.regs[C.R_CYCLES_HI]=1
        elif self.corrupt=='reserved':self.regs[0x110]=1
        elif self.corrupt=='stall':self.regs[0x60]=1
        elif self.corrupt=='pc_overflow':self.regs[0x108]=65
        elif self.corrupt=='pc_capacity':self.regs[0x108]=64
        elif self.corrupt=='aggregate_overflow':self.regs[0x68]=513


def ready():
    fake=Fake();b=C.Benchmark(fake,expected_build_id=1234);b.preload(C.make_fixture(17,9));return fake,b


def test_checked_plain_mode_repeats_and_two_engine_credit_limit():
    fake,b=ready();fake.corrupt='pc_capacity';results=[b.run() for _ in range(3)]
    assert len({tuple(r['bits']) for r in results})==1
    assert all(r['evidence_kind']=='unit-test' and r['bit_exact'] for r in results)
    assert all(r['weight_bytes']==17*9*128 for r in results)
    assert results[0]['counters']['accepted_32B_groups']==17*9*4
    assert results[0]['counters']['committed_rows']==17
    assert not hasattr(C,'lens_data')  # No fake/hardware render path until session integration.

@pytest.mark.parametrize('corrupt',['bits','integer','count','pc','groups','rows','advance','error','wrap','reserved','stall','pc_overflow','aggregate_overflow'])
def test_bad_results_or_counters_block_acceptance(corrupt):
    fake,b=ready();fake.corrupt=corrupt
    with pytest.raises(RuntimeError):b.run()

@pytest.mark.parametrize('reg,value',[(C.R_ID,0x43324d42),(C.R_ABI,2),(C.R_DEPTH,256),(C.R_CORE_KHZ,250000)])
def test_wrong_image_or_geometry_writes_nothing(reg,value):
    fake=Fake();fake.regs[reg]=value
    with pytest.raises(RuntimeError):C.Benchmark(fake,expected_build_id=1234)
    assert not fake.writes

def test_preload_fence_readback_mismatch_blocks_start():
    fake=Fake();fake.corrupt='preload';b=C.Benchmark(fake,expected_build_id=1234)
    with pytest.raises(RuntimeError,match='readback'):b.preload(C.make_fixture(1,1))
    assert (C.R_CTRL,C.START) not in fake.writes

def test_busy_mutated_and_foreign_mode_do_not_start():
    fake,b=ready();fake.writes.clear()
    with pytest.raises(ValueError):b.run(1)
    fake.regs[C.R_STATUS]=C.DRAINING
    with pytest.raises(RuntimeError,match='owns native'):b.run()
    fake.regs[C.R_STATUS]=0;b.fixture.weights[0,0,0]^=1
    with pytest.raises(RuntimeError,match='changed after'):b.run()
    assert not fake.writes

def test_timeout_drains_and_canonical_lock_identity():
    fake,b=ready();fake.hang=True
    with pytest.raises(TimeoutError):b.run(timeout=0.001)
    assert (C.R_CTRL,C.ABORT) in fake.writes
    assert not fake.reg_read(C.R_STATUS)&(C.BUSY|C.DRAINING)
    assert canonical_bdf('34:00.0')==canonical_bdf('0000:34:00.0')=='0000:34:00.0'
    with pytest.raises(ValueError):canonical_bdf('f2:0')


def test_offline_fixture_cli_and_session_helper(tmp_path):
    from tools import c1_f2_bench as B
    import json
    path=tmp_path/'matrix.npz';B.main(['--fixture',str(path),'--rows','17','--kb','3'])
    fixture=C.Fixture.load(path);record=json.loads(path.with_suffix('.json').read_text())
    assert record['hardware_measured'] is False
    assert record['useful_weight_bytes']==record['native_read_bytes']==17*3*128
    with pytest.raises(ValueError,match='fresh'):B.main(['--fixture',str(path)])
    results=list(B.checked_runs(Fake(),fixture,1234,repeats=2))
    assert [r['repeat'] for r in results]==[0,1]
    assert all(r['mode']==0 and r['evidence_kind']=='unit-test' for r in results)
    with pytest.raises(ValueError):list(B.checked_runs(Fake(),fixture,1234,repeats=0))
