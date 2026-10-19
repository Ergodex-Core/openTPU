"""Offline orchestration and monitor tests; these do not simulate vendor IP.

The full four-scenario vendor run requires Vivado 2025.2 and the exact generated
9SI/8MI receipt. Icarus stubs below test elaboration and fail-closed scoreboards
only; they cannot produce a fabric PASS or performance result.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

s = load('c2_alex_sim', ROOT / 'boards/c2_f2_alex/simulate.py')
f = load('c2_alex_build_fixture', ROOT / 'tests/test_c2_alex_build.py')


def ip_fixture(tmp_path):
    run, origin = f.make_ip_run(tmp_path)
    origin.update(schema=1, kind='c2-alex-burst200-build-inputs', part=s.build.PART,
                  core_khz=200000, hbm_mhz=450, hbm_ports=8, channel_bytes=2**29,
                  source_status='', source_revision='a'*40,
                  support_manifest_sha256=s.build.sha(s.build.SUPPORT / 'manifest.json'))
    origin['source_inputs'] = {name: {'sha256': s.build.sha(ROOT / name)} for name in
                             ('rtl/c2_alex/generate.py', 'rtl/c2_alex/create_f2_ip.tcl',
                              'boards/c2_f2_alex/build.py')}
    s.build.write_json(run / 'inputs.json', origin)
    receipt = json.loads((run / 'ip-complete.json').read_text())
    receipt['inputs_sha256'] = s.build.sha(run / 'inputs.json')
    s.build.write_json(run / 'ip-complete.json', receipt)
    return run, s.build.sha(run / 'ip-complete.json')


def pass_log(seed=17, phase=0):
    lines=[]
    for pc in range(8):
        n=610 if pc in (2,6) else 609
        lines.append(f'C2_PC pc={pc} accepted={n} returned={n} consumed={n} axi_beats={n} peak=32 hbm_reads=2000 hbm_writes=1000')
    return '\n'.join(lines)+f'\nC2_ALEX_FABRIC_PASS seed={seed} phase={phase} pcs=8 reset_pending=8 host_b=994 host_r=1229\n'


def test_ip_receipt_exact_binding_and_readonly_plan(tmp_path, monkeypatch, capsys):
    run, digest=ip_fixture(tmp_path)
    def forbidden(*a,**kw): raise AssertionError('plan must not execute')
    monkeypatch.setattr(s.build, 'command', forbidden)
    s.main(['--ip-run',str(run),'--ip-receipt-sha256',digest,'--out',str(tmp_path/'out'),'--plan'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['scenarios']==[[17,0],[17,1],[71,0],[71,1]]
    assert plan['private_ip_copy'] is True and plan['hardware_actions'] is False
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change',['receipt','inputs','product','source','topology','marker'])
def test_ip_rejects_mutations(tmp_path,change):
    run,digest=ip_fixture(tmp_path)
    if change=='receipt': (run/'ip-complete.json').write_text('{}')
    elif change=='inputs': (run/'inputs.json').write_text('{}')
    elif change=='product': (run/'ip/ip_project/child.dcp').write_text('bad')
    elif change=='marker': (run/'ip/ip_done.txt').write_text('PENDING')
    else:
        if change=='source':
            origin=json.loads((run/'inputs.json').read_text())
            origin['source_inputs']['rtl/c2_alex/generate.py']['sha256']='0'*64
            s.build.write_json(run/'inputs.json',origin)
        else:
            bd=json.loads((run/s.BD).read_text());bd['design']['components']['memory_sc']['parameters']['NUM_SI']['value']='3'
            s.build.write_json(run/s.BD,bd)
        receipt=json.loads((run/'ip-complete.json').read_text())
        receipt['inputs_sha256']=s.build.sha(run/'inputs.json');receipt['products']=s.build.ip_inventory(run)
        s.build.write_json(run/'ip-complete.json',receipt);digest=s.build.sha(run/'ip-complete.json')
    with pytest.raises((RuntimeError,KeyError)):
        s.verified_ip(run,digest)


def test_private_copy_preserves_source_and_detects_product_mutation(tmp_path):
    run,digest=ip_fixture(tmp_path);out=tmp_path/'sim';out.mkdir()
    private,files=s.copy_ip(run,out)
    assert files and private!=run
    s.verified_ip(private,digest)
    (private/'ip/ip_project/child.dcp').write_text('changed-by-simulator')
    with pytest.raises(RuntimeError,match='inventory changed'):s.verified_ip(private,digest)
    s.verified_ip(run,digest)


@pytest.mark.parametrize('change',['disk','symlink'])
def test_private_copy_refuses_unsafe_inputs(tmp_path,monkeypatch,change):
    run,_=ip_fixture(tmp_path);out=tmp_path/'sim';out.mkdir()
    if change=='disk': monkeypatch.setattr(s.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    else: (run/'ip/ip_project/escaped.v').symlink_to(run/'inputs.json')
    with pytest.raises(RuntimeError):s.copy_ip(run,out)
    assert not (out/'private-ip').exists()


def test_parse_requires_all_conservation_evidence():
    r=s.parse_result(pass_log(),17,0)
    assert len(r['pcs'])==8 and r['pcs'][2]['accepted']==610


@pytest.mark.parametrize('change',['missing','duplicate','wrong_seed','wrong_pc','lost_beat','credit','no_hbm','fatal','wrong_host'])
def test_parse_rejects_false_success(change):
    text=pass_log()
    if change=='missing':text='PASS\n'
    elif change=='duplicate':text+=text
    elif change=='wrong_seed':text=text.replace('seed=17','seed=71')
    elif change=='wrong_pc':text=text.replace('pc=7','pc=6')
    elif change=='lost_beat':text=text.replace('returned=609','returned=608',1)
    elif change=='credit':text=text.replace('peak=32','peak=31',1)
    elif change=='no_hbm':text=text.replace('hbm_reads=2000','hbm_reads=0',1)
    elif change=='fatal':text+='Fatal: assertion failed\n'
    else:text=text.replace('host_r=1229','host_r=1230')
    with pytest.raises(RuntimeError):s.parse_result(text,17,0)


def test_vendor_invocation_and_scenario_result(tmp_path):
    def fake(cmd,**kw):
        assert cmd[1:3]==['-mode','batch'] and cmd[3]=='-source'
        assert kw['timeout']==1800
        tcl=Path(cmd[4]).read_text()
        assert 'f2_mem.bd' in tcl and 'f2_pcis_pipe.xci' in tcl
        assert 'general.maxThreads 4' in tcl
        assert '-testplusarg SEED=71 -testplusarg PHASE=1' in tcl
        assert 'launch_simulation -simset sim_1 -mode behavioral' in tcl
        kw['stdout'].write(pass_log(71,1));return SimpleNamespace(returncode=0)
    result=s.run_scenario(tmp_path,tmp_path/'private',71,1,'vivado',runner=fake)
    assert result['seed']==71
    assert (tmp_path/'seed-71-phase-1/result.json').exists()


@pytest.mark.parametrize('failure',['tool','missing_pass','changed_input','timeout'])
def test_failed_scenario_never_writes_result(tmp_path,failure):
    def fake(cmd,**kw):
        if failure=='timeout':raise subprocess.TimeoutExpired(cmd,1800)
        if failure=='changed_input':(kw['cwd']/'rtl/tb_f2_mem_c2.sv').write_text('changed')
        kw['stdout'].write('simulation ended\n' if failure=='missing_pass' else pass_log())
        return SimpleNamespace(returncode=1 if failure=='tool' else 0)
    with pytest.raises((RuntimeError,subprocess.TimeoutExpired)):
        s.run_scenario(tmp_path,tmp_path/'private',17,0,'vivado',runner=fake)
    assert not (tmp_path/'seed-17-phase-0/result.json').exists()


def vendor_stubs():
    """Port declarations only, independently matching the generated BD ABI."""
    axi=dict(araddr=34,arburst=2,arcache=4,arid=1,arlen=8,arlock=1,arprot=3,arqos=4,arready=1,arsize=3,arvalid=1,
             awaddr=34,awburst=2,awcache=4,awid=1,awlen=8,awlock=1,awprot=3,awqos=4,awready=1,awsize=3,awvalid=1,
             bid=1,bready=1,bresp=2,bvalid=1,rdata=512,rid=1,rlast=1,rready=1,rresp=2,rvalid=1,wdata=512,wlast=1,wready=1,wstrb=64,wvalid=1)
    response={'arready','awready','wready','bid','bresp','bvalid','rid','rdata','rresp','rlast','rvalid'}
    host=dict(axi,araddr=64,awaddr=64,arid=16,awid=16,rid=16,bid=16)
    hbm={k:v for k,v in axi.items() if k not in ('arid','awid','rid','bid')}
    hbm.update(arlen=4,awlen=4,arlock=2,awlock=2,rdata=256,wdata=256,wstrb=32)
    def port(name,w,direction):return f'{direction} wire '+(f'[{w-1}:0] ' if w>1 else '')+name
    mem=[port(x,1,'input') for x in ('aresetn','core_clk','host_clk','hbm_clk')]
    for name,fields,is_master in [('S_HOST',host,False),*[(f'S_N{p}',axi,False) for p in range(8)],*[(f'M_H{p}',hbm,True) for p in range(8)]]:
        mem += [port(name+'_'+k,w,'input' if (k in response)==is_master else 'output') for k,w in fields.items()]
    pipe=[port(x,1,'input') for x in ('aclk','aresetn')]
    for pref,master in [('s_axi_',False),('m_axi_',True)]:
        pipe += [port(pref+k,w,'input' if (k in response)==master else 'output') for k,w in dict(host,awregion=4,arregion=4).items()]
    return 'module f2_mem_wrapper('+','.join(mem)+');endmodule\nmodule f2_pcis_pipe('+','.join(pipe)+');endmodule\n'


def compile_bench(tmp_path,injection=''):
    if not shutil.which('iverilog'):pytest.skip('Icarus required for local monitor/elaboration checks')
    stub=tmp_path/'vendor_ports.sv';stub.write_text(vendor_stubs())
    bench=tmp_path/'tb.sv';text=s.SOURCES['tb_f2_mem_c2.sv'].read_text()
    bench.write_text(text.rsplit('endmodule',1)[0]+injection+'\nendmodule\n')
    out=tmp_path/'bench.vvp'
    result=subprocess.run(['iverilog','-g2012','-s','tb_f2_mem_c2','-o',str(out),str(stub),str(bench),
                           *[str(p) for name,p in s.SOURCES.items() if name!='tb_f2_mem_c2.sv']],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'expects' not in result.stderr and 'dangling' not in result.stderr,result.stderr
    return out


def test_full_bench_elaborates_with_exact_vendor_port_abi(tmp_path):
    compile_bench(tmp_path)


@pytest.mark.parametrize('address',[0x20000000,0x100000000,0x20000])
def test_full_address_guard_rejects_alias_before_small_model(tmp_path,address):
    out=compile_bench(tmp_path,f'''
initial begin #1200;
 force H_araddr[0]=34'h{address:x};force H_arlen[0]=0;force H_arsize[0]=5;force H_arburst[0]=1;
 force H_arvalid[0]=1;force H_arready[0]=1;
 #10;$fatal(1,"address guard failed to reject");end
''')
    result=subprocess.run(['vvp',str(out)],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'HBM read wrong full address' in result.stdout,result.stdout+result.stderr


def test_independent_native_credit_monitor_rejects_unreserved_bursts(tmp_path):
    out=compile_bench(tmp_path,'''
initial begin #1200;
 force N_araddr[0]=0;force N_arlen[0]=15;force N_arvalid[0]=1;force N_arready[0]=1;
 force N_rvalid[0]=0;
 #30;$fatal(1,"credit guard failed to reject");end
''')
    result=subprocess.run(['vvp',str(out)],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'native AXI exceeds reserved receive capacity' in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('pc,address',[(0,0x3000),(7,0xe0003000)])
def test_full_address_guard_accepts_legal_physical_window(tmp_path,pc,address):
    out=compile_bench(tmp_path,f"""
initial begin #1200;
 force H_araddr[{pc}]=34'h{address:x};force H_arlen[{pc}]=0;force H_arsize[{pc}]=5;force H_arburst[{pc}]=1;
 force H_arvalid[{pc}]=1;force H_arready[{pc}]=1;
 #10;$display("MONITOR_ACCEPTED_LEGAL_ADDRESS");$finish;end
""")
    result=subprocess.run(['vvp',str(out)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and 'MONITOR_ACCEPTED_LEGAL_ADDRESS' in result.stdout,result.stdout+result.stderr


def test_timeout_kills_complete_vendor_process_group(monkeypatch,tmp_path):
    calls=[]
    class Process:
        pid=12345
        waits=0
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('vivado',1)
            return -15
    def popen(cmd,**kwargs):
        assert kwargs['start_new_session'] is True
        return Process()
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    monkeypatch.setattr(s.os,'killpg',lambda pid,sig:calls.append((pid,sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        s.run_vendor(['vivado'],cwd=tmp_path,stdout=None,stderr=None,timeout=1)
    assert calls==[(12345,s.signal.SIGTERM),(12345,s.signal.SIGKILL)]


@pytest.mark.parametrize('change',['include','script','added_include','removed','symlink'])
def test_all_simulation_dependencies_rechecked(tmp_path,change):
    run,_=ip_fixture(tmp_path)
    header=run/'ip/ip_project/helper.vh';header.write_text('frozen include')
    script=run/'ip/ip_project/helper.tcl';script.write_text('# frozen script')
    before=s.capture_simulation_ip_files(run)
    s.verify_simulation_ip_files(run,before)
    if change=='include':header.write_text('changed include')
    elif change=='script':script.write_text('# changed script')
    elif change=='added_include':(run/'ip/ip_project/added.svh').write_text('new include')
    elif change=='removed':header.unlink()
    else:
        header.unlink();header.symlink_to(script)
    with pytest.raises(RuntimeError):s.verify_simulation_ip_files(run,before)


@pytest.mark.parametrize('channel',['B','R'])
def test_unknown_host_response_cannot_pass(tmp_path,channel):
    if channel=='B':
        inject="""
initial begin #1200;
 force S_bvalid=1;force S_bresp=2'bxx;force S_bid=16'h8023;
 host_b(16'h8023);$display("UNKNOWN_ESCAPED");$finish;end
"""
        message='host B response/id'
    else:
        inject="""
initial begin #1200;
 expected[0][0]=0;
 force S_rvalid=1;force S_rresp=2'bxx;force S_rid=16'h8123;force S_rlast=1;force S_rdata=0;
 host_r(0,0,6,1,16'h8123);$display("UNKNOWN_ESCAPED");$finish;end
"""
        message='host R response/id/last'
    out=compile_bench(tmp_path,inject)
    result=subprocess.run(['vvp',str(out)],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and message in result.stdout,result.stdout+result.stderr
    assert 'UNKNOWN_ESCAPED' not in result.stdout


@pytest.mark.parametrize('mutate',[False,True])
def test_complete_receipt_binds_all_scenarios_and_dependencies(tmp_path,monkeypatch,mutate):
    run,digest=ip_fixture(tmp_path)
    (run/'ip/ip_project/helper.vh').write_text('extra simulation dependency')
    # The receipt's product filter intentionally omits .vh; simulator adds it.
    monkeypatch.setattr(s.build,'command',lambda *a,**kw:'Vivado v2025.2 (64-bit)')
    real=s.run_scenario
    calls=[]
    def fake_scenario(out,private,seed,phase,vivado):
        calls.append((seed,phase))
        def fake(cmd,**kw):
            kw['stdout'].write(pass_log(seed,phase))
            return SimpleNamespace(returncode=0)
        result=real(out,private,seed,phase,vivado,runner=fake)
        if mutate:(private/'ip/ip_project/helper.vh').write_text('changed while simulating')
        return result
    monkeypatch.setattr(s,'run_scenario',fake_scenario)
    out=tmp_path/'sim-out'
    argv=['--ip-run',str(run),'--ip-receipt-sha256',digest,'--out',str(out)]
    if mutate:
        with pytest.raises(RuntimeError,match='dependency inventory changed'):s.main(argv)
        assert not (out/'simulation-complete.json').exists()
        assert (out/'failure.json').exists()
        assert calls==[(17,0)]
    else:
        s.main(argv)
        receipt=json.loads((out/'simulation-complete.json').read_text())
        inputs=json.loads((out/'inputs.json').read_text())
        assert receipt['inputs_sha256']==s.build.sha(out/'inputs.json')
        assert len(receipt['results'])==4 and calls==list(s.SCENARIOS)
        assert 'ip/ip_project/helper.vh' in inputs['simulation_ip_files']
        for seed,phase in s.SCENARIOS:
            name=f'seed-{seed}-phase-{phase}/console.log'
            assert receipt['evidence'][name]['sha256']==s.build.sha(out/name)
        assert receipt['hardware_actions'] is False
