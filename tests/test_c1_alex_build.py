"""Offline tests for source binding and safe private-IP stage orchestration."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c1_alex_build', ROOT / 'boards/c1_f2_alex/build.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_frozen_archive_inventory():
    m = b.verify_support()
    assert len(m['sources']) == 33
    assert m['commit'] == '06c169cbcd1721ae8b167de992a93461697a397d'
    assert m['archive_sha256'] == 'fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc'


@pytest.mark.parametrize('change', ['bytes', 'extra', 'removed', 'provenance'])
def test_support_refuses_mutations(tmp_path, monkeypatch, change):
    support = tmp_path / 'support'
    shutil.copytree(b.SUPPORT, support)
    monkeypatch.setattr(b, 'SUPPORT', support)
    if change == 'bytes':
        (support / 'rtl/boards/f2/otpu_native_axi.sv').write_text('changed')
    elif change == 'extra':
        (support / 'injected.tcl').write_text('changed')
    elif change == 'removed':
        (support / 'rtl/boards/f2/otpu_native_axi.sv').unlink()
    else:
        m = json.loads((support / 'manifest.json').read_text())
        m['archive_sha256'] = '0' * 64
        b.write_json(support / 'manifest.json', m)
    with pytest.raises(RuntimeError):
        b.verify_support()


def test_plan_has_no_execution_or_output(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError('Read-only plan called a subprocess')
    monkeypatch.setattr(b.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(b.subprocess, 'run', forbidden)
    out = tmp_path / 'not-created'
    b.main(['--hdk', str(tmp_path / 'uninstalled'), '--out', str(out), '--stage', 'ip', '--plan'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['max_threads'] == 4 and plan['ip_jobs'] == 1
    assert plan['hardware_actions'] is False and plan['qualification'] == 'not-qualified'
    assert not out.exists()


@pytest.mark.parametrize('argv', [[], ['--ip-run', '/unused']])
def test_stage_ip_binding_arg_required(tmp_path, argv):
    stage = 'synth' if not argv else 'ip'
    with pytest.raises(SystemExit):
        b.main(['--hdk', '/unused', '--out', str(tmp_path / 'no'), '--stage', stage, '--plan', *argv])


@pytest.mark.parametrize('jobs', [1, 2, 3])
def test_thread_limits(jobs):
    text = 'set_param general.maxThreads 8\nlaunch_runs $jobs -jobs 3\nset_param general.maxThreads 6\n'
    bounded = b.bounded_script(text, jobs)
    assert bounded.count('general.maxThreads 4') == 3
    assert 'STEPS.SYNTH_DESIGN.TCL.PRE' in bounded
    assert f'launch_runs $jobs -jobs {jobs}' in bounded


@pytest.mark.parametrize('jobs', [0, 4, -1])
def test_bad_job_count(jobs):
    with pytest.raises(RuntimeError):
        b.bounded_script('', jobs)


def test_changed_anchor_fails():
    with pytest.raises(RuntimeError):
        b.replace_once('a a', 'a', 'b')
    with pytest.raises(RuntimeError):
        b.replace_once('x', 'a', 'b')


@pytest.mark.parametrize('bad', ['space name', 'x;bad', 'x[1]', 'x$bad', 'x{bad}', 'x\\bad'])
def test_tcl_unsafe_paths_refused(tmp_path, bad):
    with pytest.raises(RuntimeError):
        b.safe_path(tmp_path / bad)


def test_product_inventory_is_exact_and_excludes_logs(tmp_path):
    d = tmp_path / 'gen'
    d.mkdir()
    (d / 'child.v').write_text('module child;endmodule')
    (d / 'child.xci').write_text('{}')
    (d / 'child.dcp').write_bytes(b'dcp')
    (d / 'vivado.log').write_text('not synthesis input')
    before = b.inventory([d], tmp_path, products=True)
    assert set(before) == {'gen/child.v', 'gen/child.xci', 'gen/child.dcp'}
    (d / 'vivado.log').write_text('new log content')
    assert b.inventory([d], tmp_path, products=True) == before
    (d / 'child.v').write_text('module changed;endmodule')
    assert b.inventory([d], tmp_path, products=True) != before
    (d / 'new.mem').write_text('0')
    assert 'gen/new.mem' in b.inventory([d], tmp_path, products=True)


def test_product_directory_symlink_refused(tmp_path):
    d = tmp_path / 'gen'; d.mkdir()
    external = tmp_path / 'elsewhere'; external.mkdir()
    (external / 'hidden.v').write_text('module hidden;endmodule')
    (d / 'linked').symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match='Directory symlink'):
        b.inventory([d], tmp_path, products=True)


def test_vivado_version_banner():
    b.check_vivado('vivado v2025.2 (64-bit)')
    b.check_vivado('Vivado v2025.2 (64-bit)')
    with pytest.raises(RuntimeError):
        b.check_vivado('Vivado v2024.2; timestamp 2025.2')


def test_script_adaptation_preserves_baseline_and_removes_tmem(tmp_path):
    for stage in b.STAGES:
        (tmp_path / stage).mkdir()
    b.adapt_scripts(tmp_path, 1)
    synth = (tmp_path / 'synth/build.tcl').read_text()
    assert synth.index('C1_ALEX_SYNTH_ONLY_COMPLETE') < synth.index('close_project\ncreate_project')
    assert 'synth_design -mode out_of_context -top cl_opentpu' in synth
    assert 'CORE_KHZ mismatch' in synth
    assert 'clk_out1 1.0 clk_out2 2.0' not in synth
    for stage in b.STAGES:
        script = (tmp_path / stage / 'build.tcl').read_text()
        assert 'general.maxThreads 8' not in script and 'general.maxThreads 6' not in script
    floor = (tmp_path / 'implement/floorplan.tcl').read_text()
    for p in range(8):
        assert f'WRAPPER/CL/u_native{p}' in floor
    original = (b.SUPPORT / 'boards/f2/constraints/floorplan.tcl').read_text()
    assert floor.replace(' '.join(f'WRAPPER/CL/u_native{p}' for p in range(8)),
                         'WRAPPER/CL/u_native0 WRAPPER/CL/u_native1') == original
    timing = (tmp_path / 'implement/core_timing.tcl').read_text()
    assert 'set_multicycle_path' not in timing and 'source $run/otpu_clk2x.tcl' not in timing
    assert '200 MHz' in timing
    refine = (tmp_path / 'refine/build.tcl').read_text()
    assert 'foreach directive {Explore AggressiveExplore}' in refine
    assert 'route.optimizeLeafClkProgDlys false' in refine
    assert 'INTERNAL_VREF 0.84' in refine


def make_ip_run(tmp_path):
    out = tmp_path / 'ip-run'; out.mkdir()
    products = [
        'ip/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd',
        'ip/ip_project/f2_ip.gen/sources_1/bd/f2_mem/hdl/f2_mem_wrapper.v',
        'ip/ip_project/f2_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci',
        'ip/ip_project/f2_ip.srcs/sources_1/ip/f2_ocl_cdc/f2_ocl_cdc.xci',
        'pcis/ip_project/pcis.srcs/sources_1/ip/f2_pcis_pipe/f2_pcis_pipe.xci',
        'clocks/ip_project/clocks.srcs/sources_1/ip/f2_core_clk/f2_core_clk.xci',
        'clocks/ip_project/clocks.srcs/sources_1/ip/f2_hbm_clk/f2_hbm_clk.xci',
        'ip/ip_project/child.dcp']
    for name in products:
        p = out / name; p.parent.mkdir(parents=True, exist_ok=True);p.write_text(name)
    b.write_json(out / products[0], generated_bd())
    for stage in ('ip', 'pcis', 'clocks'):
        (out / stage / 'build.tcl').write_text(f'# {stage}')
        (out / stage / 'ip_done.txt').write_text('IP_DONE\n')
    origin = {'alex_revision': b.ALEX_REVISION, 'archive_sha256': b.ARCHIVE_SHA256,
              'support_manifest_sha256': 'a' * 64, 'installed_inputs': {'hdk': {'sha256': 'b' * 64}},
              'hdk_revision': b.HDK_REVISION, 'ip_revision': b.IP_REVISION,
              'prepared_files': b.inventory([out / s / 'build.tcl' for s in ('ip', 'pcis', 'clocks')], out)}
    b.write_json(out / 'inputs.json', origin)
    receipt = {'kind': 'c1-alex-private-ip', 'inputs_sha256': b.sha(out / 'inputs.json'),
               'products': b.ip_inventory(out)}
    b.write_json(out / 'ip-complete.json', receipt)
    return out, origin


def test_ip_receipt_binds_exact_inputs_and_products(tmp_path):
    out, origin = make_ip_run(tmp_path)
    assert b.verify_ip_run(out, origin)['products'] == b.ip_inventory(out)


@pytest.mark.parametrize('change', ['child', 'added', 'removed', 'script', 'marker', 'hdk', 'recipe', 'input_record'])
def test_ip_receipt_rejects_changes(tmp_path, change):
    out, origin = make_ip_run(tmp_path)
    if change == 'child':
        (out / 'ip/ip_project/child.dcp').write_text('changed')
    elif change == 'added':
        (out / 'ip/ip_project/new.mem').write_text('0')
    elif change == 'removed':
        (out / 'ip/ip_project/child.dcp').unlink()
    elif change == 'script':
        (out / 'ip/build.tcl').write_text('changed')
    elif change == 'marker':
        (out / 'ip/ip_done.txt').write_text('PENDING')
    elif change == 'hdk':
        origin['installed_inputs'] = {}
    elif change == 'recipe':
        origin['prepared_files']['ip/build.tcl']['sha256'] = 'c' * 64
    else:
        (out / 'inputs.json').write_text('{}')
    with pytest.raises(RuntimeError):
        b.verify_ip_run(out, origin)


def test_existing_output_refused_before_generating(tmp_path):
    with pytest.raises(RuntimeError, match='existing output'):
        b.prepare(tmp_path, tmp_path / 'hdk', '12345678', {})


def generated_bd():
    # Synthetic Vivado JSON-schema fixture only; this is not hardware evidence.
    def params(values):
        return {'parameters': {k: {'value': str(v)} for k, v in values.items()}}
    d = {'design_info': {'name': 'f2_mem', 'device': b.PART, 'tool_version': '2025.2', 'validated': 'true'},
         'components': {'memory_sc': {'vlnv': 'xilinx.com:ip:smartconnect:1.0',
                        **params({'NUM_SI': 9, 'NUM_MI': 8, 'NUM_CLKS': 3})}},
         'interface_ports': {}, 'interface_nets': {}, 'ports': {}, 'nets': {},
         'addressing': {'/': {'address_spaces': {}}}}
    for name, mode, protocol, width, addr, ident, freq, internal in [
            ('S_HOST', 'Slave', 'AXI4', 512, 64, 16, 250000000, 'S00_AXI'),
            *[(f'S_N{p}', 'Slave', 'AXI4', 512, 34, 1, 200000000, f'S{p+1:02d}_AXI') for p in range(8)],
            *[(f'M_H{p}', 'Master', 'AXI3', 256, 34, 0, 450000000, f'M{p:02d}_AXI') for p in range(8)]]:
        d['interface_ports'][name] = {'mode': mode, **params({'PROTOCOL': protocol,
            'DATA_WIDTH': width, 'ADDR_WIDTH': addr, 'ID_WIDTH': ident, 'FREQ_HZ': freq, 'MAX_BURST_LENGTH': 16})}
        d['interface_nets'][name] = {'interface_ports': [name, 'memory_sc/' + internal]}
    for name, freq, pin, associated in [('core_clk', 200000000, 'aclk', ':'.join(f'S_N{p}' for p in range(8))),
            ('host_clk', 250000000, 'aclk1', 'S_HOST'),
            ('hbm_clk', 450000000, 'aclk2', ':'.join(f'M_H{p}' for p in range(8)))]:
        d['ports'][name] = {'type': 'clk', **params({'FREQ_HZ': freq, 'ASSOCIATED_BUSIF': associated})}
        d['nets'][name] = {'ports': [name, 'memory_sc/' + pin]}
    d['ports']['aresetn'] = params({'POLARITY': 'ACTIVE_LOW'})
    d['nets']['aresetn'] = {'ports': ['aresetn', 'memory_sc/aresetn']}
    for source in ['S_HOST', *(f'S_N{p}' for p in range(8))]:
        targets = range(8) if source == 'S_HOST' else [int(source[3:])]
        segments = {}
        for p in range(8):
            segments[f'SEG_M_H{p}_Reg'] = ({'address_block': f'/M_H{p}/Reg', 'offset': hex(p * 2**29), 'range': '512M'}
                if p in targets else {'address_block': f'/M_H{p}/Reg', 'offset': '0x000000000', 'range': '1', 'is_excluded': 'TRUE'})
        d['addressing']['/']['address_spaces'][source] = {'segments': segments}
    return {'design': d}


def test_generated_bd_validation(tmp_path):
    path = tmp_path / 'f2_mem.bd'
    b.write_json(path, generated_bd())
    report = b.validate_memory_bd(path)
    assert report['slave_interfaces'] == 9 and report['master_interfaces'] == 8
    assert report['qualification'] == 'configuration-only-not-timing'


@pytest.mark.parametrize('change', ['old_topology', 'wrong_pc', 'native_extra', 'host_missing',
                                  'wrong_size', 'wrong_clock', 'wrong_net', 'wrong_width', 'wrong_id', 'unvalidated'])
def test_generated_bd_rejects_material_mismatches(tmp_path, change):
    data = generated_bd(); d = data['design']
    spaces = d['addressing']['/']['address_spaces']
    if change == 'old_topology':
        d['components']['memory_sc']['parameters']['NUM_SI']['value'] = '3'
    elif change == 'wrong_pc':
        spaces['S_N0']['segments']['SEG_M_H0_Reg']['address_block'] = '/M_H1/Reg'
    elif change == 'native_extra':
        spaces['S_N0']['segments']['extra'] = {'address_block': '/M_H1/Reg', 'offset': '0x20000000', 'range': '512M'}
    elif change == 'host_missing':
        spaces['S_HOST']['segments'].pop('SEG_M_H7_Reg')
    elif change == 'wrong_size':
        spaces['S_N0']['segments']['SEG_M_H0_Reg']['range'] = '2G'
    elif change == 'wrong_clock':
        d['ports']['core_clk']['parameters']['FREQ_HZ']['value'] = '450000000'
    elif change == 'wrong_net':
        d['interface_nets']['S_N0']['interface_ports'][-1] = 'memory_sc/S02_AXI'
    elif change == 'wrong_width':
        d['interface_ports']['S_HOST']['parameters']['ADDR_WIDTH']['value'] = '32'
    elif change == 'wrong_id':
        d['interface_ports']['M_H0']['parameters']['ID_WIDTH']['value'] = '6'
    else:
        d['design_info']['validated'] = 'false'
    path = tmp_path / 'f2_mem.bd'; b.write_json(path, data)
    with pytest.raises(RuntimeError, match='SmartConnect mismatch'):
        b.validate_memory_bd(path)


def test_source_dirty_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'command', lambda *a, **k: ' M rtl/mxu/otpu_c1_mxu.sv')
    with pytest.raises(RuntimeError, match='committed and clean'):
        b.prepare(tmp_path / 'uncreated', tmp_path, '12345678', {})
    assert not (tmp_path / 'uncreated').exists()


def test_memory_ip_jobs_forwarded(tmp_path, monkeypatch):
    calls = []
    out = tmp_path / 'run'
    monkeypatch.setattr(b, 'check_hdk', lambda _: {})
    monkeypatch.setattr(b, 'command', lambda *a, **k: 'vivado v2025.2' if a[-1] == '-version' else '12345678')
    def prepare(*a):
        for stage in b.STAGES:
            (out / stage).mkdir(parents=True)
        (out / 'inputs.json').write_text('{}')
        return {}
    monkeypatch.setattr(b, 'prepare', prepare)
    def stage(*args):
        calls.append(args)
        (out / args[1] / 'ip_done.txt').write_text('IP_DONE')
    monkeypatch.setattr(b, 'run_stage', stage)
    monkeypatch.setattr(b, 'ip_inventory', lambda _: {})
    monkeypatch.setattr(b, 'validate_memory_bd', lambda _: {})
    b.main(['--hdk', str(tmp_path / 'hdk'), '--out', str(out), '--stage', 'ip', '--ip-jobs', '2'])
    assert calls[0][1] == 'ip' and calls[0][2][-1] == 2


def test_prepare_freezes_core_vendor_adapters_and_audit(tmp_path, monkeypatch):
    hdk = tmp_path / 'aws-fpga'
    iface = hdk / 'hdk/common/shell_stable/design/interfaces'
    iface.mkdir(parents=True)
    iface.joinpath('cl_ports.vh').write_text('\n'.join(
        f'output logic [31:0] {name},' for name in ('cl_sh_id0', 'cl_sh_id1', 'cl_sh_status0',
                                                  'cl_sh_status1', 'cl_sh_status2', 'cl_sh_status_vled')))
    for name in ('unused_ddr_template.inc', 'unused_cl_sda_template.inc',
                 'unused_flr_template.inc', 'unused_apppf_irq_template.inc'):
        (iface / name).write_text('// fixture header only\n')
    installed = b.inventory([iface], hdk)
    monkeypatch.setattr(b, 'require_clean_source', lambda: '1' * 40)
    monkeypatch.setattr(b, 'hdk_inventory', lambda _: installed)
    out = tmp_path / 'prepared'
    record = b.prepare(out, hdk, '12345678', installed, 2)
    assert record['source_revision'] == '1' * 40 and record['source_status'] == ''
    assert record['qualification'] == 'not-qualified'
    for name in b.CORE_SOURCES:
        assert b.sha(out / 'source/rtl' / name) == b.sha(ROOT / 'rtl' / name)
    for name in ('otpu_native_axi.sv', 'otpu_hbm_geometry.sv'):
        assert b.sha(out / 'source/rtl/boards/f2' / name) == b.sha(b.SUPPORT / 'rtl/boards/f2' / name)
    top = (out / 'synth/rtl/cl_opentpu.sv').read_text()
    assert "BUILD_ID(32'h12345678)" in top
    assert 'otpu_c1_hbm_kernel' in top
    assert '.NUM_OF_AXI_PORTS(8)' in top
    assert b.sha(out / 'synth/audit_c1_alex.tcl') == b.sha(b.HERE / 'audit_c1_alex.tcl')
    script = (out / 'synth/build.tcl').read_text()
    assert script.index('::c1_alex::audit_synthesis $run') < script.index('write_checkpoint -force $run/post_synth.dcp')
    assert 'source/rtl/c1/otpu_c1_hbm_kernel.sv' in record['prepared_files']
    assert 'synth/audit_c1_alex.tcl' in record['prepared_files']


def test_run_stage_refuses_changed_prepared_script(tmp_path, monkeypatch):
    (tmp_path / 'ip').mkdir()
    script = tmp_path / 'ip/build.tcl'; script.write_text('# clean')
    b.write_json(tmp_path / 'inputs.json', {'prepared_files': b.inventory([script], tmp_path)})
    script.write_text('# changed')
    monkeypatch.setattr(b.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='Prepared build input changed'):
        b.run_stage(tmp_path, 'ip', [], 'vivado', tmp_path, {})


@pytest.mark.parametrize('flag', ['true', 'True', 'FALSE', 'false', '', True, False, 1, None])
def test_bd_rejects_noncanonical_exclusion_flags(tmp_path, flag):
    data = generated_bd()
    data['design']['addressing']['/']['address_spaces']['S_N0']['segments']['SEG_M_H1_Reg']['is_excluded'] = flag
    path = tmp_path / 'f2_mem.bd'; b.write_json(path, data)
    with pytest.raises(RuntimeError, match='noncanonical exclusion flag'):
        b.validate_memory_bd(path)


@pytest.mark.parametrize('change', ['missing_exclusion', 'duplicate_destination', 'unknown_destination',
                                   'own_excluded', 'host_excluded', 'enabled_crosswindow', 'excluded_offset', 'excluded_range'])
def test_bd_rejects_excluded_decode_mutations(tmp_path, change):
    data = generated_bd()
    spaces = data['design']['addressing']['/']['address_spaces']
    segs = spaces['S_N0']['segments']
    if change == 'missing_exclusion':
        segs.pop('SEG_M_H1_Reg')
    elif change == 'duplicate_destination':
        segs['SEG_M_H1_Reg']['address_block'] = '/M_H2/Reg'
    elif change == 'unknown_destination':
        segs['SEG_M_H1_Reg']['address_block'] = '/M_H8/Reg'
    elif change == 'own_excluded':
        segs['SEG_M_H0_Reg'] = {'address_block': '/M_H0/Reg', 'offset': '0', 'range': '1', 'is_excluded': 'TRUE'}
    elif change == 'host_excluded':
        spaces['S_HOST']['segments']['SEG_M_H1_Reg'] = {'address_block': '/M_H1/Reg', 'offset': '0', 'range': '1', 'is_excluded': 'TRUE'}
    elif change == 'enabled_crosswindow':
        segs['SEG_M_H1_Reg'] = {'address_block': '/M_H1/Reg', 'offset': '0x20000000', 'range': '512M'}
    elif change == 'excluded_offset':
        segs['SEG_M_H1_Reg']['offset'] = '0x20000000'
    else:
        segs['SEG_M_H1_Reg']['range'] = '512M'
    path = tmp_path / 'f2_mem.bd'; b.write_json(path, data)
    with pytest.raises(RuntimeError, match='SmartConnect mismatch'):
        b.validate_memory_bd(path)


def test_bd_rejects_duplicate_json_members(tmp_path):
    text = json.dumps(generated_bd()).replace('"S_N0":', '"S_N0": {}, "S_N0":', 1)
    path = tmp_path / 'f2_mem.bd'; path.write_text(text)
    with pytest.raises(RuntimeError, match='duplicate JSON key S_N0'):
        b.validate_memory_bd(path)


def test_bd_actual_vivado_exclusion_schema(tmp_path):
    # Literal observed schema from read-bd-aa7a23ed-001 (2026-10-19), not a
    # claimed fresh hardware result. All remaining interfaces use synthetic data.
    data = generated_bd()
    spaces = data['design']['addressing']['/']['address_spaces']
    for p in range(8):
        segs = spaces[f'S_N{p}']['segments']
        for q in range(8):
            segs[f'SEG_M_H{q}_Reg'] = ({'address_block': f'/M_H{q}/Reg',
                'offset': f'0x{p * 2**29:09X}', 'range': '512M'} if p == q else
                {'address_block': f'/M_H{q}/Reg', 'offset': '0x000000000', 'range': '1', 'is_excluded': 'TRUE'})
    path = tmp_path / 'f2_mem.bd'; b.write_json(path, data)
    assert b.validate_memory_bd(path)['master_interfaces'] == 8
