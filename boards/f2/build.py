#!/usr/bin/env python3
"""Freeze the F2 CL and run the pinned AWS HDK synthesis/implementation flow.

--target f2 (the default) builds the DDR4 CL (one slice on the slot's DIMM, sh_ddr); a target
whose platform mem is "hbm" (targets/f2-hbm*.toml) builds the HBM CL: S slices of NCH channels,
each channel on its own HBM pseudo-channel (cl_opentpu_hbm.sv as cl_opentpu, otpu_f2_hbm_sys, a
SmartConnect fabric generated first by build/scripts/create_hbm_sc.tcl, the HDK's
cl_mem_hbm_wrapper and cl_hbm IP; docs/f2_hbm.md)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opentpu import target  # noqa: E402


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_vivado_version(output: str, expected: str) -> None:
    # Linux's launcher prints lowercase "vivado"; other releases/platforms
    # capitalize it. Match the tool banner, not an unrelated build timestamp.
    if not re.search(rf'\bvivado\s+v?{re.escape(expected)}\b', output, re.IGNORECASE):
        raise RuntimeError(f'This target requires Vivado {expected}')


HBM_WRAPPER = 'hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv'
HBM_BD_SRC = 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/bd'
HBM_CHAIN_IP = ('hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/axi_register_slice/axi_register_slice.xci',
                'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_axi3_256b_reg_slice/cl_axi3_256b_reg_slice.xci',
                HBM_BD_SRC + '/cl_axi_sc_1x1/cl_axi_sc_1x1.bd',
                'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/bd/cl_axi_sc_1x1/hdl/cl_axi_sc_1x1_wrapper.v')

# The HBM CL's core clocks (cl_opentpu_hbm.sv u_mmcm, from clk_main_a0 = 250 MHz): core kHz ->
# (DIVCLK_DIVIDE, CLKFBOUT_MULT_F, CLKOUT0_DIVIDE_F = the core clock, CLKOUT1_DIVIDE = the pumped
# TMEM's, 2x the core); every VCO 800-1600 MHz, multiples of 0.125, integer CLKOUT1
HBM_CORE_MMCM = {200000: (1, 4.8, 6.0, 3),      # VCO 1200: 200 / 400
                 175000: (5, 21.0, 6.0, 3),     # PFD 50, VCO 1050: 175 / 350
                 150000: (1, 4.8, 8.0, 4)}      # VCO 1200: 150 / 300


def is_hbm(t: target.Target) -> bool:
    return t.platform.get('mem') == 'hbm'


def is_engine(t: target.Target) -> bool:
    """v5's memory path (platform mem_path "engine": a 450 MHz engine per HBM port, the host's
    cl_axi_sc_1x1 kept; docs/full_hbm.md): the hbm5 top, synthesis script and floorplan, no
    channel SmartConnect block design to generate."""
    return is_hbm(t) and t.platform.get('mem_path') == 'engine'


def check_hbm_ip(hdk: Path) -> None:
    """The HDK's cl_hbm IP as the HBM CL assumes it: both stacks, 450 MHz AXI ports, the
    switch on (every port reaches all 16 GiB; the CL keeps each channel on its own port's
    pseudo-channel, and the host on the port that owns its address), generated."""
    ip = hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci'
    if not ip.is_file():
        raise RuntimeError(f'Missing HBM IP: {ip}; source hdk_setup.sh')
    p = json.loads(ip.read_text())['ip_inst']['parameters']['component_parameters']
    # the global address switch on: the host's own port (NPORT) reaches every channel's
    # pseudo-channel; the monitor APB on: the shell's MON_APB ports are connected (the AWS Shell
    # Interface Specification: an unconnected monitor interface fails AFI creation)
    want = {'USER_HBM_STACK': '2', 'USER_AXI_CLK_FREQ': '450', 'USER_HBM_DENSITY': '16GB',
            'USER_SWITCH_ENABLE_00': 'TRUE', 'USER_SWITCH_ENABLE_01': 'TRUE',
            'USER_MON_APB_EN': 'TRUE'}
    for k, v in want.items():
        if str(p.get(k, [{}])[0].get('value')).upper() != v:   # (the HDK writes 'true' / 'TRUE')
            near = sorted(n for n in p if 'APB' in n or 'SWITCH' in n)
            raise RuntimeError(f'{ip}: {k} is not {v} (the IP\'s APB / switch parameters: {near})')
    gen = hdk / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/ip/cl_hbm/synth/cl_hbm.v'
    if not gen.is_file():
        raise RuntimeError(f'Missing HBM generated product: {gen}')
    if not (hdk / HBM_WRAPPER).is_file():
        raise RuntimeError(f'Missing HDK HBM wrapper: {hdk / HBM_WRAPPER}')
    # the memory path's prebuilt IP (AWS's cl_dram_hbm_dma chain: docs/f2_hbm.md)
    for rel in HBM_CHAIN_IP:
        if not (hdk / rel).is_file():
            raise RuntimeError(f'Missing HDK IP for the HBM chain: {hdk / rel}')


def check_hdk(hdk: Path, t: target.Target, *, shell: bool) -> Path:
    for path, wanted in [(hdk, t.platform['hdk_commit']),
                         (hdk / 'hdk/common/ip', t.platform['ip_commit'])]:
        actual = command('git', '-C', str(path), 'rev-parse', 'HEAD')
        if actual != wanted:
            raise RuntimeError(f'{path}: revision {actual}, expected {wanted}')
    sh = hdk / 'hdk/common/shell_stable'
    for rel in ['design/interfaces/cl_ports.vh', 'build/scripts/build_all.tcl']:
        if not (sh / rel).is_file():
            raise RuntimeError(f'Missing HDK input: {sh / rel}; source hdk_setup.sh')
    versions = dict(line.strip().split('=', 1) for line in
                    (sh / 'shell_version.txt').read_text().splitlines() if '=' in line)
    if versions.get('small_shell') != t.platform['shell_version']:
        raise RuntimeError(f'{sh}: small-shell version does not match the target')
    ip = hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_ddr4/cl_ddr4.xci'
    if not ip.is_file():
        raise RuntimeError(f'Missing DDR IP: {ip}; source hdk_setup.sh')
    ip_params = json.loads(ip.read_text())['ip_inst']['parameters']['component_parameters']
    if ip_params['C0.DDR4_TimePeriod'][0]['value'] != '938':
        raise RuntimeError(f'{ip}: unexpected DDR clock period (expected 938 ps)')
    generated = hdk / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/ip/cl_ddr4'
    for rel in ('cl_ddr4.xml', 'rtl/ip_top/cl_ddr4.sv',
                'sw/calibration_0/Debug/calibration_ddr.elf'):
        path = generated / rel
        if not path.is_file():
            raise RuntimeError(f'Missing DDR generated product: {path}')
        with path.open('rb') as f:
            header = f.read(128)
        if header.startswith(b'version https://git-lfs.github.com/spec/'):
            raise RuntimeError(f'{path}: Git LFS pointer; run git lfs pull in hdk/common/ip')
        if path.suffix == '.elf' and not header.startswith(b'\x7fELF'):
            raise RuntimeError(f'{path}: not an ELF calibration image')
    if is_hbm(t):
        check_hbm_ip(hdk)
    if shell:
        dcp = sh / 'build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp'
        expected = '04a5331321e86cffc7ef740fb6132b29f179e68e2d58ccae5031c989403ecca6'
        if not dcp.is_file() or sha(dcp) != expected:
            raise RuntimeError(f'{dcp}: missing or wrong small-shell checkpoint')
    return sh


def prepare(out: Path, sh: Path, t: target.Target, build_id: str,
            wrapper: Path | None = None) -> Path:
    """The CL directory of `t`'s build. An HBM target takes the HDK's cl_mem_hbm_wrapper
    (`wrapper`; default the HDK's beside `sh`)."""
    if out.exists():
        raise RuntimeError(f'{out} already exists; choose a new run directory')
    hbm, engine = is_hbm(t), is_engine(t)
    p = t.platform
    host_mux = False
    if hbm:
        nport = t.arch['S'] * p['channels_per_slice']
        host_mux = engine and nport == 32       # (the host through port 31's mux: docs/full_hbm.md)
        if (t.core_khz not in HBM_CORE_MMCM or p['channel_bytes'] != 2**29 or
                nport > (32 if engine else 31) or
                p['channels_per_slice'] & (p['channels_per_slice'] - 1) or t.uarch['TMEM_PMP'] != 1):
            raise RuntimeError('F2 HBM CL requires a 200, 175 or 150 MHz core (the pumped TMEM at '
                               '2x), 512 MiB channels, a power-of-two NCH and S x NCH <= 31 ports (the host '
                               'has its own HBM port; the engine path takes 32 with the host muxed)')
    elif t.core_khz != 100000 or t.arch['S'] != 1 or p['channel_bytes'] != 2**31:
        raise RuntimeError('F2 wrapper requires 100 MHz, one slice, 2 GiB per channel')
    cl = out / 'cl_opentpu'
    shutil.copytree(ROOT / 'boards/f2/design', cl / 'design')
    shutil.copytree(ROOT / 'boards/f2/build', cl / 'build')
    scripts, cons = cl / 'build/scripts', cl / 'build/constraints'
    if hbm:
        # the HBM CL's top, synthesis script and CDC constraints in the DDR ones' places (the
        # engine path: its own top and synthesis script, the SmartConnect path's left out)
        (cl / 'design/cl_opentpu.sv').unlink()
        if engine:
            (cl / 'design/cl_opentpu_hbm.sv').unlink()
            (scripts / 'synth_cl_opentpu_hbm.tcl').unlink()
            (scripts / 'synth_cl_opentpu_hbm5.tcl').replace(scripts / 'synth_cl_opentpu.tcl')
            (cons / 'cl_pnr_user_hbm.tcl').unlink()
            (cons / 'cl_timing_user_hbm.tcl').unlink()
            (cons / 'cl_timing_user_hbm5.tcl').replace(cons / 'cl_timing_user.tcl')
            (scripts / 'create_hbm_sc.tcl').unlink()
            (scripts / 'hbm_sc_inst.tcl').unlink()
        else:
            (cl / 'design/cl_opentpu_hbm5.sv').unlink()
            (scripts / 'synth_cl_opentpu_hbm5.tcl').unlink()
            (scripts / 'synth_cl_opentpu_hbm.tcl').replace(scripts / 'synth_cl_opentpu.tcl')
            (cons / 'cl_pnr_user_hbm5.tcl').unlink()
            (cons / 'cl_timing_user_hbm5.tcl').unlink()
            (cons / 'cl_timing_user_hbm.tcl').replace(cons / 'cl_timing_user.tcl')
        # the linked-reset exception covers the HBM clock's synchronizer too, and the HBM
        # build's floorplan (its pblocks under the HDK's pblock_CL) follows it in the one file
        # build_level_1_cl.tcl reads unmanaged at placement (docs/wide_port.md "Placement")
        pnr = cons / 'cl_pnr_user.tcl'
        old = 'foreach stem {main_reset_n bridge_reset_n core_reset_n}'
        if old not in pnr.read_text():
            raise RuntimeError(f'{pnr}: the reset synchronizer list moved')
        floorplan = cons / ('cl_pnr_user_hbm5.tcl' if engine else 'cl_pnr_user_hbm.tcl')
        pnr.write_text(pnr.read_text().replace(old, old[:-1] + ' hbm_reset_n}') + '\n' +
                       floorplan.read_text())
        floorplan.unlink()
        w = wrapper or sh.parents[2] / HBM_WRAPPER
        if not w.is_file():
            raise RuntimeError(f'Missing HDK HBM wrapper: {w}')
        shutil.copy2(w, cl / 'design/cl_mem_hbm_wrapper.sv')
        div, mult, d0, d1 = HBM_CORE_MMCM[t.core_khz]
        (cl / 'design/opentpu_geometry.vh').write_text(
            f"localparam int S = {t.arch['S']};\nlocalparam int NCH = {p['channels_per_slice']};\n"
            f"localparam int CORE_KHZ_CL = {t.core_khz};\n"
            f"localparam int MMCM_DIVCLK = {div};\nlocalparam real MMCM_MULT = {mult};\n"
            f"localparam real MMCM_DIV0 = {d0};\nlocalparam int MMCM_DIV1 = {d1};\n")
    else:
        (cl / 'design/cl_opentpu_hbm.sv').unlink()
        (cl / 'design/cl_opentpu_hbm5.sv').unlink()
        (scripts / 'synth_cl_opentpu_hbm.tcl').unlink()
        (scripts / 'synth_cl_opentpu_hbm5.tcl').unlink()
        (cons / 'cl_timing_user_hbm.tcl').unlink()
        (cons / 'cl_timing_user_hbm5.tcl').unlink()
        (cons / 'cl_pnr_user_hbm.tcl').unlink()
        (cons / 'cl_pnr_user_hbm5.tcl').unlink()
        (scripts / 'create_hbm_sc.tcl').unlink()
        (scripts / 'hbm_sc_inst.tcl').unlink()
    files = target.sources(t)
    for name in files:
        dest = cl / 'design/rtl' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'rtl' / name, dest)
    params = target.rtl_params(t)
    if hbm:                             # otpu_f2_hbm_sys takes the geometry; no DIMM
        params.pop('DDR_MTS')
        params.update(CORE_KHZ=t.core_khz, BUILD_ID=int(build_id, 16))
        sys_rtl = ROOT / 'rtl/boards/f2/otpu_f2_hbm_sys.sv'
        if engine:                      # v5's sys top: the host's mux at 32 ports
            params.update(HOST_MUX=int(host_mux))
            sys_rtl = ROOT / 'rtl/boards/f2/otpu_f2_hbmeng_sys.sv'
    else:
        for name in ('S', 'NCH', 'CH_LOG2', 'DRAM_LOG2'):
            params.pop(name)
        params.update(CORE_KHZ=t.core_khz, BUILD_ID=int(build_id, 16),
                      CH_BYTES=t.platform['channel_bytes'])
        sys_rtl = ROOT / 'rtl/boards/f2/otpu_f2_sys.sv'
    actual_params = set(re.findall(r'parameter\s+(?:int|bit|logic(?:\s*\[[^\]]+\])?)\s+(\w+)',
                                  sys_rtl.read_text()))
    if set(params) - actual_params:
        raise RuntimeError(f'Unknown RTL parameters: {set(params) - actual_params}')
    bit_params = set(re.findall(r'parameter\s+bit\s+(\w+)', sys_rtl.read_text()))
    (cl / 'design/opentpu_params.vh').write_text(',\n'.join(
        f".{k}({1 if k in bit_params else 64 if k == 'CH_BYTES' else 32}'h{v:x})"
        for k, v in params.items()) + '\n')
    (cl / 'design/sources.tcl').write_text('set otpu_sources {\n' +
        ''.join(f'  rtl/{f}\n' for f in files) + '}\n')
    shutil.copy2(ROOT / 'boards/ypcb-00338/constraints/otpu_clk2x.tcl',
                 cl / 'build/constraints/otpu_clk2x.tcl')
    for name in ('build_all.tcl',):
        (cl / 'build/scripts' / name).symlink_to(sh / 'build/scripts' / name)
    provenance = t.to_dict() | {'build_id': build_id,
        'git_commit': command('git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
        'git_status': command('git', '-C', str(ROOT), 'status', '--porcelain'),
        'rtl_parameters': params,
        'sources': {str(p.relative_to(cl)): sha(p) for p in sorted(cl.rglob('*')) if p.is_file()}}
    (out / 'inputs.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return cl


def hbm_sc_command(cl: Path, t: target.Target, log: Path, hdk: Path | None = None) -> list[str]:
    """The SmartConnect fabric's generation (an HBM target, before synthesis): Vivado batch on
    create_hbm_sc.tcl into cl/ip (the block design, its OOC checkpoints, its wrapper and the
    CL's instance of it, otpu_hbm_sc_inst.vh; docs/f2_hbm.md "The memory path")."""
    nport = t.arch['S'] * t.platform['channels_per_slice']
    return ['vivado', '-mode', 'batch', '-source', str(cl / 'build/scripts/create_hbm_sc.tcl'),
            '-log', str(log), '-journal', str(log.with_suffix('.jou')),
            '-tclargs', str(cl / 'ip'), str(nport), t.tech['part'], str(t.core_khz * 1000),
            str((hdk or Path('/HDK')) / HBM_BD_SRC)]


def load_synthesis(run: Path, expected_sha: str) -> tuple[dict, Path, str]:
    """Accept a completed normal synthesis, never an unvalidated recovery DCP."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--synth-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid synthesis build ID')
    log = (run / 'console.log').read_text()
    if (run / 'exit_code').read_text().strip() != '0' or not re.search(
            r'^OTPU_STAGE_SYNTHESIS_PASSED$', log, re.M):
        raise RuntimeError('Parent synthesis did not complete successfully')
    sentinel = re.search(r"^OTPU_DDR_CALIBRATION_INIT_2C=256'h([0-9a-fA-F]{64})$", log, re.M)
    if info.get('platform', {}).get('mem') == 'hbm':
        pass                            # no DIMM: no calibration firmware to verify
    elif not sentinel or not int(sentinel[1], 16):
        raise RuntimeError('Parent synthesis has no verified DDR calibration sentinel')
    dcp = run / f'cl_opentpu/build/checkpoints/cl_opentpu.{build_id}.post_synth.dcp'
    if sha(dcp) != expected_sha.lower():
        raise RuntimeError('Parent synthesis checkpoint SHA256 mismatch')
    return info, dcp, expected_sha.lower()


def inherit_synthesis(out: Path, cl: Path, run: Path, info: dict, dcp: Path, expected_sha: str) -> None:
    """Require identical HDL/parameters/CDC constraints; permit new implementation scripts."""
    current = json.loads((out / 'inputs.json').read_text())
    if current['rtl_parameters'] != info['rtl_parameters']:
        raise RuntimeError('RTL parameters differ from parent synthesis')
    for name in ('platform', 'tech'):
        if current.get(name) != info.get(name):
            raise RuntimeError(f'{name} differs from parent synthesis')
    required = {n for n in current['sources'] if n.startswith('design/')} | {
        'build/constraints/cl_synth_user.xdc',
        'build/constraints/cl_timing_user.tcl',
        'build/constraints/otpu_clk2x.tcl',
    }
    parent_design = {n for n in info['sources'] if n.startswith('design/')}
    if parent_design != {n for n in required if n.startswith('design/')}:
        raise RuntimeError('RTL source inventory differs from parent synthesis')
    for name in sorted(required):
        if current['sources'].get(name) != info['sources'].get(name):
            raise RuntimeError(f'{name} differs from parent synthesis')
        if sha(run / 'cl_opentpu' / name) != info['sources'][name]:
            raise RuntimeError(f'Parent source was modified after synthesis: {name}')
    dest = cl / 'build/checkpoints' / dcp.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dcp, dest)
    digest = expected_sha
    if sha(dcp) != digest or sha(dest) != digest:
        raise RuntimeError('Copied synthesis checkpoint SHA256 mismatch')
    current['parent_synthesis'] = {'run': str(run), 'inputs_sha256': sha(run / 'inputs.json'),
        'checkpoint': str(dcp), 'checkpoint_sha256': digest}
    (out / 'inputs.json').write_text(json.dumps(current, indent=2) + '\n')


INCR_DIRECTIVES = ('TimingClosure', 'RuntimeOptimized', 'Quick')


def load_incremental_reference(run: Path, expected_sha: str, t: target.Target) -> tuple[dict, Path]:
    """A finished implementation run's routed checkpoint as the reference of an incremental
    implementation (read_checkpoint -incremental before place_design): met or VIOLATED (a
    reference is never packaged), exactly one of them, pinned by digest, built for the same
    platform and device as `t`."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--incremental-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid reference build ID')
    exit_code = run / 'exit_code'
    if not exit_code.is_file() or not re.fullmatch(r'[0-9]+', exit_code.read_text().strip()):
        raise RuntimeError('Incremental reference run has no completed exit status')
    for key in ('platform', 'tech'):
        if info.get(key) != t.to_dict().get(key):
            raise RuntimeError(f'Incremental reference {key} differs from target {t.name}')
    checkpoints = run / 'cl_opentpu/build/checkpoints'
    routed = [p for p in (checkpoints / f'cl_opentpu.{build_id}.post_route.dcp',
                          checkpoints / f'cl_opentpu.{build_id}.post_route.VIOLATED.dcp')
              if p.is_file()]
    if len(routed) != 1:
        raise RuntimeError(f'Incremental reference run needs exactly one routed checkpoint, has {len(routed)}')
    if sha(routed[0]) != expected_sha.lower():
        raise RuntimeError('Incremental reference checkpoint SHA256 mismatch')
    return info, routed[0]


def attach_incremental(out: Path, run: Path, info: dict, dcp: Path, expected_sha: str,
                       directive: str, force: bool) -> dict:
    """Copy the pinned reference into the run (its provenance in inputs.json) and return the
    environment build_level_1_cl.tcl reads (OTPU_INCR_*)."""
    if directive not in INCR_DIRECTIVES:
        raise RuntimeError(f'Unknown incremental directive {directive}')
    dest = out / 'incremental_reference' / dcp.name
    dest.parent.mkdir()
    shutil.copy2(dcp, dest)
    digest = expected_sha.lower()
    if sha(dcp) != digest or sha(dest) != digest:
        raise RuntimeError('Copied incremental reference SHA256 mismatch')
    current = json.loads((out / 'inputs.json').read_text())
    current['incremental_reference'] = {'run': str(run), 'build_id': info['build_id'],
        'inputs_sha256': sha(run / 'inputs.json'), 'checkpoint': str(dcp),
        'checkpoint_sha256': digest, 'timing_met': not dcp.name.endswith('.VIOLATED.dcp'),
        'directive': directive, 'force_incr': force, 'copy': str(dest)}
    (out / 'inputs.json').write_text(json.dumps(current, indent=2) + '\n')
    return {'OTPU_INCR_DCP': str(dest), 'OTPU_INCR_SHA256': digest,
            'OTPU_INCR_DIRECTIVE': directive, 'OTPU_INCR_FORCE': '1' if force else '0'}


def attach_qor_suggestions(out: Path, rqs: Path, expected_sha: str) -> dict:
    """Copy a pinned .rqs file (write_qor_suggestions of an earlier routed run) into the run,
    record it in inputs.json and return the environment build_level_1_cl.tcl reads (OTPU_RQS*)."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--qor-suggestions-sha256 must be a SHA256 digest')
    if rqs.suffix != '.rqs' or not rqs.is_file():
        raise RuntimeError(f'{rqs}: not a QoR suggestions (.rqs) file')
    digest = expected_sha.lower()
    if sha(rqs) != digest:
        raise RuntimeError('QoR suggestions file SHA256 mismatch')
    dest = out / 'qor_suggestions' / rqs.name
    dest.parent.mkdir()
    shutil.copy2(rqs, dest)
    if sha(dest) != digest:
        raise RuntimeError('Copied QoR suggestions SHA256 mismatch')
    current = json.loads((out / 'inputs.json').read_text())
    current['qor_suggestions'] = {'file': str(rqs), 'sha256': digest, 'copy': str(dest)}
    (out / 'inputs.json').write_text(json.dumps(current, indent=2) + '\n')
    return {'OTPU_RQS': str(dest), 'OTPU_RQS_SHA256': digest}


def load_routed_run(run: Path, expected_sha: str) -> tuple[dict, Path]:
    """Repackage only a completed, qualified run with an explicitly pinned DCP."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--route-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid routed build ID')
    if ((run / 'exit_code').read_text().strip() != '0' or not re.search(
            r'^OTPU_STAGE_ROUTE_VALIDATION_PASSED$', (run / 'console.log').read_text(), re.M)):
        raise RuntimeError('Routed run did not complete qualification successfully')
    cl = run / 'cl_opentpu'
    if sha(cl / f'build/checkpoints/cl_opentpu.{build_id}.post_route.dcp') != expected_sha.lower():
        raise RuntimeError('Routed checkpoint SHA256 mismatch')
    for name, digest in info['sources'].items():
        if sha(cl / name) != digest:
            raise RuntimeError(f'Routed source snapshot was modified: {name}')
    return info, cl


def package_afi(cl: Path, sh: Path, hdk: Path, build_id: str, env: dict) -> Path:
    """Use the pinned HDK packager, then verify its actual archive and manifest."""
    checkpoints = cl / 'build/checkpoints'
    routed = checkpoints / f'cl_opentpu.{build_id}.post_route.dcp'
    if (checkpoints / f'cl_opentpu.{build_id}.post_route.VIOLATED.dcp').exists():
        raise RuntimeError('Refusing to package a timing-violated checkpoint')
    marker = checkpoints / f'cl_opentpu.{build_id}.route_validation.passed'
    if not marker.is_file() or marker.read_text().strip() != 'OTPU_STAGE_ROUTE_VALIDATION_PASSED':
        raise RuntimeError('Post-route validation has not passed')
    # The official function uses os.system; paths containing shell metacharacters
    # are unsafe for that implementation, even though the main flow uses argv.
    for value in (str(cl), str(sh), str(hdk)):
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
            raise RuntimeError('HDK packaging requires paths without spaces/metacharacters')
    # The official packager uses its tag as manifest.date. Keep this timestamp
    # separate from the immutable eight-digit identity compiled into the RTL.
    tag = datetime.now(timezone.utc).strftime('%Y_%m_%d-%H%M%S')
    archive = checkpoints / f'{tag}.Developer_CL.tar'
    if archive.exists():
        raise RuntimeError(f'{archive} already exists; retry packaging in a later second')
    module = sh / 'build/scripts/aws_build_dcp_from_cl.py'
    dcp_sha = sha(routed)
    ids_header = cl / 'design/cl_id_defines.vh'
    ids = dict(re.findall(r"`define\s+(CL_SH_ID[01])\s+32'h([0-9a-fA-F_]+)", ids_header.read_text()))
    if set(ids) != {'CL_SH_ID0', 'CL_SH_ID1'}:
        raise RuntimeError('Missing frozen PCI identity definitions')
    id0, id1 = (int(ids[name].replace('_', ''), 16) for name in ('CL_SH_ID0', 'CL_SH_ID1'))
    versions = dict(line.split('=', 1) for line in (hdk / 'release_version.txt').read_text().splitlines()
                    if '=' in line)
    expected = {'manifest_format_version': '2', 'date': tag,
        'pci_vendor_id': f'0x{id0 & 0xffff:04x}', 'pci_device_id': f'0x{id0 >> 16:04x}',
        'pci_subsystem_vendor_id': f'0x{id1 & 0xffff:04x}', 'pci_subsystem_id': f'0x{id1 >> 16:04x}',
        'shell_version': '0x10212415', 'hdk_version': versions['RELEASE_VERSION'].strip(),
        'dcp_file_name': f'{tag}.SH_CL_routed.dcp', 'dcp_hash': dcp_sha,
        'clock_recipe_a': 'A1', 'clock_recipe_b': 'B2', 'clock_recipe_c': 'C0', 'clock_recipe_hbm': 'H2'}
    with tempfile.TemporaryDirectory(prefix='.afi-package-', dir=checkpoints) as temp:
        stage = Path(temp)
        (stage / 'design').mkdir()
        shutil.copy2(ids_header, stage / 'design/cl_id_defines.vh')
        staged_checkpoints = stage / 'build/checkpoints'
        staged_checkpoints.mkdir(parents=True)
        os.link(routed, staged_checkpoints / f'cl_opentpu.{tag}.post_route.dcp')
        probes = checkpoints / f'{build_id}.debug_probes.ltx'
        if probes.is_file():
            os.link(probes, staged_checkpoints / f'{tag}.debug_probes.ltx')
        code = ('import importlib.util,datetime; '
                's=importlib.util.spec_from_file_location("aws_package",'+repr(str(module))+'); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'tag='+repr(tag)+'; '
                'assert datetime.datetime.strptime(tag,m.TIMESTAMP_FILE_FORMAT).strftime(m.TIMESTAMP_FILE_FORMAT)==tag; '
                'm.generate_dcp_tarball("cl_opentpu",tag,"A1","B2","C0","H2")')
        subprocess.run([sys.executable, '-c', code], cwd=staged_checkpoints,
                       env=env | {'AWS_FPGA_REPO_DIR': str(hdk), 'CL_DIR': str(stage)}, check=True)
        staged_archive = staged_checkpoints / archive.name
        with tarfile.open(staged_archive) as tar:
            members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}
            manifest_member = members[f'to_aws/{tag}.manifest.txt']
            manifest = dict(line.split('=', 1) for line in
                            tar.extractfile(manifest_member).read().decode().splitlines() if '=' in line)
            for key, value in expected.items():
                if manifest.get(key, '').lower() != value.lower():
                    raise RuntimeError(f'AFI manifest {key} mismatch: {manifest.get(key)!r}, expected {value!r}')
            if manifest.get('tool_version', '').lower().removeprefix('v') != env['VIVADO_TOOL_VERSION']:
                raise RuntimeError('AFI manifest tool_version mismatch')
            with tar.extractfile(members['to_aws/' + manifest['dcp_file_name']]) as stream:
                embedded_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if embedded_sha != dcp_sha or sha(routed) != dcp_sha:
                raise RuntimeError('AFI archive checkpoint/hash mismatch')
        with staged_archive.open('rb') as source, archive.open('xb') as dest:
            shutil.copyfileobj(source, dest)
    provenance = {'build_id': build_id, 'package_tag': tag, 'verified_manifest': manifest,
        'source_run': str(cl.parent), 'inputs_sha256': sha(cl.parent / 'inputs.json'),
        'route_validation_marker_sha256': sha(marker),
        'routed_checkpoint': str(routed), 'routed_checkpoint_sha256': dcp_sha,
        'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'official_packager_sha256': sha(module), 'build_script_sha256': sha(Path(__file__))}
    (checkpoints / f'{tag}.packaging.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return archive


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdk', type=Path, required=True)
    p.add_argument('--target', default='f2',
                   help='f2 (DDR4), or an HBM target: f2-hbm[:s2|:s4], f2-hbm16 (docs/f2_hbm.md)')
    p.add_argument('--out', type=Path, required=True, help='fresh run directory; existing run for --package-only')
    p.add_argument('--flow', choices=['SynthCL', 'ImplCL', 'BuildAll'], default='SynthCL')
    p.add_argument('--from-synth', type=Path, help='successful SynthCL run directory for ImplCL')
    p.add_argument('--synth-sha256', help='expected parent post_synth.dcp digest')
    p.add_argument('--package', action='store_true', help='package validated routed DCP for AFI submission')
    p.add_argument('--package-only', action='store_true', help='package a completed run without running Vivado')
    p.add_argument('--route-sha256', help='expected routed checkpoint digest; required with --package-only')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--build-id', default=None, help='eight hexadecimal digits; defaults to HEAD')
    p.add_argument('--place-directive', default='SSI_SpreadLogic_high',
        help="place_design's directive for build_all.tcl's first tclarg (SSI_SpreadLogic_high: the "
             "flow's since its first run; SSI_SpreadLogic_low keeps the slice together: docs/wide_port.md Placement)")
    p.add_argument('--incremental-from', type=Path,
        help='a finished ImplCL/BuildAll run whose routed checkpoint (met or VIOLATED) seeds an '
             'incremental place and route (read_checkpoint -incremental before place_design; '
             'docs/wide_port.md "Incremental implementation")')
    p.add_argument('--incremental-sha256', help="the reference routed checkpoint's digest")
    p.add_argument('--incremental-directive', choices=INCR_DIRECTIVES, default='TimingClosure',
        help='read_checkpoint -incremental -directive (TimingClosure: target WNS 0, the failing '
             "paths ripped up; RuntimeOptimized: keep the reference's WNS; Quick: not timing-driven)")
    p.add_argument('--incremental-force', action='store_true',
        help="read_checkpoint -force_incr: incremental even where Vivado's reuse / reference "
             'quality criteria would fall back to the default algorithms')
    p.add_argument('--qor-suggestions', type=Path,
        help='an .rqs file (write_qor_suggestions of an earlier routed run of this netlist) read '
             'before opt_design; its automatic suggestions apply at opt_design / place_design')
    p.add_argument('--qor-suggestions-sha256', help="the .rqs file's digest")
    a = p.parse_args()
    t = target.load(a.target)
    if t.platform.get('board') != 'f2':
        p.error(f'{a.target} is not an F2 target')
    hdk, out = a.hdk.resolve(), a.out.resolve()
    if a.package_only:
        if (not a.route_sha256 or a.package or a.prepare_only or a.from_synth or
                a.synth_sha256 or a.build_id or a.flow != 'SynthCL' or a.incremental_from or
                a.incremental_sha256 or a.incremental_force or a.qor_suggestions):
            p.error('--package-only requires --route-sha256 and does not accept build options')
        sh = check_hdk(hdk, t, shell=True)
        check_vivado_version(command('vivado', '-version'), t.platform['vivado'])
        info, cl = load_routed_run(out, a.route_sha256)
        env = dict(os.environ, CL_DIR=str(cl), HDK_SHELL_DIR=str(sh), SHELL_MODE='small_shell',
                   VIVADO_TOOL_VERSION=t.platform['vivado'])
        archive = package_afi(cl, sh, hdk, info['build_id'], env)
        print(f'Packaged completed build {info["build_id"]}: {archive}')
        return
    if a.route_sha256:
        p.error('--route-sha256 is only valid with --package-only')
    if (a.flow == 'ImplCL') != bool(a.from_synth and a.synth_sha256):
        p.error('ImplCL requires --from-synth and --synth-sha256; other flows do not accept them')
    if a.flow != 'ImplCL' and (a.from_synth or a.synth_sha256):
        p.error('--from-synth and --synth-sha256 are only valid with ImplCL')
    if a.package and a.flow == 'SynthCL':
        p.error('--package requires ImplCL or BuildAll')
    if bool(a.incremental_from) != bool(a.incremental_sha256):
        p.error('--incremental-from and --incremental-sha256 go together')
    if a.incremental_from and a.flow == 'SynthCL':
        p.error('--incremental-from requires ImplCL or BuildAll')
    if a.incremental_force and not a.incremental_from:
        p.error('--incremental-force requires --incremental-from')
    if bool(a.qor_suggestions) != bool(a.qor_suggestions_sha256):
        p.error('--qor-suggestions and --qor-suggestions-sha256 go together')
    if a.qor_suggestions and a.flow == 'SynthCL':
        p.error('--qor-suggestions requires ImplCL or BuildAll')
    sh = check_hdk(hdk, t, shell=a.flow != 'SynthCL')
    parent = load_synthesis(a.from_synth.resolve(), a.synth_sha256) if a.from_synth else None
    reference = (load_incremental_reference(a.incremental_from.resolve(), a.incremental_sha256, t)
                 if a.incremental_from else None)
    build_id = (parent[0]['build_id'] if parent else a.build_id or
                command('git', '-C', str(ROOT), 'rev-parse', '--short=8', 'HEAD'))
    if parent and a.build_id and a.build_id != build_id:
        p.error('--build-id must match the inherited synthesis build ID')
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        p.error('--build-id must contain exactly eight hexadecimal digits')
    cl = prepare(out, sh, t, build_id, hdk / HBM_WRAPPER if is_hbm(t) else None)
    if parent:
        inherit_synthesis(out, cl, a.from_synth.resolve(), *parent)
    incremental = (attach_incremental(out, a.incremental_from.resolve(), *reference,
                                      a.incremental_sha256, a.incremental_directive,
                                      a.incremental_force) if reference else {})
    rqs = (attach_qor_suggestions(out, a.qor_suggestions.resolve(), a.qor_suggestions_sha256)
           if a.qor_suggestions else {})
    env = dict(os.environ, CL_DIR=str(cl), CL='cl_opentpu', SHELL_MODE='small_shell',
        BUILD_FLOW=a.flow, BUILD_TAG=build_id, ENCRYPT='0',
        HDK_SHELL_DIR=str(sh), HDK_SHELL_DESIGN_DIR=str(sh / 'design'),
        HDK_COMMON_DIR=str(hdk / 'hdk/common'),
        HDK_IP_SRC_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip'),
        HDK_BD_SRC_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/bd'),
        HDK_BD_GEN_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/bd'),
        VIVADO_TOOL_VERSION=t.platform['vivado'])
    env.pop('OTPU_INCR_DCP', None)      # never inherited from the caller's shell
    env.pop('OTPU_RQS', None)
    env.update(incremental | rqs)
    cmd = ['vivado', '-mode', 'batch', '-source', 'build_all.tcl', '-log', str(out / 'vivado.log'),
        '-journal', str(out / 'vivado.jou'), '-tclargs', a.place_directive,
        'AggressiveExplore', 'AggressiveExplore', 'A1', 'B2', 'C0', 'H2']
    ip_cmd = (hbm_sc_command(cl, t, out / 'hbm_sc.log', hdk)
              if is_hbm(t) and not is_engine(t) and a.flow != 'ImplCL' else None)
    (out / 'invocation.json').write_text(json.dumps({'command': cmd, 'ip_command': ip_cmd,
        'cwd': str(cl / 'build/scripts'),
        'environment': {k: env[k] for k in env if k.startswith(('HDK_', 'OTPU_INCR_', 'OTPU_RQS')) or
            k in ('CL_DIR', 'CL', 'SHELL_MODE', 'BUILD_FLOW', 'BUILD_TAG', 'ENCRYPT', 'VIVADO_TOOL_VERSION')}}, indent=2) + '\n')
    if a.prepare_only:
        print(f'Prepared {cl}; invocation: {out / "invocation.json"}')
        return
    version = command('vivado', '-version')
    check_vivado_version(version, t.platform['vivado'])
    if ip_cmd:                          # the SmartConnect fabric first (an HBM target)
        with (out / 'hbm_sc_console.log').open('w') as log:
            r = subprocess.run(ip_cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode or not (cl / 'ip/ip_done.txt').is_file():
            raise RuntimeError(f'SmartConnect generation failed; see {out / "hbm_sc_console.log"}')
    with (out / 'console.log').open('w') as log:
        result = subprocess.run(cmd, cwd=cl / 'build/scripts', env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    (out / 'exit_code').write_text(str(result.returncode) + '\n')
    stage = 'post_synth' if a.flow == 'SynthCL' else 'post_route'
    checkpoint = cl / f'build/checkpoints/cl_opentpu.{build_id}.{stage}.dcp'
    if result.returncode or not checkpoint.is_file():
        raise RuntimeError(f'{a.flow} failed; see {out / "console.log"}')
    if reference and not re.search(r'^OTPU_INCREMENTAL_REFERENCE ',
                                     (out / 'console.log').read_text(), re.M):
        raise RuntimeError('The incremental reference was never loaded')
    if a.qor_suggestions and not re.search(r'^OTPU_QOR_SUGGESTIONS ',
                                           (out / 'console.log').read_text(), re.M):
        raise RuntimeError('The QoR suggestions were never read')
    if a.flow != 'SynthCL' and not (checkpoint.parent /
            f'cl_opentpu.{build_id}.route_validation.passed').is_file():
        raise RuntimeError(f'Post-route validation failed; see {out / "console.log"}')
    artifacts = [checkpoint]
    if a.package:
        archive = package_afi(cl, sh, hdk, build_id, env)
        artifacts.extend([archive, archive.with_name(
            archive.name.removesuffix('.Developer_CL.tar') + '.packaging.json')])
    (out / 'outputs.json').write_text(json.dumps({'flow': a.flow,
        'inputs_sha256': sha(out / 'inputs.json'),
        'artifacts': {str(f.relative_to(out)): {'sha256': sha(f), 'bytes': f.stat().st_size}
                      for f in artifacts}}, indent=2) + '\n')
    print(f'{a.flow} complete: {checkpoint}')
    print('Review timing/CDC/methodology and DDR calibration reports before hardware deployment.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(str(e))
