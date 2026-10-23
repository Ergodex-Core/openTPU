"""C2 microbenchmark RTL validation with synthetic native response service.

Outputs and counters are produced by RTL. Memory latency is a test parameter,
not measured HBM latency. The shell, AXI bridges and FPGA timing are not modeled.
"""
from pathlib import Path
import struct
import tempfile
import subprocess
import numpy as np
from . import rtlsim
from .c2_f2 import Fixture, ACT, DA, SA, DW, SW, R_MEM_ADDR, R_MEM_WDATA


def build():
    sources = [rtlsim.RTL / p for p in (
        "vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv",
        "tech/generic/otpu_tech_dsp.sv", "mxu/otpu_mxu.sv", "mxu/otpu_c2_mxu.sv",
        "c2/otpu_c2_hbm_kernel.sv")]
    sources.append(rtlsim.TB / "tb_c2_hbm.sv")
    original = rtlsim.VFLAGS
    rtlsim.VFLAGS = original.copy()
    rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = "2"
    try:
        return rtlsim.build("tb_c2_hbm", sources)
    finally:
        rtlsim.VFLAGS = original


def _write(fixture: Fixture, mode: int, path):
    fixture.validate()
    ops = []
    def window(addr, values):
        ops.append((R_MEM_ADDR, addr))
        ops.extend((R_MEM_WDATA, int(v)) for v in values)
    window(ACT, np.frombuffer(fixture.acts.tobytes(), dtype="<u4"))
    window(DA, np.frombuffer(fixture.da.tobytes(), dtype="<u4"))
    window(SA, fixture.sa)
    window(SW, fixture.sw)
    for row in range(fixture.rows):
        window(DW + row * 256, fixture.dw[row])
    banks = fixture.banks()
    with Path(path).open("wb") as f:
        for n in (fixture.rows, fixture.kb, mode, fixture.base_beat, len(banks[0])//64, len(ops)):
            f.write(struct.pack(">I", n))
        for bank in banks:
            for beat in bank.reshape(-1, 64):
                f.write(beat.tobytes()[::-1])
        for addr, value in ops:
            f.write(struct.pack(">II", addr, value))


def run(fixture, mode=1, *, seed=1, latency=30, stall=20, delay_pc0=0, fault=0, out_dir=None):
    if mode not in (0, 1) or latency < 1 or not 0 <= stall < 100 or delay_pc0 < 0 or fault not in range(4):
        raise ValueError("invalid native simulation configuration")
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="c2-hbm-") as d:
            return run(fixture, mode, seed=seed, latency=latency, stall=stall,
                       delay_pc0=delay_pc0, fault=fault, out_dir=d)
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    _write(fixture, mode, path / "input.bin")
    exe = build()
    result = subprocess.run([str(exe), f"+input={path/'input.bin'}", f"+events={path/'events.txt'}",
              f"+results={path/'results.txt'}", f"+seed={seed}", f"+latency={latency}",
              f"+stall={stall}", f"+delay_pc0={delay_pc0}", f"+fault={fault}"],
              capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    counters, bits, ints, faults = {}, [], [], []
    for line in (path / "results.txt").read_text().splitlines():
        fields = line.split()
        if fields[0] == "C":
            counters[int(fields[1])] = int(fields[2])
        elif fields[0] == "R":
            assert int(fields[1]) == len(bits)
            bits.append(int(fields[2], 16))
            integer = int(fields[3], 16)
            ints.append(integer - (1 << 48) if integer & (1 << 47) else integer)
        else:
            faults.append(tuple(map(int, fields[1:])))
    expected, integers = fixture.reference()
    if bits != expected.tolist() or ints != integers:
        raise AssertionError(f"C2 native RTL result mismatch: bits={bits} expected={expected.tolist()} ints={ints} expected_ints={integers}")
    expected_reads = 2 * fixture.rows * fixture.kb
    assert counters[0x030] == 0 and counters[0x034] == fixture.rows
    assert counters[0x048] == counters[0x04c] == expected_reads
    expected_groups = fixture.rows if mode == 0 else (fixture.rows + 3) // 4
    assert counters[0x040] == counters[0x044] == expected_groups
    for pc, beats in enumerate(fixture.expected_pc_beats()):
        assert counters[0x100+pc*32] == counters[0x104+pc*32] == beats
        assert counters[0x108+pc*32] <= 32
    events = np.loadtxt(path / "events.txt", dtype=np.int64, ndmin=2)
    completed = events[events[:, 0] == events[-1, 0]]
    assert len(completed) == counters[0x038] + (counters[0x03c] << 32)
    assert sum(int(mask).bit_count() for mask in completed[:, 2]) == expected_reads
    assert sum(int(mask).bit_count() for mask in completed[:, 3]) == expected_reads
    assert int(completed[:, 4].sum()) == counters[0x040]
    assert int(completed[:, 5].sum()) == counters[0x044]
    assert int(completed[:, 7].sum()) == counters[0x050]
    assert int(completed[:, 8].max()) == counters[0x068]
    return dict(bits=bits, integers=ints, counters=counters, events=events,
                faults=faults, cycles=counters[0x038]+(counters[0x03c]<<32),
                provenance="RTL simulation; synthetic in-order native service; not F2 hardware")


def lens_data(fixture, mode, evidence, name, source=None, service=None):
    """Render actual kernel handshake times; never synthesize board evidence."""
    from . import lens
    from .isasim import Config
    from .profile import UNITS
    events = evidence['events']
    events = events[events[:, 0] == events[-1, 0]].copy()
    origin = int(events[0, 1])
    events[:, 1] -= origin
    cycles = evidence['cycles']
    instructions, programs = [], []
    for idx, accepted in enumerate(events[events[:, 4] != 0]):
        tag = int(accepted[10])
        commits = events[(events[:, 5] != 0) & (events[:, 11] == tag)]
        advances = events[(events[:, 7] != 0) & (events[:, 12] == tag)]
        assert len(commits) == 1 and len(advances) == fixture.kb
        count = min(4, fixture.rows-tag) if mode else 1
        detail = f'row={tag} count={count} KB={fixture.kb}; kernel command, resident ACT/scales'
        op = 'MM_ROWS' if mode else 'MM'
        start, execute, end = int(accepted[1]), int(advances[0, 1]), int(commits[0, 1])+1
        programs.append([idx, op, detail, 'C2 native benchmark RTL', -1])
        instructions.append([0, idx, idx, 1, op, detail, start, start, -1,
                             execute, end, -1, {}, fixture.kb*count, fixture.kb*count, 0])
    bound = max(fixture.expected_pc_beats())
    c = evidence['counters']
    return dict(kind='rtl', name=name+' [KERNEL RTL / SYNTHETIC NATIVE SERVICE]',
        cycles=cycles, clock_mhz=200.0,
        config=lens._config(Config(S=1,D=128,MCOLS=4,ACT_BLOCKS=256,TMEM_WORDS=512),
                            {'C':1,'PCS':8,'C2_COLUMNS':4,'FIFO_DEPTH':256}),
        roofline=dict(bound=bound,efficiency=bound/cycles,
                      per_slice=[dict(bound=bound,portb=c[0x04c],porta=0)]),
        macs=fixture.rows*fixture.kb*128, bytes=c[0x04c]*64,
        peak_macs=512,peak_bytes=512,units=UNITS,
        slices=[dict(ports={},busy={},lose={},buckets={})],
        instrs=instructions,programs=[programs],sources=[],mxu_gaps={},
        c2_native_rtl=dict(source=source,service=service,fixture_sha256=fixture.digest(),
                           counters={str(k):v for k,v in c.items()},events=events.tolist(),
                           fields=['run','cycle','request_mask','response_mask','command_accept',
                                   'commit','output_mask','advance','pending','state',
                                   'accepted_tag','commit_tag','array_tag'],bit_exact=True),
        notes=[dict(level='warn',text='Actual Verilator kernel cycles with randomized synthetic native service. '
                    'No F2 shell, AXI bridge, HBM controller, physical FPGA or timing closure measurement. '
                    '200 MHz is only an assumed display clock.'),
               dict(level='info',text='START through final commit; excludes OCL/PCIS preload and result readback. '
                    'Both modes use identical physical PC-pair weights, shared ACT0 and local result SRAM. '
                    'No full-token, HBM-output-write, or production ISA claim.'),
               dict(level='info',text=f'Native request blocked cycles {c[0x054]}; '
                    f'incomplete weight-fill cycles {c[0x058]}; advances {c[0x050]}. '
                    'Fill counters overlap successful ingress and computation; they are not exclusive stall causes. '
                    'Roofline is the native per-PC transfer floor, not measured critical-path attribution.')])
