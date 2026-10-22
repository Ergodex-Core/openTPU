"""Mixed rowlane MXU and two-source VPU exercise with TMEM feedback evidence.

The PMP=2 mode is the core-clock model of pumped TMEM. Both modes receive the
same program, image and native-memory stall seed; their cycle counts need not
have any particular ordering because the two units can rephase each other.
"""

import json
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim, target
from opentpu.isasim import Machine
from test_quant_rtl import _xrows
from test_wide_port_rtl import _rl_matrix, _rl_prog
from tools.wideport_unit_feedback import validate_feedback_trace


def _feedback_records(path: Path):
    assert path.is_file() and path.stat().st_size, f"missing TMEM feedback trace: {path}"
    with path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    assert records and records[0].get("kind") == "config", "missing feedback config header"
    assert records[-1].get("kind") == "end", "missing feedback end record (trace truncated?)"
    assert not any(r.get("kind") == "truncated" for r in records), "feedback trace hit its limit"
    events = records[1:-1]
    assert events and all(r.get("kind") == "cycle" for r in events)
    return records[0], events, records[-1]


@pytest.mark.parametrize("pmp", [0, 2])
def test_wideport_mixed_mxu_vpu_tmem_feedback(have_verilator, monkeypatch, tmp_path, pmp):
    """Two live VPU operands contest TMEM with rowlane MM and native DMA under stalls.

    Preserve the full simulator inputs, trace and stdout when WP_FEEDBACK_EVIDENCE
    names a directory. The resource checker validates the actual request/replay
    stream; whole-TMEM and whole-DRAM state must also match the bit-exact ISA.
    """
    rng = np.random.default_rng(9041)
    deterministic_memory = os.environ.get("WP_DETERMINISTIC_MEMORY") == "1"
    cfg = Config(S=1, D=128, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, PAIR=True,
                 DSTEP=True,
                 PORTB=4, ROWLANES=True, ALINE=64, DRAM_BYTES=1 << 20)
    kb = 9
    rows, srows = _rl_matrix(rng, "int8", 13, kb * cfg.D)
    x = _xrows(rng, 1, kb * cfg.D, cfg.D // 4)
    prog, dram = _rl_prog(cfg, "int8", 4, 13, kb, 0, x, rows, srows)

    # TMEM word intervals [4096,12288), [12288,20480) and [50000,58192)
    # are mutually disjoint and avoid the rowlane MM outputs at 0x8000,
    # 0x9000 and 0xa000, the activation, and the scale/alpha words.
    # DRAM byte intervals [0x10000,0x18000), [0x18000,0x20000) avoid
    # X, WB=0x60000, SB=0x90000, SB2=0xa0000 and AL=0xb0000.
    n = 32 * 256
    a, b, dst = 4096, 12288, 50000
    dram_a, dram_b, dram_out = 0x10000, 0x18000, 0xD0000
    va = rng.standard_normal(n).astype(np.float32)
    vb = rng.standard_normal(n).astype(np.float32)
    dram[dram_a:dram_a + 4 * n] = va.view(np.uint8)
    dram[dram_b:dram_b + 4 * n] = vb.view(np.uint8)
    prog[:0] = [I.ld(dram_a, a, n), I.ld(dram_b, b, n)]
    prog.insert(5, I.vop(I.V_ADD, dst, a, b, 32, 256, 256, 256, 256, I.B_FULL))
    prog.insert(-1, I.st(dram_out, dst, n))
    if deterministic_memory:
        # The scoped adapter models reads only. The VPU's independent ST can
        # otherwise overtake a still-running MM (the preserved PMP2 v1 case).
        # BAR explicitly completes all older work before that first write.
        prog.insert(-2, I.bar())
    assert any(ins.op == I.MM and I.mm_layout(ins) == 4 for ins in prog)
    assert sum(ins.op == I.VOP for ins in prog) == 1

    evidence_root = Path(os.environ.get("WP_FEEDBACK_EVIDENCE", tmp_path)).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    keep = Path(tempfile.mkdtemp(prefix=f"pmp{pmp}-", dir=evidence_root))
    trace = keep / "tmfeedback.jsonl"
    plusargs = [f"+tmfeedback={trace}", "+tmfeedback_limit=32768"]
    if deterministic_memory:
        plusargs += ["+nat_deterministic=1", f"+nat_feedback={keep / 'native-feedback.jsonl'}",
                     f"+adapter_feedback={keep / 'adapter-feedback.jsonl'}"]
    original_run_sim = rtlsim.run_sim

    def record_sim_output(cmd, timeout=None):
        result = original_run_sim(cmd, timeout=timeout)
        (keep / "sim.stdout.log").write_text(result.stdout)
        (keep / "sim.stderr.log").write_text(result.stderr)
        return result

    # This feedback build is large; bound Verilator's compile parallelism on EC2.
    flags = rtlsim.VFLAGS.copy()
    flags[flags.index("-j") + 1] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setattr(rtlsim, "run_sim", record_sim_output)
    monkeypatch.setenv("OTPU_NCH", "8")
    monkeypatch.setenv("OTPU_NAXI", "0")
    monkeypatch.delenv("OTPU_SIM_PLUSARGS", raising=False)
    uarch = {**target.sim_uarch("f2-hbm:p4rl8a64"), "TMEM_PMP": pmp}
    inputs = {
        "dram_sha256": hashlib.sha256(dram.tobytes()).hexdigest(),
        "program_sha256": hashlib.sha256(np.asarray([ins.encode() for ins in prog], dtype="<u4").tobytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(rtlsim.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (rtlsim.RTL / "top/otpu_slice.sv", rtlsim.RTL / "mem/otpu_tmem.sv",
                                       rtlsim.RTL / "vpu/otpu_vpu.sv", rtlsim.RTL / "mxu/otpu_mxu.sv",
                                       rtlsim.RTL / "top/otpu_top.sv", rtlsim.TB / "otpu_native_mem.sv")},
        "stall": 50, "seed": 7, "lat": 120, "deterministic_memory": deterministic_memory,
        "read_phase_barrier": deterministic_memory,
        "plusargs": plusargs}
    original_build = rtlsim.build

    def record_build(top, sources, params=None):
        # Capture the actual compile parameters, including boot's enlarged
        # DRAM/IMEM capacities, rather than reconstructing top_params later.
        if top == "tb_top":
            inputs["parameters"] = params
            (keep / "feedback-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        return original_build(top, sources, params)

    monkeypatch.setattr(rtlsim, "build", record_build)
    isa = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, stats = rtlsim.run(
        cfg, [prog], [dram.copy()], axi=True, boot=True, stall=50, seed=7, lat=120,
        max_cycles=1_000_000, keep=keep,
        uarch=uarch,
        plusargs=plusargs,
    )
    assert np.array_equal(drams[0], isa.slices[0].dram), "DRAM differs from the ISA"
    assert np.array_equal(tmems[0], isa.slices[0].tmem), "TMEM differs from the ISA"
    assert stats["native_b"][0]["wide"] > 0, stats["native_b"]

    header, events, end = _feedback_records(trace)
    assert header["schema"] == 1 and header["pmp"] == pmp
    assert header["side_mask"] == (0xF0 if pmp == 0 else 0x0C)
    assert header["words"] == cfg.TMEM_WORDS and header["lanes"] == 8
    assert header["vpu_se"] == 1 and header["limit"] == 32768
    assert len(events) < 32768 and end["cycles"] == len(events)
    summary = validate_feedback_trace(trace, predict_producers=True)
    assert summary is not None
    assert summary["command_driven_vadd"] and summary["command_driven_mxu_row_valid"]
    assert not summary["whole_program_predictor"] and not summary["calibrated"] and summary["tok_s"] is None

    active = [e for e in events if not e["srst"] and not e["sys_rst"] and not e["halted"]]
    assert any(e["reads"][3] and e["reads"][4] and e["taken"][3] and e["taken"][4]
               for e in active), "VPU never read two independent TMEM operands together"
    assert any(e["vpu"]["en"] and (e["mxu"]["rows_live"] or e["mxu"]["drain_go"])
               for e in active), "VPU never overlapped MXU compute/drain"
    assert any(e["lost"] or e["frozen"] for e in active), "no TMEM replay exercised"
    memory_summary = None
    if deterministic_memory:
        from tools.wideport_memory_feedback import validate_native_trace
        from tools.wideport_memory_coupled import validate_memory_coupled
        memory_summary = {"native_boundary": validate_native_trace(keep / "native-feedback.jsonl"),
                          "coupled_arrivals": validate_memory_coupled(keep)}
    (keep / "feedback-summary.json").write_text(json.dumps(
        {"pmp": pmp, "cycles": stats["cycles"], "events": len(events),
         "resource": summary, "memory": memory_summary}, indent=2, default=str) + "\n")


@pytest.mark.parametrize("key,value", [("TMEM_GST", 0), ("S", 2), ("TMEM_PRK", 1)])
def test_feedback_rejects_unsupported_compile_contract(tmp_path, key, value):
    params = {**target.sim_uarch("f2-hbm:p4rl8a64tdp"), "S": 1, "LANES": 8,
              "TMEM_WORDS": 65536, "TMEM_SBP": 3}
    params[key] = value
    (tmp_path / "feedback-inputs.json").write_text(json.dumps({"parameters": params}))
    trace = tmp_path / "tmfeedback.jsonl"
    records = [{"kind": "config", "schema": 1, "pmp": 0, "side_mask": 240,
                "words": 65536, "vpu_se": 1}, {"kind": "cycle", "cycle": 0},
               {"kind": "end", "cycles": 1}]
    trace.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="config." + key):
        validate_feedback_trace(trace)
