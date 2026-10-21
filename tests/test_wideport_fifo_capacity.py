"""Force the wide-port MXU result FIFO to its 32- and 64-row capacity."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim, target
from opentpu.isasim import Machine
from test_quant_rtl import _xrows
from test_wide_port_rtl import _rl_matrix, _rl_prog
from test_wideport_slice_feedback import _feedback_records
from tools.wideport_unit_feedback import validate_feedback_trace


@pytest.mark.parametrize("mxu_cl,rf_capacity", [(8, 32), (16, 64)])
def test_wideport_mxu_result_fifo_capacity(
    have_verilator, monkeypatch, tmp_path, mxu_cl, rf_capacity
):
    """VPU all-bank writes hold the drain while one KB1 rowlane MM fills its FIFO.

    MXU_CL=16 is the f2-hbm configuration. MXU_CL=8 is a counterfactual
    32-row FIFO instance using the same program and native-memory traffic.
    """
    rng = np.random.default_rng(9041)
    cfg = Config(S=1, D=128, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, PAIR=True,
                 DSTEP=True, PORTB=4, ROWLANES=True, ALINE=64, DRAM_BYTES=1 << 20)
    # 129 row groups leave headroom for rows drained before contention starts.
    nrows, kb = 513, 1
    rows, srows = _rl_matrix(rng, "int8", nrows, kb * cfg.D)
    x = _xrows(rng, 1, kb * cfg.D, cfg.D // 4)
    base, dram = _rl_prog(cfg, "int8", 4, nrows, kb, 0, x, rows, srows)
    # Keep its QACT and first rowlane MM only. The other four MMs are not needed
    # for result-FIFO pressure and would make this a longer end-to-end test.
    prog = base[:4] + [I.halt()]
    assert prog[3].op == I.MM and I.mm_layout(prog[3]) == 4

    # The two VPU sources and destination are disjoint TMEM intervals well
    # clear of activation, MM output at 0x8000, and alpha at 0x7000.
    words = 32 * 256
    a, b, dst = 4096, 12288, 50000
    dram_a, dram_b, dram_out = 0x10000, 0x18000, 0xD0000
    va = rng.standard_normal(words).astype(np.float32)
    vb = rng.standard_normal(words).astype(np.float32)
    dram[dram_a:dram_a + 4 * words] = va.view(np.uint8)
    dram[dram_b:dram_b + 4 * words] = vb.view(np.uint8)
    prog[:0] = [I.ld(dram_a, a, words), I.ld(dram_b, b, words)]
    # Repeat the same valid source/destination row for 2048 issuing cycles.
    # The original 1024-cycle stream filled RF32 but only reached 56/64 on
    # the board configuration; preserve that failed receipt rather than
    # treating near-capacity occupancy as full-capacity coverage.
    prog.insert(5, I.vop(I.V_ADD, dst, a, b, 64, 256, 0, 0, 0, I.B_FULL))
    prog.insert(-1, I.st(dram_out, dst, words))
    assert sum(ins.op == I.MM for ins in prog) == 1
    assert sum(ins.op == I.VOP for ins in prog) == 1

    root = Path(os.environ.get("WP_FIFO_EVIDENCE", tmp_path)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    keep = Path(tempfile.mkdtemp(prefix=f"rf{rf_capacity}-", dir=root))
    trace = keep / "tmfeedback.jsonl"
    uarch = {**target.sim_uarch("f2-hbm:p4rl8a64"),
             "TMEM_PMP": 2, "MXU_CL": mxu_cl}
    inputs = {
        "dram_sha256": hashlib.sha256(dram.tobytes()).hexdigest(),
        "program_sha256": hashlib.sha256(
            np.asarray([ins.encode() for ins in prog], dtype="<u4").tobytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(rtlsim.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (rtlsim.RTL / "top/otpu_slice.sv",
                         rtlsim.RTL / "mem/otpu_tmem.sv",
                         rtlsim.RTL / "vpu/otpu_vpu.sv",
                         rtlsim.RTL / "mxu/otpu_mxu.sv")},
        "stall": 0, "seed": 7, "lat": 8, "mxu_cl": mxu_cl,
        "expected_rf_capacity": rf_capacity,
    }
    original_build, original_run_sim = rtlsim.build, rtlsim.run_sim

    def record_build(top, sources, params=None):
        if top == "tb_top":
            inputs["parameters"] = params
            (keep / "feedback-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        return original_build(top, sources, params)

    def record_sim_output(cmd, timeout=None):
        result = original_run_sim(cmd, timeout=timeout)
        (keep / "sim.stdout.log").write_text(result.stdout)
        (keep / "sim.stderr.log").write_text(result.stderr)
        return result

    flags = rtlsim.VFLAGS.copy()
    flags[flags.index("-j") + 1] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setattr(rtlsim, "build", record_build)
    monkeypatch.setattr(rtlsim, "run_sim", record_sim_output)
    monkeypatch.setenv("OTPU_NCH", "8")
    monkeypatch.setenv("OTPU_NAXI", "0")
    monkeypatch.delenv("OTPU_SIM_PLUSARGS", raising=False)

    isa = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, stats = rtlsim.run(
        cfg, [prog], [dram.copy()], axi=True, boot=True, stall=0, seed=7, lat=8,
        max_cycles=150_000, keep=keep, uarch=uarch,
        plusargs=[f"+tmfeedback={trace}", "+tmfeedback_limit=32768"],
    )
    assert np.array_equal(drams[0], isa.slices[0].dram), "DRAM differs from ISA"
    assert np.array_equal(tmems[0], isa.slices[0].tmem), "TMEM differs from ISA"
    assert stats["native_b"][0]["wide"] > 0, stats["native_b"]

    header, events, end = _feedback_records(trace)
    assert header["schema"] == 1 and header["pmp"] == 2
    assert header["side_mask"] == 0x0C and header["lanes"] == 8
    assert header["words"] == cfg.TMEM_WORDS and header["vpu_se"] == 1
    assert header["limit"] == 32768 and end["cycles"] == len(events) < 32768
    summary = validate_feedback_trace(trace, predict_producers=True)
    assert summary["command_driven_vadd"] and summary["command_driven_mxu_row_valid"]
    assert not summary["whole_program_predictor"] and not summary["calibrated"] and summary["tok_s"] is None
    active = [e for e in events if not e["srst"] and not e["sys_rst"] and not e["halted"]]
    assert active and all(e["mxu"]["rf_capacity"] == rf_capacity for e in active)
    assert summary["mxu_max_result_rows"] == rf_capacity, summary

    def capacity_blocked(e):
        m = e["mxu"]
        return (m["rf_n"] == rf_capacity and
                m["rows_live"] + m["rows_p"] >= rf_capacity and
                m["more"] and m["f_av"] and (m["c_unit"] or m["s_av"]) and
                m["first_row"] and m["row_in"] and not m["pop"])

    blocked = [e for e in active if capacity_blocked(e)]
    assert blocked, "no ready row blocked solely by MXU result-FIFO capacity"
    assert any(e["vpu"]["head_words"] and e["vpu"]["gnt"] and not e["mxu"]["t_gnt"]
               for e in blocked), "capacity was not reached during VPU write denial"
    assert any(e["mxu"]["d_pop"] for e in active if e["cycle"] > blocked[0]["cycle"]), \
        "MXU drain did not resume after saturation"
    assert summary["mxu_result_push_while_drain_denied"] > 0, summary
    (keep / "feedback-summary.json").write_text(json.dumps(
        {"mxu_cl": mxu_cl, "rf_capacity": rf_capacity, "cycles": stats["cycles"],
         "events": len(events), "first_capacity_block_cycle": blocked[0]["cycle"],
         "resource": summary}, indent=2, default=str) + "\n")
