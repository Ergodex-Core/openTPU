"""Matched one/two-credit test of MXU descriptors and early operand prefetch.

The existing simulation-only +hz_uq queue cap gates instruction dispatch:
one credit waits for completion, two permit prefetch while the first drains.
BAR alone does not prevent the RTL's intentional early weight prefetch.
This is a structural overlap check, not a claimed end-to-end speedup.
The full ISA state and three independent feedback boundaries remain the result
oracles. The native adapter checker intentionally stops before the first ST.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim, target, waits as W
from opentpu.isasim import Machine
from test_quant_rtl import _xrows
from test_wide_port_rtl import OUT, WB, SB, _rl_matrix, _rl_prog
from tools.wideport_memory_coupled import validate_memory_coupled
from tools.wideport_memory_feedback import validate_native_trace
from tools.wideport_unit_feedback import validate_feedback_trace
from tools.wideport_credit_budget import analyze_credit_budget


N, KB, L = 13, 9, 4
WB1, SB1 = 0x70000, 0x98000
OUT1 = OUT + 4096
V_A, V_B, V_DST = 4096, 12288, 50000
D_A, D_B, D_OUT0, D_OUT1 = 0x10000, 0x18000, 0xD0000, 0xD0100


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _programs():
    """Identical program and image; only the simulator's queue cap differs."""
    rng = np.random.default_rng(9041)
    cfg = Config(S=1, D=128, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16,
                 PAIR=True, DSTEP=True, PORTB=4, ROWLANES=True,
                 ALINE=64, DRAM_BYTES=1 << 20)
    rows, scales = _rl_matrix(rng, "int8", N, KB * cfg.D)
    x = _xrows(rng, 1, KB * cfg.D, cfg.D // 4)
    source, dram = _rl_prog(cfg, "int8", L, N, KB, 0, x, rows, scales)
    mm0 = source[3]
    rs, srs = mm0.w[4] & ~15, mm0.w[6] & ~1
    groups = (N + L - 1) // L
    assert rs == L * KB * cfg.D and srs == 64
    assert WB + groups * rs <= WB1 and WB1 + groups * rs < SB
    assert SB + groups * srs <= SB1 and SB1 + groups * srs < 0xA0000
    dram[WB1:WB1 + groups * rs] = dram[WB:WB + groups * rs]
    dram[SB1:SB1 + groups * srs] = dram[SB:SB + groups * srs]
    va = rng.standard_normal(8).astype(np.float32)
    vb = rng.standard_normal(8).astype(np.float32)
    dram[D_A:D_A + va.nbytes] = va.view(np.uint8)
    dram[D_B:D_B + vb.nbytes] = vb.view(np.uint8)
    mm1 = I.mm(sa=WB1, ssa=SB1, out=OUT1, n=N, kb=KB, rs=rs,
               ors=N + 1, m=1, ab=0, srs=srs, wf=I.W8, rl=L)
    # V_ADD is tiny and completes before the first MM. It exercises the
    # command-driven VPU checker without adding VPU/MXU contention here.
    prefix = [source[0], I.ld(D_A, V_A, 8), I.ld(D_B, V_B, 8),
              I.vop(I.V_ADD, V_DST, V_A, V_B, 1, 8, 8, 8, 8, I.B_FULL),
              I.bar(), source[2], mm0]
    suffix = [mm1, I.bar(), I.st(D_OUT0, OUT, N),
              I.st(D_OUT1, OUT1, N), I.halt()]
    programs = {mode: prefix + [I.nop()] + suffix
                for mode in ("one_credit", "two_credit")}
    assert sum(ins.op == I.MM for ins in programs["one_credit"]) == 2
    assert sum(ins.op == I.QACT for ins in programs["one_credit"]) == 1
    assert I.assemble(programs["one_credit"]).tobytes() == I.assemble(programs["two_credit"]).tobytes()
    return cfg, programs, dram


def _records(path):
    assert path.is_file() and path.stat().st_size, f"missing feedback trace {path}"
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert rows[0].get("kind") == "config" and rows[-1].get("kind") == "end", path
    assert rows[-1]["cycles"] == len(rows) - 2, path
    assert all(row.get("kind") == "cycle" and row["cycle"] == i
               for i, row in enumerate(rows[1:-1])), path
    return rows[0], rows[1:-1]


def _first_last(events, predicate):
    hits = [e["cycle"] for e in events if predicate(e)]
    return {"first": hits[0] if hits else None,
            "last": hits[-1] if hits else None, "cycles": len(hits)}


def _timing(events, *, mode, rs, srs):
    starts = [e["cycle"] for e in events if e["start"] & (1 << 1)]
    completed = [e["cycle"] for e in events if e["mxu"]["c_done"]]
    assert len(starts) == len(completed) == 2, (mode, starts, completed)
    wb1 = (WB1 // 4, (WB1 + ((N + L - 1) // L) * rs) // 4)
    sb1 = (SB1 // 4, (SB1 + ((N + L - 1) // L) * srs) // 4)
    second_b = [e["cycle"] for e in events if e["issue"]["b_req"]
                and wb1[0] <= e["issue"]["b_addr"] < wb1[1]]
    second_a = [e["cycle"] for e in events if e["issue"]["a_req"]
                and sb1[0] <= e["issue"]["a_addr"] < sb1[1]]
    assert second_a and second_b, (mode, "missing second-buffer requests")
    q2 = _first_last(events, lambda e: e["mxu"]["q_n"] == 2)
    if mode == "two_credit":
        assert starts[1] < completed[0], (starts, completed)
        assert second_a[0] < completed[0] and second_b[0] < completed[0], (
            second_a[0], second_b[0], completed[0])
        assert q2["cycles"] > 0, "two MXU descriptors were never resident together"
    else:
        assert starts[1] > completed[0], (starts, completed)
        assert second_a[0] > completed[0] and second_b[0] > completed[0]

    mx = lambda e: e["mxu"]
    ie = lambda e: e["issue"]
    first_head = [e for e in events if e["cycle"] <= completed[0]]
    second_head = [e for e in events if completed[0] < e["cycle"] <= completed[1]]
    return {
        "mode": mode, "mm_start": starts, "mm_c_done": completed,
        "mm2_first_a_request": second_a[0], "mm2_first_b_request": second_b[0],
        "two_descriptors": q2,
        "issue": _first_last(events, lambda e: ie(e)["a_req"] or ie(e)["b_req"]),
        "compute": _first_last(events, lambda e: mx(e)["compute"]),
        "drain": _first_last(events, lambda e: mx(e)["drain_go"]),
        "released_work": _first_last(events, lambda e: mx(e)["more"]),
        "next_pop_head": _first_last(events, lambda e: mx(e)["pn"]),
        "completion_intervals_not_producer_ownership": {
            "through_first_c_done": {key: _first_last(first_head, lambda e, k=key: mx(e)[k])
                    for key in ("compute", "drain_go", "more")},
            "first_to_second_c_done": {key: _first_last(second_head, lambda e, k=key: mx(e)[k])
                    for key in ("compute", "drain_go", "more")},
        },
        "highwater": {key: max(e[unit][key] for e in events)
                      for unit, key in (("issue", "occ"), ("issue", "tg_n"),
                                        ("mxu", "f_count"), ("mxu", "s_count"),
                                        ("mxu", "rf_n"), ("mxu", "rows_live"),
                                        ("mxu", "rows_p"))},
    }


def test_wideport_one_vs_two_credit(have_verilator, monkeypatch, tmp_path):
    cfg, programs, dram = _programs()
    board = {**target.sim_uarch("f2-hbm:p4rl8a64"), "TMEM_PMP": 0}
    evidence_root = Path(os.environ.get("WP_CREDIT_EVIDENCE", tmp_path)).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="two-credit-", dir=evidence_root))
    flags = rtlsim.VFLAGS.copy()
    flags[flags.index("-j") + 1] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setenv("OTPU_NCH", "8")
    monkeypatch.setenv("OTPU_NAXI", "0")
    monkeypatch.setenv("OTPU_WAITS", "1")
    monkeypatch.delenv("OTPU_SIM_PLUSARGS", raising=False)

    current = {"directory": None, "manifest": None}
    original_build, original_run_sim = rtlsim.build, rtlsim.run_sim

    def record_build(top, sources, params=None):
        if top == "tb_top":
            manifest = current["manifest"]
            manifest["parameters"] = dict(params or {})
            manifest["source_sha256"] = {
                str(Path(p).resolve().relative_to(rtlsim.ROOT)): _sha(Path(p).read_bytes())
                for p in sources}
            (current["directory"] / "feedback-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return original_build(top, sources, params)

    def record_sim_output(cmd, timeout=None):
        (current["directory"] / "sim-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        result = original_run_sim(cmd, timeout=timeout)
        (current["directory"] / "sim.stdout.log").write_text(result.stdout)
        (current["directory"] / "sim.stderr.log").write_text(result.stderr)
        return result

    monkeypatch.setattr(rtlsim, "build", record_build)
    monkeypatch.setattr(rtlsim, "run_sim", record_sim_output)
    results, errors = {}, {}
    for mode, prog in programs.items():
        keep = run_dir / mode
        keep.mkdir()
        current["directory"] = keep
        words = I.assemble(prog).astype("<u4")
        waits = W.wait_counts(prog, cfg.D, cfg.S).astype("<u8")
        queue_cap = 1 if mode == "one_credit" else 2
        plusargs = [f"+hz_uq=0:{queue_cap}:0:0:0", "+nat_deterministic=1", f"+nat_feedback={keep / 'native-feedback.jsonl'}",
                    f"+adapter_feedback={keep / 'adapter-feedback.jsonl'}",
                    f"+tmfeedback={keep / 'tmfeedback.jsonl'}", "+tmfeedback_limit=32768"]
        manifest = {"variant": mode, "mxu_window_cap": queue_cap, "pmp": 0, "nch": 8,
                    "program_sha256": _sha(words.tobytes()),
                    "waits_sha256": _sha(waits.tobytes()),
                    "dram_sha256": _sha(dram.tobytes()),
                    "instruction_count": len(prog), "loader_instruction_bytes": words.nbytes,
                    "loader_wait_bytes": waits.nbytes,
                    "stall": 50, "seed": 7, "lat": 120, "plusargs": plusargs,
                    "requested_uarch": board}
        current["manifest"] = manifest
        (keep / "feedback-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (keep / "waits.hex").write_text("".join(f"{int(w):016x}\n" for w in waits))
        try:
            isa = Machine(cfg, [prog], [dram.copy()]).run()
            drams, tmems, stats = rtlsim.run(
                cfg, [prog], [dram.copy()], axi=True, boot=True,
                stall=50, seed=7, lat=120, max_cycles=1_000_000,
                keep=keep, uarch=board, plusargs=plusargs)
            assert np.array_equal(drams[0], isa.slices[0].dram), "DRAM differs from ISA"
            assert np.array_equal(tmems[0], isa.slices[0].tmem), "TMEM differs from ISA"
            assert np.array_equal(tmems[0][OUT:OUT + N], tmems[0][OUT1:OUT1 + N]), (
                "identical MM inputs produced different output tiles")
            header, events = _records(keep / "tmfeedback.jsonl")
            assert (header["schema"], header["pmp"], header["side_mask"],
                    header["words"], header["lanes"], header["vpu_se"]) == (
                    1, 0, 0xF0, cfg.TMEM_WORDS, 8, 1)
            assert len(events) < 32768
            structure = _timing(events, mode=mode, rs=prog[6].w[4] & ~15,
                                srs=prog[6].w[6] & ~1)
            resource = validate_feedback_trace(keep / "tmfeedback.jsonl", predict_producers=True)
            native = validate_native_trace(keep / "native-feedback.jsonl")
            coupled = validate_memory_coupled(keep)
            budget = analyze_credit_budget(keep)
            (keep / "credit-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
            assert budget["predicates"]["ok"], budget["predicates"]
            assert resource["command_driven_vadd"] and resource["command_driven_mxu_row_valid"]
            assert coupled["all_mm_issuers_and_arrivals_drained"]
            summary = {"mode": mode, "cycles": stats["cycles"], "structure": structure,
                       "resource": resource, "native": native, "coupled": coupled,
                       "credit_budget": {k: v for k, v in budget.items() if k != "native_read_ledger"}}
            (keep / "two-credit-summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
            results[mode] = {"dram": drams[0], "tmem": tmems[0], "summary": summary}
        except Exception:
            errors[mode] = traceback.format_exc()
            (keep / "failure.txt").write_text(errors[mode])
    if errors:
        pytest.fail("two-credit variant failure (evidence retained in " + str(run_dir) + "):\n" +
                    "\n".join(f"{mode}: {error[-1600:]}" for mode, error in errors.items()))
    try:
        assert np.array_equal(results["one_credit"]["dram"], results["two_credit"]["dram"])
        assert np.array_equal(results["one_credit"]["tmem"], results["two_credit"]["tmem"])
        assert np.array_equal(results["one_credit"]["tmem"][OUT:OUT + N],
                              results["two_credit"]["tmem"][OUT1:OUT1 + N])
    except Exception:
        (run_dir / "comparison-failure.txt").write_text(traceback.format_exc())
        raise
    budgets = {mode: result["summary"]["credit_budget"] for mode, result in results.items()}
    traffic_equal = all(
        [o[key] for o in budgets["one_credit"]["owners"]] ==
        [o[key] for o in budgets["two_credit"]["owners"]]
        for key in ("native_weight_payload_bytes", "native_scale_payload_bytes",
                    "weight_issue_bytes", "scale_issue_bytes"))
    comparison = {
        "one_credit_cycles": results["one_credit"]["summary"]["cycles"],
        "two_credit_cycles": results["two_credit"]["summary"]["cycles"],
        "speedup_asserted": False, "same_initial_dram": True,
        "same_final_dram_tmem": True, "same_owner_weight_scale_traffic": traffic_equal,
        "cycles_saved": results["one_credit"]["summary"]["cycles"] - results["two_credit"]["summary"]["cycles"],
        "calibrated": False, "tok_s": None, "evidence": str(run_dir)}
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    assert traffic_equal, "scheduling A/B changed weight/scale traffic"
