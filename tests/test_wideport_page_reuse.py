"""Bounded, real DRAM-page reuse across three row-lane MM instructions.

MM0 and MM1 use distinct weight/scale pages. DMA then patches one 64-byte
line of MM0's pages from preloaded TMEM scratch, and MM2 reads those pages
again. This checks a conservative completed-write handoff, not the earliest
possible page release. The coupled checker generates the DMA ST and native
write/invalidation timing; accepted starts remain external frontend inputs.
"""

from collections import deque
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np
import pytest

from opentpu import isa as I, rtlsim, target, waits as W
from opentpu.isasim import Machine
from test_wideport_two_credit import (
    D_OUT0, D_OUT1, N, OUT1, SB1, WB1, _programs, _records, _sha,
)
from test_wide_port_rtl import OUT, SB, WB
from tools.wideport_memory_feedback import read_trace, validate_native_trace
from tools.wideport_unit_feedback import validate_feedback_trace
from tools.wideport_native_memory import logical_beat
from tools.wideport_pipeline_ownership import validate_pipeline_ownership
from tools.wideport_page_reuse import validate_page_epochs
from tools.wideport_write_coupled import validate_write_coupled


OUT2 = OUT + 8192
D_OUT2 = 0xD0200
STAGE_W, STAGE_S = 0x20000, 0x21000
SCRATCH_W, SCRATCH_S = 0xE000, 0xE100
PATCH_BYTES = 64


def _fixture():
    cfg, original, dram = _programs()
    assert all(I.assemble(p).tobytes() == I.assemble(original["two_credit"]).tobytes()
               for p in original.values())
    p = original["two_credit"]
    mm0, mm1 = (x for x in p if x.op == I.MM)
    rs, srs = mm0.w[4] & ~15, mm0.w[6] & ~1
    assert (mm0.w[0], mm0.w[1], mm1.w[0], mm1.w[1]) == (WB, SB, WB1, SB1)
    assert (SCRATCH_W + 16 <= SCRATCH_S and SCRATCH_S + 16 < cfg.TMEM_WORDS
            and OUT2 + N < SCRATCH_W)
    assert STAGE_W + PATCH_BYTES < STAGE_S < WB
    assert D_OUT2 + 4 * N <= cfg.DRAM_BYTES

    weight = np.bitwise_xor(dram[WB:WB + PATCH_BYTES], np.uint8(0xFF)).copy()
    scale = dram[SB:SB + PATCH_BYTES].copy()
    sw = np.frombuffer(scale[:16].tobytes(), dtype="<f4").copy()
    assert np.isfinite(sw).all() and np.any(sw != 0)
    doubled = (sw * np.float32(2)).astype("<f4")
    assert np.isfinite(doubled).all() and not np.array_equal(sw, doubled)
    scale[:16] = doubled.view(np.uint8)
    assert not np.array_equal(weight, dram[WB:WB + PATCH_BYTES])
    assert not np.array_equal(scale, dram[SB:SB + PATCH_BYTES])
    dram[STAGE_W:STAGE_W + PATCH_BYTES] = weight
    dram[STAGE_S:STAGE_S + PATCH_BYTES] = scale

    mm2 = I.mm(sa=WB, ssa=SB, out=OUT2, n=N, kb=9, rs=rs,
               ors=N + 1, m=1, ab=0, srs=srs, wf=I.W8, rl=4)
    # The initial V_ADD is unchanged. Its BAR also waits for the two scratch
    # loads. The two MMs and their NOP remain contiguous as in the credit test.
    program = (p[:4] + [I.ld(STAGE_W, SCRATCH_W, 16),
                        I.ld(STAGE_S, SCRATCH_S, 16)] + p[4:9] +
               [I.bar(), I.st(WB, SCRATCH_W, 16), I.st(SB, SCRATCH_S, 16), mm2] +
               p[9:-1] + [I.st(D_OUT2, OUT2, N), I.halt()])
    assert [x.op for x in program].count(I.MM) == 3
    assert [x.op for x in program].count(I.ST) == 5
    assert [x.op for x in program].count(I.LD) == 5
    assert [x.op for x in program].count(I.VOP) == 1
    assert all(OUT2 + N <= scratch or scratch + 16 <= OUT2
               for scratch in (SCRATCH_W, SCRATCH_S))
    # All MM instructions use identical activation and all pages except the
    # deliberate epoch-2 lines retain the same source bytes.
    assert mm2.w[0] == mm0.w[0] and mm2.w[1] == mm0.w[1]
    assert mm2.w[4:] == mm0.w[4:]
    return cfg, program, dram, weight, scale


def _ownership_evidence(events, native_header, native_events, program, mode):
    mm_starts = [e["cycle"] for e in events if e["start"] & 2]
    mm_done = [e["cycle"] for e in events if e["mxu"]["c_done"]]
    dma_starts = [e["cycle"] for e in events if e["start"] & 1]
    dma_done = [e["cycle"] for e in events if e["done"] & 1]
    assert len(mm_starts) == len(mm_done) == 3, (mm_starts, mm_done)
    assert len(dma_starts) == len(dma_done) == 10, (dma_starts, dma_done)
    assert all(a < b for a, b in zip(dma_starts, dma_done))
    assert all(a < b for a, b in zip(mm_starts, mm_done))
    # Five LDs precede the two patch STs; final three STs follow MM2's BAR.
    assert dma_done[5] < dma_starts[6] < dma_done[6] < mm_starts[2], (
        dma_done[5:7], mm_starts)
    assert mm_done[2] < dma_starts[7]
    if mode == "two_credit":
        assert mm_starts[1] < mm_done[0]
    else:
        assert mm_starts[1] > mm_done[0]

    # The native trace is a second boundary: a real accepted write to each
    # reused page must occur within its ST interval, before MM2 starts.
    writes = {WB: [], SB: []}
    read_queues = [deque() for _ in range(native_header["channels"])]
    first_reuse_write = None
    last_old_page_response = None
    for e in native_events:
        for channel, row in enumerate(e["channels"]):
            if row["rv"]:
                assert read_queues[channel], (e["cycle"], channel, "orphan response")
                old_page = read_queues[channel].popleft()
                if old_page:
                    last_old_page_response = e["cycle"]
            if not (row["cv"] and row["cr"]):
                continue
            byte = 64 * logical_beat(row["addr"], channel,
                                     chash=bool(native_header["chash"]))
            reused_page = WB <= byte < WB1 or SB <= byte < SB1
            if not row["we"]:
                if first_reuse_write is not None and e["cycle"] < mm_starts[2]:
                    assert not reused_page, (e["cycle"], byte, "old-page read after overwrite")
                read_queues[channel].append(reused_page and first_reuse_write is None)
            elif byte in writes:
                if first_reuse_write is None:
                    first_reuse_write = e["cycle"]
                    assert not any(any(q) for q in read_queues), (
                        e["cycle"], "old-page read still outstanding at overwrite")
                writes[byte].append(e["cycle"])
    assert writes[WB] and writes[SB], writes
    assert last_old_page_response is not None and last_old_page_response <= first_reuse_write
    assert all(dma_starts[5] <= c <= dma_done[5] for c in writes[WB]), writes[WB]
    assert all(dma_starts[6] <= c <= dma_done[6] for c in writes[SB]), writes[SB]

    # The added sequencer trace, when present, must describe the exact saved
    # straight-line program. DMA and MXU start/done ordering above is the
    # timing gate, independent of trace-internal slot encoding.
    dispatched = [(e["seq"]["pc"], e["seq"]["op"])
                  for e in events if e.get("seq", {}).get("dispatch")]
    if dispatched:
        assert all(0 <= pc < len(program) and op == program[pc].op
                   for pc, op in dispatched), dispatched
        required = {pc for pc, ins in enumerate(program) if ins.op in (I.MM, I.ST)}
        assert required <= {pc for pc, _ in dispatched}, (required, dispatched)
    return {"mm_start": mm_starts, "mm_c_done": mm_done,
            "dma_start": dma_starts, "dma_done": dma_done,
            "epoch2_native_write_cycles": {hex(k): v for k, v in writes.items()},
            "last_old_page_response": last_old_page_response,
            "sequencer_dispatch_checked": bool(dispatched),
            "conservative_completed_write_handoff": True,
            "earliest_page_release_proven": False,
            "independent_adapter_write_model": True}


def test_wideport_two_page_epoch_reuse(have_verilator, monkeypatch, tmp_path):
    cfg, program, dram, weight, scale = _fixture()
    stall=int(os.environ.get("WP_INPUT_STALL","50"))
    latency=int(os.environ.get("WP_INPUT_LAT","120"))
    board = {**target.sim_uarch("f2-hbm:p4rl8a64"), "TMEM_PMP": 0}
    assert board["FIFO_DEPTH"] == 1024
    root = Path(os.environ.get("WP_PAGE_REUSE_EVIDENCE", tmp_path)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="page-reuse-", dir=root))
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
            (current["directory"] / "feedback-inputs.json").write_text(
                json.dumps(manifest, indent=2) + "\n")
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
    for mode, queue_cap in (("one_credit", 1), ("two_credit", 2)):
        keep = run_dir / mode
        keep.mkdir()
        current["directory"] = keep
        words = I.assemble(program).astype("<u4")
        waits = W.wait_counts(program, cfg.D, cfg.S).astype("<u8")
        plusargs = [f"+hz_uq=0:{queue_cap}:0:0:0", "+nat_deterministic=1", "+nat_feedback_payload=1",
                    f"+nat_feedback={keep / 'native-feedback.jsonl'}",
                    f"+adapter_feedback={keep / 'adapter-feedback.jsonl'}",
                    f"+tmfeedback={keep / 'tmfeedback.jsonl'}", "+tmfeedback_limit=32768"]
        if os.environ.get("WP_NATIVE_WRITE_DELAY"):
            plusargs.append("+nat_wdl=" + str(int(os.environ["WP_NATIVE_WRITE_DELAY"])))
        if os.environ.get("WP_NATIVE_CQ"):
            plusargs.append("+nat_cq=" + str(int(os.environ["WP_NATIVE_CQ"])))
        manifest = {"variant": mode, "mxu_window_cap": queue_cap,
                    "pmp": 0, "nch": 8, "requested_uarch": board,
                    "program_sha256": _sha(words.tobytes()),
                    "waits_sha256": _sha(waits.tobytes()),
                    "dram_sha256": _sha(dram.tobytes()),
                    "instruction_count": len(program),
                    "loader_instruction_bytes": words.nbytes,
                    "loader_wait_bytes": waits.nbytes,
                    "stall": stall, "seed": 7, "lat": latency, "plusargs": plusargs,
                    "epoch2": {"weight_page": WB, "scale_page": SB,
                               "weight_stage": STAGE_W, "scale_stage": STAGE_S,
                               "weight_scratch": SCRATCH_W, "scale_scratch": SCRATCH_S,
                               "weight_patch_sha256": _sha(weight.tobytes()),
                               "scale_patch_sha256": _sha(scale.tobytes()),
                               "patch_bytes_each": PATCH_BYTES,
                               "outputs": [OUT, OUT1, OUT2],
                               "output_dram": [D_OUT0, D_OUT1, D_OUT2]}}
        current["manifest"] = manifest
        (keep / "feedback-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (keep / "waits.hex").write_text("".join(f"{int(w):016x}\n" for w in waits))
        try:
            isa = Machine(cfg, [program], [dram.copy()]).run()
            expected_tmem, expected_dram = isa.slices[0].tmem, isa.slices[0].dram
            assert np.array_equal(expected_tmem[OUT:OUT + N], expected_tmem[OUT1:OUT1 + N])
            assert np.any(expected_tmem[OUT:OUT + N] != expected_tmem[OUT2:OUT2 + N]), (
                "epoch-2 canaries did not change a result; fixture is not sensitive")
            assert np.array_equal(expected_dram[WB:WB + PATCH_BYTES], weight)
            assert np.array_equal(expected_dram[SB:SB + PATCH_BYTES], scale)
            drams, tmems, stats = rtlsim.run(
                cfg, [program], [dram.copy()], axi=True, boot=True,
                stall=stall, seed=7, lat=latency, max_cycles=1_000_000,
                keep=keep, uarch=board, plusargs=plusargs)
            assert np.array_equal(drams[0], expected_dram), "DRAM differs from ISA"
            assert np.array_equal(tmems[0], expected_tmem), "TMEM differs from ISA"
            assert np.array_equal(tmems[0][OUT:OUT + N], tmems[0][OUT1:OUT1 + N])
            assert np.any(tmems[0][OUT:OUT + N] != tmems[0][OUT2:OUT2 + N])
            header, events = _records(keep / "tmfeedback.jsonl")
            assert (header["schema"], header["pmp"], header["side_mask"],
                    header["words"], header["lanes"], header["vpu_se"]) == (
                    1, 0, 0xF0, cfg.TMEM_WORDS, 8, 1)
            assert len(events) < 32768
            native_header, native_events = read_trace(keep / "native-feedback.jsonl")
            assert native_header["channels"] == 8 and native_header["deterministic"]
            structure = _ownership_evidence(events, native_header, native_events, program, mode)
            resource = validate_feedback_trace(keep / "tmfeedback.jsonl", predict_producers=True)
            native = validate_native_trace(keep / "native-feedback.jsonl")
            pipeline = validate_pipeline_ownership(keep)
            epochs = validate_page_epochs(keep)
            coupled_writes = validate_write_coupled(keep)
            assert coupled_writes["new_epoch_read_ownership_closed"]
            assert resource["command_driven_vadd"] and resource["command_driven_mxu_row_valid"]
            assert native["write_commands"] >= 2
            summary = {"mode": mode, "cycles": stats["cycles"], "structure": structure,
                       "resource": resource, "native": native,
                       "pipeline_ownership": pipeline, "page_epochs": epochs,
                       "write_coupled": coupled_writes,
                       "isa_exact_tmem_dram": True,
                       "whole_program_coupled_predictor": False,
                       "initial_dram_sha256": _sha(dram.tobytes()),
                       "final_dram_sha256": _sha(drams[0].tobytes()),
                       "final_tmem_sha256": _sha(tmems[0].tobytes())}
            (keep / "page-reuse-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            results[mode] = {"dram": drams[0], "tmem": tmems[0], "summary": summary}
        except Exception:
            errors[mode] = traceback.format_exc()
            (keep / "failure.txt").write_text(errors[mode])
    if errors:
        pytest.fail("page-reuse variant failure (evidence retained in " + str(run_dir) + "):\n" +
                    "\n".join(f"{mode}: {error[-1600:]}" for mode, error in errors.items()))
    try:
        assert np.array_equal(results["one_credit"]["dram"], results["two_credit"]["dram"])
        assert np.array_equal(results["one_credit"]["tmem"], results["two_credit"]["tmem"])
    except Exception:
        (run_dir / "comparison-failure.txt").write_text(traceback.format_exc())
        raise
    (run_dir / "comparison.json").write_text(json.dumps({
        "same_program": True, "same_initial_dram": True,
        "same_final_dram_tmem": True,
        "one_credit_cycles": results["one_credit"]["summary"]["cycles"],
        "two_credit_cycles": results["two_credit"]["summary"]["cycles"],
        "speedup_asserted": False, "calibrated": False, "tok_s": None,
        "evidence": str(run_dir)}, indent=2) + "\n")
