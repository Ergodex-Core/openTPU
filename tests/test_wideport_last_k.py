"""MXU last-advance RTL/ISA checks on one F2 wide-port hardware configuration."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim, target
from opentpu.isasim import Machine
from test_quant_rtl import _put, _xrows
from test_wide_port_rtl import AL, D, OUT, SA, WA, X, _scales, _weights


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _case(fmt, mode, kbs, seed):
    """Four MMs: scaled, unit/RMAX, ACC/RMAX, unit/ACC/ASCALE."""
    cfg = Config(S=1, D=128, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, PAIR=True,
                 DSTEP=True, T2=True, PORTB=4, ROWLANES=True, ALINE=64,
                 DRAM_BYTES=1 << 20)
    assert D == cfg.D
    rng = np.random.default_rng(seed)
    kb_max, n, m = max(kbs), 11, 1
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    x = _xrows(rng, m, kb_max * D, D // 4)
    _put(dram, X, x)
    rs = _weights(rng, dram, fmt, n, kb_max)
    srs = _scales(rng, dram, fmt, n, kb_max, SA)
    _put(dram, AL, rng.uniform(-1.5, 1.5, m).astype(np.float32))
    pair, quad = mode != "plain", mode == "quad"
    common = dict(sa=WA, ssa=SA, n=n, rs=rs, ors=n + 1, m=m, ab=1, srs=srs,
                  wf=Q.mxu_wf(fmt), pair=pair, quad=quad)
    prog = [I.ld(X, 0, x.size), I.ld(AL, 0x7000, m),
            I.qact(0, m, 1, kb_max, kb_max * D, dup=pair and not quad, dup4=quad),
            I.mm(out=OUT, kb=kbs[0], **common),
            I.mm(out=OUT + 4096, kb=kbs[1], unit=True, rmax=True, **common),
            I.mm(out=OUT, kb=kbs[2], acc=True, rmax=True, **common),
            I.mm(out=OUT + 4096, kb=kbs[3], unit=True, acc=True, ascale=0x7000, **common),
            I.halt()]
    return cfg, prog, dram


def _run_case(monkeypatch, name, fmt, mode, kbs, seed):
    cfg, prog, dram = _case(fmt, mode, kbs, seed)
    board = {**target.sim_uarch("f2-hbm:p4rl8a64"), "TMEM_PMP": 0, "MXU_E": 4}
    assert board["TMEM_PMP"] == 0 and board["MXU_E"] == 4
    flags = rtlsim.VFLAGS.copy()
    flags[flags.index("-j") + 1] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setenv("OTPU_NCH", "8")
    monkeypatch.setenv("OTPU_NAXI", "0")
    monkeypatch.delenv("OTPU_SIM_PLUSARGS", raising=False)

    evidence = os.environ.get("WP_LASTK_EVIDENCE")
    keep = None
    plusargs = []
    words = I.assemble(prog).astype("<u4")
    if evidence:
        root = Path(evidence).resolve()
        root.mkdir(parents=True, exist_ok=True)
        keep = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=root))
        # Optional native feedback changes the scheduler, so it is explicit.
        if os.environ.get("WP_LASTK_NATIVE_FEEDBACK") == "1":
            plusargs = ["+nat_deterministic=1", f"+nat_feedback={keep / 'native-feedback.jsonl'}"]
        inputs = {"case": name, "format": fmt, "mode": mode, "kbs": list(kbs),
                  "target": "f2-hbm:p4rl8a64", "config": {"S": 1, "D": 128,
                  "MCOLS": 8, "PORTB": 4, "ROWLANES": True, "ALINE": 64,
                  "T2": True, "PAIR": True, "ACT_BLOCKS": 16, "ACT_ROWS": 16},
                  "uarch": board, "axi": True, "boot": True, "nch": 8, "naxi": 0,
                  "stall": 50, "seed": 7, "lat": 120, "max_cycles": 1_000_000,
                  "verilator_jobs": 4, "plusargs": plusargs,
                  "program_sha256": _sha(words.tobytes()), "dram_sha256": _sha(dram.tobytes()),
                  "instruction_count": len(prog)}
        (keep / "program.hex").write_text("".join(f"{int(w):08x}\n" for w in words))
        dram.tofile(keep / "dram-input.bin")
        original_build, original_run_sim = rtlsim.build, rtlsim.run_sim

        def record_build(top, sources, params=None):
            if top == "tb_top":
                inputs["parameters"] = dict(params or {})
                inputs["rtl_source_sha256"] = {
                    str(Path(p).resolve().relative_to(rtlsim.ROOT)): _sha(Path(p).read_bytes())
                    for p in sources}
                (keep / "inputs.json").write_text(json.dumps(inputs, indent=2, default=_json) + "\n")
            return original_build(top, sources, params)

        def record_sim_output(cmd, timeout=None):
            (keep / "sim-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            result = original_run_sim(cmd, timeout=timeout)
            (keep / "sim.stdout.log").write_text(result.stdout)
            (keep / "sim.stderr.log").write_text(result.stderr)
            return result

        monkeypatch.setattr(rtlsim, "build", record_build)
        monkeypatch.setattr(rtlsim, "run_sim", record_sim_output)

    isa = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, stats = rtlsim.run(
        cfg, [prog], [dram.copy()], axi=True, boot=True, stall=50, seed=7, lat=120,
        max_cycles=1_000_000, keep=keep, uarch=board, plusargs=plusargs)
    dram_ok = np.array_equal(drams[0], isa.slices[0].dram)
    tmem_ok = np.array_equal(tmems[0], isa.slices[0].tmem)
    if keep:
        np.save(keep / "dram-output.npy", drams[0])
        np.save(keep / "tmem-output.npy", tmems[0])
        outputs = {"dram_matches_isa": dram_ok, "tmem_matches_isa": tmem_ok,
                   "dram_sha256": _sha(drams[0].tobytes()),
                   "tmem_sha256": _sha(tmems[0].tobytes()), "stats": stats}
        (keep / "outputs.json").write_text(json.dumps(outputs, indent=2, default=_json) + "\n")
    assert dram_ok, "complete DRAM differs from ISA"
    assert tmem_ok, "complete TMEM differs from ISA"


# Eight directed cases plus one command-head transition case, all on the same F2 RTL build.
# PAIR odd/even KB, QUAD KB % 4 == 1/2/3, 4-bit and t2 weights, and plain mode.
@pytest.mark.parametrize("fmt,mode,kb", [
    ("int8", "pair", 1), ("int8", "pair", 6),
    ("int8", "quad", 5), ("int8", "quad", 6), ("int8", "quad", 7),
    ("int4", "quad", 7), ("t2", "quad", 5), ("int8", "plain", 9),
])
def test_last_advance_pair_quad_rtl(have_verilator, monkeypatch, fmt, mode, kb):
    _run_case(monkeypatch, f"{fmt}-{mode}-kb{kb}", fmt, mode, (kb,) * 4, 1000 + kb)


def test_last_advance_changes_with_back_to_back_commands(have_verilator, monkeypatch):
    """Consecutive QUAD heads use different thresholds without draining the pipeline."""
    _run_case(monkeypatch, "int8-quad-kb5-6-7-8", "int8", "quad", (5, 6, 7, 8), 7761)
