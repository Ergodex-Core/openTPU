"""tools/hwloop: trace extraction, the VisualSim model generator and its static validation, the
staleness fingerprint, the single-seat runner's refusals, the comparison report and the
nightly queue. No Verilator, no VisualSim process, no license: everything runs on fixtures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.hwloop import nightly, report, trace, vslib, vsmodel  # noqa: E402

PERF_LOG = """waits: slice 0: 276 instructions, 873 visits, 0 loops widened, 0 unreached
layers=4 pos=9 bw=100% lat=60: 3228204 cycles (244s sim), core-port roofline 3207980 cycles (port B 3207980 chunks, port A 3200096, MXU 3199056), efficiency 99.4%
DRAM read / write beats per channel: 812722 / 2309, 812144 / 2051, 812171 / 2049, 812717 / 2307, 812168 / 2049, 812809 / 2306, 812726 / 2310, 812179 / 2051; dram=off
MXU starved (chunk FIFO empty while streaming, the card's MXU_STARVE) 18 cycles = 0.0%; blocked (chunks but no MAC) 0.2%; MAC 99.1%; DRAM read beats/cycle/channel 1.007
  ch0: port A reads 648, DDR3 row opens 0, read-modify-writes 0 (port A / QST 0)
  ch1: port A reads 380, DDR3 row opens 0, read-modify-writes 0 (port A / QST 0)
useful bytes 423415280 -> 3307932 cycles at 128 B/cycle x 100% (the core's port): efficiency 102.5%
at 200 MHz: 16.14 ms/token, 62.0 tok/s (simulated cycles, no host time)
gemma-3-1b-it: 3228204 cycles, roofline 3207980 (99.4% of roofline)
  slice 0 unit busy: DMA 1%  MXU 100%  QUANT 1%  VPU 2%
  slice 0 DRAM port B busy 99% (MXU 3199056, DMA 8924)
phases: cycles, share, useful bytes -> cycles at 100% (their roofline), VPU busy, MXU busy
  other          2368742  73.4%   312486912 B ->   2441304 (103.1%)  VPU       552  MXU   2368975
  MLP             753084  23.3%    98574336 B ->    770112 (102.3%)  VPU     52939  MXU    757328
  attention       106376   3.3%    12354032 B ->     96516 ( 90.7%)  VPU     13534  MXU    108831
MXU prologue 1221 epilogue 1081 gaps 18370
"""


def _run_dir(tmp_path: Path, with_trace=True) -> Path:
    d = tmp_path / "run"
    d.mkdir(parents=True)
    (d / "perf.log").write_text(PERF_LOG)
    (d / "run_receipt.json").write_text(json.dumps({"tree": str(tmp_path / "tree"), "env": {"OTPU_NCH": "8", "OTPU_TARGET": "f2-hbm"},
                                                    "shell": "cd tree && python tools/perf_qwen.py ..."}))
    if with_trace:
        lines = []
        cyc = 9
        for i in range(4000):
            lines.append(f"{cyc} {i % 8} {1 if i % 500 == 499 else 0} {727392 + i}")
            cyc += 1 if i % 8 else 2
        (d / "nat_trace.txt").write_text("\n".join(lines) + "\n")
    return d


def _tree(tmp_path: Path) -> Path:
    t = tmp_path / "tree"
    (t / "rtl" / "mxu").mkdir(parents=True)
    (t / "sim" / "verilator").mkdir(parents=True)
    (t / "opentpu").mkdir()
    (t / "tools").mkdir()
    (t / "rtl" / "mxu" / "otpu_mxu.sv").write_text("module otpu_mxu; endmodule\n")
    (t / "sim" / "verilator" / "otpu_native_mem.sv").write_text("module otpu_native_mem; endmodule\n")
    (t / "opentpu" / "rtlsim.py").write_text("# rtlsim\n")
    (t / "opentpu" / "profile.py").write_text("# profile\n")
    (t / "tools" / "perf_qwen.py").write_text("# perf\n")
    return t


def test_parse_perf_log_numbers():
    s = trace.parse_perf_log(PERF_LOG)
    assert s["cycles"] == 3228204 and s["roofline_cycles"] == 3207980 and s["portb_chunks"] == 3207980
    assert len(s["channels"]) == 8 and s["channels"][0] == {"reads": 812722, "writes": 2309, "porta_reads": 648, "row_opens": 0, "rmw": 0}
    assert s["mxu_starved_cycles"] == 18 and s["mxu_blocked_pct"] == 0.2 and s["core_mhz"] == 200.0
    assert s["unit_busy_pct"]["MXU"] == 100 and s["portb_busy_pct"] == 99
    assert set(s["phases"]) == {"other", "MLP", "attention"} and s["phases"]["MLP"]["bytes"] == 98574336
    assert s["mxu_gaps"]["gaps"] == 18370


def test_parse_perf_log_wide_port_line():
    s = trace.parse_perf_log("layers=2 pos=9 bw=100% lat=60: 1643595 cycles (1782s sim), core-port roofline 1536296 cycles "
                             "(port B 2787778 chunks / 2 a cycle, port A 1390384, MXU 1536296), efficiency 93.5%\n"
                             "wide port B reads: 1242880 of 2485760 chunks, 87619 in two parts (a pair twice)\n")
    assert s["cycles"] == 1643595 and s["roofline_cycles"] == 1536296 and s["portb_chunks"] == 2787778 and s["mxu_cycles"] == 1536296
    assert s["wide_reads_line"].startswith("wide port B reads: 1242880")


def test_extract_writes_trace_and_windows(tmp_path):
    d = _run_dir(tmp_path)
    t = trace.extract(d, window=256, nch=8, lat=60)
    assert (d / "trace.json").is_file() and (d / "mem_windows.csv").is_file()
    assert t["label"] == vslib.MEASURED
    der = t["derived"]
    assert der["cycles"] == 3228204 and der["dram_bytes"] == 64 * sum(c["reads"] + c["writes"] for c in t["summary"]["channels"])
    assert abs(der["device_tok_s"] - 200e6 / 3228204) < 1e-6
    mt = t["memory_trace"]
    assert mt["reads"] + mt["writes"] == 4000 and mt["writes"] == 8 and mt["channels"] == 8
    assert 1 <= mt["max_outstanding_reads_est"] <= 60
    rows = (d / "mem_windows.csv").read_text().splitlines()
    assert rows[0].startswith("window,cycle_start,rd_ch0") and len(rows) == mt["csv_rows"] + 1


def test_extract_without_trace_file(tmp_path):
    d = _run_dir(tmp_path, with_trace=False)
    t = trace.extract(d, nch=8, lat=60)
    assert "memory_trace" not in t and t["derived"]["cycles"] == 3228204


@pytest.mark.parametrize("profile", ["f2-hbm-nch8-rtl", "f2-hbm-burst200", "f2-ddr4-100mhz", "ypcb-ddr3-1066", "asic-1ghz-hbm", "asic-1ghz-lpddr5"])
def test_profiles_build_static_models(tmp_path, profile):
    d = _run_dir(tmp_path)
    tree = _tree(tmp_path)
    t = trace.extract(d, nch=8, lat=60)
    prof = vsmodel.load_profile(profile)
    for k in ("name", "design", "kind", "label", "clocks", "memory", "pin"):
        assert k in prof, k
    assert {"core_mhz", "controller_mhz", "memory_axi_mhz"} <= set(prof["clocks"])
    assert "uncertain" in prof["memory"] and "refresh" in prof["memory"]
    fp = vslib.fingerprint_tree(tree, trace.RTL_GLOBS)
    m = vsmodel.build(t, prof, d / "vs" / profile, name="T_" + profile.replace("-", "_"), rtl_fingerprint=fp)
    assert m["static_validation"] == "STATIC_PASS", (d / "vs" / profile / "static-validation.json").read_text()
    assert m["label"] == vslib.UNVALIDATED and m["status"] == "PREPARED_NOT_EXECUTED_IN_VISUALSIM"
    xml = (d / "vs" / profile / "model.xml").read_text()
    nch = prof["memory"]["channels_per_slice"]
    if str(prof["memory"]["type"]).startswith("HBM"):
        assert m["memory_backend"] == "HBM_Module" and f"HBM_Native.Channel_{nch - 1}_IN" in xml
    else:
        assert m["memory_backend"] == "Memory_Controller+HW_DRAM" and f'name="DRAM_{nch - 1}"' in xml
    assert "Credits_Per_Channel" in dict(m["parameters"]) and len(m["phases"]) == 3
    exp = json.loads((d / "vs" / profile / "expected-counters.json").read_text())
    assert exp["requests_issued"] == sum(p["read_requests"] + p["write_requests"] for p in m["phases"])
    assert exp["read_bytes"] == sum(p["read_requests"] for p in m["phases"]) * int(dict(m["parameters"])["Request_Bytes"])
    assert "HBM refresh" in " ".join(m["not_modelled"]) or "DIMM" in " ".join(m["not_modelled"])


def test_static_validate_catches_bad_links(tmp_path):
    m = vslib.Model("Bad")
    m.param("Sim_Time", 0.001)
    e = m.add(vslib.vm_block("VM", "x", "LABEL: BEGIN\nGTO(Nowhere)\nEXIT", ["input"], ["out"]))
    m.link("VM.out", "Nobody.input")
    p = tmp_path / "bad.xml"
    p.write_text(m.to_xml())
    r = vslib.static_validate(p, root=tmp_path / "no-install")
    assert r["status"] == "STATIC_FAIL"
    assert any("unknown actor" in x for x in r["errors"]) and any("undefined GTO" in x for x in r["errors"])
    assert r["classes_checked_against_install"] is False


def test_staleness_detects_rtl_edits(tmp_path):
    tree = _tree(tmp_path)
    fp = vslib.fingerprint_tree(tree, trace.RTL_GLOBS)
    manifest = {"rtl_fingerprint": fp}
    assert vslib.check_stale(manifest, tree)["stale"] is False
    (tree / "rtl" / "mxu" / "otpu_mxu.sv").write_text("module otpu_mxu; // changed\nendmodule\n")
    st = vslib.check_stale(manifest, tree)
    assert st["stale"] is True and st["changed"] == ["rtl/mxu/otpu_mxu.sv"]
    (tree / "rtl" / "mxu" / "otpu_new.sv").write_text("module n; endmodule\n")
    assert vslib.check_stale(manifest, tree)["added"] == ["rtl/mxu/otpu_new.sv"]
    assert vslib.check_stale({}, tree)["stale"] is True


def test_run_batch_prepares_and_refuses_stale(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    tree = _tree(tmp_path)
    t = trace.extract(d, nch=8, lat=60)
    prof = vsmodel.load_profile("f2-hbm-nch8-rtl")
    fp = vslib.fingerprint_tree(tree, trace.RTL_GLOBS)
    md = d / "vs" / "p"
    m = vsmodel.build(t, prof, md, name="T", rtl_fingerprint=fp)
    manifest = json.loads((md / "manifest.json").read_text())
    monkeypatch.setattr(vslib, "active_visualsim", lambda: ["12345 java ... " + vslib.GUI_CLASS])
    monkeypatch.setattr(vslib, "license_manager_up", lambda ports=None: False)
    r = vslib.run_batch(md / "model.xml", md / "r1", execute=False, manifest=manifest, stale_root=tree)
    assert r["status"] == "PREPARED_NOT_EXECUTED" and (md / "r1" / "model.xml").is_file() and (md / "r1" / "command.sh").is_file()
    assert vslib.BATCH_CLASS in r["command"] and "--execute" in r["note"]
    snap = (md / "r1" / "model.xml").read_text()
    assert vslib.C_WRITER in snap and vslib.C_DISPLAY not in snap
    # the GUI holds the seat: an execute request is refused, nothing is run
    r = vslib.run_batch(md / "model.xml", md / "r2", execute=True, manifest=manifest, stale_root=tree)
    assert r["status"] == "REFUSED_SEAT_BUSY" and r["executed"] is False
    # a stale model is refused before the seat is even looked at
    (tree / "rtl" / "mxu" / "otpu_mxu.sv").write_text("module otpu_mxu; // v2\nendmodule\n")
    r = vslib.run_batch(md / "model.xml", md / "r3", execute=True, manifest=manifest, stale_root=tree)
    assert r["status"] == "REFUSED_STALE_MODEL" and r["staleness"]["changed"] == ["rtl/mxu/otpu_mxu.sv"]


def test_seat_lock_is_exclusive(tmp_path):
    lock = tmp_path / "seat.lock"
    with vslib.SeatLock(lock):
        with pytest.raises(RuntimeError):
            with vslib.SeatLock(lock, timeout_s=0.2):
                pass
    with vslib.SeatLock(lock):
        pass


def test_report_compares_and_labels(tmp_path):
    b = _run_dir(tmp_path / "b")
    c = _run_dir(tmp_path / "c", with_trace=False)
    trace.extract(b, nch=8, lat=60)
    (c / "perf.log").write_text(PERF_LOG.replace("3228204 cycles", "1643595 cycles").replace("at 200 MHz: 16.14 ms/token, 62.0 tok/s", "at 200 MHz: 8.22 ms/token, 121.7 tok/s"))
    trace.extract(c, nch=8, lat=60)
    prof = vsmodel.load_profile("f2-hbm-burst200")
    rep = report.compare(b, c, prof, "gemma3-1b-int8", tmp_path / "rep")
    assert rep["rows"][0]["label"] == vslib.MEASURED and rep["rows"][1]["cycles"] == 1643595
    assert abs(rep["delta"]["speedup"] - 3228204 / 1643595) < 1e-9
    assert rep["board"]["decode_tok_s_short"] == 22.3 and rep["board"]["label"] == vslib.MEASURED
    assert rep["calibration"]["whole_model_comparable"] is False and "rel_error_rtl_vs_board" not in rep["calibration"]
    assert any("VisualSim baseline" in w for w in rep["waiting"]) and any("nightly" in w for w in rep["waiting"])
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "unvalidated scenario" in md and "serving metric" in md and "waiting" in md
    assert (tmp_path / "rep" / "report.csv").read_text().splitlines()[1].startswith("baseline,measured,3228204")


def test_nightly_queue_and_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "QUEUE", tmp_path / "queue.json")
    e = nightly.add("portb2", "a9acb2b05da2b0ddcdbb97254b2776ca2cee2c87", "f2-hbm", "f2-hbm-burst200",
                    {"verilator": "pass"}, {"models": "gemma3-1b", "max_tokens": 128}, None)
    assert e["status"] == "QUEUED_AWAITING_APPROVAL"
    plan = nightly.plan(e)
    assert plan.startswith("# NOT ACTIVATED") and "boards/f2/build.py" in plan and "agfi-0d06cb48544a02106" in plan
    assert "aws ec2 create-fpga-image" in plan and "rollback" in plan
    q = nightly.load_queue()
    assert [c["name"] for c in q["candidates"]] == ["portb2"]


def test_cli_inventory_runs():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "hwloop" / "cli.py"), "inventory"], capture_output=True, text=True,
                         env={**os.environ, "VS_AR": "/nonexistent"})
    assert out.returncode == 0
    j = json.loads(out.stdout)
    assert j["installed"] is False and "f2-hbm-burst200" in j["profiles"]
