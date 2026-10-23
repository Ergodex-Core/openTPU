"""F2 bring-up refuses unsafe preconditions and stops at the first failed stage."""
import json

import pytest

from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport
from tools import f2_smoke as smoke


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setenv("OTPU_TARGET", "f2")
    for name in ("MCOLS", "LANES", "PAIR", "DSTEP", "STREAM"):
        monkeypatch.delenv("OTPU_" + name, raising=False)
    t = FakeTransport(regmap=5, devname=None, pair=True, dstep=True, stream=True, seq_wc=True,
                      D=128, MCOLS=4, LANES=8, act_rows=16)
    t.bdf, t.channel_bytes = "0000:34:00.0", 1 << 31
    t.closed, t.reg_writes, t.mem_writes, t.opens = False, [], [], []
    write = t.reg_write

    def reg_write(off, val):
        t.reg_writes.append((off, val))
        return write(off, val)

    def open_transport(dev, dma):
        t.opens.append((dev, dma))
        return t

    monkeypatch.setattr(t, "reg_write", reg_write)
    monkeypatch.setattr(t, "mem_write", lambda *args: t.mem_writes.append(args))
    t.close = lambda: setattr(t, "closed", True)
    monkeypatch.setattr(smoke, "F2Transport", open_transport)
    return t


def test_probe_reads_only_and_records_identity(device, tmp_path):
    report = tmp_path / "probe.json"
    assert smoke.main(["--probe", "--expect-build", "1234abcd", "--json", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["passed"] and result["identity"]["cfg"]["platform"] == 3
    assert device.opens == [("f2:0", False)]
    assert not device.reg_writes and not device.mem_writes
    assert device.closed


def test_wrong_build_refuses_all_writes(device, tmp_path):
    report = tmp_path / "wrong.json"
    assert smoke.main(["--expect-build", "76543210", "--json", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["stages"][-1]["name"] == "identity"
    assert "build identity" in result["stages"][-1]["error"]
    assert not device.reg_writes and not device.mem_writes
    assert device.closed


def test_other_platform_refuses_all_writes(device, monkeypatch):
    read = device.reg_read
    monkeypatch.setattr(device, "reg_read", lambda off: 2 if off == R.R_CFG_PLATFORM else read(off))
    assert smoke.main([]) == 1
    assert not device.reg_writes and not device.mem_writes


def test_active_program_is_not_stopped(device):
    device.run_s = 1000.0
    device.reg_write(R.R_CTRL, R.CTRL_RUN)
    device.reg_writes.clear()
    assert smoke.main([]) == 1
    assert not device.reg_writes and not device.mem_writes
    assert device.reg_read(R.R_STATUS) & R.ST_RUN


def test_uncalibrated_ddr_never_scrubs_or_transfers(device, tmp_path):
    device.calib = [True, False]
    report = tmp_path / "calib.json"
    assert smoke.main(["--calib-timeout", "0.001", "--json", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["stages"][-1]["name"] == "calibration"
    assert not device.mem_writes and device.closed


def test_kernel_failure_prevents_followup_and_model_checks(device, monkeypatch, tmp_path):
    for name in ("address_lines", "channel_patterns", "partial_writes", "pattern_test"):
        monkeypatch.setattr(smoke.checks, name, lambda *a, **kw: (True, "ok"))
    demo, later = [], []
    monkeypatch.setattr(smoke.checks, "run_demo", lambda *a: demo.append(a) or
                        (False, "one stale read", {"cycles": 100}))
    monkeypatch.setattr(smoke.checks, "model_check", lambda *a: later.append(a))
    report = tmp_path / "kernel.json"
    assert smoke.main(["--model", "qwen3", "--json", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["stages"][-1]["name"] == "kernel" and not result["passed"]
    assert len(demo) == 1 and not later and device.closed
    assert device.reg_read(R.R_SCRATCH) == 0   # BAR0 test restored the marker


def test_axi_error_is_not_mistaken_for_an_ecc_pass(device, monkeypatch):
    read = device.reg_read
    monkeypatch.setattr(device, "reg_read", lambda off: read(off) | R.ST_AXI_ERR
                        if off == R.R_STATUS else read(off))
    assert smoke.main(["--probe"]) == 1
    assert not device.reg_writes and not device.mem_writes
