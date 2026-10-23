"""Pinned board qualification without opening hardware or changing ownership."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import nanbeige_board as cli
from opentpu.target import config, load


def _profile_info():
    profile = load(cli.PROFILE)
    expected = cli.validate_profile(profile)
    info = {"cfg": {k: expected[k] for k in ("platform", "channels", "channel_bytes")},
            "core_khz": expected["core_khz"], "build_id": expected["build_id"],
            "calibrated": True, "running": False,
            "caps": {"args": True, "gen": True, "mx": True}, "status": 0}
    info["cfg"]["hash"] = expected["arch_hash"]
    return profile, config(profile), info


def _stub_board_session(monkeypatch, states):
    from opentpu.host import board
    profile, cfg, healthy = _profile_info()
    events, states = [], iter(states)
    transport = SimpleNamespace(close=lambda: events.append("transport-close"))
    class Probe:
        def __init__(self, supplied, *, lock=True, calibrate=True):
            assert supplied is transport
            events.append(("probe", lock, calibrate))
            if calibrate:
                events.append("calibration-write")
            if lock:
                events.append("quiesce")
        def info(self):
            return dict(healthy, **next(states))
    class Backend:
        def __init__(self, actual, images, **kwargs):
            assert actual == cfg and kwargs["transport"] is transport
            events.append("backend-upload")
        def close(self):
            events.append("backend-close")
    monkeypatch.setattr(board, "open_transport", lambda dev: transport)
    monkeypatch.setattr(board, "Board", Probe)
    monkeypatch.setattr(board, "device_config", lambda info: cfg)
    monkeypatch.setattr(board, "BoardBackend", Backend)
    return profile, cfg, events


def test_exact_profile_and_all_geometry_fields_are_required():
    profile, cfg, info = _profile_info()
    cli.validate_board(info, cfg, profile)
    for key in ("core_khz", "build_id"):
        with pytest.raises(ValueError, match=key):
            cli.validate_board({**info, key: info[key] + 1}, cfg, profile)
    for key in ("platform", "channels", "channel_bytes", "hash"):
        with pytest.raises(ValueError, match=key):
            cli.validate_board({**info, "cfg": {**info["cfg"], key: info["cfg"][key] + 1}},
                               cfg, profile)
    for kw in ({"MCOLS": 4}, {"PORTB": 1}, {"ROWLANES": False}, {"ALINE": 32}):
        with pytest.raises(ValueError, match="architecture"):
            cli.validate_board(info, replace(cfg, **kw), profile)


@pytest.mark.parametrize("changed", [
    {"name": "unknown"}, {"variant": "p4rl8a64"},
    {"platform": {"board": "f2", "core_mhz": 200, "channels_per_slice": 8,
                  "channel_bytes": 536870912}},
])
def test_modified_profile_rejected_before_transport(monkeypatch, changed):
    from opentpu.host import board
    profile, _, _ = _profile_info()
    monkeypatch.setattr(board, "open_transport", lambda _: pytest.fail("opened transport"))
    with pytest.raises(ValueError, match="pinned"):
        cli.BoardSession("f2:0", replace(profile, **changed))


@pytest.mark.parametrize("state,error", [
    ({"running": True}, RuntimeError), ({"calibrated": False}, ValueError),
    ({"core_khz": 200000}, ValueError), ({"build_id": 0}, ValueError),
])
def test_recheck_before_upload_does_not_calibrate_or_quiesce(monkeypatch, state, error):
    profile, cfg, events = _stub_board_session(monkeypatch, [{}, state])
    session = cli.BoardSession("f2:0", profile)
    try:
        with pytest.raises(error):
            session.factory(cfg, [])
        assert events == [("probe", False, False), ("probe", False, False)]
    finally:
        session.close()


def test_healthy_recheck_and_cleanup(monkeypatch):
    profile, cfg, events = _stub_board_session(monkeypatch, [{}, {}])
    session = cli.BoardSession("f2:0", profile)
    session.factory(cfg, [])
    session.close()
    assert events == [("probe", False, False), ("probe", False, False),
                      "backend-upload", "backend-close", "transport-close"]
