"""Structural TMEM cases with known bank collisions, not fitted elapsed penalties."""
import pytest

from tools.wideport_tmem_resource import TmemReplayConfig, TmemReplayResource, WriteLandingPipeline


def step(resource, requests=None, writes=(), untaken=()):
    requests = requests or {}
    return resource.step([requests.get(p, []) for p in range(8)],
                         [bool(requests.get(p)) and p not in untaken for p in range(8)], writes)


def test_input_write_lands_two_core_cycles_later():
    pipeline = WriteLandingPipeline()
    assert pipeline.step([16]) == ()
    assert pipeline.step([32]) == ()
    assert pipeline.step([]) == (16,)
    assert pipeline.step([]) == (32,)


@pytest.mark.parametrize("pmp, lost", [(0, 1 << 4), (1, 0), (2, 0)])
def test_write_occupancy_exposes_one_replay_cycle_with_fixed_side_assignment(pmp, lost):
    r = TmemReplayResource(TmemReplayConfig(pmp=pmp, side_mask=1 << 4))
    event = step(r, {4: [0]}, writes=[16])  # same bank0/subbank0, different word
    assert event["lost"] == lost
    assert event["frozen"] == 0
    replay = step(r)
    assert replay["frozen"] == lost
    assert replay["lost"] == 0
    assert step(r)["frozen"] == 0


@pytest.mark.parametrize("written", [0, 8, 1])
def test_same_word_or_different_subbank_or_bank_does_not_lose(written):
    r = TmemReplayResource(TmemReplayConfig(pmp=0))
    assert step(r, {4: [0]}, writes=[written])["lost"] == 0


def test_side_assignment_change_is_independent_of_write_occupancy():
    # MXU and QUANT share B under PMP0, but use different sides under PMP1.
    assert step(TmemReplayResource(TmemReplayConfig(pmp=0)), {0: [0], 2: [16]})["lost"] == 1 << 2
    assert step(TmemReplayResource(TmemReplayConfig(pmp=1)), {0: [0], 2: [16]})["lost"] == 0


def test_read_sharing_and_distinct_subbanks_need_no_replay():
    for second in (0, 8):
        r = TmemReplayResource(TmemReplayConfig(pmp=1))
        assert step(r, {0: [0], 4: [second]})["lost"] == 0


def test_replay_age_prevents_newer_higher_rank_port_from_overtaking():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    first = step(r, {1: [0], 0: [16], 6: [32]})
    assert first["lost"] == (1 << 0) | (1 << 6)
    second = step(r, {4: [48]})
    assert second["frozen"] == first["lost"]
    assert second["lost"] == (1 << 4) | (1 << 6)
    third = step(r)
    assert third["lost"] == 1 << 4  # older Q2 wins over the newer, higher-rank VPU B
    assert step(r)["lost"] == 0
    assert step(r)["frozen"] == 0


def test_pending_address_is_preserved_and_overrides_fresh_request():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    assert step(r, {1: [0], 0: [16]})["lost"] == 1
    event = step(r, {0: [8], 1: [32]})
    assert event["lost"] == 1  # pending word16 still conflicts, fresh word8 would not


def test_one_conflicting_bank_replays_entire_eight_lane_request():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    first = step(r, {0: [16, 1, 2, 3, 4, 5, 6, 7], 4: list(range(8))})
    assert first["lost"] == 1
    # A different bank can lose on retry: replay retains all banks, not just
    # the bank0 word that originally lost.
    second = step(r, {1: [17]})
    assert second["lost"] == second["frozen"] == 1
    assert step(r)["lost"] == 0


def test_guest_uses_host_asker_and_never_gets_independent_replay():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    event = step(r, {5: [0], 0: [16]})
    assert event["lost"] == 1  # QUANT3 borrowed DMA priority and beats MXU
    assert event["blocked"] & (1 << 5) == 0
    assert step(r)["frozen"] == 1


def test_request_can_take_bank_even_without_unit_grant():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    event = step(r, {0: [0], 4: [16]}, untaken=(4,))
    assert event["lost"] == 1  # RAM addresses come from requests, not unit takes


def test_nonreplay_port_reports_blocking_but_does_not_latch_replay():
    r = TmemReplayResource(TmemReplayConfig(pmp=0, side_mask=1 << 1))
    event = step(r, {1: [0]}, writes=[16])
    assert event["blocked"] == 1 << 1
    assert event["lost"] == 0
    assert step(r)["frozen"] == 0


def test_replay_word_overwrite_is_rejected_like_rtl_premise():
    r = TmemReplayResource(TmemReplayConfig(pmp=1))
    step(r, {1: [0], 0: [16]})
    with pytest.raises(ValueError, match="awaiting replay"):
        step(r, writes=[16])


@pytest.mark.parametrize("options", [{"nsb": 4}, {"pmp": 3}, {"lanes": 4}, {"side_mask": 256}])
def test_unsupported_configuration_fails_closed(options):
    with pytest.raises(ValueError):
        TmemReplayConfig(**options)


def test_invalid_lane_and_write_collisions_are_rejected():
    r = TmemReplayResource()
    with pytest.raises(ValueError, match="two different words"):
        step(r, {0: [0, 16]})
    with pytest.raises(ValueError, match="two different words"):
        step(r, writes=[0, 16])


@pytest.mark.parametrize("pmp,side_override", [(0, -1), (1, -1), (1, 0xF0)])
def test_resource_matches_cycle_level_rtl(have_verilator, monkeypatch, pmp, side_override):
    """Small bank-resource experiment; no full workload or FPGA build is involved."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    from opentpu import rtlsim

    flags = list(rtlsim.VFLAGS)
    flags[2] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    params = {"PMP": pmp, "SIDE_OVERRIDE": side_override}
    sources = [rtlsim.RTL / "mem/otpu_tmem.sv", rtlsim.RTL / "tech/generic/otpu_tech_ram.sv",
               rtlsim.RTL / "exp/tmem_pump/otpu_tmem_pump.sv", rtlsim.TB / "tb_tmem_replay_trace.sv"]
    exe = rtlsim.build("tb_tmem_replay_trace", sources, params)
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    evidence = Path(os.environ["TMEM_RESOURCE_EVIDENCE"]) if os.environ.get("TMEM_RESOURCE_EVIDENCE") else None
    name = f"pmp{pmp}-side{side_override}"
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f"{name}.log").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    events = [json.loads(line.split(":", 1)[1]) for line in run.stdout.splitlines() if line.startswith("TMEMTRACE:")]
    assert len(events) == 256
    cfg = TmemReplayConfig(pmp=pmp, words=256, side_mask=None if side_override < 0 else side_override)
    resource, pipeline = TmemReplayResource(cfg), WriteLandingPipeline()
    lost_counts, freeze_counts = [0] * 8, [0] * 8
    for cycle, event in enumerate(events):
        assert event["cycle"] == cycle
        assert event["side_mask"] == sum(s << p for p, s in enumerate(cfg.sides))
        assert sorted(pipeline.step(event["issued_writes"])) == sorted(event["writes"]), cycle
        modeled = resource.step(event["reads"], event["taken"], event["writes"])
        for got, want in (("lost", "lost"), ("frozen", "frozen"), ("blocked", "blk_x"), ("self_blocked", "blk_s")):
            assert modeled[got] == event[want], (cycle, got, modeled, event)
        for port in range(8):
            lost_counts[port] += bool(event["lost"] & (1 << port))
            freeze_counts[port] += bool(event["frozen"] & (1 << port))
    assert all(event["lost"] == 0 for event in events[:40])  # sharing and different-subbank windows
    assert all(event["lost"] == 0 for event in events[178:184])  # same-word landing writes
    assert any(event["lost"] for event in events[40:132])  # aged read/read contention
    assert any(event["reads"][5] or event["reads"][7] for event in events[132:176])
    assert any(len(event["reads"][0]) == 8 for event in events[232:248])
    write_window_losses = sum(event["lost"].bit_count() for event in events[192:232])
    assert (write_window_losses > 0) == (pmp == 0)
    assert events[-1]["lost"] == events[-1]["frozen"] == 0
    if evidence:
        result = {"cycles_checked": len(events), "mismatches": 0, "pmp": pmp,
                  "side_override": side_override, "side_mask": events[0]["side_mask"],
                  "write_window_replay_port_cycles": write_window_losses,
                  "replay_loss_port_cycles": lost_counts, "frozen_port_cycles": freeze_counts,
                  "source_sha256": {str(source.relative_to(rtlsim.ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                                    for source in sources},
                  "model_sha256": hashlib.sha256((Path(__file__).parents[1] / "tools/wideport_tmem_resource.py").read_bytes()).hexdigest(),
                  "log_sha256": hashlib.sha256((run.stdout + run.stderr).encode()).hexdigest(),
                  "scope": "bank resource and IQ->pw pipeline only; unit queues/enables/DRAM and full LPN timing remain unmodeled"}
        (evidence / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
