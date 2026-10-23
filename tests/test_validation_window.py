"""Fake-clock fault injection; these tests never open a hardware transport."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from opentpu.host.validation_window import Budget, Deadline, RunEvidence, ValidationWindow
from opentpu.llm.staged import Case


class Clock:
    value = 0.0
    def __call__(self):
        return self.value


def models():
    cfg = object()
    def model(name, cases):
        return SimpleNamespace(identity=name, ram=SimpleNamespace(sealed=True), active=False,
                               engine=SimpleNamespace(cfg=cfg), cases=cases)
    return (model("nanbeige", (Case("raw", (1, 3), (5, 7)),
                               Case("raw_reset_replay", (1, 3), (5, 7)))),
            model("gemma", (Case("verify", (2, 4), (6, 8)),)))


class Owner:
    evidence_kind = "mock"
    def __init__(self, clock):
        self.clock, self.calls = clock, []
        self.duration = dict(gate=.1, release=.1, upload=2, raw=1, raw_reset_replay=1,
                             cleanup=.2, restore=2, verify=.5, reopen=.1)
        self.fail, self.false, self.overrun = set(), set(), set()
        self.proof_change = {}
        self.busy = self.cancel = False
        self.cancel_after = None
        self.gated, self.resident = False, True

    def op(self, phase, deadline):
        self.calls.append((phase, deadline.expires, deadline.remaining()))
        if phase not in self.overrun:
            deadline.check()
        self.clock.value += self.duration.get(phase, 0)
        if phase in self.fail:
            raise RuntimeError("injected " + phase)
        if phase == self.cancel_after:
            self.cancel = True
        return phase not in self.false

    def preflight(self, *prepared):
        self.calls.append(("preflight", None, None))
        self.clock.value += 50  # preparation must not consume the outage allowance

    def gate_if_idle(self, deadline):
        value = self.op("gate", deadline)
        if self.busy or not value:
            return False
        self.gated = True
        return True

    def release_gemma(self, deadline):
        self.resident = False  # release can have happened even when acknowledgement fails
        return self.op("release", deadline)

    def upload_nanbeige(self, prepared, deadline):
        return self.op("upload", deadline)

    def evidence(self, model, case):
        return replace(RunEvidence(model.identity, case.expected, 123, "mock",
                                   True, True, True, True), **self.proof_change)

    def validate_nanbeige(self, prepared, case, deadline):
        self.op(case.name, deadline)
        return self.evidence(prepared, case)

    def cleanup_nanbeige(self, deadline):
        return self.op("cleanup", deadline)

    def restore_gemma(self, prepared, deadline):
        result = self.op("restore", deadline)
        self.resident = result
        return result

    def verify_gemma(self, prepared, case, deadline):
        self.op("verify", deadline)
        if not self.resident:
            raise RuntimeError("Gemma is not restored")
        return self.evidence(prepared, case)

    def reopen(self, deadline):
        assert self.resident
        result = self.op("reopen", deadline)
        if result:
            self.gated = False
        return result


def run(owner, clock, **kwargs):
    return ValidationWindow(owner, clock=clock, cancelled=lambda: owner.cancel, **kwargs).run(*models())


def names(owner):
    return [c[0] for c in owner.calls]


def test_complete_flow_reserves_restore_and_includes_reopening_in_one_clock():
    clock = Clock(); owner = Owner(clock)
    r = run(owner, clock)
    assert names(owner) == ["preflight", "gate", "release", "upload", "raw",
                           "raw_reset_replay", "cleanup", "restore", "verify", "reopen"]
    assert r["status"] == "simulated_success" and r["within_total_budget"]
    assert r["gemma_verified"] and r["admissions_reopened"] and not owner.gated
    assert not r["hardware_validated"] and not r["hardware_executed"]
    assert r["elapsed_seconds"] == pytest.approx(7)
    deadlines = {phase: end for phase, end, _ in owner.calls}
    assert deadlines["upload"] == 56 and deadlines["cleanup"] == 58
    assert deadlines["restore"] == deadlines["reopen"] == 80


def test_busy_card_or_active_request_never_releases_or_waits_under_the_gate():
    clock = Clock(); owner = Owner(clock); owner.busy = True
    r = run(owner, clock)
    assert r["status"] == "not_started_busy" and not r["outage_started"]
    assert names(owner) == ["preflight", "gate"] and not owner.gated


@pytest.mark.parametrize("phase", ["release", "upload", "raw", "raw_reset_replay"])
def test_partial_failure_always_cleans_then_restores_and_verifies(phase):
    clock = Clock(); owner = Owner(clock); owner.fail.add(phase)
    r = run(owner, clock)
    assert r["status"] == "failed_restored" and not r["nanbeige_validated"]
    assert names(owner)[-4:] == ["cleanup", "restore", "verify", "reopen"]
    assert r["gemma_verified"] and not owner.gated and r["errors"]


def test_test_deadline_stops_additional_model_work_and_preserves_restore_interval():
    clock = Clock(); owner = Owner(clock); owner.duration["upload"] = 7
    r = run(owner, clock)
    assert "raw" not in names(owner) and "raw_reset_replay" not in names(owner)
    assert names(owner)[-4:] == ["cleanup", "restore", "verify", "reopen"]
    assert r["status"] == "failed_restored" and r["within_total_budget"]
    assert r["errors"][0]["type"] == "DeadlineExceeded"


def test_late_gate_acknowledgement_still_recovers_the_unchanged_warm_model():
    clock = Clock(); owner = Owner(clock); owner.duration["gate"] = 7
    r = run(owner, clock)
    assert names(owner) == ["preflight", "gate", "verify", "reopen"]
    assert r["outage_started"] and r["status"] == "failed_restored" and not owner.gated


def test_late_cleanup_acknowledgement_does_not_skip_restoration():
    clock = Clock(); owner = Owner(clock); owner.duration["cleanup"] = 5
    r = run(owner, clock)
    assert r["nanbeige_cleaned"] and r["status"] == "failed_restored"
    assert names(owner)[-3:] == ["restore", "verify", "reopen"]


@pytest.mark.parametrize("phase", ["cleanup", "restore", "verify", "reopen"])
@pytest.mark.parametrize("mode", ["raise", "false"])
def test_no_false_ready_result_on_cleanup_or_restore_failures(phase, mode):
    clock = Clock(); owner = Owner(clock)
    (owner.fail if mode == "raise" else owner.false).add(phase)
    if phase == "verify" and mode == "false":
        owner.proof_change["configuration_matches"] = False
    r = run(owner, clock)
    assert r["status"] == "failed_closed" and not r["admissions_reopened"]
    assert owner.gated and not r["hardware_validated"]
    if phase == "cleanup":
        assert "restore" not in names(owner) and "verify" not in names(owner)


@pytest.mark.parametrize("change", [
    {"tokens": (8, 9)}, {"model_identity": "wrong"}, {"cycles": 0}, {"cycles": True},
    {"configuration_matches": False}, {"normal_locks_held": False},
    {"context_reset": False}, {"selected_on_device": False}, {"kind": "board"},
])
def test_result_mismatch_never_becomes_accepted_or_skips_recovery(change):
    clock = Clock(); owner = Owner(clock)
    original = owner.validate_nanbeige
    owner.validate_nanbeige = lambda *a: replace(original(*a), **change)
    r = run(owner, clock)
    assert r["status"] == "failed_restored" and not r["nanbeige_validated"]
    assert not r["hardware_executed"] and not r["hardware_validated"]


def test_cancel_before_gating_and_during_validation_keep_recovery_possible():
    clock = Clock(); owner = Owner(clock); owner.cancel = True
    r = run(owner, clock)
    assert r["status"] == "cancelled_before_gate" and names(owner) == ["preflight"]
    clock = Clock(); owner = Owner(clock); owner.cancel_after = "upload"
    r = run(owner, clock)
    assert r["status"] == "failed_restored" and "raw" not in names(owner)
    assert names(owner)[-4:] == ["cleanup", "restore", "verify", "reopen"]


def test_blocking_hook_overrun_is_reported_without_claiming_a_hard_timeout():
    clock = Clock(); owner = Owner(clock)
    owner.duration["restore"] = 35; owner.overrun.update({"restore", "verify"})
    r = run(owner, clock)
    assert r["status"] == "deadline_missed" and r["elapsed_seconds"] > 30
    assert not r["admissions_reopened"] and not r["hardware_validated"]
    assert "reopen" not in names(owner)


def test_preflight_and_budget_failures_have_no_admission_or_device_effects():
    clock = Clock(); owner = Owner(clock); n, g = models(); n.ram.sealed = False
    r = ValidationWindow(owner, clock=clock).run(n, g)
    assert r["status"] == "preflight_failed" and not owner.calls
    clock = Clock(); owner = Owner(clock); n, g = models(); n.active = True
    r = ValidationWindow(owner, clock=clock).run(n, g)
    assert r["status"] == "preflight_failed" and not owner.calls
    for kw in ({"total": 31}, {"total": True}, {"restore": float("inf")},
               {"cleanup": -1}, {"restore": 29, "cleanup": 1}):
        with pytest.raises(ValueError):
            Budget(**kw)


def test_controller_is_single_use():
    clock = Clock(); owner = Owner(clock); controller = ValidationWindow(owner, clock=clock)
    controller.run(*models())
    with pytest.raises(RuntimeError, match="single-use"):
        controller.run(*models())


def test_gate_exception_contract_does_not_touch_gemma():
    clock = Clock(); owner = Owner(clock); owner.fail.add("gate")
    r = run(owner, clock)
    assert r["status"] == "gate_failed" and not r["outage_started"]
    assert names(owner) == ["preflight", "gate"] and owner.resident and not owner.gated
