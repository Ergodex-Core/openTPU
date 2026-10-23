"""Owner-supplied validation window protocol and cooperative deadline controller.

No transport, gateway, service, lock, or device is opened by this module. The
owner must implement deadline-aware hooks and an independent recovery watchdog.
A controller cannot bound a blocking driver call or guarantee hardware recovery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Protocol


class DeadlineExceeded(TimeoutError):
    pass


class WindowCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Deadline:
    expires: float
    clock: object = time.monotonic
    cancelled: object = None

    def remaining(self):
        return max(0.0, self.expires - self.clock())

    def check(self):
        if self.cancelled is not None and self.cancelled():
            raise WindowCancelled("validation cancelled; restore the serving model")
        if self.clock() >= self.expires:
            raise DeadlineExceeded("phase deadline reached; reserve restoration time")


@dataclass(frozen=True)
class Budget:
    total: float = 30.0
    restore: float = 22.0
    cleanup: float = 2.0

    def __post_init__(self):
        values = (self.total, self.restore, self.cleanup)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in values) or \
                self.total > 30 or self.restore + self.cleanup >= self.total:
            raise ValueError("positive finite reserves must leave test time within 30 seconds")


@dataclass(frozen=True)
class RunEvidence:
    model_identity: str
    tokens: tuple[int, ...]
    cycles: int
    kind: str                 # 'mock' in software tests; 'board' only from a real owner
    normal_locks_held: bool
    configuration_matches: bool
    context_reset: bool
    selected_on_device: bool


class WindowOwner(Protocol):
    """Interface proposal; production integration must be provided by its owner.

    gate_if_idle must atomically reject active requests without changing admission
    state, or close admissions and return True. On exception it must roll back.
    All hooks receive an absolute monotonic deadline. Recovery hooks are called
    even with zero time remaining; they must report failure rather than claim
    readiness. They must not reload images or interrupt another owner's run.
    """
    evidence_kind: str

    def preflight(self, nanbeige, gemma): ...
    def gate_if_idle(self, deadline: Deadline) -> bool: ...
    def release_gemma(self, deadline: Deadline) -> bool: ...
    def upload_nanbeige(self, prepared, deadline: Deadline) -> bool: ...
    def validate_nanbeige(self, prepared, case, deadline: Deadline) -> RunEvidence: ...
    def cleanup_nanbeige(self, deadline: Deadline) -> bool: ...
    def restore_gemma(self, prepared, deadline: Deadline) -> bool: ...
    def verify_gemma(self, prepared, case, deadline: Deadline) -> RunEvidence: ...
    def reopen(self, deadline: Deadline) -> bool: ...


class ValidationWindow:
    """One attempt, with a reserved cleanup/restore interval and honest receipts.

    total runs from immediately before gate_if_idle through reopening admissions.
    Pure preparation/preflight happens first, while the service remains available.
    The default 22-second restore reserve is an unqualified policy example, not a
    measured bound. No production WindowOwner implementation is supplied here.
    """
    def __init__(self, owner: WindowOwner, *, budget=Budget(), clock=time.monotonic,
                 cancelled=lambda: False):
        self.owner, self.budget, self.clock = owner, budget, clock
        self.cancelled = cancelled
        self.used = False

    def run(self, nanbeige, gemma):
        if self.used:
            raise RuntimeError("a validation window controller is single-use")
        self.used = True
        r = dict(status="preflight", evidence_kind=self.owner.evidence_kind,
                 hardware_executed=False, hardware_validated=False, outage_started=False,
                 admissions_reopened=False, nanbeige_validated=False,
                 gemma_verified=False, nanbeige_cleaned=False, budget=asdict(self.budget),
                 events=[], errors=[], cases=[], elapsed_seconds=0.0)
        if self.owner.evidence_kind not in ("mock", "board"):
            raise ValueError("owner must label evidence as mock or board")

        def error(phase, exc):
            r["errors"].append(dict(phase=phase, type=type(exc).__name__, message=str(exc)))

        try:
            if not all(m.ram.sealed and not m.active for m in (nanbeige, gemma)) or \
                    nanbeige.engine.cfg != gemma.engine.cfg:
                raise ValueError("both models must be inactive, sealed, and use one configuration")
            if len(nanbeige.cases) != 2 or len(gemma.cases) != 1 or \
                    nanbeige.cases[0].prompt != nanbeige.cases[1].prompt or \
                    nanbeige.cases[0].expected != nanbeige.cases[1].expected:
                raise ValueError("acceptance requires identical raw/reset cases and one Gemma case")
            self.owner.preflight(nanbeige, gemma)
            if self.cancelled():
                r["status"] = "cancelled_before_gate"
                return r
        except Exception as exc:
            error("preflight", exc)
            r["status"] = "preflight_failed"
            return r

        started = self.clock()
        total = Deadline(started + self.budget.total, self.clock)
        cleanup = Deadline(total.expires - self.budget.restore, self.clock)
        test = Deadline(cleanup.expires - self.budget.cleanup, self.clock, self.cancelled)
        released_attempted = False
        phase = "gate"

        def call(name, deadline, action, *, recovery=False):
            r["events"].append(dict(phase=name, event="begin",
                elapsed=self.clock() - started, remaining=deadline.remaining()))
            if not recovery:
                deadline.check()
            value = action(deadline)
            r["events"].append(dict(phase=name, event="end",
                elapsed=self.clock() - started, remaining=deadline.remaining()))
            deadline.check()  # detect overruns even if a hook ignored its deadline
            return value

        def proof(model, case, value):
            if not isinstance(value, RunEvidence):
                raise TypeError("owner did not return structured execution evidence")
            if value.kind == "board" and self.owner.evidence_kind == "board":
                r["hardware_executed"] = True
            if value.kind != self.owner.evidence_kind or value.model_identity != model.identity or \
                    tuple(value.tokens) != case.expected or type(value.cycles) is not int or \
                    value.cycles <= 0 or not all((value.normal_locks_held,
                    value.configuration_matches, value.context_reset, value.selected_on_device)):
                raise ValueError("execution identity, tokens, counters, reset, or ownership failed")
            return asdict(value)

        def gate_action(deadline):
            value = self.owner.gate_if_idle(deadline)
            if value is True:
                r["outage_started"] = True  # retain the acknowledgement on deadline overrun
            return value

        def cleanup_action(deadline):
            value = self.owner.cleanup_nanbeige(deadline)
            r["nanbeige_cleaned"] = value is True
            return value

        def verify_action(deadline):
            value = self.owner.verify_gemma(gemma, gemma.cases[0], deadline)
            r["gemma_evidence"] = proof(gemma, gemma.cases[0], value)
            r["gemma_verified"] = True
            return value

        def reopen_action(deadline):
            value = self.owner.reopen(deadline)
            r["admissions_reopened"] = value is True
            return value

        try:
            gated = call("gate", test, gate_action)
            if gated is not True:
                r["status"] = "not_started_busy"
                return r
            phase = "release_gemma"
            released_attempted = True  # an uncertain release still requires recovery
            if call(phase, test, self.owner.release_gemma) is not True:
                raise RuntimeError("Gemma did not acknowledge cooperative release")
            phase = "upload_nanbeige"
            if call(phase, test, lambda d: self.owner.upload_nanbeige(nanbeige, d)) is not True:
                raise RuntimeError("Nanbeige upload was not acknowledged")
            for case in nanbeige.cases:
                phase = "validate_" + case.name
                def validate(deadline, c=case):
                    value = self.owner.validate_nanbeige(nanbeige, c, deadline)
                    r["cases"].append(dict(name=c.name, evidence=proof(nanbeige, c, value)))
                    return value
                call(phase, test, validate)
            r["nanbeige_validated"] = True
        except Exception as exc:
            error(phase, exc)
        finally:
            if r["outage_started"]:
                if released_attempted:
                    try:
                        if call("cleanup_nanbeige", cleanup,
                                cleanup_action, recovery=True) is not True:
                            raise RuntimeError("Nanbeige ownership cleanup not confirmed")
                    except Exception as exc:
                        error("cleanup_nanbeige", exc)
                    if r["nanbeige_cleaned"]:
                        try:
                            if call("restore_gemma", total,
                                    lambda d: self.owner.restore_gemma(gemma, d), recovery=True) is not True:
                                raise RuntimeError("Gemma restoration not acknowledged")
                        except Exception as exc:
                            error("restore_gemma", exc)
                # Never verify or reopen on unconfirmed cleanup: another owner may still run.
                if not released_attempted or r["nanbeige_cleaned"]:
                    try:
                        call("verify_gemma", total, verify_action, recovery=True)
                    except Exception as exc:
                        error("verify_gemma", exc)
                if r["gemma_verified"]:
                    try:
                        if call("reopen", total, reopen_action) is not True:
                            raise RuntimeError("admission reopening not acknowledged")
                    except Exception as exc:
                        error("reopen", exc)
            r["elapsed_seconds"] = self.clock() - started
            r["within_total_budget"] = self.clock() < total.expires
            if r["outage_started"]:
                good = not r["errors"] and r["nanbeige_validated"] and \
                    r["gemma_verified"] and r["admissions_reopened"] and r["within_total_budget"]
                r["hardware_validated"] = good and self.owner.evidence_kind == "board"
                r["status"] = ("simulated_success" if self.owner.evidence_kind == "mock" else
                               "validated_and_restored") if good else \
                    "deadline_missed" if not r["within_total_budget"] else \
                    "failed_restored" if r["admissions_reopened"] else "failed_closed"
            elif r["status"] != "not_started_busy":
                r["status"] = "gate_failed"
        return r
