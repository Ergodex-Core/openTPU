"""Bounded ownership of operand FIFO slots and streamed accumulator rows.

Generated issue/arrival/pop pulses may drive this component from the coupled
memory model. Dispatch, release, drain grants and drain epilogue are external
until qualified separately. DSP accumulators are continuously clocked streams,
not independent per-instruction banks. Row credits reserve eventual result
space; a new row must never overwrite an unfinished accumulation stream.
"""
from collections import deque
from dataclasses import dataclass
from tools.wideport_memory_feedback import require
from tools.wideport_mxu_producer import Rl4Int8Mm


@dataclass(frozen=True)
class Ticket:
    slot: int
    epoch: int
    owner: int


class SlotRing:
    """Reserve -> complete -> consume, with monotonically increasing epochs."""
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("slot capacity must be positive")
        self.capacity = capacity
        self.issued = self.completed = self.consumed = self.reuses = 0
        self.pending = deque()
        self.ready = deque()

    def reserve(self, owner):
        if self.issued - self.consumed >= self.capacity:
            raise ValueError("premature FIFO slot reuse: old owner has not consumed")
        token = Ticket(self.issued % self.capacity, self.issued // self.capacity, owner)
        self.reuses += int(token.epoch != 0)
        self.pending.append(token)
        self.issued += 1
        return token

    def complete(self, token=None):
        if not self.pending:
            raise ValueError("completion without reserved FIFO slot")
        expected = self.pending[0]
        if token is not None and token != expected:
            raise ValueError("stale epoch or out-of-order FIFO completion")
        self.pending.popleft()
        self.ready.append(expected)
        self.completed += 1
        return expected

    def consume(self, owner):
        if not self.ready or self.ready[0].owner != owner:
            raise ValueError("operand unavailable or wrong FIFO owner")
        token = self.ready.popleft()
        self.consumed += 1
        return token


class RowCredits:
    """Result reservation is acquired at first pop and freed only by a drain."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.live = deque()
        self.ready = deque()
        self.active = None
        self.rows_started = self.rows_completed = self.rows_drained = 0

    @property
    def available(self):
        return self.active is None and len(self.live) < self.capacity

    def first(self, owner):
        if not self.available:
            raise ValueError("accumulator unavailable or result row credits exhausted")
        token = Ticket(self.rows_started % self.capacity, self.rows_started // self.capacity, owner)
        self.live.append(token)
        self.active = token
        self.rows_started += 1
        return token

    def last(self, owner):
        if self.active is None or self.active.owner != owner:
            raise ValueError("last block without matching active accumulator")
        token, self.active = self.active, None
        return token

    def result(self, token):
        if token not in self.live or token in self.ready:
            raise ValueError("stale or duplicate accumulator result epoch")
        self.ready.append(token)
        self.rows_completed += 1

    def drain(self, *, grant):
        if not grant:
            raise ValueError("output backpressure forbids result drain")
        if not self.ready or self.ready[0] != self.live[0]:
            raise ValueError("drain without oldest completed accumulator result")
        token = self.ready.popleft()
        self.live.popleft()
        self.rows_drained += 1
        return token


class PipelineOwnership:
    """Pre-edge checks; only generated pulses update newly modeled slot state."""
    def __init__(self, mms, *, depth=1024, cl=16, program_pcs=None):
        self.specs = [Rl4Int8Mm.from_instr(x) for x in mms]
        self.depth = depth
        self.cls = min(cl, 16)
        self.lrow = self.cls + 8 + 5
        self.result_delay = self.lrow + 3
        self.weights, self.scales = SlotRing(depth), SlotRing(depth)
        self.rows = RowCredits(32 if self.lrow + 7 <= 32 else 64)
        self.accepted = self.released = self.issued_owner = self.pop_owner = self.drained_owner = 0
        self.owner_issues = self.owner_pops = 0
        self.popped_per_owner = [0] * len(mms)
        self.drained_per_owner = [0] * len(mms)
        self.dispatch = {}
        self.slot_epochs = {}
        self.program_pcs = program_pcs
        self.owner_slots = []
        self.released_slots = set()
        self.tags = {}
        self.results = {}
        self.rf_q = self.rf_t = self.rf_hs = 0
        self.blocked_cycles = self.checked = 0
        self.events = [dict(dispatch=None, start=None, release=None, first_pop=None,
                            last_pop=None, first_result=None, last_result=None,
                            last_drain=None, output_committed=None) for _ in mms]

    def step(self, e, *, b_issue, scale_issue, b_push, r_cpl, pop, rf_push, d_pop, cycle):
        me, oe, sq = e["mxu"], e["ownership"], e["seq"]
        if me["rst"]:
            return
        for key, value in (("f_head", self.weights.consumed % self.depth),
                           ("f_tail", self.weights.completed % self.depth),
                           ("s_head", self.scales.consumed % self.depth),
                           ("s_tail", self.scales.completed % self.depth),
                           ("rf_q", self.rf_q), ("rf_t", self.rf_t), ("rf_hs", self.rf_hs)):
            require(oe[key], value, "ownership." + key, cycle)
        for delay, prefix in ((1, "m0p"), (2, "m0")):
            tag = self.tags.get(cycle - delay, (0, 0, 0))
            for label, value in zip(("v", "first", "last"), tag):
                # first/last in m0p are zeroed in bubbles; m0 follows m0p.
                require(oe[prefix + "_" + label], value, "ownership." + prefix + label, cycle)
        expected_first = sum(1 << j for j in range(8)
                             if self.tags.get(cycle - (2 + self.cls + 4 + j), (0, 0, 0))[1])
        require(oe["acc_first"], expected_first, "ownership.acc_first", cycle)
        expected_tl = int(bool(self.tags.get(cycle - (2 + self.lrow), (0, 0, 0))[2]))
        require(oe["tl_v"], expected_tl, "ownership.tl_v", cycle)
        result = self.results.pop(cycle, None)
        require(rf_push, int(result is not None), "ownership.result_pipeline", cycle)
        require(me["rf_n"], len(self.rows.ready), "ownership.result_count", cycle)
        require(me["rows_live"] + me["rows_p"], len(self.rows.live), "ownership.reserved_rows", cycle)
        self.blocked_cycles += int(bool(me["drain_go"] and not me["t_gnt"]))
        if d_pop and not me["t_gnt"]:
            raise ValueError("output backpressure forbids result drain")
        if me["c_done"]:
            owner = self.drained_owner
            if (owner >= len(self.specs) or self.drained_per_owner[owner] != self.specs[owner].row_groups
                    or oe["pipe_busy"] or not oe["c_drained"]):
                raise ValueError("output committed before rows and epilogue drained")
            self.events[owner]["output_committed"] = cycle
            self.drained_owner += 1
        # Retire only old visible results; same-edge result cannot be drained.
        if d_pop:
            token = self.rows.drain(grant=bool(me["t_gnt"]))
            self.drained_per_owner[token.owner] += 1
            self.events[token.owner]["last_drain"] = cycle
        if result is not None:
            self.rows.result(result)
            ev = self.events[result.owner]
            if ev["first_result"] is None:
                ev["first_result"] = cycle
            ev["last_result"] = cycle
        first = last = 0
        if pop:
            owner = self.pop_owner
            if owner >= self.released:
                raise ValueError("accumulator pop before command release")
            spec = self.specs[owner]
            require(me["pn"], int(owner != self.drained_owner), "ownership.pop_head", cycle)
            require(oe["c_left"], spec.advances - self.popped_per_owner[owner], "ownership.remaining_advances", cycle)
            if not (oe["n_go"] if me["pn"] else oe["h_go"]):
                raise ValueError("accumulator pop-head has not been released")
            first, last = int(self.owner_pops % spec.kb == 0), int(self.owner_pops % spec.kb == spec.kb - 1)
            if first:
                self.rows.first(owner)
            self.weights.consume(owner)
            if not spec.unit:
                self.scales.consume(owner)
            if last:
                self.results[cycle + self.result_delay] = self.rows.last(owner)
            ev = self.events[owner]
            if ev["first_pop"] is None:
                ev["first_pop"] = cycle
            ev["last_pop"] = cycle
            self.owner_pops += 1
            self.popped_per_owner[owner] += 1
            if self.owner_pops == spec.advances:
                self.pop_owner += 1
                self.owner_pops = 0
        self.tags[cycle] = (int(pop), first, last)
        if b_push:
            self.weights.complete()
        if r_cpl:
            self.scales.complete()
        # Issuer changes owner only after all command-derived B entries issue.
        if b_issue:
            self.weights.reserve(self.issued_owner)
        if scale_issue:
            self.scales.reserve(self.issued_owner)
        if b_issue:
            self.owner_issues += 1
            if self.owner_issues == self.specs[self.issued_owner].advances:
                self.issued_owner += 1
                self.owner_issues = 0
        if e["start"] & 2:
            slot = sq["mxu_start_slot"]
            if slot not in self.dispatch or self.dispatch[slot]["op"] != 0x20:
                raise ValueError("MM start without matching dispatched slot")
            entry = self.dispatch[slot]
            if self.program_pcs is not None and entry["pc"] != self.program_pcs[self.accepted]:
                raise ValueError("MM start does not match expected PC/slot epoch")
            identity = (slot, entry["epoch"], entry["pc"])
            self.owner_slots.append(identity)
            self.events[self.accepted].update(dispatch=entry["cycle"], start=cycle,
                                              slot=slot, epoch=entry["epoch"], pc=entry["pc"])
            self.accepted += 1
        if sq["release"]:
            entry = self.dispatch.get(sq["release_slot"])
            identity = ((sq["release_slot"], entry["epoch"], entry["pc"]) if entry else None)
            if self.released >= self.accepted or identity != self.owner_slots[self.released]:
                raise ValueError("release does not match oldest unreleased MM slot")
            self.events[self.released]["release"] = cycle
            self.released += 1
        for slot in range(32):
            if sq["end_mask"] & (1 << slot):
                if slot not in self.dispatch:
                    raise ValueError("slot finished without a live dispatch epoch")
                del self.dispatch[slot]
        if sq["dispatch"]:
            slot = sq["slot"]
            if slot in self.dispatch:
                raise ValueError("premature sequencer slot reuse before completion")
            epoch = self.slot_epochs.get(slot, -1) + 1
            self.slot_epochs[slot] = epoch
            self.dispatch[slot] = dict(cycle=cycle, pc=sq["pc"], op=sq["op"], epoch=epoch)
        self.rf_q = (self.rf_q + self.rf_hs) % self.rows.capacity
        self.rf_hs = int(d_pop)
        self.rf_t = (self.rf_t + int(rf_push)) % self.rows.capacity
        self.checked += 1

    def finish(self):
        if (self.accepted != len(self.specs) or self.released != len(self.specs)
                or self.drained_owner != len(self.specs) or self.rows.live or self.results
                or self.weights.pending or self.weights.ready or self.scales.pending or self.scales.ready):
            raise ValueError("ownership trace ended with live or missing command resources")
        return dict(cycles_checked=self.checked, events=self.events,
                    operand_slot_reuses=self.weights.reuses, scale_slot_reuses=self.scales.reuses,
                    rows_started=self.rows.rows_started, rows_completed=self.rows.rows_completed,
                    rows_drained=self.rows.rows_drained, output_backpressure_cycles=self.blocked_cycles,
                    result_slot_reuses=max(0, self.rows.rows_started - self.rows.capacity),
                    dispatch_release="observed inputs, identities/order checked; timing not independently predicted",
                    calibrated=False, tok_s=None)


def validate_pipeline_ownership(directory, *, slice_events=None):
    """Oracle-input protocol audit; the coupled path uses generated pulses instead."""
    import json
    from pathlib import Path
    from opentpu import isa as I
    from tools.wideport_credit_budget import _program
    from tools.wideport_memory_feedback import read_trace
    root = Path(directory)
    _, recorded = read_trace(root / "tmfeedback.jsonl")
    events = recorded if slice_events is None else slice_events
    params = json.loads((root / "feedback-inputs.json").read_text())["parameters"]
    program = _program(root)
    model = PipelineOwnership([p for p in program if p.op == I.MM],
                              depth=params["FIFO_DEPTH"], cl=params.get("MXU_CL", 16),
                              program_pcs=[pc for pc, p in enumerate(program) if p.op == I.MM])
    for e in events:
        me, ie = e["mxu"], e["issue"]
        model.step(e, b_issue=bool(ie["b_req"]),
                   scale_issue=bool(ie["a_req"] and ie["tag_ent"] and ie["tag_cpl"]),
                   b_push=bool(me["b_push"]), r_cpl=bool(me["r_cpl"]),
                   pop=bool(me["pop"]), rf_push=bool(me["rf_push"]),
                   d_pop=bool(me["d_pop"]), cycle=e["cycle"])
    result = model.finish()
    result["input_scope"] = "observed issue/arrival/pop/drain pulses; accumulator tags, slot pointers, row credits independently advanced"
    return result
