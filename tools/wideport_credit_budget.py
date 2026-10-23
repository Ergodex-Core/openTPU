"""Oracle-only credit and bubble accounting for the bounded two-MM RL4 fixture.

This reads preserved pre-edge RTL traces. It is deliberately not a predictor:
observed responses are used only to audit the request/response ledger. Unknown
ownership transitions remain None, and no throughput improvement is inferred.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import struct

from opentpu import isa as I
from tools.wideport_memory_feedback import read_trace
from tools.wideport_mxu_issue import IssueCommand
from tools.wideport_native_memory import logical_beat


WB = (0x60000, 0x70000)
SB = (0x90000, 0x98000)
WEIGHT_BYTES = 18_432
SCALE_WINDOW_BYTES = 32_768  # reserved 32 KiB/page includes <=16 KiB APF lookahead
ADVANCES = 36
MM_START = MM_DONE = 1 << 1
Q_DONE = 1 << 2
COLL_DONE = 1 << 4


def _owner(byte_addr: int, bases: tuple[int, int], size: int) -> int | None:
    hits = [i for i, base in enumerate(bases) if base <= byte_addr < base + size]
    if len(hits) > 1:
        raise ValueError(f"ambiguous owner for byte address {byte_addr:#x}")
    return hits[0] if hits else None


def classify_byte(byte_addr: int) -> tuple[str, int] | None:
    """Classify a native logical beat or upstream word by byte address."""
    w = _owner(byte_addr, WB, WEIGHT_BYTES)
    if w is not None:
        return "weight", w
    s = _owner(byte_addr, SB, SCALE_WINDOW_BYTES)
    if s is not None:
        return "scale", s
    return None


def _first_last(cycles: list[int]) -> dict:
    return {"first": cycles[0] if cycles else None,
            "last": cycles[-1] if cycles else None, "count": len(cycles)}


def _gap(a: int | None, b: int | None) -> int | None:
    return b - a - 1 if a is not None and b is not None else None


def _pop_credit(ordinal: int) -> tuple[int, bool]:
    """Owner and row boundary of an RL4 W8 N13 KB9 M1 pop ordinal."""
    if not 0 <= ordinal < 2 * ADVANCES:
        raise ValueError("compute pop outside two-MM bounded program")
    return ordinal // ADVANCES, ordinal % 9 == 8


def _program(root: Path) -> list[I.Instr]:
    words = [int(line, 16) for line in (root / "prog_0.hex").read_text().splitlines()]
    words += [0] * (-len(words) % 8)
    digest = hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest()
    manifest = json.loads((root / "feedback-inputs.json").read_text())
    if digest != manifest["program_sha256"]:
        raise ValueError("program SHA256 differs from preserved manifest")
    return [I.Instr.decode(words[i:i + 8]) for i in range(0, len(words), 8)]


def _bounded_mms(program: list[I.Instr]) -> list[IssueCommand]:
    cmds = [IssueCommand.from_instr(ins) for ins in program if ins.op == I.MM]
    if len(cmds) != 2 or [c.sa for c in cmds] != list(WB) or [c.ssa for c in cmds] != list(SB):
        raise ValueError("budget requires two ordered MM commands at WB0/1 and SB0/1")
    if any(c.n != 13 or c.kb != 9 or c.advances != ADVANCES or c.unit for c in cmds):
        raise ValueError("budget requires N13 KB9 W8 RL4 M1 with 36 weight and scale entries")
    return cmds


def _events(root: Path):
    nh, native = read_trace(root / "native-feedback.jsonl")
    ah, adapter = read_trace(root / "adapter-feedback.jsonl")
    sh, slice_events = read_trace(root / "tmfeedback.jsonl")
    if len(native) != len(adapter) or len(native) < len(slice_events):
        raise ValueError("native/adapter traces must cover the complete slice interval")
    expected = dict(channels=8, portb=4, aline=64, qd=16, rd=128, ad=64, apf=32, apr=16)
    if any(ah.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported adapter configuration")
    if nh.get("channels") != 8 or not nh.get("deterministic"):
        raise ValueError("budget requires deterministic native NCH8 evidence")
    if sh.get("schema") != 1:
        raise ValueError("unsupported slice trace schema")
    for cycle, triplet in enumerate(zip(native, adapter, slice_events)):
        if any(e.get("kind") != "cycle" or e.get("cycle") != cycle for e in triplet):
            raise ValueError(f"misaligned trace at cycle {cycle}")
    # The top-level dump can leave a short native suffix after the slice
    # stopped tracing. Do not silently lose any outstanding owner response.
    for cycle in range(len(slice_events), len(native)):
        ne, ae = native[cycle], adapter[cycle]
        if ne.get("cycle") != cycle or ae.get("cycle") != cycle:
            raise ValueError(f"misaligned native suffix at cycle {cycle}")
        if any(row["cv"] or row["wv"] or row["rv"] for row in ne["channels"]) or any(
                ae[key] for key in ("a_req", "b_req", "a_rvalid", "b_rvalid", "sw_req")):
            raise ValueError(f"active native suffix after slice trace at cycle {cycle}")
    return nh, native[:len(slice_events)], adapter[:len(slice_events)], slice_events


def _new_owner(index: int, cmd: IssueCommand) -> dict:
    return {
        "index": index, "weight_base": cmd.sa, "scale_base": cmd.ssa,
        "weight_bytes_expected": WEIGHT_BYTES, "entries_expected": ADVANCES,
        "dispatch": None, "allocated_at_start": None, "activation_ready_qact_done": None,
        "weights_ready_first_b_push": None, "weights_ready_all_b_push": None,
        "scales_ready_first_r_cpl": None, "scales_ready_all_r_cpl": None,
        "output_committed_c_done": None, "udone": None, "explicit_release": None,
        "collective_done_candidate": None, "page_safe_release": None,
        "accumulator_available": None,
        "missing_instrumentation": ["instruction dispatch", "accumulator availability",
                                    "explicit buffer release", "collective dependency ownership"],
        "cycles": {k: [] for k in ("b_issue", "a_issue", "b_push", "r_cpl", "compute",
                                  "rf_push", "drain_go", "d_pop", "native_weight_request",
                                  "native_weight_response", "native_scale_request",
                                  "native_scale_response")},
        "weight_issue_bytes": 0, "scale_issue_bytes": 0,
        "scale_all_issue_bytes": 0,
        "native_weight_payload_bytes": 0, "native_scale_payload_bytes": 0,
        "native_weight_valid_beats": 0, "native_scale_valid_beats": 0,
    }


def _event(owner: dict, name: str, cycle: int):
    owner["cycles"][name].append(cycle)


def _read_response(queues, row, channel: int, cycle: int, owners: list[dict], errors: list[str]):
    if not row["rv"]:
        return
    if not queues[channel]:
        errors.append(f"cycle {cycle}: channel {channel} returned a read without an accepted request")
        return
    credit = queues[channel].popleft()
    credit["response_cycle"] = cycle
    kind_owner = credit["owner"]
    if kind_owner is None:
        return
    kind, index = kind_owner
    owner = owners[index]
    _event(owner, f"native_{kind}_response", cycle)
    owner[f"native_{kind}_valid_beats"] += 1


def _accepted_command(queues, row, channel: int, cycle: int, chash: bool,
                      owners: list[dict], errors: list[str], ledger: list[dict], ids: list[int]):
    if not (row["cv"] and row["cr"]):
        return
    address = 64 * logical_beat(row["addr"], channel, chash=chash)
    owner_key = classify_byte(address)
    if row["we"]:
        if owner_key is not None:
            errors.append(f"cycle {cycle}: write to owned {owner_key} page at {address:#x}")
        return
    credit = {"channel": channel, "id": ids[channel], "logical_byte": address,
              "owner": owner_key, "request_cycle": cycle, "response_cycle": None}
    ids[channel] += 1
    queues[channel].append(credit)
    ledger.append(credit)
    if owner_key is not None:
        kind, index = owner_key
        owner = owners[index]
        _event(owner, f"native_{kind}_request", cycle)
        owner[f"native_{kind}_payload_bytes"] += 64


def analyze_credit_budget(directory: str | Path) -> dict:
    """Return an auditable budget; ``predicates.ok`` must be asserted by caller.

    The full trace is required, including read responses after the final MM.
    External DMA/loader traffic is counted but never attributed to an MM by tag
    alone. Channel response identity is derived solely from accepted read order.
    """
    root = Path(directory)
    cmds = _bounded_mms(_program(root))
    header, native, adapter, slices = _events(root)
    owners = [_new_owner(i, cmd) for i, cmd in enumerate(cmds)]
    errors: list[str] = []
    dependencies: list[str] = []
    queues = [deque() for _ in range(8)]
    ids = [0] * 8
    ledger: list[dict] = []
    b_entries, a_entries = deque(), deque()
    starts: list[int] = []
    q_done: list[int] = []
    c_done: list[int] = []
    u_done: list[int] = []
    collective_done: list[int] = []
    port = [dict(active=0, no_request=0, backpressure=0, accepted=0, valid=0,
                 outstanding_highwater=0) for _ in range(8)]
    peaks = dict(f_count=0, s_count=0, occ=0, tg_n=0, rf_n=0)
    mxu_active = 0
    drain_owner = 0
    pop_ordinal = 0
    result_rows = deque()
    for cycle, (ne, ae, se) in enumerate(zip(native, adapter, slices)):
        ie, me = se["issue"], se["mxu"]
        if se["start"] & MM_START:
            starts.append(cycle)
            if len(starts) <= 2:
                owners[len(starts) - 1]["allocated_at_start"] = cycle
        if se["done"] & Q_DONE:
            q_done.append(cycle)
        if se["done"] & MM_DONE:
            u_done.append(cycle)
        if se["done"] & COLL_DONE:
            collective_done.append(cycle)
        current = len(starts) - 1
        if current >= 0 and current < 2:
            o = owners[current]
            for signal, label in (("b_req", "b_issue"), ("a_req", "a_issue")):
                if ie[signal]:
                    _event(o, label, cycle)
            if ie["b_req"]:
                b_entries.append(current)
                o["weight_issue_bytes"] += 128 * (ie["b_n"] + 1)
                if classify_byte(4 * ie["b_addr"]) != ("weight", current):
                    errors.append(f"cycle {cycle}: B issue outside current weight page")
            if ie["a_req"]:
                o["scale_all_issue_bytes"] += 4
                if classify_byte(4 * ie["a_addr"]) != ("scale", current):
                    errors.append(f"cycle {cycle}: A issue outside current scale page")
                if ie["tag_ent"] and ie["tag_cpl"]:
                    a_entries.append(current)
                    o["scale_issue_bytes"] += 4
        # The pop head may move to MM1 while MM0 is still draining. RL4 W8 M1
        # consumes one FIFO entry per pop: 36 advances, four rows of nine.
        # Row-valid is pipelined, so its owner comes from a row-completion FIFO.
        if bool(me["compute"]) != bool(me["fpop"]):
            dependencies.append(f"cycle {cycle}: bounded RL4 compute/fpop mismatch")
        if me["compute"]:
            if pop_ordinal < 2 * ADVANCES:
                pop_owner, row_complete = _pop_credit(pop_ordinal)
                _event(owners[pop_owner], "compute", cycle)
                if row_complete:
                    result_rows.append(pop_owner)
            else:
                dependencies.append(f"cycle {cycle}: compute exceeds 72 expected advances")
            pop_ordinal += 1
        if me["rf_push"]:
            if result_rows:
                _event(owners[result_rows.popleft()], "rf_push", cycle)
            else:
                dependencies.append(f"cycle {cycle}: result push without completed row pop")
        if drain_owner < 2:
            for signal, label in (("drain_go", "drain_go"), ("d_pop", "d_pop")):
                if me[signal]:
                    _event(owners[drain_owner], label, cycle)
        if me["c_done"]:
            c_done.append(cycle)
            drain_owner += 1
        if me["b_push"]:
            if b_entries:
                _event(owners[b_entries.popleft()], "b_push", cycle)
            else:
                errors.append(f"cycle {cycle}: b_push without outstanding MM B issue")
        if me["r_cpl"]:
            if a_entries:
                _event(owners[a_entries.popleft()], "r_cpl", cycle)
            else:
                errors.append(f"cycle {cycle}: r_cpl without scale-completing A tag")
        mxu_active += int(bool(ie["i_act"] or me["rows_live"] or me["rf_nz"] or me["drain_go"]))
        for key in peaks:
            peaks[key] = max(peaks[key], int(ie[key] if key in ("occ", "tg_n") else me[key]))
        for channel, row in enumerate(ne["channels"]):
            p = port[channel]
            p["active"] += int(bool(row["cv"]))
            p["no_request"] += int(not row["cv"])
            p["backpressure"] += int(bool(row["cv"] and not row["cr"]))
            p["accepted"] += int(bool(row["cv"] and row["cr"]))
            p["valid"] += int(bool(row["rv"]))
            # Pre-edge response retires an older request; a same-edge command
            # cannot produce a same-edge read-valid in this native contract.
            _read_response(queues, row, channel, cycle, owners, errors)
            _accepted_command(queues, row, channel, cycle, bool(header["chash"]), owners, errors, ledger, ids)
            p["outstanding_highwater"] = max(p["outstanding_highwater"], len(queues[channel]))
        if ae["b_req"] and ae["b_ready"] and not ae["b_we"]:
            index = _owner(4 * ae["b_addr"], WB, WEIGHT_BYTES)
            if index is not None and 4 * ae["b_addr"] + 128 * (ae["b_n"] + 1) > WB[index] + WEIGHT_BYTES:
                errors.append(f"cycle {cycle}: accepted B burst crosses weight page")
    if len(starts) != 2 or len(c_done) != 2 or len(u_done) != 2:
        errors.append(f"expected two starts/c_done/udone, got {len(starts)}/{len(c_done)}/{len(u_done)}")
    if b_entries or a_entries:
        errors.append(f"unfinished MM entry credits: B={len(b_entries)} A={len(a_entries)}")
    if pop_ordinal != 2 * ADVANCES or result_rows:
        dependencies.append(f"pop/row credits incomplete: pops={pop_ordinal}, rows={len(result_rows)}")
    if any(queues):
        errors.append("unreturned native channel read credits")
    for i, o in enumerate(owners):
        o["output_committed_c_done"] = c_done[i] if i < len(c_done) else None
        o["udone"] = u_done[i] if i < len(u_done) else None
        o["collective_done_candidate"] = next((t for t in collective_done if t >= (o["udone"] or 0)), None)
        for kind, label in (("b_push", "weights"), ("r_cpl", "scales")):
            seq = o["cycles"][kind]
            o[f"{label}_ready_first_{kind}"] = seq[0] if seq else None
            o[f"{label}_ready_all_{kind}"] = seq[-1] if len(seq) == ADVANCES else None
            if len(seq) != ADVANCES:
                errors.append(f"MM{i} {kind}: expected {ADVANCES}, got {len(seq)}")
        if len(o["cycles"]["b_issue"]) != ADVANCES or o["weight_issue_bytes"] != WEIGHT_BYTES:
            errors.append(f"MM{i}: weight issue count/bytes differ from 36/18432")
        if o["scale_issue_bytes"] != ADVANCES * 4:
            errors.append(f"MM{i}: scale issue bytes differ from {ADVANCES * 4}")
        if o["native_weight_payload_bytes"] != WEIGHT_BYTES:
            errors.append(f"MM{i}: native weight payload differs from {WEIGHT_BYTES} bytes")
        if len(o["cycles"]["r_cpl"]) != ADVANCES:
            errors.append(f"MM{i}: scale entry count differs from 36")
        if len(o["cycles"]["compute"]) != ADVANCES or len(o["cycles"]["rf_push"]) != 4:
            dependencies.append(f"MM{i}: expected 36 compute pops and four result rows")
        # This is a conservative proof boundary. It asserts that the owner has
        # no more requests and that every already-accepted native fill returned.
        last_issue = max(o["cycles"]["b_issue"] + o["cycles"]["a_issue"], default=None)
        last_response = max(o["cycles"]["native_weight_response"] + o["cycles"]["native_scale_response"], default=None)
        owner_pending = any(credit["owner"] is not None and credit["owner"][1] == i
                            for channel_queue in queues for credit in channel_queue)
        if (last_issue is not None and last_response is not None and not owner_pending
                and len(o["cycles"]["b_issue"]) == ADVANCES
                and len(o["cycles"]["r_cpl"]) == ADVANCES
                and o["native_weight_valid_beats"] == WEIGHT_BYTES // 64):
            o["page_safe_release"] = max(last_issue, last_response)
        o["phases"] = {name: _first_last(seq) for name, seq in o["cycles"].items()}
        first_compute = o["phases"]["compute"]["first"]
        o["activation_ready_qact_done"] = max((t for t in q_done if first_compute is not None
                                                 and t <= first_compute), default=None)
        o["bubble_cycles"] = {
            "start_to_first_b_issue": _gap(o["allocated_at_start"], o["phases"]["b_issue"]["first"]),
            "last_b_issue_to_last_b_push": _gap(o["phases"]["b_issue"]["last"], o["phases"]["b_push"]["last"]),
            "last_arrival_to_first_compute": _gap(max((x for x in (o["phases"]["b_push"]["last"], o["phases"]["r_cpl"]["last"]) if x is not None), default=None), o["phases"]["compute"]["first"]),
            "last_compute_to_first_drain": _gap(o["phases"]["compute"]["last"], o["phases"]["drain_go"]["first"]),
            "last_d_pop_to_c_done": _gap(o["phases"]["d_pop"]["last"], o["output_committed_c_done"]),
            "c_done_to_udone": _gap(o["output_committed_c_done"], o["udone"]),
        }
        if o["activation_ready_qact_done"] is None:
            o["missing_instrumentation"].append("QACT done preceding this MM start")
        if (o["allocated_at_start"] is not None and o["phases"]["b_issue"]["first"] is not None
                and o["phases"]["b_issue"]["first"] < o["allocated_at_start"]):
            dependencies.append(f"MM{i}: B issue before allocation/start")
        if (o["phases"]["b_push"]["first"] is not None and o["phases"]["b_issue"]["first"] is not None
                and o["phases"]["b_push"]["first"] <= o["phases"]["b_issue"]["first"]):
            dependencies.append(f"MM{i}: B completion before an earlier issue")
        if (o["phases"]["r_cpl"]["first"] is not None and o["phases"]["a_issue"]["first"] is not None
                and o["phases"]["r_cpl"]["first"] <= o["phases"]["a_issue"]["first"]):
            dependencies.append(f"MM{i}: scale completion before an earlier issue")
        if (o["output_committed_c_done"] is not None
                and any(t is not None and t > o["output_committed_c_done"] for t in
                        (o["weights_ready_all_b_push"], o["scales_ready_all_r_cpl"]))):
            dependencies.append(f"MM{i}: output committed before all input entries ready")
        if (o["udone"] is not None and o["output_committed_c_done"] is not None
                and o["udone"] < o["output_committed_c_done"]):
            dependencies.append(f"MM{i}: unit done before c_done")
        if (o["activation_ready_qact_done"] is not None and o["phases"]["compute"]["first"] is not None
                and o["phases"]["compute"]["first"] < o["activation_ready_qact_done"]):
            dependencies.append(f"MM{i}: compute before QACT completion")
    last_old = owners[0]["phases"]["native_weight_request"]["last"]
    first_new = owners[1]["phases"]["native_weight_request"]["first"]
    return {
        "scope": "oracle-only two-MM RL4 W8 N13 KB9; no fitted correction or TPS claim",
        "cycles": len(slices), "owners": owners, "ports": port,
        "port_counter_scope": "whole preserved trace, including boot and non-MM traffic",
        "page_safe_release_scope": "last native response after all owner issues; no buffer reuse/overwrite exercised",
        "activation_scope": "latest preceding QACT done; fixture has exactly one QACT and no activation overwrite",
        "native_payload_bytes": sum(o["native_weight_payload_bytes"] + o["native_scale_payload_bytes"] for o in owners),
        "native_valid_beats": sum(o["native_weight_valid_beats"] + o["native_scale_valid_beats"] for o in owners),
        "native_read_ledger": ledger,
        "mxu_active_cycles": mxu_active, "peaks": peaks,
        "old_last_to_new_first_native_weight_read_gap": _gap(last_old, first_new),
        "collective_completion_cycles": collective_done,
        "predicates": {"ok": not errors and not dependencies,
                       "violations": errors, "dependency_violations": dependencies,
                       "all_traffic_accounted": not any(queues) and not b_entries and not a_entries,
                       "weight_bytes_match": all(o["weight_issue_bytes"] == WEIGHT_BYTES for o in owners),
                       "native_weight_bytes_match": all(o["native_weight_payload_bytes"] == WEIGHT_BYTES for o in owners),
                       "scale_issue_bytes_match": all(o["scale_issue_bytes"] == ADVANCES * 4 for o in owners),
                       "scale_entries_match": all(len(o["cycles"]["r_cpl"]) == ADVANCES for o in owners)},
        "calibrated": False, "tok_s": None,
    }
