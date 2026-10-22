"""Bounded command -> native memory -> MXU arrivals, checked against RTL.

No observed ready/response/arrival event schedules a newly modeled path.
Dispatch, consumer release/drain state, TMEM drain grants, and DMA/loader
requests remain declared external inputs. The adapter's write/SW path is not
implemented: the read-only comparison must finish all MMs before its first
write. It never silently substitutes observed responses after that boundary.
"""
import hashlib
import json
from pathlib import Path
import struct

from opentpu import isa as I
from tools.wideport_memory_feedback import native_config, read_trace, require
from tools.wideport_native_memory import NativeCommand, NativeInput, NativeMemory, logical_beat
from tools.wideport_native_adapter import ARead, BRead, NativeAdapter, NativePins
from tools.wideport_mxu_issue import MxuIssueModel
from tools.wideport_mxu_producer import MxuRowProducer
from tools.wideport_unit_feedback import MxuQueueFeedback
from tools.wideport_pipeline_ownership import PipelineOwnership


def validate_memory_coupled(directory, *, mm_prefix=None):
    root = Path(directory)
    header, native_events = read_trace(root / "native-feedback.jsonl")
    ah, adapter_events = read_trace(root / "adapter-feedback.jsonl")
    _, slice_events = read_trace(root / "tmfeedback.jsonl")
    if len(native_events) != len(adapter_events) or len(native_events) < len(slice_events):
        raise ValueError("native/adapter lengths differ or omit slice cycles")
    if {k: ah[k] for k in ("channels", "portb", "aline", "qd", "rd", "ad", "apf", "apr")} != {
            "channels": 8, "portb": 4, "aline": 64, "qd": 16, "rd": 128, "ad": 64, "apf": 32, "apr": 16}:
        raise ValueError("unsupported native adapter contract")
    manifest = json.loads((root / "feedback-inputs.json").read_text())
    words = [int(line, 16) for line in (root / "prog_0.hex").read_text().splitlines()]
    words += [0] * (-len(words) % 8)
    digest = hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest()
    require(digest, manifest["program_sha256"], "program.sha256", -1)
    mms = [I.Instr.decode(words[i:i+8]) for i in range(0, len(words), 8) if words[i] & 255 == I.MM]
    total_mms = len(mms)
    if mm_prefix is not None:
        if not isinstance(mm_prefix, int) or not 0 < mm_prefix < total_mms:
            raise ValueError("MM prefix must exclude a nonempty command suffix")
        mms = mms[:mm_prefix]
    page_reads = {str(p): {"weight": 0, "scale": 0} for p in range(2)}
    issue, rows, queues = MxuIssueModel(mms), MxuRowProducer(mms), MxuQueueFeedback()
    mm_pcs = [i // 8 for i in range(0, len(words), 8) if words[i] & 255 == I.MM][:len(mms)]
    ownership = (PipelineOwnership(mms, depth=manifest["parameters"]["FIFO_DEPTH"],
                                   cl=manifest["parameters"].get("MXU_CL", 16), program_pcs=mm_pcs)
                 if "ownership" in slice_events[0] else None)
    memory, adapter = NativeMemory(native_config(header)), NativeAdapter()
    a_reg, b_reg, b_in = None, None, None
    aid = bid = 0
    counts = dict(cycles_checked=0, a_requests=0, b_requests=0, native_commands=0,
                  native_responses=0, b_push=0, r_cpl=0)
    stop = None
    traffic_seen = False
    mm_started = False
    for cycle, se in enumerate(slice_events):
        ne, ae = native_events[cycle], adapter_events[cycle]
        for name, event in (("native", ne), ("adapter", ae), ("slice", se)):
            require(event["cycle"], cycle, name + ".cycle", cycle)
        if ae["sw_req"]:
            raise ValueError(f"cycle {cycle}: SW traffic is unsupported")
        # This explicit boundary excludes writes from the scoped adapter.
        if ae["b_req"] and ae["b_we"]:
            stop = cycle
            break
        if ae["rst"]:
            if traffic_seen:
                raise ValueError(f"cycle {cycle}: reset after memory traffic is outside the cold-reset contract")
            adapter = NativeAdapter()
            adapter.cycle = cycle
            a_reg = b_reg = b_in = None
        require(ae["rst"], ne["rst"], "adapter/native reset agreement", cycle)
        require(ae["a_req"], int(a_reg is not None), "slice.a_register.valid", cycle)
        require(ae["b_req"], int(b_reg is not None), "slice.b_register.valid", cycle)
        if a_reg:
            require(ae["a_addr"], a_reg.word_addr, "slice.a_register.address", cycle)
        if b_reg:
            for key, predicted in (("b_addr", b_reg.word_addr), ("b_n", b_reg.chunks - 1), ("b_tag", int(b_reg.tag))):
                require(ae[key], predicted, "slice." + key, cycle)
        ns = memory.snapshot()
        # These independent memory outputs are the adapter's only channel inputs.
        pins = NativePins(ns.command_ready, tuple(tag if valid else None for tag, valid in zip(ns.read_tag, ns.read_valid)))
        ar = adapter.step(pins, a=a_reg, b=b_reg, a_flush=bool(ae["a_flush"]))
        for key, predicted in (("a_ready", ar.a_ready), ("b_ready", ar.b_ready), ("a_rvalid", ar.a_response_id is not None),
                               ("b_rvalid", ar.b_response is not None)):
            require(ae[key], int(predicted), "adapter." + key, cycle)
        if ar.b_response:
            for key, predicted in (("b_rtag", int(ar.b_response.tag)), ("b_rcm", ar.b_response.chunk_mask), ("b_rlast", int(ar.b_response.last))):
                require(ae[key], predicted, "adapter." + key, cycle)
        commands = {command.channel: command for command in ar.native_commands}
        native_inputs = []
        for c, observed in enumerate(ne["channels"]):
            cmd = commands.get(c)
            for key, predicted in (("cr", ns.command_ready[c]), ("wr", ns.write_ready[c]), ("rv", ns.read_valid[c]),
                                   ("wdone", ns.write_done[c]), ("cv", cmd is not None), ("wv", False)):
                require(observed[key], int(predicted), f"native.ch{c}." + key, cycle)
            if cmd:
                require(observed["we"], 0, f"native.ch{c}.read_only", cycle)
                require(observed["addr"], cmd.beat_addr, f"native.ch{c}.address", cycle)
            native_inputs.append(NativeInput(NativeCommand(False, cmd.beat_addr, cmd.id) if cmd else None))
        sources, ie, me = se["memory_sources"], se["issue"], se["mxu"]
        if me["rst"] and mm_started:
            raise ValueError(f"cycle {cycle}: warm MXU reset is outside the bounded issuer contract")
        mm_started |= bool(se["start"] & 2)
        snap = issue.snapshot()
        a_gnt = a_reg is None or ar.a_ready
        b_ready_i = b_reg is None or ar.b_ready
        b_gnt = not sources["dma_req"] and b_ready_i
        require(ie["a_gnt"], int(a_gnt), "issue.a_gnt", cycle)
        require(ie["b_gnt"], int(b_gnt), "issue.b_gnt", cycle)
        for key, predicted in (("i_ini", snap.initializing), ("i_act", snap.active), ("want", snap.want),
                               ("rv1", snap.rv1), ("tg_n", snap.tg_n), ("occ", snap.occ)):
            require(ie[key], int(predicted), "issue." + key, cycle)
        # Consumer occupancy and admission are coupled to generated arrivals.
        fields = {**me}
        if not me["rst"]:
            if rows.loading:
                require(me["more"], 0, "consumer.command_cache_load_more", cycle)
            fields.update(rows.eligibility(f_count=queues.f_count, s_count=queues.s_count, pn=me["pn"]))
        pop = bool(queues.compute_pop(fields)) if not me["rst"] else False
        b_push = bool(b_in and not b_in.tag and b_in.last)
        r_cpl = bool(snap.r_cpl)
        for key, predicted in (("b_push", b_push), ("r_cpl", r_cpl)):
            require(me[key], int(predicted), "arrivals." + key, cycle)
        rf_push = rows.step(pop=pop, start=bool(se["start"] & 2), done=bool(me["c_done"]), pn=bool(me["pn"]), rst=bool(me["rst"]))
        require(me["rf_push"], rf_push, "consumer.result_valid", cycle)
        fields.update(b_push=int(b_push), r_cpl=int(r_cpl), rf_push=rf_push)
        if rows.accepted or pop:
            require(me["fpop"], int(pop), "consumer.fpop", cycle)
            fields["fpop"] = int(pop)
        drain_pop = bool(me["t_gnt"] and queues.rf_nz and queues.rl_nz and
                         (not me["c_asc"] or me["al_st"] == 2) and me["d_fin"])
        queues.check_and_step(fields, cycle, me["t_gnt"])
        require(ie["a_rvalid"], int(ar.a_response_id is not None), "issue.a_rvalid", cycle)
        require(ie["b_rvalid"], int(bool(b_in and not b_in.tag)), "issue.b_rvalid", cycle)
        issued = issue.step(start=bool(se["start"] & 2), a_gnt=a_gnt, b_gnt=b_gnt,
                            a_rvalid=ar.a_response_id is not None, b_response=b_push,
                            fpop=bool(fields["fpop"]), scale_retire=bool(fields["fpop"] and not fields["c_unit"]), rst=bool(me["rst"]))
        require(ie["a_req"], int(issued.a_req), "issue.a_req", cycle)
        require(ie["b_req"], int(issued.b_req), "issue.b_req", cycle)
        if issued.a_req:
            require(ie["a_addr"], issued.a_addr, "issue.a_addr", cycle)
            for key, predicted in (("tag_ent", issued.tag.ent), ("tag_cpl", issued.tag.cpl)):
                require(ie[key], int(predicted), "issue." + key, cycle)
        if issued.b_req:
            require(ie["b_addr"], issued.b_addr, "issue.b_addr", cycle)
            require(ie["b_n"], issued.b_n, "issue.b_n", cycle)
        if ownership is not None:
            ownership.step(se, b_issue=issued.b_req,
                           scale_issue=bool(issued.a_req and issued.tag.ent and issued.tag.cpl),
                           b_push=b_push, r_cpl=r_cpl, pop=pop, rf_push=rf_push,
                           d_pop=drain_pop, cycle=cycle)
        # Slice request and response registers advance after this cycle's use.
        if ae["rst"]:
            a_reg = b_reg = b_in = None
        else:
            if a_gnt:
                a_reg = ARead(aid, issued.a_addr) if issued.a_req else None
                aid += int(issued.a_req)
            if b_ready_i:
                if sources["loader_busy"]:
                    b_reg = BRead(bid, sources["loader_addr"], 1, True) if sources["loader_req"] else None
                elif sources["dma_req"]:
                    if sources["dma_we"]:
                        # A write enters the slice here and is the boundary
                        # at the next cycle, before the adapter sees it.
                        stop = cycle + 1
                        b_reg = None
                    else:
                        b_reg = BRead(bid, sources["dma_addr"], 1, True)
                else:
                    b_reg = BRead(bid, issued.b_addr, issued.b_n + 1, False) if issued.b_req else None
                bid += int(b_reg is not None)
            b_in = ar.b_response
        edge = memory.step(native_inputs, reset=bool(ne["rst"]))
        if mm_prefix is not None:
            from tools.wideport_credit_budget import classify_byte
            for channel, command in enumerate(edge.accepted_commands):
                if command is not None:
                    key = classify_byte(64 * logical_beat(command.beat, channel, chash=bool(header["chash"])))
                    if key is not None:
                        kind, page = key
                        page_reads[str(page)][kind] += 1
        traffic_seen |= any(cmd is not None for cmd in edge.accepted_commands)
        counts["cycles_checked"] += 1
        counts["a_requests"] += int(issued.a_req)
        counts["b_requests"] += int(issued.b_req)
        counts["native_commands"] += sum(cmd is not None for cmd in edge.accepted_commands)
        counts["native_responses"] += sum(ns.read_valid)
        counts["b_push"] += int(b_push)
        counts["r_cpl"] += int(r_cpl)
    if stop is None:
        raise ValueError("bounded adapter comparison did not reach its explicit first-write boundary")
    issue.finish()
    rows.finish()
    ownership_result = {"ownership": ownership.finish()} if ownership is not None else {}
    prefix_result = ({"command_scope": f"first {mm_prefix} MM commands of {total_mms}; suffix excluded",
                      "predicted_initial_page_native_reads": page_reads} if mm_prefix is not None else {})
    return {**counts, **ownership_result, **prefix_result, "first_write_boundary": stop, "all_mm_issuers_and_arrivals_drained": True,
            "mismatches": 0, "whole_program_predictor": False, "calibrated": False, "tok_s": None,
            "external_inputs": ["reset/a_flush and accepted MM starts", "MXU c_done/pn/more and drain epilogue",
                                "TMEM drain grant", "DMA/loader requests and addresses"],
            "excluded": ["adapter writes/SW and post-first-write interval", "DMA/loader request generation",
                         "independent sequencer and drain epilogue"]}
