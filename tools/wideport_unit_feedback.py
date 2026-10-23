"""Structural feedback states checked at the slice's pre-core-edge boundary.

The optional narrow producer mode generates V_ADD address/mask/completion and
RL4 int8 result-valid pipelines from commands and coupled unit enables/pop.
Other-unit requests, operand/DRAM arrivals and command dispatch remain trace
inputs. This is not an independent whole-program predictor or cycle fit.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import struct

from tools.wideport_tmem_resource import TmemReplayConfig, TmemReplayResource, WriteLandingPipeline
from tools.wideport_vadd_producer import VAddCommand, VAddProducer
from tools.wideport_mxu_producer import MxuRowProducer
from opentpu import isa as I


def require(got, expected, field, cycle):
    if got != expected:
        raise ValueError(f"cycle {cycle} {field}: observed {got}, modeled {expected}")


class VpuWriteFeedback:
    """Four-slot SE WBUF, registered lookahead/hold and completion queue."""
    def __init__(self):
        self.queue = deque()
        self.en_r = self.en_q = self.en_h = self.ren_r = self.done = self.dpend = 0

    def check_and_step(self, e, cycle, rhold, gnt):
        if e["rst"]:
            self.__init__()
            return
        state = {"wn": len(self.queue), "en_r": self.en_r, "en_q": self.en_q,
                 "en_h": self.en_h, "ren_r": self.ren_r, "done": self.done, "dpend": self.dpend}
        for key, value in state.items():
            require(e[key], value, "vpu." + key, cycle)
        en = int(bool(self.en_r and not self.en_h))
        pdn = int(bool(e["done_i"] or self.dpend))
        push = int(bool(en and (e["cw_words"] or pdn)))
        pop = int(bool(self.queue and gnt))
        for key, value in (("en", en), ("ren", self.ren_r), ("push", push), ("pop", pop),
                           ("rhold", rhold), ("gnt", gnt)):
            require(e[key], value, "vpu." + key, cycle)
        head_words, head_done = self.queue[0] if self.queue else ([], 0)
        require(e["head_words"], head_words, "vpu.head_words", cycle)
        if self.queue:
            require(e["head_done"], head_done, "vpu.head_done", cycle)
        self.done = int(bool(pop and head_done))
        if pop:
            self.queue.popleft()
        if push:
            self.queue.append((e["cw_words"], pdn))
        if len(self.queue) > 4:
            raise ValueError(f"cycle {cycle}: VPU write buffer overflow")
        next_q = int(bool(e["ss_pe"] if e["ss_act_d"] else len(self.queue) < 2 and e["red_thr"]))
        self.en_r, self.ren_r = self.en_q, int(bool(self.en_q and not rhold))
        self.en_q, self.en_h, self.dpend = next_q, rhold, int(bool(pdn and not en))


class MxuQueueFeedback:
    """Operand/scale/result FIFO and independent row-compute/drain occupancy."""
    def __init__(self):
        self.rf_n = self.rf_nz = self.rows_live = self.rows_p = self.rl_nz = 0
        self.f_count = self.s_count = 0

    def compute_pop(self, e):
        return int(bool(e["more"] and e["f_av"] and (e["c_unit"] or e["s_av"])
                        and (not e["first_row"] or e["row_in"]
                             and self.rows_live + self.rows_p < e["rf_capacity"])))

    def check_and_step(self, e, cycle, t_gnt):
        if e["rst"]:
            self.__init__()
            return
        for key in ("rf_n", "rf_nz", "rows_live", "rows_p", "rl_nz", "f_count", "s_count"):
            require(e[key], getattr(self, key), "mxu." + key, cycle)
        drain = int(bool(self.rf_nz and self.rl_nz and (not e["c_asc"] or e["al_st"] == 2)))
        d_pop = int(bool(t_gnt and drain and e["d_fin"]))
        pop = self.compute_pop(e)
        for key, value in (("drain_go", drain), ("d_pop", d_pop), ("pop", pop), ("t_gnt", t_gnt)):
            require(e[key], value, "mxu." + key, cycle)
        first = int(bool(pop and e["first_row"]))
        next_p = self.rows_p + int(bool(first and e["pn"]))
        next_live = self.rows_live + int(bool(first and not e["pn"])) - d_pop
        next_rl_nz = int(bool(first and not e["pn"] or self.rl_nz and not (self.rows_live == 1 and d_pop)))
        if e["c_done"] and e["pn"]:
            next_live, next_rl_nz, next_p = next_p, int(next_p != 0), 0
        self.rf_nz = int(bool(e["rf_push"] or self.rf_nz and not (self.rf_n == 1 and d_pop)))
        self.rf_n += e["rf_push"] - d_pop
        self.rows_live, self.rows_p, self.rl_nz = next_live, next_p, next_rl_nz
        self.f_count += e["b_push"] - e["fpop"]
        self.s_count += e["r_cpl"] - int(bool(e["fpop"] and not e["c_unit"]))
        if min(self.rf_n, self.rows_live, self.rows_p, self.f_count, self.s_count) < 0:
            raise ValueError(f"cycle {cycle}: negative MXU queue occupancy")


def slice_grants(raw_reads, raw_writes, frozen, ren, dram, coll_gnt):
    """Exact ARB_MASK priority/guest/hold contract for this GST single-slice case."""
    # Group order: DMA, COLL, VPU, MXU, QUANT. Write ports: DMA, MXU, COLL, VPU.
    wports = (0, 2, 3, 1, None)
    masks = [set(word % 8 for word in raw_writes[p]) if p is not None else set() for p in wports]
    frz = [0, 0, 0, int(bool(frozen & 1)), int(bool(frozen & ((1 << 2) | (1 << 5) | (1 << 6))))]
    guest_q = bool(raw_reads[5] and raw_reads[1])
    guest_c = bool(raw_reads[7] and (raw_reads[1] or raw_reads[5]))
    grant = []
    for g in range(5):
        ok = not frz[g] and not any(grant[h] and masks[g] & masks[h] for h in range(g))
        if g == 4 and (guest_q or dram["q_want"] and not dram["write_ready"]):
            ok = False
        if g == 1 and guest_c:
            ok = False
        grant.append(int(ok))
    grant[1] = int(bool(coll_gnt))  # global collective decision is external to one slice
    filtered = [list(words) for words in raw_reads]
    for p in (0, 2, 5, 6):
        if frz[3 if p == 0 else 4]:
            filtered[p] = []
    if not ren:
        filtered[3] = filtered[4] = []
    if guest_c:
        filtered[7] = []
    rgroups = (3, 0, 4, 2, 2, 4, 4, 1)
    taken = [int(bool(raw_reads[p] and (ren if p in (3, 4) else grant[rgroups[p]]))) for p in range(8)]
    issued = [word for p, g in enumerate((0, 3, 1, 2)) if grant[g] for word in raw_writes[p]]
    return grant, frz, filtered, taken, issued


def validate_feedback_trace(path, *, predict_producers=False):
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    header, end = records[0], records[-1]
    if header.get("kind") != "config" or header.get("schema") != 1 or end.get("kind") != "end":
        raise ValueError("missing config/end or truncated feedback trace")
    # The slice trace's compact header is supplemented by the runner's actual
    # top parameters. In particular the zero COLL replay hold assumes GST.
    manifest = json.loads(Path(path).with_name("feedback-inputs.json").read_text())
    params = manifest["parameters"]
    expected = {"S": 1, "LANES": 8, "TMEM_IMPL": 3, "TMEM_NSB": 2, "TMEM_SBP": 3,
                "TMEM_RPL": 1, "TMEM_TDP": 1, "TMEM_GST": 1, "WPB": 1,
                "TMEM_PMP": header["pmp"], "TMEM_WORDS": header["words"]}
    for key, value in expected.items():
        require(params.get(key), value, "config." + key, -1)
    for key, default in (("TMEM_SBH", 0), ("TMEM_PRK", 0)):
        require(params.get(key, default), default, "config." + key, -1)
    if params.get("RPB", 0) < 64 or not header["vpu_se"]:
        raise ValueError("feedback requires the board ARB_MASK/SE configuration")
    side = params.get("TMEM_SDM", -1)
    side = side if side >= 0 else (0xF0 if header["pmp"] == 0 else 0x0C)
    require(header["side_mask"], side, "config.side_mask", -1)
    events = records[1:-1]
    if end["cycles"] != len(events) or not events:
        raise ValueError("feedback trace length mismatch")
    bank = TmemReplayResource(TmemReplayConfig(pmp=header["pmp"], words=header["words"], side_mask=header["side_mask"]))
    writes, vpu, mxu = WriteLandingPipeline(), VpuWriteFeedback(), MxuQueueFeedback()
    producer, commands, command_index, row_producer = VAddProducer(), [], 0, None
    if predict_producers:
        # Read the exact saved instruction image, not reconstructed workload
        # constants. Its little-endian words must match the runner's receipt.
        words = [int(line, 16) for line in Path(path).with_name("prog_0.hex").read_text().splitlines()]
        # rtlsim._write_hex omits trailing zero words; restore the final
        # instruction's zeros before checking the original assembled image.
        words += [0] * (-len(words) % 8)
        digest = hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest()
        require(digest, manifest["program_sha256"], "program.sha256", -1)
        commands = [VAddCommand.from_words(words[i:i+8], tmem_words=header["words"])
                    for i in range(0, len(words), 8) if words[i] & 255 == 0x30]
        if not commands:
            raise ValueError("producer validation requires a V_ADD command")
        row_producer = MxuRowProducer(
            [I.Instr.decode(words[i:i+8]) for i in range(0, len(words), 8) if words[i] & 255 == I.MM],
            d=params["D"], mcols=params["MCOLS"], cl=params.get("MXU_CL", 16))
    counts = {"cycles_checked": 0, "replay_port_cycles": 0, "vpu_hold_cycles": 0,
              "mxu_compute_while_drain_denied": 0, "mxu_result_push_while_drain_denied": 0,
              "vpu_queued_writes": 0, "mxu_max_result_rows": 0}
    for cycle, e in enumerate(events):
        require(e.get("kind"), "cycle", "record kind", cycle)
        require(e["cycle"], cycle, "cycle", cycle)
        frozen = bank._mask([p is not None for p in bank.pending])
        raw_reads, raw_writes = e["raw_reads"], e["raw_writes"]
        vpu_event = e["vpu"]
        if predict_producers:
            predicted = producer.snapshot()
            if not vpu_event["rst"]:
                for key in ("cw_words", "done_i", "red_thr"):
                    require(vpu_event[key], predicted[key], "producer.vpu." + key, cycle)
                require(vpu_event["ss_act_d"], 0, "producer.vpu.no_stream", cycle)
                require(raw_reads[3], predicted["ta_words"], "producer.vpu.read_a", cycle)
                require(raw_reads[4], predicted["tb_words"], "producer.vpu.read_b", cycle)
                head = list(vpu.queue[0][0]) if vpu.queue else []
                require(raw_writes[3], head, "producer.vpu.queued_writes", cycle)
                raw_reads = [list(w) for w in raw_reads]
                raw_reads[3], raw_reads[4] = predicted["ta_words"], predicted["tb_words"]
                raw_writes = [list(w) for w in raw_writes]
                raw_writes[3] = head
                vpu_event = {**vpu_event, **{key: predicted[key] for key in ("cw_words", "done_i", "red_thr")},
                             "ss_act_d": 0, "ss_pe": 0}
            start = None
            if e["start"] & (1 << 3):
                if command_index >= len(commands):
                    raise ValueError(f"cycle {cycle}: unexpected VPU start")
                start = commands[command_index]
                command_index += 1
            producer.step(start=start, en=bool(vpu.en_r and not vpu.en_h), reset=bool(vpu_event["rst"]))
        grant, frz, filtered, taken, issued = slice_grants(
            raw_reads, raw_writes, frozen, vpu.ren_r, e["dram"], e["coll_gnt"])
        require(e["gnt"], bank._mask(grant), "slice.gnt", cycle)
        require(e["rpl_frz"], bank._mask(frz), "slice.rpl_frz", cycle)
        require(e["rpl_hold"], bank._mask(frz), "slice.rpl_hold", cycle)
        require(e["reads"], filtered, "slice.filtered_reads", cycle)
        require(e["taken"], taken, "slice.taken", cycle)
        require(sorted(e["issued_writes"]), sorted(issued), "slice.issued_writes", cycle)
        require(sorted(e["writes"]), sorted(writes.step(issued)), "tmem.write_pipeline", cycle)
        result = bank.step(filtered, taken, e["writes"])
        for key in ("lost", "frozen", "blocked", "self_blocked", "write_conflict"):
            require(e[key], result[key], "tmem." + key, cycle)
        blockers = sum(1 << (p * 8 + other) for p, row in enumerate(result["read_blockers"]) for other in row)
        require(e["read_blockers"], blockers, "tmem.read_blockers", cycle)
        rhold = int(bool(result["lost"] & ((1 << 3) | (1 << 4))))
        vpu.check_and_step(vpu_event, cycle, rhold, grant[2])
        mxu_event = e["mxu"]
        if row_producer is not None:
            if not mxu_event["rst"]:
                require(mxu_event["q_n"], len(row_producer.accepted), "producer.mxu.q_n", cycle)
                require(mxu_event["cl_ld"], int(row_producer.loading), "producer.mxu.cl_ld", cycle)
                if row_producer.loading:
                    require(mxu_event["more"], 0, "producer.mxu.loading_more", cycle)
                prerequisites = row_producer.eligibility(f_count=mxu.f_count, s_count=mxu.s_count, pn=mxu_event["pn"])
                for key, value in prerequisites.items():
                    require(mxu_event[key], value, "producer.mxu." + key, cycle)
                mxu_event = {**mxu_event, **prerequisites}
                if prerequisites:
                    pop = mxu.compute_pop(mxu_event)
                    require(mxu_event["fpop"], pop, "producer.mxu.fpop", cycle)
                    mxu_event = {**mxu_event, "fpop": pop}
            # This slice contract starts cold and never resets live work.
            if mxu_event["rst"] and (mxu_event["pop"] or row_producer.accepted or any(row_producer.stages)):
                raise ValueError(f"cycle {cycle}: warm MXU reset is outside the slice producer contract")
            rf_push = row_producer.step(pop=bool(mxu.compute_pop(mxu_event)),
                                       start=bool(e["start"] & (1 << 1)), done=mxu_event["c_done"],
                                       pn=mxu_event["pn"], rst=mxu_event["rst"])
            require(mxu_event["rf_push"], rf_push, "producer.mxu.rf_push", cycle)
            mxu_event = {**mxu_event, "rf_push": rf_push}
        mxu.check_and_step(mxu_event, cycle, grant[3])
        counts["cycles_checked"] += 1
        counts["replay_port_cycles"] += result["lost"].bit_count()
        counts["vpu_hold_cycles"] += rhold
        counts["mxu_compute_while_drain_denied"] += bool(e["mxu"]["pop"] and not grant[3])
        counts["mxu_result_push_while_drain_denied"] += bool(e["mxu"]["rf_push"] and not grant[3])
        counts["vpu_queued_writes"] = max(counts["vpu_queued_writes"], e["vpu"]["wn"])
        counts["mxu_max_result_rows"] = max(counts["mxu_max_result_rows"], e["mxu"]["rf_n"])
    if predict_producers:
        require(command_index, len(commands), "producer.vpu.started_commands", len(events))
        row_producer.finish()
    return {**counts, "mismatches": 0, "whole_program_predictor": False, "calibrated": False, "tok_s": None,
            "command_driven_vadd": predict_producers,
            "command_driven_mxu_row_valid": predict_producers,
            "externally_supplied_signals": (
                ["reset and accepted unit starts", "MXU c_done/pn and more (sequencer release/head state)",
                 "inactive/load-edge MXU prerequisite flags (unused while more=0)",
                 "MXU b_push/r_cpl (operand/scale arrivals)", "MXU d_fin/c_asc/al_st (drain epilogue)",
                 "DMA/MXU/COLL/QUANT raw TMEM reads/writes", "DRAM q_want/write_ready", "collective grant"] if predict_producers else
                ["all raw unit requests", "VPU cw_words/done_i/stream state", "MXU rf_push",
                 "operand/instruction state", "DRAM readiness", "collective grant"])}
