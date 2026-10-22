"""Three-MM fixture with command-driven ST, native B writes and A invalidation.

New store/native request, completion and cache-response timing is generated.
Accepted command starts, MXU release/drain state and initial load/loader
requests remain explicit frontend inputs. Functional ST source words come
from the small ISA oracle, never observed RTL payload. No TPS calibration.
"""
import hashlib
import json
from pathlib import Path
import struct

from opentpu import isa as I
from tools.wideport_memory_feedback import native_config, read_trace, require
from tools.wideport_native_memory import NativeCommand, NativeInput, NativeMemory, logical_beat
from tools.wideport_native_adapter import ARead, BRead, BWrite, NativeAdapter, NativePins
from tools.wideport_dma_store import StoreCommand, DmaStore
from collections import deque
from copy import deepcopy
from tools.wideport_mxu_issue import MxuIssueModel
from tools.wideport_mxu_producer import MxuRowProducer
from tools.wideport_unit_feedback import MxuQueueFeedback
from tools.wideport_pipeline_ownership import PipelineOwnership
from tools.wideport_credit_budget import WB, SB, classify_byte


def validate_write_coupled(directory, *, require_adapter_payload=True,
                           native_oracles=None, adapter_oracles=None, slice_oracles=None):
    root = Path(directory)
    header, native_events = read_trace(root / "native-feedback.jsonl")
    ah, adapter_events = read_trace(root / "adapter-feedback.jsonl")
    _, slice_events = read_trace(root / "tmfeedback.jsonl")
    native_events = native_events if native_oracles is None else native_oracles
    adapter_events = adapter_events if adapter_oracles is None else adapter_oracles
    slice_events = slice_events if slice_oracles is None else slice_oracles
    if len(native_events) != len(adapter_events) or len(native_events) < len(slice_events):
        raise ValueError("native/adapter lengths differ or omit slice cycles")
    if {k: ah[k] for k in ("channels", "portb", "aline", "qd", "rd", "ad", "apf", "apr")} != {
            "channels": 8, "portb": 4, "aline": 64, "qd": 16, "rd": 128, "ad": 64, "apf": 32, "apr": 16}:
        raise ValueError("unsupported native adapter contract")
    manifest = json.loads((root / "feedback-inputs.json").read_text())
    expected_parameters = dict(S=1,D=128,MCOLS=8,ACT_BLOCKS=16,ACT_ROWS=16,TMEM_WORDS=65536,
                               LANES=8,WIN=16,FIFO_DEPTH=1024,TMEM_PMP=0,MXU_E=4,
                               NCH=8,PORTB=4,RL=8,ALINE=64,MXU_IMPL=2,TMEM_IMPL=3,
                               TMEM_NSB=2,TMEM_RPL=1,TMEM_TDP=1,TMEM_GST=1,SEQ_WC=1)
    for key,value in expected_parameters.items():
        require(manifest["parameters"].get(key),value,"parameters."+key,-1)
    require(manifest["parameters"]["DRAM_WORDS"],header["words"],"parameters.DRAM_WORDS",-1)
    for key in ("seed","stall","lat"):
        require(header[key],manifest[key],"memory."+key,-1)
    words = [int(line, 16) for line in (root / "prog_0.hex").read_text().splitlines()]
    words += [0] * (-len(words) % 8)
    digest = hashlib.sha256(b"".join(struct.pack("<I", w) for w in words)).hexdigest()
    require(digest, manifest["program_sha256"], "program.sha256", -1)
    mms = [I.Instr.decode(words[i:i+8]) for i in range(0, len(words), 8) if words[i] & 255 == I.MM]
    program = [I.Instr.decode(words[i:i+8]) for i in range(0, len(words), 8)]
    if len(mms) != 3 or "epoch2" not in manifest:
        raise ValueError("write coupling requires the bounded three-MM reuse fixture")
    require([m.w[0] for m in mms],[WB[0],WB[1],WB[0]],"fixture.weight_pages",-1)
    require([m.w[1] for m in mms],[SB[0],SB[1],SB[0]],"fixture.scale_pages",-1)
    require(manifest["epoch2"]["weight_page"],WB[0],"fixture.weight_patch",-1)
    require(manifest["epoch2"]["scale_page"],SB[0],"fixture.scale_patch",-1)
    import numpy as np
    from opentpu import Config
    from opentpu.isasim import Machine
    image = np.fromfile(root / "dram_0.bin", dtype=">u4").astype("<u4").view(np.uint8)
    # Boot appends instruction/wait loader storage beyond the functional DRAM
    # region. Pin exactly the original modeled image, not that boot overlay.
    require(hashlib.sha256(image[:1<<20].tobytes()).hexdigest(),manifest["dram_sha256"],"dram.sha256",-1)
    content = bytearray(image.tobytes())
    content.extend(bytes(header["words"] * 4 - len(content)))
    cfg = Config(S=1,D=128,MCOLS=8,ACT_BLOCKS=16,ACT_ROWS=16,PAIR=True,DSTEP=True,
                 PORTB=4,ROWLANES=True,ALINE=64,DRAM_BYTES=1<<20)
    functional = Machine(cfg,[program],[image[:1<<20].copy()]).run().slices[0]
    oracle = functional.tmem
    dma_commands = [ins for ins in program if ins.op in (I.LD,I.ST)]
    if [i.op for i in dma_commands] != [I.LD]*5 + [I.ST]*5:
        raise ValueError("write producer requires five initial LDs then five single-chunk STs")
    store, dma_index = DmaStore(), 0
    stores_started = False
    payloads = [deque() for _ in range(8)]
    samples = {}
    page_reads = {str(p): {"weight": 0, "scale": 0} for p in range(3)}
    pending_pages = {}
    page_responses = {str(p): {"weight": 0, "scale": 0} for p in range(3)}
    generated_write_events = []
    a_data_checks = b_data_checks = native_data_checks = 0
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
    traffic_seen = False
    mm_started = False
    last_slice = deepcopy(slice_events[-1])
    for cycle in range(len(native_events)):
        se = slice_events[cycle] if cycle < len(slice_events) else deepcopy(last_slice)
        if cycle >= len(slice_events):
            se.update(cycle=cycle,start=0,done=0)
            se["seq"].update(dispatch=0,release=0,end_mask=0)
        ne, ae = native_events[cycle], adapter_events[cycle]
        for name, event in (("native", ne), ("adapter", ae), ("slice", se)):
            require(event["cycle"], cycle, name + ".cycle", cycle)
        if ae["sw_req"]:
            raise ValueError(f"cycle {cycle}: SW traffic is unsupported")
        if ae["rst"]:
            if traffic_seen:
                raise ValueError(f"cycle {cycle}: reset after memory traffic is outside the cold-reset contract")
            a_reg = b_reg = b_in = None
        require(ae["rst"], ne["rst"], "adapter/native reset agreement", cycle)
        require(ae["a_req"], int(a_reg is not None), "slice.a_register.valid", cycle)
        require(ae["b_req"], int(b_reg is not None), "slice.b_register.valid", cycle)
        if a_reg:
            require(ae["a_addr"], a_reg.word_addr, "slice.a_register.address", cycle)
        if b_reg:
            is_write = isinstance(b_reg,BWrite)
            for key, predicted in (("b_addr", b_reg.word_addr), ("b_we",int(is_write)),
                                   ("b_n",0 if is_write else b_reg.chunks-1),
                                   ("b_tag",1 if is_write else int(b_reg.tag))):
                require(ae[key], predicted, "slice." + key, cycle)
            if is_write and "b_wmask" in ae:
                require(int(ae["b_wmask"],16),b_reg.word_mask,"slice.write_mask",cycle)
                mask = sum(0xFFFFFFFF << (32*i) for i in range(32) if b_reg.word_mask >> i & 1)
                require(int(ae["b_wdata"],16)&mask,b_reg.data&mask,"slice.write_data",cycle)
            elif is_write and require_adapter_payload:
                raise ValueError("write trace must contain adapter input payload/mask")
        ns = memory.snapshot()
        # These independent memory outputs are the adapter's only channel inputs.
        pins = NativePins(ns.command_ready, tuple(tag if valid else None for tag, valid in zip(ns.read_tag, ns.read_valid)),
                          ns.write_ready,ns.write_done)
        for key,value in (("write_invalidate",sum(1<<c for c,beat in enumerate(adapter.write_invalidate) if beat is not None)),
                          ("a_cache_valid",sum(1<<i for i,entry in enumerate(adapter.cache) if entry is not None)),
                          ("a_run_valid",sum(1<<i for i,v in enumerate(adapter.run_valid) if v))):
            if key in ae:
                require(ae[key],value,"adapter."+key,cycle)
        ar = adapter.step(pins, a=a_reg, b=b_reg, a_flush=bool(ae["a_flush"]),reset=bool(ae["rst"]))
        for key, predicted in (("a_ready", ar.a_ready), ("b_ready", ar.b_ready), ("a_rvalid", ar.a_response_id is not None),
                               ("b_rvalid", ar.b_response is not None)):
            require(ae[key], int(predicted), "adapter." + key, cycle)
        for key,value in (("wr_idle",ar.wr_idle),("rd_idle",ar.rd_idle)):
            if key in ae:
                require(ae[key],int(value),"adapter."+key,cycle)
            elif require_adapter_payload:
                raise ValueError("adapter idle output oracle missing")
        if ar.a_response_id is not None and not ae["rst"]:
            expected = samples[ar.a_response_source_id]
            if "a_rdata" in ae:
                require(int(ae["a_rdata"],16).to_bytes(64,"little"), expected,"adapter.a_data",cycle)
                a_data_checks += 1
            elif require_adapter_payload:
                raise ValueError("A cache-response payload oracle missing")
        if ar.b_response:
            for key, predicted in (("b_rtag", int(ar.b_response.tag)), ("b_rcm", ar.b_response.chunk_mask), ("b_rlast", int(ar.b_response.last))):
                require(ae[key], predicted, "adapter." + key, cycle)
            if "b_rdata" in ae:
                payload=int(ae["b_rdata"],16)
                for chunk in ar.b_response_sources:
                    expected=samples[chunk.low_id]+samples[chunk.high_id]
                    require(((payload>>(1024*chunk.chunk))&((1<<1024)-1)).to_bytes(128,"little"),
                            expected,"adapter.b_data",cycle)
                    b_data_checks += 1
            elif require_adapter_payload:
                raise ValueError("B joined-response payload oracle missing")
        commands = {command.channel: command for command in ar.native_commands}
        native_inputs = []
        for c, observed in enumerate(ne["channels"]):
            cmd = commands.get(c)
            for key, predicted in (("cr", ns.command_ready[c]), ("wr", ns.write_ready[c]), ("rv", ns.read_valid[c]),
                                   ("wdone", ns.write_done[c]), ("cv", bool(cmd and cmd.cvalid)),
                                   ("wv", bool(cmd and cmd.wvalid))):
                require(observed[key], int(predicted), f"native.ch{c}." + key, cycle)
            if cmd:
                require(observed["we"], int(cmd.write), f"native.ch{c}.write", cycle)
                require(observed["addr"], cmd.beat_addr, f"native.ch{c}.address", cycle)
                if cmd.wvalid:
                    mask = sum(0xFF << (8*i) for i in range(64) if cmd.mask >> i & 1)
                    require(int(observed["wmask"],16),cmd.mask,"native.write_mask",cycle)
                    require(int(observed["wdata"],16)&mask,cmd.payload&mask,"native.write_data",cycle)
            if ns.read_valid[c]:
                require(int(observed["rdata"],16).to_bytes(64,"little"),samples[ns.read_tag[c]],"native.read_data",cycle)
                native_data_checks += 1
                if ns.read_tag[c] in pending_pages:
                    owner,kind=pending_pages.pop(ns.read_tag[c])
                    page_responses[str(owner)][kind] += 1
            native_inputs.append(NativeInput(NativeCommand(cmd.write, cmd.beat_addr, cmd.id) if cmd and cmd.cvalid else None,
                                            bool(cmd and cmd.wvalid)))
        sources, ie, me = se["memory_sources"], se["issue"], se["mxu"]
        if me["rst"] and mm_started:
            raise ValueError(f"cycle {cycle}: warm MXU reset is outside the bounded issuer contract")
        mm_started |= bool(se["start"] & 2)
        snap = issue.snapshot()
        a_gnt = a_reg is None or ar.a_ready
        b_ready_i = b_reg is None or ar.b_ready
        write_idle_i = bool(ar.wr_idle and not isinstance(b_reg,BWrite))
        ds = store.snapshot()
        store_scope = stores_started or (bool(se["start"]&1) and dma_index >= 5)
        dma_request = ds["request"]
        if store_scope:
            require(sources["dma_req"],int(dma_request is not None),"dma.generated_write_request",cycle)
            if dma_request is not None:
                require(sources["dma_we"],1,"dma.write",cycle)
                require(sources["dma_addr"],dma_request.word_addr,"dma.write_address",cycle)
            require(int(bool(se["done"]&1)),int(ds["done"]),"dma.generated_done",cycle)
            require(se["raw_reads"][1],ds["read_words"],"dma.generated_tmem_reads",cycle)
            if ds["read_words"] and ((se["lost"]|se["frozen"])&2):
                raise ValueError("ST fixture violates conflict-free DMA TMEM-read contract")
        dma_valid = dma_request is not None if store_scope else bool(sources["dma_req"])
        b_gnt = not dma_valid and b_ready_i
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
                elif dma_valid:
                    if store_scope:
                        b_reg = BWrite(bid,dma_request.word_addr,dma_request.data,dma_request.word_mask)
                    elif sources["dma_we"]:
                        raise ValueError("external DMA writes are forbidden")
                    else:
                        b_reg = BRead(bid, sources["dma_addr"], 1, True)
                else:
                    b_reg = BRead(bid, issued.b_addr, issued.b_n + 1, False) if issued.b_req else None
                bid += int(b_reg is not None)
            b_in = ar.b_response
        start_store = None
        if se["start"]&1:
            instruction = dma_commands[dma_index]
            if instruction.op == I.ST:
                source,n=instruction.w[1:3]
                start_store = StoreCommand(dma_index,instruction,tuple(map(int,oracle[source:source+n])))
                stores_started = True
            dma_index += 1
        store.step(start=start_store,b_ready=b_ready_i,
                   wr_idle=write_idle_i,reset=bool(me["rst"] and not stores_started))
        edge = memory.step(native_inputs, reset=bool(ne["rst"]))
        for channel, took in enumerate(edge.accepted_write_data):
            if took:
                cmd=commands[channel]
                payloads[channel].append((cmd.payload,cmd.mask))
        for channel, command in enumerate(edge.performed):
            if command is None:
                continue
            address=64*logical_beat(command.beat,channel,chash=bool(header["chash"]))
            if command.write:
                if address in (WB[0],SB[0]) and any(owner < 2 for owner,kind in pending_pages.values()):
                    raise ValueError("page write while old-epoch native reads remain outstanding")
                value,mask=payloads[channel].popleft()
                for byte in range(64):
                    if mask>>byte&1:
                        content[address+byte]=(value>>(8*byte))&255
                generated_write_events.append(dict(cycle=cycle,address=address,id=str(command.tag)))
            else:
                samples[command.tag]=bytes(content[address:address+64])
        for channel,command in enumerate(edge.accepted_commands):
            if command is not None and not command.write:
                key=classify_byte(64*logical_beat(command.beat,channel,chash=bool(header["chash"])))
                if key is not None:
                    kind,page=key
                    owner=2 if page==0 and issue.next_start==3 else page
                    if owner==2 and (store.completed < 2 or
                            not {WB[0],SB[0]} <= {w["address"] for w in generated_write_events}):
                        raise ValueError("new-epoch read before both generated patch writes completed")
                    if owner==0 and any(w["address"] in (WB[0],SB[0]) for w in generated_write_events):
                        raise ValueError("old-epoch read after generated page overwrite")
                    page_reads[str(owner)][kind]+=1
                    pending_pages[command.tag]=(owner,kind)
        traffic_seen |= any(cmd is not None for cmd in edge.accepted_commands)
        counts["cycles_checked"] += 1
        counts["a_requests"] += int(issued.a_req)
        counts["b_requests"] += int(issued.b_req)
        counts["native_commands"] += sum(cmd is not None for cmd in edge.accepted_commands)
        counts["native_responses"] += sum(ns.read_valid)
        counts["b_push"] += int(b_push)
        counts["r_cpl"] += int(r_cpl)
    issue.finish()
    rows.finish()
    store_result=store.finish()
    if (pending_pages or page_reads != page_responses or any(memory.commands) or any(memory.read_due)
            or any(memory.write_due) or any(memory.write_data) or any(payloads)):
        raise ValueError("new epoch/native commands or responses did not fully close")
    if any(adapter.aq) or any(adapter.bq) or any(adapter.native_tags) or adapter.b_tags or adapter.a_order:
        raise ValueError("adapter still has unissued requests or response orders")
    if dma_index != len(dma_commands) or store_result != dict(starts=5,writes=5,completions=5):
        raise ValueError("fixture did not execute all five loads and five stores")
    if bytes(content[:cfg.DRAM_BYTES]) != functional.dram.tobytes():
        raise ValueError("generated final DRAM differs from exact ISA functional oracle")
    ownership_result = {"ownership": ownership.finish()} if ownership is not None else {}
    return {**counts, **ownership_result,"all_mm_issuers_and_arrivals_drained": True,
            "stores":store_result,"generated_write_events":generated_write_events,
            "generated_page_read_requests":page_reads,"generated_page_read_responses":page_responses,
            "native_payloads_checked":native_data_checks,"a_cache_payloads_checked":a_data_checks,
            "b_joined_chunk_payloads_checked":b_data_checks,
            "new_epoch_read_ownership_closed":True,
            "generated_final_dram_matches_isa":True,
            "mismatches": 0, "whole_program_predictor": False, "calibrated": False, "tok_s": None,
            "external_inputs": ["reset/a_flush and accepted MM/DMA command starts", "MXU c_done/pn/more and drain epilogue",
                                "TMEM drain grant", "initial read-only DMA/loader requests and addresses",
                                "immutable ST source words from exact ISA functional oracle"],
            "excluded": ["SW/A writes", "initial DMA load/loader request generation",
                         "independent sequencer and output drain epilogue"]}
