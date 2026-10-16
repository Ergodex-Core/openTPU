"""The openTPU slice as a VisualSim model: the decode token's memory traffic from a Verilator
trace, replayed through VisualSim's native memory blocks under the slice's structure.

Topology (every block an installed VisualSim class; docs/hwloop.md):

    Start_Once (DS_Source) -> Scheduler (Virtual_Machine)
    Scheduler.mem_req_<c>  -> memory channel c              c = 0 .. NCH-1
    memory channel c       -> Scheduler.mem_ret_<c>
    Scheduler.stream_request -> PortB_Stream (Smart_Timed_Resource: the MXU's port, one
                                D-byte chunk a core cycle) -> Scheduler.stream_return
    Scheduler.tail_request   -> Unit_Tail (Smart_Timed_Resource: a phase's VPU / quantizer
                                work that the trace shows after its stream) -> Scheduler.tail_return
    Scheduler.metrics        -> Completion_Counters (Display; ExpressionWriter in batch)

The memory is the profile's: HBM* -> VisualSim.actor.arch.Memory.HBM_Module (one of its
channels per openTPU channel; DRAM_Speed_Mhz, Channel_Buffer_Size, Burst_Length,
Timing_Cycles exposed); DDR* / LPDDR* -> one Memory_Controller + HW_DRAM pair per channel
(DRAM_Type, Controller_Speed_Mhz, HW_DRAM_Speed_Mhz, Mfg_Suggest_Timing, Extra_Timing,
refresh: Retention_Time / Fine_Granularity_Refresh_Time).

The scheduler issues each phase's reads as bursts (Request_Bytes = the bridge's burst) round-
robin over the channels, at most Credits_Per_Channel in flight per channel (the bridge's
outstanding reads), sequential addresses per channel; every returned burst must drain through
PortB_Stream (finite queue: backpressure) before the phase can end; the phase's non-stream
unit work follows as a dependency; writes are issued with the reads at the trace's share. It
is not the RTL: no instruction stream, no scoreboard, no TMEM, no DMA line buffer, one slice.

What the memory blocks do inside (bank state, refresh, read/write turnarounds) is the vendor's
implementation; what they do not expose (the HBM composite has no refresh parameter) is listed
in the manifest's `not_modelled`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import vslib
from .vslib import (C_ARCH_SETUP, C_DS_SOURCE, C_HBM, C_HWDRAM, C_MEMC, C_MEM_INIT, FIELD_MAP, P_ATTR,
                    P_PULLDOWN, Entity, Model, display, qstr, timed_resource, vm_block)

GENERATOR = "opentpu-hwloop-vsmodel/0.1"
HERE = Path(__file__).resolve().parent


def load_profile(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.is_file():
        p = HERE / "profiles" / f"{name_or_path}.json"
    prof = json.loads(p.read_text())
    prof["_path"] = str(p)
    prof["_sha256"] = vslib.sha256_file(p)
    return prof


def _phases(trace: dict) -> list[dict]:
    """The token's phases in trace order: name, read bytes, write bytes, stream cycles, tail
    cycles (unit work beyond the stream, from the phase's cycles over its port roofline and
    its VPU busy)."""
    s = trace.get("summary", {})
    d = trace.get("derived", {})
    ph = s.get("phases") or {}
    total_cycles = d.get("cycles") or s.get("cycles") or 1
    rbytes_total = d.get("dram_read_bytes") or 0
    wbytes_total = d.get("dram_write_bytes") or 0
    out = []
    if ph:
        span = sum(v["cycles"] for v in ph.values()) or 1
        for name, v in ph.items():
            share = v["cycles"] / span
            rb = int(v["bytes"]) if v.get("bytes") else int(rbytes_total * share)
            tail = max(0, v["cycles"] - v.get("roofline_cycles", v["cycles"]))
            out.append({"name": name, "read_bytes": rb, "write_bytes": int(wbytes_total * share),
                        "rtl_cycles": v["cycles"], "tail_cycles": int(tail), "vpu_busy": v.get("vpu_busy", 0)})
    else:
        out.append({"name": "token", "read_bytes": int(rbytes_total), "write_bytes": int(wbytes_total),
                    "rtl_cycles": int(total_cycles), "tail_cycles": 0, "vpu_busy": 0})
    return out


def _arr(vals, long=False) -> str:
    return "{" + ",".join((f"{int(v)}L" if long else str(v)) for v in vals) + "}"


def scheduler_code(nch: int, nphases: int) -> str:
    """The Virtual_Machine program (VisualSim VM syntax as the installed examples use it)."""
    send_req = []
    for c in range(nch):
        send_req.append(f"{'if' if c == 0 else 'else if'} (Ch == {c}) {{ SEND(mem_req_{c},Request) }}")
    ret_ports = " || ".join(f'port_name == "mem_ret_{c}"' for c in range(nch))
    ret_ch = []
    for c in range(nch):
        ret_ch.append(f'{"if" if c == 0 else "else if"} (port_name == "mem_ret_{c}") {{ Ch = {c} }}')
    code = f"""LABEL: BEGIN
if (port_name == "input") {{
 Started = true
 Phase = 0
 Issued = 0
 Completed = 0
 Drained = 0
 Inflight = 0
 Peak_Inflight = 0
 Read_Bytes = 0L
 Write_Bytes = 0L
 Total_Latency_Seconds = 0.0
 Credit_Blocked_Events = 0
 Stream_Backpressure_Events = 0
 Tail_Seconds = 0.0
 Ch_Inflight = newArray({nch},0)
 Ch_Next = newArray({nch},0L)
 Phase_Elapsed = newArray({nphases},0.0)
 Start_Seconds = TNow
 GTO(Start_Phase)
}}
if (!Started) {{ EXIT }}
if ({ret_ports}) {{
 {chr(10).join(' ' + x for x in ret_ch)}
 Completed = Completed + 1
 Inflight = Inflight - 1
 Ch_Inflight(Ch) = Ch_Inflight(Ch) - 1
 Total_Latency_Seconds = Total_Latency_Seconds + TNow - port_token.Trace_Issue_Time
 if (port_token.A_Command == "Read") {{
  Drain = newToken("Processor_DS")
  Drain.A_Destination = "PortB"
  Drain.Stream_Time = Request_Bytes / (Chunk_Bytes * Core_Hz)
  Drain.Trace_Phase = Phase
  SEND(stream_request,Drain)
 }}
 else {{
  Drained = Drained + 1
 }}
 GTO(Issue_More)
}}
if (port_name == "stream_return") {{
 Drained = Drained + 1
 if (Drained >= Phase_Requests && Inflight == 0 && Phase_Issued >= Phase_Requests) {{
  Tail = newToken("Processor_DS")
  Tail.A_Destination = "Units"
  Tail.Tail_Time = Phase_Tail_Cycles(Phase) * Replay_Fraction / Core_Hz
  Tail_Seconds = Tail_Seconds + Tail.Tail_Time
  SEND(tail_request,Tail)
 }}
 EXIT
}}
if (port_name == "tail_return") {{
 Phase_Elapsed(Phase) = TNow - Phase_Start
 Phase = Phase + 1
 GTO(Start_Phase)
}}
EXIT
LABEL: Start_Phase
if (Phase >= N_Phases) {{
 Elapsed = TNow - Start_Seconds
 Report = newToken("Header")
 Report.Status = "REPLAY_COMPLETE_NOT_A_BOARD_MEASUREMENT"
 Report.Requests_Issued = Issued
 Report.Requests_Completed = Completed
 Report.Requests_Drained = Drained
 Report.Inflight = Inflight
 Report.Peak_Inflight = Peak_Inflight
 Report.Read_Bytes = Read_Bytes
 Report.Write_Bytes = Write_Bytes
 Report.Elapsed_Seconds = Elapsed
 Report.Core_Cycles = Elapsed * Core_Hz
 Report.Replay_Fraction = Replay_Fraction
 Report.Token_Seconds_Est = Elapsed / Replay_Fraction
 Report.Device_Tok_s_Est = Replay_Fraction / Elapsed
 Report.Achieved_GBps = (Read_Bytes + Write_Bytes) / Elapsed / 1.0e9
 Report.Mean_Memory_Latency_Seconds = Total_Latency_Seconds / Completed
 Report.Credit_Blocked_Events = Credit_Blocked_Events
 Report.Tail_Seconds = Tail_Seconds
 Report.Phase_Elapsed_Seconds = Phase_Elapsed
 if (Issued != Completed || Inflight != 0) {{ throwMyException("replay invariant failed") }}
 SEND(metrics,Report)
 EXIT
}}
Phase_Start = TNow
Phase_Requests = Phase_Read_Requests(Phase) + Phase_Write_Requests(Phase)
Phase_Issued = 0
Drained = 0
GTO(Issue_More)
LABEL: Issue_More
Go = true
while (Go && Phase_Issued < Phase_Requests) {{
 Ch = Phase_Issued % {nch}
 if (Ch_Inflight(Ch) >= Credits_Per_Channel) {{
  Go = false
  Credit_Blocked_Events = Credit_Blocked_Events + 1
 }}
 else {{
  Request = newToken("Processor_DS")
  Request.ID = Issued
  Request.TIME = TNow
  Request.A_Task_Flag = true
  Request.A_Source = "Slice"
  Request.A_Destination = "Memory"
  Request.A_Command = "Read"
  if (Phase_Issued >= Phase_Read_Requests(Phase)) {{ Request.A_Command = "Write" }}
  Request.A_Bytes = Request_Bytes
  Request.A_Bytes_Sent = Interface_Bytes
  Request.A_Bytes_Remaining = Request_Bytes - Interface_Bytes
  Request.A_Address = Ch * Channel_Span + (Ch_Next(Ch) % Channel_Span)
  Request.A_Address_Min = Request.A_Address
  Ch_Next(Ch) = Ch_Next(Ch) + Request_Bytes
  Request.Trace_Phase = Phase
  Request.Trace_Issue_Time = TNow
  Request.Event_Name = "Native_Return"
  if (Request.A_Command == "Read") {{ Read_Bytes = Read_Bytes + Request_Bytes }}
  else {{ Write_Bytes = Write_Bytes + Request_Bytes }}
  Ch_Inflight(Ch) = Ch_Inflight(Ch) + 1
  Inflight = Inflight + 1
  if (Inflight > Peak_Inflight) {{ Peak_Inflight = Inflight }}
  {chr(10).join('  ' + x for x in send_req)}
  Issued = Issued + 1
  Phase_Issued = Phase_Issued + 1
 }}
}}
EXIT"""
    return code


def build(trace: dict, profile: dict, out_dir: Path, *, name: str, rtl_fingerprint: dict | None,
          replay_fraction: float = 1.0, variant: str = "baseline") -> dict:
    """Write out_dir/model.xml, manifest.json, expected-counters.json, static-validation.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mem = profile["memory"]
    clocks = profile["clocks"]
    nch = int(mem["channels_per_slice"])
    beat = int(mem["beat_bytes"])
    burst_beats = int(mem.get("burst_beats_max") or 1)
    req_bytes = beat * burst_beats
    out_beats = int(mem.get("outstanding_reads_per_channel") or 16)
    credits = max(1, out_beats // burst_beats)
    chunk = int(mem.get("chunk_bytes", 128))
    core_hz = float(clocks["core_mhz"]) * 1e6
    phases = _phases(trace)
    rtl_cycles = trace.get("derived", {}).get("cycles") or sum(p["rtl_cycles"] for p in phases)
    reads = [int(round(p["read_bytes"] * replay_fraction / req_bytes)) for p in phases]
    writes = [int(round(p["write_bytes"] * replay_fraction / req_bytes)) for p in phases]
    tails = [p["tail_cycles"] for p in phases]
    rtl_seconds = rtl_cycles / core_hz
    sim_time = max(1e-4, 10.0 * rtl_seconds * replay_fraction)
    is_hbm = str(mem["type"]).upper().startswith("HBM")
    channel_span = int(mem["channel_bytes"])

    m = Model(name)
    m.note = (f"{name}: openTPU slice decode-token replay ({variant}).\n"
              f"Workload: Verilator trace {trace.get('run_dir', '?')}; profile {profile['name']} ({profile['kind']}).\n"
              "Label: unvalidated scenario until a VisualSim run exists and is compared with the receipts.\n"
              "Native memory + credits + port drain + unit tails; not an ISA, RTL or checkpoint execution.")
    m.param("Sim_Time", sim_time)
    m.param("Architecture_Name", qstr(name))
    m.param("Core_Hz", core_hz)
    m.param("Controller_MHz", float(clocks.get("controller_mhz") or clocks.get("memory_axi_mhz") or 0.0))
    m.param("Memory_Data_Rate_MTps", float(mem.get("data_rate_MTps") or 0.0))
    m.param("Memory_Latency_ns_Assumed", float(mem.get("latency_ns_loaded") or (mem.get("latency_ns_loaded_est") or [0])[0] or 0.0))
    m.param("NCH", nch)
    m.param("Credits_Per_Channel", credits)
    m.param("Request_Bytes", req_bytes)
    m.param("Interface_Bytes", min(8, req_bytes))
    m.param("Chunk_Bytes", chunk)
    m.param("Channel_Span", channel_span)
    m.param("Replay_Fraction", replay_fraction)
    m.param("N_Phases", len(phases))
    m.param("Phase_Read_Requests", _arr(reads))
    m.param("Phase_Write_Requests", _arr(writes))
    m.param("Phase_Tail_Cycles", _arr(tails))
    m.param("RTL_Cycles_Reference", rtl_cycles)
    m.param("Stream_Queue_Depth", 16)
    m.param("View_Stats", True)

    arch = m.add(Entity("ArchitectureSetup", C_ARCH_SETUP, location=(80, 170)))
    arch.param("Architecture_Name", "Architecture_Name")
    arch.props.append(("Routing_Table", P_ATTR, ""))
    arch.props.append(("Field_Name_Mapping", P_ATTR, FIELD_MAP))

    src = m.add(Entity("Start_Once", C_DS_SOURCE, location=(100, 320)))
    src.param("Data_Structure_Name", qstr("Processor_DS"))
    src.param("Start_Time", 1e-9)
    src.param("Value_1", "Sim_Time*2.0")
    src.param("Number_of_Transactions", 1)
    src.props.append(("Time_Distribution", P_ATTR, "Fixed (Value_1)"))
    src.ports = [("output", "output")]

    init = m.add(Entity("Initial_State", C_MEM_INIT, location=(100, 470)))
    init.props.append(("Memory_Init_Text", P_ATTR, "Started local false;"))

    req_ports = [f"mem_req_{c}" for c in range(nch)]
    ret_ports = [f"mem_ret_{c}" for c in range(nch)]
    vm = m.add(vm_block("Scheduler", f"{name}_Scheduler", scheduler_code(nch, len(phases)),
                        ["input", "stream_return", "tail_return"] + ret_ports,
                        ["stream_request", "tail_request", "metrics"] + req_ports))
    m.link("Start_Once.output", "Scheduler.input")

    stream = m.add(timed_resource("PortB_Stream", f"{name}_PortB", "input.Stream_Time", 16, (700, 600)))
    stream.ports = [("input", "input"), ("output", "output")]
    m.link("Scheduler.stream_request", "PortB_Stream.input")
    m.link("PortB_Stream.output", "Scheduler.stream_return")
    tail = m.add(timed_resource("Unit_Tail", f"{name}_Units", "input.Tail_Time", 4, (700, 760)))
    tail.ports = [("input", "input"), ("output", "output")]
    m.link("Scheduler.tail_request", "Unit_Tail.input")
    m.link("Unit_Tail.output", "Scheduler.tail_return")

    counters = m.add(display("Completion_Counters", f"{name} completion counters", (1100, 580)))
    counters.ports = [("input", "input")]
    m.link("Scheduler.metrics", "Completion_Counters.input")

    not_modelled = ["instruction stream / scoreboard / TMEM / DMA line buffer (the RTL's; the trace carries their effect as phase tails)",
                    "host, PCIe, sampling", "multi-slice"]
    if is_hbm:
        if nch > 8:
            raise ValueError("HBM_Module exposes 8 channels; NCH > 8 needs a second module")
        hbm = m.add(Entity("HBM_Native", C_HBM, location=(900, 250)))
        hbm.param("HBM_Name", qstr("HBM"))
        hbm.param("Architecture_Name", "Architecture_Name")
        hbm.param("DRAM_Speed_Mhz", float(mem.get("array_mhz") or 400.0))
        hbm.param("Memory_Type", "HBM2", P_PULLDOWN)
        hbm.param("Address_Range", f"{{0L,{nch * channel_span - 1}L}}")
        hbm.param("Channel_Buffer_Size", out_beats)
        hbm.param("Memery_Width_Bytes", 8)
        hbm.param("Burst_Length", 4)
        hbm.param("Bank_Groups_per_Channel", 2)
        hbm.param("Number_of_Samples", 2)
        hbm.param("Timing_Cycles", "{5,5,32,16,16,6,4,6,4,2}")
        hbm.param("writeStatsToFile", False)
        for c in range(nch):
            m.link(f"Scheduler.mem_req_{c}", f"HBM_Native.Channel_{c}_IN")
            m.link(f"HBM_Native.Channel_{c}_OUT", f"Scheduler.mem_ret_{c}")
        stats = m.add(display("HBM_Statistics", f"{name} HBM statistics", (1100, 250)))
        stats.ports = [("input", "input")]
        m.link("HBM_Native.Statistics", "HBM_Statistics.input")
        not_modelled += ["HBM refresh (the installed HBM_Module exposes none)",
                         "the 450 MHz AXI port and the bridge's clock crossings (HBM_Module's array clock is DRAM_Speed_Mhz)",
                         "the pseudo-channel's 256-bit AXI3 port (HBM_Module's Memery_Width_Bytes is its own)"]
    else:
        dtype = "LPDDR5" if str(mem["type"]).upper().startswith("LPDDR") else ("DDR3" if "DDR3" in str(mem["type"]).upper() else "DDR4")
        rate = float(mem.get("data_rate_MTps") or 3200.0)
        refresh = mem.get("refresh") or {}
        for c in range(nch):
            mc = m.add(Entity(f"MemC_{c}", C_MEMC, location=(900, 150 + 120 * c)))
            mc.param("Controller_Name", qstr(f"MC{c}"))
            mc.param("DRAM_Type", dtype, P_PULLDOWN)
            mc.param("Controller_Speed_Mhz", rate)
            mc.param("Memory_Width_Bytes", int(mem.get("device_width_bits", 64)) // 8)
            mc.param("Burst_Length", 8)
            mc.param("Command_Buffer_Length", out_beats * 2)
            mc.param("HW_DRAM_Name", qstr(f"DRAM{c}"))
            mc.param("Number_of_Ranks", int(mem.get("ranks") or 1))
            mc.param("Architecture_Name", "Architecture_Name")
            mc.param("DEBUG", False)
            mc.ports = [("rd_wr_data_fm_bus", "input"), ("rd_data_wr_resp_to_bus", "output"),
                        ("rd_wr_data_to_mem", "output"), ("rd_data_fm_mem", "input"),
                        ("ctrl_to_mem", "output"), ("ctrl_fm_mem", "input")]
            dr = m.add(Entity(f"DRAM_{c}", C_HWDRAM, location=(1150, 150 + 120 * c)))
            dr.param("HW_DRAM_Name", qstr(f"DRAM{c}"))
            dr.param("DRAM_Type", dtype, P_PULLDOWN)
            dr.param("HW_DRAM_Speed_Mhz", rate)
            dr.param("Burst_Length", 8)
            dr.param("Memory_Width_Bytes", int(mem.get("device_width_bits", 64)) // 8)
            dr.param("Memory_Controller", qstr(f"MC{c}"))
            dr.param("Retention_Time", (refresh.get("tREFI_us") or 7.8) * 1e-6 * 8192)
            dr.param("Fine_Granularity_Refresh_Time", (refresh.get("tRFC_ns") or 350) * 1e-9)
            dr.param("Architecture_Name", "Architecture_Name")
            dr.ports = [("port_1", "input"), ("port_2", "output"), ("fm_ctrl", "input"), ("to_ctrl", "output")]
            m.link(f"Scheduler.mem_req_{c}", f"MemC_{c}.rd_wr_data_fm_bus")
            m.link(f"MemC_{c}.rd_data_wr_resp_to_bus", f"Scheduler.mem_ret_{c}")
            m.link(f"MemC_{c}.rd_wr_data_to_mem", f"DRAM_{c}.port_1")
            m.link(f"DRAM_{c}.port_2", f"MemC_{c}.rd_data_fm_mem")
            m.link(f"MemC_{c}.ctrl_to_mem", f"DRAM_{c}.fm_ctrl")
            m.link(f"DRAM_{c}.to_ctrl", f"MemC_{c}.ctrl_fm_mem")
        not_modelled += ["the DIMM / device part's own timings (the vendor block's defaults for the DRAM_Type stand in; the profile's refresh is applied)",
                         "the F2 DDR bridge (250 MHz) and LiteDRAM's controller (the vendor controller stands in)"]

    xml = m.to_xml()
    (out_dir / "model.xml").write_text(xml)
    (out_dir / "scheduler.vm.txt").write_text(scheduler_code(nch, len(phases)) + "\n")
    expected = {"requests_issued": sum(reads) + sum(writes), "requests_completed": sum(reads) + sum(writes),
                "inflight": 0, "read_bytes": sum(reads) * req_bytes, "write_bytes": sum(writes) * req_bytes,
                "rtl_reference_seconds_scaled": rtl_seconds * replay_fraction,
                "note": "acceptance values for the replay's own counters, not board numbers"}
    (out_dir / "expected-counters.json").write_text(json.dumps(expected, indent=2) + "\n")
    manifest = {
        "generator": GENERATOR, "name": name, "variant": variant, "label": vslib.UNVALIDATED,
        "status": "PREPARED_NOT_EXECUTED_IN_VISUALSIM",
        "profile": {"name": profile["name"], "path": profile.get("_path"), "sha256": profile.get("_sha256"),
                    "kind": profile["kind"], "label": profile["label"]},
        "trace": {"run_dir": trace.get("run_dir"), "label": trace.get("label"), "source": trace.get("source"),
                  "derived": trace.get("derived"), "memory_trace": trace.get("memory_trace")},
        "rtl_fingerprint": rtl_fingerprint,
        "parameters": dict(m.params),
        "phases": [dict(p, read_requests=r, write_requests=w) for p, r, w in zip(phases, reads, writes)],
        "memory_backend": "HBM_Module" if is_hbm else "Memory_Controller+HW_DRAM",
        "independent_parameters": ["Core_Hz", "Controller_MHz", "Memory_Data_Rate_MTps", "Memory_Latency_ns_Assumed (vendor block timing)"],
        "not_modelled": not_modelled,
        "not_established": ["VisualSim parser acceptance", "VM runtime compilation", "native request/return protocol",
                            "numerical correctness (a VisualSim run proves none)", "any board number"],
        "model_sha256": vslib.sha256_text(xml),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sv = vslib.static_validate(out_dir / "model.xml")
    (out_dir / "static-validation.json").write_text(json.dumps(sv, indent=2) + "\n")
    manifest["static_validation"] = sv["status"]
    return manifest
