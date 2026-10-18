"""Standalone C1 RTL kernel validation and Lens export.

Cycles are obtained from Verilator handshakes. Memory is an explicitly configurable
abstract service model; this module is not a full ISA machine, AXI bridge, or F2 card.
"""
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import Config, isa as I, quant as Q
from .isasim import Slice
from .profile import UNITS

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / p for p in (
    "rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv", "rtl/c1/otpu_c1_row_engine.sv",
    "rtl/c1/otpu_c1_cluster.sv", "rtl/c1/otpu_c1_epilogue.sv", "rtl/c1/otpu_c1_top.sv",
    "sim/c1/main.cpp")]


def build(*, engines=16, slots=4, harness="main"):
    """Content-keyed local build; bounded two-job make avoids shared-host saturation."""
    if harness not in ("main", "replay") or (harness == "replay" and (engines, slots) != (1, 1)):
        raise ValueError("replay harness requires E1/S1")
    sources = [*SOURCES[:-1], ROOT / "sim" / "c1" / f"{harness}.cpp"]
    if engines not in (1, 16) or slots not in (1, 2, 4, 8):
        raise ValueError("validated engine counts 1/16 and slot depths 1/2/4/8")
    if not shutil.which("verilator"):
        raise RuntimeError("C1 RTL validation requires Verilator")
    version = subprocess.check_output(["verilator", "--version"], text=True).strip()
    flags = ["--cc", "--exe", "--build", "-j", "2", "--assert", "-Wno-fatal",
             "--top-module", "otpu_c1_top", f"-GENGINES={engines}", f"-GSLOTS={slots}",
             "-CFLAGS", f"-std=c++17 -DC1_ENGINES={engines} -DC1_SLOTS={slots}"]
    digest = hashlib.sha256((version + repr(flags)).encode())
    for p in sources:
        digest.update(p.read_bytes())
    out = ROOT / "build" / "c1-rtl" / digest.hexdigest()[:16]
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "Votpu_c1_top"
    with (out / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            result = subprocess.run(["verilator", *flags, "--Mdir", str(out),
                                     *map(str, sources)], capture_output=True, text=True)
            (out / "build.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"C1 build failed; see {out / 'build.log'}")
    return binary


@dataclass
class Workload:
    n0: int
    n: int
    kb: int
    k0: int
    ab: int
    weights: np.ndarray
    scales: np.ndarray
    state: Slice
    expected: np.ndarray
    accumulators: list

    def save(self, path):
        """Decimal uint32 fixture with explicit 32-byte little-endian groups."""
        st = self.state
        sa = int(st.ascale[0, self.ab:self.ab+1].view(np.uint32)[0])
        lines = [f"{self.n0} {self.n} {self.kb} {self.k0} {self.ab} {self.kb*4}"]
        for g in range(self.kb*4):
            a = st.act[0, self.ab*128+g*32:self.ab*128+(g+1)*32].copy().view("<u4")
            lines.append(" ".join(map(str, [self.ab*4+g, int(st.adoff[0,self.ab+g//4,g%4]), sa, *a])))
        for row in range(self.n0, self.n0+self.n):
            sw = int(self.scales[row, :4].copy().view("<u4")[0])
            lines.append(f"{row} {sw}")
            for g in range(self.kb*4):
                w = self.weights[row, self.k0*128+g*32:self.k0*128+(g+1)*32].copy().view("<u4")
                lines.append(" ".join(map(str, [int(self.scales[row,4+self.k0+g//4]), *w])))
        Path(path).write_text("\n".join(lines)+"\n")
        return Path(path)


def workload(*, n=257, n0=0, kb=9, k0=0, ab=2, seed=41, mode="quantized", sa_bits=None):
    if not (1 <= n <= 65535 and n0 >= 0 and 1 <= kb and 0 <= k0 and k0+kb <= 255
            and 0 <= ab and ab+kb <= 128):
        raise ValueError("unsupported workload bounds")
    rng = np.random.default_rng(seed)
    full_blocks = k0+kb
    w, s = Q.quantize_mxu(rng.normal(size=(n0+n, full_blocks*128)).astype(np.float32), "int8", 128)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, TMEM_WORDS=max(65536,n+18000),
                 DRAM_BYTES=w.size+s.size+512)
    st = Slice(cfg, 0, [], np.zeros(cfg.DRAM_BYTES, np.uint8))
    x = rng.normal(size=kb*128).astype(np.float32)
    st.tput(np.arange(x.size), x)
    st.execute(I.qact(0, 1, ab, kb, x.size))
    if mode == "random":
        w[:] = rng.integers(0,256,w.shape,dtype=np.uint8)
        s[:,4:4+full_blocks] = rng.integers(0,22,(n0+n,full_blocks),dtype=np.uint8)
        st.act[:] = rng.integers(-128,128,st.act.shape,dtype=np.int8)
        st.adoff[:] = rng.integers(0,22,st.adoff.shape,dtype=np.uint8)
    elif mode in ("positive", "negative", "dropped"):
        w[:] = 128 if mode != "negative" else 127
        s[:,4:4+full_blocks] = 0 if mode != "dropped" else 255
        st.act[:] = -128
        st.adoff[:] = 0
    elif mode == "special":
        bits = np.array([0,0x80000000,1,0x00800000,0x3f800000,0x7f800000,0xff800000,0x7fc12345],np.uint32)
        s[:,:4] = np.resize(bits,n0+n).astype("<u4")[:,None].view(np.uint8)
    elif mode != "quantized":
        raise ValueError("unknown arithmetic mode")
    if sa_bits is not None:
        st.ascale[:] = np.array(sa_bits,np.uint32).view(np.float32)
    wa, ss = 128, w.size+256
    st.dram[wa:wa+w.size] = w.reshape(-1)
    st.dram[ss:ss+s.size] = s.reshape(-1)
    with np.errstate(invalid="ignore", over="ignore"):
        st.execute(I.mm(wa+n0*w.shape[1]+k0*128, ss+n0*s.shape[1],18000,n,kb,
                        w.shape[1],n,1,ab,s.shape[1],k0=k0))
    expected = st.tmem[18000:18000+n].copy()
    acc = []
    for row in range(n0,n0+n):
        total = 0
        for g in range(kb*4):
            shift = 16-int(s[row,4+k0+g//4])-int(st.adoff[0,ab+g//4,g%4])
            if shift >= 0:
                a = st.act[0,ab*128+g*32:ab*128+(g+1)*32].astype(np.int64)
                b = w[row,k0*128+g*32:k0*128+(g+1)*32].view(np.int8).astype(np.int64)
                total += int(a @ b) << shift
        assert -(1<<47) <= total < 1<<47
        acc.append(total & ((1<<48)-1))
    return Workload(n0,n,kb,k0,ab,w,s,st,expected,acc)


def run(case, output, *, engines=16, slots=4, latency=80, beat_interval=1,
        pc0_delay=0, seed=1, request_percent=100, output_percent=100, reset_after=-1,
        interleave=False):
    if latency < 1 or beat_interval < 1 or pc0_delay < 0 or not 1 <= request_percent <= 100 or not 1 <= output_percent <= 100:
        raise ValueError("invalid memory/backpressure settings")
    if reset_after != -1 and reset_after < 1:
        raise ValueError("reset_after is -1 or positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binary = build(engines=engines,slots=slots)
    fixture = case.save(output / "input.txt")
    trace = output / "trace.jsonl"
    result = subprocess.run([str(binary),str(fixture),str(trace),str(latency),str(beat_interval),
                             str(pc0_delay),str(seed),str(request_percent),str(output_percent),
                             str(reset_after),str(int(interleave))],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"C1 RTL simulation failed: {result.stdout}{result.stderr}")
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    summary = events[-1]
    gen = summary["generation"]
    results = [e for e in events if e["event"]=="result" and e["generation"]==gen]
    integers = [e for e in events if e["event"]=="integer" and e["generation"]==gen]
    wanted_rows = list(range(case.n0,case.n0+case.n))
    if [e["row"] for e in results] != wanted_rows or [e["row"] for e in integers] != wanted_rows:
        raise AssertionError("missing, duplicated or reordered RTL outputs")
    actual = np.array([e["bits"] for e in results],np.uint32)
    if not np.array_equal(actual,case.expected):
        bad = np.flatnonzero(actual!=case.expected)
        raise AssertionError(f"RTL fp32 differs from canonical ISA at rows {bad[:8]+case.n0}: "
                             f"actual {actual[bad[:8]]} expected {case.expected[bad[:8]]}")
    if [e["acc_bits"] for e in integers] != case.accumulators:
        raise AssertionError("RTL signed48 accumulation differs from independent group sum")
    beats = [e for e in events if e["event"]=="beat" and e["generation"]==gen and not e["drop"]]
    if len(beats) != case.n*case.kb*4:
        raise AssertionError("wrong accepted weight beat count")
    if summary["max_occupancy"] > slots:
        raise AssertionError("row slot occupancy exceeded bound")
    if reset_after >= 0 and not summary["dropped_beats"]:
        raise AssertionError("reset test failed to exercise stale response rejection")
    settings = dict(engines=engines,slots=slots,latency=latency,beat_interval=beat_interval,
                    pc0_delay=pc0_delay,seed=seed,request_percent=request_percent,
                    output_percent=output_percent,reset_after=reset_after,interleave=interleave)
    summary["all_generations_served_beats"] = summary.pop("served_beats")
    summary["all_generations_requests"] = summary.pop("requests")
    summary.update(settings=settings,outputs_bit_exact=case.n,selected_generation_weight_beats=len(beats),
                   weights_bytes_served=len(beats)*32,
                   trace=str(trace.resolve()),binary=str(binary))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return events,summary


def profile(case, events, summary, *, name="C1 kernel", mhz=200):
    """Lens v1 from measured RTL handshakes; no inferred baseline retiming/counters."""
    gen,start,end=summary["generation"],summary["command_cycle"],summary["end_cycle"]
    settings=summary["settings"]
    requests={e["row"]:e for e in events if e["event"]=="request" and e["generation"]==gen}
    results={e["row"]:e for e in events if e["event"]=="result" and e["generation"]==gen}
    instrs=[]
    for row in sorted(results):
        r,y=requests[row],results[row]
        instrs.append([0,row-case.n0,1,0,"C1_RTL_ROW",f"PC {r['pc']} slot {r['slot']}; row {row}; KB {case.kb}",
                       r["cycle"]-start,r["cycle"]-start,-1,r["cycle"]-start,y["cycle"]-start+1,
                       -1,{},case.kb*4,case.kb,0])
    cycles=end-start
    return {"kind":"rtl-c1-kernel","name":name+" [Verilator; abstract PC memory]",
            "cycles":cycles,"clock_mhz":mhz,
            "config":{"S":1,"D":128,"MCOLS":4,"LANES":16,"TMEM_WORDS":case.state.cfg.TMEM_WORDS,
                      "MEM":"hbm","PCS":settings["engines"],"C1_ENGINES":settings["engines"],"C1_SLOTS":settings["slots"]},
            "units":UNITS,"instrs":instrs,"programs":[[[0,"C1_RTL_ROW","row request to fp32 retirement","RTL protocol event, not ISA",-1]]],
            "sources":[],"mxu_gaps":{},
            "slices":[{"ports":{},"busy":{},"lose":{},"buckets":{}}],
            "macs":case.n*case.kb*128,"bytes":summary["weights_bytes_served"],
            "peak_macs":settings["engines"]*32,
            "peak_bytes":settings["engines"]*32/settings["beat_interval"],
            "notes":[{"level":"warn","text":"Measured Verilator C1 kernel cycles, NOT F2 hardware, synthesis timing, a full token, or canonical ISA execution. RTL computes signed48 and exact repository fp32 retirement. 200 MHz is only a comparison target."},
                     {"level":"info","text":"Abstract tagged per-PC memory: row latency and beat service interval are separate assumptions. Weight beat counts are actual ready/valid service events. Header parsing/scale transport, AXI/bank scheduling, CL DMA, ACT load, TMEM/ACC, VPU and host costs are excluded from these command cycles. Row timeline is request to result, not compute utilization. No fabricated port counters."}],
            "c1_rtl":{"summary":summary,"command":{"n0":case.n0,"n":case.n,"kb":case.kb,"k0":case.k0,"ab":case.ab},
                      "events":[dict(e,cycle=e["cycle"]-start) for e in events if "cycle" in e and e.get("generation")==gen]}}
