"""The performance of a pi0.5 action chunk on the f2-vla target as a Latency Petri Net (LPN,
EPFL DSLab's Petri-net performance models: dslab-epfl/lpn's `lpnlang`, and MJChku/LPNDialect's
compiled nets as the hardware loop's LPN tier, tools/hwloop/lpn on the hwloop branch),
calibrated against the Verilator cycles of the cut phases (docs/pi05.md 5.4), then run over the
whole chunk's programs for its seconds and Hz (docs/pi05.md 5.5). An estimate between the ISA
tally (tools/pi05_cost.py: each unit alone, every unit overlapped or none) and the RTL: the
tally's per-unit rates, the ISA's dependency structure, and the slice's units as resources.

    OTPU_TARGET=f2-vla OTPU_MCOLS=32 OTPU_LANES=64 OTPU_PACK=1 \
      python -I tools/pi05_lpn.py --ckpt DIR [--backend lpnlang|dialect|both] [--upstream] \
          [--phase v_out,...] [--np 558] [--rows 128,255,64] [--full] [--mhz 200] [--json out.json]

Two backends, the same slice model, so that one checks the other:

  lpnlang   dslab-epfl/lpn's Python package (`--lpn`: its clone's language/src, the Mac's
            ~/ergodex-03/lpn; it imports numpy, sympy and z3-solver; nothing is vendored). The
            net below is built from its Place / Transition / Token constructs and run with its
            event loop (lpn_simulate.lpn_sim_one_step). A phase runs in a second.
  dialect   the LPN tier's model: LPNDialect examples/opentpu's CompactOpenTPU (validated on
            Gemma-3 decode: +0.9% at PORTB 1, +1-4% on the wide-port points), driven by the ISA
            simulator's command trace of the phase program (this tree's simulator on a zero
            image, as tools/pi05_rtl.py runs the RTL), its MLIR compiled by lpn-compile into a
            native simulator (`--dialect`, `--lpn-compile`, `--llvm-bin`: ~/ergodex-03/lpn-scratch's
            clone, build and LLVM 20; `--work`: the MLIR, binaries and logs). What the tier lacks
            for f2-vla is added here in VlaCompactOpenTPU / VlaUnit / VlaMatrix (copies of its
            _slice, ExecutionUnit and MatrixEngine with the changes marked "vla:"; the clone and
            the hwloop branch untouched): the packed MM (a pass of two weight rows), the drain's
            two result rows a pass at E lanes a step, the VPU's sums at 16 a cycle and its
            composites at VPU_CL, QST T at the measured word rate, the LD / ST latencies, an ST's
            wait behind a QST's words, and the drain-first write arbitration. `--upstream` runs
            the tier's model as it is beside it (none of those: +26 to +103% on the phases).

The model (the lpnlang net; the dialect's places and transitions correspond):

  pprog --fetch--> the unit queues pq_dma / pq_mxu / pq_q / pq_vpu / pq_coll
      The sequencer: one instruction a cycle (LI / ADDI / LOOP / NOP take a fetch cycle and
      no window slot; a loop's back edge none), the dispatch window's WIN (16) slots a place
      of tokens taken at dispatch and returned at completion. At dispatch the instruction's
      footprint (opentpu/waits.py footprint(): otpu_pkg's TMEM / DRAM / ACT RAM ranges, read and
      written, from the registers the expansion resolved) is compared with every in-flight
      instruction's (waits.conflict(): RAW, WAR, WAW; conflict_dram() for an MM's stream
      start): the dependencies the scoreboard records (otpu_seq).
  pq_dma + pcap_dma --launch--> pdma_run --done--> pcap_dma, pslot
      The DMA (LD / ST): one instruction at a time (otpu_dma rdy = !busy; an ST until the
      DRAM acknowledged its writes), the oldest READY instruction of its queue (otpu_seq: not
      its oldest), in start order. Cycles: ceil(words / min(D / 4, LANES)) (profile._describe)
      + L_LD (the read latency: otpu_native_mem's +axi_lat 20 and its bank model) or + L_ST
      (the write acknowledgement), + L_WQ while a QST is in flight (wr_idle: no write of the
      slice outstanding, so behind the QST's words in the adapter's queue). A dependency on an
      older DMA instruction already started is satisfied (otpu_seq same_started).
  pq_q + pcap_q --launch--> pq_run --done--> pcap_q, pslot
      The quantizer: QACT 2 ceil(rows KB D / LANES), QST the reads at LANES a cycle (two passes
      with T) and its port A words (the K stores a word a cycle, the V^T stores a word of four
      rows, QST T) at QST_WC cycles each, + L_Q (the scale unit) + Q_ROW a row (the chunk max's
      third stage, plan 6c; QACT + Q_ROW once: its rows pipeline through the row buffer).
  pq_vpu + pvpu_issue --launch--> pvpu_issuing --issued--> pvpu_issue, pvpu_fly --done--> pslot
      The VPU: an issue pipeline (one instruction issuing at a time) and a completion
      pipeline (several in flight; a VOP starts while others are in flight only if its
      latency is at least theirs, the reductions alone: otpu_vpu's header, kept by the launch
      guard), the oldest ready VOP first. Issue cycles rows x ceil(cols / W): W = LANES for the
      elementwise functions and RMAX / ARGMAX, VPU_CL (16) for the composites (EXP2, EXP2SUB,
      RECIP, RSQRT, LOG2: the long lanes), 16 for RSUM / RSSQ / RDOT (the 16-lane
      sub-datapath); the measured rates of docs/pi05.md 5.4, where the tally has every VOP at
      LANES. Latency: the slot program's slots x SL (7) + the read / write stages.
  pq_mxu + pmxu_acc(2) --accept--> pmxu_accd --stream--> pmxu_sok, and pmxu_rel
  pmxu_rel + pmxu_core --release--> pmxu_run;  pmxu_run + pmxu_sok --done--> pmxu_core, pmxu_acc, pslot
      The MXU: strictly in order, two commands held (the issuer streams the newer one's
      weights while the consumer finishes the older: pmxu_acc), accepted once its DRAM
      dependencies are clear (ustart), released once all are (urel; a dependency on an MM
      already started is satisfied). The stream: N x KB D-byte chunks at one a cycle on port
      B (PORTB 1), after the previous stream, slowed by the DMA's chunks (the DMA goes first on
      port B: those queued or in flight at the accept, and those moved before the release),
      and at most FIFO_DEPTH pair slots ahead of the consumer. The consume: pops = ceil(N / 2)
      x KB x ceil(M / MCOLS) advances packed (N x KB x ... unpacked; a KB 1 packed pass every
      other cycle), the drain 2 ceil(N / 2) x ceil(M / MCOLS) x ceil(MCOLS / E) steps (two
      result rows a packed pass, E lanes a step; E 32 on f2-vla), the result FIFO's 64 rows
      making the pops wait for the drain: the MM ends at max(release + max(pops, drain), the
      stream's end + the last row's advances) + L_MM (the array's LROW = CL + MCOLS + 5 and the
      lanes' LLAT 9 of otpu_mxu); the DMA's LDs queued at the release add their TMEM write
      cycles to the drain (the DMA first there too).
  The TMEM write port between the drain and the VPU (the measured policy TMEM_WARB 2: the
      drain first, docs/pi05.md 5.4; the drain is denied 369 cycles on v_out): the drain is
      never held; a VOP that writes every cycle (elementwise, composites; the reductions write
      a word a row) issuing while an MM is in flight gets the write port 1 - d of the cycles,
      d the MM's drain steps over its cycles, until that MM ends. A mean-field approximation
      of the all-or-nothing bank arbiter (the VPU's 64-lane chunk meets the drain's 32 lanes
      whenever both write), computed at the VOP's issue; the dialect, without a clock in its
      DSL, bounds the loss by the MM's remaining drain cycles at its last batch instead.
  Not modelled: port A (the MXU's scale lines and QST's words have their own paths), the
      TMEM read ports (replicated / pumped on the board build: reads never conflict), the LD
      writes denying the VPU, the collective (S = 1), the host's launches between programs.

The constants are the RTL's (otpu_mxu LROW / LLAT, otpu_vpu SL and slot counts, otpu_quant's
scale unit, otpu_native_mem's read latency) except three of the memory path fitted on the
measured phases: L_ST 40 and L_WQ 120 (an ST's acknowledgement, and its wait behind a QST's
words: p_qkv's 4,464 and v_qkv's 2,048 STs, the DMA busy 85% on v_qkv) and QST_WC 1.85 (the
quantizer busy 534 K cycles for 270 K words on v_qkv). Calibration (docs/pi05.md 5.5): the
lpnlang backend within -11 .. +5% of the seven measured phases (mean |error| 4.6%), the
dialect within -18 .. +5% (6.6%); both read low on p_attend, whose profile shows what neither
carries (the LD writes denying the drain 40 K cycles, the quantizer busy 34%).

The lpnlang simulator consumes a transition's tokens when it fires, so every place here has
one consuming transition (the queues' launch guards pick the oldest ready token and rotate it
to the head), as lpn_family's VTA model does; its event loop is NumPy 1.x's (np.Inf aliased).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
SCRATCH = Path(os.environ.get("HWLOOP_LPN_SCRATCH", str(Path.home() / "ergodex-03" / "lpn-scratch")))

from opentpu import isa as I  # noqa: E402
from opentpu import profile as PF  # noqa: E402
from opentpu import waits as W  # noqa: E402

M32 = 0xFFFFFFFF
SEQ_OPS = (I.LI, I.ADDI, I.NOP, I.LOOP)
U_DMA, U_MXU, U_Q, U_VPU, U_COLL = 0, 1, 2, 3, 4
UNAMES = ("DMA", "MXU", "QUANT", "VPU", "COLL")
COMPOSITE = {I.V_EXP2, I.V_EXP2SUB, I.V_RECIP, I.V_RSQRT, I.V_LOG2}
SUMS = {I.V_RSUM, I.V_RSSQ, I.V_RDOT}
REDUCE = SUMS | {I.V_RMAX, I.V_ARGMAX}
# otpu_vpu: SL = 1 + LM + LA = 7 cycles a slot; the slot programs (its header): ADD / SUB /
# RSUB / MUL / OUTER 1, EXP2 / EXP2SUB 9 (+3 of range reduction), RECIP 6, RSQRT 10, LOG2 10,
# MAX / MIN / COPY / ABS / FILL 0 (written the cycle after the read); the sums' partial loop
# (RL 4) and the folding tree (log2 16 levels of LA 4)
SL = 7
SLOTS = {I.V_ADD: 1, I.V_SUB: 1, I.V_RSUB: 1, I.V_MUL: 1, I.V_OUTER: 1, I.V_EXP2: 9,
         I.V_EXP2SUB: 9, I.V_RECIP: 6, I.V_RSQRT: 10, I.V_LOG2: 10}
VBASE = 6                 # the TMEM read (pumped: 3), the lanes' input stage, the write

# the measured cycles (docs/pi05.md 5.4; Verilator, the target's board uarch and memory path,
# drain first, LANES 64, PACK): phase -> (MCOLS, E, cycles)
MEASURED = {"v_out": (32, 32, 627586), "p_qkv": (32, 32, 838610), "e_step": (32, 32, 466936),
            "p_attend": (32, 32, 718667), "v_qkv": (32, 32, 587037),
            "v_out@16": (16, 16, 876803), "e_step@16": (16, 16, 600209)}


@dataclasses.dataclass
class Knobs:
    """The model's latency constants (cycles)."""
    l_mm: int = 72        # otpu_mxu: LROW (CL 16 + MCOLS 32 + 5) + LLAT 9 + the result FIFO / drain
    l_ld: int = 48        # an LD: otpu_native_mem +axi_lat 20, its bank model, the DMA's buffer
    l_st: int = 40        # an ST: until the HBM model acknowledged its writes (fitted: --lst)
    l_wq: int = 120       # ... and behind a QST's words in the adapter's write queue (fitted on
                          # v_qkv, the DMA busy 85% there over 2,048 STs: --lwq)
    l_q: int = 44         # otpu_quant: the scale unit (LAT 30), the prescale (QL 5), the passes
    q_row: int = 1        # ... and a cycle a row since the chunk max's third stage (plan 6c: the
                          # row's scale, so its pass 1, a cycle later; QST's rows run one after
                          # the other, +1 each; QACT's pipeline through the row buffer, +1 once)
    qst_wc: float = 1.85  # cycles a QST port-A word on the HBM model's write path (measured:
                          # v_qkv's quantizer busy 534 K cycles for 270 K words and 33 K read
                          # cycles, docs/pi05.md 5.4)
    fifo: int = 1024      # FIFO_DEPTH pair slots the stream may run ahead
    win: int = 16         # the dispatch window
    vpu_cl: int = 16      # VPU_CL: the composites' lanes
    sum_w: int = 16       # the sums' sub-datapath
    warb: int = 2         # 2 the drain first (the target), 0 no contention modelled
    dma_lookahead: bool = True


# ------------------------------------------------------------------- the expanded program
@dataclasses.dataclass
class Dyn:
    idx: int
    pc: int
    ins: object
    unit: int
    fcyc: int             # fetch cycles charged: 1 + the sequencer-only instructions before it
    fp: object            # waits.Fp
    name: str
    cyc: int = 0          # the unit's busy cycles (the tally's rate + latency)
    lat: int = 0          # VPU: completion latency after the issue; MM: L_MM
    pb: int = 0           # port B chunks
    wr: int = 0           # VPU: writes every issue cycle (1) or a word a row (0)
    kind: str = ""        # VPU: ew / comp / red
    pops: int = 0
    drain: int = 0
    tail: int = 0         # MM: the last row's advances
    reads: int = 0        # QACT / QST: the TMEM read cycles (stretched while a VOP issues)
    deps: list = dataclasses.field(default_factory=list)
    deps_dram: list = dataclasses.field(default_factory=list)


def expand(prog, cfg, kn: Knobs, E: int, args=None) -> list:
    """The sequencer's walk of the program (isasim.Slice.step's LOOP / LI / ADDI / HALT), the
    run arguments R8..R15 (default 0), every unit instruction with its resolved footprint."""
    R = [0] * 16
    for k, v in (args or {}).items():
        R[k] = v & M32
    pc, stack, out, seq = 0, [], [], 0
    reg = lambda r: 0 if r == 0 else R[r]
    while True:
        ins = prog[pc]
        op = ins.op
        if op == I.HALT:
            break
        if op == I.LOOP:
            count = (reg(ins.ra) + ins.w[1]) & M32
            body = ins.w[0]
            seq += 1
            if count == 0:
                pc += 1 + body
                continue
            stack.append([pc + 1, pc + body, count])
            pc += 1
            continue
        if op in SEQ_OPS:
            seq += 1
            if op == I.LI and ins.rd:
                R[ins.rd] = ins.w[0] & M32
            elif op == I.ADDI and ins.rd:
                R[ins.rd] = (reg(ins.ra) + ins.w[0]) & M32
        else:
            u = W.UNIT.get(op)
            if u is None:
                raise ValueError(f"pc {pc}: op {op:#x} is not a unit instruction")
            fp = W.footprint(ins, [(r, r) for r in R], cfg.D, cfg.S)
            d = Dyn(len(out), pc, ins, u, 1 + seq, fp, "")
            cost(d, cfg, kn, E)
            out.append(d)
            seq = 0
        # advance (isasim.Slice.advance): a loop's last instruction jumps back without a bubble
        if stack and stack[-1][1] == pc:
            top = stack[-1]
            if top[2] > 1:
                top[2] -= 1
                pc = top[0]
                continue
            stack.pop()
        pc += 1
    return out


def cost(d: Dyn, cfg, kn: Knobs, E: int) -> None:
    """The instruction's cycles at the units' nominal rates (profile._describe's, tools/
    pi05_cost.py's model) and the RTL's latencies."""
    ins, w = d.ins, d.ins.w
    name, _, cyc, pb, _ = PF._describe(ins, cfg)
    d.name, d.pb = name, pb
    op = ins.op
    if op == I.MM:
        N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[5] >> 16) & 0xFF
        wf, pair, quad = I.mm_format(ins)
        pack = bool(getattr(cfg, "PACK", False)) and wf == I.W8 and not pair and not quad
        G = -(-M // cfg.MCOLS)
        passes = -(-N // 2) if pack else N
        kb = max(KB, 2) if pack else KB                   # a KB 1 packed pass pops every other cycle
        d.pops = passes * kb * G
        d.drain = passes * G * (2 if pack else 1) * -(-cfg.MCOLS // E)
        d.tail = G * kb
        d.lat = kn.l_mm
        d.cyc = max(d.pops, d.drain) + kn.l_mm
    elif op == I.VOP:
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        f = (w[5] >> 16) & 0xFF
        if f in COMPOSITE:
            d.kind, wd, d.wr = "comp", kn.vpu_cl, 1
            d.lat = VBASE + 3 + SL * SLOTS[f]
        elif f in SUMS:
            d.kind, wd, d.wr = "red", kn.sum_w, 0
            d.lat = VBASE + 4 + 4 * 4 + 4
        elif f in REDUCE:
            d.kind, wd, d.wr = "red", cfg.LANES, 0
            d.lat = VBASE + 4 * 4
        else:
            d.kind, wd, d.wr = "ew", cfg.LANES, 1
            d.lat = VBASE + SL * SLOTS.get(f, 0)
        d.cyc = rows * -(-cols // wd)
    elif op == I.QACT:
        d.reads = cyc
        d.cyc = cyc + kn.l_q + kn.q_row
    elif op == I.QST:
        # the reads at LANES a cycle (two passes with T), the words on port A at qst_wc
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        n = rows * KB * cfg.D
        reads = (2 if ins.flags & I.F_T else 1) * -(-n // cfg.LANES)
        d.reads = reads
        d.cyc = reads + int(math.ceil((cyc - reads) * kn.qst_wc)) + kn.l_q + rows * kn.q_row
    elif op == I.LD:
        d.cyc = cyc + kn.l_ld
    elif op == I.ST:
        d.cyc = cyc + kn.l_st
    else:
        d.cyc = max(cyc, 1)


# ------------------------------------------------------------------------------ the net
class Net:
    """The slice as an LPN over the expanded program `dyn`."""

    def __init__(self, dyn: list, kn: Knobs):
        from lpnlang import Place, Transition, Token, InWeightFunc, OutWeightFunc, GuardFunc
        from lpnlang import DelayFunc
        self.dyn, self.kn = dyn, kn
        self.now = 0
        self.done = {}                       # idx -> completion cycle
        self.started = set()                 # idx: started on its unit (MM: accepted)
        self.inflight = collections.OrderedDict()   # idx -> Dyn, dispatched, not completed
        self.pb_free = 0                     # port B: the MXU's streams end to end
        self.mm_run = None                   # (end cycle, drain share) of the MM in flight
        self.q_run = 0                       # the end cycle of the QST in flight (wr_idle)
        self.dma_pb = 0                      # port B chunks the DMA has moved so far
        self.dma_at_accept = {}              # MM idx -> dma_pb at its accept
        self.stream_end = {}                 # idx -> cycle the MM's chunks are all in
        self.vfly = {}                       # VOP idx -> (latency, kind) in the pipeline
        self.vbusy = collections.Counter()   # per unit: busy cycles (profile)
        self.t_disp, self.t_start = {}, {}   # idx -> dispatch / start cycle (the diagnostic)
        self.stretch = 0                     # VPU issue cycles added by the drain's writes
        self.T = []
        P = self.P = {}
        for n in ("pprog", "pslot", "pq_dma", "pq_mxu", "pq_q", "pq_vpu", "pq_coll",
                  "pcap_dma", "pcap_q", "pcap_coll", "pdma_run", "pq_run", "pcoll_run",
                  "pvpu_issue", "pvpu_issuing", "pvpu_fly", "pmxu_acc", "pmxu_accd",
                  "pmxu_sok", "pmxu_core", "pmxu_run"):
            P[n] = Place(n)
        P["pprog"].assign_marking(collections.deque(Token({"idx": d.idx}) for d in dyn))
        for n, k in (("pslot", kn.win), ("pcap_dma", 1), ("pcap_q", 1), ("pcap_coll", 1),
                     ("pvpu_issue", 1), ("pmxu_acc", 2), ("pmxu_core", 1)):
            P[n].assign_marking(collections.deque(Token() for _ in range(k)))
        net = self

        # ---- edge functions (lpnlang's constructs; the bodies read the net's state)
        one = InWeightFunc("one")
        one.install(lambda b: 1)
        empty = GuardFunc("empty")
        empty.install(lambda b: True)
        tok = OutWeightFunc("tok")
        tok.install(lambda b, p: p.push_token(Token()))
        zero = DelayFunc("zero")
        zero.install(lambda b: 0)

        def head(place):
            return dyn[place.tokens[0].idx]

        def ready(d: Dyn, unit: int, dram: bool = False) -> bool:
            deps = d.deps_dram if dram else d.deps
            for j in deps:
                if j in net.done:
                    continue
                if unit != U_VPU and j in net.started and dyn[j].unit == unit:
                    continue                   # otpu_seq same_started
                return False
            return True

        def pick(place, unit, test) -> bool:
            """The oldest token of the queue that `test` admits, rotated to the head (the
            launch consumes the head)."""
            tk = place.tokens
            for i in range(len(tk)):
                if test(dyn[tk[i].idx]):
                    if i:
                        t = tk[i]
                        del tk[i]
                        tk.appendleft(t)
                    return True
            return False

        # fetch: one instruction a cycle into the window, its dependencies on the in-flight ones
        fdelay = DelayFunc("fdelay")
        fdelay.install(lambda b: head(P["pprog"]).fcyc)
        fetch_out = {}
        for u, n in enumerate(("pq_dma", "pq_mxu", "pq_q", "pq_vpu", "pq_coll")):
            f = OutWeightFunc(f"fetch_{n}")

            def body(b, p, u=u):
                d = dyn[b["pprog.0.idx"]]
                if d.unit != u:
                    return
                for j, e in net.inflight.items():
                    if W.conflict(d.fp, e.fp):
                        d.deps.append(j)
                        if d.ins.op == I.MM and W.conflict(d.fp, e.fp, dram_only=True):
                            d.deps_dram.append(j)
                net.inflight[d.idx] = d
                net.t_disp[d.idx] = b["ts"]
                p.push_token(Token({"idx": d.idx}))
            f.install(body)
            fetch_out[n] = f
        self.T.append(Transition(id="fetch", delay_func=fdelay(), pi=[P["pprog"], P["pslot"]],
                                 pi_w=[one(), one()], pi_guard=[empty(), empty()],
                                 po=[P[n] for n in fetch_out], po_w=[f() for f in fetch_out.values()]))

        # the in-order single-server units: the DMA, the quantizer, the collective
        for u, q, cap, run in ((U_DMA, "pq_dma", "pcap_dma", "pdma_run"),
                               (U_Q, "pq_q", "pcap_q", "pq_run"),
                               (U_COLL, "pq_coll", "pcap_coll", "pcoll_run")):
            g = GuardFunc(f"ready_{q}")
            g.install(lambda b, u=u, q=q: pick(P[q], u, lambda d: ready(d, u)))
            mv = OutWeightFunc(f"launch_{q}")

            def launch(b, p, q=q):
                d = head(P[q])
                net.started.add(d.idx)
                net.t_start[d.idx] = b["ts"]
                if d.ins.op == I.QST:
                    net.q_run = b["ts"] + d.cyc
                p.push_token(Token({"idx": d.idx}))
            mv.install(launch)
            self.T.append(Transition(id=f"launch_{q}", delay_func=zero(), pi=[P[q], P[cap]],
                                     pi_w=[one(), one()], pi_guard=[g(), empty()],
                                     po=[P[run]], po_w=[mv()]))
            dl = DelayFunc(f"cyc_{run}")

            def unit_cycles(b, run=run):
                d = head(P[run])
                n = d.cyc
                if d.ins.op == I.ST and net.q_run > P[run].tokens[0].ts:
                    # an ST completes at wr_idle (otpu_dma): no DRAM write of the slice
                    # outstanding, so behind the QST in flight's words in the adapter's queue
                    n += net.kn.l_wq
                return n
            dl.install(unit_cycles)
            fin = OutWeightFunc(f"done_{run}")

            def finish(b, p, run=run):
                d = head(P[run])
                net.complete(d, b["ts"])
                net.dma_pb += d.pb
                p.push_token(Token())
            fin.install(finish)
            self.T.append(Transition(id=f"done_{run}", delay_func=dl(), pi=[P[run]], pi_w=[one()],
                                     pi_guard=[empty()], po=[P[cap], P["pslot"]],
                                     po_w=[fin(), tok()]))

        # the VPU: issue, then the pipeline
        def vcan(d: Dyn) -> bool:
            if not ready(d, U_VPU):
                return False
            if d.kind == "red":
                return not net.vfly
            return all(k != "red" and d.lat >= lat for lat, k in net.vfly.values())
        vg = GuardFunc("vready")
        vg.install(lambda b: pick(P["pq_vpu"], U_VPU, vcan))
        vmv = OutWeightFunc("vlaunch")

        def vlaunch(b, p):
            d = head(P["pq_vpu"])
            net.started.add(d.idx)
            net.t_start[d.idx] = b["ts"]
            net.vfly[d.idx] = (d.lat, d.kind)
            p.push_token(Token({"idx": d.idx}))
        vmv.install(vlaunch)
        self.T.append(Transition(id="launch_vpu", delay_func=zero(), pi=[P["pq_vpu"], P["pvpu_issue"]],
                                 pi_w=[one(), one()], pi_guard=[vg(), empty()],
                                 po=[P["pvpu_issuing"]], po_w=[vmv()]))
        vis = DelayFunc("vissue")

        def vissue(b):
            d = head(P["pvpu_issuing"])
            t0 = P["pvpu_issuing"].tokens[0].ts
            return net.issue_cycles(d, t0)
        vis.install(vissue)
        vmv2 = OutWeightFunc("vissued")

        def vissued(b, p):
            d = head(P["pvpu_issuing"])
            p.push_token(Token({"idx": d.idx}))
        vmv2.install(vissued)
        self.T.append(Transition(id="issued_vpu", delay_func=vis(), pi=[P["pvpu_issuing"]],
                                 pi_w=[one()], pi_guard=[empty()],
                                 po=[P["pvpu_issue"], P["pvpu_fly"]], po_w=[tok(), vmv2()]))
        vlat = DelayFunc("vlat")
        vlat.install(lambda b: head(P["pvpu_fly"]).lat)
        vfin = OutWeightFunc("vdone")

        def vdone(b, p):
            d = head(P["pvpu_fly"])
            net.complete(d, b["ts"])
            del net.vfly[d.idx]
            p.push_token(Token())
        vfin.install(vdone)
        self.T.append(Transition(id="done_vpu", delay_func=vlat(), pi=[P["pvpu_fly"]], pi_w=[one()],
                                 pi_guard=[empty()], po=[P["pslot"]], po_w=[vfin()]))

        # the MXU: accept (the stream), release (the consume), done
        ag = GuardFunc("mready")
        ag.install(lambda b: ready(head(P["pq_mxu"]), U_MXU, dram=True))
        amv = OutWeightFunc("maccept")

        def maccept(b, p):
            d = head(P["pq_mxu"])
            net.started.add(d.idx)
            t = b["ts"]
            start = max(t, net.pb_free)
            extra = 0
            if net.kn.dma_lookahead:
                for tk in list(P["pq_dma"].tokens) + list(P["pdma_run"].tokens):
                    extra += dyn[tk.idx].pb
            end = start + d.pb + extra
            net.pb_free = end
            net.stream_end[d.idx] = (start, end)
            net.dma_at_accept[d.idx] = net.dma_pb + extra
            p.push_token(Token({"idx": d.idx}))
        amv.install(maccept)
        # the accept puts the MM in two places: pmxu_accd (its stream, consumed by stream_mxu)
        # and pmxu_rel (its release; one consuming transition a place), the done transition
        # pairs the run's and the stream's tokens again
        P["pmxu_rel"] = Place("pmxu_rel")
        rmv = OutWeightFunc("to_release")
        rmv.install(lambda b, p: p.push_token(Token({"idx": head(P["pq_mxu"]).idx})))
        self.T.append(Transition(id="accept_mxu", delay_func=zero(), pi=[P["pq_mxu"], P["pmxu_acc"]],
                                 pi_w=[one(), one()], pi_guard=[ag(), empty()],
                                 po=[P["pmxu_accd"], P["pmxu_rel"]], po_w=[amv(), rmv()]))
        sdl = DelayFunc("stream")
        sdl.install(lambda b: max(0, net.stream_end[head(P["pmxu_accd"]).idx][1]
                                  - P["pmxu_accd"].tokens[0].ts))
        smv = OutWeightFunc("streamed")
        smv.install(lambda b, p: p.push_token(Token({"idx": head(P["pmxu_accd"]).idx})))
        self.T.append(Transition(id="stream_mxu", delay_func=sdl(), pi=[P["pmxu_accd"]], pi_w=[one()],
                                 pi_guard=[empty()], po=[P["pmxu_sok"]], po_w=[smv()]))
        rg = GuardFunc("rready")
        rg.install(lambda b: ready(head(P["pmxu_rel"]), U_MXU))
        rmv2 = OutWeightFunc("released")

        def released(b, p):
            d = head(P["pmxu_rel"])
            net.t_start[d.idx] = b["ts"]
            p.push_token(Token({"idx": d.idx}))
        rmv2.install(released)
        self.T.append(Transition(id="release_mxu", delay_func=zero(), pi=[P["pmxu_rel"], P["pmxu_core"]],
                                 pi_w=[one(), one()], pi_guard=[rg(), empty()],
                                 po=[P["pmxu_run"]], po_w=[rmv2()]))
        mdl = DelayFunc("mm")

        def mm_delay(b):
            d = head(P["pmxu_run"])
            rel = P["pmxu_run"].tokens[0].ts
            s0, s1 = net.stream_end[d.idx]
            drain = d.drain
            if net.kn.dma_lookahead:
                # the DMA's chunks since the accept (the DMA first on port B: each one a cycle
                # of the stream) beyond those counted then, and its LDs queued or in flight
                # (their TMEM writes deny the drain: the DMA first there too)
                extra = max(0, net.dma_pb - net.dma_at_accept[d.idx])
                for tk in list(P["pq_dma"].tokens) + list(P["pdma_run"].tokens):
                    e = dyn[tk.idx]
                    extra += e.pb
                    if e.ins.op == I.LD:
                        drain += e.cyc - net.kn.l_ld
                s1 += extra
            enabled = max(rel, s1)
            # the stream at most `fifo` pair slots ahead of the consumer
            ahead = min(net.kn.fifo * 2, max(0, rel - s0))
            port_bound = rel + max(0, d.pb - ahead)
            end = max(rel + max(d.pops, drain), s1 + d.tail, port_bound) + d.lat
            dur = max(1, end - rel)
            net.mm_run = (end, min(1.0, d.drain / dur))
            return end - enabled
        mdl.install(mm_delay)
        mfin = OutWeightFunc("mdone")

        def mdone(b, p):
            d = head(P["pmxu_run"])
            net.complete(d, b["ts"])
            net.mm_run = None
            p.push_token(Token())
        mfin.install(mdone)
        self.T.append(Transition(id="done_mxu", delay_func=mdl(), pi=[P["pmxu_run"], P["pmxu_sok"]],
                                 pi_w=[one(), one()], pi_guard=[empty(), empty()],
                                 po=[P["pmxu_core"], P["pmxu_acc"], P["pslot"]],
                                 po_w=[mfin(), tok(), tok()]))

    def issue_cycles(self, d: Dyn, t0: int) -> int:
        """A VOP's issue cycles, the drain's writes taking 1 - d of the write port while the
        MM in flight drains (the drain first)."""
        n = d.cyc
        if self.kn.warb == 2 and d.wr and self.mm_run is not None:
            end, share = self.mm_run
            r = max(0, end - t0)
            if n <= (1 - share) * r:
                n = int(math.ceil(n / (1 - share)))
            else:
                n = int(math.ceil(r + (n - (1 - share) * r)))
            self.stretch += n - d.cyc
        return max(1, n)

    def by_class(self) -> dict:
        """Per instruction class: count, busy (start to end), wait (dispatch to start), as
        opentpu.profile.by_class, and per unit the union of its busy intervals."""
        cls, iv = {}, collections.defaultdict(list)
        for d in self.dyn:
            a = cls.setdefault(d.name, {"count": 0, "busy": 0, "wait": 0})
            s, e = self.t_start.get(d.idx, 0), self.done.get(d.idx, 0)
            a["count"] += 1
            a["busy"] += e - s
            a["wait"] += s - self.t_disp.get(d.idx, 0)
            iv[d.unit].append((s, e))
        busy = {}
        for u, v in iv.items():
            tot, cs, ce = 0, None, None
            for x, y in sorted(v):
                if ce is None or x > ce:
                    tot += 0 if ce is None else ce - cs
                    cs, ce = x, y
                else:
                    ce = max(ce, y)
            busy[UNAMES[u]] = tot + (0 if ce is None else ce - cs)
        return {"classes": cls, "unit_busy": busy, "vpu_stretch": self.stretch}

    def complete(self, d: Dyn, t: int) -> None:
        self.done[d.idx] = t
        self.inflight.pop(d.idx, None)
        self.vbusy[d.unit] += d.cyc

    def run(self, max_events: int = 50_000_000) -> int:
        from lpnlang.lpn_simulate import lpn_sim_one_step
        last = 0
        for _ in range(max_events):
            t, en = lpn_sim_one_step(self.T)
            if t < 0:
                break
            last = t
        if len(self.done) != len(self.dyn):
            raise RuntimeError(f"LPN stopped at {last} with {len(self.done)} of {len(self.dyn)} "
                               "instructions complete (a deadlock in the net)")
        return last


# ------------------------------------------------------------------ the LPNDialect backend
# MJChku/LPNDialect's examples/opentpu (the hardware loop's LPN tier, tools/hwloop/lpn on the
# hwloop branch: CompactOpenTPU validated on Gemma-3 decode to +0.9% at PORTB 1 and +1-4% on the
# wide-port points), compiled by lpn-compile into a native simulator. The same slice, the
# same dependency rule (waits.footprint / conflict over the ISA simulator's command trace),
# driven here with the pi0.5 phase programs on this tree's ISA simulator. What the tier lacks
# for f2-vla is added below, in VlaCompactOpenTPU / VlaUnit / VlaMatrix (copies of compact.py's
# _slice and components.py's ExecutionUnit / MatrixEngine with the changes marked "vla:"), the
# clone untouched: the packed MM (a pass of two weight rows: 2 KB chunks on port B, KB advances,
# two result rows a pass, E lanes a step), the VPU's sums at 16 a cycle and the composites at
# VPU_CL, QST T at the measured word rate, the LD / ST latencies, an ST's wait behind a QST's
# words (a place counting the QSTs in flight), and the drain-first write arbitration as a rate
# factor on a writing VOP's latency (latency x 1 / (1 - d), d the drain's share of the MM in
# flight, kept in a place the matrix engine updates; the DSL has no clock to bound it by the
# MM's remaining cycles as the lpnlang backend does, so d is capped at 3/4).


def dialect_classes(kn: Knobs, E: int):
    """The subclasses, built once the clone is on sys.path (the DSL reads their source)."""
    from math import ceil, gcd
    from lpnlang_mlir import InputArray
    from examples.opentpu.compact import CompactOpenTPU, dependencies
    from examples.opentpu.components import CommandInterface, CommandTiming
    from examples.opentpu.model import DRAM, SystolicArray

    class VlaUnit:
        """components.ExecutionUnit with the ST / QST backlog and the VPU's drain factor."""
        def __init__(self, net, prefix, kind, interface, parameters, memory, xkinds, qst_active, drain):
            self.resource = resource = net.place(prefix + "_resource", initial_tokens=1)
            slots, pending, window = interface.slots, interface.pending[kind], interface.window
            select_ready, publish = interface.select_ready, interface.publish
            latencies, port_ids, services = parameters.latencies, parameters.port_ids, parameters.services
            ports = [memory.b, memory.sw]
            qst_done = net.place(prefix + "_qst_done")
            l_wq = kn.l_wq

            @net.transition(prefix + "_execute")
            def execute():
                guard(count(resource) > 0)
                guard(count(pending) > 0)
                chosen = select_ready(const_i64(kind))
                guard(chosen >= 0)
                slot_places = array(slots)
                observed = peek(slot_places[chosen])
                idx = observed.get("id")
                bus_id = array(port_ids)[idx]
                buses = array(ports)
                guard(select(bus_id >= 0, count(buses[select(bus_id >= 0, bus_id, 0)]) > 0, True, typ="i1"))
                token = take(slot_places[chosen])
                take(pending)
                emit(slot_places[chosen], create())
                xk = array(xkinds)[idx]
                base = array(latencies)[idx]
                # vla: an ST completes at wr_idle, behind a QST's words (xk 1); a writing VOP
                # (xk 3) gets 1 - d of the write port while the drain's share is d
                backlog = select(xk == 1, select(count(qst_active) > 0, l_wq, 0), 0)
                dstate = peek(drain)
                share = dstate.get("share")
                cap = dstate.get("cap")
                extra = divi(muli(base, share), 1000 - share)
                stretched = addi(base, select(extra < cap, extra, cap))
                cycles = addi(select(xk == 3, stretched, base), backlog)
                latency = sitofp(cycles)
                credit = take(resource)
                emit(resource, credit, delay=latency)
                if bus_id >= 0:
                    bus = take(buses[bus_id])
                    emit(buses[bus_id], bus, delay=sitofp(array(services)[idx]))
                if xk == 2:
                    emit(qst_active, create())
                    emit(qst_done, create(), delay=latency)
                publish(idx, latency)
                emit(window, token, delay=latency)

            @net.transition(prefix + "_qst_clear")
            def qst_clear():
                guard(count(qst_done) > 0)
                guard(count(qst_active) > 0)
                take(qst_done)
                take(qst_active)

    class VlaMatrix:
        """components.MatrixEngine, publishing the MM in flight's drain share."""
        def __init__(self, net, prefix, cfg, timing, interface, parameters, memory, kind, shares, drains,
                     drain, buffer_metadata=None):
            t = timing
            self.resources = resources = SystolicArray(net, prefix, t,
                max(b * c for b, c in zip(parameters.batches, parameters.chunks)),
                initial_buffers=0 if buffer_metadata is not None else None)
            self.prefetch_credit = prefetch_credit = net.place(prefix + "_prefetch_credit", initial_tokens=1)
            self.request = request = net.place(prefix + "_request")
            self.ready = ready = net.place(prefix + "_ready")
            slots, pending, window = interface.slots, interface.pending[kind], interface.window
            select_ready, is_done, publish = interface.select_ready, interface.is_done, interface.publish
            offsets, indices = interface.compute_offsets, interface.compute_indices
            rows, batches, chunks = parameters.rows, parameters.batches, parameters.chunks
            scaled, kbs = parameters.scaled, parameters.kb
            row_services, tails = parameters.row_services, parameters.tails
            drain_reset = net.place(prefix + "_drain_reset")

            if buffer_metadata is not None:
                initialize = net.place(prefix + "_initialize_buffers", initial_tokens=1)

                @net.transition(prefix + "_initialize_buffer_credits")
                def initialize_buffer_credits():
                    guard(count(initialize) > 0)
                    take(initialize)
                    for b in range(array(buffer_metadata)[1]):
                        emit(resources.buffers, create())

            @net.transition(prefix + "_start")
            def start():
                guard(count(resources.commands) > 0)
                guard(count(prefetch_credit) > 0)
                guard(count(pending) > 0)
                chosen = select_ready(const_i64(kind))
                guard(chosen >= 0)
                place = array(slots)[chosen]
                token = take(place)
                take(pending)
                emit(place, create())
                take(resources.commands)
                take(prefetch_credit)
                emit(request, token.set("row", 0))

            @net.transition(prefix + "_prefetch")
            def prefetch():
                guard(count(request) > 0)
                guard(count(resources.buffers) > 0)
                guard(count(memory.b) > 0)
                idx = peek(request).get("id")
                needs_scale = array(scaled)[idx]
                guard(count(memory.a) >= needs_scale)
                token = take(request)
                row = token.get("row")
                n, batch, chunk = array(rows)[idx], array(batches)[idx], array(chunks)[idx]
                width = select(row + batch < n, batch, n - row)
                take(resources.buffers)
                bus = take(memory.b)
                service = sitofp(divi(muli(muli(width, chunk), cfg.D) + t.dram_bytes_per_cycle - 1, t.dram_bytes_per_cycle))
                emit(memory.b, bus, delay=service)
                if needs_scale > 0:
                    scale_port = take(memory.a)
                    emit(memory.a, scale_port, delay=sitofp(muli(width, array(kbs)[idx])))
                emit(ready, token.set("width", width), delay=service + float(t.dram_latency + 4))
                if row + batch < n:
                    emit(request, token.set("row", row + batch), delay=service)
                else:
                    emit(prefetch_credit, create(), delay=service)

            @net.transition(prefix + "_issue_rows")
            def compute():
                guard(count(ready) > 0)
                guard(count(resources.compute) > 0)
                guard(count(drain) > 0)
                idx = peek(ready).get("id")
                ci = array(indices)
                enabled = const_i64(0) == 0
                for d in range(array(offsets)[idx], array(offsets)[idx + 1]):
                    enabled = select(enabled, is_done(ci[d]), False, typ="i1")
                guard(enabled)
                token = take(ready)
                port = take(resources.compute)
                service = sitofp(muli(token.get("width"), array(row_services)[idx]))
                emit(resources.compute, port, delay=service)
                emit(resources.buffers, create())
                # vla: the drain's share of this MM for the VPU's writers, and its remaining
                # drain cycles (the passes left after this batch) as the bound of their loss
                dtoken = take(drain)
                left = select(token.get("row") + array(batches)[idx] < array(rows)[idx],
                              array(rows)[idx] - token.get("row") - array(batches)[idx], 0)
                emit(drain, dtoken.set("share", array(shares)[idx]).set("cap", muli(left + 1, array(drains)[idx])))
                if token.get("row") + array(batches)[idx] >= array(rows)[idx]:
                    latency = service + sitofp(array(tails)[idx])
                    publish(idx, latency)
                    emit(resources.commands, create(), delay=latency)
                    emit(window, token, delay=latency)
                    emit(drain_reset, create(), delay=latency)

            @net.transition(prefix + "_drain_clear")
            def drain_clear():
                guard(count(drain_reset) > 0)
                guard(count(drain) > 0)
                take(drain_reset)
                dtoken = take(drain)
                emit(drain, dtoken.set("share", 0).set("cap", 0))

    class VlaCompactOpenTPU(CompactOpenTPU):
        """compact.CompactOpenTPU with the f2-vla items (the header above)."""
        def __init__(self, workload, timing, input_capacity=None):
            self.kn, self.E = kn, E
            super().__init__(workload, timing, input_capacity)

        def _simple_timing(self, ins, memory, I):
            cfg, t, w = self.workload.config, self.timing, ins.w
            if ins.op == I.VOP:
                rows, cols = w[3] & 65535, w[3] >> 16
                f = (w[5] >> 16) & 255
                # vla: the measured rates (docs/pi05.md 5.4): composites at VPU_CL, the sums at
                # 16 a cycle, the rest at LANES; the slot programs' latency
                if f in COMPOSITE:
                    wd, lat = kn.vpu_cl, VBASE + 3 + SL * SLOTS[f]
                elif f in SUMS:
                    wd, lat = kn.sum_w, VBASE + 24
                elif f in REDUCE:
                    wd, lat = cfg.LANES, VBASE + 16
                else:
                    wd, lat = cfg.LANES, VBASE + SL * SLOTS.get(f, 0)
                return rows * ceil(cols / wd) + lat, None, 0
            if ins.op == I.QST:
                rows, KB = w[3] & 65535, w[3] >> 16
                n = rows * KB * cfg.D
                _, _, cyc, _, _ = PF._describe(ins, cfg)
                reads = (2 if ins.flags & I.F_T else 1) * ceil(n / cfg.LANES)
                latency = reads + int(ceil((cyc - reads) * kn.qst_wc)) + kn.l_q + rows * kn.q_row
                return latency, memory.sw, latency
            if ins.op == I.QACT:
                _, _, cyc, _, _ = PF._describe(ins, cfg)
                return cyc + kn.l_q + kn.q_row, None, 0
            if ins.op in (I.LD, I.ST):
                service = max(ceil(w[2] / min(cfg.LANES, cfg.D // 4)), ceil(4 * w[2] / t.dram_bytes_per_cycle))
                return service + (kn.l_ld if ins.op == I.LD else kn.l_st), memory.b, service
            return super()._simple_timing(ins, memory, I)

        def _slice(self, sid, commands, I, W):
            cfg, t, net = self.workload.config, self.timing, self.net
            prefix = f"s{sid}"
            ncommands = len(commands)
            if not ncommands:
                raise ValueError("empty command trace")
            memory = DRAM(net, prefix + "_dram", t)
            nsig = self.input_capacity or ncommands
            banks = [net.place(prefix + f"_flags_{b}", initial_tokens=1) for b in range(nsig // 64 + 1)]
            signals = net.place(prefix + "_signals")
            finished = net.place(prefix + "_halted")

            @net.func(name=prefix + "_is_done", args=("i64",), results=("i1",))
            def is_done(idx):
                token = peek(array(banks)[divi(idx, 64)])
                mask = shli(1, remi(idx, 64))
                func_return(select(idx == nsig, True, andi(token.get("bits"), mask) != 0, typ="i1"))

            @net.func(name=prefix + "_publish", args=("i64", "f64"), results=())
            def publish(idx, latency):
                emit(signals, create().set("id", idx), delay=latency)
                func_return()
            slots = [net.place(prefix + f"_slot_{s}", initial_tokens=1) for s in range(t.window)]
            window = net.place(prefix + "_window")
            initialize = net.place(prefix + "_initialize", initial_tokens=1)
            cursor = net.place(prefix + "_cursor", initial_tokens=1)
            pipeline = net.place(prefix + "_pipeline")
            pending = [net.place(prefix + f"_pending_{u}") for u in range(6)]
            # vla: the QSTs in flight, the MM in flight's drain share (per mille)
            qst_active = net.place(prefix + "_qst_active")
            drain = net.place(prefix + "_drain_share")
            ports = [memory.b, memory.sw]
            deps, dram_deps = dependencies(commands, I, W, t.matrix_engines)
            kinds, latencies, services, port_ids = [], [], [], []
            batches, chunkses, scale_flags, kbs, ns, row_services, tails = [], [], [], [], [], [], []
            barriers, dispatch_delays, pipeline_delays = [], [], []
            xkinds, shares, drains = [], [], []
            last_register = nsig
            for idx, c in enumerate(commands):
                ins, w = c.instruction, c.instruction.w
                u = W.UNIT.get(ins.op, 5)
                kinds.append(u)
                barriers.append(last_register)
                dispatch_delays.append(2 if ins.op == I.LOOP else 1)
                pipeline_delays.append(5 if ins.op >= I.LD else 0)
                if ins.op == I.RLD:
                    last_register = idx
                latency, port, service = (0, None, 0) if u in (W.U_MXU, 5) else self._simple_timing(ins, memory, I)
                latencies.append(latency)
                port_ids.append(ports.index(port) if port is not None else -1)
                services.append(service)
                xk = 0
                if ins.op == I.ST:
                    xk = 1
                elif ins.op == I.QST:
                    xk = 2
                elif ins.op == I.VOP and ((w[5] >> 16) & 255) not in REDUCE:
                    xk = 3
                xkinds.append(xk)
                batch, chunks, scaled, kb, n, row_service, tail, share, drain_pass = 1, 1, 0, 1, 1, 1, 0, 0, 0
                if ins.op == I.MM:
                    n, kb, m = w[3] & 65535, (w[3] >> 16) & 255, (w[5] >> 16) & 255
                    wf, pair = (ins.flags >> I.WF_SHIFT) & 3, bool(ins.flags & I.F_PAIR)
                    quad = bool(getattr(I, "F_QUAD", 0) and ins.flags & I.F_QUAD)
                    pack = bool(getattr(cfg, "PACK", False)) and wf == I.W8 and not pair and not quad
                    chunks = kb if wf == I.W8 else ceil(kb / 2)
                    scaled = int(not bool(ins.flags & I.F_UNIT))
                    dw = min(self.E, cfg.LANES // gcd(w[5] & 65535, cfg.LANES))
                    drain_rows = ceil(m / dw)
                    G = ceil(m / cfg.MCOLS)
                    if pack:
                        # vla: a pass is a pair of weight rows: 2 KB chunks on port B and two
                        # scale rows, KB advances a group (a KB 1 pass every other cycle), two
                        # result rows drained
                        n = ceil(n / 2)
                        chunks, kb_a = 2 * kb, 2 * kb
                        steps = max(kb, 2) * G
                        drain_pass = 2 * drain_rows
                    else:
                        kb_a = kb
                        steps = (chunks if pair else kb) * G
                        drain_pass = drain_rows
                    if chunks > t.fifo_chunks:
                        raise ValueError("one matrix row must fit the prefetch FIFO")
                    batch = min(t.batch_rows, max(1, n // 64), max(1, t.fifo_chunks // (2 * chunks)))
                    row_service = max(steps, drain_pass)
                    tail = min(cfg.D // 8, t.chain_length) + cfg.MCOLS + 5 + 9 + (6 if ins.flags & I.F_ACC else 0) + drain_pass
                    # the drain's share of a pass: over the pass's cycles, the port's chunks
                    # when the stream binds (an MM with M <= MCOLS packed: 2 KB chunks a pass)
                    share = min(750, (1000 * drain_pass) // max(row_service, chunks)) if kn.warb == 2 else 0
                    kb = kb_a
                    self.matrix_batch_sizes.append(batch)
                batches.append(batch)
                chunkses.append(chunks)
                scale_flags.append(scaled)
                kbs.append(kb)
                ns.append(n)
                row_services.append(row_service)
                tails.append(tail)
                shares.append(share)
                drains.append(drain_pass)
            parameter_values = [latencies, services, port_ids, batches, chunkses, scale_flags, kbs, ns, row_services, tails]
            parameters = CommandTiming(*(self._table(prefix + "_" + name, values)
                for name, values in zip(CommandTiming.__dataclass_fields__, parameter_values)))
            metadata = self._table(prefix + "_metadata", [ncommands,
                max(1, t.fifo_chunks // max(b * c for b, c in zip(batches, chunkses)))])
            xkinds = self._table(prefix + "_xkinds", xkinds)
            shares = self._table(prefix + "_shares", shares)
            drains = self._table(prefix + "_drains", drains)

            def csr(rows_):
                offsets, flat = [0], []
                for row in rows_:
                    flat.extend(row)
                    offsets.append(len(flat))
                return offsets, flat or [nsig]

            starts = [dram_deps[i] if kinds[i] == W.U_MXU else deps[i] for i in range(ncommands)]
            offsets, indices = csr(starts)
            compute_offsets, compute_indices = csr(deps)
            begins = self._table(prefix + "_starts_begin", offsets[:-1])
            ends = self._table(prefix + "_starts_end", offsets[1:])
            indices = self._table(prefix + "_starts_indices", indices)
            compute_offsets = self._table(prefix + "_compute_offsets", compute_offsets)
            compute_indices = self._table(prefix + "_compute_indices", compute_indices)
            kinds = self._table(prefix + "_kinds", kinds)
            barriers = self._table(prefix + "_register_barriers", barriers)
            dispatch_delays = self._table(prefix + "_dispatch_delays", dispatch_delays)
            pipeline_delays = self._table(prefix + "_pipeline_delays", pipeline_delays)

            @net.func(name=prefix + "_select_ready", args=("i64",), results=("i64",))
            def select_ready(unit):
                slot_places = array(slots)
                kind, begin, end = array(kinds), array(begins), array(ends)
                dep_indices = array(indices)
                selected, oldest = subi(0, 1), addi(array(metadata)[0], 1)
                selected_enabled = const_i64(0) == 0
                for s in range(t.window):
                    token = peek(slot_places[s])
                    idx = token.get("id")
                    active = select(token.get("active") == 1, kind[idx] == unit, False, typ="i1")
                    enabled = active
                    for d in range(select(enabled, begin[idx], 0), select(enabled, end[idx], 0)):
                        enabled = select(enabled, is_done(dep_indices[d]), False, typ="i1")
                    candidate = select(unit == W.U_MXU, active, enabled, typ="i1")
                    eligible = select(candidate, idx < oldest, False, typ="i1")
                    selected = select(eligible, s, selected, typ="i64")
                    oldest = select(eligible, idx, oldest)
                    selected_enabled = select(eligible, enabled, selected_enabled, typ="i1")
                func_return(select(selected_enabled, selected, subi(0, 1), typ="i64"))

            @net.transition(prefix + "_initialize_window")
            def initialize_window():
                take(initialize)
                for s in range(t.window):
                    emit(window, create().set("slot", s))
                emit(drain, create().set("share", 0).set("cap", 0))

            @net.transition(prefix + "_dispatch")
            def dispatch():
                guard(count(cursor) > 0)
                guard(count(window) > 0)
                idx = peek(cursor).get("id")
                guard(idx < array(metadata)[0])
                bs, ps = array(barriers), array(pipeline_delays)
                barrier = bs[idx]
                guard(is_done(barrier))
                take(cursor)
                token = take(window).set("id", idx).set("active", 1)
                emit(pipeline, token, delay=sitofp(ps[idx]))
                emit(cursor, create().set("id", idx + 1), delay=sitofp(array(dispatch_delays)[idx]))

            @net.transition(prefix + "_arrive")
            def arrive():
                guard(count(pipeline) > 0)
                token = take(pipeline)
                place = array(slots)[token.get("slot")]
                take(place)
                emit(place, token)
                emit(array(pending)[array(kinds)[token.get("id")]], create())

            interface = CommandInterface(slots, pending, window, select_ready, is_done, publish,
                compute_offsets, compute_indices)
            units = [VlaUnit(net, prefix + f"_unit_{u}", u, interface, parameters, memory, xkinds, qst_active, drain)
                     for u in (W.U_DMA, W.U_Q, W.U_VPU, W.U_COLL, 5)]
            matrices = [VlaMatrix(net, prefix + f"_mxu_{engine}", cfg, t, interface, parameters, memory, W.U_MXU,
                shares, drains, drain, buffer_metadata=metadata if self.input_capacity is not None else None)
                for engine in range(t.matrix_engines)]
            self.components[sid] = {"memory": memory, "units": units, "matrix_engines": matrices}

            @net.transition(prefix + "_mark_done")
            def mark_done():
                guard(count(signals) > 0)
                idx = peek(signals).get("id")
                bank = array(banks)[divi(idx, 64)]
                guard(count(bank) > 0)
                take(signals)
                token = take(bank)
                emit(bank, token.set("bits", ori(token.get("bits"), shli(1, remi(idx, 64)))))
                if idx == array(metadata)[0] - 1:
                    emit(finished, create())

            self.slice_done.append(finished)

    return VlaCompactOpenTPU


NATIVE_RE = {"cycles": r"Simulation finished at t=([0-9.]+)", "wall_s": r"Wall window elapsed: ([0-9.]+)",
             "rollbacks": r"rollbacks: (\d+)", "completed": r"place program_completed: (\d+) token",
             "firings": r"body executions: (\d+)"}


def dialect_cycles(name: str, prog, img, cfg, kn: Knobs, E: int, a, upstream: bool = False) -> dict:
    """The phase through the LPNDialect model: the ISA simulator's command trace (this tree's),
    the model (upstream's CompactOpenTPU as it is, or VlaCompactOpenTPU), lpn-compile, the
    native run; {cycles, commands, wall...}."""
    import re
    import subprocess
    from examples.opentpu import workload as WL
    from examples.opentpu.compact import CompactOpenTPU
    from examples.opentpu.model import Timing
    t0 = time.time()
    work = WL.execute(cfg, [prog], [img], args=[0] * 8)
    fe = time.time() - t0
    timing = Timing(dram_latency=kn.l_ld, dram_bytes_per_cycle=cfg.D, window=kn.win,
                    fifo_chunks=2 * kn.fifo, chain_length=16, epilogue_lanes=E,
                    quant_buffer_chunks=1024, batch_rows=a.batch_rows)
    cls = CompactOpenTPU if upstream else dialect_classes(kn, E)
    t1 = time.time()
    model = cls(work, timing)
    work_dir = Path(a.work) / (name + ("_upstream" if upstream else ""))
    work_dir.mkdir(parents=True, exist_ok=True)
    mlir, exe = work_dir / "model.mlir", work_dir / "opentpu"
    mlir.write_text(model.build())
    build = time.time() - t1
    env = dict(os.environ, PATH=f"{a.llvm_bin}:{os.environ.get('PATH', '')}")
    if sys.platform == "darwin" and "SDKROOT" not in env:
        env["SDKROOT"] = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    t2 = time.time()
    comp = subprocess.run([a.lpn_compile, str(mlir), "-o", str(exe)], capture_output=True, text=True, env=env)
    (work_dir / "compile.log").write_text(comp.stdout + comp.stderr)
    if comp.returncode:
        raise RuntimeError(f"lpn-compile failed: {work_dir / 'compile.log'}")
    comp_s = time.time() - t2
    t3 = time.time()
    run = subprocess.run([str(exe), "--stats", "--wall-start=begin", "--wall-end=complete",
                          "--max-time=1000000000000"], capture_output=True, text=True)
    (work_dir / "native.log").write_text(run.stdout + run.stderr)
    out = {}
    for k, rx in NATIVE_RE.items():
        m = re.search(rx, run.stdout)
        if m:
            out[k] = float(m.group(1)) if k in ("cycles", "wall_s") else int(m.group(1))
    if run.returncode or out.get("completed") != 1 or out.get("rollbacks"):
        raise RuntimeError(f"native run failed or incomplete: {work_dir / 'native.log'}")
    out.update(cycles=int(math.ceil(out["cycles"])), commands=len(work.commands[0]), frontend_s=fe,
               model_s=build, compile_s=comp_s, native_s=time.time() - t3,
               net=model.statistics() if hasattr(model, "statistics") else None)
    return out


# ------------------------------------------------------------------------------ driving
def target_E(cfg) -> int:
    from opentpu import target as TG
    e = int(os.environ.get("OTPU_MXU_E", TG.current().uarch.get("MXU_E", 0)) or 0)
    e = e if e > 0 else max(1, cfg.MCOLS // 2)
    return min(e, cfg.MCOLS)


def tally(prog, cfg) -> dict:
    import pi05_cost as PCO
    t = PCO.tally(prog, cfg)
    return {u: int(t.get(u, 0)) for u in PCO.UNITS}


def phase_cycles(prog, cfg, kn: Knobs, E: int) -> tuple:
    dyn = expand(prog, cfg, kn, E)
    net = Net(dyn, kn)
    t0 = time.time()
    cyc = net.run()
    return cyc, len(dyn), time.time() - t0, net


def run_phase(ph: str, prog, img, cfg, sim_cfg, kn: Knobs, E: int, a, meas: int = 0) -> dict:
    """The phase through the backends asked for; prints a row each, returns {backend: cycles,
    ...} with the tally."""
    import numpy as np
    t = tally(prog, cfg)
    slow = max(t.values())
    out = {"tally": t}
    print(f"{ph:9s} tally " + " ".join(f"{u}={v}" for u, v in t.items())
          + (f"; measured {meas:,}" if meas else ""))

    def row(label, cyc, extra=""):
        err = f"; error {100 * (cyc - meas) / meas:+.1f}%" if meas else ""
        print(f"  {label:18s} {cyc:10,} cycles  ratio {cyc / slow:.2f}  MXU {100 * t['MXU'] / cyc:.1f}%"
              f"{err}  ({extra})")
    if a.backend in ("lpnlang", "both"):
        cyc, n, secs, net = phase_cycles(prog, cfg, kn, E)
        out["lpnlang"] = cyc
        row("lpnlang", cyc, f"{n} instructions, {secs:.0f} s")
        if a.diag:
            dg = net.by_class()
            print("    unit busy: " + " ".join(f"{u} {100 * v / cyc:.0f}%" for u, v in dg["unit_busy"].items())
                  + f"; VPU issue stretched {dg['vpu_stretch']:,} cycles by the drain")
            for k, v in sorted(dg["classes"].items(), key=lambda kv: -kv[1]["busy"])[:8]:
                print(f"      {k:14s} {v['count']:6d} busy {v['busy']:9d} wait {v['wait']:9d}")
    if a.backend in ("dialect", "both"):
        variants = [("dialect", False)] + ([("dialect-upstream", True)] if a.upstream else [])
        for label, up in variants:
            d = dialect_cycles(ph, prog, img, sim_cfg, kn, E, a, upstream=up)
            out[label] = d["cycles"]
            row(label, d["cycles"], f"{d['commands']} commands; frontend {d['frontend_s']:.0f} s, model "
                f"{d['model_s']:.0f} s, compile {d['compile_s']:.0f} s, native {d['native_s']:.1f} s, "
                f"{d.get('firings', 0):,} firings")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lpn", default=os.environ.get("LPN_SRC", str(SCRATCH.parent / "lpn" / "language" / "src")),
                    help="dslab-epfl/lpn's language/src (lpnlang; the lpnlang backend)")
    ap.add_argument("--backend", default="lpnlang", choices=("lpnlang", "dialect", "both"))
    ap.add_argument("--dialect", default=os.environ.get("HWLOOP_LPN", str(SCRATCH / "LPNDialect")),
                    help="MJChku/LPNDialect clone (the dialect backend)")
    ap.add_argument("--lpn-compile", default=str(SCRATCH / "lpn-build" / "tools" / "lpn-compile" / "lpn-compile"))
    ap.add_argument("--llvm-bin", default=str(SCRATCH / "llvm20-bin"), help="opt-20, llc-20, clang++-20")
    ap.add_argument("--work", default=os.environ.get("PI05_LPN_WORK", "/tmp/pi05_lpn"),
                    help="the dialect backend's MLIR, binaries and logs")
    ap.add_argument("--batch-rows", type=int, default=8, help="the dialect model's rows a prefetch event")
    ap.add_argument("--upstream", action="store_true",
                    help="the dialect backend also as the tier has it (no f2-vla items)")
    ap.add_argument("--phase", default="v_out,p_qkv,e_step,p_attend,v_qkv")
    ap.add_argument("--np", type=int, default=558)
    ap.add_argument("--np-out", type=int, default=128, help="p_out's Np in the cut runs")
    ap.add_argument("--rows", default="128,255,64")
    ap.add_argument("--full", action="store_true", help="the whole chunk (the real model)")
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--lst", type=int, default=Knobs.l_st)
    ap.add_argument("--lwq", type=int, default=Knobs.l_wq)
    ap.add_argument("--lld", type=int, default=Knobs.l_ld)
    ap.add_argument("--lmm", type=int, default=Knobs.l_mm)
    ap.add_argument("--lq", type=int, default=Knobs.l_q)
    ap.add_argument("--vpu-cl", type=int, default=0,
                    help="VPU_CL, the composites' lanes (default the target's uarch, else 16)")
    ap.add_argument("--q-row", type=int, default=Knobs.q_row,
                    help="the quantizer's cycles a row past the scale unit (0: the two-stage max of 5.4's RTL)")
    ap.add_argument("--qst-wc", type=float, default=Knobs.qst_wc)
    ap.add_argument("--warb", type=int, default=2)
    ap.add_argument("--no-lookahead", action="store_true")
    ap.add_argument("--diag", action="store_true", help="per class busy / waits, unit busy")
    ap.add_argument("--json")
    a = ap.parse_args()
    import numpy as np
    if a.backend in ("lpnlang", "both"):
        sys.path.insert(0, a.lpn)
        np.Inf = np.inf                   # lpnlang's simulator is NumPy 1.x's (np.Inf)
        import lpnlang  # noqa: F401
    if a.backend in ("dialect", "both"):
        sys.path.insert(1, a.dialect)
        import lpnlang_mlir  # noqa: F401
    from opentpu.isasim import board_config
    from opentpu.llm import pi05 as P
    from opentpu import target as TG
    vpu_cl = a.vpu_cl or int(TG.current().uarch.get("VPU_CL", Knobs.vpu_cl))
    kn = Knobs(l_st=a.lst, l_wq=a.lwq, l_ld=a.lld, l_mm=a.lmm, l_q=a.lq, q_row=a.q_row, qst_wc=a.qst_wc, warb=a.warb,
               vpu_cl=vpu_cl, dma_lookahead=not a.no_lookahead)
    cfg = board_config()
    E = target_E(cfg)
    R, RP, RE = (int(x) for x in a.rows.split(","))
    print(f"{cfg}; E {E}; row groups {R}/{RP}/{RE}; backend {a.backend}; {kn}")
    res = {"config": str(cfg), "E": E, "knobs": dataclasses.asdict(kn), "backend": a.backend, "phases": {}}
    spec = P.Spec.from_ckpt(a.ckpt)
    mhz = a.mhz * 1e6
    if not a.full:
        spec = dataclasses.replace(spec, v_layers=1, layers=1, e_layers=1, steps=1)
    image = P.Image(spec, cfg)
    # the ISA simulator's DRAM: the image's size (tools/pi05_rtl.py), the activations zeros
    sim_cfg = board_config(DRAM_BYTES=1 << (image.end - 1).bit_length())
    img = np.zeros(image.end, np.uint8) if a.backend != "lpnlang" else None
    progs = image.programs(a.np, R=R, RP=RP, RE=RE)
    if not a.full:
        progs_out = image.programs(a.np_out, R=R, RP=RP, RE=RE)
        key = "@16" if cfg.MCOLS == 16 else ""
        for ph in a.phase.split(","):
            prog, _ = (progs_out if ph == "p_out" else progs)[ph]
            meas = MEASURED.get(ph + key, (0, 0, 0))[2]
            r = run_phase(ph, prog, img, cfg, sim_cfg, kn, E, a, meas)
            r["measured"] = meas
            res["phases"][ph] = r
    else:
        runs = collections.Counter(r[0] for r in image.runs(spec.cameras) if not r[0].startswith("write"))
        backends = [b for b in ("lpnlang", "dialect", "dialect-upstream")
                    if (b == "lpnlang" and a.backend != "dialect") or (b == "dialect" and a.backend != "lpnlang")
                    or (b == "dialect-upstream" and a.upstream and a.backend != "lpnlang")]
        parts = {b: {"vision": 0, "prefix": 0, "expert": 0} for b in backends}
        tot = collections.Counter()
        for ph in sorted(runs):
            prog, _ = progs[ph]
            k = runs[ph]
            r = run_phase(f"{ph} x{k}", prog, img, cfg, sim_cfg, kn, E, a)
            part = {"v": "vision", "p": "prefix", "e": "expert"}[ph[0]]
            for b in backends:
                parts[b][part] += k * r[b]
            for u, v in r["tally"].items():
                tot[u] += k * v
            r["runs"] = k
            res["phases"][ph] = r
        res["chunk"] = {"tally_bound_s": max(tot.values()) / mhz, "tally_serial_s": sum(tot.values()) / mhz,
                        "tally": dict(tot)}
        print(f"the tally's bound {max(tot.values()) / mhz:.3f} s, serial {sum(tot.values()) / mhz:.3f} s")
        for b in backends:
            chunk = sum(parts[b].values())
            print(f"{b}: " + ", ".join(f"{p} {c / mhz:.3f} s" for p, c in parts[b].items())
                  + f"; chunk {chunk:,} cycles = {chunk / mhz:.3f} s, {mhz / chunk:.2f} Hz at {a.mhz:.0f} MHz"
                  f" (MXU {100 * tot['MXU'] / chunk:.1f}%)")
            res["chunk"][b] = {"cycles": chunk, "seconds": chunk / mhz, "hz": mhz / chunk, "parts_cycles": parts[b]}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
