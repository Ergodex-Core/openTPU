"""The wide port, the row lanes and the channel memory in LPNDialect's model of the slice
(tools/hwloop/lpn/README.md section 5, docs/wide_port.md, docs/row_lanes.md).

Runs under the venv python with LPNDialect on PYTHONPATH and the **export** tree as the ISA
(its simulator knows PORTB, int8 PAIR / QUAD, MM RL, QACT DUP8, MCOLS 8; upstream's pin
f237dab2 refuses int8 PAIR). Three things over upstream's `examples/opentpu/compact.py`:

1. **Port B as a credit of PORTB chunks a cycle** (`WideTiming.port_chunks`). An MM row is
   `requests` port-B requests (an advance's chunks each: int8 PAIR 2, QUAD 4; 4-bit QUAD 2; the
   last advance the chunks left) and `steps` MXU advances (ceil(kb / reuse)); the port credit is
   held one cycle a request, plus one cycle for every **two-part request** (a wide request whose
   chunks need one channel pair twice: `pair_of` below, otpu_native_dram's CHASH fold, counted
   exactly from the MM's addresses: `plan_mm`), so the roofline moves to the MXU where the port
   is wider than the reuse (docs/wide_port.md's 93.5% at PORTB 2) and the two-part reads cost
   what the receipts say they cost (7.0% / 56.7% of the wide reads at PORTB 2 / 4). The chunk
   FIFO's credit is in entries (`fifo_chunks`: the target's FIFO_DEPTH; an entry holds an
   advance's chunks), not chunks.
2. **Row lanes** (MM RL, `rs` bits 3:2): a group of L rows is one row of the model: ceil(kb /
   blocks-a-chunk) advances of L aligned chunks (never two-part: checked by the same count),
   one port-A line an advance plus the group's Sw line where the header and the advance's
   scale data do not share a 32-byte line (int4 L 4: H = A = 32), an issue cycle each.
3. **Channels** (`nch`, `channel_depth`, `adapter_latency`): NCH credit places, one a channel,
   each held for the beats the batch puts on it at the channel's sustained rate (a chunk is two
   64-byte beats, one on each channel of its pair; port A's lines the distinct beats they
   touch, as the adapter's runs and beat reuse fetch them), the per-command per-channel means
   from the addresses (`plan_mm` counts every beat); the receipts' per-channel read counts are
   the check (`memory_report`). The rate is one beat a cycle, or RD / round trip where the
   in-flight room (RD 64 B-read beats a channel at NCH > 2) is shorter than a read's round
   trip (`dram_latency` + the path): that is what binds at PORTB 4 with four lanes, where
   every channel must carry a beat a cycle. The return pipeline is the round trip on the
   ready FIFO; the FIFO's entries hide it.

Not modelled, by choice (each named in the receipt): the DMA's and QST's beats on the channels
(< 1% of a decode token's traffic), the adapter's A runs / bursts (the bridge's 450 MHz side),
QACT DUP8's extra copies, a wide 4-bit QUAD advance whose scale group crosses a 32-byte line
(charged one issue cycle and one line, as the RTL's first-line-alone request).

At PORTB 1 on the pin's configuration (fifo_chunks 128, nch 1) the net is upstream's: the same
cycle count on the Gemma bundle (the regression `lpn_run.py` keeps).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd

import numpy as np

from examples.opentpu.model import Timing, DRAM, SystolicArray
from examples.opentpu.compact import CompactOpenTPU, dependencies
from examples.opentpu.components import CommandInterface, ExecutionUnit

SCALE = 256          # fixed point of the per-row port / channel costs (a row's two-part share)


@dataclass(frozen=True)
class WideTiming(Timing):
  port_chunks: int = 1          # PORTB: D-byte chunks a cycle port B and the MXU stream take
  nch: int = 1                  # otpu_native_dram channels (1: the behavioral one-port DRAM)
  chash: bool = True            # CHASH: the pair fold (targets/f2-hbm.toml [uarch])
  beat_bytes: int = 64          # a channel beat
  channel_depth: int = 64       # RD: B read beats in flight a channel (otpu_board NRD: 64 at NCH > 2)
  adapter_latency: int = 4      # cycles over dram_latency on a read's round trip (the pin's behavioral
                                # DRAM: 4; the native path: the slice's port registers, BREG, the queue,
                                # the command, the memory model's in-order return jitter, the response
                                # FIFO and the merge: lpn_run passes 14)

  @property
  def round_trip(self) -> int:
    return self.dram_latency + self.adapter_latency

  @property
  def channel_rate(self) -> float:
    """Beats a cycle one channel sustains: one a cycle, or RD beats a round trip when the
    in-flight room is shorter than the round trip (RD 64 at --lat 60 plus the path: the limit
    that binds at PORTB 4, where every channel must stream a beat a cycle)."""
    return min(1.0, self.channel_depth / self.round_trip)

  def __post_init__(self):
    super().__post_init__()
    if self.port_chunks not in (1, 2, 4, 8):
      raise ValueError("port_chunks must be 1, 2, 4 or 8")
    if self.nch < 1 or self.nch & (self.nch - 1):
      raise ValueError("nch must be a power of two")
    if self.nch > 1 and self.nch < 2 * self.port_chunks:
      raise ValueError("otpu_native_dram needs NCH >= 2 PORTB")


# ----------------------------------------------------------------------------- the channel hash

def pair_of(m, nch: int, chash: bool):
  """The channel pair of chunk m (word address bits 31:5 = byte address / 128):
  otpu_native_dram.pair_of: m mod NP, with CHASH every higher log2(NP)-bit group XOR-folded in."""
  m = np.asarray(m, np.int64)
  pib = (nch.bit_length() - 1) - 1          # log2(NCH / 2)
  if pib <= 0:
    return np.zeros_like(m)
  mask = (1 << pib) - 1
  p = m & mask
  if chash:
    for k in range(pib, 27, pib):
      p ^= (m >> k) & mask
  return p


def chan_of(byte_addr, nch: int, chash: bool):
  """The channel of a 64-byte beat: its chunk's pair, then the half (address bit 6, with CHASH
  XOR the chunk index's parity)."""
  a = np.asarray(byte_addr, np.int64)
  m = a >> 7
  h = (a >> 6) & 1
  if chash:
    par = m.copy()
    par ^= par >> 16; par ^= par >> 8; par ^= par >> 4; par ^= par >> 2; par ^= par >> 1
    h ^= par & 1
  if nch <= 2:
    return h
  return 2 * pair_of(m, nch, chash) + h


# ----------------------------------------------------------------------------- an MM's plan

@dataclass
class MMPlan:
  n: int                 # rows of the model (RL: lane groups)
  rows_isa: int          # the instruction's N
  lanes: int             # L (0: no row lanes)
  kb: int
  bpc: int               # blocks a chunk (int8 1, 4-bit 2, t2 4)
  reuse: int             # 1, PAIR 2, QUAD 4
  chunks: int            # chunks a row (RL: a group's L x advances)
  requests: int          # port-B requests a row (advances)
  steps: int             # MXU advances a row
  entries: int           # chunk-FIFO entries a row
  lines: int             # port-A lines a row (0 when UNIT)
  issue_extra: int       # issue cycles without a chunk a row (a lone Sw / first-line request)
  wide_reads: int        # requests of more than one chunk, the command
  two_part: int          # of them, in two parts (a pair twice)
  chan_beats: list       # beats a channel, the command (port B chunks and port-A lines)
  scaled: int

  @property
  def port_x(self) -> int:
    """Port-B cycles a row, x SCALE: the requests, the issue cycles without a chunk, and the
    command's two-part share."""
    return ceil(SCALE * (self.requests + self.issue_extra + self.two_part / max(1, self.n)))

  def chan_x(self, c: int, rate: float = 1.0) -> int:
    """Channel c's service a row, x SCALE: its beats at the channel's sustained rate."""
    return ceil(SCALE * self.chan_beats[c] / rate / max(1, self.n)) if self.chan_beats else 0


def plan_mm(ins, cfg, t: WideTiming, I) -> MMPlan:
  w, fl = ins.w, ins.flags
  D = cfg.D
  n, kb, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[3] >> 24) & 0xFF
  rs = w[4] & ~getattr(I, "RS_MASK", 0x3)
  wf = (fl >> I.WF_SHIFT) & 3
  t2 = wf == I.W4I and bool(w[4] & getattr(I, "RS_T2", 0))
  bpc = 1 if wf == I.W8 else 4 if t2 else 2
  pair = bool(fl & I.F_PAIR)
  quad = bool(w[4] & getattr(I, "RS_QUAD", 0))
  L = I.mm_lanes(ins) if hasattr(I, "mm_lanes") else 0
  reuse = 4 if quad else 2 if pair else 1
  scaled = int(not bool(fl & I.F_UNIT))
  rsc = bool(fl & getattr(I, "F_RSC", 0))
  bs = 1 if wf == I.W8 else 8 if wf == I.WNV else 4         # a block's scale bytes
  sa, ssa, srs = w[0], w[1], w[6]
  nch, chash = t.nch, t.chash
  if L:
    groups = ceil(n / L)
    adv = ceil(kb / bpc)
    c_req = L
    requests = steps = adv
    A = L * bpc * bs
    H = max(4 * L, A)
    sw_extra = 1 if (scaled and not rsc and H + A > 32) else 0
    lines = (adv + sw_extra) if scaled else 0
    model_rows, chunks_row = groups, L * adv
    row_stride, c_step = rs, L * D
    issue_extra = sw_extra
    cross = 0
  else:
    chunks_row = ceil(kb / bpc)
    c_req = max(reuse // bpc, 1)
    requests = ceil(chunks_row / c_req)
    steps = ceil(kb / reuse)
    model_rows = n
    row_stride, c_step = rs, D
    # a 4-bit / NVFP4 QUAD advance's scale group crossing a 32-byte line: the first line alone
    cross = 0
    if scaled and not rsc and reuse > 1 and wf != I.W8:
      base = 24 if wf == I.WNV else 12
      Ab = reuse * bs
      for j in range(steps):
        so = base + (k0 + j * reuse) * bs
        cross += int(so // 32 != (so + Ab - 1) // 32)
    issue_extra = cross
    lines = (requests + cross) if scaled else 0
  # ---- the addresses: every chunk of every row (vectorized by row), the requests' pairs
  rows = np.arange(model_rows, dtype=np.int64)
  starts = sa + rows * row_stride
  wide_reads = two_part = 0
  beats = np.zeros(max(nch, 1), np.int64)
  if nch > 1:
    nreq = requests
    for j in range(nreq):
      q0 = j * c_req
      k = min(c_req, chunks_row - q0)
      addrs = starts[:, None] + (q0 + np.arange(k)) * D          # [rows, k]
      ms = addrs >> 7
      if k > 1:
        wide_reads += model_rows
        p = pair_of(ms, nch, chash)
        p.sort(axis=1)
        two_part += int(np.count_nonzero(np.any(p[:, 1:] == p[:, :-1], axis=1)))
      pr = pair_of(ms, nch, chash).ravel()
      cnt = np.bincount(pr, minlength=nch // 2)
      beats[0::2] += cnt
      beats[1::2] += cnt
    if lines and scaled:
      # the lines: a row's (group's) scale data at ssa + i srs, one line an advance (RL: the
      # header line first where it is its own request), each one beat on its channel
      sbase = ssa + rows * srs
      if L:
        A = L * bpc * bs
        H = max(4 * L, A)
        offs = [H + q * A for q in range(adv)] + ([0] if sw_extra else [])
      else:
        base = 4 if wf == I.W8 else 24 if wf == I.WNV else 12
        offs = [base + (k0 + j * reuse) * bs for j in range(requests)]
        if cross:
          offs += [o + 32 for o in offs if o // 32 != (o + reuse * bs - 1) // 32]
      # the adapter fetches port A in runs of APF channel-consecutive beats and reuses the last
      # beat: a sequential scale stream costs each 64-byte beat once, not one a line
      bts = np.unique((sbase[:, None] + np.asarray(offs, np.int64)[None, :]) >> 6)
      beats += np.bincount(chan_of(bts << 6, nch, chash), minlength=nch)
  else:
    for j in range(requests):
      k = min(c_req, chunks_row - j * c_req)
      if k > 1:
        wide_reads += model_rows
  return MMPlan(n=model_rows, rows_isa=n, lanes=L, kb=kb, bpc=bpc, reuse=reuse, chunks=chunks_row,
                requests=requests, steps=steps, entries=requests, lines=lines, issue_extra=issue_extra,
                wide_reads=wide_reads, two_part=two_part, chan_beats=beats.tolist() if nch > 1 else [],
                scaled=scaled)


def memory_report(workload, t: WideTiming, I) -> dict:
  """The static memory tally of a workload under the timing: the wide reads and the two-part
  ones, the per-channel beats of the MMs' port B and port A (the receipts' check), the chunks."""
  tot = {"port_chunks": t.port_chunks, "nch": t.nch, "chash": t.chash, "mm": 0, "mm_rl": 0, "chunks": 0,
         "requests": 0, "steps": 0, "lines": 0, "issue_extra": 0, "wide_reads": 0, "two_part": 0,
         "chan_beats": [0] * t.nch if t.nch > 1 else [], "by_lanes": {}}
  for commands in workload.commands:
    for c in commands:
      ins = c.instruction
      if ins.op != I.MM:
        continue
      p = plan_mm(ins, workload.config, t, I)
      tot["mm"] += 1
      tot["mm_rl"] += int(p.lanes > 0)
      tot["chunks"] += p.n * p.chunks
      tot["requests"] += p.n * p.requests
      tot["steps"] += p.n * p.steps
      tot["lines"] += p.n * p.lines
      tot["issue_extra"] += p.n * p.issue_extra
      tot["wide_reads"] += p.wide_reads
      tot["two_part"] += p.two_part
      if p.chan_beats:
        tot["chan_beats"] = [a + b for a, b in zip(tot["chan_beats"], p.chan_beats)]
      key = f"L{p.lanes}" if p.lanes else ("QUAD" if p.reuse == 4 else "PAIR" if p.reuse == 2 else "plain")
      tot["by_lanes"][key] = tot["by_lanes"].get(key, 0) + 1
  tot["port_b_cycles"] = tot["requests"] + tot["issue_extra"] + tot["two_part"]
  tot["two_part_share"] = tot["two_part"] / tot["wide_reads"] if tot["wide_reads"] else None
  tot["channel_rate"] = t.channel_rate
  tot["round_trip"] = t.round_trip
  tot["roofline"] = {"port_b": tot["port_b_cycles"], "mxu": tot["steps"], "port_a": tot["lines"],
                     "channel": ceil(max(tot["chan_beats"]) / t.channel_rate) if tot["chan_beats"] else None}
  return tot


# ----------------------------------------------------------------------------- the engine

class WideMatrixEngine:
  """upstream's components.MatrixEngine with the port-B credit in requests (plus the two-part
  share), the port-A credit in lines, the NCH channel credits, the FIFO credit in entries."""
  def __init__(self, net, prefix, cfg, t, interface, parameters, memory, kind, channels):
    self.resources = resources = SystolicArray(net, prefix, t,
        max(b * c for b, c in zip(parameters["batches"], parameters["entries"])))
    self.prefetch_credit = prefetch_credit = net.place(prefix + "_prefetch_credit", initial_tokens=1)
    self.request = request = net.place(prefix + "_request")
    self.ready = ready = net.place(prefix + "_ready")
    slots, pending, window = interface.slots, interface.pending[kind], interface.window
    select_ready, is_done, publish = interface.select_ready, interface.is_done, interface.publish
    offsets, indices = interface.compute_offsets, interface.compute_indices
    rows, batches = parameters["rows"], parameters["batches"]
    port_x, line_x, scaled = parameters["port_x"], parameters["lines"], parameters["scaled"]
    row_services, tails = parameters["row_services"], parameters["tails"]
    chan_x = parameters["chan_x"]          # [NCH] tables (beats a row x SCALE)
    chans = list(range(len(channels)))     # a list, not range(): unrolled at build time
    ready_delay = float(t.dram_latency + t.adapter_latency)

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
      emit(request, token.set("row", 0).set("carry", 0))

    @net.transition(prefix + "_prefetch")
    def prefetch():
      guard(count(request) > 0)
      guard(count(resources.buffers) > 0)
      guard(count(memory.b) > 0)
      idx = peek(request).get("id")
      needs_scale = array(scaled)[idx]
      guard(count(memory.a) >= needs_scale)
      for c in chans:
        guard(count(channels[c]) > 0)
      token = take(request)
      row = token.get("row")
      n, batch = array(rows)[idx], array(batches)[idx]
      width = select(row + batch < n, batch, n - row)
      take(resources.buffers)
      bus = take(memory.b)
      # the port's cycles for the batch, the fractional part (the two-part share) carried to
      # the next batch so a command's total is exact, not rounded up a cycle a batch
      acc = muli(width, array(port_x)[idx]) + token.get("carry")
      service = sitofp(divi(acc, SCALE))
      carry = remi(acc, SCALE)
      emit(memory.b, bus, delay=service)
      if needs_scale > 0:
        scale_port = take(memory.a)
        emit(memory.a, scale_port, delay=sitofp(muli(width, array(line_x)[idx])))
      for c in chans:
        ch = take(channels[c])
        emit(channels[c], ch, delay=sitofp(muli(width, array(chan_x[c])[idx])) / float(SCALE))
      emit(ready, token.set("width", width), delay=service + ready_delay)
      if row + batch < n:
        emit(request, token.set("row", row + batch).set("carry", carry), delay=service)
      else:
        emit(prefetch_credit, create(), delay=service)

    @net.transition(prefix + "_issue_rows")
    def compute():
      guard(count(ready) > 0)
      guard(count(resources.compute) > 0)
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
      if token.get("row") + array(batches)[idx] >= array(rows)[idx]:
        latency = service + sitofp(array(tails)[idx])
        publish(idx, latency)
        emit(resources.commands, create(), delay=latency)
        emit(window, token, delay=latency)


class WideCompactOpenTPU(CompactOpenTPU):
  """upstream's CompactOpenTPU with WideMatrixEngine and the per-command plans of plan_mm; the
  command window, the units, the dependency builder and the completion banks are upstream's
  (the _slice body is theirs, the matrix tables ours)."""
  supports_multiple_matrix_engines = False

  def __init__(self, workload, timing: WideTiming):
    self.plans = {}
    if timing.dram_bytes_per_cycle != workload.config.D:
      raise ValueError("the wide model's port credit is in chunks: dram_bytes_per_cycle must be D")
    super().__init__(workload, timing)

  def _slice(self, sid, commands, I, W):
    cfg, t, net = self.workload.config, self.timing, self.net
    prefix = f"s{sid}"
    ncommands = len(commands)
    if not ncommands:
      raise ValueError("empty command trace")
    memory = DRAM(net, prefix + "_dram", t)
    channels = [net.place(prefix + f"_dram_ch{c}", initial_tokens=1) for c in range(t.nch)] if t.nch > 1 else []
    nsig = ncommands
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
    ports = [memory.b, memory.sw]
    deps, dram_deps = dependencies(commands, I, W, 1)
    kinds, latencies, services, port_ids = [], [], [], []
    batches, entries, scale_flags, kbs, ns, row_services, tails = [], [], [], [], [], [], []
    port_xs, lines, chan_xs = [], [], [[] for _ in range(t.nch)]
    barriers, dispatch_delays, pipeline_delays = [], [], []
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
      batch, entry, scaled, kb, n, row_service, tail = 1, 1, 0, 1, 1, 1, 0
      port_x, line = 0, 0
      chan_x = [0] * t.nch
      if ins.op == I.MM:
        p = plan_mm(ins, cfg, t, I)
        self.plans[(sid, idx)] = p
        m = (w[5] >> 16) & 255
        if p.entries > t.fifo_chunks:
          raise ValueError("one matrix row must fit the prefetch FIFO")
        n, kb, scaled, entry = p.n, p.kb, p.scaled, p.entries
        batch = min(t.batch_rows, max(1, n // 64), max(1, t.fifo_chunks // (2 * entry)))
        if p.lanes:
          drain = ceil(p.lanes / min(t.epilogue_lanes, p.lanes))
        else:
          drain = ceil(m / min(t.epilogue_lanes, cfg.LANES // gcd(w[5] & 65535, cfg.LANES)))
        row_service = max(p.steps * ceil(m / cfg.MCOLS), drain)
        tail = min(cfg.D // 8, t.chain_length) + cfg.MCOLS + 5 + 9 + (6 if ins.flags & I.F_ACC else 0) + drain
        port_x, line = p.port_x, p.lines
        chan_x = [p.chan_x(ch, t.channel_rate) for ch in range(t.nch)]
        self.matrix_batch_sizes.append(batch)
      batches.append(batch)
      entries.append(entry)
      scale_flags.append(scaled)
      kbs.append(kb)
      ns.append(n)
      row_services.append(row_service)
      tails.append(tail)
      port_xs.append(port_x)
      lines.append(line)
      for ch in range(t.nch):
        chan_xs[ch].append(chan_x[ch])
    unit_parameters = _UnitTiming(latencies, services, port_ids)
    parameters = {"rows": ns, "batches": batches, "entries": entries, "port_x": port_xs, "lines": lines,
                  "scaled": scale_flags, "row_services": row_services, "tails": tails, "chan_x": chan_xs}
    metadata = [ncommands, max(1, t.fifo_chunks // max(b * c for b, c in zip(batches, entries)))]

    def csr(rows):
      offsets, flat = [0], []
      for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
      return offsets, flat or [nsig]

    starts = [dram_deps[i] if kinds[i] == W.U_MXU else deps[i] for i in range(ncommands)]
    offsets, indices = csr(starts)
    compute_offsets, compute_indices = csr(deps)
    begins, ends = offsets[:-1], offsets[1:]

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
    units = [ExecutionUnit(net, prefix + f"_unit_{u}", u, interface, unit_parameters, memory)
             for u in (W.U_DMA, W.U_Q, W.U_VPU, W.U_COLL, 5)]
    matrices = [WideMatrixEngine(net, prefix + "_mxu_0", cfg, t, interface, parameters, memory, W.U_MXU, channels)]
    self.components[sid] = {"memory": memory, "units": units, "matrix_engines": matrices, "channels": channels}

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


@dataclass
class _UnitTiming:
  """The three tables ExecutionUnit reads (upstream's CommandTiming has ten; the units use these)."""
  latencies: list
  services: list
  port_ids: list


# ----------------------------------------------------------------------------- the export frontend

def load_export(directory, export_root):
  """upstream's workload.load on the export's own ISA (its Config has PORTB / ROWLANES / MCOLS;
  input.json's `config_export`), without the pin's revision check."""
  import hashlib, json, sys
  from pathlib import Path
  root = Path(export_root).resolve()
  sys.path.insert(0, str(root))
  import opentpu
  if not Path(opentpu.__file__).resolve().is_relative_to(root):
    raise ValueError("another opentpu package was already imported")
  from opentpu import Config, isa as I
  from examples.opentpu.workload import execute
  directory = Path(directory)
  metadata = json.loads((directory / "input.json").read_text())
  cfg = Config(**metadata["config_export"])
  programs, images, hashes = [], [], {}
  for s in range(cfg.S):
    program, memory = directory / f"prog_{s}.hex", directory / f"dram_{s}.bin"
    words = [int(x, 16) for x in program.read_text().split()]
    count = metadata.get("instruction_counts", [])
    length = count[s] * 8 if count else ((len(words) + 7) // 8) * 8
    words.extend([0] * (length - len(words)))
    programs.append([I.Instr.decode(words[i:i + 8]) for i in range(0, len(words), 8)])
    images.append(np.fromfile(memory, dtype=">u4").astype("<u4").view(np.uint8))
    for path in (program, memory):
      hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
  return execute(cfg, programs, images, metadata.get("args", []), hashes)
