"""Standalone C2 RTL harness: real Verilator cycles, synthetic operand services.

No production opcode, HBM controller, clock closure or whole-token timing is implied.
The canonical ISA supplies expected bits; it never supplies the RTL result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import tempfile

import numpy as np

from . import fp32 as F, isa as I, lens, rtlsim
from .isasim import Config, Slice
from .profile import UNITS


@dataclass
class Group:
    decode: bool
    acc: bool
    kb: int
    count: int
    tag: int
    weights: np.ndarray
    acts: np.ndarray
    dw: np.ndarray
    da: np.ndarray
    sw: np.ndarray
    sa: np.ndarray
    old: np.ndarray
    addr: np.ndarray

    @property
    def mask(self):
        return (1 << self.count) - 1

    def validate(self):
        if not 1 <= self.kb <= 255 or not 1 <= self.count <= 4 or not 0 <= self.tag < 128:
            raise ValueError("C2 group dimensions/tag out of bounds")
        for name, shape, dtype in (
            ("weights", (4,self.kb,128), np.int8), ("acts", (4,self.kb,128), np.int8),
            ("dw", (4,self.kb), np.uint8), ("da", (4,self.kb,4), np.uint8),
            ("sw", (4,), np.uint32), ("sa", (4,), np.uint32),
            ("old", (4,), np.uint32), ("addr", (4,), np.uint32)):
            value=getattr(self,name)
            if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype:
                raise ValueError(f"C2 invalid {name} shape/type")
        if np.any(self.addr>=65536) or len(set(self.addr[:self.count]))!=self.count:
            raise ValueError("C2 invalid or overlapping output addresses")


def make_group(kb=9, count=4, *, decode=True, acc=False, tag=0, seed=1,
               corners=False) -> Group:
    if not 1 <= kb <= 255 or not 1 <= count <= 4 or not 0 <= tag < 128:
        raise ValueError("C2 group bounds: KB 1..255, count 1..4, tag 0..127")
    rng = np.random.default_rng(seed)
    weights = rng.integers(-128, 128, (4, kb, 128), dtype=np.int16).astype(np.int8)
    acts = rng.integers(-128, 128, (4, kb, 128), dtype=np.int16).astype(np.int8)
    dw = rng.integers(0, 20, (4, kb), dtype=np.uint8)
    da = rng.integers(0, 20, (4, kb, 4), dtype=np.uint8)
    # Include negative shift, shift zero, and maximum positive shift.
    dw[:, 0] = [0, 1, 16, 17]
    da[:, 0] = [0, 16, 17, 0]
    sw = F.bits(np.ldexp(rng.standard_normal(4).astype(np.float32), -9))
    sa = F.bits(np.ldexp(rng.standard_normal(4).astype(np.float32), -5))
    old = F.bits(rng.standard_normal(4).astype(np.float32))
    if corners:
        patterns = np.uint32([0, 0x80000000, 1, 0x00800000, 0x00FFFFFF,
                              0x7F800000, 0x7F800001, 0x7F7FFFFF, 0xBF800000])
        sw = np.roll(patterns, seed)[:4].copy()
        sa = np.roll(patterns, seed + 3)[:4].copy()
        old = np.roll(patterns, seed + 6)[:4].copy()
        weights[0] = 0
    addr = (1024 + tag * 32 + np.arange(4) * (1 if decode else 7)).astype(np.uint32)
    return Group(decode, acc, kb, count, tag, weights, acts, dw, da, sw, sa, old, addr)


def reference(group: Group):
    """Canonical MM, with nonzero ACT block and original weight-scale k0."""
    g = group
    ab, k0 = 2, 3
    n, m = (g.count, 1) if g.decode else (1, g.count)
    rs = g.kb * 128
    srs = ((4 + k0 + g.kb + 3) // 4) * 4
    ssa = n * rs
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=ab + g.kb,
                 DRAM_BYTES=ssa + n * srs + 128, TMEM_WORDS=65536)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    for c in range(n):
        dram[c * rs:(c + 1) * rs] = g.weights[c].reshape(-1).view(np.uint8)
        dram[ssa + c * srs:ssa + c * srs + 4] = g.sw[c:c+1].view(np.uint8)
        dram[ssa + c * srs + 4 + k0:ssa + c * srs + 4 + k0 + g.kb] = g.dw[c]
    sl = Slice(cfg, 0, [], dram)
    for c in range(m):
        sl.act[c, ab * 128:(ab + g.kb) * 128] = g.acts[c].reshape(-1)
        sl.adoff[c, ab:ab + g.kb] = g.da[c]
        sl.ascale[c, ab:ab + g.kb] = F.from_bits(g.sa[c:c+1])[0]
    sl.tmem[g.addr] = g.old
    ins = I.mm(0, ssa, int(g.addr[0]), n, g.kb, rs, 1 if g.decode else 7,
               m, ab, srs, acc=g.acc, k0=k0)
    sl._mm(ins)
    integers = []
    for c in range(g.count):
        wc, ac = (c, 0) if g.decode else (0, c)
        dots = (g.weights[wc].astype(np.int64) * g.acts[ac].astype(np.int64)).reshape(g.kb, 4, 32).sum(2)
        shifts = 16 - g.dw[wc].astype(np.int64)[:, None] - g.da[ac].astype(np.int64)
        integers.append(int(np.where(shifts >= 0, dots << np.maximum(0, shifts), 0).sum()))
    return sl.tmem[g.addr[:g.count]].copy(), integers


def _write(groups, path):
    def word(f, value):
        f.write(struct.pack(">I", int(value)))
    with Path(path).open("wb") as f:
        word(f, len(groups))
        for g in groups:
            for value in (g.decode, g.acc, g.kb, g.mask, g.tag):
                word(f, value)
            for c in range(4):
                word(f, g.addr[c]); word(f, g.old[c])
            for c in range(4):
                word(f, g.sw[c]); word(f, g.sa[c])
                for k in range(g.kb):
                    word(f, g.dw[c, k])
                    word(f, int.from_bytes(g.da[c, k].tobytes(), "little"))
                    f.write(g.weights[c, k].tobytes()[::-1])
                    f.write(g.acts[c, k].tobytes()[::-1])


def build():
    root = rtlsim.ROOT
    sources = [root / "rtl" / s for s in (
        "vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv",
        "tech/generic/otpu_tech_dsp.sv", "mxu/otpu_mxu.sv", "mxu/otpu_c2_mxu.sv")]
    sources.append(root / "sim/verilator/tb_c2_mxu.sv")
    # This process only; avoid all-core builds during the C1/C2/C3 fan-out.
    original = rtlsim.VFLAGS
    rtlsim.VFLAGS = original.copy()
    rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = "4"
    try:
        return rtlsim.build("tb_c2_mxu", sources)
    finally:
        rtlsim.VFLAGS = original


FIELDS = ("cycle", "accept", "accepted_tag", "advance", "array_tag", "weight_accept_mask",
          "scale_accept_mask", "act_accept_mask", "commit", "commit_tag", "old_wait",
          "output_wait", "mode_wait", "credit_wait", "weight_wait_mask", "scale_wait_mask",
          "act_wait_mask", "live_groups", "result_groups", "weight_peak01", "weight_peak23")


def run(groups, *, out_dir=None, seed=1, wstall=0, sstall=0, astall=0,
        ostall=0, rstall=0, hold=0, old_hold=0, delay_pc0=0, reset_at=0, reset_skip=0):
    if not groups or len(groups) > 128 or len({g.tag for g in groups}) != len(groups):
        raise ValueError("C2 requires 1..128 uniquely tagged groups")
    for group in groups:
        group.validate()
    if reset_at and not 0 < reset_skip < len(groups):
        raise ValueError("C2 reset requires a discarded prelude and surviving groups")
    for p in (wstall, sstall, astall, ostall, rstall):
        if not 0 <= p < 100:
            raise ValueError("C2 stall percentages must be 0..99")
    exe = build()
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="c2-rtl-") as temp:
            return run(groups, out_dir=temp, seed=seed, wstall=wstall, sstall=sstall,
                       astall=astall, ostall=ostall, rstall=rstall, hold=hold, old_hold=old_hold,
                       delay_pc0=delay_pc0, reset_at=reset_at, reset_skip=reset_skip)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(groups, out_dir / "input.bin")
    params = dict(seed=seed, wstall=wstall, sstall=sstall, astall=astall, ostall=ostall,
                  rstall=rstall, hold=hold, old_hold=old_hold, delay_pc0=delay_pc0,
                  reset_at=reset_at, reset_skip=reset_skip)
    command = [str(exe), f"+input={out_dir/'input.bin'}", f"+events={out_dir/'events.txt'}",
               f"+results={out_dir/'results.txt'}", *[f"+{k}={v}" for k, v in params.items()]]
    result = rtlsim.run_sim(command, timeout=120)
    if result.returncode:
        raise RuntimeError("C2 RTL failed:\n" + result.stdout + result.stderr)
    events = [dict(zip(FIELDS, map(int, line.split()))) for line in (out_dir / "events.txt").read_text().splitlines()]
    rows = (out_dir / "results.txt").read_text().splitlines()
    surviving = groups[reset_skip:] if reset_at else groups
    if len(rows) != len(surviving):
        raise AssertionError("C2 dropped or duplicated group")
    outputs = []
    for g, line in zip(surviving, rows):
        words = line.split()
        cycle, tag, decode, acc, mask = map(int, words[:5])
        assert (tag, decode, acc, mask) == (g.tag, int(g.decode), int(g.acc), g.mask)
        expected, integers = reference(g)
        bits = [int(words[6 + c * 3], 16) for c in range(g.count)]
        got_int = [int(words[7 + c * 3], 16) for c in range(g.count)]
        got_int = [v - (1 << 48) if v & (1 << 47) else v for v in got_int]
        assert [int(words[5 + c * 3]) for c in range(g.count)] == g.addr[:g.count].tolist()
        assert got_int == integers, ("integer mismatch", tag, got_int, integers)
        assert bits == expected.tolist(), ("fp32 mismatch", tag, [hex(v) for v in bits], [hex(v) for v in expected])
        outputs.append(dict(tag=tag, cycle=cycle, mask=mask, bits=bits, integers=got_int))
    counts = {k: sum(e[k] for e in events) for k in ("advance", "old_wait", "output_wait", "mode_wait", "credit_wait")}
    for source in ("weight", "scale", "act"):
        counts[source + "_wait_column_cycles"] = [sum(bool(e[source + "_wait_mask"] & (1 << c)) for e in events) for c in range(4)]
        counts[source + "_transactions"] = sum(e[source + "_accept_mask"].bit_count() for e in events)
    counts.update(cycles=events[-1]["cycle"], groups=len(surviving), outputs=sum(g.count for g in surviving),
                  peak_live_groups=max(e["live_groups"] for e in events),
                  peak_result_groups=max(e["result_groups"] for e in events),
                  peak_weight_chunks_per_column=max(max(e["weight_peak01"], e["weight_peak23"]) for e in events))
    return dict(bit_exact=True, model="standalone-c2-verilator-v1", service=params,
                counts=counts, outputs=outputs, events=events)


def lens_data(groups, evidence, name):
    events, counts = evidence["events"], evidence["counts"]
    instructions, programs = [], []
    for idx, g in enumerate(groups):
        start = next(e["cycle"] for e in events if e["accept"] and e["accepted_tag"] == g.tag) - 1
        end = next(e["cycle"] for e in events if e["commit"] and e["commit_tag"] == g.tag)
        advances = [e["cycle"] for e in events if e["advance"] and e["array_tag"] == g.tag]
        op = "MM_ROWS" if g.decode else "MM"
        detail = f"tag={g.tag} KB={g.kb} columns={g.count} ACC={g.acc}; exact RTL group"
        programs.append([idx, op, detail, "standalone dual-mode MXU RTL", -1])
        instructions.append([0, idx, idx, 1, op, detail, start, start, -1,
                             advances[0]-1, end, -1, {}, g.kb * g.count, g.kb * (g.count if g.decode else 1), 0])
    cycles = counts["cycles"]
    macs = sum(g.kb * g.count * 128 for g in groups)
    bound = counts["advance"]
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=256)
    return {"kind": "rtl", "name": name + " [STANDALONE RTL / SYNTHETIC SERVICE]",
            "cycles": cycles, "clock_mhz": 200.0,
            "config": lens._config(cfg, {"C": 1, "C2_COLUMNS": 4, "FIFO_DEPTH": 256, "MXU_E": 4}),
            "roofline": {"bound": bound, "efficiency": bound / cycles,
                         "per_slice": [{"bound": bound, "portb": counts["weight_transactions"], "porta": 0}]},
            "macs": macs, "bytes": counts["weight_transactions"] * 128,
            "peak_macs": 512, "peak_bytes": 512, "units": UNITS,
            "slices": [{"ports": {}, "busy": {u: cycles if u == "MXU" else 0 for u in UNITS},
                        "lose": {}, "buckets": {}}],
            "instrs": instructions, "programs": [programs], "sources": [], "mxu_gaps": {},
            "notes": [{"level": "warn", "text": "Actual Verilator cycles of standalone C2 RTL. "
                       "Independent synthetic operand/scale service and ACC/commit grants; no HBM "
                       "controller, production ISA dispatch, bank arbitration, or Gemma loop. "
                       "200 MHz is an assumed display clock, not achieved timing or an F2 measurement."},
                      {"level": "info", "text": "Arithmetic-only floor counts group advances; "
                       "total RTL cycles include fill, array/epilogue latency, credit and mode drains. "
                       "Old reference-slot profiles are not comparable speed measurements."}],
            "c2_rtl": evidence}
