"""Utilization estimate of an F2 HBM target per SLR (docs/full_hbm.md "Utilization"): the block
RAMs and DSPs counted from the RTL's parameters (the chunk FIFO's RAMB36 in 512 x 72 or 1K x 36
mode by its depth, the TMEM's 1K-word pieces, the ACT RAM, IMEM, the trace, the engines' read
FIFOs; the MXU's D x MCOLS PEs, 9 column-end DSPs a column, the epilogue lanes), the LUTs and
flip-flops scaled from the figures Vivado gave for the v4 builds (docs/wide_port.md "Synthesis",
"Placement, round 2"; docs/f2_hbm.md "Synthesis"), and the SLR crossing bits from the chains'
widths (otpu_f2_hbm_eng: per channel 27 down + 517 up, per slice a 580-bit write bus; the
collective's links; the control path). Every number is `estimated` until a synthesis reports it.

    python tools/v5_utilization.py f2-hbm:v5 [f2-hbm:v5s3 f2-hbm:p4rl8a64 ...] [--sll 23000]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import target as TG  # noqa: E402

# ---- the measured anchors (LUT / FF), docs/wide_port.md "Placement, round 2" at fb42c14e
# (f2-hbm:p4rl8: PORTB 4, MCOLS 8, RL 8, NCH 8) and docs/f2_hbm.md "Synthesis" at 4bef3758
# (f2-hbm: PORTB 1, MCOLS 4, NCH 8 / 16)
MXU8 = (97_272, 70_237)            # u_mxu at MCOLS 8, RL 8, PORTB 4
VPU = (36_165, 20_000)             # u_vpu, 8 lanes (FF estimated: the slice's 131K less the rest)
TMEM = (11_913, 5_129)             # u_tmem, 64K words pumped
QUANT, SEQ, DMA = (9_383, 6_000), (9_044, 5_000), (9_251, 7_000)
SLICE_MISC = (2_600, 3_000)        # the slice's own glue (u_slice 174,618 less the units above)
ADAPTER8 = (46_717, 12_000)        # otpu_native_dram at NCH 8, PORTB 4, RD 64
ADAPTER8_P1 = (39_000, 10_000)     # at PORTB 1 (docs/f2_hbm.md: 39K at NCH 8)
CTRL_TRACE = (4_000, 4_000)        # otpu_ctrl, otpu_trace, the board's glue
ENGINE = (2_000, 2_900)            # otpu_hbm_ch, one channel: the 6a52448c synthesis measured
                                   # 1,375 LUT / 1,400 FF (the generator, the FIFOs' pointers, the
                                   # write FIFO's 576 x 16 distributed RAM, the front); the read
                                   # path since (otpu_hbm_rpath: the 512 x 32 crossing FIFO in
                                   # distributed RAM, 512 LUT6; its input, pair and output
                                   # registers ~1.5K FF) estimated
ENGINE_URAM = 4                    # otpu_hbm_rpath's read FIFO: 256 bits x 4K in URAM288
HOST_CHAIN = (6_500, 13_500)       # pcis gate + axi_register_slice + cl_axi_sc_1x1 + the 450 slice
MUX2 = (500, 1_200)
HUB = (3_000, 4_000)               # otpu_coll's hub at S > 1 with its links' stages
SLR_LUT = {0: 272_000, 1: 230_000, 2: 435_000}
SLR_RAMB, SLR_DSP, SLR_URAM = 672, 3_008, 320
# ---- the v5 slice as the first synthesis measured it (full-hbm 6a52448c on the m7a box: PORTB 4,
# MCOLS 4, RL 4, ALINE 64, FIFO_DEPTH 256, ACT_ROWS 8, TRACE_DEPTH 4096, NRD 128): used in place
# of the estimate for a slice of that shape
MEASURED_SLICE = {("PORTB", 4), ("MCOLS", 4), ("RL", 4), ("ALINE", 64), ("FIFO_DEPTH", 256),
                  ("ACT_ROWS", 8), ("TRACE_DEPTH", 4096), ("NRD", 128)}
MEASURED_SLICE_LUT, MEASURED_SLICE_RAMB, MEASURED_SLICE_DSP = 132_000, 249, 727


def measured_slice(t: TG.Target) -> bool:
    """True when the target's slice is the shape the 6a52448c synthesis measured."""
    a, u = t.arch, t.uarch
    rl = TG.rl_streams(TG.config(t))
    have = {("PORTB", int(a.get("PORTB", 1))), ("MCOLS", a["MCOLS"]), ("RL", rl),
            ("ALINE", int(a.get("ALINE", 32))), ("FIFO_DEPTH", u["FIFO_DEPTH"]),
            ("ACT_ROWS", a["ACT_ROWS"]), ("TRACE_DEPTH", u["TRACE_DEPTH"]), ("NRD", int(u.get("NRD", 64)))}
    return MEASURED_SLICE <= have


def mxu_lut_ff(mcols: int, rl: int, portb: int) -> tuple[int, int]:
    """The MXU scaled from the MCOLS 8 / RL 8 figure: the per-column parts (the PEs' selects,
    the column ends, the hops, the result FIFO, the drain) ~75% of it, the rest (the issuer, the
    scale path, S1's decode) ~25%, RL 4 against RL 8 -20K (docs/row_lanes.md "Hardware cost")."""
    lut, ff = MXU8
    col = 0.75 * lut * mcols / 8 + 0.25 * lut
    colf = 0.75 * ff * mcols / 8 + 0.25 * ff
    if rl and rl < 8:
        col -= 20_000 * (8 - rl) / 4
        colf -= 25_000 * (8 - rl) / 4
    if portb == 1:
        col -= 10_000
    return int(col), int(colf)


def ramb36(bits_wide: int, depth: int) -> int:
    """RAMB36 for a simple dual-port RAM of this width and depth (Vivado's packing: 512 x 72
    up to 512 deep, 1K x 36 up to 1K, 2K x 18, 4K x 9; several deep ones cascaded)."""
    if depth <= 512:
        return math.ceil(bits_wide / 72)
    if depth <= 1024:
        return math.ceil(bits_wide / 36)
    if depth <= 2048:
        return math.ceil(bits_wide / 18)
    return math.ceil(bits_wide / 9) * math.ceil(depth / 4096)


def slice_estimate(t: TG.Target) -> dict:
    a, u = t.arch, t.uarch
    D, M, PB = a["D"], a["MCOLS"], int(a.get("PORTB", 1))
    rl = TG.rl_streams(TG.config(t))
    fifo, tmem, act_b = u["FIFO_DEPTH"], a["TMEM_WORDS"], a["ACT_BLOCKS"]
    act_r = a["ACT_ROWS"] or M
    imem, trace = a["IMEM_WORDS"], u["TRACE_DEPTH"]
    nrd = int(u.get("NRD", 64))
    nch = t.platform["channels_per_slice"]
    # ---- block RAM
    bram = {"chunk FIFO": ramb36(PB * D * 8, fifo),
            "scale FIFOs": ramb36(96, fifo) + ramb36(64, fifo) + (ramb36(128, fifo) if PB > 1 else 0)
            + (ramb36(128, fifo) if rl >= 8 else 0) + (ramb36((7 if rl >= 8 else 3) * 32, fifo) if rl else 0),
            "TMEM (1K-word pieces)": tmem // 1024,
            "ACT RAM": ramb36(D * 8, act_b * act_r) if act_b * act_r <= 1024 else
            math.ceil(act_b * act_r * D * 8 / (36 * 1024)) + math.ceil(act_b * act_r * 64 / (36 * 1024)),
            "IMEM": math.ceil(imem * 32 / (36 * 1024)) if imem > 1024 else ramb36(32, imem),
            "trace": ramb36(64, trace) if trace <= 512 else math.ceil(trace * 64 / (36 * 1024)),
            "quantizer / VPU / DMA (residue of the measured 388)": 42}
    # ---- DSP
    dsp = {"MXU PEs (D x MCOLS)": D * M, "column ends (9 a column)": 9 * M,
           "epilogue lanes (E 4: 26)": 26, "VPU, quantizer": 153}
    # ---- LUT / FF
    mx = mxu_lut_ff(M, rl, PB)
    ad = ADAPTER8 if PB > 1 else ADAPTER8_P1
    ad = (ad[0] * (1 if nch == 8 else nch / 8 * 0.95), ad[1] * nch / 8)
    ad = (ad[0] + (512 * nch * (nrd - 64) // 64 if nrd > 64 else 0), ad[1] + 200 * nch)
    lut = {"u_mxu": mx[0], "u_vpu": VPU[0], "u_tmem": TMEM[0], "u_quant": QUANT[0], "u_seq": SEQ[0],
           "u_dma": DMA[0], "slice glue": SLICE_MISC[0], "otpu_native_dram": int(ad[0])}
    ff = {"u_mxu": mx[1], "u_vpu": VPU[1], "u_tmem": TMEM[1], "u_quant": QUANT[1], "u_seq": SEQ[1],
          "u_dma": DMA[1], "slice glue": SLICE_MISC[1], "otpu_native_dram": int(ad[1])}
    return {"bram": bram, "dsp": dsp, "lut": lut, "ff": ff}


def crossing_bits(S: int, nch: int) -> dict:
    """Bits per SLR boundary with the hub in SLR1, slices 0 / 1 / 2, 3 in SLR0 / 1 / 2."""
    ch_down, ch_up, wbus = 27, 517, 580
    per_slice = nch * (ch_down + ch_up) + wbus
    link = 1_300                # the collective's request, reads, writes, ack (both directions)
    ctl = 150 + 2
    place = {0: 0, 1: 1, 2: 2, 3: 2}
    b01 = sum(per_slice for s in range(S) if place[s] >= 1) + (link if S > 1 else 0) + ctl
    b12 = sum(per_slice for s in range(S) if place[s] >= 2) + \
        (link * sum(1 for s in range(S) if place[s] >= 2) if S > 1 else 0) + 100
    return {"SLR0-SLR1": b01, "SLR1-SLR2": b12, "per slice": per_slice}


def report(name: str, sll: int) -> None:
    t = TG.load(name)
    S, nch = t.arch["S"], t.platform["channels_per_slice"]
    engine = t.platform.get("mem_path") == "engine"
    e = slice_estimate(t)
    sl, sf = sum(e["lut"].values()), sum(e["ff"].values())
    sb, sd = sum(e["bram"].values()), sum(e["dsp"].values())
    measured = measured_slice(t)
    print(f"\n== {name}: S {S} x NCH {nch}, MCOLS {t.arch['MCOLS']}, PORTB {t.arch.get('PORTB', 1)}, "
          f"FIFO_DEPTH {t.uarch['FIFO_DEPTH']}, ACT_ROWS {t.arch['ACT_ROWS']}, NRD {t.uarch.get('NRD', 64)}, "
          f"mem_path {'engine' if engine else 'smartconnect'} "
          f"({'the slice as synthesized at 6a52448c, the rest estimated' if measured else 'every figure estimated'})")
    print(f"a slice (estimated): LUT {sl:,}  FF {sf:,}  RAMB36 {sb}  DSP {sd}")
    if measured:
        sl, sb, sd = MEASURED_SLICE_LUT, MEASURED_SLICE_RAMB, MEASURED_SLICE_DSP
        print(f"a slice (measured): LUT {sl:,}  RAMB36 {sb}  DSP {sd}")
    for k in ("lut", "bram", "dsp"):
        print(f"  {k}: " + ", ".join(f"{m} {v:,}" for m, v in e[k].items()))
    nport = S * nch
    eng_lut, eng_ff = (ENGINE[0] * nport, ENGINE[1] * nport) if engine else (6_100 * nport, 13_000 * nport)
    eng_bram = 0                                   # (was 8 x 32: the read FIFO in RAMB36, 512 x 72 mode)
    eng_uram = ENGINE_URAM * nport if engine else 0
    host = HOST_CHAIN
    mux = MUX2 if engine and nport == 32 else (0, 0)
    place = {0: [0], 1: [1], 2: [2, 3]}
    print("per SLR (slices 0 / 1 / 2,3 in SLR0 / 1 / 2; the engines, host chain in SLR0; the hub, ctrl, trace in SLR1):")
    for slr in range(3):
        n = sum(1 for s in range(S) if s in place[slr])
        lut = n * sl + (eng_lut + host[0] + mux[0] if slr == 0 else 0) + \
            ((HUB[0] if S > 1 else 0) + CTRL_TRACE[0] if slr == 1 else 0)
        ff = n * sf + (eng_ff + host[1] + mux[1] if slr == 0 else 0) + \
            ((HUB[1] if S > 1 else 0) + CTRL_TRACE[1] if slr == 1 else 0) + \
            (16_000 if slr in (0, 1) and S > 1 else 0)    # the chain stages (one per SLR)
        bram = n * sb + (eng_bram if slr == 0 else 0) + (8 if slr == 1 else 0)
        uram = eng_uram if slr == 0 else 0
        dsp = n * sd
        flag = " <-- over 80%" if lut / SLR_LUT[slr] > 0.8 or bram / SLR_RAMB > 0.8 or dsp / SLR_DSP > 0.8 \
            or uram / SLR_URAM > 0.8 else ""
        print(f"  SLR{slr}: {n} slice{'s' if n != 1 else ''}: LUT {lut:,} ({100 * lut / SLR_LUT[slr]:.0f}% of {SLR_LUT[slr] // 1000}K)  "
              f"FF {ff:,}  RAMB36 {bram} ({100 * bram / SLR_RAMB:.0f}%)  DSP {dsp} ({100 * dsp / SLR_DSP:.0f}%)  "
              f"URAM {uram} ({100 * uram / SLR_URAM:.0f}%){flag}")
    x = crossing_bits(S, nch)
    print(f"SLR crossings (bits; {sll:,} SLLs a boundary assumed): SLR0-SLR1 {x['SLR0-SLR1']:,} "
          f"({100 * x['SLR0-SLR1'] / sll:.0f}%), SLR1-SLR2 {x['SLR1-SLR2']:,} ({100 * x['SLR1-SLR2'] / sll:.0f}%); "
          f"a slice's memory {x['per slice']:,} (without the shared write bus {nch * (27 + 517 + 576):,})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["f2-hbm:v5", "f2-hbm:v5s3", "f2-hbm:p4rl8a64"])
    ap.add_argument("--sll", type=int, default=23_000, help="SLL wires a boundary (the brief's figure; verify on a placed checkpoint)")
    a = ap.parse_args()
    for n in a.targets:
        report(n, a.sll)


if __name__ == "__main__":
    main()
