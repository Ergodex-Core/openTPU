# openTPU v4: an F2 HBM target for fast Gemma 3 1B decode

Design doc and plan. Status (2026-10-18): **approved to try P1, P2 and P4 in parallel**, with
Milestone 0 (the row-gating loss split in Lens) first and correctness checks (bit-exact
against the ISA simulator / per-position programs, Hugging Face greedy agreement) gating every
milestone before any F2 synthesis. P3 stays documented as an alternative. Branch
`tpubench`.

## Goal

Run google/gemma-3-1b-it in int8 (W8A8, 1.03 GiB streamed per token) at many times the
card's decode rate, on an AWS EC2 F2 instance (AMD Virtex UltraScale+ VU47P: HBM2e 16 GB in
32 pseudo-channels, about 460 GB/s; 9,024 DSP48E2, 2,016 RAMB36, 960 URAM288). The current
design has been synthesized on the F2 FPGA at 200 MHz, so every candidate is costed at
200 MHz. The FPGA run happens on EC2 F2 once a candidate is approved and built.

## What the Lens trace of today's design says

One decode token of Gemma 3 1B on the RTL (the card's configuration: one slice, D 128,
MCOLS 4, 8 VPU lanes, one 128-byte chunk a cycle; `build/lens/gemma3-decode-rtl.otpuprof`,
[gemma3.md](../gemma3.md)):

| | Cycles | At 133.33 MHz | At 200 MHz (F2 clock) |
|:--|--:|--:|--:|
| DRAM-bound minimum (1.03 GB at 128 B/cycle) | 7.85 M | 58.9 ms | 39.3 ms |
| Measured, RTL | 10.28 M | 77.1 ms, 13.0 tok/s | 51.4 ms, 19.5 tok/s |
| ISA model (units serialized, no RTL losses) | 8.28 M | 62.1 ms | 41.4 ms |

Where port B's 23.6% idle goes: MXU not consuming (row gating: a row's first pop waits for all
its chunks, scales and a result-FIFO credit) 18.2%, MXU row turnarounds 3.9%, chunk FIFO
starved 0.6%; the MXU drain lost 2.1% to TMEM arbitration. The VPU is in flight 4.5% of the
run, the quantizer 1.2%. The layers take 7.2 M cycles, the LM head 3.1 M.

Two conclusions drive v4:

1. **At batch 1 the port is the limit, not the memory.** The MXU consumes one 128-byte chunk
   a cycle, so a decode token does at most 128 int8 MACs a cycle (99 measured) of the 512
   the array has. More HBM bandwidth alone buys nothing: the port and the MXU must widen
   together, by the same factor C (`docs/scoreboard.md`: "port 256 B" alone 1.08x, "port 256 B
   + MXU x2" 1.88x).
2. **Every fixed cost grows C times in share.** At C = 8 the stream takes 5 ms, and the 5,350
   instructions' vector, quantizer, store and turnaround work, 5% today, would be 40% unless
   those units widen too. The row gating loss (18.2%) must be designed out, not scaled.

## The candidates

| | P1 wide port | P2 four lanes | P3 eight lanes, packed DSP | P4 four slices |
|:--|:--|:--|:--|:--|
| Port B | 256 B/cycle (C=2) | 512 B/cycle (C=4) | 1 KiB/cycle (C=8) | 4 x 128 B/cycle |
| HBM used at 200 MHz | 51 GB/s (11%) | 102 GB/s (22%) | 205 GB/s (45%) | 102 GB/s (22%) |
| MXU | 2 chunk lanes | 4 chunk lanes | 8 lanes, 2 MACs/DSP | 4 slices x today's |
| VPU / quantizer | 8 lanes | 16 lanes | 32 lanes | 8 per slice (x4) |
| TMEM | 64K words | 128K (URAM) | 256K (URAM) | 64K per slice |
| Gemma 3 kernels | unchanged | unchanged | unchanged | split across 4 slices (new) |
| Proposal | [proposal-1-wide-port.md](proposal-1-wide-port.md) | [proposal-2-four-lanes.md](proposal-2-four-lanes.md) | [proposal-3-eight-lanes-packed.md](proposal-3-eight-lanes-packed.md) | [proposal-4-four-slices.md](proposal-4-four-slices.md) |
| Decode bound at 200 MHz | 51 tok/s | 102 tok/s | 205 tok/s | 102 tok/s |
| Proposal's realistic decode | 38-41 tok/s (3x) | 69-79 tok/s (5-6x) | ~175 tok/s short context, ~164 at 2048 (13x) | 72-89 tok/s (6-7x) |
| Prefill (projected) | ~120 tok/s | ~300 tok/s | ~800 tok/s | 250-400 tok/s |
| VU47P: DSP / RAMB36 / URAM | 1,268 / 640 / 29 (14% / 32% / 3%) | 2,540 / 340 / 32 (28% / 17% / 3%) | 2,910 / 280 / 192 (32% / 14% / 20%) | 2,864 / 975 / 64 (32% / 48% / 7%) |
| New RTL | 2nd column group, HBM adapter, reorder FIFO, row-credit fix | 4 lanes + crossbar, HBM adapter, VPU16, TMEM 128K URAM | 8 packed-DSP lanes, HBM adapter, VPU32, TMEM 256K, WIN 32, 3 clock domains | 4-slice top + pipelined collective hub, HBM bridge; slice unchanged |
| Model / compiler work | none (ISA unchanged) | none (lanes are a uarch key; fallback: lane-striped image) | laned image layout, MM LANE flag, laned K cache | Gemma split over 4 slices in gemma4.py (cfg.S == 1 is baked in today) |
| Main risk | the row-gating bubble at 2 chunks/cycle (37% if unfixed) | 16-lane TMEM arbiter and the 200 MHz chunk network | packed-DSP column timing (167 MHz OOC at 4 columns), the 1 KiB datapath across SLRs | the collective hub (1 beat / 7 cycles today), P&R of 2 slices per SLR, the model split |
| Effort (proposal's estimate) | ~4-5 months | ~5-6 months | ~4-5 months | ~3-4 months |
| + software (card loop, row folding, overlap; [software-optimizations.md](software-optimizations.md)) | 40-48 tok/s | 90-100 tok/s | 178-197 tok/s | 88-98 tok/s |
| + fp4 layers, int8 head (bound; accuracy caveat) | 68 | 144 | 283 | 140 |
| Batched, 4 rows a weight pass (aggregate) | 142 | 298 | 583 | 256 |

Each proposal has the same sections: summary, datapath changes, ISA / compiler / host
changes, VU47P resource budget, expected Gemma 3 performance, timing and risks at 200 MHz,
implementation plan with milestones, what to verify in Lens, and a table of every other unit
resized (ACT RAM, epilogue lanes, result FIFO, chunk FIFO, port A / SW, quantizer, TMEM ports,
DMA, IMEM / sequencer window, collective, host, clock domains).

## Lens profiles of the candidates

`lens record model --model gemma3 --isa --candidate C=n,LANES=n,MHZ=200` (added for this
study) re-times the measured decode token's 5,350 instructions on a candidate machine: MM
work divided by C (C chunks a cycle through port and MXU), vector / quantizer / store work at
the candidate's lane width, the DRAM bound divided by C, at the given clock. It is the ISA
simulator's analytic timing: units serialized, no RTL overlap and none of the RTL's losses,
so it is a model for comparing candidates, not a measurement. P4 is modelled as C = 4 with
8 lanes: each slice streams a quarter of the bytes but repeats the norms and vector work; the
collective's cost is not in the model.

`build/lens/v4/v4-candidates.otpuprof` holds all seven profiles (the profile dropdown
switches between them): `python3 -m opentpu.lens open build/lens/v4/v4-candidates.otpuprof`.

| Profile | Cycles | ms at 200 MHz | tok/s bound | Of its DRAM bound | MXU | VPU | QUANT |
|:--|--:|--:|--:|--:|--:|--:|--:|
| today, RTL (measured cycles) | 10.28 M | 51.4 | 19.5 | 76.4% | 100% | 4.5% | 1.2% |
| today, ISA model | 8.28 M | 41.4 | 24.1 | 94.8% | 94.6% | 3.5% | 1.2% |
| P1: C=2, 8 lanes | 4.36 M | 21.8 | 45.8 | 90.0% | 89.8% | 6.7% | 2.3% |
| P2: C=4, 16 lanes | 2.19 M | 10.9 | 91.4 | 89.7% | 89.5% | 6.7% | 2.6% |
| P3: C=8, 32 lanes | 1.10 M | 5.5 | 181 | 89.0% | 88.9% | 6.7% | 3.2% |
| P4: 4 slices (C=4, 8 lanes) | 2.40 M | 12.0 | 83.2 | 81.7% | 81.5% | 12.1% | 4.1% |

Reading the models:

- The matmul work scales exactly with C (MM busy 7.84 M -> 3.92 M -> 1.96 M -> 0.98 M).
- The vector work scales with the lanes (VOP.mul 151 k -> 76 k at 16 lanes -> 38 k at 32),
  so P2 and P3 hold the VPU at 6.7% of the run only because they widen it with the MXU. P4
  keeps 8 lanes per slice and the vector share doubles to 12%.
- **The KV writes do not scale.** QST's DRAM-write term is a byte a cycle on port SW: 13.7 k
  cycles in every candidate. At P3 that is 1.2% of the token; at long context (more K / V
  rows a token) it grows. Port SW must widen with the port (each proposal's section 9).
- The RTL will lose more than the model: today the RTL is 0.81x the ISA model (10.28 M vs
  8.28 M), mostly the row gating. Applying today's ratio: P1 ~37 tok/s, P2 ~74, P3 ~146,
  P4 ~67. Each proposal says how its design removes the row-gating loss and what the RTL
  should then reach.

## How to inspect

```sh
python3 -m opentpu.lens open build/lens/v4/v4-candidates.otpuprof     # all seven profiles
python3 -m opentpu.lens summary build/lens/v4/p2-four-lanes.otpuprof   # one in the terminal
python3 -m opentpu.lens record model --model gemma3 --isa --candidate C=4,LANES=16,MHZ=200 \
    -o my.otpuprof                                                     # another candidate
```

In the app: Overview for the roofline fraction and the per-class table; Timeline for where the
LM head's 3 M cycles go; Floorplan to replay the measured RTL run cycle by cycle (the
candidates are ISA models: instruction mix and DRAM traffic, no cycle timeline).

The realistic numbers are each proposal's own loss model over the 133 MHz trace (section 5
of each); the Lens re-timings above are the common upper bound. All four agree on one
prerequisite: the row-gating bubble (2.6-2.9 cycles per streamed row today) has to go, by a
result-FIFO credit / scale-prefetch change in otpu_mxu.sv, before any wider port pays off.
[software-optimizations.md](software-optimizations.md) covers what the kernels and the
decode loop add on top and is applied to every candidate. Its findings: the megakernel
already exists (the resident decode program plus the card's own generate loop) and the RTL
already hides three quarters of the non-matmul work; what is left is (1) benchmarking and
shipping on the card loop, not the host loop (free; 7% at P3's 5.5 ms token), (2) row
folding in the image (the LM head stored 4 vocab rows per streamed row, the MLP's down
projection in wider chunks: 961 k streamed rows a token -> 585 k, +3% with the RTL gating
fix, +9-16% without it; not bit-exact to today's image, within the int8 head's noise, to be
verified against Hugging Face), (3) n-gram speculative decoding (1.03-1.10x on chat, up to
1.5x on code and summaries; no draft model exists for Gemma 3), and a FIFO deep enough to
hide the dependent chains between matmuls (about 3,000 chunks, 3x today's, an RTL knob).
Together: 1.14-1.29x per candidate, in the table. A one-line cycle model in that document
reproduces today's token to 0.7%.

## Decision (taken 2026-10-18)

Try **P1** (the wide port), **P2** (four lock-step chunk lanes) and **P4** (four
tensor-parallel slices), all three, and pick on measured results; **Milestone 0** first: split the RTL's "MXU not
consuming" counter by cause (result-FIFO credits / row not in the FIFO / scales missing /
drain) and measure it on the Gemma 3 trace, since every estimate hinges on it. Correctness
checks run after each implementation milestone, before any F2 synthesis. The work is carried
out by AI coding agents in separate worktrees off this branch.

**Milestone 0 is done** ([m0-gating-split.md](m0-gating-split.md)): of the 18.2%, 97% is rows
whose chunks had not all arrived with the chunk FIFO under one row deep (the memory model
delivering 0.76 chunks a cycle to a port that requests one: the card's DDR3, not the MXU's
gating), 2.6% scale entries behind their chunks, 0.1% result-FIFO credits. With the memory at
the port's rate (`--stall 0`) the token is 8.22 M cycles (95.5% of the roofline, 1% under the
ISA model) and the item is 2.1%: 0.8% scales, 1.2% rows late at the DDR3 model's hiccups, 0.07%
credits. The proposals' result-FIFO credit changes are not a performance item for Gemma 3; the
scale-line prefetch and the per-row fixed costs at C chunks a cycle are.

### The original ask

Pick one candidate for the F2 synthesis. The implementation is to be carried out by AI
coding agents after approval, so each proposal's plan is written as milestones with a
self-contained brief, the files to change, the test that proves the milestone (the repo's
standard: bit-exact against the per-position programs and the ISA simulator, `python3 -m
pytest -q`) and the Lens check that proves the performance. The order is the same for all:
the ISA simulator and the compiler first (otpu-chat runs Gemma 3 on the candidate's
configuration bit-exactly before any RTL), then the RTL under Verilator with Lens on every
milestone, then the F2 bitstream (AFI) and the EC2 F2 bring-up on AWS.
