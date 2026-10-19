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

**Measured at 200 MHz with an HBM-like memory model** (2026-10-18, `lens record model
--model gemma3 --axi --stall 0 --mhz 200` with `OTPU_LAT=30`: the Verilator board model's
AXI memory never withholding a handshake, 30 cycles of latency;
`build/lens/v4/today-rtl-200mhz-measured.otpuprof`):

| | Cycles | Of the DRAM bound | ms at 200 MHz | Device tok/s |
|:--|--:|--:|--:|--:|
| memory model with the DDR3 default, 20% random stalls | 10.28 M | 76.4% | 51.4 | 19.5 |
| memory model with 5% random stalls (HBM jitter) | 8.65 M | 90.7% | 43.3 | 23.1 |
| memory model with no stalls (HBM-like) | 8.22 M | 95.5% | 41.1 | 24.3 |

The "MXU not consuming" loss follows the stall rate (20%: 18.2%, 5%: 5.1%, 0%: 1.4%): it is
the memory's irregularity seen from the MXU's row gating.

With the smooth memory the "MXU not consuming" loss is 1.4% (from 18.2%), the turnarounds
1.5% (from 3.9%), the quantizer waits 0.8%, the VPU 0.5%, the chunk FIFO 0.2%: today's
datapath is within 5% of its port's roofline when the memory keeps up. The 18.2% in the
133 MHz trace was the DDR3 model's random stalls making rows arrive irregularly (the row is
not complete when the MXU could pop it), not result-FIFO credits. Milestone 0 measures the
split at both settings; the proposals' loss models, written against the 20%-stall trace, are
pessimistic by that amount, and the row-gating fixes they propose are worth less than they
estimated. What remains true: the port is the limit, and every fixed cost grows C times.

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
| **Implemented and measured on the RTL** (no stalls, HBM-like; `*-status.md`) | **4.64 M cycles, 91.6% of its 4.25 M bound: 43.1 tok/s** | **2.29 M cycles, 85.6% of 1.96 M: 87.3 tok/s** | not started | M1 on the ISA simulator: 2.25 M analytic cycles a slice, 89 tok/s (the RTL top not built) |
| Measured at the DDR3 model's 20% stalls | 5.41 M (78.6%): 37 tok/s | 3.01 M (65.1%): 66 tok/s | | |
| What the measured run says | half pairs on odd KB rows raise the bound 8%; VPU + quantizer waits 3.0% | VPU + quantizer waits 4.1%, turnarounds 1.3%: the chains, as predicted (VPU16 is M3) | | the collective and the per-slice VPU share are the open items |
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

## Branch state (2026-10-19)

Merged into tpubench: Milestone 0 (the blocked-cycle split), P4's Milestone 1 (the Gemma
split over slices), the `f2-hbm` port, the Qwen3.8 loader and runs. On their own branches,
complete and independently re-verified (their RTL gates re-run by the coordinator: P1
`test_token_on_rtl_is_bit_exact[cpc1,cpc2]` 2 passed, P2 the chunk-lane tests 6 passed,
P4's slice tests 3 passed): `v4-p1-wide-port` (`docs/v4/p1-status.md`) and
`v4-p2-four-lanes` (`docs/v4/p2-status.md`). They are not merged yet because both replace
the simulation memory path that the `f2-hbm` port's NCH adapter now owns (P1 rewrites
`otpu_native_dram` to 2 x CPC channels without the fold the host's channel map assumes and
adds a second `targets/f2.toml`; P2's lane ports are tied off at the board top): the next
engineering step is to reconcile each with the HBM adapter (the fork's `wide-port` PORTB
adapter is the model for P1; P2 needs per-lane queues in `otpu_f2_hbm_nch`), then build.

## Qwen3.8-27B-INT4 on the baseline (2026-10-19)

RedHatAI/Qwen3.8-27B-INT4 (a Qwen3.5-family hybrid, 64 layers, hidden 5120, compressed-tensors
int4 symmetric group-128) runs on today's design at S = 4 slices on the ISA simulator
([../qwen38.md](../qwen38.md)): a compressed-tensors loader (`opentpu/llm/ctensors.py`, the
nibbles offset-binary as the library packs them), the image 3.67 GiB a slice (int4 layers,
int8 head, the embedding table on the host; re-quantized to int4x at 4.6% relative RMS error,
since int4x's power-of-two sub-block scales cannot hold a bf16 group scale exactly), and a
host-side bug fixed on the way (the row server wrote slice 0 only; `tests/test_qwen38.py`).
Measured on the first 8 layers (`build/lens/v4/qwen38-27b-int4-8layers-isa.otpuprof`):
6.15 M analytic cycles a slice, 742 MB streamed a slice, the LM head 2.48 M of them.
Measured on all 64 layers (`build/lens/v4/qwen38-27b-int4-isa.otpuprof`, 2026-10-19):
31.67 M analytic cycles a slice a token, 158 ms at 200 MHz, **6.3 tok/s** on four of
today's slices (3.7 GB of weights streamed per slice per token, 14.8 GB over the four; the
MXU 89% busy: the MLP 53%, DeltaNet 27%, the head 8%, attention 5%); the greedy reply to
"Why is the sky blue? Answer in one sentence." is "The sky appears blue because shorter
wavelengths of" (8 tokens; 110 s a token on the ISA simulator). P2 or P3 lanes would scale it
as they scale Gemma 3.

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

## F2 status (2026-10-18, on hold by decision)

The F2 flow exists in the sibling tree `ergodex-03/openTPU-vla-rhino` (same base commit as
this branch): targets `f2` (DDR, 100 MHz, the only AFI built from that tree), `f2-hbm`
(one slice on 8 HBM pseudo-channels at 200 MHz, its own adapter `otpu_f2_hbm_*`; synthesized,
never routed), `f2-hbm:s4` (four slices on all 32: P4's hardware), `f2-hbm16`; the host
transport `opentpu/host/f2.py` (BAR0 / BAR4 over the small shell, no XDMA), the build flow
`boards/f2/build.py` (aws-fpga HDK 2.3.0, Vivado 2025.2, recipes A1 B2 C0 H2) run on the F2
instance over SSM, and `wide-port` (a PORTB 1 / 2 / 4 port through that adapter: P1's and
P2's bandwidth, its PORTB 4 / MCOLS 8 build failing timing at WNS -2.3 ns, round 6 running).
The two 200 MHz AFIs in the account (agfi-0fb51b57637d7fac0, agfi-0d06cb48544a02106
"burst200") are a separate CL with a SmartConnect, no source in either tree; Gemma 3 1B ran
on them bit-exact to the ISA simulator at 6.1 tok/s (the stream at 32 B/cycle through the
SmartConnect) and 22.3 tok/s on burst200. The branch to port is `f2-hbm` (ef33566c, 11 commits,
no core-RTL conflicts with this branch); P1's adapter rewrite must not be adopted (it drops the
host's channel map); P2's lane ports are tied off at the board top and need the adapter's
per-lane queues. Build host when resumed: `fpga-dev-f2` (i-00a83cab3c81ac10f, idle).
**2026-10-19: resumed.** The `f2-hbm` branch is merged into tpubench (61d07199,
[f2-port-status.md](f2-port-status.md): Gemma 3 bit-exact through the eight-channel board
model in 8.03 M cycles, 97.8% of the bound). The first routed build of the `f2-hbm` CL
(today's compute, one slice on 8 HBM pseudo-channels, 200 MHz) runs on `fpga-dev-f2`
(`/home/ubuntu/rachit/src/tpubench`, unit `otpu-f2hbm-build`, log
`/home/ubuntu/rachit/logs/f2hbm-build.log`, output `/home/ubuntu/rachit/f2hbm-run`); the
Gemma 3 checkpoint is staged at `/home/ubuntu/rachit/models/gemma-3-1b-it`. Next: the AFI
(upload of the Developer_CL.tar, `create_fpga_image`), load on slot 0, Gemma 3 with the
hardware Lens trace; then `f2-hbm:s4` with the Gemma split (P4 on silicon).
Build 1 (61d07199, `SSI_SpreadLogic_high`) failed timing after placement (WNS -4.03 ns on the
host write bus at 450 MHz, -2.0 on the bridge's write-done into the slice at 200 MHz, -1.0 on
the DMA-to-TMEM address at 400 MHz; [f2-timing.md](f2-timing.md)) and was stopped; the fixes
(registers per port on the host master, a write-done stage, fan-out replication of the TMEM
address sources, CDC bounds) are merged (a2af0904, all F2 Verilator suites and the Gemma 3
token bit-exact) and **build 2** runs with `--place-directive Default` (unit
`otpu-f2hbm-build2`, output `/home/ubuntu/rachit/f2hbm-run2`). Build 2 after placement and phys_opt: WNS -2.19 ns, TNS -25,900 (from
-3.66 / -68,500): the host read-return FIFO's write enable at 450 MHz (-2.6), the bridge's
queue write enable at 200 MHz (-2.4, 8 logic levels), the DMA-to-TMEM address at the pumped
clock (-1.4), a shell-side path (-1.0), the HBM reset pipe into 32 ports (-0.9); round 3 (block
RAM FIFOs / registered write enables, a pipelined bridge, pblocks for the slice and the
host side, a replicated reset pipe) is merged (8b2fd3a0, [f2-timing.md](f2-timing.md)
"Round 3": register-enabled FIFOs, a registered bridge head, skid buffers and links on the
shell side, pblocks for the slice / TMEM / host side, MAX_FANOUT on the HBM reset pipe, and
the flow fix that makes the HDK read the floorplan Tcl at all; all F2 suites pass and the
Gemma 3 token stays bit-exact) and **build 3** runs (unit `otpu-f2hbm-build3`, output
`/home/ubuntu/rachit/f2hbm-run3`). Build 2 routed at WNS -2.24 ns and was stopped.
Build 3 routed at **WNS -1.40 ns** (TNS -29,000, hold clean; the floorplan applied, the reset
exception on 12 endpoints): the host-to-port link at 450 MHz is a bare 3.4 ns wire (-1.4),
three fabric paths at 200 MHz ride 5 ns of route (-1.0), the TMEM address paths are within
0.5 ns. Round 4 is merged (ae770a3d, [f2-timing.md](f2-timing.md) "Round 4": one-beat
host writes behind link stages per port, a registered write-done compare and hazard mirror
in the adapter, a registered TMEM priority mask, a bridge pblock on the HBM ports' row and the
MXU's request registers in the TMEM pblock; cycle-exact and bit-exact in Verilator) and
**build 4** ran on fpga-dev-f2 to post-place WNS -2.18 ns (place_design complete, stopped
there). **Build host change (2026-10-19 15:00 UTC): only `opentpu-f2` (i-0c78de2253f66519d)
is used from here on; fpga-dev-f2 is abandoned** (its runs 1-4 and tree stay on its disk,
nothing further is read from it). Build 4 was restarted on opentpu-f2 from a fresh clone of
f449957a (`/home/ubuntu/rachit/src/tpubench`, venv `/home/ubuntu/opentpu-venv`, HDK
`/home/ubuntu/aws-fpga` 20fe9018 with ip 6d32be97, Vivado 2025.2 at `/opt/Xilinx/2025.2`,
checkpoint symlinked at `models/gemma-3-1b-it`): unit `otpu-f2hbm-build4`, script
`/home/ubuntu/rachit/f2hbm-build4.sh`, log `/home/ubuntu/rachit/logs/f2hbm-build4.log`, output
`/home/ubuntu/rachit/f2hbm-run4`, `--place-directive Default`, alongside two unrelated Vivado
jobs on the box (24 vCPUs, 249 GB). Round 5 (Alex's scheme: every openTPU block at 200 MHz
behind the HDK's SmartConnect) is merged (4423a24a, [f2-round5.md](f2-round5.md): every
openTPU block of the HBM fabric at the core clock, Alex's `otpu_native_axi` coalescer per
channel, one HDK `cl_axi_sc_1x1` SmartConnect per port for the 450 MHz side, a fail-closed
gate that no openTPU register is on hclk; 14 + 11 + 183 tests, Gemma 3 bit-exact at the same
8,029,592 cycles) and **build 5** runs on opentpu-f2 (unit `otpu-f2hbm-build5`, script
`/home/ubuntu/rachit/f2hbm-build5.sh`, log `/home/ubuntu/rachit/logs/f2hbm-build5.log`, output
`/home/ubuntu/rachit/f2hbm-run5`, `--recommended --floorplan five`: Alex's run-63 recipe,
ExtraNetDelay_high / AggressiveExplore / Explore / retime). Build 4 keeps running beside it for
round 4's routed number.

**Alex's burst200 sources found (2026-10-19).** Not in any git tree or S3 (S3 holds only the
DCP tars, AFI logs and host software under `opentpu-alex/`); they live in his run workspace on
fpga-dev-f2, `/home/ubuntu/Desktop/Ariane-Ibex-Scratch/alex/openTPU/runs/` (the lineage
`20261014_48_burst_fabric` -> `50_burst_link` (rtl/cl_opentpu.sv, otpu_native_axi.sv,
otpu_hbm_geometry.sv; floorplan.tcl, core_timing.tcl, platform_constraints.tcl, config.tcl)
-> `55_burst_impl` -> `63_burst_retime` -> `68_burst_package` -> `69_burst_afi`), snapshotted
to `s3://ergodex-bonsai-on-chip-438374459156-us-east-1/opentpu-alex/sources-snapshot-20261019.tar.gz`.
His design is the structural answer to our shell's timing: every openTPU-side block runs at
200 MHz (two native channels, each through `otpu_native_axi`, a read coalescer into 16-beat
512-bit AXI4 bursts), and the HDK's `cl_mem_hbm_wrapper` with its SmartConnect does the
clock crossing and 256-bit / 450 MHz side; a single SLR0 pblock with explicit ranges; place
`ExtraNetDelay_high`, phys_opt `AggressiveExplore`, route `Explore`, retiming (his retime run
-0.033 ns, the shipped AFI +0.032). Its limit is bandwidth (2 channels through the
SmartConnect: 32 B/cycle in the first image, ~20 GB/s in burst200), which is what our 8-channel
fabric fixes. Round 5 should merge the two: our 8 bridges at 200 MHz feeding vendor clock /
width converters per port instead of our own 450 MHz logic.

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
