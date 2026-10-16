# The hardware loop: LPN and VisualSim -> Verilator -> nightly FPGA

Status (2026-10-16): the tools in `tools/hwloop/` run end to end on this Mac up to the
VisualSim execution, which is **waiting for the seat** (the user's Architect GUI holds the
single license; the runner refuses to start a second instance and never stops the GUI). The
LPN stage **runs on this Mac** (section "The LPN stage": LPNDialect built with LLVM / MLIR 20
from conda-forge, upstream's GEMM / MLP rows reproduced to the cycle on Verilator 5.048, and
`cli.py lpn-run` grades this loop's PORTB 1 token at **+0.9%** of the Verilator receipts, inside
upstream's 5% gate but outside the model's validated configuration, so labelled `lpn-estimate`
and not a calibration: `tools/hwloop/lpn/README.md`). The nightly FPGA step is **prepared, not activated**. One
calibrated workload (Gemma-3-1B int8 decode on the F2 HBM configuration) is the only workload
covered; broadening comes after the first model runs have been compared with the receipts.

Rules of the loop (the user's):

1. **Models before RTL.** A candidate is first costed in the performance models (LPN for the
   slice's timing, VisualSim for the memory system) on the frozen tree. The models say whether
   the change is worth an RTL day; they never say the RTL is right.
2. **Verilator before synthesis.** Every RTL change is bit-exact against the ISA simulator in
   Verilator and gets its cycle receipt there. That receipt is also what re-calibrates the
   models: a model's number is "projected" until the Verilator run of the same change lands.
3. **Vivado only in the nightly.** Nothing in `tools/hwloop/` launches an instance, programs the
   card, submits an AFI or starts a schedule.
4. **Branches and PRs per track.** Each track (wide-port, ternary-bonsai, vla, vla-pack, hwloop)
   works in its own worktree and branch, commits as receipts land, and opens its PR when the
   change has a Verilator receipt and green tests. Reviews are short and on measured numbers.

## What exists (inventory, 2026-10-16)

| Piece | Where | State |
|---|---|---|
| ISA cost model / profile of a run | `opentpu/profile.py` (`Profile`, `parse`, `ddr3_plusargs`) | main; the per-unit busy, phases, MXU gaps |
| RTL timing run of one decode token with real weights, the board's memory path | `tools/perf_qwen.py` (`--json`, `--plus +nat_trace=FILE`, `--check` against the ISA simulator) | branches f2-hbm / wide-port / ternary-bonsai (not on main) |
| Verilator build / run | `opentpu/rtlsim.py` (`MEMORY`: `OTPU_LAT`, `OTPU_NCH`, `OTPU_STALL`, `OTPU_BW`, `OTPU_LDC`) | main (NCH on the F2 branches) |
| The RTL testbench's inputs | `opentpu/rtlsim.py` writes `prog_<slice>.hex`, `dram_<slice>.bin` (and reads back `dram_out_<slice>.bin`, `tmem_<slice>.hex`) | main; the same files the LPN model reads |
| Board memory model with a command trace | `sim/verilator/otpu_native_mem.sv` (`+nat_trace`, `+axi_lat`, `+axi_dram=1` bank model) | main |
| Card trace records | `opentpu/hwtrace.py` (`rtl/shell/otpu_trace.sv`) | f2-hbm and later |
| F2 build | `boards/f2/build.py` (`--flow SynthCL/ImplCL/BuildAll --package`) | f2-hbm and later; Vivado on the build host only |
| Remote synthesis hosts | `tools/tourney/remote.py` (`OTPU_BUILD_HOSTS`) | f2-hbm and later |
| Serving bench on the card | `tools/f2_bench.py --json` (TTFT, prefill tok/s = prompt/TTFT, decode tok/s) | ternary-bonsai |
| LPN model of the slice | `MJChku/LPNDialect`, branch `qwen-validation`, `examples/opentpu/` (`benchmark.py`, `run.py`, `e2e.py`, `replay_e2e.py`, `model.py`, `components.py`, `workload.py`, two Verilator patches) | upstream, pinned to this repository's `f237dab2` (origin/main 2026-10-14); **built on this Mac** (`b06f42d9`, `~/ergodex-03/lpn-scratch`; `tools/hwloop/lpn/README.md`), 13 / 13 ctest, GEMM / MLP rows reproduced |
| Measured: Gemma-3-1B int8 on Alex's burst200 image | `docs/wide_port.md` "Baseline on the board", `docs/ternary.md` 5 (ternary-bonsai) | 22.3 / 21.7 tok/s decode, prefill 3.7-4.5 tok/s (serving), TTFT 4.3 s / 59.6 s; the JSON receipt is on the F2 host, not in git |
| Measured: RTL at the same model, 4 layers + head | `docs/wide_port.md` "Baseline on the RTL"; `~/ergodex-03/bonsai2-out/perf_gemma3_l4.log` | 3,228,204 cycles, 99.4% of the port-B roofline |
| Measured: PORTB 1 / 2 / 4, 2 layers + head | `docs/wide_port.md` (wide-port a9acb2b0) | 2,798,478 / 1,643,595 / 1,308,691 cycles |
| Measured: Bonsai-2 2B / 4B on the board | `docs/ternary.md` 5 | 9.06 / 6.28 tok/s probe; 16.8 / 8.0 serving |
| Measured: the F2 DDR image | `bench_ours100.json` (copied to `VS_AR/ergodex_npu/evidence/measured/`) | Qwen3-0.6B 11.48, Gemma-3-1B 7.04 tok/s |
| ASIC points | `docs/asic.md` | NanGate45: 340-386 MHz on the card point |
| VisualSim Architect | `~/VisualSim/VS_AR` (2641, 2026-09-13), Java 17 at `~/Library/Java/JavaVirtualMachines/jdk-17.0.20+8`; local FlexLM on 1099 / 1900 / 2222 | installed; GUI running (PID in `cli.py inventory`) |
| Library index | `VS_AR/doc/pre-built_models/html/Library_Model.html` | HBM, DMA, AMBA_AXI, PCIe_DMA links |
| Examples reused | `Functional_HBM_Demo.xml`, `HBM_Module.xml`, `Memory_Controller.xml` + `HW_DRAM.xml`, `Using_Mem_Controller_and_HWDRAM.xml`, `TPU_6.xml` (Smart_Timed_Resource), `DRAM_LPDDR5X.xml` | read, hashed in `tools/hwloop/PROVENANCE.json` |
| Prior VisualSim work | `VS_AR/ergodex_npu` (aggregate fits that ran in batch on 2026-10-16: `results/aggregate_completed`; six native-memory smoke models, static only), `~/Coral-VisualSim-Preliminary`, `coralnpu_bonsai-arch-improvements/tools/visualsim` | preserved; this loop adds `VS_AR/ergodex_npu/hwloop/` beside them |

Not a generic TPU: both models are built from this slice's structure (port B at D = 128 bytes
a core cycle into the MXU's chunk FIFO, port A's scale lines, the DMA's writes, the native
channel adapter with NCH channels and its per-channel outstanding reads, the bridge's 8-beat
bursts) and from this RTL's programs and traces. They divide the work: **LPN** times the
slice (the sequencer's command window, the MXU's service and epilogue tails, the quantizer,
the VPU, the DMA, port credits) from the program itself; **VisualSim** times the memory system
behind the ports (HBM channels or a DDR controller with its bank state and refresh) from the
trace's phases. LPN's DRAM is a latency / bandwidth abstraction; VisualSim's core is the
phases' tails. Neither replaces the other, and neither replaces Verilator.

## Missing calibration inputs

1. The board bench JSON of the 22.3 tok/s run (on the F2 host; `tools/f2_bench.py --json`):
   the profile `f2-hbm-burst200` carries the doc's numbers only.
2. Alex's burst200 image has no RTL source in the repository (a routed DCP): its bridge's
   outstanding reads, burst formation and the loaded HBM latency are unknown. The
   repository's own f2-hbm build (NCH 8, 64 reads in flight a channel) is synthesized, not
   routed or run, so no board number matches the RTL configuration the Verilator runs use.
3. A whole-model RTL run (`--layers 0`, 26 layers: ~8 M cycles, hours of Verilator) for a
   tok/s directly comparable with the board; the 2- and 4-layer runs compare bytes/token,
   port utilization and the phases' cycles instead.
4. The card's hardware trace (`otpu_trace.sv`) of a board token, for the memory-side stalls
   the serving numbers hide.
5. HBM device parameters (refresh, bank timing under the cl_hbm IP's controller) and the
   DDR4 DIMM's part on the F2.
6. (done 2026-10-16) The LPN build on this Mac and the replay of this loop's PORTB 1 receipt:
   LPN 2,823,232 cycles against 2,798,478 (**+0.885%**) and against the pinned RTL on the same
   bundle, 2,797,992 (**+0.902%**, the model's own validated configuration); at the receipt's
   60-cycle latency 2,874,213 (+2.706%). What is still missing: a re-validation of the model on
   the **wide-port tree** itself (its RTL fingerprint differs from the pin in 16 changed and 14
   added files, so every LPN number for it is `--uncalibrated` by rule), and the model
   extensions for PORTB 2 / 4, NCH 8 at 60 cycles and the VLA shape (section 5 of
   `tools/hwloop/lpn/README.md`), each graded against its Verilator receipt before use.

## The per-change loop

The order is models, then RTL, then silicon: LPN and VisualSim on the frozen tree first (hours
of Verilator saved on candidates the models already rule out), Verilator for the one the
models favour (bit-exact, and the cycle receipt that grades the models), the nightly for what
Verilator passed.

```
# 0. freeze: the pinned source tree (git archive of one commit: never another worktree),
#    its RTL fingerprint, the checkpoint, quantization, position, memory placement
python tools/hwloop/cli.py freeze --rev a9acb2b0 --out /scratch/opentpu-a9acb2b0 --models /path/to/models \
    --layers 2 --target f2-hbm --nch 8 --lat 60 --mhz 200
# 1. the candidate: a parameter of the same tree (PORTB 2) or a commit on top of it
# 2. LPN on the frozen tree: the slice's cycles for the baseline and the candidate from the
#    program and DRAM image perf_qwen hands the testbench (section "The LPN stage"); minutes,
#    not the 30-37 min of a traced Verilator run. Output labelled projected (calibrated) when
#    the configuration is one the replay gate has passed, projected (uncalibrated) otherwise.
PYTHONPATH=$LPN ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.run --opentpu-root /scratch/opentpu-a9acb2b0 \
    --input runs/base/lpn/input --emit runs/base/lpn/model.mlir --lpn-compiler $LPN/build/tools/lpn-compile/lpn-compile \
    --output runs/base/lpn/opentpu && runs/base/lpn/opentpu --stats --wall-start=begin --wall-end=complete
# 3. VisualSim on the same tree: the memory system under the phases LPN (or the last
#    Verilator trace) gives; one model per memory profile, fingerprinted against the tree
python tools/hwloop/cli.py build-vs runs/base --profile f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0
python tools/hwloop/cli.py build-vs runs/cand --profile f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0 --variant candidate
python tools/hwloop/cli.py check-stale runs/base/vs/f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0
python tools/hwloop/cli.py vs-run runs/base/vs/f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0 [--execute]
# 4. decide: the candidate goes to RTL only if the models' delta clears the goal's margin
#    (the LLM track: cycles a token toward > 100 tok/s; the VLA track: the chunk's MXU bound)
# 5. Verilator: static checks + unit tests + the RTL regressions against the ISA simulator
#    (the independent numerical reference), then the memory-inclusive timing run + trace of
#    baseline and candidate (perf_qwen --check adds the ISA comparison of the same token)
OTPU_NCH=8 python -m pytest -q tests/test_rtl.py -k "board_memory_path or native_memory_path"
python -m pytest -q tests/test_wide_port_rtl.py
OTPU_PYTHON=~/bonsai2-venv/bin/python python tools/hwloop/cli.py run-rtl --tree /scratch/opentpu-a9acb2b0 \
    --out runs/base --layers 2 --portb 1 --nch 8 --lat 60 --mhz 200
python tools/hwloop/cli.py run-rtl --tree /scratch/opentpu-a9acb2b0 --out runs/cand --layers 2 --portb 2 ...
# 6. grade the models: LPN's and VisualSim's cycles against the Verilator receipt, the error
#    written into the run's receipt (LPN's own gate is 5%; a configuration whose error is
#    over it is "uncalibrated" until the model is extended)
# 7. the report (JSON, CSV, Markdown; every number labelled measured / projected) and the
#    nightly queue
python tools/hwloop/cli.py report --baseline runs/base --candidate runs/cand --profile f2-hbm-burst200 --out runs/report
python tools/hwloop/cli.py nightly add --name portb2 --commit <sha> --target f2-hbm --run runs/cand
# 8. the canonical copy (model XML, workload inputs, evidence, results, reports apart)
python tools/hwloop/cli.py install runs/base --dest ~/VisualSim/VS_AR/ergodex_npu/hwloop/opentpu/f2-hbm-nch8-rtl/baseline --report runs/report
```

`run-rtl` records the exact command and environment (`command.sh`, `run_receipt.json`);
`extract` turns `perf.log` + `perf.jsonl` + `nat_trace.txt` into `trace.json` (cycles, bytes,
per-channel reads / writes, port-B utilization, MXU starved / blocked, the phases, an
outstanding-reads estimate from the issue times and the latency, idle gaps) and
`mem_windows.csv` (reads and writes per channel per window of 1024 core cycles).

Step 2's wrapper is `cli.py lpn-run` (`tools/hwloop/lpn_run.py`; `tools/hwloop/lpn/README.md`
section 2): the export's program and DRAM image into LPN's input bundle (`lpn/inputs.py` on the
export's own compiler and ISA simulator), the model and native binary (`lpn/emit.py`), the
run, `trace.json` in `run-rtl`'s record shape labelled `lpn-estimate`, `compare.json` against
`--compare`'s Verilator receipt (error, the 5% gate, the configuration deltas), `--rtl-pin` for
the pinned RTL on the same bundle; the LPN commit, patch and compiler hashes and the pin's RTL
fingerprint in `lpn/manifest.json`. The upstream command above is what `emit.py` wraps.

### Trace content, memory-inclusive

From `perf_qwen` (the RTL's own counters): cycles, the core-port roofline (port B chunks,
port A, MXU), per-channel DRAM read / write beats and port-A reads, MXU starved (chunk FIFO
empty) and blocked cycles, the units' busy, port B busy (MXU vs DMA chunks), the phases
(cycles, bytes, roofline, VPU / MXU busy), the MXU gaps and what they waited on. From
`+nat_trace`: every command the channels took (cycle, channel, read / write, beat). What the
trace does not carry: the bridge's bursts (formed in the 450 MHz domain, not in
`otpu_native_mem`), the HBM controller's bank state and refresh. The DDR3 card has the
calibrated bank model (`--ddr 1066`, `--ldc`) for those.

## The LPN stage (`MJChku/LPNDialect`, `examples/opentpu`)

LPN (labelled Petri nets) is a performance-modelling framework like VisualSim with a different
formalism: places hold tokens, transitions take tokens, update their fields and emit after a
delay; a Python DSL (`@net.transition`, `take`, `peek`, `emit`, `guard`) or MLIR text is
compiled by `lpn-compile` to LLVM IR and linked with a token / event runtime, so a model runs
as a native binary. Upstream's `examples/opentpu` is a model of **this** slice, kept against
this repository.

**What it models.** The timing of LD / ST, MM, QACT / QST, VOP, scalar control, RLD and the
single-slice BAR / GATHER; the DRAM ports as shared credit places every unit draws from; each
matrix engine's prefetch credit and request queue; the MXU's service interval and the
pipeline / epilogue tails on result and command-window credits. Approximated: instruction
fetch, TMEM arbitration, VPU instruction overlap, every protocol and controller detail; DRAM
is a latency / bandwidth abstraction (the validated setting: 128 bytes a cycle, 8-cycle read
latency). **Rejected by its timing backend:** WAITW, HALT CHAIN, DSTEP / STREAM, WAITC and
multi-slice collectives. A single decode token's program at one position (what `perf_qwen`
runs, and what the receipts below are) uses none of those; the generate loop's HALT CHAIN
between buckets, the card's WAITC sequencer, Qwen3.5's DSTEP / STREAM and the MTP loop do, so
those workloads stay on Verilator until the model grows them.

**Inputs.** Not observed traces: the same `prog_<slice>.hex` and `dram_<slice>.bin` the RTL
testbench reads (`opentpu/rtlsim.py` writes them for every run) plus a JSON manifest
(configuration, instruction counts, initial registers). Control flow and effective addresses
come from this repository's ISA simulator; no RTL timestamp enters the model. Every DRAM byte
and TMEM word of a replay is checked against the ISA simulator, and the dynamic command PCs.

**Pin and configuration.** Upstream pins `f237dab2f196855a59a083b1454786df8f01291f`
(origin/main, 2026-10-14: 60 commits past this Mac's `main` at 7e42a626; the wide-port tree
`a9acb2b0` this loop froze is not that commit, so the first replay must fingerprint `rtl/**`
between the two and list what moved). Validated RTL configuration: one slice, D 128, four
matrix columns, two epilogue lanes, eight TMEM / VPU lanes, replicated TMEM, a 32-entry
range-based sequencer window, a 128-chunk prefetch FIFO, PORTB 1. Upstream's own words for
anything else: extra physical memory ports, banks or a different arbitration need matching
resource changes in the model, and passing the existing tests establishes nothing for a
modified RTL. So **PORTB 2 / 4 (wide-port), NCH 8 at 60 cycles, MCOLS 32 / LANES 64 (f2-vla)
and the ternary formats are uncalibrated in LPN today.** Each is a model extension plus a
replay against the Verilator receipt that already exists for it (PORTB 1 / 2 / 4: 2,798,478 /
1,643,595 / 1,308,691 cycles at 2 layers + head).

**Accuracy and speed, upstream's measurements (Verilator 5.052; 5.020 does not elaborate the
pinned revision; this Mac's receipts used 5.048):**

| case | RTL cycles | LPN cycles | error | core speedup over Verilator |
|---|---|---|---|---|
| seven 4 x 4096 x 512 GEMMs | 119,723 | 119,674 | -0.04% | 254x |
| compiled SwiGLU MLP | 27,673 | 27,629 | -0.16% | 53x |
| held-out 8 x 2048 x 128 GEMMs | 16,939 | 16,894 | -0.27% | ~380x |
| eight 8 x 4096 x 2048 GEMMs | 1,059,504 | 1,059,454 | -0.005% | 697x |
| MLP H 3072, F 8192, M 4 | 610,507 | 598,253 | -2.0% | 80x |
| SmolLM2-135M story, per token | | | 1.2% first token, 3.3% worst of 136 | |
| Qwen3-1.7B, per step | | | 0.53% first token, 0.61% worst of 10 | |

Whole-process (compile + run) speedups are 2.5-18x; the benchmark fails over 5% error. These
are upstream's numbers on upstream's cases; none is a number of this loop until the replay
in "Missing calibration inputs" 6 is done.

**The two Verilator patches.** `verilator-observer.patch` adds two optional host clock readings
around the core run (`+lpn_wall`) and their DPI helper, no DUT change; `verilator-large-memory.patch`
splits the simulation memory's storage array into rows past Verilator's range limit (no bank
conflicts, no DRAM timing; applied after the observer patch, in an isolated pinned worktree,
with a storage-equivalence check). The loop applies them to the frozen `git archive` tree
only, never to a worktree.

**Fingerprinting.** An LPN run's `lpn/manifest.json` carries the LPN repository commit and
branch, the two patches' SHA-256 (and whether the observer patch is on the pin tree), the
compiler's SHA-256, and the **RTL fingerprint of the pinned tree the model is validated against**
(`f237dab2` + patch, over the same globs as the VisualSim manifests); `compare.json` names the
Verilator receipt it was graded against. `check-stale RUN --tree EXPORT` and `lpn-run` apply the
VisualSim rule: an export whose fingerprinted files moved is refused with the changed / added /
removed files listed (`REFUSED_STALE_MODEL`); `--uncalibrated` runs anyway and labels the result
"outside validated configuration: RTL moved (...)". Against wide-port `a9acb2b0` the list is 16
changed and 14 added files (`receipts/2026-10-16/lpn_gemma3_portb1/staleness.json`).

**This loop's receipts (2026-10-16, Verilator 5.048; `tools/hwloop/lpn/README.md`):** upstream's
GEMM and MLP rows reproduced to the cycle (119,723 / 119,674, -0.041%; 27,673 / 27,629,
-0.159%); the Gemma-3-1B PORTB 1 token (the same 2,779,176 MM weight chunks, bytes within
0.03%, the pin's frontend and the export's ISA simulator bit-identical): **LPN 2,823,232 cycles**
at the validated Timing, **+0.885%** against the 2,798,478-cycle wide-port receipt and **+0.902%**
against the pinned RTL on the same bundle (2,797,992; the pinned behavioral-DRAM RTL and the
wide-port NCH 8 / lat 60 RTL are 486 cycles apart on this port-bound token); at Timing.dram_latency
60: 2,874,213, +2.706%. Inside the gate, outside the validated configuration (the tree, NCH 8 at
60 cycles, the board uarch): `lpn-estimate`, recorded, not a calibration. 1.15 s of native run
(13 s end to end with the frontend) against 37 min of Verilator. PORTB 2: `REFUSED_NOT_MODELLED`.

**Where it pays.** The LLM track's parameter sweeps (PORTB 2 / 4 / 8, int4 weights with
PAIR, the lm head's format, NCH 8 / 16) and the VLA track's shape sweeps (MCOLS 32 packed,
ACT_ROWS 255, LANES 64 on the 558-968-row prefix) are dozens of points; at 30-37 min a traced
Verilator run each, the models choose the two or three that get a Verilator day. The ISA
tally (`opentpu/profile.py`, the scoreboard) stays the first estimate; LPN is the second,
with the sequencer window, the chunk FIFO and the drain in it, which the tally has not.

## The VisualSim model (`tools/hwloop/vsmodel.py`)

Native blocks only: `DS_Source` -> `Virtual_Machine` (the scheduler) -> per channel the
memory block -> back; `Smart_Timed_Resource` for port B's drain (one D-byte chunk a core
cycle, 16-deep queue: backpressure on the returns) and for the phases' unit tails; `Display`
counters (an `ExpressionWriter` file in batch). Memory by profile:

- HBM profiles: `VisualSim.actor.arch.Memory.HBM_Module`, one of its 8 channels per openTPU
  channel, `Channel_Buffer_Size` = the bridge's outstanding reads, address-selected channels
  (channel c owns `[c x span, (c+1) x span)`), the installed timing vector. It exposes no
  refresh parameter (listed in the manifest's `not_modelled`).
- DDR / LPDDR profiles: one `Memory_Controller` + `HW_DRAM` pair per channel with
  `DRAM_Type`, the data rate, the profile's refresh (`Retention_Time`,
  `Fine_Granularity_Refresh_Time`); the vendor's bank / row / turnaround scheduling.

The scheduler issues each phase's bytes as bursts round-robin over the channels, at most
`Credits_Per_Channel` in flight per channel (outstanding beats / burst beats), sequential
addresses per channel; every returned read must drain through port B before the phase ends;
the phase's tail (its cycles over its port roofline, from the trace) follows as a
dependency. Reported: elapsed, core cycles, requests issued / completed, bytes, achieved GB/s,
mean memory latency, credit-blocked events, peak in flight, per-phase elapsed, an estimated
device tok/s for the replayed layers. It is not `max(compute, memory)`: the overlap comes from
the credits, the drain queue and the dependencies.

Clocks and memory are independent parameters of every model (`Core_Hz`, `Controller_MHz`,
`Memory_Data_Rate_MTps`, the vendor block's timing); no profile derives an ASIC number from
the FPGA's by a clock ratio. The ASIC profiles' every memory timing is an assumption and
says so.

With LPN in the loop, the phases' tails VisualSim takes as dependencies can come from an LPN
run of the candidate instead of the last Verilator trace, which is what lets a candidate be
costed end to end before its RTL exists; the receipt says which source the tails came from.

### Profiles (`tools/hwloop/profiles/`)

| profile | kind | label | what |
|---|---|---|---|
| `f2-hbm-burst200` | actual F2 | measured | Alex's image agfi-0d06cb48544a02106: 200 MHz, HBM over 2 channels; the 2026-10-16 board baselines |
| `f2-hbm-nch8-rtl` | RTL evaluation | measured (Verilator) | the repository's f2-hbm (NCH 8, 64 reads in flight a channel, --lat 60) at wide-port a9acb2b0 |
| `f2-ddr4-100mhz` | actual F2 | measured | build f9e0e5bf / agfi-0014449e89d7fceee, DDR4 at 100 MHz; bench_ours100.json |
| `ypcb-ddr3-1066` | actual card | measured | the production card, DDR3-1066 through LiteDRAM (the existing DDR configuration) |
| `asic-1ghz-hbm` | hypothetical | unvalidated scenario | 1 GHz core, 8 HBM2e pseudo-channels, 800 MHz controller, 120 ns |
| `asic-1ghz-lpddr5` | hypothetical | unvalidated scenario | 1 GHz core, 4 x 16-bit LPDDR5-6400 (memory-bound by construction) |

### Staleness and the single seat

`manifest.json` holds the RTL fingerprint (every file of `rtl/**`, `sim/verilator/*`,
`opentpu/rtlsim.py`, `opentpu/profile.py`, `tools/perf_qwen.py` with its SHA-256) of the tree
the trace came from. `check-stale` and `vs-run --tree` recompute it: a model whose RTL moved
is refused with the changed / added / removed files listed. `vs-run --execute` also refuses
while any `ModelBuilderApplication` or `VisualSimBatchModeSimulator` process exists, takes an
exclusive lock (`~/.visualsim/hwloop-seat.lock`, shared with coral-gemma-f2's tools), runs
under a wall-clock limit in its own session, and checks `License freed: true` in the log.

## Receipts of the smallest working integration (2026-10-16)

Pinned tree: wide-port `a9acb2b05da2b0ddcdbb97254b2776ca2cee2c87` exported with `git
archive` (not a worktree). Gemma-3-1B int8 (int8 head), `--layers 2 --pos 9`, OTPU_TARGET
f2-hbm, OTPU_NCH 8, `--lat 60`, 200 MHz, Verilator 5.048, `~/ergodex-03/bonsai2-venv`.

- The 4-layer receipt the doc quotes (`bonsai2-out/perf_gemma3_l4.log`, 2026-10-16 15:37):
  3,228,204 cycles, 99.4% of the 3,207,980-cycle port-B roofline, 6,517,068 DRAM beats
  (417 MB), 1.007 read beats/cycle/channel, MXU starved 18 cycles, 16.14 ms at 200 MHz for
  4 layers + head (**measured, Verilator**). `extract` + `build-vs` on it give STATIC_PASS
  models for all six profiles.
- Fresh baseline, PORTB 1, with `+nat_trace` (`tools/hwloop/receipts/2026-10-16/opentpu_portb1/`;
  37 min wall with the trace): **2,798,478 cycles**, the documented number to the cycle;
  99.6% of the 2,787,778-cycle port-B roofline; 5,659,048 native beats = 362 MB (361 MB
  read, 1.08 MB written; 25.9 GB/s at 200 MHz, 2.02 beats/cycle over 8 channels); MXU starved
  17 cycles, blocked 0.3%; at most 203 reads outstanding by the trace's issue times and the
  60-cycle latency (the adapter's queues over 8 channels), 18 idle gaps over 64 cycles
  (9,473 cycles); phases: head ("other") 2,368,741 cycles / 312 MB, MLP 376,542 / 49 MB,
  attention 53,193 / 6.2 MB (**measured, Verilator**). 71.5 tok/s device for 2 layers + head:
  not a whole-model number.
- Fresh candidate, PORTB 2 (the same tree, `--portb 2`; `receipts/2026-10-16/opentpu_portb2/`,
  30 min wall): **1,643,595 cycles**, the documented number to the cycle, **1.703x** the
  baseline on the same bytes (362 MB: the change moves no more data, it takes two chunks a
  cycle); 93.5% of its 1,536,296-cycle roofline (the MXU's: PAIR takes 5 advances a 9-chunk
  row); 44.1 GB/s at 200 MHz, 3.44 beats/cycle; MXU starved 1,324 cycles, blocked 5.7%;
  1,242,880 wide reads of which 87,619 (7.0%) in two parts; at most 307 reads outstanding by
  the trace (**measured, Verilator**). The comparison: `receipts/2026-10-16/report_portb1_vs_portb2/`.
  No board number: PORTB 2 is not synthesized; it is the queued nightly candidate.
- VisualSim: every model prepared (snapshot, exact batch command, static validation), none
  executed: the GUI holds the seat. Status in every receipt: `PREPARED_NOT_EXECUTED`,
  label `waiting`.
- LPN (`receipts/2026-10-16/lpn_gemma3_portb1/`, `lpn_gemma3_portb1_lat60/`, `lpn_upstream_gemm/`,
  `lpn_upstream_mlp/`, `lpn_toolchain/`): the PORTB 1 token **2,823,232 cycles, +0.885%** against
  the receipt above and +0.902% against the pinned RTL on the same bundle (2,797,992 cycles,
  `measured`, Verilator 5.048, 10 min), +2.706% at the receipt's 60-cycle latency; inside the 5%
  gate, outside the validated configuration (**lpn-estimate**, not a calibration). PORTB 2: not
  modelled, refused; it waits for the port-credit extension (`tools/hwloop/lpn/README.md` 5).
- Nightly: `tools/hwloop/nightly/queue.json` + `cli.py nightly plan` (not activated).

## The agentic loop (who runs what, and when a PR opens)

Several coding agents run side by side, one per track, each in its own worktree so none
touches another's files:

| track | branch (worktree) | goal | what a commit carries |
|---|---|---|---|
| LLM decode | `wide-port` off `ternary-bonsai` (`openTPU-ternary-bonsai`, the `wide-port` worktree) | Gemma-3-1B int8 decode 22.3 -> > 100 tok/s on the F2, **without MTP first** (PORTB 4 / 8, int4 weights with PAIR, the lm head's format, NCH 16); MTP heads (docs/mtp.md) as the second stage | cycles a token on the RTL, labelled measured; the model's projection beside it |
| VLA | `vla` (`openTPU-vla`), `vla-pack` (`openTPU-vla-pack`) | RhinoVLA >= 1 Hz a chunk on one slice, board-measured (docs/designs/vla-1hz-packed-slice.md) | Verilator bit-exact against the ISA simulator; the phase's exit criterion; the shape tally made real |
| hardware loop | `hwloop` (`openTPU-hwloop`, PR #2) | this loop: models, receipts, the nightly queue | receipts under `tools/hwloop/receipts/<date>/`, every number labelled |
| f2-hbm shell | `f2-hbm` | NCH 8 / 16 channels, the build | the synthesized point and its co-sim cycles |

The cadence:

1. An agent costs its candidate in the models on the frozen tree (steps 2-4) and writes the
   projection into the run's receipt. A candidate the models put under the goal's margin is
   recorded and dropped, not built.
2. It makes the RTL or compiler change, bit-exact in Verilator against the ISA simulator, and
   runs the timing receipt (steps 5-6). It commits as each receipt lands, on its own branch,
   with the receipt's numbers in the message, and keeps pushing: small commits, never a
   squash at the end.
3. It opens the PR when the change has its Verilator receipt and its tests are green (the
   RTL suites the change touches; `tests/test_hwloop.py` for this loop), with the measured /
   projected table in the body. A PR without a receipt is a draft.
4. Review is short and on the numbers: the receipt against the goal, the model's error
   against the receipt, what the nightly should synthesize. Design reviews (office hours)
   are for a new approach, not for a parameter change; they end when the numbers are
   measured, not when the prose is polished.
5. The nightly (with approval) synthesizes the queued candidate on the build host, benches it
   against the known-good image, and the error goes back into the profiles' `measured`
   blocks and the LPN receipt's grade.

## Staged plan

1. (done) Tools, profiles, one workload's receipts, prepared VisualSim models, tests, this doc.
2. First VisualSim execution when the seat frees: `vs-run --execute` on the baseline
   `f2-hbm-nch8-rtl` model; fix the VM / protocol errors the vendor parser reports; compare
   the replay's cycles with the RTL's (the RTL is the reference for this profile: the memory
   model should reproduce its 99.4% port utilization within a stated error); then the
   candidate, then `f2-hbm-burst200` against the board's 22.3 tok/s (89% of the RTL's
   port-bound rate: the gap the memory model has to explain, not fit).
3. (done) LPN, first replay: LPNDialect (`qwen-validation` b06f42d9) built on this Mac with
   LLVM / MLIR / Clang 20.1.8 from conda-forge (Homebrew has no Intel bottle for llvm@20 any
   more); the observer patch on a detached worktree of the pin `f237dab2`; upstream's GEMM and
   MLP rows reproduced to the cycle on Verilator 5.048; this loop's PORTB 1 token replayed from
   the frozen `a9acb2b0` tree's program and DRAM image: +0.885% against 2,798,478 cycles,
   +0.902% against the pinned RTL on the same bundle; the 16 changed / 14 added files between
   `f237dab2` and `a9acb2b0` listed. `cli.py lpn-run` wraps it and writes the receipt.
4. LPN, extensions in the order the tracks need them, each graded against an existing
   Verilator receipt before its number is used: PORTB 2 (two chunks a cycle on the port-B
   credits and drain; the 1,643,595-cycle receipt), NCH 8 at 60 cycles (the 203 / 307
   outstanding reads of the traces), PORTB 4 (1,308,691), then the VLA shape (MCOLS 32 packed,
   LANES 64, ACT_ROWS 255: the prefix run, against `vla`'s Verilator utilization numbers).
5. Validation split: calibrate on Gemma-3-1B short-prompt decode only; hold out Gemma long
   prompt, Bonsai-2 2B / 4B (int4, t2), Qwen3-0.6B, the DDR image's receipts; report per
   workload and per model (LPN, VisualSim) absolute / relative error, mean and worst case.
6. A whole-model run (`--layers 0`) on the build host's Verilator, overnight, for the direct
   tok/s comparison; the card's hardware trace of a token.
7. Nightly: with approval, the queued candidate's SynthCL / BuildAll on the build host, the
   AFI, the matched bench against the known-good image, the errors fed back into the
   profiles' `measured` blocks.
8. Broaden: prefill (tiled, chunk-64) and TTFT as their own workloads; the generate loop's
   HALT CHAIN and the MTP loop in LPN (today rejected); multi-slice; the DDR3 card through
   the LDC co-simulation.

## Limitations

- A model run that completes (LPN or VisualSim) proves the model runs, not that the RTL is
  correct or that any number is a board number. Numerical correctness is `perf_qwen --check`,
  the ISA simulator and the card's token-exact checks, nowhere else.
- LPN's accuracy on this loop's workload is one point: the Gemma-3-1B PORTB 1 token at
  +0.9%, on a tree whose RTL is not the model's pin. PORTB 2 / 4, NCH 8 at 60 cycles (the
  +2.7% at Timing.dram_latency 60 is a sensitivity, not a model of the channels), the VLA shape
  and the ternary formats are uncalibrated in it until the extensions of step 4 are graded. LPN
  rejects HALT CHAIN, WAITC, WAITW, DSTEP / STREAM and multi-slice programs.
- The VisualSim models are prepared, never parsed by VisualSim's own parser or VM compiler:
  the request / return protocol follows the installed examples and the prior prepared models
  and may need changes at the first run.
- The HBM composite exposes no refresh; the 450 MHz bridge domain, the bursts' formation and
  the clock crossings are not in either model (listed per model in `not_modelled`).
- The trace's phases are per decode token at one position; prefill and the KV cache's
  growth are not covered yet.
