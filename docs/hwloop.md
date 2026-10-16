# The hardware loop: Verilator -> VisualSim -> nightly FPGA

Status (2026-10-16): the tools in `tools/hwloop/` run end to end on this Mac up to the
VisualSim execution, which is **waiting for the seat** (the user's Architect GUI holds the
single license; the runner refuses to start a second instance and never stops the GUI). The
nightly FPGA step is **prepared, not activated**. One calibrated workload (Gemma-3-1B int8
decode on the F2 HBM configuration) is the only workload covered; broadening comes after the
first VisualSim run has been compared with the receipts.

Rule of the loop (the user's): hardware changes are validated in Verilator first; Vivado runs
only in the nightly. Nothing in `tools/hwloop/` launches an instance, programs the card,
submits an AFI or starts a schedule.

## What exists (inventory, 2026-10-16)

| Piece | Where | State |
|---|---|---|
| ISA cost model / profile of a run | `opentpu/profile.py` (`Profile`, `parse`, `ddr3_plusargs`) | main; the per-unit busy, phases, MXU gaps |
| RTL timing run of one decode token with real weights, the board's memory path | `tools/perf_qwen.py` (`--json`, `--plus +nat_trace=FILE`, `--check` against the ISA simulator) | branches f2-hbm / wide-port / ternary-bonsai (not on main) |
| Verilator build / run | `opentpu/rtlsim.py` (`MEMORY`: `OTPU_LAT`, `OTPU_NCH`, `OTPU_STALL`, `OTPU_BW`, `OTPU_LDC`) | main (NCH on the F2 branches) |
| Board memory model with a command trace | `sim/verilator/otpu_native_mem.sv` (`+nat_trace`, `+axi_lat`, `+axi_dram=1` bank model) | main |
| Card trace records | `opentpu/hwtrace.py` (`rtl/shell/otpu_trace.sv`) | f2-hbm and later |
| F2 build | `boards/f2/build.py` (`--flow SynthCL/ImplCL/BuildAll --package`) | f2-hbm and later; Vivado on the build host only |
| Remote synthesis hosts | `tools/tourney/remote.py` (`OTPU_BUILD_HOSTS`) | f2-hbm and later |
| Serving bench on the card | `tools/f2_bench.py --json` (TTFT, prefill tok/s = prompt/TTFT, decode tok/s) | ternary-bonsai |
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

Not a generic TPU: the VisualSim model below is built from this slice's structure (port B at
D = 128 bytes a core cycle into the MXU's chunk FIFO, port A's scale lines, the DMA's writes,
the native channel adapter with NCH channels and its per-channel outstanding reads, the
bridge's 8-beat bursts) and from this RTL's trace.

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

## The per-change loop

```
# 0. freeze: the pinned source tree (git archive of one commit: never another worktree),
#    its RTL fingerprint, the checkpoint, quantization, position, memory placement
python tools/hwloop/cli.py freeze --rev a9acb2b0 --out /scratch/opentpu-a9acb2b0 --models /path/to/models \
    --layers 2 --target f2-hbm --nch 8 --lat 60 --mhz 200
# 1. the candidate change: a commit on top (here PORTB 2 is a parameter of the same tree)
# 2. static checks + unit tests + Verilator regressions against the ISA simulator (the
#    independent numerical reference): the existing suites, e.g.
OTPU_NCH=8 python -m pytest -q tests/test_rtl.py -k "board_memory_path or native_memory_path"
python -m pytest -q tests/test_wide_port_rtl.py
# 3. the memory-inclusive timing run + trace, baseline and candidate (perf_qwen --check adds
#    the ISA comparison of the same token)
OTPU_PYTHON=~/bonsai2-venv/bin/python python tools/hwloop/cli.py run-rtl --tree /scratch/opentpu-a9acb2b0 \
    --out runs/base --layers 2 --portb 1 --nch 8 --lat 60 --mhz 200
python tools/hwloop/cli.py run-rtl --tree /scratch/opentpu-a9acb2b0 --out runs/cand --layers 2 --portb 2 ...
# 4. the VisualSim models (one per memory profile), fingerprinted against the tree
python tools/hwloop/cli.py build-vs runs/base --profile f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0
python tools/hwloop/cli.py build-vs runs/cand --profile f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0 --variant candidate
# 5. the staleness gate and the batch run (prepared without --execute; refused while the GUI
#    or another batch holds the seat; the seat lock serializes every repository's runs)
python tools/hwloop/cli.py check-stale runs/base/vs/f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0
python tools/hwloop/cli.py vs-run runs/base/vs/f2-hbm-nch8-rtl --tree /scratch/opentpu-a9acb2b0 [--execute]
# 6. the report (JSON, CSV, Markdown; every number labelled) and the nightly queue
python tools/hwloop/cli.py report --baseline runs/base --candidate runs/cand --profile f2-hbm-burst200 --out runs/report
python tools/hwloop/cli.py nightly add --name portb2 --commit <sha> --target f2-hbm --run runs/cand
# 7. the canonical copy (model XML, workload inputs, evidence, results, reports apart)
python tools/hwloop/cli.py install runs/base --dest ~/VisualSim/VS_AR/ergodex_npu/hwloop/opentpu/f2-hbm-nch8-rtl/baseline --report runs/report
```

`run-rtl` records the exact command and environment (`command.sh`, `run_receipt.json`);
`extract` turns `perf.log` + `perf.jsonl` + `nat_trace.txt` into `trace.json` (cycles, bytes,
per-channel reads / writes, port-B utilization, MXU starved / blocked, the phases, an
outstanding-reads estimate from the issue times and the latency, idle gaps) and
`mem_windows.csv` (reads and writes per channel per window of 1024 core cycles).

### Trace content, memory-inclusive

From `perf_qwen` (the RTL's own counters): cycles, the core-port roofline (port B chunks,
port A, MXU), per-channel DRAM read / write beats and port-A reads, MXU starved (chunk FIFO
empty) and blocked cycles, the units' busy, port B busy (MXU vs DMA chunks), the phases
(cycles, bytes, roofline, VPU / MXU busy), the MXU gaps and what they waited on. From
`+nat_trace`: every command the channels took (cycle, channel, read / write, beat). What the
trace does not carry: the bridge's bursts (formed in the 450 MHz domain, not in
`otpu_native_mem`), the HBM controller's bank state and refresh. The DDR3 card has the
calibrated bank model (`--ddr 1066`, `--ldc`) for those.

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
- Fresh baseline (PORTB 1) and candidate (PORTB 2) runs with `+nat_trace`: see
  `VS_AR/ergodex_npu/hwloop/opentpu/` (`workload/trace.json`, `reports/report.md`) and the
  PR's description for the numbers when the runs finish; the documented values are 2,798,478
  and 1,643,595 cycles (1.70x, `docs/wide_port.md`).
- VisualSim: every model prepared (snapshot, exact batch command, static validation), none
  executed: the GUI holds the seat. Status in every receipt: `PREPARED_NOT_EXECUTED`,
  label `waiting`.
- Nightly: `tools/hwloop/nightly/queue.json` + `cli.py nightly plan` (not activated).

## Staged plan

1. (done) Tools, profiles, one workload's receipts, prepared models, tests, this doc.
2. First VisualSim execution when the seat frees: `vs-run --execute` on the baseline
   `f2-hbm-nch8-rtl` model; fix the VM / protocol errors the vendor parser reports; compare
   the replay's cycles with the RTL's (the RTL is the reference for this profile: the memory
   model should reproduce its 99.4% port utilization within a stated error); then the
   candidate, then `f2-hbm-burst200` against the board's 22.3 tok/s (89% of the RTL's
   port-bound rate: the gap the memory model has to explain, not fit).
3. Validation split: calibrate on Gemma-3-1B short-prompt decode only; hold out Gemma long
   prompt, Bonsai-2 2B / 4B (int4, t2), Qwen3-0.6B, the DDR image's receipts; report per
   workload absolute / relative error, mean and worst case.
4. A whole-model run (`--layers 0`) on the build host's Verilator, overnight, for the direct
   tok/s comparison; the card's hardware trace of a token.
5. Nightly: with approval, the queued candidate's SynthCL / BuildAll on the build host, the
   AFI, the matched bench against the known-good image, the errors fed back into the
   profiles' `measured` blocks.
6. Broaden: prefill (tiled, chunk-64) and TTFT as their own workloads; multi-slice; the
   DDR3 card through the LDC co-simulation.

## Limitations

- A VisualSim run that completes proves the model runs, not that the RTL is correct or that
  any number is a board number. Numerical correctness is `perf_qwen --check`, the ISA
  simulator and the card's token-exact checks, nowhere else.
- The models are prepared, never parsed by VisualSim's own parser or VM compiler: the
  request / return protocol follows the installed examples and the prior prepared models and
  may need changes at the first run.
- The HBM composite exposes no refresh; the 450 MHz bridge domain, the bursts' formation and
  the clock crossings are not in the model (listed per model in `not_modelled`).
- The trace's phases are per decode token at one position; prefill and the KV cache's
  growth are not covered yet.
