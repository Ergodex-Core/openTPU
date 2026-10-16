# The LPN tier: MJChku/LPNDialect's model of this slice, in the loop

Status (2026-10-16): built on this Mac, upstream's GEMM / MLP rows reproduced to the cycle on
Verilator 5.048, `cli.py lpn-run` writes a receipt for this loop's Gemma-3-1B decode token and
grades it against the Verilator receipts (+0.885% against the wide-port receipt and +0.902%
against the pinned RTL on the same bundle at the model's validated DRAM latency, +2.71% at the
receipt's 60 cycles; all inside upstream's 5% gate; the wide-port comparisons are **outside the
model's validated configuration**, so recorded, not a calibration). Every number an LPN run produces is labelled
`lpn-estimate`; `measured` stays for Verilator and the board.

The rule of the loop (docs/hwloop.md): models first, Verilator for what the models favour, Vivado
only in the nightly. Nothing here launches an instance, programs a card or pays for anything.

## 1. Bring-up on this Mac

### Toolchain (receipt: `tools/hwloop/receipts/2026-10-16/lpn_toolchain/toolchain.json`)

| need | what upstream says | what this Mac has | what was done |
|---|---|---|---|
| LLVM / MLIR 20 development packages | Ubuntu `llvm-20`, `LLVM_DIR` / `MLIR_DIR` | Homebrew at `/usr/local` is the **x86_64 (Rosetta) Homebrew** with `llvm` 23.1 and `llvm@21`; no 20. `brew install llvm@20` has **no Intel bottle any more** ("Homebrew no longer builds bottles for this configuration") and started compiling LLVM from source under Rosetta: stopped after two minutes (hours of all 12 cores) | **conda-forge, osx-arm64, through the existing `~/miniconda3`**: `conda create -p lpn-scratch/llvm20-env -c conda-forge --override-channels llvmdev=20.1.8 mlir=20.1.8 clangdev=20.1.8 clang=20.1.8 clangxx=20.1.8 llvm-tools=20.1.8 lld=20.1.8 cmake ninja` (1.6 GB, ~6 min). Nothing installed with brew. |
| `opt-20`, `llc-20`, `clang++-20` on PATH | Debian's versioned names; `lpn-compile` hardcodes them (`findProgramByName`) | conda names them `opt`, `llc`, `clang++` | `lpn-scratch/llvm20-bin/`: two symlinks and two `sh` wrappers (`clang++-20`, `clang-20`) that export `SDKROOT=$(xcrun --show-sdk-path)` so conda's clang finds the Apple SDK |
| Clang 20 to build the project | | Xcode's clang would also do, but the conda clang keeps one libc++ | `-DCMAKE_C_COMPILER=$E/bin/clang -DCMAKE_CXX_COMPILER=$E/bin/clang++ -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)` |
| Verilator 5.052 (5.020 fails to elaborate the pin) | | **5.048** at `/usr/local/bin/verilator` | 5.048 elaborates the pinned RTL: `tb_top` of the benchmark configuration built in 2:51 wall (`-O3`), the Gemma configuration (512 MiB behavioral DRAM) too; **5.052 was not needed**. No `verilator-large-memory.patch` either (that one is for Qwen's 2 GiB image). |
| Python with NumPy | upstream's `.venv` | `~/ergodex-03/bonsai2-venv` (3.14.5 arm64, numpy 2.5.4) | used as `OTPU_PYTHON` |

Build (from outside the clone; the clone is downloaded code, read before running: `build.sh` is
a two-line `cmake -S . -B build`, `run.sh` the network example, the observer patch 3 hunks in
`opentpu/rtlsim.py`, `sim/verilator/tb_top.sv`, `sim/verilator/lpn_wall.cpp`, no DUT change):

```sh
S=~/ergodex-03/lpn-scratch; E=$S/llvm20-env
git clone https://github.com/MJChku/LPNDialect $S/LPNDialect
git -C $S/LPNDialect checkout -b qwen-validation origin/qwen-validation   # b06f42d9, 2026-10-16: the Qwen3-1.7B rows, ARCHITECTURE.md, matrix_engines
export SDKROOT=$(xcrun --show-sdk-path); export PATH=$S/llvm20-bin:$E/bin:$PATH
$E/bin/cmake -S $S/LPNDialect -B $S/lpn-build -G Ninja -DLLVM_DIR=$E/lib/cmake/llvm -DMLIR_DIR=$E/lib/cmake/mlir \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$E/bin/clang -DCMAKE_CXX_COMPILER=$E/bin/clang++ -DCMAKE_OSX_SYSROOT=$SDKROOT
$E/bin/cmake --build $S/lpn-build -j 8          # 66 s; one linker warning (x86_64 /usr/local/lib/libzstd.dylib ignored)
$E/bin/ctest --test-dir $S/lpn-build --output-on-failure    # 13 / 13 (lpn-elf-workload wants clang-20 on PATH too)
```

Upstream's branches all pin the same `REVISION f237dab2`; `origin/main` (f099cbf, 2026-10-15)
has the GEMM / MLP / SmolLM2 material, `qwen-validation` adds the Qwen3-1.7B campaign,
`ARCHITECTURE.md`, `components.py` with `Timing.matrix_engines`, `replay_e2e.py`; `fpga-dse`
(3 commits further) adds a weight-free Qwen architecture sweep (`FPGA_DSE.md`: narrow arrays,
parallel output-row lanes; upstream calls its evolved-architecture timing unvalidated).

The pin, in its own worktree, never one of the active ones:

```sh
git -C ~/ergodex-03/openTPU worktree add ~/ergodex-03/lpn-scratch/openTPU-lpn-pin f237dab2f196855a59a083b1454786df8f01291f --detach
git -C ~/ergodex-03/lpn-scratch/openTPU-lpn-pin apply ~/ergodex-03/lpn-scratch/LPNDialect/examples/opentpu/verilator-observer.patch
ln -s ~/ergodex-03/openTPU/models ~/ergodex-03/lpn-scratch/openTPU-lpn-pin/models
```

`f237dab2` is FeSens/openTPU `main` of 2026-10-14 (the commit is in this repository: origin has
it; 60 commits past this Mac's `main` 7e42a626, docs/asic.md on top of the port registers). It
is **not** an ancestor of the wide-port tree `a9acb2b0` the loop froze (see 3, staleness).

### The reproduced table (Verilator 5.048, this Mac, `--batch-rows 8 --runs 3 --build-jobs 8`)

```sh
cd $S; export PATH=$S/llvm20-bin:$PATH; L=$S/LPNDialect
PYTHONPATH=$L PYTHONDONTWRITEBYTECODE=1 OTPU_TARGET=ypcb-00338 ~/ergodex-03/bonsai2-venv/bin/python -m examples.opentpu.benchmark \
    --opentpu-root $S/openTPU-lpn-pin --output $S/runs/upstream-gemm --workload gemm --batch-rows 8 --runs 3 --build-jobs 8 \
    --lpn-compiler $S/lpn-build/tools/lpn-compile/lpn-compile          # and --workload mlp --output $S/runs/upstream-mlp
PYTHONPATH=$L PYTHONDONTWRITEBYTECODE=1 OTPU_TARGET=ypcb-00338 ~/ergodex-03/bonsai2-venv/bin/python $L/test/test_opentpu.py \
    --opentpu-root $S/openTPU-lpn-pin --tool $S/lpn-build/tools/lpn-compile/lpn-compile      # 13 tests OK, 203 s
```

| workload | RTL cycles | LPN cycles | error | upstream (5.052, icdslab2) | RTL core wall | LPN core wall | core speedup | single-use speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| seven 4 x 4096 x 512 int8 GEMMs (58.7 M MACs) | **119,723** | **119,674** | **-0.041%** | 119,723 / 119,674 / -0.041% | 11.35 s | 18.7 ms | 608x | 12.8x | DRAM + TMEM bit-exact vs the ISA simulator, identical command PCs, 0 rollbacks, float ref 0.79% |
| compiled SwiGLU MLP, M 4, H = F = 1024 (12.6 M MACs) | **27,673** | **27,629** | **-0.159%** | 27,673 / 27,629 / -0.159% | 4.82 s | 5.4 ms | 886x | 4.5x | same; float ref 1.13% |

Identical cycle counts on both engines to upstream's rows (`receipts/2026-10-16/lpn_upstream_gemm/comparison.json`,
`lpn_upstream_mlp/comparison.json`: every wall sample, the Verilator flags, the binary hashes).
The wall times differ from upstream's (their RTL 5.46 s / 1.32 s, LPN 21.4 / 24.7 ms): a
different host; the ratios are this Mac's. The held-out, million-cycle and 3072 x 8192 MLP rows
and the SmolLM2 / Qwen3-1.7B steps were not re-run (hours of Verilator each; upstream's
reports for them are in the clone's `examples/opentpu/validation/`).

## 2. The scoreboard tier: `cli.py lpn-run` (`tools/hwloop/lpn_run.py`)

Same inputs as `run-rtl`: the frozen export and the workload arguments (`--model gemma3 --layers 2
--pos 9 --wformat int8 --head-format int8 --portb 1 --target f2-hbm --nch 8 --lat 60 --mhz 200`).

```sh
python tools/hwloop/cli.py lpn-run --tree ~/ergodex-03/hwloop-scratch/opentpu-a9acb2b0 --out RUN --layers 2 --pos 9 --portb 1 \
    --nch 8 --lat 60 --mhz 200 --dram-latency 8 --batch-rows 8 --uncalibrated \
    --compare tools/hwloop/receipts/2026-10-16/opentpu_portb1 [--rtl-pin]
python tools/hwloop/cli.py check-stale RUN --tree ~/ergodex-03/hwloop-scratch/opentpu-a9acb2b0     # exit 2: the RTL moved, the files listed
python tools/hwloop/cli.py report --baseline RUN --out RUN/report                                   # the row labelled lpn-estimate
```

Steps and files (all under `RUN/lpn/`):

1. `manifest.json`: the LPN commit, branch, `lpn-compile`'s SHA-256, the two patches' SHA-256 and
   whether the observer patch is applied on the pin tree, the pin (`f237dab2`), and the **RTL
   fingerprint of the pinned tree** (`vslib.fingerprint_tree` over `rtl/**`, `sim/verilator/*`,
   `opentpu/rtlsim.py`, `opentpu/profile.py`, `tools/perf_qwen.py`, the VisualSim models' globs).
2. `staleness.json`: that fingerprint against the export. Stale -> `REFUSED_STALE_MODEL` with the
   changed / added / removed files, unless `--uncalibrated`, which runs and labels the result
   `outside validated configuration: RTL moved (...)`.
3. `input/`: `lpn/inputs.py` **on the export** (its compiler and ISA simulator): the decode step's
   program and DRAM image exactly as `perf_qwen` builds them (`board_config` under the run's
   `OTPU_*`, `--portb` setting PAIR, `compile_step(pos)`, IMEM grown as perf_qwen does), written as
   `prog_0.hex` / `dram_0.bin` (rtlsim's hex truncation, big-endian words) and `input.json` with
   the pin's `Config` fields (`PORTB`, `T2` recorded in `workload.json` and left out; `SEQ_WC` forced
   False if the target had it: the programs do not change with it). `workload.json` also holds the
   export's ISA simulator result (DRAM / TMEM SHA-256), the per-opcode tally, the footprints' DRAM
   bytes, the MM weight chunks, and any of the ISA features upstream rejects (the run is refused
   if the program uses one).
4. `model.mlir`, `opentpu`: `lpn/emit.py` with the clone on `PYTHONPATH` and the pin as
   `--opentpu-root` (upstream's `workload.load` requires `REVISION == f237dab2`; it decodes and
   executes the program with the pin's ISA simulator). The frontend's DRAM / TMEM hashes must equal
   the export's: the two trees' ISA simulators agree on the program (`numerical_check.match`).
   `Timing(dram_latency, dram_bytes_per_cycle, batch_rows)` is the driver's choice; upstream's
   validated values are the defaults.
5. `native.log`: `opentpu --stats --wall-start=begin --wall-end=complete`; cycles, 0 rollbacks,
   `program_completed: 1`.
6. `trace.json`: the `run-rtl` / `report` record shape (`derived.cycles`, `dram_bytes`,
   `portb_utilization` = MM weight chunks / cycles / PORTB, `device_tok_s`, ...), `label:
   lpn-estimate`, `calibration`, the staleness summary; `compare.json` per `--compare` receipt;
   `rtl-pin/` with `--rtl-pin` (below).

### The Gemma-3-1B decode token (2 layers + the 262K-row head, pos 9, int8 / int8 head)

Receipts: `tools/hwloop/receipts/2026-10-16/lpn_gemma3_portb1/` (Timing.dram_latency 8) and
`lpn_gemma3_portb1_lat60/` (60). The bundle: 276 static instructions (65 MM, 147 VOP, 15 QACT,
10 LD, 32 ST, 2 QST, LI / ADDI / LOOP / HALT), 475 dynamic commands, image 372 MB (DRAM_BYTES
512 MiB), none of the rejected features; the compact net has 14 transitions and 68 places. Wall:
inputs 20 s (the export's compile + ISA simulator 1.3 s), frontend 6.2 s, model 0.5 s, compile
0.4 s, native **1.15 s** (91,923 firings) against the receipt's **37 min** of Verilator.

| point | label | cycles | vs the PORTB 1 Verilator receipt (2,798,478, `measured`) | port B util (MXU chunks / cycles) | DRAM bytes | configuration |
|---|---|---|---|---|---|---|
| LPN, Timing.dram_latency 8, 128 B/cycle (the validated memory) | lpn-estimate | **2,823,232** | **+0.885%**, inside the 5% gate | 0.984 (RTL 0.996) | 362,063,442 (footprints; RTL beats x 64 = 362,179,072, -0.03%) | outside validated: RTL moved (16 changed / 14 added fingerprinted files), receipt on NCH 8 at 60 cycles with the board uarch |
| LPN, Timing.dram_latency 60 (the receipt's latency) | lpn-estimate | **2,874,213** | **+2.706%**, inside the gate | 0.967 | same | the same, plus a Timing value upstream never validated |
| pinned RTL on the same bundle (`--rtl-pin`: f237dab2 + patch, tb_top default uarch, behavioral DRAM lat 8, Verilator 5.048; 2:19 build + 7:50 run) | **measured** | **2,797,992** | LPN at lat 8 is **+0.902%** of it: the model's own validated configuration on this workload, the honest grade; the pinned RTL and the wide-port receipt are 486 cycles (0.017%) apart on this port-bound token | 0.993 | | validated (`lpn_gemma3_portb1/rtl-pin/rtl_pin.json`: DRAM + TMEM bit-exact vs the pin's ISA simulator, whose hashes equal the export's; 469 command PCs traced = the 475 dynamic commands less the 6 scalar ones) |
| PORTB 2 (the 1,643,595-cycle receipt) | none | `REFUSED_NOT_MODELLED` | not comparable: the model has one port-B credit, one chunk a cycle | | | outside the model (section 5) |

Where the configurations overlap: the program, the image and the MM weight chunks are identical
(2,779,176 chunks on both sides: the receipt's `portb_mxu_chunks`), the bytes within 0.03%. Where
they do not: the receipt is the **wide-port tree** (`a9acb2b0`: port B / port A / SW registers,
`otpu_native_dram` over **NCH 8** channels with 64 reads in flight each at **--lat 60**, the board
micro-architecture `rtlsim.BOARD_UARCH` with MXU_E 4 and the one-copy pumped TMEM), the model is
validated on the pin's testbench default (2 epilogue lanes, replicated TMEM, one behavioral port
at 8 cycles). The +0.9% / +2.7% say the model's slice timing carries over to that configuration
on this program; they are not a calibration of NCH 8 / lat 60 / the board uarch, and the
`calibration` field says so. The ISA tally's roofline (2,787,778 port-B chunks) is 0.4% under the
RTL; LPN at lat 8 is 0.9% over it: the model charges the row groups' drain and the QACT / VOP
tails the tally has not, slightly more than the RTL does.

Labels: `measured` = Verilator (the receipts, `--rtl-pin`); `lpn-estimate` = every LPN number;
`calibration: validated configuration` only when the export's RTL fingerprint equals the pin's
and the Timing is upstream's. `report.py` prints an LPN run's row with its label and
calibration; `report.compare`'s delta is labelled `lpn-estimate` when either side is one.

## 3. Staleness

`manifest.json`'s `rtl_fingerprint` is the pinned tree's (the VisualSim models carry the
export's); `check-stale RUN --tree EXPORT` (or `RUN/lpn`) recomputes the export's and refuses with
the list, exit 2, exactly as for a VisualSim model; `lpn-run` does the same before anything runs.
Between `f237dab2` + patch and wide-port `a9acb2b0` (`lpn_gemma3_portb1/staleness.json`):

- changed (16): `opentpu/profile.py`, `opentpu/rtlsim.py`, `rtl/asic/otpu_asic_top.sv`,
  `rtl/mem/otpu_actram.sv`, `rtl/mem/otpu_dram.sv`, `rtl/mem/otpu_native_dram.sv`,
  `rtl/mxu/otpu_mxu.sv`, `rtl/shell/otpu_board.sv`, `rtl/shell/otpu_ctrl.sv`, `rtl/top/otpu_pkg.sv`,
  `rtl/top/otpu_slice.sv`, `rtl/top/otpu_top.sv`, `rtl/vpu/otpu_quant.sv`,
  `sim/verilator/otpu_native_mem.sv`, `sim/verilator/tb_top.sv` (the patch is on the pin side),
  `tools/perf_qwen.py`
- added (14): `rtl/boards/f2/otpu_axil_cdc.sv`, `otpu_f2_hbm_host.sv`, `otpu_f2_hbm_mem.sv`,
  `otpu_f2_hbm_nch.sv`, `otpu_f2_hbm_port.sv`, `otpu_f2_hbm_sys.sv`, `otpu_f2_mem.sv`,
  `otpu_f2_nch.sv`, `otpu_f2_sys.sv`; `sim/verilator/tb_f2axil.sv`, `tb_f2hbm.sv`, `tb_f2mem.sv`,
  `tb_f2nch_protocol.sv`, `tb_native_nch.sv`
- removed: none

So every LPN number for the wide-port tree is `--uncalibrated` by construction until the model is
re-validated on it (a `--rtl-pin`-style run of the export's own RTL at the pin's configuration,
or the model's pin moved to the export: upstream's `REVISION`, which `workload.load` enforces).

## 4. Nightly

`cli.py nightly plan` prints the LPN points first ("models first, on this Mac"), then the Verilator
co-sim points, then the build host; the drift rule: an LPN point more than 5% from the Verilator
receipt it is graded on is uncalibrated, its projection leaves the decision, the model extension
goes on the list, the Verilator receipt stands; a configuration the model does not have (PORTB
2 / 4, NCH 8 at 60 cycles, the VLA shape) has no LPN point and the plan says so. The queued
`portb2-gemma3` entry carries its LPN points (`baseline` +0.88% / +2.71%, `candidate` none).
Plan only; nothing scheduled, nothing activated.

## 5. Extending the model: PORTB 2 / 4 / 8, NCH 16, the f2-vla shape (not implemented)

Upstream's rule (`examples/opentpu/ARCHITECTURE.md`): the hypothetical engines share the modelled
ACT / TMEM and DRAM ports; **extra physical memory ports, banks or a different arbitration need
corresponding resource changes**, and matching the existing tests establishes nothing for a new
RTL architecture: validate representative modified-RTL cases. What each of ours needs, against
`model.py` / `components.py` as they are:

**PORTB 2 / 4 / 8 (wide port; receipts 1,643,595 and 1,308,691 cycles for 2 / 4, none for 8).**
The model's port B is one credit place (`DRAM.b`, one token) whose service per prefetch is
`ceil(width x chunks x D / dram_bytes_per_cycle)` with `dram_bytes_per_cycle <= cfg.D` asserted;
the MXU's row service is `max(steps x groups, drain)` with `steps = chunks` for 4-bit PAIR and
`kb` otherwise. Changes: (a) a `Timing.port_chunks_per_cycle` (PORTB) lifting the assertion, the
port credit's service in chunks / PORTB, and the 128-chunk FIFO filling PORTB chunks a cycle;
(b) the MXU's advance for int8 PAIR at PORTB 2 (the RTL takes 5 advances for a 9-chunk row:
`ceil(kb / 2)` plus the odd chunk, and QUAD at PORTB 4), the roofline moving from the port to the
MXU (the receipt's: 93.5% of a 1,536,296-cycle MXU roofline); (c) the two-part wide reads (7.0%
of the PORTB 2 receipt's 1,242,880 wide reads cross a chunk pair: an extra port cycle each, from
the footprints' addresses); (d) the DMA / QST traffic sharing the wider port (upstream's
`fpga-dse` branch already shares one widened memory credit across prefetch and DMA / QST). Then
re-validate: the PORTB 2 and 4 receipts exist; PORTB 8 needs its Verilator run first (the loop's
`run-rtl --portb 8`, ~30 min) before any LPN number for it means anything.

**NCH 8 at 60 cycles (the receipts' memory) and NCH 16 (`f2-hbm16`).** The model's DRAM is one
port with `dram_latency` and `dram_bytes_per_cycle`: no channels, no per-channel outstanding
reads, no address hash. The 60-cycle run above over-charges the latency (+2.7% against +0.9%)
because the adapter keeps up to 203 reads in flight (the trace's estimate) and the model's
prefetch credit does not. Changes: per-channel credit places (NCH x 64 outstanding beats, the
address -> channel hash `CHASH` from the footprints' addresses, 64-byte beats), a return pipeline
of `dram_latency`, the DMA's writes on the same credits; the per-channel beat counts and the
outstanding-reads estimate of `mem_windows.csv` are the validation targets, plus the cycles.
NCH 16 is then a parameter, but the f2-hbm16 target has no Verilator receipt yet (one `run-rtl`
with `OTPU_TARGET=f2-hbm16`), and a channel's bandwidth share is the bridge's (450 MHz, 8-beat
bursts), which neither model has.

**The f2-vla shape (MCOLS 32 physical / 64 logical packed, LANES 64, ACT_ROWS 255).** In the
model MCOLS enters `groups = ceil(m / MCOLS)` and the pipeline tail `min(D / 8, chain) + MCOLS +
14`; LANES enters QACT's `ceil(kb x D / LANES)`, QST's and VOP's row costs (with a `LANES == 8`
special case for the SE's composites: EXP2 / RECIP / RSQRT / LOG2 as 25-cycle loops on eight
lanes), the drain width `min(epilogue_lanes, LANES / gcd(stride, LANES))` and the GATHER cost;
ACT_ROWS enters only through `m` (replay = more groups). Changes: (a) packed columns: two logical
rows per physical column per advance, so `groups = ceil(m / 64)` on a 32-column array with the
MXU's advance and epilogue rewritten for the pair (and `epilogue_lanes` = the VLA build's
MXU_E); (b) LANES 64: the composite table for the wide SE (no 8-lane loop), TMEM banks = 64 and
the TMEM arbitration the model only approximates, which the vision prefix's VOP-heavy phases
stress; (c) ACT_ROWS 255: the ACT RAM replay's re-stream per group is validated only to 16 rows
(upstream's held-out 8-row replay case) and the QACT fill of 255 rows is new; (d) a vision
prefix is a prompt (rows > 1), so `compile_rows` / tiled prefill inputs, not `compile_step`.
Validation targets: the `vla` worktree's Verilator utilization numbers for the 558-968-row prefix
(docs/designs/vla-1hz-packed-slice.md), bit-exact against the ISA simulator as everything else.
Until then: no LPN number for the VLA shape. Upstream's `fpga-dse` "parallel output-row lanes"
and narrow arrays are the nearest prior art and, in upstream's words, unvalidated against any
evolved RTL.

Also still rejected by the timing backend (for the LLM track's later workloads): HALT CHAIN
(the generate loop between buckets), WAITC (the card's sequencer wait entries), DSTEP / STREAM
(Qwen3.5), WAITW, multi-slice collectives; `lpn-run` refuses a program that uses one
(`REFUSED_REJECTED_ISA`) and it stays on Verilator.

## Files

- `tools/hwloop/lpn_run.py`: the tier (`cli.py lpn-run`); `tools/hwloop/lpn/inputs.py` (export
  context), `lpn/emit.py` (LPNDialect on PYTHONPATH, pin as root), `lpn/rtl_pin.py` (the pinned
  RTL on a bundle); `lpn/__init__.py`: the pin, the label, the gate, the validated configuration.
- Receipts: `tools/hwloop/receipts/2026-10-16/lpn_toolchain/`, `lpn_upstream_gemm/`,
  `lpn_upstream_mlp/`, `lpn_gemma3_portb1/`, `lpn_gemma3_portb1_lat60/`.
- Scratch (not in git): `~/ergodex-03/lpn-scratch/{LPNDialect, lpn-build, llvm20-env, llvm20-bin,
  openTPU-lpn-pin, runs, logs}`; the 372 MB DRAM images stay there.
- Tests: `tests/test_hwloop.py` (the native-log parser, the record and its compare, the staleness
  refusal, `check-stale` on an LPN directory, the nightly's LPN lines).
