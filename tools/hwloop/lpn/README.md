# The LPN tier: MJChku/LPNDialect's model of this slice, in the loop

Status (2026-10-17): built on this Mac, upstream's GEMM / MLP rows reproduced to the cycle on
Verilator 5.048, `cli.py lpn-run` writes a receipt for this loop's Gemma-3-1B decode token and
grades it against the Verilator receipts. Two models: upstream's (PORTB 1 on the pinned tree's
frontend: +0.885% against the wide-port receipt, +0.902% against the pinned RTL on the same
bundle, the model's validated configuration) and **the wide model** (`tools/hwloop/lpn/wide.py`,
section 5: port B as a credit of PORTB chunks a cycle with the two-part reads counted from
the addresses, the row lanes, the NCH channels), **graded on the wide-port export 96a90c2f
against fresh Verilator receipts: PORTB 2 +1.23%, PORTB 4 +0.96%, NCH 8 at 60 cycles +0.83%,
int8 RL 4 +4.11%, int4 RL 4 on MCOLS 8 +4.16%, all inside upstream's 5% gate** and accepted
only for those configurations on that export (the registry `tools/hwloop/lpn/validated.json`;
anything else `REFUSED_NOT_MODELLED` / `REFUSED_STALE_MODEL`). Every number an LPN run produces
is labelled `lpn-estimate`; `measured` stays for Verilator and the board.

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

The wide model (section 5) takes the same arguments plus `--portb 2 | 4`, `--rowlanes [--mcols
8] [--wformat int4 --head-format int4]`, or `--channels` (PORTB 1 on the NCH channels at
`--lat`), and its own rule: `--compare RECEIPT --grade` grades the configuration on the export
and writes the point into `tools/hwloop/lpn/validated.json`; afterwards `lpn-run` accepts that
configuration on that export only.

```sh
python tools/hwloop/cli.py lpn-run --tree ~/ergodex-03/hwloop-scratch/opentpu-96a90c2f --out RUN --layers 2 --portb 2 --nch 8 --lat 60 \
    --compare tools/hwloop/receipts/2026-10-17/opentpu96_portb2            # graded: +1.229%
python tools/hwloop/cli.py lpn-run --tree ... --out RUN --layers 4 --portb 4 --rowlanes --mcols 8 --wformat int4 --head-format int4 \
    --nch 8 --lat 60 --compare tools/hwloop/receipts/2026-10-17/opentpu96_rl4_int4_mcols8      # graded: +4.161%
python tools/hwloop/cli.py lpn-run --tree ... --out RUN --layers 2 --portb 8                # REFUSED_NOT_MODELLED (no graded point)
python tools/hwloop/cli.py lpn-run --tree ~/ergodex-03/hwloop-scratch/opentpu-a9acb2b0 --out RUN --layers 2 --portb 2   # REFUSED_STALE_MODEL (graded on 96a90c2f)
```

The wide run's `lpn/` adds `memory.json` (the static tally: wide and two-part reads, the
per-channel beats, the port B / MXU / port A / channel rooflines) and the manifest's
`rtl_fingerprint` is the **graded export's** (`validated_rtl_pin` its sha), so `check-stale`
refuses any other tree with the files listed.

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
| PORTB 2 (the 1,643,595-cycle receipt) | none (this model) | `REFUSED_NOT_MODELLED` | not comparable: one port-B credit, one chunk a cycle; the wide model's row is below | | | the wide model (section 5) |

### The wide model, graded (2026-10-17, the 96a90c2f export, f2-hbm NCH 8 at --lat 60, the board uarch)

Fresh Verilator receipts on the export (`receipts/2026-10-17/opentpu96_*`, Verilator 5.048,
6-8 minutes each on the shared machine): PORTB 4 **1,308,691** and PORTB 2 **1,643,595** (2
layers + head, identical to docs/wide_port.md's and to the a9acb2b0 receipt), int8 RL 4 at
PORTB 4 **949,485** and int4 RL 4 at PORTB 4 / MCOLS 8 **579,306** (4 layers + head, identical to
docs/wide_port.md's row-lane table), PORTB 1 **2,798,478** (identical to the a9acb2b0 receipt).
The LPN receipts: `receipts/2026-10-17/lpn96_*` (manifest, staleness, emit, memory, native
log, trace, compare, the bundle's workload / input manifests).

| configuration (Gemma-3-1B, pos 9) | Verilator (`measured`) | LPN (`lpn-estimate`) | error | what binds (the model's rooflines: port B / MXU / port A / channel) | two-part reads (model = receipt) | per-channel beats (model / receipt ch0) | wall (inputs / model / native) |
|---|---|---|---|---|---|---|---|
| int8, 2 layers + head, **PORTB 2** (PAIR) | 1,643,595 | **1,663,796** | **+1.229%** | port B 1,623,915 (1,536,296 requests + 87,619 two-part) / MXU 1,536,296 / port A 1,535,784 / channel 814,569 | 87,619 of 1,242,880 (7.05%) | 704,490 / 705,394 | 18 s / 9 s / 0.54 s |
| int8, 2 layers + head, **PORTB 4** (QUAD) | 1,308,691 | **1,321,225** | **+0.958%** | port B 1,283,555 (925,224 + 358,331 two-part) / MXU 925,224 / port A 924,712 / channel 814,569 | 358,331 of 631,808 (56.7%) | 704,490 / 705,394 | 41 s / 17 s / 1.3 s |
| int8, 2 layers + head, **PORTB 1 on the NCH 8 channels at 60 cycles** (`--channels`) | 2,798,478 | **2,821,563** | **+0.825%** (the single-port model at Timing.dram_latency 60: +2.706%; at 8: +0.885%) | port B 2,779,176 / channel 814,569 | - | 704,490 / 705,394 | 20 s / 9 s / 0.5 s |
| int8, 4 layers + head, **PORTB 4, RL 4** (MCOLS 4) | 949,485 | **988,507** | **+4.110%** | **channel 938,100** (811,329 beats at 0.865 a cycle) / port B 800,592 / MXU 800,592 / port A 799,568 | 0 of 799,488 | 811,329 / 814,004 | 48 s / 26 s / 0.56 s |
| int4 / int4 head, 4 layers + head, **PORTB 4, RL 4, MCOLS 8** (the > 100 tok/s point) | 579,306 | **603,411** | **+4.161%** | **channel 547,792** (473,766 beats) / port B 533,672 (441,256 advances + 92,416 Sw lines) / port A 532,648 / MXU 441,256 | 0 of 440,232 | 473,766 / 477,228 | 54 s / 20 s / 0.10 s |

What the errors are made of. PORTB 2 / 4 are port-B bound in the RTL and in the model (the
requests plus the two-part reads, reproduced to the read); the +1% is the same non-MM tail
the PORTB 1 point carries (+0.9%). The two row-lane points are **channel-bound in the model**
(RD 64 B-read beats in flight a channel over a 74-cycle round trip: 0.865 beats a cycle a
channel, against the lanes' need of one a cycle at PORTB 4) and 4.1% high: the RTL's channels
sustain a little more than the model's rate (the receipt's int8 RL 4 is 84.5% of its issue
roofline, the model 81%; 0.853 beats a cycle a channel would land on it), i.e. the round trip
is about two cycles shorter than the 14 cycles of path assumed (`WideTiming.adapter_latency`,
derived from the adapter's structure, not fitted: each cycle of it is 1.3% of a channel-bound
point). Inside the gate, so recorded as graded and not tuned; a `--adapter-latency` sweep is
the sensitivity, and the honest next step is a Verilator receipt at NCH 16 (f2-hbm16), where
the channels stop binding and the issue roofline should show. The int4 RL 4 point also
carries the group's Sw line as an issue cycle (92,416 of them), which the RTL's port A
roofline counts the same way (534,712 against the model's 532,648 + the PAIR MMs).

Whole model (26 layers + head, `--layers 0`; no receipt in this repository, so `--uncalibrated`
and no registry point; `receipts/2026-10-17/lpn96_whole_rl4_*`): int4 RL 4 at PORTB 4 / MCOLS 8
**1,502,766 cycles = 133.1 tok/s, +1.95%** of docs/wide_port.md's measured 1,474,025 (135.7
tok/s; the model's MXU 1,065,924 advances and port A 1,299,524 lines are the tally's numbers
exactly); int8 RL 4 at PORTB 4 **2,444,351 = 81.8 tok/s, +2.10%** of the measured 2,393,990
(83.5). Wall: inputs 76 / 34 s (the 26-layer image and the ISA simulator), model 35 / 23 s,
native 0.6 / 0.8 s against the doc's 288 / 699 s of Verilator simulation (plus its build).

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

So every number of upstream's model for the wide-port tree is `--uncalibrated` by construction
until the model is re-validated on it (a `--rtl-pin`-style run of the export's own RTL at the
pin's configuration, or the model's pin moved to the export: upstream's `REVISION`, which
`workload.load` enforces).

**The wide model's rule** (`tools/hwloop/lpn/validated.json`): a configuration key (model,
layers, pos, formats, PORTB, row lanes, MCOLS, target, NCH, lat) is accepted only with a graded
point (an LPN run compared with a Verilator receipt: `--compare RECEIPT --grade` writes the
point, the export's sha and its RTL fingerprint over the same globs) and only on that export:
`REFUSED_NOT_MODELLED` without a point (PORTB 8, NCH 16, 4-bit QUAD, `--layers 0`, the VLA
shape ...), `REFUSED_STALE_MODEL` on another export with the changed / added / removed files
(the run's `manifest.json` carries the graded export's fingerprint and sha, so `check-stale
RUN/lpn --tree OTHER` refuses too). A point over the 5% gate stays in the registry labelled
uncalibrated with its suspected cause; `--uncalibrated` runs any configuration, labelled so.

## 4. Nightly

`cli.py nightly plan` prints the LPN points first ("models first, on this Mac"), then the Verilator
co-sim points, then the build host; the drift rule: an LPN point more than 5% from the Verilator
receipt it is graded on is uncalibrated, its projection leaves the decision, the model extension
goes on the list, the Verilator receipt stands; a configuration the model does not have (PORTB
8, NCH 16, 4-bit QUAD, the VLA shape) has no LPN point and the plan says so. The queued
`portb2-gemma3` entry carries its LPN points (`baseline` +0.88% / +0.83% with the channels,
`candidate` PORTB 2 +1.23%), `portb4-rl4-int4-gemma3` its (`candidate` int4 RL 4 +4.16% at 4
layers + head, int8 RL 4 +4.11%, and the whole-model estimates). Plan only; nothing scheduled,
nothing activated.

## 5. The wide model: PORTB 2 / 4, the row lanes, the NCH channels (`tools/hwloop/lpn/wide.py`, graded 2026-10-17)

Upstream's rule (`examples/opentpu/ARCHITECTURE.md`): the hypothetical engines share the modelled
ACT / TMEM and DRAM ports; **extra physical memory ports, banks or a different arbitration need
corresponding resource changes**, and matching the existing tests establishes nothing for a new
RTL architecture: validate representative modified-RTL cases. The wide model is those resource
changes, as a subclass of upstream's `CompactOpenTPU` in this repository (the clone stays at
`b06f42d9`, unpatched; the command window, the units, the dependency builder and the completion
banks are upstream's; the matrix engine and its per-command tables are ours), run on the
**export's own ISA simulator** (`wide.load_export`: upstream's pin f237dab2 refuses int8 PAIR; the
export knows PORTB, QUAD for every format, MM RL, QACT DUP8, MCOLS 8). The regression that keeps
it honest: at PORTB 1, fifo 128 entries, NCH 1 and the pin's Timing the net reduces to
upstream's and gives **2,823,232** on the Gemma bundle, the same cycle count.

**PORTB 2 / 4 as a port-B credit of PORTB chunks a cycle** (`WideTiming.port_chunks`). An MM row
is `requests` port-B requests (an advance's chunks: int8 PAIR 2, QUAD 4; 4-bit QUAD 2; the last
advance the chunks left) and `steps` MXU advances (`ceil(kb / reuse)`: 5 a 9-chunk row at PAIR, 3
at QUAD), so the roofline moves from the port to the MXU where the port is wider than the reuse.
The port credit is held one cycle a request **plus one cycle for every two-part request**: a
wide request whose chunks need one channel pair twice, counted exactly from the MM's addresses
with otpu_native_dram's hash (chunk m on pair `(m mod NP) ^ fold(m / NP)`, the XOR of the
higher log2(NP)-bit groups; a request's chunks sorted by pair, a repeat = two parts). On the
2-layer bundle: 1,242,880 wide reads and **87,619 two-part at PORTB 2, 631,808 and 358,331 at
PORTB 4: the receipts' NATIVEB counters to the read**, and the MXU rooflines 1,536,296 / 925,224
to the cycle. The fractional two-part share of a row is carried across the batches in the
request token (not rounded up a cycle a batch: that alone was +1.7%). The chunk FIFO's credit
is in entries (the target's FIFO_DEPTH: 1024 on f2-hbm, from `targets/<target>.toml`), an entry
an advance (a narrow MM on a wide build: a chunk). Port A: one line an advance. Not modelled:
the DMA's and QST's requests sharing the wider port beyond their one-chunk service (unchanged
from upstream: < 1% of a token's traffic). Error it costs: none visible at this precision (the
PORTB 2 / 4 points are +1.2% / +1.0%, the PORTB 1 point +0.9%).

**NCH 8 at 60 cycles: the channels** (`nch`, `channel_depth`, `adapter_latency`, `chash`). NCH
credit places, one a channel, each held for the beats the batch puts on it at the channel's
sustained rate: a chunk is two 64-byte beats, one on each channel of its pair (`chan_of`: the
pair, then the half by address bit 6 XOR the chunk index's parity); port A's lines cost the
distinct beats they touch (the adapter's APF runs and beat reuse: a sequential scale stream
fetches each beat once; the receipts' per-channel reads are within 0.3-0.7% of the count:
704,490 / 705,394 at PORTB 2, 811,329 / 814,004 at int8 RL 4). The rate is one beat a cycle, or
RD / round trip where the in-flight room is shorter than a read's round trip: RD = 64 B-read
beats a channel at NCH > 2 (`otpu_board` NRD), the round trip `--lat` + 14 cycles of path (the
slice's two port registers, BREG, the queue, the command, `otpu_native_mem`'s in-order return
jitter of 0..7 cycles, the response FIFO, the merge: structural, not fitted), 64 / 74 = 0.865
beats a cycle a channel. The chunk FIFO (1024 entries) hides the latency itself: the ready
FIFO's delay is the round trip. With this the PORTB 1 point at the receipt's memory falls from
+2.706% (the single port at Timing.dram_latency 60: the packet credit of a 128-entry FIFO
exposed the latency) to **+0.825%**. The channel rate is what binds the row-lane points (below).
Not modelled: the DMA's beats on the channels, the A runs' refills (one miss per 32 beats a
channel: ~0.2-0.4% of a decode token), the bridge's 450 MHz side and its bursts (the card's,
not the RTL run's), NCH 16 (a parameter, but no receipt: `f2-hbm16` is refused until one
exists).

**Row lanes** (MM RL, `rs` bits 3:2; `--rowlanes`). A group of L rows is one row of the model:
`ceil(kb / blocks-a-chunk)` advances of L aligned chunks (one request, one entry, one MXU step
each; **0 two-part by the same count, as the receipts' 0 of 799,488 / 440,232**), one port-A
line an advance plus the group's Sw line where the header and the advance's scale data do not
share a 32-byte line (`H + A > 32`: int4 L 4 with H = A = 32; 92,416 of them in the 4-layer
bundle), each an issue cycle without a chunk (port B's and the MXU's), the drain `ceil(L /
min(L, E))`, the group's chunks `L x advances` for the bytes (the last group padded, as the
image). MCOLS 8 and int4 are inside what upstream's model carries (`groups = ceil(m / MCOLS)`
with m = 1; `chunks = ceil(kb / 2)` for the 4-bit formats): no resource change beyond the
group-as-row. Not modelled: QACT DUP8's extra copies (the quantizer's cost is upstream's
`(rows + 1) chunks + 60`). The two points are +4.1% / +4.2%, channel-bound in the model
(section 2's table says what each cycle of path is worth).

**Still refused** (`REFUSED_NOT_MODELLED`, no graded point): PORTB 8 (the ISA has no 8-chunk
advance; RL 8 is int8 L 8 on f2-hbm16, which has no Verilator receipt), NCH 16 (`f2-hbm16`), 4-bit
QUAD on a wide port (the first-line-alone scale request is coded as an issue cycle and a line
but not graded), the whole model (`--layers 0`: estimates only, see the queue), any other
model / position / format, and the f2-vla shape (MCOLS 32 physical / 64 logical packed, LANES
64, ACT_ROWS 255: the packed columns, the wide SE's composites, the TMEM arbitration and the
255-row replay are resource changes not made; validation targets: the `vla` worktree's
Verilator utilization numbers for the 558-968-row prefix, docs/designs/vla-1hz-packed-slice.md).
Still rejected by the timing backend (`REFUSED_REJECTED_ISA`): HALT CHAIN (the generate loop
between buckets), WAITC, DSTEP / STREAM (Qwen3.5), WAITW, multi-slice collectives.

## Files

- `tools/hwloop/lpn_run.py`: the tier (`cli.py lpn-run`); `tools/hwloop/lpn/inputs.py` (export
  context), `lpn/emit.py` (LPNDialect on PYTHONPATH, the pin or the export as root), `lpn/wide.py`
  (the wide model: `WideTiming`, `plan_mm`, `WideCompactOpenTPU`, `memory_report`, `load_export`),
  `lpn/validated.json` (the graded registry), `lpn/rtl_pin.py` (the pinned RTL on a bundle);
  `lpn/__init__.py`: the pin, the label, the gate, the validated configuration.
- Receipts: `tools/hwloop/receipts/2026-10-16/lpn_toolchain/`, `lpn_upstream_gemm/`,
  `lpn_upstream_mlp/`, `lpn_gemma3_portb1/`, `lpn_gemma3_portb1_lat60/`;
  `receipts/2026-10-17/opentpu96_*` (the fresh Verilator receipts on 96a90c2f) and `lpn96_*`
  (the graded LPN runs).
- Scratch (not in git): `~/ergodex-03/lpn-scratch/{LPNDialect, lpn-build, llvm20-env, llvm20-bin,
  openTPU-lpn-pin, runs, logs}`; the 372 MB DRAM images stay there.
- Tests: `tests/test_hwloop.py` (the native-log parser, the record and its compare, the staleness
  refusal, `check-stale` on an LPN directory, the nightly's LPN lines).
