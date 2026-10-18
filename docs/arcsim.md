# ArcSim: a second RTL simulator (OTPU_SIM=arcsim)

ArcSim is Ergodex's licensed cycle-accurate RTL simulator, built on CIRCT's arcilator: it
compiles SystemVerilog through `circt-verilog` into HW MLIR, lowers it with `arcsim` (the
arcilator) into a native model with a generated simulation driver, and compiles that with LLVM
`opt` / `llc` into an object a C++ harness links. This tree runs it as a second backend next to
Verilator: `OTPU_SIM=arcsim` in the environment, and `opentpu.rtlsim` builds and runs an ArcSim
model of the same SystemVerilog, with the same command line, files, output lines and exit status,
so `tools/perf_qwen.py`, `tools/perf_bonsai2.py`, `tools/pi05_rtl.py` and the pytest suite run
unchanged. The default stays Verilator (`docs/hwloop.md`: Verilator before synthesis).

ArcSim runs on Linux x86-64 only (glibc >= 2.38); the numbers below are from the m7a.16xlarge
(64 cores, 246 GB, Ubuntu 24.04, Verilator 5.048, clang 20), the shared box of the Verilator
test matrices, with the load noted at each measurement.

## Install

The developer bundle (`arcsim-coral-matrix-baseline-20261013.zip`, the Coral NPU team's: the
fixed `circt-verilog` frontend, the lowering compiler `arcsim`, matching `opt` / `llc`, the
runtime library and link dependencies; no license check) or the release packages
(`arcsim_0.3.9_amd64.deb` + `ergodex-site-tools_0.3.9_amd64.deb`, whose simulators exit 78
without a license: `Ergodex_Oct_15_2026_Temp.lic`). rtlsim finds the toolchain at
`$ARCSIM_HOME`, else the bundle under `~/.cache/arcsim/`, else the package's `/usr/lib/ergodex`
(`toolchain/bin` or `bin` for the tools, `lib` for the runtime).

```
# the bundle (verified by its SHA256SUMS; needs unzip, sha256sum), as the Coral repo's
# utils/arcsim/setup_arcsim.sh installs it
./setup_arcsim.sh bundle arcsim-coral-matrix-baseline-20261013.zip      # -> ~/.cache/arcsim/arcsim-coral-matrix-baseline-20261013
~/.cache/arcsim/arcsim-coral-matrix-baseline-20261013/toolchain/bin/arcsim --version   # LLVM 23.0.0git, CIRCT 00f6937
# linking needs a C++ compiler (clang++ or g++; OTPU_CXX names one) and libatomic, zlib, zstd,
# tinfo as runtime libraries (the .so.N are enough: no -dev packages)
```

The bundle's `setup_arcsim.sh` ends with a harmless `tmp: unbound variable` message from its
own cleanup trap once the bundle is in place.

## Build and run

```
OTPU_SIM=arcsim OTPU_TARGET=f2-hbm OTPU_NCH=8 python tools/arcsim_model.py --mcols 8 --portb 4 --rowlanes --aline 64
```

builds (or finds) the f2-hbm board model of the Gemma receipt's configuration (`tb_top` with
`otpu_native_mem` on 8 channels, the parameters `rtlsim.top_params` gives `tools/perf_qwen.py`)
in `build/arcsim/tb_top_<hash>/Atb_top` and prints the stages' seconds (`build_times.json` in the
build's directory; each stage's command and output in `<stage>.log`). Any tool or test does the
same on its first run under `OTPU_SIM=arcsim`:

```
OTPU_SIM=arcsim OTPU_TARGET=f2-hbm OTPU_NCH=8 python tools/perf_qwen.py --model gemma3 --layers 4 \
    --wformat int8 --head-format int8 --lat 60 --mhz 200 --portb 4 --rowlanes --mcols 8 --aline 64
OTPU_SIM=arcsim python -m pytest -q tests/test_f2_rtl.py tests/test_rtl.py -k "replay or act_reads or attention"
```

The cache key is Verilator's (the sources' names and contents, the parameters) plus the harness,
ArcSim's flags (`rtlsim.AFLAGS`), the target CPU and the toolchain's version, so a Verilator and
an ArcSim build of one configuration live side by side (`build/verilator/`, `build/arcsim/`).

The pipeline (`rtlsim._compile_arcsim`, the Coral flow's `rules/arcsim.bzl` with this tree's
flags):

| stage | command | what |
|---|---|---|
| frontend | `circt-verilog --Osim --ir-hw --single-unit --timescale=1ns/1ps --allow-dup-initial-drivers -Wno-multiple-always-assigns --allow-use-before-declare -DOTPU_ARCSIM --top=tb_top -G<param>=<value>... <sources> -o design.mlir` | SystemVerilog to HW MLIR; `--Osim` imports procedural blocks in the form arcilator schedules |
| lower | `arcsim design.mlir --emit-llvm --state-file=state.json --Osim --giant-value-bits=0 [--partition-eval-tasks=N] -o model.ll` | the arcilator: the model's eval, its initial, its state layout and a driver `tb_top_main` that runs the testbench's own clocks and initial blocks to `$finish` |
| opt | `opt -mtriple=x86_64-unknown-linux-gnu -mcpu=native -O3 -vectorize-slp=true model.ll -o model.bc` | |
| llc | `llc -mtriple=... -mcpu=native -O3 -filetype=obj -relocation-model=pic model.bc -o model.o` | |
| link | `clang++ -std=c++17 -O2 -DOTPU_ARCSIM_MAIN=tb_top_main sim/arcsim/otpu_arcsim.cpp model.o <lib>/runtime.o <lib>/libCIRCTArcRuntime.a <lib>/libCIRCTSupport.a <lib>/libLLVMSupport.a <lib>/libLLVMDemangle.a -latomic -l:libz.so.1 -l:libzstd.so.1 -l:libtinfo.so.6 -pthread -ldl -lm -o Atb_top` | |

`OTPU_ARCSIM_CPU` (default `native`: the model runs on the machine that built it; `x86-64` for a
portable one), `OTPU_ARCSIM_ARGS` (more arcilator arguments, e.g. the bundle's experimental
`--arc-*` passes), `OTPU_ARCSIM_TASKS` (below) are in the cache key.

## The contract (what the harness is, and is not)

ArcSim compiles the **whole testbench**: `sim/verilator/tb_top.sv`, the design and the memory
models, exactly the sources Verilator gets (`rtlsim.build_top`), with the same `-G` parameters.
The generated driver runs the testbench's `always #5 clk`, its initial block's `@(negedge clk)`
waits, `$value$plusargs`, `$display`, `$fopen` / `$fwrite` / `$fscanf`, `$readmemh`, SystemVerilog
queues and `$finish`, so the harness (`sim/arcsim/otpu_arcsim.cpp`) is 200 lines that are not a
testbench: it hands the `+plusargs` of its command line to the runtime (`circt_sv_set_plusargs`),
calls `tb_top_main()`, and provides two things the SystemVerilog models get through DPI-C under
`-DOTPU_ARCSIM`:

1. **The DRAM image store.** `otpu_native_mem` (the board's memory path, AXI = 1) and `otpu_dram`
   (the behavioural DRAM) keep their image in the harness (`otpu_mem_rd / wr / wrm / init /
   dump`): `circt-verilog`'s `$fread` takes memories of up to 1024 elements, and a 4 GiB array in
   the model's state would be a 4 GiB value to its compiler. The harness loads `<dir>/dram_<s>.bin`
   (big-endian words, as rtlsim writes them and `$fread` reads them) at `otpu_mem_init` and writes
   `<dir>/dram_out_<s>.bin` (native words, as `$fwrite "%u"` writes them) at `otpu_mem_dump`; the
   image is `calloc`'d, so a 4 GiB slice costs the pages the program touched. In `otpu_dram` the
   reads of a cycle are made before its writes, as the array's non-blocking writes let them; in
   `otpu_native_mem` a beat is one channel's and a channel performs one command a cycle, so the
   immediate writes change nothing.
2. **Verilator's random draws.** `otpu_native_mem` stalls its readies and jitters its read
   returns and write counts with `$urandom` (seeded by `+axi_seed`), and the receipts' cycle
   counts depend on the draws (seed 1 / 7: 949,497 / 949,587 cycles for the Gemma receipt). Under
   `-DOTPU_ARCSIM` the draws come from `otpu_urandom` in the harness: Verilator 5.048's stream
   (xoroshiro128** seeded through splitmix64 from `(seed << 32) ^ seed`, the reseeding call's
   own draw the stream's first; `verilated.cpp`), checked value for value against Verilator. The
   order of the draws is the model's own: its channel processes are one `always_ff` whose
   statement order is the order Verilator ran the per-channel processes in (every channel's two
   ready stalls, then channel by channel the command performed, the read return's stall and the
   write count's stall) -- a rewrite of `otpu_native_mem` that leaves Verilator's draws, and so
   its cycle counts, as they were (seed 1 / 7 reproduced to the cycle).

Everything else -- `RESULT cycles=... halted=... error=...`, `SLICE`, `MXSCALE`, `MXACT`, `MEM
ch<c>`, `NATIVE ch<c>`, the `T<s> ...` trace lines, `tmem_<s>.hex`, the exit status -- is the
testbench's and the design's own `$display` / `$fwrite`, so `rtlsim.run` parses an ArcSim run as
it parses Verilator's.

## Threads

ArcSim has multi-threaded evaluation: arcilator's task mode (`--partition-eval-tasks=N`) splits a
model's eval into a dependency graph of task functions of at most N operations each, which the
runtime's scheduler runs on `ARC_THREADS` threads (the run's environment; the harness passes it
through). `OTPU_ARCSIM_TASKS=N` at build time selects it; `ARC_THREADS=n` at run time sets the
threads. `--partition-eval=N` (no threads) splits the eval into partition functions, which also
shortens `opt` / `llc` on a monolithic eval. Verilator's counterpart is `--threads N`
(`OTPU_VL_THREADS=N` adds it to `VFLAGS`).

## Bit-exactness

The Gemma-3-1B int8 receipt on this tree (fd0a2c51 + this branch; `OTPU_TARGET=f2-hbm
OTPU_NCH=8`, `tools/perf_qwen.py --model gemma3 --layers 4 --wformat int8 --head-format int8
--lat 60 --mhz 200 --portb 4 --rowlanes --mcols 8 --aline 64`): Verilator **949,497 cycles**
(seed 1; 949,587 at seed 7). docs/wide_port.md's 949,474 was the tree before the round-4 WIP
commits (2b763946, 7d3da502: the MXU's entry and the ACT block a register later); the receipt
moved by 23 cycles there, not here -- the memory-model rewrite above reproduces both seeds to
the cycle on Verilator.

**ArcSim and the f2-hbm board model: open.** The receipt's model builds (70 minutes, below) and
runs, and stops at reset in the design's own replicated-register check:

```
Fatal: otpu_native_dram: channel 4 B tail copy 0 is 1, the tail 0
```

(`rtl/mem/otpu_native_dram.sv`: every copy of the B FIFO's tail, `rb_tq[c][k]`, must equal the
tail `rb_t[c]`; both are reset and incremented by adjacent statements of one `always_ff`). The
same executable says channel 2 under the Gemma receipt and channel 4 under a HALT program,
deterministically, so it is the lowering of the register, not an uninitialized state: arcilator
reports `rb_tq` as an "array register not converted to a memory: element type is not a signless
integer (nested arrays ...)" (`array<8 x array<8 x i6>>`, kept as one value updated through
inject chains), and its non-blocking element writes in nested loops come out wrong. The
Verilator model of the same configuration runs the HALT program in 3 cycles. Five lowering
variants on a shrunken configuration (small TMEM / IMEM / ACT / FIFO, the same adapter; a HALT
program) all stop at the same check with the same values: `OTPU_ARCSIM_ARGS` of
`--shadow-nba-writes=false`, `--queue-unobserved-nba=false`, `--infer-array-memories`,
`--arc-scalarize-small-array-values --arc-factor-array-updates`, and the default. A flat
`rb_tq [NCH * RBP]` in the RTL (sidestepping the nested array; an experiment on a copy of
otpu_native_dram.sv, not in the tree) was the next probe when this was written -- see the
report; if it passes, the next replicated-register or nested-array check is the next question.
Until the board model runs, the ArcSim backend is validated on the behavioural-DRAM
configurations (below), not on the board's memory path, and the receipt's ArcSim cycle count,
cycles/s and thread scaling are not available. This is the first thing to take to Ergodex: a
reproducer is `tools/arcsim_model.py` for any f2-hbm configuration plus a HALT program, and
`build/arcsim/<model>/lower.log` names the register.


`tests/test_f2_rtl.py` (the F2 fabric's own testbenches, `rtlsim.build` of `tb_f2nch_protocol`,
`tb_f2mem`, `tb_f2axil`, `tb_f2hbm`): Verilator 16 / 16; ArcSim 10 / 16 -- `tb_f2nch_protocol`
(2), `tb_f2axil` (4) and `tb_f2mem` (4) pass, the six `tb_f2hbm` cases fail inside the run
(`otpu_f2_hbm_nch: B with no write in flight`, `read returned stale or wrong data`, `R from a
port not read`: the bridges see responses the testbench's HBM model should not have given).
Both testbenches drive the DUT's response channels from SystemVerilog queues; `circt-verilog`
rejects a queue element in a continuous assignment, and with the reads moved to `always_comb`
ArcSim did not re-evaluate them on every queue mutation (tb_f2mem then failed the same way),
so both testbenches now register those channels at the end of the one `always_ff` that mutates
the queues -- the same values at every clock edge (Verilator: 16 / 16 before and after), and
tb_f2mem passes on ArcSim. tb_f2hbm (three clock domains, two read queues selected by `rsel`)
still diverges; not analysed further here.

`tests/test_rtl.py -k "replay or act_reads or attention"` (26 tests, the behavioural DRAM, no
board memory path): Verilator 25 passed, 1 skipped (5 min). ArcSim: **22 passed, 1 skipped, 3 failed** in 1 h 29 min (the three are `test_fuzz_mm_replay`
at MCOLS 2, whose lowering fails before any simulation: "failed to legalize unresolved
materialization from ('!hw.array<1xi32>') to ('i32')", a one-element array the arcilator does
not handle; the attention decode / layer, the one-copy replay and act_reads cases all pass) -- each
distinct `tb_top` parameter set is its own 20-70 minute build (see "Build times"), so this is a
run of hours, with `-n 6`. Where a test passed, the RTL's DRAM and TMEM after the run are
bit-identical to the ISA simulator's, as the tests assert (`assert_same_state`): ArcSim's
model of the design is the design.

## Benchmarks

### Build times

ArcSim's build is the price of its model: one process compiles the whole design, and `llc` on
the lowered IR takes most of an hour for the board configuration, whatever the optimization
level (`-O0`, `-O1`, `-O2` and `-O3` all within a few percent of each other: the time is not in
the optimizer but in instruction selection over this design's 512- and 1024-bit values), while
Verilator's build of the same (`-j 0`, 64 cores) is 3-4 minutes. The stages (`build_times.json`;
the box under a load of 8-12 from the other builds and runs):

| model | frontend | lower | opt | llc | total |
|---|---|---|---|---|---|
| `tests/test_rtl.py` attention decode (S 1, D 32, the behavioural DRAM) | 12 s | 16 s | 119 s | 1006 s | 19 min |
| the same at the larger parameters of its second case | 22 s | 38 s | 150 s | 2170 s | 40 min |
| the Gemma receipt's f2-hbm model (D 128, NCH 8, `--partition-eval=5000`) | 25 s | 65 s | 285 s | 3818 s | 70 min |
| the same with a monolithic eval (`OTPU_ARCSIM_PARTITION=0`) | 25 s | 74 s | 2927 s | crashed (SIGSEGV after 158 s: the giant-value class, on the design before the poke tables moved to the harness) | -- |

`--partition-eval=N` puts the model's eval into partition functions of about N operations (64
of them at 5000; the largest stays at ~25,000 lines of IR whatever N, down to 300), which took
`opt` from 49 minutes to 5. Finer partitions (1000, 300) and arcilator's
`--split-funcs-threshold` (it splits the 161,000-line initial function) did not change `llc`'s
hour, nor did `optnone` on the one-shot functions. Parallel code generation (the module split
into pieces for several `llc`s, as Verilator's make -j does) would, but the box's LLVM 20
`llvm-split` cannot read this LLVM 23 IR (`nocreateundeforpoison`, the new `llvm.lifetime`
signature) and the bundle has no `llvm-split`; a splitter of our own over the textual IR is the
next step if ArcSim stays in the loop.

### The Gemma receipt: cycles/s

The receipt (949,497 cycles) on the m7a.16xlarge, each run pinned to idle cores with `taskset`
(4 cores for a single-threaded model, N for N threads), the model built beforehand (the seconds
are `tools/perf_qwen.py`'s own "sim" time: the simulator process alone), the box's 1-minute load
in the column (64 cores; the other agents' Verilator matrices and this work's builds):

| simulator | threads | cycles | sim time | cycles/s | load |
|---|---|---|---|---|---|
| Verilator 5.048 (`--binary -O3`) | 1 | 949,497 | 251 s | 3,783 | 17-20 |
| Verilator `--threads 4` (`OTPU_VL_THREADS=4`) | 4 | 949,497 | 137 s | 6,930 | 17-20 |
| Verilator `--threads 8` | 8 | **949,568** | 136 s | 6,982 | 17-30 |
| ArcSim (`--partition-eval=5000`) | 1 | -- (stops at reset, above) | -- | -- | |
| ArcSim task mode (`OTPU_ARCSIM_TASKS=5000`, `ARC_THREADS=2 / 4 / 8`) | 2 / 4 / 8 | -- (the same model) | -- | -- | |

Verilator's 8-thread model is not cycle-exact with its own single-threaded one: the memory
model's `always_ff` is split over threads and Verilator's `$urandom` is per thread (a
thread-local stream reseeded from the same seed), so the draws land on other decisions; at 4
threads the block stayed on one thread. 8 threads buy nothing over 4 on this design (the eval
is one slice's logic with a serial memory model), and 4 threads 1.83x. Verilator's builds: 3-4
minutes (`-j 0`, 64 cores).


## Limits

- The Coral bundle's frontend is strict where Verilator is lenient: a variable an `initial`
  block sets and an `always_ff` drives needs `--allow-dup-initial-drivers`, a name used before
  its declaration `--allow-use-before-declare`; both are in `AFLAGS`.
- `otpu_seq`'s `initial` block fills the WAITC sync table (`u_sync.g_auto.mem`, 1024 x 1024
  bits) through the RAM instance's hierarchy, and the lowering carries that memory by value in
  the process; arcilator's default budget (`--giant-value-bits=262144`) refuses it, so `AFLAGS`
  sets it to 0. `--preserve-memories` at the frontend was tried instead and fails the lowering
  ("'arc.execute' op is on a combinational loop").
- No waveforms (`+trace` is the design's own text trace, as in Verilator; there is no VCD from
  this flow).
- `otpu_native_mem`'s PHYS mode (the channels' own `ch<c>.bin` files, tb_board's) needs
  Verilator (`$fatal` under `-DOTPU_ARCSIM`).
- The bundle is a developer snapshot of 13 October 2026, not the 0.3.9 release; `--version`
  says LLVM 23.0.0git, CIRCT 00f6937.
