# f2-burst200: Alex's F2 HBM flow as a target of this tree

Alex's burst200 port (branch `alex-burst200`, commit 06c169cb on 7e42a626, an ancestor of
tpubench) is the one openTPU F2 image that closed 200 MHz (+0.032 / +0.010 ns after its
refinement pass) and ran Gemma 3 1B and Qwen3-0.6B on hardware (`agfi-0d06cb48544a02106`,
2026-10-15). This note records how it was brought onto tpubench as a target of its own,
**`f2-burst200`**, next to our HDK flow (`boards/f2/build.py`: targets `f2`, `f2-hbm`,
`f2-hbm:s4`, `f2-hbm16`; round 5 in [f2-round5.md](f2-round5.md)). The two flows share no
file: his build reads its board RTL from its own directory and takes only the core and the
shell from `opentpu.target.sources`.

Read `boards/f2/burst200/README.md` (his README, path strings updated) for the flow itself:
stages ip, pcis, clocks, synth_link, implement, refine, audit; the HDK's
`small_shell=0x10212415`; Vivado 2025.2.

## What was imported

Every file of his commit, under a new path where ours occupied his:

| His (alex-burst200) | Here (v4-f2-alex) | Change |
|---|---|---|
| `boards/f2/{README.md,build.py,cl_id_defines.vh,simulate.py}` | `boards/f2/burst200/` | path strings only (table below) |
| `boards/f2/constraints/*.tcl` (4) | `boards/f2/burst200/constraints/` | byte-identical |
| `boards/f2/scripts/*.tcl` (8) | `boards/f2/burst200/scripts/` | byte-identical |
| `boards/f2/scripts/*.py` (9) | `boards/f2/burst200/scripts/` | `ROOT` depth and target path only |
| `rtl/boards/f2/{cl_opentpu.sv,otpu_native_axi.sv,otpu_hbm_geometry.sv}` | `rtl/boards/f2/burst200/` | byte-identical (his `otpu_native_axi`, not round 5's) |
| `targets/f2.toml` | `targets/f2-burst200.toml` | `name`; `T2 = false` (a Config field since his base); `rtl_board` (below) |
| `tests/f2/{axi_hbm_model.sv,tb_f2_mem.sv,tb_native_axi.sv,test_transport.py}` | `tests/f2_burst200/` | byte-identical (`test_transport.py`'s `parents[2]` is still the root) |
| `opentpu/host/{f2_transport.py,f2_mmio.c}` | same | byte-identical; no name clash with our `opentpu/host/f2.py` |
| `docs/f2/*.svg` (3) | `docs/f2-burst200/` | byte-identical |
| `.gitignore`, `README.md`, `pyproject.toml` | same | his three small edits, paths updated |

The path edits, and nothing else, in `boards/f2/burst200/`:

| File | Edit |
|---|---|
| `build.py` | `ROOT = ...parents[2]` -> `parents[3]`; `BOARD = ROOT / 'boards/f2'` -> `'boards/f2/burst200'`; `ROOT / 'rtl/boards/f2'` -> `'rtl/boards/f2/burst200'`; `targets/f2.toml` -> `targets/f2-burst200.toml` (the copy into `synth_link/` is named `f2-burst200.toml`); `--output` default `build/f2/burst200` unchanged |
| `simulate.py` | `parents[2]` -> `parents[3]`; `tests/f2/` -> `tests/f2_burst200/` (3); `rtl/boards/f2/` -> `rtl/boards/f2/burst200/` (3); `boards/f2/scripts/sim_f2_mem.tcl` -> `boards/f2/burst200/scripts/sim_f2_mem.tcl` |
| `scripts/emit_config.py`, `generate_f2_top.py`, `f2_run.py`, `smoke_f2.py`, `benchmark_f2_{memory,mxu,native}.py`, `benchmark_qwen3.py` | `parents[3]` -> `parents[4]`; `targets/f2.toml` -> `targets/f2-burst200.toml` |
| `scripts/package_f2.py`, `scripts/*.tcl`, `constraints/*.tcl`, `cl_id_defines.vh` | none |
| `README.md` | the same path strings in the prose and commands; the diagram links one level deeper |

`generate_f2_top.py`'s output and `emit_config.py`'s emitted parameters are unchanged:
with `targets/f2-burst200.toml` the emitted generics are his (NCH=2, CH_LOG2=31,
DRAM_LOG2=32 added since his base, all three the defaults his checked-in `cl_opentpu.sv`
relies on; CORE_KHZ=200000; ARCH_HASH 0x4ffd1e17 = 1341988375, the value baked into his
wrapper). `build_f2.tcl` does not use the generics: the wrapper instantiates `otpu_board`
with its parameters written out. `rtl/shell/otpu_board.sv`'s port list at NCH=2, S=1 is
what it was at 7e42a626 (`calib` / `ded` / `n_*` are `NCH*S` wide, 2 as before).

## The one change outside his files: `opentpu/target.py`

`sources()` for a target of board `f2` with `mem = "hbm"` returns our round-5 list
(`boards/f2/otpu_native_axi.sv`, `otpu_f2_hbm_*.sv`, ...). His `build_f2.tcl` reads
`$root/rtl/$f` for every file of that list and then `[glob $run/rtl/*.sv]`, his copy of
`rtl/boards/f2/burst200`: the two `otpu_native_axi` modules would meet in one synthesis.
`sources()` now honours an optional platform key, `rtl_board`, as the `BOARDS` key
(`BOARDS.get(t.platform.get("rtl_board", board), [])`; `BOARDS["f2-burst200"] = []`), and
`targets/f2-burst200.toml` sets `rtl_board = "f2-burst200"`. Its sources are the core and
the shell, the list his flow had at 7e42a626. The toml keeps `board = "f2"` and
`platform_id = 3` because `tests/test_target.py` requires the targets of one platform id
to share kind, family, board and WIN. Our targets' lists are unchanged.

Our `boards/f2/build.py` copies RTL per `target.sources()`, file by file, so the
`rtl/boards/f2/burst200/` subdirectory is invisible to our flow (and to
`tests/test_f2_build.py`'s `*hbm*`-free DDR snapshot).

## Build on the F2 instance

From the repository root, with the venv's Python (`opentpu.target` needs numpy, `tomllib`
3.11+) and Vivado 2025.2 on PATH (`build.py` runs `vivado -version` and refuses anything
else):

```bash
python boards/f2/burst200/build.py --hdk /home/ubuntu/aws-fpga/hdk --plan
python boards/f2/burst200/build.py --hdk /home/ubuntu/aws-fpga/hdk \
    --output /home/ubuntu/rachit/f2burst200-run1
```

`--output` must not exist. The driver checks the HDK (`cl_bb_routed.small_shell.dcp`,
`cl_pins.xdc`, `small_shell_level_1_fp_cl.xdc`, `cl_ports.vh`, `sh_ddr.sv`,
`interfaces.sv`, `cl_mem_hbm_wrapper.sv`, `cl_hbm.xci`, `shell_version.txt`), writes
`inputs.sha256.json` (every `rtl/**/*.sv`, his scripts, the toml), then runs the seven
stages one after another, each as `vivado -mode batch -source <stage>/build.tcl`, its
console in `<stage>/console.log`, its command in `<stage>/command.json`. It stops at the
first stage without its marker file, and after `refine` unless the console says
`ROUTED_TIMING_PASS_PENDING_REPORT_REVIEW`. The accepted checkpoint is
`refine/post_route.dcp`; `audit/` holds the independent reports; packaging is
`scripts/package_f2.py` (the HDK packager, a checksum-bound review JSON).

Files his Tcl and wrapper read from the HDK beyond the driver's check list, all standard
in aws-fpga for the small shell: `common/lib/lib_pipe.sv`;
`common/shell_stable/design/sh_ddr/synth/{sync.v, flop_ccf.sv, ccf_ctl.v, mgt_acc_axl.sv,
mgt_gen_axl.sv}`; the interface templates `cl_opentpu.sv` includes
(`unused_ddr_template.inc`, `unused_cl_sda_template.inc`, `unused_flr_template.inc`,
`unused_apppf_irq_template.inc` under `common/shell_stable/design/interfaces`).

Things to know before running it:

- Vivado threads: `create_f2_ip.tcl`, `build_f2.tcl` and `implement.tcl` set
  `general.maxThreads 8` (route and refine 6, the small stages 4). Fine on the instance's
  cores; the shell link and placement want tens of GB of memory.
- The image will report `BUILD_ID` 0x7e42a626 (`cl_sh_status1` too): his wrapper writes
  it out, and it is not regenerated. The core RTL under it is tpubench's, not 7e42a626's.
- `build_f2.tcl` waits up to an hour for `ip_done.txt` of the ip / pcis / clocks stages;
  with the sequential driver that never triggers.
- Timing closure is not implied: his +0.032 ns was on 7e42a626's slice; ours has round 5's
  changes in `otpu_slice`, `otpu_top` and `otpu_board`. The `refine` stage is the gate.
- Nothing is resumable in place: a failed stage means a fresh `--output` (or re-running
  that stage's `command.json` by hand in its directory).

## What differs from round 5

Round 5 ([f2-round5.md](f2-round5.md)) copied his structure (no openTPU RTL on the 450 MHz
HBM clock, AXI4 masters at the core clock, vendor IP for the crossing) onto our 8-channel
geometry. The two targets side by side:

| | `f2-burst200` (his) | `f2-hbm` round 5 (ours) |
|---|---|---|
| channels | 2 per slice, 2 GiB each (`channels_per_slice 2`, `channel_bytes 2 GiB`) on HBM ports 0 and 4; ports 1-3 tied off; 4 GiB of the 16 | 8 per slice, 512 MiB each, on ports 0-7 (`f2-hbm16`: 16; `f2-hbm:s4`: 32) |
| fabric | one SmartConnect block design `f2_mem` (3 SI x 2 MI: the two adapters and the PCIS as a third slave; address-routed, DECERR outside the map), his `otpu_native_axi` (16-beat read bursts, direction barrier) and `otpu_hbm_geometry` | the HDK's `cl_axi_sc_1x1` per port (clock and width conversion), round 5's `otpu_native_axi` (CH_LOG2 parameter, registered read return), `otpu_f2_hbm_port` / `_host` / `_mem` / `_sys` |
| host | PCIS into the SmartConnect behind an `axi_register_slice` (`f2_pcis_pipe`, SLR1); OCL through `axi_clock_converter` (`f2_ocl_cdc`); `opentpu/host/f2_transport.py` + `f2_mmio.c` | `otpu_f2_hbm_host`; `opentpu/host/f2.py` |
| clocks | two clk_wiz (`f2_core_clk` 200/400 from clk_main_a0, `f2_hbm_clk` 450 from clk_hbm_ref), `xpm_cdc_async_rst` per domain | the same four clocks from two MMCME4_ADV in `otpu_f2_hbm_sys` |
| flow | `boards/f2/burst200/build.py`: his own Tcl (OOC synth, link to the small shell dcp, Explore / AggressiveExplore, a refinement pass, an audit); floorplan `pblock_CL_SLR0` + `pblock_CL_SLR1` | `boards/f2/build.py`: the HDK's `build_all.tcl` (BuildAll) with our constraints and floorplans |
| bandwidth | 2 x 512 b x 200 MHz = 25.6 GB/s at the native ports; measured 23.8 GB/s weight stream, 22.1 GB/s in the Gemma decode (33.7 tok/s device) | 8 x 12.8 = 102 GB/s at the native ports for the slice's 25.6 GB/s demand |

## Verification run for this import

- Every imported file diffed against `alex-burst200`: only the path lines above differ;
  `rtl/boards/f2/burst200`, `tests/f2_burst200`, `docs/f2-burst200`, the host files are
  byte-identical.
- `boards/f2/burst200/build.py --hdk /nonexistent --plan` prints the seven stages.
- `boards/f2/burst200/scripts/emit_config.py /tmp/emitcfg --target targets/f2-burst200.toml`:
  `config.tcl` generics as above, `rtl_files` core + shell only.
- `tests/f2_burst200/tb_native_axi.sv` against his `otpu_native_axi.sv` under Verilator
  5.048 (`--binary --timing`, BURST=16, OUT=32, seeds 1 / 42 / 2026 / 4700): PASS; his
  `simulate.py` under Icarus: PASS, 16 trials. `cl_opentpu.sv` itself cannot be linted
  here (Xilinx IP, HDK includes).
- Our flow's tests (`tests/test_target.py`, `tests/test_f2_build.py`,
  `tests/test_f2_scoped_clocks.py`) pass as before.
