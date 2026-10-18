# The F2 HBM target on tpubench (v4-f2-port)

Branch `v4-f2-port`, from tpubench's bfeb2be3. Goal: today's design (one slice, D 128, MCOLS 4,
8 lanes) built as an AFI for the AWS F2's HBM and running Gemma 3 1B on it ([f2_hbm.md](../f2_hbm.md),
[f2.md](../f2.md)). The F2 work comes from the sibling repository openTPU-vla-rhino (remote
`rhino`, branch `f2-hbm` at ef33566c: f237dab2, the common base, plus 11 commits).

## What was merged

Merge commit **4ccf9dc2** (`git merge rhino/f2-hbm`, history kept): the DDR F2 target (8c6c6eb1:
`boards/f2/build.py`, `route_opt.py`, the constraints and Tcl scripts, `design/cl_opentpu.sv`,
`rtl/boards/f2/otpu_f2_{sys,mem,nch}.sv`, `otpu_axil_cdc.sv`, `targets/f2.toml`,
`opentpu/host/f2.py`, `board.py`'s NCH channel map, `tools/f2_smoke.py`, `f2_bench.py`,
`fpga_inference_check.py`, the `f2_web` gateway, `tests/test_f2_*.py`, `docs/f2.md`) and the HBM
fabric (e5ecbf86 .. ef33566c: `design/cl_opentpu_hbm.sv`, `otpu_f2_hbm_{sys,mem,nch,port,host}.sv`,
`targets/f2-hbm.toml` / `f2-hbm16.toml` / `f2-hbm200.toml`, `rtl/mem/otpu_native_dram.sv`,
`rtl/shell/otpu_board.sv` and `rtl/top/otpu_top.sv` parameterized by NCH, `sim/verilator/
otpu_native_mem.sv`'s NCH model, `tb_f2hbm.sv`, `tb_native_nch.sv`, `tests/test_native_nch.py`,
`docs/f2_hbm.md`), and with them Ternary-Bonsai 2 and the t2 weight format (4e9616a4:
`opentpu/llm/bonsai2.py`, `gguf_file.py`, `isa.py` / `isasim.py` / `quant.py`'s t2, `MM QUAD`,
`QACT DUP4`, `Config.T2` off by default, `docs/ternary.md`) and the Qwen3 YaRN RoPE. The RTL
the F2 branch changes is the memory path and the shell (`otpu_native_dram`, `otpu_board`,
`otpu_top`); it does not touch `rtl/mxu` or `otpu_slice.sv`, so tpubench's M0 counter split
(`MXBLOCK` in `otpu_slice.sv`, `rtlsim.stats["mxu_block"]`) and its P4 slices work merged
without an RTL conflict.

The f2-hbm target resolves (`OTPU_TARGET=f2-hbm`): name `f2-hbm`, platform id 3, kind fpga,
`rtl_params` S 1, D 128, MCOLS 4, LANES 8, ACT_ROWS 16, TMEM_IMPL 3 (PMP 1), NCH 8, CH_LOG2 29,
DRAM_LOG2 32, CHASH 1, SEQ_WC 1, ARCH_HASH 4ffd1e17 (the card's m4 arch: the same programs and
images as the YPCB card's, `T.config('f2-hbm') == T.config('ypcb-00338')`), 30 sources ending in
`boards/f2/otpu_axil_cdc.sv, otpu_f2_hbm_{nch,port,host,mem,sys}.sv`.

## Conflicts and how they were resolved

Six files conflicted, all Python / docs; the rule was to keep both sides' function.

| file | conflict | resolution |
|---|---|---|
| `opentpu/llm/gemma3.py` (add/add) | both branches wrote a Gemma 3: tpubench's runs on Gemma 4's `Spec` and kernels with the checkpoint's parameters folded (`spec_from_hf`, `folder`: norms 1 + w, the attention scale into q_norm, no V norm, no layer scalar); rhino's is a `gemma3.Spec` subclass with `norm_offset` / `attn_scale` / `layer_scale` fields, an HF / MLX / GGUF loader and FunctionGemma | tpubench's kept: [gemma3.md](../gemma3.md)'s Accuracy table and the P4 slices are its (gate 3 below reproduces it). Re-exports Gemma 4's `reference_logits` / `emulated_logits` for the F2 tools. Rhino's GGUF reader (`gemma3-int4`) and MLX names not carried; the `functiongemma` alias is (a `gemma3_text` checkpoint, through `spec_from_hf`) |
| `opentpu/llm/gemma4.py` | rhino's `_gain` / `attn_scale` / `layer_scale` plumbing for its Gemma 3 against tpubench's `v_norm` / `global_rope_div` and the S > 1 slices code | tpubench's file taken whole (rhino's changes were all for its Gemma 3; `tiled.py` likewise) |
| `opentpu/llm/__init__.py` | the docstring, `MODELS`, `load_spec`'s dispatch | tpubench's `load_spec` (gemma3 -> `spec_from_hf`), rhino's `yarn_rope`, Bonsai 2 and `functiongemma` / `bonsai` entries; `gemma3-int4` dropped |
| `opentpu/llm/qwen3.py` | `load_weights`'s dispatch | Bonsai 2 first (rhino), then Gemma 3 / Gemma 4 through `gemma4.load_weights` (tpubench: the fold) |
| `docs/gemma3.md`, `tests/test_gemma3.py` (add/add) | two documents, two test files | tpubench's; a section "On the F2 (the f2-hbm merge)" added to the document |

Rhino's tools adapted to tpubench's Gemma 3 in the merge commit: `tools/gemma3_check.py` takes
`load_spec` / `load_weights` (its `G.Spec.from_hf` was rhino's); `tools/fpga_inference_check.py`'s
`gemma3.reference_logits` is the re-export; `tools/validate.py` recognizes a Gemma 3 as a
`gemma4.Spec` with `v_norm` off. Auto-merged and checked: `opentpu/host/chat.py` (rhino's
`open_transport` / `--dev f2:0` and `SAMPLING["gemma3"]` beside tpubench's `one_slice_default`),
`opentpu/rtlsim.py` (rhino's `OTPU_NCH` in `top_params` and two-digit channel statistics beside
tpubench's `mxu_block`), `opentpu/target.py` (`BOARDS["f2"]` / `["f2-hbm"]`, `DRAM_LOG2` from
`slice_bytes`, `T2` in `arch_hash` only when on), `opentpu/host/board.py` (the NCH map:
`beat_at` / `split` / `join` with `nch` 2 by default, slice s on channels `nch s + c`: the S > 1
Gemma programs address each slice's logical DRAM as before), `opentpu/host/regs.py`
(`slice_bytes`, `CAP_T2`). `targets/f2.toml`, `f2-hbm.toml`, `f2-hbm16.toml`, `f2-hbm200.toml`
all have `platform_id = 3` = `opentpu/host/f2.py PLATFORM`.

Follow-up commit **b51ed4a8**: `tests/test_target.py` asserted platform ids unique over the
targets, which the four F2 targets (one board, id 3; the memory told apart by CFG_DRAM) break
(the test fails on rhino's branch too); it now asserts the targets of one id share what
`by_platform` reads from it (kind, family, board, WIN). Its pre-numerics hash helper leaves `T2`
out as it leaves `SEQ_WC` (fields the production host's Config does not have); the hashes
(4ffd1e17, 19a2400e, 89417856, ff270e6e, e170f967) are unchanged. `docs/targets.md` says so.

## Tests

All from the worktree root with `PYTHONPATH=. .../tpubench/.venv/bin/python -m pytest -q`,
Verilator 5.048, `models/gemma-3-1b-it` a symlink to tpubench's.

| suite | what | result |
|---|---|---|
| `tests/test_f2_build.py tests/test_f2_host.py tests/test_f2_smoke.py tests/test_target.py tests/test_gemma3.py tests/test_lens.py tests/test_fpga_inference_check.py tests/test_f2_web.py tests/test_f2_route_opt.py tests/test_f2_methodology.py tests/test_f2_scoped_clocks.py tests/test_f2_shell_baseline.py tests/test_quant.py tests/test_t2.py tests/test_yarn.py tests/test_gemma4.py tests/test_qwen3.py` | the F2 flow (HBM snapshot for f2-hbm, :s2, :s4, f2-hbm16), the host's NCH map (1x8, 2x8, 1x16, 4x8), the smoke tool, the targets, tpubench's Gemma 3 (tiny model vs HF, resident / prefill bit-exact, 2 and 4 slices), Lens, the F2 inference check (qwen3 / gemma3 x isa / board), t2, YaRN, Gemma 4, Qwen3 | first pass (with `-x`, before b51ed4a8): 102 passed, 1 skipped (the native F2 MMIO helper needs x86-64 and cc), stopped at `test_target`'s uniqueness assertion (fixed in b51ed4a8; `tests/test_target.py` alone: 23 passed). The rerun of the rest, 4,576 s: 352 passed, 13 skipped, 2 failed, 3 errors: the `test_target` hash assertion from before the fix (run started before it), three `tmp_path` errors "could not create numbered dir" (the host's disk full at the time: `test_gemma4.py::test_formats_scan_text`, `test_formats_scan`, `test_qwen3.py::test_from_hf_refuses_what_it_would_run_unlike_hf`; rerun below), and `test_gemma4.py::test_resident_gathers_are_bit_exact[fp4]` (CompileError: TMEM exhausted, a tile of 33,280 words with 33,537 of 65,536 in use). The three errored tests rerun: passed. The fp4 failure is tpubench's own, not the merge's: the same test fails with the same CompileError on tpubench's bfeb2be3 checkout (`/Users/.../ergodex-04/tpubench`, its `[int8]` and `[int4]` parameters passing there and here). Open on tpubench, unrelated to the F2 port |
| `tests/test_f2_rtl.py tests/test_native_nch.py` (Verilator) | tb_f2nch_protocol, tb_f2mem (ordering, hazards), tb_f2axil, tb_f2hbm (4 / 8 / 16 / 32 ports, three clock ratios, HAZ 0 detected); tb_native_nch: NCH 2 equals the two-channel adapter cycle for cycle, NCH 2 / 4 / 8 / 16 maps (CHASH on and off, port B above 4 GiB), a wrong map detected | 22 passed, 1 failed on the first run: `test_native_nch_map_and_order[16-1-1]`'s Verilator build died with "No space left on device" (the host's disk at 99%, other agents' builds); rerun alone: passed (264 s). 23 / 23 |
| `OTPU_NCH=8 OTPU_AXI=1 OTPU_BOOT=1 tests/test_rtl.py -k "board_memory_path or native_memory_path or mlp_rtl or logits_fill or dram_range or porta"` | otpu_board / otpu_native_dram on 8 channels (otpu_native_mem's model): the board memory path stress / fuzz, the native memory path (hazards, random), MLP decode through the adapter, logits fill, the DRAM range to 4 GiB, port A host writes between runs and before a WAITW: 26 tests | 24 passed, 2 failed (720 s): `test_dram_range_ending_at_4gib[ld]` / `[waitw]`, the RTL's TMEM holding the program's words. Not the merge's: the two pass with `OTPU_NCH=8` alone (f2_hbm.md's own recipe for them) and fail with `OTPU_AXI=1 OTPU_BOOT=1` on the default two channels too (the test's 4 GiB image against the boot image's program placement: a pre-existing environment interaction; f2_hbm.md's table ran `dram_range` without BOOT and the BOOT line without it) |

Gate 3, Gemma 3 on the ISA simulator in the f2-hbm configuration (`OTPU_TARGET=f2-hbm
tools/compare_hf.py --model gemma3 --chat --cfg board --resident --tokens 12 "Why is the sky
blue?" "Write a haiku about the ocean."`; `--cfg board` is the card's configuration, which is
f2-hbm's arch): the tokens of [gemma3.md](../gemma3.md)'s Accuracy table exactly. "Why is the
sky blue?": HF `The sky is blue due to a phenomenon called **Rayleigh`, device `Okay, let's
break down why the sky is blue`, the first token HF's #2 at 0.609 logits below the top, max
|diff| 2.269, min cosine 0.99869; the haiku `Blue waves crash and roar,\nSecrets deep within
the sea` identical for all 12 tokens, max |diff| 3.455, min cosine 0.99872.

Gate 4, one Gemma 3 decode token through the F2 HBM board model: `tb_f2hbm` is the fabric's
testbench (random traffic on the bridges and ports, no core), so the token ran through the board
model `tb_top` with the AXI memory path on `OTPU_NCH=8` (otpu_board, otpu_native_dram NCH 8,
otpu_native_mem's 8-channel model: [f2_hbm.md](../f2_hbm.md) "Tests"), as `opentpu.lens record
model` does: the chat prompt "Why is the sky blue?" prefilled on the ISA simulator, the step at
its last token (the one that picks the first new token) run on the RTL and on the ISA simulator
from the same resident decode program and DRAM, the logits compared bit for bit (the script
`/tmp/f2port_rtl_token.py`, its core quoted below). **Bit-exact**: the prompt's
15 tokens, the ISA step 42 s, the RTL step 722 s (Verilator, the trace on), both pick token
19058 `Okay`; `logits identical: True, max |diff| 0.0, DRAM after the step identical: True`.
8,029,592 cycles (gemma3.md's 10.28 M were the DDR3 model's 20% stalls; this is
otpu_native_mem's default on 8 channels). The 8 channels' native statistics: every channel
streams ~1,960,650 port B reads (the chunks rotated over the 4 channel pairs) and takes 2,052
port B writes; port A's runs and reads and the SW gather / RMW words fall on channels 0, 3, 5,
6 (~2,731 runs, 87,400 reads, ~1,675 SW reads and ~1,690 SW writes each) against ~1,068 runs
and 34,200 reads on 1, 2, 4, 7 (CHASH's pair fold over the A rows' addresses); no partial
beats. The M0 split's counters came through otpu_top's NCH board: `mxu_block` mb 13,265 = mbc
1,589 + mbr 8,311 + mbs 3,365 + mbo 0.

```
OTPU_NCH=8 OTPU_AXI=1 PYTHONPATH=. python /tmp/f2port_rtl_token.py <tree>
  # Engine(spec, W, cap=256, cfg=sim_config(spec, 256, board_config(PAIR=1, DSTEP=1, STREAM=1),
  #        lookup=True, wformat="int8"), backend=lens._Capture(..., {"axi": True}), resident=True)
  # eng.prefill(ids[:-1]) on the ISA; then eng.step(ids[-1]) once with _Capture.isa_mode (the
  # ISA simulator) and once traced on the RTL (rtlsim.run(cfg, programs, drams, trace=True,
  # uarch=BOARD_UARCH, axi=True)); np.array_equal on the logits and on the slices' DRAM
```

The same token on `--isa` (the ISA simulator's analytic timing) and on the DDR3 model are
gemma3.md's. The first run of this script was cut short by the host's full disk (the 1 GiB
`dram_out_0.bin` dump truncated at 1,061,234,880 bytes; `rtlsim` detected it); the second ran
with 8.6 GiB free.

## The build

On the F2 build instance (Vivado 2025.2 licensed for xcvu47p-fsvh2892-2-e, the AWS HDK at
20fe90180574464fd509c994680ec67191e9759f (2.3.0) with `hdk/common/ip` checked out at
6d32be972e6da854e61a8d3d6ec0466ab491c1b3 and `git lfs pull` run in it so the generated DDR and
HBM products are files, not LFS pointers; `hdk_setup.sh` sourced once with Vivado so the cl_ddr4
and cl_hbm IP are generated; the small shell 0x10212415 with its routed checkpoint
`cl_bb_routed.small_shell.dcp` at SHA-256 04a5331321e8...; Python 3.11+ with NumPy):

```
source /opt/Xilinx/2025.2/Vivado/settings64.sh
python boards/f2/build.py --hdk /home/ubuntu/aws-fpga --target f2-hbm --out RUN --flow BuildAll --package
```

`build.py` checks the HDK and IP revisions and products (`check_hdk`, `check_hbm_ip`: both
stacks, 450 MHz ports, the switch on), snapshots the CL (`RUN/cl_opentpu`), and runs the HDK's
`build_all.tcl` with the clock recipes A1 B2 C0 H2; `--package` writes the Developer_CL tar for
`aws ec2 create-fpga-image`. Hours; docs/f2_hbm.md's "Remaining steps" lists the timing risks
(the 450 MHz bridges across SLRs) and the on-card checks (vLED, CFG_DRAM / CFG_SLICES, BAR4,
`otpu-diag --mem`, then `tools/fpga_inference_check.py --model gemma3 --backend board --dev f2:0
--compare-isa`).

What could run here, without Vivado or the HDK: the snapshot step `build.py prepare()` for
`f2-hbm` with a stub HDK shell directory (a `build_all.tcl`) and a stub `cl_mem_hbm_wrapper.sv`,
as `tests/test_f2_build.py`'s HBM snapshot test runs it (the command line's `--prepare-only`
needs `check_hdk` first: `git rev-parse` of the HDK checkout at the pinned commits, the IP's
xci and generated products, so it cannot run without the HDK). It wrote `RUN/cl_opentpu/design/`
with `cl_opentpu_hbm.sv`, `cl_id_defines.vh`, the wrapper, `opentpu_geometry.vh` (`S = 1`,
`NCH = 8`), `opentpu_params.vh` (otpu_f2_hbm_sys's 32 parameters: `.S(32'h1) .D(32'h80)
.MCOLS(32'h4) .LANES(32'h8) .ACT_ROWS(32'h10) .TMEM_IMPL(32'h3) .TMEM_PMP(32'h1) ... .NCH(32'h8)
.CH_LOG2(32'h1d) .DRAM_LOG2(32'h20) .PLATFORM(32'h3) .ARCH_HASH(32'h4ffd1e17)
.CORE_KHZ(32'h30d40) .BUILD_ID(32'h4ccf9dc2)`), `sources.tcl` (the 30 RTL files, otpu_slice.sv
with the M0 counters among them, `rtl/boards/f2/otpu_f2_hbm_sys.sv` last), `rtl/` (the copies),
the HBM synthesis script and CDC constraints in the DDR ones' places, `small_shell_cl_pnr_user.xdc`
with `hbm_reset_n` in the linked-reset list, and `RUN/inputs.json` (the target, the parameters,
every file's SHA-256, the git commit and status).

## Not carried, open

- Rhino's Gemma 3 GGUF reader (`gemma3-int4`, `gaunernst/gemma-3-1b-it-int4-gguf`) and its MLX
  tensor names: tpubench's Gemma 3 reads HF safetensors. Bonsai 2's GGUF reader (`bonsai2.py`,
  `gguf_file.py`) is in.
- The t2 weight format is the ISA simulator's only (no RTL); `Config.T2` is off in every target.
- docs/f2_hbm.md's "Past 4 GiB a slice" (word addresses in the ISA, for f2-hbm16 / the 27B) is
  not done; f2-hbm (8 x 512 MiB = 4 GiB a slice) does not need it.
- The AFI itself: the build above has not been run from this branch (no Vivado here).
