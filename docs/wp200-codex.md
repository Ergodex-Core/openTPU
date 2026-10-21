# Codex wide-port 200 MHz candidate

This is an isolated candidate, not a timing-closure result. The qualified 150 MHz
image and the source workspaces were not modified. No hardware was loaded.

## Source and bounded change

- Baseline checkout: `0a868822a71422beb11771c9fa046af762579e81` from
  `wide-port-alexshell`. Its RTL is identical to `ba86b5df`; it includes the legal
  MMCM recipe and the later route-recovery/host support associated with qualified
  build `417243f1`.
- Candidate branch: `codex/wp200-request-mask`, in this task's independent clone.
  The baseline verification worktree belongs to that clone, not another owner's
  repository.
- RTL candidate commit: `df01e692` (the two selected RTL files only).
- Selected source: `e5fd4eedcf0e42218b59417fd25d54a9b8b1da1a`, only
  `rtl/top/otpu_slice.sv` and `rtl/mxu/otpu_mxu.sv`. The exact selected patch is
  retained in `evidence/selected-e5fd4eed.patch`.
- No VPU, collective, floorplan, clock-constraint, exception, severity, or waiver
  change. The repaired HBM/host placement remains byte-identical to the baseline.
- `bd3df787` is a different candidate: it includes the additional VPU/collective
  changes and V-A/V-B floorplan. The review found a synthesis receipt for it, not
  a verified route. Its current remote state is unverified; its checkpoint cannot
  represent this selectively ported RTL or its 200 MHz target.

## Why these two changes

The preserved **older 150 MHz routed** `ba86b5df` reports show the MXU
`lo_ad_reg[3][0]` to TMEM `rpv_reg[0]_inv/D` path at 6.415 ns (0.849 ns logic,
5.566 ns routing), request control reaching 1,089 payload enables, TMEM pblock
CLB occupancy 97.56%, and local SLR2 congestion levels 5–6. They do not describe
the qualified image's final recovered paths or prove a 200 MHz bottleneck.

The baseline reports are under
`/Users/rachittibrewal/ergodex-03/slr2-scratch/ba86b5df/`: `to_clk_raw.rpt`,
`to_clk_f_raw.rpt`, `congestion.rpt`, and `util_pb_otpu_tmem.rpt`.
Hash-verified copies are retained in `evidence/older-ba86b5df-150mhz/`.
These are the older diagnostic reports, not the missing qualified final bundle.

1. Port-A/B payload registers load when their slot is free. The expensive issue
   condition still controls valid. If a valid request is stalled, the slot is
   not free and all its payload remains held. If a request is accepted or the
   slot is empty, the next valid payload is captured on the original edge.
   Values when valid is low are deliberately unspecified. The MXU A address
   is no longer zeroed by the issue condition.
2. The MXU write-bank mask is computed beside the drain and RMW metadata. The
   drain mask has the same reset/enable and edge as its valid/address state;
   the RMW mask has the same delay length and grant enable as `u_rw`. RMAX uses
   its existing registered lane/address state. The existing simulation assertion
   compares the mask with actual enabled write addresses every cycle. The slice
   uses this mask only for `ULANES == LANES`; other configurations retain decoding.

The RMW metadata delay has the baseline reset behavior. No independent reset
semantics were introduced for its companion mask. Reset-in-flight behavior must
therefore be verified against the baseline as well as the ISA.

## Reproducible local validation

Use the existing Python environment with NumPy/pytest and Verilator 5.048.
Set `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`OTPU_TARGET=f2-hbm`, and `OTPU_NCH=8`. Test runs use separate temporary roots,
disable pytest's shared cache, and cap Verilator compilation to four jobs by
setting `rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j') + 1] = '4'` in their launcher.

- Build/clock/validation contracts: `test_f2_build.py`, `test_target.py`,
  `test_f2_route_opt.py`, `test_f2_scoped_clocks.py`.
- ISA bit-exact wide-port suite: `test_wide_port_rtl.py`.
  `tools/wp200_regression.py --out FRESH_DIRECTORY` runs it with the settings
  above and flushes each setup/call/teardown receipt, the collected test list,
  source/tool fingerprints, final JUnit and exit status to that directory.
- Replay/reset/width selection: `test_rtl.py -k
  'replay or porta_host_writes or lane_count'`.
- `tools/wp200_equivalence.py` runs the actual baseline and candidate RTL in
  separate processes. It generates a simulation-only monitor, retains exact
  valid request payloads and grant/replay/reset cycles, and compares all DRAM
  and TMEM against the ISA. It uses PORTB 4, MCOLS 8, ALINE 64, pumped TMEM,
  eight native AXI channels, three crossing stages, waiting eight-beat bursts,
  and 32 outstanding transactions. Cases include 0/40/70% memory-model stalls,
  multi-row replay, and reset during pending port-B, drain, and RMW activity.
  `--a-throttle` adds a test-only gate at both sides of the A handshake, admitting
  one request per 16 cycles to exercise held valid A payloads. The generated top
  harness is separate from production RTL. Reset release follows the board's
  `go_ok` rule: outstanding reads and writes must drain before restart.
  Compare two result files with `--compare BASE_JSON CANDIDATE_JSON`.

JUnit XML, logs, exact transaction traces, and final result JSON are in
`evidence/`. A skipped or unfinished case is not a passing case. Simulation
clocks and the memory latency model are functional test settings, not physical
timing or throughput measurements.

The initial reset probe used a fixed 32-cycle reset interval and failed on the
untouched baseline when a late A response arrived after restart. That stimulus
omitted the board's drain-before-restart protocol (`otpu_board.sv` around
`go_ok`). The retained initial log is `evidence/equivalence-baseline.log`;
the corrected baseline run is `equivalence-baseline-v2`. No production reset
logic or assertion was changed or suppressed to obtain the corrected result.

Completed validation: **491 pytest tests passed, one skipped, zero failures or
errors**. This consists of 308 build/clock/validator contract tests (one
card-specific test skipped), 15 replay/reset/width tests, and all 168 wide-port
RTL/ISA/cycle regression tests. The wide-port run completed with exit code zero;
all 504 setup/call/teardown receipts passed, and its source fingerprint matches
the final files. All 18
baseline/candidate scenarios were ISA-exact and identical in transaction cycles,
payloads, grant/replay/reset traces, DRAM and TMEM. Across those scenarios the
monitor observed 22,335 stalled A cycles, 745 stalled B cycles, 19,103 MXU write
stall cycles, 11 replay cycles, and six resets. This is simulation evidence, not
formal equivalence or physical timing proof.

Two broad wide-port runs lost their execution sessions after 87 passing progress
indicators and produced no final JUnit result. Process inspection confirmed the
candidate job and its compiler children had exited before each restart. The
replacement uses the existing build cache and durable per-test receipts under
`evidence/pytest/wide-port-journal/` and completed all 168 tests. Only that final
run counts toward the validation total. Both interrupted logs remain retained.
The machine-readable result is `evidence/validation_summary.json`.

The original equivalence result JSON records the baseline HEAD plus the then
uncommitted candidate changes. Its tested RTL contents are identified by the
SHA-256 hashes in `evidence/source_manifest.json`; the production files committed
in `df01e692` have those exact hashes. Test-result files are retained unchanged.
The earlier owner's 281-pass/one-skip and 1,027,452-cycle model receipts are not
counted as this candidate's verification. The local simulator/compiler versions
are retained in `evidence/environment.json`; the original source workspaces
were rechecked clean at their original commits in `evidence/workspace-state.json`.

## Single physical experiment after access/capacity verification

Authorized host: **opentpu-m8a-build**, `i-0ec8d33362b3100be`, `us-east-1`,
verified by EC2 tags as `m8a.16xlarge`. The only later hardware host is
**opentpu-f2**, `i-0c78de2253f66519d`, verified as `f2.6xlarge`. No other host,
instance lifecycle, permissions, credentials, security settings, or billing
operation is included.

Before any upload or build, verify current processes, owners, capacity, run
directories, and whether a matching candidate already exists. Do not stop or
crowd another owner. Use a new task-owned remote source/output directory and
record the source commit, complete source hashes, instance identity and command.
The remote HDK directory is not yet verified; do not guess it.

The intended flow is the existing pinned AWS HDK non-project flow through
`boards/f2/build.py`, with exactly one synthesis and one implementation:

Use a clean checkout of RTL commit
`df01e692f0131ce45939a0c61dd1379570ea82a2` for that build. Its default build tag is
`df01e692`; later commits on this branch only retain verification tooling and
evidence. Record both the full commit and the emitted provenance file.

```sh
python boards/f2/build.py --hdk "$VERIFIED_HDK" \
  --target f2-hbm:p4rl8a64 --out "$FRESH_RUN/synth" \
  --flow SynthCL --place-directive Default

# Only after inspecting successful synthesis, warnings and the checkpoint digest:
python boards/f2/build.py --hdk "$VERIFIED_HDK" \
  --target f2-hbm:p4rl8a64 --out "$FRESH_RUN/impl" \
  --flow ImplCL --from-synth "$FRESH_RUN/synth" \
  --synth-sha256 "$VERIFIED_SYNTH_SHA256" --place-directive Default
```

`Default` and the existing HDK AggressiveExplore strategies are the documented
150 MHz baseline settings. Do not use the CLI default SSI_SpreadLogic_high by
accident. Do not import the old 150 MHz incremental checkpoint or unreviewed QoR
suggestions across the changed clock target. The qualified final recovery used
additional implementation steps; that complete recipe/bundle is still missing
locally, so this first experiment is not an exact replay of its final placement.

Required identity: part `xcvu47p-fsvh2892-2-e`, Vivado 2025.2 (record exact build),
HDK commit `20fe90180574464fd509c994680ec67191e9759f`, IP commit
`6d32be972e6da854e61a8d3d6ec0466ab491c1b3`, small shell `0x10212415`.
`HBM_CORE_MMCM[200000]` is `(5, 24.0, 6.0, 3)`: core 200 MHz, TMEM 400 MHz,
HBM AXI 450 MHz, shell 250 MHz. Keep the floorplan fixed for this experiment.

## Evidence needed before a closure claim

One identified, fully routed checkpoint must have a hash and complete
source/constraint/tool/clock provenance. Preserve the existing route validator
and retain all its reports. Independently review setup/recovery, hold/removal,
pulse-width/minimum-period results at all applicable corners (WNS/TNS, WHS/THS,
WPWS/TPWS), unconstrained paths and clock/I/O coverage, actual clock frequencies,
exceptions, CDC/reset crossings, bus skew, route status, DRC and methodology.
The validator's marker is necessary but does not replace that report review.
Missing evidence or an unresolved negative slack prevents a closure claim.

Retain per-clock and cross-clock detailed max/min paths, logic/routing splits,
fanout and coordinates, congestion windows, SLR/hierarchy/pblock utilization,
clock interaction, ignored exceptions and applied QoR/directive provenance.
Use those paths to decide the next bounded experiment; do not start a sweep.
AFI creation, packaging for deployment, FPGA loading, inference and measured
throughput are separate later stages.

The missing qualified final bundle begins at:
`/home/ubuntu/opt/hwloop-nightly/as150-bdf0b7ae/rqs-spread2-mmcm/cl_opentpu/build/reports/cl_opentpu.417243f1.route_timing.rpt`.
Also needed are validation, congestion, utilization, CDC, bus-skew and checkpoint
provenance from that same run.

## Access blocker

EC2 identity reads succeeded. SSM inventory returned no managed-node entry for
the builder, and no commands for it in the queried period. The read-only SSH
capacity/ownership request to `ubuntu@54.174.195.152:22` was **denied before
execution by automatic approval review**, citing the instruction against
retrying denied remote access. No SSH command executed, no outcome is uncertain,
and no retry or alternate execution route was attempted after rejection.
Current builder capacity/ownership and tool/HDK paths remain unverified.
Physical synthesis/implementation was not launched.
