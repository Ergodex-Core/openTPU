# C2 on Alex's burst200 F2 shell

This separate branch adapts Alex's actual burst200 board flow to the existing C2
microbenchmark. It uses the supplied archive (SHA-256
`fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc`), commit
`06c169cbcd1721ae8b167de992a93461697a397d`. The 33 files under `support/` are
byte-for-byte copies of that commit; `support/manifest.json` binds every file.
The earlier C2 and R5 investigations remain in their separate worktree/history.

Alex's original fabric has two native channels, two 2-GiB windows and HBM ports
0/4. C2 preserves its eight 512-MiB windows and native 64-byte beat interface.
The new `rtl/c2_alex` generator creates one nine-slave/eight-master SmartConnect:
PCIS reaches all eight windows, while native p reaches only window p through HBM
port p. This is a new implementation requiring fresh verification and timing
qualification. Historical burst200 timing, bus-skew counts and hardware results
do not qualify this C2 image.

The original private HBM/SmartConnect, OCL AXI-Lite converter, PCIS register slice,
clock wizards, resets and legal floorplan form the baseline. Core/host/HBM clocks
remain 200/250/450 MHz. The original second clock-wizard output remains generated
but has no C2 datapath consumer. No TMEM multicycles or other new timing exceptions
are applied. Eight native instances replace the original two in the SLR0 region;
PCIS and OCL retain Alex's SLR1 placement. Original implementation/refinement
recipes are preserved with at most four threads, including the leaf-clock
optimizer workaround and shell bank-74 Vref restoration.

## Initial build gates

Use the pinned aws-fpga repository root, **not** its `hdk/` child. Required revisions
are HDK `20fe90180574464fd509c994680ec67191e9759f`, IP submodule
`6d32be972e6da854e61a8d3d6ec0466ab491c1b3`, Vivado 2025.2, part
`xcvu47p-fsvh2892-2-e`, small shell `0x10212415`. The builder verifies the shell
checkpoint hash, HBM properties and all consumed external source files. Small
external files are copied into the run; all inputs are hashed before and after
each stage. IP is imported/generated only into fresh private projects.

```bash
python3 boards/c2_f2_alex/build.py --hdk /opt/aws-fpga \
  --out /own/fresh/alex-c2-ip --stage ip --plan
python3 boards/c2_f2_alex/build.py --hdk /opt/aws-fpga \
  --out /own/fresh/alex-c2-ip --stage ip
python3 boards/c2_f2_alex/build.py --hdk /opt/aws-fpga \
  --out /own/fresh/alex-c2-synth --stage synth --ip-run /own/fresh/alex-c2-ip
```

`--plan` creates no files and invokes neither Vivado nor installed HDK commands.
Execution refuses an existing output. IP generation runs memory/OCL, PCIS and
clocks sequentially; default IP synthesis concurrency is one job. `--ip-jobs 2`
or `3` is only appropriate when host resources permit it; every script limits
Vivado to four threads. The selected concurrency must match when consuming IP.

The IP receipt hashes generated source, constraints, XCI, BD and DCP products;
changed, added or missing products invalidate it. Synthesis copies the C2 RTL,
records its source revision/build ID and binds the exact IP receipt. The initial
`--stage synth` stops at `post_synth.dcp`; it does not link, route, package, publish
or touch an FPGA. Only the dedicated `opentpu-f2` instance is authorized for F2
work. Hardware execution requires current qualification, device ownership and
original-image restore.

The synthesis audit reads required hierarchy properties directly; enumerating all
properties on `u_mem` is avoided because an actual Vivado 2025.2 probe showed that
query materializing internal boundary objects. The synthesis audit checks actual
leaf connectivity and per-port clock consumers.
It covers all 170 AXI handshakes, including optimized native write-response paths
and the shared reset-gated B/R-ready network. Declared widths, protocols and
address windows are checked against the generated BD: Vivado can remove constant
bits and rebuild boundary names. The audit preserves an explicitly unqualified
diagnostic checkpoint after inspecting the live design, including on failure.
Capture ordering is recorded explicitly. Post-implementation object changes still
require review against that checkpoint's actual topology and timing reports.

## Vendor simulation and checkpoint continuation

`simulate.py` consumes an independently recorded SHA256 of `ip-complete.json`.
It copies the private IP before opening Vivado and checks all copied dependencies,
including generated headers, before and after four seed/clock-phase scenarios.
The bench exercises all eight native channels, PCIS narrow and sparse writes,
burst boundaries, response ordering, backpressure and reset with reads in flight.
Its behavioral HBM model establishes protocol correctness, not silicon bandwidth.

```bash
python3 boards/c2_f2_alex/simulate.py --ip-run /own/alex-c2-ip \
  --ip-receipt-sha256 "$IP_RECEIPT_SHA256" --out /own/fresh/alex-c2-simulation
python3 boards/c2_f2_alex/implement.py --stage link --hdk /opt/aws-fpga \
  --synth-run /own/alex-c2-synth --synth-receipt-sha256 "$SYNTH_RECEIPT_SHA256" \
  --out /own/fresh/alex-c2-link
```

`implement.py` provides separate `link`, `implement`, `refine` and `qualify` stages.
Every stage uses a fresh directory and preserves its input checkpoint and receipt
chain. Later stages require `--from-run` and `--from-receipt-sha256`; they never
resynthesize RTL. An implementation with negative timing is retained as a candidate.
Refinement must pass setup and hold timing. Qualification additionally requires
`--simulation-run` and `--simulation-receipt-sha256`.

The current independent qualifier collects current-design timing, CDC, exception,
bus-skew, routing, methodology and DRC evidence. Linked and final checkpoints
also retain `supplemental.tcldata`: current finding relationships, gray-bit data
drivers and synchronizer stages, bounded per-bit timing queries, and live waiver
queries. Unsupported, missing, and capped observations remain explicitly pending.
Final evidence verification requires this report. The qualifier deliberately returns
`pending-review` (exit 3) until the actual C2 reports support reviewed acceptance
rules. Alex's historical counts are not acceptance rules. A later executor-only
commit may consume an older hardware build only while all synthesized source
bytes still match; its executor revision is recorded separately from the hardware
build ID.

The first synthesis attempt (`264d8413`) failed its interface audit. Its
supplemental dependency observer preserved the failure. Retry `d1c20797` also failed the boundary-pin assertion; its preserved netlist
revealed the rebuilt hierarchy. Fresh retries reuse the completed private IP and capture its complete generated `.srcs` and `.gen`
trees synchronously before synthesis, comparing them afterward. This binds
include files omitted by the original IP product suffix inventory without
changing the original IP receipts. See `STATUS.md` for the actual run.

## Evidence, publication and physical sessions

`collect.py` copies the explicitly identified receipts, source snapshots and
required products into a portable bundle. `evidence.py` verifies that bundle
without running bundled code or relying on the original absolute paths. Both
require completed physical qualification; a pending review cannot become a
publishable image. `package.py` invokes the pinned installed AWS packaging script
and checks the resulting archive against the qualified checkpoint.

`tools/c2_f2_alex_publish.py` uploads the package and its complete evidence through
the AWS CLI, verifies stored checksums, and records an idempotent AFI request.
`tools/c2_f2_alex_session.py` uses the existing exclusive device lock, verifies
the candidate's provenance, runs the C2 benchmark, and restores the original
image. These tools are prepared for the current build; their local tests are not
evidence that an AFI was created or an FPGA benchmark succeeded.

C2 still uses its existing host fixture/bit-check contract. Before a physical run,
the host must complete a PCIS write visibility fence/readback before OCL START;
the previous custom fabric's cross-interface hold behavior must not be assumed
for this vendor-converter interface. Hardware Lens profiles require actual
successful, restored sessions; this builder generates no performance data.
