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
or touch an FPGA. Generated link/implementation/refinement recipes remain for
review before subsequent build gates are enabled. Only the dedicated `opentpu-f2`
instance is authorized for later F2 work. Hardware execution remains separately
subject to actual new qualification, device ownership and original-image restore.

C2 still uses its existing host fixture/bit-check contract. Before a physical run,
the host must complete a PCIS write visibility fence/readback before OCL START;
the previous custom fabric's cross-interface hold behavior must not be assumed
for this vendor-converter interface. Hardware Lens profiles require actual
successful, restored sessions; this builder generates no performance data.
