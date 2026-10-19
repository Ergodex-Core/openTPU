# C2 isolated F2 HBM kernel

This build replaces the production TPU in the F2 shell with the C2 benchmark
kernel. It uses one 512-product datapath and eight HBM pseudo-channels. It does
not implement a production opcode or full-model inference.

`support/manifest.json` pins every reused F2 source byte to tpubench commit
`67dcf0c815fd7a5314730c019ffcc1f3083704d1`. These are committed sources, without
changes from another agent's working tree. `build.py` verifies those hashes and
creates a new snapshot for each run. Existing run directories are rejected.

On the licensed Vivado 2025.2 host, with the pinned AWS HDK and generated HBM IP:

```sh
source /opt/Xilinx/2025.2/Vivado/settings64.sh
python3 boards/c2_f2/build.py --hdk /home/ubuntu/aws-fpga \
  --out /path/to/new-c2-synthesis --flow SynthCL
python3 boards/c2_f2/build.py --hdk /home/ubuntu/aws-fpga \
  --out /path/to/new-c2-full-build --flow BuildAll --package
```

`--prepare-only` performs the source/HDK checks and writes the snapshot and
invocation without running Vivado. `inputs.json` records the C2 commit, dirty
status, frozen support revision, tool invocation and SHA-256 of every input.
`outputs.json` records the completed checkpoint and any qualified AFI package.
The device BUILD_ID is the C2 commit's first eight hexadecimal digits; commit
validated RTL before preparing a hardware build.

The generated system preserves the frozen PCIS/OCL/HBM fabric and raw PC map,
and synchronizes HBM readiness into the core clock. ACT/scales are resident;
native channels only read weights. Results retire into local readback RAM.
The measured interval therefore excludes PCIe setup/readback and HBM output
writes. Both benchmark modes read the same weight bytes at the same addresses.
See [the ABI and host commands](../../docs/v4-codex/F2-BENCH.md).

The image targets a 200 MHz core and 450 MHz HBM interfaces. These are build
targets until the shell-linked design passes placement, routing, setup/hold,
bus-skew, DRC and methodology checks. The dedicated HBM route validator also
requires six clock-crossing bounds and all 39 expected fabric gray-bus
constraints. The production design's DDR and pumped-TMEM rules do not apply to
this kernel; its shell methodology classifier remains unchanged and rejects
unknown violations. No packaging or hardware-performance claim is made from
an out-of-context result alone.

The build command creates no cloud resources, submits no AFI and changes no
FPGA slot. A qualified image must be submitted and loaded separately before
the host benchmark can run. The host checks the benchmark ABI and exact BUILD_ID,
takes the canonical exclusive device lock, verifies preload readback, compares
every output with the canonical reference, and checks native traffic counts
before writing hardware Lens profiles.
