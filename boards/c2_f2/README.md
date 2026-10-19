# C2 isolated F2 HBM kernel

This build replaces the production TPU in the F2 shell with the C2 benchmark
kernel. It uses one 512-product datapath and eight HBM pseudo-channels. It does
not implement a production opcode or full-model inference.

`support/manifest.json` pins every reused F2 source byte to tpubench commit
`fbf21635d7ac7ab4633eaa0953dbc39274dcb690`. These are committed sources, without
changes from another agent's working tree. `build.py` verifies those hashes and
creates a new snapshot for each run. Existing run directories are rejected.
The former Round 3 snapshot remains in git history and its failed `7028e35e`
run remains evidence: routed WNS −4.024 ns, 11,841 failing endpoints, with
independent CSR and custom 450 MHz host-path failures. No image was qualified.

This snapshot adopts Round 5: eight `otpu_native_axi` masters and the host/port
arbiters at 200 MHz, each port behind the HDK's `cl_axi_sc_1x1` SmartConnect for
512-bit AXI4 to 256-bit AXI3 and the crossing to 450 MHz. Native channel and BAR4
addressing, kernel ABI, and both benchmark modes are unchanged. The old native
450 MHz bridges are absent. This follows the architecture of the routed burst200
image; the eight-port implementation still needs its own physical validation.

The generated top corrects the frozen core MMCM's illegal `M=4.8, D=1`
attributes to `M=24, D=5`. From the 250 MHz shell clock, this produces a
50 MHz PFD, 1200 MHz VCO and unchanged 200/400 MHz outputs (dividers 6/3).
These values satisfy [AMD UG572 attribute increments and ranges](https://docs.amd.com/r/en-US/ug572-ultrascale-clocking/MMCM-Attributes)
and [DS923 frequency limits](https://docs.amd.com/r/en-US/ds923-virtex-ultrascale-plus/MMCM-Switching-Characteristics)
for the target Virtex UltraScale+ -2 device. The frozen support and 450 MHz HBM
MMCM remain byte-identical. Snapshot creation checks both MMCM recipes with
exact arithmetic; the first full-shell build stopped at `AVAL-168` before
placement. The corrected old-fabric run reached routing but failed setup; this new
fabric must pass every physical qualification gate independently.

On the licensed Vivado 2025.2 host, with the pinned AWS HDK and generated HBM IP:

```sh
source /opt/Xilinx/2025.2/Vivado/settings64.sh
/home/ubuntu/opentpu-venv/bin/python boards/c2_f2/build.py --hdk /home/ubuntu/aws-fpga \
  --out /path/to/new-c2-synthesis --flow SynthCL
```

`--prepare-only` performs the source/HDK checks and writes the snapshot and
invocation without running Vivado. `inputs.json` records the C2 commit, dirty
status, frozen support revision, implementation tuning and SHA-256 of the frozen
CL inputs; `invocation.json` holds the exact command and environment. It also
records SmartConnect interface metadata and all `.bd/.v/.sv/.vhd/.xdc/.xci/.dcp/
.mem/.dat/.coe` products under the installed source-BD and generated-BD trees.
Source products up to 1 MiB each are preserved in `vendor-inputs/`; checkpoints
and larger products retain exact hashes and sizes. Logs and unrelated cache
files are excluded. The builder checks pinned HDK/IP revisions (and the shell
DCP for a full build), SmartConnect shape and reviewed top-level BD/wrapper
bytes, then rechecks the complete selected inventory before and after Vivado.
Generated dependencies remain at installed HDK paths; evidence copies are not
used as relocated vendor IP.
`outputs.json` records the completed checkpoint and any qualified AFI package.
The device BUILD_ID is the C2 commit's first eight hexadecimal digits; commit
validated RTL before preparing a hardware build.

The generated system preserves the pinned Round 5 fabric and raw PC map,
and synchronizes HBM readiness into the core clock. ACT/scales are resident;
native channels only read weights. Results retire into local readback RAM.
The measured interval therefore excludes PCIe setup/readback and HBM output
writes. Both benchmark modes read the same weight bytes at the same addresses.
See [the ABI and host commands](../../docs/v4-codex/F2-BENCH.md).

The first migration gate is **SynthCL only** on dedicated `opentpu-f2`
(`i-0c78de2253f66519d`). The image still targets 200 MHz core and 450 MHz HBM;
these are targets until shell-linked placement, routing, setup/hold, bus-skew,
DRC and methodology pass. The initial floorplan is Round 5's four remaining
pblocks after removing the unused TMEM box: SLR0, the fabric bottom row, PCIS,
and the SLR0 link endpoints. Recommended tuning is ExtraNetDelay_high placement,
AggressiveExplore physical optimization, Explore routing and one retiming pass;
all Vivado phases are bounded to four threads.

The original Round 5 synthesis stopped because a Tcl glob misclassified valid
`g_sc[7].u_sc/...` vendor registers. C2 transforms only its generated constraints
to use anchored regular expressions for exactly ports 0–7, the HBM wrapper, and
reset bits 0–2. Tests include the actual failing hierarchy and misleading names.
The custom CDC inventory is exactly three host FIFOs and one host write counter
(seven skew constraints); SmartConnect CDC is a separate vendor inventory.
The old 47-entry fabric gate cannot qualify this design. Full-build packaging
remains blocked until actual new synthesis/post-opt CDC and clock evidence is
reviewed. No constraint, clock or timing severity is relaxed by this migration.

The dedicated host's reviewed SmartConnect metadata has two clocks, one slave
and one master, 64-bit addresses, AXI4 ID16/512-bit and AXI3 ID0/256-bit interfaces;
read/write outstanding limits are 32/64. The slave port is named `aclk_250` and
its metadata says 250 MHz; C2 drives it at 200 MHz. Metadata stays unchanged, and
actual generated timing and vendor CDC require audit before implementation.
The independently inventoried input hashes are:

- BD: `af85051845af4743b3381be1cb65b089ab05a6dc9f1f8ac970492223ac802d89`
- wrapper: `43dd040c56dfeb49d4b9e72ddee9e46be54bea9f381f5522af23907b8cdd091a`

The build command creates no cloud resources, submits no AFI and changes no
FPGA slot. A qualified image must be submitted and loaded separately before
the host benchmark can run. The host checks the benchmark ABI and exact BUILD_ID,
takes the canonical exclusive device lock, verifies preload readback, compares
every output with the canonical reference, and checks native traffic counts
before writing hardware Lens profiles.
