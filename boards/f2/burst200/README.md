# AWS F2 burst200

This port wraps openTPU in the AWS F2 Small Shell and connects its two native
memory channels to HBM. It preserves the accelerator RTL and adds a burst-read
adapter, HBM address normalization, vendor SmartConnect fabric, clock/reset
crossings and a host transport for OCL/PCIS BAR access.

The branch is based on **`7e42a626f714edd07867702c38c954082d51f587`**, the compute
revision used by the successful burst200 hardware experiment. This commit is
in the Ergodex fork's history. Later changes on the fork's `main` are not
included; their compute changes require new implementation and hardware
qualification before carrying over these performance claims.

## Contents

| Path | Purpose |
|---|---|
| `rtl/boards/f2/burst200/` | CL top, native-to-AXI burst adapter, HBM address adapter |
| `targets/f2-burst200.toml` | Tested architecture and 200 MHz core target |
| `boards/f2/burst200/constraints/` | AWS-legal floorplan, related-TMEM timing, platform constraints |
| `boards/f2/burst200/scripts/` | IP/clock generation, synthesis/link, implementation, refinement, audit, packaging and hardware tools |
| `opentpu/host/f2_transport.py`, `f2_mmio.c` | Host control/memory transport and explicit-width MMIO |
| `tests/f2_burst200/` | Randomized RTL and file-backed host-transport tests |
| `docs/f2-burst200/` | Clock, memory and read-burst schematics |

No AMD-generated IP products, DCPs, bitstreams, shared libraries, model weights,
credentials, or experiment logs are committed. Build outputs are generated
under ignored `build/f2/` directories. Scripts accept paths to externally
installed tools; no files in the installed HDK are modified.

## Clocking and bandwidth

| Interface/domain | Frequency | Combined data width | Theoretical throughput |
|---|---:|---:|---:|
| Shell / host PCIS | 250 MHz | 512 bits | Host-dependent |
| Accelerator native channels | 200 MHz | 2 × 512 bits | **25.6 GB/s** |
| Pumped TMEM | 400 MHz | Internal to the accelerator | Path-dependent |
| HBM AXI3 ports 0 and 4 | 450 MHz | 2 × 256 bits | **28.8 GB/s** |
| HBM reference / APB | 100 MHz | — | — |

The shell's 250 MHz clock drives one MMCM producing related, zero-phase
200/400 MHz core/TMEM clocks. A separate MMCM converts the shell's 100 MHz
HBM reference to 450 MHz HBM AXI. SmartConnect handles clock crossing and
512/256-bit conversion.

The two HBM ports can return **64 B per HBM AXI cycle**. The native channels
can return **128 B per core cycle**. These are different clock domains.
Direct MXU weight streaming measured about 23.7 GB/s; the native channels'
25.6 GB/s ceiling limits this design before the HBM ports' aggregate peak.
DMA transfers into TMEM have a separate **32 B/core-cycle** ceiling.

Ports 0 and 4 serve two 2 GiB windows, at native AXI addresses 0 and 0x80000000.
The five-port AWS wrapper has ports 1–3 tied off. Existing CHASH interleaving
is preserved. The design exposes 4 GiB of the physical 16 GiB HBM.
Reads coalesce into up to 16 native 64-byte beats, bounded by 4 KiB and
response credits; downstream fabric converts/splits them. Writes remain
single native beats, with read/write direction barriers preserving ordering.

Diagrams: [clocks](../../../docs/f2-burst200/05_clocks.svg),
[memory](../../../docs/f2-burst200/03_memory.svg),
[burst adapter](../../../docs/f2-burst200/08_read_bursts.svg).
Their source footers retain the original experiment workspace paths.

## Build

Requirements:

- Licensed **Vivado 2025.2**, with VU47P and HBM support.
- A complete AWS FPGA HDK with **`small_shell=0x10212415`**, including
  `hdk/common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp`.
- Linux, Python with openTPU's dependencies, and sufficient build memory/disk.

Run from the repository root after installing the Python dependencies and
loading the Vivado environment:

```bash
python3 boards/f2/burst200/build.py --hdk /path/to/aws-fpga/hdk --plan
python3 boards/f2/burst200/build.py --hdk /path/to/aws-fpga/hdk
```

Use `--output /path/to/new/experiment` to select a fresh directory and
`--vivado /path/to/vivado` to select the executable. `--target targets/f2-burst200-s3.toml`
builds the three-slice variant (one slice per SLR on six HBM ports, its top in
`rtl/boards/f2/burst200-s3/`, `constraints/floorplan_s3.tcl`; `docs/v4/f2-burst200-s3.md`);
the default target is this one-slice design and its flow is unchanged. Existing output
directories are refused. `--floorplan`, `--place-directive`, `--phys-directive` and
`--route-directive` select an implementation variant; `--from-netlist RUN` implements
an earlier run's linked netlist and skips stages 1 and 2, refusing a netlist built from
other sources; `--incremental-from DCP` implements incrementally from a routed checkpoint
(`docs/v4/f2-burst200-s3.md`). The driver saves source hashes, commands and logs,
then performs these stages sequentially:

1. Generate memory/HBM, PCIS and clock IP in private projects.
2. Synthesize the CL and link the real AWS shell checkpoint.
3. Apply the legal floorplan and timing constraints, then implement.
4. Refine the routed candidate and require passing setup/hold timing.
5. Reopen the final checkpoint and generate independent qualification reports.

The historical sequence was E05/E06/E09 → E50 → E55 → E64 → E67.
The build scripts incorporate the recovery needed by those experiments:
inline HBM IP is not launched as a separate OOC synthesis run; the first
implementation candidate may have negative timing and must pass through
the mandatory final refinement. A candidate checkpoint is not an accepted
image. Vivado's failing leaf-clock optimizer is disabled, and shell bank
74 Vref is restored to 0.84 V after placement. Related-TMEM multicycles
are retained; writes and phase paths remain timed at 2.5 ns. No timing
waivers are added.

The checked-in CL top is the hardware-tested wrapper. To regenerate it after
an intentional configuration/wiring change, supply HDK interface headers:

```bash
python3 boards/f2/burst200/scripts/generate_f2_top.py \
  --hdk /path/to/aws-fpga/hdk --output build/f2/cl_opentpu.sv
```

Changing `targets/f2-burst200.toml` alone does not update the checked-in wrapper.
Regenerate and review it before rebuilding with changed parameters.
The default generator output and emitted configuration have been verified
byte-for-byte against E50's tested inputs.

Rebuilding does not guarantee identical routing or timing closure. Review
the newly generated DRC, timing, route, CDC, bus-skew and methodology reports
before packaging. `scripts/package_f2.py` uses the official HDK packager
and requires a checksum-bound review JSON; see its `--help`. It packages
locally. Image generation and loading are separate AWS operations.

## Host support

The host uses the AppPF's OCL BAR0 and PCIS BAR4; Small Shell provides no
XDMA engine. Compile the MMIO helper locally on the x86-64 host:

```bash
gcc -std=c11 -Wall -Wextra -Werror -O2 -shared -fPIC \
  opentpu/host/f2_mmio.c -o opentpu/host/libf2_mmio.so
```

Set `OTPU_F2_BDF` to the FPGA AppPF BDF (the historical default is
`0000:34:00.0`). The process needs permission to map that function's
`resource0`, `resource4`, and optionally `resource4_wc` sysfs files.
The transport checks the AWS PCI vendor ID and openTPU register identity
before accessing HBM and uses the existing device-lock protocol.

Applications can construct `Board(transport=F2Transport())` after importing
`F2Transport` from `opentpu.host.f2_transport`. The companion runner selects
this transport for existing repository scripts:

```bash
python3 boards/f2/burst200/scripts/f2_run.py tools/validate.py --help
```

The runner defaults `OTPU_TARGET` to `targets/f2-burst200.toml` and puts local runtime
state/caches in `build/f2/`. It operates on an already loaded FPGA image;
it does not program the FPGA. A direct hardware smoke test is:

```bash
python3 boards/f2/burst200/scripts/smoke_f2.py --output build/f2/smoke.json
```

The smoke test writes HBM test regions and runs an instruction-driven copy.
Run with exclusive access between model sessions. It checks image identity,
calibration, measured core frequency, raw/hashed addresses and TMEM/DMA.
The `benchmark_f2_*.py` and `benchmark_qwen3.py` scripts provide subsequent
bandwidth/model tests; consult `--help` and their smoke/validation prerequisites.

For faster reads, `OTPU_F2_READ_MODE=stream` requires SSE4.1 and a WC mapping.
`OTPU_F2_READ_THREADS` accepts 1–16 threads. Scalar reads are the default.
`OTPU_F2_MMIO_LIBRARY` overrides the helper library path.

## Tests

Host tests compile the helper into a temporary directory and use file-backed
fake BARs; no FPGA or root permission is required:

```bash
python3 -m pytest -q tests/f2_burst200/test_transport.py tests/test_target.py
python3 boards/f2/burst200/simulate.py --output build/f2/native_sim
```

The second command requires Icarus Verilog and runs 16 combinations of burst
length, outstanding credits and random seeds. It checks stalls, data/masks,
direction ordering, burst boundaries and injected response errors. To test
the actual Vivado-generated SmartConnect/CDC/width-conversion fabric:

```bash
python3 boards/f2/burst200/simulate.py --fabric --output build/f2/fabric_sim \
  --ip-run build/f2/burst200/ip --pcis-run build/f2/burst200/pcis
```

The fabric test uses a functional HBM-side AXI memory model; it does not
model the analog HBM PHY or establish hardware bandwidth.

## Qualification and historical results

The successful 2026-10-15 implementation had setup/hold slack
**+0.032/+0.010 ns**, zero failing setup/hold/pulse-width endpoints,
272,732 fully routed nets and zero routing errors. All 49 bus-skew checks
passed. DRC had no errors/critical warnings; CDC and methodology findings
were reviewed against the unchanged shell rather than waived.

Historical final DCP SHA-256 (artifact is not committed):
`0f600fd7fecd2028a6db7767d1de5a2230e62c265baa545b749866b9e9ff15ed`.
The tested image was `agfi-0d06cb48544a02106`.

Controlled single-beat/burst testing kept clocks, compute RTL, model kernels,
caches and inputs fixed. Direct weight-stream bandwidth rose from 6.39 to
23.54–23.87 GB/s. Qwen3-0.6B cached/tiled int8 device/wall decode was
33.68/26.99 tok/s; FP4 with int8 head was 47.21/35.00 tok/s. Complete
65-token sequences matched the original image, with bit-exact ISA checks.
These are historical hardware measurements, not a fresh qualification of
this repository-layout refactoring. The branch's host/target tests and seeded
adapter simulations pass; a fresh full Vivado build remains to be run.

Raw reports and checkpoints are retained separately in the experiment handoff.
The repository contains the sources and recipes needed to generate new ones.
