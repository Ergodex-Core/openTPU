# C1 F2 build

The first isolated build synthesizes the existing C1 16-engine, four-slot kernel
with Vivado 2025.2 for `xcvu47p-fsvh2892-2-e`, using a 200 MHz target clock. It
retains exact source and output hashes. This resolves FPGA inference and resource
questions before the new native/HBM adapter enters the shell build.

This OOC stage is not shell-linked, routed, deployable, or a hardware performance
measurement. It cannot create an AFI or access the FPGA. The host launch uses
only `opentpu-f2` (`i-0c78de2253f66519d`), two CPU cores, 32 GiB memory, and a
one-hour runtime bound. That OOC job started before the user selected the
dedicated `m7a.8xlarge` CPU builder for subsequent builds. Only `opentpu-f2` is
authorized for eventual FPGA execution. C2 has priority for hardware sessions.

The isolated integration keeps the C1 row-engine arithmetic and ordered
retirement. Sixteen logical engines share eight 64-byte native
ports, preserving per-row response tags and bounded reservations. Alex's frozen
burst200 adapter, HBM geometry and private vendor-IP generation provide the shell
boundary. The first measurement is a matrix microbenchmark; full-token/ISA
integration remains separate.

## Native microbenchmark

`rtl/c1/otpu_c1_hbm_kernel.sv` now wraps the existing C1 arithmetic with an
independent AXI-Lite controller and `otpu_c1_native` memory adapter. It uses 16
engines, four row slots per engine, and 32 reserved 64-byte receive entries per
engine. Physical PC `floor((row % 16)/2)` holds a row at local byte offset
`(floor(row/16)*2 + row%2) * KB * 128`. The configurable base is in 64-byte beats.
Two engines share each PC; arbitration changes after at most 16 accepted native
commands when the other engine can progress. Each accepted command retains its
row, slot, generation and group. Its reserved entry is released only after both
32-byte halves reach that engine. Per-PC return order follows the frozen Alex
ID-zero native bridge contract.

Every 64-byte native read supplies two successive 32-byte groups from one row.
No half is discarded or fetched again. Thus `0x48 * 64 == 0x4c * 64 ==
0x40 * 32 == N * KB * 128` at successful completion. The raw native and useful
weight ceilings are both `8 * 64 * 200 MHz = 102.4 GB/s`; the eight 32-byte HBM
interfaces at 450 MHz total 115.2 GB/s before protocol losses. Alex's frozen
native bridge collects at most 16 beats, then cannot collect another beat during
the AR handshake cycle (`take_read` requires `!pending_valid`). Consequently,
even contiguous, always-ready traffic has a sustained request-generation ceiling
of `102.4 * 16 / 17 = 96.38 GB/s` before row tails, memory, credits or compute
stalls. These are source-derived ceilings, not measured HBM bandwidth or
complete-model token throughput.

C1's native ports only read weights; host PCIS retains write access for preload.
The structural audit therefore proves all 16 native AW/W valid inputs are tied
to GND at the actual SmartConnect input slices, while checking live AR/R and
B/error paths, the reset-gated B/R ready connections, and all host/HBM handshakes.
It does not require the optimized-away native write aliases to remain active.

The first ABI permits N=1..512, KB=1..128, and origin-zero plain output. ACT and
scales are preloaded through OCL and retained on chip. Header/scale HBM traffic,
TMEM, ACC, instruction dispatch and full-token execution are outside this
microbenchmark. This deliberate boundary can validate the actual C1 weight
memory path without claiming a complete v4 implementation.

CSR layout follows the benchmark control convention: magic `0x43314d42` at 0,
ABI 1 at 4, build ID at 8; START/ABORT/CLEAR at 0x10; mode must be 0 at 0x14;
KB/N/base at 0x18/0x1c/0x20; word-addressed MEM_ADDR/MEM_WDATA/MEM_RDATA at
0x24/0x28/0x2c. ACT starts at 0, packed four-byte da words at 0x10000, Sa at
0x11000, dw at 0x20000 + row*128, and Sw at 0x40000 + row. Result fp32, integer
low32 and integer high16 windows are 0x50000, 0x60000 and 0x70000.

Counters at 0x38/0x3c measure COMMAND through final result, excluding OCL preload,
ACT snapshot copy and result readback. 0x40 counts accepted 32-byte weight groups;
0x44 counts integer rows; 0x48/0x4c count 64-byte native reads issued/returned.
0x54 counts cycles blocked on native command ready, and 0x58 counts running
cycles with neither an accepted weight group nor final fp32 output. These are
observations, not exclusive causes of stalls. Per-PC 0x100 + PC*32 exposes
issued, returned, peak pending, command-blocked and credit-blocked counters;
the reserved latency field is zero. Before a real run the host must fence/read
back PCIS uploads before OCL START. ABORT stops new commands, drains all accepted
reads, and flushes the kernel before reporting DONE.

Validation:

```
python -m pytest -q tests/test_c1_hbm.py tests/test_c1_alex_build.py tests/test_c1_alex_top.py
python -m tools.c1_hbm_profile
```

The native RTL tests cover actual integer/fp32 results, randomized command/return
service, delayed PC0, row tails, N512, KB128, exceptional scales, error injection,
and abort/drain/restart. The separately retained native-adapter bench tests
response tags/data and command stability under backpressure. Generated Lens
profiles are marked synthetic native service and cannot be read as FPGA results.

`build.py` creates private IP then synthesizes the shell wrapper into a separate
output tree. It verifies the pinned Alex support, AWS HDK/IP revisions, shell
checkpoint, generated address map, source bytes and private IP products. Shell
linking, placement, routing and independent qualification remain separate gates.
No builder function publishes or loads an image.

## Host adapter

`opentpu/c1_f2.py` prepares canonical fixtures and checks the C1 ABI, arithmetic,
per-PC traffic, both halves of every native beat, and the two-engine receive
reservation limit. `opentpu/host/c1_f2.py` retains the canonical exclusive F2 lock,
checks C1 image identity before writes, and uses WC writes followed by MFENCE
and an uncached PCIS read. The client also reads back every uploaded weight bank
before START. Its clock field describes the configured clock, not a measurement.

```bash
python3 tools/c1_f2_bench.py --fixture /path/to/fresh/c1-matrix.npz --rows 17 --kb 9
python -m pytest -q tests/test_c1_f2_host.py
```

This CLI only prepares fixtures. Its `checked_runs` library function accepts a
transport supplied by a future qualified session controller; it has no image
loader or standalone hardware execution command. Host tests compare all native
bank bytes and OCL preload words against the actual RTL simulation fixture,
including row tails and negative/exceptional arithmetic. Raw counter records
retain their transport evidence kind. Hardware Lens/session integration remains
behind current physical qualification; test records cannot claim FPGA results.

## Checkpoint continuation and vendor simulation

`implement.py` consumes explicit SHA-256 values for the completed synthesis and
each previous stage receipt. It links the exact AWS shell, implements and
refines in separate fresh directories, preserving failed attempts and negative
timing candidates. The executor can be a later tooling-only revision; all
synthesized RTL, generators, IP, installed inputs and baseline recipes must
still match the original hashes. It never resynthesizes or accesses an FPGA.

```bash
python3 boards/c1_f2_alex/implement.py --stage link --hdk /path/to/aws-fpga \
  --synth-run /path/to/c1/synthesis --synth-receipt-sha256 "$SYNTH_SHA256" \
  --out /path/to/fresh/c1-link
python3 boards/c1_f2_alex/simulate.py --ip-run /path/to/c1/private-ip \
  --ip-receipt-sha256 "$IP_SHA256" --out /path/to/fresh/c1-simulation
```

The vendor simulation uses the current C1-generated 9SI/8MI SmartConnect and
PCIS IP, copied before simulation. Four actual Vivado/XSim scenarios cover all
eight native channels, narrow and sparse PCIS writes, burst boundaries, reset
with outstanding reads, and response conservation. Its HBM model is functional,
so this is protocol evidence rather than a memory-speed measurement. C1 kernel
arithmetic is checked separately by the native RTL tests above.

`--stage qualify` additionally requires the current vendor-simulation receipt
and collects physical timing, CDC, exception, bus-skew, routing, methodology and
DRC evidence. The collector deliberately retains `route_qualified=0` pending
review of the actual C1 records. No historical Alex or C2 qualification result
is transferred to C1. There is no packaging, publication or hardware loader in
this continuation.

## Recorded build milestones

At 2026-10-19 22:51:43 UTC, the isolated E16/S4 synthesis from `de1f5ac5` completed:
120,213 LUTs (73,043 logic, 46,848 LUTRAM and 322 SRLs), 23,994 FFs, six DSPs,
and no BRAM/URAM. The internal pre-placement estimate at 200 MHz has +1.802 ns
setup slack and +0.054 ns hold slack; its external ports lack I/O constraints,
so this does not establish shell timing. The retained DCP SHA-256 is
`d3e5679146b7e16600af1e7cc128b49aa50897dc03fde0ad9b52084d65f28a99`.

At 22:59:59 UTC, source `b2f45962` began the private-IP/shell synthesis build on
dedicated m7a builder `i-06cf8792fb8f15534`, unit
`otpu-c1-alex-b2f45962-ip-synth.service`, invocation
`57e2b5eb10fa481c8244496f85144404`. The bound is eight CPU cores, 80 GiB RAM and
four hours, with one IP synthesis job and four Vivado threads. This entry
records launch only; completion and FPGA results require later receipts.
