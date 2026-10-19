# C1 F2 build

The first isolated build synthesizes the existing C1 16-engine, four-slot kernel
with Vivado 2025.2 for `xcvu47p-fsvh2892-2-e`, using a 200 MHz target clock. It
retains exact source and output hashes. This resolves FPGA inference and resource
questions before the new native/HBM adapter enters the shell build.

This OOC stage is not shell-linked, routed, deployable, or a hardware performance
measurement. It cannot create an AFI or access the FPGA. The host launch uses
only `opentpu-f2` (`i-0c78de2253f66519d`), two CPU cores, 32 GiB memory, and a
one-hour runtime bound. C2 has priority for actual hardware sessions.

The following isolated integration will keep the C1 row-engine arithmetic and
ordered retirement. Sixteen logical engines will share eight 64-byte native
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
