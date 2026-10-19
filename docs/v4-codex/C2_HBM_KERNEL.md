# Isolated C2 HBM benchmark kernel, ABI 1

The user approved C2 FPGA bring-up and openTPU Lens measurement. This kernel is a
small matched M=1 benchmark for that bring-up. It is separate from the production
TPU image and does not execute full tokens. The initial validation below is RTL
simulation with synthetic native-memory service, not FPGA results or HBM timing.

`rtl/c2/otpu_c2_hbm_kernel.sv` owns the eight native 64-byte channels below the frozen
F2 memory fabric. The shell still owns PCIS, OCL CDC and HBM AXI conversion. Its
core-domain `hbm_ready` input must be synchronized by the wrapper. Native writes
are tied off; ACT and scales are resident before START, and results retire to
local readback SRAM. The original C2 MXU public interface is unchanged.

## Identical physical layout in both modes

The versioned sidecar `c2-f2-pairpc-v1` maps logical row `r` to PC pair
`2*(r%4)` and `2*(r%4)+1`. A chunk's low/high 64 bytes are at native beat address
`BASE_BEAT + floor(r/4)*KB + k` on those two PCs. Raw BAR4 addressing remains
`PC*512MiB + native_beat*64`; this is not a canonical CHASH weight image.

MODE 0 issues one row per legacy M=1 MXU command, routing **that row's physical
PC pair** to input column 0. MODE 1 issues up to four rows per decode command.
Both use the same ACT0, Sa, per-row Sw/dw, exactly `ROWS*KB*128` HBM weight bytes,
and the same local result sink. ROWS is 1..512; KB is 1..255. Partial groups are
masked and never request padding rows. Missing setup initialization is a host
contract violation, not a detected hardware error.

## Native ownership and completion

Each PC has a 32-entry synchronous receive ring. A request reserves a slot until
both 64-byte halves have arrived and the resulting 128-byte chunk is accepted by
the MXU. Independent PC ready/response delays cannot overwrite live ring data.
Responses have no ready; the frozen bridge must return them in order within each
PC and must not silently drop accepted transactions. The array has independent
ping-pong operand banks, so filling the following command can overlap computation.

ABORT stops new requests, clears pending source valids, drains every accepted
per-PC response, then flushes the MXU before releasing ownership. An unexpected
response is counted in diagnostic raw-return counters but cannot satisfy a real
outstanding request on another PC or corrupt the drain condition. Native error
and unexpected-return errors use the same drain path. If the native fabric never
returns an accepted read, recovery deliberately remains draining; a host timeout
must report failure rather than assume ownership was released. Whole-shell reset
is only legal with the native fabric reset/quiescent; it is not a live abort.

START acceptance resets counters. The measured interval contains kernel command
setup, real native request/return service, fill, compute, and local retirement,
through the last result commit. DONE freezes counters. Preload and host readback
are excluded. Aggregate wait counters overlap; in particular MXU weight/scale/ACT
fill counters include cycles with successful ingress and can overlap computation.
They must not be summed as exclusive critical-path loss.

## Register contract

All accesses are 32-bit aligned; only WSTRB=0xF writes are accepted. Illegal
register/memory accesses return AXI-Lite SLVERR and do not auto-increment MEM_ADDR.
AW/W may arrive independently; B and R payloads remain stable under backpressure.
Configuration and setup memory writes require idle. MEM_ADDR itself is writable
while busy; result reads may observe an incomplete run, so the host reads after
DONE. Successful MEM_WDATA increments once; successful MEM_RDATA increments on
AR acceptance, even if the response is stalled.

| Offset | Meaning |
|---|---|
| 000/004/008 | ID `0x43324d42`, ABI 1, BUILD_ID |
| 00c | status: busy0, done1, error2, aborted3, draining4 |
| 010 | exact control values START=1, ABORT=2, CLEAR=4 |
| 014/018/01c/020 | mode, KB, rows, base native beat |
| 024/028/02c | MEM_ADDR, MEM_WDATA, MEM_RDATA |
| 030/034 | error bits, result count |
| 038/03c | measured cycles low/high |
| 040/044 | accepted/committed groups |
| 048/04c | issued/raw-returned native beats |
| 050 | array advances |
| 054/058/05c/060/064 | any native request blocked, weight fill, scale fill, ACT fill, output wait cycles |
| 068 | peak total accepted native reads awaiting return |
| 06c/070/074/078/07c | CORE_KHZ, MAX_ROWS, DEPTH, PC credits, HBM ready |
| 100+PC*20 | per-PC issued, returned, peak outstanding, request blocked, pending without response, receive-credit blocked at offsets 0/4/8/c/10/14 |

ERROR bits are 1 invalid START configuration, 2 native error, 4 unexpected
response, 8 receive-credit overflow. Invalid START reports DONE/ERROR without
issuing native traffic. CLEAR is idle-only and clears status/counters.

Memory window addresses count **32-bit words**, not bytes:

| Address | Word contents |
|---|---|
| 00000 + k*32 + word | ACT0 chunk data, write only |
| 10000 + k | four da8 offsets, write only |
| 11000 | Sa fp32 bits, write only |
| 20000 + row*256 + k | dw8 in low byte, write only |
| 40000 + row | Sw fp32 bits, write only |
| 50000 + row | result fp32 bits, read only |
| 60000 + row | signed48 integer low32, read only |
| 70000 + row | signed48 integer high16, read only |

## Validation and limits

`tests/test_c2_hbm.py` and `sim/verilator/tb_c2_hbm.sv` exercise both modes,
independent PC service, delayed PC0, full receive credits, row tails and maximum
ROW/KB bounds. Every completed result is checked against canonical ISA fp32 bits
and independent signed integer accumulation. The harness checks response stability,
auto-increment, invalid configuration/access rejection, exact accepted/returned
traffic, command/commit counts, native writes remaining zero, and frozen cycles.
Abort, native error, and an unsolicited PC7 response while PC0 is delayed all drain
and then replay without reset. The trace reconciles handshake counts with CSRs.

These tests do not replace shell CDC validation, physical memory-interface tests,
post-route timing/hold checks, resource limits, or bit-exact board runs. The F2
build and host driver preserve separate identities and require those later gates.
