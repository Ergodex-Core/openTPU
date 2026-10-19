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
