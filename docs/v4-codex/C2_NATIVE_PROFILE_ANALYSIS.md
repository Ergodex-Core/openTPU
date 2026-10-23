# C2 native RTL memory-stall analysis

The ten profiles show two distinct limits: short rows repeatedly expose memory
latency at group boundaries, while longer rows also exhaust the 32 receive slots
per PC. A delayed PC hurts decode disproportionately because every four-row group
waits for it. These are **observations from RTL simulation with synthetic native
memory service, not physical F2/HBM measurements or full-token speedups**.

## Evidence and matched workloads

Artifacts are `build/lens/c2-native/c2-native-rtl.{otpuprof,html,json}` and each
case's `events.txt`. All ten recorded source-file hashes match commit
`4f7509123fdd2cf1eba5a2c88289a3e3c2709439`. The exporter is
[`tools/c2_hbm_profile.py`](../../tools/c2_hbm_profile.py); the service model is
[`sim/verilator/tb_c2_hbm.sv`](../../sim/verilator/tb_c2_hbm.sv).
Verilator executed the kernel and MXU. The shell, native-to-AXI bridge, HBM
controller, CDC and physical timing are absent. Lens's 200 MHz conversion is an
assumed clock, not an achieved frequency.

Every pair uses the same fixture (`seed=20261019`), ACT, scales, weight bytes,
physical PC assignment and local result sink. Legacy M=1 computes one row per
command; decode computes up to four rows. Row `r` uses PCs `2*(r%4)` and
`2*(r%4)+1`, one 64-byte beat on each per KB chunk. Decode therefore changes
request concurrency as well as arithmetic scheduling. ACT/scales are resident
before START; only weights traverse native memory. There are no native writes,
ACC reads, production TMEM/vector operations or token loop in these profiles.

Service uses independent per-PC 20% command-ready stalls and 20% eligible-response
delivery stalls, seed 921, in-order replies within each PC, and 0..5 cycles of
per-request jitter. The table's latency and extra PC0 delay are configured cycles.
The same random seed does not give identical service realizations after the two
modes generate different request schedules. Preload/readback time is excluded.

| Rows × KB | Latency / extra PC0 delay | Legacy cycles | Decode cycles | Legacy / decode cycles |
|---|---:|---:|---:|---:|
| 257 × 9 | 10 / 0 | 8,729 | 2,370 | 3.68 |
| 257 × 9 | 30 / 0 | 13,899 | 3,664 | 3.79 |
| 257 × 9 | 100 / 0 | 31,910 | 8,240 | 3.87 |
| 257 × 9 | 30 / 150 | 23,560 | 13,226 | 1.78 |
| 31 × 65 | 100 / 200 | 15,018 | 7,507 | 2.00 |

The 257-row cases transfer 296,064 bytes / 4,626 beats in either mode: 585 beats
on PC0/1 and 576 on each other PC. They execute 257 legacy or 65 decode groups,
with 2,313 or 585 array advances; the final decode group has one valid row.
The 31-row case transfers 257,920 bytes / 4,030 beats: 520 beats on PC0..5 and
455 on PC6/7. It executes 31 or 8 groups, with 2,015 or 520 advances; its tail
has three valid rows. No padding weights are fetched.

The exporter checked paired signed-integer and fp32 output equality; its runner
also checks canonical results. Independent analysis reconciled all ten raw event
traces to cycle counts, command/commit counts, arithmetic advances, aggregate and
per-PC issued/returned beats, and expected ROWS/KB traffic. All report zero error
and the expected result count.

## What the counters establish

**1. KB9 is limited by exposed group latency, not RD32 capacity.** Raising latency
from 10 to 100 adds 23,181 legacy cycles and 5,870 decode cycles, close to the
90-cycle increase times 257 or 65 groups (23,130 / 5,850). Each PC peaks at only
nine outstanding reads; every receive-credit-blocked counter is zero in all eight
KB9 profiles. Aggregate outstanding peaks are 18 legacy / 72 decode. Decode's
array-advance duty falls from 24.68% to 7.10% across that latency sweep.

The kernel issues only the current group's native reads and advances to the next
group after all current operands are delivered. The MXU requires the complete
group before starting arithmetic; its ping-pong banks can overlap this arithmetic
with the following fill. Consequently, increasing RD alone cannot expose more
than nine requests per PC for these rows. Cross-group memory lookahead or more
independent row progress is a plausible next experiment, not a demonstrated gain.

**2. One slow PC removes much of decode's concurrency benefit.** At KB9/latency30,
adding PC0 delay150 increases decode by 9,562 cycles and legacy by 9,661 cycles.
PC0 participates in all 65 decode groups but only 65 of the 257 legacy commands.
Decode's arithmetic work stays at 585 advances while its duty falls from 15.97%
to 4.42%; its matched cycle ratio falls from 3.79 to 1.78.

| Decode counter / trace observation | PC0 delay 0 | PC0 delay 150 |
|---|---:|---:|
| Any native request blocked | 645 | 658 |
| Weight fill incomplete | 3,422 | 12,984 |
| No native request/response handshake **and** no array advance | 1,438 (39.25%) | 10,444 (78.97%) |
| Any reads pending, no response that cycle | 2,094 | 11,093 |
| PC0 pending without response | 2,411 | 12,182 |
| PC1 pending without response | 2,438 | 2,449 |

The near-constant command-blocking count and large PC0 response-wait increase
point to delayed completion and group synchronization, rather than command-ready
pressure, in this controlled perturbation. The aggregate no-handshake/no-advance
interval grows by 9,006 cycles. It also includes control/setup/drain cycles; it
is neither a physical DRAM-idle measurement nor entirely recoverable stall time.
The trace's aggregate pending count is sampled after that edge's handshakes;
per-PC pending-without-response counts above are the kernel's CSR counters.

**3. KB65 additionally exposes receive-ring and pair-join pressure.** In the
latency100/PC0-delay200 case, every PC reaches 32 outstanding reads; aggregate
peaks are 64 legacy / 256 decode. Decode records 4,298 credit-blocked cycles on
PC0 and 4,287 on PC1, although only PC0 received the extra delay. PC1's
pending-without-response exposure is 2,702 cycles versus PC0's 6,835. A ring slot
stays reserved until both 64-byte halves are assembled and the 128-byte chunk is
accepted by the MXU, so a faster PC cannot release its reservation independently.
The other decode PCs record 934..1,104 credit-blocked cycles each.

This establishes that RD32 binds this workload; it does not quantify the gain
from RD64. There is no KB65 control without PC0 delay or receive-depth sweep in
this bundle. Larger rings cost memory and may encounter native-bridge limits;
the existing testbench also fixes its outstanding-read bound at 32.

**4. The counters do not show output or off-chip metadata limits.** Output-wait
is zero in all ten cases. ACT and scale fill counters are identical between
latency variants (650 each for KB9 decode) because those operands are resident.
Weight, ACT, scale and native wait counters overlap successful ingress and
arithmetic, and must not be added into an exclusive stall budget. Per-PC waits
also overlap each other. Array-advance duty is not PE utilization: the two modes
do different amounts of useful row work per advance.

## Next FPGA measurement priorities

1. **Establish a matched physical baseline.** After synthesis, route, identity
   and memory-integrity gates, run both modes on the same weight image, PC map,
   ROWS/KB, resident operands and result sink. Check exact outputs and byte counts;
   capture achieved clock, cycles and every aggregate/per-PC counter. Include KB9
   and KB65, multiple row counts, repeated trials and physical-PC rotations to
   distinguish a particular slow PC from grouping effects.
2. **Measure native service before changing concurrency.** Obtain per-PC accepted
   request/return timing and outstanding occupancy on real HBM, with the same
   address pattern. Current CSR pending-without-response counts measure exposure,
   not a latency distribution. Separate command backpressure, response latency,
   pair waiting and full-ring exposure; use a dedicated trace/ILA run if required.
3. **Test the two bottleneck hypotheses separately.** If KB9 still has spare
   credits and recurring latency gaps, compare bounded cross-group lookahead or
   independent progress. For KB65 and longer rows, first add a no-skew control,
   then compare RD32/64 within verified bridge capacity. Keep layout and outputs
   fixed and report RAM/timing cost with cycle changes. These are proposed
   experiments; this note changes no implementation.
4. **Extend traffic only after the isolated comparison is understood.** Add real
   ACT/scale delivery and production retirement/TMEM contention before making
   whole-model or token-throughput claims. The present local sink provides no
   evidence that these later stages will keep up.
