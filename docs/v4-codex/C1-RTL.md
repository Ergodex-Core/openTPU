# C1 bounded row-engine RTL milestone

User approval covers continuing C1-C3 implementation in separate branches and requires
openTPU Lens for performance measurement and visualization. This milestone executes a
standalone C1 integer/fp32 kernel in Verilator. It does not change the canonical ISA,
compiler dispatch, TMEM, original MM, or fallback eligibility.

## Implemented boundary

`rtl/c1/otpu_c1_row_engine.sv` provides a 32-product int8 pipeline and a power-of-two
ring of 1, 2, 4 or 8 row slots per engine. A slot is reserved at job allocation and
released only when its ordered integer result enters the shared fp32 pipeline. A
request carries slot, full row ID, command generation, KB and K0. Responses carry
the same tags, a sequential 32-byte group index, 32 int8 weights, dw and Sw. Different
rows may interleave; groups within a row must arrive in order. One beat per engine
per cycle can enter the pipelined reduction tree. Invalid or stale response tags are
consumed with `rsp_drop` and cannot change a live slot.

The implementation registers 32 signed products, five balanced reduction stages,
and a signed48 shifted partial sum. It drops groups whose `16-dw-da` is negative,
then accumulates all selected groups into that row's signed48 state. It never rounds
partial blocks. Each engine has a replicated 512-group ACT snapshot (128 blocks),
including da and Sa metadata. ACT writes are allowed only while the kernel is idle.
K0 selects the source weight/scale block in the external response adapter; AB selects
the local ACT block. Host validation must ensure one common Sa across the command.

`otpu_c1_cluster.sv` assigns absolute row `n` to engine `n % ENGINES`. Every engine
allocates and requests independently until its slots fill. A global cursor admits
integer results in ascending row order, including nonzero N0 and partial tails. A
slow PC can eventually fill every engine's bounded slots through ordered retirement;
more slots hide finite latency but do not remove this head-of-line limit.

`otpu_c1_epilogue.sv` reuses the exact existing MXU signed48-to-fp32 conversion and
repository `otpu_fmul` implementation. It computes `(i2f(acc)*Sw)*(Sa*2^-16)` with the
same fp32 rounding and FTZ semantics as the ISA. The module was shared with C3 and
renamed in this branch. Eight granted stages preserve valid and row tags under
output backpressure. `otpu_c1_top.sv` waits for the final fp32 output before accepting
another command. Shared pipeline capacity is eight results in addition to the bounded
per-PC row slots. The top-level ready/valid result is plain output; it has no TMEM ACC
read/add/write path. ACC remains supported by the earlier reference and canonical
fallback, not by this standalone hardware kernel.

Reset aborts the command and clears all row/pipeline valids. The caller **must advance
the external generation tag** before restarting, and must not wrap/reuse it while
old responses remain possible. Outstanding pre-reset responses can drain afterward;
full generation and row tags prevent them from corrupting reused slots. ACT memory
contents are retained across reset; the caller must load a snapshot before first use.

## Memory and performance contract

The RTL interface is an abstract row/beat protocol, not AXI or an HBM controller.
There is no sidecar parser yet: the harness supplies the row header fields as response
metadata and accesses the selected canonical bytes using K0. A future adapter must
validate the versioned sidecar, translate row addresses and descriptor PC IDs (RTL ports are logical engine indices), parse scale/header
bytes, retain transaction tags, obey AXI response ordering, handle errors, and account
for actual header traffic. No scale/header bandwidth is measured by this milestone.

`sim/c1/main.cpp` models independent PC service with separate parameters for first-row
response latency and cycles per returned 32-byte weight beat. First-response latency
runs concurrently for outstanding rows; per-PC beats share one service budget. Optional
PC0 delay, request backpressure, output backpressure and interleaved responses exercise
independent progress. Model latency is measured from accepted row request; actual beats
cannot appear before the next simulation cycle. This model omits DRAM bank timing,
AXI burst splitting, CDC, controller queues, shell placement and CL DMA.

Lens profiles are built from actual RTL request, beat, integer, fp32 result and occupancy
events. The `rtl-c1-kernel` kind and profile names distinguish them from prior
`analytic-c1` policy profiles and F2 device measurements. `bytes` includes only accepted
weight beats for the selected command; there are no invented TMEM port/cause counters. The Lens memory peak is the configured
abstract service ceiling, `ENGINES*32/beat_interval` bytes/cycle, not device bandwidth.
A row's timeline is request to fp32 retirement, not a claim of continuous compute
utilization. Command cycles exclude the measured harness ACT setup and reset/recovery
prefix. The recovery profile explicitly selects the successful post-reset generation.
The trace retains the complete run, including rejected old-generation traffic.

200 MHz is a comparison target, not achieved timing. No full-token throughput or F2
hardware bandwidth is claimed. The final F2 budget must include CL-side DMA and Small
Shell placement restrictions. Replicated ACT memories, reduction trees, variable shifts,
slot arrays and cross-engine retirement still need target synthesis and placement.
The present ACT reads are behavioral arrays; this milestone does not establish BRAM
inference or physical resource count.

## Reproduce

```sh
python -m pytest -q tests/test_c1_rtl.py tests/test_c1_rows.py tests/test_mx.py
python -m tools.c1_rtl_validate --output build/lens/c1/rtl
```

Builds are content-keyed under `build/c1-rtl/`, using Verilator with assertions and two
make jobs. Every run compares both the RTL signed48 accumulator to an independent
integer group sum and every final fp32 bit to actual canonical `Slice.execute(I.mm)`.
Tests cover tails, nonzero N0/K0/AB, QACT snapshots, KB126 positive/negative accumulators,
dropped shifts, exceptional fp32 scales, randomized flow, delayed PC0 and reset with
live stale responses. The C++ harness also checks output stability under backpressure,
ordered/unique output tags, request mapping and per-engine slot bounds. A separate
E1/S1 replay harness presents a second generation without reset while the first
command's final fp32 output is held for ten cycles; it checks complete retirement
before command acceptance and preserves output bits/tags across the stall.

The profiling sweep holds E16, N257 and KB9 fixed. It varies slot depth 1/2/4/8,
latency 20/80/160 independently of beat interval 1/2/4, then delays PC0 by 160 cycles.
A separate randomized recovery run exercises reset/backpressure. Generated traces,
summaries, the Lens `.otpuprof` bundle and its HTML are under `build/lens/c1/rtl/`.
Validation results and measured cycle tables are recorded below.

## Synthesis scope

A generic `sv2v -DSYNTHESIS` plus Yosys `hierarchy; proc; opt_clean; check -assert`
check passed on the integer row engine and cluster at default E4/S4. It establishes
process conversion and no undriven/multiple-driver problems in that scope only. The
installed sv2v/Yosys path has an early-return function-lowering limitation reproduced
on unchanged repository `otpu_fmul`; it does not establish a complete fp32-top synthesis
pass. Verilator validation covers the complete integer/fp32 simulation path. No FPGA
mapped area, timing, formal proof, CDC validation, or F2 shell build has been performed.

## Next integration gate

Implement and validate the descriptor/AXI adapter and real header/scale traffic, then
connect ACT distribution and TMEM retirement while retaining canonical fallback for
unsupported commands. ACC needs the exact repository fp32 add after each command's
complete result, with its read/write ordering verified. Only after those interfaces
pass full ISA/token comparisons should F2 synthesis and shell/DMA placement be used to
assess fit and frequency. Keep the current microkernel data separate from token traces.

## Measured RTL results (2026-10-18)

The E16/N257/KB9 sweep produced 25 Lens profiles. Each selected command served exactly
9,252 accepted 32-byte weight beats (296,064 bytes) and retired 257 bit-exact fp32
outputs. The table contains observed Verilator command cycles, not scaled baseline
cycles. ACT load, header/scale transport, TMEM and full-token costs remain outside the
kernel as described above.

| Per-PC memory assumptions | 1 slot | 2 slots | 4 slots | 8 slots |
|---|---:|---:|---:|---:|
| LAT20, one beat/cycle | 1,114 | 650 | 650 | 650 |
| LAT80, one beat/cycle | 2,134 | 1,134 | 710 | 710 |
| LAT160, one beat/cycle | 3,494 | 1,854 | 1,034 | 790 |
| LAT80, one beat/2 cycles | 2,729 | 1,449 | 1,321 | 1,321 |
| LAT80, one beat/4 cycles | 3,919 | 2,543 | 2,543 | 2,543 |
| LAT80, one beat/cycle, PC0 +160 | 4,854 | 2,574 | 1,434 | 870 |

Four slots remove most latency exposure at LAT80, reducing 2,134 cycles to 710. Eight
slots help longer latency or delayed-PC cases, while extra slots cannot improve the
2,543-cycle bandwidth-limited case beyond two slots. These results support bounded
independent outstanding rows as a useful next integration choice; they do not establish
that F2 supplies the assumed service or that a complete token sees the same speedup.

The randomized recovery profile takes 1,547 cycles for its successful post-reset command,
rejects 64 old-generation beats, and retires all 257 outputs exactly. Its full raw run
includes 9,316 served beats across both generations; the Lens byte total uses only the
9,252 accepted beats of the selected successful command. This distinction prevents
reset-prefix traffic from being mislabeled as useful result bandwidth.

Artifacts:

- `build/lens/c1/rtl/c1-rtl.otpuprof` — 25 profiles, with raw event provenance.
- `build/lens/c1/rtl/c1-rtl.html` — standalone openTPU Lens viewer.
- `build/lens/c1/rtl/validation.json` — sweep settings, cycle counts and exactness results.
- `build/lens/c1/rtl/<case>/trace.jsonl` — actual ready/valid service events and occupancy.

All 24 E16 RTL tests passed, including reset during partial accumulation and with live
fp32 stages; 104 existing C1 reference/ISA tests also passed. The executable API is
limited to E1 and E16. E1/S1 and E1/S4 scalar-versus-packed port checks also passed. Other RTL
parameterizations require their own simulation before being used as measured evidence.

Final combined validation: `python -m pytest -q tests/test_c1_rtl.py tests/test_c1_rows.py tests/test_mx.py`
passes 131 tests (27 RTL tests and 104 reference/ISA tests), using Verilator 5.048.
Read-only peer review of slot lifetime, generation reset, ordered retirement and the
shared epilogue found no control defect; its suggested no-reset command-boundary test
was added and passed.
