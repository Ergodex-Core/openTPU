# Wide-port burst200 production qualification

The current work has not produced a deployed TPS gain or a qualified 200 MHz
image. The existing combined candidate `b22dda7e` reduced the saved full
26-layer Gemma3-1B single-step RTL result from 2,564,240 to 2,232,093 cycles
(12.953% fewer cycles, exact ISA output). That is simulation evidence at the
same nominal clock, not measured FPGA throughput. The winner on the card
remains revision `417243f1` at 150 MHz.

## Retained production RTL change

`rtl/mxu/otpu_mxu.sv` precomputes each command's last advance index in its
existing command queue entry. The active pop head now compares `ck` with that
registered threshold instead of incrementing `ck` before the comparison.
The threshold travels with `KBa` through h/n/pop_q, and zero-length/high-counter
guards preserve the original semantics. This adds three eight-bit command-state
fields, with no new arithmetic pipeline stage or cycle. A simulation assertion
compares the old and new expression on every active pop-head clock.

The change targets an uncut path in the pinned 150 MHz winner's routed report:
`u_mxu/ck_reg[3]` to an ACT RAM address pin, 5.960 ns data delay, eight logic
levels, +0.054 ns setup slack at 6.667 ns. Route accounts for 5.099 ns; removing
the carry is a candidate improvement, not proof that a new route meets 5 ns.
Evidence: `evidence/wideport-417243f1/clockpaths/core_to_core.rpt`, lines144-232
in the task evidence directory. No post-change physical timing report exists.

The existing three-MM default comparison has byte-identical native, adapter
and TMEM traces and complete output files before/after the change: 2,183/1,868
cycles remain unchanged. Directed cases cover PAIR odd/even KB, QUAD remainders
1/2/3, int4/t2, plain mode, UNIT/RMAX, ACC/RMAX, ASCALE and consecutive commands
with different last indices, all on one actual F2 wide-port configuration.
The final retained-candidate regression passed 123 tests in23.36s on the
existing builder. The separate three-MM capture passed in123.76s and its full
trace/output SHA256 comparisons are in `lastk-trace-equivalence.json`.

An E4/E8 drain-width experiment was rejected before RTL execution: its proposed
RL8 command violates this PORTB4 ISA contract. The failed source/log are
preserved outside the checkout. E4 remains the production default. Existing
drain-denial counters do not prove that E8 improves full decode; do not change
write priority either (prior project evidence records a Qwen regression).

## Shortest path to measured TPS

Further standalone LPN development is not a prerequisite for collecting new
physical timing or hardware measurements. Keep its calibration/TPS guards,
while qualifying the production candidate independently:

1. Resolve the existing builder's usable Vivado license source/seat and obtain
   scope to run the isolated physical flow. At the read-only preflight it had
   243 GiB available RAM, 493 GiB free disk and no Vivado job. Use one Vivado
   job, six synthesis threads/eight implementation threads, retaining at least
   40 GiB available headroom. No new instance is needed.
2. Freeze a source revision and fresh output directory. Run SynthCL for
   `f2-hbm:p4rl8a64tdp`, core200/HBM450/shell250, then verify actual PMP0 RAM
   clocks/endpoints, resources and synthesis diagnostics. Hash its DCP.
3. Run fresh ImplCL from that exact synthesis checkpoint. Review routed setup,
   hold, pulse width, unconstrained paths, CDC/DRC and congestion. Preserve the
   150 MHz winner and compare actual critical paths; do not extrapolate Fmax.
4. Requalify supported Gemma/Qwen correctness at the frozen production revision.
   Existing full-model results predate this no-cycle last-index change; its
   assertion/directed/trace checks do not become new full-model receipts.
5. Only after explicit packaging/AFI and hardware scope, restore approved F2
   access, verify card ownership/release, load the qualified image and run the
   same accepted-output decode benchmark against the 150 MHz winner. Record
   checkpoint, precision, prompt/output length, context, device and wall TPS,
   TTFT, numerical comparison, transfer overhead and clock receipts.

No synthesis, implementation, AFI submission, load or takeover is authorized by
this document. Above200 MHz requires a supported clock recipe and new routed
evidence after200 closes.

## Concrete commands and blockers

Builder: `ubuntu@54.174.195.152`, existing `i-0ec8d33362b3100be`, m8a.16xlarge.
Vivado2025.2 is at `/opt/Xilinx/2025.2/Vivado`; HDK
`/home/ubuntu/aws-fpga` is commit `20fe90180574464fd509c994680ec67191e9759f`,
shell `0x10212415`. Use its existing Python environment from our checkout:

```sh
python boards/f2/build.py --target f2-hbm:p4rl8a64tdp \
  --hdk /home/ubuntu/aws-fpga --out <fresh-owned-synth-dir> --flow SynthCL
python boards/f2/build.py --target f2-hbm:p4rl8a64tdp \
  --hdk /home/ubuntu/aws-fpga --out <fresh-owned-route-dir> --flow ImplCL \
  --from-synth <successful-synth-dir> --synth-sha256 <recorded-dcp-sha256>
```

After sourcing Vivado settings, both license variables remain unset; none of
the three checked standard license files exists; read-only lmstat returns
`Cannot find license file`. Seat availability is unknown. No license checkout,
installation or configuration change was attempted.

Existing F2 `i-0c78de2253f66519d` is running at `54.196.118.232`; direct SSH
timed out. Its observed SSH CIDRs omit this computer's `223.185.128.4/32`.
Changing that host's allowlist requires separate explicit authorization; the
earlier approval concerned another security group. Its loaded image and card
ownership are unverified and remain gated. No network setting was changed.

Read-only preflight receipt: task
`evidence/wideport-validation/production-preflight-20261022.json`.
