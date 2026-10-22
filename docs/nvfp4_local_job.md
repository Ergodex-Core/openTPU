# Strict K16 and bounded local job controller

This third numerical gate extends `8284a5d1` with strict BF16/E2M1 tree RTL and
one checked logical job producing one private FP32 result. All operands are
synthetic local streams. This is not a memory reader, production ISA extension,
full accelerator, Holo inference implementation or measured FPGA configuration.

## Strict arithmetic

`rtl/mxu/otpu_nvfp4_strict_dot16.sv` snapshots one BF16/E2M1 K16 atom, its mask,
underflow policy, scale byte and tag. It executes sixteen separately rounded
FP32 products, followed by adjacent-pair reduction levels of eight, four, two
and one separately rounded FP32 additions. The folded implementation uses the
conformance-first arithmetic package from gate two. Its in-place partial buffer
is updated in ascending pair order, after the preceding level has completed.

This is the strict tree, not the common-exponent F24 dot. Gradual and DAZ/FTZ
profiles match `nvfp4.strict_dot16` directly. Masked terms are literal +0 products;
unused NaN, signed zero and weight bits are ignored. Active signed zero retains
the IEEE operation rules. An active nonfinite operand or overflow at any product
or addition poisons the atom. It still finishes the fixed operation sequence and
returns fault with canonical +0, before any scale can hide the failure.

The tree takes 31 arithmetic clocks after input capture and holds its output
until accepted; the test rejects retirement before 32 edges after capture.
Neither its folded latency nor the simulator clock is a fitted F2 frequency.
Scale/tag/profile are stable under stalls. Synchronous reset discards the atom,
and valid/ready are gated while reset is asserted.

## Supported logical job subset

`opentpu.mxu_job.local_job_fields` adapts an already checked immutable W4A16Job
to the internal typed record in `rtl/mxu/otpu_nvfp4_job.sv`. It supports:

- Exactly one token and one output, K from 1 through 64, at most four K16 atoms.
- Either strict gradual or strict DAZ/FTZ with the exact existing 256-bit profile
  digest. F24, RHT, post operations and dependencies are rejected.
- BF16 A, E2M1/K16/E4M3FN weights, FP32 output, multiplicative FP32 global factor,
  bypass transform, post NONE and symbolic MX_LAUNCH.
- Unsigned 64-bit buffer address/span/stride and all five context, generation,
  request, position and epoch identities.

The internal seven contract predicates represent those fixed symbolic fields.
They are not binary instruction opcodes or an assigned DMA/wire ABI. The test
packs the SystemVerilog struct solely to supply an internal RTL bench. An actual
host transport/parser must preserve the logical schema and reject unknown fields,
versions, layouts and unsupported symbolic tuples before producing this record.
No device capability register or existing MM encoding changed.

The RTL validator independently checks ABI 1.0, zero flags/reserved fields, all
contract predicates, strict profile/digest, finite nonnegative global factor,
shape, no dependencies and the caller's current session generation. Each buffer
must satisfy the same alignment, row-byte stride and span rules as W4A16Job.
Exclusive allocation ends use 65-bit arithmetic: end exactly 2^64 is legal, and
end greater than 2^64 rejects. Output overlap is checked against whole input
allocation spans, not just useful row bytes. Input allocations may overlap each
other. No address is dereferenced and validation grants no mapping/DMA permission.

Descriptor reply fault bits are: 0 fixed contract/version/global; 1 profile or
digest; 2 shape; 3 buffer range/alignment/alias; 4 session generation; 5 dependency.
Rejected descriptors produce only a held rejected reply, no operand acceptance
or private result. A transport handshake is not accepted work: `reply_accepted`
identifies whether the descriptor was validated, snapshotted and reserved.

## Ownership and result isolation

`rtl/mxu/otpu_nvfp4_local_job.sv` holds one job. Its stages are descriptor reply,
packet capture, strict tree, ordered epilogue and private terminal result. It
does not accept operands until the accepted descriptor reply drains. Later
changes to the live descriptor input cannot change the captured job.

Each packet carries raw A/W, scale, exact K-tail mask, first/last, output group,
K16 index, all five full-width identities, and independent scale group/K/epoch.
The one output group is zero. The controller checks the descriptor-derived mask,
first/last, strictly increasing index, full 64-bit identity equality and scale
coassociation. It uses its own bounded count and mask; an incorrect incoming
last/index/mask cannot change how many expected atoms are processed. A malformed
packet poisons the private result rather than rebinding another job. The source
must provide the descriptor's atom count; a missing packet requires upstream
timeout/reset handling. There is no autonomous watchdog or cancellation queue.

`packet_scale_valid` qualifies a composite packet and may arrive late. It is not
a separate reorderable scale channel. All payload and identity fields must remain
stable until packet acceptance. `buffer_free` identifies that snapshot event,
with free group/K/epoch valid then; it releases the local source staging atom,
not an entire DMA allocation or a model-state buffer.

The epilogue consumes only strict tree subtotals here. It preserves a separate
block multiply, ordered FP32 addition and final global multiply. Its internal
32-bit scheduling tags do not truncate external identity: full 64-bit checks and
the final completion use the original captured descriptor.

No result bits or result identity are exposed while a job is incomplete. A failed
terminal result exposes +0 and nonzero fault. A successful terminal result and
all five original identities are held under backpressure. A new descriptor is
not accepted until that result drains. `output_committed` means a successful
local private-result handshake after all arithmetic, not DMA visibility, KV/GDN
commit, sampling acceptance, or speculative token publication.

Result fault bits are: 0 controller packet/protocol; 1 strict tree; 2 epilogue
input; 3 epilogue conversion; 4 block scale; 5 ordered addition; 6 final global;
7 session invalidation. Several bits may diagnose the same poisoned computation.

## Reset and session changes

Reset gates descriptor, reply, packet, result, free and commit handshakes and
flushes accepted numerical work and private state. The enclosing runtime must
classify an accepted job lost on reset. Reset does not create an automatic new
generation or make old external responses safe to reuse.

An active change of `session_generation` is an explicit reset-like cancellation
boundary, exposed by `session_invalidated`. It immediately gates packet/result
handshakes and flushes both arithmetic units. The controller then retains one
failed completion with fault `0x80`, +0 and the original full descriptor identity.
This failure is sticky even if the session input returns to its earlier value.
It must drain before new work. Invalidation takes priority over a ready-high
result or operand consumer in the same sampling edge.

Ordinary stalled valid/payload stability applies except for reset or this explicit
session invalidation. A stalled successful result can be cancelled at that boundary
and replaced by the held failed record; it is never silently committed to the new
session. Repeated generation changes do not mutate an already failed completion.

The surrounding system must still drain or generation-fence every external
operand/scale/subtotal holder before a full identity key is reused or wraps.
Local invalidation cannot recognize arbitrarily old traffic after identical keys
are reused. No global allocator, memory ownership proof or CDC guarantee is claimed.

## Verification

Use the installed NumPy/pytest interpreter on this Mac:

```
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q tests/test_nvfp4.py tests/test_nvfp4_epilogue.py tests/test_nvfp4_local_job.py tests/test_mx.py tests/test_isa.py tests/test_quant.py tests/test_fp.py -k 'not local_job_rtl'
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q -s --basetemp=build/nvfp4-gate3/final-rtl tests/test_nvfp4_rtl.py tests/test_nvfp4_scale_acc_rtl.py tests/test_nvfp4_strict_rtl.py tests/test_nvfp4_local_job.py -k 'not local_adapter'
```

The strict test compares against the strict rational/per-operation oracle only,
including every finite BF16 exponent, all E2M1 codes, both signs/profiles, near-tie
cancellation, signed zeros, underflow boundaries, nonfinite and overflow policy,
masked special values, scale/tag stability, randomized stalls and 34 reset phases.

The local-job test uses the independent Python W4A16Job validator plus bounded
adapter for descriptor verdicts and strict_row for complete numerical answers.
It covers K=1..64 in both profiles, malformed contracts/digests/buffers/shapes,
addresses above 4 GiB and exclusive end 2^64, missing/extra first, incorrect tail
mask, coherent group/K/epoch mutations, independent scale mismatches, high-bit
identity mismatches, descriptor snapshot mutation, failure followed by safe reuse,
delayed scales, held replies/results, reset and transient session changes through
every stage including a stalled result and ready-high cancellation.

Independent high-reasoning Sol review identified the generation-change and
reply-reset gaps before the final tests. Both were corrected and directed cases
added; full identity retention is checked during cancellation. The review did not
perform synthesis or independent hardware execution.

Final results: **150 selected software tests passed** (the two RTL cases were
deselected there), and **12 aggregate RTL tests passed** (the adapter unit test
was deselected there). Verilator lint is clean for the new strict-tree and local
controller tops. Icarus Verilog 13.0 and Verilator 5.048 produce identical new-gate
scoreboard totals:

| Family | Per-simulator evidence |
|---|---|
| Strict tree | 1,516 vectors; 34 reset phases; 14,151 stalled-output cycles |
| Local jobs | 193 descriptors; 32 rejected before work; 161 terminal results, including 23 failures |
| Controller lifetime | 45 reset phases and 45 session-invalidation phases, including partial-row and ready-high cases |
| Controller stalls | 7,883 waiting-packet cycles and 1,764 held-result cycles |

The aggregate also reruns the prior F24 dot, FP32 primitives, ordered epilogue and
composed F24 pipeline families. Every simulator run requires explicit TEST_PASS,
clean exit and no fatal/error/watchdog text. Logs and generated vectors/benches
are retained in `build/nvfp4-gate3/final-rtl`, with software/aggregate logs and
lint reports in `build/nvfp4-gate3`. These are numerical behavioral tests, not
timing, utilization, power or inference measurements.

## Remaining gates

A bounded simulated descriptor transport, one tile slot and staged simulated Y
visibility are now described in [gate four](nvfp4_transport.md). A real authorized
memory frontend still needs a protected production descriptor/runtime, dependency
and fence semantics, source lifetime/generation allocation and actual device Y
writes/completion visibility. Then run
captured tensor/layer golden comparisons and qualify the actual F2 HDK/Vivado flow.
The conformance-first wide FP32 logic needs measured resource/latency/timing work
before replication; no throughput gain or 200 MHz closure follows from these tests.

Vivado/XSim, synthesis, resource mapping, place/route, timing/CDC, F2 shell integration,
DCP/AFI and hardware remain unrun. Mixed FP8 attention/BF16 head, RHT conversion,
transactional speculative state, full-model coverage and quality remain outside
this gate. No paid resources, model downloads, Mac inference, hardware access,
license/security changes, push or publication were used.
