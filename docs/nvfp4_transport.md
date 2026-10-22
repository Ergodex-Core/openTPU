# Simulated descriptor transport and tile ownership

Gate four adds `opentpu/nvfp4_transport.py` on top of `faac00de`. It connects
checked logical W4A16 jobs to a bounded simulated memory reader, one K16 tile
slot and a staged output writer. Arithmetic still comes from the independently
tested strict local controller. This module never imports Board/XDMA, opens a
device, maps host/device memory, executes a model or advertises device capability.

The existing `opentpu/host/board.py` BoardBackend/SimTransport use the current
card register and channel protocol. `host/fake.py` FakeTransport supplies RAM
without computation. Neither exposes the proposed checked W4A16 job protocol;
`runtime.py` allocation/weight paths do not supply its permission model either.
This gate therefore uses a separate pure Python simulation adapter. It changes
no existing MM encoding, board register, shell interface or hardware transport.

## Accepted subset and framing

The accepted subset is the existing local controller contract: one token, one
output, K=1..64, BF16 inputs, E2M1 weights, E4M3FN scales per 16 and multiplicative
FP32 tensor factor. Both strict gradual and strict DAZ/FTZ profiles retain their
existing exact hashes and separate rounding points. The F24 implementation and
its tests remain separate; this adapter rejects F24, RHT, dependencies, post
operations and other unimplemented contracts. No GGUF codec or newly named
checkpoint is assumed. Mixed attention/head formats and full-model execution
remain outside this gate.

`encode_descriptor`/`decode_descriptor` use a four-byte big-endian byte count
followed by exact UTF-8 JSON containing the complete W4A16Job mapping. This is a
repository-local test frame, **not a binary ISA or production wire/DMA ABI**.
There must be exactly one body with 1..4096 bytes. Partial headers/bodies,
trailing bytes, oversized lengths, duplicate fields at any depth, malformed
UTF-8/JSON, floats/nonfinite numbers, missing/unknown fields and unsupported
versions/profiles reject. The immutable typed validator then enforces all field,
shape, alignment, span, whole-allocation output-alias and uint64 identity rules.
The bounded local adapter additionally checks the live generation and K limit.
All validation and mapping checks precede acceptance and lease creation.

## Simulated memory and tile buffer

`SimMemory` owns at most 128 sparse windows, each 1..4096 bytes. Mapping copies
initial bytes and rejects overlap or uint64 address overflow; exclusive end
exactly 2^64 is legal. Every public access checks the whole range fits one
window, the trusted simulated context/generation and read/write permission.
Cross-window reads reject even when the windows are adjacent. These numeric
contexts model trusted caller identity; they are not OS authentication or an
IOMMU/security boundary. The API is single-threaded and supplies no concurrent
memory-ordering or cache-coherence guarantee.

Acceptance reserves all four declared allocation spans atomically. A, W and
scales require read permission; Y requires write permission. All referenced
mapped windows are leased exclusively, conservatively blocking any host write,
unmap or generation/permission rebind to them until completion is acquired.
An input read is checked again against its declared span and live lease. Windows
may include padding, but an unrelated padding write is also blocked by the lease.
Read access alone does not allow another transport to acquire a leased window.

Prefetch snapshots operand and scale data independently, so either can be delayed.
For atom i it reads 2*n BF16 bytes at A+32*i, ceil(n/2) packed weight bytes at
W+8*i, and one scale at S+i, where n=min(16,K-16*i). No out-of-tail read is issued.
Even-index weights use the low nibble, unused odd high nibbles are ignored, and
the remaining tile lanes are literal +0. Once ready, the same immutable Packet
object is held across calls until accepted. Its group/K/epoch and full job handle
bind operands, scale and subtotal ownership. An edited/copied/stale Packet cannot
free the tile. The single tile slot may be refilled only after `accept_packet`.

`BUFFER_FREE` records that packet snapshot/accept event. It releases the tile
slot; it does not release mapped allocations or publish a numerical result.
This scheduling boundary imports no GPU numerical fusion or TPS assumption.

## Output visibility and completion credit

There is one job, one tile slot and one terminal completion credit. Processing is:

1. Validate frame, local contract and complete allocation permissions; accept and lease.
2. Fetch/hold/accept each increasing-K atom, with independent scale readiness.
3. Accept one full-identity, exact-profile private core result after all atoms drain.
4. For success, stage exactly four little-endian FP32 bytes behind a unique write ticket.
5. Acknowledge that ticket: atomically copy the bytes into simulated Y, then release
   a successful Completion. A fault instead releases FAILED and leaves Y unchanged.
6. The owning context acquires the Completion, releasing mappings and the job credit.

A staged write remains invisible under backpressure. `issue_write(ready=False)`
does nothing; repeated issue after staging returns the same ticket. No successful
Completion exists before acknowledgment. Nonfinite successful results, nonzero
failed result bits, incorrect result identity/profile, early results and duplicate
results reject without publishing. Local RTL `output_committed` is only its
private-result handshake; this simulated transport's Y_VISIBLE event and subsequent
COMPLETION_RELEASE define the stronger memory-visibility boundary here. Neither
means model/KV state commit, speculative token acceptance or a real DMA fence.

Handles, memory leases and write tickets include an opaque per-memory namespace
plus monotonic uint64 counters. This prevents independent memories with equal
counters from acknowledging each other. Terminal records retain the original full
context/generation/request/position/epoch. Completion backpressure blocks new jobs
and keeps leases live. Wrong-owner completion acquisition rejects. The last 32
drained completions can be queried with `lookup`; repeating a retained logical
identity rejects so uncertain callers can query first. History is bounded and
is not durable crash recovery or an unbounded exactly-once guarantee. Even after
history eviction and logical-key reuse, old opaque handles/tickets cannot retire
new work. No counter or generation silently wraps; exhaustion rejects or fails
work safely.

## Cancellation and reset

Before publication, `cancel(handle)` discards staged data and releases a held
CANCELLED completion. `reset(new_generation)` requires a strictly larger uint64
generation and similarly produces LOST_ON_RESET for accepted pending work,
retaining its original identity. A staged write is discarded before either
terminal record is visible. Old results, packets and tickets cannot publish after
cancellation, reset or later ownership reuse. The old record must drain before
new submission, and leased windows must then be explicitly rebound to the new
generation. Cancellation/reset do not undo an already acknowledged Y write:
its already-published SUCCEEDED record remains available until acquired.

The adapter does not drive the RTL reset pin or run an asynchronous device. Its
cancellation applies to the simulated transport lifetime; the standalone local
RTL's reset/session-invalidation behavior is independently tested. A future live
frontend must fence all real outstanding reads/writes and device holders before
resource reuse, including responses already past a visibility boundary.

## Verification and limits

`tests/test_nvfp4_transport.py` covers malformed lengths/JSON/numerical contracts,
atomic permission failures for each buffer, owner/generation checks, uint64 bounds,
window/counter limits, all K tails, odd-nibble padding, delayed scales, held packets,
write backpressure, ordered visibility, leases and completion credit, wrong/stale
responses, cross-memory namespace collisions, cancellation/reset at every adapter
phase, already-visible output, explicit generation-wrap rejection, back-to-back
reuse and history eviction. Three seeds add 270 randomized stalled transactions
with both strict profiles and adversarial numerical words/scales. These tests
use the exact software oracle only as a test backend, not as module arithmetic.

`tests/test_nvfp4_transport_rtl.py` is a file-backed replay bridge. Real SimMemory
reads create packets supplied to the RTL descriptor/private-result controller.
Both Icarus and Verilator compare the resulting bits/faults with strict_row;
the actual logged RTL results then drive the adapter's staged output/Completion
API and verify memory visibility. Each runs 144 descriptors: all K=1..64 in both
profiles plus special/overflow/numerical-cancellation/subnormal/signed-zero cases. Ten
numerical failures retain sentinel Y. RTL packet/result backpressure assertions
remain active. This is not interactive cycle-by-cycle memory co-simulation;
transport reset/cancel and RTL reset/invalidation are tested in their respective
suites. Aggregate runs retain the separate F24, strict-tree, FP32 and epilogue tests.

Run with the installed interpreter (NumPy/pytest are available there):

```
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q tests/test_nvfp4_transport.py
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q -s tests/test_nvfp4_transport_rtl.py
```

Independent Sol review reproduced a cross-memory ticket collision before the
namespace fix. The directed collision regression now checks handle, lease and
ticket rejection. Review also prompted consistent uint64 validation on unmap
and rebind. Final results: **203 selected software tests passed** (two RTL cases deselected)
and **14 aggregate RTL tests passed** (one adapter unit case deselected). The
53 transport unit tests are included in the software total. Each simulator's
144-job bridge completed 134 successes and ten expected failures, with 7,199
waiting-packet and 1,434 stalled-result cycles. All prior numerical scoreboard
counts remain unchanged. No arithmetic or RTL source changed in this gate.
Commands and logs are recorded in `build/nvfp4-gate4`:

```
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q tests/test_nvfp4.py tests/test_nvfp4_epilogue.py tests/test_nvfp4_local_job.py tests/test_nvfp4_transport.py tests/test_mx.py tests/test_isa.py tests/test_quant.py tests/test_fp.py -k 'not local_job_rtl'
/Users/rachittibrewal/anaconda3/bin/python -m pytest -q -s --basetemp=build/nvfp4-gate4/final-rtl tests/test_nvfp4_rtl.py tests/test_nvfp4_scale_acc_rtl.py tests/test_nvfp4_strict_rtl.py tests/test_nvfp4_local_job.py tests/test_nvfp4_transport_rtl.py -k 'not local_adapter'
```

The next gate is a deliberately specified live/runtime frontend with protected
allocation ownership, device/host fencing, external response drain, dependency
semantics and a real descriptor ABI/capability negotiation. This gate provides
none of those hardware guarantees. Larger tiles/matrices, actual model tensors,
FP8 attention/BF16 heads, RHT conversion and speculative state commit/rollback
remain separate work. F2/VU47P at 200 MHz remains a target, with no native FP32 DSP.
Vivado/XSim, synthesis/resource mapping, place/route, timing/CDC, F2 shell/DCP/AFI,
FPGA loading, hardware measurements and full-model inference remain unrun.
No paid resources, live DMA, model download/inference, license/security change,
push or publication were used. The protected wide-port checkout is untouched.
