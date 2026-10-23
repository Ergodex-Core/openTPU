# Protected simulated host frontend

Gate five builds on `8c61a0b1` with an opt-in frontend in
`opentpu/host/nvfp4.py`. It adds negotiated admission and input-release fencing to
the existing one-job simulated transport. It supports exactly one token, one
output and K=1..64. It does not implement a full layer, production hardware
frontend, new instruction, board capability, DMA interface or model execution.

## Existing production interfaces inspected

| Repository contract | Existing behavior | Consequence for W4A16 |
|---|---|---|
| `opentpu/isa.py`, `rtl/top/otpu_pkg.sv`, `rtl/seq/otpu_seq.sv` | Eight 32-bit instruction words; sequencer resolves register-relative fields into `cmd_t` | No assigned BF16 W4A16 launch instruction or wire descriptor exists |
| MM opcode 0x20, WF=3; `rtl/mxu/otpu_mxu.sv` | Existing int8 ACT RAM and MX weight/scale contract | Reusing the encoding would change existing numerics |
| STREAM opcode 0x13; `StreamDesc`; `rtl/dma/otpu_dma.sv` | Float-safe TMEM descriptor for a different stream engine | Its descriptor is not a W4A16 job schema |
| `opentpu/host/regs.py`, `rtl/shell/otpu_ctrl.sv` | VERSION, REGMAP, CAPS and version 5 CAPS2; W4/MX bits describe existing numerics | Neither those bits nor legacy CAPS2 defaults advertise W4A16 |
| `Board.wait` in `opentpu/host/board.py` | HALTED is insufficient; host waits for ST_WR_IDLE so accepted stores have landed | Private core result acceptance cannot imply output visibility |
| WAITW, `docs/isa.md`, `otpu_dma.sv` | Host writes data before flag; a holding WAITW orders younger readers after it | A simulated release snapshot is not a hardware WAITW/coherence proof |

The alternatives were evaluated before implementation:

- **Recommended and implemented:** an additive simulated host endpoint with its own
  explicit capability tuple, existing checked logical descriptor and scoped fences.
  Legacy programs and host transports keep their current behavior.
- Extend STREAM/MM: rejected for this gate because their descriptor and arithmetic
  semantics differ. No compatible bit reinterpretation has been established.
- Assign a hardware launch opcode/register/capability and wire descriptor: defer
  until the F2 shell, descriptor ABI and actual memory/fence contract are chosen.
  This is an architectural decision beyond the current logical proposal.

No production ISA, register map, Board/XDMA code, STREAM descriptor, F24 arithmetic,
strict arithmetic or RTL source changed. Importing the new module opens no device
and runs no model. It is not imported into or enabled by the legacy Board path.

## Exact version and capability negotiation

`ProtectedFrontend` accepts only an explicit `SimEndpoint` over `SimMemory`.
Legacy transports, Board-like objects, dictionaries and missing endpoints fail
closed before any register read or memory access. W4/MX feature bits and
`CAPS2(None)` must never be used as fallback evidence for this new contract.

Both parties must match the complete immutable `Capabilities` tuple:

- Protocol `opentpu.w4a16.sim-host`, version 1.0, the existing local
  `len32be-json-4096-v1` framing, 64-bit address/identity fields.
- Both exact strict profile names and SHA256 digests from `mxu_job.py`.
- One token/output, maximum K64, one in-flight job.
- Fence contract `sync-input-release;acked-y-before-acquire-v1`.
- Cancellation contract `discard-previsibility;preserve-published-v1`.

Unknown/newer minor versions, altered digests/subsets/fence semantics and malformed
field types reject. This exact-match policy makes no untested forward-compatibility
promise. These are local simulation API capabilities, not hardware register bits
or a production negotiated wire ABI. The existing descriptor ABI 1.0 and all
symbolic validation remain independently required.

A session is an opaque object bound to its exact endpoint, trusted context and
current full-width generation. Negotiating an idle endpoint again invalidates
its previous session and input release. A pending job or held completion must
drain before renegotiation; changing capability knowledge does not strand the
original owner of already accepted work. Context values model trusted authority;
this Python API is not process isolation, OS authentication or a security boundary.

## Protected descriptor admission and input release

The host calls `release_inputs(frame)` after its synchronous SimMemory writes,
then `submit(frame, release)`. Release decodes the exact existing bounded frame,
checks the complete immutable W4A16Job/local subset and bound context/generation,
then authorizes all four full declared allocation spans. A/W/scales require read
permission; Y requires write permission. Full bounds, alignment and whole-span
output alias validation remain mandatory.

There is one outstanding opaque release token. It binds the entire typed job,
all four mapping lifetimes/revisions and every referenced input window's content
revision. Creating a later release supersedes the older one. Edited/copied,
foreign-session/endpoint, consumed or descriptor-mismatched tokens reject.
Changing whitespace or key order does not change an otherwise identical typed
job; malformed/unknown fields still reject before admission.

SimMemory now increments a separate content revision after every successful
host write, including same-byte and padding writes, and after a successful
simulated Y commit. Mapping revisions still change on map/rebind, so an
unmap/remap ABA cannot revive a release. The input binding is conservatively to
the whole mapped window: an unrelated write in that input window invalidates it.
A Y commit to a disjoint span in the same window as another input also changes
that window's content version. Prior Y contents alone are not an input dependency,
but Y's mapping lifetime/permissions are bound. Revision counters are unbounded
Python integers; they never silently wrap.

Submission rechecks the snapshot, then immediately acquires the existing complete
allocation leases. This is a synchronous, single-threaded check-to-lease operation.
It is not a concurrent TOCTOU proof, cache flush, host/device barrier or IOMMU.
An input release reserves no window or job credit; changed inputs need a new
release. Accepted work retains its leases through terminal completion acquisition.

Within one generation, accepted descriptor epochs must increase strictly without
wrap. Permission/bounds/fence/descriptor failures or failed transport acceptance
do not advance the watermark. Successful idle renegotiation preserves it.
Cancellation/failure of accepted work still consumes that epoch. This is a local
monotonic admission rule, not a global allocator or durable exactly-once service.

## Output completion, reset and ownership

The simulation device driver uses the existing endpoint transport to fetch/accept
packets and supply an independently computed or RTL private result. The host
frontend supplies no arithmetic. `BUFFER_FREE` still releases only a tile slot;
it cannot make `poll_completion` or `acquire_completion` report success.

A successful terminal Completion appears only after the staged four-byte Y write
is acknowledged and visible. Acquisition releases mappings and the job credit.
Polling/acquiring before then returns None; backpressure keeps the staged write
private and the completion credit occupied. Numerical failure/cancellation leaves
Y unchanged. The RTL's `output_committed` remains a private-result handshake;
ST_HALTED, a GPU launch or a speculative CPU issue is not substituted for the
required output-visibility boundary.

`reset(new_generation)` validates a strictly larger uint64 generation first.
Invalid resets leave session and work intact. A valid reset revokes old admission
and input-release tokens; pending accepted work yields LOST_ON_RESET with its
original identity. The original owner may still poll/acquire that terminal old
record. New negotiation remains blocked until it drains, after which windows must
be explicitly rebound and a new-generation session negotiated. A completed,
already-visible success is preserved through reset. Reset cannot roll back Y
past its visibility point. New generations may start their epoch watermark at 0.

Opaque session/fence sequences are bounded uint64 values and refuse exhaustion;
all prior transport namespaces/handles/tickets and stale-response checks remain.
This adapter does not reset a live device or guarantee cancellation of actual
writes already accepted by a hardware memory controller.

## Verification

`tests/test_nvfp4_frontend.py` adds 78 cases covering both sides of complete
version/capability negotiation, no-I/O legacy refusal, bound context/generation,
foreign/copied/revoked sessions and releases, malformed lengths/JSON/descriptors,
all four buffers' permission/owner/generation/span failures, late A/W/scale writes,
same-byte/padding writes, Y mapping replacement, shared input-window output writes,
ABA lifetime changes, descriptor mutation, superseded/consumed fences, accepted-only
epoch advancement, idle renegotiation, epoch exhaustion, pending/stalled/visible
reset phases, original completion ownership and capability changes during work.

The 144-job-per-simulator file-backed bridge in
`tests/test_nvfp4_transport_rtl.py` now negotiates this frontend, releases inputs
and submits through it; actual observed RTL results drive staged Y writes and
host completion acquisition. Both strict profiles and K=1..64 remain covered.
This is file-backed replay, not interactive cycle-level memory co-simulation.
Transport cancel/reset and RTL reset/session invalidation retain their independent
adversarial suites. No full-layer or TPS claim follows from these jobs.

Selected legacy regressions cover register-map fallback, CAPS2/GATHER behavior,
device capability-derived configuration, WAITW instruction/simulator ordering and
Board.wait's HALTED-without-WR_IDLE error through FakeTransport. They use no device
or model. Aggregate numerical regressions retain all prior F24/strict/FP32 families.
Final results: **295 selected software tests passed** (two RTL cases deselected)
and **14 aggregate RTL tests passed** (one adapter unit case deselected). The
software total includes 78 frontend tests, 53 transport tests and 14 selected legacy
host/WAITW cases. Both Icarus 13.0 and Verilator 5.048 completed 144 frontend bridge
jobs, including 134 successes and ten expected numerical failures that left Y
unchanged; prior numerical scoreboards remain unchanged. No RTL source changed.

Exact commands, logs, source hashes and independent review are retained under
`build/nvfp4-gate5` and packaged with delivery evidence.

## Remaining integration decisions

The next production gate must choose the real F2 shell/frontend, binary descriptor
layout/version policy and discoverable capability, protected allocator/context
ownership, input visibility/coherence, external response drain, write acknowledgment
and completion fencing. It must prove reset behavior for actual outstanding
transactions before enabling live memory. No new binary opcode or capability bit
has been allocated here. Larger tiles/matrices and a full layer remain separate.

Vivado/XSim, synthesis/resource mapping, place/route, timing/CDC, shell/DCP/AFI,
hardware loading and inference remain unrun. VU47P at 200 MHz remains a target without
native FP32 DSP support. RHT needs paired transformed weights; mixed FP8/BF16
model paths, speculative commit/rollback and the separately mentioned GGUF codec
are not implemented or assumed. No model download/inference, live DMA, paid
resource, license/security change, push or public publication was performed.
The protected wide-port checkout was not modified by this work.
