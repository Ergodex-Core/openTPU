# NV41 F2 OCL mailbox ABI 1.0

Code and simulation integration on `wide-port-alexshell` pinned at
`0a868822a71422beb11771c9fa046af762579e81`. This optional peripheral executes
one strict W4A16 output with K=1..64, using device-local HBM. It is not a layer
engine, mixed-model inference implementation, host DMA interface, or physically
qualified F2 image. See `nvfp4_w4a16.md` for the numerical contract and
`nvfp4_local_job.md` for the private arithmetic controller.

## Compatibility and discovery

The existing MM/WNV/STREAM binary encodings, control registers, CFG_HASH and
CAPS W4/MX meanings remain unchanged. `otpu_f2_hbm_sys.NVFP4_MAILBOX` defaults
to zero; existing targets keep it zero. The explicit `f2-hbm:nvfp4` target opts in
through `platform.nvfp4_mailbox=true`, retaining the F2 HBM platform identity,
geometry and nominal 200 MHz target. This is a capability addition, not a new
instruction set or a claim of timing closure. `Board.info()`'s existing `caps2`
dictionary is unchanged; the separate client probes the raw capability bit.

REGMAP at BAR0 0x3c must be >=5 and CAPS2 at 0x5c bit6 must be set before
accessing the mailbox. Then require the complete tuple ID=`0x3134564e` (NV41),
ABI=`0x00010000`, descriptor length=256, limits=`0x00030140`, and both exact
profile digests. Bits17:16 of limits identify gradual and DAZ/FTZ strict profiles;
bits15:8 are maximum in-flight jobs (1), bits7:0 maximum K (64). There is no F24
launch capability. Its separately labeled approximate core/reference remains
available for numerical experiments only.

The register audit at the pinned baseline finds legacy controls/configuration,
trace and performance registers below 0x400 (last allocated control 0x224).
Reserve **0x400..0x7ff** for NV41. The existing OCL crossing accepts only the
first 4 KiB; its out-of-aperture reads remain zero and writes are dropped.
No BAR resize, PCIM host-address translation, security setting or DMA opcode
is introduced.

## Ownership, memory and fence domains

The host holds the repository F2 exclusive DeviceLock and all allocation
lifetimes from session configuration until terminal completion ACK. The
register-only `Mailbox` client accepts an already-owned F2Transport or Board;
construction/import never opens a device. Every read and write checks the
ownership guard. PCIS remains the existing device-local data path. Its prior
writes must complete through F2Transport's existing write fence and the
`pcis_wbusy` OCL hold before LAUNCH. The owner must not modify inputs, windows,
outputs or legacy execution while a job/completion occupies the credit.
This is a cooperative single-owner contract, not IOMMU/OS isolation against a
malicious process writing PCIS. No live transfers are used by these tests.

The sidecar borrows only native channel0, HBM pseudo-channel0, absolute byte
window [0,2^29). Admission requires synchronized calibration, RUN=0, no loader
start/busy/pending, and every legacy read/write idle. A router after the existing
OCL CDC/SLR stages pairs independent AW/W before dispatch and holds routing
through the response. While ownership is held, legacy writes receive SLVERR;
legacy reads remain available. STATUS.WR_IDLE is masked while the new engine
has a request outstanding. Channel0 legacy ready/read-valid are masked while
owned; its write-completion count is virtualized to subtract sidecar B responses,
including the current-cycle delta and modulo-16-bit wrap. Other channels retain
the existing wiring. Held completion retains ownership until ACK.

Four allocation windows are staged and ARM commits the context/generation.
Each has a unique nonzero opaque handle, aligned nonzero base/span (multiples
of64), permissions READ=1 / WRITE=2 / both=3, reserved=0. All entries must be
well-formed. An operand resolves its handle plus aligned offset to exactly one
window. Before acceptance the RTL checks 65-bit sums, physical bank bounds,
operand span/stride, full-line read padding rights and role permissions. The
whole output allocation must not overlap any referenced input allocation;
input/input sharing is allowed. These checks precede native 25-bit beat address
truncation. There is no channel hashing or reuse of legacy 32-bit DRAM pointers
for this descriptor.

The engine issues at most one native request at a time. A/W/scales are read as
full64-byte lines (padding must be authorized), then sent with complete
context/generation/request/position/epoch and K-group tags to the unchanged
strict controller. Subtotal rounding, E4M3 block multiply, ordered FP32 group
accumulation and final global multiply retain separate rounding points. The
private result is written as one masked four-byte native store; write command
and data may handshake independently. Success is published only when the
B-derived native write count reaches the expected next value. Buffer capture
or private arithmetic handshake does not advertise consumer readiness.

`output_visible` in a SUCCESS completion means acknowledged Y visibility in
this device-local native memory domain. On a write-response error it means
**the irrevocable store may have modified Y**, while status FAILED and result
bits zero forbid consuming it. It does not mean a valid result or a CPU-memory
cache fence. A read/numerical/cancel-before-write failure leaves Y untouched.
Native bus errors are sticky and prohibit further admission until coordinated
shell reset; no automatic retry is safe.

Calibration loss after admission is outside the usable-memory contract. If the
memory fabric returns an error it is drained and reported; if it stops returning
responses the mailbox deliberately retains ownership. Recovery requires a
coordinated platform reset, not credit reuse or a fabricated completion.

## Registers (absolute BAR0 byte offsets)

All accesses are aligned32-bit little-endian, WSTRB-aware. Unknown/unaligned
mailbox accesses return SLVERR. Writes to read-only registers return SLVERR.

| Offset | Meaning |
|---|---|
|400/404/408/40c|ID / ABI / descriptor bytes / limits|
|410|STATE bits0 owned,1 completion,2 memory outstanding,3 output too late to cancel,4 reset pending|
|414|COMMAND: LAUNCH1, ACK2, CANCEL3, RESET4, ARM5 (full WSTRB only)|
|418|LAST_ERROR (last peripheral write outcome; successful writes clear)|
|41c/420|64-bit accepted sequence (low word first)|
|428/42c|staged generation|
|430/434|staged context owner|
|438/43c|last accepted epoch|
|440|held completion: status[7:0], faults[15:8], output_visible[16]|
|444|held result bits; zero until completion and on failure|
|448/44c|held completion sequence|
|450..477|five64-bit completion identities: context,generation,request,position,epoch|
|480..49f|gradual profile digest, low32-bit word first|
|4a0..4bf|DAZ/FTZ profile digest, same order|
|4c0/4c4|selected bank base64|
|4c8/4cc|selected bank bytes64|
|4d0/4d4|active generation64|
|500..57f|four32-byte staged windows|
|600..6ff|256-byte staged descriptor|

Window words0..7 are handle64,base64,span64,permissions32,reserved32.
Every byte of all four windows and every descriptor byte must have been written
before LAUNCH. Byte writes merge and mark only strobed bytes. ARM requires a
complete window table and nonzero monotonic generation. Changing context
requires a strictly higher generation. Same-generation ARM preserves the epoch
watermark. Configuration writes disarm admission. The client invalidates its cached windows before the first mutation, so an uncertain reconfiguration cannot submit against stale handles/offsets. A descriptor is snapshotted
atomically at LAUNCH; rejected descriptors never take credit or access memory.
Every accepted descriptor consumes a strictly increasing64-bit epoch in its
session. Sequence and generation exhaustion fail closed; there is no wrap.

Errors: 0OK,1BUSY,2LAYOUT/descriptor contract,3PROFILE,4SHAPE,5OWNER/GENERATION,
6EPOCH,7WINDOW/BOUNDS,8PERMISSION/ALIAS,9NOT_QUIET/native error,
10INCOMPLETE_BYTES,11TOO_LATE/NO_CANCEL,12NO_COMPLETION,13EXHAUSTED.
Some malformed descriptors violate several checks; error priority follows the
validator. All rejected launches leave accept sequence unchanged.

## Descriptor (256 bytes, 64 little-endian32-bit words)

| Words | Fields |
|---|---|
|0..7|magic0x314a344e (N4J1),ABI0x00010000,length256,flags0,K,mode0/1,globalFP32bits,reserved0|
|8..15|exact numerical-profile SHA256 digest as low32-bit word first|
|16..25|five64-bit identities: context,generation,request,position,epoch|
|26..33|A: handle64,offset64,span64,stride64|
|34..41|W: same|
|42..49|scales: same|
|50..57|Y: same|
|58..63|reserved zero|

Tokens=outputs=1; no dependencies, transforms, speculative state or optional
control extensions are encoded. Minimum span/stride: A=2K, W=ceil(K/2), scales=ceil(K/16),
Y=4. Larger registered spans/strides are allowed; only the single declared row is read. Even weight index is the low nibble. Mode0 is strict gradual; mode1 strict
DAZ/FTZ. BF16 activations, E2M1 weights, unsigned finite E4M3FN scales per16,
unsigned finite FP32 global scale. No all-NVFP4 Holo/Qwen/GGUF coverage is implied.

## Completion, cancellation and reset

Statuses: SUCCESS1,FAILED2,CANCELLED3,LOST_ON_RESET4. One completion is held
stable until explicit ACK, including its full original identity and sequence.
No new descriptor or window mutation is accepted while credit is occupied.
CANCEL before the output WRITE phase drains an already-issued read and flushes
the private core before completion. The boundary is conservative: entering
WRITE is too late even if neither native channel has yet handshaken. CANCEL
then returns error11 without hiding a successful/failed old store.

Soft RESET before output starts requests abort/drain, reporting LOST_ON_RESET
under the old identity; after output starts it lets the old store finish. The
old completion remains held. ACK then advances generation, invalidates window
and descriptor byte-valid masks, disarms the session and clears epoch history.
Idle RESET immediately advances generation and invalidates staging. Reset does
not lower generation even before the first ARM. Coordinated shell/core/memory
reset discards volatile state; no exactly-once guarantee spans image/reset loss.
Independent reset of only a memory endpoint while requests remain is unsupported.

The host stores an attempted launch identity before writing LAUNCH. If the
write/confirmation read raises, `resolve()` queries the acceptance sequence;
it never replays the doorbell. Unknown outcomes retain the credit locally.
Likewise an uncertain ACK is queried by STATE/sequence; its cached terminal outcome survives until acquire returns it. If an ordered query proves the same complete terminal record is still held, the caller may explicitly invoke retry_ack; no ACK is automatically replayed. A missing
acceptance with no definitive rejection requires observed idle RESET recovery.
Exclusive ownership and ordered synchronous register access are prerequisites.


## Required resident inference path (future gates)

The user requires complete inference on FPGA. The final host role is setup,
weight/image loading, request ingress and output-token egress. No host arithmetic,
per-layer orchestration, token selection, KV update, or speculative issuing may
be presented as complete FPGA inference. This mailbox currently still requires
the host to register windows, supply each bounded one-output descriptor, ring
LAUNCH, poll/ACK and handle reset/cancel. It is therefore a numerical/transport
validation interface, not the final inference API.

A future resident scheduler must own tiled matrix loops across full dimensions,
embedding gather, activation/residual buffers, layer sequencing, BF16/FP8 mixed
operators, normalization, attention QK/softmax/AV, rotary positions, KV read/write
allocation, MLP nonlinearities, the BF16 output head and token selection. Existing
legacy integer operators and generation loops require explicit numerical/layout
compatibility checks; their presence does not qualify this mixed NVFP4 path.
The exact selected checkpoint metadata and codec must be verified independently;
GGUF IQ3_S is not an assumed NVFP4 encoding.

The resident request API should enqueue token/context requests rather than each
output or layer. Draft and target execution must share an explicit device-owned
transaction: committed position, candidate KV/activation slots, target verification,
accepted-prefix count, atomic commit, rejected-suffix rollback/reclamation and
cancellation/reset epochs. No CPU speculative issuing substitutes for this state.
Capacity, bandwidth, numerical tests and scheduling gates must precede any
throughput or complete-model claim. Coordinate those architecture choices through
the parent task after this bounded mailbox gate; no resident inference rewrite
is included here.
