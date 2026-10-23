# Phonon-2 device runner handoff

This change prepares the physical execution path and tests its protocol offline.
No card has been opened. No image, production files, lock, service or readiness
flag has changed. `b91e839fc5a95e48acda0473f00d313b7859f094` and all its
waveform/encoder/TDT evidence are preserved. The public adapter remains blocked.

The execution target is the existing Gemma image `agfi-0001de1dd3df8e21e`,
`f2-hbm:p4rl8a64c150`, CFG_HASH `dd6b92a6`, 150 MHz. This runner does not load
images. It requires the pinned common/host runtime at
`df69c0a9d970ba4f78b8ac6e979b41dc69d4f181`; opcode equivalence with another
compiler revision is not assumed. The deployment owner must qualify any change.

The common plus nine host-source dependency closure can be reproduced from an
existing local Git object database, with no checkout/hooks or device access:

```sh
python -B -m tools.phonon2_runner_runtime \
  --prepare-from-git /existing/common-runtime-checkout \
  --destination /new/verified-common-and-host-runtime
```

Use `--include-test-fake` only for native Board protocol mocks. It adds one
pinned in-memory test source, with no hardware opening or model computation.

## Offline artifact preparation

Run on the existing EC2 with the existing checkpoint, before taking any device
lease. Do not download the model to the Mac. No remote model code is imported.

```sh
python -B -m tools.phonon2_run_shared_prepare \
  --runtime-root /existing/verified/common-runtime \
  --container /existing/ec2/model.fermion \
  --samples 112960 --output-dir /new/phonon2-artifact-112960
```

The preparer verifies the 177,438,361-byte container SHA
`4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`,
22 preserved compute/data sources, the 17-file common runtime, coefficients and
tokenizer data. It compiles all stages with two-pass device activation precision.
It exports immutable weight ranges, static device programs, explicit WAITC
tables, one PCM upload action and a bounded final-output descriptor. Programs
and blobs are content addressed. No activation is calculated during export.
The native Torch audit functions in the data-reader modules are never called.

Artifacts are specific to an exact PCM sample count. Their stored/valid frame
counts, masks, reset ranges and decoder watchdog are compiled constants. A new
length requires offline preparation; do not compile under a serving lease.
The owner pins the printed manifest SHA outside the artifact, publishes the
directory read-only, and constructs `Artifact(directory, expected_sha256)` before
any device action. Every blob is verified initially and again before use.
The manifest hash is a trusted deployment pin, not a signature or remote trust
decision. There is no pickle, executable checkpoint, shell hook or plugin load.

The artifact contains 693 device dispatches: request reset; six waveform phases;
one overlapping-safe handoff; nine subsampler phases; 24 handoffs and 24×27
encoder phases; one decoder handoff; three decoder phases. The same layer slot
is reused, with device output copied to the next device input before immutable
weights are restaged. Raw PCM bytes are the only input-dependent host upload.
All DSP, masking, learned layers, LSTM/cache, duration/token decisions and
UTF-8 piece selection execute as device instructions. The host only copies
already selected output segments and renders UTF-8.

## Deployment-owner interface

`opentpu.asr.phonon2_runner` has no hardware factory or physical CLI. The owner
supplies an **already-open**, locked, idle native `Board` through
`BoardTransport(board, lease_valid_callback)`. Constructing this wrapper does no
I/O. The owner must not construct a Board on the occupied Gemma card: the Board
constructor can quiesce/calibrate. The adapter never closes the borrowed Board,
scrubs DRAM, calibrates, changes an image, overrides a lock or starts a service.

After separate parent approval for an idle spare card, the owner constructs a
`Permit` with the approved spare-card lease ID, live image-query evidence, clean
health/ECC evidence and monotonic expiry. This is an owner attestation, not an
authorization supplied by this PR. `spare_card=False`, missing approval,
expired/lost lease, wrong image, dirty/ECC state or missing capabilities are
rejected before writes. Current warm-card access is unsupported. An owner must
not mark the occupied card as spare to bypass this gate.

The blocking library call is:

```python
artifact = Artifact(read_only_directory, externally_pinned_manifest_sha256)
transport = BoardTransport(already_open_owned_spare_board, lease_valid)
runner = Runner(artifact, transport)
result = runner.run(pcm_s16le_bytes, permit=approved_spare_permit,
                    cancelled=request_cancelled, request_id=request_id)
```

Input is mono signed 16-bit little-endian PCM at exactly 16 kHz, 320–480,000
samples, with byte length matching the artifact. The surrounding Inferlab API
must validate encoding/rate/channels and body size **before lease acquisition**.
No resampling, WAV/MP3 decoding, gain correction, dither, batching, streaming
chunks, timestamps or hotwords are supported. Pinned dither is zero; the NeMo
archive's alternative dither remains an explicit source ambiguity in the
existing semantic audit. Host audio preprocessing is forbidden in this path.

Success returns `{"text": str, "receipt": {...}}`. Text comes from the device's
32-byte UTF-8 scatter/gather segments and exact lengths. There is no host token
mapping, CTC grouping, trimming or inference. Status 1 means frames exhausted;
status 2 means EOS. Status 3 is `ASR_DECODE_INCOMPLETE`: partial text is discarded.
All control fields, lengths, byte totals, termination and UTF-8 are checked.

Cancellation is cooperative before/after operations and during register polls.
A temporary single-threaded transport proxy slices polling at 50 ms while
calling `Board.wait` once. Its timeout/cancel exception lets Board stop and drain
normally. Synchronous DMA cannot be interrupted inside a native call; its
worst-case latency has not been measured. There are no concurrent Board calls.
Every program is loaded with an explicit per-instruction `W.DEFAULT` WAITC
table. `Board.wait` observes HALTED, WR_IDLE and hardware errors before the next
upload. The original transport is restored even after an exception.

On any failure after a device operation, the session is marked dirty. Cleanup
attempts `stop(wait=0)` and verifies write-idle only while the same lease is
owned. Lost ownership never authorizes a stop or reset. No output is returned
as success; no automatic image reload, scrub or warm-context restoration occurs.
The owner quarantines the session and performs separately approved recovery.
A fresh approved request starts with a device clear over the owned graph and
each component's existing reset program. Input is uploaded after request reset.
Do not re-use a failed Runner or release/reassign its buffers before recovery.

Suggested API mapping, implemented by the deployment owner after acceptance:

| Code | HTTP | Meaning |
| --- | ---: | --- |
| ASR_INPUT_INVALID | 400 | Input does not match the prepared PCM contract |
| ASR_ARTIFACT_INVALID / ASR_RUNTIME_MISMATCH | 503 | Artifact/runtime must be repaired offline |
| ASR_PHYSICAL_ACCESS_GATED / ASR_PROFILE_MISMATCH | 503 | No authorized compatible device |
| ASR_LEASE_LOST / ASR_RECOVERY_REQUIRED | 503 | Owner intervention; session quarantined |
| ASR_CANCELLED | 499, or connection closed | No transcript; cancellation receipt retained |
| ASR_DEADLINE | 504 | No transcript; device recovery must be checked |
| ASR_DEVICE_ERROR / ASR_DEVICE_UNHEALTHY | 503 | Device/transport failure or unhealthy preflight |
| ASR_OUTPUT_INVALID / ASR_DECODE_INCOMPLETE | 502 | Bounded decoder failed; no partial success |

Inferlab should retain the existing `ready=false`, `available=false` and HTTP503
public adapter. Its owner can consume this contract without changing their
active worktree now. Queue admission, request cancellation, authentication,
artifact selection and device ownership belong to that owner. A mock or ISA
transport must never be installed as the public inference backend.

## Receipts and timing

Each request records the externally pinned artifact hash, model/revision, PCM
hash/sample count, expected image, owner lease, live register identity, owner
image/health evidence, operation order, each operation's host duration, device
counters per dispatch, upload bytes, graph/program addresses, final status,
steps and output hash/byte count. Failures record exact stage progress, error
code, drain result and recovery requirement. `backend=mock|isa` explicitly
excludes physical evidence. Even a completed physical request keeps
`physical_acceptance=false` and `ready=false`: a single receipt cannot enable
readiness. Preserve receipts outside the serving process before releasing its
owner lease; this library does not write production state.

The receipt's `host_audio_or_model_compute=false` describes the absence of host
DSP/model operators in the Runner implementation. An `isa` test transport still
executes all device programs on a CPU emulator. A `mock` transport injects
protocol responses and executes no model. Neither is physical FPGA inference
or an allowed serving fallback.

Previously measured **software-only** evidence is in `waveform_validation.json`:
five PCM-to-text/reset cases, exact speech transcript and all 53 decisions on
112,960 samples, 193 passing component tests and one optional skip. Mock timing
is protocol-test timing; ISA timing is CPU simulation timing. Neither predicts
FPGA latency. New artifact/mock qualification is recorded separately in
`runner_validation.json`; prior receipts are unchanged.

The prepared 112,960-sample artifact now passes a complete replay through this
Runner on CPU ISA: 693 dispatches, the same 98-byte speech transcript and the
same 53-step count. This runner reads final text/control only; its receipt does
not revalidate the individual token/duration trace. The preserved earlier
qualification contains that trace comparison. The actual serialized-artifact
minimum-context impulse → silence → impulse test also passes, with the repeated
50,692,224-byte final graph bit-identical and the intervening graph different.
All 33 final runner/native Board protocol tests pass on `opentpu-m8a-build`.
The existing virtual environment was reused; no installation or security change
was made. The Mac remains a source/Git/receipt/SSH control bridge.

Offline full-artifact preparation took 14.55 seconds on the existing F2 host;
the full CPU ISA replay took 832.49 seconds. Each minimum-context replay took
about 11.1 seconds. These are software measurements. The full artifact has
1,272,202,312 unique blob bytes and stages 3,191,080 logical program/wait bytes.
That is 640 fewer program/wait bytes than the earlier audited decoder run,
because the production decoder omits per-step hidden/cache audit stores. The
learned computation, cache policy, text and termination behavior are preserved.

For 30 seconds, graph capacity is 180,715,008 bytes reused across stages,
TMEM peak 38,924 words, IMEM peak 25,328 of 32,768 words, and the program/wait
slot is 163,840 bytes at `0xfffd8000`. For the existing 7.06-second speech case,
the earlier ISA path measured 1,272,022,400 immutable upload bytes and 693
dispatches. Artifact packing reduces stored duplicates, not those device weight
uploads. Physical DMA bandwidth, 693 load/wait overheads, frontend DFT time,
encoder/TDT duration, stop/drain/cancel latency and context restoration are
**unmeasured**. There is no evidence for a ≤30-second serving interruption.
Upload byte counts are logical payload bytes; native bus read/merge traffic
and transfer completion time have not been measured.

## Exact spare-card preflight and eventual acceptance

1. Obtain separate physical approval and an exclusive idle spare-card lease.
   Record slot/BDF and lock owner. Keep warm Gemma/other jobs untouched. No
   approved spare card/window exists as part of this change.
2. Prepare/hash artifacts and pinned host/runtime source before the lease.
   Confirm sufficient existing disk/RAM; do not create paid resources. If the
   installed runtime differs, stop and qualify that difference in software.
3. Owner obtains a live AWS local-image description binding that slot/BDF to
   `agfi-0001de1dd3df8e21e`. Require the image already present; never reload it.
   The owner supplies this evidence because Board registers do not report AGFI.
4. Under the lease, owner verifies idle/healthy/calibrated state, initialized
   DRAM/ECC evidence, no sticky ECC/error, CFG_HASH, full geometry/capabilities
   and 150 MHz, including platform 5 and eight 512-MiB HBM channels. Missing
   proof fails closed; this runner never scrubs/calibrates.
   Verify no other device work or monitoring mutation races the session.
5. Reserve `[0, graph_bytes)` and the top 163,840-byte program slot exclusively.
   This absolute-address program cannot coexist with Gemma's resident buffers.
   No warm-card test can begin until a measured ≤30-second interruption and
   approved restoration procedure exist; the present code rejects that target.
6. Run retained short speech, full speech, minimum silence, reduction-boundary
   silence and repeat-short-after-reset with matching prepared artifacts. Compare
   PCM hashes, final status/step counts/text against retained golden receipts.
   Exercise cancel, timeout, reset and dirty-session recovery on the spare only.
   Record actual counters, memory/AXI/ECC health and physical transfer/time data.
7. Parent/deployment owner reviews reproducible physical transcript agreement,
   mask/reset/numerical evidence, wider WER/context coverage and timing. Any
   serving readiness or production integration is a separate coordinated action.
