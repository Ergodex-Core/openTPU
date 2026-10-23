# Host preparation and a 30-second owner-controlled validation window

The existing 300-second harness is disallowed. Its checked-in copy exits before
imports or actions. The user permits at most 30 seconds of total interruption,
including restoration of warm Gemma serving. This branch implements the software
preparation and controller protocol; it provides no production owner adapter or
hardware execution command. Physical acceptance remains pending.

The branch is based on published PR #40 commit
`df69c0a9d970ba4f78b8ac6e979b41dc69d4f181`, preserving its deployment logging fix.
No production worktree, gateway, service, lease, or FPGA image is modified.

## Preparing while serving continues

`opentpu.llm.staged` builds each model through `RamBackend`. This backend retains
the complete image in host RAM and refuses inference and output reads. It
compiles the fixed prompt programs and all generation buckets, assembles the
instruction words and wait entries, and prepares chain slots. Image buffers and
assembled arrays are sealed read-only. Hashing happens before admission gating.

`PreparedModel.activate(backend)` accepts an already-uploaded owner-supplied
backend with the exact configuration and device input/greedy capabilities. It
does not open, upload, close, or release hardware. Its execution wrapper uses
the preassembled `Words` interface on a board and rejects unprepared programs,
host model input writes, and model/logit reads. `run_case()` executes only a
prepared fixed case. `detach()` releases the wrapper, not the owner's transport.

The prepare-only tool verifies checkpoint/configuration identities against saved
oracles, prepares the process-local native copy helper without opening a device,
then prepares Nanbeige cap128 and Gemma cap768. It uses cached local checkpoints
and a local tokenizer. For example, on the existing EC2 host:

```sh
python tools/nanbeige_stage.py \
  --nanbeige-model-dir "$NANBEIGE_MODEL" --gemma-model-dir "$GEMMA_MODEL" \
  --nanbeige-oracle "$NANBEIGE_ISA_RECEIPT" --gemma-oracle "$GEMMA_BASELINE" \
  --gemma-cap 768 --hold --json host-prepare.json
```

`--hold` keeps the RAM buffers alive until EOF or `{"action":"discard"}`. Its
only other command is `{"action":"status"}`. It accepts no upload, inference,
maintenance, or ownership operation. The JSON receipt contains identities and
measurements, not serialized model buffers. A process that exits discards its
prepared buffers. Integration must retain them in the eventual owner process,
or explicitly implement and qualify a shared-memory/IPC handoff.

The Nanbeige cases use raw input `[166100,15050]`, expecting `[90956,13]`, then
repeat the same input after context reset. Each case includes prefill, both
transformer loops, the full vocabulary head, and two device-selected output
tokens. The Gemma restore case uses the saved sky-blue chat prompt and the
independently checked device/ISA prefix `[818,7217]`. No logits are read to choose
these tokens. Nanbeige retains the disclosed experimental W8A8/INT8-embedding
precision; this branch introduces no further precision change.

## Proposed production-owner interface

`opentpu.host.validation_window.WindowOwner` is a protocol, not an implemented
connection to production. Its hooks are:

| Hook | Required behavior |
|---|---|
| `preflight(nanbeige, gemma)` | Check retained artifacts, qualification and authorization before the outage |
| `gate_if_idle(deadline)` | Atomically gate admissions only if no request is active; busy returns without gating |
| `release_gemma(deadline)` | Cooperatively release the warm worker and acknowledge completion |
| `upload_nanbeige(prepared, deadline)` | Acquire both normal locks, requalify the existing image, upload the exact RAM buffers and activate them |
| `validate_nanbeige(prepared, case, deadline)` | Run the complete prepared raw/reset case and return structured evidence |
| `cleanup_nanbeige(deadline)` | Stop only this owner's work, detach/close it, and confirm both lock releases |
| `restore_gemma(prepared, deadline)` | Upload the prepared Gemma image and return ownership to the serving worker |
| `verify_gemma(prepared, case, deadline)` | Verify exact restore tokens, device counters, configuration, reset and normal ownership |
| `reopen(deadline)` | Reopen admissions only after verified restoration |

The owner must adopt the restored engine/backend into the serving worker while
retaining its normal locks. Merely closing the test process and letting the old
worker reload Gemma adds another upload and does not satisfy this restoration
contract. The current production worker has no prepared-image adoption API;
that integration remains the production owner's work. Before serving arbitrary
new prompts, it must switch from the fixed-case execution wrapper to its normal
backend/compiler path while preserving the verified engine and ownership.

Every run proof contains the prepared artifact identity, exact token IDs,
positive integer device cycles, context reset, loaded-configuration match,
device selection, and ownership of both normal locks. Mock evidence is labeled
`mock` and can never set `hardware_validated=true`.

The owner must check the current image `agfi-0001de1dd3df8e21e`, architecture
hash3714814630, build `0x417243f1`, 150 MHz, eight 512 MiB channels, full
configuration, capabilities, calibration, idle state, and absence of errors.
There are no reload, synthesis, force-takeover or new-resource operations.

## Deadline and failure behavior

One monotonic clock starts immediately before `gate_if_idle`, including its
latency, and ends after admission reopening is acknowledged. Model packing,
tokenization and compilation happen first while production remains available.
Do not close admissions while waiting for an active request to finish.

The default policy divides the interval as follows:

| Phase | Absolute deadline from admission-gate attempt |
|---|---:|
| Release, Nanbeige upload, raw and reset validation | 6 seconds |
| Nanbeige cleanup | 8 seconds |
| Restore, verify Gemma and reopen admissions | 30 seconds |

The two-second cleanup and 22-second restoration reserves are unqualified policy
examples, not measured bounds. Budgets above 30 seconds or without positive
reserves are rejected. Each hook receives its absolute deadline and remaining
time. Cancellation ends model work and preserves recovery. No additional
validation starts after its deadline. Partial upload or validation failure still
triggers cleanup and restoration; unconfirmed cleanup prevents stealing the
card for a restore. Late cleanup acknowledgements remain recorded so that a
confirmed release can still be followed by restoration.

Readiness, reopening and deadline failures are reported separately. No result
claims success when restoration cannot be verified or the total deadline was
missed. The controller is cooperative: it cannot preempt a blocking C/MMIO call,
an outstanding DMA worker, or an owner hook that ignores its deadline. The owner
needs bounded transport operations and an independent recovery watchdog before
this can support a physical 30-second claim. The old 600-second driver run
timeout cannot serve as that watchdog.

## Software verification

On the existing EC2 CPU, **148 tests passed in 43.94 seconds**, with zero physical
transport opens. This includes the existing 105 strict Nanbeige/Gemma/Qwen
regressions and 43 preparation/controller/CLI tests. Complete tiny Nanbeige and
Gemma executions through an explicit ISA-backed mock transport match their
oracles and reset correctly without compiling during execution. RAM staging
itself executes no forward pass.

Fake-clock tests cover busy admission, partial release/upload, token/counter/
identity/configuration/ownership failures, cancellation, late gate and cleanup
acknowledgements, cleanup/restore/verification/reopening errors, and an owner
hook that overruns 30 seconds. The last case reports the missed deadline rather
than claiming a hard timeout. The retired 300-second harness exit is tested.

The full cached checkpoint measurement used the existing EC2 host, one CPU at
nice19, offline mode, and disabled image/program caches. Physical transport and
CPU-inference constructors were blocked. It completed with zero opening or
forward attempts:

| Host preparation | Seconds | Retained image bytes |
|---|---:|---:|
| Nanbeige cap128, two prepared programs | 102.45 | 4,247,624,448 |
| Gemma cap768, four prepared programs | 19.87 | 1,396,966,272 |
| Complete preparation including checks/tokenizer/helper | 126.73 | 5,644,590,720 |

The native copy helper took 0.034 seconds. Peak process RSS was 9,118,728,192
bytes; RSS after preparation was 8,110,280,704 bytes. Both model buffers remained
sealed and strongly referenced during the measurement, then were released on
process exit. The per-model program words were Nanbeige 7136/7576 and Gemma
5320/6328/6592/6904. This removes preparation from a future outage only if the
owner retains and adopts these buffers rather than restarting construction.

See [software test output](data/nanbeige_staged/software-tests.log),
[host preparation receipt](data/nanbeige_staged/host-prepare.json), and
[validation metadata](nanbeige_staged_validation.json). These establish software
preparation and recovery behavior. Physical upload/execution/cleanup timing,
verified same-owner warm restoration, production adoption integration, and
deadline-aware transport/recovery remain unqualified. The 30-second total outage
claim and physical acceptance remain pending.
