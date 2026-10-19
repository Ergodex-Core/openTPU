# Alex-based C2 build status

Recorded 2026-10-19 18:50 UTC. There are no C2 hardware measurements or qualified
C2 AFIs yet. Lens still contains the prior 128 profiles; none is a successful
C2 hardware run.

The supplied baseline is
`s3://ergodex-bonsai-on-chip-438374459156-us-east-1/opentpu-alex/alex-burst200-branch.tar`,
SHA256 `fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc`,
Alex revision `06c169cbcd1721ae8b167de992a93461697a397d`. The C2 adaptation is
isolated on branch `codex/v4-c2-alex-burst200`. It preserves eight 512 MiB HBM
windows and the existing host/kernel byte layout.

Build source `264d84134da11630622002cd869cf88b07e49aec` was launched on
**opentpu-f2**, instance `i-0c78de2253f66519d`, at 18:04:34 UTC. The bounded unit
`otpu-c2-alex-264d8413-ip-synth.service`, invocation
`cf718d0a345b44c89db86ab3bfd8c8a2`, generates private IP then synthesizes C2.
Its directory is `/home/ubuntu/rachit/codex-c2/alex-264d8413`. The unit does not
route, package, publish, load an FPGA, or execute benchmarks. All private IP
completed at 18:19 UTC. Its receipt SHA256 is
`c21aae37ad72606811b17c9b212aa26d0ef6f8b31196494057c86eb16bd39799`.
The actual generated BD passes the eight-window and 56-exclusion checks.

RTL synthesis finished, but its audit failed at 18:30:05 UTC on the synthesized
`S_HOST/awvalid` pin check. No `post_synth.dcp` or synthesis-complete receipt was
produced, and shell linking has not started. The failed run and logs remain
intact. A fresh synthesis attempt against the completed private IP started at
18:37:45 UTC from source `d1c2079715a14ac62b5cc81fef2317bf90840b7c`, directory
`/home/ubuntu/rachit/codex-c2/alex-synth-d1c20797`, unit
`otpu-c2-alex-d1c20797-synth-retry.service`, invocation
`6fecc7771d65463ba2daa40d6c6a9649`. The new audit accepts scalar or singleton-bit
handshakes with strict cardinality checks and preserves exact pin names plus an
explicitly unqualified diagnostic checkpoint. The earlier failure's exact
netlist spelling was not captured, so the new attempt must establish the outcome.

A supplemental dependency observer captured all generated `.srcs`/`.gen` files
before synthesis at 18:19:01 UTC. It correctly recorded failure when the build
failed; its early capture is not a successful attestation. Both records remain in
`/home/ubuntu/rachit/codex-c2/alex-264d8413-dependency-observer`.

The 105,815-byte source bundle is preserved at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/alex-264d8413-opentpu-f2/source.bundle`,
SHA256 `8ae89ceaf22a142e28000a26377ecc2355d706d47a406db64d91fe300cb389af`.
It requires Git base `7028e35e8720cb70d7b38d65010813dd23d88bcf`. Local launch,
upload, IAM and SSM receipts are under `build/c2-alex-remote/`.

The earlier Alex attempt `aa7a23ed` was deliberately stopped before continuing
with an incorrect BD parser. Vivado represents blocked native crosspoints as
explicit `is_excluded: TRUE` segments. The corrected parser checks all 16 active
windows and all 56 excluded crosspoints; it passes against the actual generated
BD. The earlier source, partial products, logs and stop reason remain in
`/home/ubuntu/rachit/codex-c2/alex-aa7a23ed`.

Local validation passed: 69 build/BD tests, 15 top/native-adapter tests, 46
synthesis-audit tests and 24 C2 kernel regressions. These establish software/RTL
checks, not physical timing or throughput. The native-adapter tests include
PC-window and 4 KiB boundaries. Hardware qualification still requires the current
9SI/8MI vendor simulation, real shell link, implementation/refinement and review
of timing, CDC, exceptions, routing and design-rule evidence.

The new evidence collector/verifier, qualified-checkpoint packager, AWS CLI
publisher and exclusive FPGA-session integration passed 208 tests together.
They are prepared for this build; no package, AFI or physical session has yet
been produced from the Alex-based C2 integration.

Vendor simulation runs independently using a private copy of that IP. Executor
`8af52428f3a07bbc627976609084c0a95b83e2d2` stopped before test traffic on a Vivado
Tcl option-parsing error. The corrected property invocation was verified directly
in Vivado 2025.2. Retry executor `1639daacc236822f91223a17874ebf5bf18a8c12` started
at 18:27:01 UTC, unit `otpu-c2-alex-1639daac-simulation.service`, invocation
`9fe6c7f7daee4b448115db965fdba6f1`. Seed17/phase0 passed all eight PC scoreboards,
reset-with-pending-reads coverage, host response counts and the RD32 limit.
All four scenarios (seeds17/71, phases0/1) passed by 18:38:43 UTC, and the unit
exited successfully. The complete receipt SHA256 is
`ee38e5ead29c8697524ac65ebba8d235632abdc99b31f85fff7251f98ae6dac2`.
This is protocol simulation, not hardware throughput or routed timing. The directory is
`/home/ubuntu/rachit/codex-c2/alex-executor-1639daac/simulation`.

The previous custom-fabric full build `7028e35e` failed routed setup timing. That
failure and the unlaunched R5 alternative remain preserved in the original
`opentpu-v4-c2-dual-mode-mxu` worktree. No work was performed on fpga-dev-f2 for
this Alex integration.

The latest Codex trace snapshot captured at 18:35 UTC contains 374 verified files
(356,041,443-byte archive), uploaded through the AWS CLI to
`s3://opentpu-agent-traces/codex/snapshots/20261019T183511.263781Z/` with SHA256,
private access, encryption and versioning verified. Its local receipt is
`/Users/rachittibrewal/ergodex-04/codex-session-archives/s3-upload-receipt-20261019T183631.062531Z.json`.
Earlier snapshots remain preserved. This is an individually timed snapshot of
each file, not a globally atomic or continuously synchronized archive.
