# Alex-based C2 build status

Recorded 2026-10-19 18:04 UTC. There are no C2 hardware measurements or qualified
C2 AFIs yet. Lens still contains the prior 128 profiles; none is a successful
C2 hardware run.

The supplied baseline is
`s3://ergodex-bonsai-on-chip-438374459156-us-east-1/opentpu-alex/alex-burst200-branch.tar`,
SHA256 `fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc`,
Alex revision `06c169cbcd1721ae8b167de992a93461697a397d`. The C2 adaptation is
isolated on branch `codex/v4-c2-alex-burst200`. It preserves eight 512 MiB HBM
windows and the existing host/kernel byte layout.

The active source is `264d84134da11630622002cd869cf88b07e49aec`, launched on
**opentpu-f2**, instance `i-0c78de2253f66519d`, at 18:04:34 UTC. The bounded unit
`otpu-c2-alex-264d8413-ip-synth.service`, invocation
`cf718d0a345b44c89db86ab3bfd8c8a2`, generates private IP then synthesizes C2.
Its directory is `/home/ubuntu/rachit/codex-c2/alex-264d8413`. The unit does not
route, package, publish, load an FPGA, or execute benchmarks.

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

The previous custom-fabric full build `7028e35e` failed routed setup timing. That
failure and the unlaunched R5 alternative remain preserved in the original
`opentpu-v4-c2-dual-mode-mxu` worktree. No work was performed on fpga-dev-f2 for
this Alex integration.

The Codex trace snapshot captured at 17:49 UTC contains 373 verified files
(350,578,160-byte archive), uploaded through the AWS CLI to
`s3://opentpu-agent-traces/codex/snapshots/20261019T174919.880987Z/` with SHA256,
private access, encryption and versioning verified. Its local receipt is
`/Users/rachittibrewal/ergodex-04/codex-session-archives/s3-upload-receipt-20261019T175052.195995Z.json`.
