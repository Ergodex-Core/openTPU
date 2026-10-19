# Alex-based C2 build status

Recorded 2026-10-19 20:56 UTC. There are no C2 hardware measurements or qualified
C2 AFIs yet. Lens still contains the prior 128 profiles; none is a successful
C2 hardware run.

Current stage: **placement and routing is running** on `opentpu-f2` only.
Formal shell link completed successfully at 20:48:46 UTC from executor
`cec9b011bf5f66e43bfd876ce6fade63f9c0863a`. Its retained unit exited zero;
receipt SHA256 is `5c79489ceb14949b1d7c5051787e44fe959a8402882d38ade93c32f6f1db7ec6`
and linked checkpoint SHA256 is
`b8f10ea1a56aa145b9cbfa8f42246e858282dc69a307a32e0b8ed1582095922c`.
The structural audit passes; this does not qualify physical timing.
Implementation started at 20:49:25 UTC in
`/home/ubuntu/rachit/codex-c2/alex-executor-cec9b011/implement`, unit
`otpu-c2-alex-cec9b011-implement-65f84275.service`, invocation
`90d0d11102904245834319b1788e3806`, bounded to four CPUs, 64 GiB and three hours.
It consumes the exact successful synthesis/link receipts, with no RTL changes,
IP regeneration, automatic continuation, or FPGA loading.

A separate corrected linked diagnostic completed at 20:39:31 UTC. All 768 gray
bits have normal paths; its 1,119 incomplete markers are 768 four-path caps and
351 cell-detail budget caps, with zero API errors. The collect-only helper now
uses a sixteen-path bound, separate bounded gray/finding cell budgets, actual
launch-C/data-Q identity, and documented pulse reports. Its 106 qualification
tests pass; caps and hidden evidence remain pending, never accepted. Actual
replay of helper `f594a7a423cc06e469eb641d1849837c2292a0eaf68077800a7ca4ecefeb4cc9`
started at 20:48:11 UTC against the unchanged earlier linked checkpoint and
completed at 20:54:46 UTC: 151,353 supplemental queries, zero API errors or
incomplete markers. The 100,950,518-byte report still records pending semantic
review; collection completeness is not physical acceptance. The running formal
implementation retains its clean cec9 executor. A subsequent collect-only helper
adds deduplicated XPM ancestor identities; 111 qualification tests pass.

The original AWS shell diagnostic completed at 20:52:31 UTC, invocation
`1d99a307be0b4a84b1c20b774dca7f53`, with the pinned original checkpoint unchanged.
It generated methodology, DRC, CDC, timing and pulse reference reports. One
incomplete supplemental query remains for explicit review. The latest Codex
session snapshot is verified at
`s3://opentpu-agent-traces/codex/snapshots/20261019T204927.807145Z/`,
archive SHA256 `6c44071fc39fa845263a11d19c0baf7459cd48d92ba69b0e6593641467e56fde`.

The earlier failed-link 85 KB diagnostic archive remains on the instance.
Its proposed S3 upload was rejected by automatic approval review pending
explicit payload/destination approval; no upload occurred. This does not block
implementation or the separately authorized Codex trace snapshots.

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
netlist spelling was not captured. The retry failed the same interface assertion
at 18:50:05 UTC, after RTL synthesis completed. Its diagnostic checkpoint is
preserved with SHA256
`cbdfff5e76f5c211d29be6e1268ac8684086c0a8d127b871d857f183b4542871`.
The actual pin inventory shows rebuilt hierarchy: some inputs are renamed or
removed and constant address bits are pruned. The audit's assumption that all
declared boundary pins survive optimization is invalid. Connectivity diagnosis
and a full corrected audit against this checkpoint are next, before another
fresh synthesis. The failed attempt and synchronous wrapper failure remain
unchanged; neither is a successful synthesis receipt.

A read-only Vivado probe completed successfully at 18:59:42 UTC, invocation
`7b93334ca02f479a9d3a3a3db6989ad8`. It confirms the intended clock domains and
no consumers on the unused 400 MHz output. The 146 retained handshake paths
have one driver each and the expected per-port owners. Native B/R ready signals
share a reset-gated network; eight native BVALID boundary names require further
internal connectivity checks. The 16-file diagnostic archive is preserved at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/alex-d1c20797-opentpu-f2/diagnostics/probe-0885b6f3c841.tar.gz`,
SHA256 `85332c327bc6b55bcfff68c66f0bf7580d80fae3d00b36f310e55f630ad70996`.
These observations are netlist diagnostics, not routed timing qualification.

A focused write-response/shared-ready probe completed at 19:14:21 UTC on the
same preserved checkpoint, invocation `c01eeed4f1b149aa84cf4eb7f5050198`. Its
10-file archive was uploaded and downloaded through the AWS CLI, with every
file and the full archive verified against SHA256. It is preserved at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/alex-d1c20797-opentpu-f2/diagnostics/write-slim-probe-1caee1894edf.tar.gz`,
SHA256 `216595213a5e94362f0418e6afd7ff5471e0eddbe5291bc877e3b93084283a9a`.
The earlier, redundant focused probe was explicitly stopped and its partial
output plus stop reason preserved separately; it is not a completed probe.
Review confirms all eight BVALID paths reach only their own native write
counters/error state plus the correct SmartConnect response channel. The shared
ready signal is a LUT1 inverter of the core reset register, with all sixteen
native B/R channel sinks present. A Vivado net-query mismatch in the draft audit
was corrected. The complete audit replay started at 19:22:34 UTC with reviewed audit SHA256
`1833a5919b249dd550b05182b097dfe58934c65fa25e468d3b07e02c774b4c40`,
unit `otpu-c2-alex-d1c20797-audit-c616b7b12036.service`, invocation
`0d6e04c36de8496db7d48c8e6c609f24`. It opens the original checkpoint unchanged
and writes separate diagnostic evidence. It failed at 19:26:24 UTC because
Vivado rejected a reconstructed pin collection passed to `get_property`.
The helper is being corrected to re-resolve typed objects after name selection.
This is an audit API failure, not a completed structural audit or evidence of a
hardware connection defect. The failed replay is preserved at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/alex-d1c20797-opentpu-f2/diagnostics/audit-c616b7b12036.tar.gz`,
SHA256 `8249358371e511ec31cff90864ca34cc4d47aa6d5471877d81e1f29eb2e87055`.
The next synthesis source will be committed only after the full replay passes.

The corrected typed-object helper (SHA256
`609c7c39a2e6751d596dd75a91b20618556cb963cc5c6793edda25b48fdee54b`)
passed the focused actual Vivado check at 19:32:23 UTC: all 154 individual paths
and sixteen shared B/R-ready connections pass, with zero failed audit checks.
Invocation `5a04f91fd8f24c71916e5f7d4865a92b` and its ten evidence files are
preserved at
`s3://cg-rachit-afi-438374459156-us-east-1/opentpu/c2/2026-10-19/alex-d1c20797-opentpu-f2/diagnostics/flow-f31e4f0ae9fa.tar.gz`,
SHA256 `ce726ad3cfd7bcde32c590cc8d694544be6aac582a74bfe5e77ebf9de342f139`.
The full audit replay restarted at 19:32:52 UTC, unit
`otpu-c2-alex-d1c20797-audit-2ab32bac0504.service`, invocation
`7e22061697414033bd778b24126c9d7f`. It failed at 19:36:57 UTC on a
`HD_IDF_PR_InsertedInst_S_HOST_awready` buffer absent from the passing focused
flow check. The full audit writes a diagnostic checkpoint before topology
inspection; checkpoint serialization was the first hypothesis tested for this
insertion side effect. No timing or route qualification has passed.

The ordering change moves diagnostic serialization after the live audit and reports,
and preserves the original failure if subsequent diagnostic capture also fails.
Audit SHA256 `0c45443615a0586a4a18221dd155b3422209e45bf93d2789fb1b4dbfd83b9706`
started replay at 19:40:35 UTC, unit
`otpu-c2-alex-d1c20797-audit-9826d16760f1.service`, invocation
`a6d8bbaf52c2497fa977219e757bb340`. It failed on the same inserted buffer at
19:44:01 UTC before serialization, so the ordering change does not resolve the
underlying observation difference. A separate checkpoint probe completed at 19:45:34 UTC, invocation
`d23dd465d6f4480aa2fd7a51cf7afc62`. It records zero inserted cells and all 154
individual flows passing before saving, after saving, and after reopening the
copy; serialization alone is ruled out. Its archive is preserved at
`diagnostics/checkpoint-1929fb94e062.tar.gz` under the same diagnostic prefix,
SHA256 `14987ef151004db917fc37ac1e0c63df7272e993c10999e04dd3991022bdd436`.
The ordered prefix probe launched at 19:50:14 UTC, invocation
`201edeeb29bb4214b3c72d0d78a488b9`, completed at 19:52:32 UTC on the unchanged
original checkpoint. Immediately before `list_property` on `u_mem`, no inserted
buffers are visible under that hierarchy; immediately afterward 87 are visible.
The subsequent checks report 77 failed individual flows and one failed
shared-ready check. The global inventory contains 608 pass-through LUT1 buffers;
other hierarchies had also been enumerated earlier in the prefix. Its ten verified files are preserved at
`diagnostics/prefix-86424527a580.tar.gz` under the same prefix, SHA256
`abe08b67c09d851c830ba8a63821d5989d217c7389b25f979dd06e204139ec7d`.
This localizes the discrepancy to hierarchical property enumeration before clock
queries or serialization. The corrected helper reads required properties directly
and retains all reference, black-box, port-ownership and clock checks. Audit SHA256
`a9d91bef071f935089dc8cbc157d053253f00fe7dd667d5e6dca7ccb8cd4d349`
started full replay at 19:54:31 UTC, invocation
`26c77a21f1f8453592fd5596cfd42497`, and completed successfully at 20:03:23 UTC.
All 154 individual handshake paths and all sixteen shared-ready connections pass.
The full report inventory was generated; there are no unresolved vendor roots,
deferred clocks, or custom HBM-clock registers. The original checkpoint hash is
unchanged. This successful diagnostic replay does not repair the earlier failed
synthesis receipt or qualify routing. The successful 24-file diagnostic archive
is preserved under the same prefix as `diagnostics/audit-1018bd310949.tar.gz`,
SHA256 `ff9ebac1520d594dd4d2d3049e75081cf2c0e3aa301bd7be6bc337197a9f37ca`.
Fresh synthesis with synchronous dependency capture started at 20:05:49 UTC from
`d75970db0e1fc00c1ae9f9eeced3ebd2d4ea6786`, unit
`otpu-c2-alex-d75970db-synth-retry.service`, invocation
`3227447bcd034ef38ad8a26dc772b9f6`, directory
`/home/ubuntu/rachit/codex-c2/alex-synth-d75970db`. It reuses the completed IP264
receipt and unchanged synchronous wrapper. Its 245,440-byte source bundle has
SHA256 `0415b39ca975fdfe74fe9a9153999dbf80f5cb601e1829c7571877dfa0acb876`.
Fresh synthesis completed successfully at 20:18:57 UTC. Its synthesis receipt
SHA256 is `6e1dc207d6fa59ccd284140294bea22987e19e2947da4a29c5c3d82ef0dcaa98`,
and its synchronous dependency attestation is verified, SHA256
`d7c3db148e10c42820fccc7146104c65c13490cf8b7e08aa3c504740d9021fad`.
The full audit again passes all 154 handshake witnesses and sixteen shared-ready
connections. The updated audit/qualification suite passes 192 tests.
Shell linking started at 20:19:31 UTC on the same source and instance, unit
`otpu-c2-alex-d75970db-link-ad580007.service`, invocation
`96960de224d641829d5e9bf62d440ac7`, output directory
`/home/ubuntu/rachit/codex-c2/alex-synth-d75970db/link`. It is bounded to four
CPUs, 64 GiB and three hours, with no automatic continuation or hardware actions.
The replay reports are pre-placement, out-of-context evidence: setup WNS is
+0.583 ns and hold WHS is -0.107 ns, with 271,900 failing hold endpoints. The
worst reported core setup path is command-validation control from
`cfg_rows_reg[4]` to `state_reg[0]/CE` (19 logic levels). These are routing risks,
not timing closure. Disabled-DDR clock I/O standard findings must be rechecked
after the frozen implementation constraints assign the exact M_CLK pair.
The report review found no new proven reason to stop before shell linking.
The collect-only linked/final qualifier adds bounded per-bit gray, synchronizer,
finding and waiver evidence with explicit incomplete/unsupported states. The
evidence verifier requires this supplemental report; 83 evidence tests and 127
collector/package/publisher/session integration tests pass.
The previous failed replay is preserved in S3 as
`diagnostics/audit-2ab32bac0504.tar.gz` under the same diagnostic prefix, SHA256
`d4a0ae3cf7296e8350b9251c964afa51403bb1e75514149e22e4dd626aa99d0e`.

Lens now labels C2 operand counters as overlapping fill occupancy, the optimistic native-interface
bound as a lower bound rather than HBM efficiency, and unavailable unit and
instruction measurements as unmeasured. Its scoped host/session/viewer tests
pass (83 tests); no measured profiles have been added.

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

The latest Codex trace snapshot captured at 19:52 UTC contains 374 verified files
(363,229,012-byte archive), uploaded through the AWS CLI to
`s3://opentpu-agent-traces/codex/snapshots/20261019T195201.475727Z/` with SHA256,
private access, encryption and versioning verified. Its local receipt is
`/Users/rachittibrewal/ergodex-04/codex-session-archives/s3-upload-receipt-20261019T195320.198905Z.json`.
Earlier snapshots remain preserved. This is an individually timed snapshot of
each file, not a globally atomic or continuously synchronized archive.

The first shell-link invocation produced `post_link.dcp` (71,076,276 bytes,
SHA256 `21aa0de424a557770012421774108dc908633a5051afe281ba9ea269f5c48918`),
but its subsequent structural audit failed at 20:23:06 UTC. The failed stage
receipt is `d3f532c8ad99c7122724a5dc349aba874dd875497536d0d050d118b8ec33a52c`.
All eight exact native cell checks passed; the following inventory regexp omitted
`-hierarchical`. A diagnostic on the unchanged linked checkpoint, launched at
20:25:16 UTC under invocation `6f1aed3bf3544feba88eafce4db1d7cc`, confirms that
the original selector returns no cells while the corrected selector returns exactly
`WRAPPER/CL/u_native0` through `u_native7`. Its corrected full audit is still
running, so there is no successful link receipt yet. Qualifier SHA256 is
`b59899c472591c2c9cd435fe86dbf26ca5eb88936db41969711d8e6b0d7ebc6b`;
the frozen synthesis audit and all hardware source inputs remain unchanged.
The qualification suite passes 94 tests, including missing and extra native
instances with realistic top-scope regexp behavior.

The failed link's eleven textual evidence files are captured separately on
`opentpu-f2` at `/home/ubuntu/rachit/codex-c2/alex-d75970db-link-failure-review`.
The archive is 85,177 bytes, SHA256
`ada265a6d973163d781f043f035e4361d016d07dad363d3c98ab7eadfd1cf236`;
the checkpoint binary is excluded but its original hash is verified. Upload to
the build-artifact bucket awaits explicit destination approval following automatic
approval review; no upload has occurred. This does not block local build analysis.
The authorized Codex trace snapshot at 20:20:15 UTC was uploaded and verified at
20:22:04 UTC under `s3://opentpu-agent-traces/codex/snapshots/20261019T202015.774257Z/`.

The `b59899c4` diagnostic confirmed the native selector correction, then failed
at 20:30:24 UTC in `findings`: an `expr` ternary converted a Vivado finding
handle into a string before its severity query (`RTSTAT-1#1`). The earlier
property snapshot had already read that severity successfully. The correction
uses the captured finding severity, falling back to the captured rule severity
only when absent, and rejects missing, empty or unknown values. Qualifier SHA256
`c072db502318cadcd4a38b47438fcaef69a5b91fe7090374a629558e502eef9d`
passes 99 qualification tests. A fresh bounded diagnostic is running against
the same unchanged linked checkpoint; neither diagnostic changes hardware inputs
or upgrades the failed formal link receipt.
