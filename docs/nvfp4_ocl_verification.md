# NV41 code/simulation gate

The implementation is an isolated derivative of pinned `wide-port-alexshell`
`0a868822a71422beb11771c9fa046af762579e81`, plus the previously tested strict
NVFP4 numerical/reference/controller commits. The protected active wide-port
worktree was not edited. No binary MM/WNV/STREAM instruction encoding changed.

## Verified scope

- Versioned NV41 discovery/descriptor, one credit, exclusive registered HBM0
  windows, complete byte staging, atomic acceptance, held completion and explicit
  cancellation/reset recovery; `docs/nvfp4_ocl.md` is the normative ABI contract.
- One strict output, K=1..64, both gradual and DAZ/FTZ profiles. Separate product,
  tree, scale, ordered accumulation and global-scale rounding remain unchanged.
- Default-off build compatibility; opt in with `f2-hbm:nvfp4`. Target remains the
  existing F2 HBM architecture/platform at nominal 200 MHz.

## Executed checks

| Check | Result |
|---|---|
| Selected numerical/job/transport/frontend/host/target/F2 preparation software | 472 passed, 5 RTL cases deselected |
| Existing NVFP4 aggregate arithmetic/controller/transport RTL | 14 passed, 1 non-RTL case deselected; Icarus and Verilator |
| New native ownership counter/routing RTL | 2 passed; 131080 legacy + 7711 sidecar B completions per backend, multiple raw-count wraps |
| New native job protocol RTL | 2 passed; each backend verifies 4 strict rows, command-first and data-first ready skew, delayed B visibility and 16-bit counter wrap |
| Production F2 HBM top with simulated AXI memory | 3 passed: enabled NXR=0 and3, disabled NXR=3 |
| Existing F2 OCL crossing/native-AXI regression | 13 passed, 21 unrelated cases deselected |
| Verilator lint/elaboration at pinned baseline, new default-off and new enabled top | All succeeded with the same 67 legacy warnings by kind/file; no new NVFP4 warning |
| Git whitespace check | Passed |

Each enabled production-top run executes 173 raw descriptors: 134 exact golden
successes, 10 expected numerical failures, 29 malformed/permission/bounds/profile/
epoch rejections. It also checks independent host AW/W timing, held R/B responses,
private output until acknowledged native writes, completion identity/sequence,
credit isolation, rejected legacy writes while owned, real legacy RUN and loader
exclusion/drain, partial-byte table/descriptor writes, invalid control strobes,
unaligned/out-of-aperture accesses, generation rollback/exhaustion, soft cancellation
and reset with held reads, reset after irreversible store, coordinated reset with
pending read, HBM readiness and injected R/B faults. Legacy loader traffic uses a
simulated two-channel burst-capable memory model. The sidecar itself stays on HBM0.

These production-top tests use S=1, NCH=2, reduced memory depths, generic tech
cells and NXR=0/3. They exercise production RTL, not proprietary SmartConnect/HBM
IP or physical crossings. Multislice admission and complete legacy program/model
execution are not qualified by this gate. Native owner tests separately prove
count preservation around ownership changes; no claim of full legacy inference
execution is made.

The software commands select `tests/test_nvfp4.py`, `test_nvfp4_epilogue.py`,
`test_nvfp4_frontend.py`, `test_nvfp4_local_job.py`, `test_nvfp4_transport.py`,
`test_nvfp4_ocl.py`, `test_target.py`, and `test_f2_build.py`, with `-k 'not rtl'`.
The F2 RTL command selects `tests/test_f2_rtl.py -k 'ocl_crossing or test_native_axi'`.
New test files and vector generators are tracked beside the existing tests.
Detailed compile logs, simulation logs, vectors and lint comparison are retained
under `build/nvfp4-gate6/` in the working checkout and packaged evidence.

Tools: Icarus Verilog13.0, Verilator5.048. Existing Anaconda Python3.10/numpy1.23.5
runs numerical RTL tests. F2 preparation tests require `hashlib.file_digest`, so
the already-installed `emamba` Python3.11 runtime/numpy is used, appending the
existing pure-Python pytest7.1.2 package path; no software installation. All runs
set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Existing F2 regression sets Verilator's
make parallelism to2 in the runner. A reproducible runner accompanies evidence.

## Independent review

An independent Sol reviewer identified and verified fixes for generation rollback
before first ARM, missing calibration admission, uncertain launch recovery,
uncertain reconfiguration retaining stale windows, lost terminal outcomes on
uncertain ACK and explicit proven recovery of an ACK that was not executed.
The final read-only review reports no unresolved blocking source findings.
Testbench oracle/parameter/simulator issues discovered during validation were
fixed in tests; numerical diagnostics and production semantics were not weakened.

## Unrun stages and next gate

No Vivado synthesis, implementation, timing, CDC report, power, vendor-IP simulation,
AFI build, FPGA load or hardware test ran. `vivado` is not on PATH;
AWS_FPGA_REPO_DIR, HDK_DIR and XILINX_VIVADO are unset, and checked local HDK/Vivado
locations are absent. Licenses were not changed or tested. The pinned flow expects
HDK `20fe90180574464fd509c994680ec67191e9759f`, IP
`6d32be972e6da854e61a8d3d6ec0466ab491c1b3`, SmallShell0x10212415 and Vivado2025.2.
The preparation tests use fake HDK stubs and do not substitute for those products.

There is no timing closure, area fit, TPS, full matrix/layer, mixed checkpoint or
complete inference claim. VU47P has no native FP32 DSP primitive; this strict
conformance RTL still requires deliberate implementation/resource work.

The next architecture gate belongs to the parent task: a device-resident tiled
scheduler/token loop with embedding, mixed precision layer operators, attention,
KV management, BF16 head/token selection and explicit speculative draft/verify/
commit/rollback state. The host must ultimately only setup/load and exchange
requests/tokens. NV41 remains a bounded diagnostic interface; today it still
requires per-output host submission/poll/ACK. No host fallback may be called
complete FPGA inference. No model weights were downloaded or executed, and no
RHT was enabled on unchanged weights.
