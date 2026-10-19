# Future qualified C2 physical session

`tools/c2_f2_session.py` is prepared and locally tested. It has **not been executed
against an FPGA**. Its default mode validates evidence, freezes inputs and writes
a plan. `--execute` is the separate physical path; no qualification override exists.

The physical path is restricted to EC2 instance `i-0c78de2253f66519d`, slot 0,
initial application BDF `0000:34:00.0`, with original image
`agfi-0d06cb48544a02106`. It checks the instance through IMDSv2 and requires Linux
root. It does not touch the shared CoralNPU image on the other build host.

## Required evidence

The complete qualified build directory must contain matching `inputs.json` and
`outputs.json`, the routed DCP, packaged `.Developer_CL.tar`, route qualification
marker, and route validation report. The script verifies hashes, exact VU47P part,
200 MHz core, eight HBM ports, `BuildAll`, zero setup/hold/timing/skew/unrouted/DRC/
methodology blockers, six CDC witnesses, and 47 fabric gray-bus constraints.
The 47 include eight retained native write-counter groups observed after real
shell optimization; the earlier 39-constraint inventory is rejected.
Synthesis or OOC reports cannot pass this gate.

The image record comes from `tools/c2_f2_publish.py` after AWS reports the candidate
available. Schema 1 binds `build_id`, `inputs_sha256`, `outputs_sha256`,
`qualified_marker_sha256`, `validation_report_sha256`, `package_sha256`,
`dcp_sha256`, `image_id`, `global_image_id`, and `client_token` to `region=us-east-1`
and `state=available`. Before loading, the physical path makes a fresh AWS CLI
`describe-fpga-images` request and requires owner `438374459156`, matching AFI/AGFI
IDs, available state, shell `0x10212415`, and exact publisher tags `c2:build_id`, `c2:inputs_sha256`,
`c2:package_sha256`, and `c2:client_token`. A manual substitution of an unrelated
AGFI in the record does not satisfy those checks.

## Workload matrix under one image load

Prepare fixtures offline using `tools/c2_f2_bench.py prepare`. The requested matrix
is N31/KB1, N257/KB9, N31/KB54, N31/KB255 and an N1 or N3 tail case. For example:

```sh
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare --fixture build/c2-fixtures/n31-kb1.npz --rows 31 --kb 1
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare --fixture build/c2-fixtures/n257-kb9.npz --rows 257 --kb 9
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare --fixture build/c2-fixtures/n31-kb54.npz --rows 31 --kb 54
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare --fixture build/c2-fixtures/n31-kb255.npz --rows 31 --kb 255
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare --fixture build/c2-fixtures/n3-kb9.npz --rows 3 --kb 9
```

Once the real qualified build and available-image record exist, write a plan with
their actual paths. The frozen hardware BUILD_ID is read from those files; later
documentation commits do not change it.

```sh
PYTHONPATH=. python3 tools/c2_f2_session.py \
  --build-dir /path/to/qualified-build \
  --image-record /path/to/deployment.json \
  --fixture build/c2-fixtures/n31-kb1.npz \
  --fixture build/c2-fixtures/n257-kb9.npz \
  --fixture build/c2-fixtures/n31-kb54.npz \
  --fixture build/c2-fixtures/n31-kb255.npz \
  --fixture build/c2-fixtures/n3-kb9.npz \
  --out build/c2-session-plan --repeats 3
```

That command performs no AWS, FPGA utility, lock or BAR operation. It writes
`plan.json`, copies the build input manifest, and freezes each fixture after its
canonical arithmetic check. For the later authorized physical session on the
dedicated host, use the same arguments, a new output directory, and `--execute`.
An installed Python with this checkout's dependencies, AWS CLI credentials for the
read-only image query, local AWS FPGA utilities and a C compiler are required.

## Ownership, cleanup and evidence

The wrapper takes `DeviceLock("f2-0000:34:00.0", wait=0)` before checking the original
slot image, foreign BAR mappings and baseline MMIO identity. It rejects a redirected
`OTPU_RUN_DIR`; the physical lock directory must be `/tmp/otpu`. The original must be
healthy, HBM-calibrated on both existing calibration flags, and write-idle, with RUN clear or HALTED set (RUN can remain set after a
completed program). It never stops an existing workload. All BAR
mappings are closed before image changes. No Vivado process or CPU build service
is stopped, reprioritized or otherwise managed by this script.

The lock spans the candidate load, every benchmark and restoration. After every
rescan it rediscovers the slot BDF. If enumeration moves, it retains the original
lock and acquires the new canonical BDF lock before touching that device. A busy
new lock or foreign BAR owner blocks further mutations. The benchmark subprocess
receives `OTPU_LOCK_HELD` and the actual owner PID, so it inherits the existing
lock instead of attempting a second flock. Owner loss blocks further device
operations. This avoids the current lock CLI's `f2:` alias pitfall.

Loads use synchronous `fpga-load-local-image`; rescan uses
`fpga-describe-local-image -R`. Those flags follow the
[AWS F2 management-tool documentation](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/sdk/userspace/fpga-mgmt-tools/README.html).
The candidate must expose the expected benchmark ID, ABI, BUILD_ID and 200 MHz
configuration and report HBM ready before the host benchmark runs.

Restoration is attempted after **any candidate load attempt**, including a utility
failure after partial image change. SIGTERM/SIGHUP/SIGINT are recorded and deferred
through management operations and cleanup. During a benchmark, cancellation sends
SIGTERM only to its own subprocess group, allowing the host's native-response drain;
after ten seconds it may kill that group. The wrapper waits for the child to exit
before restoration. It never kills a Vivado build. If management still reports an
active load, cleanup waits rather than starting a competing load.

Restoration loads the recorded original AGFI, rescans, rediscovers the BDF, and
compares ID/version/register-map/BUILD_ID/platform/core-clock with the original
snapshot. After reload it polls readiness for up to 30 seconds before declaring
the original unhealthy. `session.json` separately records benchmark completion and verified
restoration; a restore failure remains a failed session. Command arguments,
stdout/stderr, exit status, timeout state, image identities and per-fixture hardware
artifacts are retained. The existing benchmark's exact result/traffic gates and
aggregate-only Lens output apply to every fixture.

An evidence-write failure fails the session but cannot skip restoration. Cleanup
keeps management-command output in memory so image status can still be verified
when the output disk is full. It attempts remaining evidence writes and reports
logging errors to stderr where possible. Lost evidence is never reported as a
successful session, even when restoration succeeds.

Restoration restores the **image identity**, not prior HBM, ACT, program or model
contents. A caller using the original image must reload its data. SIGKILL, host
failure, uncooperative external loaders or loss of an inherited lock can prevent
cleanup. The wrapper reports failure rather than claiming restoration in those
cases. Its lock protects cooperating clients; it cannot prohibit an unrelated
administrator from bypassing that lock.

Local tests cover qualification and AWS-binding rejection, partial-load failure,
wrong candidate identity, benchmark failure, cancellation, BDF movement, restoration
failure, no-load preflight failures, real parent/child lock inheritance and only-own-
child termination, resolved sysfs BAR mappings, evidence-disk failures and bounded
calibration polling. These are offline control-flow tests, not physical qualification.
The session suite includes an explicit rejection test for the obsolete
39-constraint qualification inventory.
