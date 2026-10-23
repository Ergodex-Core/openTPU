#!/usr/bin/env python3
"""Plan, or explicitly execute, one qualified C2 load/benchmark/restore session.

Default behavior only validates local evidence and writes a reviewable plan. The
--execute path is restricted to the dedicated F2 instance and never builds an AFI.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import c2_f2 as C
from opentpu.host import regs as R
from opentpu.host.runstate import DeviceLock

INSTANCE = "i-0c78de2253f66519d"
ORIGINAL = "agfi-0d06cb48544a02106"
INITIAL_BDF = "0000:34:00.0"
SLOT = 0
AGFI_RE = re.compile(r"agfi-[0-9a-f]{17}")
BDF_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-1][0-9a-f]\.[0-7]")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")
    temp.replace(path)


def qualify(build_dir, image_record):
    """Fail closed on synthesis-only, missing physical gates or mismatched AFI provenance."""
    root = Path(build_dir).resolve()
    inputs = json.loads((root/"inputs.json").read_text())
    outputs = json.loads((root/"outputs.json").read_text())
    deployment = json.loads(Path(image_record).read_text())
    build = inputs.get("build_id", "")
    if (inputs.get("kind") != "c2-hbm-kernel" or not re.fullmatch(r"[0-9a-f]{8}", build) or
            inputs.get("part") != "xcvu47p-fsvh2892-2-e" or inputs.get("core_khz") != 200000 or
            inputs.get("hbm_ports") != 8 or outputs.get("flow") != "BuildAll" or
            outputs.get("inputs_sha256") != sha(root/"inputs.json")):
        raise ValueError("not a completed, matching C2 200 MHz BuildAll record")
    artifacts = outputs.get("artifacts", {})
    for name, info in artifacts.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != info.get("sha256"):
            raise ValueError(f"artifact missing/changed or outside the qualified build: {name}")
    dcps = [k for k in artifacts if k.endswith(f"cl_opentpu.{build}.post_route.dcp")]
    archives = [k for k in artifacts if k.endswith(".Developer_CL.tar")]
    if len(dcps) != 1 or len(archives) != 1:
        raise ValueError("qualified routed DCP and packaged AFI archive required")
    marker = root/"cl_opentpu/build/checkpoints"/f"cl_opentpu.{build}.route_validation.passed"
    report = root/"cl_opentpu/build/reports"/f"cl_opentpu.{build}.route_validation.txt"
    if marker.read_text().strip() != "OTPU_STAGE_ROUTE_VALIDATION_PASSED":
        raise ValueError("route qualification marker missing or invalid")
    values = dict(line.split("=", 1) for line in report.read_text().splitlines() if "=" in line)
    for name in ("setup_failed", "hold_failed", "timing_failed", "bus_skew_failed",
                 "unrouted_nets", "drc_blockers", "methodology_blockers"):
        if values.get(name) != "0":
            raise ValueError(f"physical qualification failed or absent: {name}")
    if values.get("gray_bus_constraints") != "47" or values.get("cdc_crossing_witnesses") != "6":
        raise ValueError("missing reviewed HBM CDC constraint inventory")
    expected = dict(schema=1, region="us-east-1", build_id=build, inputs_sha256=sha(root/"inputs.json"),
                    outputs_sha256=sha(root/"outputs.json"),
                    dcp_sha256=artifacts[dcps[0]]["sha256"],
                    package_sha256=artifacts[archives[0]]["sha256"],
                    qualified_marker_sha256=sha(marker), validation_report_sha256=sha(report), state="available")
    if any(deployment.get(k) != v for k, v in expected.items()):
        raise ValueError("deployment record does not bind this qualified build to the dedicated host")
    agfi = deployment.get("global_image_id", "")
    if not AGFI_RE.fullmatch(agfi) or agfi == ORIGINAL or not re.fullmatch(r"afi-[0-9a-f]{17}", deployment.get("image_id", "")):
        raise ValueError("deployment record requires a distinct candidate AGFI and regional AFI ID")
    token = deployment.get("client_token")
    if not isinstance(token, str) or not 1 <= len(token) <= 64:
        raise ValueError("deployment record requires its idempotent publication client token")
    return dict(**expected, agfi_id=agfi, afi_id=deployment["image_id"],
                build_manifest=str(root/"inputs.json"), deployment_sha256=sha(image_record),
                qualification_report=values, client_token=token)


def verify_aws_image(response, q):
    images = response.get("FpgaImages", [])
    if len(images) != 1:
        raise RuntimeError("AWS did not return exactly the expected candidate image")
    item = images[0]
    tags = {tag["Key"]:tag["Value"] for tag in item.get("Tags", [])}
    expected = {"c2:build_id":q["build_id"],"c2:inputs_sha256":q["inputs_sha256"],
                "c2:package_sha256":q["package_sha256"],"c2:client_token":q["client_token"]}
    if (item.get("OwnerId") != "438374459156" or item.get("State", {}).get("Code") != "available" or
            item.get("FpgaImageId") != q["afi_id"] or item.get("FpgaImageGlobalId") != q["agfi_id"] or
            item.get("ShellVersion", "").lower() != "0x10212415" or
            any(tags.get(key) != value for key,value in expected.items())):
        raise RuntimeError("AWS image state, owner, IDs or publication hash tags differ from the qualified binding")
    return dict(owner=item["OwnerId"],state=item["State"],tags=tags,
                afi_id=item["FpgaImageId"],agfi_id=item["FpgaImageGlobalId"])


def parse_image(text):
    rows = [line.split() for line in text.splitlines() if line.split()[:1] == ["AFI"]]
    if len(rows) != 1 or len(rows[0]) < 7 or rows[0][1] != str(SLOT):
        raise RuntimeError("ambiguous or unsupported slot status output")
    row = rows[0]
    return dict(agfi=row[2], state=row[3], status_code=int(row[4], 0),
                error=row[5], error_code=int(row[6], 0))


def parse_bdf(text, *, allow_unconfigured=False):
    rows = [line.split() for line in text.splitlines()
            if len(line.split()) == 5 and line.split()[:2] == ["AFIDEVICE", str(SLOT)]]
    if len(rows) != 1 or not BDF_RE.fullmatch(rows[0][4].lower()):
        raise RuntimeError("slot has no unambiguous application-PF BDF")
    vendor, device = (int(x, 0) for x in rows[0][2:4])
    if vendor != 0x1D0F or (not allow_unconfigured and device != 0xF001):
        raise RuntimeError("slot application PCI identity does not match the benchmark/baseline")
    return rows[0][4].lower()


def require_image(image, agfi):
    if image != dict(agfi=agfi, state="loaded", status_code=0, error="ok", error_code=0):
        raise RuntimeError(f"unexpected slot image/status: {image}")


def require_baseline(values):
    status = values["status"]
    ready = R.ST_WR_IDLE | R.ST_CALIB0 | R.ST_CALIB1
    if (values["id"] != R.ID_OTPU or values["platform"] != 3 or values["core_khz"] != 200000 or
            status & ready != ready or (status & R.ST_RUN and not status & R.ST_HALTED) or
            status & (R.ST_ERROR | R.ST_LOADING | R.ST_AXI_ERR | R.ST_ECC_DED)):
        raise RuntimeError(f"original image is not a healthy idle openTPU baseline: {values}")


def stderr_fallback(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass  # An unavailable evidence sink must never prevent image restoration.


def foreign_handles(bdf, *, proc_root=Path("/proc"), sysfs_root=Path("/sys/bus/pci/devices"), own_pid=None):
    """Linux exposes resolved /sys/devices paths through proc, not just sysfs aliases."""
    own_pid = os.getpid() if own_pid is None else own_pid
    device = sysfs_root/bdf
    prefixes = (str(device)+"/resource", str(device.resolve())+"/resource")
    offenders = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            for fd in (proc/"fd").iterdir():
                try:
                    if os.readlink(fd).startswith(prefixes):
                        offenders.append(dict(pid=int(proc.name),fd=fd.name))
                except FileNotFoundError:
                    pass
            if any(any(prefix in line for prefix in prefixes) for line in (proc/"maps").read_text().splitlines()):
                offenders.append(dict(pid=int(proc.name),mapping=True))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return offenders


def session(ops, qualification, fixtures, *, repeats=3):
    """Injectable control flow. `ops` owns locks, subprocesses and atomic evidence."""
    if ops.instance_id() != INSTANCE:
        raise RuntimeError("refusing hardware access outside the dedicated F2 instance")
    report = dict(instance_id=INSTANCE, slot=SLOT, qualification=qualification,
                  phase="preflight", restored=False, load_attempted=False)
    ops.record(report)
    report["aws_image_binding"] = ops.verify_image_binding(qualification)
    ops.record(report)
    with ExitStack() as stack:
        held = {}
        def own(bdf):
            key = "f2-"+bdf
            if key not in held:
                held[key] = ops.acquire(key)
                stack.callback(held[key].release)
            ops.no_foreign_handles(bdf)
            return held[key]
        initial = ops.discover()
        if initial != INITIAL_BDF:
            raise RuntimeError("dedicated slot topology changed before acquiring the expected canonical lock")
        own(initial)
        require_image(ops.describe(), ORIGINAL)
        before = ops.baseline(initial)
        require_baseline(before)
        report.update(original_identity=before, initial_bdf=initial)
        ops.record(report)
        failure, load_attempted = None, False
        logging_errors = []
        def record_cleanup():
            try:
                ops.record(report)
            except Exception as error:
                message = f"session evidence: {type(error).__name__}: {error}"
                logging_errors.append(message)
                stderr_fallback(message)
        def collect_logging_errors():
            errors = list(dict.fromkeys(logging_errors + getattr(ops, "logging_errors", [])))
            if errors:
                report["logging_errors"] = errors
            return errors
        try:
            ops.check_cancel()
            report["phase"] = "load_candidate"
            ops.record(report)
            load_attempted = True
            report["load_attempted"] = True
            ops.load(qualification["agfi_id"])
            ops.rescan()
            bdf = ops.discover()  # never reuse a pre-rescan BDF or BAR mapping
            lock = own(bdf)       # retain original lock; add new BDF lock if enumeration moved
            report["candidate_bdf"] = bdf
            require_image(ops.describe(), qualification["agfi_id"])
            report["candidate_identity"] = ops.candidate(bdf, int(qualification["build_id"], 16))
            ops.check_cancel()
            report["phase"] = "benchmark"
            ops.record(report)
            report["completed_fixtures"] = []
            for idx, fixture in enumerate(fixtures):
                ops.benchmark(bdf, lock, qualification, fixture, repeats, idx)
                report["completed_fixtures"].append(str(fixture))
                ops.record(report)
                ops.check_cancel()
            ops.check_cancel()
            report["benchmark_complete"] = True
        except BaseException as error:
            failure = error
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            ops.restoring = True
            # A load command can fail after changing the image: restoration is required
            # from the first load attempt, even if candidate verification never succeeded.
            if load_attempted:
                report["phase"] = "restore_original"
                record_cleanup()
                try:
                    ops.wait_terminal()
                    own(ops.discover(allow_unconfigured=True))
                    ops.load(ORIGINAL, restoring=True)
                    ops.rescan()
                    restored_bdf = ops.discover()
                    own(restored_bdf)
                    require_image(ops.describe(), ORIGINAL)
                    after = ops.baseline(restored_bdf, wait=True)
                    require_baseline(after)
                    if {k:v for k,v in after.items() if k != "status"} != {k:v for k,v in before.items() if k != "status"}:
                        raise RuntimeError("restored image identity differs from the original snapshot")
                    report.update(restored=True, restored_bdf=restored_bdf, restored_identity=after)
                except BaseException as error:
                    report["restore_error"] = f"{type(error).__name__}: {error}"
                    if failure is None:
                        failure = error
            errors_before = collect_logging_errors()
            if errors_before and failure is None:
                failure = RuntimeError("session evidence could not be fully recorded")
            report["phase"] = "complete" if failure is None else "failed"
            record_cleanup()
            # A failure writing the final result also fails the session, after cleanup.
            errors_after = collect_logging_errors()
            if errors_after and (errors_after != errors_before or failure is None):
                if failure is None:
                    failure = RuntimeError("session evidence could not be fully recorded")
                report["phase"] = "failed"
                record_cleanup()
        if failure is not None:
            raise RuntimeError("C2 session failed; see session.json or stderr for restoration/evidence status") from failure
    return report


class LocalOps:
    def __init__(self, out, python):
        self.out, self.python = Path(out), python
        self.cancelled, self.seq = None, 0
        self.locks = {}
        self.restoring, self.logging_errors = False, []

    def evidence(self, label, write):
        try:
            write()
        except Exception as error:
            if not self.restoring:
                raise
            message = f"{label}: {type(error).__name__}: {error}"
            self.logging_errors.append(message)
            stderr_fallback(message)

    def record(self, report):
        self.evidence("session evidence", lambda: write_json(self.out/"session.json",
            report | {"updated_utc": datetime.now(timezone.utc).isoformat(), "signal": self.cancelled}))

    def check_cancel(self):
        if self.cancelled is not None:
            raise InterruptedError(f"session interrupted by signal {self.cancelled}")

    def instance_id(self):
        if sys.platform != "linux" or os.geteuid() != 0:
            raise RuntimeError("physical session requires Linux root on the dedicated F2 host")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                         headers={"X-aws-ec2-metadata-token-ttl-seconds":"60"})
        with opener.open(request, timeout=3) as response:
            token = response.read().decode()
        request = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id",
                                         headers={"X-aws-ec2-metadata-token":token})
        with opener.open(request, timeout=3) as response:
            return response.read().decode().strip()

    def acquire(self, name):
        if os.environ.get("OTPU_RUN_DIR", "/tmp/otpu") != "/tmp/otpu":
            raise RuntimeError("physical session requires the canonical /tmp/otpu lock directory")
        lock = DeviceLock(name, wait=0)
        self.locks[name] = lock
        return lock

    def require_owner(self, bdf=None):
        if not self.locks or (bdf is not None and "f2-"+bdf not in self.locks):
            raise RuntimeError("FPGA operation has no canonical device lock")
        if any(lock.lost() or (lock.fd is None and lock.holder is None) for lock in self.locks.values()):
            raise RuntimeError("inherited FPGA lock owner exited; refusing further device operations")

    def verify_image_binding(self, q):
        raw = self.command("describe-afi", ["aws","ec2","describe-fpga-images","--region","us-east-1",
            "--fpga-image-ids",q["afi_id"],"--output","json","--no-cli-pager"],timeout=60)
        return verify_aws_image(json.loads(raw),q)

    def command(self, role, args, *, timeout=180, env=None, cancellable=False):
        self.seq += 1
        stem = f"{self.seq:02d}-{role}"
        self.evidence(stem+" command evidence", lambda: write_json(self.out/(stem+".command.json"), {"args":args,"timeout_s":timeout,
            "utc":datetime.now(timezone.utc).isoformat(),
            "lock_environment":{k:env[k] for k in ("OTPU_LOCK_HELD","OTPU_LOCK_PID") if env and k in env}}))
        # Keep management output available for parsing even if the evidence disk fills.
        child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=env, start_new_session=True)
        deadline, terminate_at = time.monotonic()+timeout, None
        timed_out = False
        while True:
            try:
                output, _ = child.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if terminate_at is None and (now >= deadline or (cancellable and self.cancelled is not None)):
                    timed_out = now >= deadline
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    terminate_at = now
                elif terminate_at is not None and now-terminate_at >= 10:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        rc = child.returncode
        self.evidence(stem+" output evidence", lambda: (self.out/(stem+".log")).write_text(output))
        self.evidence(stem+" result evidence", lambda: write_json(self.out/(stem+".result.json"),
            {"returncode":rc,"timed_out":timed_out}))
        if timed_out or rc:
            raise RuntimeError(f"{role} failed (rc={rc}, timeout={timed_out}); see {stem}.log")
        return output

    def describe(self):
        return parse_image(self.command("describe", ["fpga-describe-local-image","-S","0","-H"]))

    def discover(self, allow_unconfigured=False):
        return parse_bdf(self.command("slots", ["fpga-describe-local-image-slots","-H"]),
                         allow_unconfigured=allow_unconfigured)

    def load(self, agfi, restoring=False):
        self.require_owner()
        self.command("restore" if restoring else "load", ["fpga-load-local-image","-S","0","-I",agfi,"-H"], timeout=300)

    def rescan(self):
        self.require_owner()
        self.command("rescan", ["fpga-describe-local-image","-S","0","-R","-H"])

    def wait_terminal(self):
        deadline = time.monotonic()+120
        while self.describe()["state"] in ("loading", "clearing", "busy"):
            if time.monotonic() >= deadline:
                raise RuntimeError("slot management operation still active; refusing a competing restore load")
            time.sleep(1)

    def no_foreign_handles(self, bdf):
        self.require_owner(bdf)
        offenders = foreign_handles(bdf)
        self.evidence("BAR ownership evidence", lambda: write_json(self.out/f"handles-{self.seq:02d}.json",
            {"bdf":bdf,"offenders":offenders}))
        if offenders:
            raise RuntimeError(f"foreign BAR owner for {bdf}; refusing image changes: {offenders}")

    def read_regs(self, bdf, mapping):
        self.require_owner(bdf)
        from opentpu.host.c2_f2 import _library, _mapping, _address
        root = Path("/sys/bus/pci/devices")/bdf
        if tuple(int((root/name).read_text(), 0) for name in ("vendor","device")) != (0x1D0F,0xF001):
            raise RuntimeError("PCI identity changed before BAR read")
        lib = _library()
        with _mapping(root/"resource0",4096) as bar:
            address = _address(bar)
            return {name:int(lib.rd32(address+offset)) for name,offset in mapping.items()}

    def baseline(self, bdf, *, wait=False):
        deadline = time.monotonic()+30
        while True:
            values = self.read_regs(bdf, dict(id=R.R_ID, version=R.R_VERSION, regmap=R.R_REGMAP,
                build_id=R.R_BUILD_ID, platform=R.R_CFG_PLATFORM, core_khz=R.R_CORE_KHZ, status=R.R_STATUS))
            if not wait:
                return values
            try:
                require_baseline(values)
                return values
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.1)

    def candidate(self, bdf, build_id):
        deadline = time.monotonic()+30
        while True:
            values = self.read_regs(bdf, dict(id=C.R_ID,abi=C.R_ABI,build_id=C.R_BUILD,
                core_khz=C.R_CORE_KHZ,status=C.R_STATUS,hbm_ready=C.R_HBM_READY))
            if (values["id"],values["abi"],values["build_id"],values["core_khz"]) != (C.ID,C.ABI,build_id,200000):
                raise RuntimeError(f"candidate MMIO identity does not match qualified build: {values}")
            if values["status"] & (C.BUSY | C.DRAINING | C.ERROR):
                raise RuntimeError(f"candidate not idle: {values}")
            if values["hbm_ready"] & 1:
                return values
            if time.monotonic() >= deadline:
                raise RuntimeError(f"candidate HBM ready timeout: {values}")
            self.check_cancel()
            time.sleep(0.1)

    def benchmark(self, bdf, lock, q, fixture, repeats, idx):
        self.require_owner(bdf)
        if sha(q["build_manifest"]) != q["inputs_sha256"]:
            raise RuntimeError("frozen benchmark build manifest changed")
        env = dict(os.environ, OTPU_LOCK_HELD="f2-"+bdf,
                   OTPU_LOCK_PID=str(lock.holder or os.getpid()), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        self.command("benchmark", [self.python,str(ROOT/"tools/c2_f2_bench.py"),"run",
            "--fixture",str(fixture),"--bdf",bdf,"--expected-build-id","0x"+q["build_id"],
            "--build-manifest",q["build_manifest"],"--out",str(self.out/f"benchmark-{idx:02d}"),
            "--repeats",str(repeats)], env=env, timeout=900, cancellable=True)


@contextmanager
def deferred_signals(ops):
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda signum, frame: setattr(ops,"cancelled",signum))
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--image-record", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True, help="repeat for multiple workloads under one load/restore")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="perform the reviewed physical session; default only writes a plan")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        raise ValueError("repeats must be 1..100")
    q = qualify(args.build_dir, args.image_record)
    fixtures, loaded = [], []
    for path in args.fixture:
        fixture = C.Fixture.load(path)
        fixture.reference()
        loaded.append(fixture)
        fixtures.append(dict(fixture_sha256=fixture.digest(), fixture_file_sha256=sha(path),
                             fixture=str(path.resolve()), rows=fixture.rows, kb=fixture.kb))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/"fixtures"
    frozen.mkdir()
    for idx, (entry, fixture) in enumerate(zip(fixtures, loaded)):
        path = frozen/f"{idx:02d}.npz"
        fixture.save(path)
        entry.update(frozen_path=str(path), frozen_file_sha256=sha(path))
    frozen_inputs = out/"build-inputs.json"
    frozen_inputs.write_bytes(Path(q["build_manifest"]).read_bytes())
    if sha(frozen_inputs) != q["inputs_sha256"]:
        raise RuntimeError("build manifest changed during session preparation")
    q["build_manifest"] = str(frozen_inputs)
    write_json(out/"plan.json", dict(qualification=q, fixtures=fixtures,repeats=args.repeats, execute=args.execute,
        stages=["verify dedicated host","canonical lock + original image/idle check","load candidate",
                "rescan + rediscover + identity check","inherited-lock benchmark","restore original + identity check"],
        limitation="Restores the original image, not its previous HBM/ACT/model contents; SIGKILL/host failure can prevent cleanup."))
    if not args.execute:
        print(f"Plan only: {out/'plan.json'}; no AWS, FPGA utility, lock or BAR access")
        return
    ops = LocalOps(out, args.python)
    try:
        with deferred_signals(ops):
            session(ops,q,[Path(entry["frozen_path"]) for entry in fixtures],repeats=args.repeats)
    except BaseException as error:
        ops.restoring = True
        ops.evidence("failure evidence", lambda: write_json(out/"error.json", dict(
            error=f"{type(error).__name__}: {error}",
            instruction="Inspect session.json restoration status and command logs before any further FPGA action.")))
        raise
    print(f"Benchmark complete; original image restored. Evidence: {out}")


if __name__ == "__main__":
    main()
