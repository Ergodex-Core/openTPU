#!/usr/bin/env python3
"""C1 FPGA session. Execute requires independent qualified-artifact replay.

Lock/load/restoration and management mechanics copied from reviewed C2 session;
no C2 qualification or image-publication verifier is retained.
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
from opentpu import c1_f2 as C
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


def validate_baseline_descriptor(record):
    """Only the explicitly reviewed current F2 baseline is selectable.

    Its image source revision is f85ac03c; the frozen wrapper's BAR build ID is
    0x7e42a626. These are different identities and must not be substituted.
    """
    expected = dict(schema=1, kind="c2-f2-restoration-baseline", instance_id=INSTANCE,
        agfi_id="agfi-00c2d61ec51bd89aa", afi_id="afi-07e20bf5679cf37d7",
        owner="438374459156", shell_version="0x10212415",
        identity=dict(id=1330925653, version=8389640, regmap=5,
                      build_id=2118297126, platform=3, core_khz=200000))
    if (not isinstance(record, dict) or record != expected or
            type(record.get("schema")) is not int or
            not isinstance(record.get("identity"), dict) or
            any(type(v) is not int for v in record["identity"].values())):
        raise ValueError("Unreviewed restoration baseline descriptor")
    return expected


def require_baseline(values, descriptor=None):
    status = values["status"]
    ready = R.ST_WR_IDLE | R.ST_CALIB0 | R.ST_CALIB1
    if (values["id"] != R.ID_OTPU or values["platform"] != 3 or values["core_khz"] != 200000 or
            status & ready != ready or (status & R.ST_RUN and not status & R.ST_HALTED) or
            status & (R.ST_ERROR | R.ST_LOADING | R.ST_AXI_ERR | R.ST_ECC_DED)):
        raise RuntimeError(f"original image is not a healthy idle openTPU baseline: {values}")
    if descriptor is not None:
        expected = validate_baseline_descriptor(descriptor)["identity"]
        if ({k:v for k,v in values.items() if k != "status"} != expected or
                any(type(values.get(k)) is not int for k in expected)):
            raise RuntimeError("Original/restored BAR identity differs from pinned baseline")


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
    baseline = validate_baseline_descriptor(qualification["baseline"]) if "baseline" in qualification else None
    original = baseline["agfi_id"] if baseline is not None else ORIGINAL
    if qualification.get("agfi_id") == original:
        raise RuntimeError("Candidate must differ from the selected restoration baseline")
    if ops.instance_id() != INSTANCE:
        raise RuntimeError("refusing hardware access outside the dedicated F2 instance")
    report = dict(schema=1, kind="c1-f2-physical-session", instance_id=INSTANCE, slot=SLOT, qualification=qualification,
                  phase="preflight", restored=False, load_attempted=False)
    if baseline is not None: report["baseline"] = baseline
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
        original_image = ops.describe()
        require_image(original_image, original)
        before = ops.baseline(initial)
        require_baseline(before, baseline)
        report.update(original_image=original_image, original_identity=before, initial_bdf=initial)
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
            if {k:v for k,v in report["candidate_identity"].items() if k not in ("status","hbm_ready")} != qualification["identity"]:
                raise RuntimeError("C1 candidate identity differs from verified artifact")
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
                    ops.load(original, restoring=True)
                    ops.rescan()
                    restored_bdf = ops.discover()
                    own(restored_bdf)
                    report["restored_image"] = ops.describe()
                    require_image(report["restored_image"], original)
                    after = ops.baseline(restored_bdf, wait=True)
                    require_baseline(after, baseline)
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
            raise RuntimeError("C1 session failed; see session.json or stderr for restoration/evidence status") from failure
    return report


class LocalOps:
    def __init__(self, out, python):
        self.out, self.python = Path(out), python
        self.cancelled, self.seq = None, 0
        self.locks = {}
        self.restoring, self.logging_errors = False, []
        self.plan_sha = sha(self.out/"plan.json")
        self.qualification_sha = sha(self.out/"qualification.json")

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
        if report.get("phase") == "complete":
            verify_saved_inputs(self.out, self.plan_sha)
            if sha(self.out/"qualification.json") != self.qualification_sha:raise ValueError("Qualification changed")
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
        return artifact_verifier().verify_aws_image(json.loads(raw), q)

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
        from opentpu.host.c1_f2 import _library, _mapping, _address
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
                core_khz=C.R_CORE_KHZ,max_rows=C.R_MAX_ROWS,depth=C.R_DEPTH,credits=C.R_CREDITS,status=C.R_STATUS,hbm_ready=C.R_HBM_READY))
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
        verify_saved_inputs(self.out, self.plan_sha)
        env = {k:v for k,v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
        env.update(OTPU_LOCK_HELD="f2-"+bdf,
                   OTPU_LOCK_PID=str(lock.holder or os.getpid()), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        self.command("benchmark", [self.python,str(Path(__file__).with_name("c1_f2_alex_bench.py")),
            "--fixture",str(fixture),"--bdf",bdf,"--expected-build-id","0x"+q["build_id"],
            "--out",str(self.out/f"benchmark-{idx:02d}"),"--repeats",str(repeats)],
            env=env,timeout=900,cancellable=True)
        verify_saved_inputs(self.out, self.plan_sha)


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





LIFECYCLE_REFERENCE_SHA256 = '7766d8d476d82e1f3c6e7e3b98c849651cf3b6c17551998317109009e196b2a1'

# Fixed API. An installed adapter still requires the complete semantic verifier.
# This is not an allowlist of return flags: the reviewed implementation must
# replay portable C1 qualification, package and reconciled available-image bytes.
def artifact_verifier():
    path=ROOT/'boards/c1_f2_alex/physical_artifact.py'
    if not path.is_file():
        raise RuntimeError('C1 physical artifact verifier unavailable; no device access permitted')
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','boards/c1_f2_alex/physical_artifact.py'],check=True,capture_output=True)
    import importlib
    module=importlib.import_module('boards.c1_f2_alex.physical_artifact')
    if Path(module.__file__).resolve()!=path or module.API_VERSION!=1:
        raise RuntimeError('Wrong fixed C1 physical artifact interface')
    return module


def file_record(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file() or path.stat().st_nlink!=1:
        raise ValueError('Unsafe source/evidence file')
    if path.stat().st_size>64<<20:raise ValueError('Session input file exceeds64MiB')
    before=path.stat();digest=sha(path);after=path.stat()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stamp(before)!=stamp(after):raise ValueError('File changed while hashing')
    return dict(bytes=after.st_size,sha256=digest)


def source_snapshot(revision):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Full source revision required')
    def git(*args):return subprocess.check_output(['git','-c','safe.directory='+str(ROOT),'-c','core.hooksPath=/dev/null','-C',str(ROOT),*args],text=True).strip()
    if git('rev-parse','HEAD')!=revision or git('status','--porcelain','--untracked-files=normal'):
        raise ValueError('Source revision/clean checkout differs')
    names={'opentpu/c1_f2.py','opentpu/host/c1_f2.py','opentpu/host/runstate.py','opentpu/host/regs.py','tools/c1_f2_bench.py'}
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'boards/c1_f2_alex').iterdir() if p.suffix in ('.py','.tcl','.json'))
    names.update({'tools/c1_f2_alex_session.py','tools/c1_f2_alex_bench.py'})
    files={name:file_record(ROOT/name) for name in sorted(names)}
    for name in files:git('ls-files','--error-unmatch',name)
    helpers={p.name:file_record(p) for p in (Path(__file__),Path(__file__).with_name('c1_f2_alex_bench.py'),ROOT/'boards/c1_f2_alex/session_results.py')}
    return dict(revision=revision,root=str(ROOT),files=files,helpers=helpers)


def verify_saved_inputs(out, expected_sha=None):
    out=Path(out)
    if expected_sha is not None and file_record(out/"plan.json")["sha256"] != expected_sha:raise ValueError("Session plan changed")
    p=json.loads((out/'plan.json').read_text())
    if source_snapshot(p['source']['revision'])!=p['source']:raise ValueError('Runtime source changed')
    for name,want in p['frozen_files'].items():
        if file_record(out/name)!=want:raise ValueError('Frozen session input changed: '+name)
    return p


def prepare_plan(out,*,revision,artifact_request,artifact_request_sha256,baseline,baseline_sha256,fixtures,repeats=3):
    out=Path(out).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh unlinked output required')
    if type(repeats)is not int or not 1<=repeats<=100 or not 1<=len(fixtures)<=16:raise ValueError('Fixture/repeat bound')
    source=source_snapshot(revision)
    if file_record(artifact_request)['sha256']!=artifact_request_sha256 or file_record(baseline)['sha256']!=baseline_sha256:raise ValueError('Externally pinned input changed')
    descriptor=validate_baseline_descriptor(json.loads(Path(baseline).read_text()))
    frozen={};rows=[]
    for idx,path in enumerate(fixtures):
        info=file_record(path);fixture=C.Fixture.load(path);fixture.reference()
        if file_record(path)!=info:raise ValueError('Fixture changed during validation')
        rows.append(dict(relative_path=f'fixtures/{idx:02d}.npz',fixture_sha256=fixture.digest(),rows=fixture.rows,kb=fixture.kb,base_beat=fixture.base_beat,mode=0,original_file=info))
    out.mkdir(parents=False);(out/'fixtures').mkdir()
    for name,path in [('artifact-request.json',artifact_request),('baseline-record.json',baseline)]+[(row['relative_path'],path) for row,path in zip(rows,fixtures)]:
        raw=Path(path).read_bytes()
        with (out/name).open('xb') as f:f.write(raw)
        frozen[name]=file_record(out/name)
    if frozen['artifact-request.json']['sha256']!=artifact_request_sha256 or frozen['baseline-record.json']['sha256']!=baseline_sha256:raise ValueError('Pinned input changed during freezing')
    for row in rows:
        if frozen[row['relative_path']]!=row['original_file']:raise ValueError('Frozen fixture bytes differ')
    plan=dict(schema=1,kind='c1-f2-physical-session-plan',instance_id=INSTANCE,slot=SLOT,source=source,hardware_revision='0d654453e2c77ea9bc9d0e854c33d9166bb27375',baseline=descriptor,fixtures=rows,repeats=repeats,frozen_files=frozen,qualification_status='pending_independent_C1_artifact_verification',route_qualified=False,hardware_actions=False,stages=['independent C1 artifact verification','canonical lock and original healthy baseline','load candidate and rediscover','C1 checked runs under inherited lock','restore exact baseline and verify identity'],limitation='Restores baseline image, not previous HBM/model contents. SIGKILL/host failure can prevent cleanup.')
    write_json(out/'plan.json',plan);return plan


def execute_prepared(out,plan_sha256):
    out=Path(out).absolute()
    if file_record(out/'plan.json')['sha256']!=plan_sha256:raise ValueError('Untrusted plan')
    p=verify_saved_inputs(out,plan_sha256)
    # Fixed interface resolution and semantic verification precede even LocalOps.
    verifier=artifact_verifier()
    q=verifier.verify_session_artifact(out/'artifact-request.json',p['frozen_files']['artifact-request.json']['sha256'])
    required={'schema','kind','build_id','identity','agfi_id','afi_id','source_revision','evidence_manifest_sha256','package_sha256','image_record_sha256','qualification_sha256'}
    if not isinstance(q,dict) or not required<=set(q) or q['schema']!=1 or q['kind']!='c1-f2-verified-artifact-binding':raise ValueError('Incomplete C1 verifier result')
    if not re.fullmatch('[0-9a-f]{8}',q['build_id']) or q['source_revision']!=p['hardware_revision'] or q['build_id']!=q['source_revision'][:8]:raise ValueError('C1 hardware/runtime source differs')
    if not AGFI_RE.fullmatch(q['agfi_id']) or q['agfi_id']==p['baseline']['agfi_id'] or not re.fullmatch('afi-[0-9a-f]{17}',q['afi_id']):raise ValueError('C1 candidate image IDs')
    if any(not re.fullmatch('[0-9a-f]{64}',q[k]) for k in required if k.endswith('_sha256')):raise ValueError('Missing qualification bindings')
    ident=q['identity'];keys={'id','abi','build_id','core_khz','max_rows','depth','credits'}
    if not isinstance(ident,dict) or set(ident)!=keys or any(type(v)is not int for v in ident.values()) or ident!={**ident,'id':C.ID,'abi':C.ABI,'build_id':int(q['build_id'],16),'core_khz':200000,'max_rows':512,'depth':128} or not 1<=ident['credits']<=32:raise ValueError('Verified C1 identity invalid')
    q={**q,'baseline':p['baseline'],'baseline_record_sha256':p['frozen_files']['baseline-record.json']['sha256']}
    if (out/'qualification.json').exists() or (out/'session.json').exists():raise ValueError('Session attempt exists')
    write_json(out/'qualification.json',q)
    ops=LocalOps(out,sys.executable)
    try:
        with deferred_signals(ops):return session(ops,q,[out/row['relative_path'] for row in p['fixtures']],repeats=p['repeats'])
    except BaseException as error:
        ops.restoring=True
        ops.evidence('failure evidence',lambda:write_json(out/'error.json',dict(error=f'{type(error).__name__}: {error}',instruction='Inspect session.json restoration and evidence status before any further FPGA action.')))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan');a.add_argument('--out',type=Path,required=True);a.add_argument('--source-revision',required=True)
    for key in ('artifact-request','baseline'):a.add_argument('--'+key,type=Path,required=True);a.add_argument('--'+key+'-sha256',required=True)
    a.add_argument('--fixture',type=Path,action='append',required=True);a.add_argument('--repeats',type=int,default=3)
    a=sub.add_parser('execute');a.add_argument('--out',type=Path,required=True);a.add_argument('--plan-sha256',required=True)
    a=p.parse_args()
    if a.action=='plan':prepare_plan(a.out,revision=a.source_revision,artifact_request=a.artifact_request,artifact_request_sha256=a.artifact_request_sha256,baseline=a.baseline,baseline_sha256=a.baseline_sha256,fixtures=a.fixture,repeats=a.repeats);print('Offline plan saved; C1 artifact verification remains required.')
    else:execute_prepared(a.out,a.plan_sha256)
if __name__=='__main__':main()
