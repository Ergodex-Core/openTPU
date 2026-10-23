"""Session orchestration tests; no AWS calls, FPGA utility or MMIO is executed."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2_session", ROOT/"tools/c2_f2_session.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
AGFI, AFI = "agfi-0123456789abcdef0", "afi-0123456789abcdef0"


def qualified_fixture(tmp_path):
    root = tmp_path/"qualified"
    check = root/"cl_opentpu/build/checkpoints"
    reportdir = root/"cl_opentpu/build/reports"
    check.mkdir(parents=True)
    reportdir.mkdir()
    build = "4f750912"
    dcp = check/f"cl_opentpu.{build}.post_route.dcp"
    archive = check/"fixture.Developer_CL.tar"
    # Deliberately inert test bytes. The publisher separately checks actual tar members.
    dcp.write_bytes(b"UNIT TEST: not an FPGA checkpoint")
    archive.write_bytes(b"UNIT TEST: not an AFI archive")
    marker = check/f"cl_opentpu.{build}.route_validation.passed"
    marker.write_text("OTPU_STAGE_ROUTE_VALIDATION_PASSED\n")
    report = reportdir/f"cl_opentpu.{build}.route_validation.txt"
    report.write_text("\n".join(f"{k}=0" for k in (
        "setup_failed","hold_failed","timing_failed","bus_skew_failed","unrouted_nets",
        "drc_blockers","methodology_blockers"))+"\ngray_bus_constraints=47\ncdc_crossing_witnesses=6\n")
    inputs = dict(kind="c2-hbm-kernel",build_id=build,part="xcvu47p-fsvh2892-2-e",core_khz=200000,hbm_ports=8)
    S.write_json(root/"inputs.json",inputs)
    outputs = dict(flow="BuildAll",inputs_sha256=S.sha(root/"inputs.json"),
                   artifacts={str(p.relative_to(root)):dict(sha256=S.sha(p)) for p in (dcp,archive)})
    S.write_json(root/"outputs.json",outputs)
    record = dict(schema=1,region="us-east-1",build_id=build,inputs_sha256=S.sha(root/"inputs.json"),
        outputs_sha256=S.sha(root/"outputs.json"),dcp_sha256=S.sha(dcp),package_sha256=S.sha(archive),
        qualified_marker_sha256=S.sha(marker),validation_report_sha256=S.sha(report),state="available",
        image_id=AFI,global_image_id=AGFI,client_token="unit-test-publication-token")
    path = tmp_path/"deployment.json"
    S.write_json(path,record)
    return root,path


def test_qualification_binds_exact_files_and_available_image(tmp_path):
    root,record = qualified_fixture(tmp_path)
    q = S.qualify(root,record)
    assert q["agfi_id"] == AGFI and q["build_id"] == "4f750912"
    (root/"cl_opentpu/build/checkpoints/fixture.Developer_CL.tar").write_bytes(b"changed")
    with pytest.raises(ValueError,match="artifact"):
        S.qualify(root,record)


@pytest.mark.parametrize("change",["synth_only","failed_timing","old_39_inventory","missing_crossing","pending_image","bad_record_hash"])
def test_qualification_fail_closed(tmp_path,change):
    root,path = qualified_fixture(tmp_path)
    if change == "synth_only":
        out=json.loads((root/"outputs.json").read_text());out["flow"]="SynthCL";S.write_json(root/"outputs.json",out)
    elif change in ("failed_timing","old_39_inventory","missing_crossing"):
        report=root/"cl_opentpu/build/reports/cl_opentpu.4f750912.route_validation.txt"
        old,new={"failed_timing":("setup_failed=0","setup_failed=1"),
                 "old_39_inventory":("gray_bus_constraints=47","gray_bus_constraints=39"),
                 "missing_crossing":("cdc_crossing_witnesses=6","cdc_crossing_witnesses=5")}[change]
        text=report.read_text().replace(old,new)
        report.write_text(text)
        record=json.loads(path.read_text())
        record["validation_report_sha256"]=S.sha(report)
        S.write_json(path,record)
    else:
        r=json.loads(path.read_text());r["state" if change=="pending_image" else "package_sha256"]="pending";S.write_json(path,r)
    reason="missing reviewed HBM CDC" if change in ("old_39_inventory","missing_crossing") else None
    with pytest.raises(ValueError,match=reason):S.qualify(root,path)


def aws_response(q):
    return {"FpgaImages":[dict(OwnerId="438374459156",State={"Code":"available"},FpgaImageId=AFI,ShellVersion="0x10212415",
        FpgaImageGlobalId=AGFI,Tags=[dict(Key="c2:"+key,Value=q[key]) for key in
        ("build_id","inputs_sha256","package_sha256","client_token")])]}


@pytest.mark.parametrize("change",[None,"owner","state","agfi","tag","shell"])
def test_fresh_aws_binding_check(tmp_path,change):
    root,path=qualified_fixture(tmp_path);q=S.qualify(root,path);r=aws_response(q)
    if change=="owner":r["FpgaImages"][0]["OwnerId"]="other"
    if change=="state":r["FpgaImages"][0]["State"]["Code"]="pending"
    if change=="agfi":r["FpgaImages"][0]["FpgaImageGlobalId"]=S.ORIGINAL
    if change=="tag":r["FpgaImages"][0]["Tags"][0]["Value"]="wrong"
    if change=="shell":r["FpgaImages"][0]["ShellVersion"]="0xdeadbeef"
    if change:
        with pytest.raises(RuntimeError):S.verify_aws_image(r,q)
    else:assert S.verify_aws_image(r,q)["agfi_id"]==AGFI


def test_parse_aws_local_status_and_fresh_slot_address():
    text="AFI 0 agfi-0123456789abcdef0 loaded 0 ok 0 0x10212415\nAFIDEVICE 0 0x1d0f 0xf001 0000:34:00.0\n"
    S.require_image(S.parse_image(text),AGFI)
    assert S.parse_bdf(text)==S.INITIAL_BDF
    changed=text.replace("0xf001","0x9048")
    with pytest.raises(RuntimeError):S.parse_bdf(changed)
    assert S.parse_bdf(changed,allow_unconfigured=True)==S.INITIAL_BDF
    with pytest.raises(RuntimeError):S.parse_bdf(text+text)


class FakeLock:
    holder=None
    def __init__(self,ops,name):self.ops,self.name=ops,name
    def release(self):self.ops.events.append(("release",self.name));self.ops.held.remove(self.name)


class FakeOps:
    def __init__(self,fail=None,move=False):
        self.fail,self.move=fail,move
        self.image,self.events,self.held=S.ORIGINAL,[],set()
        self.report=None
    def instance_id(self):return "wrong" if self.fail=="instance" else S.INSTANCE
    def verify_image_binding(self,q):return {"test":"not an AWS call"}
    def record(self,r):self.report=copy.deepcopy(r)
    def acquire(self,name):
        if self.fail=="busy":raise RuntimeError("device lock busy")
        self.events.append(("lock",name));self.held.add(name);return FakeLock(self,name)
    def no_foreign_handles(self,bdf):
        assert "f2-"+bdf in self.held
        if self.fail=="foreign":raise RuntimeError("foreign user")
    def discover(self,allow_unconfigured=False):
        return "0000:35:00.0" if self.move and self.image==AGFI else S.INITIAL_BDF
    def describe(self):return dict(agfi=self.image,state="loaded",status_code=0,error="ok",error_code=0)
    def baseline(self,bdf,*,wait=False):
        return dict(id=S.R.ID_OTPU,version=1,regmap=5,build_id=123,platform=3,core_khz=200000,
                    status=0 if self.fail=="baseline_busy" else S.R.ST_HALTED|S.R.ST_WR_IDLE|S.R.ST_CALIB0|S.R.ST_CALIB1)
    def check_cancel(self):
        if self.fail=="cancel" and self.image==AGFI:raise InterruptedError("cancel")
    def load(self,image,restoring=False):
        assert "f2-"+S.INITIAL_BDF in self.held
        self.events.append(("load",image));self.image=image
        if self.fail=="load" and not restoring:raise RuntimeError("partial load failure")
        if self.fail=="restore" and restoring:raise RuntimeError("restore failed")
    def rescan(self):self.events.append(("rescan",self.image))
    def candidate(self,bdf,build):
        if self.fail=="candidate":raise RuntimeError("wrong candidate identity")
        return dict(id=S.C.ID,build_id=build,bdf=bdf)
    def wait_terminal(self):self.events.append(("wait_terminal",self.image))
    def benchmark(self,bdf,lock,q,fixture,repeats,idx):
        assert self.image==AGFI and "f2-"+bdf in self.held
        self.events.append(("benchmark",bdf,fixture,repeats,idx))
        if self.fail=="benchmark" and idx==1:raise RuntimeError("output mismatch")


def run_fake(ops):return S.session(ops,dict(agfi_id=AGFI,build_id="4f750912"),["first.npz","second.npz"],repeats=3)


@pytest.mark.parametrize("move",[False,True])
def test_one_load_multiple_fixtures_restore_and_fresh_bdf(move):
    ops=FakeOps(move=move);r=run_fake(ops)
    assert r["restored"] and r["phase"]=="complete" and r["completed_fixtures"]==["first.npz","second.npz"]
    assert [e for e in ops.events if e[0]=="load"]==[("load",AGFI),("load",S.ORIGINAL)]
    expected="0000:35:00.0" if move else S.INITIAL_BDF
    assert all(e[1]==expected for e in ops.events if e[0]=="benchmark")
    assert max(i for i,e in enumerate(ops.events) if e[0]=="load")<min(i for i,e in enumerate(ops.events) if e[0]=="release")
    assert not ops.held


@pytest.mark.parametrize("fail",["load","candidate","benchmark","cancel"])
def test_failure_after_load_attempt_restores_before_unlock(fail):
    ops=FakeOps(fail)
    with pytest.raises(RuntimeError):run_fake(ops)
    assert ops.report["restored"] and ops.report["phase"]=="failed"
    assert ops.image==S.ORIGINAL and not ops.held
    assert ops.events[-2:]==[("rescan",S.ORIGINAL),("release","f2-"+S.INITIAL_BDF)]


@pytest.mark.parametrize("fail",["instance","busy","foreign","baseline_busy"])
def test_failed_preflight_never_loads(fail):
    ops=FakeOps(fail)
    with pytest.raises(RuntimeError):run_fake(ops)
    assert not any(e[0]=="load" for e in ops.events)
    assert not ops.held


def test_restore_failure_is_preserved_and_not_reported_success():
    ops=FakeOps("restore")
    with pytest.raises(RuntimeError):run_fake(ops)
    assert ops.report["phase"]=="failed" and not ops.report["restored"]
    assert "restore_error" in ops.report and not ops.held


@pytest.mark.parametrize("when",["load_candidate","benchmark","restore_original","complete"])
def test_evidence_failure_never_prevents_required_restoration(when):
    class DiskFullOps(FakeOps):
        failed=False
        def record(self,r):
            self.report=copy.deepcopy(r)
            self.failed |= r["phase"]==when
            if self.failed:raise OSError("injected disk full")
    ops=DiskFullOps()
    with pytest.raises(RuntimeError):run_fake(ops)
    loads=[e[1] for e in ops.events if e[0]=="load"]
    assert loads==([] if when=="load_candidate" else [AGFI,S.ORIGINAL])
    assert ops.image==S.ORIGINAL and not ops.held
    assert ops.report["phase"]=="failed" and ops.report["logging_errors"]
    assert ops.report["restored"]==(when!="load_candidate")


def test_restoration_command_output_survives_disk_full(tmp_path,monkeypatch):
    ops=S.LocalOps(tmp_path,sys.executable);ops.restoring=True
    def disk_full(*args,**kwargs):raise OSError("injected disk full")
    monkeypatch.setattr(S,"write_json",disk_full)
    monkeypatch.setattr(Path,"write_text",disk_full)
    text=f"AFI 0 {S.ORIGINAL} loaded 0 ok 0 0x10212415\n"
    # This child only prints fixture text; no FPGA utility or device is invoked.
    output=ops.command("restore-status-test",[sys.executable,"-c",f"print({text!r},end='')"])
    S.require_image(S.parse_image(output),S.ORIGINAL)
    assert len(ops.logging_errors)==3
    ops.record({"phase":"failed","restored":True})
    assert len(ops.logging_errors)==4


def test_actual_child_inherits_canonical_lock_without_reacquiring(tmp_path,monkeypatch):
    from opentpu.host.runstate import DeviceLock
    monkeypatch.setenv("OTPU_RUN_DIR",str(tmp_path/"locks"))
    name="f2-"+S.INITIAL_BDF
    lock=DeviceLock(name,wait=0)
    try:
        env=dict(os.environ,OTPU_LOCK_HELD=name,OTPU_LOCK_PID=str(os.getpid()),PYTHONPATH=str(ROOT))
        code="from opentpu.host.runstate import DeviceLock; import os; l=DeviceLock(os.environ['OTPU_LOCK_HELD'],wait=0); assert l.fd is None and l.holder==os.getppid() and not l.lost(); l.release()"
        subprocess.run([sys.executable,"-c",code],env=env,check=True,timeout=10)
        # Child release must not release the parent's lock.
        outsider=dict(env);outsider.pop("OTPU_LOCK_HELD");outsider.pop("OTPU_LOCK_PID")
        code=f"from opentpu.host.runstate import DeviceLock,DeviceBusy\ntry: DeviceLock({name!r},wait=0)\nexcept DeviceBusy: pass\nelse: raise AssertionError('parent lock lost')"
        subprocess.run([sys.executable,"-c",code],env=outsider,check=True,timeout=10)
    finally:lock.release()


def test_cancel_terminates_only_own_benchmark_child_and_records_result(tmp_path):
    ops=S.LocalOps(tmp_path,sys.executable);ops.cancelled=15
    with pytest.raises(RuntimeError,match="benchmark failed"):
        ops.command("benchmark",[sys.executable,"-c","import time; time.sleep(30)"],cancellable=True)
    assert json.loads((tmp_path/"01-benchmark.result.json").read_text())["returncode"] != 0


def test_lost_inherited_owner_blocks_image_mutations(tmp_path):
    class Lost:
        fd=None
        holder=12345
        def lost(self):return True
    ops=S.LocalOps(tmp_path,sys.executable)
    ops.locks["f2-"+S.INITIAL_BDF]=Lost()
    with pytest.raises(RuntimeError,match="owner exited"):
        ops.load(AGFI)
    with pytest.raises(RuntimeError,match="owner exited"):
        ops.rescan()
    assert not list(tmp_path.iterdir())


def test_physical_lock_rejects_redirected_lock_directory(tmp_path,monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR",str(tmp_path/"noncanonical"))
    ops=S.LocalOps(tmp_path,sys.executable)
    with pytest.raises(RuntimeError,match="canonical /tmp/otpu"):
        ops.acquire("f2-"+S.INITIAL_BDF)
    assert not ops.locks and not list(tmp_path.iterdir())


def test_foreign_bar_detection_uses_resolved_sysfs_symlink_and_mapping(tmp_path):
    sysfs=tmp_path/"sys/bus/pci/devices";sysfs.mkdir(parents=True)
    actual=tmp_path/"sys/devices/pci0000:00/0000:34:00.0";actual.mkdir(parents=True)
    (sysfs/S.INITIAL_BDF).symlink_to(actual,target_is_directory=True)
    (actual/"resource4").touch()
    proc=tmp_path/"proc";fd=proc/"222/fd";fd.mkdir(parents=True)
    (fd/"7").symlink_to(actual/"resource4")
    (proc/"222/maps").write_text(f"1000-2000 rw-s 0 103:02 123 {actual}/resource4\n")
    got=S.foreign_handles(S.INITIAL_BDF,proc_root=proc,sysfs_root=sysfs,own_pid=111)
    assert {tuple(sorted(v.items())) for v in got}=={tuple(sorted(v.items())) for v in
        [dict(pid=222,fd="7"),dict(pid=222,mapping=True)]}
    assert S.foreign_handles(S.INITIAL_BDF,proc_root=proc,sysfs_root=sysfs,own_pid=222)==[]


def test_baseline_idle_accepts_reset_or_halted_run_but_never_active_run():
    values=FakeOps().baseline(S.INITIAL_BDF)
    calibration=S.R.ST_CALIB0|S.R.ST_CALIB1
    for st in (S.R.ST_WR_IDLE, S.R.ST_WR_IDLE|S.R.ST_HALTED, S.R.ST_WR_IDLE|S.R.ST_HALTED|S.R.ST_RUN):
        S.require_baseline(values|dict(status=st|calibration))
    for st in (S.R.ST_WR_IDLE|S.R.ST_RUN|calibration, S.R.ST_WR_IDLE|S.R.ST_CALIB0,
               S.R.ST_WR_IDLE|S.R.ST_CALIB1):
        with pytest.raises(RuntimeError):S.require_baseline(values|dict(status=st))


@pytest.mark.parametrize("never_ready",[False,True])
def test_restored_baseline_polls_calibration_with_a_deadline(tmp_path,monkeypatch,never_ready):
    ops=S.LocalOps(tmp_path,sys.executable)
    ready=FakeOps().baseline(S.INITIAL_BDF)
    calls=[];ticks=iter([0,1,31] if never_ready else [0,1])
    def snapshot(*args):
        calls.append(1)
        return ready if len(calls)>1 and not never_ready else ready|dict(status=S.R.ST_WR_IDLE)
    monkeypatch.setattr(ops,"read_regs",snapshot)
    monkeypatch.setattr(S.time,"monotonic",lambda:next(ticks))
    monkeypatch.setattr(S.time,"sleep",lambda _:None)
    if never_ready:
        with pytest.raises(RuntimeError,match="healthy idle"):ops.baseline(S.INITIAL_BDF,wait=True)
    else:
        assert ops.baseline(S.INITIAL_BDF,wait=True)==ready
    assert len(calls)==2


def test_default_plan_freezes_multiple_inputs_without_external_operations(tmp_path,monkeypatch):
    root,record=qualified_fixture(tmp_path)
    fixture=tmp_path/"source.npz";S.C.make_fixture(3,9).save(fixture)
    out=tmp_path/"plan"
    monkeypatch.setattr(sys,"argv",["c2_f2_session.py","--build-dir",str(root),"--image-record",str(record),
        "--fixture",str(fixture),"--fixture",str(fixture),"--out",str(out)])
    def forbidden(*args,**kwargs):raise AssertionError("plan attempted physical operations")
    monkeypatch.setattr(S,"LocalOps",forbidden)
    S.main()
    plan=json.loads((out/"plan.json").read_text())
    assert not plan["execute"] and len(plan["fixtures"])==2
    assert all(Path(f["frozen_path"]).is_file() for f in plan["fixtures"])
    assert Path(plan["qualification"]["build_manifest"])==out/"build-inputs.json"
