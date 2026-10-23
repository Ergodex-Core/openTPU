"""VisualSim side of the hardware loop: the installed Architect, MoML model writing, static
validation, the single-seat lock and batch runner, and the model-staleness fingerprint.

Shared between Ergodex-Core/openTPU and Ergodex-Core/coral-gemma-f2 (tools/hwloop/vslib.py in
both; keep the copies identical, tools/hwloop/PROVENANCE.json records the SHA-256).

Nothing here starts a VisualSim process unless `run_batch(..., execute=True)` is called, and
that refuses while any Architect GUI or batch simulator is running: the user's GUI is never
terminated. Only Python's standard library is used.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

# ----------------------------------------------------------------------------- labels

MEASURED = "measured"                       # a receipt from hardware or an RTL simulation
CALIBRATED = "calibrated estimate"          # a model value fitted to receipts, with its error
UNVALIDATED = "unvalidated scenario"        # a model value with no receipt to check it against
WAITING = "waiting"                         # the evidence does not exist yet (seat, nightly)

# ----------------------------------------------------------------------------- the install

VS_AR_DEFAULT = Path.home() / "VisualSim" / "VS_AR"
BATCH_CLASS = "VisualSim.actor.gui.VisualSimBatchModeSimulator"
GUI_CLASS = "VisualSim.ModelBuilder.ModelBuilderApplication"
LICENSE_PORTS = (1099, 1900)        # the local FlexLM manager VisualSim.sh waits for (and 2222)


def vs_ar() -> Path:
    """The installed VisualSim Architect (VS_AR; the environment's VS_AR or VS overrides)."""
    for k in ("VS_AR", "VS"):
        if os.environ.get(k):
            return Path(os.environ[k])
    return VS_AR_DEFAULT


def install_info(root: Path | None = None) -> dict:
    """What is installed: version, Java, the library index and the examples this loop reuses."""
    root = root or vs_ar()
    info = {"vs_ar": str(root), "installed": root.is_dir(), "version": None, "java": None,
            "library_index": None, "examples": {}}
    if not root.is_dir():
        return info
    v = root / "Version_Number.txt"
    if v.is_file():
        info["version"] = v.read_text().strip()
    sh = root / "VisualSim.sh"
    if sh.is_file():
        m = re.search(r"JAVA_HOME=(\S+)", sh.read_text())
        if m:
            info["java"] = str(Path(m.group(1)) / "bin" / "java")
    idx = root / "doc" / "pre-built_models" / "html" / "Library_Model.html"
    info["library_index"] = str(idx) if idx.is_file() else None
    for key, rel in {
        "functional_hbm_demo": "Training_Material/Architecture/Mem_Controller/Functional_HBM_Demo.xml",
        "hbm_module": "VisualSim/actor/arch/Memory/HBM_Module.xml",
        "hw_dram": "VisualSim/actor/arch/Memory/HW_DRAM.xml",
        "memory_controller": "VisualSim/actor/arch/Memory/Memory_Controller.xml",
        "controller_dram_demo": "demo/SoC/Flow_Control_HW/Using_Mem_Controller_and_HWDRAM.xml",
        "lpddr5_example": "User_Library/Cadence/DRAM_LPDDR5X.xml",
        "tpu_example": "demo/SoC/Processor/TPU/TPU_6.xml",
        "tpu_soc_example": "demo/SoC/Processor/TPU/TPU_CPU_SoC_Demo_Model.xml",
        "dma_example": "Training_Material/Architecture/DMA/DMA.xml",
        "axi_example": "Training_Material/Architecture/Bus/AHB_AXI/AXI_Bus.xml",
        "amba_axi": "VisualSim/actor/arch/Buses/AMBA_AXI.xml",
    }.items():
        p = root / rel
        info["examples"][key] = {"path": str(p), "present": p.is_file(),
                                 "sha256": sha256_file(p) if p.is_file() else None}
    return info


def java_path(root: Path | None = None) -> Path | None:
    j = install_info(root)["java"]
    return Path(j) if j and Path(j).is_file() else None


# ----------------------------------------------------------------------------- hashing

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def fingerprint_tree(root: Path, globs: tuple[str, ...]) -> dict:
    """A fingerprint of the files `globs` select under `root` (relative path and content hash of
    each, and one SHA-256 over all of them): what a VisualSim model was generated from, so a
    model built from an older RTL cannot evaluate a newer one unnoticed (`check_stale`)."""
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file()})
    entries = [(str(p.relative_to(root)), sha256_file(p)) for p in files]
    h = hashlib.sha256()
    for rel, d in entries:
        h.update(f"{rel}\0{d}\n".encode())
    return {"root": str(root), "globs": list(globs), "files": len(entries),
            "sha256": h.hexdigest(), "entries": entries}


def check_stale(manifest: dict, root: Path, key: str = "rtl_fingerprint") -> dict:
    """Compare the manifest's fingerprint with the tree at `root` now. Returns {"stale": bool,
    "changed": [...], "added": [...], "removed": [...]}; a model whose manifest has no
    fingerprint is stale by definition (nothing says what it was built from)."""
    fp = manifest.get(key)
    if not fp:
        return {"stale": True, "reason": "no fingerprint in the manifest", "changed": [],
                "added": [], "removed": []}
    now = fingerprint_tree(root, tuple(fp["globs"]))
    old = dict(fp["entries"])
    new = dict(now["entries"])
    changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    stale = now["sha256"] != fp["sha256"]
    return {"stale": stale, "expected": fp["sha256"], "actual": now["sha256"],
            "changed": changed, "added": added, "removed": removed}


# ----------------------------------------------------------------------------- MoML writing

MOML_HEADER = ('<?xml version="1.0" standalone="no"?>\n'
               '<!DOCTYPE entity PUBLIC "-//Mirabilis Design//DTD MoML 1//EN"\n'
               '    "http://www.mirabilisdesign.com/xml/dtd/MoML_1.dtd">\n')
P_PARAM = "VisualSim.data.expr.Parameter"
P_STRING = "VisualSim.data.expr.StringParameter"
P_ATTR = "VisualSim.kernel.util.StringAttribute"
P_PULLDOWN = "VisualSim.actor.parameters.PullDownParameter"
P_FILE = "VisualSim.data.expr.FileParameter"
P_LOC = "VisualSim.kernel.util.Location"

C_COMPOSITE = "VisualSim.actor.TypedCompositeActor"
C_DE = "VisualSim.simulators.de.kernel.DEDirector"
C_ARCH_SETUP = "VisualSim.actor.arch.Architecture_Setup"
C_DS_SOURCE = "VisualSim.actor.lib.DS_Source"
C_MEM_INIT = "VisualSim.actor.lib.Memory_Init"
C_VM = "VisualSim.actor.lib.Virtual_Machine"
C_STR = "VisualSim.actor.lib.Smart_Timed_Resource"
C_DISPLAY = "VisualSim.actor.lib.gui.Display"
C_WRITER = "VisualSim.actor.lib.io.ExpressionWriter"
C_HBM = "VisualSim.actor.arch.Memory.HBM_Module"
C_MEMC = "VisualSim.actor.arch.Memory.Memory_Controller"
C_HWDRAM = "VisualSim.actor.arch.Memory.HW_DRAM"
C_PORT = "VisualSim.actor.TypedIOPort"
C_RELATION = "VisualSim.actor.TypedIORelation"

# the Architecture_Setup field mapping the installed examples carry (the native memory blocks
# read A_Address / A_Bytes / A_Command ... through it)
FIELD_MAP = ("/* First row contains Column Names. */\n"
             "External_Field_Name Internal_Field_Name ;\n"
             "A_Address A_Address ;\nA_Bytes A_Bytes ;\nA_Data A_Data ;\nA_IDX A_IDX ;\n"
             "A_Instruction A_Instruction ;\nA_Priority A_Priority ;\nA_Source A_Source ;\n"
             "A_Destination A_Destination ;\nA_Command A_Command ;\nA_Task_Flag A_Task_Flag ;\n"
             "A_Bytes_Sent A_Bytes_Sent ;\nA_Bytes_Remaining A_Bytes_Remaining ;\n"
             "A_Address_Min A_Address_Min ;\n")


@dataclass
class Entity:
    name: str
    cls: str
    props: list[tuple[str, str, str]] = field(default_factory=list)   # (name, class, value)
    ports: list[tuple[str, str]] = field(default_factory=list)        # (name, direction)
    location: tuple[int, int] = (100, 100)

    def param(self, name: str, value, cls: str = P_PARAM) -> "Entity":
        self.props.append((name, cls, _moml_value(value)))
        return self


def _moml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v}L" if abs(v) >= 2 ** 31 else str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(_moml_value(x) for x in v) + "}"
    return str(v)


def qstr(s: str) -> str:
    """A string-typed VisualSim parameter value (quoted inside the attribute)."""
    return '"' + s.replace('"', "'") + '"'


@dataclass
class Model:
    name: str
    params: list[tuple[str, str]] = field(default_factory=list)      # top-level (name, value)
    entities: list[Entity] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)       # (entity.port, entity.port)
    note: str = ""
    stop_time: str = "Sim_Time"

    def param(self, name: str, value) -> "Model":
        self.params.append((name, _moml_value(value)))
        return self

    def add(self, e: Entity) -> Entity:
        self.entities.append(e)
        return e

    def link(self, a: str, b: str) -> None:
        self.links.append((a, b))

    def to_xml(self) -> str:
        out = [MOML_HEADER, f'<entity name={quoteattr(self.name)} class="{C_COMPOSITE}">\n',
               f'  <property name="DigitalSimulator" class="{C_DE}">\n',
               f'    <property name="stopTime" class="{P_PARAM}" value={quoteattr(self.stop_time)} />\n',
               '    <property name="timeResolution" class="VisualSim.moml.SharedParameter" value="1e-12" />\n',
               f'    <property name="_location" class="{P_LOC}" value="[70,55]" />\n',
               '  </property>\n']
        for n, v in self.params:
            out.append(f'  <property name={quoteattr(n)} class="{P_PARAM}" value={quoteattr(v)} />\n')
        out.append('  <property name="_ModelBuilderSize" class="VisualSim.actor.gui.SizeAttribute" value="[1500,900]" />\n')
        if self.note:
            out.append('  <property name="Qualification" class="VisualSim.ModelBuilder.kernel.attributes.TextAttribute">\n'
                       f'    <property name="text" class="{P_ATTR}" value={quoteattr(self.note)} />\n'
                       f'    <property name="_location" class="{P_LOC}" value="[140,20]" />\n'
                       '  </property>\n')
        for e in self.entities:
            out.append(f'  <entity name={quoteattr(e.name)} class={quoteattr(e.cls)}>\n')
            out.append(f'    <property name="_location" class="{P_LOC}" value="[{e.location[0]},{e.location[1]}]" />\n')
            for n, c, v in e.props:
                out.append(f'    <property name={quoteattr(n)} class={quoteattr(c)} value={quoteattr(v)} />\n')
            for pn, d in e.ports:
                out.append(f'    <port name={quoteattr(pn)} class="{C_PORT}">\n      <property name="{d}" />\n    </port>\n')
            out.append('  </entity>\n')
        for i, (a, b) in enumerate(self.links, 1):
            out.append(f'  <relation name="wire_{i}" class="{C_RELATION}" />\n')
        for i, (a, b) in enumerate(self.links, 1):
            out.append(f'  <link port={quoteattr(a)} relation="wire_{i}" />\n')
            out.append(f'  <link port={quoteattr(b)} relation="wire_{i}" />\n')
        out.append('</entity>\n')
        return "".join(out)


def vm_block(name: str, block_name: str, code: str, ports_in: list[str], ports_out: list[str],
             location=(400, 320)) -> Entity:
    e = Entity(name, C_VM, location=location)
    e.param("Block_Name", qstr(block_name))
    e.props.append(("Expression_List", P_ATTR, code))
    e.ports = [(p, "input") for p in ports_in] + [(p, "output") for p in ports_out]
    return e


def timed_resource(name: str, block_name: str, time_field: str, depth: int, location=(700, 700)) -> Entity:
    e = Entity(name, C_STR, location=location)
    e.param("Block_Name", qstr(block_name))
    e.param("Time_Field", time_field, P_STRING)
    e.param("Max_Queue_Length", depth)
    e.props.append(("Queue_Type", P_ATTR, "FIFO"))
    return e


def display(name: str, title: str, location=(1100, 580)) -> Entity:
    e = Entity(name, C_DISPLAY, location=location)
    e.props.append(("title", P_ATTR, title))
    e.param("ViewText", True)
    return e


# ----------------------------------------------------------------------------- static validation

VM_FORBIDDEN = re.compile(r"writeFile|appendFile|exec\(|system\(|https?://")


def static_validate(xml_path: Path, root: Path | None = None) -> dict:
    """Checks a model without VisualSim: well-formed XML, every entity's class installed (a
    .class or a library .xml under VS_AR), ports declared by the library block, point-to-point
    relations, the VMs' labels and braces, no external actions or file dependencies. This is
    not the vendor's parser or VM compiler: a STATIC_PASS says the model may load, not that it
    runs."""
    root = root or vs_ar()
    raw = xml_path.read_bytes()
    errors: list[str] = []
    try:
        top = ET.fromstring(raw)
    except ET.ParseError as e:
        return {"status": "STATIC_FAIL", "errors": [f"xml: {e}"], "sha256": hashlib.sha256(raw).hexdigest()}
    if not raw.isascii():
        errors.append("not ASCII")
    actors = {a.get("name"): a for a in top.findall("entity")}
    relations = {r.get("name") for r in top.findall("relation")}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    installed = root.is_dir()
    for a in actors.values():
        cls = a.get("class", "")
        if installed:
            stem = root / cls.replace(".", "/")
            if not any(Path(str(stem) + ext).is_file() for ext in (".class", ".xml", ".XML")):
                errors.append("missing native class " + cls)
    for ln in top.findall("link"):
        q = ln.get("port", "")
        counts[ln.get("relation", "")] = counts.get(ln.get("relation", ""), 0) + 1
        if q in seen:
            errors.append("port has multiple relations " + q)
        seen.add(q)
        if ln.get("relation") not in relations:
            errors.append("missing relation " + str(ln.get("relation")))
        if "." not in q:
            errors.append("bad link port " + q)
            continue
        an, pn = q.split(".", 1)
        if an not in actors:
            errors.append("unknown actor " + an)
            continue
        a = actors[an]
        declared = {x.get("name") for x in a.findall("port")}
        if installed:
            stem = root / a.get("class", "").replace(".", "/")
            for ext in (".xml", ".XML"):
                f = Path(str(stem) + ext)
                if f.is_file():
                    try:
                        declared |= {x.get("name") for x in ET.parse(f).getroot().findall("port")}
                    except ET.ParseError:
                        pass
        if declared and pn not in declared:
            errors.append("unknown port " + q)
    if any(v != 2 for v in counts.values()):
        errors.append("non point-to-point relation")
    for a in actors.values():
        if a.get("class") == C_VM:
            code = (a.find("property[@name='Expression_List']").get("value", "")
                    if a.find("property[@name='Expression_List']") is not None else "")
            if not code.startswith("LABEL: BEGIN"):
                errors.append(f"VM {a.get('name')} does not begin with LABEL: BEGIN")
            if code.count("{") != code.count("}"):
                errors.append(f"VM {a.get('name')} brace mismatch")
            labels = set(re.findall(r"LABEL:\s*(\w+)", code))
            targets = set(re.findall(r"GTO\((\w+)\)", code))
            if targets - labels:
                errors.append(f"VM {a.get('name')} undefined GTO {sorted(targets - labels)}")
            if VM_FORBIDDEN.search(code):
                errors.append(f"VM {a.get('name')} external action")
    for node in top.iter("property"):
        if node.get("class") == P_FILE and node.get("value", "") not in ("", '"none"'):
            errors.append("file dependency " + node.get("value", ""))
    return {"status": "STATIC_PASS" if not errors else "STATIC_FAIL", "errors": errors,
            "entities": len(actors), "relations": len(relations),
            "classes_checked_against_install": installed,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "not_performed": ["VisualSim Java parser", "VisualSim VM compiler", "licensed simulation"]}


# ----------------------------------------------------------------------------- single seat

def active_visualsim() -> list[str]:
    """Every VisualSim Java process (the GUI, a batch simulator): the seat's holders."""
    p = subprocess.run(["/bin/ps", "-axo", "pid=,args="], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError("cannot list processes: " + p.stderr.strip())
    return [ln.strip() for ln in p.stdout.splitlines()
            if any(c in ln for c in (GUI_CLASS, BATCH_CLASS))]


def default_lock_path() -> Path:
    d = Path(os.environ.get("HWLOOP_LOCK_DIR", str(Path.home() / ".visualsim")))
    d.mkdir(parents=True, exist_ok=True)
    return d / "hwloop-seat.lock"


class SeatLock:
    """An advisory lock around every batch execution (one VisualSim instance at a time across
    every repository's tools on this machine). It does not cover the user's GUI: that is what
    `active_visualsim()` is for, and the two are checked together in `run_batch`."""

    def __init__(self, path: Path | None = None, timeout_s: float = 0.0):
        self.path = path or default_lock_path()
        self.timeout_s = timeout_s
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "a+")
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 >= self.timeout_s:
                    self._f.close()
                    self._f = None
                    raise RuntimeError(f"seat lock held: {self.path} ({self.path.read_text().strip()[:200]})")
                time.sleep(0.5)
        self._f.seek(0)
        self._f.truncate()
        self._f.write(f"pid {os.getpid()} {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n")
        self._f.flush()
        return self

    def __exit__(self, *exc):
        if self._f:
            self._f.seek(0)
            self._f.truncate()
            fcntl.flock(self._f, fcntl.LOCK_UN)
            self._f.close()
            self._f = None


def license_manager_up(ports=LICENSE_PORTS) -> bool:
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        except OSError:
            return False
    return True


def batch_snapshot(model_xml: Path, dest: Path) -> Path:
    """A copy of the model for batch mode: every Display becomes an ExpressionWriter into the
    run directory (a GUI display has no file in batch; the counters have to land somewhere)."""
    tree = ET.parse(model_xml)
    for a in tree.getroot().iter("entity"):
        if a.get("class") == C_DISPLAY:
            a.set("class", C_WRITER)
            for child in list(a):
                if child.tag == "property" and child.get("name") != "_location":
                    a.remove(child)
            ET.SubElement(a, "property", name="fileName", value=str(dest / (a.get("name") + ".txt")),
                          **{"class": P_FILE})
    snap = dest / "model.xml"
    tree.write(snap, encoding="utf-8", xml_declaration=True)
    return snap


def batch_command(java: Path, root: Path, snapshot: Path, results: Path) -> list[str]:
    cp = f"{root}:{root}/lib/*:{root}/com/amity/flexlm/*"
    return [str(java), "-Xmx1024m", "-XX:ActiveProcessorCount=2", "-Dvs.lic=default",
            "-Djava.security.manager=allow", f"-Djava.security.policy={root / 'bin' / 'policyAll'}",
            "-classpath", cp, BATCH_CLASS, "-resultpath", str(results), str(snapshot)]


def run_batch(model_xml: Path, run_dir: Path, execute: bool = False, timeout_s: int = 300,
              root: Path | None = None, manifest: dict | None = None,
              stale_root: Path | None = None) -> dict:
    """Prepare (default) or execute one batch simulation of `model_xml` into `run_dir`.

    Prepared: the snapshot and the exact command are written with a receipt whose status is
    PREPARED_NOT_EXECUTED; no license is used. Executed: refused unless no Architect / batch
    process exists and the seat lock is free (never kills anything); then the batch simulator
    runs under a wall-clock limit, its log is kept, the license release is checked, and the
    receipt says EXECUTED or FAILED. With `manifest` and `stale_root`, a stale model (its RTL
    fingerprint no longer matches the tree) is refused before anything else."""
    root = root or vs_ar()
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt: dict = {"status": "PREPARED_NOT_EXECUTED", "started_utc": stamp, "label": WAITING,
                     "source_model": str(model_xml), "source_sha256": sha256_file(model_xml),
                     "run_dir": str(run_dir), "timeout_s": timeout_s, "executed": False}
    if manifest is not None and stale_root is not None:
        st = check_stale(manifest, stale_root)
        receipt["staleness"] = {k: v for k, v in st.items() if k != "entries"}
        if st["stale"]:
            receipt["status"] = "REFUSED_STALE_MODEL"
            _write_json(run_dir / "receipt.json", receipt)
            return receipt
    snap = batch_snapshot(model_xml, run_dir)
    receipt["snapshot_sha256"] = sha256_file(snap)
    java = java_path(root)
    results = run_dir / "batch-results"
    cmd = batch_command(java or Path("java"), root, snap, results)
    receipt["command"] = cmd
    (run_dir / "command.sh").write_text("#!/bin/sh\n# exact batch command (prepared by hwloop)\n"
                                        + " ".join(_sh(c) for c in cmd) + "\n")
    active = active_visualsim()
    receipt["seat"] = {"active_processes": active, "license_manager_up": license_manager_up()}
    if not execute:
        receipt["note"] = ("not executed: pass --execute when no Architect GUI or batch run holds the seat"
                           if active else "not executed: --execute not given")
        _write_json(run_dir / "receipt.json", receipt)
        return receipt
    if active:
        receipt["status"] = "REFUSED_SEAT_BUSY"
        receipt["note"] = "a VisualSim process holds the seat; nothing was stopped"
        _write_json(run_dir / "receipt.json", receipt)
        return receipt
    if java is None or not root.is_dir():
        receipt["status"] = "REFUSED_NO_INSTALL"
        _write_json(run_dir / "receipt.json", receipt)
        return receipt
    try:
        with SeatLock():
            if active_visualsim():
                receipt["status"] = "REFUSED_SEAT_BUSY"
                _write_json(run_dir / "receipt.json", receipt)
                return receipt
            results.mkdir(exist_ok=True)
            t0 = time.monotonic()
            with open(run_dir / "simulation.log", "w") as log:
                proc = subprocess.Popen(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    receipt["exit_code"] = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    receipt["timeout"] = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        receipt["exit_code"] = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        receipt["exit_code"] = proc.wait(timeout=10)
            receipt["wall_s"] = time.monotonic() - t0
    except RuntimeError as e:
        receipt["status"] = "REFUSED_SEAT_LOCKED"
        receipt["note"] = str(e)
        _write_json(run_dir / "receipt.json", receipt)
        return receipt
    log = (run_dir / "simulation.log").read_text(errors="replace")
    receipt["license_contacted"] = "License Server has been contacted" in log
    receipt["license_freed"] = "License freed: true" in log
    receipt["error"] = any(x in log for x in ("Exception", "There is an error", "Error:"))
    receipt["completed"] = "successfully completed" in log
    ok = (receipt.get("exit_code") == 0 and not receipt.get("timeout") and not receipt["error"]
          and receipt["completed"] and receipt["license_freed"])
    receipt["executed"] = True
    receipt["status"] = "EXECUTED" if ok else "FAILED"
    receipt["label"] = UNVALIDATED if ok else WAITING
    receipt["outputs"] = sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("*.txt"))
    _write_json(run_dir / "receipt.json", receipt)
    return receipt


def _sh(s: str) -> str:
    return s if re.fullmatch(r"[\w./:*=@+-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def _write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def read_counters(run_dir: Path) -> dict:
    """The `Name = value` pairs the model's writers left in the run directory."""
    out: dict = {}
    for p in sorted(run_dir.glob("*.txt")):
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([+\-\d.eE]+)L?\b", p.read_text(errors="replace")):
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out
