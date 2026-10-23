"""Bounded trusted-evaluator adapters. Candidates can never supply commands or paths."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import GateError, Queued, Uncertain, canonical, digest, file_hash, immutable_json, read_json, require

HOSTS = {"card": ("opentpu-f2", "i-0c78de2253f66519d"),
         "builder": ("opentpu-m8a-build", "i-0ec8d33362b3100be")}
LOCK = "/tmp/otpu/f2-0000:34:00.0.lock"


def verify_window(window, identity, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    require(window.get("schema") == 1, "unsupported ownership window")
    if window.get("access") != "authorized":
        raise Queued("current authorized connectivity not established; no SSH/GET/SSM attempted")
    if window.get("portal_pending", True) or window.get("other_owner_active", True):
        raise Queued("portal or existing optimization owner has priority; queue, never preempt")
    require(window.get("instance_name") == HOSTS["card"][0] and
            window.get("instance_id") == HOSTS["card"][1], "not the authorized card instance")
    require(window.get("iam_profile") == "opentpu-f2-ssm", "instance role not verified")
    require(window.get("device_lock") == LOCK and window.get("exclusive") is True, "exclusive canonical lock unverified")
    require(window.get("loaded_image") == identity["agfi"], "loaded image mismatch; image swaps prohibited")
    for k in ("verified_at", "expires_at"):
        require(isinstance(window.get(k), str), "ownership window missing " + k)
    verified, expires = (dt.datetime.fromisoformat(window[k].replace("Z", "+00:00"))
                         for k in ("verified_at", "expires_at"))
    require(verified.tzinfo is not None and expires.tzinfo is not None, "window times need timezone")
    require(verified <= now < expires and (now-verified).total_seconds() <= 120,
            "ownership/identity observation stale; history is not availability")
    require(window.get("owner") and window.get("receipt_sha256"), "owner window lacks source receipt")


def run_process(argv, request, cwd, timeout_s, *, live=False):
    """No shell. Only trusted pinned evaluator code; scoped process group termination."""
    env = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    # Avoid inheriting provider tokens, AWS credentials, proxy config or OTPU override flags.
    if live:
        for key in ("OTPU_LOCK_HELD", "OTPU_LOCK_PID"):
            if key in os.environ: env[key] = os.environ[key]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                             cwd=cwd, env=env, start_new_session=True)
        try:
            p.communicate(canonical(request), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                if live:
                    immutable_json(Path(cwd)/"transport.json", {"argv":argv,"request_sha256":digest(request),
                                   "status":"timeout_cleanup_uncertain","pid":p.pid})
                    # A separate trusted lock holder must retain the card until an owner resolves it.
                    raise Uncertain("live evaluator timeout; cleanup unconfirmed; retain ownership, no retry")
                os.killpg(p.pid, signal.SIGKILL)
                p.communicate()
            if live:
                immutable_json(Path(cwd)/"transport.json", {"argv":argv,"request_sha256":digest(request),
                               "status":"timeout_hardware_reconciliation_required","returncode":p.returncode})
                raise Uncertain("live evaluator timed out; hardware idle/rollback must be reconciled")
            immutable_json(Path(cwd)/"transport.json", {"argv":argv,"request_sha256":digest(request),
                           "status":"fixture_timeout","returncode":p.returncode})
            raise GateError("fixture evaluator timed out")
        stdout.seek(0); stderr.seek(0)
        out, err = stdout.read((1 << 20)+1), stderr.read((1 << 20)+1)
    immutable_json(Path(cwd)/"transport.json", {"argv":argv,"request_sha256":digest(request),
                   "returncode":p.returncode,"stdout":out.decode(errors="replace"),
                   "stderr":err.decode(errors="replace"),"output_truncated":len(out)>1<<20 or len(err)>1<<20})
    require(len(out) <= 1 << 20 and len(err) <= 1 << 20, "evaluator output limit exceeded")
    if p.returncode == 75:
        if live:
            try:
                busy=json.loads(out)
            except ValueError:
                raise Uncertain("busy exit without a before-action receipt")
            if busy.get("action_started") is not False or busy.get("status") != "busy":
                raise Uncertain("busy exit did not establish that no hardware action started")
        raise Queued("trusted evaluator found card busy; no action allowed")
    if p.returncode:
        if live:
            raise Uncertain("live evaluator failed; inspect its receipt before retry/rollback")
        raise GateError("fixture evaluator failed: " + err.decode(errors="replace")[-400:])
    try:
        return json.loads(out)
    except ValueError as e:
        raise Uncertain("evaluator produced no complete JSON receipt") from e


class FixtureAdapter:
    def evaluate(self, request, scratch, timeout):
        return run_process([sys.executable, str(Path(__file__).with_name("fixture_worker.py"))],
                           request, scratch, timeout)


class LiveAdapter:
    """Owner-supplied, reviewed instrumentation bridge; deliberately not auto-configured.

    This is a JSON protocol, not a remote execution or image management channel.
    No SSH, SSM, GET, process inspection, image load or credentials are implemented here.
    Run ONLY within an existing canonical otpu-lock session on opentpu-f2. The
    trusted bridge must verify the inherited lock and hold it through cleanup.
    """
    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def preflight(self, window):
        verify_window(window, self.seed["identity"])
        require(platform.system() == "Linux", "live evaluation must execute on opentpu-f2, never the Mac")
        registry_path = Path(__file__).resolve().parents[1]/"configs/eligibility.json"
        require(file_hash(registry_path) == self.config.get("registry_sha256"), "eligibility registry unpinned or changed")
        registry = read_json(registry_path)
        entry = next((s for s in registry["seeds"] if s["id"] == self.seed["id"]), None)
        require(entry is not None and entry["physical_whole_model_pass"] is True,
                "seed not independently eligible in reviewed registry")
        require(entry["enabled"] is True and entry["fresh_protocol_qualified"] is True,
                "reviewed registry has not enabled this seed/protocol")
        require(entry["track"] == "gemma3-1b-int8", "initial live bridge only supports qualified Gemma INT8; other tracks separate")
        require(entry["agfi"] == self.seed["identity"]["agfi"], "registry image mismatch")
        for source in registry["sources"]:
            require(file_hash(registry_path.parents[1]/source["snapshot"]) == source["sha256"],
                    "source receipt changed")
        identity = self.seed["identity"]
        for key in ("checkpoint_sha256","tokenizer_sha256","runtime_tree_sha256","image_manifest_sha256"):
            require(re.fullmatch(r"[a-f0-9]{64}", str(identity.get(key,""))) is not None,
                    "missing full fingerprint: " + key)
        require(identity.get("weights") == "int8" and identity.get("head") == "int8", "precision changed")
        require(self.seed.get("enabled") is True, "seed is not enabled for this frozen protocol")
        require(self.seed.get("physical_whole_model_pass") is True, "no whole-model FPGA correctness receipt")
        require(self.seed.get("fresh_protocol_qualified") is True, "fixed-length held-out protocol not qualified")
        require(os.environ.get("OTPU_LOCK_HELD") == "f2-0000:34:00.0", "must run within existing canonical lock owner")
        cfg = self.config.get("trusted_bridge") or {}
        require(cfg.get("reviewed") is True, "trusted runtime timing bridge has not been reviewed")
        path = Path(cfg.get("path", ""))
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), "trusted bridge path invalid")
        require(file_hash(path) == cfg.get("sha256"), "trusted bridge changed")
        self.bridge = str(path)

    def evaluate(self, request, scratch, timeout):
        # Do not expose expected answers to the runtime bridge. Evaluation is external.
        import copy
        wire = copy.deepcopy(request)
        wire["workload"].pop("expected_token_ids", None)
        for c in wire["heldout"]["cases"]:
            c.pop("expected_token_ids", None)
        for c in wire["heldout"].get("rollback_cases",[]):
            c.pop("committed_state_sha256",None)
            c.pop("continuation_token_ids",None)
        # A bridge must be a standalone executable owned by the evaluator owner,
        # never candidate code; it verifies host identity and canonical ownership itself.
        wire["ownership_window"] = self.config["ownership_window"]
        # Bridge hash is rechecked before EVERY trial (not only at preflight).
        require(file_hash(self.bridge) == self.config["trusted_bridge"]["sha256"], "trusted bridge changed")
        result = run_process([self.bridge], wire, scratch, timeout, live=True)
        if result.get("cleanup") != "idle_and_runtime_restored":
            raise Uncertain("trusted bridge cleanup not verified; retain lock and reconcile")
        return result
