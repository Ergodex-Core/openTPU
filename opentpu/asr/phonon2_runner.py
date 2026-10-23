"""Prepared Phonon-2 device runner. No board construction or host model math.

The deployment owner supplies an already-open, exclusively leased transport.
This module uses only bytes, JSON, hashing, integer bounds and final UTF-8 I/O.
Preparing weights/programs is a separate offline operation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import uuid

MODEL = "FermionResearch/Phonon-2"
REVISION = "fb09063c9dc8e130d97b3a2191ed5aed6b71bb31"
PROFILE = "f2-hbm:p4rl8a64c150"
AGFI = "agfi-0001de1dd3df8e21e"
CFG_HASH = "dd6b92a6"
PROGRAM_AT = (4 << 30) - 163840
MAX_GRAPH = 180715008
SOURCE_BINDING_SHA256 = "63873d3b15c46206b448f07e87d4f1b08e2885222aa450abed412591e2e623a3"


class RunnerError(RuntimeError):
    def __init__(self, code, message, receipt=None):
        super().__init__(message)
        self.code, self.receipt = code, receipt


class Cancelled(TimeoutError):
    """TimeoutError makes Board.wait abort immediately, without a grace wait."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise RunnerError("ASR_ARTIFACT_INVALID", "invalid " + name)
    return value


class Artifact:
    """Pin the manifest externally; hash every blob before any device access.

    Rehash a blob again on consumption to detect a changed artifact directory.
    The owner must publish the directory read-only before taking a card lease.
    """
    def __init__(self, root, expected_sha256):
        self.root = Path(root).resolve()
        raw = (self.root / "manifest.json").read_bytes()
        if len(raw) > 4 << 20 or digest(raw) != expected_sha256:
            raise RunnerError("ASR_ARTIFACT_INVALID", "manifest hash differs")
        self.sha256, self.manifest = expected_sha256, json.loads(raw)
        m = self.manifest
        if digest(json.dumps(m["binding"], sort_keys=True, separators=(",", ":")).encode()) != SOURCE_BINDING_SHA256:
            raise RunnerError("ASR_ARTIFACT_INVALID", "pinned compute/runtime binding differs")
        if (m.get("schema"), m.get("model"), m.get("revision"), m.get("profile"),
            m.get("agfi"), m.get("cfg_hash")) != (1, MODEL, REVISION, PROFILE, AGFI, CFG_HASH):
            raise RunnerError("ASR_ARTIFACT_INVALID", "artifact identity differs")
        samples = integer(m["samples"], 320, 480000, "samples")
        self.samples = samples
        graph = integer(m["graph_bytes"], 128, MAX_GRAPH, "graph extent")
        if graph % 128 or m["program_at"] != PROGRAM_AT or graph >= PROGRAM_AT:
            raise RunnerError("ASR_ARTIFACT_INVALID", "invalid residency/program slot")
        if m["pcm_bytes"] != samples * 2 or m["raw_padded_bytes"] != ((samples + 63) // 64) * 128:
            raise RunnerError("ASR_ARTIFACT_INVALID", "invalid PCM envelope")
        self._range(m["input_at"], m["raw_padded_bytes"])
        for key, size in (("state", 64), ("text", m["max_steps"] * 32),
                          ("text_lengths", m["max_steps"] * 4)):
            self._range(m["output"][key], size)
        frames = (samples // 160 + 1 + 7) // 8
        valid = (samples // 160 + 7) // 8
        if (m["frames"], m["valid"], m["max_steps"]) != (frames, valid, 10 * frames - 1):
            raise RunnerError("ASR_ARTIFACT_INVALID", "invalid decoder bounds")
        actions = m["actions"]
        if not isinstance(actions, list) or len(actions) > 1000:
            raise RunnerError("ASR_ARTIFACT_INVALID", "invalid action count")
        if len([a for a in actions if a["kind"] == "pcm"]) != 1:
            raise RunnerError("ASR_ARTIFACT_INVALID", "require one raw PCM upload")
        if actions[0].get("name") != "request_reset" or actions[-1].get("name") != "tdt.2":
            raise RunnerError("ASR_ARTIFACT_INVALID", "reset/decode sequence differs")
        if sum(a["kind"] == "program" for a in actions) != 693:
            raise RunnerError("ASR_ARTIFACT_INVALID", "complete graph requires 693 dispatches")
        for a in actions:
            if a["kind"] == "write":
                self._range(a["at"], a["bytes"])
                self._spec(a["blob"], a["bytes"])
            elif a["kind"] == "program":
                words = integer(a["words"], 8, 32768, "instruction words")
                if words % 8:
                    raise RunnerError("ASR_ARTIFACT_INVALID", "partial instruction")
                self._spec(a["code"], words * 4)
                self._spec(a["waits"], words)
            elif a["kind"] != "pcm":
                raise RunnerError("ASR_ARTIFACT_INVALID", "unknown action")
        for sha in m["blobs"]:
            self.blob(sha)

    def _range(self, at, count):
        integer(at, 0, self.manifest["graph_bytes"], "offset")
        integer(count, 1, self.manifest["graph_bytes"], "size")
        if at + count > self.manifest["graph_bytes"]:
            raise RunnerError("ASR_ARTIFACT_INVALID", "range exceeds owned graph")

    def _spec(self, sha, size):
        if self.manifest["blobs"].get(sha) != size:
            raise RunnerError("ASR_ARTIFACT_INVALID", "blob extent differs")

    def blob(self, sha):
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise RunnerError("ASR_ARTIFACT_INVALID", "unsafe blob name")
        path = self.root / "blobs" / sha
        if path.is_symlink() or path.resolve().parent != (self.root / "blobs").resolve():
            raise RunnerError("ASR_ARTIFACT_INVALID", "unsafe blob path")
        data = path.read_bytes()
        if len(data) != self.manifest["blobs"][sha] or digest(data) != sha:
            raise RunnerError("ASR_ARTIFACT_INVALID", "artifact blob changed: " + sha)
        return data


@dataclass(frozen=True)
class Permit:
    """Owner attestation, NOT a request for or evidence of human approval.

    No permit has been granted by this change. Physical use requires separate
    parent approval. The first supported target is an idle spare card only.
    """
    approved: bool
    spare_card: bool
    lease_id: str
    agfi: str
    image_query_evidence: str
    health_evidence: str
    expires_monotonic: float


def check_identity(info):
    c, caps, c2 = info.get("cfg") or {}, info.get("caps") or {}, info.get("caps2") or {}
    expected = {"S": 1, "TMEM_WORDS": 65536, "IMEM_WORDS": 32768,
                "ACT_BLOCKS": 128, "DRAM_BYTES": 4 << 30, "SEQ_WC": True}
    if any(c.get(k) != v for k, v in expected.items()) or c.get("hash") != int(CFG_HASH, 16):
        raise RunnerError("ASR_PROFILE_MISMATCH", "live geometry/CFG_HASH differs")
    if (info.get("D"), info.get("MCOLS"), info.get("LANES"), info.get("act_rows"),
        info.get("core_khz")) != (128, 8, 8, 16, 150000):
        raise RunnerError("ASR_PROFILE_MISMATCH", "live profile differs")
    if not all(caps.get(k) is True for k in ("mx", "pair", "gen", "dstep", "stream", "t2")):
        raise RunnerError("ASR_PROFILE_MISMATCH", "required capability absent")
    if (c2.get("portb"), c2.get("rowlanes"), c2.get("aline")) != (4, True, 64):
        raise RunnerError("ASR_PROFILE_MISMATCH", "wide-port capability differs")
    # Error/loading/ECC sticky state must be resolved by the owner, never scrubbed here.
    if info.get("running") or not info.get("calibrated") or info.get("status", 0) & (2 | 4 | 16 | 256 | 512):
        raise RunnerError("ASR_DEVICE_UNHEALTHY", "device busy, uncalibrated or error/ECC state")


class Runner:
    def __init__(self, artifact, transport, *, clock=time.monotonic):
        self.artifact, self.transport, self.clock = artifact, transport, clock
        self.dirty = False

    def run(self, pcm, *, permit, cancelled=lambda: False, request_id=None):
        m, t = self.artifact.manifest, self.transport
        if not isinstance(pcm, bytes) or len(pcm) != m["pcm_bytes"]:
            raise RunnerError("ASR_INPUT_INVALID", "require exact-length mono 16kHz PCM_S16LE bytes")
        if self.dirty:
            raise RunnerError("ASR_RECOVERY_REQUIRED", "failed runner requires owner recovery and a new session")
        started = self.clock()
        receipt = {"schema": 1, "request_id": request_id or str(uuid.uuid4()),
                   "backend": t.backend, "model": MODEL, "revision": REVISION,
                   "profile": PROFILE, "agfi_expected": AGFI, "artifact_sha256": self.artifact.sha256,
                   "pcm_sha256": digest(pcm), "samples": m["samples"], "lease_id": permit.lease_id,
                   "ready": False, "physical_acceptance": False, "hardware_used": False,
                   "host_audio_or_model_compute": False, "dispatches": 0, "immutable_bytes": 0,
                   "program_wait_bytes": 0, "events": [], "outcome": "pending",
                   "timing_kind": "host_wall_observation; physical timing unmeasured for mock",
                   "graph_bytes": m["graph_bytes"], "program_at": PROGRAM_AT}
        touched = False

        def check():
            if cancelled():
                raise Cancelled("request cancelled")
            if self.clock() >= permit.expires_monotonic:
                raise TimeoutError("request/lease deadline expired")
            if not t.lease_valid(permit.lease_id):
                raise RunnerError("ASR_LEASE_LOST", "device ownership lost")

        try:
            if not permit.approved or not permit.spare_card or not permit.lease_id:
                raise RunnerError("ASR_PHYSICAL_ACCESS_GATED", "separate approved idle spare-card lease required")
            if permit.agfi != AGFI or not permit.image_query_evidence or not permit.health_evidence:
                raise RunnerError("ASR_PROFILE_MISMATCH", "owner image/health evidence missing or wrong")
            check()
            t.verify_host_sources(m["binding"])
            info = t.identity()
            check_identity(info)
            receipt["live_identity"] = info
            receipt["image_query_evidence"] = permit.image_query_evidence
            receipt["health_evidence"] = permit.health_evidence
            with t.session(check, permit.expires_monotonic):
                for action in m["actions"]:
                    check()
                    name, kind = action["name"], action["kind"]
                    tick = self.clock()
                    counters = None
                    if kind == "write":
                        raw = self.artifact.blob(action["blob"])
                        for offset in range(0, len(raw), 1 << 20):
                            check(); touched = True
                            t.write(action["at"] + offset, raw[offset:offset + (1 << 20)])
                        receipt["immutable_bytes"] += len(raw)
                    elif kind == "pcm":
                        touched = True
                        t.write(m["input_at"], pcm + bytes(m["raw_padded_bytes"] - len(pcm)))
                    else:
                        code, waits = self.artifact.blob(action["code"]), self.artifact.blob(action["waits"])
                        touched = True
                        t.load_program(PROGRAM_AT, code, waits)
                        check()
                        t.start()
                        stats = t.wait(max(0.000001, permit.expires_monotonic - self.clock()))
                        counters = stats
                        receipt["dispatches"] += 1
                        receipt["program_wait_bytes"] += ((len(code) + 127) // 128) * 128 + len(waits)
                        if stats.get("ecc_ded") or stats.get("status", 0) & (2 | 16 | 256 | 512):
                            raise RunnerError("ASR_DEVICE_ERROR", "device reported execution/ECC error")
                    check()
                    event = {"name": name, "kind": kind, "seconds": self.clock() - tick}
                    if counters is not None:
                        event["device_counters"] = counters
                    receipt["events"].append(event)
                text, state = self._output(check)
            receipt.update(outcome="completed", device_status=state[5], steps=state[3],
                           emitted=state[4], text_sha256=digest(text), text_bytes=len(text),
                           reset_completed=True, hardware_used=(t.backend == "physical"),
                           elapsed_seconds=self.clock() - started,
                           physical_execution_completed=(t.backend == "physical"))
            # A completed single request is evidence, never a readiness decision.
            return {"text": text.decode("utf-8", errors="strict"), "receipt": receipt}
        except BaseException as exc:
            code = "ASR_CANCELLED" if isinstance(exc, Cancelled) else "ASR_DEADLINE" if isinstance(exc, TimeoutError) else getattr(exc, "code", "ASR_DEVICE_ERROR")
            self.dirty = touched
            drained = None
            if touched:
                try:
                    # The transport checks ownership before stop; lost locks are never overridden.
                    drained = bool(t.stop_and_drain(permit.lease_id))
                except BaseException:
                    drained = False
            receipt.update(outcome="error", code=code, elapsed_seconds=self.clock() - started,
                           hardware_used=touched and t.backend == "physical", reset_completed=False,
                           quiescent=drained, recovery_required=touched)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RunnerError(code, str(exc), receipt) from exc

    def _output(self, check):
        m, t = self.artifact.manifest, self.transport
        check()
        state = struct.unpack("<16f", t.read(m["output"]["state"], 64))
        if any(not math.isfinite(x) or x != int(x) or x < 0 for x in state):
            raise RunnerError("ASR_OUTPUT_INVALID", "noninteger/nonfinite device control state")
        frame, pending, initialized, steps, emitted, status, seen, byte_count, zero_run = map(int, state[:9])
        if status == 3:
            raise RunnerError("ASR_DECODE_INCOMPLETE", "device decode watchdog exhausted; partial text discarded")
        if (status not in (1, 2) or initialized != 1 or not 1 <= steps <= m["max_steps"] or
            pending > 8192 or emitted > steps or seen > emitted or zero_run > steps or
            byte_count > steps * 32 or frame > m["valid"] + 3 or
            (status == 1 and frame < m["valid"]) or (status == 2 and pending != 3)):
            raise RunnerError("ASR_OUTPUT_INVALID", "device result bounds/termination differ")
        check()
        lengths = struct.unpack("<" + "f" * steps, t.read(m["output"]["text_lengths"], steps * 4))
        if any(not math.isfinite(x) or x != int(x) or not 0 <= x <= 32 for x in lengths) or sum(lengths) != byte_count:
            raise RunnerError("ASR_OUTPUT_INVALID", "invalid device UTF-8 segment lengths")
        check()
        raw = t.read(m["output"]["text"], steps * 32)
        if len(raw) != steps * 32:
            raise RunnerError("ASR_OUTPUT_INVALID", "short device text read")
        text = b"".join(raw[i * 32:i * 32 + int(n)] for i, n in enumerate(lengths))
        try:
            text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RunnerError("ASR_OUTPUT_INVALID", "invalid device UTF-8") from exc
        check()
        return text, list(map(int, state))


class BoardTransport:
    """Borrow an existing pinned Board; never open/close/scrub/calibrate/load image.

    A single-threaded poll proxy slices transport polling, not Board.wait calls.
    Board.wait therefore preserves HALTED/WR_IDLE/error handling. DMA calls are
    synchronous and cancellation is observed at their boundaries.
    """
    backend = "physical"

    def __init__(self, board, lease_valid, *, clock=time.monotonic):
        self.board, self.lease_valid, self.clock = board, lease_valid, clock

    def verify_host_sources(self, binding):
        import inspect
        import opentpu
        root = Path(opentpu.__file__).resolve().parents[1]
        if self.board.__class__.__module__ != "opentpu.host.board" or self.board.__class__.__name__ != "Board":
            raise RunnerError("ASR_RUNTIME_MISMATCH", "requires the pinned native Board")
        if Path(inspect.getfile(type(self.board))).resolve() != root / "opentpu/host/board.py":
            raise RunnerError("ASR_RUNTIME_MISMATCH", "Board source location differs")
        for name, sha in {**binding["runtime_files"], **binding["host_files"]}.items():
            if digest((root / name).read_bytes()) != sha:
                raise RunnerError("ASR_RUNTIME_MISMATCH", "runtime/host source differs: " + name)
        if getattr(self.board, "lock", None) is None or getattr(self.board, "in_run", True):
            raise RunnerError("ASR_LEASE_LOST", "locked idle Board required")
        if getattr(self.board.t, "batched", False):
            raise RunnerError("ASR_RUNTIME_MISMATCH", "simulation cannot produce physical evidence")
        transport = type(self.board.t)
        if transport.__module__ != "opentpu.host.f2" or transport.__name__ != "F2Transport":
            raise RunnerError("ASR_RUNTIME_MISMATCH", "requires the pinned native F2 transport")

    def identity(self):
        return self.board.info()

    @contextmanager
    def session(self, check, deadline):
        original, clock = self.board.t, self.clock

        class PollProxy:
            def __getattr__(self, name):
                return getattr(original, name)

            def poll(self, off, mask, val, timeout=600.0, expect=0.0):
                end = min(deadline, clock() + timeout)
                while True:
                    try:
                        check()
                        remaining = end - clock()
                        if remaining <= 0:
                            raise TimeoutError("device poll deadline expired")
                        return original.poll(off, mask, val, min(0.05, remaining), 0.0)
                    except TimeoutError as exc:
                        # Cancelled is also TimeoutError; do not swallow it.
                        if isinstance(exc, Cancelled) or clock() >= end:
                            self.disable()
                            raise
                    except BaseException:
                        self.disable()
                        raise

            def disable(self):
                # Board.wait's abort/stop must drain without reentering cancellation.
                self.poll = original.poll

        self.board.t = PollProxy()
        try:
            yield
        finally:
            self.board.t = original

    def write(self, at, raw):
        import numpy as np
        self.board.write(at, np.frombuffer(raw, dtype=np.uint8))

    def load_program(self, at, code, waits):
        import numpy as np
        self.board.load_program(at, np.frombuffer(code, dtype="<u4"),
                                entries=np.frombuffer(waits, dtype="<u8"))

    def start(self):
        self.board.start()

    def wait(self, timeout):
        return self.board.wait(timeout=timeout)

    def read(self, at, size):
        return self.board.read(at, size).tobytes()

    def stop_and_drain(self, lease_id):
        if not self.lease_valid(lease_id):
            return False
        self.board.stop(wait=0.0)  # False means interrupted, not failed to drain.
        info = self.board.info()
        return not info["running"] and bool(info["status"] & 8)
