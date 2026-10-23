from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path


class GateError(ValueError):
    """An experiment is inadmissible; fail closed."""


class Queued(GateError):
    """Wait for an existing owner, approval, or current evidence."""


class Uncertain(GateError):
    """A started operation has no terminal receipt. Never retry implicitly."""


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise GateError("non-finite JSON number: " + value)
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate JSON key: " + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs)


def require(ok, reason):
    if not ok:
        raise GateError(reason)


def finite(value, name, positive=False):
    require(type(value) in (float, int) and math.isfinite(value), name + " must be finite")
    require(value > 0 if positive else value >= 0, name + " out of range")
    return value


def immutable_json(path, obj):
    """Create once, fsync, then read-only. Refuse symlinks and overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(canonical(obj) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o444)
    d = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(d)
    finally:
        os.close(d)


class RunLock:
    """Only serializes this harness. NOT a claim of FPGA ownership."""
    def __init__(self, root):
        self.path = Path(root) / ".writer.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise Queued("another harness writer holds this run")
        return self

    def __exit__(self, *args):
        os.close(self.fd)


class Journal:
    """Immutable hash-chained events. Resume derives state from these, never a loose pointer."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = []
        prior = "0" * 64
        for i, p in enumerate(sorted(self.root.glob("*.json"))):
            require(p.name == f"{i:06d}.json", "journal gap or unexpected entry")
            e = read_json(p)
            require(e["seq"] == i and e["previous"] == prior, "journal chain changed")
            saved = e.pop("sha256")
            require(saved == digest(e), "journal entry changed")
            e["sha256"] = saved
            self.events.append(e)
            prior = saved

    def append(self, kind, data):
        e = {"seq": len(self.events), "previous": self.events[-1]["sha256"] if self.events else "0"*64,
             "kind": kind, "data": data}
        e["sha256"] = digest(e)
        immutable_json(self.root / f"{len(self.events):06d}.json", e)
        self.events.append(e)
        return e
