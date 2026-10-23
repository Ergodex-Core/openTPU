"""Prepared acceptance harness. Run only in an explicitly coordinated idle window.

This file has NOT been run on hardware. It does not stop workers or load images.
Usage on the existing EC2 host: python this_file.py --window <coordinator-label>
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--window", required=True)
args = parser.parse_args()
root = Path("/home/ubuntu/nanbeige-wideport-20261023")
source = root / "source"
work = root / "hardware-001"
model = Path("/home/ubuntu/rachit/nanbeige-burst200/models/Nanbeige4.2-3B-GPTQ-Int8")
isa = json.loads((root / "checkpoint-002/isa.json").read_text())
assert isa["status"] == "complete" and isa["physical_open_attempts"] == 0
assert all(case["exact_tokens_match"] for case in isa["cases"])
for name, digest in isa["source_files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
assert not work.exists(), "Use a new reviewed receipt directory; never overwrite acceptance"
for key in ("OTPU_STOP_RUN", "OTPU_RUN_DIR", "OTPU_FORMATS"):
    assert not os.environ.get(key), f"Remove unsupported override {key}"
for key in list(os.environ):
    if key.startswith("OTPU_"):
        del os.environ[key]
os.environ.update(OTPU_IMG_CACHE=str(root / "img-cache"), OTPU_BUILD_JOBS="1",
                  OTPU_PROG_CACHE="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
os.nice(19)
os.sched_setaffinity(0, {21})
os.chdir(source)
sys.path.insert(0, str(source))
from opentpu import imgcache, progcache
from opentpu.llm.nanbeige import Spec, load_weights
from opentpu.llm.nanbeige_strict import StrictEngine
from opentpu.target import config, load
from tools import nanbeige_strict as cli

assert json.loads(json.dumps(imgcache.checkpoint(model))) == isa["checkpoint_identity"]
assert progcache.source() == isa["source_hash"]
profile = load(cli.PROFILE)
cfg = config(profile)
assert cli.validate_profile(profile) == isa["profile_contract"]
spec = Spec.from_hf(model)
work.mkdir()
receipt = dict(status="running", stage="packing", window=args.window, cases=[],
               started_utc=datetime.now(timezone.utc).isoformat(),
               implementation_commit="99bdc679c7d187b8d7990b0eed59650e2821dbcf",
               model=isa["model"], checkpoint_identity=isa["checkpoint_identity"],
               source_hash=isa["source_hash"], source_files=isa["source_files"],
               profile_contract=isa["profile_contract"], precision=cli.PRECISION,
               hardware_executed=False, locks_released=False)
session = None

def save():
    receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
    (work / "progress.json").write_text(json.dumps(receipt, indent=2, default=cli._json_default) + "\n")

def deadline(*_):
    raise TimeoutError("300-second bounded physical acceptance expired")

def factory(actual, images):
    global session
    receipt["stage"] = "acquiring_normal_locks"
    save()
    session = cli.StrictBoardSession("f2:0", profile)
    desc = subprocess.run(["fpga-describe-local-image", "-S", "0", "-H"],
                          check=True, capture_output=True, text=True, timeout=10)
    receipt["loaded_image_description"] = desc.stdout
    assert isa["profile_contract"]["image"] in desc.stdout and "loaded" in desc.stdout
    receipt.update(stage="uploading", board_info=session.info)
    save()
    return session.factory(actual, images)

signal.signal(signal.SIGALRM, deadline)
signal.signal(signal.SIGTERM, deadline)
signal.alarm(300)
save()
try:
    started = time.monotonic()
    engine = StrictEngine(spec, load_weights(model), cfg=cfg, backend=factory, cap=128)
    receipt.update(stage="forward", setup_seconds=time.monotonic() - started,
                   layout=engine.strict_plan, board_info=session.info)
    out = engine.image.lookup["gen"]["out"]
    read, write = engine.backend.read, engine.backend.write
    model_input_addresses = {engine.image.io[key] for key in ("x", "cos", "sin")}
    def tokens_only(sid, at, count):
        assert out <= at and at + count <= out + 4 * (engine.cap + 1)
        return read(sid, at, count)
    def no_host_model_inputs(sid, at, data):
        assert at not in model_input_addresses
        return write(sid, at, data)
    engine.backend.read = tokens_only
    engine.backend.write = no_host_model_inputs
    for case in isa["cases"]:
        receipt["active_case"] = case["name"]
        save()
        request = dict(prompt_tokens=case["result"]["prompt_tokens"], max_new_tokens=2)
        result = cli.run_request(engine, request, model, hardware=True)
        entry = dict(name=case["name"], result=result, expected_tokens=case["expected_tokens"],
                     exact_tokens_match=result["tokens"] == case["expected_tokens"])
        receipt["cases"].append(entry)
        receipt["hardware_executed"] |= result["hardware_executed"]
        save()
        assert entry["exact_tokens_match"], case["name"]
        assert result["hardware_executed"] and result["cycles_are_hardware_measured"]
    receipt.update(status="complete", stage="complete")
except BaseException as exc:
    receipt.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    if session is not None:
        session.close()
        receipt["locks_released"] = session.legacy is None and session.transport is None
    signal.alarm(0)
    save()
    if receipt["status"] == "complete":
        (work / "hardware.json").write_text(json.dumps(receipt, indent=2, default=cli._json_default) + "\n")
    print(json.dumps(receipt, default=cli._json_default), flush=True)
