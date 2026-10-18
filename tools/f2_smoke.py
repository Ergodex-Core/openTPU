#!/usr/bin/env python3
"""Bring up a loaded openTPU F2 image on its dedicated slot.

    python tools/f2_smoke.py --dev f2:0 --probe --expect-build 7e42a626
    python tools/f2_smoke.py --dev f2:0 --expect-build 7e42a626 --json f2-smoke.json
    python tools/f2_smoke.py --dev f2:0 --model qwen3 --tokens 2 --wformat int4

The full check destroys existing DRAM contents and programs. It checks calibration,
scrubs ECC once per image, then tests BAR4 and accelerator results against the ISA
simulator. Optional model checks compare BoardBackend with the same quantized ISA
model, not Hugging Face numerics. --probe only reads BAR0. Nothing here loads an AFI,
rescans PCIe, or contacts AWS. Run before loading application weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host import checks, regs as R
from opentpu.host.board import Board, config_summary, device_config
from opentpu.host.f2 import F2Transport
from opentpu.target import arch_hash, config, load


def identity(info: dict, expected_build: int | None = None):
    """Refuse a different target or build before any diagnostic memory writes."""
    target = load("f2")
    cfg = device_config(info)
    have = info.get("cfg") or {}
    if have.get("platform") != target.platform_id or have.get("hash") != arch_hash(config(target)):
        raise RuntimeError("the loaded architecture is not targets/f2.toml")
    if info.get("core_khz") != target.core_khz:
        raise RuntimeError(f"core clock identity {info.get('core_khz')} kHz != {target.core_khz}")
    if expected_build is not None and info.get("build_id") != expected_build:
        raise RuntimeError(f"build identity {info.get('build_id')!r} != {expected_build:08x}")
    return cfg


def health(t) -> dict:
    st = t.reg_read(R.R_STATUS)
    errors = st & (R.ST_ERROR | R.ST_AXI_ERR | R.ST_WAIT_TO | R.ST_ECC_DED)
    if errors:
        raise RuntimeError(f"STATUS {st:#010x} has error bits {errors:#x}; preserve diagnostics "
                           "and reload the image before retrying")
    return {"status": f"0x{st:08x}"}


def calibration(t, timeout: float) -> dict:
    mask = R.ST_CALIB0 | R.ST_CALIB1
    t.poll(R.R_STATUS, mask, mask, timeout)
    return health(t)


def scratch(t) -> dict:
    was = t.reg_read(R.R_SCRATCH)
    try:
        for val in (0, 0xFFFFFFFF, 0xA5A55A5A, 0x12345678):
            t.reg_write(R.R_SCRATCH, val)
            got = t.reg_read(R.R_SCRATCH)
            if got != val:
                raise RuntimeError(f"SCRATCH read {got:#010x}, expected {val:#010x}")
    finally:
        t.reg_write(R.R_SCRATCH, was)   # retain the once-per-image ECC initialization marker
    return {"message": "four aligned BAR0 word patterns"}


def checked(result) -> dict:
    ok, message, *stats = result
    if not ok:
        raise RuntimeError(message)
    return {"message": message, **({"stats": stats[0]} if stats else {})}


def _build_id(value: str) -> int:
    try:
        n = int(value, 16)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected an eight-digit hexadecimal build ID") from e
    if not 0 <= n <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("build ID must fit 32 bits")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="Full checks overwrite DRAM; --probe only reads BAR0.")
    ap.add_argument("--dev", default=os.environ.get("OTPU_DEVICE", "f2:0"))
    ap.add_argument("--expect-build", type=_build_id, help="BUILD_ID from this build's inputs.json")
    ap.add_argument("--probe", action="store_true", help="identity and status only; no writes")
    ap.add_argument("--calib-timeout", type=float, default=30.0)
    ap.add_argument("--force-scrub", action="store_true", help="zero 4 GiB even if DRAM_INIT is set")
    ap.add_argument("--json", type=Path, help="write stage results on success or failure")
    ap.add_argument("--model", action="append", default=[],
                    help="optional BoardBackend versus ISA check; repeat for multiple checkpoints")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--wformat", choices=("int8", "int4", "fp4", "nvfp4"), default="int4")
    a = ap.parse_args(argv)
    if not a.dev.startswith("f2:"):
        ap.error("--dev must select f2:<slot> or f2:<PCI BDF>")
    if a.calib_timeout <= 0 or a.tokens <= 0:
        ap.error("--calib-timeout and --tokens must be positive")
    if a.probe and (a.model or a.force_scrub):
        ap.error("--probe cannot be combined with --model or --force-scrub")

    report = {"device": a.dev, "probe": a.probe, "stages": [], "passed": False,
              "limitations": ["F2 ECC_DED is tied low; this check does not qualify DDR ECC "
                              "counters. Native DDR response errors set AXI_ERR. SDK metrics "
                              "can diagnose OCL/PCIS timeouts, but do not expose ECC counts."]}
    t = board = None
    current = "open"

    def stage(name, fn):
        nonlocal current
        current = name
        start = time.monotonic()
        print(f"[RUN ] {name}", flush=True)
        result = fn()
        report["stages"].append({"name": name, "passed": True,
                                 "seconds": round(time.monotonic() - start, 3), **result})
        print(f"[PASS] {name}: {result.get('message', '')}", flush=True)
        return result

    try:
        t = F2Transport(a.dev, dma=not a.probe)
        report["bdf"] = t.bdf
        # A monitor Board reads identity without quiescing another run or changing registers.
        current = "identity"
        probe = Board(t, lock=False, calibrate=False)
        info = probe.info()
        report["identity"] = info
        cfg = identity(info, a.expect_build)
        stage("identity", lambda: {"message": config_summary(info)})
        stage("status", lambda: health(t))
        if a.probe:
            report["passed"] = True
            return 0
        current = "idle"
        if info["running"]:
            raise RuntimeError("the accelerator is running; stop its owner before destructive bring-up")
        board = Board(t, calibrate=False)
        stage("calibration", lambda: calibration(t, a.calib_timeout))
        stage("scrub", lambda: {"message": "initialized 4 GiB" if board.scrub(a.force_scrub)
                                else "already initialized since image load"})
        stage("bar0", lambda: scratch(t))
        for ch in range(2):
            stage(f"address-ch{ch}", lambda ch=ch: checked(checks.address_lines(t, ch, t.channel_bytes)))
            stage(f"bar4-ch{ch}", lambda ch=ch: checked(checks.channel_patterns(t, ch, t.channel_bytes)))
            stage(f"partial-ch{ch}", lambda ch=ch: checked(checks.partial_writes(t, ch)))
        regions = [(7, 4099), (0x18003, 65539), (cfg.DRAM_BYTES - 4117, 4109)]
        stage("interleave", lambda: checked(checks.pattern_test(board, regions)))
        stage("kernel", lambda: checked(checks.run_demo(board, cfg)))
        stage("masked-stores", lambda: checked(checks.run_demo(board, cfg, checks.masked_program())))
        stage("host-ordering", lambda: checked(checks.waitw_host(board, rounds=4)))
        stage("status-after-kernels", lambda: health(t))
        for model in a.model:
            stage(f"model:{model}", lambda model=model: checked(
                checks.model_check(t, cfg, model, a.tokens, False, a.wformat)))
            stage(f"status:{model}", lambda: health(t))
        report["passed"] = True
        print("F2 SMOKE PASS", flush=True)
        return 0
    except Exception as e:
        report["stages"].append({"name": current, "passed": False,
                                 "error": f"{type(e).__name__}: {e}"})
        print(f"[FAIL] {current}: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        # F2.close runs Board cleanup while mapped, unmaps, and releases ownership last.
        try:
            if t is not None:
                t.close()
        finally:
            if board is not None:
                board.close()
            if a.json:
                a.json.write_text(json.dumps(report, indent=2, default=int) + "\n")


if __name__ == "__main__":
    sys.exit(main())
