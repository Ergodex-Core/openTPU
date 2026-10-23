#!/usr/bin/env python3
"""Prepare Nanbeige/Gemma images and programs in host RAM; never open hardware.

--hold retains the buffers and accepts status/discard JSONL only. It has no
upload, validation, service-control, ownership, or production admission command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentpu import imgcache, progcache
from opentpu.llm import load_spec
from opentpu.llm.nanbeige import Spec, load_weights
from opentpu.llm.qwen3 import load_weights as gemma_weights
from opentpu.llm.staged import Case, prepare_gemma, prepare_nanbeige
from opentpu.target import config, load
from tools.nanbeige_board import PROFILE, validate_profile, _json_default

MODEL = dict(repo_id="Nanbeige/Nanbeige4.2-3B-GPTQ-Int8",
             revision="e8c6ccdb655bbcf410952686026d3fcbf4c07392")


def json_identity(model):
    return json.loads(json.dumps(imgcache.checkpoint(model)))


def prepare_pair(nanbeige_model, gemma_model, nanbeige_oracle, gemma_oracle,
                 *, profile=PROFILE, gemma_cap=768, progress=None):
    started = time.monotonic()
    def report(stage):
        if progress is not None:
            progress(dict(stage=stage, elapsed_seconds=time.monotonic() - started,
                          hardware_executed=False))
    profile = load(profile)
    contract = validate_profile(profile)
    if (profile.name, profile.variant) != ("f2-hbm", "p4rl8a64c150"):
        raise ValueError("staged acceptance targets the current wide-port image only")
    from opentpu.host.f2 import _require_host, _mmio
    _require_host()
    cfg = config(profile)
    nmodel, gmodel = Path(nanbeige_model), Path(gemma_model)
    noracle = json.loads(Path(nanbeige_oracle).read_text())
    goracle = json.loads(Path(gemma_oracle).read_text())
    if noracle.get("status") != "complete" or noracle.get("model") != MODEL or \
            json_identity(nmodel) != noracle.get("checkpoint_identity"):
        raise ValueError("Nanbeige source oracle/checkpoint identity does not match the pinned 4.2 3B")
    if goracle.get("image") != contract["image"] or \
            goracle.get("host_revision") != "18fb978033799f0f21258f0e6e085c0360322d96" or \
            hashlib.sha256((gmodel / "config.json").read_bytes()).hexdigest() != goracle.get("config_sha256") or \
            sum(p.stat().st_size for p in gmodel.glob("*.safetensors")) != goracle.get("weight_file_bytes"):
        raise ValueError("Gemma restore checkpoint/configuration differs from the saved qualified baseline")
    if gemma_cap not in (512, 768):
        raise ValueError("Gemma restore capacity must match a qualified 512/768 configuration")
    prefix = tuple(goracle["isa_oracle"]["token_ids"][:2])
    if not goracle["isa_oracle"].get("matches_fpga_prefix") or len(prefix) != 2:
        raise ValueError("Gemma restore oracle lacks a checked two-token prefix")
    raw = next(c for c in noracle["cases"] if c["name"] == "raw")
    if not raw["exact_tokens_match"]:
        raise ValueError("Nanbeige raw oracle did not pass")
    ncases = [Case(name, tuple(raw["result"]["prompt_tokens"]), tuple(raw["expected_tokens"]))
              for name in ("raw", "raw_reset_replay")]
    common = dict(source_hash=progcache.source(), profile=contract)
    report("preparing_native_helper_without_device")
    helper_started = time.monotonic()
    _mmio()  # compile/load the process-local memcpy helper; no transport or BAR mapping
    helper_seconds = time.monotonic() - helper_started
    report("preparing_nanbeige_ram")
    n = prepare_nanbeige(Spec.from_hf(nmodel), load_weights(nmodel), cfg, ncases,
        provenance=dict(common, checkpoint_identity=json_identity(nmodel), model=MODEL,
                        precision_source_equivalent=False))
    report("preparing_gemma_ram")
    # Exact saved prompt and the independently checked device/ISA prefix, not a
    # synthetic CPU inference request. Tokenization is immutable host preparation.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(gmodel, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.apply_chat_template([dict(role="user", content=goracle["prompt"])],
                                       tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    g = prepare_gemma(load_spec(gmodel), gemma_weights(gmodel), cfg,
        [Case("verify", tuple(int(t) for t in ids), prefix)], cap=gemma_cap,
        provenance=dict(common, checkpoint_identity=json_identity(gmodel),
                        baseline_config_sha256=goracle["config_sha256"]))
    report("both_models_sealed_in_ram")
    receipt = dict(status="prepared_host_ram", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        preparation_seconds=time.monotonic() - started, profile_contract=contract,
        native_helper_preparation_seconds=helper_seconds,
        source_hash=progcache.source(), models={"nanbeige": dict(n.manifest, identity=n.identity),
                                                "gemma": dict(g.manifest, identity=g.identity)},
        retained_image_bytes=sum(sum(m.manifest["image_bytes"]) for m in (n, g)),
        hardware_executed=False, model_forward_executed=False, physical_acceptance="pending",
        production_mutated=False, old_300_second_harness_allowed=False,
        max_total_outage_seconds=30, outage_budget_qualified=False,
        owner_adapter_implemented=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                       (1024 if sys.platform.startswith("linux") else 1))
    return (n, g), receipt


def emit(value):
    print(json.dumps(value, default=_json_default, allow_nan=False), flush=True)


def hold(models, receipt, stream):
    """Strong model references are retained until EOF/discard; no activation API."""
    assert len(models) == 2
    for line in stream:
        try:
            req = json.loads(line)
            if not isinstance(req, dict) or set(req) - {"action", "id"}:
                raise ValueError("only status/discard requests are supported")
            if req.get("action") == "status":
                emit(dict(event="status", id=req.get("id"), **receipt))
            elif req.get("action") == "discard":
                emit(dict(event="discarded", id=req.get("id"), hardware_executed=False))
                return
            else:
                raise ValueError("upload/inference/maintenance are unavailable; use status or discard")
        except (ValueError, TypeError) as exc:
            emit(dict(event="error", message=str(exc), hardware_executed=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nanbeige-model-dir", type=Path, required=True)
    p.add_argument("--gemma-model-dir", type=Path, required=True)
    p.add_argument("--nanbeige-oracle", type=Path, required=True)
    p.add_argument("--gemma-oracle", type=Path, required=True)
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--gemma-cap", type=int, choices=(512, 768), default=768)
    p.add_argument("--json", type=Path)
    p.add_argument("--hold", action="store_true")
    args = p.parse_args(argv)
    if args.json and args.json.exists():
        raise FileExistsError(args.json)
    models, receipt = prepare_pair(args.nanbeige_model_dir, args.gemma_model_dir,
        args.nanbeige_oracle, args.gemma_oracle, profile=args.profile, gemma_cap=args.gemma_cap)
    if args.json:
        with args.json.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2, default=_json_default) + "\n")
    emit(dict(event="ready", **receipt))
    if args.hold:
        hold(models, receipt, sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
