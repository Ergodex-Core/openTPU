"""Measure ordinary and prompt-lookup Gemma 3 decode on the measured F2 card.

This is an actual-card, in-memory callback measurement.  Each mode has a fresh
Engine, but both modes run in one process; "cold" is an Engine-cold request, not
a process-cold or card-cold measurement.  No result is a terminal/SSE rate.
The script neither downloads a checkpoint nor substitutes an ISA/CPU backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT = "Why is the sky blue? Answer in one sentence."
CONFIG_SHA256 = "19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e"
EXPECTED_RAW = (
    818, 7217, 7412, 3730, 1547, 529, 496, 20284, 2760, 121707,
    19389, 236764, 1298, 26808, 563, 29892, 684, 506, 16383, 2634,
    13757, 528, 506, 11661, 236761, 106,
)
RUNS = ("cold", "warm1", "warm2")
BUILD_ID = 0x417243F1
CORE_KHZ = 150000
CAP = 512
MAX_NEW = 32


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _persist(path: Path, document: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _log(path: Path, event: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
        f.flush()


def _check_board(cfg, info: dict) -> None:
    expected_cfg = {
        "S": 1, "D": 128, "MCOLS": 8, "LANES": 8, "PORTB": 4,
        "ROWLANES": True, "ALINE": 64, "ACT_ROWS": 16,
        "TMEM_WORDS": 65536, "IMEM_WORDS": 32768,
        "ACT_BLOCKS": 128, "DRAM_BYTES": 4 << 30,
        "PAIR": True, "DSTEP": True, "STREAM": True, "T2": True,
        "SEQ_WC": True,
    }
    for key, value in expected_cfg.items():
        _require(getattr(cfg, key) == value,
                 f"board configuration {key}={getattr(cfg, key)!r}; expected {value!r}")
    _require(info.get("build_id") == BUILD_ID, "not the measured wideport150 bitstream")
    _require(info.get("core_khz") == CORE_KHZ, "not the measured 150 MHz clock")
    _require(info.get("act_rows") == 16, "unexpected physical ACT_ROWS")
    _require(info.get("cfg", {}).get("MXU_IMPL") == 2, "unexpected MXU implementation")
    _require(info.get("cfg", {}).get("TMEM_IMPL") == 3, "unexpected TMEM implementation")
    _require(info.get("cfg", {}).get("channels") == 8, "unexpected DRAM channel count")
    _require(info.get("caps2", {}).get("portb") == 4, "unexpected physical port B")
    _require(info.get("caps2", {}).get("rowlanes") is True, "missing physical row lanes")
    _require(info.get("caps2", {}).get("aline") == 64, "unexpected physical A line")


def _check_engine(eng, mode: str) -> None:
    from opentpu.host.board import BoardBackend

    _require(isinstance(eng.backend, BoardBackend), "backend is not the physical BoardBackend")
    _require(eng.resident and eng.device_inputs and eng.can_generate,
             "resident device-input generation contract failed")
    _require(bool(eng.backend.generates) and bool(eng.backend.chains)
             and callable(getattr(eng.backend, "run_generate", None)),
             "device HALT-chain streaming generation unavailable")
    _require(getattr(eng.backend, "host", None) is None
             and eng.server is None and eng.row_server is None,
             "host service or expert/row server is active")
    _require(not eng.image.ple_host and not getattr(eng.image, "embed_host", False)
             and (eng._host_rows is None or eng._host_rows([2]) == [])
             and eng._conv_lo == 0,
             "host PLE/embedding or pre-convolution fallback is active")
    _require(eng.image.rows == 8 and eng.cfg.S == 1 and eng.cap == CAP,
             "unexpected image rows, slices, or context cap")
    _require(eng.image.wformat == "int8" and eng.image.head_format == "int8",
             "unexpected weight or head format")
    _require("gemma_sd" in eng.image.lookup,
             "baseline and candidate must use the same prompt-lookup image layout")
    _check_board(eng.cfg, eng.backend.info)


def _one_run(eng, decoder, mode: str, label: str, prompt: list[int], tokenizer,
             partial: dict) -> dict:
    eng.reset()
    _require(eng.pos == 0, "Engine reset did not clear the prompt position")
    raw, arrivals = [], []
    partial.update(mode=mode, run=label, raw_token_ids=raw,
                   callback_arrivals_s=arrivals)

    def on_token(token) -> None:
        raw.append(int(token))
        arrivals.append(time.perf_counter())

    request_begin = time.perf_counter()
    partial["request_begin_s"] = request_begin
    stats_before = len(eng.stats)
    for _ in eng.prefill_chunks(prompt[:-1], logits=False):
        pass
    prefill_end = time.perf_counter()
    partial["prefill_end_s"] = prefill_end
    _require(eng.pos == len(prompt) - 1, "prefill did not leave exactly one unfed seed")
    prefill_stats = eng.stats[stats_before:]
    _require(prefill_stats and all(isinstance(s, dict) and isinstance(s.get("cycles"), int)
                                   for s in prefill_stats), "missing physical prefill cycles")
    decode_before = len(eng.stats)
    if mode == "baseline":
        output = eng.generate_card(prompt[-1], MAX_NEW, stop_ids=eng.spec.eos,
                                   on_token=on_token)
        sd = None
    else:
        output = decoder.generate_card(prompt[-1], MAX_NEW, stop_ids=eng.spec.eos,
                                       on_token=on_token, context=prompt)
        sd = decoder.last
    request_end = time.perf_counter()
    partial["request_end_s"] = request_end
    partial["decode_run_stats"] = eng.stats[-1] if len(eng.stats) > decode_before else None
    _require([int(x) for x in output] == raw, "callback and device output IDs differ")
    _require(tuple(raw) == EXPECTED_RAW, "raw IDs differ from the pinned 26-step receipt")
    _require(len(arrivals) == len(raw), "missing token-arrival timestamps")
    _require(len(eng.stats) == decode_before + 1,
             "decode was not exactly one physical device run")
    run = eng.stats[-1]
    _require(isinstance(run, dict) and isinstance(run.get("cycles"), int)
             and run["cycles"] > 0, "missing physical decode-cycle counter")
    eos = set(int(x) for x in eng.spec.eos)
    _require(raw[-1] in eos and all(t not in eos for t in raw[:-1]),
             "EOS placement differs from the pinned receipt")
    visible = raw[:-1]
    visible_times = arrivals[:-1]
    _require(len(visible) > 1 and visible_times[-1] > visible_times[0],
             "insufficient visible callback arrivals for rate measurement")
    clock_s = run["cycles"] / (CORE_KHZ * 1000)
    result = {
        "mode": mode, "run": label, "prompt_ids": prompt,
        "raw_token_ids": raw, "visible_token_ids": visible,
        "visible_text": tokenizer.decode(visible, skip_special_tokens=True),
        "request_begin_s": request_begin, "prefill_end_s": prefill_end,
        "request_end_s": request_end, "callback_arrivals_s": arrivals,
        "request_wall_s": request_end - request_begin,
        "prefill_wall_s": prefill_end - request_begin,
        "ttft_s": visible_times[0] - request_begin,
        "visible_decode_intervals": len(visible) - 1,
        "visible_decode_wall_s": visible_times[-1] - visible_times[0],
        "visible_decode_tps": (len(visible) - 1) /
                              (visible_times[-1] - visible_times[0]),
        "prefill_cycles": sum(s["cycles"] for s in prefill_stats),
        "decode_cycles": run["cycles"], "core_khz": CORE_KHZ,
        "raw_steps": len(raw), "raw_device_s": clock_s,
        "raw_device_steps_per_s": len(raw) / clock_s,
        "device_run_stats": run,
    }
    if sd is None:
        result["target_rows"] = len(raw)
    else:
        _require(sd.run is run, "speculative counters are not from the measured run")
        _require(sd.iterations > 0 and sd.verifies >= sd.accepted >= 0,
                 "invalid speculative iteration/accept/verify counters")
        _require(sd.iterations + sd.accepted == len(raw),
                 "speculative token accounting does not match device output")
        result.update(iterations=sd.iterations, accepted=sd.accepted,
                      verifies=sd.verifies, rejected=sd.verifies - sd.accepted,
                      target_rows=sd.iterations + sd.verifies,
                      speculative_compile_s=sd.compile_s)
    return result


def main() -> int:
    from transformers import AutoTokenizer
    from opentpu.host.chat import make_backend
    from opentpu.llm.gemma3 import Spec
    from opentpu.llm.gemma_specdec import GemmaPromptLookupDecoder
    from opentpu.llm.qwen3 import Engine, load_weights

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path,
                    default=Path("/home/ubuntu/opentpu/models/gemma-3-1b-it"))
    ap.add_argument("--dev", default="f2:0")
    ap.add_argument("--out", type=Path, required=True,
                    help="incrementally rewritten JSON receipt; .jsonl log is beside it")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_suffix(".jsonl")
    _require(not args.out.exists() and not log.exists(),
             "output or log already exists; choose a fresh --out path")
    config = args.model / "config.json"
    _require(config.is_file(), f"missing local checkpoint configuration: {config}")
    actual_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    _require(actual_hash == CONFIG_SHA256, "Gemma configuration hash differs from prior receipt")
    spec = Spec.from_hf(args.model)
    _require(spec.layers == 26 and spec.hidden == 1152 and spec.vocab == 262144,
             "checkpoint does not have the measured Gemma-3-1B architecture")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    prompt = [int(x) for x in (encoded["input_ids"] if hasattr(encoded, "keys") else encoded)]
    _require(len(prompt) == 20 and prompt[-1] == 107,
             f"chat-template prompt changed: length={len(prompt)}, last={prompt[-1]}")
    document = {
        "scope": "physical F2 board; in-memory callback; same process, fresh Engine per mode",
        "model": str(args.model), "config_sha256": actual_hash,
        "prompt": PROMPT, "prompt_ids": prompt, "expected_raw_ids": list(EXPECTED_RAW),
        "cap": CAP, "max_new": MAX_NEW, "weight_format": "int8",
        "head_format": "int8", "sampling": "greedy", "device": args.dev,
        "expected_build_id": f"0x{BUILD_ID:08x}", "expected_core_khz": CORE_KHZ,
        "runs": [], "complete": False,
    }
    _persist(args.out, document)
    try:
        for mode in ("baseline", "candidate"):
            _log(log, {"event": "engine_begin", "mode": mode,
                       "host_s": time.perf_counter()})
            backend_factory, cfg = make_backend("board", spec, CAP, args.dev,
                                                args.model.name, lookup=True,
                                                wformat="int8", head_format="int8")
            eng = None
            try:
                eng = Engine(spec, load_weights(args.model), cap=CAP, cfg=cfg,
                             backend=backend_factory, resident=True, prog_cache=False,
                             snapshots=0, pipeline=False, wformat="int8",
                             head_format="int8", gemma_prompt_lookup=True)
                _check_engine(eng, mode)
                decoder = GemmaPromptLookupDecoder(eng) if mode == "candidate" else None
                document.setdefault("board_info", eng.backend.info)
                layout = {"image_bytes": eng.backend.image_bytes,
                          "rows": eng.image.rows, "cap": eng.cap,
                          "wformat": eng.image.wformat,
                          "head_format": eng.image.head_format}
                if "image_layout" in document:
                    _require(document["image_layout"] == layout,
                             "baseline and candidate image layouts differ")
                else:
                    document["image_layout"] = layout
                for label in RUNS:
                    _log(log, {"event": "request_begin", "mode": mode,
                               "run": label, "host_s": time.perf_counter()})
                    partial = {}
                    try:
                        row = _one_run(eng, decoder, mode, label, prompt, tok, partial)
                    except BaseException:
                        document["failed_run"] = partial
                        _persist(args.out, document)
                        raise
                    document["runs"].append(row)
                    _persist(args.out, document)
                    concise = {k: row[k] for k in
                               ("mode", "run", "ttft_s", "visible_decode_tps",
                                "request_wall_s", "decode_cycles", "target_rows")}
                    _log(log, {"event": "request_complete", **concise})
                    print(json.dumps(concise), flush=True)
            finally:
                if eng is not None:
                    eng.backend.close()
        by_mode = {m: [r for r in document["runs"] if r["mode"] == m]
                   for m in ("baseline", "candidate")}
        _require(all(len(v) == len(RUNS) for v in by_mode.values()),
                 "incomplete paired measurement")
        for a, b in zip(by_mode["baseline"], by_mode["candidate"]):
            _require(a["run"] == b["run"] and
                     a["raw_token_ids"] == b["raw_token_ids"] and
                     a["visible_token_ids"] == b["visible_token_ids"],
                     "baseline and candidate diverged")
        document["complete"] = True
        _persist(args.out, document)
        _log(log, {"event": "complete", "host_s": time.perf_counter()})
        return 0
    except BaseException as exc:
        document["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _persist(args.out, document)
        _log(log, {"event": "failed", **document["error"]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
