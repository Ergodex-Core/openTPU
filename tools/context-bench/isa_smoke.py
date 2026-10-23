#!/usr/bin/env python3
"""Real cached Gemma models on the ISA simulator; NEVER performance or FPGA evidence."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("ISA checks must run on the builder")
    if a.output.exists():
        raise FileExistsError(a.output)
    for k in list(os.environ):
        if k.startswith("OTPU_"):
            del os.environ[k]
    os.environ.update(OTPU_BUILD_JOBS="1", OTPU_IMAGE_CACHE="0", OTPU_PROG_CACHE="0",
                     OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                     TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from opentpu.host import board, f2
    from opentpu.host.chat import Chat, sampler
    from opentpu.llm import load_spec, qwen3 as Q
    from opentpu.target import config
    opened = []
    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("PHYSICAL_ACCESS_FORBIDDEN")
    board.open_transport = forbidden
    f2.F2Transport.__init__ = forbidden
    def cpu_pick(*args, **kwargs):
        raise AssertionError("HOST_TOKEN_SELECTION_FORBIDDEN")
    result = {"evidence_kind": "ISA_correctness_NOT_FPGA_performance",
              "hardware_executed": False, "production_mutated": False, "models": {}}
    for name in ["gemma-3-1b-it", "gemma-3-270m-it"]:
        model = Path("/home/ubuntu/models") / name
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        spec = load_spec(model)
        eng = Q.Engine(spec, Q.load_weights(model), cap=4096, cfg=config("f2-hbm:p4rl8a64c150"),
                       resident=True, pipeline=False, snapshots=0, kv_full=True,
                       wformat="int8", head_format="int8", prog_cache=False)
        assert not eng.stateful and eng.device_inputs and eng.resident and eng.can_generate
        pick = sampler(0, 1, 1.0, 0, softcap=spec.softcap)
        cpu_pick.greedy, cpu_pick.params, cpu_pick.rng = True, pick.params, pick.rng
        runs = []
        for repeat in range(2):
            eng.reset()
            events = []
            chat = Chat(eng, tok, False, cpu_pick, 8, fpga_only=True, phase_trace=events.append)
            text, turn = chat.ask("Say hello.")
            selected = [e for e in events if e.get("source") == "chat" and
                        e.get("event") == "token_observed"]
            assert selected and all(e["selection"] == "device" for e in selected)
            runs.append({"text": text, "ids_including_eos": [e["token_id"] for e in selected],
                         "end": turn.end, "prefill_tokens": turn.prefill_tokens,
                         "note": "selection='device' exercises ISA device instructions; no physical FPGA"})
        assert runs[0] == runs[1]
        result["models"][name] = {"runs": runs, "exact_reset_replay": True,
                                  "capacity": 4096, "kv_full": True, "batch": 1,
                                  "weights_format": "int8", "head_format": "int8", "greedy": True}
        print(json.dumps({"model": name, "ISA_reset_replay": True}), flush=True)
        del eng
    assert opened == []
    result["physical_open_attempts"] = len(opened)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
