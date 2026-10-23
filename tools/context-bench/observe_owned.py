"""Observer for a production owner's ALREADY-QUALIFIED BoardBackend engine.

This module has no command line entrypoint, transport opener, lock acquisition,
image loader, service controller or credential handling. The owner must enforce
the approved window and hard restoration bound outside this observer.
"""
import hashlib
import json
import re
import time


def capture_owned_turn(eng, tok, case, identity, ownership_window_id, stop):
    from opentpu.host.board import BoardBackend
    from opentpu.host.chat import Chat, sampler
    from opentpu.target import arch_hash

    if not isinstance(eng.backend, BoardBackend):
        raise ValueError("Physical BoardBackend required; ISA/CPU observations are not FPGA evidence")
    if not ownership_window_id or not callable(stop):
        raise ValueError("Owner window and cancellation/deadline callback are required")
    assert identity["image"] == "agfi-0001de1dd3df8e21e"
    assert identity["normal_locks"] == ["f2:0", "f2-0000:34:00.0"]
    assert identity["build_id"] == 0x417243F1 and identity["clock_mhz"] == 150
    assert identity["batch"] == eng.batch == 1 and identity["greedy"] is True
    assert identity["model"] == case["model"]
    layers, hidden = {"gemma-3-1b-it": (26, 1152), "gemma-3-270m-it": (18, 640)}[case["model"]]
    assert eng.spec.layers == layers and eng.spec.hidden == hidden
    assert eng.spec.window == 512 and eng.spec.n_q == 4 and eng.spec.n_kv == 1
    assert identity["weights_format"] == identity["head_format"] == "int8"
    assert eng.image.wformat == eng.image.head_format == "int8"
    assert arch_hash(eng.cfg) == 3714814630
    assert eng.cap >= case["prompt_tokens"] + 64
    ids = [int(x) for x in tok.apply_chat_template(case["messages"], tokenize=True,
                                                   add_generation_prompt=True)]
    assert ids == case["input_ids"] and len(ids) == case["prompt_tokens"]
    assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == \
        case["input_ids_sha256"]
    assert not stop(), "Owner window expired before inference"
    eng.reset()  # Full new prompt each repetition; weights/programs stay resident.
    pick = sampler(0, 1, 1.0, 0, softcap=eng.spec.softcap)
    def strict_pick(*args, **kwargs):
        raise AssertionError("Host token selection is forbidden")
    strict_pick.greedy, strict_pick.params, strict_pick.rng = True, pick.params, pick.rng
    events = []
    def trace(e):
        if e.get("source") == "chat" and e.get("event") == "token_observed":
            assert e["selection"] == "device"
        events.append(e)
    chat = Chat(eng, tok, False, strict_pick, 64, clock_mhz=150,
                fpga_only=True, phase_trace=trace)
    chat.history = case["messages"][:-1]
    submitted_ns = time.perf_counter_ns()
    text, turn = chat.ask(case["messages"][-1]["content"], stop=stop)
    selected = [e for e in events if e.get("source") == "chat" and
                e.get("event") == "token_observed"]
    visible = [e for e in selected if e["token_id"] not in eng.spec.eos]
    tokens = [{"index": i, "token_id": e["token_id"], "selection": e["selection"],
               "observed_ns": round(e["host_s"] * 1e9)} for i, e in enumerate(visible)]
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    # A cut 64-token response may end in a partial integer; retain it verbatim
    # and check only completed prefix numbers, without editing or scoring it as HF parity.
    complete_numbers = numbers[:-1] if text and text[-1].isdigit() else numbers
    return {"evidence_kind": "physical_fpga_measurement", "case_id": case["case_id"],
            "identity": identity, "ownership_window_id": ownership_window_id,
            "prompt_tokens": len(ids), "input_ids_sha256": case["input_ids_sha256"],
            "requested_output_tokens": 64, "capacity": eng.cap, "truncated": False,
            "prefix_kv_reused": 0, "residency": "warm_resident",
            "contract": {"resident": bool(eng.resident), "device_inputs": bool(eng.device_inputs),
                         "can_generate": bool(eng.can_generate), "chains": bool(eng.backend.chains),
                         "stateless_kv": not eng.stateful, "host_service": eng.backend.host is not None,
                         "host_embeddings": bool(getattr(eng.image, "embed_host", False)),
                         "host_ple": bool(getattr(eng.image, "ple_host", False))},
            "worker": {"submit_ns": submitted_ns, "tokens": tokens},
            "prefill_tokens": turn.prefill_tokens, "prefill_cycles": turn.prefill_cycles,
            "prefill_host_s": turn.prefill_s, "generate_cycles": turn.decode_cycles,
            "generate_steps": turn.decode_steps, "end": turn.end,
            "selected_ids_including_eos": [e["token_id"] for e in selected],
            "raw_text": text, "raw_phase_trace": events,
            "quality": {"task": case["quality_task"],
                        "completed_integer_prefix_correct": complete_numbers ==
                            list(range(1, len(complete_numbers) + 1)) and bool(complete_numbers),
                        "scope": "task-prefix sanity only; not native-checkpoint accuracy parity"}}
