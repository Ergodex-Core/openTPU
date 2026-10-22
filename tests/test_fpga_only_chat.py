"""Fail-closed dense device decode, including the first reply token, without a card."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.host.chat import Chat, Turn, _Picked, main, sampler
from opentpu.isasim import board_config
from opentpu.llm.qwen3 import Engine, PREFILL_ROWS, Spec, device_config
from test_autodecode import _Tok, _tiny
from test_gemma3 import tiny as gemma3_tiny


@pytest.fixture(scope="module", params=["qwen3", "gemma3"])
def model(request):
    if request.param == "gemma3":
        _, W, spec, _ = request.getfixturevalue("gemma3_tiny")
        spec = replace(spec, eos=())
        cfg = board_config(DRAM_BYTES=1 << 25)
    else:
        W, spec = _tiny("qwen3")
        spec = replace(spec, eos=())
        cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    return W, spec, cfg


def make_chat(model, strict, args=(0, 0, 1.0, 3, 1.0), n=5):
    W, spec, cfg = model
    return Chat(Engine(spec, W, cap=512, cfg=cfg, resident=True, snapshots=1),
                _Tok(), False, sampler(*args, softcap=getattr(spec, "softcap", None)), n,
                fpga_only=strict)


@pytest.mark.parametrize("sampled", [False, True])
def test_device_first_pick_matches_legacy_without_host_logits(model, sampled):
    args = (0.8, 5, 0.9, 3, 1.1) if sampled else (0, 0, 1.0, 3, 1.0)
    old, new = make_chat(model, False, args), make_chat(model, True, args)
    ref, _ = old.ask("hello")
    pick = new.pick

    def no_host_pick(*a, **kw):
        raise AssertionError("CPU first-token pick")

    no_host_pick.greedy, no_host_pick.params, no_host_pick.rng = \
        pick.greedy, pick.params, pick.rng
    new.pick = no_host_pick
    prefill = new.eng.prefill_chunks
    calls = []

    def no_logits(tokens, **kw):
        calls.append((list(tokens), kw))
        assert kw.get("logits") is False
        yield from prefill(tokens, **kw)

    new.eng.prefill_chunks = no_logits
    shown = []
    got, turn = new.ask("hello", lambda d, t: shown.append(d))
    assert got == ref == "".join(shown) and new._reply == old._reply
    assert calls and sum(len(ts) for ts, _ in calls) == 4
    assert turn.device_first and turn.prefill_tokens == turn.prefill_total == 4
    assert turn.gen_tokens == turn.decode_steps == 5
    assert new.fed == old.fed and new.eng.pos == old.eng.pos
    new.resume()
    old.resume()
    assert new._reply == old._reply and new.fed == old.fed
    # A following turn exercises conversation rendering, rollback and KV reuse.
    new.ask("again")
    old.ask("again")
    assert new._reply == old._reply and new.fed == old.fed


def test_device_first_eos_is_counted_but_not_displayed(model):
    ref = make_chat(model, False, n=1)
    ref.ask("h")
    eos = ref._reply[0]
    chat = make_chat(model, True)
    chat.eng.spec = replace(chat.eng.spec, eos=(eos,))
    text, turn = chat.ask("h")
    assert text == "" and turn.end == "eos" and turn.gen_tokens == 0
    assert turn.decode_steps == 1 and turn.prefill_tokens == 0
    assert chat.fed == [ord("h")] and not chat.can_resume


def eligible():
    return SimpleNamespace(
        spec=Spec(128, 1, 2, 1, 64, 256, 256),
        image=SimpleNamespace(offload=None, ple_host=False, embed_host=False,
                              prompt_host_rows=False, vision=None),
        backend=SimpleNamespace(args=True, generates=True, chains=True, host=None),
        resident=True, device_inputs=True, can_generate=True, _conv_lo=0,
        server=None, row_server=None)


@pytest.mark.parametrize("part,attr,value,reason", [
    ("backend", "args", False, "CAPS.args"),
    ("backend", "generates", False, "CAPS.gen"),
    ("backend", "chains", False, "chaining"),
    (None, "resident", False, "resident decode"),
    (None, "device_inputs", False, "input tables"),
    (None, "can_generate", False, "generate program"),
    ("image", "offload", object(), "experts resident"),
    ("image", "ple_host", True, "embedding/PLE"),
    ("image", "embed_host", True, "embedding/PLE"),
    ("image", "prompt_host_rows", True, "embedding/PLE"),
    (None, "row_server", object(), "embedding/PLE"),
    ("backend", "host", lambda: None, "host service"),
    (None, "_conv_lo", 3, "convolution"),
    (None, "stateful", True, "stateless-KV"),
    ("spec", "mtp", True, "MTP/vision"),
    ("image", "vision", object(), "MTP/vision"),
])
def test_mode_rejects_unqualified_paths_before_inference(part, attr, value, reason):
    eng = eligible()
    if part == "spec":
        eng.spec = SimpleNamespace(**{attr: value})
    else:
        setattr(getattr(eng, part) if part else eng, attr, value)
    with pytest.raises(ValueError, match=reason):
        Chat(eng, _Tok(), False, sampler(0, 0, 1.0, 3), 4, fpga_only=True)


def test_mode_rechecks_sampling_and_refuses_host_logits():
    chat = Chat(eligible(), _Tok(), False, sampler(0, 0, 1.0, 3), 4, fpga_only=True)
    chat.pick = sampler(0.8, 0, 0.9, 3)
    with pytest.raises(ValueError, match="sampling supported"):
        chat.ask("must not tokenize or run")
    assert chat.history == []
    chat.pick = sampler(0, 0, 1.0, 3)
    with pytest.raises(ValueError, match="refuses host logits"):
        chat._decode_card(np.zeros(10), [], Turn(), 0, lambda *a: None, lambda: False)
    with pytest.raises(ValueError, match="refuses host logits"):
        chat._decode_card(_Picked(3), [], Turn(), 0, lambda *a: None, lambda: False)


def test_fully_cached_prompt_reexecutes_its_final_token(model):
    chat = make_chat(model, True)
    ids = list(map(ord, "hello"))
    chat._template = lambda **kw: ids
    text, _ = chat.ask("first")
    reply = list(chat._reply)
    again, turn = chat.ask("same cached prompt")
    assert again == text and chat._reply == reply
    assert turn.device_first and chat.fed[:len(ids)] == ids


def test_sampled_context_cap_and_resume_keep_device_position(model):
    args = (0.8, 5, 0.9, 3, 1.1)
    chat = make_chat(model, True, args, n=1)
    # The allocated image stays at 512; this tests the logical request cap near a prompt.
    chat.eng.cap = 7
    chat.ask("hello")
    assert chat.eng.pos == len(chat.fed) == 5 and chat.can_resume
    chat.resume()
    assert chat.eng.pos == len(chat.fed) == 6 and chat.can_resume
    _, turn = chat.resume()
    assert chat.eng.pos == len(chat.fed) == 7 and turn.gen_tokens == 1
    _, turn = chat.resume()
    assert turn.end == "cap" and turn.gen_tokens == 0 and not chat.can_resume


def test_trace_separates_first_pick_text_and_exact_numerators(model):
    chat = make_chat(model, True, n=3)
    events = []
    chat.phase_trace = events.append
    chat.ask("hello")
    summary = events[-1]
    assert summary["event"] == "turn_complete"
    assert summary["non_eos_token_ids"] == chat._reply
    assert summary["displayed_tokens"] == summary["device_steps"] == 3
    assert summary["wall_decode_intervals"] == 2 and summary["device_first"]
    assert any(e["event"] == "prepare_begin" and e["source"] == "engine" for e in events)
    observed = [e for e in events if e["event"] == "token_observed"]
    assert [e["token_id"] for e in observed] == chat._reply
    assert all(e["selection"] == "device" and not e["eos"] for e in observed)
    before = [e for e in events if e["event"] == "text_callback_begin"]
    after = [e for e in events if e["event"] == "text_callback_done"]
    assert len(before) == len(after) == 3
    assert all(a["host_s"] <= b["host_s"] for a, b in zip(before, after))
    chat.resume()
    assert events[-1]["turn_index"] == 2
    assert len(events[-1]["non_eos_token_ids"]) == 3


def test_cli_writes_trace_after_a_cpu_isa_reply(model, tmp_path, monkeypatch):
    from opentpu.host import chat as CH, gen_warm as GW
    W, spec, cfg = model
    monkeypatch.setattr(CH, "model_dir", lambda name: tmp_path)
    monkeypatch.setattr(CH, "load_spec", lambda path: spec)
    monkeypatch.setattr(CH, "load_weights", lambda path: W)
    monkeypatch.setattr(CH, "load_tokenizer", lambda path: lambda: _Tok())
    monkeypatch.setattr(CH, "make_backend", lambda *a, **kw: ("isa", cfg))
    monkeypatch.setattr(GW, "start", lambda *a: None)
    path = tmp_path / "trace.jsonl"
    main(["--raw", "--greedy", "--cap", "512", "--max-new", "3", "--prompt", "hello",
          "--decode-trace", str(path)])
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[-1]["event"] == "turn_complete"
    assert events[-1]["displayed_tokens"] == 3
    assert [e["selection"] for e in events if e["event"] == "token_observed"][0] == \
        "host_prefill"


@pytest.mark.parametrize("args", [["--fpga-only"],
                                  ["--fpga-only", "--backend", "board", "--per-position"],
                                  ["--fpga-only", "--backend", "board", "--mtp"]])
def test_cli_rejects_invalid_mode_before_loading_checkpoint(args, capsys):
    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 2 and "--fpga-only needs" in capsys.readouterr().err
