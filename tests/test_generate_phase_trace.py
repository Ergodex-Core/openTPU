"""Opt-in generate timing observations; no model weights, FPGA, or simulator needed."""
from types import SimpleNamespace

import numpy as np

from opentpu.host.board import Board, BoardBackend
from opentpu.host.fake import FakeTransport
from opentpu.llm import generate as G
from opentpu.llm.qwen3 import Engine


def test_engine_phase_trace_keeps_eos_and_the_unchanged_run(monkeypatch):
    monkeypatch.setattr(G, "state_words", lambda *args: np.zeros(8, np.float32))
    writes, calls, events, shown = [], [], [], []

    class Backend:
        generates = chains = True

        def write(self, s, addr, words):
            writes.append((s, addr, np.asarray(words).copy()))

        def run_generate(self, prog, out, n, on_token, stop, state, *, phase_trace=None):
            calls.append((prog, out, n, state))
            if phase_trace is not None:
                phase_trace({"source": "board", "event": "run_started", "host_s": 1.0})
            for t in (7, 99):
                if on_token is not None:
                    on_token(t)
            return {"cycles": 123, "instructions": [42]}, [7, 99]

    eng = object.__new__(Engine)
    eng.backend = Backend()
    eng.image = SimpleNamespace(lookup={"gen": {"state": 1024, "out": 2048}})
    eng.cfg = SimpleNamespace(S=1)
    eng.spec = SimpleNamespace(eos=(99,))
    eng.poss = [0]
    eng.resident, eng.cap, eng.pos, eng.block, eng._conv_lo = True, 16, 0, 4, 0
    eng.stats = []
    eng.gen_debug = False
    eng._generate_chain = lambda *args: None
    eng._generate_prog = lambda *args: "program"
    eng._write_host_rows = lambda tokens: None

    got = eng.generate_card(3, 3, on_token=shown.append, phase_trace=events.append)
    assert got == shown == [7, 99]
    assert calls == [("program", 2052, 3, 1024)]
    assert eng.pos == 2 and eng.stats == [{"cycles": 123, "instructions": [42]}]
    assert [e["event"] for e in events] == [
        "prepare_begin", "chain_ready", "program_ready", "sampling_inputs_ready",
        "host_rows_ready", "state_written", "output_markers_written", "run_dispatch",
        "run_started", "run_return"]
    assert all(e["run_index"] == 0 for e in events)
    assert events[-1]["token_ids"] == [7, 99] and events[-1]["cycles"] == 123
    assert events[-1]["generated_steps"] == 2
    assert writes[0][1] == 1024 and writes[1][1] == 2052

    events.clear()
    writes.clear()
    calls.clear()
    shown.clear()
    eng.pos = 0
    eng.stats.clear()
    assert eng.generate_card(3, 3, on_token=shown.append) == [7, 99]
    assert calls == [("program", 2052, 3, 1024)] and not events


def test_board_generate_trace_distinguishes_readback_and_callback():
    ids = np.array([7.0, 99.0], np.float32).view(np.uint32)
    seen, events = [], []

    class Board:
        def __init__(self):
            self.t = SimpleNamespace(batched=True, keeps_state=True)
            self._t_run = 1.25
            self.in_run = True

        def read(self, addr, nbytes):
            if addr == 4096:
                return np.r_[ids, np.uint32(G.OUT_MARK)]
            return np.array([G.OUT_MARK], np.uint32)

        def wait(self, expect=0.0, on_phase=None):
            self.in_run = False
            if on_phase is not None:
                on_phase({"event": "halt_observed", "host_s": 1.75,
                          "poll_exact": False, "status": 1})
                on_phase({"event": "write_idle_observed", "host_s": 1.8,
                          "status": 1, "batched": True})
            return {"cycles": 100, "instructions": [80], "status": 1}

    be = object.__new__(BoardBackend)
    be.board = Board()
    be._resident = None
    be._expects = {}
    be._key = 0
    be.info = {"core_khz": 100_000}
    be.host = None
    be.status = None
    be.start = lambda programs: None
    st, got = be.run_generate("program", 4096, 3, on_token=seen.append,
                              phase_trace=events.append)
    assert got == seen == [7, 99] and st["cycles"] == 100
    assert [e["event"] for e in events if e["event"] in
            ("halt_observed", "write_idle_observed")] == [
                "halt_observed", "write_idle_observed"]
    reads = [e for e in events if e["event"] == "output_readback"]
    assert reads[0]["token_ids"] == [7, 99]
    assert reads[0]["after_wait"] is True
    assert events[-1]["event"] == "run_complete" and events[-1]["token_ids"] == [7, 99]
    assert events[-1]["cycles"] == 100 and events[-1]["instructions"] == [80]
    assert events[-1]["generated_steps"] == 2 and events[-1]["device_s"] == 1e-6


def test_board_wait_reports_observed_halt_and_write_idle():
    board = Board(FakeTransport(devname=None, run_s=0.0, cycles=321),
                  check=False, lock=False)
    events = []
    board.start()
    stats = board.wait(on_phase=events.append)
    assert stats["cycles"] == 321
    assert [e["event"] for e in events] == ["halt_observed", "write_idle_observed"]
    assert events[0]["host_s"] <= events[1]["host_s"]
    assert events[0]["poll_exact"] is False


def test_host_fallback_trace_declares_cpu_picks_including_eos():
    from opentpu.host.chat import Chat
    from test_host import StubEngine, StubTok
    ids, events = iter([5, 0]), []
    chat = Chat(StubEngine(), StubTok(), False, lambda *args: next(ids), 4,
                phase_trace=events.append)
    text, turn = chat.ask("hi")
    assert text == "e" and turn.end == "eos"
    observed = [e for e in events if e["event"] == "token_observed"]
    assert [(e["token_id"], e["eos"], e["selection"]) for e in observed] == [
        (5, False, "host"), (0, True, "host")]
    assert any(e["event"] == "host_step_return" and e["cycles"] > 0 for e in events)
    assert len([e for e in events if e["event"] == "text_callback_done"]) == 1
    assert events[-1]["non_eos_token_ids"] == [5]
    assert events[-1]["wall_decode_intervals"] == 0
