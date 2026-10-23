"""The board comparison harness: real tiny model execution and oracle/report semantics."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fpga_inference_check as probe
from tools.fpga_inference_check import collect, compare_logits, main


def test_comparison_keeps_bit_exactness_separate_from_argmax():
    a = np.array([1.0, 2.0], np.float32)
    b = a.copy()
    b[0] = np.nextafter(b[0], np.float32(2))
    got = compare_logits(a, b)
    assert got["match"] and got["unequal_words"] == 1 and got["max_abs"] > 0
    with pytest.raises(ValueError, match="nonfinite"):
        compare_logits(a, [np.nan, 2])


def test_incomplete_oracle_history_is_rejected():
    logits = np.array([[1, 2, 3], [3, 2, 1]], np.float32)
    with pytest.raises(ValueError, match="batch shape mismatch"):
        probe.compare_logit_rows(logits, logits[-1:])
    assert len(probe.compare_logit_rows(logits, logits)) == 2


def test_collect_stops_after_terminator_without_feeding_it():
    class Device:
        stats = []
        fed = []

        def step(self, token):
            self.fed.append(token)
            self.stats.append({"cycles": 5})
            return np.array([0, 1, 0], np.float32)

    dev = Device()
    generated, logits, timing = collect(dev, [2, 0], 16, {1})
    assert generated == [1] and dev.fed == [2, 0]
    assert logits.shape == (1, 3)
    assert timing["prompt_cycles"] == 10 and timing["decode_runs"] == 0


def _checkpoint(tmp_path, family="qwen3"):
    torch = pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(7)
    args = dict(hidden_size=256, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=128, intermediate_size=512,
                vocab_size=256, bos_token_id=2, eos_token_id=1, tie_word_embeddings=True)
    if family == "qwen3":
        config = T.Qwen3Config(**args)
        model = T.Qwen3ForCausalLM(config)
    else:
        config = T.Gemma3TextConfig(**args, sliding_window=256,
                                   layer_types=["sliding_attention"])
        model = T.Gemma3ForCausalLM(config)
    model.save_pretrained(tmp_path)
    tok = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "<bos>": 2, "Hello": 3}, unk_token="<unk>"))
    tok.save(str(tmp_path / "tokenizer.json"))


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setenv("OTPU_BUILD_JOBS", "1")
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    monkeypatch.setenv("OTPU_IMG_CACHE", "0")


@pytest.mark.parametrize("family", ["qwen3", "gemma3"])
@pytest.mark.parametrize("backend", ["isa", "board"])
def test_tiny_inference_compares_same_history_to_hf_and_isa(tmp_path, monkeypatch, family,
                                                          backend):
    _checkpoint(tmp_path, family)
    opened, closed = [], []
    if backend == "board":
        # Exercise the board session/configuration/close path without claiming physical
        # FPGA coverage: only this test replaces its transport with the ISA backend.
        from opentpu import target
        from opentpu.llm.qwen3 import IsaBackend

        class FakeSession:
            def __init__(self, dev, model):
                opened.append(dev)
                self.transport = SimpleNamespace(dev=dev)
                self.info = {"test_fixture": True}
                self.cfg = target.config("f2", DRAM_BYTES=16 << 20)

            def factory(self, cfg, images):
                b = IsaBackend(cfg, images, adopt=True)
                b.fills = True          # board programs fill their logit output sentinel
                return b

            def close(self):
                closed.append(True)

        monkeypatch.setattr(probe, "BoardSession", FakeSession)
    report, arrays = tmp_path / "result.json", tmp_path / "logits.npz"
    assert main(["--model", str(tmp_path), "--backend", backend, "--dev", "f2:0", "--compare-isa",
                 "--golden", "hf", "--prompt", "Hello", "--steps", "2", "--cap", "256",
                 "--json", str(report), "--save-logits", str(arrays)]) == 0
    result = json.loads(report.read_text())
    assert result["backend"] == backend
    if backend == "board":
        assert result["board_info"] == {"test_fixture": True}
        assert opened == ["f2:0"] and closed == [True]
    else:
        assert result["board_info"] is None
    assert result["comparison_complete"] and result["isa_bit_exact"]
    assert len(result["golden_comparison"]) == len(result["generated"])
    assert result["prompt"] == [3]
    saved = np.load(arrays)
    assert saved["device"].shape == saved["golden"].shape == saved["isa"].shape
    assert min(r["cosine"] for r in result["golden_comparison"]) > 0.99


@pytest.mark.parametrize("failure", ["build", "device", "isa", "hf"])
def test_failure_preserves_identity_outputs_and_closes_device(tmp_path, monkeypatch, failure):
    from opentpu import target
    from opentpu.llm import qwen3
    _checkpoint(tmp_path)
    closed = []

    class FakeSession:
        def __init__(self, dev, model):
            self.transport = SimpleNamespace(dev="f2:0000:34:00.0")
            self.info = {"build_id": 0x1234, "calib": [True, True], "temp_c": None}
            self.cfg = target.config("f2", DRAM_BYTES=16 << 20)

        def factory(self, cfg, images):
            return object()

        def close(self):
            closed.append(True)

    class FakeEngine:
        def __init__(self, spec, W, *, backend, **kwargs):
            self.isa = backend == "isa"
            if failure == "build" or (failure == "isa" and self.isa):
                raise RuntimeError("injected construction failure")
            self.image = SimpleNamespace(nbytes=4096)
            self.stats = []

        def step(self, token):
            if failure == "device" and self.stats:
                raise RuntimeError("injected device failure")
            self.stats.append({"cycles": 5})
            return np.array([1, 0, 0], np.float32)

    def failed_hf(path, tokens):
        raise RuntimeError("injected HF failure")

    monkeypatch.setattr(probe, "BoardSession", FakeSession)
    monkeypatch.setattr(qwen3, "Engine", FakeEngine)
    monkeypatch.setattr(probe, "hf_logits", failed_hf)
    report, arrays = tmp_path / "result.json", tmp_path / "logits.npz"
    with pytest.raises(RuntimeError, match="injected"):
        main(["--model", str(tmp_path), "--dev", "f2:0", "--compare-isa", "--prompt", "Hello",
              "--steps", "2", "--json", str(report), "--save-logits", str(arrays)])
    result = json.loads(report.read_text())
    assert result["device"] == "f2:0000:34:00.0" and result["board_info"]["build_id"] == 0x1234
    assert result["phase"] == "failed" and not result["comparison_complete"]
    assert result["failed_phase"] == {"build": "building_uploading", "device": "device_inference",
                                      "isa": "isa_comparison", "hf": "golden_comparison"}[failure]
    assert result["error"]["type"] == "RuntimeError" and closed == [True]
    saved = np.load(arrays)
    if failure == "build":
        assert "device" not in saved and result["generated"] == []
    else:
        count = 1 if failure == "device" else 2
        assert saved["device"].shape == (count, 3) and len(result["generated"]) == count
        assert "golden" not in saved
        assert ("isa" in saved) == (failure == "hf")


def test_board_session_closes_transport_on_configuration_failure(monkeypatch):
    from opentpu.host import board
    closed = []
    tr = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(board, "open_transport", lambda dev: tr)

    def broken_board(*args, **kwargs):
        raise RuntimeError("wrong device")

    monkeypatch.setattr(board, "Board", broken_board)
    with pytest.raises(RuntimeError, match="wrong device"):
        probe.BoardSession("f2:0", "model")
    assert closed == [True]


@pytest.mark.parametrize("failure", ["identity", "configuration"])
def test_chat_backend_closes_transport_when_probe_fails(monkeypatch, failure):
    from opentpu.host import board, chat
    closed = []
    tr = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(board, "open_transport", lambda dev: tr)

    def info():
        if failure == "identity":
            raise RuntimeError("identity probe failed")
        return {}

    def config(_):
        raise RuntimeError("configuration probe failed")

    monkeypatch.setattr(board, "Board", lambda transport, **kw: SimpleNamespace(info=info))
    monkeypatch.setattr(board, "device_config", config)
    with pytest.raises(RuntimeError, match=f"{failure} probe failed"):
        chat.make_backend("board", None, 1024, "f2:0")
    assert closed == [True]
