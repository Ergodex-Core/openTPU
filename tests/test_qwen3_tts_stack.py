import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_stack import TalkerStackImage, StackISASession
from test_qwen3_tts_talker import make_weights


def tiny(layers=3):
    s = TalkerSpec(text_vocab=17, codec_vocab=13, text_hidden=64, hidden=64,
                   n_q=4, n_kv=2, head_dim=32, ffn=128)
    return TalkerStackImage(s, Config(D=32, DRAM_BYTES=8 << 20), 64, 32,
                            layers=layers, audio_vocab=10, eos=12)


def greedy(logits, audio=10, eos=12):
    masked = logits.copy()
    masked[audio:eos] = -np.inf
    masked[eos + 1:] = -np.inf
    return int(np.argmax(logits)), int(np.argmax(masked))


def test_composition_masking_reset_and_boundary():
    pytest.importorskip("torch")
    from qwen3_tts_stack_reference import StackReference
    image = tiny()
    weights = make_weights(image, 51)
    packed = image.build(weights)
    a, b = StackISASession(image, packed), StackISASession(image, packed)
    ref = StackReference(image.spec, weights, 3)
    for address, size in image.cache_regions().values():
        b.machine.slices[0].dram[address:address + size] = 0x3f
    for pos in range(35):
        pair = (pos * 7 % 17, pos * 5 % 13)
        expected = ref.step(*pair)
        outputs = []
        got = a.step(*pair, observe=lambda name, session: outputs.append(session.read("output")) if name.startswith("layer.") else None)
        other = b.step(*pair)
        np.testing.assert_array_equal(got[0].view(np.uint32), other[0].view(np.uint32))
        assert got[1:] == greedy(got[0])
        assert np.linalg.norm(got[0] - expected["logits"]) / np.linalg.norm(expected["logits"]) < 0.05
        for actual, want in zip(outputs, expected["layers"]):
            assert np.linalg.norm(actual - want) / np.linalg.norm(want) < 0.05
    a.reset()
    fresh = StackISASession(image, packed)
    for pair in [(1, 2), (8, 12)]:
        np.testing.assert_array_equal(a.step(*pair)[0], fresh.step(*pair)[0])


def test_reference_layer_zero_matches_prior_independent_numpy_reference():
    pytest.importorskip("torch")
    from qwen3_tts_stack_reference import StackReference
    from qwen3_tts_talker_reference import Reference
    image = tiny(1)
    weights = make_weights(image)
    a, b = StackReference(image.spec, weights, 1), Reference(image.spec, weights)
    for pair in [(1, 3), (16, 12)]:
        x, y = a.step(*pair), b.step(*pair)
        np.testing.assert_allclose(x["layers"][0], y["output"], rtol=1e-11, atol=1e-12)


def test_device_greedy_ties_and_fixed_suppression():
    image = tiny(1)
    weights = make_weights(image)
    weights["talker.codec_head.weight"].fill(0)
    session = StackISASession(image, image.build(weights))
    logits, raw, masked = session.step(5, 4)
    assert raw == masked == 0
    np.testing.assert_array_equal(logits, np.zeros(13))
    # Explicitly drive head input: highest row is suppressed, then EOS wins.
    from opentpu.kernels.tts_stack import codec_logits
    weights["talker.model.norm.weight"].fill(1)
    weights["talker.codec_head.weight"][11].fill(2)
    weights["talker.codec_head.weight"][12].fill(1)
    session = StackISASession(image, image.build(weights))
    address = image.desc.output.base.const
    session.machine.slices[0].dram[address:address + 256] = np.ones(64, np.float32).view(np.uint8)
    b = codec_logits.trace(image.cfg, 0, {"m": image.desc, "audio_vocab": 10, "eos": 12})
    session.machine.load([b.finish()]).run()
    assert int(session.read("greedy_raw")[1]) == 11
    assert int(session.read("greedy_masked")[1]) == 12


@pytest.mark.parametrize("capacity,position", [(512, 256), (32768, 32767)])
def test_native_residency_instruction_split_and_exact_header(capacity, position):
    from opentpu.target import load
    cfg = Config(**{k: v for k, v in load("f2").arch.items() if k in Config.__dataclass_fields__})
    image = TalkerStackImage(TalkerSpec(), cfg, capacity)
    contract = json.loads((Path(__file__).parent / "fixtures/qwen3_tts_stack_contract.json").read_text())
    tensors = {k: {"shape": list(shape), "dtype": "BF16"} for k, (_, _, shape) in image.parameters.items()}
    encoded = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    assert len(tensors) == contract["tensors"]
    assert hashlib.sha256(encoded).hexdigest() == contract["tensor_contract_sha256"]
    parts = image.compile_parts(position)
    assert len(parts) == 29 and image.nbytes < 2**32
    assert max(p[2]["imem_words"] for p in parts) <= cfg.IMEM_WORDS
    assert sum(p[2]["imem_words"] for p in parts) > cfg.IMEM_WORDS
    assert max(p[2]["tmem_peak_words"] for p in parts) <= cfg.TMEM_WORDS
    with pytest.raises(NotImplementedError, match="sequential"):
        image.compile(0)


def test_invalid_tokens_do_not_touch_cache_and_no_offload():
    image = tiny()
    session = StackISASession(image, image.build(make_weights(image)))
    before = session.machine.slices[0].dram.copy()
    with pytest.raises(ValueError):
        session.step(-1, 0)
    np.testing.assert_array_equal(before, session.machine.slices[0].dram)
    with pytest.raises(MemoryError, match="no offload"):
        TalkerStackImage(image.spec, replace(image.cfg, DRAM_BYTES=100000), 64, 32, layers=28, audio_vocab=10, eos=12)
