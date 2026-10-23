import json
from pathlib import Path
import numpy as np
import pytest
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.llm.qwen3_tts_shared import shared_tts_config, CFG_HASH


def test_source_format_prefill_has_pure_role_rows_and_exact_control_order():
    base = Path(__file__).parent / "fixtures"
    config = json.loads((base / "qwen3_tts_06b_config.json").read_text())
    fixture = json.loads((base / "qwen3_tts_prompts.json").read_text())
    rows = custom_voice_prefill(fixture["prompts"][0]["input_ids"], config)
    assert [(r.text_id, r.codec_id, r.kind) for r in rows[:3]] == [(151644, 0, "role"), (77091, 0, "role"), (198, 0, "role")]
    assert [(r.text_id, r.codec_id) for r in rows[3:]] == [
        (151671, 2154), (151671, 2156), (151671, 2050),
        (151671, 2157), (151671, 3061), (151672, 2148),
        (9707, 2148), (13, 2148), (151673, 2148), (151671, 2149)]
    assert len(rows) == 13
    auto = custom_voice_prefill(fixture["prompts"][0]["input_ids"], config, language="Auto")
    assert len(auto) == 12 and auto[3].codec_id == 2155
    dialect = custom_voice_prefill(fixture["prompts"][0]["input_ids"], config, speaker="Eric", language="Auto")
    assert dialect[5].codec_id == 2062
    with pytest.raises(ValueError, match="wrapper"):
        custom_voice_prefill([1, 2, 3], config)


def test_role_path_excludes_codec_embedding_and_matches_independent_reference():
    pytest.importorskip("torch")
    from test_qwen3_tts_stack import tiny
    from test_qwen3_tts_talker import make_weights
    from opentpu.llm.qwen3_tts_stack import StackISASession
    from qwen3_tts_stack_reference import StackReference
    image = tiny(3)
    weights = make_weights(image)
    a, b = StackISASession(image, image.build(weights)), StackReference(image.spec, weights, 3)
    expected = b.step(3, 0, input_kind="role")
    actual = a.step(3, 0, input_kind="role")
    assert np.linalg.norm(actual[0] - expected["logits"]) / np.linalg.norm(expected["logits"]) < 0.05
    np.testing.assert_array_equal(a.read("projected"), a.read("assembled"))
    assert len(image.compile_parts(0, "role")) == 5


def test_exact_shared_profile_or_fail_closed_on_old_runtime():
    from opentpu.isasim import Config
    if "PORTB" not in Config.__dataclass_fields__:
        with pytest.raises(RuntimeError, match="wide-port runtime"):
            shared_tts_config()
    else:
        from opentpu.target import arch_hash
        c = shared_tts_config()
        assert c.MCOLS == 8 and c.PORTB == 4 and c.ROWLANES and c.ALINE == 64
        assert arch_hash(c) == CFG_HASH


def test_exact_operator_replay_matches_compiled_layers():
    from test_qwen3_tts_stack import tiny
    from test_qwen3_tts_talker import make_weights
    from opentpu.llm.qwen3_tts_stack import StackISASession
    from qwen3_tts_exact_mx import ExactMX
    image = tiny(3)
    weights = make_weights(image, 83)
    ref = ExactMX(image.spec, weights, 3)
    session = StackISASession(image, image.build(weights))
    for pair in [(1, 3), (16, 12), (0, 0)]:
        want = ref.step(*pair)
        actual = session.step(*pair)[0]
        np.testing.assert_array_equal(actual.view(np.uint32), want["logits"].view(np.uint32))
