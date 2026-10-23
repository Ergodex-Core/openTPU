import numpy as np
import pytest
from opentpu.llm.qwen3_tts_talker_precision import TalkerPrecisionFrameImage
from opentpu.llm.qwen3_tts_frame import CodecFrameISASession
from opentpu.llm.qwen3_tts_prompt import PromptRow
from test_qwen3_tts_frame import tiny
from test_qwen3_tts_talker import make_weights


def corrected_tiny():
    base = tiny()
    return TalkerPrecisionFrameImage(base.spec, base.cfg, 64, 32, layers=3,
                                    predictor_layers=5, codebooks=16, audio_vocab=10, eos=12, pad_id=16)


def test_corrected_talker_source_algebra_boundary_poison_and_reset():
    from qwen3_tts_frame_reference import FrameReference
    image = corrected_tiny()
    weights = make_weights(image)
    packed = image.build(weights)
    a, b = CodecFrameISASession(image, packed), CodecFrameISASession(image, packed)
    ref = FrameReference(image, weights)
    for address, size in image.cache_regions().values():
        b.machine.slices[0].dram[address:address + size] = 0x3f
    for position in range(35):
        row = PromptRow(position * 7 % 17, position * 5 % 13, 'role' if position < 3 else 'aligned')
        expected = ref.prefill(row)
        logits = a.step(row.text_id, row.codec_id, input_kind=row.kind)[0]
        np.testing.assert_array_equal(logits, b.step(row.text_id, row.codec_id, input_kind=row.kind)[0])
        assert np.linalg.norm(logits - expected['logits']) / np.linalg.norm(expected['logits']) < .002
    for _ in range(3):
        ids, expected = a.frame(), ref.frame()
        if ids is None:
            assert expected is None
            break
        np.testing.assert_array_equal(ids, b.frame())
        np.testing.assert_array_equal(ids, expected['ids'])
        assert np.linalg.norm(a.read('logits') - expected['feedback']['logits']) / np.linalg.norm(expected['feedback']['logits']) < .002
    a.reset()
    fresh = CodecFrameISASession(image, packed)
    for session in [a, fresh]:
        session.step(1, 2, input_kind='role')
    np.testing.assert_array_equal(a.frame(), fresh.frame())
    np.testing.assert_array_equal(a.read('logits'), fresh.read('logits'))


def test_corrected_native_bound_and_representative_maximum_programs():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    from opentpu.llm.qwen3_tts_talker import TalkerSpec
    from opentpu.kernels.tts_talker_precision import corrected_qkv, corrected_attention_head, corrected_finish
    cfg = shared_tts_config()
    image = TalkerPrecisionFrameImage(TalkerSpec(), cfg)
    assert len(image.parameters) == 402
    assert image.nbytes < 3 << 30
    layer = image.layers[-1]
    parts = [image.part('qkv', corrected_qkv, m=layer, position=511),
             image.part('attention', corrected_attention_head, m=layer, position=511, head=7),
             image.part('finish', corrected_finish, m=layer)]
    assert max(p[2]['imem_words'] for p in parts) <= cfg.IMEM_WORDS
    assert max(p[2]['tmem_peak_words'] for p in parts) <= cfg.TMEM_WORDS
    with pytest.raises(ValueError, match='512'):
        TalkerPrecisionFrameImage(TalkerSpec(), cfg, 32768)
    with pytest.raises(ValueError, match='capacity'):
        image.layer_parts(512)
