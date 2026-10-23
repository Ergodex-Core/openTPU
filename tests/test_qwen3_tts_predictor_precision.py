import numpy as np
from opentpu.llm.qwen3_tts_frame import CodecFrameISASession
from test_qwen3_tts_frame import tiny
from test_qwen3_tts_talker import make_weights
from qwen3_tts_predictor_exact import PredictorExactReplay
from opentpu import language as ol
from opentpu.llm.qwen3_tts_predictor_precision import PredictorPrecisionImage
from opentpu.kernels.tts_predictor_precision import activation_pair, corrected_dot


def precise_tiny():
    base = tiny()
    return PredictorPrecisionImage(base.spec, base.cfg, 64, 32, layers=3,
                                   predictor_layers=5, codebooks=16, audio_vocab=10, eos=12, pad_id=16)


@ol.jit
def corrected_affine(m):
    x = ol.load(m.output)
    h = m.spec.hidden
    pair = activation_pair(x, m.identities[h])
    ol.store(m.output, corrected_dot(pair, m.gate[0:h, :], m.gate_residual[0:h, :]))


@ol.jit
def baseline_affine(m):
    ol.store(m.output, ol.dot(ol.load(m.output), m.gate[0:m.spec.hidden, :]))


def test_predictor_exact_replay_nonzero_heads_and_cache_restart():
    image = tiny()
    weights = make_weights(image)
    session = CodecFrameISASession(image, image.build(weights))
    replay = PredictorExactReplay(image, weights)
    session.step(1, 2, input_kind='role')
    for _ in range(2):
        assert session.frame(replay.observe) is not None
    assert len(replay.rows) == 190
    assert all(r['bitwise_equal'] for r in replay.rows)
    assert all(r['reference_greedy'] == r['device_greedy'] for r in replay.rows)


def test_device_residual_affine_reduces_error():
    image = precise_tiny()
    weights = make_weights(image)
    packed = image.build(weights)
    x = np.random.default_rng(52).standard_normal((1, image.spec.hidden)).astype(np.float32)
    want = x @ weights['talker.code_predictor.model.layers.0.mlp.gate_proj.weight'][:image.spec.hidden].T
    errors = []
    for kernel in (baseline_affine, corrected_affine):
        session = CodecFrameISASession(image, packed)
        layer = image.predictor_layers[0]
        address = layer.output.base.const
        session.machine.slices[0].dram[address:address + x.nbytes] = x.view(np.uint8).reshape(-1)
        session.execute([image.part('affine', kernel, m=layer)])
        got = session.machine.slices[0].dram[address:address + x.nbytes].view(np.float32).reshape(x.shape)
        errors.append(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert errors[1] < .0003 and errors[1] < errors[0] / 20


def test_precision_predictor_against_identical_prefix_reference_and_poison():
    from qwen3_tts_frame_reference import FrameReference
    image = precise_tiny()
    weights = make_weights(image)
    packed = image.build(weights)
    session, poisoned = CodecFrameISASession(image, packed), CodecFrameISASession(image, packed)
    reference = FrameReference(image, weights)
    for address, size in image.cache_regions().values():
        poisoned.machine.slices[0].dram[address:address + size] = 0x3f
    for s in [session, poisoned]:
        s.step(1, 2, input_kind='role')
    for _ in range(2):
        hidden = session.read('normalized')
        logits = []
        def observe(name, s):
            if name.startswith('predictor.head.'):
                desc = image.predictor.logits
                logits.append(s.machine.slices[0].dram[desc.base.const:desc.base.const + image.audio_vocab * 4].view(np.float32).copy())
        ids = session.frame(observe)
        np.testing.assert_array_equal(ids, poisoned.frame())
        _, _, expected = reference.predictor_frame(hidden, int(ids[0]), force_ids=ids)
        for got, row in zip(logits, expected):
            assert np.linalg.norm(got - row['logits']) / np.linalg.norm(row['logits']) < .0005
        np.testing.assert_array_equal(session.read('logits'), poisoned.read('logits'))


def test_precision_native_instruction_and_residency_budget():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    from opentpu.llm.qwen3_tts_talker import TalkerSpec
    cfg = shared_tts_config()
    for capacity in (512, 32768):
        image = PredictorPrecisionImage(TalkerSpec(), cfg, capacity)
        parts = image.predictor_parts(15, 14)
        assert image.nbytes < cfg.DRAM_BYTES
        assert max(p[2]['imem_words'] for p in parts) < cfg.IMEM_WORDS
        assert max(p[2]['tmem_peak_words'] for p in parts) < cfg.TMEM_WORDS
