import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_frame import CodecFrameImage, CodecFrameISASession, check_predictor_config
from test_qwen3_tts_talker import make_weights


def tiny():
    s = TalkerSpec(text_vocab=17, codec_vocab=13, text_hidden=64, hidden=64,
                   n_q=4, n_kv=2, head_dim=32, ffn=128)
    return CodecFrameImage(s, Config(D=32, DRAM_BYTES=32 << 20), 64, 32, layers=3,
                           predictor_layers=5, codebooks=16, audio_vocab=10, eos=12, pad_id=16)


def read_predictor(session, name):
    d = getattr(session.image.predictor, name)
    return session.machine.slices[0].dram[d.base.const:d.base.const + 4 * int(np.prod(d.shape))].view(np.float32).copy()


def test_full_frame_reference_feedback_reset_and_poison():
    pytest.importorskip('torch')
    from qwen3_tts_frame_reference import FrameReference
    from opentpu.llm.qwen3_tts_prompt import PromptRow
    im = tiny()
    weights = make_weights(im, 51)
    # Zero heads give deterministic first-index ties, isolating dataflow from
    # quantized low-margin greedy differences; real-weight acceptance is separate.
    weights['talker.codec_head.weight'].fill(0)
    for i in range(15):
        weights[f'talker.code_predictor.lm_head.{i}.weight'] *= 0
    packed = im.build(weights)
    a, b = CodecFrameISASession(im, packed), CodecFrameISASession(im, packed)
    reference = FrameReference(im, weights)
    for address, size in im.cache_regions().values():
        b.machine.slices[0].dram[address:address + size] = 0x3f
    for row in [PromptRow(1, 2, 'role'), PromptRow(5, 6)]:
        a.step(row.text_id, row.codec_id, input_kind=row.kind)
        b.step(row.text_id, row.codec_id, input_kind=row.kind)
        reference.prefill(row)
    for _ in range(3):
        seen = []
        ids = a.frame(lambda name, sess: seen.append(read_predictor(sess, 'logits')) if name.startswith('predictor.head.') else None)
        expected = reference.frame()
        np.testing.assert_array_equal(ids, b.frame())
        np.testing.assert_array_equal(ids, expected['ids'])
        assert len(seen) == 15 and np.all(ids[1:] == 0)
        # Independent source concat/sum validates inclusion of final embedding.
        np.testing.assert_allclose(a.read('codec_sum'), expected['codec_sum'], rtol=2e-6, atol=2e-6)
        last = expected['feedback']['layers'][-1]
        assert np.linalg.norm(a.read('output') - last) / np.linalg.norm(last) < .05
    a.reset()
    fresh = CodecFrameISASession(im, packed)
    for s in [a, fresh]:
        s.step(1, 2, input_kind='role')
    np.testing.assert_array_equal(a.frame(), fresh.frame())
    np.testing.assert_array_equal(a.read('logits'), fresh.read('logits'))


def test_predictor_cache_tail_masked_and_head_order_bounds():
    im = tiny()
    for pos in (0, 15):
        assert len(im.predictor_parts(pos, pos - 1 if pos else None)) == (7 if pos else 5)
    for args in [(0, 0), (16, 14), (1, 14)]:
        with pytest.raises(ValueError):
            im.predictor_parts(*args)
    session = CodecFrameISASession(im, im.build(make_weights(im)))
    with pytest.raises(ValueError, match='prefill'):
        session.frame()


def test_device_minimum_eos_and_no_predictor_on_stop():
    from opentpu.kernels.tts_frame import primary_select
    im = tiny()
    s = CodecFrameISASession(im, im.build(make_weights(im)))
    logits = np.zeros(13, np.float32)
    logits[11], logits[12], logits[3] = 99, 50, 1
    address = im.desc.logits.base.const
    s.machine.slices[0].dram[address:address + 52] = logits.view(np.uint8)
    for allow, want in [(False, 3), (True, 12)]:
        s.execute([im.part('select', primary_select, m=im.desc, audio_vocab=10, eos=12, allow_eos=allow)])
        assert s.read('primary')[1] == want
    s.position, s.frames = 2, 2
    before = s.machine.slices[0].dram.copy()
    assert s.frame() is None and s.stopped
    # EOS changes selected-ID outputs only; no model/KV or feedback runs.
    for address, size in im.cache_regions().values():
        np.testing.assert_array_equal(before[address:address + size], s.machine.slices[0].dram[address:address + size])


def test_execution_failure_latches_until_reset(monkeypatch):
    im = tiny()
    s = CodecFrameISASession(im, im.build(make_weights(im)))
    s.step(1, 2)
    original = s.machine.load
    def fail(programs):
        raise RuntimeError('injected diagnostic execution error')
    monkeypatch.setattr(s.machine, 'load', fail)
    with pytest.raises(RuntimeError, match='injected'):
        s.frame()
    assert s.failed
    with pytest.raises(ValueError, match='reset'):
        s.frame()
    with pytest.raises(ValueError, match='reset'):
        s.step(1, 2)
    monkeypatch.setattr(s.machine, 'load', original)
    s.reset()
    s.step(1, 2)
    assert not s.failed


def test_acceptance_provenance_with_relative_runpy_path(monkeypatch):
    import tools.qwen3_tts_frame_acceptance as tool
    monkeypatch.setattr(tool, '__file__', 'tools/qwen3_tts_frame_acceptance.py')
    hashes = tool.source_hashes()
    assert 'tools/qwen3_tts_frame_acceptance.py' in hashes
    assert all(len(v) == 64 for v in hashes.values())


def test_native_checkpoint_contract_and_shared_budget():
    from opentpu.llm.qwen3_tts_shared import shared_tts_config
    config = json.loads((Path(__file__).parent / 'fixtures/qwen3_tts_06b_config.json').read_text())
    check_predictor_config(config)
    cfg = shared_tts_config()
    im = CodecFrameImage(TalkerSpec.from_config(config), cfg)
    contract = json.loads((Path(__file__).parent / 'fixtures/qwen3_tts_frame_contract.json').read_text())
    tensors = {k: {'shape':list(s), 'dtype':'BF16'} for k, (_, _, s) in im.parameters.items()}
    assert len(tensors) == contract['tensors']
    assert hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == contract['tensor_contract_sha256']
    assert im.nbytes < 2 << 30
    parts = im.predictor_parts(15, 14) + im.feedback_parts(511)
    assert max(p[2]['imem_words'] for p in parts) < cfg.IMEM_WORDS
    assert max(p[2]['tmem_peak_words'] for p in parts) < cfg.TMEM_WORDS
