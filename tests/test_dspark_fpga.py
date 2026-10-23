"""ISA checks for the co-resident DSpark FPGA kernels, not checkpoint quality.

The draft uses its documented INT4 repack and the target's INT8 output head.
Tests require correct noncausal query-block semantics and unchanged target state.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm.dspark import Spec as DraftSpec
from opentpu.llm.nanbeige import Engine as TargetEngine, Spec as TargetSpec
from opentpu.llm.nanbeige_specdec import Decoder
from tools.nanbeige_fixture import tiny_fixture


def tiny_config():
    return Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                  TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=32 << 20,
                  PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True)


def make_pair(fixture=None, *, device_fusion_markov=False):
    from opentpu.llm.dspark_fpga import Engine as DraftEngine
    ts, tw, ds, dw = fixture if fixture is not None else tiny_fixture()
    target = TargetEngine(ts, tw, cfg=tiny_config(), cap=128, rows=4,
                          pipeline=False, prog_cache=False)
    draft = DraftEngine(target, ds, dw, target_weights=tw, rows=7, block=256,
                        **({"device_fusion_markov": True} if device_fusion_markov else {}))
    return target, draft


def context(draft, count=2, seed=17):
    s = draft.spec
    aux = np.random.default_rng(seed).normal(
        size=(count, len(s.target_layers), s.target_hidden)).astype(np.float32)
    return draft.project_context(aux, np.arange(count, dtype=np.int64))


def exact(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def test_full_seven_query_prefix_and_noncausal_future_slot():
    _, draft = make_pair()
    ctx = context(draft)
    before = ctx.hidden.copy()
    ids = [1] + [draft.spec.mask_token] * 6
    positions = np.arange(2, 9)
    full_hidden = draft.backbone(ids, positions, ctx)
    short_hidden = draft.backbone(ids[:3], positions[:3], ctx)
    changed = draft.backbone(ids[:-1] + [9], positions, ctx)
    assert full_hidden.shape == (7, draft.spec.hidden)
    assert np.max(np.abs(full_hidden[:3] - short_hidden)) > 1e-5
    assert np.max(np.abs(full_hidden[0] - changed[0])) > 1e-5
    full = draft.draft(1, 2, ctx, 7, query_tokens=7)
    prefix = draft.draft(1, 2, ctx, 3, query_tokens=7)
    exact(prefix.hidden, full.hidden[:3])
    exact(prefix.logits, full.logits[:3])
    np.testing.assert_array_equal(prefix.tokens, full.tokens[:3])
    exact(ctx.hidden, before)
    np.testing.assert_array_equal(ctx.positions, [0, 1])


def test_context_rollback_replay_replaces_device_cache_and_preserves_inputs():
    _, draft = make_pair()
    full = context(draft, 4)
    hidden_before = full.hidden.copy()
    prefix = full.truncate(2)
    # Populate future context first. A rolled-back call must not attend to it.
    draft.draft(1, 4, full, 3, query_tokens=7)
    rolled_back = draft.draft(1, 2, prefix, 3, query_tokens=7)
    _, clean = make_pair()
    clean_prefix = context(clean, 4).truncate(2)
    expected = clean.draft(1, 2, clean_prefix, 3, query_tokens=7)
    exact(rolled_back.hidden, expected.hidden)
    exact(rolled_back.logits, expected.logits)
    np.testing.assert_array_equal(rolled_back.tokens, expected.tokens)
    # Replaying the same committed context must give the same seven-query result.
    repeat = draft.draft(1, 2, prefix, 3, query_tokens=7)
    exact(repeat.hidden, rolled_back.hidden)
    exact(full.hidden, hidden_before)
    np.testing.assert_array_equal(full.positions, np.arange(4))


@pytest.mark.parametrize("device_fusion_markov", [False, True], ids=["host", "device"])
def test_markov_head_conditions_each_token_on_previous_proposal(device_fusion_markov):
    ts, tw, ds, dw = tiny_fixture()
    tw["lm_head.weight"].fill(0)
    w1 = dw["markov_head.markov_w1.weight"]
    w2 = dw["markov_head.markov_w2.weight"]
    w1.fill(0)
    w2.fill(0)
    w1[1, 0] = w1[0, 0] = w1[3, 1] = w1[5, 2] = 100
    w2[3, 0] = w2[5, 1] = w2[0, 2] = 1
    _, draft = make_pair((ts, tw, ds, dw), device_fusion_markov=device_fusion_markov)
    proposal = draft.draft(1, 2, context(draft), 7, query_tokens=7)
    np.testing.assert_array_equal(proposal.tokens, [3, 5, 0, 3, 5, 0, 3])
    np.testing.assert_array_equal(np.argmax(proposal.logits, axis=-1), proposal.tokens)


@pytest.mark.parametrize("device_fusion_markov", [False, True], ids=["host", "device"])
def test_draft_preserves_entire_target_image_and_next_token(device_fusion_markov):
    target, draft = make_pair(device_fusion_markov=device_fusion_markov)
    golden, _ = make_pair()
    captures = []
    for token in [2, 7]:
        exact(target.step(token), golden.step(token))
        captures.append(target.read_hidden(draft.spec.capture_layers)[0])
    before = target.backend.read(0, 0, target.image.nbytes)
    ctx = draft.project_context(np.stack(captures), np.arange(2))
    draft.draft(9, 2, ctx, 3, query_tokens=7)
    after = target.backend.read(0, 0, target.image.nbytes)
    np.testing.assert_array_equal(after, before)
    assert draft.image.base >= target.image.nbytes
    assert draft.image.nbytes <= target.cfg.DRAM_BYTES
    exact(target.step(9), golden.step(9))
    exact(target.read_hidden(draft.spec.capture_layers),
          golden.read_hidden(draft.spec.capture_layers))


@pytest.mark.parametrize("device_fusion_markov", [False, True], ids=["host", "device"])
def test_fpga_draft_with_isa_target_preserves_exact_greedy_tokens(device_fusion_markov):
    target, draft = make_pair(device_fusion_markov=device_fusion_markov)
    golden, _ = make_pair()
    prompt = [2, 7]
    for token in prompt:
        logits = golden.step(token)
    expected = [int(np.argmax(logits))]
    for _ in range(6):
        expected.append(int(np.argmax(golden.step(expected[-1]))))
    decoder = Decoder(target, draft, draft_tokens=3, prefill_chunk=2)
    generated = decoder.generate(prompt, max_new_tokens=7)
    assert list(generated.tokens) == expected
    assert decoder.target.pos == len(prompt) + 6
    assert decoder.context.positions.tolist() == list(range(decoder.target.pos))
    assert generated.stats.draft_calls > 0
    assert generated.stats.computed_query_tokens == generated.stats.draft_calls * 7
    assert "numpy-fp32-reference" not in generated.stats.draft_backend


@pytest.mark.parametrize("device_fusion_markov", [False, True], ids=["host", "device"])
def test_production_co_resident_layout_and_seven_query_imem_fit(tmp_path, device_fusion_markov):
    """Compile real checkpoint dimensions without materializing model weights."""
    from opentpu import target as targets, waits
    from opentpu.llm.dspark_fpga import Image as DraftImage

    data = Path(__file__).resolve().parents[1] / "tools/data/nanbeige"
    for name in ("target", "draft"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text((data / f"{name}-config.json").read_text())
    ds = DraftSpec.from_hf(tmp_path / "draft")
    ts = replace(TargetSpec.from_hf(tmp_path / "target"), capture_layers=ds.capture_layers)
    cfg = targets.config(targets.load(str(data / "burst200.toml")))
    target = ts.image(cfg, 128, rows=4)
    reservation = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    base = ((target.nbytes + 4095) // 4096 * 4096 + reservation + 4095) // 4096 * 4096
    draft = DraftImage(ds, cfg, 256, rows=7, base=base, head=target.head,
                       **({"device_fusion_markov": True} if device_fusion_markov else {}))
    assert draft.base >= target.nbytes + reservation
    assert draft.head == target.head
    assert draft.nbytes <= cfg.DRAM_BYTES
    for position in (0, 32, 127):
        programs = draft.compile_query(position, queries=7, predictions=3, block=256)
        assert programs and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)
    programs, arguments = draft.compile_query_runtime(0, queries=7, predictions=3, block=256)
    assert arguments and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)
    for rows in (1, 4, 7):
        programs = draft.compile_context(0, rows)
        assert programs and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)
        programs, arguments = draft.compile_context_runtime(0, rows)
        assert arguments and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)
    if device_fusion_markov:
        for rows in (1, draft.fusion_rows):
            programs = draft.compile_fusion(rows)
            assert programs and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)
        programs, arguments = draft.compile_markov()
        assert arguments and all(0 < len(p) * 8 <= cfg.IMEM_WORDS for p in programs)


def test_runtime_position_programs_match_static_and_reuse_query_programs():
    _, runtime = make_pair()
    _, static = make_pair()
    static.device.args = False
    cached_queries = {}
    for position in (2, 9, 17):
        ctx = context(runtime, position)
        ids = [1] + [runtime.spec.mask_token] * 6
        positions = np.arange(position, position + 7)
        exact(runtime.backbone(ids, positions, ctx), static.backbone(ids, positions, ctx))
        actual = runtime.draft(1, position, ctx, 3, query_tokens=7)
        expected = static.draft(1, position, ctx, 3, query_tokens=7)
        exact(actual.hidden, expected.hidden)
        exact(actual.logits, expected.logits)
        np.testing.assert_array_equal(actual.tokens, expected.tokens)
        query_programs = {key: value for key, value in runtime._programs.items()
                          if key[0] == "query"}
        assert len(query_programs) == 2  # Backbone-only and three-row output head.
        assert all(key[1] == "runtime" for key in query_programs)
        if cached_queries:
            assert query_programs.keys() == cached_queries.keys()
            assert all(query_programs[key] is value for key, value in cached_queries.items())
        cached_queries = query_programs
    assert len([key for key in static._programs if key[0] == "query"]) == 6


def test_changed_same_length_context_invalidates_cached_prefix():
    from opentpu.llm.dspark_fpga import ProjectedContext
    _, reused = make_pair()
    original = context(reused, 4)
    replacement = context(reused, 4, seed=42)
    for common_prefix in (0, 2):
        original_proposal = reused.draft(1, 4, original, 3, query_tokens=7)
        hidden = replacement.hidden.copy()
        hidden[:common_prefix] = original.hidden[:common_prefix]
        changed = ProjectedContext(original.positions, hidden)
        actual = reused.draft(1, 4, changed, 3, query_tokens=7)
        _, clean = make_pair()
        expected = clean.draft(1, 4, changed, 3, query_tokens=7)
        exact(actual.hidden, expected.hidden)
        exact(actual.logits, expected.logits)
        np.testing.assert_array_equal(actual.tokens, expected.tokens)
        assert np.max(np.abs(actual.hidden - original_proposal.hidden)) > 1e-5
        exact(original.hidden, context(reused, 4).hidden)


def test_runtime_arguments_use_board_start_wait_interface():
    class BoardStyleBackend:
        """BoardBackend API shape, with real ISA execution behind the adapter."""
        args = True

        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.starts = self.waits = self.runs = 0
            self.pending = None

        def write(self, *args):
            return self.wrapped.write(*args)

        def read(self, *args):
            return self.wrapped.read(*args)

        def run(self, programs):  # Intentionally rejects an args keyword.
            self.runs += 1
            return self.wrapped.run(programs)

        def start(self, programs, *, args):
            assert self.pending is None
            assert args
            self.starts += 1
            self.pending = (programs, args)

        def wait(self):
            assert self.pending is not None
            self.waits += 1
            programs, args = self.pending
            self.pending = None
            return self.wrapped.run(programs, args=args)

    _, draft = make_pair()
    adapter = BoardStyleBackend(draft.device)
    draft.device = adapter
    _, static = make_pair()
    static.device.args = False
    ctx = context(draft, 9)
    actual = draft.draft(1, 9, ctx, 3, query_tokens=7)
    expected = static.draft(1, 9, ctx, 3, query_tokens=7)
    exact(actual.hidden, expected.hidden)
    exact(actual.logits, expected.logits)
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    assert adapter.starts == adapter.waits == 3  # Two context chunks and one query.
    assert adapter.runs == 0
    assert adapter.pending is None


def quantized_projection_reference(x, weight, cfg, gamma=None, eps=1e-5):
    """Independent runtime allocation/packing and simple INT8 ISA projection."""
    from opentpu import language as ol
    from opentpu.kernels.lib import rmsnorm
    from opentpu.runtime import Input, Output, Weight, launch

    @ol.jit
    def projection(x, weight, gain, out, normalize, eps):
        y = ol.dot(ol.load(x), weight)
        if normalize:
            y = rmsnorm(y, ol.load(gain), eps)
        ol.store(out, y)

    x = np.asarray(x, np.float32).reshape(-1, weight.shape[1])
    gain = np.ones(weight.shape[0], np.float32) if gamma is None else gamma
    result = launch(projection, cfg, x=Input(x), weight=Weight(weight, fmt="int8"),
                    gain=Input(gain), out=Output((len(x), weight.shape[0])),
                    normalize=gamma is not None, eps=eps)
    return result.outputs["out"]


def test_device_fusion_matches_independent_quantized_projection():
    _, draft = make_pair(device_fusion_markov=True)
    spec = draft.spec
    aux = np.random.default_rng(31).normal(
        size=(9, len(spec.target_layers), spec.target_hidden)).astype(np.float32)
    original = aux.copy()
    positions = np.arange(9, dtype=np.int64)
    actual = draft.project_context(aux, positions)
    expected = quantized_projection_reference(
        aux.reshape(len(aux), -1), draft.weights["fc.weight"], draft.cfg,
        gamma=draft.weights["hidden_norm.weight"], eps=spec.eps)
    exact(actual.hidden, expected)
    np.testing.assert_array_equal(actual.positions, positions)
    exact(aux, original)
    # Host batching must not change projection or FP32 RMS arithmetic.
    one_by_one = np.concatenate([
        draft.project_context(aux[i:i + 1], positions[i:i + 1]).hidden
        for i in range(len(aux))])
    exact(one_by_one, actual.hidden)


def test_device_markov_matches_independent_quantized_projection_and_preserves_base():
    ts, tw, ds, dw = tiny_fixture()
    ds = replace(ds, logit_scale=.75)
    _, draft = make_pair((ts, tw, ds, dw), device_fusion_markov=True)
    base = np.random.default_rng(13).normal(size=(7, ds.vocab)).astype(np.float32)
    draft.device.write(0, draft.image.io["logits"], base)
    padded_w2 = np.pad(dw["markov_head.markov_w2.weight"],
                       ((0, 0), (0, draft.image.markov_width - ds.markov_rank)))
    for row, token in ((0, 1), (2, 7), (6, 3)):
        embedding = dw["markov_head.markov_w1.weight"][token]
        padded_embedding = np.pad(embedding, (0, draft.image.markov_width - ds.markov_rank))
        correction = quantized_projection_reference(padded_embedding, padded_w2, draft.cfg)[0]
        expected = base[row] * np.float32(ds.logit_scale) + correction * np.float32(ds.logit_scale)
        actual = draft._markov_logits(embedding, row)
        exact(actual, expected)
    preserved = draft.device.read(0, draft.image.io["logits"], base.nbytes)
    exact(preserved.view(np.float32).reshape(base.shape), base)
