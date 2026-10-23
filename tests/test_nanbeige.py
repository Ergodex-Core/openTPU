"""Loop semantics, shared storage, G=6 attention, and speculative verify rows."""
from dataclasses import replace
import json

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import load_spec
from opentpu.llm.nanbeige import Spec, Engine, reference_logits


def tiny_weights(spec, seed=4):
    rng = np.random.default_rng(seed)
    def mat(n, k):
        return (rng.normal(size=(n, k)) * (0.24 / np.sqrt(k))).astype(np.float32)
    def norm(n):
        return rng.uniform(0.7, 1.3, size=n).astype(np.float32)
    h, d = spec.hidden, spec.head_dim
    W = {"model.embed_tokens.weight": mat(spec.vocab, h) * 20,
         "lm_head.weight": mat(spec.vocab, h), "model.norm.weight": norm(h)}
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        W.update({p + "input_layernorm.weight": norm(h),
                  p + "post_attention_layernorm.weight": norm(h),
                  p + "self_attn.q_proj.weight": mat(spec.n_q * d, h),
                  p + "self_attn.k_proj.weight": mat(spec.n_kv * d, h),
                  p + "self_attn.v_proj.weight": mat(spec.n_kv * d, h),
                  p + "self_attn.o_proj.weight": mat(h, spec.n_q * d),
                  p + "mlp.gate_proj.weight": mat(spec.ffn, h),
                  p + "mlp.up_proj.weight": mat(spec.ffn, h),
                  p + "mlp.down_proj.weight": mat(h, spec.ffn)})
    return W


@pytest.fixture(scope="module")
def fixture():
    spec = Spec(hidden=128, layers=2, n_q=6, n_kv=1, head_dim=128,
                ffn=256, vocab=128, eps=1e-5, theta=7e7, capture_layers=(0, 1, 2, 3))
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=8 << 20)
    return spec, cfg, tiny_weights(spec)


def test_config_routes_and_retains_loop_contract(tmp_path):
    c = dict(model_type="nanbeige", hidden_size=3072, num_hidden_layers=22,
             num_attention_heads=48, num_key_value_heads=8, head_dim=128,
             intermediate_size=10752, vocab_size=166144, num_loops=2,
             rms_norm_eps=1e-5, rope_theta=70000000.0, skip_loop_final_norm=False)
    (tmp_path / "config.json").write_text(json.dumps(c))
    s = load_spec(tmp_path)
    assert isinstance(s, Spec)
    assert (s.layers, s.logical_layers, s.n_q * s.head_dim) == (22, 44, 6144)
    assert not s.qk_norm and not s.tied and not s.skip_loop_final_norm


def test_shared_weights_distinct_logical_caches(fixture):
    s, cfg, _ = fixture
    image = s.image(cfg, 128, rows=4)
    m = image.descriptors(0)
    a, b = m.loop_layer(0, loop=0), m.loop_layer(0, loop=1)
    assert a.wq.data.static() == b.wq.data.static()
    assert a.wd.parts[0].data.static() == b.wd.parts[0].data.static()
    assert a.kv.k(0).data.static() != b.kv.k(0).data.static()
    assert image.capture - image.physical_nbytes == s.layers * image.extra_kv_stride
    assert s.image(cfg, 128).nbytes < replace(s, loops=1, layers=4).image(cfg, 128).nbytes


def test_gqa6_loop_norm_and_multitoken_verification(fixture):
    s, cfg, W = fixture
    tokens = [4, 9, 23, 11]
    a = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    sequential = np.stack([a.step(t) for t in tokens])
    ref = reference_logits(s, W, tokens)
    cosine = (ref * sequential).sum(-1) / (np.linalg.norm(ref, axis=-1) *
                                           np.linalg.norm(sequential, axis=-1))
    assert cosine.min() > 0.9995
    b = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    rows = b.run_rows([(0, i) for i in range(4)], tokens, list(range(4)))
    np.testing.assert_array_equal(rows.view(np.uint32), sequential.view(np.uint32))
    np.testing.assert_array_equal(b.read_hidden((0,), rows=4)[:, 0],
                                  W["model.embed_tokens.weight"][tokens])
    # Force the production fallback also for a small fixture, where both forms fit.
    c = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    c.image.compile_rows = c.image.compile_tiled_rows
    tiled = c.run_rows([(0, i) for i in range(4)], tokens, list(range(4)))
    np.testing.assert_array_equal(tiled.view(np.uint32), sequential.view(np.uint32))
    np.testing.assert_array_equal(c.read_hidden((0, 1, 2, 3), rows=4).view(np.uint32),
                                  b.read_hidden((0, 1, 2, 3), rows=4).view(np.uint32))
    # The intermediate norm changes model semantics; this fixture exposes omission.
    no_norm = reference_logits(replace(s, skip_loop_final_norm=True), W, tokens)
    assert np.linalg.norm(ref - no_norm) / np.linalg.norm(ref) > 0.01


def test_rollback_replaces_rejected_verify_cache(fixture):
    s, cfg, W = fixture
    a = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    a.prefill([2, 7])
    a.run_rows([(0, 2), (0, 3)], [9, 12], [0, 1])
    a.rollback(2)
    after = a.step(19)
    b = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    b.prefill([2, 7])
    np.testing.assert_array_equal(after.view(np.uint32), b.step(19).view(np.uint32))


def test_production_layout_capacity_without_allocating_weights():
    s = Spec(3072, 22, 48, 8, 128, 10752, 166144, eps=1e-5, theta=7e7)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=4 << 30)
    image = s.image(cfg, 128, rows=1)
    assert image.nbytes < cfg.DRAM_BYTES
    with pytest.raises(ValueError, match="generic"):
        s.image(cfg, 128, spec_rows=8)


def test_resident_generate_runs_both_loops(fixture):
    s, cfg, W = fixture
    plain = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    token = 5
    expected = []
    for _ in range(3):
        token = int(np.argmax(plain.step(token)))
        expected.append(token)
    resident = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False, resident=True)
    assert resident.generate_card(5, 3, stop_ids=[]) == expected


def test_selected_capture_and_disabled_baseline_are_exact(fixture):
    s, cfg, W = fixture
    tokens = [3, 9, 20]
    all_taps = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=4, prog_cache=False)
    wanted = np.stack([all_taps.step(t) for t in tokens])
    captured = all_taps.read_hidden((1, 3))
    for taps in ((), (3, 1)):
        eng = Engine(replace(s, capture_layers=taps), W, cap=128, cfg=cfg, runtime_host=False,
                     rows=4, prog_cache=False)
        np.testing.assert_array_equal(np.stack([eng.step(t) for t in tokens]), wanted)
        assert eng.image.nbytes < all_taps.image.nbytes
        if taps:
            np.testing.assert_array_equal(eng.read_hidden((1, 3)), captured)
            with pytest.raises(ValueError, match="not enabled"):
                eng.read_hidden((0,))
        else:
            assert eng.image.capture is None and eng.image.capture_control is None
            with pytest.raises(ValueError, match="not enabled"):
                eng.read_hidden((1, 3))


def test_eight_row_tiled_memory_schedule_is_exact(fixture):
    s, cfg, W = fixture
    tokens = [3, 9, 20, 5, 10, 18, 12, 30]
    one = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=8, prog_cache=False)
    expected = np.stack([one.step(t) for t in tokens])
    wide = Engine(s, W, cap=128, cfg=cfg, runtime_host=False, rows=8, prog_cache=False)
    wide.image.compile_rows = wide.image.compile_tiled_rows
    result = wide.run_rows([(0, i) for i in range(8)], tokens, list(range(8)))
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_runtime_host_programs_reuse_and_rollback_match_static(fixture):
    s, cfg, W = fixture
    static = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False, runtime_host=False)
    runtime = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False)
    taps = (0, 1, 2, 3)
    for token in (4, 7, 9):
        np.testing.assert_array_equal(runtime.step(token).view(np.uint32),
                                      static.step(token).view(np.uint32))
        np.testing.assert_array_equal(runtime.read_hidden(taps), static.read_hidden(taps))
    assert not runtime.device_inputs and runtime.runtime_host
    assert len(runtime._host_programs) == 1
    assert not runtime.stats[0]["host_program_cache_hit"]
    assert runtime.stats[1]["host_program_cache_hit"]
    for tokens in ([3, 5, 6, 11], [17, 19, 22, 31]):
        rows = [(0, runtime.pos + i) for i in range(4)]
        actual = runtime.run_rows(rows, tokens, list(range(4)))
        expected = static.run_rows(rows, tokens, list(range(4)))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(runtime.read_hidden(taps, rows=4),
                                      static.read_hidden(taps, rows=4))
        runtime.pos += 4
        static.pos += 4
    assert len(runtime._host_programs) == 2
    assert runtime.stats[-1]["host_program_cache_hit"]
    runtime.rollback(4)
    static.rollback(4)
    np.testing.assert_array_equal(runtime.step(21).view(np.uint32),
                                  static.step(21).view(np.uint32))
    np.testing.assert_array_equal(runtime.read_hidden(taps), static.read_hidden(taps))


def test_runtime_tiled_host_rows_masks_match_static(fixture):
    from opentpu.compiler import arg_words
    from opentpu.llm.qwen3 import RunRows
    s, cfg, W = fixture
    static = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False, runtime_host=False)
    runtime = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False)
    # Exercise the additive mask tiles on GQA's 4+2 query groups,
    # including a reused program at another position with stale future KV data.
    programs, arguments = runtime.image.compile_host_rows(1, 4, range(4), tiled=True)
    for position in (0, 4, 1, 124):
        tokens = [3, 9, 20, 5]
        rows = [(0, position + i) for i in range(4)]
        runtime._host_inputs(tokens, [p for _, p in rows])
        runtime.backend.run(programs, args=arg_words(arguments, RunRows.values(tokens, position)))
        result = runtime._host_logits(range(4))
        expected = static.run_rows(rows, tokens, list(range(4)))
        np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(runtime.read_hidden((0, 1, 2, 3), rows=4),
                                      static.read_hidden((0, 1, 2, 3), rows=4))


def test_runtime_host_bucket_boundary_and_partial_last_bucket(fixture):
    s, cfg, W = fixture
    static = Engine(s, W, cap=384, cfg=cfg, rows=4, prog_cache=False, runtime_host=False)
    runtime = Engine(s, W, cap=384, cfg=cfg, rows=4, prog_cache=False)
    # Sparse history is intentional: test mask/address selection at both sides
    # of the bucket boundary without spending time filling its irrelevant middle.
    for pos in (0, 255, 256, 383):
        static.pos = runtime.pos = pos
        np.testing.assert_array_equal(runtime.step(7).view(np.uint32),
                                      static.step(7).view(np.uint32))
    assert len(runtime._host_programs) == 2
    rows = [(0, p) for p in range(254, 258)]
    np.testing.assert_array_equal(runtime.run_rows(rows, [3, 5, 7, 9], [3]),
                                  static.run_rows(rows, [3, 5, 7, 9], [3]))
    assert "runtime_host" not in runtime.stats[-1]  # exact static boundary fallback


def test_runtime_compile_failure_is_cached_without_poisoning_smaller_rows(fixture, monkeypatch):
    from opentpu.compiler import CompileError
    s, cfg, W = fixture
    runtime = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False)
    static = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False, runtime_host=False)
    compile_rows = runtime.image.compile_host_rows
    attempts = []

    def capacity_limit(bucket, rows, outputs, block):
        attempts.append(rows)
        if rows == 4:
            raise CompileError("test verification capacity limit")
        return compile_rows(bucket, rows, outputs, block)

    monkeypatch.setattr(runtime.image, "compile_host_rows", capacity_limit)
    for position in (0, 3):
        with pytest.raises(CompileError, match="capacity limit"):
            runtime.run_rows([(0, position + i) for i in range(4)], [2, 4, 6, 8], [3])
    assert attempts == [4]
    assert not runtime._host_programs and runtime.pos == 0
    for position in (0, 3):
        rows, tokens = [(0, position + i) for i in range(3)], [2, 4, 6]
        np.testing.assert_array_equal(runtime.run_rows(rows, tokens, [0, 1, 2]),
                                      static.run_rows(rows, tokens, [0, 1, 2]))
        np.testing.assert_array_equal(runtime.read_hidden((0, 1, 2, 3), rows=3),
                                      static.read_hidden((0, 1, 2, 3), rows=3))
    assert attempts == [4, 3] and len(runtime._host_programs) == 1


def test_production_runtime_verification_fits_burst200_imem():
    s = Spec(3072, 22, 48, 8, 128, 10752, 166144, eps=1e-5, theta=7e7,
             capture_layers=(2, 11, 22, 33, 41))
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=4 << 30, SEQ_WC=True)
    image = s.image(cfg, 128, rows=4)
    programs, arguments = image.compile_host_rows(1, 4, range(4))
    assert max(len(p) * 8 for p in programs) <= cfg.IMEM_WORDS
    assert arguments and image.nbytes < cfg.DRAM_BYTES


def test_runtime_rows_use_board_start_wait_argument_api(fixture):
    from opentpu.llm.qwen3 import IsaBackend
    s, cfg, W = fixture

    class BoardApi:
        args = True

        def __init__(self, config, images):
            self.isa = IsaBackend(config, images)
            self.started = 0
            self.waited = 0

        def write(self, *args):
            return self.isa.write(*args)

        def read(self, *args):
            return self.isa.read(*args)

        def run(self, programs):
            # BoardBackend.run intentionally has no runtime-arguments keyword.
            raise AssertionError("runtime rows must use start(programs, args=) + wait()")

        def start(self, programs, *, args=None):
            self.started += 1
            self.result = self.isa.run(programs, args=args)

        def wait(self):
            self.waited += 1
            return self.result

    board_api = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False, backend=BoardApi)
    static = Engine(s, W, cap=128, cfg=cfg, rows=4, prog_cache=False, runtime_host=False)
    np.testing.assert_array_equal(board_api.step(3), static.step(3))
    rows, tokens = [(0, p) for p in range(1, 5)], [5, 7, 9, 11]
    np.testing.assert_array_equal(board_api.run_rows(rows, tokens, list(range(4))),
                                  static.run_rows(rows, tokens, list(range(4))))
    np.testing.assert_array_equal(board_api.read_hidden((0, 1, 2, 3), rows=4),
                                  static.read_hidden((0, 1, 2, 3), rows=4))
    assert board_api.backend.started == board_api.backend.waited == 2
