"""C3 sidecar/reference milestone: compare its independent row engine to real ISA ops."""
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, lens, quant as Q
from opentpu.isasim import Machine
from opentpu.llm import vocab_engine as C


def _case(vocab=19, hidden=1152, seed=1):
    layout = C.Layout(vocab=vocab, hidden=hidden)
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, (vocab, hidden), dtype=np.int8).view(np.uint8)
    sw = rng.uniform(-2, 2, vocab).astype(np.float32)
    dw = rng.integers(0, 32, (vocab, layout.kb), dtype=np.uint8)
    scales = Q.pack_scales("int8", sw, dw=dw)
    sidecar = C.Sidecar.build(layout, w, scales, embedding_weights=w, embedding_scales=scales)
    x = (rng.standard_normal(hidden) * np.exp(rng.uniform(-15, 15, hidden // 32)).repeat(32)).astype(np.float32)
    return layout, w, scales, sidecar, x


def _isa(layout, weights, scales, x, raw=None):
    cfg, prog, dram, out = C.canonical_program(layout, weights, scales, x)
    m = Machine(cfg, [prog if raw is None else prog[2:]], [dram])
    if raw is not None:
        sl = m.slices[0]
        sl.act[0, :layout.hidden] = np.frombuffer(raw.values, np.int8)
        sl.ascale[0, :layout.kb] = F.from_bits(raw.scale_bits)
        sl.adoff[0, :layout.kb] = np.frombuffer(raw.offsets, np.uint8).reshape(-1, 4)
    m.run()
    return m.slices[0].dram[out:out + 4 * layout.vocab].view(np.uint32).copy()


def _isa_argmax(bits):
    bits = np.asarray(bits, np.uint32)
    cfg = Config(S=1, D=128, MCOLS=4, DRAM_BYTES=4096)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:bits.nbytes] = bits.view(np.uint8)
    prog = [I.ld(0, 0, len(bits)), I.argmax(1024, 0, 1, len(bits)), I.halt()]
    sl = Machine(cfg, [prog], [dram]).run().slices[0]
    pair = sl.tget(np.array([1024, 1025]))
    return C.Winner(int(F.bits(pair[0])), int(pair[1]))


def test_production_layout_and_uneven_tail_mapping():
    prod = C.Layout()
    assert prod.pitch == 1184 and sum(prod.pc_sizes) == 310378496
    assert prod.locate(0) == (0, 0, 0)
    assert prod.locate(65535) == (0, 3, 16383 * 1184)
    assert prod.locate(65536) == (1, 4, 0)
    assert prod.locate(262143) == (3, 15, 16383 * 1184)
    layout = C.Layout(vocab=19, hidden=256, bases=tuple(32 * i for i in range(16)))
    locations = [layout.locate(i) for i in range(layout.vocab)]
    assert len(set(locations)) == layout.vocab
    assert sum(layout.pc_sizes) == layout.vocab * layout.pitch
    for token, (e, pc, address) in enumerate(locations):
        assert token in layout.row_ids(e) and pc // 4 == e
        assert layout.bases[pc] <= address < layout.bases[pc] + layout.pc_sizes[pc]


@pytest.mark.parametrize("kwargs", [dict(vocab=0), dict(vocab=True), dict(hidden=1153),
    dict(hidden=128 * 29), dict(version=2), dict(bases=(0,) * 15),
    dict(bases=(1,) * 16), dict(limits=(1,) * 16), dict(bases=(-32,) * 16)])
def test_descriptor_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        C.Layout(**kwargs)


@pytest.mark.parametrize("token", [-1, 262144, 1.5, True])
def test_token_bounds_do_not_wrap(token):
    with pytest.raises(ValueError):
        C.Layout().locate(token)


def test_bursts_never_cross_page_or_controller_limit():
    layout = C.Layout(vocab=97, bases=(4064,) * 16)
    for token in range(layout.vocab):
        for max_beats in (1, 4, 16, 256):
            bursts = layout.bursts(token, max_beats)
            assert sum(n for _, _, n in bursts) == layout.pitch
            expected = layout.locate(token)[2]
            for pc, address, n in bursts:
                assert address == expected and address % 32 == n % 32 == 0
                assert address // 4096 == (address + n - 1) // 4096
                assert n <= max_beats * 32 and pc == layout.locate(token)[1]
                expected += n


def test_sidecar_roundtrip_preserves_every_byte_and_tied_weights():
    layout, w, s, _, x = _case()
    # Preserve the canonical padding as well as semantically active bytes.
    s[:, 4 + layout.kb:] = 0xA5
    sidecar = C.Sidecar.build(layout, w, s, embedding_weights=w, embedding_scales=s)
    for i in range(layout.vocab):
        row = sidecar.row(i)
        np.testing.assert_array_equal(row[:layout.scale_bytes], s[i])
        np.testing.assert_array_equal(row[C.HEADER:], w[i])
        assert not row[layout.scale_bytes:C.HEADER].any() and not row.flags.writeable
    restored = C.Sidecar.from_descriptor(json.loads(json.dumps(sidecar.descriptor())), sidecar.pcs)
    restored.verify_tied(w, s, w, s)
    w[0, 0] ^= 1
    with pytest.raises(ValueError, match="stale"):
        restored.verify_tied(w, s, w, s)
    with pytest.raises(ValueError, match="tied"):
        C.Sidecar.build(layout, w, s, embedding_weights=w ^ 1, embedding_scales=s)


def test_payload_corruption_cannot_be_covered_by_changing_only_payload_hash():
    _, w, s, sidecar, _ = _case(vocab=5)
    pcs = list(sidecar.pcs)
    p = bytearray(pcs[0])
    p[C.HEADER] ^= 1
    pcs[0] = bytes(p)
    with pytest.raises(ValueError, match="payload checksum"):
        C.Sidecar.from_descriptor(sidecar.descriptor(), pcs)
    changed = replace(sidecar, pcs=tuple(pcs))
    with pytest.raises(ValueError, match="source checksum"):
        C.Sidecar.from_descriptor(changed.descriptor(), pcs)
    p = bytearray(sidecar.pcs[0])
    p[31] = 1
    with pytest.raises(ValueError, match="padding"):
        replace(sidecar, pcs=(bytes(p),) + sidecar.pcs[1:]).verify_payload()


@pytest.mark.parametrize("hidden", [128, 256, 1152, 3584])
@pytest.mark.parametrize("seed", [1, 99])
def test_exact_logits_match_canonical_isa_with_dropped_terms(hidden, seed):
    layout, w, scales, sidecar, x = _case(hidden=hidden, seed=seed)
    result = C.execute(sidecar, C.Activation.quantize(x), np.random.default_rng(seed).permutation(layout.vocab))
    expected = _isa(layout, w, scales, x)
    np.testing.assert_array_equal(result.logits_bits, expected)
    assert result.winner == _isa_argmax(expected)
    assert sum(work.dropped_groups for work in result.work) > 0
    assert sum(work.rows for work in result.work) == layout.vocab
    assert sum(work.bytes for work in result.work) == layout.vocab * layout.pitch


@pytest.mark.parametrize("delta", [1, 3, -1, -3])
def test_i2f48_rounding_boundary_and_negative_accumulation(delta):
    layout = C.Layout(vocab=4, hidden=128)
    w = np.zeros((4, 128), np.int8)
    sign = 1 if delta > 0 else -1
    w[:, :3] = np.array([127, 127, 2]) * sign
    w[:, 32] = delta
    scales = Q.pack_scales("int8", np.ones(4, np.float32), dw=np.zeros((4, 1), np.uint8))
    raw = C.Activation(np.ones(128, np.int8).tobytes(), int(F.bits(np.float32(65536))), bytes([0, 16, 31, 31]))
    sidecar = C.Sidecar.build(layout, w.view(np.uint8), scales,
                             embedding_weights=w.view(np.uint8), embedding_scales=scales)
    result = C.execute(sidecar, raw)
    expected = _isa(layout, w.view(np.uint8), scales, np.zeros(128), raw=raw)
    np.testing.assert_array_equal(result.logits_bits, expected)
    np.testing.assert_array_equal(result.logits_bits, np.repeat(F.bits(F.i2f(sign * (1 << 24) + delta)), 4))


@pytest.mark.parametrize("sa", [0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC00000])
def test_epilogue_exceptional_and_flushed_values(sa):
    layout, w, scales, _, _ = _case(vocab=8, hidden=128)
    scales[:, :4] = np.array([0, 0x80000000, 1, 0x00800000, 0x3F800000,
                              0x7F800000, 0xFF800000, 0x7FC00000], np.uint32).view(np.uint8).reshape(8, 4)
    raw = C.Activation(bytes([1]) * 128, sa, bytes(4))
    sidecar = C.Sidecar.build(layout, w, scales, embedding_weights=w, embedding_scales=scales)
    np.testing.assert_array_equal(C.execute(sidecar, raw).logits_bits,
                                   _isa(layout, w, scales, np.zeros(128), raw=raw))


def test_shard_reduction_matches_isa_for_bits_ties_and_all_completion_orders():
    layout = C.Layout(vocab=16, hidden=128)
    patterns = [
        [0x3F800000] * 16,
        [0x80000000, 0, 0x80000001, 1] * 4,
        [0x7F800001, 0x7FC00000, 0xFFC00000, 0x7F800000] * 4,
        [0xFF800000] * 15 + [0x3F800000],
    ]
    rng = np.random.default_rng(77)
    patterns += [rng.integers(0, 1 << 32, 16, dtype=np.uint32) for _ in range(20)]
    for bits in patterns:
        expected = _isa_argmax(bits)
        local = []
        for e in range(4):
            best = None
            for i in reversed(layout.row_ids(e)):
                best = C.better(best, C.Winner(int(bits[i]), i))
            local.append((e, best))
        for order in itertools.permutations(local):
            assert C.reduce_shards(layout, order) == expected


def test_missing_duplicate_and_foreign_completions_are_rejected():
    layout, _, _, sidecar, x = _case(vocab=5)
    result = C.execute(sidecar, C.Activation.quantize(x))
    pairs = [(e, w) for e, w in enumerate(result.shards) if w is not None]
    for invalid in (pairs[:-1], pairs + pairs[:1], [(0, C.Winner(0, 4))] + pairs[1:]):
        with pytest.raises(ValueError):
            C.reduce_shards(layout, invalid)
    for order in ([0] * 5, range(4), [0, 1, 2, 3, -1], [0, 1, 2, 3, 5]):
        with pytest.raises(ValueError):
            C.execute(sidecar, C.Activation.quantize(x), order)


def test_dispatch_is_disabled_without_explicit_matching_capability_and_policy():
    layout = C.Layout()
    assert C.fallback_reasons(layout, model_type="gemma3_text") == ("C3 capability version",)
    assert not C.fallback_reasons(layout, model_type="gemma3_text", capability_version=1)
    options = dict(model_type="gemma4", tied=False, head_format="fp4", greedy=False,
                   repetition_penalty=1.1, keep_logits=True, capability_version=2)
    assert len(C.fallback_reasons(C.Layout(vocab=16), **options)) == 8
    for field, value in (("head_format", "fp4"), ("greedy", False), ("repetition_penalty", float("nan")),
                         ("keep_logits", True), ("tied", False), ("capability_version", None),
                         ("capability_version", True)):
        options = dict(model_type="gemma3_text", capability_version=1)
        options[field] = value
        assert C.fallback_reasons(layout, **options)


def test_canonical_oracle_crosses_multiple_mm_chunks():
    layout, w, scales, sidecar, x = _case(vocab=8201, hidden=128)
    result = C.execute(sidecar, C.Activation.quantize(x))
    np.testing.assert_array_equal(result.logits_bits, _isa(layout, w, scales, x))


def test_activation_and_input_shapes_are_checked_before_execution():
    layout, w, scales, sidecar, x = _case(hidden=256)
    for values, offsets in ((bytes(255), bytes(8)), (bytes(256), bytes(4)),
                            (bytes(256), bytes([32]) * 8)):
        with pytest.raises(ValueError):
            C.Activation(values, 0, offsets)
    with pytest.raises(ValueError, match="hidden"):
        C.execute(sidecar, C.Activation.quantize(x[:128]))
    with pytest.raises(ValueError, match="weights"):
        C.Sidecar.build(layout, w.astype(np.float32), scales,
                        embedding_weights=w, embedding_scales=scales)
    bad = scales.copy()
    bad[0, 4] = 32
    with pytest.raises(ValueError, match="offsets"):
        C.Sidecar.build(layout, w, bad, embedding_weights=w, embedding_scales=bad)


def test_lens_export_is_driven_by_executed_work_and_labels_analytic_timing(tmp_path):
    layout, w, s, sidecar, x = _case(vocab=19)
    run = C.execute(sidecar, C.Activation.quantize(x), np.random.default_rng(3).permutation(layout.vocab))
    model = C.lens_profile(run)
    assert model["kind"] == "model" and "NOT RTL" in model["name"]
    assert model["bytes"] == sum(map(len, sidecar.pcs))
    assert model["macs"] == layout.vocab * layout.hidden
    assert not model["c3"]["assumptions"]["HBM_latency_or_contention_modeled"]
    assert all(not sl["buckets"] for sl in model["slices"])
    cfg, prog, dram, _ = C.canonical_program(layout, w, s, x)
    canonical = lens.isa_data("canonical ISA head, analytic timing", cfg, [prog], [dram])
    path = lens.save([canonical, model], tmp_path / "c3.otpuprof")
    loaded = lens.load(path)
    assert loaded["profiles"][1]["cycles"] == model["cycles"]
    assert "NOT RTL" in lens.render_summary(model)
    assert "C3 executed reference" in lens.render(loaded)
    assert C.lens_profile(run, read_bytes_per_cycle=64)["cycles"] > model["cycles"]
    assert C.lens_profile(run, epilogue_cycles=16)["cycles"] > model["cycles"]
