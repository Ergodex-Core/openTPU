"""Paged temporal ISA checks; run on the CPU ISA builder, never on Mac."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_temporal import Engine, Spec
from opentpu.llm.personaplex_temporal_pages import TemporalPagedImage


def _fixture():
    spec = Spec(hidden=64, layers=2, n_heads=2, ffn=128, context=8)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 23)
    rng = np.random.default_rng(412)
    weights = []
    for _ in range(spec.layers):
        layer = {name: rng.normal(0, scale, shape).astype(np.float32)
                 for name, scale, shape in (
                     ("wq", 0.18, (64, 64)), ("wk", 0.18, (64, 64)),
                     ("wv", 0.08, (64, 64)), ("wo", 0.08, (64, 64)),
                     ("wg", 0.08, (128, 64)), ("wu", 0.08, (128, 64)),
                     ("wd", 0.08, (64, 128)))}
        layer["norm1"] = rng.uniform(0.8, 1.2, 64).astype(np.float32)
        layer["norm2"] = rng.uniform(0.8, 1.2, 64).astype(np.float32)
        weights.append(layer)
    norm = rng.uniform(0.8, 1.2, 64).astype(np.float32)
    x = rng.normal(0, 1, (3, 64)).astype(np.float32)
    image = TemporalPagedImage(spec, cfg, cap=3,
        pages=((0, 1, "int8"), (1, 2, "int8")),
        kv_at=0x10000, x_at=0x20000, frame_at=0x24000, page_at=0x24100,
        cos_at=0x24200, sin_at=0x24a00, zmask_at=0x25200,
        final_norm_at=0x25400, weight_page_base=0x40000,
        weight_page_limit=0x300000, program_base=0x300000,
        program_limit=0x400000)
    return spec, cfg, weights, norm, x, image


def _reference(spec, weights, final_norm, inputs):
    """Independent adjacent-pair causal transformer equations in float64."""
    value = np.asarray(inputs, np.float64)
    length, hidden = value.shape
    d = spec.head_dim
    angles = np.arange(length)[:, None, None] * spec.rope_base ** (
        -np.arange(0, d, 2) / d)
    cos, sin = np.cos(angles), np.sin(angles)

    def norm(x, gain):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + spec.eps) * gain

    def rotate(x):
        result = np.empty_like(x)
        even, odd = x[..., 0::2], x[..., 1::2]
        result[..., 0::2] = even * cos - odd * sin
        result[..., 1::2] = odd * cos + even * sin
        return result

    causal = np.where(np.arange(length)[None, :] <= np.arange(length)[:, None],
                      0.0, -np.inf)
    for layer in weights:
        h = norm(value, layer["norm1"])
        q = rotate((h @ layer["wq"].T).reshape(length, spec.n_heads, d))
        k = rotate((h @ layer["wk"].T).reshape(length, spec.n_heads, d))
        v = (h @ layer["wv"].T).reshape(length, spec.n_heads, d)
        attended = np.empty_like(v)
        for head in range(spec.n_heads):
            scores = q[:, head] @ k[:, head].T / np.sqrt(d) + causal
            probability = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probability /= np.sum(probability, axis=-1, keepdims=True)
            attended[:, head] = probability @ v[:, head]
        value = value + attended.reshape(length, hidden) @ layer["wo"].T
        h = norm(value, layer["norm2"])
        gate = h @ layer["wg"].T
        up = h @ layer["wu"].T
        value = value + ((gate / (1.0 + np.exp(-gate))) * up) @ layer["wd"].T
    return norm(value, final_norm)


def _read_f32(backend, address, count):
    return backend.read(0, address, 4 * count).view(np.float32).copy()


def _kv_bytes(backend, source_image, layer, head, name, size):
    block = source_image.bofs[source_image.lf[layer]]["kvs"][0][head]
    return backend.read(0, source_image._off(layer).const + block[name], size)


def test_two_page_isa_matches_unpaged_and_independent_reference(monkeypatch):
    spec, cfg, weights, norm, inputs, image = _fixture()
    programs = image.compile()
    assert len(programs) == 2 and all(not p.run_args for p in programs)
    packets = [image.pack_layer(i, weights[i]) for i in range(2)]
    assert all(len(packet) == image.pages[i].weight_bytes for i, packet in enumerate(packets))
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload_static(backend, norm)
    reference = _reference(spec, weights, norm, inputs)

    with Engine(spec, weights, norm, cfg=cfg, cap=3, wformat="int8") as unpaged:
        def stream():
            outputs = []
            for frame, source in enumerate(inputs):
                backend.write(0, image.io["x"], source.reshape(1, -1))
                expected = unpaged.step(source)
                original_read, original_write = backend.read, backend.write
                transfers = []

                def read(*args):
                    transfers.append(("read", args[1]))
                    return original_read(*args)

                def write(*args):
                    transfers.append(("write", args[1]))
                    return original_write(*args)

                with monkeypatch.context() as patch:
                    patch.setattr(backend, "read", read)
                    patch.setattr(backend, "write", write)
                    for page in range(2):
                        image.upload_page(backend, page, [packets[page]])
                        backend.run([programs[page]])
                assert transfers == [("write", image.weight_page_base)] * 2
                actual = _read_f32(backend, image.io["x"], spec.hidden)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                assert _read_f32(backend, image.io["frame"], 1)[0] == frame + 1
                assert _read_f32(backend, image.io["page"], 1)[0] == 0
                for layer in range(spec.layers):
                    page = image.pages[layer]
                    first = page.kv_offsets[0]["k"]
                    for head, fields in enumerate(page.kv_offsets):
                        for name, size in (("k", image.cap * spec.head_dim),
                                           ("ks", 4 * image.cap),
                                           ("vt", image.cap * spec.head_dim),
                                           ("vs", 4 * image.cap)):
                            at = image.kv_at + layer * image.kv_layer_bytes + fields[name] - first
                            observed = backend.read(0, at, size)
                            expected_kv = _kv_bytes(unpaged.backend, unpaged.image,
                                                    layer, head, name, size)
                            np.testing.assert_array_equal(observed, expected_kv)
                outputs.append(actual)
            return np.stack(outputs)

        first = stream()
        error = np.linalg.norm(first - reference) / np.linalg.norm(reference)
        assert error < 0.025
        image.reset(backend)
        unpaged.reset()
        second = stream()
        np.testing.assert_array_equal(first.view(np.uint32), second.view(np.uint32))
        with pytest.raises(RuntimeError, match="bound|outside"):
            backend.run([programs[0]])  # frame 3 is outside admitted context.


def test_page_order_layout_and_production_packet_budget():
    spec, cfg, weights, norm, _, image = _fixture()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload_static(backend, norm)
    image.upload_page(backend, 1, [image.pack_layer(1, weights[1])])
    with pytest.raises(RuntimeError, match="bound|outside"):
        backend.run([image.compile_page(1)])
    conflict = [dict(name="other", base=image.io["x"], bytes=128,
                     region="codec", kind="state")]
    with pytest.raises(MemoryError, match="overlap"):
        TemporalPagedImage(spec, cfg, cap=3, pages=image.plan,
            kv_at=image.kv_at, x_at=image.io["x"], frame_at=image.io["frame"],
            page_at=image.io["page"], cos_at=image.io["cos"], sin_at=image.io["sin"],
            zmask_at=image.io["zmask"], final_norm_at=image.io["final_norm"],
            weight_page_base=image.weight_page_base,
            weight_page_limit=image.weight_page_limit, program_base=0x300000,
            program_limit=0x400000, prefix_spans=conflict)

    prod = Spec()
    from opentpu import target
    pcfg = target.config("f2-hbm:p4rl8a64c150")
    full = TemporalPagedImage(prod, pcfg, cap=128,
        kv_at=0x10000, x_at=0x02110000, frame_at=0x02114000,
        page_at=0x02114100, cos_at=0x02114200, sin_at=0x0211c200,
        zmask_at=0x02124200, final_norm_at=0x02124600,
        weight_page_base=1960284160)
    assert full.kv_bytes == 34603008
    assert [p.weight_bytes for p in full.pages] == [831782912, 884473856,
        884473856, 884473856, 331677696, 207945728]
    assert full.weight_page_limit - (full.weight_page_base + full.page_bytes) == 108032000
    assert len(full.program_slots) == 6
    # The former nine-layer INT4 packet does not fit the exact shared image.
    with pytest.raises(MemoryError, match="largest temporal weight page"):
        TemporalPagedImage(prod, pcfg, cap=128,
            pages=((0,4,"int8"),(4,13,"int4"),(13,22,"int4"),
                   (22,31,"int4"),(31,32,"int8")),
            kv_at=0x10000, x_at=0x02110000, frame_at=0x02114000,
            page_at=0x02114100, cos_at=0x02114200, sin_at=0x0211c200,
            zmask_at=0x02124200, final_norm_at=0x02124600,
            weight_page_base=1960284160)


def test_exact_shared_profile_compiles_every_production_page():
    from opentpu import target
    from opentpu.llm.personaplex_duplex_layout import DuplexLayout
    layout = DuplexLayout.production()
    assert target.arch_hash(layout.cfg) == 0xdd6b92a6
    assert (layout.cfg.D, layout.cfg.IMEM_WORDS) == (128, 32768)
    programs = layout.temporal.compile()
    assert len(programs) == 6
    assert all(not program.run_args for program in programs)
    assert all(0 < len(program) * 8 <= layout.cfg.IMEM_WORDS for program in programs)


@pytest.mark.parametrize("weight_format", ["int4", "int8"])
def test_repeated_layer_page_matches_unpaged_across_frames_and_reset(weight_format):
    spec, cfg, weights, norm, inputs, template = _fixture()
    io = template.io
    image = TemporalPagedImage(spec, cfg, cap=3,
        pages=((0, 2, weight_format),), kv_at=template.kv_at,
        x_at=io["x"], frame_at=io["frame"], page_at=io["page"],
        cos_at=io["cos"], sin_at=io["sin"], zmask_at=io["zmask"],
        final_norm_at=io["final_norm"], weight_page_base=template.weight_page_base,
        weight_page_limit=template.weight_page_limit,
        program_base=0x300000, program_limit=0x400000)
    payloads = [image.pack_layer(0, layer) for layer in weights]
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload_static(backend, norm)
    program = image.compile_page(0)
    with Engine(spec, weights, norm, cfg=cfg, cap=3, wformat=weight_format) as unpaged:
        for _ in range(2):
            image.reset(backend)
            unpaged.reset()
            for source in inputs:
                backend.write(0, io["x"], source)
                image.upload_page(backend, 0, payloads)
                backend.run([program])
                expected = unpaged.step(source)
                observed = _read_f32(backend, io["x"], spec.hidden)
                np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
