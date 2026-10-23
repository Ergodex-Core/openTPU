"""Synthetic native/ISA tests for resident Mimi upsample and two-token decoder.

Execute on the CPU EC2 runner only. No checkpoint, PCM, or physical board is used.
"""
import numpy as np
import pytest

from opentpu import language as ol
from opentpu.compiler import Tensor, trace_fit
from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_depth_chain import DepthChainEngine, DepthChainSpec
from opentpu.llm.personaplex_mimi import MimiRVQSpec
from opentpu.llm.personaplex_mimi_decoder import (
    MimiDecoderImage, MimiDecoderSpec, _gelu, native_ring_positions, offline_rope,
)


def tiny(*, context=4, max_frames=4):
    spec = MimiDecoderSpec(hidden=64, n_heads=2, ffn=128, layers=1,
                           context=context, max_frames=max_frames)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 25)
    rng = np.random.default_rng(2182)
    upsample = rng.normal(0, 0.4, (64, 4)).astype(np.float32)
    layers = []
    for _ in range(spec.layers):
        layer = {name: rng.normal(0, 0.025, shape).astype(np.float32)
                 for name, shape in (("wq", (64, 64)), ("wk", (64, 64)),
                                     ("wv", (64, 64)), ("wo", (64, 64)),
                                     ("w1", (128, 64)), ("w2", (64, 128)))}
        layer.update(norm1_weight=rng.uniform(0.8, 1.2, 64).astype(np.float32),
                     norm1_bias=rng.normal(0, 0.04, 64).astype(np.float32),
                     norm2_weight=rng.uniform(0.8, 1.2, 64).astype(np.float32),
                     norm2_bias=rng.normal(0, 0.04, 64).astype(np.float32),
                     scale1=np.full(64, 0.7, np.float32),
                     scale2=np.full(64, 0.7, np.float32))
        layers.append(layer)
    latents = rng.normal(0, 0.8, (max_frames, 64)).astype(np.float32)
    return spec, cfg, upsample, layers, latents


def native_modules(spec, upsample_weight, layers):
    """Strictly populate original installed Moshi modules with synthetic weights."""
    import torch
    from moshi.modules.resample import ConvTrUpsample1d
    from moshi.modules.transformer import ProjectedTransformer

    up = ConvTrUpsample1d(2, dimension=spec.hidden, causal=True,
                          learnt=True, channel_wise=True)
    up.load_state_dict({"convtr.convtr.convtr.weight":
                        torch.from_numpy(upsample_weight[:, None, :].copy())}, strict=True)
    tr = ProjectedTransformer(
        input_dimension=spec.hidden, output_dimensions=(spec.hidden,),
        d_model=spec.hidden, num_heads=spec.n_heads, num_layers=spec.layers,
        causal=True, layer_scale=0.01, context=spec.context, conv_layout=True,
        max_period=spec.rope_base, gating="none", norm="layer_norm",
        positional_embedding="rope", dim_feedforward=spec.ffn, device="cpu")
    state = {}
    for index, layer in enumerate(layers):
        prefix = f"transformer.layers.{index}."
        for key, source in (("norm1.weight", "norm1_weight"),
                            ("norm1.bias", "norm1_bias"),
                            ("norm2.weight", "norm2_weight"),
                            ("norm2.bias", "norm2_bias"),
                            ("layer_scale_1.scale", "scale1"),
                            ("layer_scale_2.scale", "scale2"),
                            ("self_attn.out_proj.weight", "wo"),
                            ("linear1.weight", "w1"), ("linear2.weight", "w2")):
            state[prefix + key] = torch.from_numpy(layer[source].copy())
        state[prefix + "self_attn.in_proj_weight"] = torch.from_numpy(
            np.concatenate((layer["wq"], layer["wk"], layer["wv"])).copy())
    tr.load_state_dict(state, strict=True)
    up.eval(); tr.eval()
    return up, tr


def native_frames(spec, upsample_weight, layers, latents, *, tokenwise=False):
    import torch
    up, tr = native_modules(spec, upsample_weight, layers)
    frames = []
    captures = [[] for _ in range(spec.layers)]
    handles = []
    for index, module in enumerate(tr.transformer.layers):
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, value, index=index:
            captures[index].append(value.detach().numpy()[0].copy())))
    try:
        with torch.no_grad(), up.convtr.streaming(1), tr.streaming(1):
            for latent in latents:
                source = torch.from_numpy(latent.reshape(1, spec.hidden, 1).copy())
                u = up(source)
                if tokenwise:
                    parts = [tr(u[:, :, t:t + 1])[0] for t in range(2)]
                    output = torch.cat(parts, dim=2)
                else:
                    output = tr(u)[0]
                frames.append((u.detach().numpy()[0].T.copy(),
                               output.detach().numpy()[0].T.copy()))
    finally:
        for handle in handles:
            handle.remove()
    if tokenwise:
        captures = [list(np.concatenate(items[i:i + 2], axis=0)
                         for i in range(0, len(items), 2)) for items in captures]
    return frames, captures


def direct_image(spec, cfg, upsample_weight, layers):
    image = MimiDecoderImage(spec, cfg, latent_at=0x1000, state_base=0x2000,
                             state_limit=0x200000, weight_base=0x200000,
                             weight_limit=0x1000000)
    programs = image.compile()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, upsample_weight, layers)
    backend.write(0, image.io["state"], np.zeros(1, np.float32))
    return image, programs, backend


def read_floats(backend, at, shape):
    return np.asarray(backend.read(0, at, 4 * int(np.prod(shape))),
                      np.uint8).view(np.float32).reshape(shape).copy()


def run_direct(image, programs, backend, latents):
    observations = []
    for index, latent in enumerate(latents):
        backend.write(0, image.latent_at, latent)
        for program in programs:
            backend.run([program])
        observations.append((read_floats(backend, image.io["upsampled"], (2, image.spec.hidden)),
                             read_floats(backend, image.io["output"], (2, image.spec.hidden)),
                             [read_floats(backend, at, (2, image.spec.hidden))
                              for at in image.layer_outputs],
                             [np.asarray(backend.read(0, at, size), np.uint8).copy()
                              for at, size in image.cache_regions]))
        assert read_floats(backend, image.io["state"], (1,))[0] == index + 1
    return observations


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def test_native_two_token_frames_match_isa_and_reset_without_kv_scrub():
    spec, cfg, upsample, layers, latents = tiny()
    expected, native_layers = native_frames(spec, upsample, layers, latents)
    image, programs, backend = direct_image(spec, cfg, upsample, layers)
    assert len(programs) == 1 + spec.layers and all(not p.run_args for p in programs)
    actual = run_direct(image, programs, backend, latents)
    for frame, ((want_up, want_out), (got_up, got_out, layer_out, _)) in enumerate(zip(expected, actual)):
        np.testing.assert_allclose(got_up, want_up, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got_out, want_out, rtol=0.06, atol=0.06)
        for index in range(spec.layers):
            np.testing.assert_allclose(layer_out[index], native_layers[index][frame],
                                       rtol=0.06, atol=0.06)
    # Native reset discards validity/offsets. Device reset does not erase KV; its
    # source-compatible masks must hide stale bytes and reproduce every frame.
    backend.write(0, image.io["state"], np.zeros(1, np.float32))
    backend.write(0, image.io["overlap"], np.full((2, spec.hidden), -0.0, np.float32))
    replay = run_direct(image, programs, backend, latents)
    for frame, (before, after) in enumerate(zip(actual, replay)):
        for x, y in zip(before[:2], after[:2]):
            np.testing.assert_array_equal(bits(x), bits(y))
        for x, y in zip(before[2], after[2]):
            np.testing.assert_array_equal(bits(x), bits(y))
        # Reset leaves stale physical KV in masked slots. Once a whole context
        # is overwritten, the complete raw cache is bit-identical too.
        if 2 * (frame + 1) >= spec.context:
            for x, y in zip(before[3], after[3]):
                np.testing.assert_array_equal(x, y)


def test_native_chunk_two_differs_from_tokenwise_at_first_wrap():
    spec, cfg, upsample, layers, latents = tiny()
    # Uniform attention makes the physical ring's two writes-before-attend
    # observable independently of QK quantization or RoPE rounding.
    layer = layers[0]
    layer["wq"].fill(0); layer["wk"].fill(0)
    layer["wv"][:] = np.eye(spec.hidden, dtype=np.float32)
    layer["wo"][:] = np.eye(spec.hidden, dtype=np.float32)
    layer["w1"].fill(0); layer["w2"].fill(0)
    layer["scale1"].fill(1)
    native, _ = native_frames(spec, upsample, layers, latents)
    tokenwise, _ = native_frames(spec, upsample, layers, latents, tokenwise=True)
    gap = np.linalg.norm(native[2][1] - tokenwise[2][1])
    assert gap > 0.02, "the wrap probe must discriminate chunk-two from tokenwise updates"
    image, programs, backend = direct_image(spec, cfg, upsample, layers)
    actual = run_direct(image, programs, backend, latents)
    chunk_error = np.linalg.norm(actual[2][1] - native[2][1])
    tokenwise_error = np.linalg.norm(actual[2][1] - tokenwise[2][1])
    assert chunk_error < tokenwise_error
    # The pinned source has a delta<=0 ring-position defect at the full boundary.
    np.testing.assert_array_equal(native_ring_positions(4, 4), [4, 1, 2, 3])
    np.testing.assert_array_equal(native_ring_positions(6, 4), [4, 5, 6, 3])


@ol.jit
def gelu_probe(x, y):
    ol.store(y, _gelu(ol.load(x)))


def test_erf_form_gelu_lowering_matches_native_scalar_reference():
    import torch
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 20)
    source = Tensor(0, (1, 32), (32, 1))
    target = Tensor(4096, (1, 32), (32, 1))
    program = trace_fit(gelu_probe, cfg, 0, lambda: {"x": source, "y": target}).finish()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    values = np.linspace(-10, 10, 32, dtype=np.float32).reshape(1, 32)
    backend.write(0, 0, values)
    backend.run([program])
    actual = read_floats(backend, 4096, values.shape)
    expected = torch.nn.functional.gelu(torch.from_numpy(values), approximate="none").numpy()
    assert np.max(np.abs(actual - expected)) < 1e-5


def test_decoder_rope_uses_two_adjacent_pair_positions_and_image_preflights():
    spec, cfg, upsample, layers, _ = tiny()
    cos, sin = offline_rope(spec)
    assert cos.shape == sin.shape == (spec.max_frames, 2, spec.head_dim // 2)
    np.testing.assert_array_equal(cos[0, 0], np.ones(spec.head_dim // 2, np.float32))
    np.testing.assert_array_equal(sin[0, 0], np.zeros(spec.head_dim // 2, np.float32))
    assert not np.array_equal(bits(cos[0, 1]), bits(cos[0, 0]))
    image = MimiDecoderImage(spec, cfg, latent_at=0x1000, state_base=0x2000,
                             state_limit=0x200000, weight_base=0x200000,
                             weight_limit=0x1000000)
    budget = image.budget()
    assert budget["required_bytes"] <= cfg.DRAM_BYTES
    assert budget["program_words"] and len(image.compile()) == 1 + spec.layers
    with pytest.raises(MemoryError):
        MimiDecoderImage(spec, cfg, latent_at=0x1000, state_base=0x2000,
                         state_limit=0x2100, weight_base=0x200000,
                         weight_limit=0x1000000)


def chain_fixture():
    chain = DepthChainSpec(hidden=64, depth=64, n_heads=2, ffn=128,
                           steps=2, layers=1, text_vocab=3,
                           audio_vocab=4, head_vocab=4)
    mimi = MimiRVQSpec(codebooks=2, vocab=4, dim=64, latent=64)
    decoder = MimiDecoderSpec(hidden=64, n_heads=2, ffn=128, layers=1,
                              context=4, max_frames=2)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 32)
    zero = lambda shape: np.zeros(shape, np.float32)
    bank = {name: zero((64, 64)) for name in ("wq", "wk", "wv", "wo")}
    bank.update(wg=zero((128, 64)), wu=zero((128, 64)), wd=zero((64, 128)))
    depth_layers = [{"norm1": np.ones(64, np.float32),
                     "norm2": np.ones(64, np.float32),
                     "steps": [{key: value.copy() for key, value in bank.items()}
                               for _ in range(chain.steps)]}]
    decoder_layers = tiny(max_frames=2)[3]
    return chain, mimi, decoder, cfg, dict(
        projections=zero((2, 64, 64)),
        embeddings=(zero((3, 64)), zero((4, 64))),
        layers=depth_layers, head_bank=zero((2, 4, 64)),
        mimi_tables=zero((2, 4, 64)), mimi_projections=zero((2, 64, 64)),
        upsample_weight=zero((64, 4)), decoder_weights=decoder_layers)


def test_integrated_decoder_has_no_interstage_host_transfers_and_rejects_abandoned_frame(monkeypatch):
    chain, mimi, decoder, cfg, payload = chain_fixture()
    with DepthChainEngine(payload["projections"], payload["embeddings"], payload["layers"],
                          payload["head_bank"], spec=chain, cfg=cfg,
                          mimi_tables=payload["mimi_tables"],
                          mimi_projections=payload["mimi_projections"], mimi_spec=mimi,
                          upsample_weight=payload["upsample_weight"],
                          decoder_weights=payload["decoder_weights"],
                          decoder_spec=decoder) as engine:
        hidden = np.zeros(chain.hidden, np.float32)
        with pytest.raises(RuntimeError):
            engine.run_decoder()
        engine.run_frame(hidden, 0)
        engine.run_mimi()
        with pytest.raises(RuntimeError, match="pending decoder"):
            engine.begin_frame(hidden, 0)
        read, write, run = engine.backend.read, engine.backend.write, engine.backend.run
        transfers, programs = [], []
        def traced_read(*args):
            transfers.append("read")
            return read(*args)
        def traced_write(*args):
            transfers.append("write")
            return write(*args)
        def traced_run(program, *args, **kwargs):
            assert not args and kwargs.get("args") is None
            assert len(program) == 1 and not program[0].run_args
            programs.append(program[0])
            return run(program, *args, **kwargs)
        monkeypatch.setattr(engine.backend, "read", traced_read)
        monkeypatch.setattr(engine.backend, "write", traced_write)
        monkeypatch.setattr(engine.backend, "run", traced_run)
        assert engine.run_decoder() is None
        assert not transfers and len(programs) == 1 + decoder.layers
        assert engine.decoder_pos == 1
        assert engine.read_decoder().shape == (2, decoder.hidden)
        assert engine.read_upsampled().shape == (2, decoder.hidden)
        engine.run_frame(hidden, 0)
        engine.run_mimi(); engine.run_decoder()
        assert engine.decoder_pos == 2
        with pytest.raises(RuntimeError, match="limit"):
            engine.begin_frame(hidden, 0)
        engine.reset()
        assert engine.decoder_pos == 0
        with pytest.raises(RuntimeError):
            engine.read_decoder()
