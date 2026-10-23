"""Resident two-row Mimi input-transformer checks for the EC2 CPU ISA runner.

The input rows come from the tiny source upsampler equations only as synthetic
test data. The device executes a copy followed by its own FP32 transformer;
the comparison uses the independent NumPy equations from the decoder tests.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_mimi_input_transformer import MimiInputTransformerImage
from test_personaplex_decoder_precision import numpy_frames
from test_personaplex_mimi_decoder import bits, read_floats, tiny


def _resident_cache(image, backend, layer):
    k, vt = [], []
    for head in range(image.spec.n_heads):
        addresses = image.transformer.kv_heads[layer][head]
        assert set(addresses) == {"k", "vt"}
        k.append(read_floats(backend, addresses["k"],
                             (image.transformer.cap, image.spec.head_dim)))
        vt.append(read_floats(backend, addresses["vt"],
                              (image.spec.head_dim, image.transformer.cap)))
    return np.stack(k), np.stack(vt)


def _case(context, max_frames):
    original, cfg, upsample, layers, latents = tiny(context=context, max_frames=max_frames)
    spec = replace(original, precision="fp32")
    reference = numpy_frames(spec, upsample, layers, latents)
    # The wrapper starts at the native transformer's two-token input, after
    # the source causal upsampler. Round once to the real FP32 device boundary.
    inputs = np.stack([frame[0].astype(np.float32) for frame in reference])
    image = MimiInputTransformerImage(
        spec, cfg, input_at=0x1000, state_base=0x2000,
        state_limit=0x200000, weight_base=0x200000, weight_limit=0x1000000)
    programs = image.compile()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, layers)
    return spec, cfg, layers, inputs, reference, image, programs, backend


@pytest.mark.parametrize("context,max_frames", [(4, 4), (40, 24)])
def test_input_rows_transform_on_isa_with_wrap_masked_reset_and_resident_weights(
        monkeypatch, context, max_frames):
    spec, _, layers, inputs, reference, image, programs, backend = _case(context, max_frames)
    assert image.spec == spec and image.cfg == image.transformer.cfg
    assert image.input_at == 0x1000
    assert len(programs) == 1 + spec.layers
    assert all(not program.run_args for program in programs)
    assert image.io["output"] == image.transformer.io["output"]
    assert image.io["upsampled"] == image.transformer.io["upsampled"]
    assert image.cache_regions == image.transformer.cache_regions
    assert image.layer_outputs == image.transformer.layer_outputs
    np.testing.assert_array_equal(
        np.asarray(backend.read(0, image.transformer.upsample_at,
                                16 * spec.hidden), np.uint8),
        np.zeros(16 * spec.hidden, np.uint8))

    # Uploaded coefficients are resident: changing every Python source tensor
    # after upload must not change the device's ensuing stream.
    for layer in layers:
        for value in layer.values():
            value.fill(0)
    image.reset(backend)
    observations = []
    for frame, source in enumerate(inputs):
        backend.write(0, image.input_at, source)
        transfers = []
        native_read, native_write, native_run = backend.read, backend.write, backend.run

        def record_read(*args):
            transfers.append("read")
            return native_read(*args)

        def record_write(*args):
            transfers.append("write")
            return native_write(*args)

        def record_run(batch, *args, **kwargs):
            assert len(batch) == 1 and not batch[0].run_args
            assert not args and kwargs.get("args") is None
            return native_run(batch, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(backend, "read", record_read)
            patch.setattr(backend, "write", record_write)
            patch.setattr(backend, "run", record_run)
            for program in programs:
                backend.run([program])
        assert transfers == [], "all input transformer intermediates must remain on device"

        wanted_up, wanted_out, wanted_layers, wanted_caches = reference[frame]
        snapshot = read_floats(backend, image.io["upsampled"], (2, spec.hidden))
        output = read_floats(backend, image.io["output"], (2, spec.hidden))
        np.testing.assert_array_equal(bits(snapshot), bits(source))
        np.testing.assert_allclose(snapshot, wanted_up, rtol=0, atol=2e-6)
        np.testing.assert_allclose(output, wanted_out, rtol=2e-4, atol=3e-4,
                                   err_msg=f"context {context}, frame {frame}")
        actual_layers = [read_floats(backend, at, (2, spec.hidden))
                         for at in image.layer_outputs]
        actual_caches = [_resident_cache(image, backend, layer)
                         for layer in range(spec.layers)]
        for layer in range(spec.layers):
            np.testing.assert_allclose(actual_layers[layer], wanted_layers[layer],
                                       rtol=2e-4, atol=3e-4)
            keys, values_t = actual_caches[layer]
            want_keys, want_values_t = wanted_caches[layer]
            np.testing.assert_allclose(keys[:, :context], want_keys, rtol=2e-4, atol=3e-4)
            np.testing.assert_allclose(values_t[:, :, :context], want_values_t,
                                       rtol=2e-4, atol=3e-4)
            np.testing.assert_array_equal(bits(keys[:, context:]),
                                          bits(np.zeros_like(keys[:, context:])))
            np.testing.assert_array_equal(bits(values_t[:, :, context:]),
                                          bits(np.zeros_like(values_t[:, :, context:])))
        assert read_floats(backend, image.io["state"], (1,))[0] == frame + 1
        observations.append((snapshot, output, actual_layers, actual_caches))

    # Native logical reset retains raw K/VT. First frames must replay exactly
    # despite stale physical slots, including the full-ring boundary defect.
    old_cache = [np.asarray(backend.read(0, at, size), np.uint8).copy()
                 for at, size in image.cache_regions]
    image.reset(backend)
    assert read_floats(backend, image.io["state"], (1,))[0] == 0
    for (at, size), before in zip(image.cache_regions, old_cache):
        np.testing.assert_array_equal(np.asarray(backend.read(0, at, size), np.uint8), before)
    for frame, source in enumerate(inputs):
        backend.write(0, image.input_at, source)
        for program in programs:
            backend.run([program])
        snapshot = read_floats(backend, image.io["upsampled"], (2, spec.hidden))
        output = read_floats(backend, image.io["output"], (2, spec.hidden))
        np.testing.assert_array_equal(bits(snapshot), bits(observations[frame][0]))
        np.testing.assert_array_equal(bits(output), bits(observations[frame][1]))
        for layer, at in enumerate(image.layer_outputs):
            actual = read_floats(backend, at, (2, spec.hidden))
            np.testing.assert_array_equal(bits(actual), bits(observations[frame][2][layer]))
        if 2 * (frame + 1) >= context:
            for layer in range(spec.layers):
                for before, after in zip(observations[frame][3][layer],
                                         _resident_cache(image, backend, layer)):
                    np.testing.assert_array_equal(bits(after), bits(before))


def test_after_encoder_layout_appends_state_weights_and_nine_program_slots():
    from opentpu import target, waits
    from opentpu.llm.personaplex_depth_chain import DepthChainImage, DepthChainSpec
    from opentpu.llm.personaplex_mimi import MimiRVQSpec
    from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
    from opentpu.llm.personaplex_mimi_encoder import MimiEncoderImage
    from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec

    cfg = target.config("f2-hbm:p4rl8a64c150")
    output = DepthChainImage(DepthChainSpec(), cfg,
        mimi_spec=MimiRVQSpec(projection_precision="fp32"),
        decoder_spec=MimiDecoderSpec(precision="fp32"),
        pcm_spec=MimiPCMSpec(initial_precision="fp32", transpose_precision="fp32",
                             residual_precision="fp32", final_precision="fp32"))
    encoder = MimiEncoderImage.after_output_codec(output)
    output_before = [dict(span) for span in output.spans]
    encoder_before = [dict(span) for span in encoder.spans]
    image = MimiInputTransformerImage.after_encoder(encoder)
    assert image.spec.precision == "fp32"
    assert image.input_at == encoder.io["output"]
    assert output.spans == output_before and encoder.spans == encoder_before
    assert len(image.compile()) == len(image.program_slots) == 9
    assert all(not program.run_args for program in image.compile())

    spans = sorted(output.spans + encoder.spans + image.transformer.spans,
                   key=lambda span: span["base"])
    assert all(a["base"] + a["bytes"] <= b["base"]
               for a, b in zip(spans, spans[1:]))
    state = [span for span in image.transformer.spans
             if span["region"] == "decoder_state"]
    weights = [span for span in image.transformer.spans
               if span["region"] == "decoder_weights"]
    assert state and weights
    assert min(span["base"] for span in state) >= encoder.state_end
    assert min(span["base"] for span in weights) >= encoder.weight_end
    assert max(span["base"] + span["bytes"] for span in state) <= encoder.input_at + (32 << 20)

    slot_bytes = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    slots = sorted(output.program_slots + encoder.program_slots + image.program_slots)
    assert all(a + slot_bytes <= b for a, b in zip(slots, slots[1:]))
    assert image.program_slots[0] >= encoder.program_slots[-1] + slot_bytes
    budget = image.budget()
    assert len(budget["program_words"]) == 9
    assert max(budget["program_words"]) <= cfg.IMEM_WORDS
