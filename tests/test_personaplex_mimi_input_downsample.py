"""Source-faithful learned Mimi downsample checks for the EC2 CPU ISA runner.

No checkpoint or physical target is needed for these synthetic tests.
"""
import os
os.environ["NO_TORCH_COMPILE"] = "1"

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_mimi_input_downsample import (
    MimiInputDownsampleImage, MimiInputDownsampleSpec)


def tiny():
    spec = MimiInputDownsampleSpec(hidden=32)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 25)
    image = MimiInputDownsampleImage(spec, cfg, input_at=0x1000,
        state_base=0x2000, state_limit=0x100000,
        weight_base=0x100000, weight_limit=0x1000000)
    rng = np.random.default_rng(52271)
    weight = rng.normal(0, 0.065, (spec.hidden, spec.hidden, 4)).astype(np.float32)
    frames = rng.normal(0, 0.75, (4, 2, spec.hidden)).astype(np.float32)
    frames[0, 0] += np.linspace(-0.5, 0.5, spec.hidden, dtype=np.float32)
    return spec, cfg, image, weight, frames


def device_image(image, cfg, weight):
    programs = image.compile()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, weight)
    return programs, backend


def read_floats(backend, address, shape):
    count = int(np.prod(shape))
    return np.asarray(backend.read(0, address, 4 * count), np.uint8).view(
        np.float32).reshape(shape).copy()


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def source_downsample(frames, weight, *, replicate=True):
    """Global Conv1d[out,in,tap] with source replicate-left padding."""
    rows = frames.reshape(-1, frames.shape[-1]).astype(np.float64)
    coeff = weight.astype(np.float64)
    output = np.empty((len(rows) // 2, coeff.shape[0]), np.float64)
    for global_output in range(len(output)):
        value = np.zeros(coeff.shape[0], np.float64)
        for tap in range(4):
            at = 2 * global_output + tap - 2
            if at < 0:
                if not replicate:
                    continue
                at = 0
            value += np.einsum("i,oi->o", rows[at], coeff[:, :, tap])
        output[global_output] = value
    return output.reshape(len(frames), 1, coeff.shape[0])


def synthetic_native(frames, weight):
    """Original Moshi module with only its one synthetic learned tensor."""
    import torch
    from moshi.modules.resample import ConvDownsample1d

    hidden = weight.shape[0]
    native = ConvDownsample1d(2, dimension=hidden, causal=True,
        learnt=True, channel_wise=False)
    native.load_state_dict({"conv.conv.conv.weight":
                            torch.from_numpy(weight.copy())}, strict=True)
    native.eval()
    output = []
    with torch.no_grad(), native.conv.streaming(1):
        for frame in frames:
            source = torch.from_numpy(frame.T.copy()[None, :, :])
            output.append(native(source).detach().numpy()[0].T.copy())
    return np.stack(output)


def test_first_replicate_padding_continuous_history_and_isa_reset():
    spec, cfg, image, weight, frames = tiny()
    expected = source_downsample(frames, weight)
    zero_padded = source_downsample(frames, weight, replicate=False)
    assert np.max(np.abs(expected[0] - zero_padded[0])) > 1e-3
    programs, backend = device_image(image, cfg, weight)
    assert len(programs) == 1 and not programs[0].run_args
    assert (image.op.rows, image.op.output_rows, image.op.kernel,
            image.op.stride, image.op.history) == (2, 1, 4, 2, 2)
    assert read_floats(backend, image.io["first"], (1,))[0] == 1
    np.testing.assert_array_equal(
        bits(read_floats(backend, image.op.prepared.base, (2, spec.hidden))),
        bits(np.zeros((2, spec.hidden), np.float32)))

    observed = []
    for index, frame in enumerate(frames):
        backend.write(0, image.input_at, frame)
        backend.run([programs[0]])
        actual = read_floats(backend, image.io["output"], (1, spec.hidden))
        observed.append(actual)
        np.testing.assert_allclose(actual, expected[index], rtol=0, atol=1e-5)
        assert read_floats(backend, image.io["frame"], (1,))[0] == index + 1
        assert read_floats(backend, image.io["first"], (1,))[0] == 0
        history = read_floats(backend, image.op.prepared.base, (2, spec.hidden))
        np.testing.assert_array_equal(bits(history), bits(frame))
    observed = np.stack(observed)

    state = [(address, np.asarray(backend.read(0, address, size), np.uint8).copy())
             for address, size in image.reset_regions]
    image.reset(backend)
    replay = []
    for frame in frames:
        backend.write(0, image.input_at, frame)
        backend.run([programs[0]])
        replay.append(read_floats(backend, image.io["output"], (1, spec.hidden)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(observed))
    for address, before in state:
        np.testing.assert_array_equal(
            np.asarray(backend.read(0, address, len(before)), np.uint8), before)


def test_optional_native_replicate_stream_matches_independent_source():
    pytest.importorskip("torch")
    pytest.importorskip("moshi.modules.resample")
    _, _, _, weight, frames = tiny()
    native = synthetic_native(frames, weight)
    expected = source_downsample(frames, weight)
    np.testing.assert_allclose(native, expected, rtol=0, atol=5e-6)


def test_fp32_weight_rows_are_native_tap_major_and_bias_is_exact_zero():
    spec, cfg, image, weight, _ = tiny()
    _, backend = device_image(image, cfg, weight)
    packed = read_floats(backend, image.op.fp_data, image.op.matrix_shape)
    for output in range(spec.hidden):
        for tap in range(4):
            np.testing.assert_array_equal(
                bits(packed[output, tap * spec.hidden:(tap + 1) * spec.hidden]),
                bits(weight[output, :, tap]))
    np.testing.assert_array_equal(
        bits(packed[:, 4 * spec.hidden:]),
        bits(np.zeros_like(packed[:, 4 * spec.hidden:])))
    np.testing.assert_array_equal(
        bits(read_floats(backend, image.op.bias.base, (spec.hidden,))),
        bits(np.zeros(spec.hidden, np.float32)))
    assert set(image.payload_hashes) == {"fp32", "zero_bias"}


def test_raw_downsample_has_no_clipping_scaling_or_learned_bias():
    spec, cfg, image, _, frames = tiny()
    weight = np.zeros((spec.hidden, spec.hidden, 4), np.float32)
    weight[:, :, 2] = np.eye(spec.hidden, dtype=np.float32)
    frames[0, 0, 0] = 2.5
    frames[1, 0, 1] = -3.25
    programs, backend = device_image(image, cfg, weight)
    for frame in frames:
        backend.write(0, image.input_at, frame)
        backend.run([programs[0]])
        actual = read_floats(backend, image.io["output"], (1, spec.hidden))
        np.testing.assert_array_equal(bits(actual), bits(frame[:1]))
    assert abs(frames[0, 0, 0]) > 1 and abs(frames[1, 0, 1]) > 1


def test_downsample_program_has_no_host_activation_feedback(monkeypatch):
    _, cfg, image, weight, frames = tiny()
    programs, backend = device_image(image, cfg, weight)
    backend.write(0, image.input_at, frames[0])
    read, write, run = backend.read, backend.write, backend.run
    transfers = []

    def traced_read(*args):
        transfers.append("read")
        return read(*args)

    def traced_write(*args):
        transfers.append("write")
        return write(*args)

    def traced_run(batch, *args, **kwargs):
        assert len(batch) == 1 and not batch[0].run_args
        assert not args and kwargs.get("args") is None
        return run(batch, *args, **kwargs)

    monkeypatch.setattr(backend, "read", traced_read)
    monkeypatch.setattr(backend, "write", traced_write)
    monkeypatch.setattr(backend, "run", traced_run)
    backend.run([programs[0]])
    assert transfers == []


def test_malformed_weight_fails_before_any_payload_write():
    spec, cfg, image, weight, _ = tiny()

    class Spy:
        writes = 0
        def write(self, *_args):
            self.writes += 1

    spy = Spy()
    with pytest.raises(ValueError):
        image.upload(spy, weight[:, :, :3])
    assert spy.writes == 0
    bad = weight.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        image.upload(spy, bad)
    assert spy.writes == 0


def test_after_transformer_uses_next_state_weights_and_single_program_slot():
    from opentpu import target, waits
    from opentpu.llm.personaplex_depth_chain import DepthChainImage, DepthChainSpec
    from opentpu.llm.personaplex_mimi import MimiRVQSpec
    from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
    from opentpu.llm.personaplex_mimi_encoder import MimiEncoderImage
    from opentpu.llm.personaplex_mimi_input_transformer import MimiInputTransformerImage
    from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec

    cfg = target.config("f2-hbm:p4rl8a64c150")
    output = DepthChainImage(DepthChainSpec(), cfg,
        mimi_spec=MimiRVQSpec(projection_precision="fp32"),
        decoder_spec=MimiDecoderSpec(precision="fp32"),
        pcm_spec=MimiPCMSpec(initial_precision="fp32", transpose_precision="fp32",
                             residual_precision="fp32", final_precision="fp32"))
    encoder = MimiEncoderImage.after_output_codec(output)
    transformer = MimiInputTransformerImage.after_encoder(encoder)
    before = [dict(span) for span in transformer.spans]
    image = MimiInputDownsampleImage.after_transformer(transformer)
    assert transformer.spans == before
    assert image.input_at == transformer.io["output"]
    assert image.prefix_spans == transformer.prefix_spans + transformer.spans
    assert image.state_end <= transformer.input_state_limit
    assert image.op.prepared.base >= transformer.state_end
    assert image.op.fp_data >= transformer.weight_end
    assert len(image.program_slots) == len(image.compile()) == 1
    assert not image.compile()[0].run_args
    slot_bytes = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    assert image.program_slots[0] >= transformer.program_slots[-1] + slot_bytes
    spans = sorted(image.prefix_spans + image.spans, key=lambda item: item["base"])
    assert all(a["base"] + a["bytes"] <= b["base"]
               for a, b in zip(spans, spans[1:]))
    assert image.budget()["input_state_slack_bytes"] >= 0


@pytest.mark.parametrize("kwargs", [{"hidden": 0}, {"hidden": False},
                                     {"input_rows": 1}, {"output_rows": 2}])
def test_invalid_downsample_spec_rejected(kwargs):
    with pytest.raises(ValueError):
        MimiInputDownsampleSpec(**kwargs)
