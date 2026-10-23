"""Synthetic source and ISA checks for the causal input Mimi SEANet encoder.

Run on the EC2 CPU ISA builder. No checkpoint, board, or audio quality claim is
involved. The reference uses global sample positions rather than device tiles.
"""
import os
os.environ["NO_TORCH_COMPILE"] = "1"

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_mimi_encoder import MimiEncoderImage, MimiEncoderSpec


def tiny(*, output_rows=2, ratios=(2, 2, 2, 2)):
    spec = MimiEncoderSpec(hidden=64, n_filters=4, ratios=ratios,
                           output_rows=output_rows)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 26)
    image = MimiEncoderImage(spec, cfg, input_at=0x1000, state_base=0x2000,
                             state_limit=0x400000, weight_base=0x400000,
                             weight_limit=0x2000000)
    rng = np.random.default_rng(39172 + output_rows)
    weights = [dict(kind=op.kind, model_index=op.model_index,
                    subpath=op.subpath,
                    weight=rng.normal(0, 0.085,
                        (op.cout, op.cin, op.kernel)).astype(np.float32),
                    bias=rng.normal(0, 0.025, op.cout).astype(np.float32))
               for op in image.ops]
    frames = rng.normal(0, 0.7, (3, spec.samples)).astype(np.float32)
    return spec, cfg, image, weights, frames


def device_image(image, cfg, weights):
    programs = image.compile()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, weights)
    return programs, backend


def read_floats(backend, address, shape):
    count = int(np.prod(shape))
    return np.asarray(backend.read(0, address, 4 * count), np.uint8).view(
        np.float32).reshape(shape).copy()


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def device_frames(image, programs, backend, frames):
    output, stages = [], [[] for _ in range(15)]
    for frame_index, frame in enumerate(frames):
        backend.write(0, image.input_at, frame.reshape(-1, 1))
        for program in programs:
            backend.run([program])
        output.append(read_floats(backend, image.io["output"],
                                  (image.spec.output_rows, image.spec.hidden)))
        for index, (at, rows, channels) in enumerate(image.layer_outputs):
            stages[index].append(read_floats(backend, at, (rows, channels)))
        assert read_floats(backend, image.io["frame"], (1,))[0] == frame_index + 1
    return np.stack(output), [np.stack(value) for value in stages]


def _elu(value):
    return np.maximum(value, 0) + np.expm1(np.minimum(value, 0))


def _causal_source_conv(source, record, stride, streams, index):
    """Original Conv1d[out,in,tap] at continuous stream positions.

    A stride-S convolution emits at global output q from input positions
    q*S + tap - (kernel-S). Missing positions before sample zero are zero.
    Each frame here has a whole number of stride groups.
    """
    weight = record["weight"].astype(np.float64)
    bias = record["bias"].astype(np.float64)
    prior = streams[index]
    entire = np.concatenate((prior, source.astype(np.float64)), axis=0)
    assert len(prior) % stride == len(source) % stride == 0
    output = np.empty((len(source) // stride, len(bias)), np.float64)
    for row in range(len(output)):
        global_row = len(prior) // stride + row
        total = bias.copy()
        for tap in range(weight.shape[-1]):
            at = global_row * stride + tap - (weight.shape[-1] - stride)
            if 0 <= at < len(entire):
                total += np.einsum("i,oi->o", entire[at], weight[:, :, tap])
        output[row] = total
    streams[index] = entire
    return output


def source_frames(spec, weights, frames):
    """Moshi top-level order: initial, four residual/ELU/down pairs, final."""
    streams = [np.empty((0, record["weight"].shape[1]), np.float64)
               for record in weights]
    down_strides = tuple(reversed(spec.ratios))
    outputs, stages = [], [[] for _ in range(15)]
    for frame in frames:
        x = frame.reshape(-1, 1).astype(np.float64)
        x = _causal_source_conv(x, weights[0], 1, streams, 0)
        stages[0].append(x.copy())
        for stage in range(4):
            first, second, down = 1 + stage * 3, 2 + stage * 3, 3 + stage * 3
            model_block = 1 + stage * 3
            shortcut = x
            branch = _causal_source_conv(_elu(x), weights[first], 1, streams, first)
            branch = _causal_source_conv(_elu(branch), weights[second], 1, streams, second)
            x = shortcut + branch
            stages[model_block].append(x.copy())
            x = _elu(x)
            stages[model_block + 1].append(x.copy())
            x = _causal_source_conv(x, weights[down], down_strides[stage], streams, down)
            stages[model_block + 2].append(x.copy())
        x = _elu(x)
        stages[13].append(x.copy())
        x = _causal_source_conv(x, weights[13], 1, streams, 13)
        stages[14].append(x.copy())
        outputs.append(x.copy())
    return np.stack(outputs), [np.stack(value) for value in stages]


def native_frames(spec, weights, frames):
    """Strictly load synthetic tensors into the pinned original Moshi class."""
    import torch
    from moshi.modules.seanet import SEANetEncoder

    native = SEANetEncoder(channels=1, dimension=spec.hidden, causal=True,
        n_filters=spec.n_filters, n_residual_layers=1, activation="ELU", compress=2,
        dilation_base=2, disable_norm_outer_blocks=0, kernel_size=7,
        residual_kernel_size=3, last_kernel_size=3, norm="none",
        pad_mode="constant", ratios=list(spec.ratios), true_skip=True)
    state = {}
    for record in weights:
        prefix = f"model.{record['model_index']}.{record['subpath']}."
        state[prefix + "weight"] = torch.from_numpy(record["weight"].copy())
        state[prefix + "bias"] = torch.from_numpy(record["bias"].copy())
    native.load_state_dict(state, strict=True)
    native.eval()
    stages = [[] for _ in native.model]
    handles = [module.register_forward_hook(
        lambda _module, _input, value, index=index:
        stages[index].append(value.detach().numpy()[0].T.copy()))
        for index, module in enumerate(native.model)]
    try:
        outputs = []
        with torch.no_grad(), native.streaming(1):
            for frame in frames:
                source = torch.from_numpy(frame.reshape(1, 1, -1).copy())
                outputs.append(native(source).detach().numpy()[0].T.copy())
    finally:
        for handle in handles:
            handle.remove()
    return np.stack(outputs), [np.stack(values) for values in stages]


def test_all_native_boundaries_match_independent_source_and_isa_reset():
    spec, cfg, image, weights, frames = tiny()
    assert len(image.ops) == 14 and len(image.layer_outputs) == 15
    assert [op.model_index for op in image.ops] == [
        0, 1, 1, 3, 4, 4, 6, 7, 7, 9, 10, 10, 12, 14]
    assert [image.ops[index].stride for index in (3, 6, 9, 12)] == [2] * 4
    assert all(op.history == op.kernel - op.stride for op in image.ops)
    expected, expected_stages = source_frames(spec, weights, frames)
    native, native_stages = native_frames(spec, weights, frames)
    assert len(native_stages) == len(expected_stages) == 15
    np.testing.assert_allclose(native, expected, rtol=0, atol=5e-5)
    for index, (actual, reference) in enumerate(zip(native_stages, expected_stages)):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=5e-5,
                                   err_msg=f"native encoder stage {index}")

    programs, backend = device_image(image, cfg, weights)
    assert len(programs) == 14 and all(not program.run_args for program in programs)
    actual, stages = device_frames(image, programs, backend, frames)
    assert actual.shape == (3, spec.output_rows, spec.hidden)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-5)
    for index, (actual_stage, reference) in enumerate(zip(stages, expected_stages)):
        assert actual_stage.shape == reference.shape
        np.testing.assert_allclose(actual_stage, reference, rtol=0, atol=5e-5,
                                   err_msg=f"ISA encoder stage {index}")

    state = [(at, np.asarray(backend.read(0, at, size), np.uint8).copy())
             for at, size in image.reset_regions]
    image.reset(backend)
    replay, replay_stages = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(bits(replay), bits(actual))
    for before, after in zip(stages, replay_stages):
        np.testing.assert_array_equal(bits(after), bits(before))
    for at, before in state:
        np.testing.assert_array_equal(np.asarray(backend.read(0, at, len(before)), np.uint8),
                                      before)


@pytest.mark.parametrize("index", [3, 6, 9, 12])
def test_each_strided_conv_global_phase_and_retained_history(index):
    _, cfg, image, weights, _ = tiny()
    op = image.ops[index]
    assert op.stride == 2 and op.kernel == 4 and op.history == 2
    programs, backend = device_image(image, cfg, weights)
    streams = [np.empty((0, record["weight"].shape[1]), np.float64)
               for record in weights]
    frames = np.random.default_rng(48000 + index).normal(
        0, 0.65, (3, op.rows, op.cin)).astype(np.float32)
    observed = []
    for frame in frames:
        backend.write(0, op.input.base, frame)
        backend.run([programs[index]])
        expected = _causal_source_conv(_elu(frame.astype(np.float64)),
                                       weights[index], op.stride, streams, index)
        actual = read_floats(backend, op.output.base, (op.output_rows, op.cout))
        observed.append(actual)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-5)
        retained = read_floats(backend, op.prepared.base, (op.history, op.cin))
        np.testing.assert_allclose(retained, streams[index][-op.history:],
                                   rtol=0, atol=2e-6)
    image.reset(backend)
    replay = []
    for frame in frames:
        backend.write(0, op.input.base, frame)
        backend.run([programs[index]])
        replay.append(read_floats(backend, op.output.base,
                                  (op.output_rows, op.cout)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(np.stack(observed)))


def test_nonuniform_ratios_run_in_reverse_encoder_order():
    spec, cfg, image, weights, frames = tiny(ratios=(2, 3, 2, 2))
    assert spec.samples == 48
    assert [image.ops[index].stride for index in (3, 6, 9, 12)] == [2, 2, 3, 2]
    assert [image.ops[index].kernel for index in (3, 6, 9, 12)] == [4, 4, 6, 4]
    expected, expected_stages = source_frames(spec, weights, frames)
    programs, backend = device_image(image, cfg, weights)
    actual, stages = device_frames(image, programs, backend, frames)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-5)
    for index, (actual_stage, reference) in enumerate(zip(stages, expected_stages)):
        np.testing.assert_allclose(actual_stage, reference, rtol=0, atol=5e-5,
                                   err_msg=f"reverse-ratio encoder stage {index}")


def test_final_k3_history_exceeds_one_row_frame_and_replays():
    spec, cfg, image, weights, frames = tiny(output_rows=1)
    assert spec.samples == 16
    final = image.ops[-1]
    assert final.rows == 1 and final.kernel == 3 and final.history == 2
    programs, backend = device_image(image, cfg, weights)
    source, source_stages = source_frames(spec, weights, frames)
    actual, stages = [], [[] for _ in range(15)]
    for index, frame in enumerate(frames):
        backend.write(0, image.input_at, frame.reshape(-1, 1))
        for program in programs:
            backend.run([program])
        actual.append(read_floats(backend, image.io["output"],
                                  (spec.output_rows, spec.hidden)))
        for stage, (at, rows, channels) in enumerate(image.layer_outputs):
            stages[stage].append(read_floats(backend, at, (rows, channels)))
        # Final stage 13 is the native ELU input to the last Conv1d. With
        # one row per frame, the first retained history includes one zero row.
        previous = source_stages[13][:index + 1].reshape(-1, final.cin)
        expected_history = np.zeros((final.history, final.cin), np.float64)
        count = min(len(previous), final.history)
        expected_history[-count:] = previous[-count:]
        saved = read_floats(backend, final.prepared.base,
                            (final.history, final.cin))
        np.testing.assert_allclose(saved, expected_history, rtol=0, atol=2e-6)
    actual = np.stack(actual)
    stages = [np.stack(value) for value in stages]
    np.testing.assert_allclose(actual, source, rtol=0, atol=5e-5)
    image.reset(backend)
    replay, replay_stages = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(bits(replay), bits(actual))
    for before, after in zip(stages, replay_stages):
        np.testing.assert_array_equal(bits(after), bits(before))


@pytest.mark.parametrize("bias", [2.5, -2.5])
def test_encoder_returns_raw_latents_without_clipping_or_gain(bias):
    spec, cfg, image, weights, frames = tiny()
    for record in weights:
        record["weight"].fill(0)
        record["bias"].fill(0)
    weights[-1]["bias"].fill(bias)
    programs, backend = device_image(image, cfg, weights)
    output, _ = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(output, np.full_like(output, bias))
    assert output.shape == (3, spec.output_rows, spec.hidden)
    assert np.max(np.abs(output)) > 1.0


def test_encoder_programs_run_without_interstage_host_feedback(monkeypatch):
    _, cfg, image, weights, frames = tiny()
    programs, backend = device_image(image, cfg, weights)
    backend.write(0, image.input_at, frames[0].reshape(-1, 1))
    read, write, run = backend.read, backend.write, backend.run
    transfers, submitted = [], []

    def traced_read(*args):
        transfers.append("read")
        return read(*args)

    def traced_write(*args):
        transfers.append("write")
        return write(*args)

    def traced_run(batch, *args, **kwargs):
        assert len(batch) == 1 and not batch[0].run_args
        assert not args and kwargs.get("args") is None
        submitted.append(batch[0])
        return run(batch, *args, **kwargs)

    monkeypatch.setattr(backend, "read", traced_read)
    monkeypatch.setattr(backend, "write", traced_write)
    monkeypatch.setattr(backend, "run", traced_run)
    for program in programs:
        backend.run([program])
    assert transfers == [] and submitted == programs


@pytest.mark.parametrize("kwargs", [
    {"output_rows": 0}, {"output_rows": False}, {"ratios": (2, 2, 2)},
    {"ratios": (2, 2, 1, 2)}, {"hidden": 0}, {"n_filters": 3},
])
def test_invalid_encoder_spec_rejected(kwargs):
    with pytest.raises(ValueError):
        MimiEncoderSpec(**kwargs)


def test_production_encoder_fits_beside_unchanged_resident_output_codec():
    from opentpu import target
    from opentpu.llm.personaplex_depth_chain import DepthChainImage, DepthChainSpec
    from opentpu.llm.personaplex_mimi import MimiRVQSpec
    from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
    from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec

    cfg = target.config("f2-hbm:p4rl8a64c150")
    base = DepthChainImage(DepthChainSpec(), cfg,
        mimi_spec=MimiRVQSpec(projection_precision="fp32"),
        decoder_spec=MimiDecoderSpec(precision="fp32"),
        pcm_spec=MimiPCMSpec(initial_precision="fp32", transpose_precision="fp32",
                             residual_precision="fp32", final_precision="fp32"))
    original_spans = [dict(span) for span in base.spans]
    image = MimiEncoderImage.after_output_codec(base)
    budget = image.budget()
    assert base.spans == original_spans
    all_spans = sorted(base.spans + image.spans, key=lambda span: span["base"])
    assert all(a["base"] + a["bytes"] <= b["base"]
               for a, b in zip(all_spans, all_spans[1:]))
    assert budget["combined_required_bytes"] <= cfg.DRAM_BYTES
    assert budget["input_state_slack_bytes"] > 0
    assert budget["input_weight_slack_bytes"] > 0
    assert len(base.program_slots) + len(image.program_slots) == 41
    assert max(budget["program_words"]) <= cfg.IMEM_WORDS
