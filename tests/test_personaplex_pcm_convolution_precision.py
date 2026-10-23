"""Source-equation checks for the FP32 initial and transpose PCM ISA paths.

These run on the EC2 CPU ISA backend. The references use the checkpoint tensor
orientation and global streaming time, independently of the lowering layout.
"""
import os
os.environ["NO_TORCH_COMPILE"] = "1"

from dataclasses import replace

import numpy as np
import pytest

from opentpu.llm.personaplex_mimi_pcm import MimiPCMImage, MimiPCMSpec
from test_personaplex_mimi_pcm import bits, device_image, read_floats, tiny


@pytest.fixture(scope="module")
def fp32_case():
    spec, cfg, default_image, weights, _ = tiny()
    selected = replace(spec, initial_precision="fp32", transpose_precision="fp32")
    image = MimiPCMImage(selected, cfg, input_at=0x1000, state_base=0x2000,
                         state_limit=0x400000, weight_base=0x400000,
                         weight_limit=0x2000000)
    programs, backend = device_image(selected, cfg, image, weights)
    return spec, default_image, image, weights, programs, backend


def _causal_conv1d(frames, weight, bias):
    """Conv1d[out,in,tap] with six initial zero rows and continuous time."""
    source = frames.reshape(-1, frames.shape[-1]).astype(np.float64)
    coeff = weight.astype(np.float64)
    expected = np.empty((len(source), len(bias)), np.float64)
    for time in range(len(source)):
        value = bias.astype(np.float64).copy()
        for tap in range(coeff.shape[-1]):
            earlier = time + tap - (coeff.shape[-1] - 1)
            if earlier >= 0:
                value += np.einsum("i,oi->o", source[earlier], coeff[:, :, tap])
        expected[time] = value
    return expected.reshape(frames.shape[0], frames.shape[1], -1)


def _streaming_convtranspose1d(frames, weight, bias, stride):
    """ELU then ConvTranspose1d[in,out,tap], emitting one stride per row.

    Global output positions receive contributions from the current input row
    and the preceding row's upper kernel half. Bias belongs to the output
    position once, including at the first row when native partial is None.
    """
    source = frames.reshape(-1, frames.shape[-1]).astype(np.float64)
    source = np.maximum(source, 0) + np.expm1(np.minimum(source, 0))
    coeff = weight.astype(np.float64)
    samples = len(source) * stride
    expected = np.broadcast_to(bias.astype(np.float64),
                               (samples, len(bias))).copy()
    for row, values in enumerate(source):
        for tap in range(coeff.shape[-1]):
            position = row * stride + tap
            if position < samples:
                expected[position] += np.einsum("i,io->o", values, coeff[:, :, tap])
    return expected.reshape(frames.shape[0], frames.shape[1] * stride, -1)


def test_selected_precisions_and_checkpoint_payload_orientation(fp32_case):
    spec, default_image, image, weights, programs, backend = fp32_case
    assert spec.initial_precision == spec.transpose_precision == "int8"
    assert all(op.precision == "int8" for op in default_image.ops)
    assert len(programs) == len(image.ops) == 14
    assert all(not program.run_args for program in programs)
    assert image.budget()["initial_precision"] == "fp32"
    assert image.budget()["transpose_precision"] == "fp32"
    selected = (0, 1, 4, 7, 10)
    for index, op in enumerate(image.ops):
        assert op.precision == ("fp32" if index in selected else "int8")
        if index not in selected:
            continue
        assert isinstance(op.fp_data, int) and op.fp_data >= 0
        assert not hasattr(op, "weight_offsets")
        assert f"conv{index}_fp32" in image.payload_hashes
        assert f"conv{index}_codes" not in image.payload_hashes
        assert f"conv{index}_scales" not in image.payload_hashes
        packed = read_floats(backend, op.fp_data, op.matrix_shape)
        source = weights[index]["weight"]
        if index == 0:
            for channel in range(op.cout):
                for tap in range(op.kernel):
                    np.testing.assert_array_equal(
                        bits(packed[channel, tap * op.cin:(tap + 1) * op.cin]),
                        bits(source[channel, :, tap]))
            used = op.cin * op.kernel
        else:
            for tap in range(op.kernel):
                for channel in range(op.cout):
                    np.testing.assert_array_equal(
                        bits(packed[tap * op.cout + channel, :op.cin]),
                        bits(source[:, channel, tap]))
            used = op.cin
        np.testing.assert_array_equal(
            bits(packed[:, used:]), bits(np.zeros_like(packed[:, used:])))


def test_initial_k7_short_frames_history_and_reset_match_causal_source(fp32_case):
    _, _, image, weights, programs, backend = fp32_case
    op = image.ops[0]
    assert op.kernel == 7 and op.rows == 2 and op.history == 6
    image.reset(backend)
    frames = np.random.default_rng(22671).normal(
        0, 0.65, (5, op.rows, op.cin)).astype(np.float32)
    expected = _causal_conv1d(frames, weights[0]["weight"], weights[0]["bias"])
    actual = []
    for frame_index, frame in enumerate(frames):
        backend.write(0, op.input.base, frame)
        backend.run([programs[0]])
        value = read_floats(backend, op.output.base, (op.rows, op.cout))
        actual.append(value)
        np.testing.assert_allclose(value, expected[frame_index], rtol=0, atol=2e-5)

        consumed = frames[:frame_index + 1].reshape(-1, op.cin)
        history = np.zeros((op.history, op.cin), np.float32)
        count = min(len(consumed), op.history)
        history[-count:] = consumed[-count:]
        saved = read_floats(backend, op.prepared.base, history.shape)
        np.testing.assert_array_equal(bits(saved), bits(history))

    image.reset(backend)
    replay = []
    for frame in frames:
        backend.write(0, op.input.base, frame)
        backend.run([programs[0]])
        replay.append(read_floats(backend, op.output.base, (op.rows, op.cout)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(np.stack(actual)))


@pytest.mark.parametrize("index", [1, 4, 7, 10])
def test_each_transpose_phase_bias_overlap_and_reset_match_source(fp32_case, index):
    _, _, image, weights, programs, backend = fp32_case
    op = image.ops[index]
    assert op.kind == "transpose" and op.kernel == 2 * op.stride
    assert op.precision == "fp32"
    image.reset(backend)
    frames = np.random.default_rng(35000 + index).normal(
        0, 0.6, (3, op.rows, op.cin)).astype(np.float32)
    expected = _streaming_convtranspose1d(
        frames, weights[index]["weight"], weights[index]["bias"], op.stride)
    actual = []
    for frame_index, frame in enumerate(frames):
        backend.write(0, op.input.base, frame)
        backend.run([programs[index]])
        value = read_floats(backend, op.output.base,
                            (op.rows * op.stride, op.cout))
        actual.append(value)
        np.testing.assert_allclose(value, expected[frame_index],
                                   rtol=0, atol=2e-5,
                                   err_msg=f"transpose {index} frame {frame_index}")
        np.testing.assert_array_equal(read_floats(backend, op.seen.base, (1,)),
                                      np.array([1], np.float32))

        # Source ConvTranspose retains the biased upper half of its final raw
        # row. The next row adds this tail with its bias removed.
        activated = frame[-1].astype(np.float64)
        activated = np.maximum(activated, 0) + np.expm1(np.minimum(activated, 0))
        tail = np.stack([
            activated @ weights[index]["weight"][:, :, tap].astype(np.float64)
            + weights[index]["bias"].astype(np.float64)
            for tap in range(op.stride, op.kernel)
        ])
        saved = read_floats(backend, op.partial.base, (op.stride, op.cout))
        np.testing.assert_allclose(saved, tail, rtol=0, atol=2e-5)

    image.reset(backend)
    replay = []
    for frame in frames:
        backend.write(0, op.input.base, frame)
        backend.run([programs[index]])
        replay.append(read_floats(backend, op.output.base,
                                  (op.rows * op.stride, op.cout)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(np.stack(actual)))


@pytest.mark.parametrize("field", ["initial_precision", "transpose_precision"])
@pytest.mark.parametrize("invalid", ["fp16", "", None, False])
def test_unknown_convolution_precision_rejected(field, invalid):
    with pytest.raises(ValueError, match=field):
        MimiPCMSpec(**{field: invalid})


@pytest.mark.parametrize("failed_metric", ["relative_l2", "error_rms_full_scale",
                                          "peak_absolute_error_full_scale"])
def test_acceptance_requires_every_frame_not_only_aggregate(failed_metric):
    from tools.personaplex_pcm_precision_check import SEANET_LIMITS, numerical_acceptance

    boundary = dict(SEANET_LIMITS)
    metrics = dict(aggregate=boundary.copy(),
                   per_frame=[boundary.copy(), boundary.copy()])
    assert numerical_acceptance(metrics, SEANET_LIMITS)["passed"]
    metrics["per_frame"][1][failed_metric] *= 2
    assert not numerical_acceptance(metrics, SEANET_LIMITS)["passed"]
