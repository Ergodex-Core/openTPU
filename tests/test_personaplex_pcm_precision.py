"""Isolated FP32 final SEANet projection checks on the EC2 CPU ISA runner."""
import os
os.environ["NO_TORCH_COMPILE"] = "1"

from dataclasses import replace

import numpy as np
import pytest

from opentpu.llm.personaplex_mimi_pcm import MimiPCMImage, MimiPCMSpec
from test_personaplex_mimi_pcm import (bits, device_frames, device_image,
                                       native_decoder, read_floats, tiny)


def _fp32_image(spec, cfg):
    return MimiPCMImage(replace(spec, final_precision="fp32"), cfg,
                        input_at=0x1000, state_base=0x2000,
                        state_limit=0x400000, weight_base=0x400000,
                        weight_limit=0x2000000)


def _residual_fp32_image(spec, cfg):
    return MimiPCMImage(replace(spec, residual_precision="fp32", final_precision="fp32"), cfg,
                        input_at=0x1000, state_base=0x2000,
                        state_limit=0x400000, weight_base=0x400000,
                        weight_limit=0x2000000)


def _native_final_only(spec, weights, resident_stage12):
    """Fresh original ELU plus streaming final Conv1d on identical ISA input."""
    import torch

    native = native_decoder(spec, weights)
    native_elu, native_conv = native.model[13], native.model[14]
    outputs = []
    with torch.no_grad(), native_conv.streaming(1):
        for frame in resident_stage12:
            source = torch.from_numpy(frame.T.copy()[None, :, :])
            result = native_conv(native_elu(source))
            outputs.append(result.detach().numpy()[0, 0].copy())
    return np.stack(outputs)


def _native_residual_only(spec, weights, model_index, resident_input):
    """Fresh native residual stream on the identical old-ISA boundary input."""
    import torch

    native = native_decoder(spec, weights)
    block = native.model[model_index]
    outputs = []
    with torch.no_grad(), block.streaming(1):
        for frame in resident_input:
            source = torch.from_numpy(frame.T.copy()[None, :, :])
            outputs.append(block(source).detach().numpy()[0].T.copy())
    return np.stack(outputs)


def test_fp32_final_changes_only_stage14_and_replays_all_state_bit_exact():
    spec, cfg, old_image, weights, frames = tiny()
    old_programs, old_backend = device_image(spec, cfg, old_image, weights)
    old_pcm, old_stages = device_frames(old_image, old_programs, old_backend, frames)

    image = _fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    assert len(programs) == len(old_programs) == len(image.ops) == 14
    assert image.budget()["final_precision"] == "fp32"
    assert [op.precision for op in image.ops[:-1]] == ["int8"] * 13
    final = image.ops[-1]
    assert final.precision == "fp32" and final.kind == "conv" and final.cout == 1
    assert final.fp_weight.shape == (final.padded_k,)
    assert not hasattr(final, "weight_offsets")
    assert "conv13_fp32" in image.payload_hashes
    assert "conv13_codes" not in image.payload_hashes
    assert "conv13_scales" not in image.payload_hashes
    assert all(not program.run_args for program in programs)

    fp32_pcm, fp32_stages = device_frames(image, programs, backend, frames)
    for index in range(14):
        np.testing.assert_array_equal(bits(fp32_stages[index]), bits(old_stages[index]),
                                      err_msg=f"unchanged SEANet source stage {index}")
    np.testing.assert_array_equal(bits(old_stages[14][..., 0]), bits(old_pcm))
    np.testing.assert_array_equal(bits(fp32_stages[14][..., 0]), bits(fp32_pcm))
    assert old_pcm.shape == fp32_pcm.shape == (3, spec.samples)
    assert np.isfinite(fp32_pcm.astype(np.float64) - old_pcm.astype(np.float64)).all()

    persistent = [(at, np.asarray(backend.read(0, at, size), np.uint8).copy())
                  for at, size in image.reset_regions]
    image.reset(backend)
    replay_pcm, replay_stages = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(bits(replay_pcm), bits(fp32_pcm))
    for index, (before, after) in enumerate(zip(fp32_stages, replay_stages)):
        np.testing.assert_array_equal(bits(after), bits(before),
                                      err_msg=f"FP32 reset replay stage {index}")
    for at, before in persistent:
        np.testing.assert_array_equal(np.asarray(backend.read(0, at, len(before)), np.uint8),
                                      before)


def test_fp32_final_on_identical_resident_stage12_matches_native_streaming():
    spec, cfg, old_image, weights, frames = tiny()
    old_programs, old_backend = device_image(spec, cfg, old_image, weights)
    _, old_stages = device_frames(old_image, old_programs, old_backend, frames)
    identical_input = old_stages[12]
    native_pcm = _native_final_only(spec, weights, identical_input)

    image = _fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    final = image.ops[-1]
    image.reset(backend)
    resident_pcm = []
    for frame in identical_input:
        backend.write(0, final.input.base, frame)
        backend.run([programs[-1]])
        resident_pcm.append(read_floats(backend, image.io["output"], (spec.samples,)))
    resident_pcm = np.stack(resident_pcm)
    assert resident_pcm.shape == native_pcm.shape == (3, spec.samples)
    np.testing.assert_allclose(resident_pcm, native_pcm, rtol=0, atol=2e-6)

    image.reset(backend)
    replay = []
    for frame in identical_input:
        backend.write(0, final.input.base, frame)
        backend.run([programs[-1]])
        replay.append(read_floats(backend, image.io["output"], (spec.samples,)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(resident_pcm))


def test_fp32_final_program_has_no_interstage_host_feedback(monkeypatch):
    spec, cfg, _, weights, frames = tiny()
    image = _fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    backend.write(0, image.input_at, frames[0])
    original_read, original_write, original_run = backend.read, backend.write, backend.run
    transfers, submitted = [], []

    def traced_read(*args):
        transfers.append("read")
        return original_read(*args)

    def traced_write(*args):
        transfers.append("write")
        return original_write(*args)

    def traced_run(batch, *args, **kwargs):
        assert len(batch) == 1 and not batch[0].run_args
        assert not args and kwargs.get("args") is None
        submitted.append(batch[0])
        return original_run(batch, *args, **kwargs)

    monkeypatch.setattr(backend, "read", traced_read)
    monkeypatch.setattr(backend, "write", traced_write)
    monkeypatch.setattr(backend, "run", traced_run)
    for program in programs:
        backend.run([program])
    assert transfers == [] and submitted == programs
    assert read_floats(backend, image.io["output"], (spec.samples,)).shape == (spec.samples,)


def test_fp32_residual_and_final_payloads_upstream_boundary_and_reset():
    spec, cfg, old_image, weights, frames = tiny()
    old_programs, old_backend = device_image(spec, cfg, old_image, weights)
    _, old_stages = device_frames(old_image, old_programs, old_backend, frames)

    image = _residual_fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    assert len(programs) == len(image.ops) == 14
    assert image.budget()["residual_precision"] == "fp32"
    assert image.budget()["final_precision"] == "fp32"
    assert all(not program.run_args for program in programs)
    assert [image.ops[index].precision for index in (0, 1, 4, 7, 10)] == ["int8"] * 5
    residual_indices = (2, 3, 5, 6, 8, 9, 11, 12)
    for index in residual_indices:
        op = image.ops[index]
        assert op.precision == "fp32" and op.fp_weight is None
        assert isinstance(op.fp_data, int) and op.fp_data >= 0
        assert not hasattr(op, "weight_offsets")
        assert f"conv{index}_fp32" in image.payload_hashes
        assert f"conv{index}_codes" not in image.payload_hashes
        assert f"conv{index}_scales" not in image.payload_hashes

        # Every output-channel row is resident at its dynamic FP32 address.
        packed = read_floats(backend, op.fp_data, (op.cout, op.padded_k))
        source = weights[index]["weight"].transpose(0, 2, 1).reshape(op.cout, -1)
        np.testing.assert_array_equal(bits(packed[:, :source.shape[1]]), bits(source))
        np.testing.assert_array_equal(bits(packed[:, source.shape[1]:]),
                                      bits(np.zeros_like(packed[:, source.shape[1]:])))
    assert image.ops[-1].precision == "fp32"
    assert image.ops[-1].fp_weight.shape == (image.ops[-1].padded_k,)

    pcm, stages = device_frames(image, programs, backend, frames)
    for index in range(3):
        np.testing.assert_array_equal(bits(stages[index]), bits(old_stages[index]),
                                      err_msg=f"unchanged upstream stage {index}")
    persistent = [(at, np.asarray(backend.read(0, at, size), np.uint8).copy())
                  for at, size in image.reset_regions]
    image.reset(backend)
    replay_pcm, replay_stages = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(bits(replay_pcm), bits(pcm))
    for index, (before, after) in enumerate(zip(stages, replay_stages)):
        np.testing.assert_array_equal(bits(after), bits(before),
                                      err_msg=f"FP32 residual reset stage {index}")
    for at, before in persistent:
        np.testing.assert_array_equal(np.asarray(backend.read(0, at, len(before)), np.uint8),
                                      before)


@pytest.mark.parametrize("model_index,prior_stage,program_pair", [
    (3, 2, (2, 3)), (6, 5, (5, 6)), (9, 8, (8, 9)), (12, 11, (11, 12)),
])
def test_each_fp32_residual_block_matches_native_on_identical_isa_input(
        model_index, prior_stage, program_pair):
    spec, cfg, old_image, weights, frames = tiny()
    old_programs, old_backend = device_image(spec, cfg, old_image, weights)
    _, old_stages = device_frames(old_image, old_programs, old_backend, frames)
    identical_input = old_stages[prior_stage]
    native_result = _native_residual_only(spec, weights, model_index, identical_input)

    image = _residual_fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    first, second = (image.ops[index] for index in program_pair)
    assert first.model_index == second.model_index == model_index
    assert first.subpath == "block.1.conv.conv"
    assert second.subpath == "block.3.conv.conv"
    assert first.input.base == second.skip.base
    output_at, rows, channels = image.layer_outputs[model_index]
    assert output_at == second.output.base
    image.reset(backend)
    resident = []
    for frame in identical_input:
        backend.write(0, first.input.base, frame)
        backend.run([programs[program_pair[0]]])
        backend.run([programs[program_pair[1]]])
        resident.append(read_floats(backend, output_at, (rows, channels)))
    resident = np.stack(resident)
    assert resident.shape == native_result.shape
    np.testing.assert_allclose(resident, native_result, rtol=0, atol=1e-5)

    image.reset(backend)
    replay = []
    for frame in identical_input:
        backend.write(0, first.input.base, frame)
        backend.run([programs[program_pair[0]]])
        backend.run([programs[program_pair[1]]])
        replay.append(read_floats(backend, output_at, (rows, channels)))
    np.testing.assert_array_equal(bits(np.stack(replay)), bits(resident))


def test_fp32_residual_channel_rows_match_numpy():
    """No Torch: exercise dynamic coefficient rows through the actual ISA VPU."""
    spec, cfg, _, weights, _ = tiny()
    image = _residual_fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    first, pointwise = image.ops[11], image.ops[12]
    assert first.model_index == pointwise.model_index == 12
    assert first.kernel == 3 and first.cin == 4 and first.cout == 2
    assert first.padded_k > first.cin * first.kernel
    assert pointwise.kernel == 1 and pointwise.skip.base == first.input.base
    rng = np.random.default_rng(12017)
    frames = rng.normal(0, 0.6, (3, first.rows, first.cin)).astype(np.float32)
    kernel = weights[11]["weight"].astype(np.float64)
    bias = weights[11]["bias"].astype(np.float64)
    point_kernel = weights[12]["weight"][:, :, 0].astype(np.float64)
    point_bias = weights[12]["bias"].astype(np.float64)

    # Original causal K3 starts with two zeros, then retains the last two
    # activated inputs across frame boundaries. Independent einsum follows
    # native Conv1d [out,in,kernel] orientation and includes true skip.
    history = np.zeros((first.kernel - 1, first.cin), np.float64)
    expected_first, expected_block = [], []
    for frame in frames:
        source = frame.astype(np.float64)
        activated = np.maximum(source, 0) + np.expm1(np.minimum(source, 0))
        prepared = np.concatenate((history, activated))
        windows = np.stack([prepared[t:t + first.kernel] for t in range(first.rows)])
        branch = np.einsum("tki,oik->to", windows, kernel) + bias
        branch_activated = np.maximum(branch, 0) + np.expm1(np.minimum(branch, 0))
        combined = np.einsum("ti,oi->to", branch_activated, point_kernel) + point_bias + source
        expected_first.append(branch)
        expected_block.append(combined)
        history = prepared[-(first.kernel - 1):].copy()

    actual_first, actual_block = [], []
    for frame in frames:
        backend.write(0, first.input.base, frame)
        backend.run([programs[11]])
        backend.run([programs[12]])
        actual_first.append(read_floats(backend, first.output.base, (first.rows, first.cout)))
        actual_block.append(read_floats(backend, pointwise.output.base,
                                        (pointwise.rows, pointwise.cout)))
    np.testing.assert_allclose(np.stack(actual_first), np.stack(expected_first),
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.stack(actual_block), np.stack(expected_block),
                               rtol=0, atol=1e-5)


def test_fp32_residual_programs_have_no_interstage_host_feedback(monkeypatch):
    spec, cfg, _, weights, frames = tiny()
    image = _residual_fp32_image(spec, cfg)
    programs, backend = device_image(image.spec, cfg, image, weights)
    backend.write(0, image.input_at, frames[0])
    original_read, original_write, original_run = backend.read, backend.write, backend.run
    transfers, submitted = [], []

    def traced_read(*args):
        transfers.append("read")
        return original_read(*args)

    def traced_write(*args):
        transfers.append("write")
        return original_write(*args)

    def traced_run(batch, *args, **kwargs):
        assert len(batch) == 1 and not batch[0].run_args
        assert not args and kwargs.get("args") is None
        submitted.append(batch[0])
        return original_run(batch, *args, **kwargs)

    monkeypatch.setattr(backend, "read", traced_read)
    monkeypatch.setattr(backend, "write", traced_write)
    monkeypatch.setattr(backend, "run", traced_run)
    for program in programs:
        backend.run([program])
    assert transfers == [] and submitted == programs


@pytest.mark.parametrize("invalid", ["fp16", "", None, False])
def test_unknown_residual_precision_rejected(invalid):
    with pytest.raises(ValueError, match="residual_precision"):
        MimiPCMSpec(residual_precision=invalid)


@pytest.mark.parametrize("invalid", ["fp16", "", None, False])
def test_unknown_final_precision_rejected(invalid):
    with pytest.raises(ValueError, match="final_precision"):
        MimiPCMSpec(final_precision=invalid)
