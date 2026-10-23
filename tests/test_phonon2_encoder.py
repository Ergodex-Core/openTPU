"""Whole-context layer, precision correction and shared-profile contracts."""
import dataclasses
import hashlib

import numpy as np
import pytest

from opentpu import Config, isa as I, waits as W
from opentpu.isasim import Machine
from opentpu.asr.phonon2_encoder import compile_encoder_layer, scheduled_programs, shared_profile_config
from opentpu.asr.phonon2_conformer import BlockShape
from opentpu.asr.phonon2_precise_subsampler import compile_precise_subsampler
from opentpu.asr.phonon2_subsampler import SubsampleShape
from tests.test_phonon2_conformer import synthetic_weights
from tests.test_phonon2_subsampler import synthetic_weights as subsample_weights
from tools.phonon2_encoder_reference import layer_torch
from tools.phonon2_reference import conformer_reference
from tools.phonon2_subsample_reference import subsampler_reference
from tools.phonon2_qualify_encoder import SimulatorDevice
from tools.phonon2_program_device import ProgramSlotDevice


def run_layer(plan, x, machine=None):
    if machine is None:
        memory = plan.image.copy()
        memory[plan.input_at:plan.input_at + x.nbytes] = x.view(np.uint8).ravel()
        machine = Machine(plan.cfg, [plan.programs[0]], [memory])
    else:
        machine.load([plan.programs[0]])
        machine.slices[0].dram[:plan.input_at] = plan.image[:plan.input_at]
        machine.slices[0].dram[plan.input_at:plan.input_at + x.nbytes] = x.view(np.uint8).ravel()
    machine.run()
    for program in plan.programs[1:]:
        machine.load([program]).run()
    memory = machine.slices[0].dram
    actual = memory[plan.output_at:plan.output_at + x.nbytes].view(np.float32).reshape(x.shape).copy()
    probability = memory[plan.attention_scores_at:plan.attention_scores_at + 4 * len(x) ** 2].view(np.float32).reshape(len(x), len(x)).copy()
    return actual, probability, machine


@pytest.mark.parametrize("frames,valid", [(1, 1), (8, 7), (9, 1), (33, 29), (89, 88)])
@pytest.mark.parametrize("passes", [1, 2])
def test_tiled_whole_context_layer_and_mask_semantics(frames, valid, passes):
    shape = BlockShape(128, 256, 4)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20, IMEM_WORDS=32768)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(27).normal(size=(frames, 128)).astype(np.float32)
    plan = compile_encoder_layer(weights, frames, valid, cfg, shape, passes=passes, tile=8)
    actual, probability, _ = run_layer(plan, x)
    dense = conformer_reference(weights, x, valid, shape)["output"]
    relative = np.linalg.norm(actual - dense) / np.linalg.norm(dense)
    assert relative < (.03 if passes == 1 else .001)
    np.testing.assert_array_equal(probability[:valid, valid:], 0)
    if valid < frames:
        np.testing.assert_allclose(probability[valid:], 1 / frames, rtol=2e-6, atol=1e-8)
        assert np.linalg.norm(actual[valid:]) > 0
    assert plan.budget()["max_instruction_words"] <= cfg.IMEM_WORDS
    assert plan.budget()["max_tmem_live_words"] <= cfg.TMEM_WORDS


def test_tiled_layer_reset_drops_previous_valid_prefix_and_halo():
    shape = BlockShape(128, 256, 4)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20, IMEM_WORDS=32768)
    weights = synthetic_weights(shape)
    p1 = compile_encoder_layer(weights, 17, 17, cfg, shape, passes=2, tile=8)
    p2 = compile_encoder_layer(weights, 17, 3, cfg, shape, passes=2, tile=8)
    rng = np.random.default_rng(40)
    _, _, machine = run_layer(p1, rng.normal(size=(17, 128)).astype(np.float32))
    x = rng.normal(size=(17, 128)).astype(np.float32)
    reused, _, _ = run_layer(p2, x, machine)
    fresh, _, _ = run_layer(p2, x)
    np.testing.assert_array_equal(reused, fresh)


def test_native_torch_oracle_matches_retained_dense_graph():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    shape = BlockShape(128, 256, 4)
    weights = synthetic_weights(shape)
    x = np.random.default_rng(41).normal(size=(33, 128)).astype(np.float32)
    expected = conformer_reference(weights, x, 29, shape)["output"]
    actual = layer_torch(weights, x, 29, shape=shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_residual_subsampler_and_wait_image_dependency_checker():
    shape = SubsampleShape(8, 16)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20, IMEM_WORDS=32768)
    weights = subsample_weights(shape)
    x = np.random.default_rng(25).normal(size=(9, 128)).astype(np.float32)
    plan = compile_precise_subsampler(weights, 9, 8, cfg, shape)
    device = SimulatorDevice(cfg)
    device.reset(plan.image.size)
    device.constants(plan.image, plan.constant_ranges)
    device.memory[plan.input_at:plan.input_at + x.nbytes] = x.view(np.uint8).ravel()
    for program in plan.programs:
        device.phase(program, check=True)
    actual = device.read(plan.output_at, plan.stored_lengths[3], 16)
    dense = subsampler_reference(weights, x, 8, shape)["output"]
    assert np.linalg.norm(actual - dense) / np.linalg.norm(dense) < .001
    assert device.wait_checked > 0
    images = scheduled_programs(plan)
    assert len(images) == 9 and all(len(entries) * 8 <= loadable.nbytes for _, entries, loadable in images)


def test_shared_profile_is_strict_about_runtime_capabilities():
    names = {f.name for f in dataclasses.fields(Config)}
    if not {"PORTB", "ROWLANES", "ALINE", "T2"} <= names:
        with pytest.raises(RuntimeError, match="ASR_SHARED_PROFILE_RUNTIME_REQUIRED"):
            shared_profile_config()
    else:
        cfg = shared_profile_config()
        from opentpu import target
        assert f"{target.arch_hash(cfg):08x}" == "dd6b92a6"
        assert (cfg.MCOLS, cfg.PORTB, cfg.ALINE) == (8, 4, 64)
        assert cfg.ROWLANES and cfg.SEQ_WC and cfg.T2


def test_all_24_value_sets_reuse_the_same_phase_addresses_and_code():
    shape = BlockShape(128, 256, 4)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20, IMEM_WORDS=32768)
    signatures = set()
    for seed in range(24):
        plan = compile_encoder_layer(synthetic_weights(shape, seed=seed), 9, 7, cfg, shape, passes=2, tile=8)
        raw = b"".join(np.asarray([i.encode() for i in p], "<u4").tobytes() for p in plan.programs)
        signatures.add((plan.input_at, plan.output_at, hashlib.sha256(raw).hexdigest()))
    assert len(signatures) == 1


def test_full_width_max_context_phase_and_memory_limits_compile():
    shape = BlockShape()
    names = {f.name for f in dataclasses.fields(Config)}
    cfg = (shared_profile_config() if {"PORTB", "ROWLANES", "ALINE", "T2"} <= names else
           Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=128,
                  DRAM_BYTES=128 << 20, TMEM_WORDS=65536, IMEM_WORDS=32768))
    p = compile_encoder_layer(synthetic_weights(shape), 376, 375, cfg, passes=2)
    assert len(p.programs) == 27
    assert p.budget()["max_tmem_live_words"] <= 65536
    assert p.budget()["max_instruction_words"] <= 32768
    assert p.image.size < 128 << 20


def test_program_and_wait_bytes_load_from_nonoverlapping_dram_slot():
    shape = BlockShape(128, 256, 4)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=4 << 20, IMEM_WORDS=32768, SEQ_WC=True)
    plan = compile_encoder_layer(synthetic_weights(shape), 9, 7, cfg, shape, passes=2, tile=8)
    device = ProgramSlotDevice(cfg)
    device.reset(plan.image.size)
    device.constants(plan.image, ((0, plan.input_at),))
    x = np.random.default_rng(58).normal(size=(9, 128)).astype(np.float32)
    device.memory[plan.input_at:plan.input_at + x.nbytes] = x.view(np.uint8).ravel()
    for program in plan.programs:
        device.phase(program, check=True)
    actual = device.read(plan.output_at, 9, 128)
    expected, _, _ = run_layer(plan, x)
    np.testing.assert_array_equal(actual, expected)
    assert device.program_bytes_uploaded > 0 and device.wait_checked > 0
    assert device.program_at >= plan.image.size


def test_program_slot_rejects_graph_overlap_and_missing_room():
    cfg = Config(D=32, DRAM_BYTES=1 << 20, IMEM_WORDS=32768, SEQ_WC=True)
    d = ProgramSlotDevice(cfg)
    with pytest.raises(ValueError, match="overlap"):
        d.reset(d.program_at + 4)
    with pytest.raises(ValueError, match="program slot"):
        ProgramSlotDevice(dataclasses.replace(cfg, DRAM_BYTES=128))


@pytest.mark.parametrize("frames,valid", [(0, 0), (8, 0), (8, 9), (377, 376)])
def test_encoder_frame_guards(frames, valid):
    with pytest.raises(ValueError):
        compile_encoder_layer({}, frames, valid, Config())
