"""Synthetic native/ISA streaming SEANet-to-raw-PCM checks (EC2 CPU only)."""
import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_depth_chain import DepthChainEngine
from opentpu.llm.personaplex_mimi_pcm import MimiPCMImage, MimiPCMSpec
from test_personaplex_mimi_decoder import chain_fixture


def tiny(*, ratios=(2, 2, 2, 2)):
    spec = MimiPCMSpec(hidden=64, n_filters=4, ratios=ratios, input_rows=2)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 26)
    image = MimiPCMImage(spec, cfg, input_at=0x1000, state_base=0x2000,
                         state_limit=0x400000, weight_base=0x400000,
                         weight_limit=0x2000000)
    rng = np.random.default_rng(7193)
    weights = []
    for op in image.ops:
        shape = ((op.cin, op.cout, op.kernel) if op.kind == "transpose"
                 else (op.cout, op.cin, op.kernel))
        weights.append(dict(kind=op.kind, model_index=op.model_index,
                            subpath=op.subpath,
                            weight=rng.normal(0, 0.055, shape).astype(np.float32),
                            bias=rng.normal(0, 0.025, op.cout).astype(np.float32)))
    frames = rng.normal(0, 0.8, (3, spec.input_rows, spec.hidden)).astype(np.float32)
    return spec, cfg, image, weights, frames


def native_decoder(spec, weights):
    """Strictly load only synthetic tensors into the original Moshi class."""
    import torch
    from moshi.modules.seanet import SEANetDecoder

    native = SEANetDecoder(channels=1, dimension=spec.hidden, causal=True,
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
    return native


def native_frames(spec, weights, frames):
    import torch
    native = native_decoder(spec, weights)
    stages = [[] for _ in native.model]
    handles = []
    for index, module in enumerate(native.model):
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, value, index=index:
            stages[index].append(value.detach().numpy()[0].T.copy())))
    try:
        with torch.no_grad(), native.streaming(1):
            pcm = []
            for frame in frames:
                output = native(torch.from_numpy(frame.T.copy()[None, :, :]))
                pcm.append(output.detach().numpy()[0, 0].copy())
    finally:
        for handle in handles:
            handle.remove()
    return np.stack(pcm), [np.stack(values) for values in stages]


def device_image(spec, cfg, image, weights):
    programs = image.compile()
    backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
    image.upload(backend, weights)
    return programs, backend


def read_floats(backend, address, shape):
    count = int(np.prod(shape))
    return np.asarray(backend.read(0, address, 4 * count), np.uint8).view(
        np.float32).reshape(shape).copy()


def device_frames(image, programs, backend, frames):
    outputs, stages = [], []
    for index, frame in enumerate(frames):
        backend.write(0, image.input_at, frame)
        for program in programs:
            backend.run([program])
        outputs.append(read_floats(backend, image.io["output"], (image.spec.samples,)))
        stages.append([read_floats(backend, address, (rows, channels))
                       for address, rows, channels in image.layer_outputs])
        assert read_floats(backend, image.io["frame"], (1,))[0] == index + 1
    return np.stack(outputs), [np.stack([stage[i] for stage in stages])
                               for i in range(len(image.layer_outputs))]


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


def test_three_native_streaming_frames_all_fifteen_boundaries_and_reset():
    spec, cfg, image, weights, frames = tiny()
    native_pcm, native_stages = native_frames(spec, weights, frames)
    programs, backend = device_image(spec, cfg, image, weights)
    assert len(programs) == len(image.ops) == 14
    assert all(not program.run_args for program in programs)
    assert len(image.layer_outputs) == len(native_stages) == 15
    pcm, stages = device_frames(image, programs, backend, frames)
    assert pcm.shape == native_pcm.shape == (3, spec.samples)
    for index, (actual, expected) in enumerate(zip(stages, native_stages)):
        assert actual.shape == expected.shape, index
        np.testing.assert_allclose(actual, expected, rtol=0.10, atol=0.04,
                                   err_msg=f"native SEANet stage {index}")
    np.testing.assert_allclose(pcm, native_pcm, rtol=0.10, atol=0.04)

    # The first K7 history is six raw input rows, with zeros before frame 0.
    first = image.ops[0]
    retained = read_floats(backend, first.prepared.base, (first.history, spec.hidden))
    np.testing.assert_array_equal(bits(retained), bits(frames.reshape(-1, spec.hidden)[-6:]))
    for op in image.ops:
        if op.kind == "transpose":
            assert read_floats(backend, op.seen.base, (1,))[0] == 1
            last_raw = read_floats(backend, op.raw.base + 4 * (op.rows - 1) * op.kernel * op.cout,
                                   (op.kernel * op.cout,))
            saved = read_floats(backend, op.partial.base, (op.stride, op.cout))
            np.testing.assert_array_equal(bits(saved), bits(last_raw[op.stride * op.cout:].reshape(saved.shape)))

    persistent = [(at, backend.read(0, at, size).copy()) for at, size in image.reset_regions]
    image.reset(backend)
    replay_pcm, replay_stages = device_frames(image, programs, backend, frames)
    np.testing.assert_array_equal(bits(replay_pcm), bits(pcm))
    for before, after in zip(stages, replay_stages):
        np.testing.assert_array_equal(bits(before), bits(after))
    for at, before in persistent:
        np.testing.assert_array_equal(backend.read(0, at, len(before)), before)


def test_transpose_overlap_bias_is_added_once_across_rows_and_frames():
    spec, cfg, image, weights, frames = tiny()
    for record in weights:
        record["weight"].fill(0)
        record["bias"].fill(0)
    first_transpose = weights[1]
    assert first_transpose["kind"] == "transpose"
    first_transpose["bias"].fill(0.75)
    native_pcm, native_stages = native_frames(spec, weights, frames)
    programs, backend = device_image(spec, cfg, image, weights)
    _, stages = device_frames(image, programs, backend, frames)
    assert np.all(native_stages[2] == np.float32(0.75))
    np.testing.assert_array_equal(bits(stages[2]), bits(native_stages[2]))
    assert np.isfinite(native_pcm).all()


@pytest.mark.parametrize("bias", [2.5, -2.5])
def test_final_pcm_remains_raw_float_without_clipping_or_scaling(bias):
    spec, cfg, image, weights, frames = tiny()
    for record in weights:
        record["weight"].fill(0)
        record["bias"].fill(0)
    weights[-1]["bias"].fill(bias)
    native_pcm, _ = native_frames(spec, weights, frames[:2])
    programs, backend = device_image(spec, cfg, image, weights)
    pcm, _ = device_frames(image, programs, backend, frames[:2])
    np.testing.assert_array_equal(bits(pcm), bits(native_pcm))
    np.testing.assert_array_equal(pcm, np.full_like(pcm, bias))
    assert np.max(np.abs(pcm)) > 1.0


def test_pcm_capacity_payload_order_and_lazy_compile_preflight(monkeypatch):
    spec, cfg, image, weights, _ = tiny()
    budget = image.budget()
    assert budget["required_bytes"] <= cfg.DRAM_BYTES
    assert budget["samples_per_frame"] == spec.samples == 32
    with pytest.raises(MemoryError):
        MimiPCMImage(spec, cfg, input_at=0x1000, state_base=0x2000,
                     state_limit=0x2100, weight_base=0x400000,
                     weight_limit=0x2000000)

    class Spy:
        writes = 0
        def write(self, *_args):
            self.writes += 1
    spy = Spy()
    reordered = [dict(value) for value in weights]
    reordered[0]["model_index"] = 99
    with pytest.raises(ValueError, match="order"):
        image.upload(spy, reordered)
    assert spy.writes == 0
    malformed = [dict(value) for value in weights]
    malformed[-1]["weight"] = np.full_like(malformed[-1]["weight"], np.nan)
    with pytest.raises(ValueError):
        image.upload(spy, malformed)
    assert spy.writes == 0

    def fail(_self):
        raise CompileError("synthetic PCM program preflight failure")
    monkeypatch.setattr(MimiPCMImage, "compile", fail)
    def unread():
        raise AssertionError("payload read before PCM compile")
    with pytest.raises(CompileError, match="preflight failure"):
        image.upload(spy, unread)


def test_pcm_programs_run_without_host_feedback_transfers(monkeypatch):
    spec, cfg, image, weights, frames = tiny()
    programs, backend = device_image(spec, cfg, image, weights)
    backend.write(0, image.input_at, frames[0])
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
    assert not transfers and submitted == programs
    assert read_floats(backend, image.io["output"], (spec.samples,)).shape == (spec.samples,)


def test_integrated_pcm_consumes_decoder_once_without_feedback_and_latches_failure(monkeypatch):
    chain, mimi, decoder, cfg, payload = chain_fixture()
    pcm, _, _, pcm_weights, _ = tiny()
    hidden = np.zeros(chain.hidden, np.float32)
    with DepthChainEngine(payload["projections"], payload["embeddings"], payload["layers"],
                          payload["head_bank"], spec=chain, cfg=cfg,
                          mimi_tables=payload["mimi_tables"],
                          mimi_projections=payload["mimi_projections"], mimi_spec=mimi,
                          upsample_weight=payload["upsample_weight"],
                          decoder_weights=payload["decoder_weights"],
                          decoder_spec=decoder, pcm_weights=pcm_weights,
                          pcm_spec=pcm) as engine:
        with pytest.raises(RuntimeError):
            engine.run_pcm()
        with pytest.raises(RuntimeError):
            engine.read_pcm()
        engine.run_frame(hidden, 0)
        engine.run_mimi()
        with pytest.raises(RuntimeError):
            engine.run_pcm()
        engine.run_decoder()
        with pytest.raises(RuntimeError, match="pending PCM"):
            engine.begin_frame(hidden, 0)

        original_read, original_write, original_run = (
            engine.backend.read, engine.backend.write, engine.backend.run)
        transfers, submitted = [], []
        def traced_read(*args):
            transfers.append("read")
            return original_read(*args)
        def traced_write(*args):
            transfers.append("write")
            return original_write(*args)
        def traced_run(program, *args, **kwargs):
            assert len(program) == 1 and not program[0].run_args
            assert not args and kwargs.get("args") is None
            submitted.append(program[0])
            return original_run(program, *args, **kwargs)
        monkeypatch.setattr(engine.backend, "read", traced_read)
        monkeypatch.setattr(engine.backend, "write", traced_write)
        monkeypatch.setattr(engine.backend, "run", traced_run)
        assert engine.run_pcm() is None
        assert not transfers and len(submitted) == len(engine.image.pcm.ops) == 14
        assert engine.read_pcm().shape == (pcm.samples,)
        assert engine.read_pcm_layer(14).shape == (pcm.samples, 1)
        assert engine.read_pcm_state()
        with pytest.raises(RuntimeError):
            engine.run_pcm()
        monkeypatch.setattr(engine.backend, "read", original_read)
        monkeypatch.setattr(engine.backend, "write", original_write)
        monkeypatch.setattr(engine.backend, "run", original_run)

        engine.run_frame(hidden, 0)
        engine.run_mimi(); engine.run_decoder(); engine.run_pcm()
        assert engine.decoder_pos == 2
        with pytest.raises(RuntimeError, match="limit"):
            engine.begin_frame(hidden, 0)
        engine.reset()
        assert engine.decoder_pos == 0
        with pytest.raises(RuntimeError):
            engine.read_pcm()

        engine.run_frame(hidden, 0)
        engine.run_mimi(); engine.run_decoder()
        def fail(*_args, **_kwargs):
            raise RuntimeError("synthetic resident PCM execution failure")
        monkeypatch.setattr(engine.backend, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic resident PCM"):
            engine.run_pcm()
        monkeypatch.setattr(engine.backend, "run", original_run)
        with pytest.raises(RuntimeError, match="reset required"):
            engine.run_pcm()
        with pytest.raises(RuntimeError, match="reset required"):
            engine.read_pcm()
        engine.reset()
        engine.run_frame(hidden, 0)
        engine.run_mimi(); engine.run_decoder(); engine.run_pcm()
        assert engine.read_pcm().shape == (pcm.samples,)
