"""Independent tiny FP32 Mimi decoder checks on the CPU ISA runner.

The NumPy equations below do not call the decoder's implementation helpers or
need Torch. Run this file on the EC2 builder, not on the local Mac.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from opentpu.llm.personaplex_mimi_decoder import MimiDecoderImage
from test_personaplex_mimi_decoder import bits, direct_image, read_floats, tiny


def _positions_after_pair(end, context):
    """Source ring metadata, including the full-boundary delta <= 0 defect."""
    positions = []
    for slot in range(context):
        delta = slot - end % context
        position = end + delta if delta <= 0 else end + delta - context
        positions.append(position if slot < end else -1)
    return np.asarray(positions, np.int64)


def _norm(rows, gain, bias, eps):
    mean = rows.mean(axis=-1, keepdims=True)
    centered = rows - mean
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps) * gain + bias


def _rotate_and_pack(rows, positions, spec):
    """Adjacent real/imag source pairs become the resident half-split K/Q order."""
    d = spec.head_dim
    pairs = rows.reshape(2, spec.n_heads, d // 2, 2)
    exponent = np.arange(d // 2, dtype=np.float64) * (-2 * math.log(spec.rope_base) / d)
    angles = np.asarray(positions, np.float64)[:, None] * np.exp(exponent)[None, :]
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imaginary = pairs[..., 1] * cosine + pairs[..., 0] * sine
    return np.concatenate((real, imaginary), axis=-1)


def _erf_gelu(value):
    scaled = value / math.sqrt(2)
    erf = np.fromiter((math.erf(float(x)) for x in scaled.flat),
                      dtype=np.float64, count=scaled.size).reshape(scaled.shape)
    return 0.5 * value * (1 + erf)


def numpy_frames(spec, upsample, layers, latents):
    """Full two-token streaming equations and physical ring state in float64."""
    d, context = spec.head_dim, spec.context
    caches = [(np.zeros((spec.n_heads, context, d), np.float64),
               np.zeros((spec.n_heads, d, context), np.float64))
              for _ in layers]
    overlap = np.zeros((2, spec.hidden), np.float64)
    answers = []
    for frame, latent in enumerate(latents):
        source = latent.astype(np.float64)
        taps = upsample.astype(np.float64)
        up = source[None, :] * taps[:, :2].T + overlap
        overlap = source[None, :] * taps[:, 2:].T
        hidden = up.copy()
        layer_outputs = []
        cache_copies = []
        end = 2 * (frame + 1)
        positions = _positions_after_pair(end, context)
        slots = (np.arange(2) + 2 * frame) % context
        for layer_index, layer in enumerate(layers):
            weights = {name: value.astype(np.float64) for name, value in layer.items()}
            normalized = _norm(hidden, weights["norm1_weight"],
                               weights["norm1_bias"], spec.eps)
            q = _rotate_and_pack(normalized @ weights["wq"].T,
                                 (2 * frame, 2 * frame + 1), spec)
            k = _rotate_and_pack(normalized @ weights["wk"].T,
                                 (2 * frame, 2 * frame + 1), spec)
            v = (normalized @ weights["wv"].T).reshape(2, spec.n_heads, d)
            keys, values_t = caches[layer_index]
            # Native chunk processing writes both tokens before either attends.
            for token, slot in enumerate(slots):
                keys[:, slot, :] = k[token]
                values_t[:, :, slot] = v[token]
            attended = np.empty((2, spec.n_heads, d), np.float64)
            for token in range(2):
                query_position = 2 * frame + token
                visible = [slot for slot, key_position in enumerate(positions)
                           if key_position >= 0 and 0 <= query_position - key_position < context]
                assert visible, "every token must see at least one resident key"
                for head in range(spec.n_heads):
                    scores = keys[head, visible] @ q[token, head] / math.sqrt(d)
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                    attended[token, head] = probabilities @ values_t[head].T[visible]
            hidden = hidden + (attended.reshape(2, spec.hidden) @ weights["wo"].T) * weights["scale1"]
            normalized = _norm(hidden, weights["norm2_weight"],
                               weights["norm2_bias"], spec.eps)
            hidden = hidden + (_erf_gelu(normalized @ weights["w1"].T)
                               @ weights["w2"].T) * weights["scale2"]
            layer_outputs.append(hidden.copy())
            cache_copies.append((keys.copy(), values_t.copy()))
        answers.append((up, hidden.copy(), layer_outputs, cache_copies))
    return answers


def _resident_caches(image, backend, layer):
    keys, values_t = [], []
    for head in range(image.spec.n_heads):
        addresses = image.kv_heads[layer][head]
        assert set(addresses) == {"k", "vt"}
        keys.append(read_floats(backend, addresses["k"], (image.cap, image.spec.head_dim)))
        values_t.append(read_floats(backend, addresses["vt"], (image.spec.head_dim, image.cap)))
    return np.stack(keys), np.stack(values_t)


def _matrix_payloads(layers):
    """Expected raw matrix writes, including the source's Q/K row permutation."""
    layer = layers[0]
    hidden = layer["wq"].shape[0]
    # The tiny fixture has two heads. Keep this offline expectation separate
    # from the image's descriptor or payload hash implementation.
    n_heads = 2
    d = hidden // n_heads
    expected = []
    for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
        matrix = layer[name]
        if name in ("wq", "wk"):
            heads = matrix.reshape(n_heads, d, hidden)
            matrix = np.concatenate((heads[:, 0::2], heads[:, 1::2]), axis=1).reshape(matrix.shape)
        expected.append(np.ascontiguousarray(matrix, np.float32).view(np.uint8).tobytes())
    return expected


def test_fp32_decoder_payload_cache_masks_wrap_and_reset(monkeypatch):
    base, cfg, upsample, layers, latents = tiny()
    assert base.precision == "int8"
    spec = replace(base, precision="fp32")
    image, programs, backend = direct_image(spec, cfg, upsample, layers)
    assert len(programs) == 1 + spec.layers
    assert all(not program.run_args for program in programs)
    assert _positions_after_pair(4, spec.context).tolist() == [4, 1, 2, 3]
    assert _positions_after_pair(6, spec.context).tolist() == [4, 5, 6, 3]

    # Capture the uploads themselves so the test does not depend on matrix
    # descriptor fields. Every projection must be present as raw FP32 bits.
    captured = []
    original_write = backend.write
    def record_write(device, address, value):
        captured.append(np.ascontiguousarray(value).view(np.uint8).tobytes())
        return original_write(device, address, value)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "write", record_write)
        image.upload(backend, upsample, layers)
    for matrix_bytes in _matrix_payloads(layers):
        assert matrix_bytes in captured

    expected = numpy_frames(spec, upsample, layers, latents)
    observed = []
    for frame, latent in enumerate(latents):
        backend.write(0, image.latent_at, latent)
        transfers = []
        original_read, original_run = backend.read, backend.run
        def trace_read(*args):
            transfers.append("read")
            return original_read(*args)
        def trace_write(*args):
            transfers.append("write")
            return original_write(*args)
        def trace_run(batch, *args, **kwargs):
            assert len(batch) == 1 and not batch[0].run_args
            assert not args and kwargs.get("args") is None
            return original_run(batch, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "read", trace_read)
            patch.setattr(backend, "write", trace_write)
            patch.setattr(backend, "run", trace_run)
            for program in programs:
                backend.run([program])
        assert transfers == [], "decoder programs must not use host intermediate transfers"

        want_up, want_out, _, want_caches = expected[frame]
        got_up = read_floats(backend, image.io["upsampled"], (2, spec.hidden))
        got_out = read_floats(backend, image.io["output"], (2, spec.hidden))
        np.testing.assert_allclose(got_up, want_up, rtol=0, atol=2e-6)
        np.testing.assert_allclose(got_out, want_out, rtol=2e-4, atol=3e-4,
                                   err_msg=f"FP32 decoder frame {frame}")
        got_caches = [_resident_caches(image, backend, layer) for layer in range(spec.layers)]
        for layer in range(spec.layers):
            got_k, got_vt = got_caches[layer]
            want_k, want_vt = want_caches[layer]
            np.testing.assert_allclose(got_k[:, :spec.context], want_k,
                                       rtol=2e-4, atol=3e-4)
            np.testing.assert_allclose(got_vt[:, :, :spec.context], want_vt,
                                       rtol=2e-4, atol=3e-4)
            np.testing.assert_array_equal(bits(got_k[:, spec.context:]),
                                          bits(np.zeros_like(got_k[:, spec.context:])))
            np.testing.assert_array_equal(bits(got_vt[:, :, spec.context:]),
                                          bits(np.zeros_like(got_vt[:, :, spec.context:])))
        assert read_floats(backend, image.io["state"], (1,))[0] == frame + 1
        observed.append((got_up, got_out, got_caches))

    # Reset the logical stream only. Old FP32 K/VT bytes must remain resident,
    # and masks must hide them until overwritten by the new two-token frames.
    old_cache_bytes = [np.asarray(backend.read(0, at, size), np.uint8).copy()
                       for at, size in image.cache_regions]
    backend.write(0, image.io["state"], np.zeros(1, np.float32))
    backend.write(0, image.io["overlap"], np.full((2, spec.hidden), -0.0, np.float32))
    for (at, size), before in zip(image.cache_regions, old_cache_bytes):
        np.testing.assert_array_equal(np.asarray(backend.read(0, at, size), np.uint8), before)
    for frame, latent in enumerate(latents):
        backend.write(0, image.latent_at, latent)
        for program in programs:
            backend.run([program])
        np.testing.assert_array_equal(bits(read_floats(backend, image.io["upsampled"], (2, spec.hidden))),
                                      bits(observed[frame][0]))
        np.testing.assert_array_equal(bits(read_floats(backend, image.io["output"], (2, spec.hidden))),
                                      bits(observed[frame][1]))
        if 2 * (frame + 1) >= spec.context:
            for layer in range(spec.layers):
                for before, after in zip(observed[frame][2][layer],
                                         _resident_caches(image, backend, layer)):
                    np.testing.assert_array_equal(bits(after), bits(before))


def test_fp32_cache_uses_actual_head_dim_without_quantized_scale_rows():
    base, cfg, _, _, _ = tiny(max_frames=2)
    spec = replace(base, n_heads=4, precision="fp32")
    assert spec.head_dim < cfg.D
    image = MimiDecoderImage(spec, cfg, latent_at=0x1000, state_base=0x2000,
                             state_limit=0x200000, weight_base=0x200000,
                             weight_limit=0x1000000)
    assert len(image.kv_heads[0]) == spec.n_heads
    assert all(set(addresses) == {"k", "vt"} for addresses in image.kv_heads[0].values())
    assert image.cache_regions[0][1] == 2 * spec.n_heads * image.cap * spec.head_dim * 4


@pytest.mark.parametrize("invalid", ["fp16", "", None, False])
def test_unknown_decoder_precision_rejected(invalid):
    with pytest.raises(ValueError, match="precision"):
        replace(tiny(max_frames=2)[0], precision=invalid)
