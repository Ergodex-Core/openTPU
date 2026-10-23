"""Own CPU-only audit equations using installed PyTorch; no upstream code import.

Optional qualification dependency, deliberately outside the runtime adapter.
Pinned source locations and numerical policy are in docs/phonon2_conformer.md.
"""
import numpy as np

from tools.phonon2_reference import slaney_mel


def frontend_torch(pcm_s16le):
    import torch
    if len(pcm_s16le) % 2 or not 320 <= len(pcm_s16le) // 2 <= 480000:
        raise ValueError("require aligned PCM16, 320..480000 samples")
    x = torch.from_numpy(np.frombuffer(pcm_s16le, "<i2").astype(np.float32) / np.float32(32768))[None]
    pre = torch.cat((x[:, :1], x[:, 1:] - .97 * x[:, :-1]), dim=1)
    spectrum = torch.stft(pre, 512, hop_length=160, win_length=400,
                          window=torch.hann_window(400, periodic=False), return_complex=True, pad_mode="constant")
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1)).pow(2)
    log_mel = torch.log(torch.from_numpy(slaney_mel()) @ magnitude + 2 ** -24).permute(0, 2, 1)
    valid = len(pcm_s16le) // 2 // 160
    mask = torch.arange(log_mel.shape[1])[None] < valid
    masked = log_mel * mask[..., None]
    mean = masked.sum(1) / valid
    variance = ((masked - mean[:, None]) ** 2 * mask[..., None]).sum(1) / (valid - 1)
    features = (log_mel - mean[:, None]) / (torch.sqrt(variance)[:, None] + 1e-5)
    features *= mask[..., None]
    return {"preemphasis": pre[0].numpy(), "log_mel": log_mel[0].numpy(), "features": features[0].numpy(),
            "attention_mask": mask[0].numpy(), "valid_frames": valid, "stored_frames": log_mel.shape[1]}


def conformer_torch(weights, x, valid, shape):
    """Dense eval block using Torch's layer_norm/conv1d/batch_norm and literal shift."""
    import torch
    import torch.nn.functional as f
    w = {name: torch.from_numpy(value.dense() if hasattr(value, "dense") else np.array(value, dtype=np.float32))
         for name, value in weights.items()}
    x = torch.from_numpy(np.array(x, dtype=np.float32))[None]
    t, h, dim = x.shape[1], shape.hidden, shape.hidden // shape.heads
    def linear(a, name):
        return f.linear(a, w[name])
    def norm(a, name):
        return f.layer_norm(a, (h,), w[name + ".weight"], w[name + ".bias"], eps=1e-5)
    def ff(a, name):
        return a + .5 * linear(f.silu(linear(norm(a, "norm_" + name), name + ".linear1")), name + ".linear2")
    x = ff(x, "feed_forward1")
    z = norm(x, "norm_self_att")
    q, k, v = [linear(z, "self_attn." + name + "_proj").view(1, t, shape.heads, dim).transpose(1, 2) for name in ("q", "k", "v")]
    inv = 1 / (10000 ** (torch.arange(0, h, 2, dtype=torch.float32) / h))
    theta = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None] * inv
    positions = torch.stack((theta.sin(), theta.cos()), dim=-1).reshape(1, 2 * t - 1, h)
    p = linear(positions, "self_attn.relative_k_proj").view(1, 2 * t - 1, shape.heads, dim)
    bd = (q + w["self_attn.bias_v"][None, :, None, :]) @ p.permute(0, 2, 3, 1)
    bd = f.pad(bd, (1, 0)).view(1, shape.heads, -1, t)[:, :, 1:].reshape(1, shape.heads, t, 2 * t - 1)[..., :t]
    bd *= dim ** -.5
    mask = (torch.arange(t)[:, None] < valid) & (torch.arange(t)[None, :] < valid)
    bd.masked_fill_(~mask[None, None], torch.finfo(torch.float32).min)
    scores = ((q + w["self_attn.bias_u"][None, :, None, :]) @ k.transpose(-1, -2)) * (dim ** -.5) + bd
    attention = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(1, t, h)
    x = x + linear(attention, "self_attn.o_proj")
    glu = f.glu(linear(norm(x, "norm_conv"), "conv.pointwise_conv1"), dim=-1).transpose(1, 2)
    glu[:, :, valid:] = 0
    conv = f.conv1d(glu, w["conv.depthwise_conv.weight"], padding=4, groups=h)
    conv = f.batch_norm(conv, w["conv.norm.running_mean"], w["conv.norm.running_var"], w["conv.norm.weight"], w["conv.norm.bias"], training=False, eps=1e-5)
    x = x + linear(f.silu(conv).transpose(1, 2), "conv.pointwise_conv2")
    return norm(ff(x, "feed_forward2"), "norm_out")[0].numpy()
