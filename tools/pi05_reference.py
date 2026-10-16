"""The golden reference of pi0.5 (lerobot/pi05_libero_finetuned_v044) for openTPU's port:
lerobot's own PI05Policy in fp32 on the CPU, on a LIBERO-shaped synthetic observation, its
intermediate tensors saved to an .npz (docs/pi05.md 2).

    <lerobot venv>/bin/python tools/pi05_reference.py --ckpt DIR --out ref.npz [--seed 0]
        [--task "put the bowl on the plate"]

Runs in a venv with `lerobot[pi]` (torch, transformers 5), not openTPU's: the openTPU side
only reads the .npz. The observation is the dataset's shapes: two 256 x 256 RGB cameras
(`observation.images.image`, `image2`: smooth random fields in [0, 1], a fixed seed), an
8-dim state (normal(0, 0.5) in the normalized space, mapped back through the checkpoint's
MEAN_STD statistics so the policy's own normalizer gives it), a task string. The empty third
camera is the policy's (all -1, masked). The flow-matching noise [1, 50, 32] is drawn here
with the same seed and passed to sample_actions, so the device run can start from it.

Saved (fp32 unless said; L the prefix's padded length 256 x 3 + 200 = 968):
  inputs:   img0, img1 [3, 224, 224] (the policy's preprocessed pixels in [-1, 1]: after its
            resize to 224 and x * 2 - 1), state_raw [8], state_norm [8], tokens [200] int64,
            token_mask [200] bool, prompt (str), noise [50, 32]
  vision:   vis_embed0 [256, 1152] (camera 0's patch + position embedding), vis_layer0_<i>
            [256, 1152] (camera 0 after encoder layer i), vis_post0 (after post_layernorm),
            vis_feat<c> [256, 2048] (camera c's projector output, c = 0, 1, 2)
  prefix:   prefix_embs [L, 2048] (images then sqrt(2048)-scaled token embeddings),
            prefix_pad [L] bool, prefix_pos [L] int64, prefix_h<i> [L, 2048] (after
            PaliGemma layer i), prefix_k<i>, prefix_v<i> [L, 256] (layer i's cache: K after
            RoPE, V)
  expert:   times [10], x<s> [50, 32] (x_t into step s), v<s> [50, 32] (its velocity),
            exp_in<s> [50, 1024] (the suffix embedding), exp_h<s>_<i> [50, 1024] (after
            expert layer i of step s), exp_out<s> [50, 1024] (after the final adaRMS norm),
            adarms_cond<s> [1024] (time MLP output)
  actions:  actions_model [50, 32] (sample_actions' output), actions [50, 7] (unnormalized
            by the postprocessor's MEAN_STD statistics), action_mean, action_std [7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def smooth_image(rng, h: int = 256, w: int = 256) -> np.ndarray:
    """[3, h, w] in [0, 1]: a few random low-frequency waves per channel plus a little noise
    (an image-like field: SigLIP sees structure rather than white noise)."""
    y, x = np.mgrid[0:h, 0:w] / np.float32(h)
    out = np.zeros((3, h, w), np.float32)
    for c in range(3):
        f = np.zeros((h, w))
        for _ in range(6):
            kx, ky = rng.uniform(-6, 6, 2)
            f += rng.uniform(0.2, 1) * np.sin(2 * np.pi * (kx * x + ky * y) + rng.uniform(0, 6.3))
        f = (f - f.min()) / (f.max() - f.min())
        out[c] = 0.9 * f + 0.1 * rng.uniform(0, 1, (h, w))
    return np.clip(out, 0, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="put the black bowl on the plate")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401  (registers its steps)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    torch.set_grad_enabled(False)
    ckpt = Path(args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.device, cfg.dtype, cfg.compile_model, cfg.gradient_checkpointing = \
        "cpu", "float32", False, False
    policy = PI05Policy.from_pretrained(str(ckpt), config=cfg)
    policy.eval()
    model = policy.model

    # the policy's normalization statistics (MEAN_STD for state and action)
    stats = {}
    for name in ("policy_preprocessor_step_2_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        for k, v in load_file(str(ckpt / name)).items():
            stats[k] = v
    # the checkpoint's own pipelines (the normalizer's saved statistics, the tokenizer step)
    from lerobot.processor import PolicyProcessorPipeline
    pre = PolicyProcessorPipeline.from_pretrained(str(ckpt), "policy_preprocessor.json",
                                                  overrides={"device_processor": {"device": "cpu"}})

    rng = np.random.default_rng(args.seed)
    img0, img1 = smooth_image(rng), smooth_image(rng)
    mean, std = stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()
    state_norm = np.clip(rng.normal(0, 0.5, 8), -0.95, 0.95).astype(np.float32)
    state_raw = (state_norm * (std + 1e-8) + mean).astype(np.float32)
    noise = torch.from_numpy(rng.standard_normal((1, cfg.chunk_size, cfg.max_action_dim))
                             .astype(np.float32))

    batch = {"observation.images.image": torch.from_numpy(img0),
             "observation.images.image2": torch.from_numpy(img1),
             "observation.state": torch.from_numpy(state_raw),
             "task": args.task}
    batch = pre(batch)
    save = {"state_raw": state_raw, "noise": noise[0].numpy(),
            "tokens": batch[OBS_LANGUAGE_TOKENS][0].numpy().astype(np.int64),
            "token_mask": batch[OBS_LANGUAGE_ATTENTION_MASK][0].numpy().astype(bool)}
    save["state_norm"] = batch["observation.state"][0].numpy().astype(np.float32)

    images, img_masks = policy._preprocess_images(batch)
    save["img0"], save["img1"] = images[0][0].numpy(), images[1][0].numpy()

    # ---- hooks
    pg = model.paligemma_with_expert.paligemma
    ex = model.paligemma_with_expert.gemma_expert
    vt = pg.model.vision_tower.vision_model
    hooks, state = [], {"cam": 0, "step": -1}

    def keep(name):
        def h(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            save[name()] = o[0].detach().float().numpy().copy()
        return h
    hooks.append(vt.embeddings.register_forward_hook(
        lambda m, i, o: save.__setitem__("vis_embed0", o[0].numpy().copy())
        if state["cam"] == 0 else None))
    for i, layer in enumerate(vt.encoder.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, o, i=i: save.__setitem__(
                f"vis_layer0_{i}", (o[0] if isinstance(o, tuple) else o)[0].numpy().copy())
            if state["cam"] == 0 else None))
    hooks.append(vt.post_layernorm.register_forward_hook(
        lambda m, i, o: save.__setitem__("vis_post0", o[0].numpy().copy())
        if state["cam"] == 0 else None))
    for i, layer in enumerate(pg.model.language_model.layers):
        hooks.append(layer.register_forward_hook(keep(lambda i=i: f"prefix_h{i}")))
    for i, layer in enumerate(ex.model.layers):
        hooks.append(layer.register_forward_hook(
            keep(lambda i=i: f"exp_h{state['step']}_{i}")))
    hooks.append(ex.model.norm.register_forward_hook(
        lambda m, i, o: save.__setitem__(f"exp_out{state['step']}", o[0][0].numpy().copy())))

    real_embed_image = model.paligemma_with_expert.embed_image

    def embed_image(img):
        f = real_embed_image(img)
        save[f"vis_feat{state['cam']}"] = f[0].numpy().copy()
        state["cam"] += 1
        return f
    model.paligemma_with_expert.embed_image = embed_image

    real_embed_suffix = model.embed_suffix

    def embed_suffix(x_t, timestep):
        state["step"] += 1
        s = state["step"]
        out = real_embed_suffix(x_t, timestep)
        save[f"x{s}"] = x_t[0].numpy().copy()
        save[f"exp_in{s}"] = out[0][0].numpy().copy()
        save[f"adarms_cond{s}"] = out[3][0].numpy().copy()
        save.setdefault("_times", []).append(float(timestep[0]))
        return out
    model.embed_suffix = embed_suffix

    real_out = model.action_out_proj

    class OutProj(torch.nn.Module):
        def forward(self, x):
            v = real_out(x)
            save[f"v{state['step']}"] = v[0].numpy().copy()
            return v
    model.action_out_proj = OutProj()

    # prefix embeddings and cache: recomputed here exactly as sample_actions does
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    embs, pad, att = model.embed_prefix(images, img_masks, tokens, masks)
    state["cam"] = 0                       # (sample_actions embeds the images again)
    save["prefix_embs"] = embs[0].numpy().copy()
    save["prefix_pad"] = pad[0].numpy().astype(bool)
    save["prefix_pos"] = (torch.cumsum(pad, dim=1) - 1)[0].numpy().astype(np.int64)

    real_fwd = model.paligemma_with_expert.forward

    def fwd(*a, **kw):
        out = real_fwd(*a, **kw)
        pkv = out[1]
        if pkv is not None:
            for i, (k, v, *_) in enumerate(pkv):
                save[f"prefix_k{i}"] = k[0, 0].numpy().copy()
                save[f"prefix_v{i}"] = v[0, 0].numpy().copy()
        return out
    model.paligemma_with_expert.forward = fwd

    actions = model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    save["actions_model"] = actions[0].numpy().copy()
    a7 = actions[:, :, :cfg.output_features["action"].shape[0]]
    # the postprocessor's MEAN_STD unnormalization (its saved statistics; eps as the step's)
    am, asd = stats["action.mean"].numpy(), stats["action.std"].numpy()
    save["actions"] = (a7[0].numpy() * asd + am).astype(np.float32)
    save["action_mean"], save["action_std"] = am.astype(np.float32), asd.astype(np.float32)
    save["times"] = np.array(save.pop("_times"), np.float32)
    save["prompt"] = np.array(json.dumps(args.task))
    for h in hooks:
        h.remove()
    np.savez(args.out, **save)
    print(f"saved {len(save)} arrays to {args.out}; valid prefix tokens "
          f"{int(save['prefix_pad'].sum())}, actions[0] {save['actions'][0]}")


if __name__ == "__main__":
    main()
