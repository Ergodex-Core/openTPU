"""The golden reference of RhinoVLA V3 for openTPU's port: the repo's own PyTorch model
(HuixiAI/RhinoVLA, rhinovla/model/framework/RhinoVLA.py predict_action) in fp32 on the CPU,
on a synthetic observation, its intermediate tensors saved to an .npz (docs/rhinovla.md 2).

    python tools/rhinovla_reference.py --repo DIR --tiny --out ref.npz [--seed 0]
    python tools/rhinovla_reference.py --repo DIR --ckpt rhinovla_pretrain.ckpt --out ref.npz
        [--config configs/training/demo_full_finetune.yaml] [--task "..."]

--tiny builds the repo's model at opentpu.llm.rhinovla.TINY (the assets' config.json with
the tiny sizes, in a temporary copy of rhinovla/assets/qwen3_vl_processor) and loads
rhinovla.random_weights(seed) into it strictly: every tensor name of the port is checked
against the model's. --ckpt builds the configured model (the yaml's action_expert) and loads
the checkpoint through the repo's own loader (TrainerUtils.load_pretrained_backbones), then
casts it to fp32. The repo is imported from --repo with its two GPU-side imports stubbed
(qwen_vl_utils.process_vision_info: the PIL images of the messages; accelerate.logging).

The observation: `views` smooth random RGB images (image x image, the configs' 256; the tiny
Spec's 64), a 96-D state (normal(0, 0.5)) with every slot live, the task string, the noise
[chunk, 96] drawn here (seed) and given to the model (sample_noise replaced), the action mask
all ones (--mask-dims N: the first N slots live, the rest 0, as a robot's mapping would set).

Saved (fp32 unless said; L the prefix length, V the views, N the patches a view, M = N / 4):
  inputs:   img<v> [3, image, image] (pixels in [-1, 1], as the processor scales them), ids
            [L] int64 (the processor's token ids, the image pads in place), state [96],
            state_mask, action_mask [96], noise [chunk, 96], task (str), times [steps]
  vision:   vis_embed [V N, C] (after the patch and position embedding), vis_layer<i> [V N,
            C] (after block i), vis_deep<k> [V M, hidden] (deepstack merger k), vis_merged
            [V M, hidden] (the merger: the prefix's image rows)
  prefix:   prefix_pos [3, L] int64 (the mRoPE positions), prefix_h<i> [L, hidden] (after
            text layer i, with the deepstack rows added), prefix_k<i>, prefix_v<i>
            [L, n_kv, d] (layer i's cache), rope_delta int64
  expert:   x<s> [chunk, 96] (x_t into step s), v<s> (its velocity), exp_in<s> [rows, E]
            (the suffix embedding), exp_h<s>_<i> (after expert layer i), exp_out<s> (after
            the final adaRMS), cond<s> [E] (the time embedding), suffix_pos<s> [3, rows]
  actions:  actions [chunk, 96] (predict_action's output: normalized slots)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.llm import rhinovla as RV  # noqa: E402

VIEWS = (("top_head", "rgb"), ("hand_left", "rgb"), ("hand_right", "rgb"))


def smooth_image(rng, h: int, w: int) -> np.ndarray:
    """[h, w, 3] uint8: a few random low-frequency waves per channel plus a little noise."""
    y, x = np.mgrid[0:h, 0:w] / np.float32(h)
    out = np.zeros((3, h, w), np.float32)
    for c in range(3):
        f = np.zeros((h, w))
        for _ in range(6):
            kx, ky = rng.uniform(-6, 6, 2)
            f += rng.uniform(0.2, 1) * np.sin(2 * np.pi * (kx * x + ky * y) + rng.uniform(0, 6.3))
        f = (f - f.min()) / (f.max() - f.min())
        out[c] = 0.9 * f + 0.1 * rng.uniform(0, 1, (h, w))
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)


class Cfg(dict):
    """An attribute dict standing in for the repo's OmegaConf config."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        v = super().get(k, default)
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def stub_imports() -> None:
    """The repo's imports that need a GPU-side package: qwen_vl_utils.process_vision_info
    (the messages' PIL images, in order; no resize: the images are already at the pixel
    budget) and accelerate.logging.get_logger."""
    qvu = types.ModuleType("qwen_vl_utils")

    def process_vision_info(messages, **kw):
        imgs = []
        for m in messages:
            for turn in m:
                for c in turn["content"]:
                    if c.get("type") == "image":
                        imgs.append(c["image"])
        return imgs, None
    qvu.process_vision_info = process_vision_info
    sys.modules.setdefault("qwen_vl_utils", qvu)
    try:
        import accelerate.logging  # noqa: F401
        return
    except ImportError:
        pass
    import importlib.machinery
    import transformers  # noqa: F401  (its import probes accelerate: before the stub)
    acc = types.ModuleType("accelerate")
    accl = types.ModuleType("accelerate.logging")
    accl.get_logger = logging.getLogger
    acc.logging = accl
    acc.__spec__ = importlib.machinery.ModuleSpec("accelerate", None)
    accl.__spec__ = importlib.machinery.ModuleSpec("accelerate.logging", None)
    sys.modules.setdefault("accelerate", acc)
    sys.modules.setdefault("accelerate.logging", accl)


def training_config(spec: RV.Spec, assets: str, yaml_path=None) -> Cfg:
    base = {}
    if yaml_path:
        import yaml
        base = yaml.safe_load(Path(yaml_path).read_text())
    cfg = Cfg(json.loads(json.dumps(base)))
    cfg.setdefault("framework", {})
    cfg["framework"]["name"] = "RhinoVLA"
    cfg["framework"]["qwenvl"] = dict(base_vlm=assets, attn_implementation="eager",
                                      freeze_qwen=True, image_min_pixels=spec.image ** 2,
                                      image_max_pixels=spec.image ** 2)
    ae = dict(cfg["framework"].get("action_expert", {}))
    ae.update(action_dim=spec.action_dim, state_dim=spec.state_dim, action_horizon=spec.chunk,
              width=spec.e_hidden, depth=spec.e_layers, mlp_dim=spec.e_ffn, use_state_token=True,
              use_mask_condition=spec.mask_condition, use_adarms=True,
              prefix_layer_offset=spec.offset, num_inference_timesteps=spec.steps,
              init_from_qwen_last_n_layers=False)
    cfg["framework"]["action_expert"] = ae
    cfg.setdefault("datasets", {})
    vd = dict(cfg["datasets"].get("vla_data", {}))
    vd.update(image_size=[spec.image, spec.image], prompt_template="v2",
              use_view_registry_prompt=True,
              view_role_vocab=[r for r, _ in VIEWS], view_modality_vocab=["rgb", "fisheye"])
    vd.pop("CoT_prompt", None)
    cfg["datasets"]["vla_data"] = vd
    cfg.setdefault("trainer", {})
    cfg["trainer"]["enable_gradient_checkpointing"] = False
    return cfg


def build_model(repo: str, spec: RV.Spec, tiny: bool, ckpt=None, yaml_path=None, seed: int = 0):
    """The repo's RhinoVLA at spec, fp32 on the CPU: random_weights(seed) loaded strictly
    (tiny) or the checkpoint through the repo's loader."""
    import torch
    sys.path.insert(0, repo)
    stub_imports()
    from rhinovla.model.framework.RhinoVLA import RhinoVLA
    assets = str(Path(repo) / "rhinovla" / "assets" / "qwen3_vl_processor")
    tmp = None
    if tiny:
        tmp = tempfile.mkdtemp(prefix="rhino_tiny_")
        for f in Path(assets).iterdir():
            shutil.copy(f, tmp)
        base = json.loads((Path(assets) / "config.json").read_text())
        (Path(tmp) / "config.json").write_text(json.dumps(RV.tiny_config(spec, base), indent=1))
        assets = tmp
    torch.manual_seed(seed)
    model = RhinoVLA(training_config(spec, assets, yaml_path))
    model = model.float().eval()
    if tiny:
        W = RV.random_weights(spec, np.random.default_rng(seed))
        sd = {k: torch.from_numpy(v) for k, v in W.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("lm_head.weight")]
        if missing or unexpected:
            raise SystemExit(f"tensor names: missing {missing[:10]}, unexpected {unexpected[:10]}")
        with torch.no_grad():                   # the tied head is the table (as the 2B's)
            model.qwen.model.lm_head.weight = model.qwen.model.model.language_model.embed_tokens.weight
    else:
        from rhinovla.training.trainer_utils.trainer_tools import TrainerUtils
        TrainerUtils.load_pretrained_backbones(model, ckpt)
        model = model.float().eval()
    return model, tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the RhinoVLA repository (branch main, V3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--config", help="the training yaml (the action_expert section)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="put the black bowl on the plate")
    ap.add_argument("--mask-dims", type=int, default=0, help="live action slots (0: all 96)")
    ap.add_argument("--save-weights", help="the tiny model's tensors to this .safetensors")
    args = ap.parse_args()
    import torch
    from PIL import Image
    torch.set_grad_enabled(False)
    if args.tiny:
        spec = RV.TINY
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt or --tiny")
        assets = str(Path(args.repo) / "rhinovla" / "assets" / "qwen3_vl_processor")
        spec = RV.Spec.from_config(assets, args.config)
    model, tmp = build_model(args.repo, spec, args.tiny, args.ckpt, args.config, args.seed)
    rng = np.random.default_rng(args.seed)
    imgs = [smooth_image(rng, spec.image, spec.image) for _ in range(spec.views)]
    state = np.clip(rng.normal(0, 0.5, spec.state_dim), -0.95, 0.95).astype(np.float32)
    noise = rng.standard_normal((spec.chunk, spec.action_dim)).astype(np.float32)
    amask = np.ones(spec.action_dim, np.float32)
    if args.mask_dims:
        amask[args.mask_dims:] = 0
    smask = np.ones(spec.state_dim, np.float32)
    save = {"state": state, "state_mask": smask, "action_mask": amask, "noise": noise,
            "task": np.array(json.dumps(args.task))}
    for v, im in enumerate(imgs):
        save[f"img{v}"] = ((im.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)

    # ---- hooks
    qm = model.qwen.model.model
    vis, lm, ex = qm.visual, qm.language_model, model.action_expert
    hooks, st = [], {"step": -1}

    def keep(name, first=False):
        def h(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o[0] if first else o
            save[name()] = o.detach().float().numpy().copy()
        return h

    def keep_in(name, batch=True):
        def h(_m, inp):
            x = inp[0][0] if batch else inp[0]
            save[name()] = x.detach().float().numpy().copy()
        return h
    hooks.append(vis.patch_embed.register_forward_hook(
        lambda m, i, o: save.__setitem__("_patch", o.detach().numpy().copy())))
    for i, blk in enumerate(vis.blocks):
        hooks.append(blk.register_forward_pre_hook(keep_in(lambda i=i: "vis_embed" if i == 0
                                                           else f"vis_layer{i - 1}", False)))
        hooks.append(blk.register_forward_hook(keep(lambda i=i: f"vis_layer{i}")))
    for k, mg in enumerate(vis.deepstack_merger_list):
        hooks.append(mg.register_forward_hook(keep(lambda k=k: f"vis_deep{k}")))
    hooks.append(vis.merger.register_forward_hook(keep(lambda: "vis_merged")))
    for i, layer in enumerate(lm.layers):
        if i:
            hooks.append(layer.register_forward_pre_hook(keep_in(lambda i=i: f"prefix_h{i - 1}")))
    hooks.append(lm.norm.register_forward_pre_hook(keep_in(lambda: f"prefix_h{len(lm.layers) - 1}")))
    for i, layer in enumerate(ex.transformer.layers):
        hooks.append(layer.register_forward_hook(keep(lambda i=i: f"exp_h{st['step']}_{i}", True)))
    hooks.append(ex.transformer.norm.register_forward_hook(
        lambda m, i, o: save.__setitem__(f"exp_out{st['step']}", o[0][0].numpy().copy())))

    real_embed = ex.io.embed_suffix

    def embed_suffix(state_, x_t, timestep):
        st["step"] += 1
        s = st["step"]
        out = real_embed(state_, x_t, timestep)
        save[f"x{s}"] = x_t[0].numpy().copy()
        save[f"exp_in{s}"] = out[0][0].numpy().copy()
        save[f"cond{s}"] = out[2][0].numpy().copy()
        save.setdefault("_times", []).append(float(timestep[0]))
        return out
    ex.io.embed_suffix = embed_suffix
    real_decode = ex.io.decode_actions

    def decode(h):
        v = real_decode(h)
        save[f"v{st['step']}"] = v[0].numpy().copy()
        return v
    ex.io.decode_actions = decode
    import rhinovla.model.modules.action_expert as AEM
    real_pos = AEM.build_suffix_position_ids

    def pos_ids(*a, **kw):
        p = real_pos(*a, **kw)
        save[f"suffix_pos{st['step']}"] = p[:, 0].numpy().astype(np.int64)
        return p
    AEM.build_suffix_position_ids = pos_ids
    real_encode = model._encode_prefix

    def encode(*a, **kw):
        pkv, pmask, delta = real_encode(*a, **kw)
        for i in range(len(lm.layers)):
            kk, vv = AEM.get_cache_layer(pkv, i)
            save[f"prefix_k{i}"] = kk[0].permute(1, 0, 2).numpy().copy()
            save[f"prefix_v{i}"] = vv[0].permute(1, 0, 2).numpy().copy()
        save["rope_delta"] = np.int64(delta[0, 0].item())
        return pkv, pmask, delta
    model._encode_prefix = encode
    real_fwd = qm.forward

    def fwd(*a, **kw):
        save["ids"] = kw["input_ids"][0].numpy().astype(np.int64)
        out = real_fwd(*a, **kw)
        return out
    qm.forward = fwd
    real_lm = lm.forward

    def lmf(*a, **kw):
        save["prefix_pos"] = kw["position_ids"][:, 0].numpy().astype(np.int64)
        return real_lm(*a, **kw)
    lm.forward = lmf
    model.sample_noise = lambda shape, device: torch.from_numpy(noise)[None].to(device)

    ex_ = {"image": [Image.fromarray(im) for im in imgs], "lang": args.task,
           "view_roles": [r for r, _ in VIEWS][:spec.views],
           "view_modalities": [m for _, m in VIEWS][:spec.views],
           "state": state, "state_mask": smask, "action_mask": amask}
    actions = model.predict_action([ex_])
    save["actions"] = actions[0].numpy().copy()
    save["times"] = np.array(save.pop("_times"), np.float32)
    save.pop("_patch", None)
    for h in hooks:
        h.remove()
    np.savez(args.out, **save)
    print(f"saved {len(save)} arrays to {args.out}; prefix {len(save['ids'])} tokens, "
          f"rope delta {int(save['rope_delta'])}, actions[0, :4] {save['actions'][0, :4]}")
    if args.save_weights:
        from safetensors.torch import save_file
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()
                   if not k.endswith("lm_head.weight")}, args.save_weights)
        print(f"weights saved to {args.save_weights}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
