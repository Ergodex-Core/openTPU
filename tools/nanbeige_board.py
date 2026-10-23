"""Pinned Nanbeige board contracts and tokenizer helpers; no hybrid inference."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from opentpu.target import arch_hash, config


PROFILE = "f2-hbm:p4rl8a64c150"
QUALIFIED_PROFILES = {
    ("f2-hbm", "p4rl8a64c150"): {
        "arch_hash": 3714814630, "core_khz": 150000, "platform": 5,
        "channels": 8, "channel_bytes": 536870912, "build_id": 0x417243F1,
        "image": "agfi-0001de1dd3df8e21e"},
    ("burst200", ""): {
        "arch_hash": 1341988375, "core_khz": 200000, "platform": 3,
        "channels": 2, "channel_bytes": 2147483648, "build_id": 0x7E42A626,
        "image": "agfi-0d06cb48544a02106"},
}


def validate_profile(profile):
    expected = QUALIFIED_PROFILES.get((profile.name, profile.variant))
    if expected is None or profile.platform.get("board") != "f2":
        raise ValueError("strict Nanbeige requires a pinned F2 profile")
    if arch_hash(config(profile)) != expected["arch_hash"] or \
            profile.core_khz != expected["core_khz"] or profile.platform_id != expected["platform"] or \
            any(profile.platform[k] != expected[k2] for k, k2 in
                (("channels_per_slice", "channels"), ("channel_bytes", "channel_bytes"))):
        raise ValueError("strict Nanbeige profile differs from the pinned architecture/clock/memory")
    return dict(expected)


def validate_board(info, actual, profile):
    expected = validate_profile(profile)
    wanted = config(profile)
    differences = {k: {"expected": v, "actual": asdict(actual)[k]}
                   for k, v in asdict(wanted).items() if asdict(actual)[k] != v}
    if differences or arch_hash(actual) != expected["arch_hash"]:
        raise ValueError(f"loaded FPGA architecture does not match pinned profile: {differences}")
    for key in ("core_khz", "build_id"):
        if info.get(key) != expected[key]:
            raise ValueError(f"loaded FPGA {key} differs from pinned profile")
    for key in ("platform", "channels", "channel_bytes", "hash"):
        value = expected["arch_hash"] if key == "hash" else expected[key]
        if (info.get("cfg") or {}).get(key) != value:
            raise ValueError(f"loaded FPGA {key} differs from pinned profile")
    if not info.get("calibrated"):
        raise ValueError("FPGA memory is not calibrated")
    if info.get("running"):
        raise RuntimeError("FPGA is already running; refusing to overwrite its model")


class BoardSession:
    """Canonical transport ownership and requalification immediately before upload."""
    def __init__(self, dev, profile):
        from opentpu.host.board import open_transport
        validate_profile(profile)
        self.profile = profile
        self.transport = open_transport(dev)
        self.backend = None
        try:
            self.info, self.cfg = self._probe()
        except BaseException:
            self.close()
            raise

    def _probe(self):
        from opentpu.host.board import Board, device_config
        info = Board(self.transport, lock=False, calibrate=False).info()
        cfg = device_config(info)
        validate_board(info, cfg, self.profile)
        return info, cfg

    def factory(self, cfg, images):
        from opentpu.host.board import BoardBackend
        if cfg != self.cfg:
            raise ValueError("backend configuration differs from the qualified profile")
        self.info, self.cfg = self._probe()
        self.backend = BoardBackend(cfg, images, transport=self.transport,
                                    model="Nanbeige4.2 strict W8A8", own_transport=False)
        return self.backend

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            self.transport.close()


def _tokenize(model_dir, prompt):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True,
                                               local_files_only=True)
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, preserve_thinking=False,
                                           return_dict=False)
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    return tokenizer, [int(t) for t in tokens]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
