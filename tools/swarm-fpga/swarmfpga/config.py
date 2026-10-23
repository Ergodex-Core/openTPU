"""A deliberately small, strict declarative interface."""
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32
    context: int = 32
    width: int = 16
    layers: int = 1
    heads: int = 2
    ff_width: int = 32
    learning_rate: float = 0.03
    batch_size: int = 2
    sequence_length: int = 8
    steps: int = 100
    seed: int = 7
    forward_dtype: str = "fp32"
    backward_dtype: str = "fp32"
    backend: str = "reference"
    image: str = ""

    def __post_init__(self):
        for key in ("vocab_size", "context", "width", "layers", "heads", "ff_width",
                    "batch_size", "sequence_length", "steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.width % self.heads or self.sequence_length > self.context:
            raise ValueError("width must divide by heads; sequence_length must fit context")
        if type(self.learning_rate) not in (int, float) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.forward_dtype not in ("fp32", "int8") or self.backward_dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError("unsupported forward/backward precision")
        if self.backend not in ("reference", "f2") or self.image not in ("", "burst200"):
            raise ValueError("backend must be reference or f2 with the burst200 image")
        if self.backend == "reference" and self.image:
            raise ValueError("reference backend cannot use an FPGA selector")
        if self.backend == "f2" and not self.image:
            raise ValueError("f2 requires an explicit native image")

    @classmethod
    def from_dict(cls, data):
        allowed = {"model": {"type", "vocab_size", "context", "width", "layers", "heads", "ff_width"},
                   "trainer": {"learning_rate", "batch_size", "sequence_length", "steps", "seed", "forward_dtype", "backward_dtype"},
                   "backend": {"type", "image"}}
        features = {"input_features": [{"name": "tokens", "type": "token_ids"}],
                    "output_features": [{"name": "next_token", "type": "token_ids"}]}
        if not isinstance(data, dict) or set(data) - (set(allowed) | set(features)):
            raise ValueError("unknown configuration section")
        values = {}
        for section, keys in allowed.items():
            part = data.get(section, {})
            if not isinstance(part, dict) or set(part) - keys:
                raise ValueError(f"unknown {section} option")
            if section == "model" and part.get("type", "causal_lm") != "causal_lm":
                raise ValueError("only causal_lm is implemented")
            values.update({k: v for k, v in part.items() if k != "type"})
        values["backend"] = data.get("backend", {}).get("type", "reference")
        for name, expected in features.items():
            if data.get(name, expected) != expected:
                raise ValueError(f"{name} must be {expected}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return Config.from_dict(yaml.safe_load(Path(path).read_text()))
