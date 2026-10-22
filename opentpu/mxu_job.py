"""Proposed W4A16 logical ISA interface; no binary opcode or device capability.

MM already owns opcode 0x20 and WF=3 means the existing int8-activation
NVFP4 path. This separate immutable descriptor cannot be encoded as an Instr.
Validation checks structure/ranges only; a driver must authorize memory mappings.
"""
from dataclasses import asdict, dataclass, fields
from hashlib import sha256

from .nvfp4 import STRICT_GRADUAL, STRICT_FTZ, fp32_value, NumericalFault

_COMMON = ("W4A16;BF16->FP32;E2M1;K16;E4M3FN unsigned finite;global unsigned finite FP32;"
           "even-low-nibble;tail=+0;tree16 adjacent pairs;RNE32 each product/add;"
           "block multiply;increasing K16 accumulator from +0;global multiply last;"
           "overflow=fail;nonfinite=fail;post=none;RHT=bypass;v1;")
PROFILE_SPECS = {STRICT_GRADUAL: _COMMON + "subnormal=gradual;",
                 STRICT_FTZ: _COMMON + "subnormal=DAZ-before-op,FTZ-after-RNE;"}
PROFILE_DIGESTS = {p: sha256(s.encode("ascii")).hexdigest() for p, s in PROFILE_SPECS.items()}
SOFTWARE_CAPABILITIES = tuple(PROFILE_DIGESTS.items())


def _integer(v, hi, name, lo=0):
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(f"{name} outside [{lo}, {hi}]")


@dataclass(frozen=True)
class Buffer:
    address: int
    span: int
    stride: int

    def validate(self, rows, row_bytes, alignment, name):
        for k in ("address", "span", "stride"):
            _integer(getattr(self, k), (1 << 64) - 1, name + "." + k)
        if self.address % alignment or self.stride % alignment:
            raise ValueError(name + ": misaligned address/stride")
        if self.stride < row_bytes or self.span < (rows - 1) * self.stride + row_bytes:
            raise ValueError(name + ": short stride/span")
        if self.address + self.span > 1 << 64:
            raise ValueError(name + ": address64 span overflow")


@dataclass(frozen=True)
class W4A16Job:
    a: Buffer
    w: Buffer
    scales: Buffer
    y: Buffer
    tokens: int
    outputs: int
    k: int
    global_bits: int
    profile: str = STRICT_GRADUAL
    profile_digest: str = PROFILE_DIGESTS[STRICT_GRADUAL]
    abi_major: int = 1
    abi_minor: int = 0
    operation: str = "MX_LAUNCH"
    weight_format: str = "E2M1_K16_E4M3FN"
    activation_format: str = "BF16"
    output_format: str = "FP32"
    scale_direction: str = "multiply"
    transform: str = "bypass"
    post_operation: str = "NONE"
    flags: int = 0
    reserved: int = 0
    context: int = 0
    generation: int = 0
    request: int = 0
    position: int = 0
    epoch: int = 0
    dependencies: tuple = ()

    def __post_init__(self):
        for name, want in (("abi_major", 1), ("abi_minor", 0), ("flags", 0), ("reserved", 0)):
            _integer(getattr(self, name), want, name, want)
        for name, want in (("operation", "MX_LAUNCH"), ("weight_format", "E2M1_K16_E4M3FN"),
                           ("activation_format", "BF16"), ("output_format", "FP32"),
                           ("scale_direction", "multiply"), ("transform", "bypass"),
                           ("post_operation", "NONE")):
            if getattr(self, name) != want:
                raise ValueError("unsupported " + name)
        if not isinstance(self.profile, str) or self.profile not in PROFILE_DIGESTS:
            raise ValueError("unsupported complete numerical capability tuple")
        if self.profile_digest != PROFILE_DIGESTS[self.profile]:
            raise ValueError("profile digest mismatch")
        for name in ("tokens", "outputs", "k"):
            _integer(getattr(self, name), (1 << 32) - 1, name, 1)
        for name in ("context", "generation", "request", "position", "epoch"):
            _integer(getattr(self, name), (1 << 64) - 1, name)
        if type(self.dependencies) is not tuple:
            raise ValueError("dependencies must be an immutable tuple")
        for dep in self.dependencies:
            _integer(dep, (1 << 64) - 1, "dependency")
        _integer(self.global_bits, (1 << 32) - 1, "global_bits")
        if self.global_bits >> 31 or fp32_value(self.global_bits) < 0:
            raise NumericalFault("global scale must be unsigned finite FP32")
        for name, rows, width, align in (("a", self.tokens, 2 * self.k, 2),
                                       ("w", self.outputs, (self.k + 1) // 2, 1),
                                       ("scales", self.outputs, (self.k + 15) // 16, 1),
                                       ("y", self.tokens, 4 * self.outputs, 4)):
            b = getattr(self, name)
            if type(b) is not Buffer:
                raise ValueError(name + " must be a Buffer")
            b.validate(rows, width, align, name)
        for b in (self.a, self.w, self.scales):
            if max(b.address, self.y.address) < min(b.address + b.span,
                                                  self.y.address + self.y.span):
                raise ValueError("private output aliases an input allocation")

    def to_mapping(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, value):
        """Reject unknown fields before constructing the immutable snapshot."""
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("unknown job layout/fields")
        v = dict(value)
        try:
            for name in ("a", "w", "scales", "y"):
                b = v[name]
                if type(b) is not dict or set(b) != {"address", "span", "stride"}:
                    raise ValueError("unknown buffer layout")
                v[name] = Buffer(**b)
            if "dependencies" in v:
                if type(v["dependencies"]) not in (list, tuple):
                    raise ValueError("invalid dependencies")
                v["dependencies"] = tuple(v["dependencies"])
            return cls(**v)
        except (KeyError, TypeError) as e:
            raise ValueError("incomplete or malformed job") from e
