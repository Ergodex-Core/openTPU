"""Exact 0.6B CustomVoice contract and a bounded text-projection compiler slice.

No production generation, checkpoint fetching, CPU/GPU model fallback, board
transport, or external API is provided. `synthesize` always fails closed until
the complete text-to-waveform graph has a validated FPGA implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SPEAKERS = ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden",
            "Ono_Anna", "Sohee")
LANGUAGES = ("Auto", "Chinese", "English", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian")
BLOCKERS = (
    "device_text_embedding_and_prompt_assembly",
    "talker_28_layer_transformer_and_codec_head",
    "code_predictor_5_layers_and_15_codebook_heads",
    "device_sampling_rng_and_generation_control",
    "rvq_and_audio_decoder_transformer",
    "causal_convolution_transpose_convolution_and_stream_state",
    "decoder_exact_gelu_and_snakebeta",
    "checkpoint_precision_audio_quality_and_shared_image_validation",
)


class FPGAOnlyUnavailable(RuntimeError):
    """The complete model cannot execute under the required FPGA-only policy."""


@dataclass(frozen=True)
class TTSRequest:
    text: str
    speaker: str = "Ryan"
    language: str = "English"
    instruct: str | None = None
    stream: bool = False

    def validated(self) -> "TTSRequest":
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a nonempty string")
        if self.instruct not in (None, ""):
            raise ValueError("0.6B CustomVoice does not support style instructions")
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        def choice(value, values, name):
            if isinstance(value, str):
                for canonical in values:
                    if value.casefold() == canonical.casefold():
                        return canonical
            raise ValueError(f"unsupported {name}: {value!r}")
        return replace(self, speaker=choice(self.speaker, SPEAKERS, "speaker"),
                       language=choice(self.language, LANGUAGES, "language"))


def portal_metadata() -> dict:
    """JSON-compatible catalog data; availability is independent of hardware discovery."""
    return {
        "id": MODEL_ID, "revision": MODEL_REVISION, "task": "text-to-speech",
        "display_name": "Qwen3-TTS 0.6B CustomVoice", "license": "Apache-2.0",
        "execution_target": "fpga-only", "status": "blocked", "enabled": False,
        "readiness": "text_projection_slice_only", "hardware_validated": False,
        "speakers": list(SPEAKERS), "languages": list(LANGUAGES),
        "style_instructions": False, "voice_cloning": False,
        "streaming_available": False, "sample_rate_hz": 24000,
        "channels": 1, "codec_frames_per_second": 12.5,
        "samples_per_codec_frame": 1920, "codebooks_per_frame": 16,
        "implemented_slices": ["text_projection_int8_silu_fp32"],
        "blockers": list(BLOCKERS), "fallback": None,
        "reference_url": f"https://huggingface.co/{MODEL_ID}",
    }


def synthesize(request: TTSRequest):
    """Fail before any download, model execution, or device/service side effect."""
    request.validated()
    raise FPGAOnlyUnavailable(
        f"{MODEL_ID}: complete FPGA text-to-waveform inference is unavailable; "
        "only the text projection compiler slice exists. No host fallback. "
        "Remaining stages: " + ", ".join(BLOCKERS))


@dataclass(frozen=True)
class ProjectionSpec:
    input_size: int = 2048
    intermediate_size: int = 2048
    output_size: int = 1024

    @classmethod
    def from_config(cls, config: dict) -> "ProjectionSpec":
        """Accept only the audited variant and its actual projection semantics."""
        talker = config.get("talker_config", {})
        expected = {
            "model_type": "qwen3_tts", "tts_model_size": "0b6",
            "tts_model_type": "custom_voice", "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        }
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("expected Qwen3-TTS 12Hz 0.6B CustomVoice configuration")
        if (talker.get("text_hidden_size"), talker.get("hidden_size"),
                talker.get("hidden_act")) != (2048, 1024, "silu"):
            raise ValueError("unsupported text projection dimensions or activation")
        return cls()

    def compile(self, inputs, weights, cfg):
        """Compile diagnostic input embeddings into device projection programs/images.

        Checkpoint conversion/packing happens offline on the host; input embeddings
        are supplied by the test harness. This API is intentionally not exposed as
        synthesis. Device embedding lookup and the rest of TTS remain unimplemented.
        Only int8 weights/activations and fp32 biases/SiLU are implemented here.
        """
        import numpy as np
        from ..kernels.tts_projection import text_projection
        from ..runtime import Input, Output, Weight, compile_kernel

        sizes = (self.input_size, self.intermediate_size, self.output_size)
        if any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError("projection dimensions must be positive integers")
        if (self.input_size % cfg.D or self.intermediate_size % (cfg.S * cfg.D)
                or self.output_size % cfg.S):
            raise ValueError("projection dimensions do not fit device block/shard alignment")
        if max(self.input_size, self.intermediate_size) > cfg.D * cfg.ACT_BLOCKS:
            raise ValueError("projection exceeds ACT RAM inner dimension")
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_size or not 0 < x.shape[0] <= cfg.MCOLS:
            raise ValueError("inputs must be [1..MCOLS, input_size]")
        prefix = "talker.text_projection."
        shapes = {"linear_fc1.weight": (self.intermediate_size, self.input_size),
                  "linear_fc1.bias": (self.intermediate_size,),
                  "linear_fc2.weight": (self.output_size, self.intermediate_size),
                  "linear_fc2.bias": (self.output_size,)}
        arrays = {}
        for name, shape in shapes.items():
            value = np.asarray(weights[prefix + name], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{prefix + name}: expected {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{prefix + name}: non-finite checkpoint values")
            arrays[name] = value
        if not np.isfinite(x).all():
            raise ValueError("non-finite input embeddings")
        return compile_kernel(
            text_projection, cfg, x=Input(x),
            fc1=Weight(arrays["linear_fc1.weight"], shard=0, fmt="int8"),
            bias1=Input(arrays["linear_fc1.bias"]),
            fc2=Weight(arrays["linear_fc2.weight"], shard=0, fmt="int8"),
            bias2=Input(arrays["linear_fc2.bias"]),
            out=Output((x.shape[0], self.output_size)))
