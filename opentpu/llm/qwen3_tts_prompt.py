"""Source-format CustomVoice prefill schedule; token/control processing only.

Learned text projection and codec embeddings belong to the device kernels.
Supports the source's non-streaming text prefill, not audio generation.
"""
from dataclasses import dataclass
from .qwen3_tts import ProjectionSpec


@dataclass(frozen=True)
class PromptRow:
    text_id: int
    codec_id: int
    kind: str = "aligned"


def custom_voice_prefill(input_ids, config, *, speaker="Ryan", language="English"):
    """Exact non_streaming_mode=True, fixed-speaker/no-instruct source schedule."""
    ProjectionSpec.from_config(config)
    ids = list(input_ids)
    role, suffix = [151644, 77091, 198], [151645, 198, 151644, 77091, 198]
    if len(ids) <= 8 or ids[:3] != role or ids[-5:] != suffix:
        raise ValueError("expected source assistant text wrapper with a nonempty body")
    c = config["talker_config"]
    if any(type(i) is not int or not 0 <= i < c["text_vocab_size"] for i in ids):
        raise ValueError("invalid text token ID")
    speaker, language = speaker.lower(), language.lower()
    if speaker not in c["spk_id"] or language not in ("auto", *c["codec_language_id"]):
        raise ValueError("unsupported speaker/language")
    language_id = None if language == "auto" else c["codec_language_id"][language]
    dialect = c["spk_is_dialect"][speaker]
    if language in ("auto", "chinese") and dialect is not False:
        language_id = c["codec_language_id"][dialect]
    codec = ([c["codec_nothink_id"], c["codec_think_bos_id"], c["codec_think_eos_id"]]
             if language_id is None else
             [c["codec_think_id"], c["codec_think_bos_id"], language_id, c["codec_think_eos_id"]])
    codec += [c["spk_id"][speaker], c["codec_pad_id"], c["codec_bos_id"]]
    rows = [PromptRow(i, 0, "role") for i in role]
    text = [config["tts_pad_token_id"]] * (len(codec) - 2) + [config["tts_bos_token_id"]]
    rows += [PromptRow(t, k) for t, k in zip(text, codec[:-1])]
    rows += [PromptRow(t, c["codec_pad_id"]) for t in ids[3:-5] + [config["tts_eos_token_id"]]]
    rows.append(PromptRow(config["tts_pad_token_id"], c["codec_bos_id"]))
    return tuple(rows)
