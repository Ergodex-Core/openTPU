"""Validate local PersonaPlex service assets and prevent implicit Hub downloads.

The NVIDIA server always requests its config.json through hf_hub_download, even
when all runtime paths are supplied. ``offline_hub`` resolves that single request
to the provided local config and rejects every other Hub request. It neither
downloads assets nor substitutes synthetic model metadata.

The caller must provide the upstream config, original Mimi codec, browser build,
and voice prompts. Validation checks local structure, not checkpoint authenticity
or compatibility; model loading still performs those runtime checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path


UPSTREAM_REPO = "nvidia/personaplex-7b-v1"
MIMI_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_FILENAME = "tokenizer_spm_32k_3.model"


class AssetError(ValueError):
    """A required local service asset is absent or structurally invalid."""


class OfflineDownloadError(RuntimeError):
    """An upstream component requested a Hub asset outside the local allowlist."""


def _directory(value, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise AssetError(f"{label} must be an existing local directory: {path}")
    return path


def _file(path: Path, label: str) -> Path:
    # Existing Hugging Face snapshots use symlinks into their content cache.
    # Resolve them normally; no requirement to duplicate files within a bundle.
    path = path.resolve()
    if not path.is_file():
        raise AssetError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise AssetError(f"Empty {label}: {path}")
    return path


@dataclass(frozen=True)
class ServiceAssets:
    root: Path
    model_dir: Path
    config: Path
    mimi_weight: Path
    voice_prompt_dir: Path
    static: Path
    tokenizer: Path
    voices: tuple[Path, ...]

    @classmethod
    def resolve(cls, root, model_dir) -> "ServiceAssets":
        """Validate the complete offline bundle before opening the FPGA.

        Expected layout::

            assets/config.json
            assets/tokenizer-e351c8d8-checkpoint125.safetensors
            assets/voices/<voice-name>.pt
            assets/dist/index.html
            model/tokenizer_spm_32k_3.model

        At least one named voice is required. The browser must select an available
        voice; this validator does not invent or alias missing presets. Files may
        be symlinks to a local cache. No pickle voice files are deserialized here.
        """
        root_path = _directory(root, "PersonaPlex assets root")
        model_path = _directory(model_dir, "PersonaPlex model directory")
        config = _file(root_path / "config.json", "upstream PersonaPlex config.json")
        try:
            content = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AssetError(f"Invalid upstream PersonaPlex JSON config: {config}") from exc
        if not isinstance(content, dict):
            raise AssetError(f"Upstream PersonaPlex config must be a JSON object: {config}")
        mimi = _file(root_path / MIMI_FILENAME, "original Mimi codec checkpoint")
        tokenizer = _file(model_path / TOKENIZER_FILENAME, "SentencePiece tokenizer")
        voice_dir = _directory(root_path / "voices", "PersonaPlex voice directory")
        voices = tuple(path for path in sorted(voice_dir.glob("*.pt"))
                       if path.stem and not path.name.startswith("."))
        for voice in voices:
            _file(voice, f"voice prompt {voice.name}")
        # Retain request-visible filenames even if a voice points to a cache
        # blob whose basename is a content hash instead of NATF2.pt, etc.
        if not voices:
            raise AssetError(f"No named .pt voice prompts found in {voice_dir}")
        static = _directory(root_path / "dist", "PersonaPlex browser asset directory")
        _file(static / "index.html", "PersonaPlex browser index.html")
        return cls(root_path, model_path, config, mimi, voice_dir, static, tokenizer, voices)


@contextmanager
def offline_hub(assets: ServiceAssets):
    """Scope a no-network Hub resolver around the upstream runpy server launch.

    Install before importing/running moshi.server, whose module imports the Hub
    function directly. The server must receive explicit local Mimi/tokenizer/
    voice/static paths. Only its config-counter request is allowed here; missing
    server arguments therefore fail instead of downloading a full original LM.

    This patch is process-global for its scope, as is the upstream CLI wrapper.
    Use it while launching this single-purpose server, not concurrently with
    unrelated Hub jobs. The original function is restored even on exceptions.
    """
    import huggingface_hub

    if not isinstance(assets, ServiceAssets):
        raise TypeError("offline_hub requires validated ServiceAssets")
    previous = huggingface_hub.hf_hub_download

    def local_download(repo_id, filename, *, subfolder=None, repo_type=None,
                       revision=None, **kwargs):
        if (repo_id != UPSTREAM_REPO or filename != "config.json"
                or subfolder not in (None, "") or repo_type not in (None, "model")
                or revision not in (None, "main")):
            raise OfflineDownloadError(
                f"Offline PersonaPlex service cannot download {repo_id!r}/{filename!r}; "
                "only the local upstream config.json is allowlisted. "
                "Supply local paths for the codec, tokenizer, voices and browser assets."
            )
        if kwargs.get("force_download") or kwargs.get("local_dir") is not None:
            raise OfflineDownloadError("Offline PersonaPlex resolver does not download or copy Hub files")
        return str(_file(assets.config, "upstream PersonaPlex config.json"))

    huggingface_hub.hf_hub_download = local_download
    try:
        yield local_download
    finally:
        huggingface_hub.hf_hub_download = previous
