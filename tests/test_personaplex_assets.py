"""Offline service assets must fail before FPGA use or network access."""
from pathlib import Path
import sys
from types import ModuleType

import pytest

from opentpu.llm.personaplex_assets import (
    AssetError, MIMI_FILENAME, OfflineDownloadError, ServiceAssets,
    TOKENIZER_FILENAME, UPSTREAM_REPO, offline_hub,
)


@pytest.fixture
def bundle(tmp_path):
    root, model = tmp_path / "assets", tmp_path / "model"
    for path in (root, model, root / "voices", root / "dist"):
        path.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"source": "test-fixture"}')
    (root / MIMI_FILENAME).write_bytes(b"codec-fixture")
    (model / TOKENIZER_FILENAME).write_bytes(b"tokenizer-fixture")
    (root / "voices/NATF2.pt").write_bytes(b"voice-fixture")
    (root / "dist/index.html").write_text("<!doctype html><title>Test fixture</title>")
    return root, model


@pytest.fixture
def hub(monkeypatch):
    module = ModuleType("huggingface_hub")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Original hf_hub_download must never be called")
    module.hf_hub_download = forbidden
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module, forbidden, calls


def test_resolve_returns_absolute_local_paths(bundle):
    root, model = bundle
    assets = ServiceAssets.resolve(root, model)
    assert assets.config == root / "config.json"
    assert assets.mimi_weight == root / MIMI_FILENAME
    assert assets.tokenizer == model / TOKENIZER_FILENAME
    assert assets.voice_prompt_dir == root / "voices"
    assert assets.static == root / "dist"
    assert assets.voices == (root / "voices/NATF2.pt",)
    assert all(path.is_absolute() for path in assets.voices)


@pytest.mark.parametrize("relative,label", [
    ("config.json", "config.json"),
    (MIMI_FILENAME, "Mimi codec"),
    ("dist/index.html", "index.html"),
])
def test_missing_required_asset_is_explicit(bundle, relative, label):
    root, model = bundle
    (root / relative).unlink()
    with pytest.raises(AssetError, match=label):
        ServiceAssets.resolve(root, model)


def test_missing_tokenizer_or_voice_fails_locally(bundle):
    root, model = bundle
    (model / TOKENIZER_FILENAME).unlink()
    with pytest.raises(AssetError, match="tokenizer"):
        ServiceAssets.resolve(root, model)
    (model / TOKENIZER_FILENAME).write_bytes(b"tokenizer")
    (root / "voices/NATF2.pt").unlink()
    with pytest.raises(AssetError, match="No named .pt"):
        ServiceAssets.resolve(root, model)


@pytest.mark.parametrize("content", ["", "not json", "[]", "null", "\xff"])
def test_invalid_config_rejected(bundle, content):
    root, model = bundle
    (root / "config.json").write_bytes(content.encode("latin-1"))
    with pytest.raises(AssetError, match="config"):
        ServiceAssets.resolve(root, model)


def test_empty_voice_rejected_without_deserializing_pickle(bundle):
    root, model = bundle
    (root / "voices/NATF2.pt").write_bytes(b"")
    with pytest.raises(AssetError, match="Empty voice prompt"):
        ServiceAssets.resolve(root, model)


def test_local_cache_symlinks_are_supported(bundle, tmp_path):
    root, model = bundle
    blob = tmp_path / "hf-cache-blob"
    blob.write_bytes(b"codec-fixture")
    codec = root / MIMI_FILENAME
    codec.unlink()
    codec.symlink_to(blob)
    assert ServiceAssets.resolve(root, model).mimi_weight == blob
    voice = root / "voices/NATF2.pt"
    voice.unlink()
    voice.symlink_to(blob)
    assert ServiceAssets.resolve(root, model).voices[0].name == "NATF2.pt"


def test_offline_hub_uses_only_local_config_and_restores(bundle, hub):
    root, model = bundle
    module, original, calls = hub
    assets = ServiceAssets.resolve(root, model)
    with offline_hub(assets) as resolver:
        # Mirrors `from huggingface_hub import hf_hub_download` in moshi.server.
        from huggingface_hub import hf_hub_download
        assert hf_hub_download is resolver
        assert hf_hub_download(UPSTREAM_REPO, "config.json") == str(assets.config)
        assert hf_hub_download(repo_id=UPSTREAM_REPO, filename="config.json",
                               revision="main", repo_type="model") == str(assets.config)
    assert module.hf_hub_download is original
    assert calls == []


@pytest.mark.parametrize("repo,filename,kwargs", [
    (UPSTREAM_REPO, "model.safetensors", {}),
    (UPSTREAM_REPO, MIMI_FILENAME, {}),
    (UPSTREAM_REPO, "../config.json", {}),
    (UPSTREAM_REPO, "/config.json", {}),
    ("other/repo", "config.json", {}),
    (UPSTREAM_REPO, "config.json", {"subfolder": ".."}),
    (UPSTREAM_REPO, "config.json", {"repo_type": "dataset"}),
    (UPSTREAM_REPO, "config.json", {"revision": "unrelated-commit"}),
    (UPSTREAM_REPO, "config.json", {"force_download": True}),
    (UPSTREAM_REPO, "config.json", {"local_dir": "/somewhere"}),
])
def test_non_allowlisted_requests_never_download(bundle, hub, repo, filename, kwargs):
    root, model = bundle
    module, original, calls = hub
    with offline_hub(ServiceAssets.resolve(root, model)):
        with pytest.raises(OfflineDownloadError, match="Offline PersonaPlex"):
            module.hf_hub_download(repo, filename, **kwargs)
    assert module.hf_hub_download is original
    assert calls == []


def test_exception_and_nested_scopes_restore_original(bundle, hub):
    root, model = bundle
    module, original, calls = hub
    assets = ServiceAssets.resolve(root, model)
    with pytest.raises(RuntimeError, match="startup failed"):
        with offline_hub(assets) as outer:
            with offline_hub(assets):
                assert module.hf_hub_download(UPSTREAM_REPO, "config.json") == str(assets.config)
            assert module.hf_hub_download is outer
            raise RuntimeError("startup failed")
    assert module.hf_hub_download is original
    assert calls == []


def test_deleted_config_does_not_fall_back_to_hub(bundle, hub):
    root, model = bundle
    module, original, calls = hub
    assets = ServiceAssets.resolve(root, model)
    assets.config.unlink()
    with offline_hub(assets):
        with pytest.raises(AssetError, match="Missing.*config.json"):
            module.hf_hub_download(UPSTREAM_REPO, "config.json")
    assert calls == [] and module.hf_hub_download is original


def test_missing_root_is_clear(tmp_path):
    with pytest.raises(AssetError, match="assets root.*local directory"):
        ServiceAssets.resolve(tmp_path / "missing", tmp_path)
