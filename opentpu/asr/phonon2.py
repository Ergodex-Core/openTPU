"""Phonon-2's strict FPGA-only serving contract.

This module does not load a model, import a remote runtime, or open a device.
The learned encoder and TDT path have CPU/ISA evidence; audio-to-transcript remains gated.
"""
from __future__ import annotations

MODEL_ID = "FermionResearch/Phonon-2"
MODEL_REVISION = "fb09063c9dc8e130d97b3a2191ed5aed6b71bb31"
BASE_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
ARCHIVE_SHA256 = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"
CONTAINER_SHA256 = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"

BLOCKERS = (
    ("audio_frontend", "Device PCM conversion, preemphasis, STFT, mel, log and masked normalization are missing."),
    ("subsampling", "The complete learned 8x subsampler is shared-profile ISA-qualified; its device waveform-frontend and physical runtime handoff are not integrated."),
    ("encoder_integration", "All 24 whole-context Conformer layers and one-slot residency are shared-profile ISA-qualified with real checkpoint weights; physical program/wait loading and execution remain unverified."),
    ("predictor_integration", "INT6 embedding, both LSTM layers and the Transformers cache policy are shared-profile ISA-qualified; physical state/cache acceptance remains gated."),
    ("tdt_decode", "Joint head, token/duration selection, EOS and zero-duration/watchdog behavior pass real-checkpoint ISA tests; physical runtime integration remains unverified."),
    ("text_output", "Device-selected UTF-8 segments and byte lengths match the pinned text policy; physical output I/O/API integration remains unqualified."),
    ("numerics", "Learned encoder and TDT/text path pass real-speech CPU/ISA comparisons; device waveform parity, larger-context coverage, physical transcript and WER acceptance remain missing."),
    ("hardware_evidence", "No real audio-to-transcript FPGA run; a coordinated hardware window is required."),
)


def model_status() -> dict:
    """Fresh, JSON-safe metadata for Inferlab; never advertise a kernel test as inference."""
    return {
        "schema_version": 1,
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "display_name": "Phonon-2",
        "inferlab_id": "phonon2",
        "task": "automatic-speech-recognition",
        "language": "en",
        "architecture": "FastConformer + LSTM Token Duration Transducer",
        "parameters": 627008134,
        "checkpoint_download_bytes": 163515201,
        "status": "blocked",
        "readiness": "learned_encoder_tdt_isa_only",
        "available": False,
        "ready": False,
        "execution_policy": "fpga_only",
        "host_model_compute_allowed": False,
        "cpu_gpu_fallback": False,
        "hardware_validated": False,
        "audio_to_transcript_validated": False,
        "license": {"weights": "CC-BY-4.0", "code": "Apache-2.0",
                    "attribution_url": f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/NOTICE"},
        "input_contract": {"encoding": "pcm_s16le", "sample_rate_hz": 16000,
                           "channels": 1, "min_samples": 320, "max_seconds": 30,
                           "resampling": "unsupported", "hotwords": "unsupported",
                           "note": "Proposed first integration envelope; endpoint remains unavailable."},
        "implemented_slice": "Learned 8x subsampler, all 24 Conformer layers, LSTM predictor and autonomous TDT/UTF-8 selection on the shared-profile ISA; waveform frontend remains offline oracle",
        "shared_image": {"profile": "f2-hbm:p4rl8a64c150", "agfi": "agfi-0001de1dd3df8e21e",
                         "declared_cfg_hash": "dd6b92a6", "hardware_identity_verified": False,
                         "per_model_image_reload": False, "maximum_serving_interruption_seconds": 30,
                         "runtime_source_revision": "df69c0a9d970ba4f78b8ac6e979b41dc69d4f181",
                         "manifest": "docs/phonon2/shared_profile.json"},
        "qualification": {"conformer_blocks": 24, "max_stored_encoder_frames": 376,
                          "max_real_checkpoint_tested_encoder_frames": 89,
                          "longer_context_status": "compile/resource qualification only",
                          "frontend_device_implemented": False, "tdt_device_implemented": True,
                          "predictor_isa_implemented": True, "text_isa_implemented": True,
                          "tdt_evidence": "docs/phonon2/tdt_validation.json",
                          "tdt_semantic_policy": "docs/phonon2/tdt_semantic_policy.json",
                          "runtime_dependency": "docs/phonon2/tdt_runtime_dependency.json",
                          "real_speech_token_duration_frame_and_text_parity": True,
                          "tdt_reset_bitwise_passed": True,
                          "text_output_format": "Device-selected UTF-8 scatter/gather bytes plus lengths; host output I/O only",
                          "word_timestamps_implemented": False,
                          "evidence": "docs/phonon2/encoder_validation.json",
                          "activation_passes": 2, "encoder_reset_bitwise_passed": True,
                          "subsampler_isa_implemented": True,
                          "subsampler_max_stored_mel_frames": 3001,
                          "subsampler_max_stored_encoder_frames": 376,
                          "real_speech_feature_vectors_tested": True,
                          "subsampler_evidence": "docs/phonon2/subsampler_validation.json",
                          "frontend_policy": "docs/phonon2/semantic_policy.json"},
        "error": {"code": "ASR_FPGA_NOT_READY", "http_status": 503},
        "blockers": [{"code": code, "detail": detail} for code, detail in BLOCKERS],
    }


def inferlab_model_status() -> dict:
    """Fields matching Inferlab's existing model-card contract, with pinned provenance."""
    return {
        "id": "phonon2", "name": "Phonon-2", "kind": "asr", "capability": "transcription",
        "size": "627M · five-value encoder · INT6 tables",
        "execution": "Blocked · ASR integration incomplete", "ready": False,
        "description": "English ASR; full FPGA audio-to-transcript inference is not implemented.",
        "reason": "Shared-profile learned encoder and LSTM/TDT/text path pass CPU/ISA checks; device waveform frontend and physical audio-to-transcript acceptance remain incomplete.",
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "execution_policy": "fpga_only", "cpu_gpu_fallback": False,
        "shared_image_profile": "f2-hbm:p4rl8a64c150", "image_reload_required": False,
        "hardware_validated": False, "error_code": "ASR_FPGA_NOT_READY",
    }


class ASRNotReady(RuntimeError):
    code = "ASR_FPGA_NOT_READY"
    http_status = 503

    def __init__(self):
        super().__init__("Phonon-2 FPGA audio-to-transcript inference is not implemented; no host fallback is permitted.")


def transcribe(pcm_s16le: bytes, *, sample_rate_hz: int = 16000, channels: int = 1):
    """Reserved adapter entry point. Fail before file/model/device access or host compute."""
    raise ASRNotReady()
