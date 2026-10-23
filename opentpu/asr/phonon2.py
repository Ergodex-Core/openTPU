"""Phonon-2's strict FPGA-only serving contract.

This module does not load a model, import a remote runtime, or open a device. The
five-value projection slice is in phonon2_linear; it is not an ASR engine.
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
    ("subsampling", "Device 8x Conv2D/depthwise subsampling and its length masks are missing."),
    ("conformer", "One complete full-width block is ISA-qualified for 1..8 stored frames; the full-length 24-block encoder and device graph integration remain missing."),
    ("predictor", "Resident two-layer LSTM, token embedding and state commit on non-blank are missing."),
    ("tdt_decode", "Device joint head, separate token/duration selection, blank/zero-duration control and termination are missing."),
    ("text_output", "Device token-to-text output and timestamp conventions are not implemented or qualified."),
    ("numerics", "A real-weight block is qualified on synthetic activations; frontend numerical parity, full-encoder and transcript validation remain missing."),
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
        "readiness": "kernel_slice_only",
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
        "implemented_slice": "Five-value projections and one complete bounded Conformer block on the existing ISA; offline audited frontend golden vectors",
        "qualification": {"conformer_blocks": 1, "max_stored_encoder_frames": 8,
                          "frontend_device_implemented": False, "tdt_device_implemented": False,
                          "evidence": "docs/phonon2/conformer_validation.json",
                          "frontend_policy": "docs/phonon2/semantic_policy.json"},
        "error": {"code": "ASR_FPGA_NOT_READY", "http_status": 503},
        "blockers": [{"code": code, "detail": detail} for code, detail in BLOCKERS],
    }


def inferlab_model_status() -> dict:
    """Fields matching Inferlab's existing model-card contract, with pinned provenance."""
    return {
        "id": "phonon2", "name": "Phonon-2", "kind": "asr", "capability": "transcription",
        "size": "627M · five-value encoder · INT6 tables",
        "execution": "Blocked · device projection slice only", "ready": False,
        "description": "English ASR; full FPGA audio-to-transcript inference is not implemented.",
        "reason": "Device frontend, Conformer, LSTM predictor and TDT decoding are incomplete; audio input stays disabled.",
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "execution_policy": "fpga_only", "cpu_gpu_fallback": False,
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
