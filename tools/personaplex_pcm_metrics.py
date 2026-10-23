"""Raw FP32 PCM diagnostics for offline PersonaPlex qualification.

Inputs and full-scale units are used exactly as supplied. No waveform gain,
normalization, clipping, resampling, or perceptual-quality judgment is applied.
"""

import numpy as np


def _summary(actual, reference):
    # Float64 arithmetic avoids overflow when subtracting finite FP32 samples.
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    error = a - r
    reference_rms = float(np.sqrt(np.mean(r * r)))
    actual_rms = float(np.sqrt(np.mean(a * a)))
    error_rms = float(np.sqrt(np.mean(error * error)))
    zero_reference = reference_rms == 0.0
    zero_error = error_rms == 0.0
    result = {
        "samples": int(a.size),
        "reference_rms_full_scale": reference_rms,
        "actual_rms_full_scale": actual_rms,
        "error_rms_full_scale": error_rms,
        "peak_absolute_error_full_scale": float(np.max(np.abs(error))),
        "mean_error_full_scale": float(np.mean(error)),
        "reference_mean_full_scale": float(np.mean(r)),
        "actual_mean_full_scale": float(np.mean(a)),
        "snr_db": (None if zero_reference or zero_error else
                   float(20.0 * np.log10(reference_rms / error_rms))),
        "error_rms_dbfs": (None if zero_error else float(20.0 * np.log10(error_rms))),
        "signal_rms_dbfs": (None if zero_reference else float(20.0 * np.log10(reference_rms))),
        "relative_l2": None if zero_reference else error_rms / reference_rms,
        "raw_rms_gain_ratio_diagnostic_only": (None if zero_reference else
                                                actual_rms / reference_rms),
        "zero_reference_rms": zero_reference,
        "zero_error_rms": zero_error,
        "zero_actual_rms": actual_rms == 0.0,
    }
    if any(isinstance(value, float) and not np.isfinite(value)
           for value in result.values()):
        raise ValueError("PCM diagnostic metric is nonfinite")
    return result


def waveform_metrics(actual, reference):
    """Return aggregate and per-frame raw waveform diagnostics.

    Both inputs must be finite FP32 arrays of identical shape [frames,samples]
    with positive dimensions. Zero reference/error RMS makes undefined dB or
    ratio values ``None`` and sets explicit flags, keeping JSON output finite.
    The reported RMS gain ratio is diagnostic only and is never applied.
    """
    a, r = np.asarray(actual), np.asarray(reference)
    if (a.dtype != np.float32 or r.dtype != np.float32 or a.ndim != 2
            or r.ndim != 2 or a.shape != r.shape or not all(a.shape)
            or not np.isfinite(a).all() or not np.isfinite(r).all()):
        raise ValueError("PCM metrics require finite FP32 equal [frames,samples] arrays")
    return {"units": "raw FP32 full scale = 1; no normalization or gain applied",
            "aggregate": _summary(a, r),
            "per_frame": [_summary(frame_a, frame_r) for frame_a, frame_r in zip(a, r)]}
