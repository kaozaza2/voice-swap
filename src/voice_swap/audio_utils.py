"""Audio smoothing and post-processing utilities."""

import numpy as np
import torch
import torch.nn.functional as F


def crossfade_segments(
    segments: list[np.ndarray],
    overlap: int = 1024,
) -> np.ndarray:
    """Crossfade overlapping segments for smooth output."""
    if len(segments) == 0:
        return np.array([])

    if len(segments) == 1:
        return segments[0]

    result = segments[0]
    for i in range(1, len(segments)):
        if len(result) < overlap:
            result = np.concatenate([result, segments[i]])
            continue

        fade_out = np.linspace(1, 0, overlap)
        fade_in = np.linspace(0, 1, overlap)

        result[-overlap:] = result[-overlap:] * fade_out + segments[i][:overlap] * fade_in
        result = np.concatenate([result, segments[i][overlap:]])

    return result


def smooth_pitch(pitch: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth pitch contour to reduce artifacts."""
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(pitch, kernel, mode="same")

    mask = pitch > 0
    smoothed[~mask] = 0

    return smoothed


def normalize_volume(
    audio: np.ndarray,
    target_db: float = -20.0,
    gain: float = 0.9,
) -> np.ndarray:
    """Normalize audio volume."""
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio

    target_rms = 10 ** (target_db / 20)
    audio = audio * (target_rms / rms)
    audio = np.clip(audio, -gain, gain)
    return audio


def highpass_filter(
    audio: np.ndarray,
    cutoff: float = 80.0,
    sr: int = 44100,
) -> np.ndarray:
    """Apply highpass filter to remove low-frequency rumble."""
    from scipy.signal import butter, sosfilt

    sos = butter(5, cutoff / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, audio)


def lowpass_filter(
    audio: np.ndarray,
    cutoff: float = 16000.0,
    sr: int = 44100,
) -> np.ndarray:
    """Apply lowpass filter to reduce high-frequency noise."""
    from scipy.signal import butter, sosfilt

    sos = butter(5, cutoff / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, audio)


def denoise_audio(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Simple noise reduction using spectral gating."""
    import librosa

    # Compute STFT
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Estimate noise floor
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)

    # Spectral gate
    gate = np.maximum(0, magnitude - noise_floor * 1.5)
    gate = gate / (magnitude + 1e-8)

    # Smooth gate
    gate = librosa.decompose.nn_filter(gate, aggregate=np.median, metric="cosine")
    gate = np.minimum(gate, magnitude / (noise_floor * 2 + 1e-8))

    # Apply gate
    magnitude = magnitude * gate

    # Reconstruct
    D_denoised = magnitude * np.exp(1j * phase)
    audio_denoised = librosa.istft(D_denoised, hop_length=512)

    return audio_denoised


def pitch_shift_audio(
    audio: np.ndarray,
    semitones: float,
    sr: int = 44100,
) -> np.ndarray:
    """Shift pitch by semitones."""
    import librosa

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def time_stretch(
    audio: np.ndarray,
    rate: float,
    sr: int = 44100,
) -> np.ndarray:
    """Time stretch audio without changing pitch."""
    import librosa

    return librosa.effects.time_stretch(audio, rate=rate)


def add_reverb(
    audio: np.ndarray,
    wet_level: float = 0.2,
    decay: float = 0.5,
    sr: int = 44100,
) -> np.ndarray:
    """Add simple reverb effect."""
    n_samples = len(audio)
    ir_length = int(sr * decay)
    ir = np.zeros(ir_length)
    ir[0] = 1.0

    for i in range(1, ir_length):
        ir[i] = ir[i - 1] * 0.5 + np.random.randn() * 0.01

    ir = ir / np.sum(np.abs(ir))

    reverb = np.convolve(audio, ir, mode="full")[:n_samples]
    return audio * (1 - wet_level) + reverb * wet_level


def final_mix(
    original: np.ndarray,
    converted: np.ndarray,
    mix_ratio: float = 0.9,
) -> np.ndarray:
    """Mix original and converted audio for natural sound."""
    min_len = min(len(original), len(converted))
    original = original[:min_len]
    converted = converted[:min_len]

    return original * (1 - mix_ratio) + converted * mix_ratio
