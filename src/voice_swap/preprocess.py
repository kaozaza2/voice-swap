"""Audio preprocessing utilities."""

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .config import AudioConfig

# Supported audio formats
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".wma", ".opus", ".aiff", ".aif", ".ape", ".wv",
    ".ac3", ".dts", ".alac", ".dsf", ".dff",
}


def is_audio_file(path: Path) -> bool:
    """Check if file is a supported audio format."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def load_audio(path: str | Path, sr: int = 44100) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, orig_sr = librosa.load(str(path), sr=None, mono=True)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


def split_audio(
    audio: np.ndarray,
    segment_length: int,
    hop_length: int,
) -> list[np.ndarray]:
    """Split audio into overlapping segments."""
    segments = []
    for start in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start : start + segment_length]
        if np.abs(segment).max() > 0.01:  # skip silent segments
            segments.append(segment)
    return segments


def extract_features(
    audio: np.ndarray,
    config: AudioConfig,
) -> dict[str, np.ndarray]:
    """Extract mel spectrogram and other features from audio."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    pitch, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=config.sample_rate,
        hop_length=config.hop_length,
    )
    pitch = np.nan_to_num(pitch)

    return {"mel": mel, "pitch": pitch}


def preprocess_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    config: AudioConfig | None = None,
) -> int:
    """Preprocess all audio files in input directory."""
    if config is None:
        config = AudioConfig()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in input_dir.iterdir() if f.is_file() and is_audio_file(f)]
    if not audio_files:
        supported = ", ".join(sorted(AUDIO_EXTENSIONS))
        raise FileNotFoundError(
            f"No audio files found in {input_dir}\n"
            f"Supported formats: {supported}"
        )

    all_segments = []
    all_features = []

    for audio_path in tqdm(audio_files, desc="Loading audio"):
        audio = load_audio(audio_path, sr=config.sample_rate)
        segments = split_audio(audio, config.segment_length, config.hop_length)
        all_segments.extend(segments)

    print(f"Found {len(all_segments)} segments from {len(audio_files)} files")

    for i, segment in enumerate(tqdm(all_segments, desc="Extracting features")):
        features = extract_features(segment, config)
        all_features.append(features)

        if i % 100 == 0:
            save_features(
                all_features[: i + 1],
                output_dir / "features.pt",
            )

    save_features(all_features, output_dir / "features.pt")
    print(f"Saved {len(all_features)} segments to {output_dir / 'features.pt'}")
    return len(all_features)


def save_features(features: list[dict], path: Path) -> None:
    """Save extracted features to disk."""
    batch = {
        "mel": torch.tensor(np.array([f["mel"] for f in features])),
        "pitch": torch.tensor(np.array([f["pitch"] for f in features])),
    }
    torch.save(batch, path)


def load_features(path: Path) -> dict[str, torch.Tensor]:
    """Load features from disk."""
    return torch.load(path, weights_only=True)
