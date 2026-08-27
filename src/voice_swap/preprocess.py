"""Audio preprocessing utilities."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
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
FEATURE_MANIFEST = "features_manifest.json"
DEFAULT_SHARD_SIZE = 2048


def is_audio_file(path: Path) -> bool:
    """Check if file is a supported audio format."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def load_audio_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """Decode audio with ffmpeg when libsndfile cannot read its format."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            f"Could not decode {path}. Install ffmpeg to read this audio format."
        )

    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {details}") from error

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path}.")

    return audio.copy()


def load_audio(path: str | Path, sr: int = 44100) -> np.ndarray:
    """Load audio and resample it, falling back to ffmpeg for unsupported formats."""
    path = Path(path)

    try:
        audio, orig_sr = librosa.load(str(path), sr=None, mono=True)
    except sf.LibsndfileError:
        return load_audio_with_ffmpeg(path, sr)

    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


def split_audio(
    audio: np.ndarray,
    segment_length: int,
    segment_hop_length: int,
) -> list[np.ndarray]:
    """Split audio into overlapping segments."""
    if segment_hop_length <= 0:
        raise ValueError("segment_hop_length must be greater than zero.")

    segments = []
    for start in range(0, len(audio) - segment_length + 1, segment_hop_length):
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


def extract_file_features(
    audio_path: Path,
    config: AudioConfig,
) -> list[dict[str, np.ndarray]]:
    """Load one file and extract features for its non-silent segments."""
    audio = load_audio(audio_path, sr=config.sample_rate)
    segments = split_audio(
        audio,
        config.segment_length,
        config.segment_hop_length,
    )
    return [extract_features(segment, config) for segment in segments]


def preprocess_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    config: AudioConfig | None = None,
    workers: int | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> int:
    """Preprocess audio files into bounded feature shards."""
    if config is None:
        config = AudioConfig()
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than zero.")

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

    if workers is None:
        workers = min(len(audio_files), os.cpu_count() or 1)
    if workers <= 0:
        raise ValueError("workers must be greater than zero.")

    print(f"Processing {len(audio_files)} files with {workers} worker(s)")
    feature_sets = _iter_feature_sets(audio_files, config, workers)
    shard_features: list[dict[str, np.ndarray]] = []
    shard_paths: list[str] = []
    n_segments = 0

    for file_features in tqdm(
        feature_sets,
        total=len(audio_files),
        desc="Preprocessing",
        unit="file",
    ):
        for features in file_features:
            shard_features.append(features)
            n_segments += 1

            if len(shard_features) == shard_size:
                shard_paths.append(
                    _save_feature_shard(output_dir, shard_paths, shard_features)
                )
                shard_features = []

    if shard_features:
        shard_paths.append(_save_feature_shard(output_dir, shard_paths, shard_features))

    if n_segments == 0:
        raise ValueError(
            "No non-silent audio segments found. Provide longer or louder recordings."
        )

    manifest = {
        "format_version": 2,
        "n_segments": n_segments,
        "shards": shard_paths,
    }
    (output_dir / FEATURE_MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"Saved {n_segments} segments across {len(shard_paths)} feature shards")
    return n_segments


def _iter_feature_sets(
    audio_files: list[Path],
    config: AudioConfig,
    workers: int,
):
    """Yield each file's features, optionally extracting files in parallel."""
    if workers == 1:
        yield from (extract_file_features(audio_path, config) for audio_path in audio_files)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(extract_file_features, audio_path, config)
            for audio_path in audio_files
        ]
        for future in as_completed(futures):
            yield future.result()


def _save_feature_shard(
    output_dir: Path,
    shard_paths: list[str],
    features: list[dict[str, np.ndarray]],
) -> str:
    """Save a feature shard and return its filename."""
    filename = f"features_{len(shard_paths):05d}.pt"
    save_features(features, output_dir / filename)
    return filename


def save_features(features: list[dict], path: Path) -> None:
    """Save extracted features to disk."""
    # librosa.pyin returns float64; store float32 so the pitch loss does not
    # promote the whole generator loss to double (CUDA rejects double grads).
    batch = {
        "mel": torch.tensor(np.array([f["mel"] for f in features]), dtype=torch.float32),
        "pitch": torch.tensor(np.array([f["pitch"] for f in features]), dtype=torch.float32),
    }
    torch.save(batch, path)


def load_features(path: Path) -> dict[str, torch.Tensor]:
    """Load features from disk."""
    return torch.load(path, weights_only=True)


def load_feature_paths(path: Path) -> list[Path]:
    """Find sharded or legacy preprocessed features."""
    if path.is_file():
        return [path]

    manifest_path = path / FEATURE_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return [path / shard for shard in manifest["shards"]]

    legacy_path = path / "features.pt"
    if legacy_path.exists():
        return [legacy_path]

    raise FileNotFoundError(f"No preprocessed features found in {path}")
