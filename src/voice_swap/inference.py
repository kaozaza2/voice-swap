"""Inference and song conversion with high-quality output."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .config import Config
from .model import VoiceSwapModel
from .hifigan import HiFiGAN
from .preprocess import extract_features, load_audio
from .audio_utils import (
    crossfade_segments,
    smooth_pitch,
    normalize_volume,
    highpass_filter,
    denoise_audio,
    add_reverb,
)
from .separator import separate_audio


OUTPUT_FORMATS = {".wav", ".flac", ".mp3", ".m4a"}
LOSSY_ENCODING_OPTIONS = {
    ".mp3": ["-codec:a", "libmp3lame", "-q:a", "2"],
    ".m4a": ["-codec:a", "aac", "-b:a", "192k"],
}


def write_output_audio(
    output_path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Write audio as WAV, FLAC, MP3, or M4A based on the output extension."""
    output_path = Path(output_path)
    output_format = output_path.suffix.lower()

    if output_format not in OUTPUT_FORMATS:
        supported_formats = ", ".join(sorted(OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output format '{output_path.suffix}'. "
            f"Use one of: {supported_formats}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in {".wav", ".flac"}:
        sf.write(str(output_path), audio, sample_rate)
        return

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("MP3 and M4A output require ffmpeg to be installed and available in PATH.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        sf.write(str(temp_path), audio, sample_rate)
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                *LOSSY_ENCODING_OPTIONS[output_format],
                str(output_path),
            ],
            check=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def load_model(
    checkpoint_path: str | Path,
    config: Config | None = None,
) -> tuple[VoiceSwapModel, HiFiGAN]:
    """Load trained model and vocoder from checkpoint."""
    if config is None:
        config = Config()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = VoiceSwapModel(
        content_dim=config.audio.n_mels,
        hidden_dim=config.model.hidden_dim,
        spk_dim=config.model.spk_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
    ).to(device)

    vocoder = HiFiGAN(in_channels=config.audio.n_mels).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    vocoder.load_state_dict(checkpoint["vocoder_state_dict"])

    model.eval()
    vocoder.eval()

    return model, vocoder


def convert_song(
    model: VoiceSwapModel,
    vocoder: HiFiGAN,
    input_path: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    reference_path: str | Path | None = None,
    separate_stems: bool = True,
    mix_instrumental: bool = True,
    instrumental_mix: float = 0.15,
    smooth: bool = True,
    denoise: bool = True,
    normalize: bool = True,
) -> None:
    """Convert a song to use the trained voice.

    Args:
        model: Voice conversion model
        vocoder: HiFi-GAN vocoder
        input_path: Path to input song
        output_path: Path for output file
        config: Configuration
        reference_path: Reference audio for voice quality
        separate_stems: Whether to separate vocals first
        mix_instrumental: Whether to mix back instrumental
        instrumental_mix: Mix ratio for instrumental (0.0-1.0)
        smooth: Whether to apply smoothing
        denoise: Whether to apply denoising
        normalize: Whether to normalize output
    """
    if config is None:
        config = Config()

    device = next(model.parameters()).device
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate stems if requested
    if separate_stems:
        print("Separating audio stems...")
        temp_dir = Path(tempfile.mkdtemp())
        stems = separate_audio(input_path, temp_dir, device=str(device))
        vocal_path = stems.get("vocals", input_path)
        instrumental_path = stems.get("instrumental")
    else:
        vocal_path = input_path
        instrumental_path = None

    # Load audio
    source_audio = load_audio(vocal_path, sr=config.audio.sample_rate)

    if reference_path:
        ref_audio = load_audio(reference_path, sr=config.audio.sample_rate)
    else:
        ref_audio = source_audio[: config.audio.sample_rate * 10]

    # Extract features
    ref_features = extract_features(ref_audio, config)
    # extract_features returns (n_mels, frames); the model works on (batch, frames, n_mels).
    ref_mel = torch.tensor(ref_features["mel"]).T.unsqueeze(0).to(device)

    # Process in overlapping segments
    segment_length = config.audio.segment_length
    overlap = config.audio.hop_length * 16  # ~0.37s overlap

    converted_segments = []
    segment_times = []

    for i in tqdm(range(0, len(source_audio), segment_length - overlap), desc="Converting"):
        segment = source_audio[i : i + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))

        if np.abs(segment).max() < 0.01:
            continue

        features = extract_features(segment, config)
        source_mel = torch.tensor(features["mel"]).T.unsqueeze(0).to(device)

        with torch.no_grad():
            converted_mel = model.convert(source_mel, ref_mel)
            converted_audio = vocoder(converted_mel.transpose(1, 2)).squeeze().cpu().numpy()

        if smooth:
            converted_audio = smooth_audio(converted_audio, config.audio.sample_rate)

        converted_segments.append(converted_audio)
        segment_times.append(i / config.audio.sample_rate)

    # Crossfade segments
    if converted_segments:
        output_audio = crossfade_segments(converted_segments, overlap=overlap)
    else:
        output_audio = np.zeros(len(source_audio))

    # Trim to original length
    output_audio = output_audio[: len(source_audio)]

    # Mix with instrumental if available
    if mix_instrumental and instrumental_path:
        instrumental = load_audio(instrumental_path, sr=config.audio.sample_rate)
        instrumental = instrumental[: len(output_audio)]
        output_audio = output_audio * (1 - instrumental_mix) + instrumental * instrumental_mix

    # Post-processing
    if denoise:
        output_audio = denoise_audio(output_audio, sr=config.audio.sample_rate)

    output_audio = highpass_filter(output_audio, cutoff=60.0, sr=config.audio.sample_rate)

    if normalize:
        output_audio = normalize_volume(output_audio, target_db=-18.0)

    output_audio = np.clip(output_audio, -0.99, 0.99)

    write_output_audio(output_path, output_audio, config.audio.sample_rate)
    print(f"Saved converted audio to {output_path}")


def smooth_audio(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Apply smoothing to converted audio."""
    from scipy.signal import savgol_filter

    if len(audio) < 15:
        return audio

    smoothed = savgol_filter(audio, 15, 3)
    return smoothed


def convert_with_instrumental(
    model: VoiceSwapModel,
    vocoder: HiFiGAN,
    song_path: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    reference_path: str | Path | None = None,
    instrumental_path: str | Path | None = None,
    instrumental_mix: float = 0.15,
    separation_method: str = "demucs",
    device: str | None = None,
) -> None:
    """Full pipeline: separate -> convert -> mix back.

    This is the recommended workflow for best results.

    Args:
        instrumental_path: If provided, use this instrumental track instead of separating
    """
    if config is None:
        config = Config()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    song_path = Path(song_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp())

    # Step 1: Get vocals and instrumental
    if instrumental_path:
        print("Using provided instrumental track...")
        vocal_path = song_path
        instrumental_path = Path(instrumental_path)
    else:
        print("Step 1: Separating vocal and instrumental...")
        stems = separate_audio(song_path, temp_dir, method=separation_method, device=device)
        vocal_path = stems["vocals"]
        instrumental_path = stems["instrumental"]

    # Step 2: Convert vocals
    print("Step 2: Converting vocals to your voice...")
    converted_path = temp_dir / "converted_vocals.wav"
    convert_song(
        model=model,
        vocoder=vocoder,
        input_path=vocal_path,
        output_path=converted_path,
        config=config,
        reference_path=reference_path,
        separate_stems=False,
        smooth=True,
        denoise=True,
        normalize=True,
    )

    # Step 3: Mix with instrumental
    print("Step 3: Mixing with instrumental...")
    converted_vocals, _ = librosa.load(str(converted_path), sr=config.audio.sample_rate)
    instrumental, _ = librosa.load(str(instrumental_path), sr=config.audio.sample_rate)

    min_len = min(len(converted_vocals), len(instrumental))
    converted_vocals = converted_vocals[:min_len]
    instrumental = instrumental[:min_len]

    mixed = converted_vocals * (1 - instrumental_mix) + instrumental * instrumental_mix
    mixed = normalize_volume(mixed, target_db=-18.0)
    mixed = np.clip(mixed, -0.99, 0.99)

    write_output_audio(output_path, mixed, config.audio.sample_rate)
    print(f"Saved final cover to {output_path}")

    # Cleanup temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
