"""Audio source separation for vocal and instrumental tracks."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


class DemucsSeparator:
    """Use Demucs for high-quality source separation."""

    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        from demucs.pretrained import get_model
        from demucs.apply import BagOfModels

        self._model = get_model(self.model_name)
        if isinstance(self._model, BagOfModels):
            self._model = self._model.models[0]
        self._model.to(self.device)
        self._model.eval()

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        stems: bool = True,
    ) -> dict[str, Path]:
        """Separate audio into vocal and instrumental tracks.

        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated tracks
            stems: If True, save individual stems; if False, save vocal + instrumental only

        Returns:
            Dictionary mapping stem names to output paths
        """
        self._load_model()

        from demucs.apply import apply_model
        from demucs.audio import AudioFile

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav, sr = AudioFile(str(input_path)).read(streams=0)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        wav = wav.to(self.device)

        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                device=self.device,
                overlap=0.25,
                progress=True,
            )

        sources = sources[0]
        sources = sources * ref.std() + ref.mean()

        source_names = self._model.sources
        outputs = {}

        if stems:
            for i, name in enumerate(source_names):
                out_path = output_dir / f"{input_path.stem}_{name}.wav"
                sf.write(str(out_path), sources[i].cpu().numpy().T, sr)
                outputs[name] = out_path
        else:
            vocal_idx = source_names.index("vocals") if "vocals" in source_names else 0
            vocal = sources[vocal_idx]

            instrumental = torch.zeros_like(vocal)
            for i, name in enumerate(source_names):
                if name != "vocals":
                    instrumental += sources[i]

            vocal_path = output_dir / f"{input_path.stem}_vocals.wav"
            inst_path = output_dir / f"{input_path.stem}_instrumental.wav"

            sf.write(str(vocal_path), vocal.cpu().numpy().T, sr)
            sf.write(str(inst_path), instrumental.cpu().numpy().T, sr)

            outputs["vocals"] = vocal_path
            outputs["instrumental"] = inst_path

        return outputs

    def separate_vocals(
        self,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Extract only vocals from audio."""
        outputs = self.separate(input_path, output_dir, stems=False)
        return outputs["vocals"]

    def separate_instrumental(
        self,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Extract only instrumental from audio."""
        outputs = self.separate(input_path, output_dir, stems=False)
        return outputs["instrumental"]


def separate_audio(
    input_path: str | Path,
    output_dir: str | Path,
    method: str = "demucs",
    device: str | None = None,
) -> dict[str, Path]:
    """Separate audio into vocal and instrumental tracks.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated tracks
        method: Separation method (only "demucs" supported)
        device: Device to use for inference

    Returns:
        Dictionary mapping stem names to output paths
    """
    if method != "demucs":
        raise ValueError(f"Only demucs is supported, got: {method}")

    separator = DemucsSeparator(device=device)
    return separator.separate(input_path, output_dir)
