"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    fmin: int = 0
    fmax: int = 8000
    segment_length: int = 16384  # ~0.37s at 44100Hz


@dataclass
class ModelConfig:
    hidden_dim: int = 192
    n_layers: int = 6
    n_heads: int = 8
    codebook_size: int = 1024
    n_codes: int = 4
    content_dim: int = 768
    spk_dim: int = 256


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 2e-4
    epochs: int = 1000
    save_every: int = 100
    log_every: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 4000
    fp16: bool = True


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls()
        if "audio" in data:
            config.audio = AudioConfig(**data["audio"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "train" in data:
            config.train = TrainConfig(**data["train"])
        if "device" in data:
            config.device = data["device"]
        return config

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
