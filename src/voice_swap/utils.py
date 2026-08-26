"""Utility functions with cross-platform support."""

import platform
import sys
from pathlib import Path

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get appropriate device based on platform and preference."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def get_platform() -> str:
    """Get current platform."""
    return platform.system().lower()


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_linux() -> bool:
    return platform.system() == "Linux"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_temp_dir() -> Path:
    """Get platform-appropriate temp directory."""
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    return temp_dir


def get_ffmpeg_path() -> str | None:
    """Find ffmpeg binary path across platforms."""
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # Common paths on Windows
    if is_windows():
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        for path in common_paths:
            if Path(path).exists():
                return path

    return None


def check_dependencies() -> dict[str, bool]:
    """Check if required dependencies are available."""
    deps = {}

    try:
        import torch
        deps["torch"] = True
        deps["cuda"] = torch.cuda.is_available()
        deps["mps"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        deps["torch"] = False
        deps["cuda"] = False
        deps["mps"] = False

    try:
        import librosa
        deps["librosa"] = True
    except ImportError:
        deps["librosa"] = False

    try:
        import soundfile
        deps["soundfile"] = True
    except ImportError:
        deps["soundfile"] = False

    deps["ffmpeg"] = get_ffmpeg_path() is not None

    return deps


def print_system_info() -> None:
    """Print system information for debugging."""
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS: Available")
    else:
        print("Using CPU only")


def safe_path(path: str | Path) -> Path:
    """Convert path to platform-appropriate format."""
    p = Path(path)
    if is_windows():
        # Normalize Windows paths
        return Path(str(p).replace("/", "\\"))
    return p
