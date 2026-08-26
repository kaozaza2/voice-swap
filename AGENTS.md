# AGENTS.md

## Project Overview

**Voice Swap** — AI singing voice conversion tool. Use your voice to cover any song.

- **Repo**: `git@github.com:kaozaza2/voice-swap.git`
- **Author**: MokiMikore <kittisakphiri@gmail.com>
- **License**: WTFPL

## Architecture

```
src/voice_swap/
├── cli.py           # CLI commands (cover, convert, separate, preprocess, train-model, info)
├── config.py        # YAML config management
├── model.py         # Conformer + ECAPA-TDNN + Flow decoder
├── hifigan.py       # HiFi-GAN vocoder + Multi-scale/Period discriminators
├── losses.py        # Mel, perceptual, adversarial, feature matching losses
├── preprocess.py    # Audio loading, feature extraction, dataset preprocessing
├── train.py         # Training loop with GAN adversarial training
├── inference.py     # Song conversion pipeline
├── separator.py     # Demucs vocal/instrumental separation
├── audio_utils.py   # Crossfade, smoothing, denoising, normalization
└── utils.py         # Cross-platform utilities
```

## Key Decisions

- **Separation**: Demucs (htdemucs) — better quality than Spleeter
- **Vocoder**: HiFi-GAN — replaces Griffin-Lim for high-quality audio
- **Model**: Conformer blocks + ECAPA-TDNN speaker encoder + Flow decoder
- **Training**: GAN adversarial training with multi-scale + multi-period discriminators
- **Formats**: 17 audio formats supported via librosa/soundfile
- **Cross-platform**: Windows, macOS (MPS), Linux (CUDA)

## CLI Commands

| Command | Description |
|---------|-------------|
| `voice-swap cover -m model -s song [-i inst] -o out [-f]` | Full pipeline |
| `voice-swap convert -m model -i vocals -o out` | Direct conversion |
| `voice-swap separate -i song -o dir` | Stem separation |
| `voice-swap preprocess -i dir -o dir` | Feature extraction |
| `voice-swap train-model -d dir -o dir [-e n]` | Train model |
| `voice-swap info` | System info |

## Current Status

- [x] Project structure with uv
- [x] Model architecture (Conformer + ECAPA + Flow)
- [x] HiFi-GAN vocoder
- [x] Training with adversarial losses
- [x] Demucs source separation
- [x] Audio smoothing and post-processing
- [x] Cross-platform support
- [x] WTFPL license
- [x] Pushed to GitHub

## TODO

- [ ] Add tests (pytest)
- [ ] Add linting (ruff, mypy)
- [ ] Add GitHub Actions CI
- [ ] Add batch processing
- [ ] Add voice presets
- [ ] Add web UI (Gradio/Streamlit)
- [ ] Add multiple output formats (mp3, flac)
- [ ] Add progress bar with ETA
- [ ] Add voice similarity scoring

## Run Commands

```bash
# Setup
uv venv && uv pip install -e .

# Check system
voice-swap info

# Workflow
voice-swap preprocess -i data/raw -o data/processed
voice-swap train-model -d data/processed -e 1000
voice-swap cover -m checkpoints/best.pth -s song.wav -o cover.wav
```
