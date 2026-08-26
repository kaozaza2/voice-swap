# Voice Swap

AI singing voice conversion tool — use your voice to cover any song.

---

## Features

### Voice Cloning
- **Conformer-based content encoder** captures what is being sung
- **ECAPA-TDNN speaker encoder** learns your unique voice characteristics
- **Flow-based decoder** generates high-quality converted audio

### Source Separation
- **Demucs neural network** separates vocals from instrumentals
- Clean vocal extraction for accurate voice conversion
- Automatic mixing of converted vocals with original instrumental

### High-Quality Output
- **HiFi-GAN vocoder** produces crystal-clear audio
- **Multi-scale adversarial training** for realistic results
- **Crossfade smoothing** eliminates segment boundaries
- **Pitch smoothing** and **denoising** for natural sound

### Cross-Platform
- Windows, macOS, and Linux support
- CUDA (NVIDIA), MPS (Apple Silicon), and CPU inference
- Automatic device detection

---

## Installation

### Prerequisites

- Python 3.11 or higher
- ffmpeg installed and in PATH
- GPU recommended (NVIDIA CUDA or Apple Silicon)

### Setup

```bash
# Clone or navigate to the project
cd voice-swap

# Create virtual environment
uv venv

# Install dependencies
uv pip install -e .

# Verify installation
voice-swap info
```

### Windows-Specific Setup

```powershell
# Install uv (if not installed)
pip install uv

# Create virtual environment
uv venv

# Install dependencies
uv pip install -e .

# Verify ffmpeg is in PATH
ffmpeg -version
```

---

## Quick Start

### 1. Record Your Voice

Record 10-30 minutes of yourself singing. Requirements:
- Clean vocals only (no background music)
- Quiet room with minimal echo
- Various pitches and dynamics
- WAV format, 44.1kHz sample rate

```bash
mkdir -p data/raw
# Place your .wav files in data/raw/
```

### 2. Preprocess Audio

```bash
voice-swap preprocess --input data/raw --output data/processed
```

This extracts:
- Mel spectrograms
- Pitch contours
- Speaker features

### 3. Train Model

```bash
voice-swap train-model --data data/processed --epochs 1000
```

Training time:
- GPU: ~2-4 hours
- CPU: ~24-48 hours (not recommended)

The CLI shows progress and an estimated time remaining while preprocessing,
training, conversion, and source separation run. Training progress covers all
epochs, and shows the current generator and discriminator losses.

Checkpoints are saved to `checkpoints/`.

### 4. Cover a Song

```bash
voice-swap cover \
  --model checkpoints/best.pth \
  --song song.wav \
  --output my_cover.wav
```

---

## Commands Reference

### `voice-swap info`

Show system information and dependency status.

```bash
voice-swap info
```

Output:
```
Platform: Windows-10-10.0.19041-SP0
Python: 3.11.5
PyTorch: 2.13.0
CUDA: 11.8
GPU: NVIDIA GeForce RTX 3080

Dependencies:
  torch: OK
  cuda: OK
  mps: MISSING
  librosa: OK
  soundfile: OK
  ffmpeg: OK
```

### `voice-swap preprocess`

Extract features from audio files for training.

```bash
voice-swap preprocess --input <input_dir> --output <output_dir>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Directory with .wav files |
| `--output` | `data/processed` | Output directory |

### `voice-swap train-model`

Train the voice conversion model.

```bash
voice-swap train-model --data <data_dir> --output <output_dir> --epochs <n>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | (required) | Processed features directory |
| `--output` | `checkpoints` | Checkpoint output directory |
| `--epochs` | `1000` | Number of training epochs |

### `voice-swap cover`

Full pipeline: separate → convert → mix (recommended).

```bash
voice-swap cover \
  --model <checkpoint> \
  --song <input_song> \
  --output <output_file>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Model checkpoint path |
| `--song` | (required) | Input song path |
| `--output` | (required) | Output file path |
| `--reference` | None | Reference audio for voice quality |
| `--instrumental-mix` | `0.15` | Instrumental mix ratio (0.0-1.0) |

### `voice-swap convert`

Convert vocals to your voice (no separation).

```bash
voice-swap convert \
  --model <checkpoint> \
  --input <vocals> \
  --output <output_file>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Model checkpoint path |
| `--input` | (required) | Input audio path |
| `--output` | (required) | Output file path |
| `--reference` | None | Reference audio for voice quality |
| `--instrumental-mix` | `0.15` | Mix with instrumental (0.0 = off) |
| `--no-separate` | False | Skip stem separation |
| `--no-smooth` | False | Skip audio smoothing |
| `--no-denoise` | False | Skip denoising |
| `--no-normalize` | False | Skip volume normalization |

### Output Formats

The `convert` and `cover` commands select the output format from the `--output`
file extension. WAV, FLAC, and MP3 are supported. MP3 encoding requires ffmpeg,
which is already listed as an installation prerequisite.

```bash
voice-swap cover --model checkpoints/best.pth --song song.wav --output my_cover.flac
voice-swap convert --model checkpoints/best.pth --input vocals.wav --output converted.mp3
```

### `voice-swap separate`

Separate audio into vocal and instrumental tracks.

```bash
voice-swap separate --input <audio> --output <output_dir>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input audio file |
| `--output` | (required) | Output directory |
| `--stems` | False | Save individual stems |

---

## Guides

### Guide 1: Recording Quality

**Best practices for recording:**

1. **Microphone**: Use a condenser or dynamic microphone (not laptop mic)
2. **Room**: Quiet room with soft surfaces (carpet, curtains)
3. **Distance**: 6-12 inches from microphone
4. **Pop filter**: Use to reduce plosives
5. **Sample rate**: 44.1kHz, 16-bit or 24-bit WAV

**What to sing:**
- Various pitches (low, mid, high)
- Different dynamics (soft, loud)
- Multiple styles (legato, staccato)
- Include vowels and consonants

**Duration**: 10-30 minutes recommended

### Guide 2: Training Tips

**Epochs:**
- 500 epochs: Quick test, lower quality
- 1000 epochs: Good balance (recommended)
- 2000+ epochs: Higher quality, longer training

**Monitoring loss:**
```bash
# Loss should decrease over time
# Typical values:
# - Initial: 5.0-10.0
# - After 100 epochs: 2.0-4.0
# - After 500 epochs: 0.5-1.5
# - After 1000 epochs: 0.2-0.8
```

**Overfitting signs:**
- Loss increases after decreasing
- Converted audio sounds robotic
- Solution: Use fewer epochs or add more data

### Guide 3: Converting Songs

**Recommended workflow:**

```bash
# Step 1: Separate stems
voice-swap separate --input song.wav --output separated/

# Step 2: Convert vocals
voice-swap convert \
  --model checkpoints/best.pth \
  --input separated/song_vocals.wav \
  --output converted_vocals.wav

# Step 3: Mix manually (optional)
# Use audio software to mix converted_vocals.wav with song_instrumental.wav
```

**Or use the full pipeline:**

```bash
voice-swap cover \
  --model checkpoints/best.pth \
  --song song.wav \
  --output my_cover.wav \
  --instrumental-mix 0.15
```

### Guide 4: Instrumental Mix Ratio

The `--instrumental-mix` option controls how much of the original instrumental is mixed back:

| Ratio | Effect |
|-------|--------|
| `0.0` | Pure converted vocals (may sound dry) |
| `0.1` | Subtle instrumental presence |
| `0.15` | Balanced (recommended) |
| `0.2` | More instrumental |
| `0.3+` | Strong instrumental (may overpower vocals) |

### Guide 5: Reference Audio

A reference audio file helps preserve voice quality:

- Use a clean recording of your voice
- 5-10 seconds is enough
- Match the pitch range of the target song
- Avoid recordings with effects (reverb, delay)

```bash
voice-swap cover \
  --model checkpoints/best.pth \
  --song song.wav \
  --output my_cover.wav \
  --reference my_clean_sample.wav
```

---

## Configuration

Create a `config.yaml` file for custom settings:

```yaml
audio:
  sample_rate: 44100
  hop_length: 512
  n_fft: 2048
  n_mels: 128

model:
  hidden_dim: 256
  n_layers: 6
  n_heads: 8
  spk_dim: 512

train:
  batch_size: 16
  learning_rate: 0.0002
  epochs: 1000
  save_every: 100
  log_every: 10
```

Use with:
```bash
voice-swap --config config.yaml cover --model best.pth --song song.wav --output out.wav
```

---

## Troubleshooting

### "No audio files found"
- Ensure .wav files are in the input directory
- Check file extensions are lowercase

### "CUDA out of memory"
- Reduce batch size in config
- Use shorter audio segments
- Close other GPU-intensive applications

### "ffmpeg not found"
- Install ffmpeg: `brew install ffmpeg` (macOS) or `choco install ffmpeg` (Windows)
- Ensure ffmpeg is in system PATH

### Poor quality output
- Record more training data (20-30 minutes)
- Train for more epochs
- Use a reference audio file
- Adjust instrumental mix ratio

### Robotic sounding output
- Reduce training epochs (overfitting)
- Add more diverse singing samples
- Ensure clean vocal recordings

---

## Architecture

```
Source Audio → Content Encoder → ─┐
                                   ├→ Decoder → Mel → HiFi-GAN → Audio
Reference Audio → Speaker Encoder → ┘
```

**Components:**
- **Content Encoder**: Conformer blocks capture musical content
- **Speaker Encoder**: ECAPA-TDNN captures voice identity
- **Decoder**: Flow-based model combines content + voice
- **Vocoder**: HiFi-GAN generates waveform from mel spectrogram

---

## License

MIT License

---

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) for source separation
- [HiFi-GAN](https://github.com/jik876/hifi-gan) for neural vocoding
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) for speaker encoding
- [Conformer](https://arxiv.org/abs/2005.08100) for content encoding
