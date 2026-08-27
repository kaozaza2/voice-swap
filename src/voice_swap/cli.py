"""CLI entry point for voice-swap."""

from pathlib import Path

import click

from .config import Config
from .preprocess import preprocess_dataset, AUDIO_EXTENSIONS
from .train import train
from .inference import load_model, convert_song, convert_with_instrumental
from .separator import separate_audio
from .utils import print_system_info, check_dependencies


@click.group()
@click.option("--config", type=click.Path(exists=True), default=None, help="Config YAML file")
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto")
@click.pass_context
def main(ctx: click.Context, config: str | None, device: str) -> None:
    """Voice Swap - AI singing voice conversion."""
    ctx.ensure_object(dict)
    if config:
        ctx.obj["config"] = Config.from_yaml(config)
    else:
        ctx.obj["config"] = Config()
    ctx.obj["config"].device = device


@main.command()
@click.option("--input", "-i", required=True, help="Input directory with audio files")
@click.option("--output", "-o", default="data/processed", help="Output directory")
@click.option(
    "--workers",
    "-w",
    type=click.IntRange(min=0),
    default=0,
    help="Parallel worker count (0: choose automatically)",
)
@click.pass_context
def preprocess(ctx: click.Context, input: str, output: str, workers: int) -> None:
    """Preprocess audio files for training."""
    config = ctx.obj["config"]
    n_segments = preprocess_dataset(
        input,
        output,
        config.audio,
        workers=workers or None,
    )
    click.echo(f"Preprocessing complete: {n_segments} segments extracted")


@main.command()
@click.option("--data", "-d", required=True, help="Processed features directory")
@click.option("--output", "-o", default="checkpoints", help="Output directory for checkpoints")
@click.option("--epochs", "-e", type=int, default=None, help="Number of epochs")
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Checkpoint to resume training from",
)
@click.pass_context
def train_model(
    ctx: click.Context,
    data: str,
    output: str,
    epochs: int | None,
    resume: str | None,
) -> None:
    """Train voice conversion model."""
    config = ctx.obj["config"]
    if epochs:
        config.train.epochs = epochs
    train(data, output, config, resume=resume)
    click.echo("Training complete!")


@main.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--input", "-i", required=True, help="Input song path")
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output path (.wav, .flac, .mp3, or .m4a)",
)
@click.option("--reference", "-r", default=None, help="Reference audio for voice quality")
@click.option("--instrumental-mix", default=0.15, help="Mix ratio for instrumental (0.0-1.0)")
@click.option("--no-separate", is_flag=True, help="Skip stem separation")
@click.option("--no-smooth", is_flag=True, help="Skip smoothing")
@click.option("--no-denoise", is_flag=True, help="Skip denoising")
@click.option("--no-normalize", is_flag=True, help="Skip volume normalization")
@click.pass_context
def convert(
    ctx: click.Context,
    model: str,
    input: str,
    output: str,
    reference: str | None,
    instrumental_mix: float,
    no_separate: bool,
    no_smooth: bool,
    no_denoise: bool,
    no_normalize: bool,
) -> None:
    """Convert a song to use your voice."""
    config = ctx.obj["config"]
    model, vocoder = load_model(model, config)
    convert_song(
        model=model,
        vocoder=vocoder,
        input_path=input,
        output_path=output,
        config=config,
        reference_path=reference,
        separate_stems=not no_separate,
        mix_instrumental=instrumental_mix > 0,
        instrumental_mix=instrumental_mix,
        smooth=not no_smooth,
        denoise=not no_denoise,
        normalize=not no_normalize,
    )
    click.echo(f"Conversion complete: {output}")


@main.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--song", "-s", required=True, help="Input song path")
@click.option("--instrumental", "-i", default=None, help="External instrumental track (skips separation)")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (.wav, .flac, .mp3, or .m4a; default: <song>_cover.wav)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite output if exists")
@click.option("--reference", "-r", default=None, help="Reference audio for voice quality")
@click.option("--instrumental-mix", default=0.15, help="Mix ratio for instrumental (0.0-1.0)")
@click.pass_context
def cover(
    ctx: click.Context,
    model: str,
    song: str,
    instrumental: str | None,
    output: str | None,
    force: bool,
    reference: str | None,
    instrumental_mix: float,
) -> None:
    """Full pipeline: separate vocals, convert, mix back (recommended)."""
    from pathlib import Path

    config = ctx.obj["config"]
    song_path = Path(song)

    if output is None:
        output = str(song_path.parent / f"{song_path.stem}_cover.wav")

    if not force and Path(output).exists():
        click.echo(f"Error: {output} already exists. Use -f to overwrite.")
        raise SystemExit(1)

    model, vocoder = load_model(model, config)
    convert_with_instrumental(
        model=model,
        vocoder=vocoder,
        song_path=song,
        output_path=output,
        config=config,
        reference_path=reference,
        instrumental_path=instrumental,
        instrumental_mix=instrumental_mix,
    )
    click.echo(f"Cover complete: {output}")


@main.command()
@click.option("--input", "-i", required=True, help="Input audio file")
@click.option("--output", "-o", required=True, help="Output directory")
@click.option("--stems", is_flag=True, help="Save individual stems")
@click.pass_context
def separate(ctx: click.Context, input: str, output: str, stems: bool) -> None:
    """Separate audio into vocal and instrumental tracks."""
    results = separate_audio(input, output)
    for name, path in results.items():
        click.echo(f"{name}: {path}")


@main.command()
def info() -> None:
    """Show system information and dependencies."""
    print_system_info()
    click.echo("\nDependencies:")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "OK" if available else "MISSING"
        click.echo(f"  {name}: {status}")
    click.echo("\nSupported audio formats:")
    click.echo(f"  {', '.join(sorted(AUDIO_EXTENSIONS))}")


if __name__ == "__main__":
    main()
