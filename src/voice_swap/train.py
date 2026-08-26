"""Training loop for voice conversion model."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .model import VoiceSwapModel
from .hifigan import HiFiGAN, MultiScaleDiscriminator, MultiPeriodDiscriminator
from .losses import TotalLoss, AdversarialLoss, FeatureMatchingLoss
from .preprocess import load_features


class VoiceDataset(Dataset):
    """Dataset for voice conversion training."""

    def __init__(self, features_path: Path):
        self.features = load_features(features_path)
        self.n_samples = self.features["mel"].shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mel = self.features["mel"][idx]
        pitch = self.features["pitch"][idx]

        if self.n_samples > 1:
            ref_idx = torch.randint(0, self.n_samples, (1,)).item()
            ref_mel = self.features["mel"][ref_idx]
        else:
            ref_mel = mel

        return {
            "mel": mel.unsqueeze(0),
            "pitch": pitch,
            "ref_mel": ref_mel.unsqueeze(0),
        }


def train(
    data_path: str | Path,
    output_dir: str | Path,
    config: Config | None = None,
) -> None:
    """Train the voice conversion model."""
    if config is None:
        config = Config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    model = VoiceSwapModel(
        content_dim=config.audio.n_mels,
        hidden_dim=config.model.hidden_dim,
        spk_dim=config.model.spk_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
    ).to(device)

    vocoder = HiFiGAN(in_channels=config.audio.n_mels).to(device)

    msd = MultiScaleDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    # Initialize optimizers
    gen_params = list(model.parameters()) + list(vocoder.parameters())
    optimizer_g = torch.optim.AdamW(
        gen_params,
        lr=config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
    )

    optimizer_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        lr=config.train.learning_rate * 0.5,
        betas=(0.8, 0.99),
        eps=1e-9,
    )

    scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_g,
        max_lr=config.train.learning_rate,
        epochs=config.train.epochs,
        steps_per_epoch=1000,
    )

    scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_d,
        max_lr=config.train.learning_rate * 0.5,
        epochs=config.train.epochs,
        steps_per_epoch=1000,
    )

    # Initialize losses
    criterion = TotalLoss()
    adv_loss = AdversarialLoss()
    feat_match_loss = FeatureMatchingLoss()

    dataset = VoiceDataset(Path(data_path))
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_loss = float("inf")
    total_batches = len(dataloader) * config.train.epochs
    progress_bar = tqdm(
        total=total_batches,
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
    )

    for epoch in range(config.train.epochs):
        model.train()
        vocoder.train()
        msd.train()
        mpd.train()

        total_gen_loss = 0
        total_disc_loss = 0

        progress_bar.set_description(f"Epoch {epoch + 1}/{config.train.epochs}")
        for batch in dataloader:
            source_mel = batch["mel"].to(device)
            ref_mel = batch["ref_mel"].to(device)
            target_mel = batch["mel"].to(device)
            target_pitch = batch["pitch"].to(device)

            # Generator forward
            output_mel, log_det, pred_pitch = model(source_mel, ref_mel)

            # Generate audio
            fake_audio = vocoder(output_mel)
            real_audio = vocoder(target_mel)

            # Discriminator forward
            real_outputs_msd, real_features_msd = msd(real_audio)
            fake_outputs_msd, fake_features_msd = msd(fake_audio)
            real_outputs_mpd, real_features_mpd = mpd(real_audio)
            fake_outputs_mpd, fake_features_mpd = mpd(fake_audio)

            # Generator loss
            losses = criterion(
                output_mel,
                target_mel,
                pred_pitch,
                target_pitch,
                log_det,
                real_outputs_msd + real_outputs_mpd,
                fake_outputs_msd + fake_outputs_mpd,
                real_features_msd + real_features_mpd,
                fake_features_msd + fake_features_mpd,
            )

            gen_loss = (
                losses["mel"] * 10
                + losses["perceptual"] * 5
                + losses["pitch"] * 2
                + losses["kl"] * 0.5
                + losses["gen_adv"] * 2
                + losses["feature_match"] * 5
            )

            optimizer_g.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, config.train.grad_clip)
            optimizer_g.step()
            scheduler_g.step()

            # Discriminator forward
            with torch.no_grad():
                output_mel, _, _ = model(source_mel, ref_mel)
                fake_audio = vocoder(output_mel)

            fake_outputs_msd, _ = msd(fake_audio)
            fake_outputs_mpd, _ = mpd(fake_audio)

            disc_real_msd, disc_fake_msd = adv_loss.discriminator_loss(
                real_outputs_msd, fake_outputs_msd
            )
            disc_real_mpd, disc_fake_mpd = adv_loss.discriminator_loss(
                real_outputs_mpd, fake_outputs_mpd
            )

            disc_loss = (disc_real_msd + disc_fake_msd + disc_real_mpd + disc_fake_mpd) * 0.5

            optimizer_d.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(msd.parameters()) + list(mpd.parameters()),
                config.train.grad_clip,
            )
            optimizer_d.step()
            scheduler_d.step()

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            progress_bar.update()
            progress_bar.set_postfix(
                gen=f"{gen_loss.item():.4f}",
                disc=f"{disc_loss.item():.4f}",
            )

        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)

        if (epoch + 1) % config.train.log_every == 0:
            print(
                f"Epoch {epoch + 1}: gen_loss={avg_gen_loss:.4f}, disc_loss={avg_disc_loss:.4f}"
            )

        if (epoch + 1) % config.train.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "vocoder_state_dict": vocoder.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "gen_loss": avg_gen_loss,
                "disc_loss": avg_disc_loss,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")

            if avg_gen_loss < best_loss:
                best_loss = avg_gen_loss
                torch.save(checkpoint, output_dir / "best.pth")
                print(f"Saved best model (gen_loss={best_loss:.4f})")

    progress_bar.close()
    print("Training complete!")
