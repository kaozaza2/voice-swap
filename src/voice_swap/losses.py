"""Loss functions for voice conversion training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel spectrogram loss."""

    def __init__(self, n_mels_list: list[int] = [64, 128, 256]):
        super().__init__()
        import torchaudio

        self.n_mels_list = n_mels_list
        # Registered as submodules so the filterbanks are built once and follow
        # the module onto whichever device training runs on.
        self.transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=44100,
                n_fft=2048,
                hop_length=512,
                n_mels=n_mels,
            )
            for n_mels in n_mels_list
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for transform in self.transforms:
            pred_mel = self._compute_mel(transform, pred)
            target_mel = self._compute_mel(transform, target)
            loss += F.l1_loss(pred_mel, target_mel)
        return loss / len(self.transforms)

    @staticmethod
    def _compute_mel(transform: nn.Module, audio: torch.Tensor) -> torch.Tensor:
        return torch.log(transform(audio) + 1e-7)


class PerceptualLoss(nn.Module):
    """Perceptual loss using spectrogram features."""

    def __init__(self):
        super().__init__()
        self.n_fft = 2048
        self.hop_length = 512

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_spec = torch.stft(
            pred.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        ).abs()
        target_spec = torch.stft(
            target.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        ).abs()
        return F.l1_loss(pred_spec, target_spec)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""

    def __init__(self):
        super().__init__()

    def generator_loss(self, disc_outputs: list[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for fake in disc_outputs:
            loss += torch.mean((fake - 1) ** 2)
        return loss / len(disc_outputs)

    def discriminator_loss(
        self,
        real_outputs: list[torch.Tensor],
        fake_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_loss = 0.0
        fake_loss = 0.0

        for real, fake in zip(real_outputs, fake_outputs):
            real_loss += torch.mean((real - 1) ** 2)
            fake_loss += torch.mean(fake**2)

        return real_loss / len(real_outputs), fake_loss / len(fake_outputs)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for stable GAN training."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_features: list[list[torch.Tensor]],
        fake_features: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        loss = 0.0
        for real_feats, fake_feats in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feats, fake_feats):
                loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss


class PitchLoss(nn.Module):
    """Pitch prediction loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_pitch: torch.Tensor, target_pitch: torch.Tensor) -> torch.Tensor:
        mask = target_pitch > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_pitch.device)
        return F.mse_loss(pred_pitch[mask], target_pitch[mask])


class TotalLoss(nn.Module):
    """Combined loss for voice conversion."""

    def __init__(self):
        super().__init__()
        self.mel_loss = MelSpectrogramLoss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.pitch_loss = PitchLoss()

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        log_det: torch.Tensor | None = None,
        disc_real_outputs: list | None = None,
        disc_fake_outputs: list | None = None,
        real_features: list | None = None,
        fake_features: list | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # Reconstruction losses
        losses["mel"] = self.mel_loss(pred_mel, target_mel)
        losses["perceptual"] = self.perceptual_loss(pred_mel, target_mel)
        losses["pitch"] = self.pitch_loss(pred_pitch, target_pitch)

        # Flow Jacobian penalty. `-0.5 * mean(1 + log_det - log_det.exp())`
        # overflowed to inf within a few updates: log_det is summed over every
        # frame and channel, so exp() of it saturates. Penalise the per-frame
        # magnitude instead, which keeps the affine scales near unity.
        if log_det is not None:
            n_frames = pred_pitch.shape[-1]
            losses["kl"] = (log_det / n_frames).pow(2).mean()

        # Adversarial losses
        if disc_real_outputs is not None and disc_fake_outputs is not None:
            losses["gen_adv"] = self.adversarial_loss.generator_loss(disc_fake_outputs)
            losses["disc_real"], losses["disc_fake"] = self.adversarial_loss.discriminator_loss(
                disc_real_outputs, disc_fake_outputs
            )

        # Feature matching
        if real_features is not None and fake_features is not None:
            losses["feature_match"] = self.feature_matching_loss(real_features, fake_features)

        return losses
