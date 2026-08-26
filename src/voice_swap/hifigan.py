"""HiFi-GAN Vocoder for high-quality audio synthesis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels, channels, kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        )
        self.act1 = nn.LeakyReLU(0.1)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x + residual


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for adversarial training."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._build_discriminator(),
            self._build_discriminator(),
            self._build_discriminator(),
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def _build_discriminator(self) -> nn.Module:
        return nn.ModuleDict({
            "conv1": weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            "conv2": weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            "conv3": weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            "conv4": weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            "conv5": weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            "conv6": weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            "conv7": weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            "out": weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
        })

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        features = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)

            feat = F.leaky_relu(disc["conv1"](x), 0.1)
            feat_list = [feat]

            feat = F.leaky_relu(disc["conv2"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv3"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv4"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv5"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv6"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv7"](feat), 0.1)
            feat_list.append(feat)

            out = disc["out"](feat)
            outputs.append(out)
            features.append(feat_list)

        return outputs, features


class HiFiGAN(nn.Module):
    """HiFi-GAN vocoder for mel-to-audio conversion."""

    def __init__(
        self,
        in_channels: int = 128,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        channels: int = 512,
    ):
        super().__init__()

        self.pre_conv = weight_norm(nn.Conv1d(in_channels, channels, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        ch = channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2))
            )
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(ch, k, d))

        self.post_conv = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = None
            for j in range(3):
                if xs is None:
                    xs = self.resblocks[i * 3 + j](x)
                else:
                    xs = xs + self.resblocks[i * 3 + j](x)
            x = xs / 3

        x = F.leaky_relu(x)
        x = self.post_conv(x)
        return torch.tanh(x)

    def discriminator_forward(self, audio: torch.Tensor) -> tuple[list, list]:
        """Forward pass for discriminator training."""
        return self.discriminator(audio)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for better quality."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._build_discriminator(2),
            self._build_discriminator(3),
            self._build_discriminator(5),
            self._build_discriminator(7),
            self._build_discriminator(11),
        ])

    def _build_discriminator(self, period: int) -> nn.Module:
        return nn.ModuleDict({
            "conv1": weight_norm(nn.Conv2d(1, 32, (1, period), (1, 1), (0, period // 2))),
            "conv2": weight_norm(nn.Conv2d(32, 128, (1, period), (1, 1), (0, period // 2))),
            "conv3": weight_norm(nn.Conv2d(128, 512, (1, period), (1, 1), (0, period // 2))),
            "conv4": weight_norm(nn.Conv2d(512, 1024, (1, period), (1, 1), (0, period // 2))),
            "conv5": weight_norm(nn.Conv2d(1024, 1024, (1, period), (1, 1), (0, period // 2))),
            "out": weight_norm(nn.Conv2d(1024, 1, (1, 1), (1, 1), (0, 0))),
        })

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        features = []

        for disc in self.discriminators:
            feat = F.leaky_relu(disc["conv1"](x), 0.1)
            feat_list = [feat]

            feat = F.leaky_relu(disc["conv2"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv3"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv4"](feat), 0.1)
            feat_list.append(feat)

            feat = F.leaky_relu(disc["conv5"](feat), 0.1)
            feat_list.append(feat)

            out = disc["out"](feat)
            outputs.append(out)
            features.append(feat_list)

        return outputs, features
