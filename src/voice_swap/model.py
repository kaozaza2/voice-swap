"""Voice conversion model with advanced architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_rel: int = 128):
        super().__init__()
        self.dim = dim
        self.max_rel = max_rel
        self.weight = nn.Parameter(torch.randn(2 * max_rel + 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_rel, self.max_rel) + self.max_rel
        rel_emb = self.weight[rel_pos]
        return x + rel_emb.mean(dim=-1).unsqueeze(-1)


class ConformerBlock(nn.Module):
    """Conformer block with feed-forward, attention, and convolution."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 15, padding=7, groups=dim),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, 1),
        )
        self.norm2 = nn.LayerNorm(dim)

        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)

        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = residual + x

        x = x + 0.5 * self.ff2(x)
        return self.norm3(x)


class ContentEncoder(nn.Module):
    """Encode audio content with Conformer blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList(
            [ConformerBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ECAPAEncoder(nn.Module):
    """ECAPA-TDNN style speaker encoder for better voice cloning."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList()
        self.residual_norms = nn.ModuleList()
        for i in range(n_layers):
            channels = hidden_dim
            self.residual_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                    nn.Conv1d(channels, channels, 1),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                    nn.Conv1d(channels, channels, 1),
                    nn.BatchNorm1d(channels),
                )
            )
            self.residual_norms.append(nn.LayerNorm(channels))

        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, 1),
            nn.Tanh(),
            nn.Conv1d(128, 1, 1),
            nn.Softmax(dim=-1),
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.first(x)

        for block, norm in zip(self.residual_blocks, self.residual_norms):
            residual = x
            x = block(x)
            x = residual + x
            x = norm(x.transpose(1, 2)).transpose(1, 2)

        w = self.attention(x)
        x = (x * w).sum(dim=-1)
        return self.out(x)


class PitchPredictor(nn.Module):
    """Predict pitch contour for smoother intonation."""

    def __init__(self, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
            ])
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(out).squeeze(-1)


class FlowDecoder(nn.Module):
    """Flow-based decoder for better audio quality."""

    def __init__(self, hidden_dim: int, spk_dim: int, out_dim: int, n_flows: int = 4):
        super().__init__()
        self.combine = nn.Linear(hidden_dim + spk_dim, hidden_dim)

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                )
            )

        self.mel_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        content: torch.Tensor,
        spk_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spk = spk_embedding.unsqueeze(1).expand(-1, content.size(1), -1)
        x = torch.cat([content, spk], dim=-1)
        x = self.combine(x)

        log_det = torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            h = flow(x)
            mean, log_std = h.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -5, 5)
            x = x * torch.exp(log_std) + mean
            log_det = log_det + log_std.sum(dim=(1, 2))

        return self.mel_proj(x), log_det

    def inference(
        self,
        content: torch.Tensor,
        spk_embedding: torch.Tensor,
    ) -> torch.Tensor:
        spk = spk_embedding.unsqueeze(1).expand(-1, content.size(1), -1)
        x = torch.cat([content, spk], dim=-1)
        x = self.combine(x)

        for flow in self.flows:
            h = flow(x)
            mean, log_std = h.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -5, 5)
            x = x * torch.exp(log_std) + mean

        return self.mel_proj(x)


class VoiceSwapModel(nn.Module):
    """Advanced voice conversion model with Conformer + ECAPA + Flow."""

    def __init__(
        self,
        content_dim: int = 128,
        hidden_dim: int = 256,
        spk_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.content_encoder = ContentEncoder(
            content_dim, hidden_dim, n_layers, n_heads, dropout
        )
        self.speaker_encoder = ECAPAEncoder(content_dim, hidden_dim, spk_dim)
        self.decoder = FlowDecoder(hidden_dim, spk_dim, content_dim)
        self.pitch_predictor = PitchPredictor(hidden_dim)

    def forward(
        self,
        source_mel: torch.Tensor,
        reference_mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content = self.content_encoder(source_mel)
        spk_embed = self.speaker_encoder(reference_mel)
        output, log_det = self.decoder(content, spk_embed)
        pitch = self.pitch_predictor(content)
        return output, log_det, pitch

    def convert(
        self,
        source_mel: torch.Tensor,
        reference_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Inference mode - no gradients."""
        content = self.content_encoder(source_mel)
        spk_embed = self.speaker_encoder(reference_mel)
        return self.decoder.inference(content, spk_embed)
