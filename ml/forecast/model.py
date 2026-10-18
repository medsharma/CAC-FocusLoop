"""LapseForecaster network. Raw inputs; per-feature normalisation is baked in as buffers so the
exported Core ML model takes raw FeatureFrame / ContentFeatures values."""
from __future__ import annotations

import torch
from torch import nn

from data import N_CONTENT, N_FEAT


class Standardize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("inv_std", 1.0 / torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-2).reshape(1, 1, -1))

    def forward(self, x):
        return torch.clamp((x - self.mean) * self.inv_std, -8.0, 8.0)


class LapseNet(nn.Module):
    """features (B,200,42) [+ content (B,20,11)] -> logit (B,1)."""

    def __init__(self, use_content: bool, f_mean, f_std, c_mean=None, c_std=None, width: int = 48):
        super().__init__()
        self.use_content = use_content
        self.fnorm = Standardize(f_mean, f_std)
        self.fconv = nn.Sequential(
            nn.Conv1d(N_FEAT, width, 5, stride=2, padding=2), nn.ReLU(),   # 200 -> 100
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.ReLU(),    # 100 -> 50
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.ReLU(),    # 50 -> 25
            nn.AvgPool1d(5),                                               # 25 -> 5 (keeps a coarse trend)
        )
        d = width * 5
        if use_content:
            self.cnorm = Standardize(c_mean, c_std)
            self.cconv = nn.Sequential(nn.Conv1d(N_CONTENT, 16, 3, padding=1), nn.ReLU(), nn.AvgPool1d(4))  # 20 -> 5
            d += 16 * 5
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def logits(self, features, content=None):
        h = self.fconv(self.fnorm(features).transpose(1, 2)).flatten(1)
        if self.use_content:
            c = self.cconv(self.cnorm(content).transpose(1, 2)).flatten(1)
            h = torch.cat([h, c], 1)
        return self.head(h)

    def forward(self, features, content=None):
        return torch.sigmoid(self.logits(features, content))


def focal_loss(logits, y, alpha: float, gamma: float = 2.0):
    """Binary focal loss (Lin et al. 2017). alpha weights the positive class."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()
