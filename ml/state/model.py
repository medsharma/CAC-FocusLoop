"""Attention-state temporal network: a small dilated 1D-CNN over 10 s windows (100 x 42 raw features).

Per-feature normalisation (mean / std from the training split) and the calibration (temperature +
per-class bias) live inside the module as buffers / parameters, so the exported Core ML model takes
RAW FeatureFrame.vector values.

Pooling = [mean over the window, max over the window, mean over the last `tail` steps]. The target
is the label of the window's LAST frame, so the tail branch lets the network tell "lapse now" from
"lapse earlier in the window" (plain global pooling cannot).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data import CLASSES, N_FEAT, WINDOW

EMBED_DIM = 24
CLIP = 8.0  # clamp normalised inputs (robustness to out-of-range real-world values; fp16-safe)


class AttentionStateNet(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, width: int = 48, embed: int = EMBED_DIM,
                 n_cls: int = len(CLASSES), dropout: float = 0.2, layers: int = 4, tail: int = 20):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_feat))
        self.register_buffer("std", torch.ones(n_feat))
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("class_bias", torch.zeros(n_cls))   # post-hoc calibration offset (logits / T + b)
        self.layers, self.tail = layers, tail
        k = 5
        mods, c_in = [], n_feat
        for i in range(layers):
            d = 2 ** i
            mods += [nn.Conv1d(c_in, width, k, padding=2 * d, dilation=d), nn.BatchNorm1d(width), nn.ReLU()]
            c_in = width
        self.conv = nn.Sequential(*mods)
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Linear((3 if tail else 2) * width, embed)
        self.head = nn.Linear(embed, n_cls)

    def set_norm(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-3))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 100, 42) raw -> embedding (B, E)."""
        z = torch.clamp((x - self.mean) / self.std, -CLIP, CLIP)
        h = self.conv(z.transpose(1, 2))                       # (B, C, T)
        pools = [h.mean(dim=2), h.amax(dim=2)]
        if self.tail:
            pools.append(h[:, :, -self.tail:].mean(dim=2))
        return torch.relu(self.embed(self.drop(torch.cat(pools, dim=1))))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward(self, x: torch.Tensor):
        """Returns (probabilities (B,4) calibrated softmax, embedding (B,E))."""
        e = self.features(x)
        p = torch.softmax(self.head(e) / self.temperature + self.class_bias, dim=1)
        return p, e


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(path, model: AttentionStateNet, extra: dict):
    torch.save({"state_dict": model.state_dict(),
                "config": {"width": model.conv[0].out_channels, "embed": model.embed.out_features,
                           "window": WINDOW, "n_feat": model.mean.numel(), "layers": model.layers,
                           "tail": model.tail},
                **extra}, path)


def load_checkpoint(path) -> tuple[AttentionStateNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = AttentionStateNet(n_feat=cfg["n_feat"], width=cfg["width"], embed=cfg["embed"],
                          layers=cfg.get("layers", 3), tail=cfg.get("tail", 0))
    sd = ck["state_dict"]
    sd.setdefault("class_bias", torch.zeros(len(CLASSES)))
    m.load_state_dict(sd)
    m.eval()
    return m, ck
