"""Causally Weighted Variational Information Bottleneck (CW-VIB).

Inputs:
    target: target feature map, shape [N, C, H, W].
    background: background prototype vectors, shape [N, K, C].
    labels: optional zero-based foreground class labels, shape [N].

Outputs:
    features: context-regulated feature map, shape [N, C, H, W].
    risk: prototype risk scores, shape [N, K].
    gate: fusion gates, shape [N, K, 1].
    kl: scalar information-bottleneck loss.
    auxiliary: optional on/off classification supervision.

Implementation settings:
    Background prototypes are sum-pooled. The gate is scalar per prototype.
    Risk is detached by default; labels provide auxiliary supervision for the
    shared risk head. Add the returned auxiliary loss to the training objective
    when using that setting. Its coefficient is set by the training caller.
    Training uses Gaussian reparameterization; evaluation uses the latent mean.

Example:
    module = CWVIB(channels=64, num_classes=12)
    output = module(target, background, labels=labels)
    # The detector consumes output["features"].
    loss = detection_loss + 0.001 * output["kl"] + 0.1 * output["auxiliary"]
    # These loss coefficients illustrate caller configuration.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def js_divergence(p, q):
    """Jensen-Shannon divergence of probability vectors, measured in nats."""
    eps = torch.finfo(p.dtype).eps
    mean = (p + q) / 2
    log_mean = mean.clamp_min(eps).log()
    return 0.5 * (
        (p * (p.clamp_min(eps).log() - log_mean)).sum(-1)
        + (q * (q.clamp_min(eps).log() - log_mean)).sum(-1)
    )


class CWVIB(nn.Module):
    """Prototype-wise risk estimation, variational encoding and gated fusion.

    Modes:
        cwvib: risk-weighted KL and risk-modulated fusion.
        vib: unweighted KL and learned fusion gates.
        no_vib: deterministic latent encoding with zero KL loss.
    """

    def __init__(self, channels, num_classes, latent_dim=None,
                 alpha=1.0, detach_risk=True):
        super().__init__()
        latent_dim = channels if latent_dim is None else latent_dim
        if channels < 1 or num_classes < 2 or latent_dim < 1 or alpha < 0:
            raise ValueError("Require positive dimensions, >=2 classes and alpha >=0")
        self.channels = channels
        self.alpha = alpha
        self.detach_risk = detach_risk
        self.head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, num_classes),
        )
        self.encoder = nn.Linear(channels, 2 * latent_dim)
        self.gate = nn.Linear(channels + latent_dim, 1)
        self.project = nn.Linear(latent_dim, channels, bias=False)

    def intervention(self, target_summary, background):
        """Keep/remove each prototype while holding all other inputs fixed."""
        total = background.sum(1)
        count = background.shape[1]
        target_repeated = target_summary[:, None].expand(-1, count, -1)
        on_logits = self.head(torch.cat((target_summary, total), dim=-1))
        off_logits = self.head(torch.cat(
            (target_repeated, total[:, None] - background), dim=-1))
        standalone_logits = self.head(torch.cat(
            (torch.zeros_like(background), background), dim=-1))
        on = on_logits.softmax(-1)[:, None].expand_as(off_logits)
        off = off_logits.softmax(-1)
        influence = js_divergence(on, off).clamp(0, math.log(2))
        standalone = standalone_logits.softmax(-1).amax(-1)
        return influence * standalone, on_logits, off_logits

    def forward(self, target, background, mode="cwvib", labels=None):
        if target.ndim != 4 or background.ndim != 3:
            raise ValueError("Expected target [N,C,H,W] and background [N,K,C]")
        if (target.shape[0] != background.shape[0]
                or target.shape[1] != self.channels
                or background.shape[2] != self.channels
                or background.shape[1] == 0):
            raise ValueError("Incompatible feature dimensions or empty prototype set")
        if mode not in ("cwvib", "vib", "no_vib"):
            raise ValueError(f"Unknown mode: {mode}")

        summary = target.mean((2, 3))
        risk_raw, on_logits, off_logits = self.intervention(summary, background)
        risk = risk_raw.detach() if self.detach_risk else risk_raw

        # q(z_k | b_k) = N(mu_k, diag(sigma_k^2)).
        mu, logvar = self.encoder(background).chunk(2, dim=-1)
        logvar = logvar.clamp(-12, 8)
        if self.training and mode != "no_vib":
            latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            latent = mu

        target_repeated = summary[:, None].expand(-1, latent.shape[1], -1)
        gate = self.gate(torch.cat((target_repeated, latent), dim=-1)).sigmoid()
        if mode == "cwvib":
            gate = gate * (1 - risk[..., None])
        context = self.project((gate * latent).sum(1))
        fused = target + context[:, :, None, None]

        # Gaussian KL to N(0,I), summed over latent dimensions.
        kl_per_prototype = 0.5 * (
            mu.square() + logvar.exp() - 1 - logvar
        ).sum(-1)
        if mode == "cwvib":
            kl = ((1 + self.alpha * risk) * kl_per_prototype).mean()
        elif mode == "vib":
            kl = kl_per_prototype.mean()
        else:
            kl = mu.sum() * 0

        auxiliary = on_logits.sum() * 0
        if labels is not None:
            off_labels = labels[:, None].expand(-1, background.shape[1]).reshape(-1)
            auxiliary = F.cross_entropy(on_logits, labels)
            auxiliary = auxiliary + F.cross_entropy(
                off_logits.reshape(-1, on_logits.shape[-1]), off_labels)

        return {
            "features": fused,
            "risk": risk_raw,
            "gate": gate,
            "kl": kl,
            "auxiliary": auxiliary,
            "on_logits": on_logits,
            "off_logits": off_logits,
        }
