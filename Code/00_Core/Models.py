"""Model variants for H2T-CBD: BlackBox, FeatConcat, and H2TCBD with anchor heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class BlackBox(nn.Module):
    """Plain MLP regressor. 768 -> 256 -> 256 -> 1 (sigmoid). No bottleneck."""

    def __init__(self, in_dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, feats=None):
        h = self.net(x)
        y = torch.sigmoid(self.head(h)).squeeze(-1)
        return {"score": y}


class FeatConcat(nn.Module):
    """Ablation variant: concat raw features into the MLP, no bottleneck.
    Only usable when 19d features are available (feat dataset)."""

    def __init__(self, in_dim=768, n_feat=19, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_feat, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, feats=None):
        if feats is None:
            feats = torch.zeros(x.size(0), self.net[0].in_features - x.size(1), device=x.device)
        h = self.net(torch.cat([x, feats], dim=-1))
        y = torch.sigmoid(self.head(h)).squeeze(-1)
        return {"score": y}


class H2TCBD(nn.Module):
    """Holistic-to-Trait Concept-Bottleneck Distillation.

    emb -> encoder -> z (K=6 bottleneck dims) -> readout -> holistic score.

    When use_anchor=True, K small linear heads each regress one bottleneck
    dimension onto its corresponding anchor feature group (pre-normalised).
    The anchor loss ties each z_k to a known linguistic competency.
    """

    def __init__(self, in_dim=768, K=6, hidden=256, dropout=0.2,
                 anchor_dims=None, use_anchor=True):
        super().__init__()
        self.K = K
        self.use_anchor = use_anchor
        self.anchor_dims = anchor_dims or []
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.bottleneck = nn.Linear(hidden, K)
        self.readout = nn.Linear(K, 1)
        if use_anchor and anchor_dims:
            self.anchor_heads = nn.ModuleList([
                nn.Linear(1, len(dims)) for dims in anchor_dims
            ])
        else:
            self.anchor_heads = None

    def forward(self, x, feats=None):
        h = self.encoder(x)
        z = self.bottleneck(h)                # (B, K)
        y = torch.sigmoid(self.readout(z)).squeeze(-1)
        out = {"score": y, "z": z}
        if self.anchor_heads is not None:
            preds = []
            for k, head in enumerate(self.anchor_heads):
                preds.append(head(z[:, k:k + 1]))
            out["anchor_pred"] = preds        # list[(B, m_k)]
        return out
