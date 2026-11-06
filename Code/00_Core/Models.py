import torch
from torch import nn

import Common as C


def encoder(in_dim, hidden, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))


class BlackBox(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=.2):
        super().__init__()
        self.encoder = encoder(in_dim, hidden, dropout)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x, feats=None):
        return {'score': torch.sigmoid(self.readout(self.encoder(x))).squeeze(-1)}


class FeatConcat(BlackBox):
    def __init__(self, in_dim, hidden=256, dropout=.2):
        super().__init__(in_dim + len(C.FEAT_COLS), hidden, dropout)

    def forward(self, x, feats=None):
        return super().forward(torch.cat([x, feats], dim=-1))


class H2TCBD(nn.Module):
    def __init__(self, in_dim, kept_groups, score_mask=True, hidden=256, dropout=.2):
        super().__init__()
        self.kept_groups = tuple(sorted(kept_groups))
        self.encoder = encoder(in_dim, hidden, dropout)
        self.bottleneck = nn.Linear(hidden, 6)
        self.readout = nn.Linear(6, 1)
        heads = [nn.Linear(1, len(dims)) for dims in C.ANCHOR_DIMS]
        self.anchor_heads = nn.ModuleDict({str(k): heads[k] for k in self.kept_groups})
        mask = [float(not score_mask or k in self.kept_groups) for k in range(6)]
        self.register_buffer('score_mask', torch.tensor(mask))

    def forward(self, x, feats=None):
        z = self.bottleneck(self.encoder(x))
        active = z * self.score_mask
        preds = {int(k): head(z[:, int(k):int(k) + 1]) for k, head in self.anchor_heads.items()}
        return {'score': torch.sigmoid(self.readout(active)).squeeze(-1), 'z': z,
                'anchor_pred': preds, 'score_terms': active * self.readout.weight[0]}


def make_model(variant, in_dim, kept_groups=None, hidden=256, dropout=.2):
    if variant == 'blackbox':
        return BlackBox(in_dim, hidden, dropout)
    if variant == 'featconcat':
        return FeatConcat(in_dim, hidden, dropout)
    if variant == 'h2tcbd':
        return H2TCBD(in_dim, range(6), True, hidden, dropout)
    if variant == 'no_anchor':
        return H2TCBD(in_dim, [], False, hidden, dropout)
    if variant in ('selective', 'soft_selective'):
        if not kept_groups:
            raise ValueError('ARS retained no groups; no fallback scoring model is defined.')
        return H2TCBD(in_dim, kept_groups, variant == 'selective', hidden, dropout)
    raise ValueError(f'Unknown variant: {variant}')
