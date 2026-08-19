"""Training engine: load cached embeddings and train/evaluate model variants.

Protocols:
  P-in:     in-dataset 80/10/10 stratified split.
  P-LOPO:   leave-one-prompt-out (feat only, 8 folds).
  P-cross:  train on source, zero-shot evaluation on target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common as C
from models import BlackBox, FeatConcat, H2TCBD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Column indices of each anchor group within the 19d normalised feature vector.
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]


def load_emb(name):
    d = np.load(C.EMB / f"{name}.npz")
    out = {"emb": d["emb"], "score": d["score"]}
    if "essay_set" in d:
        out["essay_set"] = d["essay_set"]
    if "feats" in d:
        out["feats"] = d["feats"]
    return out


class FeatNorm:
    """Z-score feature normalisation; fit on the train split, applied to val/test."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def make_model(variant, in_dim=768, K=6):
    """Model factory. Supported variants: blackbox, h2tcbd, no_anchor, featconcat."""
    if variant == "blackbox":
        return BlackBox(in_dim=in_dim)
    if variant == "featconcat":
        return FeatConcat(in_dim=in_dim, n_feat=len(C.FEAT_COLS))
    if variant == "h2tcbd":
        return H2TCBD(in_dim=in_dim, K=K, anchor_dims=ANCHOR_DIMS, use_anchor=True)
    if variant == "no_anchor":
        return H2TCBD(in_dim=in_dim, K=K, anchor_dims=ANCHOR_DIMS, use_anchor=False)
    raise ValueError(variant)


def train_eval(variant, tr, va, te, scoremap_tr, K=6, lam_anchor=0.5,
               epochs=60, lr=1e-3, bs=256, seed=42, ml=None, feat_norm=None,
               te_scoremap=None):
    """Train a model and evaluate on the test set.

    tr/va/te are dicts with keys: emb, score, [feats], [essay_set].
    scoremap_tr is the training-set ScaleMap; te_scoremap inverse-maps test
    predictions when the test set has a different score range (P-cross).
    """
    C.set_seed(seed)

    model = make_model(variant, in_dim=tr["emb"].shape[1], K=K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xtr = torch.tensor(tr["emb"], dtype=torch.float32)
    ytr = torch.tensor(scoremap_tr.to_unit(tr["score"]), dtype=torch.float32)
    has_feat_tr = "feats" in tr
    if has_feat_tr and feat_norm is not None:
        Ftr = torch.tensor(feat_norm.transform(tr["feats"]), dtype=torch.float32)
    else:
        Ftr = None

    Xva = torch.tensor(va["emb"], dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(te["emb"], dtype=torch.float32).to(DEVICE)
    Fte = None
    if "feats" in te and feat_norm is not None and variant == "featconcat":
        Fte = torch.tensor(feat_norm.transform(te["feats"]), dtype=torch.float32).to(DEVICE)
    Fva = None
    if "feats" in va and feat_norm is not None and variant == "featconcat":
        Fva = torch.tensor(feat_norm.transform(va["feats"]), dtype=torch.float32).to(DEVICE)

    n = len(Xtr)
    mse = nn.MSELoss()
    best_qwk, best_state, best_ep = -1e9, None, -1
    te_map = te_scoremap or scoremap_tr

    def evaluate(X, F, raw_scores, smap):
        model.eval()
        with torch.no_grad():
            feats_arg = F if (variant == "featconcat") else None
            pred_u = model(X, feats=feats_arg)["score"].cpu().numpy()
        pred_int = smap.from_unit(pred_u)
        return C.qwk(raw_scores, pred_int, smap.smin, smap.smax), pred_int, pred_u

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(DEVICE)
            yb = ytr[idx].to(DEVICE)
            fb = Ftr[idx].to(DEVICE) if Ftr is not None else None
            feats_arg = fb if (variant == "featconcat") else None
            out = model(xb, feats=feats_arg)
            loss = mse(out["score"], yb)

            # Anchor loss: each z_k regresses its feature group (feat only).
            if "anchor_pred" in out and fb is not None and lam_anchor > 0:
                a_loss = 0.0
                for k, dims in enumerate(ANCHOR_DIMS):
                    target = fb[:, dims]
                    a_loss = a_loss + mse(out["anchor_pred"][k], target)
                loss = loss + lam_anchor * a_loss / len(ANCHOR_DIMS)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        train_loss = tot / n
        vq, _, _ = evaluate(Xva, Fva, va["score"], scoremap_tr)
        if ml is not None:
            ml.log(epoch=ep, train_loss=round(train_loss, 5), val_qwk=round(vq, 4),
                   variant=variant, seed=seed)
        if vq > best_qwk:
            best_qwk, best_ep = vq, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 12:   # early stopping
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        # Persist the best model weights next to the run's metrics so the
        # representation-alignment analysis can use the real trained models.
        if ml is not None:
            torch.save(best_state, Path(ml.exp_dir) / "model.pt")
    tq, pred_int, pred_u = evaluate(Xte, Fte, te["score"], te_map)
    metrics = {
        "test_qwk": round(tq, 4),
        "test_rmse": round(C.rmse(te["score"], pred_int), 4),
        "test_pearson": round(C.pearson(te["score"], pred_int), 4),
        "val_qwk": round(best_qwk, 4),
        "best_epoch": best_ep,
        "n_train": int(n), "n_test": int(len(te["score"])),
    }
    return metrics, model
