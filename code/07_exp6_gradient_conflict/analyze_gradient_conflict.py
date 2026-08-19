"""Gradient conflict analysis: MSE gradient vs anchor gradient on each LOPO fold.

For each fold, runs one batch through an untrained H2T-CBD, computes
gradients of MSE loss and anchor loss on shared parameters, then measures
cosine similarity. Compares benefit folds (1/2/8) vs hurt folds (3/4/5/6).
"""

from __future__ import annotations

import sys, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))

import common as C
from engine import load_emb, FeatNorm, make_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]
SEEDS = [13, 42, 123]


def get_shared_params(model):
    """encoder + bottleneck"""
    params = []
    params.extend(model.encoder.parameters())
    params.extend(model.bottleneck.parameters())
    return params


def compute_cosine(g_vecs_1, g_vecs_2):
    """Flatten two gradient lists and return their cosine similarity."""
    g1 = torch.cat([g.flatten() for g in g_vecs_1])
    g2 = torch.cat([g.flatten() for g in g_vecs_2])
    dot = (g1 * g2).sum()
    norm1 = g1.norm()
    norm2 = g2.norm()
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def analyze_fold(d, fold, seed=42):
    """Analyze the MSE-vs-anchor gradient conflict on one LOPO fold."""
    C.set_seed(seed)
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]

    # Train/test split.
    te_mask = d["essay_set"] == fold
    tr_mask = ~te_mask
    tr_idx = np.where(tr_mask)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(tr_idx)
    n_va = max(1, int(0.1 * len(tr_idx)))
    itr = tr_idx[n_va:]

    # Feature standardisation.
    fn = FeatNorm().fit(d["feats"][itr])

    # Take one batch (up to 256 essays).
    n_batch = min(256, len(itr))
    batch_idx = itr[:n_batch]

    X = torch.tensor(d["emb"][batch_idx], dtype=torch.float32).to(DEVICE)
    F = torch.tensor(fn.transform(d["feats"][batch_idx]), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(
        C.ScaleMap(smin, smax).fit(d["score"][itr]).to_unit(d["score"][batch_idx]),
        dtype=torch.float32,
    ).to(DEVICE)

    # Average over 10 random initialisations per seed (3 seeds in main).
    cosines = []
    for _ in range(10):
        model = make_model("h2tcbd", in_dim=X.shape[1], K=6).to(DEVICE)
        mse = nn.MSELoss()
        shared = get_shared_params(model)

        # ---- g_MSE ----
        model.zero_grad()
        out = model(X)
        loss_mse = mse(out["score"], y)
        loss_mse.backward(retain_graph=True)
        g_mse = [p.grad.detach().clone() for p in shared]

        # ---- g_anchor ----
        model.zero_grad()
        out2 = model(X)
        anchor_loss = 0.0
        for k, head in enumerate(model.anchor_heads):
            anchor_loss = anchor_loss + mse(head(out2["z"][:, k:k + 1]), F[:, ANCHOR_DIMS[k]])
        anchor_loss = anchor_loss / len(ANCHOR_DIMS)
        anchor_loss.backward()
        g_anchor = [p.grad.detach().clone() for p in shared]

        cos = compute_cosine(g_mse, g_anchor)
        cosines.append(cos)

    return cosines, len(itr)


def main():
    d = load_emb("feat")
    sets = sorted(np.unique(d["essay_set"]).tolist())

    results = {}
    for fold in sets:
        all_cos = []
        n_train = None
        for s in SEEDS:
            cos_list, n_train = analyze_fold(d, fold, seed=s)
            all_cos.extend(cos_list)
        mean_cos = float(np.mean(all_cos))
        std_cos = float(np.std(all_cos))
        results[fold] = {"cosine": mean_cos, "std": std_cos, "n_train": int(n_train)}
        print(f"  fold {fold}: cos = {mean_cos:+.4f} +- {std_cos:.4f}  n_train={n_train}")

    benefit = [1, 2, 8]
    hurt = [3, 4, 5, 6]

    b_cos = np.mean([results[f]["cosine"] for f in benefit])
    h_cos = np.mean([results[f]["cosine"] for f in hurt])
    print(f"\n  benefit folds mean cos = {b_cos:+.4f}")
    print(f"  hurt folds mean cos    = {h_cos:+.4f}")
    print(f"  diff = {b_cos - h_cos:+.4f}")

    payload = {
        "folds": {
            str(k): {
                "cosine": float(v["cosine"]),
                "std": float(v["std"]),
                "n_train": int(v["n_train"]),
            }
            for k, v in results.items()
        },
        "benefit_mean_cos": float(b_cos),
        "hurt_mean_cos": float(h_cos),
        "diff": float(b_cos - h_cos),
    }
    C.save_result("Exp6_gradient_conflict", payload)
    print("saved: runs/Exp6_gradient_conflict/result.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
