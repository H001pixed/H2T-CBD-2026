"""Rescaled-lambda LOPO runs for the ARS threshold theta (Exp2.8, unified K/lambda).

Design principle: the per-group anchor weight is kept constant at lambda/6
(the 6-group setting) across all configurations. For K kept groups the anchor
weight is therefore lambda = 0.5 * K / 6:
  K=6 -> 0.5,   K=5 -> 0.4167,   K=4 -> 0.3333,   K=3 -> 0.25.

This script trains the 4-group (drop coherence + rhetoric, theta=0.15) and
3-group (additionally drop mechanics, theta=0.20) configurations under P-LOPO
and rebuilds the theta-sensitivity table using the rescaled 5-group runs
(B1_5grp_rescaled_LOPO_feat_*) and the 6-group runs. The table is saved to
results/runs/theta_sensitivity/result.json (overwriting the previous table,
whose 4/3-group rows used the original lambda=0.5 convention).

Usage: python Run_Lopo_Theta_Sensitivity_Rescaled.py
Outputs: results/runs/B1_{4,3}grp_rescaled_LOPO_feat_* and
         results/runs/theta_sensitivity/result.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

for core in ("00_core",):
    p = Path(__file__).resolve().parents[1] / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import Common as C  # noqa: E402
from Engine import load_emb, FeatNorm  # noqa: E402
from Models import H2TCBD  # noqa: E402
from Metrics_Logger import MetricsLogger  # noqa: E402

SEEDS = [13, 42, 123]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_LAM = 0.5
GROUP_NAMES = list(C.ANCHOR_GROUPS.keys())
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]
HARMED_FOLDS = [3, 4, 5, 6]

# theta -> (n kept, kept indices, dropped groups)
THETA_CONFIGS = [
    (0.05, 6, list(range(6)), []),
    (0.08, 5, [0, 1, 2, 3, 4], ["rhetoric_engagement"]),
    (0.10, 5, [0, 1, 2, 3, 4], ["rhetoric_engagement"]),
    (0.15, 4, [0, 1, 2, 3], ["coherence_readability", "rhetoric_engagement"]),
    (0.20, 3, [0, 1, 2],
     ["mechanics", "coherence_readability", "rhetoric_engagement"]),
]


def train_selective(tr, va, te, scoremap_tr, seed, feat_norm, kept, lam, ml=None):
    """Train H2T-CBD with the kept anchor heads and the given anchor weight."""
    C.set_seed(seed)
    model = H2TCBD(in_dim=tr["emb"].shape[1], K=6, anchor_dims=ANCHOR_DIMS,
                   use_anchor=True).to(DEVICE)
    model.anchor_heads = nn.ModuleList([model.anchor_heads[k] for k in kept])
    kept_dims = [ANCHOR_DIMS[k] for k in kept]
    n_kept = len(kept)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    Xtr = torch.tensor(tr["emb"], dtype=torch.float32)
    ytr = torch.tensor(scoremap_tr.to_unit(tr["score"]), dtype=torch.float32)
    Ftr = torch.tensor(feat_norm.transform(tr["feats"]), dtype=torch.float32)
    Xva = torch.tensor(va["emb"], dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(te["emb"], dtype=torch.float32).to(DEVICE)
    n = len(Xtr)

    def evaluate(X, raw, smap):
        model.eval()
        with torch.no_grad():
            pred_u = model(X)["score"].cpu().numpy()
        pred_i = smap.from_unit(pred_u)
        return C.qwk(raw, pred_i, smap.smin, smap.smax), pred_i

    best_qwk, best_state, best_ep = -1e9, None, -1
    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb = Xtr[idx].to(DEVICE)
            yb = ytr[idx].to(DEVICE)
            fb = Ftr[idx].to(DEVICE)
            out = model(xb)
            loss = mse_fn(out["score"], yb)
            a_loss = 0.0
            for k, head in enumerate(model.anchor_heads):
                pred = head(out["z"][:, kept[k]:kept[k] + 1])
                a_loss = a_loss + mse_fn(pred, fb[:, kept_dims[k]])
            loss = loss + lam * a_loss / n_kept
            opt.zero_grad()
            loss.backward()
            opt.step()
        vq, _ = evaluate(Xva, va["score"], scoremap_tr)
        if ml is not None:
            ml.log(epoch=ep, val_qwk=round(vq, 4))
        if vq > best_qwk:
            best_qwk, best_ep = vq, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep - best_ep >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        if ml is not None:
            torch.save(best_state, Path(ml.exp_dir) / "model.pt")
    tq, pred_i = evaluate(Xte, te["score"], scoremap_tr)
    return {"test_qwk": round(tq, 4),
            "test_rmse": round(C.rmse(te["score"], pred_i), 4),
            "test_pearson": round(C.pearson(te["score"], pred_i), 4),
            "val_qwk": round(best_qwk, 4),
            "best_epoch": best_ep}, model


def load_qwk(exp_id):
    p = C.RUNS / exp_id / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["test_qwk"]


def config_stats(prefix):
    qwks, harmed = [], []
    for seed in SEEDS:
        for fold in range(1, 9):
            q = load_qwk(f"{prefix}_s{seed}_fold{fold}")
            qb = load_qwk(f"B1_LOPO_feat_blackbox_s{seed}_fold{fold}")
            if q is None or qb is None:
                return None
            qwks.append(q)
            if fold in HARMED_FOLDS:
                harmed.append(q - qb)
    return round(float(np.mean(qwks)), 4), round(float(np.mean(harmed)), 4)


def main():
    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    sets = sorted(np.unique(d["essay_set"]).tolist())

    for n_grp, kept in ((4, [0, 1, 2, 3]), (3, [0, 1, 2])):
        lam = BASE_LAM * n_grp / 6
        print(f"training {n_grp}-group, lambda={lam:.6f} (per-group = lambda/6)")
        for seed in SEEDS:
            for hold in sets:
                te_mask = d["essay_set"] == hold
                tr_idx = np.where(~te_mask)[0]
                te_idx = np.where(te_mask)[0]
                rng = np.random.default_rng(seed)
                rng.shuffle(tr_idx)
                n_va = max(1, int(0.1 * len(tr_idx)))
                iva, itr = tr_idx[:n_va], tr_idx[n_va:]
                fn = FeatNorm().fit(d["feats"][itr])
                smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
                sub = lambda i: {"emb": d["emb"][i], "score": d["score"][i],
                                 "feats": d["feats"][i]}
                exp_id = f"B1_{n_grp}grp_rescaled_LOPO_feat_s{seed}_fold{hold}"
                if C.already_done(exp_id):
                    r = json.loads(C.result_path(exp_id).read_text())
                    print(f"[skip] {exp_id} qwk={r['test_qwk']}")
                else:
                    ml = MetricsLogger(C.RUNS / exp_id)
                    metrics, _ = train_selective(sub(itr), sub(iva), sub(te_idx),
                                                 smap, seed=seed, feat_norm=fn,
                                                 kept=kept, lam=lam, ml=ml)
                    r = {"exp_id": exp_id, "seed": seed, "fold": int(hold),
                         "n_groups": n_grp, "lambda": round(lam, 6), **metrics}
                    C.save_result(exp_id, r)
                    print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")

    # Rescaled theta table.
    prefix_map = {
        6: "B1_LOPO_feat_h2tcbd",
        5: "B1_5grp_rescaled_LOPO_feat",
        4: "B1_4grp_rescaled_LOPO_feat",
        3: "B1_3grp_rescaled_LOPO_feat",
    }
    rows = []
    print(f"\n{'theta':>6s} {'n':>2s} {'lambda':>8s} {'QWK':>8s} {'harmed':>8s}")
    for theta, n_grp, kept, dropped in THETA_CONFIGS:
        lam = BASE_LAM * n_grp / 6
        st = config_stats(prefix_map[n_grp])
        if st is None:
            print(f"{theta:.2f}: missing runs for {n_grp}-group")
            continue
        rows.append({"theta": theta, "n_groups": n_grp,
                     "lambda": round(lam, 4),
                     "retained": [GROUP_NAMES[i] for i in kept],
                     "dropped": dropped,
                     "overall_qwk": st[0],
                     "harmed_mean_dqwk": st[1]})
        print(f"{theta:.2f} {n_grp:>2d} {lam:>8.4f} {st[0]:>8.4f} {st[1]:>8.4f}")

    out_dir = C.RUNS / "theta_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "note": ("ARS threshold sensitivity on ASAP-1 P-LOPO with rescaled "
                 "lambda (per-group anchor weight constant at lambda/6): "
                 "K=6->0.5, K=5->0.4167, K=4->0.3333, K=3->0.25."),
        "rows": rows,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: (",".join(v) if isinstance(v, list) else v)
                   for k, v in r.items()} for r in rows]).to_csv(
        out_dir / "theta_sensitivity.csv", index=False)
    print(f"\nsaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
