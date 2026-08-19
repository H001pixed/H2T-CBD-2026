"""LOPO sensitivity runs for the ARS threshold theta (Exp8 theta sensitivity).

ARS values on ASAP-1: length/fluency 0.590, lexical 0.390, syntactic 0.333,
mechanics 0.175, coherence/readability 0.110, rhetoric 0.070. The threshold
decisions are:
  theta = 0.05 -> keep 6 groups (existing 6-group runs)
  theta = 0.08 -> drop rhetoric (existing 5-group runs)
  theta = 0.10 -> drop rhetoric (existing 5-group runs)
  theta = 0.15 -> drop coherence + rhetoric (keep 4 groups, new runs)
  theta = 0.20 -> drop mechanics + coherence + rhetoric (keep 3 groups, new
                  runs)

This script trains the 4-group and 3-group selective-anchoring H2T-CBD models
under P-LOPO (8 prompts x 3 seeds, same protocol as run_lopo_5groups.py), then
builds the full sensitivity table: theta, retained groups, dropped groups,
overall P-LOPO QWK, and harmed-prompt (prompts 3/4/5/6) mean delta-QWK versus
the black-box baseline.

Usage: python run_lopo_theta_sensitivity.py
Outputs: results/runs/B1_{4,3}grp_LOPO_feat_* (result.json + metrics + model.pt)
         results/runs/theta_sensitivity/result.json and
         results/runs/theta_sensitivity/theta_sensitivity.csv
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

import common as C  # noqa: E402
from engine import load_emb, FeatNorm  # noqa: E402
from models import H2TCBD  # noqa: E402
from metrics_logger import MetricsLogger  # noqa: E402

SEEDS = [13, 42, 123]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM = 0.5
GROUP_NAMES = list(C.ANCHOR_GROUPS.keys())
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]
HARMED_FOLDS = [3, 4, 5, 6]

# theta -> (n_groups kept, kept indices, label)
THETA_CONFIGS = [
    (0.05, 6, list(range(6))),
    (0.08, 5, [0, 1, 2, 3, 4]),
    (0.10, 5, [0, 1, 2, 3, 4]),
    (0.15, 4, [0, 1, 2, 3]),
    (0.20, 3, [0, 1, 2]),
]


def train_selective(tr, va, te, scoremap_tr, seed, feat_norm, kept, ml=None):
    """Train H2T-CBD with only the kept anchor heads (P-LOPO protocol)."""
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
            loss = loss + LAM * a_loss / n_kept
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


def main():
    # ---- Train the 4-group and 3-group LOPO runs (theta 0.15 / 0.20). ----
    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    sets = sorted(np.unique(d["essay_set"]).tolist())
    for n_grp, kept in ((4, [0, 1, 2, 3]), (3, [0, 1, 2])):
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
                exp_id = f"B1_{n_grp}grp_LOPO_feat_s{seed}_fold{hold}"
                if C.already_done(exp_id):
                    r = json.loads(C.result_path(exp_id).read_text())
                    print(f"[skip] {exp_id} qwk={r['test_qwk']}")
                else:
                    ml = MetricsLogger(C.RUNS / exp_id)
                    metrics, _ = train_selective(sub(itr), sub(iva), sub(te_idx),
                                                 smap, seed=seed, feat_norm=fn,
                                                 kept=kept, ml=ml)
                    r = {"exp_id": exp_id, "seed": seed, "fold": int(hold),
                         "n_groups": n_grp, **metrics}
                    C.save_result(exp_id, r)
                    print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")

    # ---- Build the sensitivity table. ----
    rows = []
    for theta, n_grp, kept in THETA_CONFIGS:
        dropped = [g for i, g in enumerate(GROUP_NAMES) if i not in kept]
        prefix = {6: "B1_LOPO_feat_h2tcbd", 5: "B1_5grp_LOPO_feat",
                  4: "B1_4grp_LOPO_feat", 3: "B1_3grp_LOPO_feat"}[n_grp]
        qwks, dq_harmed = [], []
        for seed in SEEDS:
            for fold in sets:
                q = load_qwk(f"{prefix}_s{seed}_fold{fold}")
                qb = load_qwk(f"B1_LOPO_feat_blackbox_s{seed}_fold{fold}")
                if q is None or qb is None:
                    continue
                qwks.append(q)
                if fold in HARMED_FOLDS:
                    dq_harmed.append(q - qb)
        overall = float(np.mean(qwks)) if qwks else float("nan")
        harmed = float(np.mean(dq_harmed)) if dq_harmed else float("nan")
        rows.append({
            "theta": theta,
            "n_groups": n_grp,
            "retained": [g for i, g in enumerate(GROUP_NAMES) if i in kept],
            "dropped": dropped,
            "overall_qwk": round(overall, 4),
            "harmed_mean_dqwk": round(harmed, 4),
        })
        print(f"theta={theta:.2f} n_grp={n_grp} "
              f"overall_qwk={overall:.4f} harmed_dqwk={harmed:.4f}")

    out_dir = C.RUNS / "theta_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "note": ("ARS threshold sensitivity on ASAP-1 P-LOPO. theta decisions "
                 "use the full-data ARS (0.590/0.390/0.333/0.175/0.110/0.070); "
                 "theta=0.05/0.08/0.10 reuse the existing 6/5-group runs, "
                 "theta=0.15/0.20 train new 4/3-group runs (script "
                 "run_lopo_theta_sensitivity.py)."),
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
